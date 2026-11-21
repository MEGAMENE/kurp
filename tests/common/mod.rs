#![allow(dead_code)]

pub mod synthetic_images;
pub use synthetic_images::*;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Duration;

use arc_swap::ArcSwap;
use axum::{
    Json, Router,
    body::Body,
    extract::{Path, State},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
    routing::{any, get, patch},
};
use bytes::Bytes;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::{Semaphore, broadcast, oneshot};

use kurp::app_state::AppState;
use kurp::clients::komga_client::KomgaClient;
use kurp::clients::proxy_client::ProxyClient;
use kurp::config::app_config::AppConfig;
use kurp::server;
use kurp::tags_provider::UpscaleTagChecker;
use kurp::upscaler::book_profile::BookProfileManager;
use kurp::upscaler::upscaler::UpscaleEngine;

#[derive(Clone, Debug)]
pub struct MockPageEntry {
    pub bytes: Bytes,
    pub content_type: String,
    pub etag: Option<String>,
    pub content_disposition: Option<String>,
}

impl MockPageEntry {
    pub fn png(bytes: Bytes) -> Self {
        Self {
            bytes,
            content_type: "image/png".to_string(),
            etag: None,
            content_disposition: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RecordedRequest {
    pub method: String,
    pub path: String,
    pub headers: HashMap<String, String>,
    pub body: Bytes,
}

#[derive(Clone, Default)]
pub struct MockKomgaState {
    pub pages: Arc<RwLock<HashMap<(String, usize), MockPageEntry>>>,
    pub book_page_counts: Arc<RwLock<HashMap<String, usize>>>,
    pub book_tags: Arc<RwLock<HashMap<String, Vec<String>>>>,
    pub series_tags: Arc<RwLock<HashMap<String, Vec<String>>>>,
    pub book_series: Arc<RwLock<HashMap<String, String>>>,
    pub error_status: Arc<RwLock<Option<StatusCode>>>,
    pub patch_status: Arc<RwLock<Option<StatusCode>>>,
    pub require_auth: Arc<RwLock<bool>>,
    pub page_delay: Arc<RwLock<Duration>>,
    pub partial_206: Arc<RwLock<bool>>,
    pub truncated_body: Arc<RwLock<bool>>,
    pub return_empty_pages: Arc<RwLock<bool>>,
    pub request_count: Arc<AtomicUsize>,
    pub page_fetch_count: Arc<AtomicUsize>,
    pub recorded_requests: Arc<RwLock<Vec<RecordedRequest>>>,
    pub self_base_url: Arc<RwLock<String>>,
}

pub struct MockKomgaServer {
    pub base_url: String,
    pub state: MockKomgaState,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl MockKomgaServer {
    pub async fn start() -> Self {
        let state = MockKomgaState::default();

        let app = Router::new()
            .route("/api/v1/books/{book_id}", get(handle_book_meta))
            .route("/api/v1/series/{series_id}", get(handle_series_meta))
            .route("/api/v1/books/{book_id}/pages", get(handle_book_pages))
            .route(
                "/api/v1/books/{book_id}/pages/{page_number}",
                get(handle_page_image)
                    .head(handle_page_head)
                    .options(handle_options),
            )
            .route(
                "/opds/v1.2/books/{book_id}/pages/{page_number}",
                get(handle_page_image)
                    .head(handle_page_head)
                    .options(handle_options),
            )
            .route(
                "/api/v1/series/{series_id}/metadata",
                patch(handle_patch_metadata).options(handle_options),
            )
            .route(
                "/api/v1/books/{book_id}/metadata",
                patch(handle_patch_metadata).options(handle_options),
            )
            .route(
                "/api/v1/books/metadata",
                patch(handle_patch_metadata).options(handle_options),
            )
            .route("/redirect-test", get(handle_redirect))
            .route("/proxy-echo", any(handle_proxy_echo))
            .with_state(state.clone());

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("Failed to bind mock Komga ephemeral port");
        let local_addr = listener.local_addr().expect("Failed to get local addr");
        let base_url = format!("http://{}", local_addr);
        *state.self_base_url.write().unwrap() = base_url.clone();

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async {
                    let _ = shutdown_rx.await;
                })
                .await
                .ok();
        });

        Self {
            base_url,
            state,
            shutdown_tx: Some(shutdown_tx),
        }
    }

    pub fn set_book_tags(&self, book_id: &str, series_id: &str, tags: Vec<&str>) {
        self.state
            .book_series
            .write()
            .unwrap()
            .insert(book_id.to_string(), series_id.to_string());
        self.state.book_tags.write().unwrap().insert(
            book_id.to_string(),
            tags.into_iter().map(str::to_string).collect(),
        );
    }

    pub fn set_series_tags(&self, series_id: &str, tags: Vec<&str>) {
        self.state.series_tags.write().unwrap().insert(
            series_id.to_string(),
            tags.into_iter().map(str::to_string).collect(),
        );
    }

    pub fn set_page(&self, book_id: &str, page_number: usize, entry: MockPageEntry) {
        self.state
            .pages
            .write()
            .unwrap()
            .insert((book_id.to_string(), page_number), entry);
    }

    pub fn set_default_book_pages(&self, book_id: &str, count: usize, entry: MockPageEntry) {
        self.state
            .book_page_counts
            .write()
            .unwrap()
            .insert(book_id.to_string(), count);
        for p in 1..=count {
            self.set_page(book_id, p, entry.clone());
        }
    }

    pub fn set_error(&self, status: StatusCode) {
        *self.state.error_status.write().unwrap() = Some(status);
    }

    pub fn clear_error(&self) {
        *self.state.error_status.write().unwrap() = None;
    }

    pub fn set_require_auth(&self, require: bool) {
        *self.state.require_auth.write().unwrap() = require;
    }

    pub fn set_page_delay(&self, delay: Duration) {
        *self.state.page_delay.write().unwrap() = delay;
    }

    pub fn page_fetch_count(&self) -> usize {
        self.state.page_fetch_count.load(Ordering::SeqCst)
    }

    pub fn recorded_requests(&self) -> Vec<RecordedRequest> {
        self.state.recorded_requests.read().unwrap().clone()
    }

    pub fn clear_recorded(&self) {
        self.state.recorded_requests.write().unwrap().clear();
    }
}

impl Drop for MockKomgaServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

fn record_headers(
    state: &MockKomgaState,
    method: &str,
    path: &str,
    headers: &HeaderMap,
    body: Bytes,
) {
    let mut map = HashMap::new();
    for (k, v) in headers {
        if let Ok(s) = v.to_str() {
            map.insert(k.as_str().to_ascii_lowercase(), s.to_string());
        }
    }
    state.recorded_requests.write().unwrap().push(RecordedRequest {
        method: method.to_string(),
        path: path.to_string(),
        headers: map,
        body,
    });
}

fn has_any_auth(headers: &HeaderMap) -> bool {
    headers.contains_key("x-api-key")
        || headers.contains_key("x-auth-token")
        || headers.contains_key(header::AUTHORIZATION)
        || headers.contains_key(header::COOKIE)
}

async fn handle_book_meta(
    State(state): State<MockKomgaState>,
    Path(book_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    state.request_count.fetch_add(1, Ordering::SeqCst);
    record_headers(
        &state,
        "GET",
        &format!("/api/v1/books/{book_id}"),
        &headers,
        Bytes::new(),
    );

    if *state.require_auth.read().unwrap() && !has_any_auth(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    if let Some(status) = *state.error_status.read().unwrap() {
        return status.into_response();
    }

    let series_id = state
        .book_series
        .read()
        .unwrap()
        .get(&book_id)
        .cloned()
        .unwrap_or_else(|| "series-default".to_string());
    let tags = state
        .book_tags
        .read()
        .unwrap()
        .get(&book_id)
        .cloned()
        .unwrap_or_else(|| vec!["upscale".to_string()]);

    Json(json!({
        "id": book_id,
        "seriesId": series_id,
        "name": "Test Volume",
        "number": 1,
        "metadata": {
            "title": "Test Volume",
            "tags": tags
        }
    }))
    .into_response()
}

async fn handle_series_meta(
    State(state): State<MockKomgaState>,
    Path(series_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    state.request_count.fetch_add(1, Ordering::SeqCst);
    record_headers(
        &state,
        "GET",
        &format!("/api/v1/series/{series_id}"),
        &headers,
        Bytes::new(),
    );

    if *state.require_auth.read().unwrap() && !has_any_auth(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    if let Some(status) = *state.error_status.read().unwrap() {
        return status.into_response();
    }

    let tags = state
        .series_tags
        .read()
        .unwrap()
        .get(&series_id)
        .cloned()
        .unwrap_or_default();

    Json(json!({
        "id": series_id,
        "metadata": {
            "tags": tags
        }
    }))
    .into_response()
}

async fn handle_book_pages(
    State(state): State<MockKomgaState>,
    Path(book_id): Path<String>,
    headers: HeaderMap,
) -> Response {
    state.request_count.fetch_add(1, Ordering::SeqCst);
    record_headers(
        &state,
        "GET",
        &format!("/api/v1/books/{book_id}/pages"),
        &headers,
        Bytes::new(),
    );

    if *state.require_auth.read().unwrap() && !has_any_auth(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    if let Some(status) = *state.error_status.read().unwrap() {
        return status.into_response();
    }
    if *state.return_empty_pages.read().unwrap() {
        return Json(Vec::<Value>::new()).into_response();
    }

    let pages_map = state.pages.read().unwrap();
    let mut keys: Vec<usize> = pages_map
        .keys()
        .filter(|(b, _)| b == &book_id)
        .map(|(_, p)| *p)
        .collect();
    keys.sort();

    let list: Vec<Value> = keys
        .into_iter()
        .map(|num| {
            let entry = &pages_map[&(book_id.clone(), num)];
            json!({
                "number": num,
                "fileName": format!("page_{:03}.png", num),
                "mediaType": entry.content_type.split(';').next().unwrap_or("image/png").trim(),
                "width": 400,
                "height": 400,
                "sizeBytes": entry.bytes.len(),
                "size": format!("{} KB", entry.bytes.len() / 1024)
            })
        })
        .collect();

    Json(list).into_response()
}

async fn handle_page_head(
    State(state): State<MockKomgaState>,
    Path((book_id, page_number)): Path<(String, usize)>,
    headers: HeaderMap,
) -> Response {
    state.request_count.fetch_add(1, Ordering::SeqCst);
    record_headers(
        &state,
        "HEAD",
        &format!("/api/v1/books/{book_id}/pages/{page_number}"),
        &headers,
        Bytes::new(),
    );

    let len = state
        .pages
        .read()
        .unwrap()
        .get(&(book_id, page_number))
        .map(|e| e.bytes.len())
        .unwrap_or(0);

    if len == 0 {
        return StatusCode::NOT_FOUND.into_response();
    }

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/png")
        .header(header::CONTENT_LENGTH, len)
        .body(Body::empty())
        .unwrap()
}

async fn handle_page_image(
    State(state): State<MockKomgaState>,
    Path((book_id, page_number)): Path<(String, usize)>,
    headers: HeaderMap,
) -> Response {
    state.request_count.fetch_add(1, Ordering::SeqCst);
    state.page_fetch_count.fetch_add(1, Ordering::SeqCst);
    record_headers(
        &state,
        "GET",
        &format!("/api/v1/books/{book_id}/pages/{page_number}"),
        &headers,
        Bytes::new(),
    );

    if *state.require_auth.read().unwrap() && !has_any_auth(&headers) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    if let Some(status) = *state.error_status.read().unwrap() {
        return status.into_response();
    }

    let delay = *state.page_delay.read().unwrap();
    if !delay.is_zero() {
        tokio::time::sleep(delay).await;
    }

    if *state.truncated_body.read().unwrap() {
        return StatusCode::BAD_GATEWAY.into_response();
    }

    let entry = state
        .pages
        .read()
        .unwrap()
        .get(&(book_id, page_number))
        .cloned();
    let Some(entry) = entry else {
        return StatusCode::NOT_FOUND.into_response();
    };

    if *state.partial_206.read().unwrap() {
        let slice = entry.bytes.slice(0..entry.bytes.len().min(32));
        return Response::builder()
            .status(StatusCode::PARTIAL_CONTENT)
            .header(header::CONTENT_TYPE, &entry.content_type)
            .header(
                header::CONTENT_RANGE,
                format!("bytes 0-{}/{}", slice.len().saturating_sub(1), entry.bytes.len()),
            )
            .header(header::CONTENT_LENGTH, slice.len())
            .body(Body::from(slice))
            .unwrap();
    }

    if let (Some(req_inm), Some(etag)) = (
        headers.get(header::IF_NONE_MATCH).and_then(|v| v.to_str().ok()),
        entry.etag.as_deref(),
    ) && req_inm.split(',').any(|t| t.trim() == etag)
    {
        return Response::builder()
            .status(StatusCode::NOT_MODIFIED)
            .header(header::ETAG, etag)
            .body(Body::empty())
            .unwrap();
    }

    let mut builder = Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, &entry.content_type)
        .header(header::CONTENT_LENGTH, entry.bytes.len());

    if let Some(ref etag) = entry.etag {
        builder = builder.header(header::ETAG, etag);
    }
    if let Some(ref disp) = entry.content_disposition {
        builder = builder.header(header::CONTENT_DISPOSITION, disp);
    }

    builder.body(Body::from(entry.bytes)).unwrap()
}

/// Spawns a raw TCP upstream server that replies `200 OK` with `Content-Length: 50000`
/// and immediately drops the connection after 8 bytes so `to_bytes` sees a real EOF error.
pub async fn spawn_truncated_body_upstream() -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("Failed to bind truncated upstream port");
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        while let Ok((mut sock, _)) = listener.accept().await {
            tokio::spawn(async move {
                let mut buf = [0u8; 2048];
                let _ = tokio::io::AsyncReadExt::read(&mut sock, &mut buf).await;
                let req_text = String::from_utf8_lossy(&buf);
                if req_text.contains("/pages/") {
                    let resp = b"HTTP/1.1 200 OK\r\nContent-Type: image/png\r\nContent-Length: 50000\r\nConnection: close\r\n\r\n\x89PNG\r\n\x1a\n";
                    let _ = sock.write_all(resp).await;
                } else if req_text.contains("/pages") {
                    let body = "[]";
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                } else {
                    let body = r#"{"id":"trunc-book","seriesId":"s1","metadata":{"tags":["upscale"]}}"#;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = sock.write_all(resp.as_bytes()).await;
                }
                let _ = sock.shutdown().await;
            });
        }
    });
    format!("http://{addr}")
}

async fn handle_options(
    State(state): State<MockKomgaState>,
    headers: HeaderMap,
    req_uri: axum::http::Uri,
) -> Response {
    record_headers(&state, "OPTIONS", req_uri.path(), &headers, Bytes::new());
    Response::builder()
        .status(StatusCode::NO_CONTENT)
        .header(header::ACCESS_CONTROL_ALLOW_ORIGIN, "*")
        .header(header::ACCESS_CONTROL_ALLOW_METHODS, "GET, HEAD, PATCH, OPTIONS")
        .body(Body::empty())
        .unwrap()
}

async fn handle_patch_metadata(
    State(state): State<MockKomgaState>,
    headers: HeaderMap,
    req_uri: axum::http::Uri,
    body: Bytes,
) -> Response {
    record_headers(&state, "PATCH", req_uri.path(), &headers, body);
    if let Some(status) = *state.patch_status.read().unwrap() {
        return status.into_response();
    }
    StatusCode::NO_CONTENT.into_response()
}

async fn handle_redirect(State(state): State<MockKomgaState>) -> Response {
    let base = state.self_base_url.read().unwrap().clone();
    Response::builder()
        .status(StatusCode::FOUND)
        .header(header::LOCATION, format!("{base}/login?next=/dashboard"))
        .body(Body::empty())
        .unwrap()
}

async fn handle_proxy_echo(headers: HeaderMap) -> Response {
    let mut map = serde_json::Map::new();
    for (k, v) in &headers {
        if let Ok(s) = v.to_str() {
            map.insert(k.as_str().to_ascii_lowercase(), Value::String(s.to_string()));
        }
    }
    Json(Value::Object(map)).into_response()
}

/// Starts a real in-process Kurp HTTP proxy server using `kurp::server::make_routes`.
pub struct TestKurpServer {
    pub base_url: String,
    pub state: AppState,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl TestKurpServer {
    pub async fn start(config: AppConfig, upscaler: Arc<UpscaleEngine>) -> Self {
        let reqwest_client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("Failed to build reqwest client");

        let upstream = config.upstream_url.trim_end_matches('/').to_string();
        let komga_client = Arc::new(KomgaClient::new(reqwest_client.clone(), upstream.clone()));
        let proxy_client = Arc::new(ProxyClient::new(reqwest_client, upstream));
        let upscale_tag_checker =
            Arc::new(UpscaleTagChecker::new(config.upscale_tag.clone(), komga_client.clone()));
        let book_profile_manager = Arc::new(BookProfileManager::new(
            komga_client.clone(),
            &config.auto_levels.book_profile,
        ));
        let pipeline_semaphore = Arc::new(Semaphore::new(config.max_concurrent_pipelines.max(1)));
        let (reload_tx, _) = broadcast::channel::<()>(4);

        let state = AppState {
            config: Arc::new(ArcSwap::from_pointee(config)),
            upscaler,
            komga_client,
            proxy_client,
            upscale_tag_checker,
            book_profile_manager,
            pipeline_semaphore,
            shutdown_tx: reload_tx,
        };

        let routes = server::make_routes(state.clone());
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("Failed to bind Kurp ephemeral test port");
        let addr = listener.local_addr().expect("local_addr");
        let base_url = format!("http://{addr}");

        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        tokio::spawn(async move {
            axum::serve(
                listener,
                routes.into_make_service_with_connect_info::<SocketAddr>(),
            )
            .with_graceful_shutdown(async {
                let _ = shutdown_rx.await;
            })
            .await
            .ok();
        });

        Self {
            base_url,
            state,
            shutdown_tx: Some(shutdown_tx),
        }
    }
}

impl Drop for TestKurpServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

pub fn create_cpu_upscale_config(upstream_url: &str) -> AppConfig {
    let models_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("realcugan-ncnn-vulkan-rs")
        .join("src")
        .join("realcugan-ncnn-vulkan")
        .join("models");

    let mut cfg = AppConfig {
        upstream_url: upstream_url.to_string(),
        size_threshold_enabled: false,
        ..AppConfig::default()
    };
    cfg.realcugan.gpuid = -1;
    cfg.realcugan.tile_size = 400;
    cfg.realcugan.num_threads = Some(2);
    cfg.realcugan.models_path = models_dir;
    cfg
}
