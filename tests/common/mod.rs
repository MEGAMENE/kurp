//! Shared Test Harness & Mock Utilities for Kurp Integration Tests
//!
//! Provides in-process mock Komga HTTP server, headless mock upscaler, and configuration fixtures.

pub mod synthetic_images;
pub use synthetic_images::*;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::get,
    Json, Router,
};
use bytes::Bytes;
use image::DynamicImage;
use serde_json::json;
use std::collections::HashMap;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, RwLock,
};
use tokio::net::TcpListener;

/// State for the in-process mock Komga server
#[derive(Clone)]
pub struct MockKomgaState {
    pub pages: Arc<RwLock<HashMap<usize, Bytes>>>,
    pub error_status: Arc<RwLock<Option<StatusCode>>>,
    pub request_count: Arc<AtomicUsize>,
}

/// In-Process Mock Komga Server simulating Komga REST API
pub struct MockKomgaServer {
    pub base_url: String,
    pub state: MockKomgaState,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
}

impl MockKomgaServer {
    /// Starts an in-memory mock Komga server bound to an ephemeral localhost port
    pub async fn start() -> Self {
        let state = MockKomgaState {
            pages: Arc::new(RwLock::new(HashMap::new())),
            error_status: Arc::new(RwLock::new(None)),
            request_count: Arc::new(AtomicUsize::new(0)),
        };

        let app = Router::new()
            .route("/api/v1/books/{book_id}", get(handle_book_meta))
            .route("/api/v1/books/{book_id}/pages", get(handle_book_pages))
            .route("/api/v1/books/{book_id}/pages/{page_number}", get(handle_page_image))
            .with_state(state.clone());

        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("Failed to bind ephemeral test port");
        let local_addr = listener.local_addr().expect("Failed to get local addr");
        let base_url = format!("http://{}", local_addr);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();

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

    /// Registers a synthetic page image with the mock server
    pub fn set_page(&self, page_number: usize, image_bytes: Bytes) {
        let mut map = self.state.pages.write().unwrap();
        map.insert(page_number, image_bytes);
    }

    /// Injects an HTTP error (e.g. 500 Internal Server Error)
    pub fn set_error(&self, status: StatusCode) {
        let mut err = self.state.error_status.write().unwrap();
        *err = Some(status);
    }

    /// Clears any injected error condition
    pub fn clear_error(&self) {
        let mut err = self.state.error_status.write().unwrap();
        *err = None;
    }

    /// Returns the number of HTTP requests handled by the mock server
    pub fn request_count(&self) -> usize {
        self.state.request_count.load(Ordering::SeqCst)
    }
}

impl Drop for MockKomgaServer {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(());
        }
    }
}

async fn handle_book_meta(
    State(state): State<MockKomgaState>,
    Path(book_id): Path<String>,
) -> Response {
    state.request_count.fetch_add(1, Ordering::SeqCst);
    if let Some(status) = *state.error_status.read().unwrap() {
        return status.into_response();
    }

    Json(json!({
        "id": book_id,
        "seriesId": "series-123",
        "name": "Test Volume 1",
        "number": 1,
        "metadata": {
            "title": "Test Volume 1",
            "tags": ["upscale"]
        }
    }))
    .into_response()
}

async fn handle_book_pages(
    State(state): State<MockKomgaState>,
    Path(_book_id): Path<String>,
) -> Response {
    state.request_count.fetch_add(1, Ordering::SeqCst);
    if let Some(status) = *state.error_status.read().unwrap() {
        return status.into_response();
    }

    let pages_map = state.pages.read().unwrap();
    let mut page_list = Vec::new();
    let mut keys: Vec<_> = pages_map.keys().cloned().collect();
    keys.sort();

    for num in keys {
        let size = pages_map.get(&num).map(|b| b.len() as u64).unwrap_or(0);
        page_list.push(json!({
            "number": num,
            "fileName": format!("page_{:03}.png", num),
            "mediaType": "image/png",
            "width": 1600,
            "height": 2400,
            "sizeBytes": size
        }));
    }

    Json(page_list).into_response()
}

async fn handle_page_image(
    State(state): State<MockKomgaState>,
    Path((_book_id, page_number)): Path<(String, usize)>,
) -> Response {
    state.request_count.fetch_add(1, Ordering::SeqCst);
    if let Some(status) = *state.error_status.read().unwrap() {
        return status.into_response();
    }

    let pages_map = state.pages.read().unwrap();
    if let Some(bytes) = pages_map.get(&page_number) {
        (
            [("content-type", "image/png")],
            bytes.clone(),
        )
            .into_response()
    } else {
        StatusCode::NOT_FOUND.into_response()
    }
}

/// Headless Mock Upscaler for tests running without physical GPU
pub struct HeadlessMockUpscaler {
    pub scale_factor: u32,
    pub calls: Arc<AtomicUsize>,
}

impl HeadlessMockUpscaler {
    pub fn new(scale_factor: u32) -> Self {
        Self {
            scale_factor,
            calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn upscale_mock(&self, image: DynamicImage) -> DynamicImage {
        self.calls.fetch_add(1, Ordering::SeqCst);
        if self.scale_factor <= 1 {
            image
        } else {
            let (w, h) = (image.width(), image.height());
            image.resize_exact(
                w * self.scale_factor,
                h * self.scale_factor,
                image::imageops::FilterType::Nearest,
            )
        }
    }

    pub fn call_count(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

/// Hierarchical test configuration matching the Master Reference Specification
#[derive(Debug, Clone)]
pub struct TestAutoLevelsConfig {
    pub enabled: bool,
    pub confidence_threshold: f32,
    pub black_clip_percent: f32,
    pub white_clip_percent: f32,
    pub surge_ratio_threshold: f32,
    pub max_black_shift: u8,
    pub min_white_threshold: u8,
    pub correct_paper_cast: bool,
    pub gamma: f32,
    pub protect_sepia_artwork: bool,
}

impl Default for TestAutoLevelsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            confidence_threshold: 0.75,
            black_clip_percent: 0.1,
            white_clip_percent: 0.1,
            surge_ratio_threshold: 2.8,
            max_black_shift: 85,
            min_white_threshold: 200,
            correct_paper_cast: true,
            gamma: 1.0,
            protect_sepia_artwork: true,
        }
    }
}
