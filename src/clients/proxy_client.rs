use std::net::SocketAddr;
use std::sync::LazyLock;

use axum::body::Body;
use axum::extract::{ConnectInfo, Request};
use axum::http::header::{
    CONNECTION, HOST, LOCATION, PROXY_AUTHENTICATE, PROXY_AUTHORIZATION, TE, TRAILER,
    TRANSFER_ENCODING, UPGRADE,
};
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use axum::response::Response;
use sync_wrapper::SyncStream;

use crate::models::errors::ProxyError;

static KEEP_ALIVE: HeaderName = HeaderName::from_static("keep-alive");
static PROXY_CONNECTION: HeaderName = HeaderName::from_static("proxy-connection");
static TRAILERS: HeaderName = HeaderName::from_static("trailers");
static X_FORWARDED_HOST: HeaderName = HeaderName::from_static("x-forwarded-host");
static X_FORWARDED_PROTO: HeaderName = HeaderName::from_static("x-forwarded-proto");
static X_FORWARDED_FOR: HeaderName = HeaderName::from_static("x-forwarded-for");

static STATIC_HOP_HEADERS: LazyLock<[HeaderName; 10]> = LazyLock::new(|| {
    [
        CONNECTION,
        KEEP_ALIVE.clone(),
        PROXY_AUTHENTICATE,
        PROXY_AUTHORIZATION,
        PROXY_CONNECTION.clone(),
        TRAILER,
        TRAILERS.clone(),
        TRANSFER_ENCODING,
        UPGRADE,
        HOST,
    ]
});

#[derive(Clone, Debug)]
struct ClientOrigin {
    scheme: String,
    host: Option<String>,
}

#[derive(Clone)]
pub struct ProxyClient {
    client: reqwest::Client,
    base_url: std::sync::Arc<std::sync::RwLock<String>>,
}

impl ProxyClient {
    pub fn new(client: reqwest::Client, base_url: String) -> Self {
        Self {
            client,
            base_url: std::sync::Arc::new(std::sync::RwLock::new(
                base_url.trim_end_matches('/').to_string(),
            )),
        }
    }

    pub fn set_base_url(&self, base_url: String) {
        let mut guard = self.base_url.write().unwrap_or_else(|e| e.into_inner());
        *guard = base_url.trim_end_matches('/').to_string();
    }

    pub fn base_url(&self) -> String {
        self.base_url
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    pub async fn proxy_request(&self, req: Request) -> Result<Response, ProxyError> {
        let (request, origin, base_url) = self.to_proxy_request(req)?;
        let response = self.call_proxy_request(request).await?;
        self.response_to_reply(response, Some(&origin), &base_url).await
    }

    async fn response_to_reply(
        &self,
        response: reqwest::Response,
        origin: Option<&ClientOrigin>,
        base_url: &str,
    ) -> Result<Response, ProxyError> {
        let status = response.status();
        let filtered_headers = filter_hop_headers(response.headers());
        let mut builder = Response::builder();

        for (k, v) in filtered_headers.iter() {
            if k == LOCATION
                && let Ok(loc_str) = v.to_str()
                && let Some(rewritten) = rewrite_location(loc_str, base_url, origin)
                && let Ok(hv) = HeaderValue::from_str(&rewritten)
            {
                builder = builder.header(k, hv);
                continue;
            }
            builder = builder.header(k, v);
        }

        let body = Body::from_stream(response.bytes_stream());
        builder
            .status(status)
            .body(body)
            .map_err(|err| ProxyError::internal(err.to_string()))
    }

    fn to_proxy_request(
        &self,
        request: Request,
    ) -> Result<(reqwest::Request, ClientOrigin, String), ProxyError> {
        let base_url = self.base_url();
        let path = request
            .uri()
            .path_and_query()
            .map(|pq| pq.as_str())
            .unwrap_or_else(|| request.uri().path());
        let url = if path.starts_with('/') {
            format!("{}{}", base_url, path)
        } else {
            format!("{}/{}", base_url, path)
        };

        let scheme = request
            .headers()
            .get(&X_FORWARDED_PROTO)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.split(',').next())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .or_else(|| request.uri().scheme_str())
            .unwrap_or("http")
            .to_string();

        let original_host = request.headers().get(HOST).cloned();
        let forwarded_host = request
            .headers()
            .get(&X_FORWARDED_HOST)
            .and_then(|v| v.to_str().ok())
            .and_then(|s| s.split(',').next())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .or_else(|| {
                original_host
                    .as_ref()
                    .and_then(|v| v.to_str().ok())
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
            });

        let client_ip = request
            .extensions()
            .get::<ConnectInfo<SocketAddr>>()
            .map(|ConnectInfo(addr)| addr.ip().to_string());

        let method = request.method().clone();
        let mut headers = filter_hop_headers(request.headers());

        if !headers.contains_key(&X_FORWARDED_HOST)
            && let Some(host_val) = original_host
        {
            headers.insert(X_FORWARDED_HOST.clone(), host_val);
        }
        if !headers.contains_key(&X_FORWARDED_PROTO)
            && let Ok(proto_val) = HeaderValue::from_str(&scheme)
        {
            headers.insert(X_FORWARDED_PROTO.clone(), proto_val);
        }
        if let Some(ip) = client_ip {
            let new_xff = match headers
                .get(&X_FORWARDED_FOR)
                .and_then(|v| v.to_str().ok())
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                Some(existing) => format!("{}, {}", existing, ip),
                None => ip,
            };
            if let Ok(xff_val) = HeaderValue::from_str(&new_xff) {
                headers.insert(X_FORWARDED_FOR.clone(), xff_val);
            }
        }

        let body_stream = SyncStream::new(request.into_body().into_data_stream());
        let reqwest_req = self
            .client
            .request(method, url)
            .headers(headers)
            .body(reqwest::Body::wrap_stream(body_stream))
            .build()
            .map_err(|err| ProxyError::bad_gateway(err.to_string()))?;

        Ok((
            reqwest_req,
            ClientOrigin {
                scheme,
                host: forwarded_host,
            },
            base_url,
        ))
    }

    async fn call_proxy_request(
        &self,
        request: reqwest::Request,
    ) -> Result<reqwest::Response, ProxyError> {
        self.client
            .execute(request)
            .await
            .map_err(|err| ProxyError::from_reqwest(&err))
    }
}

fn filter_hop_headers(headers: &HeaderMap<HeaderValue>) -> HeaderMap<HeaderValue> {
    let mut dynamic_hops: Vec<String> = Vec::new();
    for conn_val in headers.get_all(CONNECTION) {
        if let Ok(conn_str) = conn_val.to_str() {
            for token in conn_str.split(',') {
                let trimmed = token.trim();
                if !trimmed.is_empty() {
                    dynamic_hops.push(trimmed.to_ascii_lowercase());
                }
            }
        }
    }

    let mut filtered = HeaderMap::with_capacity(headers.len());
    for (k, v) in headers.iter() {
        // RFC 9110 §7.6.1 / HTTP/2: preserve `TE: trailers`, strip all other `TE` values.
        if k == TE {
            if v.to_str()
                .ok()
                .is_some_and(|s| s.trim().eq_ignore_ascii_case("trailers"))
            {
                filtered.append(k.clone(), v.clone());
            }
            continue;
        }

        if STATIC_HOP_HEADERS.iter().any(|h| h == k) {
            continue;
        }

        let key_str = k.as_str();
        if dynamic_hops
            .iter()
            .any(|hop| key_str.eq_ignore_ascii_case(hop))
        {
            continue;
        }

        filtered.append(k.clone(), v.clone());
    }

    filtered
}

fn rewrite_location(
    location: &str,
    base_url: &str,
    origin: Option<&ClientOrigin>,
) -> Option<String> {
    if base_url.is_empty() || location.len() < base_url.len() {
        return None;
    }

    let (prefix, suffix) = location.split_at(base_url.len());
    if !prefix.eq_ignore_ascii_case(base_url) {
        return None;
    }

    if !suffix.is_empty()
        && !suffix.starts_with('/')
        && !suffix.starts_with('?')
        && !suffix.starts_with('#')
    {
        return None;
    }

    let normalized_suffix = if suffix.is_empty() { "/" } else { suffix };
    if let Some(origin) = origin
        && let Some(ref host) = origin.host
    {
        return Some(format!(
            "{}://{}{}",
            origin.scheme, host, normalized_suffix
        ));
    }

    if normalized_suffix.starts_with('/') {
        Some(normalized_suffix.to_string())
    } else {
        Some(format!("/{}", normalized_suffix))
    }
}
