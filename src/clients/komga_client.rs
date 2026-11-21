use axum::http::header::{ACCEPT_ENCODING, AUTHORIZATION, COOKIE};
use axum::http::{HeaderMap, HeaderName, HeaderValue};
use bytes::Bytes;
use log::{info, warn};

use crate::models::errors::HttpError;
use crate::models::komga::{KomgaBook, KomgaPageInfo, KomgaSeries};

static X_API_KEY: HeaderName = HeaderName::from_static("x-api-key");
static X_AUTH_TOKEN: HeaderName = HeaderName::from_static("x-auth-token");

/// Credential headers extracted safely from an incoming client request and forwarded
/// verbatim to Komga API calls so `KomgaClient` mirrors `ProxyClient` auth parity
/// (`X-API-Key`, `X-Auth-Token`, raw `Authorization` Basic/Bearer, `Cookie`, or
/// reverse-proxy pre-auth with empty headers).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ForwardAuthHeaders {
    pub authorization: Option<HeaderValue>,
    pub x_api_key: Option<HeaderValue>,
    pub x_auth_token: Option<HeaderValue>,
    pub cookie: Option<HeaderValue>,
}

impl ForwardAuthHeaders {
    pub fn from_headers(headers: &HeaderMap) -> Self {
        Self {
            authorization: headers.get(AUTHORIZATION).cloned(),
            x_api_key: headers.get(&X_API_KEY).cloned(),
            x_auth_token: headers.get(&X_AUTH_TOKEN).cloned(),
            cookie: combine_cookie_headers(headers),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.authorization.is_none()
            && self.x_api_key.is_none()
            && self.x_auth_token.is_none()
            && self.cookie.is_none()
    }

    pub fn apply_to(&self, mut req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        if let Some(ref auth) = self.authorization {
            req = req.header(AUTHORIZATION, auth);
        }
        if let Some(ref api_key) = self.x_api_key {
            req = req.header(&X_API_KEY, api_key);
        }
        if let Some(ref auth_token) = self.x_auth_token {
            req = req.header(&X_AUTH_TOKEN, auth_token);
        }
        if let Some(ref cookie) = self.cookie {
            req = req.header(COOKIE, cookie);
        }
        req
    }
}

impl From<&HeaderMap> for ForwardAuthHeaders {
    fn from(headers: &HeaderMap) -> Self {
        Self::from_headers(headers)
    }
}

fn combine_cookie_headers(headers: &HeaderMap) -> Option<HeaderValue> {
    let mut all = headers.get_all(COOKIE).iter();
    let first = all.next()?;
    let Some(second) = all.next() else {
        return (!first.as_bytes().is_empty()).then(|| first.clone());
    };

    let mut buf = Vec::new();
    for val in [first, second].into_iter().chain(all) {
        let mut bytes = val.as_bytes().trim_ascii();
        while let Some(stripped) = bytes.strip_suffix(b";") {
            bytes = stripped.trim_ascii();
        }
        if bytes.is_empty() {
            continue;
        }
        if !buf.is_empty() {
            buf.extend_from_slice(b"; ");
        }
        buf.extend_from_slice(bytes);
    }

    if buf.is_empty() {
        None
    } else {
        HeaderValue::from_bytes(&buf).ok()
    }
}

pub struct KomgaClient {
    client: reqwest::Client,
    base_uri: std::sync::RwLock<String>,
}

impl KomgaClient {
    pub fn new(client: reqwest::Client, base_uri: String) -> Self {
        Self {
            client,
            base_uri: std::sync::RwLock::new(base_uri.trim_end_matches('/').to_string()),
        }
    }

    pub fn set_base_uri(&self, base_uri: String) {
        let mut guard = self.base_uri.write().unwrap_or_else(|e| e.into_inner());
        *guard = base_uri.trim_end_matches('/').to_string();
    }

    fn base_uri(&self) -> String {
        self.base_uri.read().unwrap_or_else(|e| e.into_inner()).clone()
    }

    pub async fn get_book(
        &self,
        book_id: &str,
        auth: &ForwardAuthHeaders,
    ) -> Result<KomgaBook, HttpError> {
        let url = format!("{}/api/v1/books/{}", self.base_uri(), book_id);
        let req = auth
            .apply_to(self.client.get(&url))
            .header(ACCEPT_ENCODING, "identity");
        let result = req.send().await.map_err(|err| HttpError::from_reqwest(&err))?;

        info!("GET {} {}", result.url(), result.status());
        if !result.status().is_success() {
            let status = result.status();
            let resp_url = result.url().clone();
            warn!("Komga GET {} returned {}", resp_url, status);
            return Err(HttpError::upstream_status(
                status,
                format!("{}, {}", status, resp_url),
            ));
        }

        result
            .json::<KomgaBook>()
            .await
            .map_err(|err| HttpError::from_reqwest(&err))
    }

    pub async fn get_series(
        &self,
        series_id: &str,
        auth: &ForwardAuthHeaders,
    ) -> Result<KomgaSeries, HttpError> {
        let url = format!("{}/api/v1/series/{}", self.base_uri(), series_id);
        let req = auth
            .apply_to(self.client.get(&url))
            .header(ACCEPT_ENCODING, "identity");
        let result = req.send().await.map_err(|err| HttpError::from_reqwest(&err))?;

        info!("GET {} {}", result.url(), result.status());
        if !result.status().is_success() {
            let status = result.status();
            let resp_url = result.url().clone();
            warn!("Komga GET {} returned {}", resp_url, status);
            return Err(HttpError::upstream_status(
                status,
                format!("{}, {}", status, resp_url),
            ));
        }

        result
            .json::<KomgaSeries>()
            .await
            .map_err(|err| HttpError::from_reqwest(&err))
    }

    pub async fn get_book_pages(
        &self,
        book_id: &str,
        auth: &ForwardAuthHeaders,
    ) -> Result<Vec<KomgaPageInfo>, HttpError> {
        let url = format!("{}/api/v1/books/{}/pages", self.base_uri(), book_id);
        let req = auth
            .apply_to(self.client.get(&url))
            .header(ACCEPT_ENCODING, "identity");
        let result = req.send().await.map_err(|err| HttpError::from_reqwest(&err))?;

        info!("GET {} {}", result.url(), result.status());
        if !result.status().is_success() {
            let status = result.status();
            let resp_url = result.url().clone();
            warn!("Komga GET {} returned {}", resp_url, status);
            return Err(HttpError::upstream_status(
                status,
                format!("{}, {}", status, resp_url),
            ));
        }

        result
            .json::<Vec<KomgaPageInfo>>()
            .await
            .map_err(|err| HttpError::from_reqwest(&err))
    }

    pub async fn get_page_image(
        &self,
        book_id: &str,
        page_number: usize,
        auth: &ForwardAuthHeaders,
    ) -> Result<Bytes, HttpError> {
        let url = format!(
            "{}/api/v1/books/{}/pages/{}",
            self.base_uri(),
            book_id,
            page_number
        );
        let req = auth
            .apply_to(self.client.get(&url))
            .header(ACCEPT_ENCODING, "identity");
        let result = req.send().await.map_err(|err| HttpError::from_reqwest(&err))?;

        info!("GET {} {}", result.url(), result.status());
        if !result.status().is_success() {
            let status = result.status();
            let resp_url = result.url().clone();
            warn!("Komga GET {} returned {}", resp_url, status);
            return Err(HttpError::upstream_status(
                status,
                format!("{}, {}", status, resp_url),
            ));
        }

        result
            .bytes()
            .await
            .map_err(|err| HttpError::from_reqwest(&err))
    }
}
