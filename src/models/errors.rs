use std::fmt::{self, Display, Formatter};

use axum::http::StatusCode;

#[derive(Debug, Clone)]
pub struct ProxyError {
    pub status: StatusCode,
    pub message: String,
}

impl ProxyError {
    pub fn bad_gateway(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            message: message.into(),
        }
    }

    pub fn gateway_timeout(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::GATEWAY_TIMEOUT,
            message: message.into(),
        }
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }

    pub fn from_reqwest(err: &reqwest::Error) -> Self {
        if err.is_timeout() {
            Self::gateway_timeout(err.to_string())
        } else {
            Self::bad_gateway(err.to_string())
        }
    }

    pub fn status_code(&self) -> StatusCode {
        self.status
    }
}

#[derive(Debug, Clone)]
pub struct HttpError {
    pub status: Option<StatusCode>,
    pub message: String,
}

impl HttpError {
    pub fn upstream_status(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status: Some(status),
            message: message.into(),
        }
    }

    pub fn transport(message: impl Into<String>) -> Self {
        Self {
            status: None,
            message: message.into(),
        }
    }

    pub fn timeout(message: impl Into<String>) -> Self {
        Self {
            status: Some(StatusCode::GATEWAY_TIMEOUT),
            message: message.into(),
        }
    }

    pub fn from_reqwest(err: &reqwest::Error) -> Self {
        if err.is_timeout() {
            Self::timeout(err.to_string())
        } else {
            Self::transport(err.to_string())
        }
    }

    pub fn is_auth_error(&self) -> bool {
        matches!(
            self.status,
            Some(StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
        )
    }

    /// Preserves upstream 4xx client statuses (such as 401/403/404) while mapping
    /// upstream 5xx or transport errors to 502 Bad Gateway (or 504 on timeout).
    pub fn response_status(&self) -> StatusCode {
        match self.status {
            Some(StatusCode::GATEWAY_TIMEOUT) => StatusCode::GATEWAY_TIMEOUT,
            Some(status) if status.is_client_error() => status,
            Some(_) | None => StatusCode::BAD_GATEWAY,
        }
    }
}

impl Display for ProxyError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "{} ({})", self.message, self.status)
    }
}

impl Display for HttpError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self.status {
            Some(status) => write!(f, "{} ({})", self.message, status),
            None => write!(f, "{}", self.message),
        }
    }
}

impl std::error::Error for ProxyError {}
impl std::error::Error for HttpError {}
