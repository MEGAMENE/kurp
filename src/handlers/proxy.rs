use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};

use crate::app_state::AppState;

pub async fn proxy_handler(State(state): State<AppState>, req: Request) -> Response {
    match state.proxy_client.proxy_request(req).await {
        Ok(resp) => resp,
        Err(err) => {
            log::error!("Proxy error ({}): {}", err.status_code(), err.message);
            err.status_code().into_response()
        }
    }
}
