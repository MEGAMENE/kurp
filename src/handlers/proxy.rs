use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;

use crate::app_state::AppState;

pub async fn proxy_handler(
    State(state): State<AppState>,
    req: Request,
) -> impl IntoResponse {
    match state.proxy_client.proxy_request(req).await {
        Ok(resp) => resp.into_response(),
        Err(_) => StatusCode::BAD_GATEWAY.into_response()
    }
}

