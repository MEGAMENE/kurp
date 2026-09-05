use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::Response;

use crate::app_state::AppState;
use crate::models::komga::{KomgaBookMetadataUpdate, KomgaSeriesMetadataUpdate};

pub async fn check_tags_on_series_metadata_update(
    State(state): State<AppState>,
    request: Request,
) -> Result<Response, StatusCode> {
    let (parts, body) = request.into_parts();
    let bytes = to_bytes(body, usize::MAX).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let json: KomgaSeriesMetadataUpdate = serde_json::from_slice(&bytes)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let request = Request::from_parts(parts, Body::from(bytes));
    let response = state.proxy_client.proxy_request(request).await.map_err(|_| StatusCode::BAD_GATEWAY)?;

    if let Some(_) = json.tags {
        state.upscale_tag_checker.invalidate_cache();
        state.upscale_call_history_cache.invalidate_all();
    }

    Ok(response)
}

pub async fn check_tags_on_book_metadata_update(
    State(state): State<AppState>,
    request: Request,
) -> Result<Response, StatusCode> {
    let (parts, body) = request.into_parts();
    let bytes = to_bytes(body, usize::MAX).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let json: KomgaBookMetadataUpdate = serde_json::from_slice(&bytes)
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    let request = Request::from_parts(parts, Body::from(bytes));
    let response = state.proxy_client.proxy_request(request).await.map_err(|_| StatusCode::BAD_GATEWAY)?;

    if let Some(_) = json.tags {
        state.upscale_tag_checker.invalidate_cache();
        state.upscale_call_history_cache.invalidate_all();
    }

    Ok(response)
}
