use axum::body::{to_bytes, Body};
use axum::extract::{Path, Request, State};
use axum::http::StatusCode;
use axum::response::Response;

use crate::app_state::AppState;

const MAX_METADATA_BODY_BYTES: usize = 16 * 1024 * 1024;

pub async fn check_tags_on_series_metadata_update(
    State(state): State<AppState>,
    request: Request,
) -> Result<Response, StatusCode> {
    let (parts, body) = request.into_parts();
    let bytes = to_bytes(body, MAX_METADATA_BODY_BYTES).await.map_err(|err| {
        log::warn!("Failed to read series metadata PATCH body: {}", err);
        if err.to_string().contains("length limit") {
            StatusCode::PAYLOAD_TOO_LARGE
        } else {
            StatusCode::BAD_REQUEST
        }
    })?;

    // Best-effort JSON inspection: never fail malformed/compressed bodies locally;
    // proxy verbatim to Komga and inspect `"tags"` presence (even `"tags": null`).
    let touches_tags = inspect_single_metadata_tags(&bytes);

    let request = Request::from_parts(parts, Body::from(bytes));
    let response = state
        .proxy_client
        .proxy_request(request)
        .await
        .map_err(|err| {
            log::error!("Series metadata PATCH proxy error: {}", err);
            err.status_code()
        })?;

    if response.status().is_success() && touches_tags {
        state.upscale_tag_checker.invalidate_cache();
    }

    Ok(response)
}

pub async fn check_tags_on_book_metadata_update(
    State(state): State<AppState>,
    Path(book_id): Path<String>,
    request: Request,
) -> Result<Response, StatusCode> {
    let (parts, body) = request.into_parts();
    let bytes = to_bytes(body, MAX_METADATA_BODY_BYTES).await.map_err(|err| {
        log::warn!("Failed to read book metadata PATCH body: {}", err);
        if err.to_string().contains("length limit") {
            StatusCode::PAYLOAD_TOO_LARGE
        } else {
            StatusCode::BAD_REQUEST
        }
    })?;

    let touches_tags = inspect_single_metadata_tags(&bytes);

    let request = Request::from_parts(parts, Body::from(bytes));
    let response = state
        .proxy_client
        .proxy_request(request)
        .await
        .map_err(|err| {
            log::error!("Book metadata PATCH proxy error: {}", err);
            err.status_code()
        })?;

    if response.status().is_success() && touches_tags {
        state.upscale_tag_checker.invalidate_book(&book_id).await;
        state.book_profile_manager.invalidate(&book_id).await;
    }

    Ok(response)
}

pub async fn check_tags_on_bulk_books_metadata_update(
    State(state): State<AppState>,
    request: Request,
) -> Result<Response, StatusCode> {
    let (parts, body) = request.into_parts();
    let bytes = to_bytes(body, MAX_METADATA_BODY_BYTES).await.map_err(|err| {
        log::warn!("Failed to read bulk book metadata PATCH body: {}", err);
        if err.to_string().contains("length limit") {
            StatusCode::PAYLOAD_TOO_LARGE
        } else {
            StatusCode::BAD_REQUEST
        }
    })?;

    let book_ids = inspect_bulk_metadata_tags(&bytes);

    let request = Request::from_parts(parts, Body::from(bytes));
    let response = state
        .proxy_client
        .proxy_request(request)
        .await
        .map_err(|err| {
            log::error!("Bulk book metadata PATCH proxy error: {}", err);
            err.status_code()
        })?;

    if response.status().is_success() && !book_ids.is_empty() {
        for book_id in book_ids {
            state.upscale_tag_checker.invalidate_book(&book_id).await;
            state.book_profile_manager.invalidate(&book_id).await;
        }
    }

    Ok(response)
}

fn inspect_single_metadata_tags(bytes: &[u8]) -> bool {
    let Ok(serde_json::Value::Object(obj)) = serde_json::from_slice::<serde_json::Value>(bytes)
    else {
        return false;
    };
    obj.contains_key("tags")
}

fn inspect_bulk_metadata_tags(bytes: &[u8]) -> Vec<String> {
    let Ok(serde_json::Value::Object(map)) = serde_json::from_slice::<serde_json::Value>(bytes)
    else {
        return Vec::new();
    };
    map.into_iter()
        .filter_map(|(book_id, entry)| {
            let serde_json::Value::Object(fields) = entry else {
                return None;
            };
            fields.contains_key("tags").then_some(book_id)
        })
        .collect()
}
