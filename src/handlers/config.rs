use std::str::FromStr;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::header::{AUTHORIZATION, WWW_AUTHENTICATE};
use axum::http::{HeaderMap, HeaderName, StatusCode, Uri};
use axum::response::{IntoResponse, Response};

use crate::app_state::AppState;
use crate::config::app_config::AppConfig;

static X_KURP_TOKEN: HeaderName = HeaderName::from_static("x-kurp-token");

pub async fn update_config(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(new_config): Json<AppConfig>,
) -> Response {
    let current_cfg = state.config.load_full();
    if !current_cfg.allow_config_updates {
        return StatusCode::NOT_FOUND.into_response();
    }

    if !authorize_config_request(&headers, current_cfg.config_api_token.as_deref()) {
        return unauthorized_response();
    }

    if let Err(reason) = new_config.validate() {
        log::warn!("Rejected invalid configuration update: {}", reason);
        return (StatusCode::BAD_REQUEST, reason).into_response();
    }

    if let Err(e) = AppConfig::update_config(&new_config) {
        log::error!("Failed to persist configuration file: {}", e);
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    }

    if let Err(e) = state.upscaler.reload(&new_config).await {
        log::error!("Failed to hot-reload Real-CUGAN engine: {}", e);
        if let Err(revert_err) = AppConfig::update_config(&current_cfg) {
            log::error!(
                "Failed to revert config.yml after reload failure: {}",
                revert_err
            );
        }
        return (StatusCode::INTERNAL_SERVER_ERROR, e).into_response();
    }

    state.config.store(Arc::new(new_config.clone()));
    state
        .upscale_tag_checker
        .update_upscale_tag(new_config.upscale_tag.clone());

    if new_config.auto_levels != current_cfg.auto_levels {
        state.book_profile_manager.invalidate_all();
    }

    if new_config.upstream_url != current_cfg.upstream_url
        && let Ok(uri) = Uri::from_str(new_config.upstream_url.trim())
    {
        let raw = uri.to_string();
        let clean = raw.strip_suffix('/').unwrap_or(&raw).to_string();
        state.komga_client.set_base_uri(clean.clone());
        state.proxy_client.set_base_url(clean);
        state.upscale_tag_checker.invalidate_cache();
        state.book_profile_manager.invalidate_all();
    }

    if new_config.port != current_cfg.port
        && let Err(e) = state.shutdown_tx.send(())
    {
        log::warn!("Failed to signal HTTP listener restart on port change: {}", e);
    }

    StatusCode::OK.into_response()
}

pub async fn get_config(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let current_cfg = state.config.load_full();
    if !current_cfg.allow_config_updates {
        return StatusCode::NOT_FOUND.into_response();
    }

    if !authorize_config_request(&headers, current_cfg.config_api_token.as_deref()) {
        return unauthorized_response();
    }

    Json((*current_cfg).clone()).into_response()
}

fn unauthorized_response() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        [(WWW_AUTHENTICATE, "Bearer realm=\"kurp-config\"")],
        "Unauthorized",
    )
        .into_response()
}

fn authorize_config_request(headers: &HeaderMap, configured_token: Option<&str>) -> bool {
    let Some(expected) = configured_token.filter(|t| !t.is_empty()) else {
        return true;
    };

    if let Some(token) = headers
        .get(&X_KURP_TOKEN)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        && constant_time_eq(token.as_bytes(), expected.as_bytes())
    {
        return true;
    }

    if let Some(auth_val) = headers.get(AUTHORIZATION).and_then(|v| v.to_str().ok()) {
        let trimmed = auth_val.trim();
        if trimmed.len() > 7 && trimmed[..7].eq_ignore_ascii_case("bearer ") {
            let token = trimmed[7..].trim();
            if constant_time_eq(token.as_bytes(), expected.as_bytes()) {
                return true;
            }
        }
    }

    false
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let max_len = a.len().max(b.len());
    let mut diff = a.len() ^ b.len();
    for i in 0..max_len {
        let x = *a.get(i).unwrap_or(&0);
        let y = *b.get(i).unwrap_or(&0);
        diff |= usize::from(x ^ y);
    }
    diff == 0
}
