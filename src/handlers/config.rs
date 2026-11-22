use std::str::FromStr;
use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use axum::http::{StatusCode, Uri};
use axum::response::{IntoResponse, Response};

use crate::app_state::AppState;
use crate::config::app_config::AppConfig;

pub async fn update_config(
    State(state): State<AppState>,
    Json(new_config): Json<AppConfig>,
) -> Response {
    let current_cfg = state.config.load_full();
    if !current_cfg.allow_config_updates {
        return StatusCode::NOT_FOUND.into_response();
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

pub async fn get_config(State(state): State<AppState>) -> Response {
    let current_cfg = state.config.load_full();
    if !current_cfg.allow_config_updates {
        return StatusCode::NOT_FOUND.into_response();
    }

    Json((*current_cfg).clone()).into_response()
}
