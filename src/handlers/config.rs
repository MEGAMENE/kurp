use std::ops::Deref;

use axum::extract::State;
use axum::Json;
use axum::response::IntoResponse;
use axum::http::StatusCode;

use crate::app_state::AppState;
use crate::config::app_config::AppConfig;

pub async fn update_config(
    State(state): State<AppState>,
    Json(new_config): Json<AppConfig>,
) -> impl IntoResponse {
    if let Err(e) = AppConfig::write_config(new_config) {
        log::error!("Failed to write configuration file: {}", e);
        return StatusCode::INTERNAL_SERVER_ERROR;
    }
    if let Err(e) = state.shutdown_tx.send(()) {
        log::warn!("Failed to signal server reload: {}", e);
    }

    StatusCode::OK
}

pub async fn get_config(
    State(state): State<AppState>
) -> impl IntoResponse {
    Json(state.config.deref().clone())
}
