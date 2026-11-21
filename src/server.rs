use std::net::SocketAddr;
use std::time::Duration;

use axum::Router;
use axum::routing::{any, get, patch};
use log::{error, info};
use tokio::sync::broadcast::Receiver;
use tokio::time::sleep;

use crate::app_state::AppState;
use crate::handlers::config::{get_config, update_config};
use crate::handlers::komga::{
    check_tags_on_book_metadata_update, check_tags_on_bulk_books_metadata_update,
    check_tags_on_series_metadata_update,
};
use crate::handlers::proxy::proxy_handler;
use crate::handlers::upscale::upscale_komga;

pub async fn start(state: AppState, mut shutdown_rx: Receiver<()>) -> std::io::Result<()> {
    let port = state.config.load().port;
    let mut force_shutdown_rx = state.shutdown_tx.subscribe();

    let routes = make_routes(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(err) => {
            error!("Failed to bind HTTP listener on {}: {}", addr, err);
            return Err(err);
        }
    };
    info!("Kurp HTTP proxy listening on {}", addr);

    let server = axum::serve(
        listener,
        routes.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        let _ = shutdown_rx.recv().await;
    });

    let force_shutdown = async move {
        let _ = force_shutdown_rx.recv().await;
        sleep(Duration::from_secs(5)).await;
    };

    tokio::select! {
        res = server => {
            if let Err(err) = res {
                error!("HTTP server exited with error: {}", err);
                return Err(err);
            }
            info!("Graceful server shutdown");
        }
        _ = force_shutdown => {
            info!("Forced server shutdown after drain timeout");
        }
    }

    Ok(())
}

pub fn make_routes(state: AppState) -> Router {
    let allow_config_updates = state.config.load().allow_config_updates;

    let mut routes = Router::new()
        .route(
            "/api/v1/books/{book_id}/pages/{page_number}",
            get(upscale_komga)
                .head(proxy_handler)
                .fallback(proxy_handler),
        )
        .route(
            "/opds/v1.2/books/{book_id}/pages/{page_number}",
            get(upscale_komga)
                .head(proxy_handler)
                .fallback(proxy_handler),
        )
        .route(
            "/api/v1/series/{series_id}/metadata",
            patch(check_tags_on_series_metadata_update).fallback(proxy_handler),
        )
        .route(
            "/api/v1/books/{book_id}/metadata",
            patch(check_tags_on_book_metadata_update).fallback(proxy_handler),
        )
        .route(
            "/api/v1/books/metadata",
            patch(check_tags_on_bulk_books_metadata_update).fallback(proxy_handler),
        )
        .route("/", any(proxy_handler))
        .route("/{*any}", any(proxy_handler));

    if allow_config_updates {
        routes = routes.route("/kurp/config", get(get_config).post(update_config));
    }

    routes.with_state(state)
}
