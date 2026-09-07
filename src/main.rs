use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;
use axum::http::Uri;
use log::LevelFilter;
use moka::future::Cache;
use ractor::Actor;
use reqwest::redirect::Policy;
use tokio::sync::broadcast;

use crate::app_state::AppState;
use crate::clients::komga_client::KomgaClient;
use crate::clients::proxy_client::ProxyClient;
use crate::config::app_config::AppConfig;
use crate::tags_provider::UpscaleTagChecker;
use crate::upscaler::upscale_actor::{UpscaleSupervisorActor, UpscaleSupervisorMessage};

mod config;
mod upscaler;
mod http_compression;
mod models;
mod handlers;
mod clients;
mod tags_provider;
mod app_state;
mod server;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

#[tokio::main]
async fn main() {
    env_logger::Builder::new()
        .filter_level(LevelFilter::Info)
        .filter_module("ractor", LevelFilter::Warn)
        .parse_default_env()
        .init();

    let (upscale_actor, _) = Actor::spawn(None, UpscaleSupervisorActor, ())
        .await
        .expect("Failed to start Upscale Actor!");

    let mut current_config = match AppConfig::new() {
        Ok(cfg) => Arc::new(cfg),
        Err(e) => {
            log::error!("Failed to load configuration: {}. Exiting.", e);
            return;
        }
    };

    loop {
        upscale_actor.send_message(UpscaleSupervisorMessage::Destroy)
            .expect("Failed to send Upscaler Destroy message");

        match AppConfig::new() {
            Ok(cfg) => {
                current_config = Arc::new(cfg);
            }
            Err(e) => {
                log::error!("Failed to reload config: {}. Retaining previous configuration.", e);
            }
        }
        let config = current_config.clone();
        upscale_actor.send_message(UpscaleSupervisorMessage::Init(config.clone()))
            .expect("Failed to send Upscaler Init message");

        let (tx, graceful_shutdown_rx) = broadcast::channel::<()>(10);

        let reqwest_client = reqwest::Client::builder()
            .redirect(Policy::none())
            .tcp_nodelay(true)
            .pool_max_idle_per_host(32)
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(60))
            .build()
            .expect("Reqwest client couldn't build");

        let upstream_url = match Uri::from_str(config.upstream_url.as_str()) {
            Ok(u) => u,
            Err(e) => {
                log::error!("Invalid upstream URL '{}': {}. Falling back to http://localhost:8080", config.upstream_url, e);
                Uri::from_static("http://localhost:8080")
            }
        };
        let upstream_url_raw = upstream_url.to_string();
        let upstream_url_str = upstream_url_raw.strip_suffix('/').unwrap_or(&upstream_url_raw).to_string();
        let komga_client = Arc::new(KomgaClient::new(reqwest_client.clone(), upstream_url_str.clone()));

        let tag_provider = Arc::new(UpscaleTagChecker::new(
            config.upscale_tag.clone(),
            komga_client,
        ));

        let upscale_call_cache = Cache::new(1_000);
        let proxy_client = ProxyClient::new(reqwest_client, upstream_url_str);

        let state = AppState {
            config,
            upscaler: upscale_actor.clone(),
            proxy_client: Arc::new(proxy_client),
            upscale_call_history_cache: Arc::new(upscale_call_cache),
            upscale_tag_checker: tag_provider,
            shutdown_tx: tx,
        };
        server::start(state, graceful_shutdown_rx).await;
    }
}