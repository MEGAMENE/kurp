use std::process::ExitCode;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use arc_swap::ArcSwap;
use axum::http::Uri;
use log::{LevelFilter, error, info};
use reqwest::redirect::Policy;
use tokio::sync::{Semaphore, broadcast};

use kurp::app_state::AppState;
use kurp::clients::komga_client::KomgaClient;
use kurp::clients::proxy_client::ProxyClient;
use kurp::config::app_config::AppConfig;
use kurp::server;
use kurp::tags_provider::UpscaleTagChecker;
use kurp::upscaler::book_profile::BookProfileManager;
use kurp::upscaler::upscaler::UpscaleEngine;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

async fn wait_for_os_shutdown() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sig) => {
                let _ = sig.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };

    #[cfg(windows)]
    let terminate = async {
        match tokio::signal::windows::ctrl_close() {
            Ok(mut sig) => {
                let _ = sig.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };

    #[cfg(not(any(unix, windows)))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            info!("Received Ctrl+C / SIGINT; initiating graceful shutdown");
        }
        _ = terminate => {
            info!("Received OS termination signal; initiating graceful shutdown");
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    env_logger::Builder::new()
        .filter_level(LevelFilter::Info)
        .parse_default_env()
        .init();

    // Load and validate configuration once at startup.
    let initial_config = match AppConfig::new() {
        Ok(cfg) => cfg,
        Err(e) => {
            error!("Failed to load configuration: {e}. Exiting.");
            return ExitCode::FAILURE;
        }
    };

    if let Err(e) = initial_config.validate() {
        error!("Invalid configuration: {e}. Exiting.");
        return ExitCode::FAILURE;
    }

    // Configure Rayon only when explicitly overridden; otherwise use hardware-agnostic default.
    if let Some(threads) = initial_config.rayon_threads {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
        {
            error!("Failed to configure Rayon thread pool ({threads} threads): {e}");
        } else {
            info!("Configured Rayon thread pool with {threads} threads");
        }
    }

    let upscaler = match UpscaleEngine::init(&initial_config).await {
        Ok(engine) => Arc::new(engine),
        Err(e) => {
            error!("Failed to initialize Real-CUGAN upscaler: {e}. Exiting.");
            return ExitCode::FAILURE;
        }
    };

    let shared_config = Arc::new(ArcSwap::from_pointee(initial_config));
    let os_exit_flag = Arc::new(AtomicBool::new(false));
    let (os_shutdown_tx, _) = broadcast::channel::<()>(4);

    {
        let os_flag = os_exit_flag.clone();
        let os_tx = os_shutdown_tx.clone();
        tokio::spawn(async move {
            wait_for_os_shutdown().await;
            os_flag.store(true, Ordering::SeqCst);
            let _ = os_tx.send(());
        });
    }

    let mut exit_code = ExitCode::SUCCESS;

    loop {
        if os_exit_flag.load(Ordering::SeqCst) {
            break;
        }

        let cfg = shared_config.load_full();
        let reqwest_client = match reqwest::Client::builder()
            .redirect(Policy::none())
            .connect_timeout(Duration::from_secs(10))
            .tcp_nodelay(true)
            .pool_max_idle_per_host(32)
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(60))
            .build()
        {
            Ok(client) => client,
            Err(e) => {
                error!("Failed to build HTTP client: {e}");
                exit_code = ExitCode::FAILURE;
                break;
            }
        };

        let upstream_url = match Uri::from_str(cfg.upstream_url.trim()) {
            Ok(u) => u,
            Err(e) => {
                error!("Invalid upstream URL '{}': {e}", cfg.upstream_url);
                exit_code = ExitCode::FAILURE;
                break;
            }
        };
        let upstream_url_raw = upstream_url.to_string();
        let upstream_url_str = upstream_url_raw
            .strip_suffix('/')
            .unwrap_or(&upstream_url_raw)
            .to_string();

        let komga_client = Arc::new(KomgaClient::new(
            reqwest_client.clone(),
            upstream_url_str.clone(),
        ));
        let tag_provider = Arc::new(UpscaleTagChecker::new(
            cfg.upscale_tag.clone(),
            komga_client.clone(),
        ));
        let proxy_client = Arc::new(ProxyClient::new(reqwest_client, upstream_url_str));
        let book_profile_manager = Arc::new(BookProfileManager::new(
            komga_client.clone(),
            &cfg.auto_levels.book_profile,
        ));
        let pipeline_semaphore = Arc::new(Semaphore::new(cfg.max_concurrent_pipelines.max(1)));

        let (reload_tx, mut reload_rx) = broadcast::channel::<()>(8);
        let (server_shutdown_tx, server_shutdown_rx) = broadcast::channel::<()>(4);

        let mut os_rx = os_shutdown_tx.subscribe();
        let forward_tx = server_shutdown_tx.clone();
        let reload_tx_for_os = reload_tx.clone();
        let forwarder = tokio::spawn(async move {
            tokio::select! {
                _ = os_rx.recv() => {
                    let _ = forward_tx.send(());
                    let _ = reload_tx_for_os.send(());
                }
                _ = reload_rx.recv() => {
                    let _ = forward_tx.send(());
                }
            }
        });

        let state = AppState {
            config: shared_config.clone(),
            upscaler: upscaler.clone(),
            komga_client,
            proxy_client,
            upscale_tag_checker: tag_provider,
            book_profile_manager,
            pipeline_semaphore,
            shutdown_tx: reload_tx,
        };

        let server_result = server::start(state, server_shutdown_rx).await;
        forwarder.abort();

        if let Err(e) = server_result {
            error!("HTTP server failed on port {}: {e}", cfg.port);
            exit_code = ExitCode::FAILURE;
            break;
        }

        if os_exit_flag.load(Ordering::SeqCst) {
            break;
        }

        info!("Listener restarted for updated configuration");
    }

    // Explicitly drop Real-CUGAN handles before returning from `main()` so NCNN's C `atexit`
    // Vulkan teardown never races with a live handle.
    upscaler.shutdown();
    exit_code
}
