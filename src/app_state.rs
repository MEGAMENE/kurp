use std::sync::Arc;

use arc_swap::ArcSwap;
use tokio::sync::Semaphore;
use tokio::sync::broadcast::Sender;

use crate::clients::komga_client::KomgaClient;
use crate::clients::proxy_client::ProxyClient;
use crate::config::app_config::AppConfig;
use crate::tags_provider::UpscaleTagChecker;
use crate::upscaler::book_profile::BookProfileManager;
use crate::upscaler::upscaler::UpscaleEngine;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ArcSwap<AppConfig>>,
    pub upscaler: Arc<UpscaleEngine>,
    pub komga_client: Arc<KomgaClient>,
    pub proxy_client: Arc<ProxyClient>,
    pub upscale_tag_checker: Arc<UpscaleTagChecker>,
    pub book_profile_manager: Arc<BookProfileManager>,
    pub pipeline_semaphore: Arc<Semaphore>,
    pub shutdown_tx: Sender<()>,
}
