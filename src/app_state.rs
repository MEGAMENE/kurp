use std::sync::Arc;

use moka::future::Cache;
use ractor::ActorRef;
use tokio::sync::broadcast::Sender;

use crate::clients::proxy_client::ProxyClient;
use crate::config::app_config::AppConfig;
use crate::tags_provider::UpscaleTagChecker;
use crate::upscaler::upscale_actor::UpscaleSupervisorMessage;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub upscaler: ActorRef<UpscaleSupervisorMessage>,
    pub proxy_client: Arc<ProxyClient>,
    pub upscale_call_history_cache: Arc<Cache<String, ()>>,
    pub upscale_tag_checker: Arc<UpscaleTagChecker>,
    pub shutdown_tx: Sender<()>,
}