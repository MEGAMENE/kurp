use std::sync::{Arc, RwLock};
use std::time::Duration;

use moka::future::Cache;

use crate::clients::komga_client::{ForwardAuthHeaders, KomgaClient};
use crate::models::errors::HttpError;

pub struct UpscaleTagChecker {
    upscale_tag: RwLock<Option<String>>,
    komga: Arc<KomgaClient>,
    cache: Cache<String, bool>,
}

impl UpscaleTagChecker {
    pub fn new(upscale_tag: Option<String>, komga: Arc<KomgaClient>) -> Self {
        let cache = Cache::builder()
            .max_capacity(10_000)
            .time_to_idle(Duration::from_secs(30 * 60))
            .build();

        Self {
            upscale_tag: RwLock::new(upscale_tag),
            komga,
            cache,
        }
    }

    pub fn update_upscale_tag(&self, new_tag: Option<String>) {
        let mut guard = self.upscale_tag.write().unwrap_or_else(|e| e.into_inner());
        if *guard != new_tag {
            *guard = new_tag;
            self.cache.invalidate_all();
        }
    }

    pub async fn komga_contains_upscale_tag(
        &self,
        book_id: &str,
        auth: &ForwardAuthHeaders,
    ) -> Result<bool, HttpError> {
        let upscale_tag = {
            let guard = self.upscale_tag.read().unwrap_or_else(|e| e.into_inner());
            match guard.as_deref() {
                Some(t) if !t.is_empty() => t.to_string(),
                _ => return Ok(true),
            }
        };

        self.cache
            .try_get_with(book_id.to_string(), async move {
                self.check_komga_tags(&upscale_tag, book_id, auth).await
            })
            .await
            .map_err(|arc_err| (*arc_err).clone())
    }

    async fn check_komga_tags(
        &self,
        upscale_tag: &str,
        book_id: &str,
        auth: &ForwardAuthHeaders,
    ) -> Result<bool, HttpError> {
        let book = self.komga.get_book(book_id, auth).await?;
        if book
            .metadata
            .tags
            .iter()
            .any(|el| el.eq_ignore_ascii_case(upscale_tag))
        {
            return Ok(true);
        }

        let series = self.komga.get_series(&book.series_id, auth).await?;
        let contains_tag = series
            .metadata
            .tags
            .iter()
            .any(|el| el.eq_ignore_ascii_case(upscale_tag));

        Ok(contains_tag)
    }

    pub fn invalidate_cache(&self) {
        self.cache.invalidate_all();
    }

    pub async fn invalidate_book(&self, book_id: &str) {
        self.cache.invalidate(book_id).await;
    }
}
