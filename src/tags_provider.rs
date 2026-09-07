use std::sync::Arc;
use std::time::Duration;

use headers::{Authorization, Cookie};
use headers::authorization::Basic;
use moka::future::Cache;
use unicase::Ascii;

use crate::clients::komga_client::KomgaClient;
use crate::models::errors::HttpError;

pub struct UpscaleTagChecker {
    upscale_tag: Option<String>,
    komga: Arc<KomgaClient>,
    cache: Cache<String, bool>,
}

impl UpscaleTagChecker {
    pub fn new(upscale_tag: Option<String>, komga: Arc<KomgaClient>) -> Self {
        let cache = Cache::builder()
            .time_to_live(Duration::from_secs(3 * 60))
            .build();

        Self { upscale_tag, komga, cache }
    }

    pub async fn komga_contains_upscale_tag(
        &self,
        book_id: &str,
        cookie: Option<Cookie>,
        auth: Option<Authorization<Basic>>,
    ) -> Result<bool, HttpError> {
        match &self.upscale_tag {
            None => Ok(true),
            Some(upscale_tag) => {
                match self.cache.get(book_id).await {
                    None => Ok(self.check_komga_tags(upscale_tag, book_id, cookie, auth).await?),
                    Some(contains) => Ok(contains)
                }
            }
        }
    }

    async fn check_komga_tags(
        &self,
        upscale_tag: &String,
        book_id: &str,
        cookie: Option<Cookie>,
        auth: Option<Authorization<Basic>>,
    ) -> Result<bool, HttpError> {
        let upscale_tag = Ascii::new(upscale_tag);
        let book = self.komga.get_book(book_id, cookie.clone(), auth.clone()).await?;
        let series = self.komga.get_series(&book.series_id, cookie, auth).await?;
        let contains_tag = book.metadata.tags.iter().chain(series.metadata.tags.iter())
            .map(|el| Ascii::new(el))
            .any(|el| el == upscale_tag);

        self.cache.insert(book_id.to_string(), contains_tag).await;

        Ok(contains_tag)
    }

    pub fn invalidate_cache(&self) {
        self.cache.invalidate_all()
    }
}