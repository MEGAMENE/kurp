use std::sync::Arc;
use std::time::{Duration, Instant};

use image::{DynamicImage, ImageFormat};
use log::{info, warn};
use moka::future::Cache;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::clients::komga_client::{ForwardAuthHeaders, KomgaClient};
use crate::config::app_config::{AutoLevelsConfig, BookProfileConfig};
use crate::upscaler::auto_levels::{PageClassification, analyze_page};
use crate::upscaler::avif::{decode_image_with_metadata, probe_image_dimensions};

/// Safety ceiling on page dimensions decoded during lightweight book profiling.
const PROFILE_MAX_DECODE_DIMENSION: u32 = 16384;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ClassificationGroupStats {
    pub page_count: usize,
    pub median_black_floor: u8,
}

impl ClassificationGroupStats {
    pub fn from_summaries(summaries: &[PageAnalysisSummary]) -> Self {
        let page_count = summaries.len();
        if page_count == 0 {
            return Self::default();
        }

        let mut dist = [0u32; 256];
        for s in summaries {
            dist[s.black_floor as usize] += 1;
        }

        let target = (page_count / 2) as u32;
        let mut acc = 0u32;
        let mut median_black_floor = 255u8;
        for (bin, &freq) in dist.iter().enumerate() {
            acc += freq;
            if acc > target {
                median_black_floor = bin as u8;
                break;
            }
        }

        Self {
            page_count,
            median_black_floor,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageAnalysisSummary {
    pub page_number: usize,
    pub classification: PageClassification,
    pub black_floor: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BookProfile {
    pub book_id: String,
    pub total_book_pages: usize,
    pub sampled_pages: usize,
    pub monochrome_group: ClassificationGroupStats,
    pub mixed_group: ClassificationGroupStats,
    pub color_group: ClassificationGroupStats,
}

impl BookProfile {
    pub fn empty(book_id: &str) -> Self {
        Self {
            book_id: book_id.to_string(),
            total_book_pages: 0,
            sampled_pages: 0,
            monochrome_group: ClassificationGroupStats::default(),
            mixed_group: ClassificationGroupStats::default(),
            color_group: ClassificationGroupStats::default(),
        }
    }

    pub fn from_samples(
        book_id: &str,
        total_pages: usize,
        analyses: &[PageAnalysisSummary],
    ) -> Self {
        let mut mono = Vec::new();
        let mut mixed = Vec::new();
        let mut color = Vec::new();

        for &a in analyses {
            match a.classification {
                PageClassification::Monochrome => mono.push(a),
                PageClassification::Mixed => mixed.push(a),
                PageClassification::Color => color.push(a),
            }
        }

        Self {
            book_id: book_id.to_string(),
            total_book_pages: total_pages,
            sampled_pages: analyses.len(),
            monochrome_group: ClassificationGroupStats::from_summaries(&mono),
            mixed_group: ClassificationGroupStats::from_summaries(&mixed),
            color_group: ClassificationGroupStats::from_summaries(&color),
        }
    }
}

/// Selects up to `sample_count` evenly-spaced 0-based indices across `0..total_pages`.
pub fn select_sample_indices(total_pages: usize, sample_count: usize) -> Vec<usize> {
    if total_pages == 0 || sample_count == 0 {
        return Vec::new();
    }
    if total_pages <= sample_count {
        return (0..total_pages).collect();
    }
    if sample_count == 1 {
        return vec![total_pages / 2];
    }

    let mut indices = Vec::with_capacity(sample_count);
    let max_idx = total_pages - 1;
    let denom = sample_count - 1;
    for i in 0..sample_count {
        let idx = (i * max_idx) / denom;
        if indices.last().copied() != Some(idx) {
            indices.push(idx);
        }
    }
    indices
}

/// Runs lightweight per-page analysis using the live `AutoLevelsConfig`.
pub fn analyze_page_lightweight(
    image: &DynamicImage,
    page_number: usize,
    config: &AutoLevelsConfig,
) -> PageAnalysisSummary {
    let analysis = analyze_page(image, page_number, config);
    PageAnalysisSummary {
        page_number,
        classification: analysis.classification,
        black_floor: analysis.black_floor,
    }
}

fn format_from_media_type(media_type: &str) -> Option<ImageFormat> {
    let mime = media_type.split(';').next()?.trim();
    if mime.eq_ignore_ascii_case("image/png") {
        Some(ImageFormat::Png)
    } else if mime.eq_ignore_ascii_case("image/jpeg") || mime.eq_ignore_ascii_case("image/jpg") {
        Some(ImageFormat::Jpeg)
    } else if mime.eq_ignore_ascii_case("image/webp") {
        Some(ImageFormat::WebP)
    } else if mime.eq_ignore_ascii_case("image/gif") {
        Some(ImageFormat::Gif)
    } else if mime.eq_ignore_ascii_case("image/avif") {
        Some(ImageFormat::Avif)
    } else {
        None
    }
}

pub struct BookProfileManager {
    profile_cache: Cache<String, Arc<BookProfile>>,
    negative_cache: Cache<String, Instant>,
    komga_client: Arc<KomgaClient>,
    fetch_semaphore: Arc<Semaphore>,
    sample_pages: usize,
    timeout_seconds: u64,
}

impl BookProfileManager {
    pub fn new(komga_client: Arc<KomgaClient>, config: &BookProfileConfig) -> Self {
        let profile_cache = Cache::builder()
            .max_capacity(config.cache_size.max(1) as u64)
            .time_to_idle(Duration::from_secs(config.cache_ttl_seconds.max(60)))
            .build();

        let negative_cache = Cache::builder()
            .time_to_live(Duration::from_secs(60))
            .max_capacity(1000)
            .build();

        Self {
            profile_cache,
            negative_cache,
            komga_client,
            fetch_semaphore: Arc::new(Semaphore::new(config.parallel_page_fetches.max(1))),
            sample_pages: config.sample_pages.max(1),
            timeout_seconds: config.timeout_seconds.max(1),
        }
    }

    /// Synchronously resolves or builds a sampled `BookProfile` (sampling `config.sample_pages`
    /// evenly-spaced pages with `JoinSet` cancellation on timeout/drop and a shared global
    /// fetch semaphore). Auth errors (`401`/`403`) and empty page lists are never poisoned into
    /// shared caches across users.
    pub async fn get_or_build_profile(
        &self,
        book_id: &str,
        auth: &ForwardAuthHeaders,
        auto_levels_config: &AutoLevelsConfig,
    ) -> Arc<BookProfile> {
        if self.negative_cache.get(book_id).await.is_some() {
            return Arc::new(BookProfile::empty(book_id));
        }

        let book_id_owned = book_id.to_string();
        let komga_client = self.komga_client.clone();
        let fetch_sem = self.fetch_semaphore.clone();
        let sample_pages = self.sample_pages;
        let timeout_dur = Duration::from_secs(self.timeout_seconds);
        let neg_cache = self.negative_cache.clone();
        let auth_owned = auth.clone();
        let al_cfg = *auto_levels_config;

        let book_id_for_timeout = book_id_owned.clone();
        let neg_cache_for_timeout = neg_cache.clone();

        let build_fut = async move {
            let start_time = Instant::now();
            let page_list = match komga_client.get_book_pages(&book_id_owned, &auth_owned).await {
                Ok(pages) => pages,
                Err(e) => {
                    warn!(
                        "[BookProfile] Failed to fetch page list for {}: {}",
                        book_id_owned, e
                    );
                    if !e.is_auth_error() {
                        neg_cache.insert(book_id_owned.clone(), Instant::now()).await;
                    }
                    return Err(format!("Failed to fetch page list: {e}"));
                }
            };

            if page_list.is_empty() {
                warn!(
                    "[BookProfile] Book {} returned empty page list; skipping cache",
                    book_id_owned
                );
                return Err(format!("Empty page list for {book_id_owned}"));
            }

            let total_pages = page_list.len();
            let indices = select_sample_indices(total_pages, sample_pages);
            let mut join_set: JoinSet<Result<Option<PageAnalysisSummary>, bool>> = JoinSet::new();

            for idx in indices {
                let page_info = &page_list[idx];
                let sem = fetch_sem.clone();
                let client = komga_client.clone();
                let b_id = book_id_owned.clone();
                let p_num = page_info.number;
                let fallback_fmt = format_from_media_type(&page_info.media_type);
                let auth_cl = auth_owned.clone();

                join_set.spawn(async move {
                    let _permit = match sem.acquire_owned().await {
                        Ok(p) => p,
                        Err(_) => return Ok(None),
                    };

                    let image_bytes = match client.get_page_image(&b_id, p_num, &auth_cl).await {
                        Ok(b) => b,
                        Err(e) => {
                            warn!(
                                "[BookProfile] Failed to fetch page {} for {}: {}",
                                p_num, b_id, e
                            );
                            return Err(e.is_auth_error());
                        }
                    };

                    let summary = tokio::task::spawn_blocking(move || {
                        if let Ok((w, h, _)) = probe_image_dimensions(&image_bytes, fallback_fmt)
                            && (w > PROFILE_MAX_DECODE_DIMENSION || h > PROFILE_MAX_DECODE_DIMENSION)
                        {
                            warn!(
                                "[BookProfile] Skipping oversized page {} ({}x{}) for {}",
                                p_num, w, h, b_id
                            );
                            return None;
                        }

                        let decoded = match decode_image_with_metadata(&image_bytes, fallback_fmt) {
                            Ok(img) => img,
                            Err(e) => {
                                warn!(
                                    "[BookProfile] Failed to decode page {} for {}: {}",
                                    p_num, b_id, e
                                );
                                return None;
                            }
                        };
                        drop(image_bytes);

                        let summary = analyze_page_lightweight(&decoded.image, p_num, &al_cfg);
                        drop(decoded);
                        Some(summary)
                    })
                    .await
                    .ok()
                    .flatten();

                    Ok(summary)
                });
            }

            let mut analyses = Vec::with_capacity(sample_pages.min(total_pages));
            let mut saw_auth_error = false;

            while let Some(res) = join_set.join_next().await {
                match res {
                    Ok(Ok(Some(summary))) => analyses.push(summary),
                    Ok(Err(true)) => saw_auth_error = true,
                    _ => {}
                }
            }

            if analyses.is_empty() {
                warn!(
                    "[BookProfile] Failed to analyze any sampled pages for {}",
                    book_id_owned
                );
                if !saw_auth_error {
                    neg_cache.insert(book_id_owned.clone(), Instant::now()).await;
                }
                return Err(format!("Failed to analyze any pages for {book_id_owned}"));
            }

            let elapsed_ms = start_time.elapsed().as_millis() as u64;
            let profile = BookProfile::from_samples(&book_id_owned, total_pages, &analyses);
            info!(
                "[BookProfile] book_id={} total_pages={} sampled={} build_time_ms={} mono={} mixed={} color={}",
                book_id_owned,
                total_pages,
                profile.sampled_pages,
                elapsed_ms,
                profile.monochrome_group.page_count,
                profile.mixed_group.page_count,
                profile.color_group.page_count
            );

            Ok(Arc::new(profile))
        };

        let timed_build_fut = async move {
            match tokio::time::timeout(timeout_dur, build_fut).await {
                Ok(res) => res,
                Err(_) => {
                    warn!(
                        "[BookProfile] Timeout profiling {} (all in-flight sample tasks aborted)",
                        book_id_for_timeout
                    );
                    neg_cache_for_timeout
                        .insert(book_id_for_timeout.clone(), Instant::now())
                        .await;
                    Err(format!("Timeout profiling book {book_id_for_timeout}"))
                }
            }
        };

        match self
            .profile_cache
            .try_get_with::<_, String>(book_id.to_string(), timed_build_fut)
            .await
        {
            Ok(profile) => profile,
            Err(e) => {
                warn!(
                    "[BookProfile] Profile building failed for {}: {}",
                    book_id, e
                );
                Arc::new(BookProfile::empty(book_id))
            }
        }
    }

    pub async fn invalidate(&self, book_id: &str) {
        self.profile_cache.invalidate(book_id).await;
        self.negative_cache.invalidate(book_id).await;
    }

    pub fn invalidate_all(&self) {
        self.profile_cache.invalidate_all();
        self.negative_cache.invalidate_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_sample_indices_evenly_spaced() {
        assert_eq!(select_sample_indices(0, 12), Vec::<usize>::new());
        assert_eq!(select_sample_indices(5, 12), vec![0, 1, 2, 3, 4]);
        assert_eq!(select_sample_indices(12, 12), (0..12).collect::<Vec<_>>());

        let sampled = select_sample_indices(200, 12);
        assert_eq!(sampled.len(), 12);
        assert_eq!(sampled[0], 0);
        assert_eq!(*sampled.last().unwrap(), 199);
        assert!(sampled.windows(2).all(|w| w[0] < w[1]));

        assert_eq!(select_sample_indices(50, 1), vec![25]);
    }

    #[test]
    fn test_classification_group_stats_median() {
        let summaries = vec![
            PageAnalysisSummary {
                page_number: 1,
                classification: PageClassification::Monochrome,
                black_floor: 10,
            },
            PageAnalysisSummary {
                page_number: 2,
                classification: PageClassification::Monochrome,
                black_floor: 24,
            },
            PageAnalysisSummary {
                page_number: 3,
                classification: PageClassification::Monochrome,
                black_floor: 18,
            },
        ];
        let stats = ClassificationGroupStats::from_summaries(&summaries);
        assert_eq!(stats.page_count, 3);
        assert_eq!(stats.median_black_floor, 18);
    }
}
