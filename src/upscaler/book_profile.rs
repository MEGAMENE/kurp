use std::sync::Arc;
use std::time::{Duration, Instant};
use headers::{Authorization, Cookie};
use headers::authorization::Basic;
use image::{DynamicImage, GenericImageView};
use log::{info, warn};
use moka::future::Cache;
use tokio::sync::Semaphore;

use crate::clients::komga_client::KomgaClient;
use crate::upscaler::auto_levels::PageClassification;

#[derive(Debug, Clone)]
pub struct ClassificationGroupStats {
    pub page_count: usize,
    pub page_indices: Vec<usize>,
    pub black_floor_distribution: [u32; 256],
    pub white_point_distribution: [u32; 256],
    pub paper_cast_detected_count: usize,
    pub noise_floor_values: Vec<f32>,
    pub median_black_floor: u8,
    pub median_white_point: u8,
    pub median_dynamic_range: u8,
}

impl Default for ClassificationGroupStats {
    fn default() -> Self {
        Self {
            page_count: 0,
            page_indices: Vec::new(),
            black_floor_distribution: [0u32; 256],
            white_point_distribution: [0u32; 256],
            paper_cast_detected_count: 0,
            noise_floor_values: Vec::new(),
            median_black_floor: 0,
            median_white_point: 255,
            median_dynamic_range: 255,
        }
    }
}

impl ClassificationGroupStats {
    pub fn from_summaries(summaries: &[PageAnalysisSummary]) -> Self {
        let page_count = summaries.len();
        if page_count == 0 {
            return Self::default();
        }

        let mut page_indices = Vec::with_capacity(page_count);
        let mut black_floor_distribution = [0u32; 256];
        let mut white_point_distribution = [0u32; 256];
        let mut paper_cast_detected_count = 0;
        let mut noise_floor_values = Vec::new();
        let mut black_floors = Vec::with_capacity(page_count);
        let mut white_points = Vec::with_capacity(page_count);
        let mut dynamic_ranges = Vec::with_capacity(page_count);

        for s in summaries {
            page_indices.push(s.page_number);
            black_floor_distribution[s.black_floor as usize] += 1;
            white_point_distribution[s.white_ceiling as usize] += 1;
            if s.paper_cast_detected {
                paper_cast_detected_count += 1;
            }
            if let Some(n) = s.noise_floor {
                noise_floor_values.push(n);
            }
            black_floors.push(s.black_floor);
            white_points.push(s.white_ceiling);
            dynamic_ranges.push(s.dynamic_range);
        }

        black_floors.sort_unstable();
        white_points.sort_unstable();
        dynamic_ranges.sort_unstable();

        let median_black_floor = black_floors[page_count / 2];
        let median_white_point = white_points[page_count / 2];
        let median_dynamic_range = dynamic_ranges[page_count / 2];

        Self {
            page_count,
            page_indices,
            black_floor_distribution,
            white_point_distribution,
            paper_cast_detected_count,
            noise_floor_values,
            median_black_floor,
            median_white_point,
            median_dynamic_range,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PageAnalysisSummary {
    pub page_number: usize,
    pub classification: PageClassification,
    pub black_floor: u8,
    pub white_ceiling: u8,
    pub dynamic_range: u8,
    pub paper_cast_detected: bool,
    pub noise_floor: Option<f32>,
}

#[derive(Debug, Clone)]
pub struct BookProfile {
    pub book_id: String,
    pub total_book_pages: usize,
    pub monochrome_group: ClassificationGroupStats,
    pub mixed_group: ClassificationGroupStats,
    pub color_group: ClassificationGroupStats,
    pub page_analyses: Vec<PageAnalysisSummary>,
    pub build_time_ms: u64,
}

impl BookProfile {
    pub fn empty(book_id: &str) -> Self {
        Self {
            book_id: book_id.to_string(),
            total_book_pages: 0,
            monochrome_group: ClassificationGroupStats::default(),
            mixed_group: ClassificationGroupStats::default(),
            color_group: ClassificationGroupStats::default(),
            page_analyses: Vec::new(),
            build_time_ms: 0,
        }
    }

    pub fn from_all_pages(
        book_id: &str,
        total_pages: usize,
        analyses: Vec<PageAnalysisSummary>,
        build_time_ms: u64,
    ) -> Self {
        let mut mono = Vec::new();
        let mut mixed = Vec::new();
        let mut color = Vec::new();

        for a in &analyses {
            match a.classification {
                PageClassification::Monochrome => mono.push(a.clone()),
                PageClassification::Mixed => mixed.push(a.clone()),
                PageClassification::Color => color.push(a.clone()),
            }
        }

        let profile = Self {
            book_id: book_id.to_string(),
            total_book_pages: total_pages,
            monochrome_group: ClassificationGroupStats::from_summaries(&mono),
            mixed_group: ClassificationGroupStats::from_summaries(&mixed),
            color_group: ClassificationGroupStats::from_summaries(&color),
            page_analyses: analyses,
            build_time_ms,
        };

        info!(
            "[BookProfile] book_id={} total_pages={} build_time_ms={} mono_pages={} mixed_pages={} color_pages={}",
            book_id, total_pages, build_time_ms,
            profile.monochrome_group.page_count,
            profile.mixed_group.page_count,
            profile.color_group.page_count
        );

        profile
    }
}

pub fn analyze_page_lightweight(image: &DynamicImage, page_number: usize) -> PageAnalysisSummary {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return PageAnalysisSummary {
            page_number,
            classification: PageClassification::Color,
            black_floor: 0,
            white_ceiling: 255,
            dynamic_range: 255,
            paper_cast_detected: false,
            noise_floor: None,
        };
    }

    let is_already_luma = matches!(
        image,
        DynamicImage::ImageLuma8(_) | DynamicImage::ImageLumaA8(_)
    );

    let mut hist_y = [0u32; 256];
    let mut hist_y_neutral = [0u32; 256];
    let mut neutral_count: usize = 0;
    let mut total_valid_pixels: usize = 0;
    let mut max_chroma: u8 = 0;
    let mut high_chroma_count: usize = 0;

    let mut highlight_count: usize = 0;
    let mut sum_r_high: u64 = 0;
    let mut _sum_g_high: u64 = 0;
    let mut sum_b_high: u64 = 0;

    match image {
        DynamicImage::ImageLuma8(luma) => {
            for &p in luma.as_raw() {
                hist_y[p as usize] += 1;
                hist_y_neutral[p as usize] += 1;
            }
            total_valid_pixels = luma.len();
            neutral_count = total_valid_pixels;
        }
        DynamicImage::ImageLumaA8(luma_a) => {
            for chunk in luma_a.as_raw().chunks_exact(2) {
                if chunk[1] >= 128 {
                    let p = chunk[0];
                    hist_y[p as usize] += 1;
                    hist_y_neutral[p as usize] += 1;
                    total_valid_pixels += 1;
                    neutral_count += 1;
                }
            }
        }
        DynamicImage::ImageRgb8(rgb) => {
            for chunk in rgb.as_raw().chunks_exact(3) {
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];
                let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as usize;
                hist_y[y] += 1;
                total_valid_pixels += 1;

                let max_c = r.max(g).max(b);
                let min_c = r.min(g).min(b);
                let chroma = max_c - min_c;
                if chroma > max_chroma {
                    max_chroma = chroma;
                }
                if chroma > 35 {
                    high_chroma_count += 1;
                }

                let is_neutral = if y <= 50 { chroma <= 25 } else { chroma <= 12 };
                if is_neutral {
                    neutral_count += 1;
                    hist_y_neutral[y] += 1;
                }

                if y >= 205 && r >= g && g > b && chroma <= 55 {
                    highlight_count += 1;
                    sum_r_high += r as u64;
                    _sum_g_high += g as u64;
                    sum_b_high += b as u64;
                }
            }
        }
        DynamicImage::ImageRgba8(rgba) => {
            for chunk in rgba.as_raw().chunks_exact(4) {
                if chunk[3] < 128 {
                    continue;
                }
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];
                let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as usize;
                hist_y[y] += 1;
                total_valid_pixels += 1;

                let max_c = r.max(g).max(b);
                let min_c = r.min(g).min(b);
                let chroma = max_c - min_c;
                if chroma > max_chroma {
                    max_chroma = chroma;
                }
                if chroma > 35 {
                    high_chroma_count += 1;
                }

                let is_neutral = if y <= 50 { chroma <= 25 } else { chroma <= 12 };
                if is_neutral {
                    neutral_count += 1;
                    hist_y_neutral[y] += 1;
                }

                if y >= 205 && r >= g && g > b && chroma <= 55 {
                    highlight_count += 1;
                    sum_r_high += r as u64;
                    _sum_g_high += g as u64;
                    sum_b_high += b as u64;
                }
            }
        }
        _ => {
            let rgb = image.to_rgb8();
            return analyze_page_lightweight(&DynamicImage::ImageRgb8(rgb), page_number);
        }
    }

    if total_valid_pixels == 0 {
        return PageAnalysisSummary {
            page_number,
            classification: PageClassification::Color,
            black_floor: 0,
            white_ceiling: 255,
            dynamic_range: 255,
            paper_cast_detected: false,
            noise_floor: None,
        };
    }

    let neutral_ratio = neutral_count as f32 / total_valid_pixels as f32;
    let classification = if is_already_luma {
        PageClassification::Monochrome
    } else if neutral_ratio >= 0.85 {
        if max_chroma <= 35
            || high_chroma_count <= 50
            || ((high_chroma_count as f32 / total_valid_pixels as f32) <= 0.00005)
        {
            PageClassification::Monochrome
        } else {
            PageClassification::Mixed
        }
    } else {
        PageClassification::Color
    };

    let hist = match classification {
        PageClassification::Monochrome => &hist_y,
        PageClassification::Mixed => &hist_y_neutral,
        PageClassification::Color => &hist_y,
    };

    // Bimodal shelf detection in [10..=28]
    let mut max_peak_count = 0u32;
    let mut max_peak_bin = 0usize;
    for bin in 10..=28 {
        if hist[bin] > max_peak_count {
            max_peak_count = hist[bin];
            max_peak_bin = bin;
        }
    }

    let mut min_valley_count = u32::MAX;
    if max_peak_bin >= 10 {
        for bin in 3..max_peak_bin {
            if hist[bin] < min_valley_count {
                min_valley_count = hist[bin];
            }
        }
    }

    let surge_ratio = if min_valley_count == 0 {
        50.0
    } else {
        (max_peak_count as f32) / (min_valley_count as f32)
    };

    let is_shelf = classification != PageClassification::Color
        && max_peak_bin >= 10
        && (max_peak_count as f64) >= (total_valid_pixels as f64 * 0.004)
        && surge_ratio >= 2.8;

    let black_floor = if is_shelf {
        max_peak_bin as u8
    } else {
        let clip_target = (total_valid_pixels as f64 * 0.001).round() as u64;
        let mut bp = 0u8;
        let mut acc = 0u64;
        for (level, &count) in hist.iter().enumerate() {
            acc += count as u64;
            if acc >= clip_target {
                bp = level as u8;
                break;
            }
        }
        bp
    };

    let white_clip_target = (total_valid_pixels as f64 * 0.001).round() as u64;
    let mut white_ceiling = 255u8;
    let mut acc_h = 0u64;
    for (level, &count) in hist.iter().enumerate().rev() {
        acc_h += count as u64;
        if acc_h >= white_clip_target {
            white_ceiling = level as u8;
            break;
        }
    }

    let dynamic_range = white_ceiling.saturating_sub(black_floor);

    let paper_cast_detected = if highlight_count >= (total_valid_pixels / 200) && highlight_count > 0 {
        let avg_r = sum_r_high as f32 / highlight_count as f32;
        let avg_b = sum_b_high as f32 / highlight_count as f32;
        (avg_r - avg_b) >= 6.0
    } else {
        false
    };

    PageAnalysisSummary {
        page_number,
        classification,
        black_floor,
        white_ceiling,
        dynamic_range,
        paper_cast_detected,
        noise_floor: None,
    }
}

pub struct BookProfileManager {
    profile_cache: Cache<String, Arc<BookProfile>>,
    negative_cache: Cache<String, Instant>,
    komga_client: Arc<KomgaClient>,
    parallel_page_fetches: usize,
    timeout_seconds: u64,
}

impl BookProfileManager {
    pub fn new(
        komga_client: Arc<KomgaClient>,
        cache_size: usize,
        parallel_page_fetches: usize,
        timeout_seconds: u64,
    ) -> Self {
        let profile_cache = Cache::builder()
            .max_capacity(cache_size.max(1) as u64)
            .build();

        let negative_cache = Cache::builder()
            .time_to_live(Duration::from_secs(60))
            .max_capacity(1000)
            .build();

        Self {
            profile_cache,
            negative_cache,
            komga_client,
            parallel_page_fetches: parallel_page_fetches.max(1),
            timeout_seconds: timeout_seconds.max(5),
        }
    }

    pub async fn get_or_build_profile(
        &self,
        book_id: &str,
        cookie: Option<Cookie>,
        auth: Option<Authorization<Basic>>,
    ) -> Arc<BookProfile> {
        if self.negative_cache.get(book_id).await.is_some() {
            return Arc::new(BookProfile::empty(book_id));
        }

        let book_id_owned = book_id.to_string();
        let komga_client = self.komga_client.clone();
        let parallel_fetches = self.parallel_page_fetches;
        let timeout_dur = Duration::from_secs(self.timeout_seconds);
        let neg_cache = self.negative_cache.clone();

        let book_id_for_timeout = book_id_owned.clone();
        let neg_cache_for_timeout = neg_cache.clone();

        let build_fut = async move {
            let start_time = Instant::now();
            let page_list = match komga_client.get_book_pages(&book_id_owned, cookie.clone(), auth.clone()).await {
                Ok(pages) => pages,
                Err(e) => {
                    warn!("[BookProfile] Failed to fetch page list for {}: {}", book_id_owned, e);
                    neg_cache.insert(book_id_owned.clone(), Instant::now()).await;
                    return Err(format!("Failed to fetch page list: {}", e));
                }
            };

            if page_list.is_empty() {
                return Ok(Arc::new(BookProfile::empty(&book_id_owned)));
            }

            let total_pages = page_list.len();
            let semaphore = Arc::new(Semaphore::new(parallel_fetches));
            let mut tasks = Vec::with_capacity(total_pages);

            for page_info in page_list {
                let sem = semaphore.clone();
                let client = komga_client.clone();
                let b_id = book_id_owned.clone();
                let p_num = page_info.number;
                let c_opt = cookie.clone();
                let a_opt = auth.clone();

                tasks.push(tokio::spawn(async move {
                    let _permit = sem.acquire().await.ok()?;
                    let image_bytes = match client.get_page_image(&b_id, p_num, c_opt, a_opt).await {
                        Ok(b) => b,
                        Err(e) => {
                            warn!("[BookProfile] Failed to fetch page {} for {}: {}", p_num, b_id, e);
                            return None;
                        }
                    };

                    // Streaming drop sequence:
                    // 1. Decode image
                    let decoded = match image::load_from_memory(&image_bytes) {
                        Ok(img) => img,
                        Err(e) => {
                            warn!("[BookProfile] Failed to decode page {} for {}: {}", p_num, b_id, e);
                            return None;
                        }
                    };
                    // Drop compressed bytes immediately
                    drop(image_bytes);

                    // 2. Lightweight extraction
                    let summary = analyze_page_lightweight(&decoded, p_num);

                    // 3. Drop uncompressed pixel buffer immediately
                    drop(decoded);

                    Some(summary)
                }));
            }

            let mut analyses = Vec::with_capacity(total_pages);
            for t in tasks {
                if let Ok(Some(s)) = t.await {
                    analyses.push(s);
                }
            }

            if analyses.is_empty() && total_pages > 0 {
                warn!("[BookProfile] Failed to decode any pages for {}", book_id_owned);
                neg_cache.insert(book_id_owned.clone(), Instant::now()).await;
                return Err(format!("Failed to analyze any pages for {}", book_id_owned));
            }

            let elapsed_ms = start_time.elapsed().as_millis() as u64;
            let profile = BookProfile::from_all_pages(&book_id_owned, total_pages, analyses, elapsed_ms);
            Ok(Arc::new(profile))
        };

        let timed_build_fut = async move {
            match tokio::time::timeout(timeout_dur, build_fut).await {
                Ok(res) => res,
                Err(_) => {
                    warn!("[BookProfile] Timeout fetching/building profile for {}", book_id_for_timeout);
                    neg_cache_for_timeout.insert(book_id_for_timeout.clone(), Instant::now()).await;
                    Err(format!("Timeout profiling book {}", book_id_for_timeout))
                }
            }
        };

        match self.profile_cache.try_get_with::<_, String>(book_id.to_string(), timed_build_fut).await {
            Ok(profile) => profile,
            Err(e) => {
                warn!("[BookProfile] Profile building failed for {}: {}", book_id, e);
                Arc::new(BookProfile::empty(book_id))
            }
        }
    }

    pub async fn invalidate(&self, book_id: &str) {
        self.profile_cache.invalidate(book_id).await;
        self.negative_cache.invalidate(book_id).await;
    }
}
