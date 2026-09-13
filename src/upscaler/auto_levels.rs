use std::collections::HashMap;
use std::time::Instant;

use bytes::Bytes;
use image::{DynamicImage, GenericImageView};
use log::{info, warn};
use palette::{IntoColor, Lab, LinSrgb, Oklab, Srgb};
use rayon::prelude::*;
use serde_derive::{Deserialize, Serialize};

use crate::config::app_config::{AutoLevelsConfig, AutoLevelsMode};
use crate::upscaler::book_profile::BookProfile;

// ============================================================================
// Data Structures (Section 2 of Master Reference Specification)
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PageClassification {
    Monochrome,
    Mixed,
    Color,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DefectSeverity {
    None,
    Mild,
    Moderate,
    Severe,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DefectType {
    ElevatedBlackFloor {
        floor_value: u8,
        surge_ratio: f32,
        is_bimodal: bool,
    },
    CompressedDynamicRange {
        effective_range: u8,
        p5: u8,
        p95: u8,
        tail_emptiness: f32,
    },
    PaperYellowing {
        cast_magnitude: f32,
        highlight_rgb: (f32, f32, f32),
        is_differential: bool,
    },
    ClippedHighlights {
        white_point: u8,
        has_plateau: bool,
    },
    ElevatedNoiseFloor {
        noise_sigma: f32,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DefectDetection {
    pub defect_type: DefectType,
    pub raw_confidence: f32,
    pub adjusted_confidence: f32,
    pub severity: DefectSeverity,
    pub evidence: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrectionPlan {
    pub black_point: u8,
    pub white_point: u8,
    pub correction_fraction: f32,
    pub gamma: f32,
    pub per_channel_black: Option<(u8, u8, u8)>,
    pub paper_cast_gains: Option<(f32, f32, f32)>,
    pub paper_cast_ramp_start: u8,
    pub use_oklab: bool,
    pub skip_correction: bool,
}

impl Default for CorrectionPlan {
    fn default() -> Self {
        Self {
            black_point: 0,
            white_point: 255,
            correction_fraction: 1.0,
            gamma: 1.0,
            per_channel_black: None,
            paper_cast_gains: None,
            paper_cast_ramp_start: 180,
            use_oklab: false,
            skip_correction: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Outcome {
    Pass,
    Reject,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TierResult {
    pub tier: u8,
    pub outcome: Outcome,
    pub metrics: HashMap<String, f64>,
    pub reason: Option<String>,
}

impl TierResult {
    pub fn pass(tier: u8) -> Self {
        Self {
            tier,
            outcome: Outcome::Pass,
            metrics: HashMap::new(),
            reason: None,
        }
    }

    pub fn reject(tier: u8, reason: impl Into<String>) -> Self {
        Self {
            tier,
            outcome: Outcome::Reject,
            metrics: HashMap::new(),
            reason: Some(reason.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityVerdict {
    pub outcome: Outcome,
    pub attenuation_attempted: bool,
    pub tier_results: Vec<TierResult>,
}

impl QualityVerdict {
    pub fn pass() -> Self {
        Self {
            outcome: Outcome::Pass,
            attenuation_attempted: false,
            tier_results: Vec::new(),
        }
    }

    pub fn reject(tier_results: Vec<TierResult>, attenuation_attempted: bool) -> Self {
        Self {
            outcome: Outcome::Reject,
            attenuation_attempted,
            tier_results,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LevelInfo {
    pub was_leveled: bool,
    pub is_monochrome: bool,
    pub classification: PageClassification,
    pub black_point: u8,
    pub white_point: u8,
    pub channel_black_points: Option<[u8; 3]>,
    pub quality_verdict: Option<QualityVerdict>,
}

impl Default for LevelInfo {
    fn default() -> Self {
        Self {
            was_leveled: false,
            is_monochrome: false,
            classification: PageClassification::Color,
            black_point: 0,
            white_point: 255,
            channel_black_points: None,
            quality_verdict: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DualZoneSample {
    pub count: usize,
    pub avg_rgb: (f32, f32, f32),
    pub chroma: f32,
}

#[derive(Debug, Clone)]
pub struct PageAnalysis {
    pub page_number: usize,
    pub classification: PageClassification,
    pub luminance_histogram: [u32; 256],
    pub neutral_luminance_histogram: [u32; 256],
    pub neutral_channel_histograms: ([u32; 256], [u32; 256], [u32; 256]),
    pub total_pixels: usize,
    pub neutral_pixel_count: usize,
    pub neutral_ratio: f32,
    pub max_chroma: u8,
    pub high_chroma_count: usize,
    pub highlight_sample: DualZoneSample,
    pub shadow_sample: DualZoneSample,
    pub defects: Vec<DefectDetection>,
    pub black_floor: u8,
    pub white_ceiling: u8,
    pub is_shelf_detected: bool,
    pub per_channel_black: Option<(u8, u8, u8)>,
}

// ============================================================================
// Page Analysis & Classification (Section 3b)
// ============================================================================

#[inline]
pub fn is_neutral_pixel(r: u8, g: u8, b: u8, y: u8) -> bool {
    let chroma = r.max(g).max(b) - r.min(g).min(b);
    if y <= 50 {
        chroma <= 25 // Dark ink region: accommodates JPEG chroma ringing & subsampling
    } else {
        chroma <= 12 // Midtones & highlights: strict neutrality
    }
}

pub fn analyze_page(
    image: &DynamicImage,
    page_number: usize,
    config: &AutoLevelsConfig,
) -> PageAnalysis {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return PageAnalysis {
            page_number,
            classification: PageClassification::Color,
            luminance_histogram: [0; 256],
            neutral_luminance_histogram: [0; 256],
            neutral_channel_histograms: ([0; 256], [0; 256], [0; 256]),
            total_pixels: 0,
            neutral_pixel_count: 0,
            neutral_ratio: 0.0,
            max_chroma: 0,
            high_chroma_count: 0,
            highlight_sample: DualZoneSample::default(),
            shadow_sample: DualZoneSample::default(),
            defects: Vec::new(),
            black_floor: 0,
            white_ceiling: 255,
            is_shelf_detected: false,
            per_channel_black: None,
        };
    }

    let is_already_luma = matches!(
        image,
        DynamicImage::ImageLuma8(_) | DynamicImage::ImageLumaA8(_)
    );

    let mut hist_y = [0u32; 256];
    let mut hist_y_neutral = [0u32; 256];
    let mut hist_r_neutral = [0u32; 256];
    let mut hist_g_neutral = [0u32; 256];
    let mut hist_b_neutral = [0u32; 256];
    let mut neutral_count: usize = 0;
    let mut total_valid_pixels: usize = 0;
    let mut max_chroma: u8 = 0;
    let mut high_chroma_count: usize = 0;

    let mut hl_count: usize = 0;
    let mut hl_sum_r: u64 = 0;
    let mut hl_sum_g: u64 = 0;
    let mut hl_sum_b: u64 = 0;

    let mut sh_count: usize = 0;
    let mut sh_sum_r: u64 = 0;
    let mut sh_sum_g: u64 = 0;
    let mut sh_sum_b: u64 = 0;

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

                let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8;
                hist_y[y as usize] += 1;
                total_valid_pixels += 1;

                let chroma = r.max(g).max(b) - r.min(g).min(b);
                if chroma > max_chroma {
                    max_chroma = chroma;
                }
                if chroma > 35 {
                    high_chroma_count += 1;
                }

                if is_neutral_pixel(r, g, b, y) {
                    neutral_count += 1;
                    hist_y_neutral[y as usize] += 1;
                }

                if y <= 40 && chroma <= 25 {
                    hist_r_neutral[r as usize] += 1;
                    hist_g_neutral[g as usize] += 1;
                    hist_b_neutral[b as usize] += 1;
                }

                // Highlight sample: y >= 205, relaxed chroma <= 55, r >= g && g > b
                if y >= 205 && chroma <= 55 && r >= g && g > b {
                    hl_count += 1;
                    hl_sum_r += r as u64;
                    hl_sum_g += g as u64;
                    hl_sum_b += b as u64;
                }

                // Shadow sample: y in [8, 45]
                if (8..=45).contains(&y) {
                    sh_count += 1;
                    sh_sum_r += r as u64;
                    sh_sum_g += g as u64;
                    sh_sum_b += b as u64;
                }
            }
        }
        DynamicImage::ImageRgba8(rgba) => {
            for chunk in rgba.as_raw().chunks_exact(4) {
                if chunk[3] < 128 {
                    continue; // Skip transparent pixels
                }
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];

                let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8;
                hist_y[y as usize] += 1;
                total_valid_pixels += 1;

                let chroma = r.max(g).max(b) - r.min(g).min(b);
                if chroma > max_chroma {
                    max_chroma = chroma;
                }
                if chroma > 35 {
                    high_chroma_count += 1;
                }

                if is_neutral_pixel(r, g, b, y) {
                    neutral_count += 1;
                    hist_y_neutral[y as usize] += 1;
                }

                if y <= 40 && chroma <= 25 {
                    hist_r_neutral[r as usize] += 1;
                    hist_g_neutral[g as usize] += 1;
                    hist_b_neutral[b as usize] += 1;
                }

                if y >= 205 && chroma <= 55 && r >= g && g > b {
                    hl_count += 1;
                    hl_sum_r += r as u64;
                    hl_sum_g += g as u64;
                    hl_sum_b += b as u64;
                }

                if (8..=45).contains(&y) {
                    sh_count += 1;
                    sh_sum_r += r as u64;
                    sh_sum_g += g as u64;
                    sh_sum_b += b as u64;
                }
            }
        }
        _ => {
            let rgb = image.to_rgb8();
            return analyze_page(&DynamicImage::ImageRgb8(rgb), page_number, config);
        }
    }

    if total_valid_pixels == 0 {
        return PageAnalysis {
            page_number,
            classification: PageClassification::Color,
            luminance_histogram: hist_y,
            neutral_luminance_histogram: hist_y_neutral,
            neutral_channel_histograms: (hist_r_neutral, hist_g_neutral, hist_b_neutral),
            total_pixels: 0,
            neutral_pixel_count: 0,
            neutral_ratio: 0.0,
            max_chroma: 0,
            high_chroma_count: 0,
            highlight_sample: DualZoneSample::default(),
            shadow_sample: DualZoneSample::default(),
            defects: Vec::new(),
            black_floor: 0,
            white_ceiling: 255,
            is_shelf_detected: false,
            per_channel_black: None,
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

    let highlight_sample = if hl_count > 0 {
        let avg_r = hl_sum_r as f32 / hl_count as f32;
        let avg_g = hl_sum_g as f32 / hl_count as f32;
        let avg_b = hl_sum_b as f32 / hl_count as f32;
        let chroma = avg_r.max(avg_g).max(avg_b) - avg_r.min(avg_g).min(avg_b);
        DualZoneSample {
            count: hl_count,
            avg_rgb: (avg_r, avg_g, avg_b),
            chroma,
        }
    } else {
        DualZoneSample::default()
    };

    let shadow_sample = if sh_count > 0 {
        let avg_r = sh_sum_r as f32 / sh_count as f32;
        let avg_g = sh_sum_g as f32 / sh_count as f32;
        let avg_b = sh_sum_b as f32 / sh_count as f32;
        let chroma = avg_r.max(avg_g).max(avg_b) - avg_r.min(avg_g).min(avg_b);
        DualZoneSample {
            count: sh_count,
            avg_rgb: (avg_r, avg_g, avg_b),
            chroma,
        }
    } else {
        DualZoneSample::default()
    };

    let min_y = hist_y.iter().position(|&c| c > 0).unwrap_or(0) as u8;
    let max_y = hist_y.iter().rposition(|&c| c > 0).unwrap_or(255) as u8;
    let dynamic_range = max_y.saturating_sub(min_y);

    let mut analysis = PageAnalysis {
        page_number,
        classification,
        luminance_histogram: hist_y,
        neutral_luminance_histogram: hist_y_neutral,
        neutral_channel_histograms: (hist_r_neutral, hist_g_neutral, hist_b_neutral),
        total_pixels: total_valid_pixels,
        neutral_pixel_count: neutral_count,
        neutral_ratio,
        max_chroma,
        high_chroma_count,
        highlight_sample,
        shadow_sample,
        defects: Vec::new(),
        black_floor: 0,
        white_ceiling: 255,
        is_shelf_detected: false,
        per_channel_black: None,
    };

    // Sections 6.1 & 6.2: Solid black pages (bins 0-5) or solid white pages (bins 250-255) with dynamic range < 10
    // must skip correction and pass through unmodified with no defects.
    if dynamic_range < 10 {
        return analysis;
    }

    // Detect defects
    let hist_for_shelf = match classification {
        PageClassification::Monochrome => &analysis.luminance_histogram,
        PageClassification::Mixed => &analysis.neutral_luminance_histogram,
        PageClassification::Color => &analysis.luminance_histogram,
    };

    let mut defects = Vec::new();

    if let Some(d) = detect_elevated_black_floor(hist_for_shelf, classification, total_valid_pixels, config) {
        if let DefectType::ElevatedBlackFloor { floor_value, is_bimodal, .. } = d.defect_type {
            analysis.black_floor = floor_value;
            analysis.is_shelf_detected = is_bimodal;
        }
        defects.push(d);
    }

    if let Some(d) = detect_compressed_dynamic_range(&analysis.luminance_histogram, total_valid_pixels) {
        defects.push(d);
    }

    if let Some(d) = detect_paper_yellowing(&analysis, classification, config) {
        defects.push(d);
    }

    if let Some(d) = detect_clipped_highlights(&analysis.luminance_histogram, classification, total_valid_pixels, config) {
        if let DefectType::ClippedHighlights { white_point, .. } = d.defect_type {
            analysis.white_ceiling = white_point;
        }
        defects.push(d);
    }

    if let Some(d) = detect_elevated_noise_floor(image, classification) {
        defects.push(d);
    }

    analysis.defects = defects;
    analysis
}

// ============================================================================
// Defect Detection Engine (Section 3c)
// ============================================================================

pub fn detect_elevated_black_floor(
    hist: &[u32; 256],
    classification: PageClassification,
    _total_pixels: usize,
    config: &AutoLevelsConfig,
) -> Option<DefectDetection> {
    let n_samples: u32 = hist.iter().sum();
    if n_samples == 0 {
        return None;
    }

    // 1. Find highest peak in elevated linework range [10..=28]
    let mut max_peak_count = 0u32;
    let mut max_peak_bin = 0usize;
    for bin in 10..=28 {
        if hist[bin] > max_peak_count {
            max_peak_count = hist[bin];
            max_peak_bin = bin;
        }
    }

    // 2. Find minimum valley between bin 3 and max_peak_bin
    let mut min_valley_count = u32::MAX;
    if max_peak_bin >= 3 {
        for bin in 3..max_peak_bin {
            if hist[bin] < min_valley_count {
                min_valley_count = hist[bin];
            }
        }
    }

    let surge_ratio = if min_valley_count == 0 {
        50.0
    } else {
        max_peak_count as f32 / min_valley_count as f32
    };

    // 3. Peak compactness FWHM check (Max FWHM = 6 bins)
    let half_max = max_peak_count / 2;
    let mut left_fwhm = max_peak_bin;
    while left_fwhm > 0 && hist[left_fwhm] >= half_max {
        left_fwhm -= 1;
    }
    let mut right_fwhm = max_peak_bin;
    while right_fwhm < 255 && hist[right_fwhm] >= half_max {
        right_fwhm += 1;
    }
    let fwhm = (right_fwhm.saturating_sub(left_fwhm + 1)).max(1);

    // Bimodal Shelf Criteria:
    // Level-0 text or black borders do NOT disqualify the shelf!
    let is_shelf = classification != PageClassification::Color
        && max_peak_bin >= 10
        && (max_peak_count as f64) >= (n_samples as f64 * 0.004)
        && surge_ratio >= config.detection.surge_ratio_threshold
        && fwhm <= 6;

    let (floor_value, mut raw_confidence) = if is_shelf {
        (max_peak_bin as u8, 0.88f32)
    } else {
        let clip_target = (n_samples as f64 * (config.detection.black_clip_percent as f64 / 100.0)).round() as u64;
        let mut bp = 0u8;
        let mut acc = 0u64;
        for (level, &count) in hist.iter().enumerate() {
            acc += count as u64;
            if acc >= clip_target {
                bp = level as u8;
                break;
            }
        }
        (bp, 0.50f32)
    };

    if floor_value <= 1 {
        return None;
    }

    let severity = if floor_value <= 5 {
        DefectSeverity::Mild
    } else if floor_value <= 15 {
        DefectSeverity::Moderate
    } else if floor_value <= 30 {
        DefectSeverity::Severe
    } else {
        return None;
    };

    if (16..=20).contains(&floor_value) {
        raw_confidence = (raw_confidence + 0.10).min(0.95);
    }

    Some(DefectDetection {
        defect_type: DefectType::ElevatedBlackFloor {
            floor_value,
            surge_ratio,
            is_bimodal: is_shelf,
        },
        raw_confidence,
        adjusted_confidence: raw_confidence,
        severity,
        evidence: format!("Ink floor={}, surge={:.1}, shelf={}", floor_value, surge_ratio, is_shelf),
    })
}

pub fn detect_compressed_dynamic_range(
    hist: &[u32; 256],
    total_pixels: usize,
) -> Option<DefectDetection> {
    let n_samples: u32 = hist.iter().sum();
    if n_samples == 0 || total_pixels == 0 {
        return None;
    }

    let p5_target = (n_samples as f64 * 0.05).round() as u64;
    let p95_target = (n_samples as f64 * 0.95).round() as u64;

    let mut p5 = 0u8;
    let mut acc = 0u64;
    for (level, &count) in hist.iter().enumerate() {
        acc += count as u64;
        if acc >= p5_target {
            p5 = level as u8;
            break;
        }
    }

    let mut p95 = 255u8;
    let mut acc_h = 0u64;
    for (level, &count) in hist.iter().enumerate() {
        acc_h += count as u64;
        if acc_h >= p95_target {
            p95 = level as u8;
            break;
        }
    }

    let effective_range = p95.saturating_sub(p5);
    // Range >= 200 is healthy; effective_range < 10 indicates solid black/white page, not compression
    if effective_range >= 200 || effective_range < 10 {
        return None;
    }

    let empty_below = (0..=p5 as usize).filter(|&b| hist[b] == 0).count();
    let empty_above = (p95 as usize..=255).filter(|&b| hist[b] == 0).count();
    let tail_bins = (p5 as usize) + (255 - p95 as usize);
    let tail_emptiness = if tail_bins > 0 {
        (empty_below + empty_above) as f32 / tail_bins as f32
    } else {
        1.0
    };

    if tail_emptiness < 0.70 {
        return None; // Moody atmospheric artwork
    }

    let severity = if effective_range >= 180 {
        DefectSeverity::Mild
    } else if effective_range >= 130 {
        DefectSeverity::Moderate
    } else {
        DefectSeverity::Severe
    };

    let raw_confidence = (0.40 + (tail_emptiness * 0.40)).clamp(0.40, 0.80);

    Some(DefectDetection {
        defect_type: DefectType::CompressedDynamicRange {
            effective_range,
            p5,
            p95,
            tail_emptiness,
        },
        raw_confidence,
        adjusted_confidence: raw_confidence,
        severity,
        evidence: format!("Dynamic range {}, tail_emptiness={:.2}", effective_range, tail_emptiness),
    })
}

pub fn detect_paper_yellowing(
    analysis: &PageAnalysis,
    classification: PageClassification,
    config: &AutoLevelsConfig,
) -> Option<DefectDetection> {
    if analysis.highlight_sample.count < (analysis.total_pixels as f64 * 0.005).round() as usize {
        return None;
    }

    let (hl_r, hl_g, hl_b) = analysis.highlight_sample.avg_rgb;

    // Physical Lignin Absorption Pattern: R >= G > B
    if !(hl_r >= hl_g && hl_g > hl_b) {
        return None;
    }

    let cast_magnitude = hl_r - hl_b;
    let rg_diff = hl_r - hl_g;
    if cast_magnitude < 6.0 || cast_magnitude > 55.0 || rg_diff > 0.65 * cast_magnitude {
        return None;
    }

    // Dual-Zone Check: Compare highlights against dark ink shadows
    if config.detection.protect_sepia_artwork && analysis.shadow_sample.count >= 50 {
        let (sh_r, _, sh_b) = analysis.shadow_sample.avg_rgb;
        if sh_r > sh_b + 4.0 {
            info!(
                "[PaperYellowing] Skipped: shadows are warm ({:.1}, {:.1}) -> intentional sepia art",
                sh_r, sh_b
            );
            return None;
        }
    }

    let severity = if cast_magnitude < 12.0 {
        DefectSeverity::Mild
    } else if cast_magnitude < 25.0 {
        DefectSeverity::Moderate
    } else {
        DefectSeverity::Severe
    };

    let mut raw_confidence: f32 = if classification == PageClassification::Color { 0.75f32 } else { 0.55f32 };
    if cast_magnitude > 15.0 {
        raw_confidence = (raw_confidence + 0.15f32).min(0.90f32);
    }

    Some(DefectDetection {
        defect_type: DefectType::PaperYellowing {
            cast_magnitude,
            highlight_rgb: (hl_r, hl_g, hl_b),
            is_differential: true,
        },
        raw_confidence,
        adjusted_confidence: raw_confidence,
        severity,
        evidence: format!("Differential paper cast: magnitude={:.1}", cast_magnitude),
    })
}

pub fn detect_clipped_highlights(
    hist: &[u32; 256],
    classification: PageClassification,
    total_pixels: usize,
    config: &AutoLevelsConfig,
) -> Option<DefectDetection> {
    if total_pixels == 0 {
        return None;
    }

    let clip_target = (total_pixels as f64 * (config.detection.white_clip_percent as f64 / 100.0)).round() as u64;
    let mut white_point = 255u8;
    let mut acc_h = 0u64;
    for (level, &count) in hist.iter().enumerate().rev() {
        acc_h += count as u64;
        if acc_h >= clip_target {
            white_point = level as u8;
            break;
        }
    }

    if white_point >= 254 {
        return None;
    }
    if classification == PageClassification::Color && white_point >= 250 {
        return None;
    }

    let min_white_threshold = config.correction.min_white_threshold;
    let plateau_threshold = (total_pixels as f64 * 0.005).round() as u32;
    let has_plateau = hist[white_point as usize] >= plateau_threshold;

    if white_point < min_white_threshold && !has_plateau {
        return None; // Natural dark scene trailing off gradually
    }

    let severity = if white_point >= 250 {
        DefectSeverity::Mild
    } else if white_point >= 240 {
        DefectSeverity::Moderate
    } else if white_point >= min_white_threshold || has_plateau {
        DefectSeverity::Severe
    } else {
        return None;
    };

    Some(DefectDetection {
        defect_type: DefectType::ClippedHighlights {
            white_point,
            has_plateau,
        },
        raw_confidence: 0.60,
        adjusted_confidence: 0.60,
        severity,
        evidence: format!("White point at {}, plateau={}", white_point, has_plateau),
    })
}

pub fn detect_elevated_noise_floor(
    image: &DynamicImage,
    classification: PageClassification,
) -> Option<DefectDetection> {
    let (width, height) = image.dimensions();
    if width < 3 || height < 3 {
        return None;
    }

    let luma = image.to_luma8();
    let mut laplacian_values = Vec::new();

    for y in (1..height - 1).step_by(2) {
        for x in (1..width - 1).step_by(2) {
            let p00 = luma.get_pixel(x - 1, y - 1)[0] as i32;
            let p01 = luma.get_pixel(x, y - 1)[0] as i32;
            let p02 = luma.get_pixel(x + 1, y - 1)[0] as i32;
            let p10 = luma.get_pixel(x - 1, y)[0] as i32;
            let p11 = luma.get_pixel(x, y)[0] as i32;
            let p12 = luma.get_pixel(x + 1, y)[0] as i32;
            let p20 = luma.get_pixel(x - 1, y + 1)[0] as i32;
            let p21 = luma.get_pixel(x, y + 1)[0] as i32;
            let p22 = luma.get_pixel(x + 1, y + 1)[0] as i32;

            let lap = ((p00 + p02 + p20 + p22) - 2 * (p01 + p21 + p10 + p12) + 4 * p11).abs() as f32;
            laplacian_values.push(lap);
        }
    }

    if laplacian_values.is_empty() {
        return None;
    }

    laplacian_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_lap = laplacian_values[laplacian_values.len() / 2];

    let mut deviations: Vec<f32> = laplacian_values.iter().map(|&v| (v - median_lap).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = deviations[deviations.len() / 2];
    let noise_sigma = 1.4826 * mad;

    if noise_sigma < 4.0 {
        return None;
    }

    let severity = if noise_sigma < 8.0 {
        DefectSeverity::Mild
    } else if noise_sigma < 15.0 {
        DefectSeverity::Moderate
    } else {
        DefectSeverity::Severe
    };

    let raw_confidence = if classification == PageClassification::Monochrome { 0.30 } else { 0.40 };

    Some(DefectDetection {
        defect_type: DefectType::ElevatedNoiseFloor { noise_sigma },
        raw_confidence,
        adjusted_confidence: raw_confidence,
        severity,
        evidence: format!("Noise sigma={:.2} (MAD={:.2})", noise_sigma, mad),
    })
}

// ============================================================================
// Book-Level Confidence Adjustment (Section 3c.6)
// ============================================================================

pub fn adjust_confidence_with_book_context(
    defects: &mut [DefectDetection],
    classification: PageClassification,
    book_profile: &BookProfile,
) {
    let group = match classification {
        PageClassification::Monochrome => &book_profile.monochrome_group,
        PageClassification::Mixed => &book_profile.mixed_group,
        PageClassification::Color => &book_profile.color_group,
    };

    if group.page_count <= 1 {
        return;
    }

    for defect in defects.iter_mut() {
        match defect.defect_type {
            DefectType::ElevatedBlackFloor { floor_value, .. } => {
                let low = floor_value.saturating_sub(3) as usize;
                let high = (floor_value as usize + 3).min(255);
                let similar_count: u32 = group.black_floor_distribution[low..=high].iter().sum();
                let freq = similar_count as f32 / group.page_count as f32;

                if freq >= 0.30 {
                    let boost = ((freq - 0.30) / 0.70) * 0.15;
                    defect.adjusted_confidence = (defect.raw_confidence + boost).min(0.98);
                } else if freq < 0.10 {
                    defect.adjusted_confidence = defect.raw_confidence * 0.60;
                }
            }
            DefectType::CompressedDynamicRange { .. } => {
                if group.median_dynamic_range < 200 {
                    defect.adjusted_confidence = (defect.raw_confidence + 0.15).min(0.95);
                }
            }
            DefectType::PaperYellowing { .. } => {
                let cast_freq = group.paper_cast_detected_count as f32 / group.page_count as f32;
                if cast_freq >= 0.50 {
                    defect.adjusted_confidence = (defect.raw_confidence + 0.15).min(0.95);
                } else if cast_freq < 0.10 {
                    defect.adjusted_confidence = defect.raw_confidence * 0.50;
                }
            }
            DefectType::ClippedHighlights { white_point, .. } => {
                let low = white_point.saturating_sub(3) as usize;
                let high = (white_point as usize + 3).min(255);
                let similar_count: u32 = group.white_point_distribution[low..=high].iter().sum();
                let freq = similar_count as f32 / group.page_count as f32;

                if freq >= 0.30 {
                    let boost = ((freq - 0.30) / 0.70) * 0.15;
                    defect.adjusted_confidence = (defect.raw_confidence + boost).min(0.98);
                } else if freq < 0.10 {
                    defect.adjusted_confidence = defect.raw_confidence * 0.60;
                }
            }
            DefectType::ElevatedNoiseFloor { .. } => {}
        }
    }
}

// ============================================================================
// Correction Engine (Section 3d)
// ============================================================================

pub fn compute_correction_fraction(
    confidence: f32,
    severity: DefectSeverity,
    config: &AutoLevelsConfig,
) -> f32 {
    if confidence < config.detection.confidence_threshold {
        return 0.0;
    }

    let severity_multiplier = match severity {
        DefectSeverity::None => 0.0,
        DefectSeverity::Mild => 0.40,
        DefectSeverity::Moderate => 0.75,
        DefectSeverity::Severe => 1.00,
    };

    let confidence_scale = ((confidence - config.detection.confidence_threshold)
        / (1.0 - config.detection.confidence_threshold))
        .clamp(0.0, 1.0);

    severity_multiplier * confidence_scale
}

pub fn generate_correction_plan(
    defects: &[DefectDetection],
    classification: PageClassification,
    config: &AutoLevelsConfig,
) -> CorrectionPlan {
    let mut plan = CorrectionPlan::default();

    for defect in defects {
        let fraction = compute_correction_fraction(defect.adjusted_confidence, defect.severity, config);
        if fraction == 0.0 {
            continue;
        }

        match defect.defect_type {
            DefectType::ElevatedBlackFloor { floor_value, .. } => {
                let clamped_floor = floor_value.min(config.correction.max_black_shift);
                plan.black_point = plan.black_point.max((clamped_floor as f32 * fraction).round() as u8);
                plan.correction_fraction = plan.correction_fraction.max(fraction);
            }
            DefectType::ClippedHighlights { white_point, .. } => {
                let corrected_wp = 255 - ((255.0 - white_point as f32) * fraction).round() as u8;
                plan.white_point = plan.white_point.min(corrected_wp);
                plan.correction_fraction = plan.correction_fraction.max(fraction);
            }
            DefectType::CompressedDynamicRange { p5, p95, .. } => {
                if plan.black_point == 0 {
                    plan.black_point = plan.black_point.max((p5 as f32 * fraction).round() as u8);
                }
                if plan.white_point == 255 {
                    plan.white_point = plan.white_point.min(255 - ((255.0 - p95 as f32) * fraction).round() as u8);
                }
                plan.correction_fraction = plan.correction_fraction.max(fraction);
            }
            DefectType::PaperYellowing { highlight_rgb, .. } => {
                if config.correction.correct_paper_cast {
                    let max_ch = highlight_rgb.0.max(highlight_rgb.1).max(highlight_rgb.2);
                    if highlight_rgb.0 > 0.0 && highlight_rgb.1 > 0.0 && highlight_rgb.2 > 0.0 {
                        plan.paper_cast_gains = Some((
                            1.0 + ((max_ch / highlight_rgb.0) - 1.0) * fraction,
                            1.0 + ((max_ch / highlight_rgb.1) - 1.0) * fraction,
                            1.0 + ((max_ch / highlight_rgb.2) - 1.0) * fraction,
                        ));
                        plan.paper_cast_ramp_start = 180;
                        plan.correction_fraction = plan.correction_fraction.max(fraction);
                    }
                }
            }
            DefectType::ElevatedNoiseFloor { .. } => {}
        }
    }

    // Safeguards
    if classification == PageClassification::Color {
        if plan.black_point > 0 && (plan.black_point <= 1 || plan.black_point > 12) {
            plan.black_point = 0;
        }
        if plan.white_point < 255 && (plan.white_point >= 254 || plan.white_point < 250) {
            plan.white_point = 255;
        }
    } else {
        if plan.black_point <= 1 {
            plan.black_point = 0;
        }
        if plan.black_point > config.correction.max_black_shift {
            plan.black_point = config.correction.max_black_shift;
        }
        if plan.white_point >= 254 {
            plan.white_point = 255;
        }
        if plan.white_point < config.correction.min_white_threshold {
            plan.white_point = 255;
        }
    }

    plan.use_oklab = classification == PageClassification::Color;
    plan.skip_correction = plan.black_point == 0
        && plan.white_point == 255
        && plan.paper_cast_gains.is_none()
        && plan.per_channel_black.is_none();
    plan.gamma = config.correction.gamma;

    plan
}

pub fn detect_per_channel_shelf(
    analysis: &PageAnalysis,
    plan: &CorrectionPlan,
) -> Option<(u8, u8, u8)> {
    if !analysis.is_shelf_detected || analysis.classification == PageClassification::Color {
        return None;
    }

    let (r_hist, g_hist, b_hist) = &analysis.neutral_channel_histograms;

    let mut peak_r = 0u32;
    let mut peak_r_bin = 0u8;
    let mut peak_g = 0u32;
    let mut peak_g_bin = 0u8;
    let mut peak_b = 0u32;
    let mut peak_b_bin = 0u8;

    for bin in 2..=35 {
        if r_hist[bin] > peak_r {
            peak_r = r_hist[bin];
            peak_r_bin = bin as u8;
        }
        if g_hist[bin] > peak_g {
            peak_g = g_hist[bin];
            peak_g_bin = bin as u8;
        }
        if b_hist[bin] > peak_b {
            peak_b = b_hist[bin];
            peak_b_bin = bin as u8;
        }
    }

    let channel_spread = peak_r_bin.max(peak_g_bin).max(peak_b_bin)
        - peak_r_bin.min(peak_g_bin).min(peak_b_bin);

    if channel_spread < 4 || plan.white_point < 250 {
        return None;
    }

    Some((peak_r_bin, peak_g_bin, peak_b_bin))
}

// ----------------------------------------------------------------------------
// Color Page Correction (Oklab with Radial Chord Bisection)
// ----------------------------------------------------------------------------

fn build_oklab_l_lut(l_black: f32, l_white: f32, gamma: f32) -> [f32; 4096] {
    let inv_gamma = if (gamma - 1.0).abs() > 0.001 && gamma > 0.1 {
        1.0 / gamma
    } else {
        1.0
    };

    let mut lut = [0.0f32; 4096];
    let range = (l_white - l_black).max(0.001);

    for i in 0..4096 {
        let l = i as f32 / 4095.0;
        if l <= l_black {
            lut[i] = 0.0;
        } else if l >= l_white {
            lut[i] = 1.0;
        } else {
            let norm = (l - l_black) / range;
            let mapped = if inv_gamma != 1.0 {
                norm.powf(inv_gamma)
            } else {
                norm
            };
            lut[i] = mapped.clamp(0.0, 1.0);
        }
    }
    lut
}

pub fn apply_color_correction_oklab(image: &mut DynamicImage, plan: &CorrectionPlan) {
    let bp = plan.black_point as f32 / 255.0;
    let wp = plan.white_point as f32 / 255.0;

    let srgb_black = Srgb::new(bp, bp, bp);
    let oklab_black: Oklab = srgb_black.into_color();
    let l_black = oklab_black.l;

    let srgb_white = Srgb::new(wp, wp, wp);
    let oklab_white: Oklab = srgb_white.into_color();
    let l_white = oklab_white.l;

    let l_lut = build_oklab_l_lut(l_black, l_white, plan.gamma);

    match image {
        DynamicImage::ImageRgb8(rgb) => {
            (&mut **rgb)
                .par_chunks_exact_mut(3)
                .for_each(|chunk| {
                    let r = chunk[0];
                    let g = chunk[1];
                    let b = chunk[2];

                    let srgb = Srgb::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
                    let mut oklab: Oklab = srgb.into_color();

                    let orig_l = oklab.l;
                    if orig_l <= 0.001 {
                        return;
                    }

                    let lut_idx = ((orig_l * 4095.0).round() as usize).min(4095);
                    let new_l = l_lut[lut_idx];
                    oklab.l = new_l;

                    // Step 1: Fast in-gamut test on linear sRGB (<5ns)
                    let lin_rgb: LinSrgb = oklab.into_color();
                    if lin_rgb.red >= 0.0 && lin_rgb.red <= 1.0
                        && lin_rgb.green >= 0.0 && lin_rgb.green <= 1.0
                        && lin_rgb.blue >= 0.0 && lin_rgb.blue <= 1.0
                    {
                        let out_srgb: Srgb = lin_rgb.into_color();
                        chunk[0] = (out_srgb.red * 255.0).round().clamp(0.0, 255.0) as u8;
                        chunk[1] = (out_srgb.green * 255.0).round().clamp(0.0, 255.0) as u8;
                        chunk[2] = (out_srgb.blue * 255.0).round().clamp(0.0, 255.0) as u8;
                        return;
                    }

                    // Step 2: Out-of-gamut boundary compression along constant hue ray (10 binary search iterations)
                    let mut t_low = 0.0f32;
                    let mut t_high = 1.0f32;
                    let mut best_lin = lin_rgb;

                    for _ in 0..10 {
                        let t_mid = 0.5 * (t_low + t_high);
                        let test_oklab = Oklab::new(new_l, oklab.a * t_mid, oklab.b * t_mid);
                        let test_lin: LinSrgb = test_oklab.into_color();
                        if test_lin.red >= 0.0 && test_lin.red <= 1.0
                            && test_lin.green >= 0.0 && test_lin.green <= 1.0
                            && test_lin.blue >= 0.0 && test_lin.blue <= 1.0
                        {
                            t_low = t_mid;
                            best_lin = test_lin;
                        } else {
                            t_high = t_mid;
                        }
                    }

                    let out_srgb: Srgb = best_lin.into_color();
                    chunk[0] = (out_srgb.red * 255.0).round().clamp(0.0, 255.0) as u8;
                    chunk[1] = (out_srgb.green * 255.0).round().clamp(0.0, 255.0) as u8;
                    chunk[2] = (out_srgb.blue * 255.0).round().clamp(0.0, 255.0) as u8;
                });
        }
        DynamicImage::ImageRgba8(rgba) => {
            (&mut **rgba)
                .par_chunks_exact_mut(4)
                .for_each(|chunk| {
                    if chunk[3] < 128 {
                        return;
                    }
                    let r = chunk[0];
                    let g = chunk[1];
                    let b = chunk[2];

                    let srgb = Srgb::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0);
                    let mut oklab: Oklab = srgb.into_color();

                    let orig_l = oklab.l;
                    if orig_l <= 0.001 {
                        return;
                    }

                    let lut_idx = ((orig_l * 4095.0).round() as usize).min(4095);
                    let new_l = l_lut[lut_idx];
                    oklab.l = new_l;

                    let lin_rgb: LinSrgb = oklab.into_color();
                    if lin_rgb.red >= 0.0 && lin_rgb.red <= 1.0
                        && lin_rgb.green >= 0.0 && lin_rgb.green <= 1.0
                        && lin_rgb.blue >= 0.0 && lin_rgb.blue <= 1.0
                    {
                        let out_srgb: Srgb = lin_rgb.into_color();
                        chunk[0] = (out_srgb.red * 255.0).round().clamp(0.0, 255.0) as u8;
                        chunk[1] = (out_srgb.green * 255.0).round().clamp(0.0, 255.0) as u8;
                        chunk[2] = (out_srgb.blue * 255.0).round().clamp(0.0, 255.0) as u8;
                        return;
                    }

                    let mut t_low = 0.0f32;
                    let mut t_high = 1.0f32;
                    let mut best_lin = lin_rgb;

                    for _ in 0..10 {
                        let t_mid = 0.5 * (t_low + t_high);
                        let test_oklab = Oklab::new(new_l, oklab.a * t_mid, oklab.b * t_mid);
                        let test_lin: LinSrgb = test_oklab.into_color();
                        if test_lin.red >= 0.0 && test_lin.red <= 1.0
                            && test_lin.green >= 0.0 && test_lin.green <= 1.0
                            && test_lin.blue >= 0.0 && test_lin.blue <= 1.0
                        {
                            t_low = t_mid;
                            best_lin = test_lin;
                        } else {
                            t_high = t_mid;
                        }
                    }

                    let out_srgb: Srgb = best_lin.into_color();
                    chunk[0] = (out_srgb.red * 255.0).round().clamp(0.0, 255.0) as u8;
                    chunk[1] = (out_srgb.green * 255.0).round().clamp(0.0, 255.0) as u8;
                    chunk[2] = (out_srgb.blue * 255.0).round().clamp(0.0, 255.0) as u8;
                });
        }
        _ => {}
    }

    if let Some((gr, gg, gb)) = plan.paper_cast_gains {
        apply_paper_cast_ramp(image, gr, gg, gb, plan.paper_cast_ramp_start);
    }
}

// ----------------------------------------------------------------------------
// Monochrome & Mixed Page Correction (RGB LUT with Hermite Smoothstep)
// ----------------------------------------------------------------------------

fn build_channel_lut(bp: u8, wp: u8, gamma: f32) -> [u8; 256] {
    let inv_gamma = if (gamma - 1.0).abs() > 0.001 && gamma > 0.1 {
        1.0 / gamma
    } else {
        1.0
    };

    let range = (wp as f32 - bp as f32).max(1.0);
    let mut res = [0u8; 256];
    for y in 0..=255 {
        if (y as u8) <= bp {
            res[y] = 0;
        } else if (y as u8) >= wp {
            res[y] = 255;
        } else {
            let norm = (y as f32 - bp as f32) / range;
            let mapped = if inv_gamma != 1.0 {
                norm.powf(inv_gamma)
            } else {
                norm
            };
            res[y] = (mapped * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
    res
}

pub fn apply_monochrome_mixed_correction(
    image: &mut DynamicImage,
    plan: &CorrectionPlan,
    classification: PageClassification,
) {
    let bp_r = plan.per_channel_black.map(|p| p.0).unwrap_or(plan.black_point);
    let bp_g = plan.per_channel_black.map(|p| p.1).unwrap_or(plan.black_point);
    let bp_b = plan.per_channel_black.map(|p| p.2).unwrap_or(plan.black_point);

    let lut_r = build_channel_lut(bp_r, plan.white_point, plan.gamma);
    let lut_g = build_channel_lut(bp_g, plan.white_point, plan.gamma);
    let lut_b = build_channel_lut(bp_b, plan.white_point, plan.gamma);

    match image {
        DynamicImage::ImageLuma8(luma) => {
            for p in &mut **luma {
                *p = lut_r[*p as usize];
            }
        }
        DynamicImage::ImageLumaA8(luma_a) => {
            for chunk in (&mut **luma_a).chunks_exact_mut(2) {
                if chunk[1] >= 128 {
                    chunk[0] = lut_r[chunk[0] as usize];
                }
            }
        }
        DynamicImage::ImageRgb8(rgb) => {
            for chunk in (&mut **rgb).chunks_exact_mut(3) {
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];
                let chroma = r.max(g).max(b) - r.min(g).min(b);

                if classification == PageClassification::Mixed && chroma >= 35 {
                    continue; // Preserve colored logos / titles completely
                }

                let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8;
                let low_thresh = if y <= 50 { 25.0 } else { 12.0 };

                if (chroma as f32) <= low_thresh {
                    chunk[0] = lut_r[r as usize];
                    chunk[1] = lut_g[g as usize];
                    chunk[2] = lut_b[b as usize];
                } else {
                    let w = (35.0 - chroma as f32) / (35.0 - low_thresh);
                    let w_smooth = w * w * (3.0 - 2.0 * w);
                    chunk[0] = (w_smooth * lut_r[r as usize] as f32 + (1.0 - w_smooth) * r as f32).round().clamp(0.0, 255.0) as u8;
                    chunk[1] = (w_smooth * lut_g[g as usize] as f32 + (1.0 - w_smooth) * g as f32).round().clamp(0.0, 255.0) as u8;
                    chunk[2] = (w_smooth * lut_b[b as usize] as f32 + (1.0 - w_smooth) * b as f32).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        DynamicImage::ImageRgba8(rgba) => {
            for chunk in (&mut **rgba).chunks_exact_mut(4) {
                if chunk[3] < 128 {
                    continue;
                }
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];
                let chroma = r.max(g).max(b) - r.min(g).min(b);

                if classification == PageClassification::Mixed && chroma >= 35 {
                    continue;
                }

                let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8;
                let low_thresh = if y <= 50 { 25.0 } else { 12.0 };

                if (chroma as f32) <= low_thresh {
                    chunk[0] = lut_r[r as usize];
                    chunk[1] = lut_g[g as usize];
                    chunk[2] = lut_b[b as usize];
                } else {
                    let w = (35.0 - chroma as f32) / (35.0 - low_thresh);
                    let w_smooth = w * w * (3.0 - 2.0 * w);
                    chunk[0] = (w_smooth * lut_r[r as usize] as f32 + (1.0 - w_smooth) * r as f32).round().clamp(0.0, 255.0) as u8;
                    chunk[1] = (w_smooth * lut_g[g as usize] as f32 + (1.0 - w_smooth) * g as f32).round().clamp(0.0, 255.0) as u8;
                    chunk[2] = (w_smooth * lut_b[b as usize] as f32 + (1.0 - w_smooth) * b as f32).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        _ => {}
    }

    if let Some((gr, gg, gb)) = plan.paper_cast_gains {
        apply_paper_cast_ramp(image, gr, gg, gb, plan.paper_cast_ramp_start);
    }
}

pub fn apply_paper_cast_ramp(
    image: &mut DynamicImage,
    gr: f32,
    gg: f32,
    gb: f32,
    ramp_start: u8,
) {
    let ramp_range = (255.0 - ramp_start as f32).max(1.0);

    match image {
        DynamicImage::ImageRgb8(rgb) => {
            for chunk in (&mut **rgb).chunks_exact_mut(3) {
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];
                let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8;
                if y >= ramp_start {
                    let factor = ((y as f32 - ramp_start as f32) / ramp_range).clamp(0.0, 1.0);
                    chunk[0] = (r as f32 * (1.0 + (gr - 1.0) * factor)).round().clamp(0.0, 255.0) as u8;
                    chunk[1] = (g as f32 * (1.0 + (gg - 1.0) * factor)).round().clamp(0.0, 255.0) as u8;
                    chunk[2] = (b as f32 * (1.0 + (gb - 1.0) * factor)).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        DynamicImage::ImageRgba8(rgba) => {
            for chunk in (&mut **rgba).chunks_exact_mut(4) {
                if chunk[3] < 128 {
                    continue;
                }
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];
                let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8;
                if y >= ramp_start {
                    let factor = ((y as f32 - ramp_start as f32) / ramp_range).clamp(0.0, 1.0);
                    chunk[0] = (r as f32 * (1.0 + (gr - 1.0) * factor)).round().clamp(0.0, 255.0) as u8;
                    chunk[1] = (g as f32 * (1.0 + (gg - 1.0) * factor)).round().clamp(0.0, 255.0) as u8;
                    chunk[2] = (b as f32 * (1.0 + (gb - 1.0) * factor)).round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        _ => {}
    }
}

// ============================================================================
// Quality Guard Verification Pipeline (Section 3e)
// ============================================================================

pub fn compute_luminance_ssim(orig: &DynamicImage, leveled: &DynamicImage) -> f32 {
    let (w, h) = orig.dimensions();
    if w == 0 || h == 0 {
        return 1.0;
    }
    let orig_luma = orig.to_luma8();
    let lev_luma = leveled.to_luma8();

    let c1 = 6.5025f64; // (0.01 * 255)^2
    let c2 = 58.5225f64; // (0.03 * 255)^2

    let block_size = 8u32;
    let blocks_x = w / block_size;
    let blocks_y = h / block_size;

    if blocks_x == 0 || blocks_y == 0 {
        return 1.0;
    }

    let ssim_sum: f64 = (0..blocks_y)
        .into_par_iter()
        .map(|by| {
            let mut row_ssim = 0.0f64;
            let y_start = by * block_size;
            for bx in 0..blocks_x {
                let x_start = bx * block_size;
                let mut sum_x = 0.0f64;
                let mut sum_y = 0.0f64;
                let mut sum_x_sq = 0.0f64;
                let mut sum_y_sq = 0.0f64;
                let mut sum_xy = 0.0f64;
                let n = (block_size * block_size) as f64;

                for dy in 0..block_size {
                    for dx in 0..block_size {
                        let px = orig_luma.get_pixel(x_start + dx, y_start + dy)[0] as f64;
                        let py = lev_luma.get_pixel(x_start + dx, y_start + dy)[0] as f64;
                        sum_x += px;
                        sum_y += py;
                        sum_x_sq += px * px;
                        sum_y_sq += py * py;
                        sum_xy += px * py;
                    }
                }

                let mu_x = sum_x / n;
                let mu_y = sum_y / n;
                let sigma_x_sq = (sum_x_sq / n) - (mu_x * mu_x);
                let sigma_y_sq = (sum_y_sq / n) - (mu_y * mu_y);
                let sigma_xy = (sum_xy / n) - (mu_x * mu_y);

                let numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * sigma_xy + c2);
                let denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x_sq + sigma_y_sq + c2);

                let block_ssim = if denominator > 0.0 {
                    numerator / denominator
                } else {
                    1.0
                };
                row_ssim += block_ssim;
            }
            row_ssim
        })
        .sum();

    (ssim_sum / (blocks_x * blocks_y) as f64) as f32
}

pub fn compute_strided_gradient_ncc_parallel(
    original: &DynamicImage,
    leveled: &DynamicImage,
    stride: usize,
) -> f32 {
    let (w, h) = original.dimensions();
    if w < 3 || h < 3 {
        return 1.0;
    }
    let orig_luma = original.to_luma8();
    let lev_luma = leveled.to_luma8();

    let y_indices: Vec<u32> = (1..h - 1).step_by(stride.max(1)).collect();

    let (sum_prod, sum_sq1, sum_sq2) = y_indices
        .into_par_iter()
        .map(|y| {
            let mut row_prod = 0.0f64;
            let mut row_sq1 = 0.0f64;
            let mut row_sq2 = 0.0f64;

            for x in (1..w - 1).step_by(stride.max(1)) {
                let o00 = orig_luma.get_pixel(x - 1, y - 1)[0] as f64;
                let o01 = orig_luma.get_pixel(x, y - 1)[0] as f64;
                let o02 = orig_luma.get_pixel(x + 1, y - 1)[0] as f64;
                let o10 = orig_luma.get_pixel(x - 1, y)[0] as f64;
                let o12 = orig_luma.get_pixel(x + 1, y)[0] as f64;
                let o20 = orig_luma.get_pixel(x - 1, y + 1)[0] as f64;
                let o21 = orig_luma.get_pixel(x, y + 1)[0] as f64;
                let o22 = orig_luma.get_pixel(x + 1, y + 1)[0] as f64;

                let gx1 = (o02 + 2.0 * o12 + o22) - (o00 + 2.0 * o10 + o20);
                let gy1 = (o20 + 2.0 * o21 + o22) - (o00 + 2.0 * o01 + o02);

                let l00 = lev_luma.get_pixel(x - 1, y - 1)[0] as f64;
                let l01 = lev_luma.get_pixel(x, y - 1)[0] as f64;
                let l02 = lev_luma.get_pixel(x + 1, y - 1)[0] as f64;
                let l10 = lev_luma.get_pixel(x - 1, y)[0] as f64;
                let l12 = lev_luma.get_pixel(x + 1, y)[0] as f64;
                let l20 = lev_luma.get_pixel(x - 1, y + 1)[0] as f64;
                let l21 = lev_luma.get_pixel(x, y + 1)[0] as f64;
                let l22 = lev_luma.get_pixel(x + 1, y + 1)[0] as f64;

                let gx2 = (l02 + 2.0 * l12 + l22) - (l00 + 2.0 * l10 + l20);
                let gy2 = (l20 + 2.0 * l21 + l22) - (l00 + 2.0 * l01 + l02);

                let mag1 = (gx1 * gx1 + gy1 * gy1).sqrt();
                let mag2 = (gx2 * gx2 + gy2 * gy2).sqrt();
                if mag1 >= 12.0 || mag2 >= 12.0 {
                    row_prod += gx1 * gx2 + gy1 * gy2;
                    row_sq1 += gx1 * gx1 + gy1 * gy1;
                    row_sq2 += gx2 * gx2 + gy2 * gy2;
                }
            }
            (row_prod, row_sq1, row_sq2)
        })
        .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

    let denom = (sum_sq1 * sum_sq2).sqrt();
    if denom > 1e-6 {
        (sum_prod / denom).clamp(-1.0, 1.0) as f32
    } else {
        1.0
    }
}

pub fn compute_active_block_contrast_ratios(
    original: &DynamicImage,
    leveled: &DynamicImage,
    block_size: u32,
    min_dynamic_range: u8,
    epsilon: f32,
) -> Vec<f32> {
    let (w, h) = original.dimensions();
    let orig_luma = original.to_luma8();
    let lev_luma = leveled.to_luma8();

    let blocks_x = w / block_size;
    let blocks_y = h / block_size;

    if blocks_x == 0 || blocks_y == 0 {
        return Vec::new();
    }

    (0..blocks_y)
        .into_par_iter()
        .flat_map(|by| {
            let y_start = by * block_size;
            let mut row_ratios = Vec::new();
            for bx in 0..blocks_x {
                let x_start = bx * block_size;

                let mut o_min = 255u8;
                let mut o_max = 0u8;
                let mut l_min = 255u8;
                let mut l_max = 0u8;

                for dy in 0..block_size {
                    for dx in 0..block_size {
                        let po = orig_luma.get_pixel(x_start + dx, y_start + dy)[0];
                        let pl = lev_luma.get_pixel(x_start + dx, y_start + dy)[0];
                        o_min = o_min.min(po);
                        o_max = o_max.max(po);
                        l_min = l_min.min(pl);
                        l_max = l_max.max(pl);
                    }
                }

                if (o_max as i32 - o_min as i32) < min_dynamic_range as i32 {
                    continue; // Skip blank gutters / solid black borders
                }

                let c_orig = (o_max as f32 - o_min as f32) / (o_max as f32 + o_min as f32 + epsilon);
                let c_lev = (l_max as f32 - l_min as f32) / (l_max as f32 + l_min as f32 + epsilon);

                if c_orig > 1e-4 {
                    row_ratios.push(c_lev / c_orig);
                }
            }
            row_ratios
        })
        .collect()
}

pub fn compute_image_luminance_histogram(image: &DynamicImage) -> [u32; 256] {
    let mut hist = [0u32; 256];
    match image {
        DynamicImage::ImageLuma8(luma) => {
            for &p in luma.as_raw() {
                hist[p as usize] += 1;
            }
        }
        DynamicImage::ImageLumaA8(luma_a) => {
            for chunk in luma_a.as_raw().chunks_exact(2) {
                if chunk[1] >= 128 {
                    hist[chunk[0] as usize] += 1;
                }
            }
        }
        DynamicImage::ImageRgb8(rgb) => {
            for chunk in rgb.as_raw().chunks_exact(3) {
                let y = ((54 * chunk[0] as u32 + 183 * chunk[1] as u32 + 19 * chunk[2] as u32 + 128) >> 8) as usize;
                hist[y] += 1;
            }
        }
        DynamicImage::ImageRgba8(rgba) => {
            for chunk in rgba.as_raw().chunks_exact(4) {
                if chunk[3] >= 128 {
                    let y = ((54 * chunk[0] as u32 + 183 * chunk[1] as u32 + 19 * chunk[2] as u32 + 128) >> 8) as usize;
                    hist[y] += 1;
                }
            }
        }
        _ => {
            let rgb = image.to_rgb8();
            for chunk in rgb.as_raw().chunks_exact(3) {
                let y = ((54 * chunk[0] as u32 + 183 * chunk[1] as u32 + 19 * chunk[2] as u32 + 128) >> 8) as usize;
                hist[y] += 1;
            }
        }
    }
    hist
}

pub fn quality_tier_1(
    orig_hist: &[u32; 256],
    lev_hist: &[u32; 256],
    original: &DynamicImage,
    leveled: &DynamicImage,
    plan: &CorrectionPlan,
    total_pixels: usize,
    config: &AutoLevelsConfig,
) -> TierResult {
    let mut metrics = HashMap::new();

    if config.quality_guards.tier1.check_tail_erosion && plan.black_point > 0 {
        let bp = plan.black_point as usize;
        let bp_plus_4 = (bp + 4).min(255);
        let orig_slope = orig_hist[bp_plus_4] as i64 - orig_hist[bp] as i64;
        let lev_slope = lev_hist[4] as i64 - lev_hist[0] as i64;

        if orig_slope > (total_pixels as f64 * 0.01).round() as i64 && lev_slope < 0 {
            let mut res = TierResult::reject(1, "Midtone gradient crushed into black floor");
            res.metrics.insert("orig_slope".into(), orig_slope as f64);
            res.metrics.insert("lev_slope".into(), lev_slope as f64);
            return res;
        }

        if orig_slope > 0 {
            let r_s = lev_slope as f32 / orig_slope as f32;
            metrics.insert("r_s".into(), r_s as f64);
            if orig_slope > (total_pixels as f64 * 0.01).round() as i64 && r_s < 0.40 {
                let mut res = TierResult::reject(1, format!("Slope ratio {:.3} < 0.40 (tail erosion)", r_s));
                res.metrics = metrics;
                return res;
            }
        }
    }

    let (sum_sq, valid_count) = match (original, leveled) {
        (DynamicImage::ImageRgba8(orig_rgba), DynamicImage::ImageRgba8(lev_rgba)) => {
            let (s, c) = orig_rgba
                .as_raw()
                .par_chunks_exact(4)
                .zip(lev_rgba.as_raw().par_chunks_exact(4))
                .filter(|(o, _)| o[3] >= 128)
                .map(|(o, l)| {
                    let yo = ((54 * o[0] as u32 + 183 * o[1] as u32 + 19 * o[2] as u32 + 128) >> 8) as f64;
                    let yl = ((54 * l[0] as u32 + 183 * l[1] as u32 + 19 * l[2] as u32 + 128) >> 8) as f64;
                    let diff = yo - yl;
                    (diff * diff, 1u64)
                })
                .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            (s, c as f64)
        }
        (DynamicImage::ImageLumaA8(orig_la), DynamicImage::ImageLumaA8(lev_la)) => {
            let (s, c) = orig_la
                .as_raw()
                .par_chunks_exact(2)
                .zip(lev_la.as_raw().par_chunks_exact(2))
                .filter(|(o, _)| o[1] >= 128)
                .map(|(o, l)| {
                    let diff = o[0] as f64 - l[0] as f64;
                    (diff * diff, 1u64)
                })
                .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            (s, c as f64)
        }
        _ => {
            let orig_luma = original.to_luma8();
            let lev_luma = leveled.to_luma8();
            let total = (orig_luma.width() * orig_luma.height()) as f64;
            let s: f64 = orig_luma
                .as_raw()
                .par_iter()
                .zip(lev_luma.as_raw().par_iter())
                .map(|(&a, &b)| {
                    let diff = a as f64 - b as f64;
                    diff * diff
                })
                .sum();
            (s, total)
        }
    };

    if valid_count == 0.0 {
        return TierResult::pass(1);
    }

    let mse = sum_sq / valid_count;
    let psnr = if mse > 1e-6 {
        10.0 * (65025.0 / mse).log10()
    } else {
        99.0
    };

    metrics.insert("mse".into(), mse);
    metrics.insert("psnr".into(), psnr);

    if mse > config.quality_guards.tier1.mse_threshold as f64 {
        let mut res = TierResult::reject(1, format!("MSE {:.1} exceeded threshold {:.1}", mse, config.quality_guards.tier1.mse_threshold));
        res.metrics = metrics;
        return res;
    }

    if psnr < config.quality_guards.tier1.psnr_min as f64 {
        let mut res = TierResult::reject(1, format!("PSNR {:.1}dB below minimum {:.1}dB", psnr, config.quality_guards.tier1.psnr_min));
        res.metrics = metrics;
        return res;
    }

    let mut res = TierResult::pass(1);
    res.metrics = metrics;
    res
}

pub fn quality_tier_2(
    original: &DynamicImage,
    leveled: &DynamicImage,
    classification: PageClassification,
    plan: &CorrectionPlan,
    config: &AutoLevelsConfig,
) -> TierResult {
    let mut metrics = HashMap::new();

    let ssim = compute_luminance_ssim(original, leveled);
    metrics.insert("ssim".into(), ssim as f64);

    if ssim < config.quality_guards.tier2.ssim_threshold {
        let mut res = TierResult::reject(
            2,
            format!("SSIM {:.3} below threshold {:.2}", ssim, config.quality_guards.tier2.ssim_threshold),
        );
        res.metrics = metrics;
        return res;
    }

    if classification == PageClassification::Color {
        let orig_rgb = original.to_rgb8();
        let lev_rgb = leveled.to_rgb8();
        let total_pixels = (orig_rgb.width() * orig_rgb.height()) as usize;
        let step = (total_pixels / 2000).max(1);

        let mut chroma_deltas = Vec::new();
        let has_paper_cast = plan.paper_cast_gains.is_some();
        let ramp_start = plan.paper_cast_ramp_start;

        for (op, lp) in orig_rgb.pixels().step_by(step).zip(lev_rgb.pixels().step_by(step)) {
            let (or, og, ob) = (op[0], op[1], op[2]);
            let orig_y = ((54 * or as u32 + 183 * og as u32 + 19 * ob as u32 + 128) >> 8) as u8;
            let orig_chroma = or.max(og).max(ob) - or.min(og).min(ob);

            if has_paper_cast && orig_y >= ramp_start && orig_chroma <= 55 {
                continue;
            }

            let srgb_orig = Srgb::new(or as f32 / 255.0, og as f32 / 255.0, ob as f32 / 255.0);
            let srgb_lev = Srgb::new(lp[0] as f32 / 255.0, lp[1] as f32 / 255.0, lp[2] as f32 / 255.0);

            let lab_orig: Lab = srgb_orig.into_color();
            let lab_lev: Lab = srgb_lev.into_color();

            let delta_a = lab_orig.a - lab_lev.a;
            let delta_b = lab_orig.b - lab_lev.b;
            let delta_chroma = (delta_a * delta_a + delta_b * delta_b).sqrt();
            chroma_deltas.push(delta_chroma);
        }

        if !chroma_deltas.is_empty() {
            chroma_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p95_idx = ((chroma_deltas.len() as f32 * 0.95).floor() as usize).min(chroma_deltas.len() - 1);
            let med_idx = chroma_deltas.len() / 2;

            let p95_delta = chroma_deltas[p95_idx];
            let median_delta = chroma_deltas[med_idx];

            metrics.insert("chroma_delta_p95".into(), p95_delta as f64);
            metrics.insert("chroma_delta_median".into(), median_delta as f64);

            if p95_delta > config.quality_guards.tier2.chroma_delta_p95 {
                let mut res = TierResult::reject(
                    2,
                    format!("Color drift: p95 Chroma Delta={:.2} > {:.2}", p95_delta, config.quality_guards.tier2.chroma_delta_p95),
                );
                res.metrics = metrics;
                return res;
            }

            if median_delta > config.quality_guards.tier2.chroma_delta_median {
                let mut res = TierResult::reject(
                    2,
                    format!("Color drift: median Chroma Delta={:.2} > {:.2}", median_delta, config.quality_guards.tier2.chroma_delta_median),
                );
                res.metrics = metrics;
                return res;
            }
        }
    }

    let mut res = TierResult::pass(2);
    res.metrics = metrics;
    res
}

pub fn quality_tier_3(
    original: &DynamicImage,
    leveled: &DynamicImage,
    config: &AutoLevelsConfig,
) -> TierResult {
    let mut metrics = HashMap::new();

    let stride = config.quality_guards.tier3.stride as usize;
    let ncc = compute_strided_gradient_ncc_parallel(original, leveled, stride);
    metrics.insert("gradient_ncc".into(), ncc as f64);

    if ncc < config.quality_guards.tier3.gradient_ncc {
        let mut res = TierResult::reject(
            3,
            format!("Gradient NCC {:.3} damaged fine linework/screentones", ncc),
        );
        res.metrics = metrics;
        return res;
    }

    let block_ratios = compute_active_block_contrast_ratios(original, leveled, 32, 25, 1.0);
    if !block_ratios.is_empty() {
        let mut sorted = block_ratios;
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p05_idx = ((sorted.len() as f32 * 0.05).floor() as usize).min(sorted.len() - 1);
        let p05_ratio = sorted[p05_idx];
        metrics.insert("contrast_ratio_p05".into(), p05_ratio as f64);

        if p05_ratio < config.quality_guards.tier3.min_contrast_ratio {
            let mut res = TierResult::reject(
                3,
                format!("Local contrast collapsed: p05 ratio {:.2} < {:.2}", p05_ratio, config.quality_guards.tier3.min_contrast_ratio),
            );
            res.metrics = metrics;
            return res;
        }
    }

    let mut res = TierResult::pass(3);
    res.metrics = metrics;
    res
}

// ============================================================================
// Main Level & Verify Pipeline (Pre-Actor / CPU)
// ============================================================================

pub fn level_and_verify(
    image: DynamicImage,
    original_bytes: &Bytes,
    book_profile: Option<&BookProfile>,
    config: &AutoLevelsConfig,
) -> (DynamicImage, LevelInfo, QualityVerdict) {
    level_and_verify_with_context(image, original_bytes, None, None, book_profile, config)
}

pub fn level_and_verify_with_context(
    mut image: DynamicImage,
    _original_bytes: &Bytes,
    book_id: Option<&str>,
    page_number: Option<usize>,
    book_profile: Option<&BookProfile>,
    config: &AutoLevelsConfig,
) -> (DynamicImage, LevelInfo, QualityVerdict) {
    let b_id = book_id.unwrap_or("-");
    let p_num = page_number.map(|p| p.to_string()).unwrap_or_else(|| "-".to_string());

    if !config.enabled || config.mode == AutoLevelsMode::Off {
        return (image, LevelInfo::default(), QualityVerdict::pass());
    }

    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return (image, LevelInfo::default(), QualityVerdict::pass());
    }

    // Edge case 6.8: Thumbnail bypass (<400px in both dimensions)
    if width < 400 && height < 400 {
        return (image, LevelInfo::default(), QualityVerdict::pass());
    }

    let start_time = Instant::now();
    let analysis = analyze_page(&image, page_number.unwrap_or(0), config);

    // Filter by mode
    if config.mode == AutoLevelsMode::Manga && analysis.classification == PageClassification::Color {
        return (
            image,
            LevelInfo {
                was_leveled: false,
                is_monochrome: false,
                classification: analysis.classification,
                black_point: 0,
                white_point: 255,
                channel_black_points: None,
                quality_verdict: Some(QualityVerdict::pass()),
            },
            QualityVerdict::pass(),
        );
    }

    if config.mode == AutoLevelsMode::Color && analysis.classification != PageClassification::Color {
        return (
            image,
            LevelInfo {
                was_leveled: false,
                is_monochrome: analysis.classification == PageClassification::Monochrome,
                classification: analysis.classification,
                black_point: 0,
                white_point: 255,
                channel_black_points: None,
                quality_verdict: Some(QualityVerdict::pass()),
            },
            QualityVerdict::pass(),
        );
    }

    // Modulate defect confidence using BookProfile if available
    let mut defects = analysis.defects.clone();
    if let Some(profile) = book_profile {
        adjust_confidence_with_book_context(&mut defects, analysis.classification, profile);
    }

    // Generate initial correction plan
    let mut plan = generate_correction_plan(&defects, analysis.classification, config);

    // Per-channel dark shelf alignment check
    let channel_shelf = detect_per_channel_shelf(&analysis, &plan);
    if let Some(pcs) = channel_shelf {
        plan.per_channel_black = Some(pcs);
    }

    let is_monochrome = analysis.classification == PageClassification::Monochrome;

    if plan.skip_correction {
        let elapsed_us = start_time.elapsed().as_micros();
        info!(
            "[AutoLevels] book_id={} page={} class={:?} black={}->{} white={}->{} oklab={} leveled={} time_us={}",
            b_id, p_num, analysis.classification,
            analysis.black_floor, plan.black_point,
            analysis.white_ceiling, plan.white_point,
            plan.use_oklab, false, elapsed_us
        );
        return (
            image,
            LevelInfo {
                was_leveled: false,
                is_monochrome,
                classification: analysis.classification,
                black_point: 0,
                white_point: 255,
                channel_black_points: None,
                quality_verdict: Some(QualityVerdict::pass()),
            },
            QualityVerdict::pass(),
        );
    }

    // Keep pristine original reference for rollback / verification
    let pristine_reference = image.clone();

    // Helper for rollback: return pristine decoded and oriented frame
    let redecode_pristine = || -> DynamicImage {
        pristine_reference.clone()
    };

    // Apply correction plan in-place to image
    if plan.use_oklab {
        apply_color_correction_oklab(&mut image, &plan);
    } else {
        apply_monochrome_mixed_correction(&mut image, &plan, analysis.classification);
    }

    // Compute leveled luminance histogram for Tier 1 slope erosion check (respecting alpha transparency)
    let lev_hist = compute_image_luminance_histogram(&image);

    // Quality Guard Evaluation
    let t1 = quality_tier_1(
        &analysis.luminance_histogram,
        &lev_hist,
        &pristine_reference,
        &image,
        &plan,
        analysis.total_pixels,
        config,
    );

    let mut tier_results = Vec::new();
    if t1.outcome == Outcome::Reject {
        warn!(
            "[QualityGuard] REJECT book_id={} page={} tier=1 reason=\"{}\" metrics={:?}",
            b_id, p_num,
            t1.reason.as_deref().unwrap_or(""),
            t1.metrics
        );
        tier_results.push(t1);
        let verdict = QualityVerdict::reject(tier_results, false);
        let rolled_back = redecode_pristine();
        let elapsed_us = start_time.elapsed().as_micros();
        info!(
            "[AutoLevels] book_id={} page={} class={:?} black={}->{} white={}->{} oklab={} leveled={} time_us={}",
            b_id, p_num, analysis.classification,
            analysis.black_floor, plan.black_point,
            analysis.white_ceiling, plan.white_point,
            plan.use_oklab, false, elapsed_us
        );
        return (
            rolled_back,
            LevelInfo {
                was_leveled: false,
                is_monochrome,
                classification: analysis.classification,
                black_point: 0,
                white_point: 255,
                channel_black_points: None,
                quality_verdict: Some(verdict.clone()),
            },
            verdict,
        );
    }
    tier_results.push(t1);

    let t2 = quality_tier_2(&pristine_reference, &image, analysis.classification, &plan, config);
    let mut retry_needed = false;
    if t2.outcome == Outcome::Reject {
        warn!(
            "[QualityGuard] REJECT book_id={} page={} tier=2 reason=\"{}\" metrics={:?}",
            b_id, p_num,
            t2.reason.as_deref().unwrap_or(""),
            t2.metrics
        );
        tier_results.push(t2);
        retry_needed = true;
    } else {
        tier_results.push(t2);
    }

    if !retry_needed {
        let t3 = quality_tier_3(&pristine_reference, &image, config);
        if t3.outcome == Outcome::Reject {
            warn!(
                "[QualityGuard] REJECT book_id={} page={} tier=3 reason=\"{}\" metrics={:?}",
                b_id, p_num,
                t3.reason.as_deref().unwrap_or(""),
                t3.metrics
            );
            tier_results.push(t3);
            retry_needed = true;
        } else {
            tier_results.push(t3);
        }
    }

    // Adaptive Attenuation Policy: Retry once with 50% strength on Tier 2 or 3 rejection
    if retry_needed {
        let mut attenuated_plan = plan.clone();
        attenuated_plan.black_point = (plan.black_point as f32 * 0.5).round() as u8;
        attenuated_plan.white_point = 255 - ((255.0 - plan.white_point as f32) * 0.5).round() as u8;
        if let Some((r, g, b)) = plan.per_channel_black {
            attenuated_plan.per_channel_black = Some((
                (r as f32 * 0.5).round() as u8,
                (g as f32 * 0.5).round() as u8,
                (b as f32 * 0.5).round() as u8,
            ));
        }
        if let Some((gr, gg, gb)) = plan.paper_cast_gains {
            attenuated_plan.paper_cast_gains = Some((
                1.0 + (gr - 1.0) * 0.5,
                1.0 + (gg - 1.0) * 0.5,
                1.0 + (gb - 1.0) * 0.5,
            ));
        }

        // Re-level from pristine image
        let mut retry_image = redecode_pristine();
        if attenuated_plan.use_oklab {
            apply_color_correction_oklab(&mut retry_image, &attenuated_plan);
        } else {
            apply_monochrome_mixed_correction(&mut retry_image, &attenuated_plan, analysis.classification);
        }

        let retry_lev_hist = compute_image_luminance_histogram(&retry_image);

        let rt1 = quality_tier_1(
            &analysis.luminance_histogram,
            &retry_lev_hist,
            &pristine_reference,
            &retry_image,
            &attenuated_plan,
            analysis.total_pixels,
            config,
        );
        let rt2 = quality_tier_2(&pristine_reference, &retry_image, analysis.classification, &attenuated_plan, config);
        let rt3 = quality_tier_3(&pristine_reference, &retry_image, config);

        if rt1.outcome == Outcome::Pass && rt2.outcome == Outcome::Pass && rt3.outcome == Outcome::Pass {
            let elapsed_us = start_time.elapsed().as_micros();
            info!(
                "[AutoLevels] book_id={} page={} class={:?} black={}->{} white={}->{} oklab={} leveled=true (Attenuated 50%) time_us={}",
                b_id, p_num,
                analysis.classification,
                analysis.black_floor,
                attenuated_plan.black_point,
                analysis.white_ceiling,
                attenuated_plan.white_point,
                attenuated_plan.use_oklab,
                elapsed_us
            );
            let verdict = QualityVerdict {
                outcome: Outcome::Pass,
                attenuation_attempted: true,
                tier_results: vec![rt1, rt2, rt3],
            };
            return (
                retry_image,
                LevelInfo {
                    was_leveled: true,
                    is_monochrome,
                    classification: analysis.classification,
                    black_point: attenuated_plan.black_point,
                    white_point: attenuated_plan.white_point,
                    channel_black_points: attenuated_plan.per_channel_black.map(|p| [p.0, p.1, p.2]),
                    quality_verdict: Some(verdict.clone()),
                },
                verdict,
            );
        } else {
            warn!(
                "[QualityGuard] REJECT (Attenuated retry rejected) book_id={} page={}",
                b_id, p_num
            );
            let rolled_back = redecode_pristine();
            let verdict = QualityVerdict {
                outcome: Outcome::Reject,
                attenuation_attempted: true,
                tier_results: vec![rt1, rt2, rt3],
            };
            let elapsed_us = start_time.elapsed().as_micros();
            info!(
                "[AutoLevels] book_id={} page={} class={:?} black={}->{} white={}->{} oklab={} leveled=false time_us={}",
                b_id, p_num, analysis.classification,
                analysis.black_floor, plan.black_point,
                analysis.white_ceiling, plan.white_point,
                plan.use_oklab, elapsed_us
            );
            return (
                rolled_back,
                LevelInfo {
                    was_leveled: false,
                    is_monochrome,
                    classification: analysis.classification,
                    black_point: 0,
                    white_point: 255,
                    channel_black_points: None,
                    quality_verdict: Some(verdict.clone()),
                },
                verdict,
            );
        }
    }

    let elapsed_us = start_time.elapsed().as_micros();
    info!(
        "[AutoLevels] book_id={} page={} class={:?} black={}->{} white={}->{} oklab={} leveled=true time_us={}",
        b_id, p_num,
        analysis.classification,
        analysis.black_floor,
        plan.black_point,
        analysis.white_ceiling,
        plan.white_point,
        plan.use_oklab,
        elapsed_us
    );

    let verdict = QualityVerdict {
        outcome: Outcome::Pass,
        attenuation_attempted: false,
        tier_results,
    };

    (
        image,
        LevelInfo {
            was_leveled: true,
            is_monochrome,
            classification: analysis.classification,
            black_point: plan.black_point,
            white_point: plan.white_point,
            channel_black_points: plan.per_channel_black.map(|p| [p.0, p.1, p.2]),
            quality_verdict: Some(verdict.clone()),
        },
        verdict,
    )
}

/// Legacy / convenience wrapper around `level_and_verify`.
pub fn analyze_and_level_image(
    image: DynamicImage,
    config: &AutoLevelsConfig,
) -> (DynamicImage, LevelInfo) {
    let dummy_bytes = Bytes::new();
    let (img, info, _) = level_and_verify(image, &dummy_bytes, None, config);
    (img, info)
}
