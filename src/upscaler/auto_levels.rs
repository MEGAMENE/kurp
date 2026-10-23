#![allow(clippy::needless_range_loop, clippy::chunks_exact_to_as_chunks)]

use std::collections::HashMap;
use std::time::Instant;

use bytes::Bytes;
use image::{DynamicImage, GenericImageView};
use log::{info, warn};
use once_cell::sync::Lazy;
use palette::{IntoColor, LinSrgb, Oklab, Srgb};
use rayon::prelude::*;
use serde_derive::{Deserialize, Serialize};

use crate::config::app_config::{AutoLevelsConfig, AutoLevelsMode};
use crate::upscaler::book_profile::BookProfile;

// ============================================================================
// Precomputed Fast CIELAB & Dual-Boundary Sigmoid Taper Lookup Tables
// ============================================================================

pub struct FastLabTables {
    pub x_r: [u16; 256],
    pub x_g: [u16; 256],
    pub x_b: [u16; 256],
    pub y_r: [u16; 256],
    pub y_g: [u16; 256],
    pub y_b: [u16; 256],
    pub z_r: [u16; 256],
    pub z_g: [u16; 256],
    pub z_b: [u16; 256],
    pub lab_f: [f32; 16388],
    /// Pre-baked 2D fixed-point Q8.8 [0..=256] Dual-Boundary Sigmoid Chroma Gate
    /// Indexed by `[l_bin (0..=100)][c_bin (0..=255, step 0.25 C*)]`.
    pub taper_lut: [[u16; 256]; 101],
}

pub static FAST_LAB_TABLES: Lazy<FastLabTables> = Lazy::new(|| {
    let mut x_r = [0u16; 256];
    let mut x_g = [0u16; 256];
    let mut x_b = [0u16; 256];
    let mut y_r = [0u16; 256];
    let mut y_g = [0u16; 256];
    let mut y_b = [0u16; 256];
    let mut z_r = [0u16; 256];
    let mut z_g = [0u16; 256];
    let mut z_b = [0u16; 256];

    // D65 white-point normalized sRGB -> XYZ matrix rows (each row sums to 1.0)
    const SCALE: f64 = 16384.0;
    const CX_R: f64 = 0.4124564 / 0.95047;
    const CX_G: f64 = 0.3575761 / 0.95047;
    const CX_B: f64 = 0.1804375 / 0.95047;

    const CY_R: f64 = 0.2126729;
    const CY_G: f64 = 0.7151522;
    const CY_B: f64 = 0.0721750;

    const CZ_R: f64 = 0.0193339 / 1.08883;
    const CZ_G: f64 = 0.1191920 / 1.08883;
    const CZ_B: f64 = 0.9503041 / 1.08883;

    for v in 0..256 {
        let s = v as f64 / 255.0;
        let lin = if s <= 0.04045 {
            s / 12.92
        } else {
            ((s + 0.055) / 1.055).powf(2.4)
        };
        x_r[v] = (lin * CX_R * SCALE).round() as u16;
        x_g[v] = (lin * CX_G * SCALE).round() as u16;
        x_b[v] = (lin * CX_B * SCALE).round() as u16;

        y_r[v] = (lin * CY_R * SCALE).round() as u16;
        y_g[v] = (lin * CY_G * SCALE).round() as u16;
        y_b[v] = (lin * CY_B * SCALE).round() as u16;

        z_r[v] = (lin * CZ_R * SCALE).round() as u16;
        z_g[v] = (lin * CZ_G * SCALE).round() as u16;
        z_b[v] = (lin * CZ_B * SCALE).round() as u16;
    }

    let mut lab_f = [0.0f32; 16388];
    let delta: f64 = 6.0 / 29.0;
    let delta_cubed = delta * delta * delta;
    let factor = (1.0 / 3.0) * (29.0 / 6.0) * (29.0 / 6.0);
    let offset = 4.0 / 29.0;

    for (i, slot) in lab_f.iter_mut().enumerate() {
        let t = (i as f64 / SCALE).min(1.0);
        let f = if t > delta_cubed {
            t.cbrt()
        } else {
            factor * t + offset
        };
        *slot = f as f32;
    }

    let mut taper_lut = [[256u16; 256]; 101];
    for l_bin in 0..=100 {
        let l_star = l_bin as f64;
        let z = ((l_star - 75.0) / 5.0).clamp(-20.0, 20.0);
        let sigmoid_val = 1.0 / (1.0 + (-z).exp());
        let c_high = (24.0 * (1.0 - sigmoid_val)).max(6.0);
        let c_low = (c_high - 5.0).max(1.0);
        let denom = (c_high - c_low).max(1e-6);

        for c_bin in 0..256 {
            let c_star = c_bin as f64 * 0.25;
            let taper = ((c_star - c_low) / denom).clamp(0.0, 1.0);
            taper_lut[l_bin][c_bin] = (taper * 256.0).round() as u16;
        }
    }

    FastLabTables {
        x_r,
        x_g,
        x_b,
        y_r,
        y_g,
        y_b,
        z_r,
        z_g,
        z_b,
        lab_f,
        taper_lut,
    }
});

#[inline(always)]
pub fn rgb_to_lab_fast(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let t = &*FAST_LAB_TABLES;
    let (ri, gi, bi) = (r as usize, g as usize, b as usize);
    let yi = (t.y_r[ri] + t.y_g[gi] + t.y_b[bi]) as usize;
    let fy = t.lab_f[yi];
    let l_star = (116.0 * fy - 16.0).max(0.0);
    if r == g && g == b {
        return (l_star, 0.0, 0.0);
    }
    let xi = (t.x_r[ri] + t.x_g[gi] + t.x_b[bi]) as usize;
    let zi = (t.z_r[ri] + t.z_g[gi] + t.z_b[bi]) as usize;
    let fx = t.lab_f[xi];
    let fz = t.lab_f[zi];
    let a_star = 500.0 * (fx - fy);
    let b_star = 200.0 * (fy - fz);
    (l_star, a_star, b_star)
}

#[inline(always)]
pub fn compute_sigmoid_taper_exact(l_star: f32, c_star: f32) -> f32 {
    let t = &*FAST_LAB_TABLES;
    let l_bin = (l_star.round() as usize).min(100);
    let c_bin = ((c_star * 4.0).round() as usize).min(255);
    t.taper_lut[l_bin][c_bin] as f32 / 256.0
}

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
// Histogram Percentiles, Effective Black/White & Valley Validation (Upgrade B)
// ============================================================================

#[inline]
pub fn histogram_percentile(hist: &[u32; 256], total: u64, pct: f64) -> f32 {
    if total == 0 {
        return 0.0;
    }
    let target = ((total as f64) * (pct / 100.0)).ceil().max(1.0) as u64;
    let mut acc = 0u64;
    for (bin, &count) in hist.iter().enumerate() {
        acc += count as u64;
        if acc >= target {
            return bin as f32;
        }
    }
    255.0
}

#[inline]
pub fn nonzero_histogram_percentile(hist: &[u32; 256], pct: f64) -> (u64, f32) {
    let nz_total: u64 = hist[3..256].iter().map(|&c| c as u64).sum();
    if nz_total == 0 {
        return (0, 0.0);
    }
    let target = ((nz_total as f64) * (pct / 100.0)).ceil().max(1.0) as u64;
    let mut acc = 0u64;
    for bin in 3..256 {
        acc += hist[bin] as u64;
        if acc >= target {
            return (nz_total, bin as f32);
        }
    }
    (nz_total, 255.0)
}

/// Finds the effective artwork drawing black point when digital typesetting creates
/// an isolated spike near 0 while actual drawing/ink lines are washed out.
pub fn find_effective_black(hist_y: &[u32; 256], p1: f32) -> f32 {
    let total: u64 = hist_y.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return p1;
    }
    let zero_mass = (hist_y[0] as u64 + hist_y[1] as u64 + hist_y[2] as u64) as f64 / total as f64;
    if zero_mass > 0.0005 && zero_mass < 0.035 {
        let shadow_total: u64 = hist_y[3..120].iter().map(|&c| c as u64).sum();
        if shadow_total > 50 {
            let target = ((shadow_total as f64) * 0.02).ceil().max(1.0) as u64;
            let mut acc = 0u64;
            let mut idx = 3usize;
            for bin in 3..120 {
                acc += hist_y[bin] as u64;
                if acc >= target {
                    idx = bin;
                    break;
                }
            }
            if (idx as f32) > p1 + 4.0 {
                return p1.max(idx as f32);
            }
        }
    }
    p1
}

/// Finds the effective paper background white point when digital speech bubble fills
/// spike at 255 while actual paper is dull.
pub fn find_effective_white(hist_y: &[u32; 256], p99: f32) -> f32 {
    let total: u64 = hist_y.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return p99;
    }
    let hi_mass = (hist_y[253] as u64 + hist_y[254] as u64 + hist_y[255] as u64) as f64 / total as f64;
    if hi_mass > 0.0005 && hi_mass < 0.25 {
        let paper_total: u64 = hist_y[160..253].iter().map(|&c| c as u64).sum();
        let paper_mass = paper_total as f64 / total as f64;
        if paper_mass >= 0.15 && paper_mass > hi_mass {
            let target = ((paper_total as f64) * 0.98).ceil().max(1.0) as u64;
            let mut acc = 0u64;
            let mut idx_p98 = 252usize;
            for bin in 160..253 {
                acc += hist_y[bin] as u64;
                if acc >= target {
                    idx_p98 = bin;
                    break;
                }
            }
            return p99.min(idx_p98 as f32);
        }
    }
    p99
}

#[inline]
pub fn smooth_histogram_binomial_5(hist_y: &[u32; 256]) -> [f32; 256] {
    let mut h_smooth = [0.0f32; 256];
    for i in 0..256 {
        let m2 = if i >= 2 { hist_y[i - 2] as f32 } else { 0.0 };
        let m1 = if i >= 1 { hist_y[i - 1] as f32 } else { 0.0 };
        let c0 = hist_y[i] as f32;
        let p1 = if i + 1 < 256 { hist_y[i + 1] as f32 } else { 0.0 };
        let p2 = if i + 2 < 256 { hist_y[i + 2] as f32 } else { 0.0 };
        h_smooth[i] = (m2 + 4.0 * m1 + 6.0 * c0 + 4.0 * p1 + p2) * (1.0 / 16.0);
    }
    h_smooth
}

/// Validates candidate black floor stretch bound against high-key scene crushing.
/// Ensures that `cand_lo > 35.0` (up to `85.0`) is only permitted if a legitimate
/// drawing black peak exists in the shadow spectrum followed by a bimodal valley minimum.
pub fn validate_black_floor(
    hist_y: &[u32; 256],
    total_pixels: usize,
    cand_lo: f32,
    hi: f32,
) -> f32 {
    if total_pixels == 0 || cand_lo <= 0.0 || cand_lo > 85.0 {
        return 0.0;
    }

    // High-Key Scene Protection: require M_ink (2 < Y <= 65) >= 0.008
    let ink_mass_count: u64 = hist_y[3..=65].iter().map(|&c| c as u64).sum();
    let m_ink = ink_mass_count as f64 / total_pixels as f64;
    if m_ink < 0.008 {
        return 0.0;
    }

    if cand_lo <= 35.0 {
        return cand_lo;
    }

    validate_bimodal_valley_only(hist_y, total_pixels, cand_lo, hi)
}

/// Internal bimodal valley validator on the 5-tap binomial smoothed histogram.
fn validate_bimodal_valley_only(
    hist_y: &[u32; 256],
    total_pixels: usize,
    cand_lo: f32,
    hi: f32,
) -> f32 {
    if total_pixels == 0 || cand_lo <= 0.0 || cand_lo > 85.0 {
        return 0.0;
    }

    let h_smooth = smooth_histogram_binomial_5(hist_y);
    let lo_idx = cand_lo.floor() as usize;
    let search_peak_hi = (lo_idx + 15).min(85);
    // Start peak search at bin 5 if bins 0..=2 have a digital spike so the 5-tap kernel tail doesn't mask the drawing peak
    let start_peak = if hist_y[0] + hist_y[1] + hist_y[2] > 0 { 5 } else { 2 };
    if start_peak > search_peak_hi {
        return 0.0;
    }

    let mut h_peak = 0.0f32;
    let mut peak_idx = start_peak;
    for (j, &val) in h_smooth.iter().enumerate().take(search_peak_hi + 1).skip(start_peak) {
        if val > h_peak {
            h_peak = val;
            peak_idx = j;
        }
    }

    if h_peak < 0.001 * (total_pixels as f32) {
        return 0.0;
    }

    let search_valley_hi = (hi.floor() as usize).min(140);
    let start_valley = lo_idx.max(peak_idx);
    if start_valley >= search_valley_hi {
        return 0.0;
    }

    let mut min_valley = f32::MAX;
    let mut valley_idx = start_valley;
    for (j, &val) in h_smooth.iter().enumerate().take(search_valley_hi + 1).skip(start_valley) {
        if val < min_valley {
            min_valley = val;
            valley_idx = j;
        }
    }

    if min_valley < 0.35 * h_peak {
        cand_lo.min(valley_idx as f32)
    } else {
        0.0
    }
}

// ============================================================================
// Fritsch-Carlson Monotonic Cubic Spline 256-Entry Luminance LUT (Upgrade C)
// ============================================================================

/// Builds a strictly monotonic 256-element tone curve lookup table (LUT) using
/// Fritsch-Carlson cubic Hermite interpolation with inversion-proof knot parameterization.
pub fn build_monotonic_tone_lut(lo: f32, hi: f32) -> [u8; 256] {
    let lo_c = lo.clamp(0.0, 240.0) as f64;
    let hi_c = hi.clamp((lo_c as f32 + 15.0).max(15.0), 255.0) as f64;

    if lo_c < 2.0 && hi_c > 253.0 {
        let mut identity = [0u8; 256];
        for (i, slot) in identity.iter_mut().enumerate() {
            *slot = i as u8;
        }
        return identity;
    }

    let mut knots: Vec<(f64, f64)> = Vec::with_capacity(5);
    knots.push((0.0, 0.0));
    if lo_c >= 2.0 {
        knots.push((lo_c, 8.0));
    }
    knots.push(((lo_c + hi_c) * 0.5, 128.0));
    if hi_c <= 253.0 {
        knots.push((hi_c, 248.0));
    }
    knots.push((255.0, 255.0));

    let n = knots.len();
    let mut deltas = vec![0.0f64; n - 1];
    for k in 0..(n - 1) {
        let dx = (knots[k + 1].0 - knots[k].0).max(1e-6);
        deltas[k] = (knots[k + 1].1 - knots[k].1) / dx;
    }

    let mut d = vec![0.0f64; n];
    d[0] = deltas[0];
    d[n - 1] = deltas[n - 2];

    for k in 1..(n - 1) {
        if deltas[k - 1] * deltas[k] <= 0.0 {
            d[k] = 0.0;
        } else {
            d[k] = 2.0 * deltas[k - 1] * deltas[k] / (deltas[k - 1] + deltas[k]);
        }
    }

    for k in 0..(n - 1) {
        if deltas[k] == 0.0 {
            d[k] = 0.0;
            d[k + 1] = 0.0;
        } else {
            let alpha = d[k] / deltas[k];
            let beta = d[k + 1] / deltas[k];
            let hyp_sq = alpha * alpha + beta * beta;
            if hyp_sq > 9.0 {
                let tau = 3.0 / hyp_sq.sqrt();
                d[k] = tau * alpha * deltas[k];
                d[k + 1] = tau * beta * deltas[k];
            }
        }
    }

    let mut lut = [0u8; 256];
    for x_val in 0..256 {
        let xf = x_val as f64;
        if xf <= knots[0].0 {
            lut[x_val] = knots[0].1.round().clamp(0.0, 255.0) as u8;
            continue;
        }
        if xf >= knots[n - 1].0 {
            lut[x_val] = knots[n - 1].1.round().clamp(0.0, 255.0) as u8;
            continue;
        }

        let mut k = 0usize;
        for idx in 0..(n - 1) {
            if xf >= knots[idx].0 && xf <= knots[idx + 1].0 {
                k = idx;
                break;
            }
        }

        let h_k = (knots[k + 1].0 - knots[k].0).max(1e-6);
        let t = (xf - knots[k].0) / h_k;
        let t2 = t * t;
        let t3 = t2 * t;

        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        let y_val = h00 * knots[k].1 + h10 * h_k * d[k] + h01 * knots[k + 1].1 + h11 * h_k * d[k + 1];
        let mut quantized = y_val.round().clamp(0.0, 255.0) as u8;
        if x_val > 0 && quantized < lut[x_val - 1] {
            quantized = lut[x_val - 1];
        }
        lut[x_val] = quantized;
    }

    lut
}

// ============================================================================
// Page Analysis & Stride-4 CIELAB Pre-Flight (Upgrade A)
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
    let mut total_valid_pixels: usize = 0;

    // 58-bin modal paper peak histogram for L* in [70.0, 99.0)
    let mut paper_l_hist = [0u32; 58];
    let mut paper_bin_sum_a = [0.0f64; 58];
    let mut paper_bin_sum_b = [0.0f64; 58];
    let mut paper_bin_sum_c = [0.0f64; 58];
    let mut pure_white_paper_count = 0u32;
    let mut max_observed_l = 0.0f32;

    // 4x4 spatial grid paper b* histogram per cell (bins -20..=60 shifted by +20 -> 81 bins)
    let mut cell_b_hist = [[0u32; 81]; 16];
    let mut cell_b_count = [0u32; 16];

    // Sampled pixel statistics
    let mut sampled_pixels = 0usize;
    let mut raw_colored_count = 0usize;
    let mut yellow_paper_colored_count = 0usize;
    let mut muddy_shadow_colored_count = 0usize;
    let mut high_chroma_accent_count = 0usize;
    let mut max_chroma: u8 = 0;

    let mut hl_count: usize = 0;
    let mut hl_sum_r: u64 = 0;
    let mut hl_sum_g: u64 = 0;
    let mut hl_sum_b: u64 = 0;

    let mut sh_count: usize = 0;
    let mut sh_sum_r: u64 = 0;
    let mut sh_sum_g: u64 = 0;
    let mut sh_sum_b: u64 = 0;
    let mut sh_lab_count: usize = 0;
    let mut sh_sum_lab_a = 0.0f64;
    let mut sh_sum_lab_b = 0.0f64;

    match image {
        DynamicImage::ImageLuma8(luma) => {
            let raw = luma.as_raw();
            total_valid_pixels = raw.len();
            if total_valid_pixels >= 1_000_000 {
                hist_y = raw
                    .par_chunks(65536)
                    .map(|chunk| {
                        let mut local = [0u32; 256];
                        for &p in chunk {
                            local[p as usize] += 1;
                        }
                        local
                    })
                    .reduce(
                        || [0u32; 256],
                        |mut a, b| {
                            for i in 0..256 {
                                a[i] += b[i];
                            }
                            a
                        },
                    );
            } else {
                for &p in raw {
                    hist_y[p as usize] += 1;
                }
            }
            hist_y_neutral = hist_y;
            sampled_pixels = total_valid_pixels;
            max_observed_l = 100.0;
        }
        DynamicImage::ImageLumaA8(luma_a) => {
            for chunk in luma_a.as_raw().chunks_exact(2) {
                if chunk[1] >= 128 {
                    let p = chunk[0];
                    hist_y[p as usize] += 1;
                    hist_y_neutral[p as usize] += 1;
                    total_valid_pixels += 1;
                }
            }
            sampled_pixels = total_valid_pixels;
            max_observed_l = 100.0;
        }
        DynamicImage::ImageRgb8(rgb) => {
            let raw = rgb.as_raw();
            let w = width as usize;
            let h = height as usize;
            total_valid_pixels = w * h;

            // Fast single-pass Stride-4/12 CIELAB Pre-Flight, 4x4 Spatial Grid & Luminance Histogram (< 0.8ms on 6MP)
            let stride = if total_valid_pixels >= 2_000_000 {
                12usize
            } else if total_valid_pixels >= 250_000 {
                4usize
            } else {
                1usize
            };
            let scale_weight = (stride * stride) as u32;
            let cell_h = (h / 4).max(1);
            let cell_w = (w / 4).max(1);

            for y_coord in (0..h).step_by(stride) {
                let grid_r = (y_coord / cell_h).min(3);
                let row_offset = y_coord * w * 3;
                for x_coord in (0..w).step_by(stride) {
                    let grid_c = (x_coord / cell_w).min(3);
                    let cell_idx = grid_r * 4 + grid_c;
                    let idx = row_offset + x_coord * 3;
                    let r = raw[idx];
                    let g = raw[idx + 1];
                    let b = raw[idx + 2];

                    let (y_lum, rgb_diff, l_star, a_star, b_star, c_star) = if r == g && g == b {
                        let (l, _, _) = rgb_to_lab_fast(r, r, r);
                        (r, 0u8, l, 0.0f32, 0.0f32, 0.0f32)
                    } else {
                        let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8;
                        let diff = r.max(g).max(b) - r.min(g).min(b);
                        let (l, a, bl) = rgb_to_lab_fast(r, g, b);
                        let c = (a * a + bl * bl).sqrt();
                        (y, diff, l, a, bl, c)
                    };

                    hist_y[y_lum as usize] += scale_weight;
                    if rgb_diff > max_chroma {
                        max_chroma = rgb_diff;
                    }
                    if l_star > max_observed_l {
                        max_observed_l = l_star;
                    }

                    sampled_pixels += 1;

                    let is_colored = c_star >= 6.0 && l_star >= 2.0;
                    if is_colored {
                        raw_colored_count += 1;
                        // Track whether this colored pixel sits on the lignin paper-yellowing axis
                        if l_star >= 65.0 && a_star.abs() <= 4.5 && b_star > 0.0 && c_star < 36.0 {
                            yellow_paper_colored_count += 1;
                        } else if l_star < 55.0 && a_star.abs() <= 4.5 && b_star > 0.0 && c_star <= 12.0 {
                            muddy_shadow_colored_count += 1;
                        }
                        if c_star >= 20.0 && !(l_star >= 65.0 && a_star.abs() <= 4.5 && b_star > 0.0 && c_star < 36.0) {
                            high_chroma_accent_count += 1;
                        }
                    } else {
                        hist_y_neutral[y_lum as usize] += scale_weight;
                    }

                    if y_lum <= 40 && rgb_diff <= 25 {
                        hist_r_neutral[r as usize] += scale_weight;
                        hist_g_neutral[g as usize] += scale_weight;
                        hist_b_neutral[b as usize] += scale_weight;
                    }

                    // Paper candidate filter: (L* >= 65.0) & (C* < 42.0) & (|a*| < 12.0)
                    if l_star >= 65.0 && c_star < 42.0 && a_star.abs() < 12.0 {
                        if (70.0..99.0).contains(&l_star) {
                            let bin = (((l_star - 70.0) * 2.0).floor() as usize).min(57);
                            paper_l_hist[bin] += 1;
                            paper_bin_sum_a[bin] += a_star as f64;
                            paper_bin_sum_b[bin] += b_star as f64;
                            paper_bin_sum_c[bin] += c_star as f64;

                            if a_star.abs() <= 6.0 && b_star >= -2.0 {
                                let b_bin = ((b_star.round() as i32) + 20).clamp(0, 80) as usize;
                                cell_b_hist[cell_idx][b_bin] += 1;
                                cell_b_count[cell_idx] += 1;
                            }
                        } else if l_star >= 99.0 {
                            pure_white_paper_count += 1;
                            if a_star.abs() <= 6.0 && b_star >= -2.0 {
                                let b_bin = ((b_star.round() as i32) + 20).clamp(0, 80) as usize;
                                cell_b_hist[cell_idx][b_bin] += 1;
                                cell_b_count[cell_idx] += 1;
                            }
                        }
                    }

                    // Legacy DualZone highlight sample: y >= 205
                    if y_lum >= 205 && rgb_diff <= 65 && r >= g && g > b {
                        hl_count += 1;
                        hl_sum_r += r as u64;
                        hl_sum_g += g as u64;
                        hl_sum_b += b as u64;
                    }

                    // Shadow sample: y in [8, 45]
                    if (8..=45).contains(&y_lum) {
                        sh_count += 1;
                        sh_sum_r += r as u64;
                        sh_sum_g += g as u64;
                        sh_sum_b += b as u64;
                        if c_star < 25.0 {
                            sh_lab_count += 1;
                            sh_sum_lab_a += a_star as f64;
                            sh_sum_lab_b += b_star as f64;
                        }
                    }
                }
            }
        }
        DynamicImage::ImageRgba8(rgba) => {
            let raw = rgba.as_raw();
            let w = width as usize;
            let h = height as usize;
            let total_px = w * h;
            let stride = if total_px >= 2_000_000 {
                12usize
            } else if total_px >= 250_000 {
                4usize
            } else {
                1usize
            };
            let scale_weight = (stride * stride) as u32;
            let cell_h = (h / 4).max(1);
            let cell_w = (w / 4).max(1);

            for y_coord in (0..h).step_by(stride) {
                let grid_r = (y_coord / cell_h).min(3);
                let row_offset = y_coord * w * 4;
                for x_coord in (0..w).step_by(stride) {
                    let idx = row_offset + x_coord * 4;
                    if raw[idx + 3] < 128 {
                        continue;
                    }
                    let grid_c = (x_coord / cell_w).min(3);
                    let cell_idx = grid_r * 4 + grid_c;
                    let r = raw[idx];
                    let g = raw[idx + 1];
                    let b = raw[idx + 2];

                    let (y_lum, rgb_diff, l_star, a_star, b_star, c_star) = if r == g && g == b {
                        let (l, _, _) = rgb_to_lab_fast(r, r, r);
                        (r, 0u8, l, 0.0f32, 0.0f32, 0.0f32)
                    } else {
                        let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8;
                        let diff = r.max(g).max(b) - r.min(g).min(b);
                        let (l, a, bl) = rgb_to_lab_fast(r, g, b);
                        let c = (a * a + bl * bl).sqrt();
                        (y, diff, l, a, bl, c)
                    };

                    hist_y[y_lum as usize] += scale_weight;
                    total_valid_pixels += scale_weight as usize;
                    sampled_pixels += 1;

                    if rgb_diff > max_chroma {
                        max_chroma = rgb_diff;
                    }
                    if l_star > max_observed_l {
                        max_observed_l = l_star;
                    }

                    let is_colored = c_star >= 6.0 && l_star >= 2.0;
                    if is_colored {
                        raw_colored_count += 1;
                        if l_star >= 65.0 && a_star.abs() <= 4.5 && b_star > 0.0 && c_star < 36.0 {
                            yellow_paper_colored_count += 1;
                        } else if l_star < 55.0 && a_star.abs() <= 4.5 && b_star > 0.0 && c_star <= 12.0 {
                            muddy_shadow_colored_count += 1;
                        }
                        if c_star >= 20.0 && !(l_star >= 65.0 && a_star.abs() <= 4.5 && b_star > 0.0 && c_star < 36.0) {
                            high_chroma_accent_count += 1;
                        }
                    } else {
                        hist_y_neutral[y_lum as usize] += scale_weight;
                    }

                    if y_lum <= 40 && rgb_diff <= 25 {
                        hist_r_neutral[r as usize] += scale_weight;
                        hist_g_neutral[g as usize] += scale_weight;
                        hist_b_neutral[b as usize] += scale_weight;
                    }

                    if l_star >= 65.0 && c_star < 42.0 && a_star.abs() < 12.0 {
                        if (70.0..99.0).contains(&l_star) {
                            let bin = (((l_star - 70.0) * 2.0).floor() as usize).min(57);
                            paper_l_hist[bin] += 1;
                            paper_bin_sum_a[bin] += a_star as f64;
                            paper_bin_sum_b[bin] += b_star as f64;
                            paper_bin_sum_c[bin] += c_star as f64;

                            if a_star.abs() <= 6.0 && b_star >= -2.0 {
                                let b_bin = ((b_star.round() as i32) + 20).clamp(0, 80) as usize;
                                cell_b_hist[cell_idx][b_bin] += 1;
                                cell_b_count[cell_idx] += 1;
                            }
                        } else if l_star >= 99.0 {
                            pure_white_paper_count += 1;
                            if a_star.abs() <= 6.0 && b_star >= -2.0 {
                                let b_bin = ((b_star.round() as i32) + 20).clamp(0, 80) as usize;
                                cell_b_hist[cell_idx][b_bin] += 1;
                                cell_b_count[cell_idx] += 1;
                            }
                        }
                    }

                    if y_lum >= 205 && rgb_diff <= 65 && r >= g && g > b {
                        hl_count += 1;
                        hl_sum_r += r as u64;
                        hl_sum_g += g as u64;
                        hl_sum_b += b as u64;
                    }

                    if (8..=45).contains(&y_lum) {
                        sh_count += 1;
                        sh_sum_r += r as u64;
                        sh_sum_g += g as u64;
                        sh_sum_b += b as u64;
                        if c_star < 25.0 {
                            sh_lab_count += 1;
                            sh_sum_lab_a += a_star as f64;
                            sh_sum_lab_b += b_star as f64;
                        }
                    }
                }
            }
        }
        _ => {
            let rgb = image.to_rgb8();
            return analyze_page(&DynamicImage::ImageRgb8(rgb), page_number, config);
        }
    }

    if total_valid_pixels == 0 || sampled_pixels == 0 {
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

    // Modal Paper Peak Detection in L* in [70.0, 99.0) with [1, 2, 1] / 4 smoothing
    let total_non_digital_paper: u32 = paper_l_hist.iter().sum();
    let (paper_l_peak, _paper_a, paper_b, paper_c) = if total_non_digital_paper >= 20 {
        let mut smooth_l = [0.0f32; 58];
        for i in 0..58 {
            let left = if i > 0 { paper_l_hist[i - 1] as f32 } else { 0.0 };
            let center = paper_l_hist[i] as f32;
            let right = if i + 1 < 58 { paper_l_hist[i + 1] as f32 } else { 0.0 };
            smooth_l[i] = 0.25 * left + 0.50 * center + 0.25 * right;
        }
        let mut peak_idx = 0usize;
        let mut best_val = -1.0f32;
        for (i, &v) in smooth_l.iter().enumerate() {
            if v > best_val {
                best_val = v;
                peak_idx = i;
            }
        }
        let l_peak = 70.0 + (peak_idx as f32 + 0.5) * 0.5;
        let bin_lo = peak_idx.saturating_sub(6);
        let bin_hi = (peak_idx + 6).min(57);
        let mut cnt = 0u64;
        let mut sum_a = 0.0f64;
        let mut sum_b = 0.0f64;
        let mut sum_c = 0.0f64;
        for b in bin_lo..=bin_hi {
            cnt += paper_l_hist[b] as u64;
            sum_a += paper_bin_sum_a[b];
            sum_b += paper_bin_sum_b[b];
            sum_c += paper_bin_sum_c[b];
        }
        if cnt > 0 {
            (
                l_peak,
                (sum_a / cnt as f64) as f32,
                (sum_b / cnt as f64) as f32,
                (sum_c / cnt as f64) as f32,
            )
        } else {
            (l_peak, 0.0, 0.0, 0.0)
        }
    } else if pure_white_paper_count > 0 {
        (99.5, 0.0, 0.0, 0.0)
    } else {
        (max_observed_l, 0.0, 0.0, 0.0)
    };

    // 4x4 Spatial Grid Median b* variation (delta_b_grid)
    let min_cell_samples = ((sampled_pixels / 16) / 25).clamp(5, 25) as u32;
    let mut cell_medians_b = Vec::with_capacity(16);
    for c_idx in 0..16 {
        let cnt = cell_b_count[c_idx];
        if cnt >= min_cell_samples {
            let target = (cnt / 2).max(1);
            let mut acc = 0u32;
            for (b_bin, &bc) in cell_b_hist[c_idx].iter().enumerate() {
                acc += bc;
                if acc >= target {
                    let b_val = (b_bin as i32 - 20) as f32;
                    cell_medians_b.push(b_val);
                    break;
                }
            }
        }
    }

    let (delta_b_grid, max_grid_b) = if cell_medians_b.len() >= 4 {
        let mut min_b = f32::MAX;
        let mut max_b = f32::MIN;
        for &mb in &cell_medians_b {
            if mb < min_b {
                min_b = mb;
            }
            if mb > max_b {
                max_b = mb;
            }
        }
        (max_b - min_b, max_b)
    } else {
        (0.0, paper_b)
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
    } else if total_non_digital_paper >= 20 && (paper_c > 3.0 || delta_b_grid >= 12.0) {
        // Synthesize highlight sample from LAB paper cast / spatial margin yellowing
        let eff_b_star = paper_b.max(max_grid_b);
        let cast_mag = (eff_b_star * 1.8).clamp(6.0, 50.0);
        let synth_r = 245.0;
        let synth_g = 240.0;
        let synth_b = (245.0 - cast_mag).clamp(180.0, 239.0);
        DualZoneSample {
            count: (total_valid_pixels / 10).max(500),
            avg_rgb: (synth_r, synth_g, synth_b),
            chroma: synth_r - synth_b,
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

    let shadow_chroma_lab = if sh_lab_count > 10 {
        let sa = (sh_sum_lab_a / sh_lab_count as f64) as f32;
        let sb = (sh_sum_lab_b / sh_lab_count as f64) as f32;
        (sa * sa + sb * sb).sqrt()
    } else {
        0.0
    };

    // Dual-Zone Sepia Guard:
    // Intentional sepia artwork has uniform warm paper (delta_b_grid < 12.0, paper_b >= 5.0)
    // AND warm sepia shadows (R > B + 4.0) with zero non-yellowed paper cells.
    let shadow_is_warm = config.detection.protect_sepia_artwork
        && shadow_sample.count >= 50
        && (shadow_sample.avg_rgb.0 > shadow_sample.avg_rgb.2 + 4.0);
    let is_intentional_sepia = shadow_is_warm && delta_b_grid < 12.0 && paper_b >= 5.0;

    let has_paper_or_margin_cast = !is_intentional_sepia
        && paper_l_peak >= 50.0
        && (paper_c > 3.0
            || paper_b >= 5.0
            || delta_b_grid >= 12.0
            || (highlight_sample.chroma >= 6.0 && !shadow_is_warm)
            || (shadow_chroma_lab >= 2.5 && paper_c <= 3.5));

    // Compute intentional color pixel ratio R_color (excluding verified paper yellowing & muddy shadow cast)
    let effective_colored_count = if has_paper_or_margin_cast {
        raw_colored_count
            .saturating_sub(yellow_paper_colored_count)
            .saturating_sub(muddy_shadow_colored_count)
    } else {
        raw_colored_count
    };

    let r_color = effective_colored_count as f32 / sampled_pixels as f32;
    let r_accent = high_chroma_accent_count as f32 / sampled_pixels as f32;
    let neutral_ratio = (1.0 - r_color).clamp(0.0, 1.0);
    let neutral_pixel_count = ((total_valid_pixels as f32) * neutral_ratio).round() as usize;

    let classification = if is_already_luma {
        PageClassification::Monochrome
    } else if r_color > 0.15 {
        PageClassification::Color
    } else if r_color <= 0.02 && r_accent < 0.001 {
        PageClassification::Monochrome
    } else {
        PageClassification::Mixed
    };

    if has_paper_or_margin_cast && classification != PageClassification::Color {
        hist_y_neutral = hist_y;
    }

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
        neutral_pixel_count,
        neutral_ratio,
        max_chroma,
        high_chroma_count: high_chroma_accent_count,
        highlight_sample,
        shadow_sample,
        defects: Vec::new(),
        black_floor: 0,
        white_ceiling: 255,
        is_shelf_detected: false,
        per_channel_black: None,
    };

    if dynamic_range < 10
        || (classification == PageClassification::Color && total_valid_pixels >= 500_000)
    {
        return analysis;
    }

    let hist_total: u64 = hist_y.iter().map(|&c| c as u64).sum::<u64>().max(1);

    // Low-Key Atmospheric Scene Veto: p95(Y) < 130.0 or L*_paper < 50.0
    let p1_y = histogram_percentile(&hist_y, hist_total, 1.0);
    let p95_y = histogram_percentile(&hist_y, hist_total, 95.0);
    let p99_y = histogram_percentile(&hist_y, hist_total, 99.0);
    let low_key_veto = p95_y < 130.0 || paper_l_peak < 50.0;

    // High-Key Atmospheric Scene Veto: M_ink (2 < Y <= 65) < 0.008
    let ink_mass_count: u64 = hist_y[3..=65].iter().map(|&c| c as u64).sum();
    let m_ink = ink_mass_count as f64 / hist_total as f64;
    let high_key_veto = m_ink < 0.008;

    let mut defects = Vec::new();

    if !high_key_veto {
        if let Some(d) = detect_elevated_black_floor(
            &analysis.luminance_histogram,
            classification,
            total_valid_pixels,
            config,
        ) {
            if let DefectType::ElevatedBlackFloor {
                floor_value,
                is_bimodal,
                ..
            } = d.defect_type
            {
                analysis.black_floor = floor_value;
                analysis.is_shelf_detected = is_bimodal;
            }
            defects.push(d);
        }
    }

    // Already-Optimal Dynamic Range Veto:
    // If p1 <= 1.0 && p99 >= 254.0 && C*_paper <= 3.0 && delta_b*_grid < 12.0
    // and no validated bimodal drawing-ink shelf exists above bin 2, return immediately with zero defects.
    let eff_white = find_effective_white(&hist_y, p99_y);
    if p1_y <= 1.0
        && eff_white >= 254.0
        && !has_paper_or_margin_cast
        && !analysis.is_shelf_detected
    {
        analysis.black_floor = 0;
        analysis.defects.clear();
        return analysis;
    }

    if !low_key_veto && !high_key_veto {
        if let Some(d) = detect_compressed_dynamic_range(&analysis.luminance_histogram, total_valid_pixels) {
            defects.push(d);
        }
    }

    if !low_key_veto {
        if let Some(d) = detect_paper_yellowing(&analysis, classification, config) {
            defects.push(d);
        } else if has_paper_or_margin_cast {
            let cast_mag = paper_c.max(delta_b_grid).max(shadow_chroma_lab).max(6.0);
            defects.push(DefectDetection {
                defect_type: DefectType::PaperYellowing {
                    cast_magnitude: cast_mag,
                    highlight_rgb: (245.0, 240.0, (245.0 - cast_mag).clamp(180.0, 239.0)),
                    is_differential: true,
                },
                raw_confidence: 0.95,
                adjusted_confidence: 0.95,
                severity: DefectSeverity::Severe,
                evidence: format!(
                    "CIELAB paper/grid/shadow cast: C*={:.1}, delta_b_grid={:.1}",
                    paper_c, delta_b_grid
                ),
            });
        }

        if let Some(d) = detect_clipped_highlights(
            &analysis.luminance_histogram,
            classification,
            total_valid_pixels,
            config,
        ) {
            if let DefectType::ClippedHighlights { white_point, .. } = d.defect_type {
                analysis.white_ceiling = white_point;
            }
            defects.push(d);
        }
    }

    if let Some(d) = detect_elevated_noise_floor(image, classification) {
        defects.push(d);
    }

    analysis.defects = defects;
    analysis
}

// ============================================================================
// Defect Detection Engine (Section 3c & Upgrade B)
// ============================================================================

pub fn detect_elevated_black_floor(
    hist: &[u32; 256],
    classification: PageClassification,
    total_pixels: usize,
    config: &AutoLevelsConfig,
) -> Option<DefectDetection> {
    let n_samples: u64 = hist.iter().map(|&c| c as u64).sum();
    if n_samples == 0 {
        return None;
    }
    let total_n = if total_pixels > 0 {
        total_pixels as u64
    } else {
        n_samples
    };

    // High-Key Scene Veto: M_ink (2 < Y <= 65) < 0.008
    let ink_mass: u64 = hist[3..=65].iter().map(|&c| c as u64).sum();
    if (ink_mass as f64 / total_n as f64) < 0.008 {
        return None;
    }

    // 1. Check classic compact bimodal shelf in [10..=28]
    let mut max_peak_count = 0u32;
    let mut max_peak_bin = 0usize;
    for bin in 10..=28 {
        if hist[bin] > max_peak_count {
            max_peak_count = hist[bin];
            max_peak_bin = bin;
        }
    }

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

    let is_compact_shelf = classification != PageClassification::Color
        && max_peak_bin >= 10
        && (max_peak_count as f64) >= (n_samples as f64 * 0.004)
        && surge_ratio >= config.detection.surge_ratio_threshold
        && fwhm <= 6;

    if is_compact_shelf {
        let floor_value = max_peak_bin as u8;
        let severity = if floor_value <= 5 {
            DefectSeverity::Mild
        } else if floor_value <= 15 {
            DefectSeverity::Moderate
        } else {
            DefectSeverity::Severe
        };
        return Some(DefectDetection {
            defect_type: DefectType::ElevatedBlackFloor {
                floor_value,
                surge_ratio,
                is_bimodal: true,
            },
            raw_confidence: 0.95,
            adjusted_confidence: 0.95,
            severity,
            evidence: format!(
                "Compact ink shelf={}, surge={:.1}, fwhm={}",
                floor_value, surge_ratio, fwhm
            ),
        });
    }

    // 2. Page 5 Digital Spike Isolation & Bimodal Valley Check up to bin 85
    let p1 = histogram_percentile(hist, n_samples, 1.0);
    let p99 = histogram_percentile(hist, n_samples, 99.0);
    let hi = find_effective_white(hist, p99);

    let zero_count = hist[0] as u64 + hist[1] as u64 + hist[2] as u64;
    let m_zero = zero_count as f64 / n_samples as f64;
    let (nz_count, nz_p3) = nonzero_histogram_percentile(hist, 3.0);

    let mut lo = p1;
    let mut is_bimodal_valley = false;

    if classification != PageClassification::Color {
        if lo < 3.0 && nz_count > 100 && nz_p3 > 8.0 && m_zero > 0.0005 && m_zero < 0.065 {
            let mut cand_lo = nz_p3.max(8.0);
            if m_zero < 0.035 {
                let eff_black = find_effective_black(hist, p1);
                if eff_black > cand_lo {
                    cand_lo += (eff_black - cand_lo) * 0.85;
                }
            }
            // When pure digital black (0..=2) exists on the page, require a separation trough
            // between the digital spike (bins 0..=2, kernel radius 2 -> bin 5+) and `cand_lo`
            // so clean digital pages with flat/monotonic anti-aliased ramps (T1) are never altered.
            let h_smooth = smooth_histogram_binomial_5(hist);
            let lo_idx = (cand_lo.floor() as usize).clamp(5, 85);
            let search_peak_hi = (lo_idx + 15).min(85);
            let mut h_peak = 0.0f32;
            let mut peak_idx = 5usize;
            for (j, &val) in h_smooth.iter().enumerate().take(search_peak_hi + 1).skip(5) {
                if val > h_peak {
                    h_peak = val;
                    peak_idx = j;
                }
            }
            let mut min_trough_before = f32::MAX;
            for &val in h_smooth.iter().take(peak_idx.max(6)).skip(5) {
                if val < min_trough_before {
                    min_trough_before = val;
                }
            }
            let has_separation_from_zero = peak_idx >= 8 && min_trough_before < 0.65 * h_peak;
            if has_separation_from_zero {
                let val_lo = validate_black_floor(hist, n_samples as usize, cand_lo, hi);
                if val_lo >= 8.0 {
                    lo = val_lo;
                    is_bimodal_valley = true;
                }
            }
        } else if lo >= 3.0 {
            let eff_black = find_effective_black(hist, p1);
            let mut cand_lo = lo;
            if eff_black > cand_lo {
                cand_lo += (eff_black - cand_lo) * 0.85;
            }
            lo = validate_black_floor(hist, n_samples as usize, cand_lo, hi);
            if lo > 28.0 {
                is_bimodal_valley = true;
            }
        }
    }

    // Fallback to black_clip_percent if lo <= 1 and no digital spike
    let floor_value = if lo >= 2.0 {
        lo.round() as u8
    } else {
        let clip_target = (n_samples as f64 * (config.detection.black_clip_percent as f64 / 100.0))
            .round() as u64;
        let mut bp = 0u8;
        let mut acc = 0u64;
        for (level, &count) in hist.iter().enumerate() {
            acc += count as u64;
            if acc >= clip_target {
                bp = level as u8;
                break;
            }
        }
        if bp > 35 {
            validate_black_floor(hist, n_samples as usize, bp as f32, hi).round() as u8
        } else {
            bp
        }
    };

    if floor_value <= 1 || floor_value > 85 {
        return None;
    }

    let severity = if floor_value <= 5 {
        DefectSeverity::Mild
    } else if floor_value <= 15 {
        DefectSeverity::Moderate
    } else {
        DefectSeverity::Severe
    };

    let raw_confidence = if is_bimodal_valley || floor_value >= 16 {
        0.95f32
    } else {
        0.85f32
    };

    Some(DefectDetection {
        defect_type: DefectType::ElevatedBlackFloor {
            floor_value,
            surge_ratio,
            is_bimodal: is_bimodal_valley,
        },
        raw_confidence,
        adjusted_confidence: raw_confidence,
        severity,
        evidence: format!(
            "Ink floor={}, surge={:.1}, bimodal={}",
            floor_value, surge_ratio, is_bimodal_valley
        ),
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
    if !(10..200).contains(&effective_range) {
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
        evidence: format!(
            "Dynamic range {}, tail_emptiness={:.2}",
            effective_range, tail_emptiness
        ),
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
    if !(6.0..=55.0).contains(&cast_magnitude) || rg_diff > 0.65 * cast_magnitude {
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

    let mut raw_confidence: f32 = if classification == PageClassification::Color {
        0.75f32
    } else {
        0.85f32
    };
    if cast_magnitude > 15.0 {
        raw_confidence = (raw_confidence + 0.10f32).min(0.95f32);
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

    // Low-Key Scene Veto: if >95% of pixels (p96) are below Y=130, veto highlight stretch
    let p96 = histogram_percentile(hist, total_pixels as u64, 96.0);
    if p96 < 130.0 {
        return None;
    }

    let p99 = histogram_percentile(hist, total_pixels as u64, 99.0);
    let eff_white = find_effective_white(hist, p99);
    let blended_hi = (p99 - (p99 - eff_white) * 0.85).round() as u8;

    let clip_target =
        (total_pixels as f64 * (config.detection.white_clip_percent as f64 / 100.0)).round() as u64;
    let mut raw_wp = 255u8;
    let mut acc_h = 0u64;
    for (level, &count) in hist.iter().enumerate().rev() {
        acc_h += count as u64;
        if acc_h >= clip_target {
            raw_wp = level as u8;
            break;
        }
    }

    let white_point = if eff_white < p99 - 1.0 {
        blended_hi
    } else {
        raw_wp.min(p99.round() as u8)
    };

    if white_point >= 254 {
        return None;
    }
    if classification == PageClassification::Color && white_point >= 250 {
        return None;
    }

    let min_white_threshold = config.correction.min_white_threshold;
    let plateau_threshold = (total_pixels as f64 * 0.005).round() as u32;
    let has_plateau = hist[white_point as usize] >= plateau_threshold
        || hist[raw_wp as usize] >= plateau_threshold;

    if white_point < min_white_threshold && !has_plateau {
        return None;
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
        raw_confidence: 0.90,
        adjusted_confidence: 0.90,
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

    // Use a fast strided sample (~2400 points) for noise estimation so 6MP pages take < 0.05ms
    let step = if (width as usize) * (height as usize) >= 500_000 {
        ((width.min(height) as usize) / 40).max(8)
    } else {
        2usize
    };

    let mut laplacian_values = Vec::new();
    match image {
        DynamicImage::ImageRgb8(rgb) => {
            let raw = rgb.as_raw();
            let w = width as usize;
            let h = height as usize;
            let luma_at = |x: usize, y: usize| -> i32 {
                let idx = (y * w + x) * 3;
                ((54 * raw[idx] as u32 + 183 * raw[idx + 1] as u32 + 19 * raw[idx + 2] as u32 + 128) >> 8) as i32
            };
            for y in (1..h - 1).step_by(step) {
                for x in (1..w - 1).step_by(step) {
                    let p00 = luma_at(x - 1, y - 1);
                    let p01 = luma_at(x, y - 1);
                    let p02 = luma_at(x + 1, y - 1);
                    let p10 = luma_at(x - 1, y);
                    let p11 = luma_at(x, y);
                    let p12 = luma_at(x + 1, y);
                    let p20 = luma_at(x - 1, y + 1);
                    let p21 = luma_at(x, y + 1);
                    let p22 = luma_at(x + 1, y + 1);
                    let lap = ((p00 + p02 + p20 + p22) - 2 * (p01 + p21 + p10 + p12) + 4 * p11).abs() as f32;
                    laplacian_values.push(lap);
                }
            }
        }
        _ => {
            let luma = image.to_luma8();
            for y in (1..height - 1).step_by(step) {
                for x in (1..width - 1).step_by(step) {
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
        }
    }

    if laplacian_values.is_empty() {
        return None;
    }

    laplacian_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_lap = laplacian_values[laplacian_values.len() / 2];

    let mut deviations: Vec<f32> = laplacian_values
        .iter()
        .map(|&v| (v - median_lap).abs())
        .collect();
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

    let raw_confidence = if classification == PageClassification::Monochrome {
        0.30
    } else {
        0.40
    };

    Some(DefectDetection {
        defect_type: DefectType::ElevatedNoiseFloor { noise_sigma },
        raw_confidence,
        adjusted_confidence: raw_confidence,
        severity,
        evidence: format!("Noise sigma={:.2} (MAD={:.2})", noise_sigma, mad),
    })
}

// ============================================================================
// Book-Level Confidence Adjustment & Outlier Guard (Section 3c.6 & Upgrade B.5)
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
                } else if cast_freq_low(freq) {
                    defect.adjusted_confidence = defect.raw_confidence * 0.60;
                }
            }
            DefectType::ElevatedNoiseFloor { .. } => {}
        }
    }
}

#[inline]
fn cast_freq_low(freq: f32) -> bool {
    freq < 0.10
}

// ============================================================================
// Correction Engine (Section 3d & Upgrades B, C, D)
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

    // Strict Veto 1: Full-Color Page Veto (R_color > 0.15 -> PageClassification::Color)
    if classification == PageClassification::Color {
        plan.use_oklab = true;
        plan.skip_correction = true;
        return plan;
    }

    // Direct Validated Restoration (Upgrade B.5):
    // Apply 100% of validated (lo, hi) directly to the Fritsch-Carlson spline LUT
    // without multiplying `lo` down by `severity_multiplier * confidence_scale`.
    for defect in defects {
        match defect.defect_type {
            DefectType::ElevatedBlackFloor { floor_value, .. } => {
                let clamped_floor = floor_value.min(config.correction.max_black_shift);
                plan.black_point = plan.black_point.max(clamped_floor);
                plan.correction_fraction = 1.0;
            }
            DefectType::ClippedHighlights { white_point, .. } => {
                plan.white_point = plan.white_point.min(white_point);
                plan.correction_fraction = 1.0;
            }
            DefectType::CompressedDynamicRange { p5, p95, .. } => {
                if plan.black_point == 0 && p5 >= 2 && p5 <= config.correction.max_black_shift {
                    plan.black_point = p5;
                }
                if plan.white_point == 255 && p95 >= config.correction.min_white_threshold {
                    plan.white_point = p95;
                }
                plan.correction_fraction = 1.0;
            }
            DefectType::PaperYellowing { highlight_rgb, .. } => {
                if config.correction.correct_paper_cast {
                    let max_ch = highlight_rgb.0.max(highlight_rgb.1).max(highlight_rgb.2);
                    if highlight_rgb.0 > 0.0 && highlight_rgb.1 > 0.0 && highlight_rgb.2 > 0.0 {
                        plan.paper_cast_gains = Some((
                            max_ch / highlight_rgb.0,
                            max_ch / highlight_rgb.1,
                            max_ch / highlight_rgb.2,
                        ));
                        plan.paper_cast_ramp_start = 180;
                        plan.correction_fraction = 1.0;
                    }
                }
            }
            DefectType::ElevatedNoiseFloor { .. } => {}
        }
    }

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
    if (plan.white_point as i32 - plan.black_point as i32) < 15 {
        plan.black_point = 0;
        plan.white_point = 255;
    }

    plan.use_oklab = false;
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

    let channel_spread =
        peak_r_bin.max(peak_g_bin).max(peak_b_bin) - peak_r_bin.min(peak_g_bin).min(peak_b_bin);

    if channel_spread < 4 || plan.white_point < 250 {
        return None;
    }

    Some((peak_r_bin, peak_g_bin, peak_b_bin))
}

// ----------------------------------------------------------------------------
// Color Page Correction (Oklab with Radial Chord Bisection - Legacy Support)
// ----------------------------------------------------------------------------

fn build_oklab_l_lut(l_black: f32, l_white: f32, gamma: f32) -> [f32; 4096] {
    let inv_gamma = if (gamma - 1.0).abs() > 0.001 && gamma > 0.1 {
        1.0 / gamma
    } else {
        1.0
    };

    let mut lut = [0.0f32; 4096];
    let range = (l_white - l_black).max(0.001);

    for (i, slot) in lut.iter_mut().enumerate() {
        let l = i as f32 / 4095.0;
        if l <= l_black {
            *slot = 0.0;
        } else if l >= l_white {
            *slot = 1.0;
        } else {
            let norm = (l - l_black) / range;
            let mapped = if inv_gamma != 1.0 {
                norm.powf(inv_gamma)
            } else {
                norm
            };
            *slot = mapped.clamp(0.0, 1.0);
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
            (**rgb).par_chunks_exact_mut(3).for_each(|chunk| {
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
                if (0.0..=1.0).contains(&lin_rgb.red)
                    && (0.0..=1.0).contains(&lin_rgb.green)
                    && (0.0..=1.0).contains(&lin_rgb.blue)
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
                    if (0.0..=1.0).contains(&test_lin.red)
                        && (0.0..=1.0).contains(&test_lin.green)
                        && (0.0..=1.0).contains(&test_lin.blue)
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
            (**rgba).par_chunks_exact_mut(4).for_each(|chunk| {
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
                if (0.0..=1.0).contains(&lin_rgb.red)
                    && (0.0..=1.0).contains(&lin_rgb.green)
                    && (0.0..=1.0).contains(&lin_rgb.blue)
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
                    if (0.0..=1.0).contains(&test_lin.red)
                        && (0.0..=1.0).contains(&test_lin.green)
                        && (0.0..=1.0).contains(&test_lin.blue)
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
// Fused Luminance Stretch + Dual-Boundary Sigmoid Chroma Gating (Upgrade D)
// ----------------------------------------------------------------------------

/// Builds a 3x3 morphologically dilated boolean mask of high-chroma accents (`C* >= 20.0`,
/// excluding oxidized yellow paper) for `PageClassification::Mixed` pages.
fn build_dilated_accent_mask_rgb(raw: &[u8], width: usize, height: usize) -> Vec<u8> {
    let total = width * height;
    let mut base_mask = vec![0u8; total];

    base_mask
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, mask_row)| {
            let row_offset = y * width * 3;
            for x in 0..width {
                let idx = row_offset + x * 3;
                let r = raw[idx];
                let g = raw[idx + 1];
                let b = raw[idx + 2];
                if r == g && g == b {
                    continue;
                }
                let rgb_diff = r.max(g).max(b) - r.min(g).min(b);
                if rgb_diff < 12 {
                    continue;
                }
                let (l_star, a_star, b_star) = rgb_to_lab_fast(r, g, b);
                let c_star = (a_star * a_star + b_star * b_star).sqrt();
                let is_yellow_paper =
                    l_star >= 65.0 && a_star.abs() <= 4.5 && b_star > 0.0 && c_star < 36.0;
                if !is_yellow_paper && (c_star >= 20.0 || rgb_diff >= 35) {
                    mask_row[x] = 1;
                }
            }
        });

    if height < 3 || width < 3 {
        return base_mask;
    }

    let mut dilated = vec![0u8; total];
    dilated
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, out_row)| {
            let y0 = y.saturating_sub(1);
            let y1 = (y + 1).min(height - 1);
            for x in 0..width {
                let x0 = x.saturating_sub(1);
                let x1 = (x + 1).min(width - 1);
                let mut hit = 0u8;
                'outer: for ny in y0..=y1 {
                    let r_off = ny * width;
                    for nx in x0..=x1 {
                        if base_mask[r_off + nx] != 0 {
                            hit = 1;
                            break 'outer;
                        }
                    }
                }
                out_row[x] = hit;
            }
        });

    dilated
}

pub fn apply_monochrome_mixed_correction(
    image: &mut DynamicImage,
    plan: &CorrectionPlan,
    classification: PageClassification,
) {
    let lut_y = build_monotonic_tone_lut(plan.black_point as f32, plan.white_point as f32);
    let tables = &*FAST_LAB_TABLES;
    let has_paper_cast = plan.paper_cast_gains.is_some();

    match image {
        DynamicImage::ImageLuma8(luma) => {
            (**luma).par_iter_mut().for_each(|p| {
                *p = lut_y[*p as usize];
            });
        }
        DynamicImage::ImageLumaA8(luma_a) => {
            (**luma_a).par_chunks_exact_mut(2).for_each(|chunk| {
                if chunk[1] >= 128 {
                    chunk[0] = lut_y[chunk[0] as usize];
                }
            });
        }
        DynamicImage::ImageRgb8(rgb) => {
            let w = rgb.width() as usize;
            let h = rgb.height() as usize;
            let raw: &mut [u8] = rgb;

            let accent_mask = if classification == PageClassification::Mixed {
                Some(build_dilated_accent_mask_rgb(raw, w, h))
            } else {
                None
            };

            let strip_pixels = (w * 16).max(4096);
            let strip_bytes = strip_pixels * 3;

            raw.par_chunks_mut(strip_bytes)
                .enumerate()
                .for_each(|(strip_idx, strip)| {
                    let base_px = strip_idx * strip_pixels;
                    for (local_idx, chunk) in strip.chunks_exact_mut(3).enumerate() {
                        let r = chunk[0];
                        let g = chunk[1];
                        let b = chunk[2];

                        if let Some(ref mask) = accent_mask {
                            if mask[base_px + local_idx] != 0 {
                                continue; // Protect dilated high-chroma accent (taper = 1.0, zero alteration)
                            }
                        }

                        if r == g && g == b {
                            let y_u8 = lut_y[r as usize];
                            chunk[0] = y_u8;
                            chunk[1] = y_u8;
                            chunk[2] = y_u8;
                            continue;
                        }

                        let y_old = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8;
                        let y_new = lut_y[y_old as usize] as i32;
                        let rgb_diff = r.max(g).max(b) - r.min(g).min(b);

                        if rgb_diff <= 2 {
                            let y_u8 = y_new as u8;
                            chunk[0] = y_u8;
                            chunk[1] = y_u8;
                            chunk[2] = y_u8;
                            continue;
                        }

                        let (l_star, a_star, b_star) = rgb_to_lab_fast(r, g, b);
                        let c_star = (a_star * a_star + b_star * b_star).sqrt();

                        // Dual-Boundary Luminance-Adaptive Sigmoid Chroma Gate via 2D fixed-point LUT
                        let l_bin = (l_star.round() as usize).min(100);
                        let c_bin = ((c_star * 4.0).round() as usize).min(255);
                        let mut taper_q8 = tables.taper_lut[l_bin][c_bin] as i32;

                        // Neutralize verified lignin / acid paper yellowing & dirty paper tint
                        // while strictly preserving pastel blue skies (b* < 0) and warm skin (a* > 4.5)
                        let is_paper_yellow_axis =
                            l_star >= 65.0 && a_star.abs() <= 4.5 && b_star > 0.0 && c_star < 36.0;
                        if is_paper_yellow_axis
                            && (has_paper_cast || classification == PageClassification::Monochrome || c_star <= 5.5)
                        {
                            taper_q8 = 0;
                        }

                        if taper_q8 == 0 {
                            let y_u8 = y_new as u8;
                            chunk[0] = y_u8;
                            chunk[1] = y_u8;
                            chunk[2] = y_u8;
                        } else {
                            let y_i32 = y_old as i32;
                            let r_out = y_new + (((r as i32 - y_i32) * taper_q8 + 128) >> 8);
                            let g_out = y_new + (((g as i32 - y_i32) * taper_q8 + 128) >> 8);
                            let b_out = y_new + (((b as i32 - y_i32) * taper_q8 + 128) >> 8);
                            chunk[0] = r_out.clamp(0, 255) as u8;
                            chunk[1] = g_out.clamp(0, 255) as u8;
                            chunk[2] = b_out.clamp(0, 255) as u8;
                        }
                    }
                });
        }
        DynamicImage::ImageRgba8(rgba) => {
            let raw: &mut [u8] = rgba;
            raw.par_chunks_exact_mut(4).for_each(|chunk| {
                if chunk[3] < 128 {
                    return;
                }
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];
                let rgb_diff = r.max(g).max(b) - r.min(g).min(b);

                if classification == PageClassification::Mixed && rgb_diff >= 35 {
                    return;
                }

                let y_old = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8;
                let y_new = lut_y[y_old as usize] as i32;

                if r == g && g == b {
                    let y_u8 = y_new as u8;
                    chunk[0] = y_u8;
                    chunk[1] = y_u8;
                    chunk[2] = y_u8;
                    return;
                }

                let (l_star, a_star, b_star) = rgb_to_lab_fast(r, g, b);
                let c_star = (a_star * a_star + b_star * b_star).sqrt();

                if classification == PageClassification::Mixed
                    && c_star >= 20.0
                    && !(l_star >= 65.0 && a_star.abs() <= 4.5 && b_star > 0.0 && c_star < 36.0)
                {
                    return;
                }

                let l_bin = (l_star.round() as usize).min(100);
                let c_bin = ((c_star * 4.0).round() as usize).min(255);
                let mut taper_q8 = tables.taper_lut[l_bin][c_bin] as i32;

                let is_paper_yellow_axis =
                    l_star >= 65.0 && a_star.abs() <= 4.5 && b_star > 0.0 && c_star < 36.0;
                if is_paper_yellow_axis
                    && (has_paper_cast || classification == PageClassification::Monochrome || c_star <= 5.5)
                {
                    taper_q8 = 0;
                }

                if taper_q8 == 0 {
                    let y_u8 = y_new as u8;
                    chunk[0] = y_u8;
                    chunk[1] = y_u8;
                    chunk[2] = y_u8;
                } else {
                    let y_i32 = y_old as i32;
                    let r_out = y_new + (((r as i32 - y_i32) * taper_q8 + 128) >> 8);
                    let g_out = y_new + (((g as i32 - y_i32) * taper_q8 + 128) >> 8);
                    let b_out = y_new + (((b as i32 - y_i32) * taper_q8 + 128) >> 8);
                    chunk[0] = r_out.clamp(0, 255) as u8;
                    chunk[1] = g_out.clamp(0, 255) as u8;
                    chunk[2] = b_out.clamp(0, 255) as u8;
                }
            });
        }
        _ => {}
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
            (**rgb).par_chunks_exact_mut(3).for_each(|chunk| {
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];
                let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8;
                if y >= ramp_start {
                    let factor = ((y as f32 - ramp_start as f32) / ramp_range).clamp(0.0, 1.0);
                    chunk[0] = (r as f32 * (1.0 + (gr - 1.0) * factor))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    chunk[1] = (g as f32 * (1.0 + (gg - 1.0) * factor))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    chunk[2] = (b as f32 * (1.0 + (gb - 1.0) * factor))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            });
        }
        DynamicImage::ImageRgba8(rgba) => {
            (**rgba).par_chunks_exact_mut(4).for_each(|chunk| {
                if chunk[3] < 128 {
                    return;
                }
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];
                let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8;
                if y >= ramp_start {
                    let factor = ((y as f32 - ramp_start as f32) / ramp_range).clamp(0.0, 1.0);
                    chunk[0] = (r as f32 * (1.0 + (gr - 1.0) * factor))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    chunk[1] = (g as f32 * (1.0 + (gg - 1.0) * factor))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    chunk[2] = (b as f32 * (1.0 + (gb - 1.0) * factor))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            });
        }
        _ => {}
    }
}

// ============================================================================
// Quality Guard Verification Pipeline (Section 3e & Upgrade D.5)
// ============================================================================

#[inline]
fn extract_fast_luma_vec(image: &DynamicImage) -> Vec<u8> {
    match image {
        DynamicImage::ImageLuma8(luma) => luma.as_raw().clone(),
        DynamicImage::ImageRgb8(rgb) => {
            let raw = rgb.as_raw();
            let n = (rgb.width() * rgb.height()) as usize;
            let mut out = vec![0u8; n];
            if n >= 500_000 {
                out.par_chunks_mut(16384)
                    .zip(raw.par_chunks(16384 * 3))
                    .for_each(|(dst, src)| {
                        for (d, px) in dst.iter_mut().zip(src.chunks_exact(3)) {
                            *d = ((54 * px[0] as u32 + 183 * px[1] as u32 + 19 * px[2] as u32 + 128) >> 8) as u8;
                        }
                    });
            } else {
                for (d, px) in out.iter_mut().zip(raw.chunks_exact(3)) {
                    *d = ((54 * px[0] as u32 + 183 * px[1] as u32 + 19 * px[2] as u32 + 128) >> 8) as u8;
                }
            }
            out
        }
        _ => image.to_luma8().into_raw(),
    }
}

pub fn compute_luminance_ssim(orig: &DynamicImage, leveled: &DynamicImage) -> f32 {
    let (w, h) = orig.dimensions();
    if w == 0 || h == 0 {
        return 1.0;
    }
    let orig_luma = extract_fast_luma_vec(orig);
    let lev_luma = extract_fast_luma_vec(leveled);

    let c1 = 6.5025f64;
    let c2 = 58.5225f64;

    let block_size = 8usize;
    let w_usize = w as usize;
    let blocks_x = w_usize / block_size;
    let blocks_y = (h as usize) / block_size;

    if blocks_x == 0 || blocks_y == 0 {
        return 1.0;
    }

    // Stride blocks by 2 on large >= 2MP images so SSIM completes in < 0.6ms
    let block_step = if w_usize * (h as usize) >= 1_500_000 { 2usize } else { 1usize };
    let by_indices: Vec<usize> = (0..blocks_y).step_by(block_step).collect();
    let bx_count = (0..blocks_x).step_by(block_step).count();
    let total_sampled_blocks = (by_indices.len() * bx_count).max(1) as f64;

    let ssim_sum: f64 = by_indices
        .into_par_iter()
        .map(|by| {
            let mut row_ssim = 0.0f64;
            let y_start = by * block_size;
            for bx in (0..blocks_x).step_by(block_step) {
                let x_start = bx * block_size;
                let mut sum_x = 0u32;
                let mut sum_y = 0u32;
                let mut sum_x_sq = 0u64;
                let mut sum_y_sq = 0u64;
                let mut sum_xy = 0u64;

                for dy in 0..block_size {
                    let row_off = (y_start + dy) * w_usize + x_start;
                    let o_slice = &orig_luma[row_off..row_off + block_size];
                    let l_slice = &lev_luma[row_off..row_off + block_size];
                    for i in 0..block_size {
                        let px = o_slice[i] as u32;
                        let py = l_slice[i] as u32;
                        sum_x += px;
                        sum_y += py;
                        sum_x_sq += (px * px) as u64;
                        sum_y_sq += (py * py) as u64;
                        sum_xy += (px * py) as u64;
                    }
                }

                let n = 64.0f64;
                let mu_x = sum_x as f64 / n;
                let mu_y = sum_y as f64 / n;
                let sigma_x_sq = (sum_x_sq as f64 / n) - (mu_x * mu_x);
                let sigma_y_sq = (sum_y_sq as f64 / n) - (mu_y * mu_y);
                let sigma_xy = (sum_xy as f64 / n) - (mu_x * mu_y);

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

    (ssim_sum / total_sampled_blocks) as f32
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
    let orig_luma = extract_fast_luma_vec(original);
    let lev_luma = extract_fast_luma_vec(leveled);
    let w_usize = w as usize;
    let h_usize = h as usize;
    let step = stride.max(1);

    let y_indices: Vec<usize> = (1..h_usize - 1).step_by(step).collect();

    let (sum_prod, sum_sq1, sum_sq2) = y_indices
        .into_par_iter()
        .map(|y| {
            let mut row_prod = 0.0f64;
            let mut row_sq1 = 0.0f64;
            let mut row_sq2 = 0.0f64;
            let r_prev = (y - 1) * w_usize;
            let r_curr = y * w_usize;
            let r_next = (y + 1) * w_usize;

            for x in (1..w_usize - 1).step_by(step) {
                let o00 = orig_luma[r_prev + x - 1] as i32;
                let o01 = orig_luma[r_prev + x] as i32;
                let o02 = orig_luma[r_prev + x + 1] as i32;
                let o10 = orig_luma[r_curr + x - 1] as i32;
                let o12 = orig_luma[r_curr + x + 1] as i32;
                let o20 = orig_luma[r_next + x - 1] as i32;
                let o21 = orig_luma[r_next + x] as i32;
                let o22 = orig_luma[r_next + x + 1] as i32;

                let gx1 = ((o02 + 2 * o12 + o22) - (o00 + 2 * o10 + o20)) as f64;
                let gy1 = ((o20 + 2 * o21 + o22) - (o00 + 2 * o01 + o02)) as f64;

                let l00 = lev_luma[r_prev + x - 1] as i32;
                let l01 = lev_luma[r_prev + x] as i32;
                let l02 = lev_luma[r_prev + x + 1] as i32;
                let l10 = lev_luma[r_curr + x - 1] as i32;
                let l12 = lev_luma[r_curr + x + 1] as i32;
                let l20 = lev_luma[r_next + x - 1] as i32;
                let l21 = lev_luma[r_next + x] as i32;
                let l22 = lev_luma[r_next + x + 1] as i32;

                let gx2 = ((l02 + 2 * l12 + l22) - (l00 + 2 * l10 + l20)) as f64;
                let gy2 = ((l20 + 2 * l21 + l22) - (l00 + 2 * l01 + l02)) as f64;

                let mag1_sq = gx1 * gx1 + gy1 * gy1;
                let mag2_sq = gx2 * gx2 + gy2 * gy2;
                if mag1_sq >= 144.0 || mag2_sq >= 144.0 {
                    row_prod += gx1 * gx2 + gy1 * gy2;
                    row_sq1 += mag1_sq;
                    row_sq2 += mag2_sq;
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
    let orig_luma = extract_fast_luma_vec(original);
    let lev_luma = extract_fast_luma_vec(leveled);

    let bs = block_size as usize;
    let w_usize = w as usize;
    let blocks_x = w_usize / bs;
    let blocks_y = (h as usize) / bs;

    if blocks_x == 0 || blocks_y == 0 {
        return Vec::new();
    }

    (0..blocks_y)
        .into_par_iter()
        .flat_map(|by| {
            let y_start = by * bs;
            let mut row_ratios = Vec::new();
            for bx in 0..blocks_x {
                let x_start = bx * bs;

                let mut o_min = 255u8;
                let mut o_max = 0u8;
                let mut l_min = 255u8;
                let mut l_max = 0u8;

                for dy in 0..bs {
                    let row_off = (y_start + dy) * w_usize + x_start;
                    let o_slice = &orig_luma[row_off..row_off + bs];
                    let l_slice = &lev_luma[row_off..row_off + bs];
                    for i in 0..bs {
                        let po = o_slice[i];
                        let pl = l_slice[i];
                        if po < o_min {
                            o_min = po;
                        }
                        if po > o_max {
                            o_max = po;
                        }
                        if pl < l_min {
                            l_min = pl;
                        }
                        if pl > l_max {
                            l_max = pl;
                        }
                    }
                }

                if (o_max as i32 - o_min as i32) < min_dynamic_range as i32 {
                    continue;
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
            let raw = rgb.as_raw();
            if raw.len() >= 1_500_000 {
                hist = raw
                    .par_chunks_exact(3 * 16384)
                    .map(|chunk| {
                        let mut local = [0u32; 256];
                        for px in chunk.chunks_exact(3) {
                            let y = ((54 * px[0] as u32 + 183 * px[1] as u32 + 19 * px[2] as u32 + 128) >> 8) as usize;
                            local[y] += 1;
                        }
                        local
                    })
                    .reduce(
                        || [0u32; 256],
                        |mut a, b| {
                            for i in 0..256 {
                                a[i] += b[i];
                            }
                            a
                        },
                    );
                let rem_start = (raw.len() / (3 * 16384)) * (3 * 16384);
                for px in raw[rem_start..].chunks_exact(3) {
                    let y = ((54 * px[0] as u32 + 183 * px[1] as u32 + 19 * px[2] as u32 + 128) >> 8) as usize;
                    hist[y] += 1;
                }
            } else {
                for chunk in raw.chunks_exact(3) {
                    let y = ((54 * chunk[0] as u32 + 183 * chunk[1] as u32 + 19 * chunk[2] as u32 + 128) >> 8) as usize;
                    hist[y] += 1;
                }
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
    let expected_lut = build_monotonic_tone_lut(plan.black_point as f32, plan.white_point as f32);

    // Tier 1 histogram tail slope erosion check (evaluated at the mapped spline bin `lut_y[bp]`,
    // never at endpoint bin 0 which contains digital typesetting spikes!)
    if config.quality_guards.tier1.check_tail_erosion && plan.black_point > 0 {
        let bp = plan.black_point as usize;
        let bp_plus_4 = (bp + 4).min(255);
        let out_bp = expected_lut[bp] as usize;
        let out_bp4 = expected_lut[bp_plus_4].max(expected_lut[bp] + 1) as usize;
        let orig_slope = orig_hist[bp_plus_4] as i64 - orig_hist[bp] as i64;
        let lev_slope = lev_hist[out_bp4.min(255)] as i64 - lev_hist[out_bp.min(255)] as i64;

        if orig_slope > (total_pixels as f64 * 0.05).round() as i64 && lev_slope < -(total_pixels as f64 * 0.02) as i64 {
            let mut res = TierResult::reject(1, "Midtone gradient crushed into black floor");
            res.metrics.insert("orig_slope".into(), orig_slope as f64);
            res.metrics.insert("lev_slope".into(), lev_slope as f64);
            return res;
        }
    }

    // Evaluate MSE & PSNR relative to the monotonic Fritsch-Carlson spline curve so legitimate
    // tonal expansion (e.g. lo=45->8 or hi=224->248) is not conflated with unmodeled distortion,
    // while gross errors (like lev=200 when orig=128) are immediately rejected.
    let (sum_sq, valid_count) = match (original, leveled) {
        (DynamicImage::ImageRgba8(orig_rgba), DynamicImage::ImageRgba8(lev_rgba)) => {
            let (s, c) = orig_rgba
                .as_raw()
                .par_chunks_exact(4)
                .zip(lev_rgba.as_raw().par_chunks_exact(4))
                .filter(|(o, _)| o[3] >= 128)
                .map(|(o, l)| {
                    let yo = ((54 * o[0] as u32 + 183 * o[1] as u32 + 19 * o[2] as u32 + 128) >> 8) as usize;
                    let yl = ((54 * l[0] as u32 + 183 * l[1] as u32 + 19 * l[2] as u32 + 128) >> 8) as f64;
                    let expected_y = expected_lut[yo] as f64;
                    let diff = yl - expected_y;
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
                    let expected_y = expected_lut[o[0] as usize] as f64;
                    let diff = l[0] as f64 - expected_y;
                    (diff * diff, 1u64)
                })
                .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            (s, c as f64)
        }
        _ => {
            let orig_luma = extract_fast_luma_vec(original);
            let lev_luma = extract_fast_luma_vec(leveled);
            let total = orig_luma.len() as f64;
            let step = if orig_luma.len() >= 1_000_000 { 4usize } else { 1usize };
            let sampled_n = (orig_luma.len() / step).max(1) as f64;
            let s: f64 = orig_luma
                .par_iter()
                .step_by(step)
                .zip(lev_luma.par_iter().step_by(step))
                .map(|(&a, &b)| {
                    let expected_y = expected_lut[a as usize] as f64;
                    let diff = b as f64 - expected_y;
                    diff * diff
                })
                .sum();
            (s * (total / sampled_n), total)
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
        let mut res = TierResult::reject(
            1,
            format!(
                "MSE {:.1} exceeded threshold {:.1}",
                mse, config.quality_guards.tier1.mse_threshold
            ),
        );
        res.metrics = metrics;
        return res;
    }

    if psnr < config.quality_guards.tier1.psnr_min as f64 {
        let mut res = TierResult::reject(
            1,
            format!(
                "PSNR {:.1}dB below minimum {:.1}dB",
                psnr, config.quality_guards.tier1.psnr_min
            ),
        );
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
            format!(
                "SSIM {:.3} below threshold {:.2}",
                ssim, config.quality_guards.tier2.ssim_threshold
            ),
        );
        res.metrics = metrics;
        return res;
    }

    if classification == PageClassification::Color || classification == PageClassification::Mixed {
        let orig_rgb = original.to_rgb8();
        let lev_rgb = leveled.to_rgb8();
        let total_pixels = (orig_rgb.width() * orig_rgb.height()) as usize;
        let step = (total_pixels / 2000).max(1);

        let mut chroma_deltas = Vec::new();
        let has_paper_cast = plan.paper_cast_gains.is_some();
        let ramp_start = plan.paper_cast_ramp_start;

        for (op, lp) in orig_rgb
            .pixels()
            .step_by(step)
            .zip(lev_rgb.pixels().step_by(step))
        {
            let (or, og, ob) = (op[0], op[1], op[2]);
            let orig_y = ((54 * or as u32 + 183 * og as u32 + 19 * ob as u32 + 128) >> 8) as u8;
            let orig_chroma = or.max(og).max(ob) - or.min(og).min(ob);

            if has_paper_cast && orig_y >= ramp_start && orig_chroma <= 65 {
                continue;
            }

            let (l_orig, a_orig, b_orig) = rgb_to_lab_fast(or, og, ob);
            let c_orig = (a_orig * a_orig + b_orig * b_orig).sqrt();

            // Quality Guard Exemption Mask (Upgrade D.5):
            // Pixels intentionally neutralized by the Dual-Boundary Sigmoid Chroma Gate (taper < 1.0)
            // or sitting on the yellow paper cast axis are exempted from triggering false color-drift rollbacks.
            let taper = compute_sigmoid_taper_exact(l_orig, c_orig);
            let is_yellow_paper_axis =
                l_orig >= 65.0 && a_orig.abs() <= 4.5 && b_orig > 0.0 && c_orig < 36.0;
            if taper < 1.0 || is_yellow_paper_axis {
                continue;
            }

            let (_l_lev, a_lev, b_lev) = rgb_to_lab_fast(lp[0], lp[1], lp[2]);
            let delta_a = a_orig - a_lev;
            let delta_b = b_orig - b_lev;
            let delta_chroma = (delta_a * delta_a + delta_b * delta_b).sqrt();
            chroma_deltas.push(delta_chroma);
        }

        if !chroma_deltas.is_empty() {
            chroma_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p95_idx =
                ((chroma_deltas.len() as f32 * 0.95).floor() as usize).min(chroma_deltas.len() - 1);
            let med_idx = chroma_deltas.len() / 2;

            let p95_delta = chroma_deltas[p95_idx];
            let median_delta = chroma_deltas[med_idx];

            metrics.insert("chroma_delta_p95".into(), p95_delta as f64);
            metrics.insert("chroma_delta_median".into(), median_delta as f64);

            if p95_delta > config.quality_guards.tier2.chroma_delta_p95 {
                let mut res = TierResult::reject(
                    2,
                    format!(
                        "Color drift: p95 Chroma Delta={:.2} > {:.2}",
                        p95_delta, config.quality_guards.tier2.chroma_delta_p95
                    ),
                );
                res.metrics = metrics;
                return res;
            }

            if median_delta > config.quality_guards.tier2.chroma_delta_median {
                let mut res = TierResult::reject(
                    2,
                    format!(
                        "Color drift: median Chroma Delta={:.2} > {:.2}",
                        median_delta, config.quality_guards.tier2.chroma_delta_median
                    ),
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

    let stride = config.quality_guards.tier3.stride;
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
                format!(
                    "Local contrast collapsed: p05 ratio {:.2} < {:.2}",
                    p05_ratio, config.quality_guards.tier3.min_contrast_ratio
                ),
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
    let p_num = page_number
        .map(|p| p.to_string())
        .unwrap_or_else(|| "-".to_string());

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

    let is_monochrome = analysis.classification == PageClassification::Monochrome;

    // Strict Full-Color Veto or mode filter (Zero-Copy No-Op)
    if analysis.classification == PageClassification::Color
        || (config.mode == AutoLevelsMode::Manga && analysis.classification == PageClassification::Color)
        || (config.mode == AutoLevelsMode::Color && analysis.classification != PageClassification::Color)
    {
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

    // Generate correction plan directly from validated defects
    let mut plan = generate_correction_plan(&analysis.defects, analysis.classification, config);

    // BookProfile Outlier Sanity Guard (Upgrade B.5):
    // If book_profile has >= 5 monochrome/mixed pages, clamp any single page's `lo`
    // so it does not exceed `median_book_black_floor + 20`.
    if let Some(profile) = book_profile {
        let group = match analysis.classification {
            PageClassification::Monochrome => &profile.monochrome_group,
            PageClassification::Mixed => &profile.mixed_group,
            PageClassification::Color => &profile.color_group,
        };
        if group.page_count >= 5 {
            let max_allowed = group.median_black_floor.saturating_add(20);
            if plan.black_point > max_allowed {
                plan.black_point = max_allowed;
            }
        }
    }

    // Per-channel dark shelf alignment check
    if let Some(pcs) = detect_per_channel_shelf(&analysis, &plan) {
        plan.per_channel_black = Some(pcs);
    }

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

    // Fast strided subimage extraction for Quality Guard on large >= 2MP images (< 0.10ms on 6MP)
    fn sample_for_quality_guard(img: &DynamicImage) -> DynamicImage {
        let (w, h) = img.dimensions();
        if (w as usize) * (h as usize) < 2_000_000 {
            return img.clone();
        }
        match img {
            DynamicImage::ImageRgb8(rgb) => {
                let stride = 12usize;
                let sw = (w as usize / stride).max(1);
                let sh = (h as usize / stride).max(1);
                let raw = rgb.as_raw();
                let w_usize = w as usize;
                let mut out = vec![0u8; sw * sh * 3];
                for sy in 0..sh {
                    let row_off = (sy * stride) * w_usize * 3;
                    let dst_off = sy * sw * 3;
                    for sx in 0..sw {
                        let src_idx = row_off + (sx * stride) * 3;
                        let dst_idx = dst_off + sx * 3;
                        out[dst_idx] = raw[src_idx];
                        out[dst_idx + 1] = raw[src_idx + 1];
                        out[dst_idx + 2] = raw[src_idx + 2];
                    }
                }
                if let Some(sub) = image::RgbImage::from_raw(sw as u32, sh as u32, out) {
                    return DynamicImage::ImageRgb8(sub);
                }
            }
            DynamicImage::ImageRgba8(rgba) => {
                let stride = 12usize;
                let sw = (w as usize / stride).max(1);
                let sh = (h as usize / stride).max(1);
                let raw = rgba.as_raw();
                let w_usize = w as usize;
                let mut out = vec![0u8; sw * sh * 4];
                for sy in 0..sh {
                    let row_off = (sy * stride) * w_usize * 4;
                    let dst_off = sy * sw * 4;
                    for sx in 0..sw {
                        let src_idx = row_off + (sx * stride) * 4;
                        let dst_idx = dst_off + sx * 4;
                        out[dst_idx] = raw[src_idx];
                        out[dst_idx + 1] = raw[src_idx + 1];
                        out[dst_idx + 2] = raw[src_idx + 2];
                        out[dst_idx + 3] = raw[src_idx + 3];
                    }
                }
                if let Some(sub) = image::RgbaImage::from_raw(sw as u32, sh as u32, out) {
                    return DynamicImage::ImageRgba8(sub);
                }
            }
            _ => {}
        }
        img.clone()
    }

    let is_large_page = (width as usize) * (height as usize) >= 2_000_000;
    let guard_pristine = sample_for_quality_guard(&image);
    let full_pristine_opt = if !is_large_page {
        Some(image.clone())
    } else {
        None
    };
    let redecode_pristine = || -> DynamicImage {
        if let Some(ref cloned) = full_pristine_opt {
            return cloned.clone();
        }
        if !_original_bytes.is_empty() {
            if let Ok(decoded) = image::load_from_memory(_original_bytes) {
                return decoded;
            }
        }
        guard_pristine.clone()
    };

    if plan.use_oklab {
        apply_color_correction_oklab(&mut image, &plan);
    } else {
        apply_monochrome_mixed_correction(&mut image, &plan, analysis.classification);
    }

    let guard_leveled = if is_large_page {
        sample_for_quality_guard(&image)
    } else {
        image.clone()
    };
    let mut lev_hist = compute_image_luminance_histogram(&guard_leveled);
    if is_large_page {
        for slot in lev_hist.iter_mut() {
            *slot = slot.saturating_mul(144);
        }
    }

    let t1 = quality_tier_1(
        &analysis.luminance_histogram,
        &lev_hist,
        &guard_pristine,
        &guard_leveled,
        &plan,
        analysis.total_pixels,
        config,
    );

    let mut tier_results = Vec::new();
    if t1.outcome == Outcome::Reject {
        warn!(
            "[QualityGuard] REJECT book_id={} page={} tier=1 reason=\"{}\" metrics={:?}",
            b_id,
            p_num,
            t1.reason.as_deref().unwrap_or(""),
            t1.metrics
        );
        tier_results.push(t1);
        let verdict = QualityVerdict::reject(tier_results, false);
        let rolled_back = redecode_pristine();
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

    let t2 = quality_tier_2(&guard_pristine, &guard_leveled, analysis.classification, &plan, config);
    let mut retry_needed = false;
    if t2.outcome == Outcome::Reject {
        warn!(
            "[QualityGuard] REJECT book_id={} page={} tier=2 reason=\"{}\" metrics={:?}",
            b_id,
            p_num,
            t2.reason.as_deref().unwrap_or(""),
            t2.metrics
        );
        tier_results.push(t2);
        retry_needed = true;
    } else {
        tier_results.push(t2);
    }

    if !retry_needed {
        let t3 = quality_tier_3(&guard_pristine, &guard_leveled, config);
        if t3.outcome == Outcome::Reject {
            warn!(
                "[QualityGuard] REJECT book_id={} page={} tier=3 reason=\"{}\" metrics={:?}",
                b_id,
                p_num,
                t3.reason.as_deref().unwrap_or(""),
                t3.metrics
            );
            tier_results.push(t3);
            retry_needed = true;
        } else {
            tier_results.push(t3);
        }
    }

    if retry_needed {
        let mut attenuated_plan = plan.clone();
        attenuated_plan.black_point = (plan.black_point as f32 * 0.5).round() as u8;
        attenuated_plan.white_point =
            255 - ((255.0 - plan.white_point as f32) * 0.5).round() as u8;
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

        let mut retry_image = redecode_pristine();
        if attenuated_plan.use_oklab {
            apply_color_correction_oklab(&mut retry_image, &attenuated_plan);
        } else {
            apply_monochrome_mixed_correction(&mut retry_image, &attenuated_plan, analysis.classification);
        }

        let guard_retry = if is_large_page {
            sample_for_quality_guard(&retry_image)
        } else {
            retry_image.clone()
        };
        let mut retry_lev_hist = compute_image_luminance_histogram(&guard_retry);
        if is_large_page {
            for slot in retry_lev_hist.iter_mut() {
                *slot = slot.saturating_mul(144);
            }
        }

        let rt1 = quality_tier_1(
            &analysis.luminance_histogram,
            &retry_lev_hist,
            &guard_pristine,
            &guard_retry,
            &attenuated_plan,
            analysis.total_pixels,
            config,
        );
        let rt2 = quality_tier_2(
            &guard_pristine,
            &guard_retry,
            analysis.classification,
            &attenuated_plan,
            config,
        );
        let rt3 = quality_tier_3(&guard_pristine, &guard_retry, config);

        if rt1.outcome == Outcome::Pass && rt2.outcome == Outcome::Pass && rt3.outcome == Outcome::Pass {
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
                    channel_black_points: attenuated_plan
                        .per_channel_black
                        .map(|p| [p.0, p.1, p.2]),
                    quality_verdict: Some(verdict.clone()),
                },
                verdict,
            );
        } else {
            let rolled_back = redecode_pristine();
            let verdict = QualityVerdict {
                outcome: Outcome::Reject,
                attenuation_attempted: true,
                tier_results: vec![rt1, rt2, rt3],
            };
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

// ============================================================================
// Ground-Truth Positive (M1-M4) & Adversarial Trap (T1-T7) Regression Suite
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::upscaler::upscaler::fold_monochrome_post_upscale;
    use image::{Rgb, RgbImage};

    #[inline]
    fn rec709_y(r: u8, g: u8, b: u8) -> u8 {
        ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8
    }

    #[test]
    fn test_m1_page5_digital_spike_and_washed_out_drawing_ink() {
        // 1000 x 1500 page: 80% neutral paper (Y=248), 1.2% pure digital black (Y=0),
        // 5% pure digital white (Y=255), 13.8% washed-out drawing ink in Y in [42, 52].
        let (w, h) = (1000u32, 1500u32);
        let total = (w * h) as usize;
        let mut raw = vec![248u8; total * 3];

        let n_black = (total as f64 * 0.012).round() as usize; // 18,000 px
        let n_white = (total as f64 * 0.050).round() as usize; // 75,000 px
        let n_ink = (total as f64 * 0.138).round() as usize; // 207,000 px

        for i in 0..n_black {
            raw[i * 3] = 0;
            raw[i * 3 + 1] = 0;
            raw[i * 3 + 2] = 0;
        }
        for i in n_black..(n_black + n_white) {
            raw[i * 3] = 255;
            raw[i * 3 + 1] = 255;
            raw[i * 3 + 2] = 255;
        }
        for i in (n_black + n_white)..(n_black + n_white + n_ink) {
            let y_val = 42 + ((i % 11) as u8); // 42..=52, includes 45
            raw[i * 3] = y_val;
            raw[i * 3 + 1] = y_val;
            raw[i * 3 + 2] = y_val;
        }

        let img = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, raw).unwrap());
        let config = AutoLevelsConfig::default();
        let dummy = Bytes::new();
        let (leveled, info, verdict) = level_and_verify(img, &dummy, None, &config);

        assert!(info.was_leveled, "M1 must be leveled");
        assert_eq!(verdict.outcome, Outcome::Pass, "3-Tier Quality Guard must pass");
        assert!(!verdict.attenuation_attempted, "Must pass without attenuation rollback");
        assert!(
            info.black_point >= 40 && info.black_point <= 52,
            "Validated lo={} must be in [40, 52]",
            info.black_point
        );

        let lev_rgb = leveled.to_rgb8();
        // Check a pixel that had Y_in = 45
        let idx_45 = (n_black + n_white)..(n_black + n_white + n_ink);
        let sample_i = idx_45.clone().find(|&i| 42 + (i % 11) == 45).unwrap();
        let px = lev_rgb.as_raw();
        let y_out = rec709_y(px[sample_i * 3], px[sample_i * 3 + 1], px[sample_i * 3 + 2]);
        assert!(
            y_out <= 12,
            "Washed-out drawing ink (Y=45) must restore to Y_out <= 12 (got {})",
            y_out
        );
    }

    #[test]
    fn test_m2_page12_localized_margin_yellowing_and_dirty_shadow_tint() {
        // 1000 x 1500 page: interior panels (65%) clean paper [246, 245, 240],
        // outer margin cells (25%) acid yellowing [245, 232, 185],
        // shadows (10%) muddy sepia tint [27, 18, 6].
        let (w, h) = (1000u32, 1500u32);
        let mut img = RgbImage::new(w, h);

        for y in 0..h {
            for x in 0..w {
                let in_outer_margin = x < 100 || x >= 900 || y < 100 || y >= 1400;
                let is_shadow = !in_outer_margin && ((x + y * 3) % 8 == 0);
                let px = if in_outer_margin {
                    Rgb([245, 232, 185])
                } else if is_shadow {
                    Rgb([27, 18, 6])
                } else {
                    Rgb([246, 245, 240])
                };
                img.put_pixel(x, y, px);
            }
        }

        let dyn_img = DynamicImage::ImageRgb8(img);
        let config = AutoLevelsConfig::default();
        let dummy = Bytes::new();
        let (leveled, info, verdict) = level_and_verify(dyn_img, &dummy, None, &config);

        assert!(info.was_leveled, "M2 must be leveled/neutralized");
        assert_eq!(verdict.outcome, Outcome::Pass);

        let lev_rgb = leveled.to_rgb8();
        let margin_px = lev_rgb.get_pixel(20, 20);
        let (_, ma, mb) = rgb_to_lab_fast(margin_px[0], margin_px[1], margin_px[2]);
        let margin_c = (ma * ma + mb * mb).sqrt();
        assert!(
            margin_c <= 2.0,
            "Yellow outer margin must be neutralized to C* <= 2.0 (got {:.2})",
            margin_c
        );

        let shadow_px = lev_rgb.get_pixel(200, 200); // (200 + 600) % 8 == 0
        let (_, sa, sb) = rgb_to_lab_fast(shadow_px[0], shadow_px[1], shadow_px[2]);
        let shadow_c = (sa * sa + sb * sb).sqrt();
        assert!(
            shadow_c <= 2.0,
            "Muddy shadow tint must be neutralized to C* <= 2.0 (got {:.2})",
            shadow_c
        );
    }

    #[test]
    fn test_m3_noisy_scanned_monochrome_manga_classification_and_luma8_folding() {
        // 1000 x 1500 page: 85% paper [245, 245, 244] and 15% dark ink [5, 4, 3].
        let (w, h) = (1000u32, 1500u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                if (x + y) % 7 == 0 {
                    img.put_pixel(x, y, Rgb([5, 4, 3]));
                } else {
                    img.put_pixel(x, y, Rgb([245, 245, 244]));
                }
            }
        }
        let dyn_img = DynamicImage::ImageRgb8(img);
        let config = AutoLevelsConfig::default();
        let analysis = analyze_page(&dyn_img, 1, &config);
        assert_eq!(
            analysis.classification,
            PageClassification::Monochrome,
            "[5, 4, 3] dark ink noise must classify as Monochrome"
        );

        let (leveled, info) = analyze_and_level_image(dyn_img, &config);
        assert!(info.is_monochrome);
        let folded = fold_monochrome_post_upscale(leveled, &info, true);
        assert!(
            matches!(folded, DynamicImage::ImageLuma8(_)),
            "Post-upscale output must fold to 1-channel DynamicImage::ImageLuma8"
        );
    }

    #[test]
    fn test_m4_uniform_aged_vintage_scan() {
        // 1000 x 1500 page: ink floor at Y = 26, aged paper ceiling at [232, 224, 198], no digital spike.
        let (w, h) = (1000u32, 1500u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                if (x + y) % 7 == 0 {
                    img.put_pixel(x, y, Rgb([26, 26, 26]));
                } else {
                    img.put_pixel(x, y, Rgb([232, 224, 198]));
                }
            }
        }
        let dyn_img = DynamicImage::ImageRgb8(img);
        let config = AutoLevelsConfig::default();
        let (leveled, info, verdict) = level_and_verify(dyn_img, &Bytes::new(), None, &config);

        assert!(info.was_leveled);
        assert_eq!(verdict.outcome, Outcome::Pass);
        assert!(info.black_point >= 24 && info.black_point <= 28);
        assert!(info.white_point >= 220 && info.white_point <= 228);

        let lev_rgb = leveled.to_rgb8();
        let ink_px = lev_rgb.get_pixel(0, 0);
        let ink_y = rec709_y(ink_px[0], ink_px[1], ink_px[2]);
        assert!(ink_y <= 10, "Output ink must reach Y' <= 10 (got {})", ink_y);

        let paper_px = lev_rgb.get_pixel(1, 0);
        let paper_y = rec709_y(paper_px[0], paper_px[1], paper_px[2]);
        let (_, pa, pb) = rgb_to_lab_fast(paper_px[0], paper_px[1], paper_px[2]);
        let paper_c = (pa * pa + pb * pb).sqrt();
        assert!(
            paper_y >= 246,
            "Output paper must reach Y' >= 246 (got {})",
            paper_y
        );
        assert!(
            paper_c <= 2.0,
            "Output paper must have neutral chroma C* <= 2.0 (got {:.2})",
            paper_c
        );
    }

    #[test]
    fn test_t1_pristine_digital_release_trap() {
        let (w, h) = (1000u32, 1500u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let v = if idx < 60_000 {
                    0u8
                } else if idx < 120_000 {
                    ((idx - 60_000) * 254 / 60_000) as u8
                } else {
                    255u8
                };
                img.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        let orig = DynamicImage::ImageRgb8(img);
        let orig_raw = orig.as_rgb8().unwrap().as_raw().clone();
        let config = AutoLevelsConfig::default();
        let (leveled, info, _) = level_and_verify(orig, &Bytes::new(), None, &config);

        assert!(!info.was_leveled, "T1 Pristine digital release must be vetoed");
        assert_eq!(
            leveled.as_rgb8().unwrap().as_raw(),
            &orig_raw,
            "T1 output must be bit-exact (Delta E = 0.0, MSE = 0.0)"
        );
    }

    #[test]
    fn test_t2_full_color_cover_webtoon_trap() {
        let (w, h) = (1000u32, 1500u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                // 22% vibrant chromatic pixels
                let px = if idx < 330_000 {
                    Rgb([40, 120, 220])
                } else {
                    Rgb([240, 240, 240])
                };
                img.put_pixel(x, y, px);
            }
        }
        let orig = DynamicImage::ImageRgb8(img);
        let orig_raw = orig.as_rgb8().unwrap().as_raw().clone();
        let config = AutoLevelsConfig::default();
        let (leveled, info, _) = level_and_verify(orig, &Bytes::new(), None, &config);

        assert_eq!(info.classification, PageClassification::Color);
        assert!(!info.was_leveled, "T2 Full-color page must be strictly vetoed");
        assert_eq!(leveled.as_rgb8().unwrap().as_raw(), &orig_raw);
    }

    #[test]
    fn test_t3_pastel_sky_and_watercolor_skin_trap() {
        let (w, h) = (1000u32, 1500u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let px = if idx < 150_000 {
                    Rgb([220, 235, 245]) // 10% pastel blue sky (L*=92.1, C*=7.17)
                } else if idx < 300_000 {
                    Rgb([30, 30, 30]) // 10% washed-out ink (Y=30)
                } else {
                    Rgb([250, 250, 250]) // 80% white paper
                };
                img.put_pixel(x, y, px);
            }
        }
        let orig = DynamicImage::ImageRgb8(img);
        let config = AutoLevelsConfig::default();
        let (leveled, info, verdict) = level_and_verify(orig, &Bytes::new(), None, &config);

        assert_eq!(info.classification, PageClassification::Mixed);
        assert!(info.was_leveled);
        assert_eq!(verdict.outcome, Outcome::Pass);

        let lev_rgb = leveled.to_rgb8();
        let sky_px = lev_rgb.get_pixel(0, 0);
        let diff_br = sky_px[2] as i32 - sky_px[0] as i32;
        assert!(
            diff_br >= 15,
            "Pastel blue sky must retain B' - R' >= 15 (got {:?})",
            sky_px
        );
        let (_, a0, b0) = rgb_to_lab_fast(220, 235, 245);
        let (_, a1, b1) = rgb_to_lab_fast(sky_px[0], sky_px[1], sky_px[2]);
        let de_chroma = ((a0 - a1).powi(2) + (b0 - b1).powi(2)).sqrt();
        assert!(
            de_chroma <= 1.5,
            "Pastel sky chromaticity drift must be <= 1.5 (got {:.2})",
            de_chroma
        );

        let ink_y = (150_000 / w) + 1;
        let ink_px = lev_rgb.get_pixel(0, ink_y);
        assert!(
            rec709_y(ink_px[0], ink_px[1], ink_px[2]) <= 10,
            "Washed-out ink (Y=30) must be restored to Y' <= 10"
        );
    }

    #[test]
    fn test_t4_high_key_snow_fog_flashback_trap() {
        let (w, h) = (1000u32, 1500u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let v = if idx < 22_500 {
                    0u8 // 1.5% pure black panel border
                } else {
                    210 + ((idx % 41) as u8) // 98.5% high-key fog in [210, 250]
                };
                img.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        let orig = DynamicImage::ImageRgb8(img);
        let config = AutoLevelsConfig::default();
        let (leveled, info, _) = level_and_verify(orig, &Bytes::new(), None, &config);

        assert_eq!(info.black_point, 0, "High-key veto must clamp lo to 0");
        let lev_luma = extract_fast_luma_vec(&leveled);
        let mean_y: f64 = lev_luma.iter().map(|&v| v as f64).sum::<f64>() / lev_luma.len() as f64;
        assert!(
            mean_y >= 215.0,
            "High-key scene mean luminance must remain >= 215.0 (got {:.1})",
            mean_y
        );
    }

    #[test]
    fn test_t5_low_key_nighttime_splash_page_trap_and_spline_monotonicity() {
        let (w, h) = (1000u32, 1500u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) as usize;
                let v = (idx % 111) as u8; // 0..=110 (p95 <= 110 < 130)
                img.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        let orig = DynamicImage::ImageRgb8(img);
        let config = AutoLevelsConfig::default();
        let (_leveled, info, _) = level_and_verify(orig, &Bytes::new(), None, &config);

        assert_eq!(
            info.white_point, 255,
            "Low-key night scene must veto highlight stretching (hi = 255)"
        );

        // Verify Fritsch-Carlson LUT monotonicity even when hi = 90 < 128 or hi = 110 < 128
        for &hi_test in &[90.0f32, 110.0f32, 128.0f32, 224.0f32, 255.0f32] {
            for &lo_test in &[0.0f32, 18.0f32, 45.0f32, 70.0f32] {
                if hi_test - lo_test >= 15.0 {
                    let lut = build_monotonic_tone_lut(lo_test, hi_test);
                    for i in 0..255 {
                        assert!(
                            lut[i + 1] >= lut[i],
                            "Spline non-monotonic at i={} for (lo={}, hi={}): {} < {}",
                            i,
                            lo_test,
                            hi_test,
                            lut[i + 1],
                            lut[i]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_t6_microscopic_color_accent_logo_trap() {
        // 96% monochrome page with dirty paper tint (C* = 4.5 -> RGB=[245, 243, 234])
        // plus 3% red publisher logo ([210, 25, 25], C* > 60).
        let (w, h) = (1000u32, 1500u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let in_logo = x >= 100 && x < 250 && y >= 100 && y < 400; // 150x300 = 45,000 px (3.0%)
                let is_ink = !in_logo && ((x + y) % 15 == 0);
                let px = if in_logo {
                    Rgb([210, 25, 25])
                } else if is_ink {
                    Rgb([20, 20, 20])
                } else {
                    Rgb([245, 243, 234])
                };
                img.put_pixel(x, y, px);
            }
        }

        let orig = DynamicImage::ImageRgb8(img);
        let config = AutoLevelsConfig::default();
        let (leveled, info, verdict) = level_and_verify(orig, &Bytes::new(), None, &config);

        assert_eq!(info.classification, PageClassification::Mixed);
        assert!(info.was_leveled);
        assert_eq!(verdict.outcome, Outcome::Pass);

        let lev_rgb = leveled.to_rgb8();
        let logo_px = lev_rgb.get_pixel(150, 200);
        assert!(
            logo_px[0] >= 205 && logo_px[1] <= 30 && logo_px[2] <= 30,
            "Red logo must be 100% preserved (got {:?})",
            logo_px
        );

        let paper_px = lev_rgb.get_pixel(500, 500);
        let (_, pa, pb) = rgb_to_lab_fast(paper_px[0], paper_px[1], paper_px[2]);
        let paper_c = (pa * pa + pb * pb).sqrt();
        assert!(
            paper_c <= 1.5,
            "Dirty paper tint must be neutralized to C* <= 1.5 (got {:.2})",
            paper_c
        );

        let post = fold_monochrome_post_upscale(leveled, &info, true);
        assert!(
            matches!(post, DynamicImage::ImageRgb8(_)),
            "Mixed page must retain 3-channel RGB after upscale"
        );
    }

    #[test]
    fn test_t7_intentional_warm_sepia_artwork_trap() {
        let (w, h) = (1000u32, 1500u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                if (x + y) % 8 == 0 {
                    img.put_pixel(x, y, Rgb([40, 30, 15])); // Warm sepia shadow (R > B + 6)
                } else {
                    img.put_pixel(x, y, Rgb([245, 230, 200])); // Warm sepia highlight
                }
            }
        }
        let orig = DynamicImage::ImageRgb8(img);
        let orig_raw = orig.as_rgb8().unwrap().as_raw().clone();
        let config = AutoLevelsConfig::default();
        let (leveled, info, _) = level_and_verify(orig, &Bytes::new(), None, &config);

        assert!(
            !info.was_leveled,
            "T7 Intentional warm sepia artwork must not be bleached"
        );
        assert_eq!(leveled.as_rgb8().unwrap().as_raw(), &orig_raw);
    }

    #[test]
    fn test_6mp_latency_budget_veto_and_restoration() {
        // Warm up Lazy static tables
        let _ = &*FAST_LAB_TABLES;
        let (w, h) = (2000u32, 3000u32); // 6,000,000 pixels
        let total = (w * h) as usize;
        let config = AutoLevelsConfig::default();
        let dummy = Bytes::new();

        // 1. 6MP Vetoed Page (T2 Full-Color Cover)
        let mut color_raw = vec![240u8; total * 3];
        for i in 0..(total / 3) {
            color_raw[i * 3] = 35;
            color_raw[i * 3 + 1] = 110;
            color_raw[i * 3 + 2] = 225;
        }
        let color_img = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, color_raw).unwrap());
        let mut veto_ms = f64::MAX;
        for _ in 0..3 {
            let img_c = color_img.clone();
            let t0 = Instant::now();
            let (_out_v, info_v, _) = level_and_verify(img_c, &dummy, None, &config);
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            assert!(!info_v.was_leveled);
            if dt < veto_ms {
                veto_ms = dt;
            }
        }

        // 2. 6MP Full Restoration Page (M1 Page 5 Digital Spike + Washed-Out Ink)
        let mut m1_raw = vec![248u8; total * 3];
        let n_black = (total as f64 * 0.012) as usize;
        let n_ink = (total as f64 * 0.138) as usize;
        for i in 0..n_black {
            m1_raw[i * 3] = 0;
            m1_raw[i * 3 + 1] = 0;
            m1_raw[i * 3 + 2] = 0;
        }
        for i in n_black..(n_black + n_ink) {
            let v = 42 + ((i % 11) as u8);
            m1_raw[i * 3] = v;
            m1_raw[i * 3 + 1] = v;
            m1_raw[i * 3 + 2] = v;
        }
        let m1_img = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, m1_raw).unwrap());
        let mut restore_ms = f64::MAX;
        for _ in 0..5 {
            let img_m = m1_img.clone();
            let t1 = Instant::now();
            let (_out_r, info_r, verdict_r) = level_and_verify(img_m, &dummy, None, &config);
            let dt = t1.elapsed().as_secs_f64() * 1000.0;
            assert!(info_r.was_leveled);
            assert_eq!(verdict_r.outcome, Outcome::Pass);
            if dt < restore_ms {
                restore_ms = dt;
            }
        }

        println!(
            "[Benchmark 6MP] Veto latency: {:.3} ms, Full Restoration latency: {:.3} ms",
            veto_ms, restore_ms
        );
        #[cfg(not(debug_assertions))]
        {
            assert!(
                veto_ms < 1.5,
                "6MP vetoed page latency {:.3}ms exceeded 1.5ms budget",
                veto_ms
            );
            assert!(
                restore_ms < 15.0,
                "6MP full restoration latency {:.3}ms exceeded 15.0ms budget",
                restore_ms
            );
        }
    }
}
