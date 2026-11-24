#![allow(clippy::needless_range_loop)]

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::LazyLock;
use std::time::Instant;

use image::{DynamicImage, GenericImageView, GrayAlphaImage, GrayImage, RgbImage, RgbaImage};
use log::{info, warn};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::config::app_config::AutoLevelsConfig;
use crate::upscaler::book_profile::BookProfile;

// Calibrated pipeline & Quality Guard constants (kept internal so `config.yml` stays minimal)
const BLACK_CLIP_PERCENT: f64 = 0.1;
const WHITE_CLIP_PERCENT: f64 = 0.1;
const PROTECT_SEPIA_ARTWORK: bool = true;
const MAX_BLACK_SHIFT: u8 = 85;
const MIN_WHITE_THRESHOLD: u8 = 200;
const DEFAULT_GAMMA: f32 = 1.0;

const GUARD_TIER1_CHECK_TAIL_EROSION: bool = true;
const GUARD_TIER1_MIN_DYNAMIC_RANGE: f32 = 64.0;
const GUARD_TIER2_SSIM_THRESHOLD: f32 = 0.92;
const GUARD_TIER2_CHROMA_DELTA_P95: f32 = 2.5;
const GUARD_TIER2_CHROMA_DELTA_MEDIAN: f32 = 1.5;
const GUARD_TIER3_GRADIENT_NCC: f32 = 0.90;
const GUARD_TIER3_MIN_CONTRAST_RATIO: f32 = 0.50;
const GUARD_TIER3_STRIDE: usize = 4;

// ============================================================================
// Precomputed Fast CIELAB Lookup Tables (D65, std::sync::LazyLock)
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
    /// Direct 1-cycle `L*` lookup for neutral gray pixels `r == g == b`.
    pub gray_l: [f32; 256],
    /// L1d-resident 2050-entry cube-root table (`8.2 KB`, total struct `13.7 KB` fits inside 32 KB L1d).
    /// Branchless linear interpolation across `[0..=16384]` (exact on the linear toe `v <= 144`, `< 0.008` max CIELAB error).
    pub lab_f: [f32; 2050],
}

pub static FAST_LAB_TABLES: LazyLock<FastLabTables> = LazyLock::new(|| {
    let mut x_r = [0u16; 256];
    let mut x_g = [0u16; 256];
    let mut x_b = [0u16; 256];
    let mut y_r = [0u16; 256];
    let mut y_g = [0u16; 256];
    let mut y_b = [0u16; 256];
    let mut z_r = [0u16; 256];
    let mut z_g = [0u16; 256];
    let mut z_b = [0u16; 256];
    let mut gray_l = [0.0f32; 256];

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

    let delta: f64 = 6.0 / 29.0;
    let delta_cubed = delta * delta * delta;
    let factor = (1.0 / 3.0) * (29.0 / 6.0) * (29.0 / 6.0);
    let offset = 4.0 / 29.0;

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

        let fy = if lin > delta_cubed {
            lin.cbrt()
        } else {
            factor * lin + offset
        };
        gray_l[v] = (116.0 * fy - 16.0).max(0.0) as f32;
    }

    let mut lab_f = [0.0f32; 2050];

    for (i, slot) in lab_f.iter_mut().enumerate() {
        let t = ((i * 8) as f64 / SCALE).min(1.0);
        let f = if t > delta_cubed {
            t.cbrt()
        } else {
            factor * t + offset
        };
        *slot = f as f32;
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
        gray_l,
        lab_f,
    }
});

#[inline(always)]
pub fn rec709_y(r: u8, g: u8, b: u8) -> u8 {
    ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8
}

#[inline(always)]
fn eval_lab_f(t: &FastLabTables, val: usize) -> f32 {
    let idx = (val >> 3).min(2048);
    let frac = (val & 7) as f32 * 0.125;
    let f0 = t.lab_f[idx];
    f0 + (t.lab_f[idx + 1] - f0) * frac
}

#[inline(always)]
fn rgb_to_lab_with_tables(t: &FastLabTables, r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    if r == g && g == b {
        return (t.gray_l[r as usize], 0.0, 0.0);
    }
    let (ri, gi, bi) = (r as usize, g as usize, b as usize);
    let yi = (t.y_r[ri] + t.y_g[gi] + t.y_b[bi]) as usize;
    let fy = eval_lab_f(t, yi);
    let l_star = (116.0 * fy - 16.0).max(0.0);
    let xi = (t.x_r[ri] + t.x_g[gi] + t.x_b[bi]) as usize;
    let zi = (t.z_r[ri] + t.z_g[gi] + t.z_b[bi]) as usize;
    let fx = eval_lab_f(t, xi);
    let fz = eval_lab_f(t, zi);
    let a_star = 500.0 * (fx - fy);
    let b_star = 200.0 * (fy - fz);
    (l_star, a_star, b_star)
}

#[inline(always)]
pub fn rgb_to_lab_fast(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    rgb_to_lab_with_tables(&FAST_LAB_TABLES, r, g, b)
}

#[inline(always)]
fn smoothstep_inv(edge0: f32, inv_span: f32, x: f32) -> f32 {
    let t = ((x - edge0) * inv_span).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ============================================================================
// Data Structures
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PageClassification {
    Monochrome,
    Mixed,
    Color,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DefectType {
    ElevatedBlackFloor {
        floor_value: u8,
        is_bimodal: bool,
    },
    PaperCast {
        cast_a: f32,
        cast_b: f32,
        cast_magnitude: f32,
        delta_grid: f32,
        is_localized_margin: bool,
    },
    DullHighlights {
        white_point: u8,
    },
}

/// Spatial paper-cast field measured over the page's 4x4 cell grid (excluding pure digital white L* >= 99).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PaperCastField {
    pub active: bool,
    pub localized_margin: bool,
    pub global_a: f32,
    pub global_b: f32,
    pub global_c: f32,
    pub dom_a: f32,
    pub dom_b: f32,
    pub dom_u_a: f32,
    pub dom_u_b: f32,
    pub c_max: f32,
    pub cell_a: [f32; 16],
    pub cell_b: [f32; 16],
}

impl Default for PaperCastField {
    fn default() -> Self {
        Self {
            active: false,
            localized_margin: false,
            global_a: 0.0,
            global_b: 0.0,
            global_c: 0.0,
            dom_a: 0.0,
            dom_b: 0.0,
            dom_u_a: 0.0,
            dom_u_b: 0.0,
            c_max: 0.0,
            cell_a: [0.0; 16],
            cell_b: [0.0; 16],
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CorrectionPlan {
    pub black_point: u8,
    pub white_point: u8,
    pub gamma: f32,
    /// Scaling factor in `[0.0, 1.0]` for cast neutralization (`0.5` on attenuated retry).
    pub cast_strength: f32,
    pub cast_field: PaperCastField,
    pub has_muddy_shadow_cast: bool,
    pub has_high_chroma_accents: bool,
    pub skip_correction: bool,
}

impl Default for CorrectionPlan {
    fn default() -> Self {
        Self {
            black_point: 0,
            white_point: 255,
            gamma: 1.0,
            cast_strength: 0.0,
            cast_field: PaperCastField::default(),
            has_muddy_shadow_cast: false,
            has_high_chroma_accents: false,
            skip_correction: true,
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
    /// True when `correct_paper_cast == false` intentionally preserved warm/tinted paper chroma on a
    /// Monochrome RGB page, signaling `upscaler.rs` to skip post-upscale `Luma8` folding.
    pub paper_cast_preserved: bool,
    pub classification: PageClassification,
    pub black_point: u8,
    pub white_point: u8,
    pub quality_verdict: Option<QualityVerdict>,
}

impl Default for LevelInfo {
    fn default() -> Self {
        Self {
            was_leveled: false,
            is_monochrome: false,
            paper_cast_preserved: false,
            classification: PageClassification::Color,
            black_point: 0,
            white_point: 255,
            quality_verdict: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DualZoneSample {
    pub count: usize,
    pub avg_rgb: (f32, f32, f32),
}

#[derive(Debug, Clone)]
pub struct PageAnalysis {
    pub classification: PageClassification,
    pub luminance_histogram: [u32; 256],
    pub total_pixels: usize,
    pub highlight_sample: DualZoneSample,
    pub shadow_sample: DualZoneSample,
    pub paper_cast: PaperCastField,
    pub has_paper_cast: bool,
    pub has_muddy_shadow_cast: bool,
    pub has_high_chroma_accents: bool,
    pub defects: Vec<DefectType>,
    pub black_floor: u8,
    pub white_ceiling: u8,
    pub is_shelf_detected: bool,
}

// ============================================================================
// Fast Grayscale Check (Used on Early Bypass / Disabled Auto-Levels)
// ============================================================================

/// Fast pre-flight grayscale check (`< 0.5 ms` on 6MP) used when auto-levels is disabled
/// or bypassed on small images so post-upscale `Luma8` folding remains accurate.
pub fn is_image_grayscale(image: &DynamicImage) -> bool {
    match image {
        DynamicImage::ImageLuma8(_)
        | DynamicImage::ImageLumaA8(_)
        | DynamicImage::ImageLuma16(_)
        | DynamicImage::ImageLumaA16(_) => true,
        DynamicImage::ImageRgb8(rgb) => is_rgb_slice_grayscale::<3>(rgb.as_raw(), rgb.width() as usize, rgb.height() as usize),
        DynamicImage::ImageRgba8(rgba) => is_rgb_slice_grayscale::<4>(rgba.as_raw(), rgba.width() as usize, rgba.height() as usize),
        DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgb32F(_) => {
            let rgb = image.to_rgb8();
            is_rgb_slice_grayscale::<3>(rgb.as_raw(), rgb.width() as usize, rgb.height() as usize)
        }
        DynamicImage::ImageRgba16(_) | DynamicImage::ImageRgba32F(_) => {
            let rgba = image.to_rgba8();
            is_rgb_slice_grayscale::<4>(rgba.as_raw(), rgba.width() as usize, rgba.height() as usize)
        }
        _ => {
            let rgb = image.to_rgb8();
            is_rgb_slice_grayscale::<3>(rgb.as_raw(), rgb.width() as usize, rgb.height() as usize)
        }
    }
}

fn is_rgb_slice_grayscale<const CHANNELS: usize>(raw: &[u8], w: usize, h: usize) -> bool {
    let total_px = w * h;
    if total_px == 0 {
        return false;
    }
    let stride = if total_px >= 4_000_000 {
        17usize
    } else if total_px >= 1_500_000 {
        11usize
    } else if total_px >= 250_000 {
        5usize
    } else {
        1usize
    };

    let tables = &*FAST_LAB_TABLES;
    let mut sampled = 0usize;
    let mut colored = 0usize;
    let mut accents = 0usize;

    for y in (0..h).step_by(stride) {
        let row_off = y * w * CHANNELS;
        let x_start = if stride > 1 { ((y / stride) * 3) % stride } else { 0 };
        for x in (x_start..w).step_by(stride) {
            let idx = row_off + x * CHANNELS;
            if CHANNELS == 4 && raw[idx + 3] < 128 {
                continue;
            }
            sampled += 1;
            let (r, g, b) = (raw[idx], raw[idx + 1], raw[idx + 2]);
            if r == g && g == b {
                continue;
            }
            let (l, a, bl) = rgb_to_lab_with_tables(tables, r, g, b);
            let c_sq = a * a + bl * bl;
            if c_sq >= 36.0 && l >= 2.0 {
                colored += 1;
                if c_sq >= 225.0 {
                    accents += 1;
                }
            }
        }
    }

    if sampled == 0 {
        return false;
    }
    let r_color = colored as f32 / sampled as f32;
    let r_accent = accents as f32 / sampled as f32;
    r_color <= 0.015 && r_accent < 0.0005
}

// ============================================================================
// Histogram Percentiles, Effective Black/White & Dual-Valley Validation
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
pub fn histogram_reverse_percentile(hist: &[u32; 256], total: u64, pct_from_top: f64) -> f32 {
    if total == 0 {
        return 255.0;
    }
    let target = ((total as f64) * (pct_from_top / 100.0)).ceil().max(1.0) as u64;
    let mut acc = 0u64;
    for (bin, &count) in hist.iter().enumerate().rev() {
        acc += count as u64;
        if acc >= target {
            return bin as f32;
        }
    }
    0.0
}

#[inline]
pub fn nonzero_histogram_percentile(hist: &[u32; 256], pct: f64) -> (u64, f32) {
    let nz_total: u64 = hist[3..256].iter().map(|&c| c as u64).sum();
    if nz_total == 0 {
        return (0, 0.0);
    }
    let target = ((nz_total as f64) * (pct / 100.0)).ceil().max(1.0) as u64;
    let mut acc = 0u64;
    for (offset, &count) in hist[3..256].iter().enumerate() {
        acc += count as u64;
        if acc >= target {
            return (nz_total, (offset + 3) as f32);
        }
    }
    (nz_total, 255.0)
}

/// Computes the 2% cumulative density bin in the shadow band `hist[3..120]` if shadow mass >= 0.8%.
#[inline]
fn shadow_band_p2(hist_y: &[u32; 256], hist_total: u64) -> Option<f32> {
    let shadow_total: u64 = hist_y[3..120].iter().map(|&c| c as u64).sum();
    if (shadow_total as f64 / hist_total as f64) < 0.008 {
        return None;
    }
    let target = ((shadow_total as f64) * 0.02).ceil().max(1.0) as u64;
    let mut acc = 0u64;
    for (offset, &count) in hist_y[3..120].iter().enumerate() {
        acc += count as u64;
        if acc >= target {
            return Some((offset + 3) as f32);
        }
    }
    Some(3.0)
}

/// Computes the 2% cumulative density bin in the shadow band `hist[3..120]` when a digital
/// typesetting black spike (`0.0005 < m_zero < 0.035`) is present.
pub fn find_effective_black(hist_y: &[u32; 256], p1: f32) -> f32 {
    let total: u64 = hist_y.iter().map(|&c| c as u64).sum();
    if total == 0 {
        return p1;
    }
    let zero_mass = (hist_y[0] as u64 + hist_y[1] as u64 + hist_y[2] as u64) as f64 / total as f64;
    if zero_mass > 0.0005
        && zero_mass < 0.035
        && let Some(p2) = shadow_band_p2(hist_y, total)
        && p2 > p1 + 4.0
    {
        return p2;
    }
    p1
}

/// Finds the effective paper white ceiling when digital speech bubble fills spike at `253..=255`
/// while the actual scanned paper forms a validated bimodal peak in `[200..=251]` with a clear
/// separation valley below the digital white spike (protecting light-gray screentones on clean pages).
pub fn find_effective_white(hist_y: &[u32; 256], p99: f32) -> f32 {
    let total: u64 = hist_y.iter().map(|&c| c as u64).sum();
    if total == 0 || p99 < 254.0 {
        return p99;
    }
    let digital_white_mass =
        (hist_y[253] as u64 + hist_y[254] as u64 + hist_y[255] as u64) as f64 / total as f64;
    if !(0.0005..0.25).contains(&digital_white_mass) {
        return p99;
    }

    let paper_total: u64 = hist_y[200..253].iter().map(|&c| c as u64).sum();
    let paper_mass = paper_total as f64 / total as f64;
    // Actual page paper occupies a substantial fraction of the page (>= 40%) and dominates digital bubbles.
    if paper_mass < 0.40 || paper_mass <= digital_white_mass {
        return p99;
    }

    let h_smooth = smooth_histogram_binomial_5(hist_y);
    let mut w_peak = 0.0f32;
    let mut w_peak_idx = 200usize;
    for (j, &val) in h_smooth.iter().enumerate().take(251).skip(200) {
        if val > w_peak {
            w_peak = val;
            w_peak_idx = j;
        }
    }
    if w_peak < 0.005 * total as f32 {
        return p99;
    }

    // Require separation valley between the dull paper peak and the digital white spike (253..=255)
    let mut min_after = f32::MAX;
    for &val in h_smooth.iter().take(253).skip(w_peak_idx + 1) {
        if val < min_after {
            min_after = val;
        }
    }
    // Require valley below the dull paper peak so continuous bright ramps aren't falsely flagged
    let mut min_before = f32::MAX;
    for &val in h_smooth.iter().take(w_peak_idx).skip(140) {
        if val < min_before {
            min_before = val;
        }
    }
    if min_after >= 0.35 * w_peak || min_before >= 0.35 * w_peak {
        return p99;
    }

    let target = ((paper_total as f64) * 0.98).ceil().max(1.0) as u64;
    let mut acc = 0u64;
    let mut idx_p98 = 252usize;
    for (offset, &count) in hist_y[200..253].iter().enumerate() {
        acc += count as u64;
        if acc >= target {
            idx_p98 = offset + 200;
            break;
        }
    }
    p99.min(idx_p98 as f32)
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

struct ValidatedShadowPeak {
    left_foot: usize,
    valley_idx: usize,
}

/// Unified pre-peak valley, detached dark-ink, attached FWHM-skirt, and post-peak valley
/// validation shared by `detect_compact_ink_shelf` (`pre_trough_ratio = 0.25`, `compact = true`)
/// and `validate_black_floor` (`pre_trough_ratio = 0.35`, `compact = false`) whenever a digital
/// black spike (`0.0005 < m_zero < 0.035`) is present.
fn validate_spike_separated_peak(
    hist: &[u32; 256],
    h_smooth: &[f32; 256],
    hist_total: u64,
    peak_idx: usize,
    hi: f32,
    pre_trough_ratio: f32,
    compact: bool,
) -> Option<ValidatedShadowPeak> {
    const START_TROUGH: usize = 5; // Kernel radius 2 excludes the 0..=2 digital black spike
    if peak_idx < 8 {
        return None;
    }
    let h_peak = h_smooth[peak_idx];

    // 1. FWHM boundaries around peak_idx on the 5-tap smoothed histogram
    let half_max = h_peak * 0.5;
    let mut left_fwhm = peak_idx;
    while left_fwhm > 0 && h_smooth[left_fwhm] >= half_max {
        left_fwhm -= 1;
    }
    let mut right_fwhm = peak_idx;
    while right_fwhm < 255 && h_smooth[right_fwhm] >= half_max {
        right_fwhm += 1;
    }
    let fwhm = right_fwhm.saturating_sub(left_fwhm + 1).max(1);
    if compact && fwhm > 8 {
        return None;
    }

    // 2. Descend LEFT from peak_idx to the immediate local valley minimum (stopping on climb or
    // flat plateau once below pre_trough_ratio, without any absolute flat_eps freeze that breaks
    // monotonically decaying JPEG DCT Gibbs ringing skirts).
    let mut min_before = h_peak;
    let mut trough_idx = peak_idx;
    for j in (START_TROUGH..peak_idx).rev() {
        let val = h_smooth[j];
        if val < min_before {
            min_before = val;
            trough_idx = j;
        } else if min_before < pre_trough_ratio * h_peak {
            break;
        }
    }
    if trough_idx == peak_idx || min_before >= pre_trough_ratio * h_peak {
        return None;
    }

    // 3. Peak core mass & left half-max foot (strictly disjoint from hist[0..=trough_idx])
    let left_foot = (left_fwhm + 1).max(trough_idx + 1).min(peak_idx);
    let (core_lo, core_hi) = if compact {
        (
            peak_idx.saturating_sub(2).max(trough_idx + 1),
            (peak_idx + 2).min(255),
        )
    } else {
        (
            (left_fwhm + 1)
                .min(peak_idx.saturating_sub(2))
                .max(trough_idx + 1),
            right_fwhm
                .saturating_sub(1)
                .max((peak_idx + 2).min(255))
                .min(255),
        )
    };
    let peak_core_mass: u64 = hist[core_lo..=core_hi].iter().map(|&c| c as u64).sum();
    if compact && (peak_core_mass as f64 / hist_total as f64) < 0.010 {
        return None;
    }

    // 4. Detached pre-valley dark-ink caps (eliminates 0..=2 / 3..=trough split-cap loopholes and
    // blocks jumping over detached authentic dark ink in Y = 3..=trough_idx).
    let below_trough_count: u64 = hist[0..=trough_idx].iter().map(|&c| c as u64).sum();
    let below_trough_mass = below_trough_count as f64 / hist_total as f64;
    if below_trough_mass >= 0.035 {
        return None;
    }
    let detached_nz_count: u64 = hist[3..=trough_idx].iter().map(|&c| c as u64).sum();
    if (detached_nz_count as f64 / hist_total as f64) >= 0.015 {
        return None;
    }
    if below_trough_mass >= 0.002
        && peak_core_mass < (below_trough_count as f64 * 1.25).round() as u64
    {
        return None;
    }

    // 5. Attached pre-foot skirt dominance: JPEG DCT Gibbs undershoot occurs along the 1-3px edge
    // of a washed panel (bounded as <= 35% of the peak's core mass), whereas broad/sloped/flat
    // dark-charcoal artwork below the peak exceeds this ratio and is strictly vetoed.
    let pre_foot_skirt: u64 = hist[3..left_foot].iter().map(|&c| c as u64).sum();
    if pre_foot_skirt > (peak_core_mass as f64 * 0.35).round() as u64 {
        return None;
    }

    // 6. Post-peak bimodal valley validation (GEMINI.md §2.2)
    let search_valley_hi = (hi.floor() as usize).min(140);
    let start_valley = (peak_idx + 1).min(search_valley_hi);
    if start_valley >= search_valley_hi {
        return None;
    }
    let mut min_after = f32::MAX;
    let mut valley_idx = start_valley;
    for (j, &val) in h_smooth
        .iter()
        .enumerate()
        .take(search_valley_hi + 1)
        .skip(start_valley)
    {
        if val < min_after {
            min_after = val;
            valley_idx = j;
        }
    }
    if min_after < 0.35 * h_peak {
        Some(ValidatedShadowPeak {
            left_foot,
            valley_idx,
        })
    } else {
        None
    }
}

/// Validates a candidate black floor in `[2.0, 85.0]` against High-Key scenes, digital black
/// spikes, and flat midtone ramps using BOTH pre-peak trough + FWHM-skirt validation (when
/// `m_zero > 0.0005`) and post-peak bimodal valley validation across the full `[2..=85]` range.
pub fn validate_black_floor(hist_y: &[u32; 256], cand_lo: f32, hi: f32) -> f32 {
    let hist_total: u64 = hist_y.iter().map(|&c| c as u64).sum();
    if hist_total == 0 || !cand_lo.is_finite() || cand_lo < 2.0 || cand_lo > 85.0 {
        return 0.0;
    }

    // High-Key Scene Protection: require M_ink (3..=85) >= 0.008 (aligned with max ink floor 85)
    let ink_mass_count: u64 = hist_y[3..=85].iter().map(|&c| c as u64).sum();
    if (ink_mass_count as f64 / hist_total as f64) < 0.008 {
        return 0.0;
    }

    let zero_count = hist_y[0] as u64 + hist_y[1] as u64 + hist_y[2] as u64;
    let m_zero = zero_count as f64 / hist_total as f64;
    if m_zero >= 0.035 {
        return 0.0;
    }

    let h_smooth = smooth_histogram_binomial_5(hist_y);
    let lo_idx = cand_lo.floor() as usize;
    let search_peak_hi = (lo_idx + 15).min(85);
    // Start at bin 5 if a digital black spike exists so the 5-tap kernel radius (2) doesn't bleed in
    let start_peak = if m_zero > 0.0005 { 5 } else { 2 };
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
    // If argmax landed on a rising flank at the search boundary, climb to the actual local peak
    // (within the shadow band <= 85) so FWHM and skirt checks evaluate the real peak.
    while peak_idx < 85 && h_smooth[peak_idx + 1] > h_peak {
        peak_idx += 1;
        h_peak = h_smooth[peak_idx];
    }
    if h_smooth[peak_idx + 1] > h_peak || h_peak < 0.001 * hist_total as f32 {
        return 0.0;
    }

    if m_zero > 0.0005 {
        return validate_spike_separated_peak(
            hist_y, &h_smooth, hist_total, peak_idx, hi, 0.35, false,
        )
        .map_or(0.0, |v| cand_lo.min(v.valley_idx as f32));
    }

    // Post-peak bimodal valley validation (guards ALL cand_lo in [2..=85], including <= 35)
    let search_valley_hi = (hi.floor() as usize).min(140);
    let start_valley = (peak_idx + 1).min(search_valley_hi);
    if start_valley >= search_valley_hi {
        return 0.0;
    }

    let mut min_after = f32::MAX;
    let mut valley_idx = start_valley;
    for (j, &val) in h_smooth.iter().enumerate().take(search_valley_hi + 1).skip(start_valley) {
        if val < min_after {
            min_after = val;
            valley_idx = j;
        }
    }

    if min_after < 0.35 * h_peak {
        cand_lo.min(valley_idx as f32)
    } else {
        0.0
    }
}

// ============================================================================
// Inversion-Proof Adaptive Fritsch-Carlson Monotonic Cubic Spline LUT
// ============================================================================

/// Builds a strictly monotonic 256-entry tone curve LUT using Fritsch-Carlson cubic Hermite
/// interpolation with unit gamma (`gamma = 1.0`).
pub fn build_monotonic_tone_lut(lo: f32, hi: f32) -> [u8; 256] {
    build_monotonic_tone_lut_gamma(lo, hi, 1.0)
}

/// Builds a strictly monotonic 256-entry tone curve LUT with inversion-proof adaptive soft-toe
/// and soft-shoulder knots and gamma-aware midpoint `P2`.
pub fn build_monotonic_tone_lut_gamma(lo: f32, hi: f32, gamma: f32) -> [u8; 256] {
    let identity = || -> [u8; 256] {
        let mut id = [0u8; 256];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = i as u8;
        }
        id
    };

    if !lo.is_finite() || !hi.is_finite() {
        return identity();
    }

    let gamma_c = if gamma.is_finite() {
        gamma.clamp(0.2, 5.0) as f64
    } else {
        1.0
    };

    let lo_c = lo.clamp(0.0, 240.0) as f64;
    let hi_c = hi.clamp((lo_c as f32 + 15.0).max(15.0), 255.0) as f64;

    if lo_c < 2.0 && hi_c > 253.0 && (gamma_c - 1.0).abs() < 1e-4 {
        return identity();
    }

    // Inversion-proof adaptive knots: strictly darkens lo -> [0..=2] and brightens hi -> [253..=255]
    let y_lo = if lo_c < 2.0 {
        0.0
    } else {
        (0.15 * lo_c).min(2.0)
    };
    let y_hi = if hi_c > 253.0 {
        255.0
    } else {
        255.0 - (0.15 * (255.0 - hi_c)).min(2.0)
    };
    let y_mid = y_lo + (y_hi - y_lo) * 0.5_f64.powf(1.0 / gamma_c);

    let mut knots = [(0.0f64, 0.0f64); 5];
    let mut n = 0usize;
    knots[n] = (0.0, 0.0);
    n += 1;
    if lo_c >= 2.0 {
        knots[n] = (lo_c, y_lo);
        n += 1;
    }
    knots[n] = ((lo_c + hi_c) * 0.5, y_mid);
    n += 1;
    if hi_c <= 253.0 {
        knots[n] = (hi_c, y_hi);
        n += 1;
    }
    knots[n] = (255.0, 255.0);
    n += 1;

    let mut deltas = [0.0f64; 4];
    for k in 0..(n - 1) {
        let dx = (knots[k + 1].0 - knots[k].0).max(1e-6);
        deltas[k] = (knots[k + 1].1 - knots[k].1) / dx;
    }

    let mut d = [0.0f64; 5];
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
    for (x_val, slot) in lut.iter_mut().enumerate() {
        let xf = x_val as f64;
        if xf <= knots[0].0 {
            *slot = knots[0].1.round().clamp(0.0, 255.0) as u8;
            continue;
        }
        if xf >= knots[n - 1].0 {
            *slot = knots[n - 1].1.round().clamp(0.0, 255.0) as u8;
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
        *slot = y_val.round().clamp(0.0, 255.0) as u8;
    }

    // Final integer monotonicity guard
    for i in 1..256 {
        if lut[i] < lut[i - 1] {
            lut[i] = lut[i - 1];
        }
    }

    lut
}

// ============================================================================
// Page Analysis: 4x4 Grid Modal Paper Peak, Residual Chroma & Sepia Guard
// ============================================================================

#[derive(Clone, Copy)]
struct ChromaticSample {
    l: f32,
    a: f32,
    b: f32,
}

struct SamplingAcc {
    hist_y: [u32; 256],
    total_valid_pixels: usize,
    sampled_pixels: usize,
    // 58-bin modal paper peak histogram for L* in [70.0, 99.0)
    paper_l_hist: [u32; 58],
    paper_bin_sum_a: [f64; 58],
    paper_bin_sum_b: [f64; 58],
    pure_white_paper_count: u32,
    max_low_key_l: f32,
    // 4x4 spatial grid paper (a*, b*) histograms strictly excluding L* >= 99.0:
    // a* in -12..=12 (25 bins, +12 offset), b* in -42..=42 (85 bins, +42 offset)
    cell_a_hist: [[u32; 25]; 16],
    cell_b_hist: [[u32; 85]; 16],
    cell_count: [u32; 16],
    // Chromatic samples (C* >= 6.0 && L* >= 2.0) for exact residual chroma classification
    chromatic_samples: Vec<ChromaticSample>,
    chromatic_midtone_count: usize,
    raw_colored_count: usize,
    raw_accent_count: usize,
    midtone_cutoff: usize,
    // 1-entry RGB -> (Y, L*, a*, b*, C*^2) cache across strided reads
    prev_key: u32,
    prev_y: u8,
    prev_lab_csq: (f32, f32, f32, f32),
    // Highlight & shadow dual-zone stats
    hl_count: usize,
    hl_sum_r: u64,
    hl_sum_g: u64,
    hl_sum_b: u64,
    sh_count: usize,
    sh_sum_r: u64,
    sh_sum_g: u64,
    sh_sum_b: u64,
    sh_lab_count: usize,
    sh_sum_lab_a: f64,
    sh_sum_lab_b: f64,
}

impl SamplingAcc {
    fn new() -> Self {
        Self {
            hist_y: [0u32; 256],
            total_valid_pixels: 0,
            sampled_pixels: 0,
            paper_l_hist: [0u32; 58],
            paper_bin_sum_a: [0.0; 58],
            paper_bin_sum_b: [0.0; 58],
            pure_white_paper_count: 0,
            max_low_key_l: 0.0,
            cell_a_hist: [[0u32; 25]; 16],
            cell_b_hist: [[0u32; 85]; 16],
            cell_count: [0u32; 16],
            chromatic_samples: Vec::new(),
            chromatic_midtone_count: 0,
            raw_colored_count: 0,
            raw_accent_count: 0,
            midtone_cutoff: usize::MAX,
            prev_key: u32::MAX,
            prev_y: 0,
            prev_lab_csq: (0.0, 0.0, 0.0, 0.0),
            hl_count: 0,
            hl_sum_r: 0,
            hl_sum_g: 0,
            hl_sum_b: 0,
            sh_count: 0,
            sh_sum_r: 0,
            sh_sum_g: 0,
            sh_sum_b: 0,
            sh_lab_count: 0,
            sh_sum_lab_a: 0.0,
            sh_sum_lab_b: 0.0,
        }
    }

    #[inline(always)]
    fn push_rgb(&mut self, tables: &FastLabTables, rgb: [u8; 3], cell_idx: usize, scale_weight: u32) {
        let [r, g, b] = rgb;
        let (y_lum, l_star, a_star, b_star, c_sq) = if r == g && g == b {
            (r, tables.gray_l[r as usize], 0.0f32, 0.0f32, 0.0f32)
        } else {
            let key = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
            if key == self.prev_key {
                let (l, a, bl, csq) = self.prev_lab_csq;
                (self.prev_y, l, a, bl, csq)
            } else {
                let y = rec709_y(r, g, b);
                let (l, a, bl) = rgb_to_lab_with_tables(tables, r, g, b);
                let csq = a * a + bl * bl;
                self.prev_key = key;
                self.prev_y = y;
                self.prev_lab_csq = (l, a, bl, csq);
                (y, l, a, bl, csq)
            }
        };

        self.hist_y[y_lum as usize] += scale_weight;
        self.total_valid_pixels += scale_weight as usize;
        self.sampled_pixels += 1;

        if l_star < 70.0 && l_star > self.max_low_key_l {
            self.max_low_key_l = l_star;
        }

        if c_sq >= 36.0 && l_star >= 2.0 {
            self.raw_colored_count += 1;
            if c_sq >= 225.0 {
                self.raw_accent_count += 1;
            }
            if (25.0..=70.0).contains(&l_star) && c_sq >= 100.0 {
                self.chromatic_midtone_count += 1;
            }
            // Once midtone chroma mass is guaranteed > 5% of all possible samples, cast exclusion
            // cannot activate, so we skip buffering further chromatic samples on full-color pages.
            if self.chromatic_midtone_count <= self.midtone_cutoff {
                if self.chromatic_samples.is_empty() {
                    self.chromatic_samples.reserve(1024);
                }
                self.chromatic_samples.push(ChromaticSample {
                    l: l_star,
                    a: a_star,
                    b: b_star,
                });
            }
        }

        // Paper candidate filter: (L* >= 65.0) & (C* < 42.0) & (|a*| < 12.0)
        // Strictly exclude pure digital white (L* >= 99.0) from BOTH paper_l_hist AND 4x4 cell histograms!
        if l_star >= 65.0 && c_sq < 1764.0 && a_star.abs() < 12.0 {
            if (70.0..99.0).contains(&l_star) {
                let bin = (((l_star - 70.0) * 2.0).floor() as usize).min(57);
                self.paper_l_hist[bin] += 1;
                self.paper_bin_sum_a[bin] += a_star as f64;
                self.paper_bin_sum_b[bin] += b_star as f64;

                let a_bin = (a_star.round() as i32 + 12).clamp(0, 24) as usize;
                let b_bin = (b_star.round() as i32 + 42).clamp(0, 84) as usize;
                self.cell_a_hist[cell_idx][a_bin] += 1;
                self.cell_b_hist[cell_idx][b_bin] += 1;
                self.cell_count[cell_idx] += 1;

                self.hl_count += 1;
                self.hl_sum_r += r as u64;
                self.hl_sum_g += g as u64;
                self.hl_sum_b += b as u64;
            } else if l_star >= 99.0 {
                self.pure_white_paper_count += 1;
            }
        }

        if (8..=45).contains(&y_lum) {
            self.sh_count += 1;
            self.sh_sum_r += r as u64;
            self.sh_sum_g += g as u64;
            self.sh_sum_b += b as u64;
            if c_sq < 625.0 {
                self.sh_lab_count += 1;
                self.sh_sum_lab_a += a_star as f64;
                self.sh_sum_lab_b += b_star as f64;
            }
        }
    }
}

fn sample_rgb_slice<const CHANNELS: usize>(raw: &[u8], w: usize, h: usize, acc: &mut SamplingAcc) {
    let total_px = w * h;
    let stride = if total_px >= 4_000_000 {
        17usize
    } else if total_px >= 1_500_000 {
        11usize
    } else if total_px >= 250_000 {
        5usize
    } else {
        1usize
    };
    let scale_weight = (stride * stride) as u32;
    let max_samples = h.div_ceil(stride) * w.div_ceil(stride);
    acc.midtone_cutoff = max_samples / 20;

    let cell_h = (h / 4).max(1);
    let cell_w = (w / 4).max(1);
    let tables = &*FAST_LAB_TABLES;

    for y_coord in (0..h).step_by(stride) {
        let grid_r = (y_coord / cell_h).min(3);
        let row_offset = y_coord * w * CHANNELS;
        let x_start = if stride > 1 {
            ((y_coord / stride) * 3) % stride
        } else {
            0
        };
        for x_coord in (x_start..w).step_by(stride) {
            let idx = row_offset + x_coord * CHANNELS;
            if CHANNELS == 4 && raw[idx + 3] < 128 {
                continue;
            }
            let grid_c = (x_coord / cell_w).min(3);
            let cell_idx = grid_r * 4 + grid_c;
            acc.push_rgb(
                tables,
                [raw[idx], raw[idx + 1], raw[idx + 2]],
                cell_idx,
                scale_weight,
            );
        }
    }
}

fn empty_page_analysis(hist_y: [u32; 256]) -> PageAnalysis {
    PageAnalysis {
        classification: PageClassification::Color,
        luminance_histogram: hist_y,
        total_pixels: 0,
        highlight_sample: DualZoneSample::default(),
        shadow_sample: DualZoneSample::default(),
        paper_cast: PaperCastField::default(),
        has_paper_cast: false,
        has_muddy_shadow_cast: false,
        has_high_chroma_accents: false,
        defects: Vec::new(),
        black_floor: 0,
        white_ceiling: 255,
        is_shelf_detected: false,
    }
}

pub fn analyze_page(
    image: &DynamicImage,
    _page_number: usize,
    _config: &AutoLevelsConfig,
) -> PageAnalysis {
    analyze_page_with_min_white(image, MIN_WHITE_THRESHOLD)
}

fn analyze_page_with_min_white(image: &DynamicImage, min_white_threshold: u8) -> PageAnalysis {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return empty_page_analysis([0; 256]);
    }

    // Normalize 16-bit and 32F formats cleanly without dropping luma or alpha semantics
    match image {
        DynamicImage::ImageLuma16(_) => {
            return analyze_page_with_min_white(
                &DynamicImage::ImageLuma8(image.to_luma8()),
                min_white_threshold,
            );
        }
        DynamicImage::ImageLumaA16(_) => {
            return analyze_page_with_min_white(
                &DynamicImage::ImageLumaA8(image.to_luma_alpha8()),
                min_white_threshold,
            );
        }
        DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgb32F(_) => {
            return analyze_page_with_min_white(
                &DynamicImage::ImageRgb8(image.to_rgb8()),
                min_white_threshold,
            );
        }
        DynamicImage::ImageRgba16(_) | DynamicImage::ImageRgba32F(_) => {
            return analyze_page_with_min_white(
                &DynamicImage::ImageRgba8(image.to_rgba8()),
                min_white_threshold,
            );
        }
        _ => {}
    }

    let is_already_luma = matches!(
        image,
        DynamicImage::ImageLuma8(_) | DynamicImage::ImageLumaA8(_)
    );

    let mut acc = SamplingAcc::new();

    match image {
        DynamicImage::ImageLuma8(luma) => {
            let raw = luma.as_raw();
            let (w_u, h_u) = (width as usize, height as usize);
            let total_px = w_u * h_u;
            let stride = if total_px >= 4_000_000 {
                17usize
            } else if total_px >= 1_500_000 {
                11usize
            } else if total_px >= 250_000 {
                5usize
            } else {
                1usize
            };
            let scale_weight = (stride * stride) as u32;
            if stride == 1 {
                for &p in raw {
                    acc.hist_y[p as usize] += 1;
                }
                acc.total_valid_pixels = total_px;
                acc.sampled_pixels = total_px;
            } else {
                for y_coord in (0..h_u).step_by(stride) {
                    let row_off = y_coord * w_u;
                    let x_start = ((y_coord / stride) * 3) % stride;
                    for x_coord in (x_start..w_u).step_by(stride) {
                        acc.hist_y[raw[row_off + x_coord] as usize] += scale_weight;
                        acc.sampled_pixels += 1;
                    }
                }
                acc.total_valid_pixels = acc.sampled_pixels * scale_weight as usize;
            }
            populate_luma_paper_peak(&acc.hist_y, scale_weight, &mut acc.paper_l_hist, &mut acc.pure_white_paper_count, &mut acc.max_low_key_l);
        }
        DynamicImage::ImageLumaA8(luma_a) => {
            let raw = luma_a.as_raw();
            let (w_u, h_u) = (width as usize, height as usize);
            let total_px = w_u * h_u;
            let stride = if total_px >= 4_000_000 {
                17usize
            } else if total_px >= 1_500_000 {
                11usize
            } else if total_px >= 250_000 {
                5usize
            } else {
                1usize
            };
            let scale_weight = (stride * stride) as u32;
            if stride == 1 {
                for chunk in raw.as_chunks::<2>().0 {
                    if chunk[1] >= 128 {
                        acc.hist_y[chunk[0] as usize] += 1;
                        acc.total_valid_pixels += 1;
                    }
                }
                acc.sampled_pixels = acc.total_valid_pixels;
            } else {
                for y_coord in (0..h_u).step_by(stride) {
                    let row_off = y_coord * w_u * 2;
                    let x_start = ((y_coord / stride) * 3) % stride;
                    for x_coord in (x_start..w_u).step_by(stride) {
                        let idx = row_off + x_coord * 2;
                        if raw[idx + 1] >= 128 {
                            acc.hist_y[raw[idx] as usize] += scale_weight;
                            acc.sampled_pixels += 1;
                        }
                    }
                }
                acc.total_valid_pixels = acc.sampled_pixels * scale_weight as usize;
            }
            populate_luma_paper_peak(&acc.hist_y, scale_weight, &mut acc.paper_l_hist, &mut acc.pure_white_paper_count, &mut acc.max_low_key_l);
        }
        DynamicImage::ImageRgb8(rgb) => {
            sample_rgb_slice::<3>(rgb.as_raw(), width as usize, height as usize, &mut acc);
        }
        DynamicImage::ImageRgba8(rgba) => {
            sample_rgb_slice::<4>(rgba.as_raw(), width as usize, height as usize, &mut acc);
        }
        _ => unreachable!("16-bit/32F formats normalized above"),
    }

    let hist_total: u64 = acc.hist_y.iter().map(|&c| c as u64).sum();
    if hist_total == 0 || acc.sampled_pixels == 0 {
        return empty_page_analysis(acc.hist_y);
    }

    // 1. Global Modal Paper Peak Detection in L* in [70.0, 99.0)
    let total_non_digital_paper: u64 = acc.paper_l_hist.iter().map(|&c| c as u64).sum();
    let min_paper_samples = ((acc.sampled_pixels / 200) as u64).clamp(10, 20);
    let (paper_l_peak, paper_a, paper_b, paper_c) = if total_non_digital_paper >= min_paper_samples {
        let mut smooth_l = [0.0f32; 58];
        for i in 0..58 {
            let left = if i > 0 { acc.paper_l_hist[i - 1] as f32 } else { 0.0 };
            let center = acc.paper_l_hist[i] as f32;
            let right = if i + 1 < 58 { acc.paper_l_hist[i + 1] as f32 } else { 0.0 };
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
        for b in bin_lo..=bin_hi {
            cnt += acc.paper_l_hist[b] as u64;
            sum_a += acc.paper_bin_sum_a[b];
            sum_b += acc.paper_bin_sum_b[b];
        }
        if cnt > 0 && !is_already_luma {
            let pa = (sum_a / cnt as f64) as f32;
            let pb = (sum_b / cnt as f64) as f32;
            (l_peak, pa, pb, pa.hypot(pb))
        } else {
            (l_peak, 0.0, 0.0, 0.0)
        }
    } else if (total_non_digital_paper + acc.pure_white_paper_count as u64) >= min_paper_samples {
        (99.5, 0.0, 0.0, 0.0)
    } else {
        (acc.max_low_key_l, 0.0, 0.0, 0.0)
    };

    // 2. 4x4 Spatial Grid Paper Medians (strictly excluding L* >= 99.0, arbitrary cast hue)
    let min_cell_samples = ((acc.sampled_pixels / 16) / 25).clamp(5, 25) as u32;
    let mut cell_a = [paper_a; 16];
    let mut cell_b = [paper_b; 16];
    let mut cell_valid = [false; 16];
    let mut valid_cell_count = 0usize;

    let mut min_a = f32::MAX;
    let mut max_a = f32::MIN;
    let mut min_b = f32::MAX;
    let mut max_b = f32::MIN;
    let mut max_valid_cell_c = 0.0f32;

    let mut max_border_c = 0.0f32;
    let mut min_border_c = f32::MAX;
    let mut max_border_ab = (paper_a, paper_b);
    let mut core_c_sum = 0.0f32;
    let mut core_a_sum = 0.0f32;
    let mut core_b_sum = 0.0f32;
    let mut valid_core_count = 0usize;

    if !is_already_luma {
        for c_idx in 0..16 {
            let cnt = acc.cell_count[c_idx];
            if cnt < min_cell_samples {
                continue;
            }
            let target = (cnt / 2).max(1);
            let mut med_a = 0.0f32;
            let mut run = 0u32;
            for (bin, &bc) in acc.cell_a_hist[c_idx].iter().enumerate() {
                run += bc;
                if run >= target {
                    med_a = (bin as i32 - 12) as f32;
                    break;
                }
            }
            let mut med_b = 0.0f32;
            run = 0;
            for (bin, &bc) in acc.cell_b_hist[c_idx].iter().enumerate() {
                run += bc;
                if run >= target {
                    med_b = (bin as i32 - 42) as f32;
                    break;
                }
            }
            let c_cell = med_a.hypot(med_b);
            if med_a.abs() >= 12.0 || c_cell >= 42.0 {
                continue;
            }

            cell_a[c_idx] = med_a;
            cell_b[c_idx] = med_b;
            cell_valid[c_idx] = true;
            valid_cell_count += 1;

            min_a = min_a.min(med_a);
            max_a = max_a.max(med_a);
            min_b = min_b.min(med_b);
            max_b = max_b.max(med_b);
            if c_cell > max_valid_cell_c {
                max_valid_cell_c = c_cell;
            }

            let r = c_idx / 4;
            let c = c_idx % 4;
            let is_border = r == 0 || r == 3 || c == 0 || c == 3;
            if is_border {
                if c_cell > max_border_c {
                    max_border_c = c_cell;
                    max_border_ab = (med_a, med_b);
                }
                if c_cell < min_border_c {
                    min_border_c = c_cell;
                }
            } else {
                core_c_sum += c_cell;
                core_a_sum += med_a;
                core_b_sum += med_b;
                valid_core_count += 1;
            }
        }
    }

    // Populate invalid interior core cells (e.g. obscured by dark panels or L* >= 99 speech bubbles)
    // from valid interior core cells rather than global border-skewed paper averages.
    if valid_core_count > 0 {
        let core_mean_a = core_a_sum / valid_core_count as f32;
        let core_mean_b = core_b_sum / valid_core_count as f32;
        for r in 1..=2 {
            for c in 1..=2 {
                let idx = r * 4 + c;
                if !cell_valid[idx] {
                    cell_a[idx] = core_mean_a;
                    cell_b[idx] = core_mean_b;
                }
            }
        }
    }

    let delta_b_grid = if valid_cell_count >= 4 { max_b - min_b } else { 0.0 };
    let delta_a_grid = if valid_cell_count >= 4 { max_a - min_a } else { 0.0 };
    let delta_grid = delta_b_grid.max(delta_a_grid);
    let core_ref_c = if valid_core_count > 0 {
        core_c_sum / valid_core_count as f32
    } else if min_border_c < f32::MAX {
        min_border_c
    } else {
        paper_c
    };
    let border_minus_core_c = (max_border_c - core_ref_c).max(0.0);

    // Require true perimeter margin geometry with consistent hue alignment (cos >= 0.70):
    // >= 3 cast corners OR opposite mid-edge pairs ((top & bottom) or (left & right)).
    // Prevents a single pastel sky/panel along one edge from impersonating localized margin
    // paper yellowing and stripping a color panel to Monochrome!
    let mut cast_corners = 0usize;
    let mut has_top_mid_cast = false;
    let mut has_bottom_mid_cast = false;
    let mut has_left_mid_cast = false;
    let mut has_right_mid_cast = false;
    if max_border_c > 1e-4 && border_minus_core_c >= 10.0 {
        for c_idx in 0..16 {
            if !cell_valid[c_idx] {
                continue;
            }
            let r = c_idx / 4;
            let c = c_idx % 4;
            let is_border = r == 0 || r == 3 || c == 0 || c == 3;
            if !is_border {
                continue;
            }
            let ma = cell_a[c_idx];
            let mb = cell_b[c_idx];
            let c_cell = ma.hypot(mb);
            if c_cell < core_ref_c + 10.0 {
                continue;
            }
            let cos_align =
                (ma * max_border_ab.0 + mb * max_border_ab.1) / (c_cell * max_border_c);
            if cos_align < 0.70 {
                continue;
            }
            let is_corner = (r == 0 || r == 3) && (c == 0 || c == 3);
            if is_corner {
                cast_corners += 1;
            } else if r == 0 {
                has_top_mid_cast = true;
            } else if r == 3 {
                has_bottom_mid_cast = true;
            } else if c == 0 {
                has_left_mid_cast = true;
            } else {
                has_right_mid_cast = true;
            }
        }
    }
    let has_perimeter_margin = cast_corners >= 3
        || (has_top_mid_cast && has_bottom_mid_cast)
        || (has_left_mid_cast && has_right_mid_cast);

    let has_localized_margin_cast =
        delta_grid >= 12.0 && border_minus_core_c >= 10.0 && has_perimeter_margin;
    let has_global_cast = paper_c > 3.5;

    let c_max = paper_c.max(max_valid_cell_c);
    let (dom_a, dom_b) = if has_localized_margin_cast {
        max_border_ab
    } else {
        (paper_a, paper_b)
    };
    let dom_c = dom_a.hypot(dom_b);
    let (dom_u_a, dom_u_b) = if dom_c > 1e-4 {
        (dom_a / dom_c, dom_b / dom_c)
    } else {
        (0.0, 0.0)
    };

    // 3. CIELAB Dual-Zone Sepia Guard (stride-invariant)
    let min_sh_samples = (acc.sampled_pixels / 1000).max(20);
    let (shadow_a, shadow_b, shadow_c) = if acc.sh_lab_count >= min_sh_samples {
        let sa = (acc.sh_sum_lab_a / acc.sh_lab_count as f64) as f32;
        let sb = (acc.sh_sum_lab_b / acc.sh_lab_count as f64) as f32;
        (sa, sb, sa.hypot(sb))
    } else {
        (0.0, 0.0, 0.0)
    };

    let is_intentional_sepia = !is_already_luma
        && PROTECT_SEPIA_ARTWORK
        && acc.sh_lab_count >= min_sh_samples
        && paper_b >= 6.0
        && shadow_b >= 5.0
        && shadow_a >= 0.5
        && delta_b_grid < 12.0;

    // 4. Residual Chroma Classification (R_res & R_accent)
    let p95_y = histogram_percentile(&acc.hist_y, hist_total, 95.0);
    let bright_paper_mass = acc.hist_y[200..=255]
        .iter()
        .map(|&c| c as u64)
        .sum::<u64>() as f64
        / hist_total as f64;

    let chromatic_midtone_mass = acc.chromatic_midtone_count as f32 / acc.sampled_pixels as f32;
    let allow_cast_exclusion = !is_intentional_sepia
        && p95_y >= 130.0
        && paper_l_peak >= 65.0
        && (has_global_cast || has_localized_margin_cast)
        && chromatic_midtone_mass <= 0.05
        && dom_c > 1e-4;
    let is_cast_muddy_shadow = |s: &ChromaticSample| -> bool {
        let c_sq = s.a * s.a + s.b * s.b;
        s.l < 45.0 && c_sq <= 100.0 && s.a.abs() <= 4.5 && s.b > 0.0
    };
    // On neutral/white paper (no verified paper cast), only treat true dark warm shadows
    // (L* < 22.0, sRGB Y <= 50) as muddy ink — never brown/olive midtones (Y = 51..106) —
    // and require real white paper (>= 30% in Y >= 200), non-low-key p95 >= 130, and
    // <= 5% non-muddy saturated pixels (C* >= 12 across ALL L*, catching dark blues & pastels).
    let is_neutral_muddy_shadow = |s: &ChromaticSample| -> bool {
        let c_sq = s.a * s.a + s.b * s.b;
        s.l < 22.0 && c_sq <= 100.0 && s.a.abs() <= 4.5 && s.b > 0.0
    };
    let allow_muddy_shadow_exclusion = !is_intentional_sepia
        && !allow_cast_exclusion
        && p95_y >= 130.0
        && paper_l_peak >= 65.0
        && bright_paper_mass >= 0.30
        && chromatic_midtone_mass <= 0.05
        && !acc.chromatic_samples.is_empty()
        && {
            let mut muddy_count = 0usize;
            let mut non_muddy_sat_count = 0usize;
            for s in &acc.chromatic_samples {
                if is_neutral_muddy_shadow(s) {
                    muddy_count += 1;
                } else if (s.a * s.a + s.b * s.b) >= 144.0 {
                    non_muddy_sat_count += 1;
                }
            }
            (muddy_count as f32 / acc.chromatic_samples.len() as f32) >= 0.60
                && (non_muddy_sat_count as f32 / acc.sampled_pixels as f32) <= 0.05
        };

    let (res_colored_count, res_accent_count) =
        if allow_cast_exclusion || allow_muddy_shadow_exclusion {
            let mut colored = 0usize;
            let mut accents = 0usize;
            for s in &acc.chromatic_samples {
                let c_par = s.a * dom_u_a + s.b * dom_u_b;
                // (a) Paper-cast ray (covers global cast, corner cells, and edge strips along cast axis)
                if allow_cast_exclusion && s.l >= 65.0 {
                    let c_perp = (s.a * dom_u_b - s.b * dom_u_a).abs();
                    if c_par >= 0.0 && c_par <= c_max * 1.15 && c_perp <= 5.0 {
                        continue;
                    }
                }
                // (b) Low-chroma muddy shadow ink (stricter L* < 22 on neutral paper)
                if (allow_cast_exclusion && is_cast_muddy_shadow(s))
                    || (allow_muddy_shadow_exclusion && is_neutral_muddy_shadow(s))
                {
                    continue;
                }
                colored += 1;
                let c_res = if allow_cast_exclusion && s.l >= 65.0 && c_par > 0.0 {
                    let sub = c_par.min(c_max);
                    (s.a - sub * dom_u_a).hypot(s.b - sub * dom_u_b)
                } else {
                    s.a.hypot(s.b)
                };
                if c_res >= 15.0 {
                    accents += 1;
                }
            }
            (colored, accents)
        } else {
            (acc.raw_colored_count, acc.raw_accent_count)
        };

    let r_res = res_colored_count as f32 / acc.sampled_pixels as f32;
    let r_accent = res_accent_count as f32 / acc.sampled_pixels as f32;

    // Never allow neutral-paper muddy-shadow exclusion to downgrade a page to Monochrome
    // (which would trigger post-upscale Luma8 grayscale folding even when was_leveled == false!).
    let classification = if is_already_luma {
        PageClassification::Monochrome
    } else if r_res > 0.15 || is_intentional_sepia {
        PageClassification::Color
    } else if r_res <= 0.015 && r_accent < 0.0005 && !allow_muddy_shadow_exclusion {
        PageClassification::Monochrome
    } else {
        PageClassification::Mixed
    };

    let highlight_sample = if acc.hl_count > 0 {
        DualZoneSample {
            count: acc.hl_count,
            avg_rgb: (
                acc.hl_sum_r as f32 / acc.hl_count as f32,
                acc.hl_sum_g as f32 / acc.hl_count as f32,
                acc.hl_sum_b as f32 / acc.hl_count as f32,
            ),
        }
    } else {
        DualZoneSample::default()
    };

    let shadow_sample = if acc.sh_count > 0 {
        DualZoneSample {
            count: acc.sh_count,
            avg_rgb: (
                acc.sh_sum_r as f32 / acc.sh_count as f32,
                acc.sh_sum_g as f32 / acc.sh_count as f32,
                acc.sh_sum_b as f32 / acc.sh_count as f32,
            ),
        }
    } else {
        DualZoneSample::default()
    };

    let has_true_paper_cast = !is_already_luma
        && !is_intentional_sepia
        && paper_l_peak >= 65.0
        && (has_global_cast || has_localized_margin_cast);

    let paper_cast = PaperCastField {
        active: has_true_paper_cast,
        localized_margin: has_localized_margin_cast,
        global_a: paper_a,
        global_b: paper_b,
        global_c: paper_c,
        dom_a,
        dom_b,
        dom_u_a,
        dom_u_b,
        c_max,
        cell_a,
        cell_b,
    };

    let mut analysis = PageAnalysis {
        classification,
        luminance_histogram: acc.hist_y,
        total_pixels: acc.total_valid_pixels,
        highlight_sample,
        shadow_sample,
        paper_cast,
        has_paper_cast: has_true_paper_cast,
        has_muddy_shadow_cast: false,
        has_high_chroma_accents: r_accent > 0.0,
        defects: Vec::new(),
        black_floor: 0,
        white_ceiling: 255,
        is_shelf_detected: false,
    };

    let min_y = acc.hist_y.iter().position(|&c| c > 0).unwrap_or(0) as u8;
    let max_y = acc.hist_y.iter().rposition(|&c| c > 0).unwrap_or(255) as u8;
    let dynamic_range = max_y.saturating_sub(min_y);

    // Unconditional Full-Color Veto & Blank/Uniform Page Veto (no resolution gate!)
    if dynamic_range < 10 || classification == PageClassification::Color {
        return analysis;
    }

    // 5. Unified Atmospheric Vetoes (High-Key & Low-Key)
    let p1_y = histogram_percentile(&acc.hist_y, hist_total, 1.0);
    let p99_y = histogram_percentile(&acc.hist_y, hist_total, 99.0);
    let low_key_veto = p95_y < 130.0 || paper_l_peak < 65.0;

    let ink_mass_count: u64 = acc.hist_y[3..=85].iter().map(|&c| c as u64).sum();
    let m_ink = ink_mass_count as f64 / hist_total as f64;
    let high_key_veto = m_ink < 0.008;

    let mut defects = Vec::new();

    if !high_key_veto
        && !low_key_veto
        && let Some(d) = detect_elevated_black_floor(&acc.hist_y, classification)
    {
        if let DefectType::ElevatedBlackFloor {
            floor_value,
            is_bimodal,
        } = d
        {
            analysis.black_floor = floor_value;
            analysis.is_shelf_detected = is_bimodal;
        }
        defects.push(d);
    }

    let has_muddy_shadow_cast = !is_already_luma
        && !is_intentional_sepia
        && paper_l_peak >= 65.0
        && acc.sh_lab_count >= min_sh_samples
        && shadow_b >= 2.0
        && shadow_a.abs() <= 4.5
        && shadow_c <= 12.0
        && (has_true_paper_cast || analysis.is_shelf_detected || analysis.black_floor >= 8);
    analysis.has_muddy_shadow_cast = has_muddy_shadow_cast;

    // 6. Already-Optimal Dynamic Range Veto
    let eff_white = find_effective_white(&acc.hist_y, p99_y);
    let gamma_is_identity = (DEFAULT_GAMMA - 1.0).abs() < 1e-3;
    if p1_y <= 1.0
        && p99_y >= 254.0
        && eff_white >= 254.0
        && !has_true_paper_cast
        && !has_muddy_shadow_cast
        && !analysis.is_shelf_detected
        && gamma_is_identity
    {
        analysis.black_floor = 0;
        analysis.defects.clear();
        return analysis;
    }

    if !low_key_veto {
        if has_true_paper_cast || has_muddy_shadow_cast {
            analysis.has_paper_cast = true;
            defects.push(DefectType::PaperCast {
                cast_a: dom_a,
                cast_b: dom_b,
                cast_magnitude: c_max.max(shadow_c),
                delta_grid,
                is_localized_margin: has_localized_margin_cast,
            });
        }

        if !high_key_veto
            && let Some(d) =
                detect_dull_highlights_with_min_white(&acc.hist_y, classification, min_white_threshold)
        {
            if let DefectType::DullHighlights { white_point } = d {
                analysis.white_ceiling = white_point;
            }
            defects.push(d);
        }
    }

    analysis.defects = defects;
    analysis
}

fn populate_luma_paper_peak(
    hist_y: &[u32; 256],
    scale_weight: u32,
    paper_l_hist: &mut [u32; 58],
    pure_white_paper_count: &mut u32,
    max_low_key_l: &mut f32,
) {
    let divisor = scale_weight.max(1);
    for (y, &cnt) in hist_y.iter().enumerate() {
        let unscaled = cnt / divisor;
        if unscaled == 0 {
            continue;
        }
        let (l_star, _, _) = rgb_to_lab_fast(y as u8, y as u8, y as u8);
        if l_star < 70.0 && l_star > *max_low_key_l {
            *max_low_key_l = l_star;
        }
        if (70.0..99.0).contains(&l_star) {
            let bin = (((l_star - 70.0) * 2.0).floor() as usize).min(57);
            paper_l_hist[bin] = paper_l_hist[bin].saturating_add(unscaled);
        } else if l_star >= 99.0 {
            *pure_white_paper_count = pure_white_paper_count.saturating_add(unscaled);
        }
    }
}

// ============================================================================
// Unified Defect Detectors (Shadow Floor & Highlight Ceiling)
// ============================================================================

/// Detects a compact bimodal washed-out black ink shelf in `[8..=35]` (e.g. lifted scan/rip
/// black floors at `Y = 15..28`) specifically when a digital black spike (`0.0005 < m_zero < 0.035`)
/// has anti-aliased text fringes (`Y = 3..10`) that pull `shadow_p2` below the actual ink peak.
fn detect_compact_ink_shelf(
    hist: &[u32; 256],
    h_smooth: &[f32; 256],
    hist_total: u64,
    hi: f32,
) -> Option<u8> {
    // Scan from low to high for the FIRST validated compact ink peak in [8..=35] so an
    // earlier washed ink floor (e.g. Y=12) is never skipped in favor of a taller dark-gray tone (Y=30).
    for peak_idx in 8..=35 {
        let h_peak = h_smooth[peak_idx];
        if h_peak < h_smooth[peak_idx - 1] || h_peak < h_smooth[peak_idx + 1] {
            continue;
        }
        if h_peak < 0.003 * hist_total as f32 {
            continue;
        }
        if let Some(v) =
            validate_spike_separated_peak(hist, h_smooth, hist_total, peak_idx, hi, 0.25, true)
        {
            return Some(v.left_foot as u8);
        }
    }

    None
}

/// Unified shadow detector over `[2..=85]` with strict `m_zero < 0.035` cap, low-key veto (`p95 >= 130`),
/// compact ink shelf detection in `[8..=35]` (for `0.0005 < m_zero < 0.035` AA text tails), no `p1 < 3`
/// dead zone, shadow-band `shadow_band_p2` primary candidate, unified pre/post-peak valley + FWHM-skirt
/// validation, and ratio-normalized counts.
pub fn detect_elevated_black_floor(
    hist: &[u32; 256],
    classification: PageClassification,
) -> Option<DefectType> {
    if classification == PageClassification::Color {
        return None;
    }
    let hist_total: u64 = hist.iter().map(|&c| c as u64).sum();
    if hist_total == 0 {
        return None;
    }

    // Low-Key Scene Veto: do not crush night splashes / dark low-key scenes
    let p95 = histogram_percentile(hist, hist_total, 95.0);
    if p95 < 130.0 {
        return None;
    }

    // High-Key Veto aligned to [3..=85]
    let ink_mass: u64 = hist[3..=85].iter().map(|&c| c as u64).sum();
    if (ink_mass as f64 / hist_total as f64) < 0.008 {
        return None;
    }

    let zero_count = hist[0] as u64 + hist[1] as u64 + hist[2] as u64;
    let m_zero = zero_count as f64 / hist_total as f64;
    // Strict digital black cap: >= 3.5% authentic pure black ink vetoes black floor elevation
    if m_zero >= 0.035 {
        return None;
    }

    let p1 = histogram_percentile(hist, hist_total, 1.0);
    let p99 = histogram_percentile(hist, hist_total, 99.0);
    let hi = find_effective_white(hist, p99);

    let (validated_lo, is_bimodal) = if m_zero > 0.0005 {
        // 1. Compact bimodal ink shelf in [8..=35] (handles real-world lifted blacks alongside AA text)
        let h_smooth = smooth_histogram_binomial_5(hist);
        if let Some(shelf_floor) = detect_compact_ink_shelf(hist, &h_smooth, hist_total, hi) {
            return Some(DefectType::ElevatedBlackFloor {
                floor_value: shelf_floor,
                is_bimodal: true,
            });
        }

        // 2. Digital black spike present (0.05% < m_zero < 3.5%): no p1 < 3 branch split!
        let (nz_total, nz_p3) = nonzero_histogram_percentile(hist, 3.0);
        if (nz_total as f64 / hist_total as f64) < 0.02 || nz_p3 <= 8.0 {
            return None;
        }
        // Use shadow-band CDF as primary candidate so sparse ink (1-2%) never overshoots into midtones
        let shadow_p2 = shadow_band_p2(hist, hist_total)?;
        let cand_lo = nz_p3.min(shadow_p2);
        if cand_lo < 8.0 {
            return None;
        }
        let val = validate_black_floor(hist, cand_lo, hi);
        (val, val >= 8.0)
    } else {
        // No digital black spike (m_zero <= 0.05%)
        let clip_bp = histogram_percentile(hist, hist_total, BLACK_CLIP_PERCENT);
        let cand_lo = if p1 >= 2.0 { p1.max(clip_bp) } else { clip_bp };
        if cand_lo < 2.0 {
            return None;
        }
        let val = validate_black_floor(hist, cand_lo, hi);
        (val, val > 28.0)
    };

    let floor_value = validated_lo.round() as u8;
    if floor_value <= 1 || floor_value > 85 {
        return None;
    }

    Some(DefectType::ElevatedBlackFloor {
        floor_value,
        is_bimodal,
    })
}

/// Detects dull paper highlights using `hist_total`, unified `p95 >= 130` low-key guard,
/// validated bimodal `find_effective_white`, and strict `MIN_WHITE_THRESHOLD` enforcement.
pub fn detect_dull_highlights(
    hist: &[u32; 256],
    classification: PageClassification,
) -> Option<DefectType> {
    detect_dull_highlights_with_min_white(hist, classification, MIN_WHITE_THRESHOLD)
}

fn detect_dull_highlights_with_min_white(
    hist: &[u32; 256],
    classification: PageClassification,
    min_white_threshold: u8,
) -> Option<DefectType> {
    if classification == PageClassification::Color {
        return None;
    }
    let hist_total: u64 = hist.iter().map(|&c| c as u64).sum();
    if hist_total == 0 {
        return None;
    }

    // Unified Low-Key Scene Veto: p95(Y) < 130.0
    let p95 = histogram_percentile(hist, hist_total, 95.0);
    if p95 < 130.0 {
        return None;
    }

    let p99 = histogram_percentile(hist, hist_total, 99.0);
    let eff_white = find_effective_white(hist, p99);
    let raw_wp = histogram_reverse_percentile(hist, hist_total, WHITE_CLIP_PERCENT);

    let white_point = if eff_white < p99 - 1.0 {
        (p99 - (p99 - eff_white) * 0.85).round() as u8
    } else {
        (raw_wp.min(p99)).round() as u8
    };

    if white_point >= 254 || white_point < min_white_threshold {
        return None;
    }

    Some(DefectType::DullHighlights { white_point })
}

// ============================================================================
// Correction Plan Generation
// ============================================================================

pub fn generate_correction_plan(
    analysis: &PageAnalysis,
    config: &AutoLevelsConfig,
) -> CorrectionPlan {
    let mut plan = CorrectionPlan {
        gamma: DEFAULT_GAMMA.clamp(0.2, 5.0),
        cast_field: analysis.paper_cast.clone(),
        has_muddy_shadow_cast: analysis.has_muddy_shadow_cast,
        has_high_chroma_accents: analysis.has_high_chroma_accents,
        ..CorrectionPlan::default()
    };

    if analysis.classification == PageClassification::Color {
        plan.skip_correction = true;
        return plan;
    }

    for defect in &analysis.defects {
        match *defect {
            DefectType::ElevatedBlackFloor { floor_value, .. } => {
                let clamped = floor_value.min(MAX_BLACK_SHIFT);
                plan.black_point = plan.black_point.max(clamped);
            }
            DefectType::DullHighlights { white_point } => {
                plan.white_point = plan.white_point.min(white_point);
            }
            DefectType::PaperCast { .. } => {
                if config.correct_paper_cast {
                    plan.cast_strength = 1.0;
                }
            }
        }
    }

    if plan.black_point <= 1 {
        plan.black_point = 0;
    }
    if plan.black_point > MAX_BLACK_SHIFT {
        plan.black_point = MAX_BLACK_SHIFT;
    }
    if plan.white_point >= 254 || plan.white_point < MIN_WHITE_THRESHOLD {
        plan.white_point = 255;
    }
    if (plan.white_point as i32 - plan.black_point as i32) < 15 {
        plan.black_point = 0;
        plan.white_point = 255;
    }

    // For verified Monochrome RGB pages with correct_paper_cast enabled, neutralize minor 1-bit
    // chroma noise during leveling even when no macroscopic paper cast defect was raised.
    let tonal_active =
        plan.black_point > 0 || plan.white_point < 255 || (plan.gamma - 1.0).abs() >= 1e-3;
    if analysis.classification == PageClassification::Monochrome
        && config.correct_paper_cast
        && tonal_active
    {
        plan.cast_strength = 1.0;
    }

    plan.skip_correction = !tonal_active && plan.cast_strength <= 0.0;
    plan
}

// ============================================================================
// Arbitrary-Hue Localized Directional Cast Weight & Format-Agnostic Core
// ============================================================================

/// Pre-computed coordinate mapping from a Quality Guard tile sample back to full-resolution pixels
/// so spatially-varying 4x4 cell cast fields and margin rings evaluate identically on tiles and full frames.
struct TileCoords {
    full_w: usize,
    full_h: usize,
    src_x: Vec<usize>,
    src_y: Vec<usize>,
    tile_size: Option<usize>,
}

#[derive(Clone, Copy)]
struct ColInterp {
    c0: usize,
    c1: usize,
    fx: f32,
    dx_edge: f32,
}

#[inline(always)]
fn eval_local_cast_envelope(
    field: &PaperCastField,
    r0: usize,
    r1: usize,
    fy: f32,
    dy_edge: f32,
    col: ColInterp,
) -> (f32, f32) {
    if !field.localized_margin {
        return (field.global_a, field.global_b);
    }

    let idx00 = r0 * 4 + col.c0;
    let idx01 = r0 * 4 + col.c1;
    let idx10 = r1 * 4 + col.c0;
    let idx11 = r1 * 4 + col.c1;

    let a_top = field.cell_a[idx00] + (field.cell_a[idx01] - field.cell_a[idx00]) * col.fx;
    let a_bot = field.cell_a[idx10] + (field.cell_a[idx11] - field.cell_a[idx10]) * col.fx;
    let a_bilin = a_top + (a_bot - a_top) * fy;

    let b_top = field.cell_b[idx00] + (field.cell_b[idx01] - field.cell_b[idx00]) * col.fx;
    let b_bot = field.cell_b[idx10] + (field.cell_b[idx11] - field.cell_b[idx10]) * col.fx;
    let b_bilin = b_top + (b_bot - b_top) * fy;

    // Smoothstep over the outer ~12% margin ring (full within 10%, C1 taper 10%..14%) toward the
    // dominant border cell cast so both corners and side-edge strips are cleaned uniformly while
    // keeping clean interior cells strictly bounded to their local paper envelope.
    let d_edge = dy_edge.min(col.dx_edge);
    let ring_w = 1.0 - smoothstep_inv(0.10, 25.0, d_edge);
    let a_env = a_bilin + (field.dom_a - a_bilin) * ring_w;
    let b_env = b_bilin + (field.dom_b - b_bilin) * ring_w;
    (a_env, b_env)
}

#[inline(always)]
fn compute_cast_weight_with_unit(
    l_star: f32,
    a_star: f32,
    b_star: f32,
    c_env: f32,
    ua: f32,
    ub: f32,
    plan: &CorrectionPlan,
) -> f32 {
    if plan.cast_strength <= 0.0 {
        return 0.0;
    }

    let mut w_paper = 0.0f32;
    if plan.cast_field.active && l_star >= 55.0 && c_env > 1e-4 {
        let c_par = a_star * ua + b_star * ub;
        if c_par > 0.0 {
            let c_perp = (a_star * ub - b_star * ua).abs();
            let w_l = smoothstep_inv(55.0, 1.0 / 15.0, l_star);
            let w_par_lo = smoothstep_inv(0.0, 1.0, c_par);
            let hi_cap = c_env * 1.15;
            let w_par_hi = 1.0 - smoothstep_inv(hi_cap, 1.0 / 6.0, c_par);
            let w_perp = 1.0 - smoothstep_inv(3.5, 1.0 / 3.0, c_perp);
            w_paper = w_l * w_par_lo * w_par_hi * w_perp * plan.cast_strength;
        }
    }

    let mut w_shadow = 0.0f32;
    let sh_l_max = if plan.cast_field.active { 58.0 } else { 38.0 };
    if plan.has_muddy_shadow_cast && l_star < sh_l_max && b_star > 0.0 {
        let c_sq = a_star * a_star + b_star * b_star;
        if c_sq < 132.25 && a_star.abs() < 5.5 {
            let w_sh_l = if plan.cast_field.active {
                // Smoothly bridges into w_paper [55..70] with zero L* in [50, 55) dead zone
                1.0 - smoothstep_inv(35.0, 1.0 / 23.0, l_star)
            } else {
                // On neutral/white-paper Mixed pages, align full weight with L* < 22 and C1-taper to 38
                1.0 - smoothstep_inv(22.0, 1.0 / 16.0, l_star)
            };
            let w_sh_c = if c_sq <= 72.25 {
                1.0
            } else {
                1.0 - smoothstep_inv(8.5, 1.0 / 3.0, c_sq.sqrt())
            };
            let w_sh_a = 1.0 - smoothstep_inv(3.5, 0.5, a_star.abs());
            let w_sh_b = smoothstep_inv(0.0, 1.0 / 1.5, b_star);
            w_shadow = w_sh_l * w_sh_c * w_sh_a * w_sh_b * plan.cast_strength;
        }
    }

    w_paper.max(w_shadow).clamp(0.0, 1.0)
}

#[inline(always)]
fn compute_cast_weight_with_env(
    l_star: f32,
    a_star: f32,
    b_star: f32,
    a_env: f32,
    b_env: f32,
    plan: &CorrectionPlan,
) -> f32 {
    let c_env = (a_env * a_env + b_env * b_env).sqrt();
    let (ua, ub) = if c_env > 1e-4 {
        (a_env / c_env, b_env / c_env)
    } else {
        (0.0, 0.0)
    };
    compute_cast_weight_with_unit(l_star, a_star, b_star, c_env, ua, ub, plan)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn compute_cast_weight_at(
    orig_x: usize,
    orig_y: usize,
    full_w: usize,
    full_h: usize,
    l_star: f32,
    a_star: f32,
    b_star: f32,
    plan: &CorrectionPlan,
) -> f32 {
    if plan.cast_strength <= 0.0 {
        return 0.0;
    }
    let fw = full_w.max(1) as f32;
    let fh = full_h.max(1) as f32;
    let u = (((orig_x as f32 + 0.5) / fw) * 4.0 - 0.5).clamp(0.0, 3.0);
    let v = (((orig_y as f32 + 0.5) / fh) * 4.0 - 0.5).clamp(0.0, 3.0);
    let c0 = (u.floor() as usize).min(3);
    let c1 = (c0 + 1).min(3);
    let r0 = (v.floor() as usize).min(3);
    let r1 = (r0 + 1).min(3);
    let col = ColInterp {
        c0,
        c1,
        fx: u - c0 as f32,
        dx_edge: (orig_x.min(full_w.saturating_sub(1 + orig_x)) as f32) / fw,
    };
    let dy_edge = (orig_y.min(full_h.saturating_sub(1 + orig_y)) as f32) / fh;
    let (a_env, b_env) = eval_local_cast_envelope(&plan.cast_field, r0, r1, v - r0 as f32, dy_edge, col);
    compute_cast_weight_with_env(l_star, a_star, b_star, a_env, b_env, plan)
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn eval_accent_px<const CHANNELS: usize>(
    px: &[u8; CHANNELS],
    prev_key0: &mut u32,
    prev_acc0: &mut bool,
    prev_key1: &mut u32,
    prev_acc1: &mut bool,
    cast_active: bool,
    dom_u_a: f32,
    dom_u_b: f32,
    cast_cap: f32,
    tables: &FastLabTables,
) -> bool {
    if CHANNELS == 4 && px[3] == 0 {
        return false;
    }
    let (r, g, b) = (px[0], px[1], px[2]);
    // Exact sRGB bound across all 256^3 colors: C* >= 20.0 requires rgb_diff >= 24 (use < 22 with margin; subsumes r==g==b)
    let rgb_diff = r.max(g).max(b) - r.min(g).min(b);
    if rgb_diff < 22 {
        return false;
    }
    let key = (r as u32) | ((g as u32) << 8) | ((b as u32) << 16);
    if key == *prev_key0 {
        return *prev_acc0;
    }
    if key == *prev_key1 {
        return *prev_acc1;
    }
    let (l_star, a_star, b_star) = rgb_to_lab_with_tables(tables, r, g, b);
    let c_sq = a_star * a_star + b_star * b_star;
    let hit = if c_sq < 400.0 {
        false
    } else if cast_active && l_star >= 65.0 {
        let c_par = a_star * dom_u_a + b_star * dom_u_b;
        let c_perp = (a_star * dom_u_b - b_star * dom_u_a).abs();
        !(c_par >= 0.0 && c_par <= cast_cap && c_perp <= 5.0)
    } else {
        true
    };
    *prev_key1 = *prev_key0;
    *prev_acc1 = *prev_acc0;
    *prev_key0 = key;
    *prev_acc0 = hit;
    hit
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn scan_accent_16<const CHANNELS: usize>(
    b16: &[[u8; CHANNELS]; 16],
    prev_key0: &mut u32,
    prev_acc0: &mut bool,
    prev_key1: &mut u32,
    prev_acc1: &mut bool,
    cast_active: bool,
    dom_u_a: f32,
    dom_u_b: f32,
    cast_cap: f32,
    tables: &FastLabTables,
) -> u64 {
    let mut any_cand = 0u8;
    for px in b16 {
        let d = px[0].max(px[1]).max(px[2]) - px[0].min(px[1]).min(px[2]);
        any_cand |= (d >= 22) as u8;
    }
    if any_cand == 0 {
        return 0;
    }
    let p0 = b16[0];
    let mut all_eq = true;
    for px in &b16[1..] {
        all_eq &= *px == p0;
    }
    if all_eq {
        let hit = eval_accent_px(
            &p0,
            prev_key0,
            prev_acc0,
            prev_key1,
            prev_acc1,
            cast_active,
            dom_u_a,
            dom_u_b,
            cast_cap,
            tables,
        );
        return if hit { 0xFFFF } else { 0 };
    }
    let mut sub = 0u64;
    for (k, px) in b16.iter().enumerate() {
        if eval_accent_px(
            px,
            prev_key0,
            prev_acc0,
            prev_key1,
            prev_acc1,
            cast_active,
            dom_u_a,
            dom_u_b,
            cast_cap,
            tables,
        ) {
            sub |= 1u64 << k;
        }
    }
    sub
}

#[inline(always)]
fn scan_accent_row<const CHANNELS: usize>(
    row_raw: &[u8],
    mask_row: &mut [u64],
    cast_active: bool,
    dom_u_a: f32,
    dom_u_b: f32,
    cast_cap: f32,
    tables: &FastLabTables,
) -> u8 {
    let chunks = row_raw.as_chunks::<CHANNELS>().0;
    let (blocks16, tail) = chunks.as_chunks::<16>();
    let mut prev_key0 = u32::MAX;
    let mut prev_acc0 = false;
    let mut prev_key1 = u32::MAX;
    let mut prev_acc1 = false;
    let mut any = 0u8;
    let mut wi = 0usize;
    let (word_chunks, rem_blocks) = blocks16.as_chunks::<4>();

    for word_blocks in word_chunks {
        let m0 = scan_accent_16(
            &word_blocks[0],
            &mut prev_key0,
            &mut prev_acc0,
            &mut prev_key1,
            &mut prev_acc1,
            cast_active,
            dom_u_a,
            dom_u_b,
            cast_cap,
            tables,
        );
        let m1 = scan_accent_16(
            &word_blocks[1],
            &mut prev_key0,
            &mut prev_acc0,
            &mut prev_key1,
            &mut prev_acc1,
            cast_active,
            dom_u_a,
            dom_u_b,
            cast_cap,
            tables,
        );
        let m2 = scan_accent_16(
            &word_blocks[2],
            &mut prev_key0,
            &mut prev_acc0,
            &mut prev_key1,
            &mut prev_acc1,
            cast_active,
            dom_u_a,
            dom_u_b,
            cast_cap,
            tables,
        );
        let m3 = scan_accent_16(
            &word_blocks[3],
            &mut prev_key0,
            &mut prev_acc0,
            &mut prev_key1,
            &mut prev_acc1,
            cast_active,
            dom_u_a,
            dom_u_b,
            cast_cap,
            tables,
        );
        let word = m0 | (m1 << 16) | (m2 << 32) | (m3 << 48);
        mask_row[wi] = word;
        if word != 0 {
            any = 1;
        }
        wi += 1;
    }

    if !rem_blocks.is_empty() || !tail.is_empty() {
        let mut word = 0u64;
        let mut bit_off = 0usize;
        for b16 in rem_blocks {
            word |= scan_accent_16(
                b16,
                &mut prev_key0,
                &mut prev_acc0,
                &mut prev_key1,
                &mut prev_acc1,
                cast_active,
                dom_u_a,
                dom_u_b,
                cast_cap,
                tables,
            ) << bit_off;
            bit_off += 16;
        }
        for (k, px) in tail.iter().enumerate() {
            if eval_accent_px(
                px,
                &mut prev_key0,
                &mut prev_acc0,
                &mut prev_key1,
                &mut prev_acc1,
                cast_active,
                dom_u_a,
                dom_u_b,
                cast_cap,
                tables,
            ) {
                word |= 1u64 << (bit_off + k);
            }
        }
        mask_row[wi] = word;
        if word != 0 {
            any = 1;
        }
    }
    any
}

#[inline(always)]
fn dilate_3row_bits(
    r0: &[u64],
    r1: &[u64],
    r2: &[u64],
    out_row: &mut [u64],
    last_word_mask: u64,
) {
    let row_words = out_row.len();
    let mut v_prev = 0u64;
    let mut v_curr = r0[0] | r1[0] | r2[0];
    for wi in 0..row_words {
        let v_next = if wi + 1 < row_words {
            r0[wi + 1] | r1[wi + 1] | r2[wi + 1]
        } else {
            0
        };
        let mut d = v_curr | (v_curr << 1) | (v_curr >> 1) | (v_prev >> 63) | (v_next << 63);
        if wi + 1 == row_words {
            d &= last_word_mask;
        }
        out_row[wi] = d;
        v_prev = v_curr;
        v_curr = v_next;
    }
}

/// Applies target luminance `y_new` with chroma offsets `(dr, dg, db)` while preserving 100%
/// chromaticity and hue inside the sRGB cube: clamps `y_eff` into the exact valid headroom
/// `[-min(dr,dg,db), 255 - max(dr,dg,db)]` so bright pastels (cream skin, pastel sky) and muted
/// dark tones are never desaturated or flat-clipped against the sRGB gamut apex.
#[inline(always)]
fn apply_chroma_preserving_luma_shift(y_new: i32, dr: i32, dg: i32, db: i32) -> (u8, u8, u8) {
    let min_d = dr.min(dg).min(db);
    let max_d = dr.max(dg).max(db);
    let lo_y = (-min_d).max(0);
    let hi_y = (255 - max_d).min(255);
    let y_eff = if lo_y <= hi_y {
        y_new.clamp(lo_y, hi_y)
    } else {
        y_new.clamp(0, 255)
    };
    (
        (y_eff + dr).clamp(0, 255) as u8,
        (y_eff + dg).clamp(0, 255) as u8,
        (y_eff + db).clamp(0, 255) as u8,
    )
}

#[derive(Clone, Copy)]
struct MixedSpanCtx<'a> {
    lut_y: &'a [u8; 256],
    plan: &'a CorrectionPlan,
    tables: &'a FastLabTables,
    do_cast: bool,
    has_muddy_shadow: bool,
    c_env: f32,
    ua: f32,
    ub: f32,
}

#[derive(Clone, Copy)]
struct EnvRun {
    x0: usize,
    x1: usize,
    c_env: f32,
    ua: f32,
    ub: f32,
}

struct MixedPixelCache {
    /// Highlight/paper cache (`y_old >= 120`): caches spatially-invariant `(rgb, y_old, y_new, L*, a*, b*)`
    /// so envelope transitions refresh `hi_out` in 10 float ops without invalidating `hi_key` or re-running CIELAB!
    hi_key: u32,
    hi_rgb: (u8, u8, u8),
    hi_y_old: i32,
    hi_y_new: i32,
    hi_lab: (f32, f32, f32),
    hi_out: [u8; 4],
    /// Shadow/dark cache (`y_old < 120`, spatially invariant — never invalidated!).
    lo_key: u32,
    lo_out: (u8, u8, u8),
}

impl MixedPixelCache {
    #[inline(always)]
    fn new() -> Self {
        Self {
            hi_key: u32::MAX,
            hi_rgb: (0, 0, 0),
            hi_y_old: 0,
            hi_y_new: 0,
            hi_lab: (0.0, 0.0, 0.0),
            hi_out: [0, 0, 0, 255],
            lo_key: u32::MAX,
            lo_out: (0, 0, 0),
        }
    }

    #[inline(always)]
    fn refresh_hi_for_env(&mut self, ctx: &MixedSpanCtx<'_>) {
        if self.hi_key == u32::MAX {
            return;
        }
        let (r, g, b) = self.hi_rgb;
        let y_old = self.hi_y_old;
        let y_new = self.hi_y_new;
        let retain_q8 = if ctx.do_cast && (ctx.c_env > 1e-4 || ctx.has_muddy_shadow) {
            let (l_star, a_star, b_star) = self.hi_lab;
            let w_cast = compute_cast_weight_with_unit(
                l_star, a_star, b_star, ctx.c_env, ctx.ua, ctx.ub, ctx.plan,
            );
            ((1.0 - w_cast) * 256.0).round() as i32
        } else {
            256
        };
        let out = if retain_q8 <= 0 {
            let o = y_new as u8;
            (o, o, o)
        } else {
            let dr = ((r as i32 - y_old) * retain_q8 + 128) >> 8;
            let dg = ((g as i32 - y_old) * retain_q8 + 128) >> 8;
            let db = ((b as i32 - y_old) * retain_q8 + 128) >> 8;
            apply_chroma_preserving_luma_shift(y_new, dr, dg, db)
        };
        self.hi_out = [out.0, out.1, out.2, 255];
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn apply_mixed_pixel_slow<const CHANNELS: usize>(
    chunk: &mut [u8; CHANNELS],
    r: u8,
    g: u8,
    b: u8,
    key: u32,
    ctx: &MixedSpanCtx<'_>,
    cache: &mut MixedPixelCache,
    hi_key: &mut u32,
    hi_out: &mut [u8; 4],
    lo_key: &mut u32,
    lo_out: &mut (u8, u8, u8),
) {
    if r == g && g == b {
        let out = ctx.lut_y[r as usize];
        if r < 120 {
            *lo_key = key;
            *lo_out = (out, out, out);
        }
        chunk[0] = out;
        chunk[1] = out;
        chunk[2] = out;
        return;
    }

    let y_old = rec709_y(r, g, b) as i32;
    let y_new = ctx.lut_y[y_old as usize] as i32;
    let is_hi = y_old >= 120;

    let mut lab = (0.0f32, 0.0f32, 0.0f32);
    let retain_q8 = if ctx.do_cast
        && (if is_hi {
            ctx.c_env > 1e-4 || ctx.has_muddy_shadow
        } else {
            ctx.has_muddy_shadow
        })
    {
        lab = rgb_to_lab_with_tables(ctx.tables, r, g, b);
        let w_cast = compute_cast_weight_with_unit(
            lab.0, lab.1, lab.2, ctx.c_env, ctx.ua, ctx.ub, ctx.plan,
        );
        ((1.0 - w_cast) * 256.0).round() as i32
    } else {
        if is_hi && ctx.do_cast {
            lab = rgb_to_lab_with_tables(ctx.tables, r, g, b);
        }
        256
    };

    let out = if retain_q8 <= 0 {
        let o = y_new as u8;
        (o, o, o)
    } else {
        let dr = ((r as i32 - y_old) * retain_q8 + 128) >> 8;
        let dg = ((g as i32 - y_old) * retain_q8 + 128) >> 8;
        let db = ((b as i32 - y_old) * retain_q8 + 128) >> 8;
        apply_chroma_preserving_luma_shift(y_new, dr, dg, db)
    };

    if is_hi {
        *hi_key = key;
        *hi_out = [out.0, out.1, out.2, 255];
        cache.hi_rgb = (r, g, b);
        cache.hi_y_old = y_old;
        cache.hi_y_new = y_new;
        cache.hi_lab = lab;
    } else {
        *lo_key = key;
        *lo_out = out;
    }
    chunk[0] = out.0;
    chunk[1] = out.1;
    chunk[2] = out.2;
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn apply_mixed_one<const CHANNELS: usize>(
    chunk: &mut [u8; CHANNELS],
    ctx: &MixedSpanCtx<'_>,
    cache: &mut MixedPixelCache,
    hi_key: &mut u32,
    hi_out: &mut [u8; 4],
    lo_key: &mut u32,
    lo_out: &mut (u8, u8, u8),
    hi_u32: &mut u32,
) {
    if CHANNELS == 4 {
        let raw32 = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        if *hi_key != u32::MAX && raw32 == *hi_u32 {
            chunk[0] = hi_out[0];
            chunk[1] = hi_out[1];
            chunk[2] = hi_out[2];
            chunk[3] = 255;
            return;
        }
        if (raw32 >> 24) == 0 {
            return;
        }
        let key = raw32 & 0x00FF_FFFF;
        if key == *hi_key {
            chunk[0] = hi_out[0];
            chunk[1] = hi_out[1];
            chunk[2] = hi_out[2];
            return;
        }
        if key == *lo_key {
            chunk[0] = lo_out.0;
            chunk[1] = lo_out.1;
            chunk[2] = lo_out.2;
            return;
        }
        apply_mixed_pixel_slow(
            chunk,
            chunk[0],
            chunk[1],
            chunk[2],
            key,
            ctx,
            cache,
            hi_key,
            hi_out,
            lo_key,
            lo_out,
        );
        *hi_u32 = if *hi_key == u32::MAX {
            0u32
        } else {
            *hi_key | 0xFF00_0000
        };
    } else {
        let (r, g, b) = (chunk[0], chunk[1], chunk[2]);
        let key = (r as u32) | ((g as u32) << 8) | ((b as u32) << 16);
        if key == *hi_key {
            chunk[0] = hi_out[0];
            chunk[1] = hi_out[1];
            chunk[2] = hi_out[2];
            return;
        }
        if key == *lo_key {
            chunk[0] = lo_out.0;
            chunk[1] = lo_out.1;
            chunk[2] = lo_out.2;
            return;
        }
        apply_mixed_pixel_slow(
            chunk, r, g, b, key, ctx, cache, hi_key, hi_out, lo_key, lo_out,
        );
    }
}

#[inline(always)]
fn apply_mixed_span_unmasked<const CHANNELS: usize>(
    chunks: &mut [[u8; CHANNELS]],
    ctx: &MixedSpanCtx<'_>,
    cache: &mut MixedPixelCache,
) {
    let mut hi_key = cache.hi_key;
    let mut hi_out = cache.hi_out;
    let mut lo_key = cache.lo_key;
    let mut lo_out = cache.lo_out;
    let mut hi_u32 = if hi_key == u32::MAX {
        0u32
    } else {
        hi_key | 0xFF00_0000
    };

    let (groups, tail) = chunks.as_chunks_mut::<4>();
    for group in groups {
        if CHANNELS == 4 && hi_key != u32::MAX {
            let p0 = u32::from_le_bytes([group[0][0], group[0][1], group[0][2], group[0][3]]);
            let p1 = u32::from_le_bytes([group[1][0], group[1][1], group[1][2], group[1][3]]);
            let p2 = u32::from_le_bytes([group[2][0], group[2][1], group[2][2], group[2][3]]);
            let p3 = u32::from_le_bytes([group[3][0], group[3][1], group[3][2], group[3][3]]);
            if ((p0 ^ hi_u32) | (p1 ^ hi_u32) | (p2 ^ hi_u32) | (p3 ^ hi_u32)) == 0 {
                for px in group.iter_mut() {
                    px[0] = hi_out[0];
                    px[1] = hi_out[1];
                    px[2] = hi_out[2];
                    px[3] = 255;
                }
                continue;
            }
        } else if hi_key != u32::MAX {
            let hi_r = hi_key as u8;
            let hi_g = (hi_key >> 8) as u8;
            let hi_b = (hi_key >> 16) as u8;
            let m0 = (group[0][0] == hi_r) & (group[0][1] == hi_g) & (group[0][2] == hi_b);
            let m1 = (group[1][0] == hi_r) & (group[1][1] == hi_g) & (group[1][2] == hi_b);
            let m2 = (group[2][0] == hi_r) & (group[2][1] == hi_g) & (group[2][2] == hi_b);
            let m3 = (group[3][0] == hi_r) & (group[3][1] == hi_g) & (group[3][2] == hi_b);
            if m0 & m1 & m2 & m3 {
                for px in group.iter_mut() {
                    px[0] = hi_out[0];
                    px[1] = hi_out[1];
                    px[2] = hi_out[2];
                }
                continue;
            }
        }
        for px in group.iter_mut() {
            apply_mixed_one(
                px,
                ctx,
                cache,
                &mut hi_key,
                &mut hi_out,
                &mut lo_key,
                &mut lo_out,
                &mut hi_u32,
            );
        }
    }
    for px in tail {
        apply_mixed_one(
            px,
            ctx,
            cache,
            &mut hi_key,
            &mut hi_out,
            &mut lo_key,
            &mut lo_out,
            &mut hi_u32,
        );
    }

    cache.hi_key = hi_key;
    cache.hi_out = hi_out;
    cache.lo_key = lo_key;
    cache.lo_out = lo_out;
}

#[inline(always)]
fn apply_mixed_span<const CHANNELS: usize, const HAS_MASK: bool>(
    chunks: &mut [[u8; CHANNELS]],
    x_offset: usize,
    mask_row: &[u64],
    ctx: &MixedSpanCtx<'_>,
    cache: &mut MixedPixelCache,
) {
    if !HAS_MASK {
        apply_mixed_span_unmasked(chunks, ctx, cache);
        return;
    }

    let len = chunks.len();
    let mut pos = 0usize;
    while pos < len {
        let x = x_offset + pos;
        let wi = x >> 6;
        let bit_off = x & 63;
        if bit_off != 0 || pos + 64 > len {
            let step = (64 - bit_off).min(len - pos);
            let mut word = (mask_row[wi] >> bit_off)
                & if step == 64 { u64::MAX } else { (1u64 << step) - 1 };
            if word == 0 {
                apply_mixed_span_unmasked(&mut chunks[pos..pos + step], ctx, cache);
            } else if word != (if step == 64 { u64::MAX } else { (1u64 << step) - 1 }) {
                let mut cur = 0usize;
                while cur < step {
                    if (word & 1) == 0 {
                        let z_run = word.trailing_zeros().min((step - cur) as u32) as usize;
                        apply_mixed_span_unmasked(
                            &mut chunks[pos + cur..pos + cur + z_run],
                            ctx,
                            cache,
                        );
                        cur += z_run;
                        word >>= z_run;
                    } else {
                        let o_run = word.trailing_ones().min((step - cur) as u32) as usize;
                        cur += o_run;
                        word >>= o_run;
                    }
                }
            }
            pos += step;
            continue;
        }

        // Aligned 64-pixel words: skip full masked words and batch consecutive zero words!
        let mut w_end = wi;
        let max_words = (len - pos) >> 6;
        if mask_row[w_end] == 0 {
            w_end += 1;
            while (w_end - wi) < max_words && mask_row[w_end] == 0 {
                w_end += 1;
            }
            let px_count = (w_end - wi) << 6;
            apply_mixed_span_unmasked(&mut chunks[pos..pos + px_count], ctx, cache);
            pos += px_count;
        } else if mask_row[w_end] == u64::MAX {
            w_end += 1;
            while (w_end - wi) < max_words && mask_row[w_end] == u64::MAX {
                w_end += 1;
            }
            pos += (w_end - wi) << 6;
        } else {
            let mut word = mask_row[w_end];
            let mut cur = 0usize;
            while cur < 64 {
                if (word & 1) == 0 {
                    let z_run = word.trailing_zeros().min((64 - cur) as u32) as usize;
                    apply_mixed_span_unmasked(&mut chunks[pos + cur..pos + cur + z_run], ctx, cache);
                    cur += z_run;
                    word >>= z_run;
                } else {
                    let o_run = word.trailing_ones().min((64 - cur) as u32) as usize;
                    cur += o_run;
                    word >>= o_run;
                }
            }
            pos += 64;
        }
    }
}

#[inline(always)]
fn eval_span_unit_env(
    field: &PaperCastField,
    r0: usize,
    r1: usize,
    fy: f32,
    dy_edge: f32,
    col: ColInterp,
    dom: (f32, f32, f32),
) -> (f32, f32, f32) {
    if col.dx_edge <= 0.10 {
        dom
    } else {
        let (ae, be) = eval_local_cast_envelope(field, r0, r1, fy, dy_edge, col);
        let ce = (ae * ae + be * be).sqrt();
        if ce > 1e-4 {
            (ce, ae / ce, be / ce)
        } else {
            (0.0, 0.0, 0.0)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_merged_env_runs(
    field: &PaperCastField,
    r0: usize,
    r1: usize,
    fy: f32,
    dy_edge: f32,
    span_cols: &[ColInterp],
    span_w: usize,
    w: usize,
    dom: (f32, f32, f32),
    out: &mut Vec<EnvRun>,
) {
    out.clear();
    for (si, &col) in span_cols.iter().enumerate() {
        let (nc, nua, nub) = eval_span_unit_env(field, r0, r1, fy, dy_edge, col, dom);
        let x0 = si * span_w;
        let x1 = (x0 + span_w).min(w);
        if let Some(last) = out.last_mut()
            && (nc - last.c_env).abs() <= 1.25
            && (nua - last.ua).abs() <= 0.06
            && (nub - last.ub).abs() <= 0.06
        {
            last.x1 = x1;
        } else {
            out.push(EnvRun {
                x0,
                x1,
                c_env: nc,
                ua: nua,
                ub: nub,
            });
        }
    }
}

fn apply_rgb_core<const CHANNELS: usize>(
    raw: &mut [u8],
    w: usize,
    h: usize,
    lut_y: &[u8; 256],
    plan: &CorrectionPlan,
    classification: PageClassification,
    coords: Option<&TileCoords>,
) {
    if w == 0 || h == 0 {
        return;
    }

    const STRIP_ROWS: usize = 192;
    let row_bytes = w * CHANNELS;
    let use_parallel = w * h >= 64_000;

    // Fast 1-pass Monochrome path (zero per-pixel LAB math, sequential on tile samples / coarse Rayon strips on full frames)
    if classification == PageClassification::Monochrome {
        if plan.cast_strength >= 0.999 {
            let process_strip = |strip: &mut [u8]| {
                for chunk in strip.as_chunks_mut::<CHANNELS>().0 {
                    if CHANNELS == 4 && chunk[CHANNELS - 1] == 0 {
                        continue;
                    }
                    let y = rec709_y(chunk[0], chunk[1], chunk[2]);
                    let out = lut_y[y as usize];
                    chunk[0] = out;
                    chunk[1] = out;
                    chunk[2] = out;
                }
            };
            if use_parallel {
                raw.par_chunks_mut(STRIP_ROWS * row_bytes).for_each(process_strip);
            } else {
                process_strip(raw);
            }
        } else {
            let retain_q8 = ((1.0 - plan.cast_strength.clamp(0.0, 1.0)) * 256.0).round() as i32;
            let process_strip = |strip: &mut [u8]| {
                for chunk in strip.as_chunks_mut::<CHANNELS>().0 {
                    if CHANNELS == 4 && chunk[CHANNELS - 1] == 0 {
                        continue;
                    }
                    let (r, g, b) = (chunk[0], chunk[1], chunk[2]);
                    if r == g && g == b {
                        let out = lut_y[r as usize];
                        chunk[0] = out;
                        chunk[1] = out;
                        chunk[2] = out;
                        continue;
                    }
                    let y_old = rec709_y(r, g, b) as i32;
                    let y_new = lut_y[y_old as usize] as i32;
                    let dr = ((r as i32 - y_old) * retain_q8 + 128) >> 8;
                    let dg = ((g as i32 - y_old) * retain_q8 + 128) >> 8;
                    let db = ((b as i32 - y_old) * retain_q8 + 128) >> 8;
                    let (ro, go, bo) = apply_chroma_preserving_luma_shift(y_new, dr, dg, db);
                    chunk[0] = ro;
                    chunk[1] = go;
                    chunk[2] = bo;
                }
            };
            if use_parallel {
                raw.par_chunks_mut(STRIP_ROWS * row_bytes).for_each(process_strip);
            } else {
                process_strip(raw);
            }
        }
        return;
    }

    let tables = &*FAST_LAB_TABLES;
    let do_accents = plan.has_high_chroma_accents;
    let do_dilate = do_accents && w >= 3 && h >= 3;
    let row_words = w.div_ceil(64).max(1);
    let rem = w & 63;
    let last_word_mask = if rem == 0 { u64::MAX } else { (1u64 << rem) - 1 };

    let cast_active = plan.cast_field.active;
    let dom_u_a = plan.cast_field.dom_u_a;
    let dom_u_b = plan.cast_field.dom_u_b;
    let cast_cap = plan.cast_field.c_max * 1.15;

    // Single-pass fused 3x3 dilated accent mask: when running in parallel across STRIP_ROWS strips,
    // pre-scan ONLY the 2 boundary seam rows per strip (~4% of image) from pristine `raw` so each
    // strip dilates in a tiny L1d-resident 3-row rolling bitmask with ZERO full-frame extra pass!
    let strip_rows = if !use_parallel {
        h
    } else if w * h <= 300_000 {
        h.div_ceil(4).clamp(32, 128)
    } else {
        STRIP_ROWS
    };
    let num_strips = h.div_ceil(strip_rows);
    let num_seams = if do_accents && num_strips > 1 {
        num_strips - 1
    } else {
        0
    };
    let mut seam_above = vec![0u64; num_seams * row_words];
    let mut seam_above_has = vec![0u8; num_seams];
    let mut seam_below = vec![0u64; num_seams * row_words];
    let mut seam_below_has = vec![0u8; num_seams];

    if num_seams >= 4 {
        seam_above
            .par_chunks_mut(row_words)
            .zip(seam_above_has.par_iter_mut())
            .zip(seam_below.par_chunks_mut(row_words))
            .zip(seam_below_has.par_iter_mut())
            .enumerate()
            .for_each(|(seam_idx, (((ab, ab_has), bl), bl_has))| {
                let seam_y = (seam_idx + 1) * strip_rows;
                let row_above = &raw[(seam_y - 1) * row_bytes..seam_y * row_bytes];
                *ab_has = scan_accent_row::<CHANNELS>(
                    row_above,
                    ab,
                    cast_active,
                    dom_u_a,
                    dom_u_b,
                    cast_cap,
                    tables,
                );
                let row_below = &raw[seam_y * row_bytes..(seam_y + 1) * row_bytes];
                *bl_has = scan_accent_row::<CHANNELS>(
                    row_below,
                    bl,
                    cast_active,
                    dom_u_a,
                    dom_u_b,
                    cast_cap,
                    tables,
                );
            });
    } else {
        for seam_idx in 0..num_seams {
            let seam_y = (seam_idx + 1) * strip_rows;
            let off = seam_idx * row_words;
            let row_above = &raw[(seam_y - 1) * row_bytes..seam_y * row_bytes];
            seam_above_has[seam_idx] = scan_accent_row::<CHANNELS>(
                row_above,
                &mut seam_above[off..off + row_words],
                cast_active,
                dom_u_a,
                dom_u_b,
                cast_cap,
                tables,
            );
            let row_below = &raw[seam_y * row_bytes..(seam_y + 1) * row_bytes];
            seam_below_has[seam_idx] = scan_accent_row::<CHANNELS>(
                row_below,
                &mut seam_below[off..off + row_words],
                cast_active,
                dom_u_a,
                dom_u_b,
                cast_cap,
                tables,
            );
        }
    }

    let do_cast = plan.cast_strength > 0.0 && (plan.cast_field.active || plan.has_muddy_shadow_cast);
    let localized = do_cast && plan.cast_field.localized_margin;
    let has_muddy_shadow = plan.has_muddy_shadow_cast;

    let global_c_env = (plan.cast_field.global_a * plan.cast_field.global_a
        + plan.cast_field.global_b * plan.cast_field.global_b)
        .sqrt();
    let (global_ua, global_ub) = if global_c_env > 1e-4 {
        (
            plan.cast_field.global_a / global_c_env,
            plan.cast_field.global_b / global_c_env,
        )
    } else {
        (0.0, 0.0)
    };
    let dom_c = (plan.cast_field.dom_a * plan.cast_field.dom_a
        + plan.cast_field.dom_b * plan.cast_field.dom_b)
        .sqrt();
    let dom_tuple = (dom_c, dom_u_a, dom_u_b);

    let full_w = coords.map_or(w, |c| c.full_w).max(1);
    let full_h = coords.map_or(h, |c| c.full_h).max(1);
    let fw_f = full_w as f32;
    let fh_f = full_h as f32;

    // Span width evenly divides 32px tile boundaries (16 on >=1024px wide full-res images, 8 on smaller/tile buffers)
    let span_w = if w >= 1024 && coords.is_none() { 16usize } else { 8usize };
    let num_spans = w.div_ceil(span_w);

    let span_cols: Vec<ColInterp> = if localized {
        (0..num_spans)
            .map(|si| {
                let x = (si * span_w + (span_w / 2)).min(w - 1);
                let ox = coords.map_or(x, |c| c.src_x[x]);
                let u = (((ox as f32 + 0.5) / fw_f) * 4.0 - 0.5).clamp(0.0, 3.0);
                let c0 = (u.floor() as usize).min(3);
                ColInterp {
                    c0,
                    c1: (c0 + 1).min(3),
                    fx: u - c0 as f32,
                    dx_edge: (ox.min(full_w.saturating_sub(1 + ox)) as f32) / fw_f,
                }
            })
            .collect()
    } else {
        Vec::new()
    };

    let row_y_interp = |y: usize| -> (usize, usize, f32, f32) {
        let oy = coords.map_or(y, |c| c.src_y[y]);
        let v = (((oy as f32 + 0.5) / fh_f) * 4.0 - 0.5).clamp(0.0, 3.0);
        let r0 = (v.floor() as usize).min(3);
        (
            r0,
            (r0 + 1).min(3),
            v - r0 as f32,
            (oy.min(full_h.saturating_sub(1 + oy)) as f32) / fh_f,
        )
    };

    let process_strip = |(strip_idx, strip): (usize, &mut [u8])| {
        let y_base = strip_idx * strip_rows;
        let rows_in_strip = strip.len() / row_bytes;
        let y_end = y_base + rows_in_strip;

        // Precompute run-length-merged horizontal cast envelope ONCE per interior full-res strip
        let mut strip_runs: Vec<EnvRun> = Vec::new();
        let use_strip_runs = if localized && coords.is_none() && rows_in_strip > 0 {
            let (_, _, _, dy_top) = row_y_interp(y_base);
            let (_, _, _, dy_bot) = row_y_interp(y_end - 1);
            if dy_top >= 0.145 && dy_bot >= 0.145 {
                let (r0, r1, fy, dy_mid) = row_y_interp(y_base + rows_in_strip / 2);
                build_merged_env_runs(
                    &plan.cast_field,
                    r0,
                    r1,
                    fy,
                    dy_mid,
                    &span_cols,
                    span_w,
                    w,
                    dom_tuple,
                    &mut strip_runs,
                );
                true
            } else {
                false
            }
        } else {
            false
        };
        let mut row_runs: Vec<EnvRun> = Vec::new();

        // L1d-resident 3-row rolling accent bitmask buffers (~1 KB total on 2000px width)
        let mut prev_bits = if do_accents { vec![0u64; row_words] } else { Vec::new() };
        let mut cur_bits = if do_accents { vec![0u64; row_words] } else { Vec::new() };
        let mut next_bits = if do_accents { vec![0u64; row_words] } else { Vec::new() };
        let mut dilated_bits = if do_dilate { vec![0u64; row_words] } else { Vec::new() };
        let mut prev_has = 0u8;
        let mut cur_has = 0u8;
        let mut next_dirty = 0u8;

        if do_accents && rows_in_strip > 0 {
            if strip_idx > 0 {
                let off = (strip_idx - 1) * row_words;
                prev_has = seam_above_has[strip_idx - 1];
                if prev_has != 0 {
                    prev_bits.copy_from_slice(&seam_above[off..off + row_words]);
                }
                cur_has = seam_below_has[strip_idx - 1];
                if cur_has != 0 {
                    cur_bits.copy_from_slice(&seam_below[off..off + row_words]);
                }
            } else {
                cur_has = scan_accent_row::<CHANNELS>(
                    &strip[0..row_bytes],
                    &mut cur_bits,
                    cast_active,
                    dom_u_a,
                    dom_u_b,
                    cast_cap,
                    tables,
                );
            }
        }

        let mut cache = MixedPixelCache::new();
        let mut last_c = -1.0f32;
        let mut last_ua = 0.0f32;
        let mut last_ub = 0.0f32;

        for local_y in 0..rows_in_strip {
            let y = y_base + local_y;

            let mut next_has = 0u8;
            let has_mask = if do_accents {
                if do_dilate {
                    if local_y + 1 < rows_in_strip {
                        let next_row_raw =
                            &strip[(local_y + 1) * row_bytes..(local_y + 2) * row_bytes];
                        next_has = scan_accent_row::<CHANNELS>(
                            next_row_raw,
                            &mut next_bits,
                            cast_active,
                            dom_u_a,
                            dom_u_b,
                            cast_cap,
                            tables,
                        );
                    } else if y_end < h {
                        let off = strip_idx * row_words;
                        next_has = seam_below_has[strip_idx];
                        if next_has != 0 {
                            next_bits.copy_from_slice(&seam_below[off..off + row_words]);
                        } else if next_dirty != 0 {
                            next_bits.fill(0);
                        }
                    } else if next_dirty != 0 {
                        next_bits.fill(0);
                    }

                    if (prev_has | cur_has | next_has) != 0 {
                        dilate_3row_bits(
                            &prev_bits,
                            &cur_bits,
                            &next_bits,
                            &mut dilated_bits,
                            last_word_mask,
                        );
                        true
                    } else {
                        false
                    }
                } else {
                    cur_has != 0
                }
            } else {
                false
            };

            let mask_slice: &[u64] = if has_mask {
                if do_dilate { &dilated_bits } else { &cur_bits }
            } else {
                &[]
            };

            let row = &mut strip[local_y * row_bytes..(local_y + 1) * row_bytes];
            let row_chunks = row.as_chunks_mut::<CHANNELS>().0;

            if !localized {
                let ctx = MixedSpanCtx {
                    lut_y,
                    plan,
                    tables,
                    do_cast,
                    has_muddy_shadow,
                    c_env: global_c_env,
                    ua: global_ua,
                    ub: global_ub,
                };
                if has_mask {
                    apply_mixed_span::<CHANNELS, true>(row_chunks, 0, mask_slice, &ctx, &mut cache);
                } else {
                    apply_mixed_span::<CHANNELS, false>(row_chunks, 0, &[], &ctx, &mut cache);
                }
            } else if use_strip_runs {
                if has_mask {
                    for run in &strip_runs {
                        let ctx = MixedSpanCtx {
                            lut_y,
                            plan,
                            tables,
                            do_cast,
                            has_muddy_shadow,
                            c_env: run.c_env,
                            ua: run.ua,
                            ub: run.ub,
                        };
                        if run.c_env != last_c || run.ua != last_ua || run.ub != last_ub {
                            last_c = run.c_env;
                            last_ua = run.ua;
                            last_ub = run.ub;
                            cache.refresh_hi_for_env(&ctx);
                        }
                        apply_mixed_span::<CHANNELS, true>(
                            &mut row_chunks[run.x0..run.x1],
                            run.x0,
                            mask_slice,
                            &ctx,
                            &mut cache,
                        );
                    }
                } else {
                    for run in &strip_runs {
                        let ctx = MixedSpanCtx {
                            lut_y,
                            plan,
                            tables,
                            do_cast,
                            has_muddy_shadow,
                            c_env: run.c_env,
                            ua: run.ua,
                            ub: run.ub,
                        };
                        if run.c_env != last_c || run.ua != last_ua || run.ub != last_ub {
                            last_c = run.c_env;
                            last_ua = run.ua;
                            last_ub = run.ub;
                            cache.refresh_hi_for_env(&ctx);
                        }
                        apply_mixed_span::<CHANNELS, false>(
                            &mut row_chunks[run.x0..run.x1],
                            run.x0,
                            &[],
                            &ctx,
                            &mut cache,
                        );
                    }
                }
            } else {
                let (r0, r1, fy, dy_edge) = row_y_interp(y);
                if dy_edge <= 0.10 {
                    let ctx = MixedSpanCtx {
                        lut_y,
                        plan,
                        tables,
                        do_cast,
                        has_muddy_shadow,
                        c_env: dom_c,
                        ua: dom_u_a,
                        ub: dom_u_b,
                    };
                    if dom_c != last_c || dom_u_a != last_ua || dom_u_b != last_ub {
                        last_c = dom_c;
                        last_ua = dom_u_a;
                        last_ub = dom_u_b;
                        cache.refresh_hi_for_env(&ctx);
                    }
                    if has_mask {
                        apply_mixed_span::<CHANNELS, true>(row_chunks, 0, mask_slice, &ctx, &mut cache);
                    } else {
                        apply_mixed_span::<CHANNELS, false>(row_chunks, 0, &[], &ctx, &mut cache);
                    }
                } else {
                    build_merged_env_runs(
                        &plan.cast_field,
                        r0,
                        r1,
                        fy,
                        dy_edge,
                        &span_cols,
                        span_w,
                        w,
                        dom_tuple,
                        &mut row_runs,
                    );
                    if has_mask {
                        for run in &row_runs {
                            let ctx = MixedSpanCtx {
                                lut_y,
                                plan,
                                tables,
                                do_cast,
                                has_muddy_shadow,
                                c_env: run.c_env,
                                ua: run.ua,
                                ub: run.ub,
                            };
                            if run.c_env != last_c || run.ua != last_ua || run.ub != last_ub {
                                last_c = run.c_env;
                                last_ua = run.ua;
                                last_ub = run.ub;
                                cache.refresh_hi_for_env(&ctx);
                            }
                            apply_mixed_span::<CHANNELS, true>(
                                &mut row_chunks[run.x0..run.x1],
                                run.x0,
                                mask_slice,
                                &ctx,
                                &mut cache,
                            );
                        }
                    } else {
                        for run in &row_runs {
                            let ctx = MixedSpanCtx {
                                lut_y,
                                plan,
                                tables,
                                do_cast,
                                has_muddy_shadow,
                                c_env: run.c_env,
                                ua: run.ua,
                                ub: run.ub,
                            };
                            if run.c_env != last_c || run.ua != last_ua || run.ub != last_ub {
                                last_c = run.c_env;
                                last_ua = run.ua;
                                last_ub = run.ub;
                                cache.refresh_hi_for_env(&ctx);
                            }
                            apply_mixed_span::<CHANNELS, false>(
                                &mut row_chunks[run.x0..run.x1],
                                run.x0,
                                &[],
                                &ctx,
                                &mut cache,
                            );
                        }
                    }
                }
            }

            if do_accents {
                if do_dilate {
                    let old_prev_has = prev_has;
                    std::mem::swap(&mut prev_bits, &mut cur_bits);
                    std::mem::swap(&mut cur_bits, &mut next_bits);
                    prev_has = cur_has;
                    cur_has = next_has;
                    next_dirty = old_prev_has;
                } else if local_y + 1 < rows_in_strip {
                    let next_row_raw =
                        &strip[(local_y + 1) * row_bytes..(local_y + 2) * row_bytes];
                    cur_has = scan_accent_row::<CHANNELS>(
                        next_row_raw,
                        &mut cur_bits,
                        cast_active,
                        dom_u_a,
                        dom_u_b,
                        cast_cap,
                        tables,
                    );
                }
            }
        }
    };

    if use_parallel {
        raw.par_chunks_mut(strip_rows * row_bytes)
            .enumerate()
            .for_each(process_strip);
    } else {
        process_strip((0, raw));
    }
}

/// Applies the validated tonal LUT and arbitrary-hue directional cast subtraction in-place.
/// Normalizes 16-bit and 32F formats explicitly to their matching 8-bit representation (never no-op!).
pub fn apply_monochrome_mixed_correction(
    image: &mut DynamicImage,
    plan: &CorrectionPlan,
    classification: PageClassification,
) {
    apply_correction_internal(image, plan, classification, None);
}

fn apply_correction_internal(
    image: &mut DynamicImage,
    plan: &CorrectionPlan,
    classification: PageClassification,
    coords: Option<&TileCoords>,
) {
    // Explicit 16-bit / 32F normalization (never silent no-op!)
    match image {
        DynamicImage::ImageLuma16(_) => *image = DynamicImage::ImageLuma8(image.to_luma8()),
        DynamicImage::ImageLumaA16(_) => *image = DynamicImage::ImageLumaA8(image.to_luma_alpha8()),
        DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgb32F(_) => {
            *image = DynamicImage::ImageRgb8(image.to_rgb8());
        }
        DynamicImage::ImageRgba16(_) | DynamicImage::ImageRgba32F(_) => {
            *image = DynamicImage::ImageRgba8(image.to_rgba8());
        }
        _ => {}
    }

    let lut_y = build_monotonic_tone_lut_gamma(
        plan.black_point as f32,
        plan.white_point as f32,
        plan.gamma,
    );

    match image {
        DynamicImage::ImageLuma8(luma) => {
            if luma.len() > 250_000 {
                (**luma).par_chunks_mut(65536).for_each(|strip| {
                    for p in strip {
                        *p = lut_y[*p as usize];
                    }
                });
            } else {
                for p in &mut **luma {
                    *p = lut_y[*p as usize];
                }
            }
        }
        DynamicImage::ImageLumaA8(luma_a) => {
            // Smooth straight-alpha LUT for all A > 0 (no A=127 halo seam!), alpha untouched
            if luma_a.len() > 500_000 {
                (**luma_a).par_chunks_mut(65536).for_each(|strip| {
                    for chunk in strip.as_chunks_mut::<2>().0 {
                        if chunk[1] > 0 {
                            chunk[0] = lut_y[chunk[0] as usize];
                        }
                    }
                });
            } else {
                for chunk in (**luma_a).as_chunks_mut::<2>().0 {
                    if chunk[1] > 0 {
                        chunk[0] = lut_y[chunk[0] as usize];
                    }
                }
            }
        }
        DynamicImage::ImageRgb8(rgb) => {
            let (w, h) = (rgb.width() as usize, rgb.height() as usize);
            apply_rgb_core::<3>(rgb, w, h, &lut_y, plan, classification, coords);
        }
        DynamicImage::ImageRgba8(rgba) => {
            let (w, h) = (rgba.width() as usize, rgba.height() as usize);
            apply_rgb_core::<4>(rgba, w, h, &lut_y, plan, classification, coords);
        }
        _ => {}
    }
}

// ============================================================================
// Pre-Flight Tile Sampling & Shared Luma/Alpha Extraction
// ============================================================================

fn sample_for_quality_guard(img: &DynamicImage) -> (DynamicImage, TileCoords, bool) {
    let (w, h) = img.dimensions();
    let w_usize = w as usize;
    let h_usize = h as usize;
    let total_px = w_usize * h_usize;

    let block_stride = if total_px >= 4_000_000 && w_usize >= 192 && h_usize >= 192 {
        6usize
    } else if total_px >= 2_000_000 && w_usize >= 128 && h_usize >= 128 {
        4usize
    } else if total_px >= 500_000 && w_usize >= 128 && h_usize >= 128 {
        2usize
    } else {
        1usize
    };

    if block_stride == 1 {
        let normalized = match img {
            DynamicImage::ImageLuma16(_) => DynamicImage::ImageLuma8(img.to_luma8()),
            DynamicImage::ImageLumaA16(_) => DynamicImage::ImageLumaA8(img.to_luma_alpha8()),
            DynamicImage::ImageRgb16(_) | DynamicImage::ImageRgb32F(_) => DynamicImage::ImageRgb8(img.to_rgb8()),
            DynamicImage::ImageRgba16(_) | DynamicImage::ImageRgba32F(_) => DynamicImage::ImageRgba8(img.to_rgba8()),
            other => other.clone(),
        };
        let coords = TileCoords {
            full_w: w_usize,
            full_h: h_usize,
            src_x: (0..w_usize).collect(),
            src_y: (0..h_usize).collect(),
            tile_size: None,
        };
        return (normalized, coords, true);
    }

    const BS: usize = 32;
    let bx_count = (w_usize / BS).div_ceil(block_stride);
    let by_count = (h_usize / BS).div_ceil(block_stride);
    let sw = (bx_count * BS).max(BS);
    let sh = (by_count * BS).max(BS);

    let mut src_x = vec![0usize; sw];
    for bx in 0..bx_count {
        let origin_x = (bx * block_stride * BS).min(w_usize - BS);
        for dx in 0..BS {
            src_x[bx * BS + dx] = origin_x + dx;
        }
    }
    let mut src_y = vec![0usize; sh];
    for by in 0..by_count {
        let origin_y = (by * block_stride * BS).min(h_usize - BS);
        for dy in 0..BS {
            src_y[by * BS + dy] = origin_y + dy;
        }
    }

    let coords = TileCoords {
        full_w: w_usize,
        full_h: h_usize,
        src_x,
        src_y,
        tile_size: Some(BS),
    };

    #[allow(clippy::too_many_arguments)]
    fn extract_tiles<const CHANNELS: usize>(
        raw: &[u8],
        w_usize: usize,
        h_usize: usize,
        bx_count: usize,
        by_count: usize,
        block_stride: usize,
        sw: usize,
        sh: usize,
    ) -> Vec<u8> {
        let mut out = vec![0u8; sw * sh * CHANNELS];
        for by in 0..by_count {
            let src_by = (by * block_stride * BS).min(h_usize - BS);
            for dy in 0..BS {
                let src_row_off = (src_by + dy) * w_usize * CHANNELS;
                let dst_row_off = (by * BS + dy) * sw * CHANNELS;
                for bx in 0..bx_count {
                    let src_bx = (bx * block_stride * BS).min(w_usize - BS);
                    let src_idx = src_row_off + src_bx * CHANNELS;
                    let dst_idx = dst_row_off + (bx * BS) * CHANNELS;
                    out[dst_idx..dst_idx + BS * CHANNELS]
                        .copy_from_slice(&raw[src_idx..src_idx + BS * CHANNELS]);
                }
            }
        }
        out
    }

    let sample = match img {
        DynamicImage::ImageRgb8(rgb) => {
            let out = extract_tiles::<3>(rgb.as_raw(), w_usize, h_usize, bx_count, by_count, block_stride, sw, sh);
            DynamicImage::ImageRgb8(RgbImage::from_raw(sw as u32, sh as u32, out).unwrap())
        }
        DynamicImage::ImageRgba8(rgba) => {
            let out = extract_tiles::<4>(rgba.as_raw(), w_usize, h_usize, bx_count, by_count, block_stride, sw, sh);
            DynamicImage::ImageRgba8(RgbaImage::from_raw(sw as u32, sh as u32, out).unwrap())
        }
        DynamicImage::ImageLuma8(luma) => {
            let out = extract_tiles::<1>(luma.as_raw(), w_usize, h_usize, bx_count, by_count, block_stride, sw, sh);
            DynamicImage::ImageLuma8(GrayImage::from_raw(sw as u32, sh as u32, out).unwrap())
        }
        DynamicImage::ImageLumaA8(luma_a) => {
            let out = extract_tiles::<2>(luma_a.as_raw(), w_usize, h_usize, bx_count, by_count, block_stride, sw, sh);
            DynamicImage::ImageLumaA8(GrayAlphaImage::from_raw(sw as u32, sh as u32, out).unwrap())
        }
        DynamicImage::ImageLuma16(_) => {
            let luma = img.to_luma8();
            let out = extract_tiles::<1>(luma.as_raw(), w_usize, h_usize, bx_count, by_count, block_stride, sw, sh);
            DynamicImage::ImageLuma8(GrayImage::from_raw(sw as u32, sh as u32, out).unwrap())
        }
        DynamicImage::ImageLumaA16(_) => {
            let la = img.to_luma_alpha8();
            let out = extract_tiles::<2>(la.as_raw(), w_usize, h_usize, bx_count, by_count, block_stride, sw, sh);
            DynamicImage::ImageLumaA8(GrayAlphaImage::from_raw(sw as u32, sh as u32, out).unwrap())
        }
        DynamicImage::ImageRgba16(_) | DynamicImage::ImageRgba32F(_) => {
            let rgba = img.to_rgba8();
            let out = extract_tiles::<4>(rgba.as_raw(), w_usize, h_usize, bx_count, by_count, block_stride, sw, sh);
            DynamicImage::ImageRgba8(RgbaImage::from_raw(sw as u32, sh as u32, out).unwrap())
        }
        _ => {
            let rgb = img.to_rgb8();
            let out = extract_tiles::<3>(rgb.as_raw(), w_usize, h_usize, bx_count, by_count, block_stride, sw, sh);
            DynamicImage::ImageRgb8(RgbImage::from_raw(sw as u32, sh as u32, out).unwrap())
        }
    };

    (sample, coords, false)
}

/// Extracts luma buffer (zero-copy borrow on `ImageLuma8`) and optional opaque alpha mask
/// (`1` iff `A >= 128`) ONCE for sharing across Tiers 1-3.
fn extract_luma_and_alpha<'a>(image: &'a DynamicImage) -> (Cow<'a, [u8]>, Option<Vec<u8>>) {
    match image {
        DynamicImage::ImageLuma8(luma) => (Cow::Borrowed(luma.as_raw().as_slice()), None),
        DynamicImage::ImageLumaA8(la) => {
            let raw = la.as_raw();
            let n = raw.len() / 2;
            let mut y_buf = vec![0u8; n];
            let mut a_buf = vec![0u8; n];
            let mut any_trans = false;
            for (i, px) in raw.as_chunks::<2>().0.iter().enumerate() {
                y_buf[i] = px[0];
                let ok = px[1] >= 128;
                any_trans |= !ok;
                a_buf[i] = u8::from(ok);
            }
            (Cow::Owned(y_buf), if any_trans { Some(a_buf) } else { None })
        }
        DynamicImage::ImageRgb8(rgb) => {
            let raw = rgb.as_raw();
            let n = (rgb.width() * rgb.height()) as usize;
            let mut out = vec![0u8; n];
            if n >= 64_000 {
                out.par_chunks_mut(16384)
                    .zip(raw.par_chunks(16384 * 3))
                    .for_each(|(dst, src)| {
                        for (d, px) in dst.iter_mut().zip(src.as_chunks::<3>().0) {
                            *d = rec709_y(px[0], px[1], px[2]);
                        }
                    });
            } else {
                for (d, px) in out.iter_mut().zip(raw.as_chunks::<3>().0) {
                    *d = rec709_y(px[0], px[1], px[2]);
                }
            }
            (Cow::Owned(out), None)
        }
        DynamicImage::ImageRgba8(rgba) => {
            let raw = rgba.as_raw();
            let n = (rgba.width() * rgba.height()) as usize;
            let mut y_buf = vec![0u8; n];
            let mut a_buf = vec![0u8; n];
            let any_trans = if n >= 64_000 {
                use std::sync::atomic::{AtomicBool, Ordering};
                let flag = AtomicBool::new(false);
                y_buf
                    .par_chunks_mut(16384)
                    .zip(a_buf.par_chunks_mut(16384))
                    .zip(raw.par_chunks(16384 * 4))
                    .for_each(|((yd, ad), src)| {
                        let mut local_trans = false;
                        for ((y_out, a_out), px) in
                            yd.iter_mut().zip(ad.iter_mut()).zip(src.as_chunks::<4>().0)
                        {
                            *y_out = rec709_y(px[0], px[1], px[2]);
                            let ok = px[3] >= 128;
                            local_trans |= !ok;
                            *a_out = u8::from(ok);
                        }
                        if local_trans {
                            flag.store(true, Ordering::Relaxed);
                        }
                    });
                flag.load(Ordering::Relaxed)
            } else {
                let mut local_trans = false;
                for (i, px) in raw.as_chunks::<4>().0.iter().enumerate() {
                    y_buf[i] = rec709_y(px[0], px[1], px[2]);
                    let ok = px[3] >= 128;
                    local_trans |= !ok;
                    a_buf[i] = u8::from(ok);
                }
                local_trans
            };
            (Cow::Owned(y_buf), if any_trans { Some(a_buf) } else { None })
        }
        _ => (Cow::Owned(image.to_luma8().into_raw()), None),
    }
}

/// Extracts only luma (zero-copy borrow on `ImageLuma8`, no redundant alpha mask allocation)
/// for the leveled candidate image whose alpha channel is untouched.
fn extract_luma_only<'a>(image: &'a DynamicImage) -> Cow<'a, [u8]> {
    match image {
        DynamicImage::ImageLuma8(luma) => Cow::Borrowed(luma.as_raw().as_slice()),
        DynamicImage::ImageLumaA8(la) => {
            let raw = la.as_raw();
            let mut y_buf = Vec::with_capacity(raw.len() / 2);
            for px in raw.as_chunks::<2>().0 {
                y_buf.push(px[0]);
            }
            Cow::Owned(y_buf)
        }
        DynamicImage::ImageRgb8(rgb) => {
            let raw = rgb.as_raw();
            let n = (rgb.width() * rgb.height()) as usize;
            let mut out = vec![0u8; n];
            if n >= 64_000 {
                out.par_chunks_mut(16384)
                    .zip(raw.par_chunks(16384 * 3))
                    .for_each(|(dst, src)| {
                        for (d, px) in dst.iter_mut().zip(src.as_chunks::<3>().0) {
                            *d = rec709_y(px[0], px[1], px[2]);
                        }
                    });
            } else {
                for (d, px) in out.iter_mut().zip(raw.as_chunks::<3>().0) {
                    *d = rec709_y(px[0], px[1], px[2]);
                }
            }
            Cow::Owned(out)
        }
        DynamicImage::ImageRgba8(rgba) => {
            let raw = rgba.as_raw();
            let n = (rgba.width() * rgba.height()) as usize;
            let mut out = vec![0u8; n];
            if n >= 64_000 {
                out.par_chunks_mut(16384)
                    .zip(raw.par_chunks(16384 * 4))
                    .for_each(|(dst, src)| {
                        for (d, px) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                            *d = rec709_y(px[0], px[1], px[2]);
                        }
                    });
            } else {
                for (d, px) in out.iter_mut().zip(raw.as_chunks::<4>().0) {
                    *d = rec709_y(px[0], px[1], px[2]);
                }
            }
            Cow::Owned(out)
        }
        _ => Cow::Owned(image.to_luma8().into_raw()),
    }
}

#[inline]
fn histogram_from_luma(luma: &[u8], alpha: Option<&[u8]>) -> [u32; 256] {
    let mut hist = [0u32; 256];
    match alpha {
        Some(a) => {
            for (&y, &ok) in luma.iter().zip(a.iter()) {
                if ok != 0 {
                    hist[y as usize] += 1;
                }
            }
        }
        None => {
            for &y in luma {
                hist[y as usize] += 1;
            }
        }
    }
    hist
}

// ============================================================================
// Fixed 3-Tier Quality Guard Verification Engine
// ============================================================================

/// Computes structure-only SSIM `S(x,y) = (sigma_xy + c3) / (sigma_x * sigma_y + c3)` on active
/// 8x8 blocks with a one-sided contrast-collapse penalty (`sigma_y < sigma_x`) and alpha awareness.
fn compute_structure_ssim_from_luma(
    orig_luma: &[u8],
    lev_luma: &[u8],
    alpha_mask: Option<&[u8]>,
    w_usize: usize,
    h_usize: usize,
) -> f32 {
    let block_size = 8usize;
    let blocks_x = w_usize / block_size;
    let blocks_y = h_usize / block_size;
    if blocks_x == 0 || blocks_y == 0 {
        return 1.0;
    }

    const C2: f64 = 58.5225;
    const C3: f64 = C2 * 0.5; // 29.26125

    let eval_block_row = |by: usize| -> (f64, u64) {
        let mut row_sum = 0.0f64;
        let mut row_active = 0u64;
        let y_start = by * block_size;
        for bx in 0..blocks_x {
            let x_start = bx * block_size;
            let mut sum_x = 0u32;
            let mut sum_y = 0u32;
            let mut sum_x_sq = 0u64;
            let mut sum_y_sq = 0u64;
            let mut sum_xy = 0u64;
            let mut count = 0u32;

            for dy in 0..block_size {
                let row_off = (y_start + dy) * w_usize + x_start;
                let o_slice = &orig_luma[row_off..row_off + block_size];
                let l_slice = &lev_luma[row_off..row_off + block_size];
                if let Some(am) = alpha_mask {
                    let a_slice = &am[row_off..row_off + block_size];
                    for i in 0..block_size {
                        if a_slice[i] == 0 {
                            continue;
                        }
                        let px = o_slice[i] as u32;
                        let py = l_slice[i] as u32;
                        sum_x += px;
                        sum_y += py;
                        sum_x_sq += (px * px) as u64;
                        sum_y_sq += (py * py) as u64;
                        sum_xy += (px * py) as u64;
                        count += 1;
                    }
                } else {
                    for i in 0..block_size {
                        let px = o_slice[i] as u32;
                        let py = l_slice[i] as u32;
                        sum_x += px;
                        sum_y += py;
                        sum_x_sq += (px * px) as u64;
                        sum_y_sq += (py * py) as u64;
                        sum_xy += (px * py) as u64;
                    }
                    count += block_size as u32;
                }
            }

            if count < 16 {
                continue;
            }

            let n = count as f64;
            let mu_x = sum_x as f64 / n;
            let mu_y = sum_y as f64 / n;
            let var_x = ((sum_x_sq as f64 / n) - (mu_x * mu_x)).max(0.0);
            let var_y = ((sum_y_sq as f64 / n) - (mu_y * mu_y)).max(0.0);
            let sigma_x = var_x.sqrt();
            let sigma_y = var_y.sqrt();

            // Evaluate active blocks only so solid flat ink/paper fills don't dilute structure signal
            if sigma_x < 2.0 && sigma_y < 2.0 {
                continue;
            }

            let sigma_xy = (sum_xy as f64 / n) - (mu_x * mu_y);
            let s_term = (sigma_xy + C3) / (sigma_x * sigma_y + C3);
            // One-sided contrast-loss penalty: never penalizes contrast expansion (sigma_y >= sigma_x),
            // but immediately catches crushed detail where sigma_y collapses toward 0.
            let c_loss = if sigma_y < sigma_x {
                (2.0 * sigma_x * sigma_y + C2) / (var_x + var_y + C2)
            } else {
                1.0
            };
            row_sum += (s_term * c_loss).clamp(-1.0, 1.0);
            row_active += 1;
        }
        (row_sum, row_active)
    };

    let (ssim_sum, active_blocks) = if w_usize * h_usize >= 64_000 {
        (0..blocks_y)
            .into_par_iter()
            .map(eval_block_row)
            .reduce(|| (0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    } else {
        (0..blocks_y)
            .map(eval_block_row)
            .fold((0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
    };

    if active_blocks == 0 {
        1.0
    } else {
        (ssim_sum / active_blocks as f64) as f32
    }
}

pub fn compute_luminance_ssim(orig: &DynamicImage, leveled: &DynamicImage) -> f32 {
    let (w, h) = orig.dimensions();
    if w == 0 || h == 0 || leveled.dimensions() != (w, h) {
        return 1.0;
    }
    let (orig_luma, alpha_mask) = extract_luma_and_alpha(orig);
    let lev_luma = extract_luma_only(leveled);
    compute_structure_ssim_from_luma(&orig_luma, &lev_luma, alpha_mask.as_deref(), w as usize, h as usize)
}

fn compute_strided_gradient_ncc_from_luma(
    orig_luma: &[u8],
    lev_luma: &[u8],
    alpha_mask: Option<&[u8]>,
    w_usize: usize,
    h_usize: usize,
    stride: usize,
    tile_size: Option<usize>,
) -> f32 {
    if w_usize < 3 || h_usize < 3 {
        return 1.0;
    }
    let step = stride.max(1);

    let eval_row = |y: usize| -> (f64, f64, f64) {
        let mut row_prod = 0.0f64;
        let mut row_sq1 = 0.0f64;
        let mut row_sq2 = 0.0f64;
        let r_prev = (y - 1) * w_usize;
        let r_curr = y * w_usize;
        let r_next = (y + 1) * w_usize;

        for x in (1..w_usize - 1).step_by(step) {
            if let Some(ts) = tile_size
                && (x.is_multiple_of(ts) || x % ts == ts - 1)
            {
                continue;
            }
            if let Some(am) = alpha_mask
                && (am[r_prev + x - 1] == 0
                    || am[r_prev + x] == 0
                    || am[r_prev + x + 1] == 0
                    || am[r_curr + x - 1] == 0
                    || am[r_curr + x] == 0
                    || am[r_curr + x + 1] == 0
                    || am[r_next + x - 1] == 0
                    || am[r_next + x] == 0
                    || am[r_next + x + 1] == 0)
            {
                continue;
            }

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
    };

    let (sum_prod, sum_sq1, sum_sq2) = if w_usize * h_usize > 400_000 {
        let y_indices: Vec<usize> = (1..h_usize - 1)
            .step_by(step)
            .filter(|&y| match tile_size {
                Some(ts) => {
                    let m = y % ts;
                    m != 0 && m != ts - 1
                }
                None => true,
            })
            .collect();
        y_indices
            .into_par_iter()
            .map(eval_row)
            .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2))
    } else {
        let mut acc = (0.0f64, 0.0f64, 0.0f64);
        for y in (1..h_usize - 1).step_by(step) {
            if let Some(ts) = tile_size {
                let m = y % ts;
                if m == 0 || m == ts - 1 {
                    continue;
                }
            }
            let r = eval_row(y);
            acc.0 += r.0;
            acc.1 += r.1;
            acc.2 += r.2;
        }
        acc
    };

    let denom = (sum_sq1 * sum_sq2).sqrt();
    if denom > 1e-6 {
        (sum_prod / denom).clamp(-1.0, 1.0) as f32
    } else {
        1.0
    }
}

pub fn compute_strided_gradient_ncc_parallel(
    original: &DynamicImage,
    leveled: &DynamicImage,
    stride: usize,
) -> f32 {
    let (w, h) = original.dimensions();
    if w < 3 || h < 3 || leveled.dimensions() != (w, h) {
        return 1.0;
    }
    let (orig_luma, alpha_mask) = extract_luma_and_alpha(original);
    let lev_luma = extract_luma_only(leveled);
    compute_strided_gradient_ncc_from_luma(
        &orig_luma,
        &lev_luma,
        alpha_mask.as_deref(),
        w as usize,
        h as usize,
        stride,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn compute_active_block_contrast_ratios_from_luma(
    orig_luma: &[u8],
    lev_luma: &[u8],
    alpha_mask: Option<&[u8]>,
    w_usize: usize,
    h_usize: usize,
    block_size: usize,
    min_dynamic_range: u8,
    epsilon: f32,
    exempt_black_max: Option<u8>,
    exempt_white_min: Option<u8>,
) -> Vec<f32> {
    let blocks_x = w_usize / block_size;
    let blocks_y = h_usize / block_size;
    if blocks_x == 0 || blocks_y == 0 {
        return Vec::new();
    }

    let eval_row_blocks = |by: usize, out: &mut Vec<f32>| {
        let y_start = by * block_size;
        for bx in 0..blocks_x {
            let x_start = bx * block_size;
            let mut o_min = 255u8;
            let mut o_max = 0u8;
            let mut l_min = 255u8;
            let mut l_max = 0u8;
            let mut opaque_count = 0usize;

            for dy in 0..block_size {
                let row_off = (y_start + dy) * w_usize + x_start;
                let o_slice = &orig_luma[row_off..row_off + block_size];
                let l_slice = &lev_luma[row_off..row_off + block_size];
                if let Some(am) = alpha_mask {
                    let a_slice = &am[row_off..row_off + block_size];
                    for i in 0..block_size {
                        if a_slice[i] == 0 {
                            continue;
                        }
                        let po = o_slice[i];
                        let pl = l_slice[i];
                        o_min = o_min.min(po);
                        o_max = o_max.max(po);
                        l_min = l_min.min(pl);
                        l_max = l_max.max(pl);
                        opaque_count += 1;
                    }
                } else {
                    for i in 0..block_size {
                        let po = o_slice[i];
                        let pl = l_slice[i];
                        o_min = o_min.min(po);
                        o_max = o_max.max(po);
                        l_min = l_min.min(pl);
                        l_max = l_max.max(pl);
                    }
                    opaque_count += block_size;
                }
            }

            if opaque_count < 64 || (o_max as i32 - o_min as i32) < min_dynamic_range as i32 {
                continue;
            }

            // Exempt blocks lying strictly inside intentionally compressed paper whitening
            // boundaries (e.g. speech bubble edge on aged paper) or ink toe boundaries.
            if exempt_white_min.is_some_and(|w_min| o_min >= w_min)
                || exempt_black_max.is_some_and(|b_max| o_max <= b_max)
            {
                continue;
            }

            let c_orig = (o_max as f32 - o_min as f32) / (o_max as f32 + o_min as f32 + epsilon);
            let c_lev = (l_max as f32 - l_min as f32) / (l_max as f32 + l_min as f32 + epsilon);

            if c_orig > 1e-4 {
                out.push(c_lev / c_orig);
            }
        }
    };

    if w_usize * h_usize > 400_000 {
        (0..blocks_y)
            .into_par_iter()
            .flat_map(|by| {
                let mut row_ratios = Vec::with_capacity(blocks_x);
                eval_row_blocks(by, &mut row_ratios);
                row_ratios
            })
            .collect()
    } else {
        let mut ratios = Vec::with_capacity(blocks_x * blocks_y);
        for by in 0..blocks_y {
            eval_row_blocks(by, &mut ratios);
        }
        ratios
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
    if w == 0 || h == 0 || leveled.dimensions() != (w, h) {
        return Vec::new();
    }
    let (orig_luma, alpha_mask) = extract_luma_and_alpha(original);
    let lev_luma = extract_luma_only(leveled);
    compute_active_block_contrast_ratios_from_luma(
        &orig_luma,
        &lev_luma,
        alpha_mask.as_deref(),
        w as usize,
        h as usize,
        block_size.max(1) as usize,
        min_dynamic_range,
        epsilon,
        None,
        None,
    )
}

pub fn compute_image_luminance_histogram(image: &DynamicImage) -> [u32; 256] {
    let (luma, alpha) = extract_luma_and_alpha(image);
    histogram_from_luma(&luma, alpha.as_deref())
}

/// Tier 1 Quality Guard: Non-tautological monotonic-affine residual distortion, dynamic range
/// collapse check, and midtone-crush guard (fused single integer pass, zero u8 overflow, zero tautology).
fn quality_tier_1_from_luma(
    orig_luma: &[u8],
    lev_luma: &[u8],
    alpha_mask: Option<&[u8]>,
    plan: &CorrectionPlan,
) -> TierResult {
    let mut metrics = HashMap::new();
    let min_mid = (plan.black_point as u16 + 12).min(200) as u8;
    let max_mid = plan.white_point.saturating_sub(12).max(55);
    let check_midtone = GUARD_TIER1_CHECK_TAIL_EROSION && min_mid < max_mid;

    let mut orig_hist = [0u32; 256];
    let mut lev_hist = [0u32; 256];
    let mut sum_x = 0u64;
    let mut sum_y = 0u64;
    let mut sum_xx = 0u64;
    let mut sum_xy = 0u64;
    let mut n = 0u64;
    let mut midtone_count = 0u64;
    let mut midtone_crushed = 0u64;

    if let Some(am) = alpha_mask {
        for i in 0..orig_luma.len() {
            if am[i] == 0 {
                continue;
            }
            let xo = orig_luma[i];
            let yl = lev_luma[i];
            orig_hist[xo as usize] += 1;
            lev_hist[yl as usize] += 1;
            let xu = xo as u64;
            let yu = yl as u64;
            sum_x += xu;
            sum_y += yu;
            sum_xx += xu * xu;
            sum_xy += xu * yu;
            n += 1;
            if check_midtone && xo >= min_mid && xo <= max_mid {
                midtone_count += 1;
                if yl <= 2 || yl >= 253 {
                    midtone_crushed += 1;
                }
            }
        }
    } else {
        for (&xo, &yl) in orig_luma.iter().zip(lev_luma.iter()) {
            orig_hist[xo as usize] += 1;
            lev_hist[yl as usize] += 1;
            let xu = xo as u64;
            let yu = yl as u64;
            sum_x += xu;
            sum_y += yu;
            sum_xx += xu * xu;
            sum_xy += xu * yu;
            if check_midtone && xo >= min_mid && xo <= max_mid {
                midtone_count += 1;
                if yl <= 2 || yl >= 253 {
                    midtone_crushed += 1;
                }
            }
        }
        n = orig_luma.len() as u64;
    }

    if n == 0 {
        return TierResult::pass(1);
    }

    // 1. Dynamic Range Collapse Check (aligned with Tier1Config.min_dynamic_range)
    let orig_p01 = histogram_percentile(&orig_hist, n, 1.0);
    let orig_p99 = histogram_percentile(&orig_hist, n, 99.0);
    let lev_p01 = histogram_percentile(&lev_hist, n, 1.0);
    let lev_p99 = histogram_percentile(&lev_hist, n, 99.0);
    let orig_range = (orig_p99 - orig_p01).max(0.0);
    let lev_range = (lev_p99 - lev_p01).max(0.0);
    let min_dr = GUARD_TIER1_MIN_DYNAMIC_RANGE;
    metrics.insert("orig_range".into(), orig_range as f64);
    metrics.insert("lev_range".into(), lev_range as f64);

    if (orig_range >= 20.0 && lev_range < orig_range * 0.60)
        || (orig_range >= min_dr && lev_range < min_dr)
    {
        let mut res = TierResult::reject(
            1,
            format!(
                "Dynamic range collapsed from {:.1} to {:.1} (min_dynamic_range={:.0})",
                orig_range, lev_range, min_dr
            ),
        );
        res.metrics = metrics;
        return res;
    }

    // 2. Midtone-Crush Check (replaces broken single-bin u8-overflowing slope check)
    if check_midtone && midtone_count >= n / 20 {
        let crush_ratio = midtone_crushed as f64 / midtone_count as f64;
        metrics.insert("midtone_crush_ratio".into(), crush_ratio);
        if crush_ratio > 0.25 {
            let mut res = TierResult::reject(
                1,
                format!("Midtone pixels crushed into extreme black/white ({:.1}%)", crush_ratio * 100.0),
            );
            res.metrics = metrics;
            return res;
        }
    }

    // 3. Clamped Positive-Slope Affine Residual MSE & PSNR (independent of plan LUT — zero tautology!)
    let nf = n as f64;
    let mu_x = sum_x as f64 / nf;
    let mu_y = sum_y as f64 / nf;
    let var_x = (sum_xx as f64 / nf) - mu_x * mu_x;
    let cov_xy = (sum_xy as f64 / nf) - mu_x * mu_y;

    let (a, b) = if var_x < 1.0 {
        (1.0, 0.0)
    } else {
        let slope = (cov_xy / var_x).clamp(0.5, 1.85);
        (slope, mu_y - slope * mu_x)
    };

    let mut pred_lut = [0.0f32; 256];
    for (x, slot) in pred_lut.iter_mut().enumerate() {
        *slot = (a * x as f64 + b).clamp(0.0, 255.0) as f32;
    }

    let mut sum_sq = 0.0f64;
    if let Some(am) = alpha_mask {
        for i in 0..orig_luma.len() {
            if am[i] == 0 {
                continue;
            }
            let diff = lev_luma[i] as f32 - pred_lut[orig_luma[i] as usize];
            sum_sq += (diff * diff) as f64;
        }
    } else {
        for (&xo, &yl) in orig_luma.iter().zip(lev_luma.iter()) {
            let diff = yl as f32 - pred_lut[xo as usize];
            sum_sq += (diff * diff) as f64;
        }
    }

    let mse = sum_sq / nf;
    let psnr = if mse > 1e-6 {
        10.0 * (65025.0 / mse).log10()
    } else {
        99.0
    };

    metrics.insert("mse".into(), mse);
    metrics.insert("psnr".into(), psnr);

    const MAX_AFFINE_RESIDUAL_MSE: f64 = 220.0;

    if mse > MAX_AFFINE_RESIDUAL_MSE {
        let mut res = TierResult::reject(
            1,
            format!(
                "Residual distortion MSE {:.1} exceeded ceiling {:.1}",
                mse, MAX_AFFINE_RESIDUAL_MSE
            ),
        );
        res.metrics = metrics;
        return res;
    }

    let mut res = TierResult::pass(1);
    res.metrics = metrics;
    res
}

pub fn quality_tier_1(
    original: &DynamicImage,
    leveled: &DynamicImage,
    plan: &CorrectionPlan,
) -> TierResult {
    let (orig_luma, alpha_mask) = extract_luma_and_alpha(original);
    let lev_luma = extract_luma_only(leveled);
    quality_tier_1_from_luma(&orig_luma, &lev_luma, alpha_mask.as_deref(), plan)
}

/// Tier 2 Quality Guard: Structure-only SSIM (`>= ssim_threshold`) + 2D prime-lattice CIELAB
/// chromaticity drift check (no `or+dy` shortcut, alpha-aware, exact cast-only exemption mask).
#[allow(clippy::too_many_arguments)]
fn quality_tier_2_internal(
    original: &DynamicImage,
    leveled: &DynamicImage,
    orig_luma: &[u8],
    lev_luma: &[u8],
    alpha_mask: Option<&[u8]>,
    classification: PageClassification,
    plan: &CorrectionPlan,
    coords: Option<&TileCoords>,
    ssim_threshold: f32,
) -> TierResult {
    let mut metrics = HashMap::new();
    let (w, h) = original.dimensions();
    let (w_usize, h_usize) = (w as usize, h as usize);

    let ssim = compute_structure_ssim_from_luma(orig_luma, lev_luma, alpha_mask, w_usize, h_usize);
    metrics.insert("ssim".into(), ssim as f64);

    if ssim < ssim_threshold {
        let mut res = TierResult::reject(
            2,
            format!(
                "Structure SSIM {:.3} below threshold {:.2}",
                ssim, ssim_threshold
            ),
        );
        res.metrics = metrics;
        return res;
    }

    if classification == PageClassification::Color || classification == PageClassification::Mixed {
        let mut chroma_deltas = Vec::with_capacity(4096);
        let full_w = coords.map_or(w_usize, |c| c.full_w).max(1);
        let full_h = coords.map_or(h_usize, |c| c.full_h).max(1);

        // 2D coprime prime-lattice stride (~3,600 samples across the 2D plane — zero column aliasing!)
        let raw_step = (((w_usize * h_usize) as f64) / 3600.0).sqrt().floor() as usize;
        let step = if raw_step <= 1 {
            1usize
        } else if raw_step.is_multiple_of(2) {
            raw_step + 1
        } else {
            raw_step
        };

        let tables = &*FAST_LAB_TABLES;
        let sample_chroma = |orig_raw: &[u8], lev_raw: &[u8], channels: usize, out: &mut Vec<f32>| {
            let mut last_orig_key = u32::MAX;
            let mut last_orig_lab = (0.0f32, 0.0f32, 0.0f32);
            let mut last_lev_key = u32::MAX;
            let mut last_lev_ab = (0.0f32, 0.0f32);
            for y in (0..h_usize).step_by(step) {
                let x_start = if step > 1 { ((y / step) * 3) % step } else { 0 };
                let row_off = y * w_usize * channels;
                let oy = coords.map_or(y, |c| c.src_y[y]);
                for x in (x_start..w_usize).step_by(step) {
                    let idx = row_off + x * channels;
                    if channels == 4 && orig_raw[idx + 3] < 128 {
                        continue;
                    }
                    let (or, og, ob) = (orig_raw[idx], orig_raw[idx + 1], orig_raw[idx + 2]);
                    let (lr, lg, lb) = (lev_raw[idx], lev_raw[idx + 1], lev_raw[idx + 2]);
                    if or == lr && og == lg && ob == lb {
                        out.push(0.0);
                        continue;
                    }
                    let orig_key = (or as u32) | ((og as u32) << 8) | ((ob as u32) << 16);
                    let (l_orig, a_orig, b_orig) = if orig_key == last_orig_key {
                        last_orig_lab
                    } else {
                        let lab = rgb_to_lab_with_tables(tables, or, og, ob);
                        last_orig_key = orig_key;
                        last_orig_lab = lab;
                        lab
                    };
                    let ox = coords.map_or(x, |c| c.src_x[x]);
                    // Exact cast-only exemption mask: exempts ONLY pixels targeted by directional cast
                    // subtraction (never blindly exempts pastel blue sky, pale skin, cream skin, or blonde!)
                    if compute_cast_weight_at(ox, oy, full_w, full_h, l_orig, a_orig, b_orig, plan) >= 0.15 {
                        continue;
                    }
                    let lev_key = (lr as u32) | ((lg as u32) << 8) | ((lb as u32) << 16);
                    let (a_lev, b_lev) = if lev_key == last_lev_key {
                        last_lev_ab
                    } else {
                        let (_, a_l, b_l) = rgb_to_lab_with_tables(tables, lr, lg, lb);
                        last_lev_key = lev_key;
                        last_lev_ab = (a_l, b_l);
                        last_lev_ab
                    };
                    let delta_chroma = (a_orig - a_lev).hypot(b_orig - b_lev);
                    out.push(delta_chroma);
                }
            }
        };

        match (original, leveled) {
            (DynamicImage::ImageRgb8(o), DynamicImage::ImageRgb8(l)) => {
                sample_chroma(o.as_raw(), l.as_raw(), 3, &mut chroma_deltas);
            }
            (DynamicImage::ImageRgba8(o), DynamicImage::ImageRgba8(l)) => {
                sample_chroma(o.as_raw(), l.as_raw(), 4, &mut chroma_deltas);
            }
            _ => {
                let o_rgba = original.to_rgba8();
                let l_rgba = leveled.to_rgba8();
                sample_chroma(o_rgba.as_raw(), l_rgba.as_raw(), 4, &mut chroma_deltas);
            }
        }

        if !chroma_deltas.is_empty() {
            chroma_deltas.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let p95_idx = ((chroma_deltas.len() as f32 * 0.95).floor() as usize).min(chroma_deltas.len() - 1);
            let med_idx = chroma_deltas.len() / 2;
            let p95_delta = chroma_deltas[p95_idx];
            let median_delta = chroma_deltas[med_idx];

            metrics.insert("chroma_delta_p95".into(), p95_delta as f64);
            metrics.insert("chroma_delta_median".into(), median_delta as f64);

            if p95_delta > GUARD_TIER2_CHROMA_DELTA_P95 {
                let mut res = TierResult::reject(
                    2,
                    format!(
                        "Color drift: p95 Chroma Delta={:.2} > {:.2}",
                        p95_delta, GUARD_TIER2_CHROMA_DELTA_P95
                    ),
                );
                res.metrics = metrics;
                return res;
            }

            if median_delta > GUARD_TIER2_CHROMA_DELTA_MEDIAN {
                let mut res = TierResult::reject(
                    2,
                    format!(
                        "Color drift: median Chroma Delta={:.2} > {:.2}",
                        median_delta, GUARD_TIER2_CHROMA_DELTA_MEDIAN
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

pub fn quality_tier_2(
    original: &DynamicImage,
    leveled: &DynamicImage,
    classification: PageClassification,
    plan: &CorrectionPlan,
) -> TierResult {
    let (orig_luma, alpha_mask) = extract_luma_and_alpha(original);
    let lev_luma = extract_luma_only(leveled);
    quality_tier_2_internal(
        original,
        leveled,
        &orig_luma,
        &lev_luma,
        alpha_mask.as_deref(),
        classification,
        plan,
        None,
        GUARD_TIER2_SSIM_THRESHOLD,
    )
}

/// Tier 3 Quality Guard: Tile-seam-safe strided Sobel NCC + active 32x32 block Michelson contrast
/// ratio (`p01`, exempting intentional paper/ink boundary whitening blocks).
#[allow(clippy::too_many_arguments)]
fn quality_tier_3_from_luma(
    orig_luma: &[u8],
    lev_luma: &[u8],
    alpha_mask: Option<&[u8]>,
    w_usize: usize,
    h_usize: usize,
    plan: &CorrectionPlan,
    tile_size: Option<usize>,
) -> TierResult {
    let mut metrics = HashMap::new();

    let ncc = compute_strided_gradient_ncc_from_luma(
        orig_luma,
        lev_luma,
        alpha_mask,
        w_usize,
        h_usize,
        GUARD_TIER3_STRIDE,
        tile_size,
    );
    metrics.insert("gradient_ncc".into(), ncc as f64);

    if ncc < GUARD_TIER3_GRADIENT_NCC {
        let mut res = TierResult::reject(
            3,
            format!("Gradient NCC {:.3} damaged fine linework/screentones", ncc),
        );
        res.metrics = metrics;
        return res;
    }

    let exempt_black_max = if plan.black_point > 1 {
        Some(plan.black_point.saturating_add(4))
    } else {
        None
    };
    let exempt_white_min = if plan.white_point < 254 {
        Some(plan.white_point.saturating_sub(4))
    } else {
        None
    };

    let mut block_ratios = compute_active_block_contrast_ratios_from_luma(
        orig_luma,
        lev_luma,
        alpha_mask,
        w_usize,
        h_usize,
        32,
        25,
        1.0,
        exempt_black_max,
        exempt_white_min,
    );

    if !block_ratios.is_empty() {
        block_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p01_idx = ((block_ratios.len() as f32 * 0.01).floor() as usize).min(block_ratios.len() - 1);
        let p01_ratio = block_ratios[p01_idx];
        metrics.insert("contrast_ratio_p01".into(), p01_ratio as f64);

        if p01_ratio < GUARD_TIER3_MIN_CONTRAST_RATIO {
            let mut res = TierResult::reject(
                3,
                format!(
                    "Local contrast collapsed: p01 ratio {:.2} < {:.2}",
                    p01_ratio, GUARD_TIER3_MIN_CONTRAST_RATIO
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

pub fn quality_tier_3(
    original: &DynamicImage,
    leveled: &DynamicImage,
    plan: &CorrectionPlan,
) -> TierResult {
    let (w, h) = original.dimensions();
    let (orig_luma, alpha_mask) = extract_luma_and_alpha(original);
    let lev_luma = extract_luma_only(leveled);
    quality_tier_3_from_luma(
        &orig_luma,
        &lev_luma,
        alpha_mask.as_deref(),
        w as usize,
        h as usize,
        plan,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn evaluate_guard_on_sample(
    guard_pristine: &DynamicImage,
    guard_candidate: &DynamicImage,
    orig_luma: &[u8],
    alpha_mask: Option<&[u8]>,
    classification: PageClassification,
    plan: &CorrectionPlan,
    coords: &TileCoords,
    ssim_threshold: f32,
    book_id: &str,
    page_str: &str,
) -> (Outcome, Vec<TierResult>, bool) {
    let (sw, sh) = guard_pristine.dimensions();
    let cand_luma = extract_luma_only(guard_candidate);
    let mut tier_results = Vec::with_capacity(3);

    let t1 = quality_tier_1_from_luma(orig_luma, &cand_luma, alpha_mask, plan);
    if t1.outcome == Outcome::Reject {
        warn!(
            "[QualityGuard] REJECT book_id={} page={} tier=1 reason=\"{}\" metrics={:?}",
            book_id,
            page_str,
            t1.reason.as_deref().unwrap_or(""),
            t1.metrics
        );
        tier_results.push(t1);
        return (Outcome::Reject, tier_results, false); // Tier 1 gross failure -> immediate rollback
    }
    tier_results.push(t1);

    let (t2, t3) = if (sw as usize) * (sh as usize) >= 64_000 {
        rayon::join(
            || {
                quality_tier_2_internal(
                    guard_pristine,
                    guard_candidate,
                    orig_luma,
                    &cand_luma,
                    alpha_mask,
                    classification,
                    plan,
                    Some(coords),
                    ssim_threshold,
                )
            },
            || {
                quality_tier_3_from_luma(
                    orig_luma,
                    &cand_luma,
                    alpha_mask,
                    sw as usize,
                    sh as usize,
                    plan,
                    coords.tile_size,
                )
            },
        )
    } else {
        let t2 = quality_tier_2_internal(
            guard_pristine,
            guard_candidate,
            orig_luma,
            &cand_luma,
            alpha_mask,
            classification,
            plan,
            Some(coords),
            ssim_threshold,
        );
        if t2.outcome == Outcome::Reject {
            warn!(
                "[QualityGuard] REJECT book_id={} page={} tier=2 reason=\"{}\" metrics={:?}",
                book_id,
                page_str,
                t2.reason.as_deref().unwrap_or(""),
                t2.metrics
            );
            tier_results.push(t2);
            return (Outcome::Reject, tier_results, true);
        }
        let t3 = quality_tier_3_from_luma(
            orig_luma,
            &cand_luma,
            alpha_mask,
            sw as usize,
            sh as usize,
            plan,
            coords.tile_size,
        );
        (t2, t3)
    };

    if t2.outcome == Outcome::Reject {
        warn!(
            "[QualityGuard] REJECT book_id={} page={} tier=2 reason=\"{}\" metrics={:?}",
            book_id,
            page_str,
            t2.reason.as_deref().unwrap_or(""),
            t2.metrics
        );
        tier_results.push(t2);
        return (Outcome::Reject, tier_results, true);
    }
    tier_results.push(t2);

    if t3.outcome == Outcome::Reject {
        warn!(
            "[QualityGuard] REJECT book_id={} page={} tier=3 reason=\"{}\" metrics={:?}",
            book_id,
            page_str,
            t3.reason.as_deref().unwrap_or(""),
            t3.metrics
        );
        tier_results.push(t3);
        return (Outcome::Reject, tier_results, true);
    }
    tier_results.push(t3);

    (Outcome::Pass, tier_results, false)
}

// ============================================================================
// Main Pre-Flight Level & Verify Pipeline (Zero Clones on Rollback!)
// ============================================================================

pub fn level_and_verify(
    image: DynamicImage,
    book_profile: Option<&BookProfile>,
    config: &AutoLevelsConfig,
) -> (DynamicImage, LevelInfo, QualityVerdict) {
    level_and_verify_with_context(image, None, None, book_profile, config)
}

pub fn level_and_verify_with_context(
    image: DynamicImage,
    book_id: Option<&str>,
    page_number: Option<usize>,
    book_profile: Option<&BookProfile>,
    config: &AutoLevelsConfig,
) -> (DynamicImage, LevelInfo, QualityVerdict) {
    level_and_verify_impl(
        image,
        book_id,
        page_number,
        book_profile,
        config,
        GUARD_TIER2_SSIM_THRESHOLD,
    )
}

#[doc(hidden)]
pub fn level_and_verify_with_ssim_override(
    image: DynamicImage,
    config: &AutoLevelsConfig,
    ssim_threshold: f32,
) -> (DynamicImage, LevelInfo, QualityVerdict) {
    level_and_verify_impl(image, None, None, None, config, ssim_threshold)
}

fn level_and_verify_impl(
    mut image: DynamicImage,
    book_id: Option<&str>,
    page_number: Option<usize>,
    book_profile: Option<&BookProfile>,
    config: &AutoLevelsConfig,
    ssim_threshold: f32,
) -> (DynamicImage, LevelInfo, QualityVerdict) {
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return (image, LevelInfo::default(), QualityVerdict::pass());
    }

    // Early bypass BEFORE analyze_page when disabled or thumbnail (< 400x400)
    if !config.enabled || (width < 400 && height < 400) {
        let is_mono = is_image_grayscale(&image);
        let classification = if is_mono {
            PageClassification::Monochrome
        } else {
            PageClassification::Color
        };
        return (
            image,
            LevelInfo {
                was_leveled: false,
                is_monochrome: is_mono,
                paper_cast_preserved: false,
                classification,
                black_point: 0,
                white_point: 255,
                quality_verdict: Some(QualityVerdict::pass()),
            },
            QualityVerdict::pass(),
        );
    }

    let b_id = book_id.unwrap_or("-");
    let p_str = page_number
        .map(|p| p.to_string())
        .unwrap_or_else(|| "-".to_string());
    let start_time = Instant::now();

    let is_already_luma = matches!(
        image,
        DynamicImage::ImageLuma8(_)
            | DynamicImage::ImageLumaA8(_)
            | DynamicImage::ImageLuma16(_)
            | DynamicImage::ImageLumaA16(_)
    );

    let analysis = analyze_page(&image, page_number.unwrap_or(0), config);
    let is_monochrome = analysis.classification == PageClassification::Monochrome;
    // Any un-leveled RGB Monochrome page with a detected tint/cast must preserve its RGB channels
    // post-upscale (never silently folded to 1-channel Luma8 after a veto or Quality Guard rollback).
    let has_unleveled_cast = is_monochrome
        && !is_already_luma
        && (analysis.has_paper_cast || analysis.has_muddy_shadow_cast);
    let paper_cast_preserved = has_unleveled_cast && !config.correct_paper_cast;

    // Strict Full-Color Veto (Zero-Copy No-Op)
    if analysis.classification == PageClassification::Color {
        return (
            image,
            LevelInfo {
                was_leveled: false,
                is_monochrome: false,
                paper_cast_preserved: false,
                classification: PageClassification::Color,
                black_point: 0,
                white_point: 255,
                quality_verdict: Some(QualityVerdict::pass()),
            },
            QualityVerdict::pass(),
        );
    }

    let mut plan = generate_correction_plan(&analysis, config);

    // BookProfile Outlier Sanity Guard:
    // Prevent a single anomalous page from drifting > 20 levels above the volume's median black floor.
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

    // Re-check skip_correction after BookProfile clamping
    let tonal_active =
        plan.black_point > 0 || plan.white_point < 255 || (plan.gamma - 1.0).abs() >= 1e-3;
    if !tonal_active && plan.cast_strength <= 0.0 {
        plan.skip_correction = true;
    }

    if plan.skip_correction {
        let elapsed_us = start_time.elapsed().as_micros();
        info!(
            "[AutoLevels] book_id={} page={} class={:?} black={}->0 white={}->255 cast={:.2} leveled=false time_us={}",
            b_id, p_str, analysis.classification, analysis.black_floor, analysis.white_ceiling, plan.cast_strength, elapsed_us
        );
        return (
            image,
            LevelInfo {
                was_leveled: false,
                is_monochrome,
                paper_cast_preserved: has_unleveled_cast,
                classification: analysis.classification,
                black_point: 0,
                white_point: 255,
                quality_verdict: Some(QualityVerdict::pass()),
            },
            QualityVerdict::pass(),
        );
    }

    // ========================================================================
    // Pre-Flight Tile Quality Guard (NEVER mutates full-res `image` before Pass!)
    // ========================================================================
    let (guard_pristine, tile_coords, guard_is_full_res) = sample_for_quality_guard(&image);
    let (orig_luma, alpha_mask) = extract_luma_and_alpha(&guard_pristine);

    let mut guard_leveled = guard_pristine.clone();
    apply_correction_internal(
        &mut guard_leveled,
        &plan,
        analysis.classification,
        Some(&tile_coords),
    );

    let (outcome1, tiers1, allow_retry) = evaluate_guard_on_sample(
        &guard_pristine,
        &guard_leveled,
        &orig_luma,
        alpha_mask.as_deref(),
        analysis.classification,
        &plan,
        &tile_coords,
        ssim_threshold,
        b_id,
        &p_str,
    );

    let (winning_plan, winning_sample, verdict) = if outcome1 == Outcome::Pass {
        (
            plan,
            guard_leveled,
            QualityVerdict {
                outcome: Outcome::Pass,
                attenuation_attempted: false,
                tier_results: tiers1,
            },
        )
    } else if !allow_retry {
        // Tier 1 gross failure -> immediate untouched full-res rollback (0 clones, 0 re-decodes!)
        let verdict = QualityVerdict::reject(tiers1, false);
        return (
            image,
            LevelInfo {
                was_leveled: false,
                is_monochrome,
                paper_cast_preserved: has_unleveled_cast,
                classification: analysis.classification,
                black_point: 0,
                white_point: 255,
                quality_verdict: Some(verdict.clone()),
            },
            verdict,
        );
    } else {
        // 50% Attenuation Retry on the lightweight tile sample (attenuates black, white, gamma, AND cast!)
        let mut attenuated = plan.clone();
        attenuated.black_point = (plan.black_point as f32 * 0.5).round() as u8;
        if attenuated.black_point <= 1 {
            attenuated.black_point = 0;
        }
        attenuated.white_point = 255 - ((255.0 - plan.white_point as f32) * 0.5).round() as u8;
        if attenuated.white_point >= 254 {
            attenuated.white_point = 255;
        }
        attenuated.gamma = 1.0 + (plan.gamma - 1.0) * 0.5;
        let atten_tonal_active = attenuated.black_point > 0
            || attenuated.white_point < 255
            || (attenuated.gamma - 1.0).abs() >= 1e-3;
        if analysis.classification == PageClassification::Monochrome {
            if !atten_tonal_active && !analysis.has_paper_cast && !analysis.has_muddy_shadow_cast {
                attenuated.cast_strength = 0.0;
            }
        } else {
            attenuated.cast_strength = plan.cast_strength * 0.5;
        }

        // Guard against mild plans collapsing into an identity no-op after 50% attenuation
        if !atten_tonal_active && attenuated.cast_strength <= 0.0 {
            let verdict = QualityVerdict::reject(tiers1, false);
            return (
                image,
                LevelInfo {
                    was_leveled: false,
                    is_monochrome,
                    paper_cast_preserved: has_unleveled_cast,
                    classification: analysis.classification,
                    black_point: 0,
                    white_point: 255,
                    quality_verdict: Some(verdict.clone()),
                },
                verdict,
            );
        }

        let mut guard_retry = guard_pristine.clone();
        apply_correction_internal(
            &mut guard_retry,
            &attenuated,
            analysis.classification,
            Some(&tile_coords),
        );

        let (outcome2, tiers2, _) = evaluate_guard_on_sample(
            &guard_pristine,
            &guard_retry,
            &orig_luma,
            alpha_mask.as_deref(),
            analysis.classification,
            &attenuated,
            &tile_coords,
            ssim_threshold,
            b_id,
            &p_str,
        );

        if outcome2 == Outcome::Pass {
            (
                attenuated,
                guard_retry,
                QualityVerdict {
                    outcome: Outcome::Pass,
                    attenuation_attempted: true,
                    tier_results: tiers2,
                },
            )
        } else {
            // Rejected after attenuation retry -> return untouched full-res `image` directly!
            let verdict = QualityVerdict::reject(tiers2, true);
            return (
                image,
                LevelInfo {
                    was_leveled: false,
                    is_monochrome,
                    paper_cast_preserved: has_unleveled_cast,
                    classification: analysis.classification,
                    black_point: 0,
                    white_point: 255,
                    quality_verdict: Some(verdict.clone()),
                },
                verdict,
            );
        }
    };

    // ONLY AFTER Quality Guard acceptance: apply winning plan in-place to full-res `image`
    // (or move `winning_sample` directly if the image was < 500K and guarded at full resolution!).
    let final_image = if guard_is_full_res {
        winning_sample
    } else {
        drop(winning_sample);
        drop(guard_pristine);
        apply_correction_internal(&mut image, &winning_plan, analysis.classification, None);
        image
    };

    let elapsed_us = start_time.elapsed().as_micros();
    info!(
        "[AutoLevels] book_id={} page={} class={:?} black={}->{} white={}->{} cast={:.2} leveled=true attenuated={} time_us={}",
        b_id,
        p_str,
        analysis.classification,
        analysis.black_floor,
        winning_plan.black_point,
        analysis.white_ceiling,
        winning_plan.white_point,
        winning_plan.cast_strength,
        verdict.attenuation_attempted,
        elapsed_us
    );

    (
        final_image,
        LevelInfo {
            was_leveled: true,
            is_monochrome,
            paper_cast_preserved,
            classification: analysis.classification,
            black_point: winning_plan.black_point,
            white_point: winning_plan.white_point,
            quality_verdict: Some(verdict.clone()),
        },
        verdict,
    )
}

// ============================================================================
// Adversarial Regression & Unit Tests (Part A: M1-M4, Part B: T1-T7, Part C)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::upscaler::upscaler::fold_monochrome_post_upscale;
    use image::{ImageBuffer, Luma, Rgb, Rgba};
    use std::sync::Mutex;

    static TEST_RAYON_MUTEX: Mutex<()> = Mutex::new(());

    #[test]
    fn test_m1_page5_digital_spike_and_08_percent_dead_zone_eliminated() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        for &black_ratio in &[0.012f64, 0.008f64] {
            let (w, h) = (1000u32, 1500u32);
            let total = (w * h) as usize;
            let mut raw = vec![248u8; total * 3];

            let n_black = (total as f64 * black_ratio).round() as usize;
            let n_white = (total as f64 * 0.050).round() as usize;
            let n_ink = (total as f64 * 0.138).round() as usize;

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
                let y_val = 42 + ((i % 11) as u8); // 42..=52
                raw[i * 3] = y_val;
                raw[i * 3 + 1] = y_val;
                raw[i * 3 + 2] = y_val;
            }

            let img = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, raw).unwrap());
            let config = AutoLevelsConfig::default();
            let (leveled, info, verdict) = level_and_verify(img, None, &config);

            assert!(info.was_leveled, "M1 (m_zero={}) must be leveled", black_ratio);
            assert_eq!(verdict.outcome, Outcome::Pass);
            assert!(!verdict.attenuation_attempted);
            assert!(
                (40..=52).contains(&info.black_point),
                "Validated lo={} must be in [40, 52] for m_zero={}",
                info.black_point,
                black_ratio
            );

            let lev_rgb = leveled.to_rgb8();
            let sample_i = ((n_black + n_white)..(n_black + n_white + n_ink))
                .find(|&i| 42 + (i % 11) == 45)
                .unwrap();
            let px = lev_rgb.as_raw();
            let y_out = rec709_y(px[sample_i * 3], px[sample_i * 3 + 1], px[sample_i * 3 + 2]);
            assert!(
                y_out <= 8,
                "Washed-out drawing ink (Y=45) must restore cleanly to Y_out <= 8 (got {})",
                y_out
            );
        }
    }

    #[test]
    fn test_m1_sparse_ink_no_midtone_overshoot_and_clean_digital_gray_preserved() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        // 1. Sparse ink: 1.0% digital black + 1.5% drawing ink (Y=25) + 5.0% midtone (Y=75) + 92.5% paper
        let (w, h) = (1000u32, 1000u32);
        let total = (w * h) as usize;
        let mut raw = vec![250u8; total * 3];
        let n_black = total / 100; // 1.0%
        let n_ink = (total * 15) / 1000; // 1.5%
        let n_mid = total / 20; // 5.0%
        for i in 0..n_black {
            raw[i * 3..i * 3 + 3].copy_from_slice(&[0, 0, 0]);
        }
        for i in n_black..(n_black + n_ink) {
            raw[i * 3..i * 3 + 3].copy_from_slice(&[25, 25, 25]);
        }
        for i in (n_black + n_ink)..(n_black + n_ink + n_mid) {
            raw[i * 3..i * 3 + 3].copy_from_slice(&[75, 75, 75]);
        }
        let img = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, raw).unwrap());
        let config = AutoLevelsConfig::default();
        let (_leveled, info, _) = level_and_verify(img, None, &config);
        assert!(
            info.black_point <= 30,
            "Sparse ink black floor must stay near 25, never overshoot to midtone 75 (got {})",
            info.black_point
        );

        // 2. Clean digital manga: 4% authentic black ink (Y=0) + 6% dark-gray fill (Y=45) + 90% white (Y=255)
        let mut clean_raw = vec![255u8; total * 3];
        for i in 0..(total * 4 / 100) {
            clean_raw[i * 3..i * 3 + 3].copy_from_slice(&[0, 0, 0]);
        }
        for i in (total * 4 / 100)..(total * 10 / 100) {
            clean_raw[i * 3..i * 3 + 3].copy_from_slice(&[45, 45, 45]);
        }
        let clean_img = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, clean_raw).unwrap());
        let (_out, clean_info, _) = level_and_verify(clean_img, None, &config);
        assert!(
            !clean_info.was_leveled,
            "Clean digital page with 4% authentic black + 6% dark-gray art must not be altered"
        );
    }

    #[test]
    fn test_m2_localized_margin_yellowing_rgb_rgba_and_mixed_interior_skin_preservation() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let (w, h) = (1000u32, 1500u32);
        let config = AutoLevelsConfig::default();

        // 1. Monochrome M2 (RGB8)
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let in_outer_margin = !(100..900).contains(&x) || !(100..1400).contains(&y);
                let is_shadow = !in_outer_margin && (x + y * 3).is_multiple_of(8);
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
        let (leveled, info, verdict) = level_and_verify(DynamicImage::ImageRgb8(img), None, &config);
        assert_eq!(info.classification, PageClassification::Monochrome);
        assert!(info.was_leveled);
        assert_eq!(verdict.outcome, Outcome::Pass);

        let lev_rgb = leveled.to_rgb8();
        for &(px_x, px_y) in &[(20u32, 20u32), (500, 50), (200, 200)] {
            let p = lev_rgb.get_pixel(px_x, px_y);
            let (_, a, b) = rgb_to_lab_fast(p[0], p[1], p[2]);
            assert!(
                a.hypot(b) <= 2.0,
                "M2 pixel ({},{}) must neutralize to C* <= 2.0 (got {:.2})",
                px_x,
                px_y,
                a.hypot(b)
            );
        }

        // 2. Mixed M2 (RGBA8): yellow outer margins + red logo + interior cream skin [245, 232, 205]
        let mut rgba_img = RgbaImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let in_outer_margin = !(100..900).contains(&x) || !(100..1400).contains(&y);
                let in_logo = (150..280).contains(&x) && (150..380).contains(&y);
                let in_skin = (400..650).contains(&x) && (500..750).contains(&y);
                let is_shadow = !in_outer_margin && !in_logo && !in_skin && (x + y * 3).is_multiple_of(8);
                let px = if in_outer_margin {
                    Rgba([245, 232, 185, 255])
                } else if in_logo {
                    Rgba([210, 25, 25, 255])
                } else if in_skin {
                    Rgba([245, 232, 205, 255])
                } else if is_shadow {
                    Rgba([27, 18, 6, 255])
                } else {
                    Rgba([246, 245, 240, 255])
                };
                rgba_img.put_pixel(x, y, px);
            }
        }
        let (lev_rgba, rgba_info, rgba_verdict) =
            level_and_verify(DynamicImage::ImageRgba8(rgba_img), None, &config);
        assert_eq!(rgba_info.classification, PageClassification::Mixed);
        assert!(rgba_info.was_leveled);
        assert_eq!(rgba_verdict.outcome, Outcome::Pass);

        let out_rgba = lev_rgba.as_rgba8().unwrap();
        // Corner and top edge strip cleaned
        for &(mx, my) in &[(20u32, 20u32), (500, 50)] {
            let p = out_rgba.get_pixel(mx, my);
            let (_, a, b) = rgb_to_lab_fast(p[0], p[1], p[2]);
            assert!(a.hypot(b) <= 2.0, "Mixed RGBA margin ({},{}) C*={:.2} > 2.0", mx, my, a.hypot(b));
        }
        // Red logo untouched
        let logo_p = out_rgba.get_pixel(200, 250);
        assert!(logo_p[0] >= 205 && logo_p[1] <= 30 && logo_p[2] <= 30);
        // Interior cream skin preserved (NOT bleached to gray!)
        let skin_p = out_rgba.get_pixel(500, 600);
        let (_, sa, sb) = rgb_to_lab_fast(skin_p[0], skin_p[1], skin_p[2]);
        assert!(
            sa.hypot(sb) >= 10.0,
            "Interior cream skin on Mixed M2 must NOT be bleached (got {:?}, C*={:.2})",
            skin_p,
            sa.hypot(sb)
        );
    }

    #[test]
    fn test_m3_noisy_scanned_monochrome_and_luma8_folding() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let (w, h) = (1000u32, 1500u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                if (x + y).is_multiple_of(7) {
                    img.put_pixel(x, y, Rgb([5, 4, 3]));
                } else {
                    img.put_pixel(x, y, Rgb([245, 245, 244]));
                }
            }
        }
        let dyn_img = DynamicImage::ImageRgb8(img);
        let config = AutoLevelsConfig::default();
        let analysis = analyze_page(&dyn_img, 1, &config);
        assert_eq!(analysis.classification, PageClassification::Monochrome);

        let (leveled, info, _) = level_and_verify(dyn_img, None, &config);
        assert!(info.is_monochrome);
        assert!(!info.paper_cast_preserved);
        let folded = fold_monochrome_post_upscale(leveled, &info, true);
        assert!(matches!(folded, DynamicImage::ImageLuma8(_)));
    }

    #[test]
    fn test_m4_uniform_aged_scan_and_correct_paper_cast_false_respect() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let (w, h) = (1000u32, 1500u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                if (x + y).is_multiple_of(7) {
                    img.put_pixel(x, y, Rgb([26, 26, 26]));
                } else {
                    img.put_pixel(x, y, Rgb([232, 224, 198]));
                }
            }
        }
        let dyn_img = DynamicImage::ImageRgb8(img);

        // 1. Default correct_paper_cast = true
        let config = AutoLevelsConfig::default();
        let (leveled, info, verdict) = level_and_verify(dyn_img.clone(), None, &config);
        assert!(info.was_leveled);
        assert_eq!(verdict.outcome, Outcome::Pass);
        assert!((24..=28).contains(&info.black_point));
        assert!((220..=228).contains(&info.white_point));
        assert!(!info.paper_cast_preserved);

        let lev_rgb = leveled.to_rgb8();
        let ink_px = lev_rgb.get_pixel(0, 0);
        assert!(rec709_y(ink_px[0], ink_px[1], ink_px[2]) <= 6);
        let paper_px = lev_rgb.get_pixel(1, 0);
        assert!(rec709_y(paper_px[0], paper_px[1], paper_px[2]) >= 250);
        let (_, pa, pb) = rgb_to_lab_fast(paper_px[0], paper_px[1], paper_px[2]);
        assert!(pa.hypot(pb) <= 1.0);

        // 2. correct_paper_cast = false preserves paper tint & sets paper_cast_preserved = true
        let preserve_cfg = AutoLevelsConfig {
            correct_paper_cast: false,
            ..Default::default()
        };
        let (pres_img, pres_info, _) = level_and_verify(dyn_img, None, &preserve_cfg);
        assert!(pres_info.paper_cast_preserved);
        let pres_paper = pres_img.as_rgb8().unwrap().get_pixel(1, 0);
        assert!(
            pres_paper[0] > pres_paper[2],
            "Paper tint must be preserved when correct_paper_cast=false (got {:?})",
            pres_paper
        );
    }

    #[test]
    fn test_arbitrary_hue_casts_blue_green_pink_on_mixed() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let cases = [
            ("blue", [225u8, 235, 248], [210u8, 25, 25]),
            ("green", [225u8, 245, 230], [210u8, 25, 25]),
            ("pink", [248u8, 230, 240], [25u8, 90, 220]),
        ];
        let config = AutoLevelsConfig::default();
        for (name, paper_rgb, logo_rgb) in cases {
            let (w, h) = (600u32, 600u32);
            let mut img = RgbImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let in_logo = (50..150).contains(&x) && (50..160).contains(&y);
                    let is_ink = !in_logo && (x + y).is_multiple_of(12);
                    let px = if in_logo {
                        Rgb(logo_rgb)
                    } else if is_ink {
                        Rgb([20, 20, 20])
                    } else {
                        Rgb(paper_rgb)
                    };
                    img.put_pixel(x, y, px);
                }
            }
            let (leveled, info, verdict) = level_and_verify(DynamicImage::ImageRgb8(img), None, &config);
            assert_eq!(info.classification, PageClassification::Mixed, "case {}", name);
            assert!(info.was_leveled, "case {}", name);
            assert_eq!(verdict.outcome, Outcome::Pass, "case {}", name);

            let out = leveled.as_rgb8().unwrap();
            let pp = out.get_pixel(400, 400);
            let (_, pa, pb) = rgb_to_lab_fast(pp[0], pp[1], pp[2]);
            assert!(
                pa.hypot(pb) <= 2.0,
                "{} cast paper must neutralize to C* <= 2.0 (got {:.2}, {:?})",
                name,
                pa.hypot(pb),
                pp
            );
            let lp = out.get_pixel(100, 100);
            assert_eq!(lp.0, logo_rgb, "{} cast logo must be 100% preserved", name);
        }
    }

    #[test]
    fn test_t1_to_t7_adversarial_traps_zero_false_positives() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let config = AutoLevelsConfig::default();

        // T1: Pristine digital release + light-gray screentone counterexample
        {
            let (w, h) = (1000u32, 1500u32);
            let mut img = RgbImage::new(w, h);
            let total = (w * h) as usize;
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) as usize;
                    let v = if idx < total * 15 / 100 {
                        255u8 // 15% white bubbles
                    } else if idx < total * 50 / 100 {
                        200u8 // 35% light-gray screentone
                    } else {
                        (idx % 60) as u8 // 50% dark art starting at 0
                    };
                    img.put_pixel(x, y, Rgb([v, v, v]));
                }
            }
            let orig = DynamicImage::ImageRgb8(img);
            let orig_raw = orig.as_rgb8().unwrap().as_raw().clone();
            let (leveled, info, _) = level_and_verify(orig, None, &config);
            assert!(!info.was_leveled, "T1 clean page with gray screentone must be vetoed");
            assert_eq!(leveled.as_rgb8().unwrap().as_raw(), &orig_raw);
        }

        // T2: Full-color cover (even < 500K pixels!)
        {
            let (w, h) = (500u32, 500u32);
            let mut img = RgbImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let px = if y < 120 { Rgb([40, 120, 220]) } else { Rgb([240, 240, 240]) };
                    img.put_pixel(x, y, px);
                }
            }
            let orig = DynamicImage::ImageRgb8(img);
            let orig_raw = orig.as_rgb8().unwrap().as_raw().clone();
            let (leveled, info, _) = level_and_verify(orig, None, &config);
            assert_eq!(info.classification, PageClassification::Color);
            assert!(!info.was_leveled);
            assert_eq!(leveled.as_rgb8().unwrap().as_raw(), &orig_raw);
        }

        // T3: Pastel sky & watercolor skin
        {
            let (w, h) = (1000u32, 1500u32);
            let mut img = RgbImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) as usize;
                    let px = if idx < 150_000 {
                        Rgb([220, 235, 245])
                    } else if idx < 300_000 {
                        Rgb([30, 30, 30])
                    } else {
                        Rgb([250, 250, 250])
                    };
                    img.put_pixel(x, y, px);
                }
            }
            let (leveled, info, verdict) = level_and_verify(DynamicImage::ImageRgb8(img), None, &config);
            assert_eq!(info.classification, PageClassification::Mixed);
            assert!(info.was_leveled);
            assert_eq!(verdict.outcome, Outcome::Pass);
            let lev_rgb = leveled.to_rgb8();
            let sky = lev_rgb.get_pixel(0, 0);
            assert!(sky[2] as i32 - sky[0] as i32 >= 15);
            let (_, a0, b0) = rgb_to_lab_fast(220, 235, 245);
            let (_, a1, b1) = rgb_to_lab_fast(sky[0], sky[1], sky[2]);
            assert!((a0 - a1).hypot(b0 - b1) <= 1.5);
            let ink = lev_rgb.get_pixel(0, 151);
            assert!(rec709_y(ink[0], ink[1], ink[2]) <= 8);
        }

        // T4: High-key fog
        {
            let (w, h) = (1000u32, 1500u32);
            let mut img = RgbImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let idx = (y * w + x) as usize;
                    let v = if idx < 22_500 { 0u8 } else { 210 + ((idx % 41) as u8) };
                    img.put_pixel(x, y, Rgb([v, v, v]));
                }
            }
            let (leveled, info, _) = level_and_verify(DynamicImage::ImageRgb8(img), None, &config);
            assert_eq!(info.black_point, 0);
            let (luma, _) = extract_luma_and_alpha(&leveled);
            let mean_y = luma.iter().map(|&v| v as f64).sum::<f64>() / luma.len() as f64;
            assert!(mean_y >= 215.0);
        }

        // T5: Low-key nighttime splash (Luma8 included!)
        {
            let (w, h) = (800u32, 800u32);
            let luma_img = ImageBuffer::from_fn(w, h, |x, y| Luma([((x + y) % 111) as u8]));
            let (_leveled, info, _) = level_and_verify(DynamicImage::ImageLuma8(luma_img), None, &config);
            assert_eq!(info.white_point, 255);
        }

        // T6: Microscopic red logo on dirty paper
        {
            let (w, h) = (1000u32, 1500u32);
            let mut img = RgbImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let in_logo = (100..250).contains(&x) && (100..400).contains(&y);
                    let is_ink = !in_logo && (x + y).is_multiple_of(15);
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
            let (leveled, info, verdict) = level_and_verify(DynamicImage::ImageRgb8(img), None, &config);
            assert_eq!(info.classification, PageClassification::Mixed);
            assert!(info.was_leveled);
            assert_eq!(verdict.outcome, Outcome::Pass);
            let out = leveled.as_rgb8().unwrap();
            let logo = out.get_pixel(150, 200);
            assert!(logo[0] >= 205 && logo[1] <= 30 && logo[2] <= 30);
            let paper = out.get_pixel(500, 500);
            let (_, pa, pb) = rgb_to_lab_fast(paper[0], paper[1], paper[2]);
            assert!(pa.hypot(pb) <= 1.5);
        }

        // T7: Intentional warm sepia WITH pure-white digital speech bubbles in corner cell!
        {
            let (w, h) = (1000u32, 1500u32);
            let mut img = RgbImage::new(w, h);
            for y in 0..h {
                for x in 0..w {
                    let in_white_bubble = x < 240 && y < 360; // >90% of cell (0,0) is pure white!
                    let px = if in_white_bubble {
                        Rgb([255, 255, 255])
                    } else if (x + y).is_multiple_of(8) {
                        Rgb([40, 30, 15])
                    } else {
                        Rgb([245, 230, 200])
                    };
                    img.put_pixel(x, y, px);
                }
            }
            let orig = DynamicImage::ImageRgb8(img);
            let orig_raw = orig.as_rgb8().unwrap().as_raw().clone();
            let (leveled, info, _) = level_and_verify(orig, None, &config);
            assert!(
                !info.was_leveled,
                "T7 Sepia with pure-white speech bubble cell must remain strictly vetoed"
            );
            assert_eq!(leveled.as_rgb8().unwrap().as_raw(), &orig_raw);
        }
    }

    #[test]
    fn test_fritsch_carlson_zero_inversion_nan_safety_and_gamma() {
        // NaN / Inf safety -> identity
        let nan_lut = build_monotonic_tone_lut(f32::NAN, 255.0);
        let inf_lut = build_monotonic_tone_lut(0.0, f32::INFINITY);
        for i in 0..256 {
            assert_eq!(nan_lut[i], i as u8);
            assert_eq!(inf_lut[i], i as u8);
        }

        // Zero direction inversion on mild stretches!
        let lut_mild_lo = build_monotonic_tone_lut(5.0, 255.0);
        assert!(lut_mild_lo[5] <= 5, "lo=5 must darken, not lighten to 8 (got {})", lut_mild_lo[5]);

        let lut_mild_hi = build_monotonic_tone_lut(0.0, 252.0);
        assert!(lut_mild_hi[252] >= 252, "hi=252 must brighten, not darken to 248 (got {})", lut_mild_hi[252]);

        let lut_both = build_monotonic_tone_lut(6.0, 250.0);
        assert!(lut_both[250] - lut_both[6] > 244, "Range (6,250) must expand, never shrink");

        // Monotonicity across extreme (lo, hi, gamma) including hi < 128
        for &gamma in &[0.5f32, 1.0, 1.5, 2.2] {
            for &hi_t in &[90.0f32, 110.0, 128.0, 224.0, 255.0] {
                for &lo_t in &[0.0f32, 5.0, 26.0, 45.0, 70.0] {
                    if hi_t - lo_t >= 15.0 {
                        let lut = build_monotonic_tone_lut_gamma(lo_t, hi_t, gamma);
                        for i in 0..255 {
                            assert!(lut[i + 1] >= lut[i]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_preflight_guard_rollback_preserves_full_resolution_without_mosaic() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        // Force Tier 2 SSIM rejection on a >= 2MP page (2000x1200 = 2.4MP) and verify the returned
        // image is bit-exact full resolution (NEVER a 1/16th 32x32 tile mosaic!).
        let (w, h) = (2000u32, 1200u32);
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y).is_multiple_of(7) { 30u8 } else { 225u8 };
                img.put_pixel(x, y, Rgb([v, v, v]));
            }
        }
        let orig = DynamicImage::ImageRgb8(img);
        let orig_raw = orig.as_rgb8().unwrap().as_raw().clone();

        let (out, info, verdict) =
            level_and_verify_with_ssim_override(orig, &AutoLevelsConfig::default(), 1.01);

        assert!(!info.was_leveled);
        assert_eq!(verdict.outcome, Outcome::Reject);
        assert!(verdict.attenuation_attempted);
        assert_eq!(out.dimensions(), (w, h), "Rollback must NEVER return a shrunk tile mosaic!");
        assert_eq!(out.as_rgb8().unwrap().as_raw(), &orig_raw, "Rollback must be bit-exact original");
    }

    #[test]
    fn test_6mp_latency_budget_mono_mixed_luma8() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let _ = &*FAST_LAB_TABLES;
        let (w, h) = (2000u32, 3000u32);
        let total = (w * h) as usize;
        let config = AutoLevelsConfig::default();

        // 1. Vetoed 6MP Color page (clone BEFORE Instant::now() and drop AFTER elapsed())
        let mut color_raw = vec![240u8; total * 3];
        for i in 0..(total / 3) {
            color_raw[i * 3..i * 3 + 3].copy_from_slice(&[35, 110, 225]);
        }
        let color_img = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, color_raw).unwrap());
        let mut veto_ms = f64::MAX;
        for _ in 0..5 {
            let input = color_img.clone();
            let t0 = Instant::now();
            let (out, info, _) = level_and_verify(input, None, &config);
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            assert!(!info.was_leveled);
            drop(out);
            veto_ms = veto_ms.min(dt);
        }

        // 2. 6MP Monochrome M1 restoration
        let mut m1_raw = vec![248u8; total * 3];
        let n_black = (total as f64 * 0.012) as usize;
        let n_ink = (total as f64 * 0.138) as usize;
        for i in 0..n_black {
            m1_raw[i * 3..i * 3 + 3].copy_from_slice(&[0, 0, 0]);
        }
        for i in n_black..(n_black + n_ink) {
            let v = 42 + ((i % 11) as u8);
            m1_raw[i * 3..i * 3 + 3].copy_from_slice(&[v, v, v]);
        }
        let m1_img = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, m1_raw).unwrap());
        let mut mono_ms = f64::MAX;
        for _ in 0..3 {
            let input = m1_img.clone();
            let t1 = Instant::now();
            let (out, info, verdict) = level_and_verify(input, None, &config);
            let dt = t1.elapsed().as_secs_f64() * 1000.0;
            assert!(info.was_leveled && verdict.outcome == Outcome::Pass);
            drop(out);
            mono_ms = mono_ms.min(dt);
        }

        // 3. 6MP Mixed restoration (dirty paper + 3% red logo)
        let mut mixed_raw = vec![0u8; total * 3];
        for y in 0..h as usize {
            let row_off = y * (w as usize) * 3;
            let in_logo_y = (100..500).contains(&y);
            for x in 0..w as usize {
                let idx = row_off + x * 3;
                if in_logo_y && (100..550).contains(&x) {
                    mixed_raw[idx..idx + 3].copy_from_slice(&[210, 25, 25]);
                } else if (x + y).is_multiple_of(15) {
                    mixed_raw[idx..idx + 3].copy_from_slice(&[20, 20, 20]);
                } else {
                    mixed_raw[idx..idx + 3].copy_from_slice(&[245, 243, 234]);
                }
            }
        }
        let mixed_img = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, mixed_raw).unwrap());
        let mut mixed_ms = f64::MAX;
        for _ in 0..3 {
            let input = mixed_img.clone();
            let t_m = Instant::now();
            let (out, info, verdict) = level_and_verify(input, None, &config);
            let dt = t_m.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(info.classification, PageClassification::Mixed);
            assert!(info.was_leveled && verdict.outcome == Outcome::Pass);
            drop(out);
            mixed_ms = mixed_ms.min(dt);
        }

        // 4. 6MP Luma8 restoration
        let mut luma_raw = vec![228u8; total];
        for i in 0..n_ink {
            luma_raw[i] = 28;
        }
        let luma_img = DynamicImage::ImageLuma8(GrayImage::from_raw(w, h, luma_raw).unwrap());
        let mut luma_ms = f64::MAX;
        for _ in 0..3 {
            let input = luma_img.clone();
            let t2 = Instant::now();
            let (out, info, verdict) = level_and_verify(input, None, &config);
            let dt = t2.elapsed().as_secs_f64() * 1000.0;
            assert!(info.was_leveled && verdict.outcome == Outcome::Pass);
            drop(out);
            luma_ms = luma_ms.min(dt);
        }

        println!(
            "[Benchmark 6MP] Veto: {:.3}ms | Mono: {:.3}ms | Mixed: {:.3}ms | Luma8: {:.3}ms",
            veto_ms, mono_ms, mixed_ms, luma_ms
        );
        #[cfg(not(debug_assertions))]
        {
            assert!(veto_ms < 1.5, "6MP veto {:.3}ms >= 1.5ms", veto_ms);
            assert!(mono_ms < 15.0, "6MP mono {:.3}ms >= 15.0ms", mono_ms);
            assert!(mixed_ms < 15.0, "6MP mixed {:.3}ms >= 15.0ms", mixed_ms);
            assert!(luma_ms < 15.0, "6MP luma8 {:.3}ms >= 15.0ms", luma_ms);
        }
    }

    #[test]
    fn test_16bit_format_parity_and_early_bypass_grayscale_detection() {
        let config = AutoLevelsConfig::default();

        // 16-bit Luma16 with elevated black floor (Y=30) & dull paper (Y=220) must genuinely level (not no-op!)
        let (w, h) = (450u32, 450u32);
        let luma16 = ImageBuffer::from_fn(w, h, |x, y| {
            let v8: u16 = if (x + y).is_multiple_of(8) { 30 } else { 220 };
            Luma([v8 * 257])
        });
        let (leveled, info, verdict) = level_and_verify(DynamicImage::ImageLuma16(luma16), None, &config);
        assert!(info.was_leveled, "16-bit Luma16 must be genuinely leveled, never silent no-op");
        assert_eq!(verdict.outcome, Outcome::Pass);
        let out_luma = leveled.as_luma8().expect("16-bit Luma16 normalizes cleanly to Luma8");
        assert!(out_luma.get_pixel(0, 0)[0] <= 8);
        assert!(out_luma.get_pixel(1, 0)[0] >= 250);

        // Early bypass (< 400px or disabled) preserves accurate is_monochrome classification
        let small_mono = DynamicImage::ImageRgb8(RgbImage::from_pixel(200, 200, Rgb([120, 120, 120])));
        let (_, small_info, _) = level_and_verify(small_mono, None, &config);
        assert!(!small_info.was_leveled);
        assert!(small_info.is_monochrome);

        let small_color = DynamicImage::ImageRgb8(RgbImage::from_pixel(200, 200, Rgb([210, 30, 30])));
        let (_, color_info, _) = level_and_verify(small_color, None, &config);
        assert!(!color_info.was_leveled);
        assert!(!color_info.is_monochrome);
    }

    #[test]
    fn test_quality_guard_genuine_rejections_tier1_tier2_tier3() {
        let (w, h) = (128u32, 128u32);

        // Tier 1: Inverted image must be rejected (zero tautology!)
        let orig_grad = DynamicImage::ImageLuma8(ImageBuffer::from_fn(w, h, |x, _| Luma([(x * 2).min(255) as u8])));
        let inv_grad = DynamicImage::ImageLuma8(ImageBuffer::from_fn(w, h, |x, _| Luma([255u8.saturating_sub((x * 2).min(255) as u8)])));
        let t1 = quality_tier_1(&orig_grad, &inv_grad, &CorrectionPlan::default());
        assert_eq!(t1.outcome, Outcome::Reject, "Tier 1 must reject inverted luminance");

        // Tier 2: Bleaching pastel blue sky [220,235,245] to gray [234,234,234] must be rejected (never exempted!)
        let sky_orig = DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, Rgb([220, 235, 245])));
        let sky_bleached = DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, Rgb([234, 234, 234])));
        let cast_plan = CorrectionPlan {
            cast_strength: 1.0,
            cast_field: PaperCastField {
                active: true,
                global_a: -0.5,
                global_b: 8.0,
                global_c: 8.0,
                dom_a: -0.5,
                dom_b: 8.0,
                dom_u_a: -0.06,
                dom_u_b: 0.998,
                c_max: 8.0,
                ..PaperCastField::default()
            },
            ..CorrectionPlan::default()
        };
        let t2 = quality_tier_2(&sky_orig, &sky_bleached, PageClassification::Mixed, &cast_plan);
        assert_eq!(t2.outcome, Outcome::Reject, "Tier 2 must reject bleached pastel sky even when paper cast is active");

        // Tier 3: Crushed midtone checkerboard must reject, while speech-bubble-on-aged-paper block is exempted
        let checker_orig = DynamicImage::ImageLuma8(ImageBuffer::from_fn(w, h, |x, y| {
            Luma([if (x + y).is_multiple_of(2) { 90 } else { 150 }])
        }));
        let checker_crushed = DynamicImage::ImageLuma8(ImageBuffer::from_fn(w, h, |x, y| {
            Luma([if (x + y).is_multiple_of(2) { 118 } else { 122 }])
        }));
        let t3_fail = quality_tier_3(&checker_orig, &checker_crushed, &CorrectionPlan::default());
        assert_eq!(t3_fail.outcome, Outcome::Reject, "Tier 3 must reject collapsed midtone contrast");

        let bubble_orig = DynamicImage::ImageLuma8(ImageBuffer::from_fn(w, h, |x, _| {
            Luma([if x % 32 < 16 { 224 } else { 255 }])
        }));
        let bubble_whitened = DynamicImage::ImageLuma8(ImageBuffer::from_fn(w, h, |x, _| {
            Luma([if x % 32 < 16 { 253 } else { 255 }])
        }));
        let white_plan = CorrectionPlan {
            white_point: 224,
            ..CorrectionPlan::default()
        };
        let t3_pass = quality_tier_3(&bubble_orig, &bubble_whitened, &white_plan);
        assert_eq!(t3_pass.outcome, Outcome::Pass, "Tier 3 must exempt intentional aged-paper/speech-bubble whitening");
    }

    #[test]
    fn test_scan_accent_row_alternating_accent_and_yellow_paper() {
        let tables = &*FAST_LAB_TABLES;
        let mut row_rgb = vec![0u8; 128 * 3];
        for x in 0..128 {
            let px = if x % 2 == 0 {
                [210u8, 25, 25]
            } else {
                [245u8, 232, 185]
            };
            row_rgb[x * 3..x * 3 + 3].copy_from_slice(&px);
        }
        let mut mask = [0u64; 2];
        let any = scan_accent_row::<3>(&row_rgb, &mut mask, true, 0.08, 0.997, 30.0, tables);
        assert_eq!(any, 1);
        for x in 0..128usize {
            let bit = (mask[x >> 6] >> (x & 63)) & 1;
            let expected = u64::from(x.is_multiple_of(2));
            assert_eq!(bit, expected, "unexpected accent bit at x={}", x);
        }

        // Verify Rgba8: transparent [210, 25, 25, 0] on even pixels must remain 0
        // even when prev_key1 (x=2) or prev_key0 (x=4) holds [210, 25, 25] with prev_acc=true.
        let mut row_rgba = vec![0u8; 64 * 4];
        for x in 0..64 {
            let px = match x {
                0 | 3 => [210u8, 25, 25, 255],
                1 => [245u8, 232, 185, 255],
                2 | 4 => [210u8, 25, 25, 0],
                _ => [245u8, 232, 185, 255],
            };
            row_rgba[x * 4..x * 4 + 4].copy_from_slice(&px);
        }
        let mut mask_rgba = [0u64; 1];
        let any_rgba = scan_accent_row::<4>(&row_rgba, &mut mask_rgba, true, 0.08, 0.997, 30.0, tables);
        assert_eq!(any_rgba, 1);
        assert_eq!(mask_rgba[0] & 1, 1);
        assert_eq!((mask_rgba[0] >> 1) & 1, 0);
        assert_eq!((mask_rgba[0] >> 2) & 1, 0, "transparent copy hitting prev_key1 must not be masked");
        assert_eq!((mask_rgba[0] >> 3) & 1, 1);
        assert_eq!((mask_rgba[0] >> 4) & 1, 0, "transparent copy hitting prev_key0 must not be masked");
    }

    #[test]
    fn test_luma8_low_key_single_speck_does_not_defeat_paper_peak() {
        let (w, h) = (2000u32, 2000u32);
        let mut buf = ImageBuffer::from_pixel(w, h, Luma([40u8]));
        for y in 1880..h {
            for x in 0..w {
                buf.put_pixel(x, y, Luma([150u8]));
            }
        }
        buf.put_pixel(0, 0, Luma([200u8]));
        buf.put_pixel(17, 0, Luma([255u8]));

        let analysis = analyze_page_with_min_white(&DynamicImage::ImageLuma8(buf), 140);
        assert_eq!(
            analysis.white_ceiling, 255,
            "low-key Luma8 scene with isolated bright specks must remain vetoed"
        );
        assert!(
            !analysis
                .defects
                .iter()
                .any(|d| matches!(d, DefectType::DullHighlights { .. })),
            "isolated bright specks must not trigger DullHighlights on low-key Luma8 scenes: {:?}",
            analysis.defects
        );
    }

    #[test]
    fn test_neutral_mono_rgb_still_folds_luma8_when_correct_paper_cast_false() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let (w, h) = (500u32, 500u32);
        let img = ImageBuffer::from_fn(w, h, |x, _y| {
            let phase = x % 40;
            if phase < 8 {
                Rgb([30u8, 30u8, 30u8])
            } else if (12..16).contains(&phase) {
                Rgb([128u8, 128u8, 128u8])
            } else {
                Rgb([225u8, 225u8, 225u8])
            }
        });
        let config = AutoLevelsConfig {
            correct_paper_cast: false,
            ..Default::default()
        };
        let dyn_img = DynamicImage::ImageRgb8(img);
        let analysis = analyze_page(&dyn_img, 0, &config);
        assert_eq!(analysis.classification, PageClassification::Monochrome);
        assert!(!analysis.has_paper_cast);
        assert!(!analysis.has_muddy_shadow_cast);

        let (leveled, info, _) = level_and_verify(dyn_img, None, &config);
        assert!(info.is_monochrome);
        assert!(info.was_leveled);
        assert!(
            !info.paper_cast_preserved,
            "neutral grayscale RGB page must not claim a preserved paper cast"
        );
        let folded = fold_monochrome_post_upscale(leveled, &info, true);
        assert!(
            matches!(folded, DynamicImage::ImageLuma8(_)),
            "neutral monochrome RGB page must fold to Luma8 even when correct_paper_cast == false"
        );
    }

    #[test]
    fn test_antialiased_typeset_text_compact_ink_shelf_and_dark_ink_guards() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let total = 100_000u32;

        // 1. Real-world Page 32 archetype: crisp digital typeset text (0.6% Y=0..2) +
        // anti-aliased text fringe (0.7% across Y=3..9 that drags shadow_p2 below 8!) +
        // compact washed-out black ink floor peaking at Y=21 (14% mass) + midtones + Y=255 paper.
        let mut hist_p32 = [0u32; 256];
        hist_p32[0] = 400;
        hist_p32[1] = 100;
        hist_p32[2] = 100;
        for bin in 3..=9 {
            hist_p32[bin] = 100; // 0.7% AA fringe -> pulls shadow_p2 down to bin 7 (< 8)!
        }
        hist_p32[20] = 1_200;
        hist_p32[21] = 11_500;
        hist_p32[22] = 1_200;
        for bin in 70..=150 {
            hist_p32[bin] = 120; // midtone screentones
        }
        hist_p32[255] = total - hist_p32.iter().sum::<u32>();
        let det = detect_elevated_black_floor(&hist_p32, PageClassification::Monochrome)
            .expect("Compact Y=21 washed-out ink shelf with AA text fringe must be detected");
        assert!(
            matches!(
                det,
                DefectType::ElevatedBlackFloor {
                    floor_value: 20..=22,
                    is_bimodal: true
                }
            ),
            "Expected floor_value in 20..=22, got {:?}",
            det
        );

        // 2. End-to-end 2D image verification (ensures Quality Guard passes without attenuation)
        let (w, h) = (720u32, 1024u32);
        let img = ImageBuffer::from_fn(w, h, |x, y| {
            let v = if y < 60 && x < 260 && (x % 4 == 0) {
                0u8 // crisp digital black text (~0.6%)
            } else if y < 60 && x < 260 && (x % 4 == 1) {
                (3 + ((x + y) % 7)) as u8 // anti-aliased fringe in Y=3..9 (~0.6%)
            } else if (300..520).contains(&y) && (20..520).contains(&x) {
                21u8 // large washed-out black panel background (~15%)
            } else if (600..820).contains(&y) && (x + y) % 3 == 0 {
                130u8 // screentone midtones
            } else {
                255u8
            };
            Rgb([v, v, v])
        });
        let config = AutoLevelsConfig::default();
        let (leveled, info, verdict) = level_and_verify(DynamicImage::ImageRgb8(img), None, &config);
        assert!(info.was_leveled && verdict.outcome == Outcome::Pass);
        assert!(!verdict.attenuation_attempted);
        assert!(
            (20..=22).contains(&info.black_point),
            "Expected black_point in 20..=22 on Page-32 archetype, got {}",
            info.black_point
        );
        let bg_px = leveled.to_rgb8().get_pixel(150, 400).0;
        assert!(
            bg_px[0] <= 4 && bg_px[1] <= 4 && bg_px[2] <= 4,
            "Washed-out Y=21 panel background must restore to near-black (<=4), got {:?}",
            bg_px
        );

        // 3. Adversarial trap: 4.0% authentic dark ink at Y=4 + 15% dark-gray fill at Y=25
        // Must NOT jump over authentic dark ink at Y=4 to crush Y=25!
        let mut hist_dark_ink = [0u32; 256];
        hist_dark_ink[4] = 4_000;
        hist_dark_ink[25] = 15_000;
        hist_dark_ink[255] = total - hist_dark_ink[4] - hist_dark_ink[25];
        if let Some(DefectType::ElevatedBlackFloor { floor_value, .. }) =
            detect_elevated_black_floor(&hist_dark_ink, PageClassification::Monochrome)
        {
            assert!(
                floor_value <= 6,
                "Must not jump over 4% dark ink at Y=4 to crush Y=25 tone (got {})",
                floor_value
            );
        }

        // 4. Dual-peak shadow trap: 10% washed ink at Y=12 + 12% dark-gray tone at Y=30
        // Must lock onto the first ink peak (Y=12), never skip to the taller Y=30 tone!
        let mut hist_dual = [0u32; 256];
        hist_dual[0] = 500;
        hist_dual[12] = 10_000;
        hist_dual[30] = 12_000;
        hist_dual[255] = total - hist_dual[0] - hist_dual[12] - hist_dual[30];
        let det_dual = detect_elevated_black_floor(&hist_dual, PageClassification::Monochrome)
            .expect("First washed ink peak at Y=12 must be detected");
        assert!(
            matches!(
                det_dual,
                DefectType::ElevatedBlackFloor {
                    floor_value: 11..=13,
                    ..
                }
            ),
            "Must lock onto first ink peak Y=12, not jump to taller Y=30 tone (got {:?})",
            det_dual
        );
    }

    #[test]
    fn test_white_paper_dominant_muddy_shadow_ink_classifies_mixed_not_color() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let config = AutoLevelsConfig::default();

        // 1. White paper ([252, 252, 250], C* < 3.5 -> no global paper cast) +
        // 22.8% warm/muddy lifted dark border/background ([27, 17, 7], Y=17, L*=5.9, C*=6.7) +
        // 0.35% colored title/credit badge ([215, 35, 35], C* >= 20).
        // Must NOT trigger the Full-Color Veto! Must classify Mixed, level black_floor ~ 17,
        // neutralize muddy shadow tint, preserve colored logo untouched, and pass Quality Guard.
        let (w, h) = (720u32, 1024u32);
        let img = ImageBuffer::from_fn(w, h, |x, y| {
            if (20..65).contains(&y) && (80..140).contains(&x) {
                Rgb([215u8, 35u8, 35u8]) // ~0.35% red title logo
            } else if y >= 790 || (y >= 420 && !(40..680).contains(&x)) {
                Rgb([27u8, 17u8, 7u8]) // ~25% warm/muddy lifted dark frame
            } else if (200..400).contains(&y) && (x + y) % 5 == 0 {
                Rgb([135u8, 135u8, 135u8]) // neutral halftone screentones
            } else {
                Rgb([252u8, 252u8, 250u8]) // clean neutral white paper
            }
        });

        let (leveled, info, verdict) =
            level_and_verify(DynamicImage::ImageRgb8(img), None, &config);
        assert_eq!(
            info.classification,
            PageClassification::Mixed,
            "White paper + dominant muddy shadow manga page with small logo must classify Mixed"
        );
        assert!(info.was_leveled && verdict.outcome == Outcome::Pass);
        assert!(!verdict.attenuation_attempted);
        assert!(
            (15..=19).contains(&info.black_point),
            "Expected black_point near 17, got {}",
            info.black_point
        );

        let rgb = leveled.to_rgb8();
        let border_px = rgb.get_pixel(20, 880).0;
        let (_, bd_a, bd_b) = rgb_to_lab_fast(border_px[0], border_px[1], border_px[2]);
        assert!(
            border_px[0] <= 4 && border_px[1] <= 4 && border_px[2] <= 4,
            "Muddy lifted dark frame [27,17,7] must restore to <= 4, got {:?}",
            border_px
        );
        assert!(
            bd_a.hypot(bd_b) <= 2.0,
            "Muddy shadow chroma must neutralize (C*={:.2}, pixel={:?})",
            bd_a.hypot(bd_b),
            border_px
        );
        let logo_px = rgb.get_pixel(110, 42).0;
        assert_eq!(logo_px, [215, 35, 35], "Red logo accent must stay bit-exact");

        // 2. Moody dark full-color illustration ([20, 35, 95] deep blue darks covering 25%)
        // MUST still veto unconditionally as PageClassification::Color!
        let dark_color = ImageBuffer::from_fn(w, h, |_, y| {
            if y >= 760 {
                Rgb([20u8, 35u8, 95u8])
            } else {
                Rgb([252u8, 252u8, 250u8])
            }
        });
        let (_out_color, color_info, _) =
            level_and_verify(DynamicImage::ImageRgb8(dark_color), None, &config);
        assert_eq!(color_info.classification, PageClassification::Color);
        assert!(!color_info.was_leveled);
    }

    #[test]
    fn test_compact_ink_shelf_adversarial_counterexamples() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let total = 100_000u32;

        // 1.1a: m_zero == 0, 10% dark ink at Y=7 + 15% dark-gray fill at Y=25.
        // Must NOT jump over Y=7 ink to crush Y=25 tone (floor_value <= 8).
        let mut hist_1_1a = [0u32; 256];
        hist_1_1a[7] = 10_000;
        hist_1_1a[25] = 15_000;
        hist_1_1a[255] = total - 25_000;
        if let Some(DefectType::ElevatedBlackFloor { floor_value, .. }) =
            detect_elevated_black_floor(&hist_1_1a, PageClassification::Monochrome)
        {
            assert!(
                floor_value <= 8,
                "1.1a: must not jump over 10% dark ink at Y=7 to Y=25 (got {})",
                floor_value
            );
        }

        // 1.1b: m_zero = 0.6% (Y=0) + 22% broad charcoal shading + 0.1% at Y=26 + 5% coat at Y=30.
        // (i) Authentic charcoal starting at Y=3..=24 must veto completely (None).
        let mut hist_1_1b_auth = [0u32; 256];
        hist_1_1b_auth[0] = 600;
        for bin in &mut hist_1_1b_auth[3..=24] {
            *bin = 1_000; // 22 * 1000 = 22,000 (22% continuous dark charcoal)
        }
        hist_1_1b_auth[26] = 100;
        hist_1_1b_auth[30] = 5_000;
        hist_1_1b_auth[255] = total - hist_1_1b_auth.iter().sum::<u32>();
        assert!(
            detect_elevated_black_floor(&hist_1_1b_auth, PageClassification::Monochrome).is_none(),
            "1.1b(i): broad authentic charcoal shading Y=3..24 before Y=30 coat must veto"
        );

        // (ii) Lifted broad charcoal Y=10..=24 + 0.1% at Y=26 + 5% coat at Y=30:
        // compact shelf detector MUST veto Y=30, and unified detector must NEVER jump to Y=30!
        let mut hist_1_1b_lifted = [0u32; 256];
        hist_1_1b_lifted[0] = 600;
        for bin in &mut hist_1_1b_lifted[10..=24] {
            *bin = 1_467; // 15 * 1467 = 22,005 (~22%)
        }
        hist_1_1b_lifted[26] = 100;
        hist_1_1b_lifted[30] = 5_000;
        hist_1_1b_lifted[255] = total - hist_1_1b_lifted.iter().sum::<u32>();
        let smooth_1_1b = smooth_histogram_binomial_5(&hist_1_1b_lifted);
        assert!(
            detect_compact_ink_shelf(&hist_1_1b_lifted, &smooth_1_1b, total as u64, 255.0)
                .is_none(),
            "1.1b(ii): compact ink shelf must NOT jump over Y=10..24 charcoal mound to Y=30"
        );
        if let Some(DefectType::ElevatedBlackFloor { floor_value, .. }) =
            detect_elevated_black_floor(&hist_1_1b_lifted, PageClassification::Monochrome)
        {
            assert!(
                floor_value <= 12,
                "1.1b(ii): must stay at the Y=10 charcoal foot, never jump to Y=30 (got {})",
                floor_value
            );
        }

        // 1.1c: Flat dark-art plateau before Y=30 coat (must not slide trough_idx into an earlier zero bin).
        // (i) Authentic flat dark plateau Y=3..=27 (3.75% total) + 0.6% Y=0 + 6% Y=30 coat -> strict veto.
        let mut hist_1_1c_auth = [0u32; 256];
        hist_1_1c_auth[0] = 600;
        for bin in &mut hist_1_1c_auth[3..=27] {
            *bin = 150; // 25 * 150 = 3,750 (3.75% flat plateau)
        }
        hist_1_1c_auth[30] = 6_000;
        hist_1_1c_auth[255] = total - hist_1_1c_auth.iter().sum::<u32>();
        assert!(
            detect_elevated_black_floor(&hist_1_1c_auth, PageClassification::Monochrome).is_none(),
            "1.1c(i): authentic flat 3.75% dark plateau Y=3..27 before Y=30 coat must veto"
        );

        // (ii) Lifted flat plateau Y=10..=27 (3.6% total) with zero bins at Y=6..=9 + 0.6% Y=0 + 6% Y=30:
        // compact shelf detector MUST veto Y=30 (no sliding across flat plateau to Y=8 zero bin!).
        let mut hist_1_1c_lifted = [0u32; 256];
        hist_1_1c_lifted[0] = 600;
        for bin in &mut hist_1_1c_lifted[10..=27] {
            *bin = 200; // 18 * 200 = 3,600 (3.6% flat plateau)
        }
        hist_1_1c_lifted[30] = 6_000;
        hist_1_1c_lifted[255] = total - hist_1_1c_lifted.iter().sum::<u32>();
        let smooth_1_1c = smooth_histogram_binomial_5(&hist_1_1c_lifted);
        assert!(
            detect_compact_ink_shelf(&hist_1_1c_lifted, &smooth_1_1c, total as u64, 255.0)
                .is_none(),
            "1.1c(ii): compact ink shelf must NOT slide across flat 3.6% plateau Y=10..27 to Y=30"
        );
        if let Some(DefectType::ElevatedBlackFloor { floor_value, .. }) =
            detect_elevated_black_floor(&hist_1_1c_lifted, PageClassification::Monochrome)
        {
            assert!(
                floor_value <= 12,
                "1.1c(ii): must stay at Y=10 plateau foot, never jump to Y=30 (got {})",
                floor_value
            );
        }

        // 1.2: m_zero == 0, 2.5% charcoal ink at Y=4 + 10% dark-gray fill at Y=25.
        // Must NOT jump over Y=4 charcoal ink to crush Y=25.
        let mut hist_1_2 = [0u32; 256];
        hist_1_2[4] = 2_500;
        hist_1_2[25] = 10_000;
        hist_1_2[255] = total - 12_500;
        if let Some(DefectType::ElevatedBlackFloor { floor_value, .. }) =
            detect_elevated_black_floor(&hist_1_2, PageClassification::Monochrome)
        {
            assert!(
                floor_value <= 5,
                "1.2: must not jump over 2.5% charcoal ink at Y=4 to Y=25 (got {})",
                floor_value
            );
        }

        // 1.4: Split dark-ink loophole: 2.5% at Y=0 + 2.5% at Y=4 (5.0% total) + 8% dark-gray fill at Y=28.
        // Unified hist[0..=trough_idx] < 3.5% cap must veto.
        let mut hist_1_4 = [0u32; 256];
        hist_1_4[0] = 2_500;
        hist_1_4[4] = 2_500;
        hist_1_4[28] = 8_000;
        hist_1_4[255] = total - 13_000;
        assert!(
            detect_elevated_black_floor(&hist_1_4, PageClassification::Monochrome).is_none(),
            "1.4: split 2.5% Y=0 + 2.5% Y=4 dark ink must veto Y=28 fill"
        );

        // 1.5: Wide Gaussian washed ink (sigma = 2.5 centered at Y=22, 15% total) + 0.6% Y=0 text.
        // Must anchor at left half-max foot (< 22) rather than raw peak center 22 so left-tail
        // shadow detail in Y=19..22 is preserved instead of flattened to <= 2.
        let mut hist_1_5 = [0u32; 256];
        hist_1_5[0] = 600;
        let mut gauss_sum = 0.0f64;
        let mut weights = [0.0f64; 256];
        for (y, w) in weights.iter_mut().enumerate().take(36).skip(12) {
            let d = (y as f64 - 22.0) / 2.5;
            *w = (-0.5 * d * d).exp();
            gauss_sum += *w;
        }
        for (y, &w) in weights.iter().enumerate().take(36).skip(12) {
            hist_1_5[y] = ((w / gauss_sum) * 15_000.0).round() as u32;
        }
        hist_1_5[255] = total - hist_1_5.iter().sum::<u32>();
        let det_1_5 = detect_elevated_black_floor(&hist_1_5, PageClassification::Monochrome)
            .expect("1.5: wide Gaussian washed ink shelf at Y=22 must be detected");
        assert!(
            matches!(
                det_1_5,
                DefectType::ElevatedBlackFloor {
                    floor_value: 18..=20,
                    is_bimodal: true,
                }
            ),
            "1.5: wide Gaussian washed ink at Y=22 must anchor at left foot (18..=20), got {:?}",
            det_1_5
        );

        // 1.7: Low-key night splash (p95 < 130): 15% compact dark sky at Y=18 + 1.5% white stars +
        // 83.5% dark gradient Y=40..90. Both detector and full pipeline must veto bit-exact.
        let mut hist_1_7 = [0u32; 256];
        hist_1_7[0] = 600;
        hist_1_7[18] = 15_000;
        hist_1_7[255] = 1_500;
        let rem = total - hist_1_7[0] - hist_1_7[18] - hist_1_7[255];
        for (i, bin) in hist_1_7[40..=89].iter_mut().enumerate() {
            *bin = rem / 50 + u32::from(i < (rem % 50) as usize);
        }
        assert!(
            detect_elevated_black_floor(&hist_1_7, PageClassification::Monochrome).is_none(),
            "1.7: low-key night splash (p95 < 130) must veto black-floor shift"
        );

        let night_img = ImageBuffer::from_fn(480u32, 640u32, |x, y| {
            let v = if y < 10 && x < 288 {
                0u8 // ~0.9% dark border
            } else if y < 106 {
                18u8 // ~15% compact dark night sky
            } else if y < 116 && x < 460 {
                255u8 // ~1.5% pure-white stars
            } else {
                (40 + ((x + y) % 50)) as u8 // 83% dark clouds Y=40..89 -> p95 < 130
            };
            Rgb([v, v, v])
        });
        let orig_bytes = night_img.as_raw().clone();
        let (out_night, info_night, _) =
            level_and_verify(DynamicImage::ImageRgb8(night_img), None, &AutoLevelsConfig::default());
        assert!(!info_night.was_leveled, "1.7: low-key night splash must not be leveled");
        assert_eq!(out_night.to_rgb8().as_raw(), &orig_bytes, "1.7: must remain bit-exact");
    }

    #[test]
    fn test_neutral_muddy_shadow_adversarial_counterexamples() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let config = AutoLevelsConfig::default();
        let (w, h) = (600u32, 800u32);

        // 2.1a: White paper + 30% warm earth/tan mid-darks RGB [80, 70, 57] (Y=71, L*=30.3, C*=9.5).
        // Strictly outside neutral-paper L* < 22.0 shadow limit -> MUST veto as Color bit-exact!
        let earth_img = ImageBuffer::from_fn(w, h, |_, y| {
            if y >= 560 {
                Rgb([80u8, 70u8, 57u8]) // 30% warm earth/tan mid-darks
            } else {
                Rgb([252u8, 252u8, 252u8])
            }
        });
        let earth_raw = earth_img.as_raw().clone();
        let (earth_out, earth_info, _) =
            level_and_verify(DynamicImage::ImageRgb8(earth_img), None, &config);
        assert_eq!(
            earth_info.classification,
            PageClassification::Color,
            "2.1a: 30% warm earth/tan mid-darks (L*=30.3) must veto as Color"
        );
        assert!(!earth_info.was_leveled);
        assert_eq!(earth_out.to_rgb8().as_raw(), &earth_raw);

        // 2.1b: Digital white paper [255,255,255] + 4% authentic black [0,0,0] + 25% intentional
        // warm-brown ink [35,24,14] (Y=26, L*=9.3, C*=9.0), ZERO colored logos. Must veto leveling
        // bit-exact AND must NEVER classify Monochrome (so upscaler.rs never folds to Luma8!).
        let warm_ink_img = ImageBuffer::from_fn(w, h, |_, y| {
            if y < 32 {
                Rgb([0u8, 0u8, 0u8]) // 4% authentic black ink
            } else if y >= 600 {
                Rgb([35u8, 24u8, 14u8]) // 25% intentional warm-brown ink
            } else {
                Rgb([255u8, 255u8, 255u8]) // optimal digital white paper
            }
        });
        let warm_ink_raw = warm_ink_img.as_raw().clone();
        let (warm_ink_out, warm_ink_info, _) =
            level_and_verify(DynamicImage::ImageRgb8(warm_ink_img), None, &config);
        assert!(!warm_ink_info.was_leveled, "2.1b: 4% authentic black must veto leveling");
        assert_ne!(
            warm_ink_info.classification,
            PageClassification::Monochrome,
            "2.1b: neutral-paper muddy shadow rescue must never classify Monochrome"
        );
        assert!(!warm_ink_info.is_monochrome);
        assert!(
            !crate::upscaler::upscaler::should_fold_luma(&warm_ink_info, true),
            "2.1b: upscaler must NOT fold un-leveled warm-brown ink to 1-channel Luma8"
        );
        assert_eq!(warm_ink_out.to_rgb8().as_raw(), &warm_ink_raw);

        // 2.2a: White paper (64%) + 22% warm shadows [27,17,7] + 14% deep navy [20,35,95] (L*=16.5 < 25).
        // Must classify Color and veto bit-exact via non-muddy C* >= 12 guard across ALL L*!
        let navy_img = ImageBuffer::from_fn(w, h, |_, y| {
            if y >= 624 {
                Rgb([27u8, 17u8, 7u8]) // 22% warm shadows
            } else if y >= 512 {
                Rgb([20u8, 35u8, 95u8]) // 14% deep navy (L* < 25, C* >= 30)
            } else {
                Rgb([252u8, 252u8, 252u8])
            }
        });
        let navy_raw = navy_img.as_raw().clone();
        let (navy_out, navy_info, _) =
            level_and_verify(DynamicImage::ImageRgb8(navy_img), None, &config);
        assert_eq!(
            navy_info.classification,
            PageClassification::Color,
            "2.2a: 14% deep navy + 22% warm shadows must veto as Color"
        );
        assert!(!navy_info.was_leveled);
        assert_eq!(navy_out.to_rgb8().as_raw(), &navy_raw);

        // 2.2b: White paper (64%) + 22% warm shadows [27,17,7] + 14% pastel sky [180,220,250] (L*=86 > 70).
        // Must classify Color and veto bit-exact via non-muddy C* >= 12 guard across ALL L*!
        let pastel_img = ImageBuffer::from_fn(w, h, |_, y| {
            if y >= 624 {
                Rgb([27u8, 17u8, 7u8]) // 22% warm shadows
            } else if y >= 512 {
                Rgb([180u8, 220u8, 250u8]) // 14% bright pastel blue (L* > 70, C* ~ 17.7)
            } else {
                Rgb([252u8, 252u8, 252u8])
            }
        });
        let pastel_raw = pastel_img.as_raw().clone();
        let (pastel_out, pastel_info, _) =
            level_and_verify(DynamicImage::ImageRgb8(pastel_img), None, &config);
        assert_eq!(
            pastel_info.classification,
            PageClassification::Color,
            "2.2b: 14% pastel sky + 22% warm shadows must veto as Color"
        );
        assert!(!pastel_info.was_leveled);
        assert_eq!(pastel_out.to_rgb8().as_raw(), &pastel_raw);

        // 2.3: Low-key warm night splash (98.5% warm darks [27,17,7]..[55,42,28] + 1.5% white stars).
        // Must NOT trigger neutral-paper muddy shadow rescue -> must classify Color and veto bit-exact!
        let warm_night_img = ImageBuffer::from_fn(w, h, |x, y| {
            if y < 12 {
                Rgb([255u8, 255u8, 255u8]) // 1.5% pure-white stars
            } else {
                let t = ((x + y) % 28) as u8;
                Rgb([27 + t, 17 + (t * 25 / 28), 7 + (t * 21 / 28)])
            }
        });
        let warm_night_raw = warm_night_img.as_raw().clone();
        let (warm_night_out, warm_night_info, _) =
            level_and_verify(DynamicImage::ImageRgb8(warm_night_img), None, &config);
        assert_eq!(
            warm_night_info.classification,
            PageClassification::Color,
            "2.3: low-key warm night splash with 1.5% white stars must veto as Color"
        );
        assert!(!warm_night_info.was_leveled);
        assert_eq!(warm_night_out.to_rgb8().as_raw(), &warm_night_raw);

        // 2.4: Aged-paper Mixed gradient continuity across L* in [50, 55) (no yellow dead-zone ring),
        // while neutral-paper Mixed stays strictly zero above L* = 38.
        let aged_plan = CorrectionPlan {
            cast_strength: 1.0,
            has_muddy_shadow_cast: true,
            cast_field: PaperCastField {
                active: true,
                dom_u_a: 0.0,
                dom_u_b: 1.0,
                c_max: 14.0,
                ..PaperCastField::default()
            },
            ..CorrectionPlan::default()
        };
        for l_mid in [50.0f32, 52.5, 54.9, 55.5] {
            let w_aged =
                compute_cast_weight_with_unit(l_mid, 0.0, 7.5, 14.0, 0.0, 1.0, &aged_plan);
            assert!(
                w_aged > 0.0,
                "2.4: aged-paper Mixed gradient must remain active at L*={l_mid} (got {w_aged})"
            );
        }

        let neutral_plan = CorrectionPlan {
            cast_strength: 1.0,
            has_muddy_shadow_cast: true,
            cast_field: PaperCastField {
                active: false,
                ..PaperCastField::default()
            },
            ..CorrectionPlan::default()
        };
        assert_eq!(
            compute_cast_weight_with_unit(39.0, 0.0, 7.5, 0.0, 0.0, 1.0, &neutral_plan),
            0.0,
            "2.4: neutral-paper Mixed shadow cast weight must be strictly 0 above L*=38"
        );
    }

    #[test]
    fn test_coherence_audit_invariants() {
        let _guard = TEST_RAYON_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        let config = AutoLevelsConfig::default();

        // 1. Rgba8 transparent-black [0, 0, 0, 0] pixels at the start of a Mixed strip must never
        //    match an uninitialized MixedPixelCache sentinel and become opaque black [0, 0, 0, 255].
        let rgba_img = ImageBuffer::from_fn(480u32, 480u32, |x, y| {
            if y < 64 {
                Rgba([0u8, 0u8, 0u8, 0u8]) // leading transparent strip
            } else if (80..120).contains(&y) && (80..120).contains(&x) {
                Rgba([215u8, 30u8, 30u8, 255u8]) // small high-chroma accent -> Mixed
            } else if y >= 380 {
                Rgba([25u8, 25u8, 25u8, 255u8]) // lifted neutral shadow floor
            } else {
                Rgba([250u8, 250u8, 250u8, 255u8]) // neutral bright background
            }
        });
        let (rgba_out, rgba_info, rgba_verdict) =
            level_and_verify(DynamicImage::ImageRgba8(rgba_img), None, &config);
        assert_eq!(rgba_info.classification, PageClassification::Mixed);
        assert!(rgba_info.was_leveled && rgba_verdict.outcome == Outcome::Pass);
        let out_rgba = rgba_out.to_rgba8();
        for x in [0u32, 1, 2, 3, 120, 479] {
            assert_eq!(
                out_rgba.get_pixel(x, 0).0,
                [0, 0, 0, 0],
                "Transparent black pixel at ({x}, 0) must keep alpha=0 unchanged"
            );
        }

        // 2. Unified shadow validation:
        //    (a) Compressed exponential ringing skirt attached to a compact washed-black shelf
        //        must restore the shelf floor cleanly (~19..=21).
        let total = 100_000u32;
        let mut hist_ringed = [0u32; 256];
        hist_ringed[0] = 550;
        for (idx, bin) in hist_ringed.iter_mut().enumerate().take(19).skip(6) {
            let d = (18 - idx) as f64;
            *bin = (590.0 * 0.86f64.powf(d)).round() as u32; // ~3.3% exponential edge skirt
        }
        hist_ringed[19] = 825;
        hist_ringed[20] = 1_125;
        hist_ringed[21] = 11_560;
        hist_ringed[22] = 1_175;
        hist_ringed[23] = 670;
        hist_ringed[255] = total - hist_ringed.iter().sum::<u32>();
        let det_ringed = detect_elevated_black_floor(&hist_ringed, PageClassification::Monochrome)
            .expect("Compact washed-black shelf with compression skirt must be detected");
        assert!(
            matches!(
                det_ringed,
                DefectType::ElevatedBlackFloor {
                    floor_value: 19..=21,
                    is_bimodal: true,
                }
            ),
            "Expected floor_value in 19..=21, got {:?}",
            det_ringed
        );

        //    (b) Continuous exponential dark shading before a dark-gray band must veto in BOTH
        //        compact-shelf detection AND Path-2 fallback (never crushing dark shading).
        let mut hist_sloped = [0u32; 256];
        hist_sloped[0] = 1_500;
        for (idx, bin) in hist_sloped.iter_mut().enumerate().take(29).skip(5) {
            let d = (28 - idx) as f64;
            *bin = (850.0 * 0.87f64.powf(d)).round() as u32; // ~6.3% continuous dark shading
        }
        hist_sloped[29] = 2_000;
        hist_sloped[30] = 7_500;
        hist_sloped[31] = 2_500;
        hist_sloped[255] = total - hist_sloped.iter().sum::<u32>();
        assert!(
            detect_elevated_black_floor(&hist_sloped, PageClassification::Monochrome).is_none(),
            "Continuous dark shading before dark-gray band must veto across both shadow paths"
        );

        //    (c) Detached dark ink at Y=4 alongside a small Y=0 spike must block jumping to Y=25.
        let mut hist_detached = [0u32; 256];
        hist_detached[0] = 600;
        hist_detached[4] = 2_500;
        hist_detached[25] = 10_000;
        hist_detached[255] = total - hist_detached.iter().sum::<u32>();
        assert!(
            detect_elevated_black_floor(&hist_detached, PageClassification::Monochrome).is_none(),
            "Detached 2.5% dark ink at Y=4 must veto jumping to Y=25"
        );

        //    (d) Split pre-valley dark mass (3.0% at Y=0 + 0.8% at Y=5 = 3.8% >= 3.5%) must veto
        //        in validate_black_floor as well.
        let mut hist_split = [0u32; 256];
        hist_split[0] = 3_000;
        hist_split[5] = 800;
        hist_split[45] = 45_000;
        hist_split[255] = total - hist_split.iter().sum::<u32>();
        assert_eq!(
            validate_black_floor(&hist_split, 45.0, 255.0),
            0.0,
            "Split 3.8% pre-valley dark mass must veto in validate_black_floor"
        );

        // 3. Un-leveled tinted RGB scenes must never be folded to 1-channel Luma8 post-upscale:
        //    (a) Low-key dark olive scene with small warm bright highlights
        let low_key_tinted = ImageBuffer::from_fn(480u32, 480u32, |_, y| {
            if y < 8 {
                Rgb([250u8, 242u8, 220u8]) // ~1.6% warm highlights
            } else {
                Rgb([32u8, 30u8, 16u8]) // low-key dark olive field (a* < 0)
            }
        });
        let (_, lk_info, _) =
            level_and_verify(DynamicImage::ImageRgb8(low_key_tinted), None, &config);
        assert!(!lk_info.was_leveled);
        assert!(
            !crate::upscaler::upscaler::should_fold_luma(&lk_info, true),
            "Low-key tinted RGB scene must never fold to 1-channel Luma8"
        );

        //    (b) Uniform tinted blank page (dynamic_range < 10)
        let tinted_blank = ImageBuffer::from_pixel(480u32, 480u32, Rgb([245u8, 232u8, 185u8]));
        let (_, tb_info, _) =
            level_and_verify(DynamicImage::ImageRgb8(tinted_blank), None, &config);
        assert!(!tb_info.was_leveled);
        assert!(
            !crate::upscaler::upscaler::should_fold_luma(&tb_info, true),
            "Vetoed uniform tinted RGB page must preserve its tint and skip Luma8 folding"
        );

        // 4. Dark saturated accents with rgb_diff in 22..29 (e.g. [48, 23, 48], C* >= 20.0)
        //    must pass the accent mask pre-filter.
        let tables = &*FAST_LAB_TABLES;
        let (mut k0, mut a0, mut k1, mut a1) = (u32::MAX, false, u32::MAX, false);
        assert!(
            eval_accent_px(
                &[48u8, 23u8, 48u8],
                &mut k0,
                &mut a0,
                &mut k1,
                &mut a1,
                false,
                0.0,
                0.0,
                0.0,
                tables,
            ),
            "Dark saturated accent [48, 23, 48] (rgb_diff=25, C*>=20) must be protected"
        );

        // 5. Mild plan (white_point=253) whose 50% attenuation collapses to identity (254 -> 255)
        //    must reject cleanly (was_leveled == false) rather than passing an identity retry.
        let mild_img = ImageBuffer::from_fn(480u32, 480u32, |x, y| {
            let v = if y < 96 {
                0u8 // 20% pure black -> black_point=0
            } else if y < 240 {
                (60 + ((x * 7 + y * 13) % 120)) as u8 // 30% textured midtones
            } else {
                253u8 // 50% slightly dull paper -> white_point=253
            };
            Rgb([v, v, v])
        });
        let (_, mild_info, mild_verdict) = level_and_verify_with_ssim_override(
            DynamicImage::ImageRgb8(mild_img),
            &config,
            1.0001, // force Tier-2 SSIM failure on first pass
        );
        assert!(
            !mild_info.was_leveled && mild_verdict.outcome == Outcome::Reject,
            "Attenuated no-op (white_point 253 -> 255) must reject rather than passing identity LUT"
        );
    }
}
