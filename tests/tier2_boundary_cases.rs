//! Tier 2: Boundary & Corner Cases Test Suite
//!
//! Validates exact numerical boundaries, transitions, and stress conditions across real production algorithms.

mod common;
use common::*;

use bytes::Bytes;
use image::{DynamicImage, Rgb, RgbImage};
use kurp::config::app_config::AutoLevelsConfig;
use kurp::upscaler::auto_levels::{
    apply_monochrome_mixed_correction, detect_clipped_highlights, detect_compressed_dynamic_range,
    detect_elevated_black_floor, detect_paper_yellowing, level_and_verify, quality_tier_1,
    quality_tier_2, quality_tier_3, CorrectionPlan, DefectType, DualZoneSample, Outcome,
    PageAnalysis, PageClassification,
};

// ============================================================================
// Boundary Test: Bimodal Surge Ratio (Threshold = 2.8x)
// ============================================================================

#[test]
fn test_tier2_bimodal_surge_ratio_exact_boundaries() {
    let config = AutoLevelsConfig::default();
    let n_samples = 100_000u32;

    // Sub-threshold: peak = 2790, valley = 1000 -> surge = 2.79 < 2.80
    let mut hist_sub = [0u32; 256];
    for b in 3..18 {
        hist_sub[b] = 1000;
    }
    hist_sub[18] = 2790;
    hist_sub[250] = n_samples - hist_sub.iter().sum::<u32>();
    let d_sub = detect_elevated_black_floor(&hist_sub, PageClassification::Monochrome, n_samples as usize, &config);
    if let Some(d) = d_sub {
        if let DefectType::ElevatedBlackFloor { is_bimodal, .. } = d.defect_type {
            assert!(!is_bimodal, "Surge ratio 2.79x must not be accepted as bimodal");
        }
    }

    // Exact/Above threshold: peak = 2810, valley = 1000 -> surge = 2.81 >= 2.80
    let mut hist_above = [0u32; 256];
    for b in 3..18 {
        hist_above[b] = 1000;
    }
    hist_above[18] = 2810;
    hist_above[250] = n_samples - hist_above.iter().sum::<u32>();
    let d_above = detect_elevated_black_floor(&hist_above, PageClassification::Monochrome, n_samples as usize, &config)
        .expect("Must detect elevated black floor");
    if let DefectType::ElevatedBlackFloor { is_bimodal, surge_ratio, .. } = d_above.defect_type {
        assert!(is_bimodal, "Surge ratio 2.81x must be accepted as bimodal");
        assert!(surge_ratio >= 2.80);
    } else {
        panic!("Expected ElevatedBlackFloor");
    }
}

// ============================================================================
// Boundary Test: Peak Compactness FWHM Check (Max FWHM = 6 bins)
// ============================================================================

#[test]
fn test_tier2_fwhm_peak_compactness_boundaries() {
    let config = AutoLevelsConfig::default();
    let n_samples = 100_000u32;

    // Compact peak: FWHM <= 6 (e.g. bins 16..=20, width 5)
    let mut hist_compact = [0u32; 256];
    for b in 16..=20 {
        hist_compact[b] = 1000;
    }
    hist_compact[18] = 2000; // Peak 2000, half_max = 1000, left=15, right=21, fwhm = 21-16 = 5 <= 6
    hist_compact[5] = 100;
    hist_compact[250] = n_samples - hist_compact.iter().sum::<u32>();
    let d_compact = detect_elevated_black_floor(&hist_compact, PageClassification::Monochrome, n_samples as usize, &config)
        .expect("Compact peak must be detected");
    if let DefectType::ElevatedBlackFloor { is_bimodal, .. } = d_compact.defect_type {
        assert!(is_bimodal, "Compact peak (FWHM <= 6) must pass as bimodal");
    }

    // Broad hump: FWHM > 6 bins (e.g. bins 13..=23 >= half_max)
    let mut hist_broad = [0u32; 256];
    for b in 13..=23 {
        hist_broad[b] = 1000;
    }
    hist_broad[18] = 1500; // half_max = 750, so bins 13..=23 >= 750, FWHM > 6
    hist_broad[5] = 100;
    hist_broad[250] = n_samples - hist_broad.iter().sum::<u32>();
    let d_broad = detect_elevated_black_floor(&hist_broad, PageClassification::Monochrome, n_samples as usize, &config);
    if let Some(d) = d_broad {
        if let DefectType::ElevatedBlackFloor { is_bimodal, .. } = d.defect_type {
            assert!(!is_bimodal, "Broad hump (FWHM > 6) must be rejected from bimodal shelf");
        }
    }
}

// ============================================================================
// Boundary Test: Authentic Black Linework (Threshold = 3.0%)
// ============================================================================

#[test]
fn test_tier2_authentic_black_percentage_boundaries() {
    let config = AutoLevelsConfig::default();
    let n_samples = 100_000u32;

    // Both 2.99% and 3.01% (and even 10%) at bin 0 do NOT disqualify the shelf at bin 18
    for &pct in &[0.0299, 0.0301, 0.1000] {
        let mut hist = [0u32; 256];
        hist[0] = (n_samples as f64 * pct) as u32;
        hist[18] = 3000;
        hist[8] = 100; // valley
        hist[250] = n_samples - hist.iter().sum::<u32>();

        let d = detect_elevated_black_floor(&hist, PageClassification::Monochrome, n_samples as usize, &config)
            .expect("Shelf at bin 18 must be detected");
        if let DefectType::ElevatedBlackFloor { floor_value, is_bimodal, .. } = d.defect_type {
            assert_eq!(floor_value, 18);
            assert!(is_bimodal, "Authentic black at bin 0 must never disqualify shelf at 18");
        }
    }
}

// ============================================================================
// Boundary Test: Tail Emptiness Check (Threshold = 0.70)
// ============================================================================

#[test]
fn test_tier2_tail_emptiness_exact_boundaries() {
    let total = 10_000usize;

    // Moody art: low tail emptiness (<0.70) returns None
    let hist_moody = [10u32; 256];
    let d_moody = detect_compressed_dynamic_range(&hist_moody, total);
    assert!(d_moody.is_none(), "Moody art with filled tails must return None");

    // Compressed scan: tail emptiness >= 0.70
    let mut hist_comp = [0u32; 256];
    for b in 40..=210 {
        hist_comp[b] = 50;
    }
    let d_comp = detect_compressed_dynamic_range(&hist_comp, total);
    assert!(d_comp.is_some(), "Compressed scan with empty tails must be detected");
}

// ============================================================================
// Boundary Test: Dynamic Range Effective Range (Threshold = 200)
// ============================================================================

#[test]
fn test_tier2_effective_range_boundaries() {
    let total = 10_000usize;

    // Effective range >= 200: healthy range, returns None
    let mut hist_healthy = [0u32; 256];
    for b in 10..=240 {
        hist_healthy[b] = 40;
    }
    let d_healthy = detect_compressed_dynamic_range(&hist_healthy, total);
    assert!(d_healthy.is_none(), "Healthy dynamic range (>=200) must return None");

    // Effective range < 10: solid/blank page protection, returns None
    let mut hist_solid = [0u32; 256];
    hist_solid[0] = 10_000;
    let d_solid = detect_compressed_dynamic_range(&hist_solid, total);
    assert!(d_solid.is_none(), "Solid page (<10 range) must return None");
}

// ============================================================================
// Boundary Test: Physical Lignin Filter (R >= G > B, R-B in [6.0, 55.0])
// ============================================================================

#[test]
fn test_tier2_physical_lignin_chroma_bounds() {
    let config = AutoLevelsConfig::default();

    let make_analysis = |r: f32, g: f32, b: f32| -> PageAnalysis {
        PageAnalysis {
            page_number: 1,
            classification: PageClassification::Color,
            luminance_histogram: [0; 256],
            neutral_luminance_histogram: [0; 256],
            neutral_channel_histograms: ([0; 256], [0; 256], [0; 256]),
            total_pixels: 10_000,
            neutral_pixel_count: 9_000,
            neutral_ratio: 0.9,
            max_chroma: 20,
            high_chroma_count: 0,
            highlight_sample: DualZoneSample {
                count: 500,
                avg_rgb: (r, g, b),
                chroma: r - b,
            },
            shadow_sample: DualZoneSample {
                count: 500,
                avg_rgb: (15.0, 15.0, 15.0),
                chroma: 0.0,
            },
            defects: Vec::new(),
            black_floor: 0,
            white_ceiling: 255,
            is_shelf_detected: false,
            per_channel_black: None,
        }
    };

    // Sub-threshold: R-B = 5.0 -> None
    let a_sub = make_analysis(245.0, 242.0, 240.0);
    assert!(detect_paper_yellowing(&a_sub, PageClassification::Color, &config).is_none());

    // In-range: R-B = 10.0 -> Some
    let a_valid = make_analysis(245.0, 240.0, 235.0);
    assert!(detect_paper_yellowing(&a_valid, PageClassification::Color, &config).is_some());

    // Upper boundary: R-B = 56.0 -> None (saturated color)
    let a_high = make_analysis(255.0, 230.0, 198.0);
    assert!(detect_paper_yellowing(&a_high, PageClassification::Color, &config).is_none());
}

// ============================================================================
// Boundary Test: Dual-Zone Sepia Check (Shadow R > B + 4.0)
// ============================================================================

#[test]
fn test_tier2_dual_zone_sepia_shadow_boundary() {
    let config = AutoLevelsConfig::default();

    let make_analysis = |sh_r: f32, sh_b: f32| -> PageAnalysis {
        PageAnalysis {
            page_number: 1,
            classification: PageClassification::Color,
            luminance_histogram: [0; 256],
            neutral_luminance_histogram: [0; 256],
            neutral_channel_histograms: ([0; 256], [0; 256], [0; 256]),
            total_pixels: 10_000,
            neutral_pixel_count: 9_000,
            neutral_ratio: 0.9,
            max_chroma: 20,
            high_chroma_count: 0,
            highlight_sample: DualZoneSample {
                count: 500,
                avg_rgb: (245.0, 240.0, 230.0), // R-B = 15.0
                chroma: 15.0,
            },
            shadow_sample: DualZoneSample {
                count: 500,
                avg_rgb: (sh_r, 15.0, sh_b),
                chroma: (sh_r - sh_b).abs(),
            },
            defects: Vec::new(),
            black_floor: 0,
            white_ceiling: 255,
            is_shelf_detected: false,
            per_channel_black: None,
        }
    };

    // Neutral shadow: R = 15.0, B = 12.0 (diff 3.0 <= 4.0) -> detected as paper yellowing
    let a_neutral = make_analysis(15.0, 12.0);
    assert!(detect_paper_yellowing(&a_neutral, PageClassification::Color, &config).is_some());

    // Warm shadow: R = 17.0, B = 11.0 (diff 6.0 > 4.0) -> protected sepia artwork, returns None
    let a_sepia = make_analysis(17.0, 11.0);
    assert!(detect_paper_yellowing(&a_sepia, PageClassification::Color, &config).is_none());
}

// ============================================================================
// Boundary Test: Clipped Highlights White Point (wp = 200 & plateau)
// ============================================================================

#[test]
fn test_tier2_clipped_highlights_white_point_boundary() {
    let config = AutoLevelsConfig::default();
    let total = 10_000usize;

    // Healthy white point >= 254 -> None
    let mut hist_healthy = [0u32; 256];
    hist_healthy[254] = 200;
    hist_healthy[100] = 9800;
    assert!(detect_clipped_highlights(&hist_healthy, PageClassification::Monochrome, total, &config).is_none());

    // Valid clipped highlight at 240 -> Some
    let mut hist_clipped = [0u32; 256];
    hist_clipped[240] = 500;
    hist_clipped[100] = 9500;
    assert!(detect_clipped_highlights(&hist_clipped, PageClassification::Monochrome, total, &config).is_some());
}

// ============================================================================
// Boundary Test: Hermite Smoothstep Blending Transitions
// ============================================================================

#[test]
fn test_tier2_hermite_smoothstep_chroma_boundaries() {
    let mut img = DynamicImage::ImageRgb8(RgbImage::new(3, 1));
    let rgb = img.as_mut_rgb8().unwrap();
    rgb.put_pixel(0, 0, Rgb([20, 20, 20]));     // Pure monochrome linework (chroma = 0)
    rgb.put_pixel(1, 0, Rgb([100, 80, 80]));    // Transition zone (chroma = 20)
    rgb.put_pixel(2, 0, Rgb([255, 0, 0]));      // Pure color element (chroma = 255)

    let plan = CorrectionPlan {
        black_point: 20,
        white_point: 255,
        ..Default::default()
    };

    apply_monochrome_mixed_correction(&mut img, &plan, PageClassification::Mixed);

    let res = img.to_rgb8();
    // Monochrome linework at level 20 mapped to Fritsch-Carlson shadow knot (8)
    assert!(res.get_pixel(0, 0)[0] <= 8);
    // Saturated red element untouched (chroma >= 35)
    assert_eq!(res.get_pixel(2, 0), &Rgb([255, 0, 0]));
}

// ============================================================================
// Boundary Test: Quality Guard Tier 1 (Slope Erosion, MSE, PSNR)
// ============================================================================

#[test]
fn test_tier2_tier1_quality_guard_boundaries() {
    let config = AutoLevelsConfig::default();
    let total = 10_000usize;
    let plan = CorrectionPlan {
        black_point: 18,
        white_point: 255,
        ..Default::default()
    };

    let orig = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, Rgb([128, 128, 128])));
    // Slight shift (pass)
    let lev_pass = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, Rgb([130, 130, 130])));
    let h1 = [10u32; 256];
    let res_pass = quality_tier_1(&h1, &h1, &orig, &lev_pass, &plan, total, &config);
    assert_eq!(res_pass.outcome, Outcome::Pass);

    // Severe shift (MSE > 220, fail)
    let lev_fail = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, Rgb([200, 200, 200])));
    let res_fail = quality_tier_1(&h1, &h1, &orig, &lev_fail, &plan, total, &config);
    assert_eq!(res_fail.outcome, Outcome::Reject);
}

// ============================================================================
// Boundary Test: Quality Guard Tier 2 (SSIM, Delta E chroma)
// ============================================================================

#[test]
fn test_tier2_tier2_quality_guard_boundaries() {
    let config = AutoLevelsConfig::default();
    let plan = CorrectionPlan::default();

    let orig = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, Rgb([128, 128, 128])));
    let lev = DynamicImage::ImageRgb8(RgbImage::from_pixel(100, 100, Rgb([128, 128, 128])));

    let res = quality_tier_2(&orig, &lev, PageClassification::Monochrome, &plan, &config);
    assert_eq!(res.outcome, Outcome::Pass);
}

// ============================================================================
// Boundary Test: Quality Guard Tier 3 (Sobel NCC, Block Michelson)
// ============================================================================

#[test]
fn test_tier2_tier3_quality_guard_boundaries() {
    let config = AutoLevelsConfig::default();
    let orig = generate_bimodal_ink_shelf_page(100, 100, 18);
    let lev = orig.clone();

    let res = quality_tier_3(&orig, &lev, &config);
    assert_eq!(res.outcome, Outcome::Pass);
}

// ============================================================================
// Boundary Test: Small Image Fast Bypass (<400px)
// ============================================================================

#[test]
fn test_tier2_thumbnail_dimensions_boundaries() {
    let config = AutoLevelsConfig::default();

    // 399x399: bypassed
    let img_399 = generate_bimodal_ink_shelf_page(399, 399, 18);
    let bytes_399 = Bytes::from(encode_to_png_bytes(&img_399));
    let (_, info_399, _) = level_and_verify(img_399, &bytes_399, None, &config);
    assert!(!info_399.was_leveled, "399x399 must be bypassed");

    // 400x400: processed
    let img_400 = generate_bimodal_ink_shelf_page(400, 400, 18);
    let bytes_400 = Bytes::from(encode_to_png_bytes(&img_400));
    let (_, info_400, _) = level_and_verify(img_400, &bytes_400, None, &config);
    assert!(info_400.was_leveled, "400x400 must be leveled");
}
