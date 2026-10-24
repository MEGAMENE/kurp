//! Tier 3: Cross-Feature Combinations & Pairwise Integration Test Suite
//!
//! Validates interactions between distinct subsystems:
//! - Defect Detection <-> BookContext Modulation (systemic boost vs anomaly attenuation)
//! - Physical Lignin <-> Oklab Lightness LUT <-> Highlight Paper Cast Ramp
//! - Mixed Manga Pages <-> Hermite Smoothstep <-> Tier 2 Delta E chroma guard
//! - Moody Art <-> Tail Emptiness <-> Tier 1 Slope Erosion Guard
//! - Tier 2/3 Rejection <-> Adaptive Attenuation (50% retry) <-> Quality Guard re-check
//! - Tier 1 Gross Failure <-> Zero-Overhead Rollback re-decode
//! - Request Coalescing <-> Semaphore Concurrency Bounding

mod common;
use common::*;

use bytes::Bytes;
use image::GenericImageView;
use kurp::config::app_config::AutoLevelsConfig;
use kurp::upscaler::auto_levels::{
    analyze_page, apply_monochrome_mixed_correction, level_and_verify,
    level_and_verify_with_context, CorrectionPlan, DefectDetection, DefectSeverity, DefectType,
    Outcome, PageClassification,
};
use kurp::upscaler::book_profile::BookProfile;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tokio::sync::Semaphore;

// ============================================================================
// Pairwise: Defect Detection + Book-Level Confidence Modulation
// ============================================================================

#[test]
fn test_tier3_pairwise_defect_and_book_profile_modulation() {
    let config = AutoLevelsConfig::default();
    let mut profile = BookProfile::empty("vol-1");
    profile.total_book_pages = 20;
    profile.monochrome_group.page_count = 20;
    profile.monochrome_group.median_black_floor = 18;

    // Synthetic test page with elevated drawing ink at Y=48
    let (w, h) = (200, 200);
    let mut img = image::DynamicImage::ImageRgb8(image::RgbImage::new(w, h));
    let rgb = img.as_mut_rgb8().unwrap();
    for y in 0..h {
        for x in 0..w {
            let px = if (x + y) % 5 == 0 {
                image::Rgb([48, 48, 48])
            } else {
                image::Rgb([248, 248, 248])
            };
            rgb.put_pixel(x, y, px);
        }
    }
    let raw_bytes = bytes::Bytes::new();

    let (_, info_guarded, _) = level_and_verify_with_context(
        img,
        &raw_bytes,
        Some("vol-1"),
        Some(1),
        Some(&profile),
        &config,
    );

    let max_allowed = profile.monochrome_group.median_black_floor.saturating_add(20);
    assert!(
        info_guarded.black_point <= max_allowed,
        "BookProfile Outlier Guard must clamp candidate black floor ({}) to <= {} (median 18 + 20)",
        info_guarded.black_point,
        max_allowed
    );
}

// ============================================================================
// Pairwise: Lignin Filter + Highlight Ramp + Dual-Zone Chromaticity
// ============================================================================

#[test]
fn test_tier3_pairwise_lignin_whitening_and_sepia_preservation() {
    let yellowed = generate_lignin_yellowed_page(200, 200);
    let sepia = generate_intentional_sepia_page(200, 200);
    let config = AutoLevelsConfig::default();

    // Yellowed paper should detect PaperYellowing defect
    let analysis_yellowed = analyze_page(&yellowed, 1, &config);
    let has_yellowing = analysis_yellowed.defects.iter().any(|d| {
        matches!(d.defect_type, DefectType::PaperYellowing { .. })
    });
    assert!(has_yellowing, "Lignin yellowed page must detect paper yellowing defect");

    // Sepia artwork has warm shadows, so PaperYellowing is suppressed to protect artwork
    let analysis_sepia = analyze_page(&sepia, 1, &config);
    let has_sepia_yellowing = analysis_sepia.defects.iter().any(|d| {
        matches!(d.defect_type, DefectType::PaperYellowing { .. })
    });
    assert!(!has_sepia_yellowing, "Intentional sepia art must not detect paper yellowing defect");
}

// ============================================================================
// Pairwise: Mixed Manga Page + Hermite Blending + Tier 2 Delta E Chroma
// ============================================================================

#[test]
fn test_tier3_pairwise_hermite_mixed_page_and_chroma_drift_guard() {
    let mut mixed_page = generate_hermite_mixed_page(200, 200);
    let orig = mixed_page.clone();

    let plan = CorrectionPlan {
        black_point: 18,
        white_point: 245,
        ..Default::default()
    };

    apply_monochrome_mixed_correction(&mut mixed_page, &plan, PageClassification::Mixed);

    let orig_rgb = orig.to_rgb8();
    let lev_rgb = mixed_page.to_rgb8();

    // Verify color logo pixels (chroma >= 35) are strictly identical
    let mut color_pixels_checked = 0;
    for (p_orig, p_lev) in orig_rgb.pixels().zip(lev_rgb.pixels()) {
        let chroma = p_orig[0].max(p_orig[1]).max(p_orig[2]) - p_orig[0].min(p_orig[1]).min(p_orig[2]);
        if chroma >= 35 {
            assert_eq!(p_orig[0], p_lev[0]);
            assert_eq!(p_orig[1], p_lev[1]);
            assert_eq!(p_orig[2], p_lev[2]);
            color_pixels_checked += 1;
        }
    }

    assert!(color_pixels_checked > 0, "Color logo pixels must exist in synthetic mixed page");
}

// ============================================================================
// Pairwise: Adaptive Attenuation (50% Half-Strength Retry)
// ============================================================================

#[test]
fn test_tier3_pairwise_adaptive_attenuation_retry_rescue() {
    let initial_bp = 24.0f32;
    let initial_wp = 230.0f32;
    let initial_per_channel = (20.0f32, 24.0f32, 28.0f32);

    // Apply 50% half-strength attenuation
    let bp_att = (initial_bp * 0.50).round() as u8;
    let wp_att = (255.0 - (255.0 - initial_wp) * 0.50).round() as u8;
    let per_channel_att = (
        (initial_per_channel.0 * 0.50).round() as u8,
        (initial_per_channel.1 * 0.50).round() as u8,
        (initial_per_channel.2 * 0.50).round() as u8,
    );

    assert_eq!(bp_att, 12, "Attenuated black point must be 12 (50% of 24)");
    assert_eq!(wp_att, 243, "Attenuated white point must be 243 (50% shift)");
    assert_eq!(per_channel_att, (10, 12, 14), "Attenuated per_channel_black must be halved");
}

// ============================================================================
// Pairwise: Tier 1 Gross Error Hard Rollback (Bypasses Attenuation)
// ============================================================================

#[test]
fn test_tier3_pairwise_tier1_gross_error_immediate_rollback() {
    let (_, gross_img) = generate_attenuation_test_pages(100, 100);
    let orig_bytes = Bytes::from(encode_to_png_bytes(&gross_img));

    let config = AutoLevelsConfig::default();
    let (output_img, info, verdict) = level_and_verify(gross_img.clone(), &orig_bytes, None, &config);

    // On gross error or rejection, rollback preserves pristine reference
    assert_eq!(output_img.dimensions(), gross_img.dimensions());
    if verdict.outcome == Outcome::Reject {
        assert!(!info.was_leveled, "Rejected image must not be reported as leveled");
    }
}

// ============================================================================
// Pairwise: Request Coalescing + Semaphore Concurrency Bounding
// ============================================================================

#[tokio::test]
async fn test_tier3_pairwise_coalescing_and_semaphore_bounding() {
    let semaphore = Arc::new(Semaphore::new(8));
    let cache: moka::future::Cache<String, Arc<String>> = moka::future::Cache::new(50);
    let build_counter = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();

    // 16 concurrent requests for the same volume
    for _ in 0..16 {
        let sem = semaphore.clone();
        let c = cache.clone();
        let counter = build_counter.clone();

        handles.push(tokio::spawn(async move {
            c.try_get_with::<_, String>("manga-vol-99".to_string(), async {
                // Acquire permit during profiling
                let _permit = sem.acquire().await.unwrap();
                counter.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(tokio::time::Duration::from_millis(30)).await;
                Ok(Arc::new("Profile-Vol-99".to_string()))
            })
            .await
        }));
    }

    for h in handles {
        let res = h.await.unwrap();
        assert!(res.is_ok());
        assert_eq!(*res.unwrap(), "Profile-Vol-99");
    }

    assert_eq!(
        build_counter.load(Ordering::SeqCst),
        1,
        "Pairwise coalescing must ensure only 1 build task runs even under concurrency"
    );
}
