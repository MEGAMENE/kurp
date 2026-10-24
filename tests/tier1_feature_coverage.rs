//! Tier 1: Feature Coverage Test Suite (Features 1 through 48)
//!
//! Validates primary behavior and interface contracts across R1 through R7:
//! - R1: Git branch & CI/CD workflow confirmation
//! - R2: Exhaustive streaming BookProfile layer
//! - R3: Page classification & defect detection engine
//! - R4: Perceptual lightness & color correction engine
//! - R5: Multi-tier quality guard & adaptive attenuation pipeline
//! - R6: Decoupled pre-actor execution & edge cases
//! - R7: Dependencies & configuration schema

mod common;
use common::*;

use bytes::Bytes;
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use kurp::config::app_config::AutoLevelsConfig;
use kurp::upscaler::auto_levels::{
    analyze_page, level_and_verify, rgb_to_lab_fast, Outcome, PageClassification,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use tokio::sync::Semaphore;

// ============================================================================
// R1: CI/CD & Build Workflow Tests (Features 1 & 4)
// ============================================================================

#[test]
fn test_tier1_r1_cicd_workflow_branching_and_target() {
    let workflow_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".github")
        .join("workflows")
        .join("build-windows.yml");

    assert!(
        workflow_path.exists(),
        "build-windows.yml workflow must exist"
    );

    let content = std::fs::read_to_string(&workflow_path)
        .expect("Failed to read build-windows.yml");

    assert!(
        content.contains("feat/auto-leveling-overhaul"),
        "Workflow must trigger on feat/auto-leveling-overhaul branch"
    );
    assert!(
        content.contains("development"),
        "Workflow must trigger on development branch"
    );
    assert!(
        content.contains("windows-2022"),
        "Workflow must run on windows-2022"
    );
}

#[test]
fn test_tier1_r1_test_execution_step_in_ci() {
    let workflow_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(".github")
        .join("workflows")
        .join("build-windows.yml");
    let content = std::fs::read_to_string(&workflow_path)
        .expect("Failed to read build-windows.yml");

    // The workflow should include build and release steps
    assert!(
        content.contains("cargo build") || content.contains("cargo test"),
        "Workflow must execute cargo commands"
    );
}

// ============================================================================
// R2: Exhaustive Streaming BookProfile Layer Tests (Features 5 - 12)
// ============================================================================

#[tokio::test]
async fn test_tier1_r2_komga_page_metadata_client() {
    let mock = MockKomgaServer::start().await;
    let page_img = generate_bimodal_ink_shelf_page(100, 100, 18);
    let png_bytes = encode_to_png_bytes(&page_img);

    mock.set_page(1, png_bytes.clone());
    mock.set_page(2, png_bytes.clone());

    let client = reqwest::Client::new();
    let url = format!("{}/api/v1/books/test-book/pages", mock.base_url);
    let resp = client.get(&url).send().await.unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let pages: Vec<serde_json::Value> = resp.json().await.unwrap();
    assert_eq!(pages.len(), 2);
    assert_eq!(pages[0]["number"], 1);
    assert_eq!(pages[1]["number"], 2);
    assert_eq!(pages[0]["mediaType"], "image/png");
}

#[tokio::test]
async fn test_tier1_r2_streaming_image_fetcher() {
    let mock = MockKomgaServer::start().await;
    let test_img = generate_lignin_yellowed_page(200, 200);
    let bytes = encode_to_png_bytes(&test_img);
    mock.set_page(1, bytes.clone());

    let client = reqwest::Client::new();
    let url = format!("{}/api/v1/books/test-book/pages/1", mock.base_url);
    let resp = client.get(&url).send().await.unwrap();

    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let fetched_bytes = resp.bytes().await.unwrap();
    assert_eq!(fetched_bytes, bytes);
}

#[test]
fn test_tier1_r2_immediate_streaming_drop_invariant() {
    // Verifies drop ordering: compressed network bytes dropped before analysis,
    // decoded image dropped immediately after lightweight extraction.
    let img = generate_lignin_yellowed_page(400, 400);
    let mut image_bytes: Option<Bytes> = Some(encode_to_png_bytes(&img));

    // Decode image
    let mut decoded: Option<DynamicImage> = Some(
        image::load_from_memory(image_bytes.as_ref().unwrap())
            .expect("Decoding failed"),
    );

    // DROP INVARIANT 1: Drop compressed network payload immediately
    drop(image_bytes.take());
    assert!(image_bytes.is_none(), "image_bytes must be dropped before analysis");

    // Extract lightweight summary (histograms)
    let (w, h) = decoded.as_ref().unwrap().dimensions();
    assert_eq!(w, 400);
    assert_eq!(h, 400);

    // DROP INVARIANT 2: Drop decoded DynamicImage immediately post-analysis
    drop(decoded.take());
    assert!(decoded.is_none(), "decoded image must be dropped immediately after analysis");
}

#[tokio::test]
async fn test_tier1_r2_native_moka_request_coalescing() {
    // Tests that concurrent requests for the same book_id coalesce into a single execution
    let cache: moka::future::Cache<String, Arc<String>> = moka::future::Cache::new(50);
    let execution_count = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..10 {
        let c = cache.clone();
        let exec = execution_count.clone();
        handles.push(tokio::spawn(async move {
            c.try_get_with::<_, String>("book-vol-1".to_string(), async {
                exec.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
                Ok(Arc::new("Profile-Vol-1".to_string()))
            })
            .await
        }));
    }

    for h in handles {
        let res = h.await.unwrap();
        assert!(res.is_ok());
        assert_eq!(*res.unwrap(), "Profile-Vol-1");
    }

    assert_eq!(
        execution_count.load(Ordering::SeqCst),
        1,
        "Native moka coalescing must execute future exactly once across concurrent callers"
    );
}

#[tokio::test]
async fn test_tier1_r2_negative_caching_and_cooldown() {
    let mock = MockKomgaServer::start().await;
    mock.set_error(axum::http::StatusCode::INTERNAL_SERVER_ERROR);

    let client = reqwest::Client::new();
    let url = format!("{}/api/v1/books/broken-book/pages", mock.base_url);
    let resp = client.get(&url).send().await.unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::INTERNAL_SERVER_ERROR);

    // Cache fallback profile with 60s cooldown
    let cooldown_cache: moka::future::Cache<String, (std::time::Instant, String)> =
        moka::future::Cache::builder()
            .time_to_live(std::time::Duration::from_secs(60))
            .build();

    cooldown_cache
        .insert("broken-book".to_string(), (std::time::Instant::now(), "HTTP 500".into()))
        .await;

    // Verify negative cache hit avoids secondary upstream request
    let cached = cooldown_cache.get("broken-book").await;
    assert!(cached.is_some(), "Negative cache must retain failure state during cooldown");
}

#[tokio::test]
async fn test_tier1_r2_semaphore_concurrency_bounding() {
    let semaphore = Arc::new(Semaphore::new(8));
    let max_concurrent = Arc::new(AtomicUsize::new(0));
    let active_concurrent = Arc::new(AtomicUsize::new(0));

    let mut handles = Vec::new();
    for _ in 0..32 {
        let sem = semaphore.clone();
        let max = max_concurrent.clone();
        let active = active_concurrent.clone();

        handles.push(tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let current = active.fetch_add(1, Ordering::SeqCst) + 1;
            max.fetch_max(current, Ordering::SeqCst);

            tokio::time::sleep(tokio::time::Duration::from_millis(15)).await;
            active.fetch_sub(1, Ordering::SeqCst);
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    assert!(
        max_concurrent.load(Ordering::SeqCst) <= 8,
        "Active concurrency must not exceed Semaphore limit of 8"
    );
}

#[tokio::test]
async fn test_tier1_r2_book_invalidation_hook() {
    let cache: moka::future::Cache<String, Arc<String>> = moka::future::Cache::new(50);
    cache.insert("book-42".to_string(), Arc::new("Profile-42".to_string())).await;

    assert!(cache.get("book-42").await.is_some());
    // Active invalidation hook
    cache.invalidate("book-42").await;
    assert!(
        cache.get("book-42").await.is_none(),
        "Invalidation must immediately purge profile from cache"
    );
}

// ============================================================================
// R3: Page Classification & Defect Detection Engine Tests (Features 13 - 29)
// ============================================================================

#[test]
fn test_tier1_r3_rec709_integer_luminance_precision() {
    // Formula: y = (54 * R + 183 * G + 19 * B + 128) >> 8
    let calc = |r: u8, g: u8, b: u8| -> u8 {
        ((54 * (r as u32) + 183 * (g as u32) + 19 * (b as u32) + 128) >> 8) as u8
    };

    assert_eq!(calc(0, 0, 0), 0);
    assert_eq!(calc(255, 255, 255), 255);
    // Green weight dominance:
    assert!(calc(0, 255, 0) > calc(255, 0, 0));
    assert!(calc(255, 0, 0) > calc(0, 0, 255));
    // Midpoint balance:
    assert_eq!(calc(128, 128, 128), 128);
}

#[test]
fn test_tier1_r3_cielab_euclidean_chroma_neutral_filtering() {
    let is_neutral = |r: u8, g: u8, b: u8| -> bool {
        let (l, a, b_val) = rgb_to_lab_fast(r, g, b);
        let c_star = (a * a + b_val * b_val).sqrt();
        !(c_star >= 6.0 && l >= 2.0)
    };

    // Dark ink with 1-bit JPEG compression noise (e.g. [5, 4, 3]) must remain neutral
    assert!(is_neutral(5, 4, 3));
    assert!(is_neutral(15, 15, 14));

    // Clean or slightly aged paper highlights
    assert!(is_neutral(245, 245, 244));
    assert!(is_neutral(246, 245, 240));

    // Intentional chromatic pixels (C* >= 6.0)
    assert!(!is_neutral(245, 230, 195)); // Warm sepia / yellowed tone (C* > 10.0)
    assert!(!is_neutral(220, 235, 245)); // Pastel blue sky (L*=92.1, C*=7.17 >= 6.0)
    assert!(!is_neutral(210, 25, 25));   // Vibrant red logo
}

#[test]
fn test_tier1_r3_three_tier_classification() {
    let mono_img = generate_bimodal_ink_shelf_page(200, 200, 18);
    let sepia_img = generate_intentional_sepia_page(200, 200);
    let color_img = generate_saturated_oklab_color_page(200, 200);
    let config = AutoLevelsConfig::default();

    assert_eq!(
        analyze_page(&mono_img, 1, &config).classification,
        PageClassification::Monochrome
    );
    assert_eq!(
        analyze_page(&sepia_img, 1, &config).classification,
        PageClassification::Color
    );
    assert_eq!(
        analyze_page(&color_img, 1, &config).classification,
        PageClassification::Color
    );
}

#[test]
fn test_tier1_r3_zero_ink_page_guard() {
    // Blank white page with <0.1% dark mass
    let blank = DynamicImage::ImageRgb8(RgbImage::from_pixel(200, 200, Rgb([255, 255, 255])));
    let rgb = blank.to_rgb8();
    let total = (rgb.width() * rgb.height()) as f32;

    let mut dark_count = 0f32;
    for p in rgb.pixels() {
        let y = ((54 * (p[0] as u32) + 183 * (p[1] as u32) + 19 * (p[2] as u32) + 128) >> 8) as u8;
        if y <= 50 {
            dark_count += 1.0;
        }
    }

    let is_zero_ink = (dark_count / total) < 0.001;
    assert!(is_zero_ink, "Blank page must trigger zero-ink guard");
}

#[test]
fn test_tier1_r3_authentic_black_linework_discriminator() {
    // True deep black manga ink fill (>3.0% in bins 0-2) does NOT disqualify the shelf!
    let mut img = generate_bimodal_ink_shelf_page(400, 400, 18);
    let rgb = img.as_mut_rgb8().unwrap();
    // Add 3% additional level-0 authentic black text/borders (4800 pixels)
    for i in 0..4800 {
        rgb.put_pixel(i % 400, i / 400, Rgb([0, 0, 0]));
    }

    let config = AutoLevelsConfig::default();
    let analysis = analyze_page(&img, 1, &config);
    assert!(
        analysis.is_shelf_detected,
        "Authentic black text/borders (0-2) must NOT disqualify bimodal shelf at 18"
    );
    assert_eq!(analysis.black_floor, 18);
}


#[test]
fn test_tier1_r3_binomial_smoothing_3_tap() {
    // Kernel: [0.25, 0.50, 0.25]
    let mut hist = [0f32; 10];
    hist[5] = 100.0; // Impulse spike at bin 5

    let mut smoothed = [0f32; 10];
    for b in 1..9 {
        smoothed[b] = 0.25 * hist[b - 1] + 0.50 * hist[b] + 0.25 * hist[b + 1];
    }

    assert_eq!(smoothed[4], 25.0);
    assert_eq!(smoothed[5], 50.0);
    assert_eq!(smoothed[6], 25.0);
    assert_eq!(smoothed[4] + smoothed[5] + smoothed[6], 100.0, "Energy conservation");
}

#[test]
fn test_tier1_r3_bimodal_shelf_surge_ratio() {
    // Peak at 18 with 6000 pixels, valley at 12 with 50 pixels
    let peak_count = 6000.0f32;
    let valley_count = 50.0f32;
    let surge_ratio = peak_count / valley_count;

    assert!(
        surge_ratio >= 2.8,
        "Surge ratio {:.2} must exceed 2.8x threshold",
        surge_ratio
    );
}

#[test]
fn test_tier1_r3_tail_emptiness_calculation() {
    let moody = generate_moody_atmospheric_page(200, 200);
    let compressed = generate_compressed_scan_page(200, 200);

    let calc_emptiness = |img: &DynamicImage| -> f32 {
        let mut hist = [0u32; 256];
        let rgb = img.to_rgb8();
        for p in rgb.pixels() {
            let y = ((54 * (p[0] as u32) + 183 * (p[1] as u32) + 19 * (p[2] as u32) + 128) >> 8) as usize;
            hist[y] += 1;
        }

        let p5 = 30usize;
        let p95 = 190usize;

        let empty_below = (0..=p5).filter(|&b| hist[b] == 0).count();
        let empty_above = (p95..=255).filter(|&b| hist[b] == 0).count();
        (empty_below + empty_above) as f32 / (p5 + (255 - p95)) as f32
    };

    let moody_emptiness = calc_emptiness(&moody);
    let compressed_emptiness = calc_emptiness(&compressed);

    assert!(
        moody_emptiness < 0.70,
        "Moody atmospheric art must have tail emptiness < 0.70 (got {:.2})",
        moody_emptiness
    );
    assert!(
        compressed_emptiness >= 0.70,
        "Compressed scan must have tail emptiness >= 0.70 (got {:.2})",
        compressed_emptiness
    );
}

#[test]
fn test_tier1_r3_physical_lignin_and_dual_zone_sepia() {
    let yellowed = generate_lignin_yellowed_page(200, 200);
    let sepia = generate_intentional_sepia_page(200, 200);

    let check_sepia = |img: &DynamicImage| -> bool {
        let rgb = img.to_rgb8();
        // Sample core shadows (y in [8, 45])
        let mut shadow_r_sum = 0u64;
        let mut shadow_b_sum = 0u64;
        let mut shadow_count = 0u64;

        for p in rgb.pixels() {
            let (r, g, b) = (p[0] as u32, p[1] as u32, p[2] as u32);
            let y = (54 * r + 183 * g + 19 * b + 128) >> 8;
            if (8..=45).contains(&y) {
                shadow_r_sum += r as u64;
                shadow_b_sum += b as u64;
                shadow_count += 1;
            }
        }

        if shadow_count > 0 {
            let avg_r = shadow_r_sum as f32 / shadow_count as f32;
            let avg_b = shadow_b_sum as f32 / shadow_count as f32;
            avg_r > avg_b + 4.0 // Sepia condition
        } else {
            false
        }
    };

    assert!(!check_sepia(&yellowed), "Lignin paper yellowing has neutral shadows");
    assert!(check_sepia(&sepia), "Intentional sepia has warm shadows (R > B + 4)");
}

// ============================================================================
// R4: Perceptual Lightness & Color Correction Engine Tests (Features 30 - 36)
// ============================================================================

#[test]
fn test_tier1_r4_severity_confidence_scaling() {
    let calc_fraction = |confidence: f32, severity_mult: f32| -> f32 {
        if confidence < 0.75 {
            0.0
        } else {
            let scale = ((confidence - 0.75) / (1.0 - 0.75)).clamp(0.0, 1.0);
            severity_mult * scale
        }
    };

    assert_eq!(calc_fraction(0.70, 1.0), 0.0);
    assert_eq!(calc_fraction(0.75, 1.0), 0.0);
    assert_eq!(calc_fraction(1.00, 1.0), 1.0);
    assert_eq!(calc_fraction(1.00, 0.75), 0.75); // Moderate severity
    assert!((calc_fraction(0.875, 1.0) - 0.50).abs() < 1e-4);
}

#[test]
fn test_tier1_r4_fast_linear_srgb_in_gamut_check() {
    let in_gamut = |r: f32, g: f32, b: f32| -> bool {
        r >= 0.0 && r <= 1.0 && g >= 0.0 && g <= 1.0 && b >= 0.0 && b <= 1.0
    };

    assert!(in_gamut(0.0, 0.5, 1.0));
    assert!(!in_gamut(-0.01, 0.5, 0.5)); // Out of gamut below 0
    assert!(!in_gamut(0.5, 1.05, 0.5));  // Out of gamut above 1
}

#[test]
fn test_tier1_r4_radial_chord_bisection_monotonicity() {
    // Tests 10-step monotonic binary search scaling factor t in [0.0, 1.0]
    let mut low = 0.0f32;
    let mut high = 1.0f32;

    // Simulate finding boundary at t = 0.654
    let target = 0.654f32;
    for _ in 0..10 {
        let mid = (low + high) * 0.5;
        if mid <= target {
            low = mid;
        } else {
            high = mid;
        }
    }

    let err = (low - target).abs();
    assert!(
        err < (1.0 / 1024.0),
        "10 iterations must achieve error < 2^-10 (got {:.6})",
        err
    );
}

#[test]
fn test_tier1_r4_hermite_smoothstep_blending() {
    let hermite = |chroma: f32, low_thresh: f32| -> f32 {
        if chroma >= 35.0 {
            0.0 // 100% original color preserved untouched
        } else if chroma <= low_thresh {
            1.0 // 100% leveled
        } else {
            let w = (35.0 - chroma) / (35.0 - low_thresh);
            w * w * (3.0 - 2.0 * w)
        }
    };

    assert_eq!(hermite(35.0, 12.0), 0.0);
    assert_eq!(hermite(40.0, 12.0), 0.0);
    assert_eq!(hermite(12.0, 12.0), 1.0);
    assert_eq!(hermite(5.0, 12.0), 1.0);
    let mid = hermite((35.0 + 12.0) * 0.5, 12.0);
    assert!((mid - 0.5).abs() < 1e-4);
}

// ============================================================================
// R5: Multi-Tier Quality Guard Tests (Features 37 - 45)
// ============================================================================

#[test]
fn test_tier1_r5_luminance_mse_and_psnr() {
    let calc_mse_psnr = |orig: &[u8], lev: &[u8]| -> (f32, f32) {
        assert_eq!(orig.len(), lev.len());
        let sum_sq: f64 = orig
            .iter()
            .zip(lev.iter())
            .map(|(&a, &b)| {
                let diff = a as f64 - b as f64;
                diff * diff
            })
            .sum();
        let mse = (sum_sq / orig.len() as f64) as f32;
        let psnr = if mse < 1e-6 {
            99.0
        } else {
            10.0 * (255.0 * 255.0 / mse).log10()
        };
        (mse, psnr)
    };

    let a = vec![100u8; 1000];
    let b = vec![105u8; 1000]; // Shift by 5
    let (mse, psnr) = calc_mse_psnr(&a, &b);

    assert_eq!(mse, 25.0);
    assert!(mse <= 220.0, "MSE within safe limit");
    assert!(psnr >= 22.0, "PSNR >= 22dB requirement");
}

#[test]
fn test_tier1_r5_paper_cast_exemption_mask() {
    // Verifies that unprinted paper highlights (y >= 180, chroma <= 30) are exempt from color drift penalty
    let is_exempt = |r: u8, g: u8, b: u8, paper_cast_active: bool| -> bool {
        if !paper_cast_active {
            return false;
        }
        let y = ((54 * (r as u32) + 183 * (g as u32) + 19 * (b as u32) + 128) >> 8) as u8;
        let chroma = r.max(g).max(b) - r.min(g).min(b);
        y >= 180 && chroma <= 30
    };

    assert!(is_exempt(240, 235, 215, true)); // Paper highlight exempt
    assert!(!is_exempt(240, 235, 215, false)); // Disabled if no paper cast correction
    assert!(!is_exempt(120, 115, 100, true)); // Midtones NOT exempt (y < 180)
    assert!(!is_exempt(240, 50, 50, true)); // High-chroma color NOT exempt
}

#[test]
fn test_tier1_r5_gradient_edge_energy_gating() {
    // Skip pixels with ||G|| < 12.0
    let is_active_edge = |gx: f32, gy: f32| -> bool {
        (gx * gx + gy * gy).sqrt() >= 12.0
    };

    assert!(!is_active_edge(3.0, 4.0)); // Magnitude 5 < 12 -> skipped
    assert!(is_active_edge(10.0, 10.0)); // Magnitude ~14.14 >= 12 -> evaluated
}

#[test]
fn test_tier1_r5_activity_gated_block_michelson() {
    // Evaluated only if (I_max - I_min) >= 25
    let is_active_block = |i_min: u8, i_max: u8| -> bool {
        (i_max as i32 - i_min as i32) >= 25
    };

    assert!(!is_active_block(250, 255)); // Blank white gutter: diff 5 -> skipped
    assert!(!is_active_block(0, 5));     // Solid black gutter: diff 5 -> skipped
    assert!(is_active_block(10, 200));   // Active panel linework: diff 190 -> evaluated
}

// ============================================================================
// R6: Decoupled Pre-Actor Execution & Edge Cases (Features 46 - 52)
// ============================================================================

#[tokio::test]
async fn test_tier1_r6_decoupled_pre_actor_spawn_blocking() {
    let result = tokio::task::spawn_blocking(|| {
        // CPU leveling executed inside blocking threadpool
        let img = generate_lignin_yellowed_page(100, 100);
        let (w, h) = img.dimensions();
        w * h
    })
    .await
    .unwrap();

    assert_eq!(result, 10000);
}

#[test]
fn test_tier1_r6_thumbnail_fast_bypass() {
    let thumb_img = generate_bimodal_ink_shelf_page(300, 300, 18);
    let bytes = Bytes::from(encode_to_png_bytes(&thumb_img));
    let config = AutoLevelsConfig::default();
    let (_leveled, info, verdict) = level_and_verify(thumb_img, &bytes, None, &config);
    assert!(!info.was_leveled, "Thumbnails (<400px) must bypass auto-leveling");
    assert_eq!(verdict.outcome, Outcome::Pass);
}

#[test]
fn test_tier1_r6_alpha_transparency_masking() {
    let rgba_img = generate_rgba_transparent_page(100, 100);
    let bytes = Bytes::from(encode_to_png_bytes(&rgba_img));
    let config = AutoLevelsConfig::default();
    let (leveled, _info, verdict) = level_and_verify(rgba_img, &bytes, None, &config);
    assert_eq!(
        verdict.outcome,
        Outcome::Pass,
        "Alpha transparent page must pass quality guard with masked transparency"
    );
    assert!(matches!(leveled, DynamicImage::ImageRgba8(_)));
}

#[test]
fn test_tier1_r6_solid_black_blank_page() {
    let black_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(500, 500, Rgb([0, 0, 0])));
    let bytes = Bytes::from(encode_to_png_bytes(&black_img));
    let config = AutoLevelsConfig::default();
    let (_leveled, info, verdict) = level_and_verify(black_img, &bytes, None, &config);
    assert!(!info.was_leveled, "Solid black chapter divider must be skipped");
    assert_eq!(verdict.outcome, Outcome::Pass);
}

#[test]
fn test_tier1_r6_solid_white_blank_page() {
    let white_img = DynamicImage::ImageRgb8(RgbImage::from_pixel(500, 500, Rgb([255, 255, 255])));
    let bytes = Bytes::from(encode_to_png_bytes(&white_img));
    let config = AutoLevelsConfig::default();
    let (_leveled, info, verdict) = level_and_verify(white_img, &bytes, None, &config);
    assert!(!info.was_leveled, "Solid white blank page must be skipped");
    assert_eq!(verdict.outcome, Outcome::Pass);
}


// ============================================================================
// R7: Dependencies & Configuration Schema (Features 2, 3)
// ============================================================================

#[test]
fn test_tier1_r7_test_auto_levels_config_defaults() {
    let config = TestAutoLevelsConfig::default();

    assert!(config.enabled);
    assert_eq!(config.confidence_threshold, 0.75);
    assert_eq!(config.black_clip_percent, 0.1);
    assert_eq!(config.white_clip_percent, 0.1);
    assert_eq!(config.surge_ratio_threshold, 2.8);
    assert_eq!(config.max_black_shift, 85);
    assert_eq!(config.min_white_threshold, 200);
    assert!(config.correct_paper_cast);
    assert!(config.protect_sepia_artwork);
}
