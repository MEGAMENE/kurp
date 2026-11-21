//! Tier 4: End-to-End Real-CUGAN CPU + Auto-Levels + BookProfile Sampling + Concurrency Integration

mod common;
use common::*;

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::http::{StatusCode, header};
use kurp::clients::komga_client::ForwardAuthHeaders;
use kurp::upscaler::auto_levels::{PageClassification, level_and_verify_with_context, rec709_y};
use kurp::upscaler::avif::decode_image_with_metadata;
use kurp::upscaler::upscaler::UpscaleEngine;

#[tokio::test]
async fn test_tier4_e2e_komga_realcugan_cpu_auto_levels_and_lossless_webp() {
    let komga = MockKomgaServer::start().await;
    komga.set_book_tags("manga-vol-1", "manga-series-1", vec!["upscale"]);

    // 400x400 M1 page (>= 400px so auto-levels executes genuinely, then 2x CPU Real-CUGAN -> 800x800 Luma8 WebP)
    let m1 = generate_m1_spike_page(400, 400, 0.012, 42);
    let png_bytes = encode_to_png_bytes(&m1);
    komga.set_default_book_pages(
        "manga-vol-1",
        5,
        MockPageEntry {
            bytes: png_bytes,
            content_type: "image/png".to_string(),
            etag: Some("\"manga-p1-v1\"".to_string()),
            content_disposition: Some("inline; filename=\"001.png\"".to_string()),
        },
    );

    let mut cfg = create_cpu_upscale_config(&komga.base_url);
    cfg.upscale_tag = Some("upscale".to_string());
    let expected_suffix = format!("-kurp-{:016x}", cfg.config_hash());

    let upscaler = Arc::new(UpscaleEngine::init(&cfg).await.expect("CPU upscaler init"));
    let kurp = TestKurpServer::start(cfg, upscaler.clone()).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("{}/api/v1/books/manga-vol-1/pages/1", kurp.base_url))
        .header("X-API-Key", "reader-key")
        .send()
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers().get(header::CONTENT_TYPE).unwrap(), "image/webp");
    assert_eq!(resp.headers().get(header::VARY).unwrap(), "Accept");
    assert_eq!(
        resp.headers().get(header::ETAG).unwrap().to_str().unwrap(),
        format!("\"manga-p1-v1{expected_suffix}\"")
    );
    assert_eq!(
        resp.headers().get(header::CONTENT_DISPOSITION).unwrap().to_str().unwrap(),
        "inline; filename=\"001.webp\""
    );

    let webp_bytes = resp.bytes().await.unwrap();
    let decoded = decode_image_with_metadata(&webp_bytes, None).expect("WebP decode must succeed");
    assert_eq!((decoded.image.width(), decoded.image.height()), (800, 800));

    // Sample paper background and verify monochrome restoration produced clean bright paper
    let rgb = decoded.image.to_rgb8();
    let paper = rgb.get_pixel(790, 790);
    assert!(rec709_y(paper[0], paper[1], paper[2]) >= 245);

    upscaler.shutdown();
}

#[tokio::test]
async fn test_tier4_book_profile_12_of_200_even_sampling_and_outlier_guard() {
    let komga = MockKomgaServer::start().await;
    komga.set_book_tags("vol-200", "series-200", vec!["upscale"]);

    // Register 200 pages with M4 aged paper (black floor ~26)
    let page_bytes = encode_to_png_bytes(&generate_m4_aged_paper_page(400, 400));
    komga.set_default_book_pages("vol-200", 200, MockPageEntry::png(page_bytes));

    let mut cfg = create_cpu_upscale_config(&komga.base_url);
    cfg.auto_levels.book_profile.sample_pages = 12;
    cfg.auto_levels.book_profile.parallel_page_fetches = 4;

    let kurp = TestKurpServer::start(cfg.clone(), Arc::new(UpscaleEngine::new_empty())).await;
    let auth = ForwardAuthHeaders::default();

    let profile = kurp
        .state
        .book_profile_manager
        .get_or_build_profile("vol-200", &auth, &cfg.auto_levels)
        .await;

    assert_eq!(profile.total_book_pages, 200);
    assert_eq!(profile.sampled_pages, 12);
    assert_eq!(profile.monochrome_group.page_count, 12);
    assert!(
        (24..=28).contains(&profile.monochrome_group.median_black_floor),
        "Expected median black floor ~26, got {}",
        profile.monochrome_group.median_black_floor
    );
    assert_eq!(
        komga.page_fetch_count(),
        12,
        "BookProfile must fetch ONLY 12 evenly-spaced pages out of 200 (never 200!)"
    );

    // BookProfile Outlier Sanity Guard clamps an anomalous page with black floor ~68 to <= median + 20
    let anomalous = generate_m1_spike_page(450, 450, 0.012, 68);
    let (_out, info, _) = level_and_verify_with_context(
        anomalous,
        Some("vol-200"),
        Some(99),
        Some(&profile),
        &cfg.auto_levels,
    );
    assert_eq!(info.classification, PageClassification::Monochrome);
    assert!(
        info.black_point <= profile.monochrome_group.median_black_floor.saturating_add(20),
        "Outlier guard must clamp black_point ({}) <= median({}) + 20",
        info.black_point,
        profile.monochrome_group.median_black_floor
    );
}

#[tokio::test]
async fn test_tier4_book_profile_joinset_timeout_cancellation_and_empty_pages_nocache() {
    let komga = MockKomgaServer::start().await;
    komga.set_book_tags("slow-vol", "s-slow", vec!["upscale"]);
    let page_bytes = encode_to_png_bytes(&generate_m4_aged_paper_page(400, 400));
    komga.set_default_book_pages("slow-vol", 20, MockPageEntry::png(page_bytes));
    komga.set_page_delay(Duration::from_secs(5));

    let mut cfg = create_cpu_upscale_config(&komga.base_url);
    cfg.auto_levels.book_profile.timeout_seconds = 1;
    cfg.auto_levels.book_profile.sample_pages = 10;

    let kurp = TestKurpServer::start(cfg.clone(), Arc::new(UpscaleEngine::new_empty())).await;
    let auth = ForwardAuthHeaders::default();

    // Timeout must abort all in-flight JoinSet tasks promptly (~1s, never 5s!)
    let t0 = Instant::now();
    let timed_out = kurp
        .state
        .book_profile_manager
        .get_or_build_profile("slow-vol", &auth, &cfg.auto_levels)
        .await;
    let elapsed = t0.elapsed();
    assert!(
        elapsed < Duration::from_secs(3),
        "JoinSet timeout must abort promptly near 1s (took {:?})",
        elapsed
    );
    assert_eq!(timed_out.sampled_pages, 0);

    // Empty page list must NOT be permanently cached in profile_cache:
    komga.set_page_delay(Duration::ZERO);
    *komga.state.return_empty_pages.write().unwrap() = true;
    let empty_prof = kurp
        .state
        .book_profile_manager
        .get_or_build_profile("empty-then-filled", &auth, &cfg.auto_levels)
        .await;
    assert_eq!(empty_prof.sampled_pages, 0);

    *komga.state.return_empty_pages.write().unwrap() = false;
    komga.set_book_tags("empty-then-filled", "s-ef", vec!["upscale"]);
    let m4_bytes = encode_to_png_bytes(&generate_m4_aged_paper_page(400, 400));
    komga.set_default_book_pages("empty-then-filled", 5, MockPageEntry::png(m4_bytes));

    let recovered = kurp
        .state
        .book_profile_manager
        .get_or_build_profile("empty-then-filled", &auth, &cfg.auto_levels)
        .await;
    assert_eq!(
        recovered.sampled_pages, 5,
        "Empty page list must not poison profile_cache once pages are indexed"
    );
}

#[tokio::test]
async fn test_tier4_upscale_false_bypass_and_size_threshold_passthrough() {
    let komga = MockKomgaServer::start().await;
    komga.set_book_tags("book-pass", "series-pass", vec!["upscale"]);
    let orig_bytes = encode_to_png_bytes(&generate_m1_spike_page(400, 400, 0.012, 42));
    komga.set_page(
        "book-pass",
        1,
        MockPageEntry {
            bytes: orig_bytes.clone(),
            content_type: "image/png".to_string(),
            etag: Some("\"orig-pass\"".to_string()),
            content_disposition: None,
        },
    );

    // 1. upscale: false bypasses BookProfile, auto_levels, and GPU completely
    let mut cfg_off = create_cpu_upscale_config(&komga.base_url);
    cfg_off.upscale = false;
    let kurp_off = TestKurpServer::start(cfg_off, Arc::new(UpscaleEngine::new_empty())).await;
    let client = reqwest::Client::new();

    komga.clear_recorded();
    let r_off = client
        .get(format!("{}/api/v1/books/book-pass/pages/1", kurp_off.base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(r_off.status(), StatusCode::OK);
    assert_eq!(r_off.headers().get(header::ETAG).unwrap(), "\"orig-pass\"");
    assert_eq!(r_off.bytes().await.unwrap(), orig_bytes);
    assert!(
        !komga
            .recorded_requests()
            .iter()
            .any(|r| r.path.ends_with("/pages")),
        "upscale: false must skip BookProfile"
    );

    // 2. size_threshold_enabled bypasses oversized PNG without re-encoding
    let mut cfg_thresh = create_cpu_upscale_config(&komga.base_url);
    cfg_thresh.size_threshold_enabled = true;
    cfg_thresh.size_threshold_png = 0; // force bypass on any >=1KB PNG
    cfg_thresh.auto_levels.enabled = false;
    let kurp_thresh =
        TestKurpServer::start(cfg_thresh, Arc::new(UpscaleEngine::new_empty())).await;

    let r_thresh = client
        .get(format!("{}/api/v1/books/book-pass/pages/1", kurp_thresh.base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(r_thresh.status(), StatusCode::OK);
    assert_eq!(r_thresh.headers().get(header::CONTENT_TYPE).unwrap(), "image/png");
    assert_eq!(r_thresh.bytes().await.unwrap(), orig_bytes);
}

#[tokio::test]
async fn test_tier4_parallel_page1_body_drain_and_pipeline_backpressure() {
    let komga = MockKomgaServer::start().await;
    komga.set_book_tags("burst-vol", "s-burst", vec!["upscale"]);
    let page_bytes = encode_to_png_bytes(&generate_m1_spike_page(64, 64, 0.012, 42));
    komga.set_default_book_pages("burst-vol", 6, MockPageEntry::png(page_bytes));

    let mut cfg = create_cpu_upscale_config(&komga.base_url);
    cfg.max_concurrent_pipelines = 2;
    cfg.auto_levels.book_profile.sample_pages = 6;
    cfg.auto_levels.book_profile.parallel_page_fetches = 3;

    let upscaler = Arc::new(UpscaleEngine::init(&cfg).await.expect("CPU upscaler init"));
    let kurp = TestKurpServer::start(cfg, upscaler.clone()).await;
    let client = reqwest::Client::new();

    // Concurrent requests across all 6 pages share BookProfile coalescing + bounded pipeline semaphore
    let mut handles = Vec::new();
    for p in 1..=6 {
        let c = client.clone();
        let url = format!("{}/api/v1/books/burst-vol/pages/{}", kurp.base_url, p);
        handles.push(tokio::spawn(async move {
            let r = c.get(&url).send().await.unwrap();
            assert_eq!(r.status(), StatusCode::OK);
            let b = r.bytes().await.unwrap();
            let dec = image::load_from_memory(&b).unwrap();
            assert_eq!((dec.width(), dec.height()), (128, 128));
        }));
    }

    for h in handles {
        h.await.unwrap();
    }

    upscaler.shutdown();
}
