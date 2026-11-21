//! Tier 2: Mathematical Boundaries, ISOBMFF Overflow Fuzzing, Quality Guards & Latency Budget

mod common;
use common::*;

use std::time::Instant;

use image::{DynamicImage, GrayImage, ImageBuffer, Luma, Rgb, RgbImage};
use kurp::config::app_config::AutoLevelsConfig;
use kurp::upscaler::auto_levels::{
    CorrectionPlan, DefectType, Outcome, PageClassification, PaperCastField,
    build_monotonic_tone_lut, build_monotonic_tone_lut_gamma, detect_elevated_black_floor,
    level_and_verify, quality_tier_1, quality_tier_2, quality_tier_3,
};
use kurp::upscaler::avif::{
    decode_image_with_metadata, display_p3_icc, is_avif_bytes, parse_avif_color_info,
    probe_image_dimensions, rec2020_icc, rec2020_srgb_icc, synthesize_icc_from_cicp,
};

fn make_box(fourcc: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let total = (8 + payload.len()) as u32;
    let mut out = Vec::with_capacity(total as usize);
    out.extend_from_slice(&total.to_be_bytes());
    out.extend_from_slice(fourcc);
    out.extend_from_slice(payload);
    out
}

fn make_ftyp(major: &[u8; 4], compat: &[&[u8; 4]]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(major);
    payload.extend_from_slice(&0u32.to_be_bytes());
    for c in compat {
        payload.extend_from_slice(*c);
    }
    make_box(b"ftyp", &payload)
}

fn make_nclx(p: u16, t: u16, m: u16, full: bool) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(b"nclx");
    payload.extend_from_slice(&p.to_be_bytes());
    payload.extend_from_slice(&t.to_be_bytes());
    payload.extend_from_slice(&m.to_be_bytes());
    payload.push(if full { 0x80 } else { 0x00 });
    make_box(b"colr", &payload)
}

fn make_colr_icc(tag: &[u8; 4], icc: &[u8]) -> Vec<u8> {
    let mut payload = Vec::new();
    payload.extend_from_slice(tag);
    payload.extend_from_slice(icc);
    make_box(b"colr", &payload)
}

#[test]
fn test_tier2_avif_64bit_largesize_overflow_fuzzing_and_malformed_safety() {
    let ftyp = make_ftyp(b"avif", &[b"mif1"]);
    assert!(is_avif_bytes(&ftyp));

    // Fuzz crafted 64-bit largesize (size == 1) and invalid box sizes: MUST finish immediately without hang/panic
    let edge_sizes: [u64; 7] = [
        u64::MAX,
        u64::MAX - 7,
        u64::MAX - 15,
        usize::MAX as u64,
        0,
        8,
        15,
    ];
    for &largesize in &edge_sizes {
        let mut buf = ftyp.clone();
        buf.extend_from_slice(&1u32.to_be_bytes());
        buf.extend_from_slice(b"meta");
        buf.extend_from_slice(&largesize.to_be_bytes());
        buf.extend_from_slice(&[0u8; 32]);

        let t0 = Instant::now();
        assert!(parse_avif_color_info(&buf).is_none());
        assert!(t0.elapsed().as_millis() < 50);
        assert!(decode_image_with_metadata(&buf, None).is_err());
        assert!(probe_image_dimensions(&buf, None).is_err());
    }

    // Crafted invalid 32-bit box size (< 8 and overflowing) + nested moov recursion beyond depth 16
    let mut small_box = ftyp.clone();
    small_box.extend_from_slice(&4u32.to_be_bytes());
    small_box.extend_from_slice(b"meta");
    assert!(parse_avif_color_info(&small_box).is_none());

    // Deeply nested trak inside moov (> MAX_BOX_DEPTH=16) must stop safely without stack overflow
    let mut nested = make_colr_icc(b"rICC", b"DEEP_ICC");
    for _ in 0..24 {
        nested = make_box(b"trak", &nested);
    }
    let mut deep_avis = make_ftyp(b"avis", &[]);
    deep_avis.extend_from_slice(&make_box(b"moov", &nested));
    assert!(parse_avif_color_info(&deep_avis).is_none());

    // Non-AVIF HEIC mif1 brand must NOT be misclassified as AVIF
    let heic_only = make_ftyp(b"mif1", &[b"heic"]);
    assert!(!is_avif_bytes(&heic_only));
}

#[test]
fn test_tier2_avif_hdr_pq_hlg_non_corruption_and_multi_colr_precedence() {
    // HDR PQ (t=16), HLG (t=18), and Linear (t=8) must NEVER receive SDR ICC profiles
    for transfer in [16u16, 18, 8] {
        assert!(synthesize_icc_from_cicp(9, transfer).is_none());
        assert!(synthesize_icc_from_cicp(12, transfer).is_none());

        let mut avif = make_ftyp(b"avif", &[]);
        let ipco = make_box(b"ipco", &make_nclx(9, transfer, 9, false));
        let iprp = make_box(b"iprp", &ipco);
        let mut meta_payload = vec![0u8; 4];
        meta_payload.extend_from_slice(&iprp);
        avif.extend_from_slice(&make_box(b"meta", &meta_payload));

        let info = parse_avif_color_info(&avif).expect("nclx should parse");
        assert!(info.icc_profile.is_none());
        assert_eq!(info.is_hdr, matches!(transfer, 16 | 18));
    }

    // Valid SDR synthesis checks
    assert_eq!(
        synthesize_icc_from_cicp(12, 13).as_deref(),
        Some(display_p3_icc())
    );
    assert_eq!(
        synthesize_icc_from_cicp(9, 14).as_deref(),
        Some(rec2020_icc())
    );
    assert_eq!(
        synthesize_icc_from_cicp(9, 13).as_deref(),
        Some(rec2020_srgb_icc())
    );

    // Multi-colr precedence + pitm/ipma primary scoping:
    // prop 1: thumbnail prof (item 2)
    // prop 2: primary nclx (item 1)
    // prop 3: primary prof (item 1, AFTER nclx -> must STILL override nclx!)
    let thumb_icc = b"THUMB_ICC_IGNORED";
    let primary_icc = b"PRIMARY_PROF_ICC_WINS";
    let mut ipco_payload = Vec::new();
    ipco_payload.extend_from_slice(&make_colr_icc(b"prof", thumb_icc));
    ipco_payload.extend_from_slice(&make_nclx(12, 13, 6, true));
    ipco_payload.extend_from_slice(&make_colr_icc(b"prof", primary_icc));
    let ipco = make_box(b"ipco", &ipco_payload);

    let mut ipma_payload = vec![0, 0, 0, 0];
    ipma_payload.extend_from_slice(&2u32.to_be_bytes());
    ipma_payload.extend_from_slice(&2u16.to_be_bytes());
    ipma_payload.push(1);
    ipma_payload.push(1);
    ipma_payload.extend_from_slice(&1u16.to_be_bytes());
    ipma_payload.push(2);
    ipma_payload.push(2);
    ipma_payload.push(3);
    let ipma = make_box(b"ipma", &ipma_payload);

    let mut iprp_payload = Vec::new();
    iprp_payload.extend_from_slice(&ipco);
    iprp_payload.extend_from_slice(&ipma);
    let iprp = make_box(b"iprp", &iprp_payload);

    let mut pitm_payload = vec![0, 0, 0, 0];
    pitm_payload.extend_from_slice(&1u16.to_be_bytes());
    let pitm = make_box(b"pitm", &pitm_payload);

    let mut meta_payload = vec![0u8; 4];
    meta_payload.extend_from_slice(&pitm);
    meta_payload.extend_from_slice(&iprp);

    let mut avif = make_ftyp(b"avif", &[]);
    avif.extend_from_slice(&make_box(b"meta", &meta_payload));
    let info = parse_avif_color_info(&avif).expect("should parse primary colr");
    assert_eq!(info.icc_profile.as_deref(), Some(primary_icc.as_slice()));

    // Animated AVIF (avis) moov/.../stsd/av01/colr
    let mut av01_payload = vec![0u8; 78];
    av01_payload.extend_from_slice(&make_nclx(12, 13, 6, true));
    let mut stsd_payload = vec![0u8; 4];
    stsd_payload.extend_from_slice(&1u32.to_be_bytes());
    stsd_payload.extend_from_slice(&make_box(b"av01", &av01_payload));
    let moov = make_box(
        b"moov",
        &make_box(
            b"trak",
            &make_box(
                b"mdia",
                &make_box(b"minf", &make_box(b"stbl", &make_box(b"stsd", &stsd_payload))),
            ),
        ),
    );
    let mut avis = make_ftyp(b"avis", &[]);
    avis.extend_from_slice(&moov);
    let avis_info = parse_avif_color_info(&avis).expect("should parse avis moov colr");
    assert_eq!(avis_info.icc_profile.as_deref(), Some(display_p3_icc()));
}

#[test]
fn test_tier2_fritsch_carlson_zero_inversion_nan_and_gamma_monotonicity() {
    for (lo, hi) in [(f32::NAN, 255.0), (0.0, f32::INFINITY), (f32::NEG_INFINITY, 200.0)] {
        let lut = build_monotonic_tone_lut(lo, hi);
        for (i, &v) in lut.iter().enumerate() {
            assert_eq!(v, i as u8, "non-finite knots must return identity LUT");
        }
    }

    assert!(build_monotonic_tone_lut(5.0, 255.0)[5] <= 5);
    assert!(build_monotonic_tone_lut(0.0, 252.0)[252] >= 252);
    let both = build_monotonic_tone_lut(6.0, 250.0);
    assert!(both[250] - both[6] > 244);

    for &gamma in &[0.2f32, 0.6, 1.0, 1.8, 5.0] {
        for &hi in &[90.0f32, 128.0, 224.0, 255.0] {
            for &lo in &[0.0f32, 5.0, 26.0, 65.0] {
                if hi - lo >= 15.0 {
                    let lut = build_monotonic_tone_lut_gamma(lo, hi, gamma);
                    for i in 0..255 {
                        assert!(lut[i + 1] >= lut[i]);
                    }
                }
            }
        }
    }
}

#[test]
fn test_tier2_shadow_detector_bounds_3_5pct_cap_and_85_ink_ceiling() {
    let config = AutoLevelsConfig::default();
    let total = 100_000u32;

    // 1. M_zero >= 0.035 strictly vetoes black floor elevation
    let mut hist_cap = [0u32; 256];
    hist_cap[0] = 3500; // 3.5%
    hist_cap[22] = 10_000;
    hist_cap[255] = total - hist_cap[0] - hist_cap[22];
    assert!(
        detect_elevated_black_floor(&hist_cap, PageClassification::Monochrome, &config).is_none()
    );

    // 2. Ink peak at Y=74 (> 65, <= 85) MUST be reachable (M_ink integrates [3..=85])
    let mut hist_74 = [0u32; 256];
    hist_74[0] = 800; // 0.8% spike
    hist_74[74] = 8000; // 8.0% washed ink in [66..=85]
    hist_74[250] = total - hist_74[0] - hist_74[74];
    let det_74 = detect_elevated_black_floor(&hist_74, PageClassification::Monochrome, &config)
        .expect("Washed ink peak at Y=74 (<= 85) must be reachable and validated");
    assert!(matches!(
        det_74,
        DefectType::ElevatedBlackFloor { floor_value, is_bimodal: true } if (70..=76).contains(&floor_value)
    ));

    // 3. No post-peak valley (continuous midtone ramp) must reject shelf
    let mut hist_no_valley = [0u32; 256];
    hist_no_valley[0] = 500;
    hist_no_valley[20..=145].fill(500);
    hist_no_valley[255] = total - hist_no_valley.iter().sum::<u32>();
    assert!(
        detect_elevated_black_floor(&hist_no_valley, PageClassification::Monochrome, &config)
            .is_none()
    );
}

#[test]
fn test_tier2_quality_guards_genuine_rejections_and_2mp_untouched_rollback() {
    let config = AutoLevelsConfig::default();

    // Genuine Tier 1 rejection: inverted luminance
    let (w, h) = (128u32, 128u32);
    let orig_grad =
        DynamicImage::ImageLuma8(ImageBuffer::from_fn(w, h, |x, _| Luma([(x * 2).min(255) as u8])));
    let inv_grad = DynamicImage::ImageLuma8(ImageBuffer::from_fn(w, h, |x, _| {
        Luma([255u8.saturating_sub((x * 2).min(255) as u8)])
    }));
    assert_eq!(
        quality_tier_1(&orig_grad, &inv_grad, &CorrectionPlan::default(), &config).outcome,
        Outcome::Reject
    );

    // Genuine Tier 2 rejection: bleached pastel sky even when paper cast is active
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
    assert_eq!(
        quality_tier_2(
            &sky_orig,
            &sky_bleached,
            PageClassification::Mixed,
            &cast_plan,
            &config
        )
        .outcome,
        Outcome::Reject
    );

    // Genuine Tier 3 rejection vs speech-bubble whitening exemption
    let mid_orig = DynamicImage::ImageLuma8(ImageBuffer::from_fn(w, h, |x, y| {
        Luma([if (x + y).is_multiple_of(2) { 90 } else { 150 }])
    }));
    let mid_crushed = DynamicImage::ImageLuma8(ImageBuffer::from_fn(w, h, |x, y| {
        Luma([if (x + y).is_multiple_of(2) { 118 } else { 122 }])
    }));
    assert_eq!(
        quality_tier_3(&mid_orig, &mid_crushed, &CorrectionPlan::default(), &config).outcome,
        Outcome::Reject
    );

    let bubble_orig = DynamicImage::ImageLuma8(ImageBuffer::from_fn(w, h, |x, _| {
        Luma([if x % 32 < 16 { 224 } else { 255 }])
    }));
    let bubble_white = DynamicImage::ImageLuma8(ImageBuffer::from_fn(w, h, |x, _| {
        Luma([if x % 32 < 16 { 253 } else { 255 }])
    }));
    let wp_plan = CorrectionPlan {
        white_point: 224,
        ..CorrectionPlan::default()
    };
    assert_eq!(
        quality_tier_3(&bubble_orig, &bubble_white, &wp_plan, &config).outcome,
        Outcome::Pass
    );

    // >= 2MP Rollback MUST preserve full resolution bit-exact (NO 32x32 mosaic corruption!)
    let _guard = RAYON_BENCH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let m4_2mp = generate_m4_aged_paper_page(2000, 1200);
    let orig_raw = m4_2mp.as_rgb8().unwrap().as_raw().clone();
    let mut reject_cfg = AutoLevelsConfig::default();
    reject_cfg.quality_guards.tier2.ssim_threshold = 1.01;
    let (rolled_back, info, verdict) = level_and_verify(m4_2mp, None, &reject_cfg);
    assert!(!info.was_leveled);
    assert_eq!(verdict.outcome, Outcome::Reject);
    assert!(verdict.attenuation_attempted);
    assert_eq!((rolled_back.width(), rolled_back.height()), (2000, 1200));
    assert_eq!(rolled_back.as_rgb8().unwrap().as_raw(), &orig_raw);
}

static RAYON_BENCH_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn test_tier2_thumbnail_399_vs_400_boundary() {
    let _guard = RAYON_BENCH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let config = AutoLevelsConfig::default();

    // Neutral monochrome <400px thumbnail: bypasses auto-levels while preserving `is_monochrome = true`.
    let mono_399 = generate_m3_noisy_mono_page(399, 399);
    let (_, mono_info_399, _) = level_and_verify(mono_399, None, &config);
    assert!(!mono_info_399.was_leveled);
    assert!(mono_info_399.is_monochrome);

    // Tinted M4 <400px thumbnail: bypasses auto-levels AND intentionally preserves `is_monochrome = false`
    // so post-upscale Luma8 folding never strips un-neutralized tint or sepia on small thumbnails.
    let m4_399 = generate_m4_aged_paper_page(399, 399);
    let (_, m4_info_399, _) = level_and_verify(m4_399, None, &config);
    assert!(!m4_info_399.was_leveled);
    assert!(!m4_info_399.is_monochrome);

    // At 400px (above `< 400 && < 400` early bypass), M4 levels and passes all quality tiers.
    let ok_400 = generate_m4_aged_paper_page(400, 400);
    let (_, info_400, verdict_400) = level_and_verify(ok_400, None, &config);
    assert!(info_400.was_leveled);
    assert_eq!(verdict_400.outcome, Outcome::Pass);
}

#[test]
fn test_tier2_6mp_latency_budget_veto_mono_mixed_luma8() {
    let _guard = RAYON_BENCH_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let config = AutoLevelsConfig::default();
    let (w, h) = (2000u32, 3000u32);

    // 1. Vetoed 6MP Color page (clone/drop outside timed block since production passes owned DynamicImage)
    let color_img = generate_color_cover_page(w, h);
    let mut veto_ms = f64::MAX;
    for _ in 0..5 {
        let input = color_img.clone();
        let t0 = Instant::now();
        let (out, info, _) = level_and_verify(input, None, &config);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        assert!(!info.was_leveled);
        drop(out);
        veto_ms = veto_ms.min(elapsed);
    }

    // 2. 6MP Monochrome M1
    let mono_img = generate_m1_spike_page(w, h, 0.012, 42);
    let mut mono_ms = f64::MAX;
    for _ in 0..5 {
        let input = mono_img.clone();
        let t0 = Instant::now();
        let (out, info, verdict) = level_and_verify(input, None, &config);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        assert!(info.was_leveled && verdict.outcome == Outcome::Pass);
        drop(out);
        mono_ms = mono_ms.min(elapsed);
    }

    // 3. 6MP Mixed (M2 Mixed RGBA)
    let mixed_img = generate_m2_mixed_rgba_page(w, h);
    let mut mixed_ms = f64::MAX;
    for _ in 0..5 {
        let input = mixed_img.clone();
        let t0 = Instant::now();
        let (out, info, verdict) = level_and_verify(input, None, &config);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        assert!(info.was_leveled && verdict.outcome == Outcome::Pass);
        drop(out);
        mixed_ms = mixed_ms.min(elapsed);
    }

    // 4. 6MP Luma8
    let total = (w * h) as usize;
    let mut luma_raw = vec![228u8; total];
    for slot in luma_raw.iter_mut().take((total * 14) / 100) {
        *slot = 28;
    }
    let luma_img = DynamicImage::ImageLuma8(GrayImage::from_raw(w, h, luma_raw).unwrap());
    let mut luma_ms = f64::MAX;
    for _ in 0..5 {
        let input = luma_img.clone();
        let t0 = Instant::now();
        let (out, info, verdict) = level_and_verify(input, None, &config);
        let elapsed = t0.elapsed().as_secs_f64() * 1000.0;
        assert!(info.was_leveled && verdict.outcome == Outcome::Pass);
        drop(out);
        luma_ms = luma_ms.min(elapsed);
    }

    println!(
        "[6MP Benchmark] Veto={veto_ms:.2}ms | Mono={mono_ms:.2}ms | Mixed={mixed_ms:.2}ms | Luma8={luma_ms:.2}ms"
    );
    #[cfg(not(debug_assertions))]
    {
        assert!(veto_ms < 1.5, "6MP veto {veto_ms:.2}ms >= 1.5ms");
        assert!(mono_ms < 15.0, "6MP mono {mono_ms:.2}ms >= 15.0ms");
        assert!(mixed_ms < 15.0, "6MP mixed {mixed_ms:.2}ms >= 15.0ms");
        assert!(luma_ms < 15.0, "6MP luma8 {luma_ms:.2}ms >= 15.0ms");
    }
}
