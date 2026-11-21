//! Tier 1: Genuine CV Restoration, Colorimetry & Image Codec Integration Tests
//!
//! All auto-levels tests use >= 400px images so none trivially pass via the `<400px` bypass.

mod common;
use common::*;

use std::io::Cursor;

use image::{
    DynamicImage, ImageBuffer, ImageDecoder, ImageEncoder, ImageFormat, ImageReader, Luma, LumaA,
    Rgb, RgbImage, Rgba, RgbaImage, codecs::jpeg::JpegEncoder, codecs::png::PngEncoder,
    metadata::Orientation,
};
use kurp::config::app_config::{AutoLevelsConfig, Format};
use kurp::upscaler::auto_levels::{
    LevelInfo, Outcome, PageClassification, analyze_page, is_image_grayscale, level_and_verify,
    rec709_y, rgb_to_lab_fast,
};
use kurp::upscaler::avif::{decode_image_with_metadata, display_p3_icc, probe_image_dimensions};
use kurp::upscaler::upscaler::{
    WEBP_MAX_DIMENSION, encode_image, encode_upscaled_output, fold_monochrome_post_upscale,
    should_fold_luma,
};

#[test]
fn test_tier1_m1_digital_spike_dead_zone_sparse_ink_and_clean_gray() {
    let config = AutoLevelsConfig::default();

    // M1 at 1.2% and 0.8% (eliminated dead zone)
    for &black_ratio in &[0.012f64, 0.008f64] {
        let img = generate_m1_spike_page(1000, 1500, black_ratio, 42);
        let (leveled, info, verdict) = level_and_verify(img, None, &config);

        assert!(
            info.was_leveled,
            "M1 (m_zero={black_ratio}) must be leveled"
        );
        assert_eq!(verdict.outcome, Outcome::Pass);
        assert!(
            !verdict.attenuation_attempted,
            "M1 (m_zero={black_ratio}) should pass without attenuation"
        );
        assert!(
            (40..=52).contains(&info.black_point),
            "Validated black_point={} out of [40, 52] for m_zero={black_ratio}",
            info.black_point
        );

        let lev_rgb = leveled.to_rgb8();
        // Find a pixel that started at washed-out drawing ink Y=45 (offset 3 into ink region)
        let total = 1000usize * 1500;
        let n_black = (total as f64 * black_ratio).round() as usize;
        let n_white = (total as f64 * 0.050).round() as usize;
        let idx = n_black + n_white + 3;
        let x = (idx % 1000) as u32;
        let y = (idx / 1000) as u32;
        let px = lev_rgb.get_pixel(x, y);
        let y_out = rec709_y(px[0], px[1], px[2]);
        assert!(
            y_out <= 8,
            "Washed ink Y=45 must restore to Y_out <= 8 (got {y_out})"
        );
    }

    // Sparse ink (1.0% Y=0 + 1.5% ink Y=25 + 5.0% midtone Y=75) must NOT overshoot to 75
    {
        let (w, h) = (1000u32, 1000u32);
        let total = (w * h) as usize;
        let mut raw = vec![250u8; total * 3];
        let n_black = total / 100;
        let n_ink = (total * 15) / 1000;
        let n_mid = total / 20;
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
        let (_leveled, info, _) = level_and_verify(img, None, &config);
        assert!(
            info.black_point <= 30,
            "Sparse ink black floor must stay near 25, not overshoot to 75 (got {})",
            info.black_point
        );
    }

    // Clean digital page (4% authentic black Y=0 + 6% dark-gray art Y=45 + 90% white) -> strict veto
    {
        let (w, h) = (1000u32, 1000u32);
        let total = (w * h) as usize;
        let mut raw = vec![255u8; total * 3];
        for i in 0..(total * 4 / 100) {
            raw[i * 3..i * 3 + 3].copy_from_slice(&[0, 0, 0]);
        }
        for i in (total * 4 / 100)..(total * 10 / 100) {
            raw[i * 3..i * 3 + 3].copy_from_slice(&[45, 45, 45]);
        }
        let orig = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, raw.clone()).unwrap());
        let (out, info, _) = level_and_verify(orig, None, &config);
        assert!(
            !info.was_leveled,
            "Clean digital art with 4% authentic black + 6% dark-gray tone must be vetoed"
        );
        assert_eq!(out.as_rgb8().unwrap().as_raw(), &raw);
    }
}

#[test]
fn test_tier1_m2_localized_yellow_margin_rgb_and_rgba_mixed_skin_preservation() {
    let config = AutoLevelsConfig::default();

    // 1. Monochrome M2 (Rgb8)
    let mono = generate_m2_localized_yellow_page(1000, 1500);
    let (leveled, info, verdict) = level_and_verify(mono, None, &config);
    assert_eq!(info.classification, PageClassification::Monochrome);
    assert!(info.was_leveled);
    assert_eq!(verdict.outcome, Outcome::Pass);

    let lev_rgb = leveled.to_rgb8();
    for &(px_x, px_y) in &[(20u32, 20u32), (500, 50), (200, 200)] {
        let p = lev_rgb.get_pixel(px_x, px_y);
        let (_, a, b) = rgb_to_lab_fast(p[0], p[1], p[2]);
        assert!(
            a.hypot(b) <= 2.0,
            "M2 pixel ({px_x},{px_y}) must neutralize to C* <= 2.0 (got {:.2})",
            a.hypot(b)
        );
    }

    // 2. Mixed M2 (Rgba8): yellowed outer margin + red logo + interior cream skin + semi-transparent edge
    let mixed_rgba = generate_m2_mixed_rgba_page(1000, 1500);
    let (lev_rgba, rgba_info, rgba_verdict) = level_and_verify(mixed_rgba, None, &config);
    assert_eq!(rgba_info.classification, PageClassification::Mixed);
    assert!(rgba_info.was_leveled);
    assert_eq!(rgba_verdict.outcome, Outcome::Pass);

    let out_rgba = lev_rgba.as_rgba8().expect("must remain Rgba8");
    for &(mx, my) in &[(20u32, 20u32), (500, 50)] {
        let p = out_rgba.get_pixel(mx, my);
        let (_, a, b) = rgb_to_lab_fast(p[0], p[1], p[2]);
        assert!(
            a.hypot(b) <= 2.0,
            "Mixed Rgba8 outer margin ({mx},{my}) must neutralize to C* <= 2.0 (got {:.2})",
            a.hypot(b)
        );
    }

    // Red logo protected by 3x3 dilated accent mask
    let logo_p = out_rgba.get_pixel(200, 300);
    assert!(logo_p[0] >= 205 && logo_p[1] <= 30 && logo_p[2] <= 30);
    assert_eq!(logo_p[3], 255);

    // Interior cream skin preserved (NEVER bleached to gray!)
    let skin_p = out_rgba.get_pixel(500, 700);
    let (_, sa, sb) = rgb_to_lab_fast(skin_p[0], skin_p[1], skin_p[2]);
    assert!(
        sa.hypot(sb) >= 10.0,
        "Interior cream skin must NOT be bleached (got {:?}, C*={:.2})",
        skin_p,
        sa.hypot(sb)
    );

    // Alpha preserved untouched on semi-transparent A=64 edge pixels (no A=127 halo)
    let edge_p = out_rgba.get_pixel(750, 675);
    assert_eq!(edge_p[3], 64, "Alpha channel must remain untouched");
}

#[test]
fn test_tier1_m3_noisy_mono_and_m4_aged_paper_correct_paper_cast_toggle() {
    let config = AutoLevelsConfig::default();

    // M3: noisy [5, 4, 3] ink (HSV S=40%, CIELAB C*=0.49) -> Monochrome -> Luma8 fold
    let m3 = generate_m3_noisy_mono_page(1000, 1500);
    let (m3_leveled, m3_info, m3_verdict) = level_and_verify(m3, None, &config);
    assert_eq!(m3_info.classification, PageClassification::Monochrome);
    assert!(m3_info.is_monochrome);
    assert!(!m3_info.paper_cast_preserved);
    assert_eq!(m3_verdict.outcome, Outcome::Pass);
    let m3_folded = fold_monochrome_post_upscale(m3_leveled, &m3_info, true);
    assert!(matches!(m3_folded, DynamicImage::ImageLuma8(_)));

    // M4: aged paper [232, 224, 198] + ink [26, 26, 26]
    let m4 = generate_m4_aged_paper_page(1000, 1500);

    // Default correct_paper_cast: true
    let (m4_leveled, m4_info, m4_verdict) = level_and_verify(m4.clone(), None, &config);
    assert!(m4_info.was_leveled);
    assert_eq!(m4_verdict.outcome, Outcome::Pass);
    assert!((24..=28).contains(&m4_info.black_point));
    assert!((220..=228).contains(&m4_info.white_point));
    assert!(!m4_info.paper_cast_preserved);
    let m4_rgb = m4_leveled.as_rgb8().unwrap();
    let ink_px = m4_rgb.get_pixel(0, 0);
    assert!(rec709_y(ink_px[0], ink_px[1], ink_px[2]) <= 6);
    let paper_px = m4_rgb.get_pixel(1, 0);
    assert!(rec709_y(paper_px[0], paper_px[1], paper_px[2]) >= 250);
    let (_, pa, pb) = rgb_to_lab_fast(paper_px[0], paper_px[1], paper_px[2]);
    assert!(pa.hypot(pb) <= 1.0);

    // Disabled correct_paper_cast: false -> preserves tint & skips Luma8 folding
    let mut preserve_cfg = AutoLevelsConfig::default();
    preserve_cfg.correction.correct_paper_cast = false;
    let (pres_img, pres_info, _) = level_and_verify(m4, None, &preserve_cfg);
    assert!(pres_info.paper_cast_preserved);
    assert!(!should_fold_luma(&pres_info, true));
    let pres_paper = pres_img.as_rgb8().unwrap().get_pixel(1, 0);
    assert!(
        pres_paper[0] > pres_paper[2],
        "Warm paper tint must be preserved when correct_paper_cast=false (got {:?})",
        pres_paper
    );
    let kept_rgb = fold_monochrome_post_upscale(pres_img, &pres_info, true);
    assert!(
        kept_rgb.as_rgb8().is_some(),
        "Luma8 fold must be skipped when paper_cast_preserved=true"
    );
}

#[test]
fn test_tier1_arbitrary_hue_casts_blue_green_pink_on_mixed() {
    let config = AutoLevelsConfig::default();
    let cases = [
        ("blue", [225u8, 235, 248], [210u8, 25, 25]),
        ("green", [225u8, 245, 230], [210u8, 25, 25]),
        ("pink", [248u8, 230, 240], [25u8, 90, 220]),
    ];

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
        let (leveled, info, verdict) =
            level_and_verify(DynamicImage::ImageRgb8(img), None, &config);
        assert_eq!(info.classification, PageClassification::Mixed, "{name} cast");
        assert!(info.was_leveled, "{name} cast");
        assert_eq!(verdict.outcome, Outcome::Pass, "{name} cast");

        let out = leveled.as_rgb8().unwrap();
        let pp = out.get_pixel(400, 400);
        let (_, pa, pb) = rgb_to_lab_fast(pp[0], pp[1], pp[2]);
        assert!(
            pa.hypot(pb) <= 2.0,
            "{name} paper cast must neutralize to C* <= 2.0 (got {:.2})",
            pa.hypot(pb)
        );
        assert_eq!(
            out.get_pixel(100, 100).0,
            logo_rgb,
            "{name} logo must stay bit-exact"
        );
    }
}

#[test]
fn test_tier1_t1_to_t7_adversarial_traps_zero_false_positives() {
    let config = AutoLevelsConfig::default();

    // T1: Pristine digital page with 15% white bubbles + 35% light-gray screentone (Y=200) + 50% dark art
    {
        let (w, h) = (1000u32, 1500u32);
        let total = (w * h) as usize;
        let mut raw = vec![0u8; total * 3];
        for idx in 0..total {
            let v = if idx < total * 15 / 100 {
                255u8
            } else if idx < total * 50 / 100 {
                200u8
            } else {
                (idx % 60) as u8
            };
            raw[idx * 3..idx * 3 + 3].copy_from_slice(&[v, v, v]);
        }
        let orig = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, raw.clone()).unwrap());
        let (out, info, _) = level_and_verify(orig, None, &config);
        assert!(!info.was_leveled, "T1 clean screentone page must be vetoed");
        assert_eq!(out.as_rgb8().unwrap().as_raw(), &raw);
    }

    // T2: Full-color cover at 500x500 (unconditional veto regardless of pixel count)
    {
        let color = generate_color_cover_page(500, 500);
        let orig_raw = color.as_rgb8().unwrap().as_raw().clone();
        let (out, info, _) = level_and_verify(color, None, &config);
        assert_eq!(info.classification, PageClassification::Color);
        assert!(!info.was_leveled);
        assert_eq!(out.as_rgb8().unwrap().as_raw(), &orig_raw);
    }

    // T3: Pastel blue sky [220, 235, 245] + washed ink Y=30
    {
        let (w, h) = (1000u32, 1500u32);
        let mut raw = vec![250u8; (w * h) as usize * 3];
        for i in 0..150_000 {
            raw[i * 3..i * 3 + 3].copy_from_slice(&[220, 235, 245]);
        }
        for i in 150_000..300_000 {
            raw[i * 3..i * 3 + 3].copy_from_slice(&[30, 30, 30]);
        }
        let img = DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, raw).unwrap());
        let (leveled, info, verdict) = level_and_verify(img, None, &config);
        assert_eq!(info.classification, PageClassification::Mixed);
        assert!(info.was_leveled);
        assert_eq!(verdict.outcome, Outcome::Pass);
        let rgb = leveled.as_rgb8().unwrap();
        let sky = rgb.get_pixel(0, 0);
        assert!(i32::from(sky[2]) - i32::from(sky[0]) >= 15);
        let (_, a0, b0) = rgb_to_lab_fast(220, 235, 245);
        let (_, a1, b1) = rgb_to_lab_fast(sky[0], sky[1], sky[2]);
        assert!((a0 - a1).hypot(b0 - b1) <= 1.5);
        let ink = rgb.get_pixel(0, 151);
        assert!(rec709_y(ink[0], ink[1], ink[2]) <= 8);
    }

    // T4: High-key fog (M_ink[3..=85] == 0.0 -> black floor vetoed)
    {
        let (w, h) = (1000u32, 1500u32);
        let total = (w * h) as usize;
        let mut raw = vec![0u8; total * 3];
        for idx in 0..total {
            let v = if idx < 22_500 {
                0u8
            } else {
                210 + ((idx % 41) as u8)
            };
            raw[idx * 3..idx * 3 + 3].copy_from_slice(&[v, v, v]);
        }
        let (leveled, info, _) =
            level_and_verify(DynamicImage::ImageRgb8(RgbImage::from_raw(w, h, raw).unwrap()), None, &config);
        assert_eq!(info.black_point, 0);
        let avg_y: f64 = leveled
            .to_luma8()
            .as_raw()
            .iter()
            .map(|&v| f64::from(v))
            .sum::<f64>()
            / total as f64;
        assert!(avg_y >= 215.0);
    }

    // T5: Low-key nighttime splash in BOTH Rgb8 and Luma8
    {
        let (w, h) = (800u32, 800u32);
        let luma_night = ImageBuffer::from_fn(w, h, |x, y| Luma([((x + y) % 111) as u8]));
        let (_, info_luma, _) = level_and_verify(DynamicImage::ImageLuma8(luma_night), None, &config);
        assert_eq!(info_luma.white_point, 255);

        let rgb_night = ImageBuffer::from_fn(w, h, |x, y| {
            let v = ((x + y) % 111) as u8;
            Rgb([v, v, v])
        });
        let (_, info_rgb, _) = level_and_verify(DynamicImage::ImageRgb8(rgb_night), None, &config);
        assert_eq!(info_rgb.white_point, 255);
    }

    // T6: Microscopic 3% red logo on dirty paper [245, 243, 234]
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
        let (leveled, info, verdict) =
            level_and_verify(DynamicImage::ImageRgb8(img), None, &config);
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

    // T7: Intentional warm sepia + pure-white speech bubbles dominating corner cell (0,0)
    {
        let sepia = generate_sepia_with_white_bubbles_page(1000, 1500);
        let orig_raw = sepia.as_rgb8().unwrap().as_raw().clone();
        let (out, info, _) = level_and_verify(sepia, None, &config);
        assert_eq!(info.classification, PageClassification::Color);
        assert!(!info.was_leveled, "T7 sepia + white bubbles must be vetoed");
        assert_eq!(out.as_rgb8().unwrap().as_raw(), &orig_raw);
    }
}

#[test]
fn test_tier1_16bit_format_parity_and_disabled_grayscale_check() {
    let config = AutoLevelsConfig::default();

    // Luma16
    let luma16 = ImageBuffer::from_fn(450u32, 450u32, |x, y| {
        let v: u16 = if (x + y).is_multiple_of(8) { 30 } else { 220 };
        Luma([v * 257])
    });
    let (lev_l16, info_l16, verdict_l16) =
        level_and_verify(DynamicImage::ImageLuma16(luma16), None, &config);
    assert!(info_l16.was_leveled);
    assert_eq!(verdict_l16.outcome, Outcome::Pass);
    let out_l16 = lev_l16.as_luma8().unwrap();
    assert!(out_l16.get_pixel(0, 0)[0] <= 8);
    assert!(out_l16.get_pixel(1, 0)[0] >= 250);

    // Rgb16
    let rgb16 = ImageBuffer::from_fn(450u32, 450u32, |x, y| {
        if (x + y).is_multiple_of(8) {
            Rgb([30 * 257, 30 * 257, 30 * 257])
        } else {
            Rgb([220 * 257, 215 * 257, 195 * 257])
        }
    });
    let (lev_r16, info_r16, _) = level_and_verify(DynamicImage::ImageRgb16(rgb16), None, &config);
    assert!(info_r16.was_leveled);
    let out_r16 = lev_r16.as_rgb8().unwrap();
    assert!(rec709_y(out_r16.get_pixel(1, 0)[0], out_r16.get_pixel(1, 0)[1], out_r16.get_pixel(1, 0)[2]) >= 250);

    // Rgba16
    let rgba16 = ImageBuffer::from_fn(450u32, 450u32, |x, y| {
        if (x + y).is_multiple_of(8) {
            Rgba([30 * 257, 30 * 257, 30 * 257, 65535])
        } else {
            Rgba([220 * 257, 220 * 257, 220 * 257, 65535])
        }
    });
    let (_lev_a16, info_a16, _) =
        level_and_verify(DynamicImage::ImageRgba16(rgba16), None, &config);
    assert!(info_a16.was_leveled);

    // auto_levels.enabled = false still preserves accurate is_image_grayscale
    let disabled = AutoLevelsConfig {
        enabled: false,
        ..AutoLevelsConfig::default()
    };
    let m3 = generate_m3_noisy_mono_page(500, 500);
    assert!(is_image_grayscale(&m3));
    let (_, dis_mono, _) = level_and_verify(m3, None, &disabled);
    assert!(!dis_mono.was_leveled);
    assert!(dis_mono.is_monochrome);

    let cover = generate_color_cover_page(500, 500);
    assert!(!is_image_grayscale(&cover));
    let (_, dis_col, _) = level_and_verify(cover, None, &disabled);
    assert!(!dis_col.was_leveled);
    assert!(!dis_col.is_monochrome);
}

#[test]
fn test_tier1_libwebp_lossless_icc_png_fallback_and_alpha_compositing() {
    let mut rgb = RgbImage::new(32, 32);
    for (x, y, px) in rgb.enumerate_pixels_mut() {
        *px = Rgb([(x * 7) as u8, (y * 7) as u8, 128]);
    }
    let icc = display_p3_icc();
    let (webp_bytes, out_fmt) = encode_image(&DynamicImage::ImageRgb8(rgb.clone()), ImageFormat::WebP, Some(icc), 90)
        .expect("libwebp lossless encode must succeed");
    assert_eq!(out_fmt, ImageFormat::WebP);

    let decoded = decode_image_with_metadata(&webp_bytes, None).expect("WebP decode must succeed");
    assert_eq!(decoded.detected_format, ImageFormat::WebP);
    assert_eq!(decoded.icc_profile.as_deref(), Some(icc));
    assert_eq!(decoded.image.to_rgb8().as_raw(), rgb.as_raw());

    // Tall webtoon > 16383px falls back cleanly to PNG without panic
    let tall = DynamicImage::ImageLuma8(ImageBuffer::from_pixel(4, WEBP_MAX_DIMENSION + 1, Luma([200])));
    let (fb_bytes, fb_fmt) = encode_image(&tall, ImageFormat::WebP, None, 90)
        .expect("WebP >16383px PNG fallback must succeed");
    assert_eq!(fb_fmt, ImageFormat::Png);
    assert!(fb_bytes.starts_with(b"\x89PNG\r\n\x1a\n"));

    // Rgba8 monochrome fold composites transparent [0,0,0,0] over white (255), NOT black (0)
    let mut rgba = RgbaImage::new(8, 8);
    rgba.put_pixel(0, 0, Rgba([0, 0, 0, 0]));
    rgba.put_pixel(1, 0, Rgba([0, 0, 0, 255]));
    let mono_info = LevelInfo {
        was_leveled: true,
        is_monochrome: true,
        paper_cast_preserved: false,
        classification: PageClassification::Monochrome,
        black_point: 0,
        white_point: 255,
        quality_verdict: None,
    };
    let (folded_webp, _) = encode_upscaled_output(
        DynamicImage::ImageRgba8(rgba.clone()),
        ImageFormat::Png,
        &mono_info,
        Some(icc),
        Format::WebP,
        90,
        true,
    )
    .unwrap();
    let decoded_mono = decode_image_with_metadata(&folded_webp, None).unwrap();
    assert!(decoded_mono.icc_profile.is_none(), "Folded Luma8 drops RGB ICC");
    let dec_rgb = decoded_mono.image.to_luma8();
    assert_eq!(dec_rgb.get_pixel(0, 0)[0], 255);
    assert_eq!(dec_rgb.get_pixel(1, 0)[0], 0);

    // JPEG LumaA8 white-compositing emits 1-channel L8 JPEG
    let mut la = ImageBuffer::<LumaA<u8>, Vec<u8>>::new(8, 8);
    la.put_pixel(0, 0, LumaA([0, 0]));
    la.put_pixel(1, 0, LumaA([0, 255]));
    let (jpg_bytes, jpg_fmt) =
        encode_image(&DynamicImage::ImageLumaA8(la), ImageFormat::Jpeg, None, 95).unwrap();
    assert_eq!(jpg_fmt, ImageFormat::Jpeg);
    let dec_jpg = image::load_from_memory(&jpg_bytes).unwrap();
    assert!(matches!(dec_jpg, DynamicImage::ImageLuma8(_)));
}

#[test]
fn test_tier1_decode_image_with_metadata_exif_orientation_and_icc() {
    // 1. PNG with ICC profile preserved through decode_image_with_metadata
    let img = DynamicImage::ImageRgb8(RgbImage::from_pixel(40, 20, Rgb([10, 20, 30])));
    let mut png_buf = Vec::new();
    let mut enc = PngEncoder::new(&mut png_buf);
    enc.set_icc_profile(display_p3_icc().to_vec()).unwrap();
    enc.write_image(img.as_bytes(), 40, 20, image::ExtendedColorType::Rgb8)
        .unwrap();
    let dec = decode_image_with_metadata(&png_buf, None).unwrap();
    assert_eq!(dec.detected_format, ImageFormat::Png);
    assert_eq!(dec.icc_profile.as_deref(), Some(display_p3_icc()));
    assert_eq!(probe_image_dimensions(&png_buf, None).unwrap(), (40, 20, ImageFormat::Png));

    // 2. JPEG with EXIF Orientation::Rotate90 swaps 40x20 -> 20x40 in both probe and decode
    let mut jpg_buf = Vec::new();
    let jenc = JpegEncoder::new_with_quality(&mut jpg_buf, 95);
    img.write_with_encoder(jenc).unwrap();
    // Inject minimal APP1 EXIF Orientation=6 (Rotate90) after SOI (0xFF, 0xD8)
    let exif_tiff: [u8; 32] = [
        b'E', b'x', b'i', b'f', 0, 0, // Exif\0\0
        b'M', b'M', 0, 42, 0, 0, 0, 8, // Big-endian TIFF header, IFD0 @ offset 8
        0, 1, // 1 IFD entry
        0x01, 0x12, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x06, 0x00, 0x00, // Orientation = 6
        0, 0, 0, 0, // next IFD = 0
    ];
    let app1_len = (2 + exif_tiff.len()) as u16;
    let mut oriented_jpg = Vec::with_capacity(jpg_buf.len() + exif_tiff.len() + 4);
    oriented_jpg.extend_from_slice(&jpg_buf[..2]);
    oriented_jpg.extend_from_slice(&[0xFF, 0xE1]);
    oriented_jpg.extend_from_slice(&app1_len.to_be_bytes());
    oriented_jpg.extend_from_slice(&exif_tiff);
    oriented_jpg.extend_from_slice(&jpg_buf[2..]);

    let mut dec_reader = ImageReader::new(Cursor::new(&oriented_jpg))
        .with_guessed_format()
        .unwrap()
        .into_decoder()
        .unwrap();
    assert_eq!(dec_reader.orientation().unwrap(), Orientation::Rotate90);

    let probed = probe_image_dimensions(&oriented_jpg, None).unwrap();
    assert_eq!(probed, (20, 40, ImageFormat::Jpeg));

    let dec_oriented = decode_image_with_metadata(&oriented_jpg, None).unwrap();
    assert_eq!(dec_oriented.detected_format, ImageFormat::Jpeg);
    assert_eq!((dec_oriented.image.width(), dec_oriented.image.height()), (20, 40));

    // Reference unused shared generators cleanly in analyze_page smoke check
    let cfg = AutoLevelsConfig::default();
    let cover_analysis = analyze_page(&generate_color_cover_page(400, 400), 1, &cfg);
    assert_eq!(cover_analysis.classification, PageClassification::Color);
    let sepia_analysis = analyze_page(&generate_sepia_with_white_bubbles_page(400, 400), 1, &cfg);
    assert_eq!(sepia_analysis.classification, PageClassification::Color);
    assert!(!encode_to_png_bytes(&img).is_empty());
}
