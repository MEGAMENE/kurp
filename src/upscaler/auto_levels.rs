use image::{DynamicImage, GenericImageView};
use log::info;
use crate::config::app_config::{AutoLevelsConfig, AutoLevelsMode};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PageClassification {
    Monochrome,
    Mixed,
    Color,
}

#[derive(Debug, Clone, Copy)]
pub struct LevelInfo {
    pub was_leveled: bool,
    pub is_monochrome: bool,
    pub classification: PageClassification,
    pub black_point: u8,
    pub white_point: u8,
}

impl Default for LevelInfo {
    fn default() -> Self {
        Self {
            was_leveled: false,
            is_monochrome: false,
            classification: PageClassification::Color,
            black_point: 0,
            white_point: 255,
        }
    }
}

/// Analyzes the input image, classifies its content, determines black/white point anchors,
/// and applies contrast stretching with midtone gamma and color cast correction.
///
/// Runs pre-upscale on CPU to feed clean, high-contrast edges to Real-CUGAN.
pub fn analyze_and_level_image(
    mut image: DynamicImage,
    config: &AutoLevelsConfig,
) -> (DynamicImage, LevelInfo) {
    if !config.enabled || config.mode == AutoLevelsMode::Off {
        return (image, LevelInfo::default());
    }

    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return (image, LevelInfo::default());
    }

    let mut hist_y = [0u32; 256];
    let mut hist_y_neutral = [0u32; 256];
    let mut neutral_count: usize = 0;
    let mut total_valid_pixels: usize = 0;
    let mut max_chroma: u8 = 0;
    let mut high_chroma_count: usize = 0;

    let mut highlight_count: usize = 0;
    let mut sum_r_high: u64 = 0;
    let mut sum_g_high: u64 = 0;
    let mut sum_b_high: u64 = 0;

    let is_already_luma = matches!(
        image,
        DynamicImage::ImageLuma8(_) | DynamicImage::ImageLumaA8(_)
    );

    match &image {
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

                // Rec.709 integer luminance: (54*R + 183*G + 19*B + 128) >> 8
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

                // Ink luma (y <= 50) commonly has subtle scanner sensor noise or JPEG chroma ringing (up to ~25).
                // Midtones & highlights use strict neutral threshold (<= 12) to exclude colored artwork.
                let is_neutral = if y <= 50 { chroma <= 25 } else { chroma <= 12 };
                if is_neutral {
                    neutral_count += 1;
                    hist_y_neutral[y] += 1;
                }

                // Genuine neutral paper highlights (chroma <= 10)
                if y >= 220 && chroma <= 10 {
                    highlight_count += 1;
                    sum_r_high += r as u64;
                    sum_g_high += g as u64;
                    sum_b_high += b as u64;
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

                if y >= 220 && chroma <= 10 {
                    highlight_count += 1;
                    sum_r_high += r as u64;
                    sum_g_high += g as u64;
                    sum_b_high += b as u64;
                }
            }
        }
        _ => {
            let rgb = image.to_rgb8();
            return analyze_and_level_image(DynamicImage::ImageRgb8(rgb), config);
        }
    }

    if total_valid_pixels == 0 {
        return (image, LevelInfo::default());
    }

    let neutral_ratio = neutral_count as f32 / total_valid_pixels as f32;

    // Content classification:
    // - Monochrome: B&W scans without genuine color (scanner/JPEG noise chroma <= 35, high_chroma_count < 50).
    //   Only Monochrome pages are converted to 1-channel grayscale post-upscale.
    // - Mixed: B&W manga containing color titles, watermarks, scanlator stamps, or chapter splash art.
    //   Never converted to grayscale, preserving all color elements.
    // - Color: Full-color comics and Western graphic novels (neutral_ratio < 0.85).
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

    let is_monochrome = classification == PageClassification::Monochrome;

    if config.mode == AutoLevelsMode::Manga && classification == PageClassification::Color {
        return (
            image,
            LevelInfo {
                was_leveled: false,
                is_monochrome: false,
                classification,
                black_point: 0,
                white_point: 255,
            },
        );
    }

    if config.mode == AutoLevelsMode::Color && classification != PageClassification::Color {
        return (
            image,
            LevelInfo {
                was_leveled: false,
                is_monochrome,
                classification,
                black_point: 0,
                white_point: 255,
            },
        );
    }

    let (hist, n_samples) = match classification {
        PageClassification::Monochrome => (&hist_y, total_valid_pixels),
        PageClassification::Mixed => (&hist_y_neutral, neutral_count),
        PageClassification::Color => (&hist_y, total_valid_pixels),
    };

    if n_samples == 0 {
        return (image, LevelInfo::default());
    }

    let black_clip_target =
        (n_samples as f64 * (config.black_clip_percent.max(0.0) as f64 / 100.0)).round() as u64;
    let white_clip_target =
        (n_samples as f64 * (config.white_clip_percent.max(0.0) as f64 / 100.0)).round() as u64;

    // --- Ink Shelf / Surge Detection ---
    // When manga or comics are scanned with elevated black levels (e.g. limited video range 16-235
    // or CMYK prepress mapping), linework is elevated to bins 12..=24. If digital vector text or headers
    // were typeset, bin 0 has a small spike followed by an empty valley (bins 2..=14) before the true ink floor.
    // We restrict peak detection strictly to [12..=24] to avoid catching dark gray screentones / halftones (30+).

    let dark_0_3: u64 = hist[0..=3].iter().map(|&c| c as u64).sum();
    let dark_0_3_pct = (dark_0_3 as f64 / n_samples as f64) * 100.0;

    let mut max_peak_count = 0u32;
    let mut max_peak_bin = 0usize;
    for bin in 10..=25 {
        if hist[bin] > max_peak_count {
            max_peak_count = hist[bin];
            max_peak_bin = bin;
        }
    }

    let mut min_valley_count = u32::MAX;
    if max_peak_bin >= 10 {
        for bin in 2..max_peak_bin {
            if hist[bin] < min_valley_count {
                min_valley_count = hist[bin];
            }
        }
    }

    // Shelf criteria:
    // - Applied only to Monochrome and Mixed content (scanned ink with digital typesetting).
    // - Peak in [10..=25] contains significant ink density: >= 0.5% of samples in this single bin
    // - Peak surges by at least 3.0x over the valley floor preceding it (or infinite surge if clean empty valley)
    // - Linework is not already pristine dark (dark_0_3_pct < 3.0%)
    let surge = if min_valley_count == 0 {
        f32::INFINITY
    } else {
        (max_peak_count as f32) / (min_valley_count as f32)
    };

    let is_shelf = classification != PageClassification::Color
        && max_peak_bin >= 10
        && (max_peak_count as f64) >= (n_samples as f64 * 0.005)
        && surge >= 3.0
        && dark_0_3_pct < 3.0;

    let mut b_point = if is_shelf {
        max_peak_bin as u8
    } else {
        let mut bp = 0u8;
        let mut acc_low = 0u64;
        for (level, &count) in hist.iter().enumerate() {
            acc_low += count as u64;
            if acc_low >= black_clip_target {
                bp = level as u8;
                break;
            }
        }
        bp
    };

    let mut w_point = 255u8;
    let mut acc_high = 0u64;
    for (level, &count) in hist.iter().enumerate().rev() {
        acc_high += count as u64;
        if acc_high >= white_clip_target {
            w_point = level as u8;
            break;
        }
    }

    // Safeguards
    if classification == PageClassification::Color {
        // Color comics: protect intentional shadow palettes and low-contrast pastel scenes
        if b_point > 12 {
            b_point = 0;
        }
        if w_point < 250 {
            w_point = 255;
        }
    } else if classification == PageClassification::Mixed {
        // Mixed content (manga with color elements or atmospheric scenes):
        // If no scanner shelf is confirmed, treat as atmospheric artwork and protect midtones.
        if !is_shelf && b_point > 12 {
            b_point = 0;
        } else {
            if b_point <= 1 {
                b_point = 0;
            } else if b_point > config.max_black_shift.max(45) {
                b_point = 0;
            } else if b_point > config.max_black_shift {
                b_point = config.max_black_shift;
            }
        }

        if w_point >= 254 {
            w_point = 255;
        } else if w_point < config.min_white_threshold {
            w_point = 255; // Don't blow out dark/night scenes
        }
    } else {
        if b_point <= 1 {
            b_point = 0;
        } else if b_point > config.max_black_shift.max(45) {
            b_point = 0;
        } else if b_point > config.max_black_shift {
            b_point = config.max_black_shift;
        }

        if w_point >= 254 {
            w_point = 255;
        } else if w_point < config.min_white_threshold {
            w_point = 255; // Don't blow out dark/night scenes
        }
    }

    // Paper cast detection for color comic pages
    let mut paper_cast_gains = [1.0f32, 1.0f32, 1.0f32];
    let mut apply_paper_cast = false;

    if classification == PageClassification::Color
        && config.correct_paper_cast
        && highlight_count > 0
        && highlight_count >= (total_valid_pixels / 100)
    {
        let avg_r = (sum_r_high as f32) / (highlight_count as f32);
        let avg_g = (sum_g_high as f32) / (highlight_count as f32);
        let avg_b = (sum_b_high as f32) / (highlight_count as f32);

        // Paper aging is warm yellow/sepia: R >= G > B (depressed blue)
        let is_yellow_paper = avg_r >= avg_g && avg_g > avg_b && (avg_r - avg_b) > 6.0;

        if is_yellow_paper && avg_r >= 210.0 && avg_g >= 200.0 && avg_b >= 170.0 {
            let max_val = avg_r.max(avg_g).max(avg_b);
            paper_cast_gains[0] = max_val / avg_r;
            paper_cast_gains[1] = max_val / avg_g;
            paper_cast_gains[2] = max_val / avg_b;
            apply_paper_cast = true;
        }
    }

    // Skip if already pristine
    if b_point == 0 && w_point == 255 && !apply_paper_cast {
        info!(
            "[AutoLevels] Skipped (already pristine: black <= 1, white >= 254, classification: {:?})",
            classification
        );
        return (
            image,
            LevelInfo {
                was_leveled: false,
                is_monochrome,
                classification,
                black_point: 0,
                white_point: 255,
            },
        );
    }

    let range = (w_point as f32 - b_point as f32).max(1.0);
    let inv_gamma = if (config.gamma - 1.0).abs() > 0.001 && config.gamma > 0.1 {
        1.0 / config.gamma
    } else {
        1.0
    };

    let mut lut = [0u8; 256];
    for y in 0..=255 {
        if (y as u8) <= b_point {
            lut[y] = 0;
        } else if (y as u8) >= w_point {
            lut[y] = 255;
        } else {
            let norm = (y as f32 - b_point as f32) / range;
            let mapped = if inv_gamma != 1.0 {
                norm.powf(inv_gamma)
            } else {
                norm
            };
            lut[y] = (mapped * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }

    match &mut image {
        DynamicImage::ImageLuma8(luma) => {
            for p in &mut **luma {
                *p = lut[*p as usize];
            }
        }
        DynamicImage::ImageLumaA8(luma_a) => {
            for chunk in (&mut **luma_a).chunks_exact_mut(2) {
                if chunk[1] >= 128 {
                    chunk[0] = lut[chunk[0] as usize];
                }
            }
        }
        DynamicImage::ImageRgb8(rgb) => {
            for chunk in (&mut **rgb).chunks_exact_mut(3) {
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];

                let max_c = r.max(g).max(b);
                let min_c = r.min(g).min(b);
                let chroma = max_c - min_c;

                let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as usize;

                // Scanner dark pedestal floor clamping:
                // For non-color pages with an elevated black shelf (e.g. CCD scanner offset R=25, G=15, B=3),
                // clamp dark ink pixels (lum <= b_point, chroma <= 25) directly to pitch black (0, 0, 0).
                if b_point > 0 && classification != PageClassification::Color && y <= b_point && chroma <= 25 {
                    chunk[0] = 0;
                    chunk[1] = 0;
                    chunk[2] = 0;
                    continue;
                }

                // Microsecond fast-path for pure neutral pixels and pure color pixels
                if !apply_paper_cast {
                    if chroma <= 12 {
                        chunk[0] = lut[r as usize];
                        chunk[1] = lut[g as usize];
                        chunk[2] = lut[b as usize];
                        continue;
                    }
                    if classification != PageClassification::Color && chroma >= 35 {
                        continue;
                    }
                }

                if classification == PageClassification::Color {
                    let y_lev = lut[y] as f32;
                    let (mut r_lev, mut g_lev, mut b_lev) = if y == 0 {
                        (0.0f32, 0.0f32, 0.0f32)
                    } else {
                        let scale = y_lev / (y as f32);
                        (
                            (r as f32 * scale).clamp(0.0, 255.0),
                            (g as f32 * scale).clamp(0.0, 255.0),
                            (b as f32 * scale).clamp(0.0, 255.0),
                        )
                    };

                    if apply_paper_cast && y >= 180 {
                        let cast_factor = ((y as f32 - 180.0) / 55.0).clamp(0.0, 1.0);
                        r_lev = (r_lev * (1.0 + (paper_cast_gains[0] - 1.0) * cast_factor)).clamp(0.0, 255.0);
                        g_lev = (g_lev * (1.0 + (paper_cast_gains[1] - 1.0) * cast_factor)).clamp(0.0, 255.0);
                        b_lev = (b_lev * (1.0 + (paper_cast_gains[2] - 1.0) * cast_factor)).clamp(0.0, 255.0);
                    }

                    chunk[0] = r_lev.round() as u8;
                    chunk[1] = g_lev.round() as u8;
                    chunk[2] = b_lev.round() as u8;
                    continue;
                }

                let low_thresh = if y <= 50 { 20 } else { 12 };
                let high_thresh = 35;

                let w_neutral = if chroma <= low_thresh {
                    1.0f32
                } else if chroma >= high_thresh {
                    0.0f32
                } else {
                    (high_thresh - chroma) as f32 / (high_thresh - low_thresh) as f32
                };

                if w_neutral <= 0.001 {
                    // Pure colorful pixel (red title, watermark, colored art) - 100% untouched!
                    continue;
                }

                let mut r_lev = lut[r as usize] as f32;
                let mut g_lev = lut[g as usize] as f32;
                let mut b_lev = lut[b as usize] as f32;

                if apply_paper_cast && y >= 180 {
                    let cast_factor = ((y as f32 - 180.0) / 55.0).clamp(0.0, 1.0);
                    r_lev = (r_lev * (1.0 + (paper_cast_gains[0] - 1.0) * cast_factor)).clamp(0.0, 255.0);
                    g_lev = (g_lev * (1.0 + (paper_cast_gains[1] - 1.0) * cast_factor)).clamp(0.0, 255.0);
                    b_lev = (b_lev * (1.0 + (paper_cast_gains[2] - 1.0) * cast_factor)).clamp(0.0, 255.0);
                }

                if w_neutral >= 0.999 {
                    chunk[0] = r_lev.round() as u8;
                    chunk[1] = g_lev.round() as u8;
                    chunk[2] = b_lev.round() as u8;
                } else {
                    chunk[0] = (w_neutral * r_lev + (1.0 - w_neutral) * (r as f32))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    chunk[1] = (w_neutral * g_lev + (1.0 - w_neutral) * (g as f32))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    chunk[2] = (w_neutral * b_lev + (1.0 - w_neutral) * (b as f32))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
        DynamicImage::ImageRgba8(rgba) => {
            for chunk in (&mut **rgba).chunks_exact_mut(4) {
                if chunk[3] < 128 {
                    continue; // Skip transparent and semi-transparent pixels
                }

                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];

                let max_c = r.max(g).max(b);
                let min_c = r.min(g).min(b);
                let chroma = max_c - min_c;

                let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as usize;

                // Scanner dark pedestal floor clamping:
                // For non-color pages with an elevated black shelf (e.g. CCD scanner offset R=25, G=15, B=3),
                // clamp dark ink pixels (lum <= b_point, chroma <= 25) directly to pitch black (0, 0, 0).
                if b_point > 0 && classification != PageClassification::Color && y <= b_point && chroma <= 25 {
                    chunk[0] = 0;
                    chunk[1] = 0;
                    chunk[2] = 0;
                    continue;
                }

                // Microsecond fast-path for pure neutral pixels and pure color pixels
                if !apply_paper_cast {
                    if chroma <= 12 {
                        chunk[0] = lut[r as usize];
                        chunk[1] = lut[g as usize];
                        chunk[2] = lut[b as usize];
                        continue;
                    }
                    if classification != PageClassification::Color && chroma >= 35 {
                        continue;
                    }
                }

                if classification == PageClassification::Color {
                    let y_lev = lut[y] as f32;
                    let (mut r_lev, mut g_lev, mut b_lev) = if y == 0 {
                        (0.0f32, 0.0f32, 0.0f32)
                    } else {
                        let scale = y_lev / (y as f32);
                        (
                            (r as f32 * scale).clamp(0.0, 255.0),
                            (g as f32 * scale).clamp(0.0, 255.0),
                            (b as f32 * scale).clamp(0.0, 255.0),
                        )
                    };

                    if apply_paper_cast && y >= 180 {
                        let cast_factor = ((y as f32 - 180.0) / 55.0).clamp(0.0, 1.0);
                        r_lev = (r_lev * (1.0 + (paper_cast_gains[0] - 1.0) * cast_factor)).clamp(0.0, 255.0);
                        g_lev = (g_lev * (1.0 + (paper_cast_gains[1] - 1.0) * cast_factor)).clamp(0.0, 255.0);
                        b_lev = (b_lev * (1.0 + (paper_cast_gains[2] - 1.0) * cast_factor)).clamp(0.0, 255.0);
                    }

                    chunk[0] = r_lev.round() as u8;
                    chunk[1] = g_lev.round() as u8;
                    chunk[2] = b_lev.round() as u8;
                    continue;
                }

                let low_thresh = if y <= 50 { 20 } else { 12 };
                let high_thresh = 35;

                let w_neutral = if chroma <= low_thresh {
                    1.0f32
                } else if chroma >= high_thresh {
                    0.0f32
                } else {
                    (high_thresh - chroma) as f32 / (high_thresh - low_thresh) as f32
                };

                if w_neutral <= 0.001 {
                    continue;
                }

                let mut r_lev = lut[r as usize] as f32;
                let mut g_lev = lut[g as usize] as f32;
                let mut b_lev = lut[b as usize] as f32;

                if apply_paper_cast && y >= 180 {
                    let cast_factor = ((y as f32 - 180.0) / 55.0).clamp(0.0, 1.0);
                    r_lev = (r_lev * (1.0 + (paper_cast_gains[0] - 1.0) * cast_factor)).clamp(0.0, 255.0);
                    g_lev = (g_lev * (1.0 + (paper_cast_gains[1] - 1.0) * cast_factor)).clamp(0.0, 255.0);
                    b_lev = (b_lev * (1.0 + (paper_cast_gains[2] - 1.0) * cast_factor)).clamp(0.0, 255.0);
                }

                if w_neutral >= 0.999 {
                    chunk[0] = r_lev.round() as u8;
                    chunk[1] = g_lev.round() as u8;
                    chunk[2] = b_lev.round() as u8;
                } else {
                    chunk[0] = (w_neutral * r_lev + (1.0 - w_neutral) * (r as f32))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    chunk[1] = (w_neutral * g_lev + (1.0 - w_neutral) * (g as f32))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                    chunk[2] = (w_neutral * b_lev + (1.0 - w_neutral) * (b as f32))
                        .round()
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
        _ => unreachable!(),
    }

    info!(
        "[AutoLevels] Leveled ({:?}): black {}->0{}, white {}->255, gamma={:.2}{}",
        classification,
        b_point,
        if is_shelf { " (ink shelf detected)" } else { "" },
        w_point,
        config.gamma,
        if apply_paper_cast {
            " [paper cast corrected]"
        } else {
            ""
        }
    );

    (
        image,
        LevelInfo {
            was_leveled: true,
            is_monochrome,
            classification,
            black_point: b_point,
            white_point: w_point,
        },
    )
}
