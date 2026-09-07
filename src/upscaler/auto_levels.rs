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

                // Ink luma (y <= 50) commonly has subtle scanner sensor noise or JPEG chroma ringing (up to ~25).
                // Midtones & highlights use strict neutral threshold (<= 12) to exclude colored artwork.
                let is_neutral = if y <= 50 { chroma <= 25 } else { chroma <= 12 };
                if is_neutral {
                    neutral_count += 1;
                    hist_y_neutral[y] += 1;
                }

                if y >= 220 && chroma <= 25 {
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

                let is_neutral = if y <= 50 { chroma <= 25 } else { chroma <= 12 };
                if is_neutral {
                    neutral_count += 1;
                    hist_y_neutral[y] += 1;
                }

                if y >= 220 && chroma <= 25 {
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
    let classification = if is_already_luma || neutral_ratio >= 0.995 {
        PageClassification::Monochrome
    } else if neutral_ratio >= 0.70 {
        PageClassification::Mixed
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
    // When manga or comics are scanned with elevated black levels and later typeset with
    // digital text (speech bubbles, credits headers), bin 0 has a small spike of digital font pixels,
    // followed by an empty valley (bins 2..=14), followed by a massive surge/cliff at the true ink floor
    // (bins 8..=35). A naive percentile stops at bin 0 and incorrectly classifies the page as pristine.
    // Here we detect this signature ink shelf and anchor the black point to the true artwork floor.

    // 1. Check if the page is already rich in true pristine blacks:
    // If bins 0..=3 already contain >= 5.0% of the samples, the page has rich, calibrated black linework.
    let dark_0_3: u64 = hist[0..=3].iter().map(|&c| c as u64).sum();
    let dark_0_3_pct = (dark_0_3 as f64 / n_samples as f64) * 100.0;

    // 2. Search for ink shelf / surge peak in bins [8..=35]:
    let mut max_peak_count = 0u32;
    let mut max_peak_bin = 0usize;
    for bin in 8..=35 {
        if hist[bin] > max_peak_count {
            max_peak_count = hist[bin];
            max_peak_bin = bin;
        }
    }

    // 3. Search for minimum valley count between bin 2 and max_peak_bin:
    let mut min_valley_count = u32::MAX;
    if max_peak_bin >= 8 {
        for bin in 2..max_peak_bin {
            if hist[bin] < min_valley_count {
                min_valley_count = hist[bin];
            }
        }
    }

    // 4. Shelf criteria:
    // - Applied only to Monochrome and Mixed content (scanned ink with digital typesetting).
    //   On full-color comics, dark clusters (e.g. slate walls, night skies) are the colorist's
    //   intentional artistic palette and must NOT be mistaken for a scanning defect.
    // - Peak is at or above bin 8 (meaning ink was lifted by at least 8 levels)
    // - Peak contains significant ink density: >= 0.5% (0.005) of samples in this single bin
    // - Peak surges by at least 3.0x over the valley floor preceding it
    // - The page is not already pristine dark (dark_0_3_pct < 5.0%)
    let is_shelf = classification != PageClassification::Color
        && max_peak_bin >= 8
        && (max_peak_count as f64) >= (n_samples as f64 * 0.005)
        && min_valley_count > 0
        && ((max_peak_count as f32) / (min_valley_count as f32) >= 3.0)
        && dark_0_3_pct < 5.0;

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
    if b_point <= 4 {
        b_point = 0;
    } else if b_point > 45 {
        // If the black point would exceed 45 (e.g. high-key pastel art, blank white endpapers),
        // there is no black ink intended on this page; do not crush midtones!
        b_point = 0;
    } else if b_point > config.max_black_shift {
        b_point = config.max_black_shift;
    }

    if w_point >= 252 {
        w_point = 255;
    } else if w_point < config.min_white_threshold {
        w_point = 255; // Don't blow out dark/night scenes
    }

    // Paper cast detection for color comic pages
    let mut paper_cast_gains = [1.0f32, 1.0f32, 1.0f32];
    let mut apply_paper_cast = false;

    if classification == PageClassification::Color
        && config.correct_paper_cast
        && highlight_count >= (total_valid_pixels / 100)
    {
        let avg_r = (sum_r_high as f32) / (highlight_count as f32);
        let avg_g = (sum_g_high as f32) / (highlight_count as f32);
        let avg_b = (sum_b_high as f32) / (highlight_count as f32);

        let max_dev = (avg_r - avg_g)
            .abs()
            .max((avg_g - avg_b).abs())
            .max((avg_b - avg_r).abs());

        // Significant highlight paper cast (e.g. oxidized newsprint yellowing)
        if max_dev > 5.0 && avg_r >= 190.0 && avg_g >= 190.0 && avg_b >= 170.0 {
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
            "[AutoLevels] Skipped (already pristine: black <= 4, white >= 252, classification: {:?})",
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
                chunk[0] = lut[chunk[0] as usize];
            }
        }
        DynamicImage::ImageRgb8(rgb) => {
            for chunk in (&mut **rgb).chunks_exact_mut(3) {
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];
                let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as usize;
                let y_new = lut[y] as f32;

                if y == 0 {
                    chunk[0] = 0;
                    chunk[1] = 0;
                    chunk[2] = 0;
                } else {
                    let gain = y_new / (y as f32);
                    let mut r_out = (r as f32 * gain).round().clamp(0.0, 255.0);
                    let mut g_out = (g as f32 * gain).round().clamp(0.0, 255.0);
                    let mut b_out = (b as f32 * gain).round().clamp(0.0, 255.0);

                    if apply_paper_cast && y >= 180 {
                        let cast_factor = ((y as f32 - 180.0) / 55.0).clamp(0.0, 1.0);
                        r_out = (r_out * (1.0 + (paper_cast_gains[0] - 1.0) * cast_factor))
                            .clamp(0.0, 255.0);
                        g_out = (g_out * (1.0 + (paper_cast_gains[1] - 1.0) * cast_factor))
                            .clamp(0.0, 255.0);
                        b_out = (b_out * (1.0 + (paper_cast_gains[2] - 1.0) * cast_factor))
                            .clamp(0.0, 255.0);
                    }

                    chunk[0] = r_out as u8;
                    chunk[1] = g_out as u8;
                    chunk[2] = b_out as u8;
                }
            }
        }
        DynamicImage::ImageRgba8(rgba) => {
            for chunk in (&mut **rgba).chunks_exact_mut(4) {
                let r = chunk[0];
                let g = chunk[1];
                let b = chunk[2];
                let y = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as usize;
                let y_new = lut[y] as f32;

                if y == 0 {
                    chunk[0] = 0;
                    chunk[1] = 0;
                    chunk[2] = 0;
                } else {
                    let gain = y_new / (y as f32);
                    let mut r_out = (r as f32 * gain).round().clamp(0.0, 255.0);
                    let mut g_out = (g as f32 * gain).round().clamp(0.0, 255.0);
                    let mut b_out = (b as f32 * gain).round().clamp(0.0, 255.0);

                    if apply_paper_cast && y >= 180 {
                        let cast_factor = ((y as f32 - 180.0) / 55.0).clamp(0.0, 1.0);
                        r_out = (r_out * (1.0 + (paper_cast_gains[0] - 1.0) * cast_factor))
                            .clamp(0.0, 255.0);
                        g_out = (g_out * (1.0 + (paper_cast_gains[1] - 1.0) * cast_factor))
                            .clamp(0.0, 255.0);
                        b_out = (b_out * (1.0 + (paper_cast_gains[2] - 1.0) * cast_factor))
                            .clamp(0.0, 255.0);
                    }

                    chunk[0] = r_out as u8;
                    chunk[1] = g_out as u8;
                    chunk[2] = b_out as u8;
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
