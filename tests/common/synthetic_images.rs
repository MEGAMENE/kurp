#![allow(dead_code)]

use bytes::Bytes;
use image::{DynamicImage, Rgb, RgbImage, Rgba, RgbaImage};
use std::io::Cursor;

/// M1: Digital typesetting spike (`Y=0` at `black_ratio`) + 5% pure-white speech bubbles (`Y=255`)
/// + 13.8% washed-out drawing ink (`Y in [ink_level..=ink_level+10]`) on neutral paper (`Y=248`).
pub fn generate_m1_spike_page(
    width: u32,
    height: u32,
    black_ratio: f64,
    ink_level: u8,
) -> DynamicImage {
    let total = (width as usize) * (height as usize);
    let mut raw = vec![248u8; total * 3];

    let n_black = (total as f64 * black_ratio).round() as usize;
    let n_white = (total as f64 * 0.050).round() as usize;
    let n_ink = (total as f64 * 0.138).round() as usize;

    for i in 0..n_black.min(total) {
        raw[i * 3] = 0;
        raw[i * 3 + 1] = 0;
        raw[i * 3 + 2] = 0;
    }
    let white_end = (n_black + n_white).min(total);
    for i in n_black..white_end {
        raw[i * 3] = 255;
        raw[i * 3 + 1] = 255;
        raw[i * 3 + 2] = 255;
    }
    let ink_end = (white_end + n_ink).min(total);
    for (offset, i) in (white_end..ink_end).enumerate() {
        let v = ink_level.saturating_add((offset % 11) as u8);
        raw[i * 3] = v;
        raw[i * 3 + 1] = v;
        raw[i * 3 + 2] = v;
    }

    DynamicImage::ImageRgb8(RgbImage::from_raw(width, height, raw).expect("valid M1 dimensions"))
}

/// M2 (Monochrome Rgb8): 65% clean interior `[246, 245, 240]`, 25% yellowed outer margins
/// `[245, 232, 185]`, and 10% muddy shadow ink `[27, 18, 6]`.
pub fn generate_m2_localized_yellow_page(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    let mx = (width as f32 * 0.10).round() as u32;
    let my = (height as f32 * 0.10).round() as u32;

    for y in 0..height {
        for x in 0..width {
            let in_outer_margin = x < mx || x >= width - mx || y < my || y >= height - my;
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

    DynamicImage::ImageRgb8(img)
}

/// M2 (Mixed Rgba8): localized yellow outer margins, muddy shadow ink, red logo `[210, 25, 25]`,
/// interior cream skin `[245, 232, 205]` (`C* = 14.7`), plus semi-transparent edge strip (`A = 64`).
pub fn generate_m2_mixed_rgba_page(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbaImage::new(width, height);
    let mx = (width as f32 * 0.10).round() as u32;
    let my = (height as f32 * 0.10).round() as u32;

    for y in 0..height {
        for x in 0..width {
            let in_outer_margin = x < mx || x >= width - mx || y < my || y >= height - my;
            let in_logo = x >= (width * 15 / 100)
                && x < (width * 28 / 100)
                && y >= (height * 15 / 100)
                && y < (height * 30 / 100);
            let in_skin = x >= (width * 40 / 100)
                && x < (width * 65 / 100)
                && y >= (height * 40 / 100)
                && y < (height * 60 / 100);
            let in_aa_edge = !in_outer_margin
                && !in_logo
                && !in_skin
                && x >= (width * 72 / 100)
                && x < (width * 78 / 100)
                && y >= (height * 40 / 100)
                && y < (height * 50 / 100);
            let is_shadow =
                !in_outer_margin && !in_logo && !in_skin && !in_aa_edge && ((x + y * 3) % 8 == 0);

            let px = if in_outer_margin {
                Rgba([245, 232, 185, 255])
            } else if in_logo {
                Rgba([210, 25, 25, 255])
            } else if in_skin {
                Rgba([245, 232, 205, 255])
            } else if in_aa_edge {
                Rgba([246, 245, 240, 64])
            } else if is_shadow {
                Rgba([27, 18, 6, 255])
            } else {
                Rgba([246, 245, 240, 255])
            };
            img.put_pixel(x, y, px);
        }
    }

    DynamicImage::ImageRgba8(img)
}

/// M3: Noisy scanned monochrome manga (`85%` `[245, 245, 244]` + `15%` `[5, 4, 3]` dark ink).
pub fn generate_m3_noisy_mono_page(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            if (x + y).is_multiple_of(7) {
                img.put_pixel(x, y, Rgb([5, 4, 3]));
            } else {
                img.put_pixel(x, y, Rgb([245, 245, 244]));
            }
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// M4: Uniform aged paper `[232, 224, 198]` with elevated ink `[26, 26, 26]`.
pub fn generate_m4_aged_paper_page(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            if (x + y).is_multiple_of(7) {
                img.put_pixel(x, y, Rgb([26, 26, 26]));
            } else {
                img.put_pixel(x, y, Rgb([232, 224, 198]));
            }
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// T2: Full-color comic/webtoon cover (warm midtones + saturated accents, `R_res > 0.15`).
pub fn generate_color_cover_page(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let px = match ((x * 3) / width.max(1), (y * 3) / height.max(1)) {
                (0, 0) => Rgb([35, 110, 225]),
                (1, 0) => Rgb([225, 45, 45]),
                (2, 0) => Rgb([35, 195, 75]),
                (0, 1) => Rgb([215, 175, 85]), // warm golden midtone
                (1, 1) => Rgb([180, 95, 65]),  // warm terracotta midtone
                _ => Rgb([240, 238, 232]),
            };
            img.put_pixel(x, y, px);
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// T7: Intentional warm sepia artwork (`[245, 230, 200]` + `[40, 30, 15]`) with a corner 4x4 cell
/// dominated (>90%) by pure-white digital speech bubbles `[255, 255, 255]`.
pub fn generate_sepia_with_white_bubbles_page(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    let bubble_w = (width * 24) / 100;
    let bubble_h = (height * 24) / 100;
    for y in 0..height {
        for x in 0..width {
            let px = if x < bubble_w && y < bubble_h {
                Rgb([255, 255, 255])
            } else if (x + y).is_multiple_of(8) {
                Rgb([40, 30, 15])
            } else {
                Rgb([245, 230, 200])
            };
            img.put_pixel(x, y, px);
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// Encodes a `DynamicImage` into compressed PNG bytes.
pub fn encode_to_png_bytes(img: &DynamicImage) -> Bytes {
    let mut buf = Vec::new();
    img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
        .expect("Failed to encode synthetic PNG image");
    Bytes::from(buf)
}
