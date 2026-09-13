//! Synthetic Test Image Generators for Kurp Auto-Leveling E2E & Integration Test Suite
//!
//! Provides deterministic programmatic generation of test images covering all 9 canonical defect
//! and quality guard scenarios specified in TEST_INFRA.md and Master Reference Specification.

use image::{DynamicImage, Rgb, RgbImage, Rgba, RgbaImage};
use bytes::Bytes;
use std::io::Cursor;

/// Scenario 1: Physical Lignin Paper Yellowing
///
/// Simulates genuine cellulose paper oxidation:
/// - Unprinted paper in Zone M (outer 2-6% perimeter) and gutters: R=248, G=242, B=215
///   (Lignin signature: R >= G > B, (R-B)=33 in [6, 55], (R-G)=6 <= 0.65*(R-B), y >= 205).
/// - Linework in Zone C (inner 88% canvas): R=15, G=15, B=15
///   (Neutral shadow: R <= B + 4, shadow_is_warm = false).
pub fn generate_lignin_yellowed_page(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    let margin_x = (width as f32 * 0.06).round() as u32;
    let margin_y = (height as f32 * 0.06).round() as u32;

    for y in 0..height {
        for x in 0..width {
            let in_core = x >= margin_x && x < width - margin_x && y >= margin_y && y < height - margin_y;
            // Draw linework panels in core canvas
            let is_linework = in_core && ((x % 60 < 4) || (y % 80 < 4) || ((x + y) % 120 < 5));

            let pixel = if is_linework {
                Rgb([15, 15, 15]) // Neutral dark ink shadow
            } else {
                Rgb([248, 242, 215]) // Yellowed paper background (R-B = 33)
            };
            img.put_pixel(x, y, pixel);
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// Scenario 2: Intentional Sepia / Flashback Artwork
///
/// Simulates intentional artistic warm tones:
/// - Highlight paper: R=245, G=230, B=200 (warm highlights, R - B = 45.0).
/// - Ink shadows in Zone C: R=40, G=30, B=15 (warm shadows: R = 40 > B + 4 = 19).
/// Dual-zone chromaticity check detects warm shadow ink and skips paper yellowing correction.
pub fn generate_intentional_sepia_page(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    let margin_x = (width as f32 * 0.06).round() as u32;
    let margin_y = (height as f32 * 0.06).round() as u32;

    for y in 0..height {
        for x in 0..width {
            let in_core = x >= margin_x && x < width - margin_x && y >= margin_y && y < height - margin_y;
            let is_linework = in_core && ((x % 50 < 5) || (y % 70 < 5));

            let pixel = if is_linework {
                Rgb([40, 30, 15]) // Warm sepia ink shadow (R > B + 4)
            } else {
                Rgb([245, 230, 200]) // Warm sepia highlight paper
            };
            img.put_pixel(x, y, pixel);
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// Scenario 3: Bimodal Ink Shelf with Digital Typeset Text and Letterbox Borders
///
/// Simulates elevated scanner linework pedestal (e.g. broadcast 16-235 offset):
/// - Outer 1.5% perimeter letterbox borders: R=G=B=0 (solid black, subtracted by perimeter mask).
/// - Digital typeset dialogue text: R=G=B=0 (<2.0% area, does NOT disqualify shelf).
/// - Elevated linework pedestal: R=G=B=shelf_bin (typically 18) with surge ratio >= 2.8x.
/// - Valley trough: bins 3..shelf_bin-1 have 0 count.
/// - Background paper: R=G=B=255.
pub fn generate_bimodal_ink_shelf_page(width: u32, height: u32, shelf_bin: u8) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    let border_x = ((width as f32 * 0.015).round() as u32).max(1);
    let border_y = ((height as f32 * 0.015).round() as u32).max(1);

    for y in 0..height {
        for x in 0..width {
            let is_border = x < border_x || x >= width - border_x || y < border_y || y >= height - border_y;
            let is_typeset_text = !is_border && (x >= 100 && x < 200 && y >= 100 && y < 130 && (x % 5 == 0 || y % 5 == 0));
            let is_elevated_linework = !is_border && !is_typeset_text && ((x % 35 < 3) || (y % 45 < 3));

            let pixel = if is_border || is_typeset_text {
                Rgb([0, 0, 0]) // Level 0 black
            } else if is_elevated_linework {
                Rgb([shelf_bin, shelf_bin, shelf_bin]) // Elevated linework pedestal
            } else {
                Rgb([255, 255, 255]) // Pure white background
            };
            img.put_pixel(x, y, pixel);
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// Scenario 4: High-Frequency Screentone (Halftone Dots at 60-85 LPI)
///
/// Simulates manga screentones with periodic 2D cosine dot grid:
/// Density peaks near level 20-25, but spatial 8-neighbor Laplacian detects >60% high gradient pixels
/// and/or broad FWHM, aborting shelf detection to prevent tonal flattening.
pub fn generate_screentone_page(width: u32, height: u32, dot_pitch: f32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    let pitch = dot_pitch.max(4.0);

    for y in 0..height {
        for x in 0..width {
            // Periodic 2D cosine dot simulation
            let cx = (2.0 * std::f32::consts::PI * (x as f32) / pitch).cos();
            let cy = (2.0 * std::f32::consts::PI * (y as f32) / pitch).cos();
            let val = ((cx * cy * 0.5 + 0.5) * 40.0 + 10.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(x, y, Rgb([val, val, val]));
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// Scenario 5: Moody Atmospheric Art with Continuous Tails
///
/// Simulates dark/foggy scenes with smooth, continuous tonal decay:
/// - Effective range (p95 - p5) < 200.
/// - Continuous Gaussian roll-off across tail bins 0..p5 and p95..255.
/// - tail_emptiness < 0.70 skips range expansion to protect the artist's atmosphere.
pub fn generate_moody_atmospheric_page(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    let max_dist = ((cx * cx) + (cy * cy)).sqrt();

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let dist = (dx * dx + dy * dy).sqrt() / max_dist;
            // Smooth continuous gradient rolling off gently across full tonal range
            let val = (255.0 * (1.0 - dist.min(1.0)).powf(1.5)).clamp(0.0, 255.0) as u8;
            img.put_pixel(x, y, Rgb([val, val, val]));
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// Scenario 6: Artificially Compressed Scan (High Tail Emptiness)
///
/// Hard-clamped tonal range [35, 185] with strictly zero counts in tails:
/// - tail_emptiness == 1.0 >= 0.70 -> triggers Dynamic Range Expansion.
pub fn generate_compressed_scan_page(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let t = (x as f32 + y as f32) / (width as f32 + height as f32);
            let val = (35.0 + t * (185.0 - 35.0)).clamp(35.0, 185.0) as u8;
            img.put_pixel(x, y, Rgb([val, val, val]));
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// Scenario 7: Saturated Primary Colors for Oklab Radial Chord Bisection
///
/// Contains highly saturated colors testing gamut cusp compression:
/// - Pure saturated primary blue: R=10, G=40, B=230
/// - Saturated vermillion red SFX lettering: R=240, G=20, B=20
/// - Saturated emerald green: R=20, G=220, B=30
/// 10-iteration radial chord bisection guarantees 100% hue angle preservation without RGB clipping.
pub fn generate_saturated_oklab_color_page(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let pixel = match (x * 3 / width, y * 3 / height) {
                (0, 0) => Rgb([10, 40, 230]),   // Saturated primary blue
                (1, 0) => Rgb([240, 20, 20]),   // Saturated vermillion red
                (2, 0) => Rgb([20, 220, 30]),   // Saturated emerald green
                (0, 1) => Rgb([240, 220, 10]),  // Saturated yellow
                (1, 1) => Rgb([220, 30, 220]),  // Saturated magenta
                (2, 1) => Rgb([20, 230, 230]),  // Saturated cyan
                _ => {
                    let gray = ((x + y) % 256) as u8;
                    Rgb([gray, gray, gray])
                }
            };
            img.put_pixel(x, y, pixel);
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// Scenario 8: Mixed Manga Page with Hermite Smoothstep Blending
///
/// Contains monochrome manga linework alongside high-chroma color title logo:
/// - Monochrome art: R=G=B (chroma = 0)
/// - Red chapter logo: R=235, G=20, B=25 (chroma = 215 >= 35)
/// - Smooth transition: pixels with chroma between 12 and 35
/// Hermite smoothstep w^2(3 - 2w) preserves color logo 100% unaltered.
pub fn generate_hermite_mixed_page(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            // Top-left title box contains high-chroma red logo
            let in_logo = x >= 30 && x < 200 && y >= 30 && y < 100;
            let pixel = if in_logo && (x % 6 < 4 || y % 6 < 4) {
                Rgb([235, 20, 25]) // High-chroma red logo (chroma = 215 >= 35)
            } else if in_logo {
                Rgb([200, 100, 110]) // Transition chroma element
            } else {
                // Monochrome linework & paper
                let gray = if (x % 40 < 3) || (y % 50 < 3) { 18 } else { 245 };
                Rgb([gray, gray, gray])
            };
            img.put_pixel(x, y, pixel);
        }
    }
    DynamicImage::ImageRgb8(img)
}

/// Scenario 9: Alpha Transparency Masking
///
/// RGBA8 canvas containing transparent background, anti-aliased edge, and opaque art:
/// - Transparent background: A = 0, R=G=B=0 (masked out of statistics).
/// - Edge boundary: A = 64, 100, 127 (masked out, A < 128).
/// - Opaque linework: A = 255, R=G=B=18.
/// Guarantees alpha channel is preserved untouched on output.
pub fn generate_rgba_transparent_page(width: u32, height: u32) -> DynamicImage {
    let mut img = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let in_badge = (x as i32 - width as i32 / 2).pow(2) + (y as i32 - height as i32 / 2).pow(2) < (width as i32 / 3).pow(2);
            let pixel = if in_badge {
                Rgba([20, 20, 20, 255]) // Opaque interior linework
            } else if x % 20 == 0 {
                Rgba([100, 100, 100, 64]) // Semi-transparent edge (A < 128)
            } else {
                Rgba([0, 0, 0, 0]) // Fully transparent canvas
            };
            img.put_pixel(x, y, pixel);
        }
    }
    DynamicImage::ImageRgba8(img)
}

/// Scenario 10: Quality Guard Attenuation & Rollback Test Pairs
///
/// Returns a tuple of (borderline_image, gross_error_image):
/// - Borderline image: Slightly exceeds Tier 2 chroma drift or Tier 3 NCC; rescued by 50% attenuation retry.
/// - Gross error image: Drastic slope collapse and MSE > 220; triggers immediate zero-overhead rollback.
pub fn generate_attenuation_test_pages(width: u32, height: u32) -> (DynamicImage, DynamicImage) {
    let mut borderline = RgbImage::new(width, height);
    let mut gross = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            // Borderline: gentle color gradient with elevated pedestal
            let val = ((x * 160 / width) + 20) as u8;
            borderline.put_pixel(x, y, Rgb([val, val + 5, val + 10]));

            // Gross error: sharp inverted step function designed to fail MSE / PSNR
            let gross_val = if (x / 20) % 2 == 0 { 0 } else { 255 };
            gross.put_pixel(x, y, Rgb([gross_val, gross_val, gross_val]));
        }
    }

    (DynamicImage::ImageRgb8(borderline), DynamicImage::ImageRgb8(gross))
}

/// Helper: Encodes DynamicImage to compressed PNG Bytes
pub fn encode_to_png_bytes(img: &DynamicImage) -> Bytes {
    let mut buf = Vec::new();
    img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
        .expect("Failed to encode synthetic PNG image");
    Bytes::from(buf)
}

/// Helper: Encodes DynamicImage to compressed JPEG Bytes
pub fn encode_to_jpeg_bytes(img: &DynamicImage, quality: u8) -> Bytes {
    let mut buf = Vec::new();
    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
    img.write_with_encoder(encoder)
        .expect("Failed to encode synthetic JPEG image");
    Bytes::from(buf)
}
