use std::io::Cursor;
use std::sync::Arc;

use bytes::Bytes;
use image::{DynamicImage, ImageFormat};
use img_parts::{DynImage, ImageICC};
use log::{info, warn};
use realcugan_ncnn_vulkan_rs::RealCugan;
use waifu2x_ncnn_vulkan_rs::Waifu2x;
use zenpixels_convert::PixelBufferConvertTypedExt;

use crate::config::app_config::{AppConfig, Format};

#[derive(Copy, Clone)]
pub struct UpscalerConfig {
    threshold_enabled: bool,
    threshold: u32,
    threshold_png: u32,
    return_format: Format,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum SourceClass {
    Grayscale,
    GrayscaleWithColor,
    Color,
}

#[derive(Copy, Clone)]
struct ChromaStats {
    average_spread: f64,
    max_spread: u8,
    pixels_over_tolerance: usize,
    total_pixels: usize,
    has_localized_color: bool,
}

impl ChromaStats {
    fn percentage_over_tolerance(self) -> f64 {
        if self.total_pixels == 0 { 0.0 } else { self.pixels_over_tolerance as f64 * 100.0 / self.total_pixels as f64 }
    }
}

pub trait Upscaler: Send {
    fn upscale(&self, input: Bytes, image_format: ImageFormat, source_name: &str) -> (Bytes, ImageFormat) {
        let config = self.get_config();
        if config.threshold_enabled {
            let input_kb = (input.len() / 1024) as u32;
            let threshold = if image_format == ImageFormat::Png { config.threshold_png } else { config.threshold };
            if input_kb > threshold {
                info!("image {}: size {} is bigger than threshold {}. skipping upscale", source_name, input_kb, threshold);
                return (input, image_format);
            }
        }

        let image = match image_format {
            ImageFormat::Avif => match decode_avif(&input) {
                Ok(image) => image,
                Err(error) => {
                    warn!("image {}: can't decode AVIF image: {error}", source_name);
                    return (input, image_format);
                }
            },
            _ => {
                let mut reader = image::io::Reader::new(Cursor::new(input.clone()));
                reader.set_format(image_format);
                match reader.decode().or_else(|_| {
                    image::io::Reader::new(Cursor::new(input.clone()))
                        .with_guessed_format()
                        .map_err(image::ImageError::IoError)
                        .and_then(|reader| reader.decode())
                }) {
                    Ok(image) => image,
                    Err(error) => {
                        warn!("image {}: can't decode image: {error}", source_name);
                        return (input, image_format);
                    }
                }
            }
        };

        // Classify the ORIGINAL pixels before the neural upscaler runs. The
        // upscaler can introduce chroma into an otherwise grayscale page, so
        // post-processing must never re-classify the upscaled result.
        let input_stats = chroma_stats(&image);
        let source_class = classify_source(input_stats);
        info!(
            "grayscale diagnostic [{}]: input avg chroma {:.3}, max chroma {}, pixels over tolerance {:.3}%, localized color {} -> {:?}",
            source_name,
            input_stats.average_spread,
            input_stats.max_spread,
            input_stats.percentage_over_tolerance(),
            input_stats.has_localized_color,
            source_class
        );

        let upscaled = self.upscale_image(image);
        let output_stats = chroma_stats(&upscaled);
        info!(
            "grayscale diagnostic [{}]: upscaled avg chroma {:.3}, max chroma {}, pixels over tolerance {:.3}%",
            source_name,
            output_stats.average_spread,
            output_stats.max_spread,
            output_stats.percentage_over_tolerance()
        );

        let upscaled = match source_class {
            SourceClass::Grayscale => {
                info!("grayscale diagnostic [{}]: pure grayscale source; enforcing neutral RGB and correcting levels", source_name);
                normalize_grayscale(&upscaled)
            }
            SourceClass::GrayscaleWithColor => {
                info!("grayscale diagnostic [{}]: grayscale source with localized color; correcting luminance levels while preserving color", source_name);
                normalize_luminance_preserve_color(&upscaled)
            }
            SourceClass::Color => {
                info!("grayscale diagnostic [{}]: color source; no grayscale/levels post-processing", source_name);
                upscaled
            }
        };

        let final_stats = chroma_stats(&upscaled);
        info!(
            "grayscale diagnostic [{}]: final avg chroma {:.3}, max chroma {}, pixels over tolerance {:.3}%",
            source_name,
            final_stats.average_spread,
            final_stats.max_spread,
            final_stats.percentage_over_tolerance()
        );

        let mut buf = Cursor::new(Vec::new());
        let format_to = match config.return_format {
            Format::Png => ImageFormat::Png,
            Format::Jpeg => ImageFormat::Jpeg,
            Format::WebP => ImageFormat::WebP,
            Format::Original => image_format,
        };

        if let Err(error) = upscaled.write_to(&mut buf, format_to) {
            warn!("image {}: can't write upscaled image: {error}", source_name);
            return (input, image_format);
        }

        let output = Bytes::from(buf.into_inner());
        let output = if image_format == ImageFormat::Avif { output } else { preserve_icc_profile(&input, output) };
        (output, format_to)
    }

    fn upscale_image(&self, image: DynamicImage) -> DynamicImage;
    fn get_config(&self) -> UpscalerConfig;
}

fn classify_source(stats: ChromaStats) -> SourceClass {
    const MAX_GRAYSCALE_AVERAGE_SPREAD: f64 = 2.0;
    const MAX_GRAYSCALE_PIXELS_OVER_TOLERANCE_PERCENT: f64 = 10.0;
    const MAX_BW_COLOR_AVERAGE_SPREAD: f64 = 12.0;
    const MAX_BW_COLOR_PIXELS_OVER_TOLERANCE_PERCENT: f64 = 35.0;

    let is_low_chroma = stats.average_spread <= MAX_GRAYSCALE_AVERAGE_SPREAD
        && stats.percentage_over_tolerance() <= MAX_GRAYSCALE_PIXELS_OVER_TOLERANCE_PERCENT;

    if is_low_chroma {
        if stats.has_localized_color { SourceClass::GrayscaleWithColor } else { SourceClass::Grayscale }
    } else if stats.average_spread <= MAX_BW_COLOR_AVERAGE_SPREAD
        && stats.percentage_over_tolerance() <= MAX_BW_COLOR_PIXELS_OVER_TOLERANCE_PERCENT
        && stats.has_localized_color
    {
        SourceClass::GrayscaleWithColor
    } else {
        SourceClass::Color
    }
}

fn chroma_stats(image: &DynamicImage) -> ChromaStats {
    let rgb = image.to_rgb8();
    let width = rgb.width() as usize;
    let height = rgb.height() as usize;
    let total_pixels = width * height;
    let mut total_spread = 0u64;
    let mut max_spread = 0u8;
    let mut pixels_over_tolerance = 0usize;
    let mut colour_mask = vec![false; total_pixels];

    for (index, pixel) in rgb.pixels().enumerate() {
        let r = pixel[0] as i16;
        let g = pixel[1] as i16;
        let b = pixel[2] as i16;
        let spread = (r - g).abs().max((r - b).abs()).max((g - b).abs()) as u8;
        total_spread += spread as u64;
        max_spread = max_spread.max(spread);
        if spread > 3 { pixels_over_tolerance += 1; }
        colour_mask[index] = spread >= 12;
    }

    let has_localized_color = has_significant_color_component(&colour_mask, width, height);

    ChromaStats {
        average_spread: if total_pixels == 0 { 0.0 } else { total_spread as f64 / total_pixels as f64 },
        max_spread,
        pixels_over_tolerance,
        total_pixels,
        has_localized_color,
    }
}

fn has_significant_color_component(mask: &[bool], width: usize, height: usize) -> bool {
    const MIN_COMPONENT_PIXELS: usize = 8;
    if width == 0 || height == 0 { return false; }

    let mut visited = vec![false; mask.len()];
    let mut stack = Vec::with_capacity(64);

    for start in 0..mask.len() {
        if !mask[start] || visited[start] { continue; }
        visited[start] = true;
        stack.push(start);
        let mut component_size = 0usize;

        while let Some(index) = stack.pop() {
            component_size += 1;
            if component_size >= MIN_COMPONENT_PIXELS { return true; }

            let x = index % width;
            let y = index / width;
            let x_start = x.saturating_sub(1);
            let x_end = (x + 1).min(width - 1);
            let y_start = y.saturating_sub(1);
            let y_end = (y + 1).min(height - 1);

            for ny in y_start..=y_end {
                for nx in x_start..=x_end {
                    let neighbour = ny * width + nx;
                    if mask[neighbour] && !visited[neighbour] {
                        visited[neighbour] = true;
                        stack.push(neighbour);
                    }
                }
            }
        }
    }

    false
}

fn normalize_grayscale(image: &DynamicImage) -> DynamicImage {
    let gray = normalize_luma_channel(&image.to_luma8());
    let mut rgb = image::RgbImage::new(gray.width(), gray.height());
    for (x, y, pixel) in gray.enumerate_pixels() {
        let value = pixel[0];
        rgb.put_pixel(x, y, image::Rgb([value, value, value]));
    }
    DynamicImage::ImageRgb8(rgb)
}

fn normalize_luminance_preserve_color(image: &DynamicImage) -> DynamicImage {
    let rgb = image.to_rgb8();
    let mut luminance = image::GrayImage::new(rgb.width(), rgb.height());

    for (x, y, pixel) in rgb.enumerate_pixels() {
        luminance.put_pixel(x, y, image::Luma([rgb_luminance(pixel[0], pixel[1], pixel[2])]));
    }

    let normalized_luminance = normalize_luma_channel(&luminance);
    let mut output = image::RgbImage::new(rgb.width(), rgb.height());

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let old_y = luminance.get_pixel(x, y)[0] as f32;
        let new_y = normalized_luminance.get_pixel(x, y)[0] as f32;
        let (r, g, b) = ycbcr_rescale(pixel[0], pixel[1], pixel[2], old_y, new_y);
        output.put_pixel(x, y, image::Rgb([r, g, b]));
    }

    DynamicImage::ImageRgb8(output)
}

fn normalize_luma_channel(image: &image::GrayImage) -> image::GrayImage {
    let (black_point, white_point) = percentile_points(image);
    if black_point == 0 && white_point == 255 { return image.clone(); }
    if white_point <= black_point { return image.clone(); }

    let range = (white_point - black_point) as u16;
    let mut output = image::GrayImage::new(image.width(), image.height());

    for (x, y, pixel) in image.enumerate_pixels() {
        let value = pixel[0] as i16;
        let mapped = if value <= black_point as i16 {
            0
        } else if value >= white_point as i16 {
            255
        } else {
            (((value - black_point as i16) as u16 * 255 + range / 2) / range) as u8
        };
        output.put_pixel(x, y, image::Luma([mapped]));
    }
    output
}

fn percentile_points(image: &image::GrayImage) -> (u8, u8) {
    const LOW_PERCENTILE: f64 = 0.005;
    const HIGH_PERCENTILE: f64 = 0.995;

    let mut histogram = [0u64; 256];
    for pixel in image.pixels() { histogram[pixel[0] as usize] += 1; }

    let total = image.width() as u64 * image.height() as u64;
    if total == 0 { return (0, 255); }

    let low_target = ((total as f64 * LOW_PERCENTILE).ceil() as u64).max(1);
    let high_target = ((total as f64 * HIGH_PERCENTILE).ceil() as u64).min(total);

    let mut cumulative = 0u64;
    let mut low = 0u8;
    for (value, count) in histogram.iter().enumerate() {
        cumulative += *count;
        if cumulative >= low_target { low = value as u8; break; }
    }

    cumulative = 0;
    let mut high = 255u8;
    for (value, count) in histogram.iter().enumerate() {
        cumulative += *count;
        if cumulative >= high_target { high = value as u8; break; }
    }
    (low, high)
}

fn rgb_luminance(r: u8, g: u8, b: u8) -> u8 {
    ((299u32 * r as u32 + 587u32 * g as u32 + 114u32 * b as u32 + 500) / 1000) as u8
}

fn ycbcr_rescale(r: u8, g: u8, b: u8, old_y: f32, new_y: f32) -> (u8, u8, u8) {
    let cb = b as f32 - old_y;
    let cr = r as f32 - old_y;
    let out_r = new_y + 1.402f32 * cr;
    let out_g = new_y - 0.344136f32 * cb - 0.714136f32 * cr;
    let out_b = new_y + 1.772f32 * cb;
    (clamp_u8(out_r), clamp_u8(out_g), clamp_u8(out_b))
}

fn clamp_u8(value: f32) -> u8 { value.round().clamp(0.0, 255.0) as u8 }

fn decode_avif(input: &Bytes) -> Result<DynamicImage, String> {
    let decoded = zenavif::decode(input.as_ref()).map_err(|error| format!("{error:?}"))?;
    let width = decoded.width();
    let height = decoded.height();
    let has_alpha = decoded.descriptor().format.channels() == 4;
    if has_alpha {
        let rgba = decoded.to_rgba8();
        let pixels = rgba.copy_to_contiguous_bytes();
        image::RgbaImage::from_raw(width, height, pixels)
            .map(DynamicImage::ImageRgba8)
            .ok_or_else(|| "decoded AVIF has an invalid RGBA pixel buffer".to_string())
    } else {
        let rgb = decoded.to_rgb8();
        let pixels = rgb.copy_to_contiguous_bytes();
        image::RgbImage::from_raw(width, height, pixels)
            .map(DynamicImage::ImageRgb8)
            .ok_or_else(|| "decoded AVIF has an invalid RGB pixel buffer".to_string())
    }
}

fn preserve_icc_profile(input: &Bytes, output: Bytes) -> Bytes {
    let input_image = match DynImage::from_bytes(input.to_vec().into()) {
        Ok(Some(image)) => image,
        Ok(None) | Err(_) => return output,
    };
    let profile = match input_image.icc_profile() {
        Some(profile) => profile.to_vec(),
        None => return output,
    };
    let mut output_image = match DynImage::from_bytes(output.to_vec().into()) {
        Ok(Some(image)) => image,
        Ok(None) => return output,
        Err(error) => {
            warn!("can't parse upscaled image for ICC profile preservation: {error}");
            return output;
        }
    };
    output_image.set_icc_profile(Some(profile.into()));
    let mut writer = Cursor::new(Vec::new());
    if let Err(error) = output_image.encoder().write_to(&mut writer) {
        warn!("can't write upscaled image after restoring ICC profile: {error}");
        return output;
    }
    Bytes::from(writer.into_inner())
}

pub struct Waifu2xUpscaler { config: UpscalerConfig, waifu2x: Waifu2x }
pub struct RealCuganUpscaler { config: UpscalerConfig, realcugan: RealCugan }

impl Waifu2xUpscaler {
    pub fn new(config: Arc<AppConfig>) -> Self {
        let waifu2x = Waifu2x::new(config.waifu2x.gpuid, config.waifu2x.noise, config.waifu2x.scale, config.waifu2x.model, config.waifu2x.tile_size, config.waifu2x.tta_mode, config.waifu2x.num_threads, config.waifu2x.models_path.clone());
        let upscaler_config = UpscalerConfig { threshold_enabled: config.size_threshold_enabled, threshold: config.size_threshold, threshold_png: config.size_threshold_png, return_format: config.return_format };
        Self { config: upscaler_config, waifu2x }
    }
}

impl RealCuganUpscaler {
    pub fn new(config: Arc<AppConfig>) -> Self {
        let realcugan = RealCugan::new(config.realcugan.gpuid, config.realcugan.noise, config.realcugan.scale, config.realcugan.model, config.realcugan.tile_size, config.realcugan.sync_gap, config.realcugan.tta_mode, config.realcugan.num_threads, config.realcugan.models_path.clone());
        let upscaler_config = UpscalerConfig { threshold_enabled: config.size_threshold_enabled, threshold: config.size_threshold, threshold_png: config.size_threshold_png, return_format: config.return_format };
        Self { config: upscaler_config, realcugan }
    }
}

impl Upscaler for Waifu2xUpscaler {
    fn upscale_image(&self, image: DynamicImage) -> DynamicImage { self.waifu2x.proc_image(image) }
    fn get_config(&self) -> UpscalerConfig { self.config }
}

impl Upscaler for RealCuganUpscaler {
    fn upscale_image(&self, image: DynamicImage) -> DynamicImage { self.realcugan.proc_image(image) }
    fn get_config(&self) -> UpscalerConfig { self.config }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_range_grayscale_is_unchanged() {
        let mut image = image::GrayImage::new(2, 1);
        image.put_pixel(0, 0, image::Luma([0]));
        image.put_pixel(1, 0, image::Luma([255]));
        assert_eq!(percentile_points(&image), (0, 255));
        assert_eq!(normalize_luma_channel(&image), image);
    }

    #[test]
    fn narrow_grayscale_range_is_expanded() {
        let mut image = image::GrayImage::new(100, 1);
        for x in 0..100 { image.put_pixel(x, 0, image::Luma([50 + (x as u8 / 2)])); }
        let normalized = normalize_luma_channel(&image);
        assert_eq!(normalized.get_pixel(0, 0)[0], 0);
        assert_eq!(normalized.get_pixel(99, 0)[0], 255);
    }

    #[test]
    fn localized_color_is_classified_as_grayscale_with_color() {
        let mut image = image::RgbImage::from_pixel(100, 100, image::Rgb([128, 128, 128]));
        for y in 40..60 { for x in 40..60 { image.put_pixel(x, y, image::Rgb([255, 0, 0])); } }
        let stats = chroma_stats(&DynamicImage::ImageRgb8(image));
        assert_eq!(classify_source(stats), SourceClass::GrayscaleWithColor);
    }

    #[test]
    fn full_color_image_is_not_treated_as_grayscale() {
        let image = image::RgbImage::from_pixel(10, 10, image::Rgb([255, 0, 0]));
        let stats = chroma_stats(&DynamicImage::ImageRgb8(image));
        assert_eq!(classify_source(stats), SourceClass::Color);
    }
}
