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

#[derive(Copy, Clone)]
struct ChromaStats {
    average_spread: f64,
    max_spread: u8,
    pixels_over_tolerance: usize,
    total_pixels: usize,
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

        // Real-CUGAN is a colour model. For pages that are effectively grayscale,
        // its reconstruction can introduce a small chroma bias. Detect grayscale
        // input and normalize the model output back to neutral RGB. This is
        // deliberately based on the decoded pixels, so it applies equally to
        // AVIF, PNG, WebP, JPEG, etc. and is not an AVIF-specific conversion.
        let input_stats = chroma_stats(&image);
        let preserve_grayscale = is_effectively_grayscale(input_stats);
        info!(
            "grayscale diagnostic [{}]: input avg chroma {:.3}, max chroma {}, pixels over tolerance {:.3}% -> {}",
            source_name,
            input_stats.average_spread,
            input_stats.max_spread,
            input_stats.percentage_over_tolerance(),
            if preserve_grayscale { "GRAYSCALE" } else { "COLOR" }
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

        let upscaled = if preserve_grayscale {
            info!("grayscale diagnostic [{}]: applying grayscale normalization", source_name);
            let normalized = grayscale_rgb(&upscaled);
            let normalized_stats = chroma_stats(&normalized);
            info!(
                "grayscale diagnostic [{}]: normalized avg chroma {:.3}, max chroma {}, pixels over tolerance {:.3}%",
                source_name,
                normalized_stats.average_spread,
                normalized_stats.max_spread,
                normalized_stats.percentage_over_tolerance()
            );
            normalized
        } else {
            info!("grayscale diagnostic [{}]: no grayscale normalization applied", source_name);
            upscaled
        };

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

fn is_effectively_grayscale(stats: ChromaStats) -> bool {
    const MAX_AVERAGE_SPREAD: f64 = 2.0;
    const MAX_PIXELS_OVER_TOLERANCE_PERCENT: f64 = 10.0;

    stats.average_spread <= MAX_AVERAGE_SPREAD
        && stats.percentage_over_tolerance() <= MAX_PIXELS_OVER_TOLERANCE_PERCENT
}

fn chroma_stats(image: &DynamicImage) -> ChromaStats {
    let rgb = image.to_rgb8();
    let mut total_spread = 0u64;
    let mut max_spread = 0u8;
    let mut pixels_over_tolerance = 0usize;
    let total_pixels = rgb.width() as usize * rgb.height() as usize;

    for pixel in rgb.pixels() {
        let r = pixel[0] as i16;
        let g = pixel[1] as i16;
        let b = pixel[2] as i16;
        let spread = (r - g).abs().max((r - b).abs()).max((g - b).abs()) as u8;
        total_spread += spread as u64;
        max_spread = max_spread.max(spread);
        if spread > 3 { pixels_over_tolerance += 1; }
    }

    ChromaStats {
        average_spread: if total_pixels == 0 { 0.0 } else { total_spread as f64 / total_pixels as f64 },
        max_spread,
        pixels_over_tolerance,
        total_pixels,
    }
}

fn grayscale_rgb(image: &DynamicImage) -> DynamicImage {
    let gray = image.to_luma8();
    let width = gray.width();
    let height = gray.height();
    let mut rgb = image::RgbImage::new(width, height);
    for (x, y, pixel) in gray.enumerate_pixels() {
        let value = pixel[0];
        rgb.put_pixel(x, y, image::Rgb([value, value, value]));
    }
    DynamicImage::ImageRgb8(rgb)
}

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
