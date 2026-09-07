use std::io::Cursor;
use std::sync::Arc;

use bytes::Bytes;
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::PngEncoder;
use image::codecs::webp::WebPEncoder;
use image::{DynamicImage, ExtendedColorType, ImageDecoder, ImageEncoder, ImageFormat, ImageReader};
use log::{error, info};
use realcugan_ncnn_vulkan_rs::{RealCugan, RealCuganError};

use crate::config::app_config::{AppConfig, Format};

#[derive(Copy, Clone)]
pub struct UpscalerConfig {
    threshold_enabled: bool,
    threshold: u32,
    threshold_png: u32,
    return_format: Format,
}

fn encode_image(
    image: &DynamicImage,
    target_format: ImageFormat,
    icc_profile: Option<&[u8]>,
) -> Bytes {
    let mut buf = Cursor::new(Vec::new());
    match target_format {
        ImageFormat::WebP => {
            let mut encoder = WebPEncoder::new_lossless(&mut buf);
            if let Some(icc) = icc_profile {
                if let Err(e) = encoder.set_icc_profile(icc.to_vec()) {
                    error!("failed to set ICC profile on WebP encoder: {}", e);
                }
            }
            let (w, h) = (image.width(), image.height());
            let color_type: ExtendedColorType = image.color().into();
            encoder
                .write_image(image.as_bytes(), w, h, color_type)
                .expect("can't write lossless WebP image");
        }
        ImageFormat::Png => {
            let mut encoder = PngEncoder::new(&mut buf);
            if let Some(icc) = icc_profile {
                if let Err(e) = encoder.set_icc_profile(icc.to_vec()) {
                    error!("failed to set ICC profile on PNG encoder: {}", e);
                }
            }
            let (w, h) = (image.width(), image.height());
            let color_type: ExtendedColorType = image.color().into();
            encoder
                .write_image(image.as_bytes(), w, h, color_type)
                .expect("can't write PNG image");
        }
        ImageFormat::Jpeg => {
            let mut encoder = JpegEncoder::new(&mut buf);
            if let Some(icc) = icc_profile {
                if let Err(e) = encoder.set_icc_profile(icc.to_vec()) {
                    error!("failed to set ICC profile on JPEG encoder: {}", e);
                }
            }
            let (w, h) = (image.width(), image.height());
            match image {
                DynamicImage::ImageRgb8(ref rgb) => {
                    encoder
                        .write_image(rgb.as_raw(), w, h, ExtendedColorType::Rgb8)
                        .expect("can't write JPEG image");
                }
                DynamicImage::ImageLuma8(ref luma) => {
                    encoder
                        .write_image(luma.as_raw(), w, h, ExtendedColorType::L8)
                        .expect("can't write JPEG image");
                }
                _ => {
                    let rgb = image.to_rgb8();
                    encoder
                        .write_image(rgb.as_raw(), w, h, ExtendedColorType::Rgb8)
                        .expect("can't write JPEG image");
                }
            }
        }
        other => {
            image.write_to(&mut buf, other).expect("can't write image");
        }
    }
    Bytes::from(buf.into_inner())
}

pub trait Upscaler: Send {
    fn upscale(&self, input: Bytes, image_format: ImageFormat) -> (Bytes, ImageFormat) {
        let config = self.get_config();
        if config.threshold_enabled {
            let input_kb = (input.len() / 1024) as u32;
            let threshold = if image_format == ImageFormat::Png { config.threshold_png } else { config.threshold };
            if input_kb > threshold {
                info!("image size {} is bigger than threshold {}. skipping upscale", input_kb, threshold);
                return (input, image_format);
            }
        }

        let mut reader = match ImageReader::new(Cursor::new(input.clone())).with_guessed_format() {
            Ok(r) => r,
            Err(e) => {
                info!("failed to guess format: {}. Returning original", e);
                return (input, image_format);
            }
        };
        if reader.format().is_none() {
            reader.set_format(image_format);
        }
        let detected_format = reader.format().unwrap_or(image_format);

        let is_avif = detected_format == ImageFormat::Avif || image_format == ImageFormat::Avif;
        let avif_color_info = if is_avif {
            crate::upscaler::avif::parse_avif_color_info(&input)
        } else {
            None
        };

        let mut decoder = match reader.into_decoder() {
            Ok(d) => d,
            Err(_) => {
                let mut fallback_reader = ImageReader::new(Cursor::new(input.clone()));
                fallback_reader.set_format(image_format);
                match fallback_reader.into_decoder() {
                    Ok(d) => d,
                    Err(e) => {
                        info!("failed to create decoder for image: {}. Returning original", e);
                        return (input, image_format);
                    }
                }
            }
        };

        let icc_profile = decoder.icc_profile().ok().flatten().or_else(|| {
            avif_color_info.as_ref().and_then(|info| info.icc_profile.clone())
        });
        if let Some(ref icc) = icc_profile {
            info!("detected embedded/synthesized ICC color profile ({} bytes), preserving in output", icc.len());
        }

        let mut image = match DynamicImage::from_decoder(decoder) {
            Ok(img) => img,
            Err(e) => {
                info!("failed to decode image: {}. Returning original", e);
                return (input, image_format);
            }
        };

        if let Some(ref info) = avif_color_info {
            if info.should_correct_matrix() {
                info!("applying BT.601 matrix correction for AVIF image (fixing SMPTE 170M bug)");
                crate::upscaler::avif::correct_avif_matrix(&mut image);
            }
        }

        let upscaled = self.upscale_image(image);

        let (output_bytes, final_format) = match config.return_format {
            Format::LosslessWebP | Format::WebP => {
                (encode_image(&upscaled, ImageFormat::WebP, icc_profile.as_deref()), ImageFormat::WebP)
            }
            Format::Png => {
                (encode_image(&upscaled, ImageFormat::Png, icc_profile.as_deref()), ImageFormat::Png)
            }
            Format::Jpeg => {
                (encode_image(&upscaled, ImageFormat::Jpeg, icc_profile.as_deref()), ImageFormat::Jpeg)
            }
            Format::Original => {
                let target_format = match image_format {
                    ImageFormat::Avif => ImageFormat::WebP,
                    other => other,
                };
                (encode_image(&upscaled, target_format, icc_profile.as_deref()), target_format)
            }
        };

        (output_bytes, final_format)
    }

    fn upscale_image(&self, image: DynamicImage) -> DynamicImage;

    fn get_config(&self) -> UpscalerConfig;
}

pub struct RealCuganUpscaler {
    config: UpscalerConfig,
    realcugan: RealCugan,
}

impl RealCuganUpscaler {
    pub fn new(config: Arc<AppConfig>) -> Result<Self, RealCuganError> {
        let realcugan = RealCugan::new(
            config.realcugan.gpuid,
            config.realcugan.noise,
            config.realcugan.scale,
            config.realcugan.model,
            config.realcugan.tile_size,
            config.realcugan.sync_gap,
            config.realcugan.tta_mode,
            config.realcugan.num_threads,
            config.realcugan.models_path.clone(),
        )?;

        let upscaler_config = UpscalerConfig {
            threshold_enabled: config.size_threshold_enabled,
            threshold: config.size_threshold,
            threshold_png: config.size_threshold_png,
            return_format: config.return_format,
        };

        Ok(Self {
            config: upscaler_config,
            realcugan,
        })
    }
}

impl Upscaler for RealCuganUpscaler {
    fn upscale_image(&self, image: DynamicImage) -> DynamicImage {
        match self.realcugan.proc_image(&image) {
            Ok(upscaled) => upscaled,
            Err(e) => {
                error!("Real-CUGAN upscale error: {}. Falling back to original image.", e);
                image
            }
        }
    }

    fn get_config(&self) -> UpscalerConfig {
        self.config
    }
}