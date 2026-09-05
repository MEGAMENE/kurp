use std::io::Cursor;
use std::sync::Arc;

use bytes::Bytes;
use image::codecs::webp::WebPEncoder;
use image::{DynamicImage, ExtendedColorType, ImageEncoder, ImageFormat, ImageReader};
use log::info;
use realcugan_ncnn_vulkan_rs::RealCugan;

use crate::config::app_config::{AppConfig, Format};

#[derive(Copy, Clone)]
pub struct UpscalerConfig {
    threshold_enabled: bool,
    threshold: u32,
    threshold_png: u32,
    return_format: Format,
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

        let mut reader = ImageReader::new(Cursor::new(input.clone()));
        reader.set_format(image_format);
        let image = match reader.decode().or_else(|_| {
            ImageReader::new(Cursor::new(input.clone()))
                .with_guessed_format()
                .unwrap()
                .decode()
        }) {
            Ok(img) => img,
            Err(e) => {
                info!("failed to decode image: {}. Returning original", e);
                return (input, image_format);
            }
        };

        let upscaled = self.upscale_image(image);
        let mut buf = Cursor::new(Vec::new());

        match config.return_format {
            Format::LosslessWebP => {
                let encoder = WebPEncoder::new_lossless(&mut buf);
                let (w, h) = (upscaled.width(), upscaled.height());
                let color_type: ExtendedColorType = upscaled.color().into();
                encoder
                    .write_image(upscaled.as_bytes(), w, h, color_type)
                    .expect("can't write lossless WebP image");
                (Bytes::from(buf.into_inner()), ImageFormat::WebP)
            }
            Format::Png => {
                upscaled.write_to(&mut buf, ImageFormat::Png).expect("can't write image");
                (Bytes::from(buf.into_inner()), ImageFormat::Png)
            }
            Format::Jpeg => {
                upscaled.write_to(&mut buf, ImageFormat::Jpeg).expect("can't write image");
                (Bytes::from(buf.into_inner()), ImageFormat::Jpeg)
            }
            Format::WebP => {
                upscaled.write_to(&mut buf, ImageFormat::WebP).expect("can't write image");
                (Bytes::from(buf.into_inner()), ImageFormat::WebP)
            }
            Format::Original => {
                upscaled.write_to(&mut buf, image_format).expect("can't write image");
                (Bytes::from(buf.into_inner()), image_format)
            }
        }
    }

    fn upscale_image(&self, image: DynamicImage) -> DynamicImage;

    fn get_config(&self) -> UpscalerConfig;
}

pub struct RealCuganUpscaler {
    config: UpscalerConfig,
    realcugan: RealCugan,
}

impl RealCuganUpscaler {
    pub fn new(config: Arc<AppConfig>) -> Self {
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
        );

        let upscaler_config = UpscalerConfig {
            threshold_enabled: config.size_threshold_enabled,
            threshold: config.size_threshold,
            threshold_png: config.size_threshold_png,
            return_format: config.return_format,
        };

        Self {
            config: upscaler_config,
            realcugan,
        }
    }
}

impl Upscaler for RealCuganUpscaler {
    fn upscale_image(&self, image: DynamicImage) -> DynamicImage {
        self.realcugan.proc_image(image)
    }

    fn get_config(&self) -> UpscalerConfig {
        self.config
    }
}