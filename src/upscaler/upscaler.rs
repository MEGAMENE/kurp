use std::io::Cursor;
use std::num::NonZeroU32;
use std::sync::Arc;

use arc_swap::ArcSwap;
use bytes::Bytes;
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::{CompressionType, FilterType, PngEncoder};
use image::{DynamicImage, ExtendedColorType, GrayImage, ImageEncoder, ImageFormat};
use log::{error, info, warn};
use rayon::prelude::*;
use realcugan_ncnn_vulkan_rs::{
    RealCugan, RealCuganConfig as Layer2RealCuganConfig, RealCuganError, RealCuganGpuType,
    RealCuganNoise, RealCuganScale, RealCuganSyncGap, RealCuganTtaLevel,
};
use tokio::sync::{Mutex, Semaphore};

use crate::config::app_config::{AppConfig, Format};
use crate::upscaler::auto_levels::{LevelInfo, PageClassification};

/// Maximum dimension supported by the WebP container/VP8L bitstream (`16383 x 16383`).
pub const WEBP_MAX_DIMENSION: u32 = 16383;

struct LoadedUpscaler {
    realcugan: Arc<RealCugan>,
    semaphore: Arc<Semaphore>,
    concurrent_jobs: u32,
    scale: u32,
}

/// Lock-free hot-swappable Real-CUGAN inference engine backed by Layer 2 `2706874`
/// (`Arc<RealCugan>` is `Send + Sync`) with bounded `Semaphore` concurrency and
/// `tokio::task::spawn_blocking` dispatch.
pub struct UpscaleEngine {
    loaded: Arc<ArcSwap<Option<LoadedUpscaler>>>,
    reload_lock: Mutex<()>,
    active_config: Mutex<Option<AppConfig>>,
}

impl UpscaleEngine {
    /// Creates an unloaded engine (used when `config.upscale == false` or in unit/HTTP tests).
    pub fn new_empty() -> Self {
        Self {
            loaded: Arc::new(ArcSwap::from_pointee(None)),
            reload_lock: Mutex::new(()),
            active_config: Mutex::new(None),
        }
    }

    /// Initializes the upscaler engine asynchronously on a blocking worker thread.
    /// When `config.upscale == false`, GPU/model initialization is skipped cleanly.
    pub async fn init(config: &AppConfig) -> Result<Self, String> {
        let engine = Self::new_empty();
        engine.reload(config).await?;
        Ok(engine)
    }

    #[inline]
    pub fn is_loaded(&self) -> bool {
        self.loaded.load().is_some()
    }

    #[inline]
    pub fn scale(&self) -> Option<u32> {
        self.loaded.load().as_ref().as_ref().map(|l| l.scale)
    }

    /// Runs Real-CUGAN upscaling asynchronously without blocking Tokio worker threads.
    ///
    /// Acquires a bounded permit from the `concurrent_jobs` semaphore before spawning onto the
    /// blocking pool, so disconnected/dropped client futures cancel cleanly before touching the
    /// GPU. Never silently returns un-upscaled pixels on error.
    pub async fn upscale(&self, image: DynamicImage) -> Result<DynamicImage, RealCuganError> {
        let guard = self.loaded.load_full();
        let Some(loaded) = guard.as_ref() else {
            return Err(RealCuganError::GpuInitFailed);
        };

        let permit = loaded
            .semaphore
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| RealCuganError::GpuInitFailed)?;
        let realcugan = loaded.realcugan.clone();

        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            realcugan.proc_image(&image)
        })
        .await
        .map_err(|e| {
            error!("Real-CUGAN blocking worker panicked or was cancelled: {e}");
            RealCuganError::ProcessFailed(-1)
        })?
    }

    /// Validates `config`, drains in-flight jobs to release VRAM cleanly (preventing dual-model
    /// VRAM OOM overlap), builds the new `RealCugan` on `spawn_blocking`, and swaps atomically.
    pub async fn reload(&self, config: &AppConfig) -> Result<(), String> {
        let _reload_guard = self.reload_lock.lock().await;
        config.validate()?;

        if !config.upscale {
            let prev = self.loaded.swap(Arc::new(None));
            drain_loaded(prev).await;
            *self.active_config.lock().await = Some(config.clone());
            info!("Real-CUGAN upscaling disabled (`upscale: false`); GPU instance unloaded");
            return Ok(());
        }

        // Pre-verify target model files exist on disk BEFORE draining/unloading the active model
        // so a misconfigured reload never breaks a working upscaler.
        let (param_path, bin_path) = config.realcugan.resolve_model_paths()?;
        if !param_path.is_file() {
            return Err(format!(
                "Real-CUGAN param file not found: {}",
                param_path.display()
            ));
        }
        if !bin_path.is_file() {
            return Err(format!(
                "Real-CUGAN model binary not found: {}",
                bin_path.display()
            ));
        }

        // Drain & drop any previous model BEFORE allocating the new Vulkan/NCNN model so VRAM
        // never peaks at 2x during hot-reload.
        let prev = self.loaded.swap(Arc::new(None));
        drain_loaded(prev).await;

        let cfg_clone = config.clone();
        let build_res = tokio::task::spawn_blocking(move || build_loaded_upscaler(&cfg_clone))
            .await
            .map_err(|e| format!("Real-CUGAN initialization worker panicked: {e}"))?;

        match build_res {
            Ok(new_loaded) => {
                self.loaded.store(Arc::new(Some(new_loaded)));
                *self.active_config.lock().await = Some(config.clone());
                Ok(())
            }
            Err(err) => {
                error!("Failed to initialize new Real-CUGAN instance during reload: {err}");
                // Best-effort restore of previous valid configuration if one was running
                let prev_cfg_opt = self.active_config.lock().await.clone();
                if let Some(prev_cfg) = prev_cfg_opt
                    && prev_cfg.upscale
                {
                    warn!("Attempting to restore previous Real-CUGAN instance after failed reload");
                    if let Ok(Ok(restored)) =
                        tokio::task::spawn_blocking(move || build_loaded_upscaler(&prev_cfg)).await
                    {
                        self.loaded.store(Arc::new(Some(restored)));
                    }
                }
                Err(err)
            }
        }
    }

    /// Drops the active `Arc<RealCugan>` cleanly before `main()` exits so NCNN's C `atexit`
    /// Vulkan device teardown never races with a live `RealCUGAN` destructor.
    pub fn shutdown(&self) {
        let prev = self.loaded.swap(Arc::new(None));
        drop(prev);
        info!("Real-CUGAN engine shut down cleanly");
    }
}

async fn drain_loaded(prev: Arc<Option<LoadedUpscaler>>) {
    if let Some(ref loaded) = *prev {
        let _permits = loaded
            .semaphore
            .acquire_many(loaded.concurrent_jobs)
            .await;
    }
    drop(prev);
}

fn build_loaded_upscaler(config: &AppConfig) -> Result<LoadedUpscaler, String> {
    let rc = &config.realcugan;
    let scale = RealCuganScale::try_from(rc.scale).map_err(|e| e.to_string())?;
    let noise = RealCuganNoise::try_from(rc.noise).map_err(|e| e.to_string())?;
    let sync_gap = RealCuganSyncGap::try_from(rc.sync_gap).map_err(|e| e.to_string())?;
    let tta_level = RealCuganTtaLevel::try_from(rc.tta_level).map_err(|e| e.to_string())?;
    let concurrent_jobs = NonZeroU32::new(rc.concurrent_jobs)
        .ok_or_else(|| "realcugan.concurrent_jobs must be >= 1".to_string())?;
    let num_threads = NonZeroU32::new(rc.num_threads);

    if rc.gpuid >= 0 && !RealCugan::should_use_cpu(rc.gpuid) {
        let gpu_count = RealCugan::get_gpu_count();
        let dev_type = RealCugan::get_gpu_device_type(rc.gpuid);
        let heap_mb = RealCugan::get_heap_budget(rc.gpuid);
        let dev_type_str = match dev_type {
            Some(RealCuganGpuType::Discrete) => "Discrete GPU",
            Some(RealCuganGpuType::Integrated) => "Integrated GPU",
            Some(RealCuganGpuType::Virtual) => "Virtual GPU",
            Some(RealCuganGpuType::Cpu) => "CPU",
            Some(RealCuganGpuType::Unknown(_)) | None => "Unknown",
        };
        info!(
            "Real-CUGAN GPU #{} detected: {} (VRAM heap budget: {} MB, total GPUs: {})",
            rc.gpuid, dev_type_str, heap_mb, gpu_count
        );

        if rc.tile_size == 0 {
            let prepadding = RealCugan::default_prepadding(scale);
            let auto_tile =
                RealCugan::compute_auto_tilesize(rc.gpuid, scale, tta_level, concurrent_jobs);
            info!(
                "Real-CUGAN auto tile size: {:?}px (scale: {}x, tta: {}x, concurrent_jobs: {}, prepadding: {})",
                auto_tile,
                scale.as_u32(),
                tta_level.as_i32(),
                concurrent_jobs.get(),
                prepadding
            );
        }
    } else if rc.gpuid < 0 {
        info!("Real-CUGAN configured in explicit CPU execution mode (gpuid={})", rc.gpuid);
    } else {
        info!(
            "Real-CUGAN GPU #{} auto-routed to CPU execution mode (software/unavailable Vulkan device)",
            rc.gpuid
        );
    }

    let layer2_cfg = Layer2RealCuganConfig {
        gpuid: rc.gpuid,
        model: rc.model,
        scale,
        noise,
        models_dir: rc.models_path.clone(),
        tile_size: rc.tile_size,
        sync_gap,
        tta_level,
        num_threads,
        concurrent_jobs,
        use_fp16_arithmetic: rc.use_fp16_arithmetic,
    };

    let realcugan = RealCugan::new(layer2_cfg).map_err(|e| e.to_string())?;
    let scale_u32 = realcugan.scale().as_u32();

    info!(
        "Real-CUGAN initialized successfully: scale={}x, model={:?}, noise={}, tta_level={}x, fp16={}, concurrent_jobs={}",
        scale_u32,
        rc.model,
        rc.noise,
        tta_level.as_i32(),
        rc.use_fp16_arithmetic,
        concurrent_jobs.get(),
    );

    Ok(LoadedUpscaler {
        realcugan: Arc::new(realcugan),
        semaphore: Arc::new(Semaphore::new(concurrent_jobs.get() as usize)),
        concurrent_jobs: concurrent_jobs.get(),
        scale: scale_u32,
    })
}

/// Unified post-upscale monochrome folding guard used by both `fold_monochrome_post_upscale`
/// and `encode_upscaled_output`.
#[inline]
pub fn should_fold_luma(level_info: &LevelInfo, output_grayscale_for_monochrome: bool) -> bool {
    output_grayscale_for_monochrome
        && !level_info.paper_cast_preserved
        && level_info.is_monochrome
        && level_info.classification == PageClassification::Monochrome
}

/// Folds `PageClassification::Monochrome` images to 1-channel `DynamicImage::ImageLuma8`
/// after Real-CUGAN upscaling when `should_fold_luma` is true. Transparent pixels in `Rgba8` /
/// `LumaA8` are composited over white before luminance extraction so transparency never turns black.
pub fn fold_monochrome_post_upscale(
    upscaled: DynamicImage,
    level_info: &LevelInfo,
    output_grayscale_for_monochrome: bool,
) -> DynamicImage {
    if !should_fold_luma(level_info, output_grayscale_for_monochrome) {
        return upscaled;
    }

    match upscaled {
        DynamicImage::ImageLuma8(_) => upscaled,
        DynamicImage::ImageRgb8(rgb) => {
            let (w, h) = (rgb.width(), rgb.height());
            let raw = rgb.as_raw();
            let mut luma_buf = vec![0u8; (w as usize) * (h as usize)];
            luma_buf
                .par_chunks_mut(16384)
                .zip(raw.par_chunks(16384 * 3))
                .for_each(|(dst, src)| {
                    for (d, &[r, g, b]) in dst.iter_mut().zip(src.as_chunks::<3>().0) {
                        *d = ((54 * r as u32 + 183 * g as u32 + 19 * b as u32 + 128) >> 8) as u8;
                    }
                });
            match GrayImage::from_raw(w, h, luma_buf) {
                Some(gray) => DynamicImage::ImageLuma8(gray),
                None => DynamicImage::ImageLuma8(DynamicImage::ImageRgb8(rgb).to_luma8()),
            }
        }
        DynamicImage::ImageRgba8(rgba) => {
            let (w, h) = (rgba.width(), rgba.height());
            let raw = rgba.as_raw();
            let mut luma_buf = vec![0u8; (w as usize) * (h as usize)];
            luma_buf
                .par_chunks_mut(16384)
                .zip(raw.par_chunks(16384 * 4))
                .for_each(|(dst, src)| {
                    for (d, &[pr, pg, pb, pa]) in dst.iter_mut().zip(src.as_chunks::<4>().0) {
                        let a = pa as u32;
                        let inv_a = 255 - a;
                        let r = (pr as u32 * a + 255 * inv_a + 127) / 255;
                        let g = (pg as u32 * a + 255 * inv_a + 127) / 255;
                        let b = (pb as u32 * a + 255 * inv_a + 127) / 255;
                        *d = ((54 * r + 183 * g + 19 * b + 128) >> 8) as u8;
                    }
                });
            match GrayImage::from_raw(w, h, luma_buf) {
                Some(gray) => DynamicImage::ImageLuma8(gray),
                None => DynamicImage::ImageLuma8(DynamicImage::ImageRgba8(rgba).to_luma8()),
            }
        }
        DynamicImage::ImageLumaA8(la) => {
            let (w, h) = (la.width(), la.height());
            let raw = la.as_raw();
            let mut luma_buf = vec![0u8; (w as usize) * (h as usize)];
            luma_buf
                .par_chunks_mut(16384)
                .zip(raw.par_chunks(16384 * 2))
                .for_each(|(dst, src)| {
                    for (d, &[py, pa]) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                        let a = pa as u32;
                        let inv_a = 255 - a;
                        *d = ((py as u32 * a + 255 * inv_a + 127) / 255) as u8;
                    }
                });
            match GrayImage::from_raw(w, h, luma_buf) {
                Some(gray) => DynamicImage::ImageLuma8(gray),
                None => DynamicImage::ImageLuma8(DynamicImage::ImageLumaA8(la).to_luma8()),
            }
        }
        other => DynamicImage::ImageLuma8(other.to_luma8()),
    }
}

/// Encodes `image` to `target_format`, returning `(bytes, actual_format)`.
///
/// - **WebP**: Strictly lossless via `libwebp` (`webpx`) with `ICCP` chunk preservation.
///   Non-8-bit images are normalized to 8-bit before encoding. Images exceeding `16383px` on
///   either axis gracefully fall back to PNG (`ImageFormat::Png`) without panicking.
/// - **JPEG**: Fast slice-based white compositing for `Rgba8 -> Rgb8` and `LumaA8 -> L8`.
/// - **PNG / Other**: Preserves ICC profiles cleanly with zero `.expect()` panics.
pub fn encode_image(
    image: &DynamicImage,
    target_format: ImageFormat,
    icc_profile: Option<&[u8]>,
    jpeg_quality: u8,
) -> Result<(Bytes, ImageFormat), String> {
    let (w, h) = (image.width(), image.height());
    if w == 0 || h == 0 {
        return Err("cannot encode empty image (0x0)".to_string());
    }

    match target_format {
        ImageFormat::WebP => {
            if w > WEBP_MAX_DIMENSION || h > WEBP_MAX_DIMENSION {
                warn!(
                    "Image dimensions ({}x{}) exceed WebP limit ({}); falling back to PNG",
                    w, h, WEBP_MAX_DIMENSION
                );
                return encode_image(image, ImageFormat::Png, icc_profile, jpeg_quality);
            }
            let bytes = encode_webp_lossless(image, icc_profile)?;
            Ok((Bytes::from(bytes), ImageFormat::WebP))
        }
        ImageFormat::Png => {
            let mut buf = Cursor::new(Vec::with_capacity(512 * 1024));
            let mut encoder = PngEncoder::new_with_quality(
                &mut buf,
                CompressionType::Fast,
                FilterType::Adaptive,
            );
            if let Some(icc) = icc_profile
                && let Err(e) = encoder.set_icc_profile(icc.to_vec())
            {
                warn!("failed to set ICC profile on PNG encoder: {e}");
            }

            match image {
                DynamicImage::ImageRgb32F(_) => {
                    let rgb16 = image.to_rgb16();
                    let raw: Vec<u8> = rgb16
                        .as_raw()
                        .iter()
                        .flat_map(|v| v.to_be_bytes())
                        .collect();
                    encoder
                        .write_image(&raw, w, h, ExtendedColorType::Rgb16)
                        .map_err(|e| format!("failed to encode PNG (Rgb32F->Rgb16): {e}"))?;
                }
                DynamicImage::ImageRgba32F(_) => {
                    let rgba16 = image.to_rgba16();
                    let raw: Vec<u8> = rgba16
                        .as_raw()
                        .iter()
                        .flat_map(|v| v.to_be_bytes())
                        .collect();
                    encoder
                        .write_image(&raw, w, h, ExtendedColorType::Rgba16)
                        .map_err(|e| format!("failed to encode PNG (Rgba32F->Rgba16): {e}"))?;
                }
                _ => {
                    let color_type: ExtendedColorType = image.color().into();
                    encoder
                        .write_image(image.as_bytes(), w, h, color_type)
                        .map_err(|e| format!("failed to encode PNG image: {e}"))?;
                }
            }

            Ok((Bytes::from(buf.into_inner()), ImageFormat::Png))
        }
        ImageFormat::Jpeg => {
            let mut buf = Cursor::new(Vec::with_capacity(256 * 1024));
            let mut encoder = JpegEncoder::new_with_quality(&mut buf, jpeg_quality.clamp(1, 100));
            if let Some(icc) = icc_profile
                && let Err(e) = encoder.set_icc_profile(icc.to_vec())
            {
                warn!("failed to set ICC profile on JPEG encoder: {e}");
            }

            match image {
                DynamicImage::ImageRgb8(rgb) => {
                    encoder
                        .write_image(rgb.as_raw(), w, h, ExtendedColorType::Rgb8)
                        .map_err(|e| format!("failed to encode JPEG (Rgb8): {e}"))?;
                }
                DynamicImage::ImageLuma8(luma) => {
                    encoder
                        .write_image(luma.as_raw(), w, h, ExtendedColorType::L8)
                        .map_err(|e| format!("failed to encode JPEG (Luma8): {e}"))?;
                }
                DynamicImage::ImageRgba8(rgba) => {
                    let raw = rgba.as_raw();
                    let pixel_count = (w as usize) * (h as usize);
                    let mut rgb_buf = vec![0u8; pixel_count * 3];
                    rgb_buf
                        .par_chunks_mut(16384 * 3)
                        .zip(raw.par_chunks(16384 * 4))
                        .for_each(|(dst, src)| {
                            for (d, &[pr, pg, pb, pa]) in
                                dst.as_chunks_mut::<3>().0.iter_mut().zip(src.as_chunks::<4>().0)
                            {
                                let a = pa as u32;
                                let inv_a = 255 - a;
                                d[0] = ((pr as u32 * a + 255 * inv_a + 127) / 255) as u8;
                                d[1] = ((pg as u32 * a + 255 * inv_a + 127) / 255) as u8;
                                d[2] = ((pb as u32 * a + 255 * inv_a + 127) / 255) as u8;
                            }
                        });
                    encoder
                        .write_image(&rgb_buf, w, h, ExtendedColorType::Rgb8)
                        .map_err(|e| format!("failed to encode JPEG (Rgba8->Rgb8): {e}"))?;
                }
                DynamicImage::ImageLumaA8(luma_a) => {
                    let raw = luma_a.as_raw();
                    let pixel_count = (w as usize) * (h as usize);
                    let mut luma_buf = vec![0u8; pixel_count];
                    luma_buf
                        .par_chunks_mut(16384)
                        .zip(raw.par_chunks(16384 * 2))
                        .for_each(|(dst, src)| {
                            for (d, &[py, pa]) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                                let a = pa as u32;
                                let inv_a = 255 - a;
                                *d = ((py as u32 * a + 255 * inv_a + 127) / 255) as u8;
                            }
                        });
                    encoder
                        .write_image(&luma_buf, w, h, ExtendedColorType::L8)
                        .map_err(|e| format!("failed to encode JPEG (LumaA8->L8): {e}"))?;
                }
                DynamicImage::ImageLuma16(_) | DynamicImage::ImageLumaA16(_) => {
                    let luma = image.to_luma8();
                    encoder
                        .write_image(luma.as_raw(), w, h, ExtendedColorType::L8)
                        .map_err(|e| format!("failed to encode JPEG (L16->L8): {e}"))?;
                }
                _ => {
                    let rgb = image.to_rgb8();
                    encoder
                        .write_image(rgb.as_raw(), w, h, ExtendedColorType::Rgb8)
                        .map_err(|e| format!("failed to encode JPEG (fallback Rgb8): {e}"))?;
                }
            }

            Ok((Bytes::from(buf.into_inner()), ImageFormat::Jpeg))
        }
        other => {
            let mut buf = Cursor::new(Vec::with_capacity(256 * 1024));
            image
                .write_to(&mut buf, other)
                .map_err(|e| format!("failed to encode image ({other:?}): {e}"))?;
            Ok((Bytes::from(buf.into_inner()), other))
        }
    }
}

fn run_webp_encoder<'a>(
    mut encoder: webpx::Encoder<'a>,
    cfg: webpx::EncoderConfig,
    icc_profile: Option<&'a [u8]>,
) -> Result<Vec<u8>, String> {
    encoder = encoder.config(cfg);
    if let Some(icc) = icc_profile {
        encoder = encoder.icc_profile(icc);
    }
    encoder
        .encode(webpx::Unstoppable)
        .map_err(|e| format!("libwebp lossless encode failed: {e}"))
}

/// Strictly lossless `libwebp` encoding via `webpx` (no lossy option), preserving any ICC
/// profile in a WebP `ICCP` chunk.
fn encode_webp_lossless(image: &DynamicImage, icc_profile: Option<&[u8]>) -> Result<Vec<u8>, String> {
    let (w, h) = (image.width(), image.height());
    // Lossless preset level 3 (method=3, quality=70, near_lossless=100 OFF) + multi-threaded
    // entropy coding gives 3-5x faster encoding and 30-50% smaller files than pure-Rust RLE.
    let cfg = webpx::EncoderConfig::new_lossless_level(3).thread_level(1);

    match image {
        DynamicImage::ImageRgb8(rgb) => {
            run_webp_encoder(webpx::Encoder::new_rgb(rgb.as_raw(), w, h), cfg, icc_profile)
        }
        DynamicImage::ImageRgba8(rgba) => run_webp_encoder(
            webpx::Encoder::new_rgba(rgba.as_raw(), w, h),
            cfg,
            icc_profile,
        ),
        DynamicImage::ImageLuma8(luma) => {
            #[cfg(target_endian = "little")]
            {
                let raw = luma.as_raw();
                let mut argb = vec![0u32; raw.len()];
                argb.par_chunks_mut(16384)
                    .zip(raw.par_chunks(16384))
                    .for_each(|(dst, src)| {
                        for (d, &y) in dst.iter_mut().zip(src.iter()) {
                            *d = 0xFF00_0000u32 | ((y as u32) * 0x0001_0101u32);
                        }
                    });
                run_webp_encoder(webpx::Encoder::new_argb(&argb, w, h), cfg, icc_profile)
            }
            #[cfg(not(target_endian = "little"))]
            {
                let rgb = image.to_rgb8();
                run_webp_encoder(webpx::Encoder::new_rgb(rgb.as_raw(), w, h), cfg, icc_profile)
            }
        }
        DynamicImage::ImageLumaA8(la) => {
            #[cfg(target_endian = "little")]
            {
                let raw = la.as_raw();
                let mut argb = vec![0u32; (w as usize) * (h as usize)];
                argb.par_chunks_mut(16384)
                    .zip(raw.par_chunks(16384 * 2))
                    .for_each(|(dst, src)| {
                        for (d, &[py, pa]) in dst.iter_mut().zip(src.as_chunks::<2>().0) {
                            let y = py as u32;
                            let a = pa as u32;
                            *d = (a << 24) | (y * 0x0001_0101u32);
                        }
                    });
                run_webp_encoder(webpx::Encoder::new_argb(&argb, w, h), cfg, icc_profile)
            }
            #[cfg(not(target_endian = "little"))]
            {
                let rgba = image.to_rgba8();
                run_webp_encoder(
                    webpx::Encoder::new_rgba(rgba.as_raw(), w, h),
                    cfg,
                    icc_profile,
                )
            }
        }
        other => {
            if other.color().has_alpha() {
                let rgba = other.to_rgba8();
                run_webp_encoder(
                    webpx::Encoder::new_rgba(rgba.as_raw(), w, h),
                    cfg,
                    icc_profile,
                )
            } else {
                let rgb = other.to_rgb8();
                run_webp_encoder(webpx::Encoder::new_rgb(rgb.as_raw(), w, h), cfg, icc_profile)
            }
        }
    }
}

/// Folds eligible monochrome pages to 1-channel `Luma8` (dropping RGB ICC when folded) and
/// encodes the upscaled image to the resolved output format.
pub fn encode_upscaled_output(
    upscaled: DynamicImage,
    image_format: ImageFormat,
    level_info: &LevelInfo,
    icc_profile: Option<&[u8]>,
    return_format: Format,
    jpeg_quality: u8,
    output_grayscale_for_monochrome: bool,
) -> Result<(Bytes, ImageFormat), String> {
    let fold = should_fold_luma(level_info, output_grayscale_for_monochrome);
    let (final_image, final_icc) = if fold {
        (
            fold_monochrome_post_upscale(upscaled, level_info, output_grayscale_for_monochrome),
            None,
        )
    } else {
        (upscaled, icc_profile)
    };

    let target_format = match return_format {
        Format::WebP => ImageFormat::WebP,
        Format::Png => ImageFormat::Png,
        Format::Jpeg => ImageFormat::Jpeg,
        Format::Original => match image_format {
            ImageFormat::Avif => ImageFormat::WebP,
            other => other,
        },
    };

    encode_image(&final_image, target_format, final_icc, jpeg_quality)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::upscaler::avif::display_p3_icc;
    use image::{ImageDecoder, ImageReader, Rgb, RgbImage, Rgba, RgbaImage};

    fn create_cpu_test_config() -> AppConfig {
        let models_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("realcugan-ncnn-vulkan-rs")
            .join("src")
            .join("realcugan-ncnn-vulkan")
            .join("models");

        AppConfig {
            size_threshold_enabled: false,
            realcugan: crate::config::app_config::RealCuganConfig {
                gpuid: -1,
                tile_size: 400,
                num_threads: 2,
                models_path: models_dir,
                ..crate::config::app_config::RealCuganConfig::default()
            },
            ..AppConfig::default()
        }
    }

    #[tokio::test]
    async fn test_upscale_engine_disabled_skips_gpu_init() {
        let cfg = AppConfig {
            upscale: false,
            realcugan: crate::config::app_config::RealCuganConfig {
                models_path: std::path::PathBuf::from("nonexistent_dir"),
                ..crate::config::app_config::RealCuganConfig::default()
            },
            ..AppConfig::default()
        };
        let engine = UpscaleEngine::init(&cfg)
            .await
            .expect("upscale: false must skip GPU/model initialization cleanly");
        assert!(!engine.is_loaded());
        assert_eq!(engine.scale(), None);
    }

    #[tokio::test]
    async fn test_upscale_engine_rejects_invalid_tta_level() {
        let mut cfg = create_cpu_test_config();
        cfg.realcugan.tta_level = 3;
        let res = UpscaleEngine::init(&cfg).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn test_upscale_engine_cpu_upscale_and_shutdown() {
        let cfg = create_cpu_test_config();
        let engine = UpscaleEngine::init(&cfg)
            .await
            .expect("Failed to initialize UpscaleEngine in CPU mode");
        assert!(engine.is_loaded());
        assert_eq!(engine.scale(), Some(2));

        let test_img = DynamicImage::ImageRgb8(RgbImage::new(16, 16));
        let upscaled = engine
            .upscale(test_img)
            .await
            .expect("CPU upscale should succeed");
        assert_eq!(upscaled.width(), 32);
        assert_eq!(upscaled.height(), 32);

        engine.shutdown();
        assert!(!engine.is_loaded());
    }

    #[test]
    fn test_webp_lossless_roundtrip_with_icc_and_png_fallback() {
        let mut rgb = RgbImage::new(16, 16);
        for (x, y, px) in rgb.enumerate_pixels_mut() {
            *px = Rgb([(x * 15) as u8, (y * 15) as u8, 128]);
        }
        let img = DynamicImage::ImageRgb8(rgb.clone());
        let icc = display_p3_icc();

        let (bytes, fmt) = encode_image(&img, ImageFormat::WebP, Some(icc), 90)
            .expect("lossless WebP encode should succeed");
        assert_eq!(fmt, ImageFormat::WebP);

        let mut decoder = ImageReader::new(Cursor::new(&bytes))
            .with_guessed_format()
            .unwrap()
            .into_decoder()
            .unwrap();
        let extracted_icc = decoder.icc_profile().unwrap();
        assert_eq!(extracted_icc.as_deref(), Some(icc));

        let decoded = DynamicImage::from_decoder(decoder).unwrap().to_rgb8();
        assert_eq!(decoded.as_raw(), rgb.as_raw(), "WebP output must be strictly lossless");

        // Dimensions > 16383 must fall back to PNG cleanly without panicking
        let tall = DynamicImage::ImageLuma8(GrayImage::new(2, WEBP_MAX_DIMENSION + 1));
        let (fallback_bytes, fallback_fmt) =
            encode_image(&tall, ImageFormat::WebP, None, 90).expect("PNG fallback should succeed");
        assert_eq!(fallback_fmt, ImageFormat::Png);
        assert!(fallback_bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
    }

    #[test]
    fn test_fold_monochrome_rgba_composites_white_and_respects_paper_cast_preserved() {
        let mut rgba = RgbaImage::new(4, 4);
        rgba.put_pixel(0, 0, Rgba([0, 0, 0, 0])); // fully transparent -> white (255), NOT black (0)
        rgba.put_pixel(1, 0, Rgba([0, 0, 0, 255])); // opaque black -> 0

        let mono_info = LevelInfo {
            was_leveled: true,
            is_monochrome: true,
            paper_cast_preserved: false,
            classification: PageClassification::Monochrome,
            black_point: 0,
            white_point: 255,
            quality_verdict: None,
        };
        let folded = fold_monochrome_post_upscale(
            DynamicImage::ImageRgba8(rgba.clone()),
            &mono_info,
            true,
        );
        let luma = folded.as_luma8().expect("should fold to Luma8");
        assert_eq!(luma.get_pixel(0, 0)[0], 255);
        assert_eq!(luma.get_pixel(1, 0)[0], 0);

        let preserved_info = LevelInfo {
            paper_cast_preserved: true,
            ..mono_info
        };
        let kept = fold_monochrome_post_upscale(
            DynamicImage::ImageRgba8(rgba),
            &preserved_info,
            true,
        );
        assert!(kept.as_rgba8().is_some(), "paper_cast_preserved must skip L8 fold");
    }
}
