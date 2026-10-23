use std::env;
use std::fs;
use std::path::PathBuf;

use config::{Config, ConfigError, Environment, File};
use realcugan_ncnn_vulkan_rs::RealCuganModelType;
use serde_derive::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone, Debug)]
#[allow(unused)]
pub struct AppConfig {
    pub port: u16,
    pub upstream_url: String,
    pub upscale: bool,
    pub return_format: Format,
    pub size_threshold_enabled: bool,
    pub size_threshold: u32,
    pub size_threshold_png: u32,
    pub max_upscale_dimension: u32,
    pub jpeg_quality: u8,
    pub realcugan: RealCuganConfig,
    #[serde(default)]
    pub auto_levels: AutoLevelsConfig,
    pub upscale_tag: Option<String>,
    pub allow_config_updates: bool,
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq)]
pub enum AutoLevelsMode {
    All,
    Manga,
    Color,
    Off,
}

impl Default for AutoLevelsMode {
    fn default() -> Self {
        AutoLevelsMode::All
    }
}

/// Production Hierarchical Configuration for Kurp Auto-Leveling System Overhaul.
#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct AutoLevelsConfig {
    /// Master toggle enabling/disabling the auto-leveling pipeline.
    pub enabled: bool,

    /// Legacy mode selector retained for M1-M2 transitional compilation. Default: All.
    #[serde(default)]
    pub mode: AutoLevelsMode,

    /// Strips redundant 3-channel RGB to 1-channel Luma8 for monochrome manga (20-40% size reduction).
    pub output_grayscale_for_monochrome: bool,

    /// Book-level streaming profiling and caching configuration.
    #[serde(default)]
    pub book_profile: BookProfileConfig,

    /// Defect detection engine thresholds.
    #[serde(default)]
    pub detection: DetectionConfig,

    /// Color and contrast correction parameters and safeguards.
    #[serde(default)]
    pub correction: CorrectionConfig,

    /// Multi-Tier Quality Guard verification thresholds.
    #[serde(default)]
    pub quality_guards: QualityGuardsConfig,
}

impl Default for AutoLevelsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: AutoLevelsMode::All,
            output_grayscale_for_monochrome: true,
            book_profile: BookProfileConfig::default(),
            detection: DetectionConfig::default(),
            correction: CorrectionConfig::default(),
            quality_guards: QualityGuardsConfig::default(),
        }
    }
}

/// Volume profiling, caching, and parallel prefetch configuration.
#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct BookProfileConfig {
    /// Maximum number of book profiles retained in the Moka cache (default: 100 volumes).
    pub cache_size: usize,
    /// Maximum concurrent page fetches and decodes during book profiling (default: 8).
    pub parallel_page_fetches: usize,
    /// Global timeout in seconds for profiling a volume before fallback (default: 45s).
    pub timeout_seconds: u64,
}

impl Default for BookProfileConfig {
    fn default() -> Self {
        Self {
            cache_size: 100,
            parallel_page_fetches: 8,
            timeout_seconds: 45,
        }
    }
}

/// Defect detection engine thresholds.
#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct DetectionConfig {
    /// Minimum modulated defect confidence required to generate a correction plan (default: 0.75).
    pub confidence_threshold: f32,
    /// Cumulative histogram percentile for raw ink black floor candidate (default: 0.1%).
    pub black_clip_percent: f32,
    /// Reverse cumulative percentile from white ceiling for paper white candidate (default: 0.1%).
    pub white_clip_percent: f32,
    /// Minimum peak-to-valley surge ratio required to classify an elevated ink shelf (default: 2.8x).
    pub surge_ratio_threshold: f32,
    /// When true, checks shadow ink chromaticity to preserve intentional sepia art (default: true).
    pub protect_sepia_artwork: bool,
}

impl Default for DetectionConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.75,
            black_clip_percent: 0.1,
            white_clip_percent: 0.1,
            surge_ratio_threshold: 2.8,
            protect_sepia_artwork: true,
        }
    }
}

/// Contrast expansion anchors, black shift clamping, and white balance settings.
#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct CorrectionConfig {
    /// Maximum black point shift allowed, safeguarding against crushing halftones (default: 85).
    pub max_black_shift: u8,
    /// Lower bound on white point calculation, protecting dark scenes from blowout (default: 200).
    pub min_white_threshold: u8,
    /// When true, applies white balance gains to neutralize physical paper yellowing (default: true).
    pub correct_paper_cast: bool,
    /// Midtone gamma power-law exponent; 1.0 = linear (default: 1.0).
    pub gamma: f32,
}

impl Default for CorrectionConfig {
    fn default() -> Self {
        Self {
            max_black_shift: 85,
            min_white_threshold: 200,
            correct_paper_cast: true,
            gamma: 1.0,
        }
    }
}

/// Multi-Tier Quality Guard verification container.
#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct QualityGuardsConfig {
    /// Tier 1: Fast mathematical bounds (<1ms: MSE, PSNR, tail slope erosion).
    #[serde(default)]
    pub tier1: Tier1Config,
    /// Tier 2: Perceptual fidelity (<5ms: SSIM, lightness-neutralized chromaticity drift).
    #[serde(default)]
    pub tier2: Tier2Config,
    /// Tier 3: Structural fidelity (<8ms: strided Sobel NCC, activity-gated block Michelson contrast).
    #[serde(default)]
    pub tier3: Tier3Config,
}

impl Default for QualityGuardsConfig {
    fn default() -> Self {
        Self {
            tier1: Tier1Config::default(),
            tier2: Tier2Config::default(),
            tier3: Tier3Config::default(),
        }
    }
}

/// Tier 1 Quality Guard configuration: Fast sub-millisecond mathematical sanity checks.
#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Tier1Config {
    /// Maximum allowable Mean Squared Error across luminance channel (default: 220.0).
    pub mse_threshold: f32,
    /// Minimum allowable Peak Signal-to-Noise Ratio in decibels (default: 22.0 dB).
    pub psnr_min: f32,
    /// When true, enables histogram tail slope erosion gradient check (default: true).
    pub check_tail_erosion: bool,
}

impl Default for Tier1Config {
    fn default() -> Self {
        Self {
            mse_threshold: 220.0,
            psnr_min: 22.0,
            check_tail_erosion: true,
        }
    }
}

/// Tier 2 Quality Guard configuration: Perceptual fidelity checks.
#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Tier2Config {
    /// Minimum allowable luminance Structural Similarity Index via dssim-core (default: 0.88).
    pub ssim_threshold: f32,
    /// Maximum allowable 95th percentile lightness-neutralized chromaticity drift (default: 2.5).
    pub chroma_delta_p95: f32,
    /// Maximum allowable median lightness-neutralized chromaticity drift (default: 1.5).
    pub chroma_delta_median: f32,
}

impl Default for Tier2Config {
    fn default() -> Self {
        Self {
            ssim_threshold: 0.88,
            chroma_delta_p95: 2.5,
            chroma_delta_median: 1.5,
        }
    }
}

/// Tier 3 Quality Guard configuration: Structural and edge-frequency fidelity checks.
#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
#[serde(default)]
pub struct Tier3Config {
    /// Minimum Normalized Cross-Correlation of Sobel gradient magnitudes (default: 0.90).
    pub gradient_ncc: f32,
    /// Minimum 1st percentile of active block Michelson contrast ratios (default: 0.50).
    pub min_contrast_ratio: f32,
    /// Spatial sampling stride for Sobel gradient NCC computation (default: 4).
    pub stride: usize,
}

impl Default for Tier3Config {
    fn default() -> Self {
        Self {
            gradient_ncc: 0.90,
            min_contrast_ratio: 0.50,
            stride: 4,
        }
    }
}

pub type QualityGuardTier1Config = Tier1Config;
pub type QualityGuardTier2Config = Tier2Config;
pub type QualityGuardTier3Config = Tier3Config;

impl AutoLevelsConfig {
    #[inline] pub fn black_clip_percent(&self) -> f32 { self.detection.black_clip_percent }
    #[inline] pub fn white_clip_percent(&self) -> f32 { self.detection.white_clip_percent }
    #[inline] pub fn max_black_shift(&self) -> u8 { self.correction.max_black_shift }
    #[inline] pub fn min_white_threshold(&self) -> u8 { self.correction.min_white_threshold }
    #[inline] pub fn correct_paper_cast(&self) -> bool { self.correction.correct_paper_cast }
    #[inline] pub fn gamma(&self) -> f32 { self.correction.gamma }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub enum Format {
    Png,
    Jpeg,
    WebP,
    Original,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(remote = "RealCuganModelType")]
enum RealCuganModelTypeDef {
    Nose,
    Pro,
    Se,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct RealCuganConfig {
    pub gpuid: i32,
    pub scale: u32,
    pub noise: i32,

    #[serde(with = "RealCuganModelTypeDef")]
    pub model: RealCuganModelType,
    pub tile_size: u32,
    pub sync_gap: u32,
    pub tta_mode: bool,
    pub num_threads: i32,
    pub models_path: String,
}

impl AppConfig {
    pub fn new() -> Result<Self, ConfigError> {
        let config_dir = AppConfig::get_config_directory()
            .map_err(|e| ConfigError::Message(e.to_string()))?;
        let models_default_dir = config_dir.join("models");

        let mut config = Config::builder();
        if config_dir.join("config.yml").exists() {
            config = config.add_source(File::from(config_dir.join("config.yml")))
        }

        config = config.add_source(Environment::with_prefix("kurp"))
            .set_default("port", 3030)?
            .set_default("upstream_url", "http://localhost:8080")?
            .set_default("upscale", true)?
            .set_default("return_format", "WebP")?
            .set_default("size_threshold_enabled", true)?
            .set_default("size_threshold", 500)?
            .set_default("size_threshold_png", 1000)?
            .set_default("max_upscale_dimension", 4000)?
            .set_default("jpeg_quality", 90)?
            .set_default("realcugan.gpuid", 0)?
            .set_default("realcugan.scale", 2)?
            .set_default("realcugan.noise", -1)?
            .set_default("realcugan.model", "Se")?
            .set_default("realcugan.tile_size", 0)?
            .set_default("realcugan.sync_gap", 3)?
            .set_default("realcugan.tta_mode", false)?
            .set_default("realcugan.num_threads", 2)?
            .set_default("realcugan.models_path", models_default_dir.to_str().unwrap())?
            .set_default("auto_levels.enabled", true)?
            .set_default("auto_levels.mode", "All")?
            .set_default("auto_levels.output_grayscale_for_monochrome", true)?
            .set_default("auto_levels.book_profile.cache_size", 100)?
            .set_default("auto_levels.book_profile.parallel_page_fetches", 8)?
            .set_default("auto_levels.book_profile.timeout_seconds", 45)?
            .set_default("auto_levels.detection.confidence_threshold", 0.75)?
            .set_default("auto_levels.detection.black_clip_percent", 0.1)?
            .set_default("auto_levels.detection.white_clip_percent", 0.1)?
            .set_default("auto_levels.detection.surge_ratio_threshold", 2.8)?
            .set_default("auto_levels.detection.protect_sepia_artwork", true)?
            .set_default("auto_levels.correction.max_black_shift", 85)?
            .set_default("auto_levels.correction.min_white_threshold", 200)?
            .set_default("auto_levels.correction.correct_paper_cast", true)?
            .set_default("auto_levels.correction.gamma", 1.0)?
            .set_default("auto_levels.quality_guards.tier1.mse_threshold", 220.0)?
            .set_default("auto_levels.quality_guards.tier1.psnr_min", 22.0)?
            .set_default("auto_levels.quality_guards.tier1.check_tail_erosion", true)?
            .set_default("auto_levels.quality_guards.tier2.ssim_threshold", 0.88)?
            .set_default("auto_levels.quality_guards.tier2.chroma_delta_p95", 2.5)?
            .set_default("auto_levels.quality_guards.tier2.chroma_delta_median", 1.5)?
            .set_default("auto_levels.quality_guards.tier3.gradient_ncc", 0.90)?
            .set_default("auto_levels.quality_guards.tier3.min_contrast_ratio", 0.50)?
            .set_default("auto_levels.quality_guards.tier3.stride", 4)?
            .set_default("allow_config_updates", false)?;

        config.build()?.try_deserialize()
    }

    pub fn write_config(config: AppConfig) -> std::io::Result<()> {
        let yaml = serde_yaml::to_string(&config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let config_path = AppConfig::get_config_directory()?.join("config.yml");
        fs::write(config_path, yaml)
    }

    fn get_config_directory() -> std::io::Result<PathBuf> {
        let current_dir = env::current_dir()?;
        let dir_env = env::var("KURP_CONF_DIR");
        let config_dir: PathBuf = dir_env.map(|path| PathBuf::from(path))
            .unwrap_or_else(|_| current_dir);

        fs::create_dir_all(&config_dir)?;

        Ok(config_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autolevels_config_default_values() {
        let cfg = AutoLevelsConfig::default();
        assert!(cfg.enabled);
        assert!(cfg.output_grayscale_for_monochrome);
        assert_eq!(cfg.mode, AutoLevelsMode::All);

        // BookProfile defaults
        assert_eq!(cfg.book_profile.cache_size, 100);
        assert_eq!(cfg.book_profile.parallel_page_fetches, 8);
        assert_eq!(cfg.book_profile.timeout_seconds, 45);

        // Detection defaults
        assert!((cfg.detection.confidence_threshold - 0.75).abs() < f32::EPSILON);
        assert!((cfg.detection.black_clip_percent - 0.1).abs() < f32::EPSILON);
        assert!((cfg.detection.white_clip_percent - 0.1).abs() < f32::EPSILON);
        assert!((cfg.detection.surge_ratio_threshold - 2.8).abs() < f32::EPSILON);
        assert!(cfg.detection.protect_sepia_artwork);

        // Correction defaults
        assert_eq!(cfg.correction.max_black_shift, 85);
        assert_eq!(cfg.min_white_threshold(), 200);
        assert!(cfg.correction.correct_paper_cast);
        assert!((cfg.correction.gamma - 1.0).abs() < f32::EPSILON);

        // QualityGuards Tier 1 defaults
        assert!((cfg.quality_guards.tier1.mse_threshold - 220.0).abs() < f32::EPSILON);
        assert!((cfg.quality_guards.tier1.psnr_min - 22.0).abs() < f32::EPSILON);
        assert!(cfg.quality_guards.tier1.check_tail_erosion);

        // QualityGuards Tier 2 defaults
        assert!((cfg.quality_guards.tier2.ssim_threshold - 0.88).abs() < f32::EPSILON);
        assert!((cfg.quality_guards.tier2.chroma_delta_p95 - 2.5).abs() < f32::EPSILON);
        assert!((cfg.quality_guards.tier2.chroma_delta_median - 1.5).abs() < f32::EPSILON);

        // QualityGuards Tier 3 defaults
        assert!((cfg.quality_guards.tier3.gradient_ncc - 0.90).abs() < f32::EPSILON);
        assert!((cfg.quality_guards.tier3.min_contrast_ratio - 0.50).abs() < f32::EPSILON);
        assert_eq!(cfg.quality_guards.tier3.stride, 4);
    }

    #[test]
    fn test_autolevels_config_copy_clone() {
        let cfg = AutoLevelsConfig::default();
        let copy_cfg = cfg; // Tests Copy
        let clone_cfg = copy_cfg.clone(); // Tests Clone
        assert_eq!(cfg, clone_cfg);
    }

    #[test]
    fn test_autolevels_backward_compatibility_accessors() {
        let cfg = AutoLevelsConfig::default();
        assert_eq!(cfg.black_clip_percent(), cfg.detection.black_clip_percent);
        assert_eq!(cfg.white_clip_percent(), cfg.detection.white_clip_percent);
        assert_eq!(cfg.max_black_shift(), cfg.correction.max_black_shift);
        assert_eq!(cfg.min_white_threshold(), cfg.correction.min_white_threshold);
        assert_eq!(cfg.correct_paper_cast(), cfg.correction.correct_paper_cast);
        assert_eq!(cfg.gamma(), cfg.correction.gamma);
    }

    #[test]
    fn test_hierarchical_yaml_deserialization_full() {
        let yaml = r#"
        enabled: true
        output_grayscale_for_monochrome: false
        book_profile:
          cache_size: 50
          parallel_page_fetches: 4
          timeout_seconds: 30
        detection:
          confidence_threshold: 0.80
          black_clip_percent: 0.2
          white_clip_percent: 0.05
          surge_ratio_threshold: 3.2
          protect_sepia_artwork: false
        correction:
          max_black_shift: 25
          min_white_threshold: 210
          correct_paper_cast: false
          gamma: 1.1
        quality_guards:
          tier1:
            mse_threshold: 180.0
            psnr_min: 24.0
            check_tail_erosion: false
          tier2:
            ssim_threshold: 0.91
            chroma_delta_p95: 2.0
            chroma_delta_median: 1.2
          tier3:
            gradient_ncc: 0.93
            min_contrast_ratio: 0.55
            stride: 2
        "#;
        let cfg: AutoLevelsConfig = serde_yaml::from_str(yaml).expect("Failed to deserialize full YAML");
        assert!(!cfg.output_grayscale_for_monochrome);
        assert_eq!(cfg.book_profile.cache_size, 50);
        assert_eq!(cfg.book_profile.parallel_page_fetches, 4);
        assert!((cfg.detection.confidence_threshold - 0.80).abs() < f32::EPSILON);
        assert_eq!(cfg.correction.max_black_shift, 25);
        assert!((cfg.quality_guards.tier2.ssim_threshold - 0.91).abs() < f32::EPSILON);
        assert_eq!(cfg.quality_guards.tier3.stride, 2);

        // Serialize back to YAML and deserialize again
        let serialized = serde_yaml::to_string(&cfg).expect("Serialization failed");
        let roundtripped: AutoLevelsConfig = serde_yaml::from_str(&serialized)
            .expect("Roundtrip deserialization failed");
        assert_eq!(cfg, roundtripped);
    }

    #[test]
    fn test_hierarchical_yaml_deserialization_partial_defaults() {
        let yaml = r#"
        detection:
          surge_ratio_threshold: 3.5
        "#;
        let cfg: AutoLevelsConfig = serde_yaml::from_str(yaml).expect("Failed to deserialize partial YAML");
        assert!(cfg.enabled); // Inherited default
        assert_eq!(cfg.book_profile.cache_size, 100); // Inherited default
        assert!((cfg.detection.surge_ratio_threshold - 3.5).abs() < f32::EPSILON); // Overridden
        assert!((cfg.detection.confidence_threshold - 0.75).abs() < f32::EPSILON); // Inherited default
        assert_eq!(cfg.correction.max_black_shift, 85); // Inherited default
        assert_eq!(cfg.quality_guards.tier3.stride, 4); // Inherited default
    }

    #[test]
    fn test_autolevels_config_empty_yaml_fallback() {
        let yaml = "{}";
        let cfg: AutoLevelsConfig = serde_yaml::from_str(yaml).expect("Failed to deserialize empty YAML");
        assert_eq!(cfg, AutoLevelsConfig::default());
    }

    #[test]
    fn test_legacy_mode_key_tolerance() {
        let yaml = r#"
        enabled: true
        mode: Manga
        "#;
        let cfg: AutoLevelsConfig = serde_yaml::from_str(yaml).expect("Failed to deserialize legacy YAML");
        assert!(cfg.enabled);
        assert_eq!(cfg.mode, AutoLevelsMode::Manga);
        assert_eq!(cfg.correction.max_black_shift, 85); // Defaults still intact
    }

    #[test]
    fn test_app_config_serde_default_when_auto_levels_omitted() {
        let yaml_str = r#"
port: 8080
upstream_url: "http://komga:8080"
upscale: false
return_format: "WebP"
size_threshold_enabled: false
size_threshold: 200
size_threshold_png: 500
max_upscale_dimension: 3000
jpeg_quality: 85
realcugan:
  gpuid: 0
  scale: 2
  noise: 0
  model: "Se"
  tile_size: 0
  sync_gap: 3
  tta_mode: false
  num_threads: 2
  models_path: "./models"
allow_config_updates: false
"#;
        let app_config: AppConfig = serde_yaml::from_str(yaml_str)
            .expect("AppConfig should deserialize with default AutoLevelsConfig when omitted");
        assert_eq!(app_config.auto_levels, AutoLevelsConfig::default());
    }
}
