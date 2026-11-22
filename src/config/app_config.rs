use std::collections::HashMap;
use std::env;
use std::fs::{self, File as FsFile};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use axum::http::Uri;
use config::{Config, ConfigError, Environment, File};
use realcugan_ncnn_vulkan_rs::{
    RealCuganModelType, RealCuganNoise, RealCuganScale, RealCuganSyncGap, RealCuganTtaLevel,
};
use serde::{Deserialize, Serialize};

fn default_port() -> u16 {
    3030
}

fn default_upstream_url() -> String {
    "http://localhost:8080".to_string()
}

fn default_true() -> bool {
    true
}

fn default_size_threshold() -> u32 {
    500
}

fn default_size_threshold_png() -> u32 {
    1000
}

fn default_max_upscale_dimension() -> u32 {
    4000
}

fn default_jpeg_quality() -> u8 {
    90
}

fn default_max_concurrent_pipelines() -> usize {
    6
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct AppConfig {
    pub port: u16,
    pub upstream_url: String,
    pub allow_config_updates: bool,
    pub upscale: bool,
    pub upscale_tag: Option<String>,
    pub size_threshold_enabled: bool,
    pub size_threshold: u32,
    pub size_threshold_png: u32,
    pub max_upscale_dimension: u32,
    pub return_format: Format,
    pub jpeg_quality: u8,
    pub max_concurrent_pipelines: usize,
    pub rayon_threads: usize,
    pub realcugan: RealCuganConfig,
    pub auto_levels: AutoLevelsConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            port: default_port(),
            upstream_url: default_upstream_url(),
            allow_config_updates: false,
            upscale: default_true(),
            upscale_tag: None,
            size_threshold_enabled: default_true(),
            size_threshold: default_size_threshold(),
            size_threshold_png: default_size_threshold_png(),
            max_upscale_dimension: default_max_upscale_dimension(),
            return_format: Format::WebP,
            jpeg_quality: default_jpeg_quality(),
            max_concurrent_pipelines: default_max_concurrent_pipelines(),
            rayon_threads: 0,
            realcugan: RealCuganConfig::default(),
            auto_levels: AutoLevelsConfig::default(),
        }
    }
}

/// User-facing configuration for Kurp's D65 CIELAB auto-leveling & page restoration pipeline.
#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct AutoLevelsConfig {
    /// Master toggle enabling/disabling the auto-leveling pipeline.
    pub enabled: bool,

    /// Neutralizes physical paper casts / margin yellowing (`false` preserves original paper tint).
    pub correct_paper_cast: bool,

    /// Strips redundant 3-channel RGB to 1-channel Luma8 for monochrome manga after upscaling.
    pub output_grayscale_for_monochrome: bool,
}

impl Default for AutoLevelsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            correct_paper_cast: true,
            output_grayscale_for_monochrome: true,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq, Eq, Hash, Default)]
pub enum Format {
    Png,
    Jpeg,
    #[default]
    WebP,
    Original,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(remote = "RealCuganModelType")]
enum RealCuganModelTypeDef {
    Nose,
    Pro,
    Se,
}

/// Real-CUGAN upscaler settings aligned 1:1 with `realcugan-ncnn-vulkan-rs` Layer 2 (`2706874`).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct RealCuganConfig {
    /// Device index (`-1` for CPU, `>= 0` for Vulkan GPU; software adapters auto-route to CPU).
    pub gpuid: i32,
    /// Model architecture family (`Nose`, `Pro`, or `Se`).
    #[serde(with = "RealCuganModelTypeDef")]
    pub model: RealCuganModelType,
    /// Upscaling factor (`2`, `3`, or `4`).
    pub scale: u32,
    /// Denoise strength (`-1` conservative, `0` no-denoise, `1..=3` denoise1x..3x).
    pub noise: i32,
    /// Tile size in pixels (`0` = automatic VRAM-aware sizing, or `>= 32`).
    pub tile_size: u32,
    /// Cross-tile synchronization mode (`0`..=`3`).
    pub sync_gap: u32,
    /// Test-Time Augmentation pass count (`1`, `2`, `4`, or `8`).
    pub tta_level: u32,
    /// Enable FP16 shader arithmetic when supported by the Vulkan device.
    pub use_fp16_arithmetic: bool,
    /// Maximum concurrent GPU/CPU Real-CUGAN jobs (scales Layer 1 VRAM tile budgets & semaphore permits).
    pub concurrent_jobs: u32,
    /// CPU thread count (`0` = automatic in Layer 2, or `>= 1`).
    pub num_threads: u32,
    /// Root directory containing `models-se`, `models-pro`, and/or `models-nose`.
    pub models_path: PathBuf,
}

impl Default for RealCuganConfig {
    fn default() -> Self {
        Self {
            gpuid: 0,
            model: RealCuganModelType::Se,
            scale: 2,
            noise: -1,
            tile_size: 0,
            sync_gap: 2,
            tta_level: 1,
            use_fp16_arithmetic: false,
            concurrent_jobs: 1,
            num_threads: 0,
            models_path: PathBuf::from("./models"),
        }
    }
}

impl RealCuganConfig {
    /// Checks whether `(model, scale, noise)` is a valid combination in Layer 2 (`2706874`).
    pub fn is_valid_model_combo(
        model: RealCuganModelType,
        scale: RealCuganScale,
        noise: RealCuganNoise,
    ) -> bool {
        matches!(
            (model, scale, noise),
            (
                RealCuganModelType::Nose,
                RealCuganScale::X2,
                RealCuganNoise::NoDenoise
            ) | (
                RealCuganModelType::Pro,
                RealCuganScale::X2 | RealCuganScale::X3,
                RealCuganNoise::Conservative
                    | RealCuganNoise::NoDenoise
                    | RealCuganNoise::Denoise3x
            ) | (RealCuganModelType::Se, RealCuganScale::X2, _)
                | (
                    RealCuganModelType::Se,
                    RealCuganScale::X3 | RealCuganScale::X4,
                    RealCuganNoise::Conservative
                        | RealCuganNoise::NoDenoise
                        | RealCuganNoise::Denoise3x
                )
        )
    }

    /// Resolves the expected `(param_path, bin_path)` on disk for this configuration.
    pub fn resolve_model_paths(&self) -> Result<(PathBuf, PathBuf), String> {
        let scale = RealCuganScale::try_from(self.scale).map_err(|e| e.to_string())?;
        let noise = RealCuganNoise::try_from(self.noise).map_err(|e| e.to_string())?;
        let subdir = match self.model {
            RealCuganModelType::Nose => "models-nose",
            RealCuganModelType::Pro => "models-pro",
            RealCuganModelType::Se => "models-se",
        };
        let stem = match noise {
            RealCuganNoise::Conservative => format!("up{}x-conservative", scale.as_u32()),
            RealCuganNoise::NoDenoise => format!("up{}x-no-denoise", scale.as_u32()),
            RealCuganNoise::Denoise1x => format!("up{}x-denoise1x", scale.as_u32()),
            RealCuganNoise::Denoise2x => format!("up{}x-denoise2x", scale.as_u32()),
            RealCuganNoise::Denoise3x => format!("up{}x-denoise3x", scale.as_u32()),
        };
        let dir = self.models_path.join(subdir);
        Ok((
            dir.join(format!("{stem}.param")),
            dir.join(format!("{stem}.bin")),
        ))
    }
}

/// Deterministic 64-bit FNV-1a hasher for stable ETag config hashing across restarts.
struct Fnv1a64(u64);

impl Fnv1a64 {
    const OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const PRIME: u64 = 0x100000001b3;

    fn new() -> Self {
        Self(Self::OFFSET_BASIS)
    }

    #[inline]
    fn write_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    #[inline]
    fn write_bool(&mut self, v: bool) {
        self.write_bytes(&[u8::from(v)]);
    }

    #[inline]
    fn write_u8(&mut self, v: u8) {
        self.write_bytes(&[v]);
    }

    #[inline]
    fn write_u32(&mut self, v: u32) {
        self.write_bytes(&v.to_le_bytes());
    }

    #[inline]
    fn write_i32(&mut self, v: i32) {
        self.write_bytes(&v.to_le_bytes());
    }

    #[inline]
    fn finish(self) -> u64 {
        self.0
    }
}

impl AppConfig {
    /// Loads configuration from `<config_dir>/config.yml` (if present) overlaid with POSIX-valid
    /// `KURP_*` environment variables (`__` hierarchy separator, e.g. `KURP_REALCUGAN__GPUID=0`).
    pub fn new() -> Result<Self, ConfigError> {
        let config_dir =
            AppConfig::get_config_directory().map_err(|e| ConfigError::Message(e.to_string()))?;

        Self::from_dir_and_env(&config_dir, env::vars().collect())
    }

    /// Loads configuration from an explicit directory and environment map (race-free for tests
    /// under Rust 2024 without mutating process-wide environment variables).
    pub fn from_dir_and_env(
        config_dir: &Path,
        env_map: HashMap<String, String>,
    ) -> Result<Self, ConfigError> {
        let mut builder = Config::builder();
        let yaml_path = config_dir.join("config.yml");
        if yaml_path.exists() {
            builder = builder.add_source(File::from(yaml_path));
        }

        // Filter out KURP_CONF_DIR so `#[serde(deny_unknown_fields)]` does not reject the
        // directory selector environment variable as an unknown AppConfig field.
        let filtered_env: HashMap<String, String> = env_map
            .into_iter()
            .filter(|(k, _)| !k.eq_ignore_ascii_case("KURP_CONF_DIR"))
            .collect();

        let env_source = Environment::with_prefix("kurp")
            .prefix_separator("_")
            .separator("__")
            .ignore_empty(true)
            .source(Some(filtered_env));

        let built = builder.add_source(env_source).build()?;
        let has_explicit_models_path = built.get_string("realcugan.models_path").is_ok();

        let mut cfg: AppConfig = built.try_deserialize()?;

        // Avoid `.to_str().unwrap()` panics on non-UTF-8 config directories by assigning
        // `PathBuf` directly after deserialization when `realcugan.models_path` was not configured.
        if !has_explicit_models_path {
            cfg.realcugan.models_path = config_dir.join("models");
        }

        if cfg
            .upscale_tag
            .as_deref()
            .is_some_and(|tag| tag.trim().is_empty())
        {
            cfg.upscale_tag = None;
        }

        Ok(cfg)
    }

    /// Validates all runtime invariants before startup or persisting `/kurp/config` updates.
    pub fn validate(&self) -> Result<(), String> {
        if self.port == 0 {
            return Err("port must be greater than 0".to_string());
        }

        let uri = Uri::from_str(self.upstream_url.trim())
            .map_err(|e| format!("invalid upstream_url '{}': {e}", self.upstream_url))?;
        match uri.scheme_str() {
            Some("http" | "https") => {}
            other => {
                return Err(format!(
                    "upstream_url must use http or https scheme (got {other:?})"
                ));
            }
        }
        if uri.authority().is_none() {
            return Err(format!(
                "upstream_url '{}' is missing host authority",
                self.upstream_url
            ));
        }

        if !(1..=100).contains(&self.jpeg_quality) {
            return Err(format!(
                "jpeg_quality must be in 1..=100 (got {})",
                self.jpeg_quality
            ));
        }

        if self.max_concurrent_pipelines == 0 {
            return Err("max_concurrent_pipelines must be >= 1".to_string());
        }

        // Real-CUGAN validation (aligned 1:1 with Layer 2 `2706874`)
        let rc = &self.realcugan;
        if rc.gpuid < -1 {
            return Err(format!(
                "realcugan.gpuid must be >= -1 (got {})",
                rc.gpuid
            ));
        }
        let scale = RealCuganScale::try_from(rc.scale).map_err(|e| e.to_string())?;
        let noise = RealCuganNoise::try_from(rc.noise).map_err(|e| e.to_string())?;
        if !RealCuganConfig::is_valid_model_combo(rc.model, scale, noise) {
            return Err(format!(
                "unsupported realcugan model combination: model={:?}, scale={}, noise={}",
                rc.model, rc.scale, rc.noise
            ));
        }
        RealCuganSyncGap::try_from(rc.sync_gap).map_err(|e| e.to_string())?;
        RealCuganTtaLevel::try_from(rc.tta_level).map_err(|e| e.to_string())?;
        if rc.tile_size != 0 && !(32..=i32::MAX as u32).contains(&rc.tile_size) {
            return Err(format!(
                "realcugan.tile_size must be 0 (auto) or 32..={} (got {})",
                i32::MAX,
                rc.tile_size
            ));
        }
        if rc.concurrent_jobs == 0 {
            return Err("realcugan.concurrent_jobs must be >= 1".to_string());
        }
        if self.upscale && rc.models_path.as_os_str().is_empty() {
            return Err("realcugan.models_path must not be empty when upscale is enabled".to_string());
        }

        Ok(())
    }

    /// Deterministic hash of all image-output-affecting fields (upscaling, encoding, auto-levels)
    /// used to construct config-scoped ETags (`-kurp-<hash:016x>`).
    pub fn config_hash(&self) -> u64 {
        let mut h = Fnv1a64::new();
        h.write_bool(self.upscale);
        h.write_u8(match self.return_format {
            Format::Png => 1,
            Format::Jpeg => 2,
            Format::WebP => 3,
            Format::Original => 4,
        });
        h.write_bool(self.size_threshold_enabled);
        h.write_u32(self.size_threshold);
        h.write_u32(self.size_threshold_png);
        h.write_u32(self.max_upscale_dimension);
        h.write_u8(self.jpeg_quality);

        // Real-CUGAN output-affecting fields
        h.write_u32(self.realcugan.scale);
        h.write_i32(self.realcugan.noise);
        h.write_u8(match self.realcugan.model {
            RealCuganModelType::Nose => 1,
            RealCuganModelType::Pro => 2,
            RealCuganModelType::Se => 3,
        });
        h.write_u32(self.realcugan.tile_size);
        h.write_u32(self.realcugan.sync_gap);
        h.write_u32(self.realcugan.tta_level);
        h.write_bool(self.realcugan.use_fp16_arithmetic);

        // Auto-levels output-affecting fields
        let al = &self.auto_levels;
        h.write_bool(al.enabled);
        h.write_bool(al.correct_paper_cast);
        h.write_bool(al.output_grayscale_for_monochrome);

        h.finish()
    }

    /// Atomically persists configuration to `<config_dir>/config.yml` using `serde-saphyr`
    /// via a synced temporary file + atomic rename.
    pub fn update_config(config: &AppConfig) -> std::io::Result<()> {
        let config_dir = AppConfig::get_config_directory()?;
        Self::write_config_to_dir(config, &config_dir)
    }

    /// Atomically persists configuration into an explicit directory (testable without mutating
    /// `KURP_CONF_DIR`).
    pub fn write_config_to_dir(config: &AppConfig, config_dir: &Path) -> std::io::Result<()> {
        fs::create_dir_all(config_dir)?;
        let yaml = serde_saphyr::to_string(config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;

        let target_path = config_dir.join("config.yml");
        let tmp_name = format!(
            ".config.yml.tmp.{}.{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        );
        let tmp_path = config_dir.join(tmp_name);

        let write_result = (|| -> std::io::Result<()> {
            let mut file = FsFile::create(&tmp_path)?;
            file.write_all(yaml.as_bytes())?;
            file.sync_all()?;
            Ok(())
        })();

        if let Err(e) = write_result {
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }

        if let Err(e) = fs::rename(&tmp_path, &target_path) {
            let _ = fs::remove_file(&tmp_path);
            return Err(e);
        }

        Ok(())
    }

    pub fn get_config_directory() -> std::io::Result<PathBuf> {
        let current_dir = env::current_dir()?;
        let config_dir: PathBuf = env::var_os("KURP_CONF_DIR")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or(current_dir);

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
        assert!(cfg.correct_paper_cast);
        assert!(cfg.output_grayscale_for_monochrome);
    }

    #[test]
    fn test_checked_in_config_yml_matches_default() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let loaded = AppConfig::from_dir_and_env(&manifest_dir, HashMap::new())
            .expect("checked-in config.yml must deserialize cleanly under deny_unknown_fields");
        assert_eq!(loaded, AppConfig::default());
        assert!(loaded.validate().is_ok());
    }

    #[test]
    fn test_saphyr_yaml_roundtrip_and_deny_unknown_fields() {
        let cfg = AppConfig::default();
        assert!(cfg.validate().is_ok());

        let yaml = serde_saphyr::to_string(&cfg).expect("serde-saphyr serialize failed");
        let parsed: AppConfig =
            serde_saphyr::from_str(&yaml).expect("serde-saphyr deserialize failed");
        assert_eq!(cfg, parsed);

        let bad_yaml = "port: 3030\nlegacy_unknown_field: true\n";
        let err = serde_saphyr::from_str::<AppConfig>(bad_yaml);
        assert!(err.is_err(), "unknown fields must be rejected");
    }

    #[test]
    fn test_posix_env_overrides_and_kurp_conf_dir_compatibility() {
        let tmp_dir = std::env::temp_dir().join(format!(
            "kurp_cfg_test_{}",
            std::process::id()
        ));
        let _ = fs::create_dir_all(&tmp_dir);

        let mut env_map = HashMap::new();
        env_map.insert("KURP_CONF_DIR".to_string(), "/config".to_string());
        env_map.insert("KURP_PORT".to_string(), "4040".to_string());
        env_map.insert("KURP_RAYON_THREADS".to_string(), "4".to_string());
        env_map.insert("KURP_REALCUGAN__GPUID".to_string(), "-1".to_string());
        env_map.insert("KURP_REALCUGAN__CONCURRENT_JOBS".to_string(), "3".to_string());
        env_map.insert("KURP_REALCUGAN__NUM_THREADS".to_string(), "2".to_string());
        env_map.insert("KURP_AUTO_LEVELS__CORRECT_PAPER_CAST".to_string(), "false".to_string());

        let cfg = AppConfig::from_dir_and_env(&tmp_dir, env_map)
            .expect("POSIX __ env overrides should deserialize cleanly");
        assert_eq!(cfg.port, 4040);
        assert_eq!(cfg.rayon_threads, 4);
        assert_eq!(cfg.realcugan.gpuid, -1);
        assert_eq!(cfg.realcugan.concurrent_jobs, 3);
        assert_eq!(cfg.realcugan.num_threads, 2);
        assert!(!cfg.auto_levels.correct_paper_cast);
        assert_eq!(cfg.realcugan.models_path, tmp_dir.join("models"));
        assert!(cfg.validate().is_ok());

        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_validate_catches_invalid_configs() {
        assert!(
            AppConfig {
                port: 0,
                ..AppConfig::default()
            }
            .validate()
            .is_err()
        );

        assert!(
            AppConfig {
                upstream_url: "ftp://komga:8080".to_string(),
                ..AppConfig::default()
            }
            .validate()
            .is_err()
        );

        assert!(
            AppConfig {
                jpeg_quality: 0,
                ..AppConfig::default()
            }
            .validate()
            .is_err()
        );

        assert!(
            AppConfig {
                max_concurrent_pipelines: 0,
                ..AppConfig::default()
            }
            .validate()
            .is_err()
        );

        assert!(
            AppConfig {
                realcugan: RealCuganConfig {
                    model: RealCuganModelType::Pro,
                    scale: 4,
                    ..RealCuganConfig::default()
                },
                ..AppConfig::default()
            }
            .validate()
            .is_err()
        );

        assert!(
            AppConfig {
                realcugan: RealCuganConfig {
                    tta_level: 3,
                    ..RealCuganConfig::default()
                },
                ..AppConfig::default()
            }
            .validate()
            .is_err()
        );

        assert!(
            AppConfig {
                realcugan: RealCuganConfig {
                    tile_size: 16,
                    ..RealCuganConfig::default()
                },
                ..AppConfig::default()
            }
            .validate()
            .is_err()
        );

        assert!(
            AppConfig {
                realcugan: RealCuganConfig {
                    concurrent_jobs: 0,
                    ..RealCuganConfig::default()
                },
                ..AppConfig::default()
            }
            .validate()
            .is_err()
        );
    }

    #[test]
    fn test_config_hash_deterministic_and_sensitive() {
        let cfg1 = AppConfig::default();
        let cfg2 = AppConfig::default();
        assert_eq!(cfg1.config_hash(), cfg2.config_hash());

        let cfg3 = AppConfig {
            realcugan: RealCuganConfig {
                noise: 3,
                ..RealCuganConfig::default()
            },
            ..AppConfig::default()
        };
        assert_ne!(cfg1.config_hash(), cfg3.config_hash());

        // Non-image-affecting fields (e.g. port) should not alter image ETag hash
        let cfg4 = AppConfig {
            port: 9999,
            ..AppConfig::default()
        };
        assert_eq!(cfg1.config_hash(), cfg4.config_hash());
    }
}
