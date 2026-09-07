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

#[derive(Serialize, Deserialize, Debug, Copy, Clone, PartialEq)]
pub enum AutoLevelsMode {
    All,
    Manga,
    Off,
}

#[derive(Serialize, Deserialize, Copy, Clone, Debug, PartialEq)]
pub struct AutoLevelsConfig {
    pub enabled: bool,
    pub mode: AutoLevelsMode,
    pub black_clip_percent: f32,
    pub white_clip_percent: f32,
    pub max_black_shift: u8,
    pub min_white_threshold: u8,
    pub correct_paper_cast: bool,
    pub gamma: f32,
    pub output_grayscale_for_monochrome: bool,
}

impl Default for AutoLevelsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: AutoLevelsMode::All,
            black_clip_percent: 0.5,
            white_clip_percent: 0.5,
            max_black_shift: 50,
            min_white_threshold: 200,
            correct_paper_cast: true,
            gamma: 1.0,
            output_grayscale_for_monochrome: true,
        }
    }
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
            .set_default("auto_levels.black_clip_percent", 0.5)?
            .set_default("auto_levels.white_clip_percent", 0.5)?
            .set_default("auto_levels.max_black_shift", 50)?
            .set_default("auto_levels.min_white_threshold", 200)?
            .set_default("auto_levels.correct_paper_cast", true)?
            .set_default("auto_levels.gamma", 1.0)?
            .set_default("auto_levels.output_grayscale_for_monochrome", true)?
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
