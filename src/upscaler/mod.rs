pub mod auto_levels;
pub mod avif;
pub mod book_profile;
#[allow(clippy::module_inception)]
pub mod upscaler;

pub use book_profile::{BookProfile, BookProfileManager, ClassificationGroupStats, PageAnalysisSummary};
pub use upscaler::UpscaleEngine;
