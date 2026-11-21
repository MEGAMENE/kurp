use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct KomgaSeries {
    pub id: String,
    pub metadata: KomgaSeriesMetadata,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct KomgaSeriesMetadata {
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct KomgaBook {
    pub id: String,
    #[serde(rename = "seriesId")]
    pub series_id: String,
    pub metadata: KomgaBookMetadata,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct KomgaBookMetadata {
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KomgaPageInfo {
    pub number: usize,
    #[serde(rename = "fileName", default)]
    pub file_name: String,
    #[serde(rename = "mediaType", default)]
    pub media_type: String,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(rename = "sizeBytes", default)]
    pub size_bytes: Option<u64>,
    #[serde(default)]
    pub size: Option<String>,
}
