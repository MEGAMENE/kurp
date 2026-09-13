use std::future::Future;
use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::Response;
use axum_extra::TypedHeader;
use bytes::Bytes;
use headers::{Authorization, Cookie};
use headers::authorization::Basic;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader};
use log::info;
use moka::future::Cache;
use once_cell::sync::Lazy;
use ractor::call;
use regex::Regex;
use unicase::Ascii;

use crate::app_state::AppState;
use crate::http_compression;
use crate::http_compression::{Algorithm, compress};
use crate::models::errors::HttpError;
use crate::upscaler::auto_levels::level_and_verify_with_context;
use crate::upscaler::book_profile::BookProfile;
use crate::upscaler::upscale_actor::UpscaleSupervisorMessage;

pub async fn upscale_komga(
    State(state): State<AppState>,
    authorization: Option<TypedHeader<Authorization<Basic>>>,
    cookie: Option<TypedHeader<Cookie>>,
    req: Request,
) -> Result<Response, StatusCode> {
    let uri = req.uri().clone();
    let tag_checker = state.upscale_tag_checker.clone();
    let cookie_val = cookie.map(|c| c.0);
    let auth_val = authorization.map(|a| a.0);

    let path_segments: Vec<&str> = uri.path().split('/').collect();
    let book_id = path_segments.windows(2)
        .find(|path| path[0] == "books")
        .map(|path| path[1].to_string());
    let page_number = path_segments.windows(2)
        .find(|path| path[0] == "pages")
        .and_then(|path| path[1].parse::<usize>().ok());

    let c_val = cookie_val.clone();
    let a_val = auth_val.clone();
    let bid_check = book_id.clone();
    let upscale_condition = || async move {
        if let Some(ref id) = bid_check {
            tag_checker.komga_contains_upscale_tag(id, c_val, a_val).await
        } else {
            Ok(false)
        }
    };

    upscale_internal(state, req, upscale_condition, book_id, page_number, cookie_val, auth_val).await
}

pub async fn upscale<F, Fut>(
    state: AppState,
    request: Request,
    upscale_condition: F,
) -> Result<Response, StatusCode>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<bool, HttpError>>,
{
    let path_segments: Vec<&str> = request.uri().path().split('/').collect();
    let book_id = path_segments.windows(2)
        .find(|path| path[0] == "books")
        .map(|path| path[1].to_string());
    let page_number = path_segments.windows(2)
        .find(|path| path[0] == "pages")
        .and_then(|path| path[1].parse::<usize>().ok());

    upscale_internal(state, request, upscale_condition, book_id, page_number, None, None).await
}

async fn upscale_internal<F, Fut>(
    state: AppState,
    request: Request,
    upscale_condition: F,
    book_id: Option<String>,
    page_number: Option<usize>,
    cookie: Option<Cookie>,
    auth: Option<Authorization<Basic>>,
) -> Result<Response, StatusCode>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<bool, HttpError>>,
{
    let request = to_proxy_request(state.upscale_call_history_cache.clone(), request).await;
    let request_path = request.uri().path_and_query()
        .map(|path| path.to_string())
        .unwrap_or("/".to_string());

    let uri_str = format!("{} {}", request.method().as_str(), request.uri().path());

    let response = state.proxy_client.proxy_request(request).await
        .map_err(|_| StatusCode::BAD_GATEWAY)?;
    info!("{}: upstream response: {}", uri_str, response.status());

    if response.status() == 304 || !response.status().is_success() {
        return Ok(response);
    }
    let should_upscale = upscale_condition().await.map_err(|_| StatusCode::BAD_GATEWAY)?;
    if !should_upscale {
        return Ok(response);
    }

    // Resolve BookProfile concurrently / ahead of leveling if auto-levels is enabled
    let auto_levels_enabled = state.config.auto_levels.enabled;
    let book_profile = if auto_levels_enabled {
        if let Some(ref bid) = book_id {
            Some(state.book_profile_manager.get_or_build_profile(bid, cookie, auth).await)
        } else {
            None
        }
    } else {
        None
    };

    let upscaled = upscale_response(response, &state, book_id, page_number, book_profile).await;
    info!("{} finished upscaling", uri_str);
    state.upscale_call_history_cache.insert(request_path, ()).await;
    Ok(upscaled)
}

async fn upscale_response(
    response: Response,
    state: &AppState,
    book_id: Option<String>,
    page_number: Option<usize>,
    book_profile: Option<Arc<BookProfile>>,
) -> Response {
    let status = response.status();
    let headers = response.headers().clone();
    let content_type = headers.get("content-type").and_then(|v| v.to_str().ok()).unwrap_or("");
    let image_format = match content_type {
        "image/png" => ImageFormat::Png,
        "image/jpeg" | "image/jpg" => ImageFormat::Jpeg,
        "image/webp" => ImageFormat::WebP,
        "image/gif" => ImageFormat::Gif,
        "image/avif" => ImageFormat::Avif,
        _ => ImageFormat::from_extension(content_type.trim_start_matches("image/"))
            .unwrap_or(ImageFormat::Png),
    };

    let encoding = headers.get("content-encoding");
    let response_bytes = match to_bytes(response.into_body(), 50 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            log::error!("failed to read response body: {}", e);
            return Response::builder().status(status).body(Body::empty()).unwrap();
        }
    };

    let algo = encoding.and_then(parse_encoding_header);

    let to_upscale = if let Some(a) = algo {
        match http_compression::decompress(response_bytes.clone(), a).await {
            Ok(dec) => dec,
            Err(e) => {
                log::error!("Failed to decompress image body: {}. Returning original response.", e);
                return to_response(status, response_bytes, &headers, image_format);
            }
        }
    } else {
        response_bytes.clone()
    };

    let auto_levels_config = state.config.auto_levels;

    // Decoupled Pre-Actor CPU Execution (Milestone 4):
    // Image decoding, histogram generation, color correction, and Quality Guard verification
    // run inside tokio::task::spawn_blocking on the CPU threadpool before dispatching to GPU actor.
    let preprocessed = if auto_levels_config.enabled {
        let to_upscale_cl = to_upscale.clone();
        let bp_cl = book_profile.clone();
        let b_id = book_id.clone();
        let p_num = page_number;
        tokio::task::spawn_blocking(move || {
            let mut reader = match ImageReader::new(Cursor::new(to_upscale_cl.clone())).with_guessed_format() {
                Ok(r) => r,
                Err(e) => {
                    log::info!("Failed to guess format for pre-actor leveling: {}. Falling back.", e);
                    return None;
                }
            };
            if reader.format().is_none() {
                reader.set_format(image_format);
            }
            let detected_format = reader.format().unwrap_or(image_format);
            let is_avif = detected_format == ImageFormat::Avif || image_format == ImageFormat::Avif;
            let avif_color_info = if is_avif {
                crate::upscaler::avif::parse_avif_color_info(&to_upscale_cl)
            } else {
                None
            };

            let mut decoder = match reader.into_decoder() {
                Ok(d) => d,
                Err(_) => {
                    let mut fallback = ImageReader::new(Cursor::new(to_upscale_cl.clone()));
                    fallback.set_format(image_format);
                    match fallback.into_decoder() {
                        Ok(d) => d,
                        Err(e) => {
                            log::info!("Failed to decode image for pre-actor leveling: {}. Falling back.", e);
                            return None;
                        }
                    }
                }
            };

            let orientation = decoder.orientation().ok();
            let icc_profile = decoder.icc_profile().ok().flatten().or_else(|| {
                avif_color_info.as_ref().and_then(|info| info.icc_profile.clone())
            });

            let mut image = match DynamicImage::from_decoder(decoder) {
                Ok(img) => img,
                Err(e) => {
                    log::info!("Failed to decode DynamicImage for pre-actor leveling: {}. Falling back.", e);
                    return None;
                }
            };

            if let Some(orientation) = orientation {
                image.apply_orientation(orientation);
            }

            let (leveled_image, level_info, _verdict) = level_and_verify_with_context(
                image,
                &to_upscale_cl,
                b_id.as_deref(),
                p_num,
                bp_cl.as_deref(),
                &auto_levels_config,
            );

            Some((leveled_image, level_info, icc_profile))
        }).await.unwrap_or(None)
    } else {
        None
    };

    let (upscaled, format) = if let Some((img, level_info, icc)) = preprocessed {
        match call!(state.upscaler, UpscaleSupervisorMessage::UpscalePreprocessed, img, image_format, level_info, icc) {
            Ok(res) => res,
            Err(e) => {
                log::error!("Upscale actor call failed: {}. Returning original response.", e);
                return to_response(status, response_bytes, &headers, image_format);
            }
        }
    } else {
        match call!(state.upscaler, UpscaleSupervisorMessage::Upscale, to_upscale, image_format) {
            Ok(res) => res,
            Err(e) => {
                log::error!("Upscale actor call failed: {}. Returning original response.", e);
                return to_response(status, response_bytes, &headers, image_format);
            }
        }
    };

    let response_body = if let Some(a) = algo {
        match compress(upscaled.clone(), a).await {
            Ok(comp) => comp,
            Err(e) => {
                log::error!("Failed to recompress upscaled image: {}. Returning uncompressed.", e);
                upscaled
            }
        }
    } else {
        upscaled
    };

    to_response(status, response_body, &headers, format)
}

fn to_response(
    status: StatusCode,
    bytes: Bytes,
    headers: &HeaderMap<HeaderValue>,
    format: ImageFormat,
) -> Response {
    let mime_type = match format {
        ImageFormat::Png => { Some(("image/png", "png")) }
        ImageFormat::Jpeg => { Some(("image/jpeg", "jpeg")) }
        ImageFormat::WebP => { Some(("image/webp", "webp")) }
        ImageFormat::Avif => { Some(("image/avif", "avif")) }
        _ => { None }
    };
    let mut builder = Response::builder();
    for (k, v) in headers {
        if Ascii::new("Content-Length") == k {
            builder = builder.header("Content-Length", bytes.len())
        } else if Ascii::new("Content-Type") == k && mime_type.is_some() {
            builder = builder.header("Content-Type", mime_type.unwrap().0)
        } else if Ascii::new("Content-Disposition") == k && mime_type.is_some() {
            if let Ok(val_str) = v.to_str() {
                let new_value: String = val_str.split("; ")
                    .map(|param| if param.starts_with("filename=") || param.starts_with("filename*=") {
                        with_new_file_extension(param, mime_type.unwrap().1)
                    } else {
                        param.to_string()
                    })
                    .collect::<Vec<String>>().join("; ");
                builder = builder.header("Content-Disposition", new_value);
            } else {
                builder = builder.header(k, v);
            }
        } else {
            builder = builder.header(k, v);
        }
    }
    builder
        .status(status)
        .body(Body::from(bytes))
        .unwrap()
}

static FILENAME_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)(filename\*=UTF-8''|filename=)(.+)").unwrap()
});

fn with_new_file_extension(name: &str, extension: &str) -> String {
    if let Some(captures) = FILENAME_REGEX.captures(name) {
        let param_name = captures.get(1).map(|m| m.as_str()).unwrap_or("filename=");
        let raw_filename = captures.get(2).map(|m| m.as_str()).unwrap_or("");
        let trimmed = raw_filename.trim().trim_matches('"');
        let new_filename = Path::new(trimmed)
            .with_extension(extension)
            .to_string_lossy()
            .to_string();
        if raw_filename.starts_with('"') && raw_filename.ends_with('"') {
            format!("{}\"{}\"", param_name, new_filename)
        } else {
            format!("{}{}", param_name, new_filename)
        }
    } else {
        name.to_string()
    }
}

fn parse_encoding_header(encoding: &HeaderValue) -> Option<Algorithm> {
    let encoding = encoding.to_str().ok()?;
    match encoding.trim().to_lowercase().as_str() {
        "gzip" => Some(Algorithm::Gzip),
        "deflate" => Some(Algorithm::Deflate),
        "br" => Some(Algorithm::Brotli),
        _ => None,
    }
}

async fn to_proxy_request(
    call_cache: Arc<Cache<String, ()>>,
    req: Request,
) -> Request {
    let request_path = req.uri().path_and_query()
        .map(|path| path.to_string())
        .unwrap_or("/".to_string());

    let (mut parts, body) = req.into_parts();

    parts.headers = remove_uncached_conditional_headers(
        parts.headers,
        call_cache.clone(),
        request_path,
    ).await;

    Request::from_parts(parts, body)
}

fn is_conditional_header(header_name: &str) -> bool {
    static CONDITIONAL_HEADERS: Lazy<Vec<Ascii<&'static str>>> = Lazy::new(|| {
        vec![Ascii::new("If-Modified-Since"), Ascii::new("If-None-Match")]
    });
    CONDITIONAL_HEADERS.iter().any(|h| h == &header_name)
}

async fn remove_uncached_conditional_headers(
    headers: HeaderMap<HeaderValue>,
    call_cache: Arc<Cache<String, ()>>,
    request_path: String,
) -> HeaderMap<HeaderValue> {
    if call_cache.get(&request_path).await.is_some() {
        return headers;
    }

    headers.iter()
        .filter_map(|(k, v)|
            if is_conditional_header(k.as_str()) {
                None
            } else {
                Some((k.clone(), v.clone()))
            }).collect()
}
