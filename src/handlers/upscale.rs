use axum::body::{Body, to_bytes};
use axum::extract::{Path, Request, State};
use axum::http::header::{
    ACCEPT, ACCEPT_ENCODING, ACCEPT_RANGES, CONTENT_DISPOSITION, CONTENT_ENCODING,
    CONTENT_LENGTH, CONTENT_TYPE, ETAG, IF_MODIFIED_SINCE, IF_NONE_MATCH, VARY,
};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use image::ImageFormat;
use log::{error, info, warn};

use crate::app_state::AppState;
use crate::clients::komga_client::ForwardAuthHeaders;
use crate::config::app_config::Format;
use crate::http_compression::{self, Algorithm};
use crate::upscaler::auto_levels::{
    LevelInfo, PageClassification, is_image_grayscale, level_and_verify_with_context,
};
use crate::upscaler::avif::{decode_image_with_metadata, probe_image_dimensions};
use crate::upscaler::upscaler::encode_upscaled_output;

const MAX_PAGE_BODY_BYTES: usize = 50 * 1024 * 1024;
const CONTENT_MD5: HeaderName = HeaderName::from_static("content-md5");

#[derive(Debug, Default, Copy, Clone)]
struct PageQueryParams {
    zero_based: bool,
    convert: Option<ImageFormat>,
}

pub async fn upscale_komga(
    State(state): State<AppState>,
    Path((book_id, page_number)): Path<(String, usize)>,
    mut req: Request,
) -> Result<Response, StatusCode> {
    let config = state.config.load_full();
    if !config.upscale {
        return proxy_raw(&state, req).await;
    }

    let auth = ForwardAuthHeaders::from_headers(req.headers());
    let query_params = parse_page_query(req.uri().query());
    let effective_page_number = if query_params.zero_based {
        page_number.saturating_add(1)
    } else {
        page_number
    };
    let accept_parts: Vec<&str> = req
        .headers()
        .get_all(ACCEPT)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .collect();
    let accept_header = (!accept_parts.is_empty()).then(|| accept_parts.join(", "));

    let uri_str = format!("{} {}", req.method().as_str(), req.uri().path());

    // Check upscale tag BEFORE touching conditional headers so untagged books retain 100% 304 caching.
    let should_upscale = state
        .upscale_tag_checker
        .komga_contains_upscale_tag(&book_id, &auth)
        .await
        .map_err(|err| {
            warn!("Upscale tag lookup failed for book {}: {}", book_id, err);
            err.response_status()
        })?;

    if !should_upscale {
        return proxy_raw(&state, req).await;
    }

    let cfg_hash = config.config_hash();
    let etag_suffix = build_etag_suffix(
        cfg_hash,
        config.return_format,
        query_params.convert,
        accept_header.as_deref(),
    );

    let matched_current_config = rewrite_conditional_headers(req.headers_mut(), &etag_suffix);
    req.headers_mut()
        .insert(ACCEPT_ENCODING, HeaderValue::from_static("identity"));

    let response = state.proxy_client.proxy_request(req).await.map_err(|err| {
        error!("Page upstream proxy failed for {}: {}", uri_str, err);
        err.status_code()
    })?;
    info!("{}: upstream response: {}", uri_str, response.status());

    if response.status() == StatusCode::NOT_MODIFIED {
        return Ok(rewrite_not_modified_response(
            response,
            matched_current_config.then_some(etag_suffix.as_str()),
        ));
    }

    // Only upscale full 200 OK page payloads (passthrough 206 Partial Content, 4xx, 5xx untouched).
    if response.status() != StatusCode::OK {
        return Ok(response);
    }

    // Acquire backpressure permit BEFORE reading/buffering up to 50 MiB body.
    let _pipeline_permit = state.pipeline_semaphore.acquire().await.map_err(|_| {
        error!("Pipeline semaphore closed");
        StatusCode::SERVICE_UNAVAILABLE
    })?;

    let status = response.status();
    let headers = response.headers().clone();
    let body = response.into_body();
    let auto_levels_enabled = config.auto_levels.enabled;

    // Drain Page 1 TCP stream concurrently while BookProfileManager samples 9-15 pages.
    let read_body_fut = to_bytes(body, MAX_PAGE_BODY_BYTES);
    let profile_fut = async {
        if auto_levels_enabled {
            Some(
                state
                    .book_profile_manager
                    .get_or_build_profile(&book_id, &auth, &config.auto_levels)
                    .await,
            )
        } else {
            None
        }
    };

    let (body_res, book_profile) = tokio::join!(read_body_fut, profile_fut);
    let response_bytes = match body_res {
        Ok(b) => b,
        Err(err) => {
            error!("Failed to read upstream page body for {}: {}", uri_str, err);
            return Err(if err.to_string().contains("length limit") {
                StatusCode::PAYLOAD_TOO_LARGE
            } else {
                StatusCode::BAD_GATEWAY
            });
        }
    };

    let mime_format = headers
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .and_then(parse_content_type_format);

    let to_upscale = match parse_encoding_header(headers.get(CONTENT_ENCODING)) {
        ContentEncoding::Identity => response_bytes.clone(),
        ContentEncoding::Supported(algo) => {
            match http_compression::decompress(response_bytes.clone(), algo).await {
                Ok(dec) => dec,
                Err(e) => {
                    error!(
                        "Failed to decompress upstream image body ({:?}): {}. Passing through original.",
                        algo, e
                    );
                    return Ok(passthrough_response(status, response_bytes, &headers));
                }
            }
        }
        ContentEncoding::Unsupported(raw) => {
            warn!(
                "Unsupported Content-Encoding '{}' on upstream page; passing through original.",
                raw
            );
            return Ok(passthrough_response(status, response_bytes, &headers));
        }
    };

    if config.size_threshold_enabled {
        let input_kb = (to_upscale.len() / 1024) as u32;
        let threshold = if mime_format == Some(ImageFormat::Png) {
            config.size_threshold_png
        } else {
            config.size_threshold
        };
        if input_kb > threshold {
            info!(
                "image size {} KiB is bigger than threshold {} KiB; skipping upscale",
                input_kb, threshold
            );
            return Ok(passthrough_response(status, response_bytes, &headers));
        }
    }

    let auto_levels_config = config.auto_levels;
    let max_upscale_dimension = config.max_upscale_dimension;
    let b_id = book_id.clone();

    // Stage 1 (CPU spawn_blocking): Probe dimensions, decode (with AVIF/EXIF orientation & ICC),
    // and execute Auto-Levels (or fast grayscale check if auto_levels is disabled).
    let decoded_stage = tokio::task::spawn_blocking(move || {
        if max_upscale_dimension > 0
            && let Ok((pw, ph, _)) = probe_image_dimensions(&to_upscale, mime_format)
            && (pw > max_upscale_dimension || ph > max_upscale_dimension)
        {
            info!(
                "image dimensions ({}x{}) exceed max_upscale_dimension ({}); returning original",
                pw, ph, max_upscale_dimension
            );
            return None;
        }

        let decoded = match decode_image_with_metadata(&to_upscale, mime_format) {
            Ok(d) => d,
            Err(e) => {
                info!("Failed to decode image: {}. Returning original.", e);
                return None;
            }
        };
        drop(to_upscale);

        let (dim_w, dim_h) = (decoded.image.width(), decoded.image.height());
        if max_upscale_dimension > 0
            && (dim_w > max_upscale_dimension || dim_h > max_upscale_dimension)
        {
            info!(
                "image dimensions ({}x{}) exceed max_upscale_dimension ({}); returning original",
                dim_w, dim_h, max_upscale_dimension
            );
            return None;
        }

        let detected_format = decoded.detected_format;
        let icc_profile = decoded.icc_profile;

        let (processed_image, level_info) = if auto_levels_config.enabled {
            let (leveled_image, info, _verdict) = level_and_verify_with_context(
                decoded.image,
                Some(&b_id),
                Some(effective_page_number),
                book_profile.as_deref(),
                &auto_levels_config,
            );
            (leveled_image, info)
        } else {
            let is_gray = is_image_grayscale(&decoded.image);
            let info = LevelInfo {
                is_monochrome: is_gray,
                classification: if is_gray {
                    PageClassification::Monochrome
                } else {
                    PageClassification::Color
                },
                ..LevelInfo::default()
            };
            (decoded.image, info)
        };

        Some((processed_image, level_info, detected_format, icc_profile))
    })
    .await
    .unwrap_or(None);

    let (img, level_info, detected_format, icc) = match decoded_stage {
        Some(res) => res,
        None => return Ok(passthrough_response(status, response_bytes, &headers)),
    };

    // Stage 2 (GPU / CPU Real-CUGAN inference)
    let upscaled_image = match state.upscaler.upscale(img).await {
        Ok(res) => res,
        Err(e) => {
            error!(
                "Real-CUGAN upscale failed for {}: {}. Returning original response.",
                uri_str, e
            );
            return Ok(passthrough_response(status, response_bytes, &headers));
        }
    };

    // Stage 3 (CPU spawn_blocking): Resolve output format (?convert / Accept / return_format) and encode.
    let target_return_format = resolve_output_format(
        config.return_format,
        query_params.convert,
        accept_header.as_deref(),
        detected_format,
    );
    let jpeg_quality = config.jpeg_quality;
    let output_grayscale_for_monochrome = auto_levels_config.output_grayscale_for_monochrome;

    let encoded_res = tokio::task::spawn_blocking(move || {
        encode_upscaled_output(
            upscaled_image,
            detected_format,
            &level_info,
            icc.as_deref(),
            target_return_format,
            jpeg_quality,
            output_grayscale_for_monochrome,
        )
    })
    .await;

    let (upscaled_bytes, out_format) = match encoded_res {
        Ok(Ok(res)) => res,
        Ok(Err(e)) => {
            error!("Image encoding failed for {}: {}. Returning original.", uri_str, e);
            return Ok(passthrough_response(status, response_bytes, &headers));
        }
        Err(e) => {
            error!(
                "Image encoding task panicked for {}: {}. Returning original.",
                uri_str, e
            );
            return Ok(passthrough_response(status, response_bytes, &headers));
        }
    };

    info!("{} finished upscaling ({:?}, {} bytes)", uri_str, out_format, upscaled_bytes.len());
    Ok(modified_response(
        status,
        upscaled_bytes,
        &headers,
        out_format,
        &etag_suffix,
    ))
}

async fn proxy_raw(state: &AppState, req: Request) -> Result<Response, StatusCode> {
    state.proxy_client.proxy_request(req).await.map_err(|err| {
        error!("Proxy error ({}): {}", err.status_code(), err.message);
        err.status_code()
    })
}

fn passthrough_response(
    status: StatusCode,
    bytes: Bytes,
    original_headers: &HeaderMap<HeaderValue>,
) -> Response {
    let mut builder = Response::builder().status(status);
    for (k, v) in original_headers.iter() {
        builder = builder.header(k, v);
    }
    builder.body(Body::from(bytes)).unwrap_or_else(|err| {
        error!("Failed to build passthrough response: {}", err);
        StatusCode::INTERNAL_SERVER_ERROR.into_response()
    })
}

fn modified_response(
    status: StatusCode,
    bytes: Bytes,
    original_headers: &HeaderMap<HeaderValue>,
    format: ImageFormat,
    etag_suffix: &str,
) -> Response {
    let mime_and_ext = format_mime_and_ext(format);
    let mut builder = Response::builder().status(status);
    let mut saw_content_length = false;
    let mut saw_content_type = false;
    let mut vary_values: Vec<String> = Vec::new();

    for (k, v) in original_headers.iter() {
        if k == CONTENT_ENCODING || k == CONTENT_MD5 || k == ACCEPT_RANGES {
            continue;
        }
        if k == CONTENT_LENGTH {
            saw_content_length = true;
            builder = builder.header(CONTENT_LENGTH, bytes.len());
            continue;
        }
        if k == CONTENT_TYPE {
            saw_content_type = true;
            if let Some((mime, _)) = mime_and_ext {
                builder = builder.header(CONTENT_TYPE, mime);
            } else {
                builder = builder.header(k, v);
            }
            continue;
        }
        if k == CONTENT_DISPOSITION
            && let Some((_, ext)) = mime_and_ext
            && let Ok(val_str) = v.to_str()
        {
            let rewritten = rewrite_content_disposition(val_str, ext);
            builder = builder.header(CONTENT_DISPOSITION, rewritten);
            continue;
        }
        if k == ETAG
            && let Ok(etag_str) = v.to_str()
        {
            let suffixed = suffix_etag(etag_str, etag_suffix);
            if let Ok(hv) = HeaderValue::from_str(&suffixed) {
                builder = builder.header(ETAG, hv);
                continue;
            }
        }
        if k == VARY
            && let Ok(vary_str) = v.to_str()
        {
            vary_values.push(vary_str.to_string());
            continue;
        }

        builder = builder.header(k, v);
    }

    if !saw_content_length {
        builder = builder.header(CONTENT_LENGTH, bytes.len());
    }
    if !saw_content_type && let Some((mime, _)) = mime_and_ext {
        builder = builder.header(CONTENT_TYPE, mime);
    }

    if let Some(merged_vary) = merge_vary_accept(&vary_values) {
        builder = builder.header(VARY, merged_vary);
    }

    builder.body(Body::from(bytes)).unwrap_or_else(|err| {
        error!("Failed to build upscaled response: {}", err);
        StatusCode::INTERNAL_SERVER_ERROR.into_response()
    })
}

fn rewrite_not_modified_response(mut response: Response, etag_suffix: Option<&str>) -> Response {
    let headers = response.headers_mut();
    if let Some(suffix) = etag_suffix
        && let Some(raw_etag) = headers.get(ETAG).and_then(|v| v.to_str().ok()).map(str::to_string)
    {
        let suffixed = suffix_etag(&raw_etag, suffix);
        if let Ok(hv) = HeaderValue::from_str(&suffixed) {
            headers.insert(ETAG, hv);
        }
    }

    let existing_vary: Vec<String> = headers
        .get_all(VARY)
        .iter()
        .filter_map(|v| v.to_str().ok().map(str::to_string))
        .collect();
    if let Some(merged) = merge_vary_accept(&existing_vary)
        && let Ok(hv) = HeaderValue::from_str(&merged)
    {
        headers.insert(VARY, hv);
    }

    response
}

fn format_mime_and_ext(format: ImageFormat) -> Option<(&'static str, &'static str)> {
    match format {
        ImageFormat::Png => Some(("image/png", "png")),
        ImageFormat::Jpeg => Some(("image/jpeg", "jpeg")),
        ImageFormat::WebP => Some(("image/webp", "webp")),
        ImageFormat::Gif => Some(("image/gif", "gif")),
        ImageFormat::Avif => Some(("image/avif", "avif")),
        _ => None,
    }
}

fn format_ext_tag(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::WebP => "webp",
        ImageFormat::Gif => "gif",
        ImageFormat::Avif => "avif",
        _ => "other",
    }
}

fn format_to_image_format(format: Format) -> Option<ImageFormat> {
    match format {
        Format::Png => Some(ImageFormat::Png),
        Format::Jpeg => Some(ImageFormat::Jpeg),
        Format::WebP => Some(ImageFormat::WebP),
        Format::Original => None,
    }
}

fn image_format_to_format(format: ImageFormat) -> Format {
    match format {
        ImageFormat::Png => Format::Png,
        ImageFormat::Jpeg => Format::Jpeg,
        ImageFormat::WebP => Format::WebP,
        _ => Format::Original,
    }
}

fn parse_page_query(query: Option<&str>) -> PageQueryParams {
    let mut params = PageQueryParams::default();
    let Some(q) = query else {
        return params;
    };

    for pair in q.split('&') {
        let (key, val) = pair.split_once('=').unwrap_or((pair, ""));
        if key.eq_ignore_ascii_case("zero_based") {
            params.zero_based = val.eq_ignore_ascii_case("true") || val == "1";
        } else if key.eq_ignore_ascii_case("convert") {
            if val.eq_ignore_ascii_case("png") {
                params.convert = Some(ImageFormat::Png);
            } else if val.eq_ignore_ascii_case("jpeg") || val.eq_ignore_ascii_case("jpg") {
                params.convert = Some(ImageFormat::Jpeg);
            } else if val.eq_ignore_ascii_case("webp") {
                params.convert = Some(ImageFormat::WebP);
            }
        }
    }
    params
}

fn parse_content_type_format(content_type: &str) -> Option<ImageFormat> {
    let mime = content_type.split(';').next()?.trim();
    if mime.eq_ignore_ascii_case("image/png") {
        Some(ImageFormat::Png)
    } else if mime.eq_ignore_ascii_case("image/jpeg") || mime.eq_ignore_ascii_case("image/jpg") {
        Some(ImageFormat::Jpeg)
    } else if mime.eq_ignore_ascii_case("image/webp") {
        Some(ImageFormat::WebP)
    } else if mime.eq_ignore_ascii_case("image/gif") {
        Some(ImageFormat::Gif)
    } else if mime.eq_ignore_ascii_case("image/avif") {
        Some(ImageFormat::Avif)
    } else {
        None
    }
}

enum ContentEncoding {
    Identity,
    Supported(Algorithm),
    Unsupported(String),
}

fn parse_encoding_header(encoding: Option<&HeaderValue>) -> ContentEncoding {
    let Some(val) = encoding else {
        return ContentEncoding::Identity;
    };
    let Ok(raw) = val.to_str() else {
        return ContentEncoding::Unsupported("<non-utf8>".to_string());
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("identity") {
        ContentEncoding::Identity
    } else if trimmed.eq_ignore_ascii_case("gzip") || trimmed.eq_ignore_ascii_case("x-gzip") {
        ContentEncoding::Supported(Algorithm::Gzip)
    } else if trimmed.eq_ignore_ascii_case("deflate") {
        ContentEncoding::Supported(Algorithm::Deflate)
    } else if trimmed.eq_ignore_ascii_case("br") {
        ContentEncoding::Supported(Algorithm::Brotli)
    } else {
        ContentEncoding::Unsupported(trimmed.to_string())
    }
}

/// Evaluates client `Accept` header per RFC 9110 §12.5.1 media-range specificity.
/// Returns `q` quality in `[0.0, 1.0]` for the requested `(target_type, target_subtype)`.
fn accept_quality_for(accept: Option<&str>, target_type: &str, target_subtype: &str) -> f32 {
    let Some(raw) = accept.map(str::trim).filter(|s| !s.is_empty()) else {
        return 1.0;
    };

    let mut best_spec = -1i8;
    let mut best_q = 0.0f32;
    let mut saw_any_range = false;

    for item in raw.split(',') {
        let mut parts = item.split(';');
        let Some(media_range) = parts.next().map(str::trim).filter(|s| !s.is_empty()) else {
            continue;
        };
        let Some((m_type, m_sub)) = media_range.split_once('/') else {
            continue;
        };
        let m_type = m_type.trim();
        let m_sub = m_sub.trim();
        saw_any_range = true;

        let spec = if m_type.eq_ignore_ascii_case(target_type)
            && m_sub.eq_ignore_ascii_case(target_subtype)
        {
            2
        } else if m_type.eq_ignore_ascii_case(target_type) && m_sub == "*" {
            1
        } else if m_type == "*" && m_sub == "*" {
            0
        } else {
            continue;
        };

        let mut q = 1.0f32;
        for param in parts {
            if let Some((pk, pv)) = param.split_once('=')
                && pk.trim().eq_ignore_ascii_case("q")
                && let Ok(parsed) = pv.trim().parse::<f32>()
                && parsed.is_finite()
            {
                q = parsed.clamp(0.0, 1.0);
                break;
            }
        }

        if spec > best_spec || (spec == best_spec && q > best_q) {
            best_spec = spec;
            best_q = q;
        }
    }

    if !saw_any_range {
        1.0
    } else if best_spec < 0 {
        0.0
    } else {
        best_q
    }
}

fn negotiate_accepted_format(preferred: ImageFormat, accept: Option<&str>) -> ImageFormat {
    let preferred_sub = match preferred {
        ImageFormat::WebP => "webp",
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::Gif => "gif",
        _ => return preferred,
    };

    if accept_quality_for(accept, "image", preferred_sub) > 0.0 {
        return preferred;
    }

    let candidates = [
        (ImageFormat::Png, "png"),
        (ImageFormat::Jpeg, "jpeg"),
        (ImageFormat::WebP, "webp"),
    ];
    let mut best_fmt = ImageFormat::Png;
    let mut best_q = 0.0f32;
    for (fmt, sub) in candidates {
        let q = accept_quality_for(accept, "image", sub);
        if q > best_q {
            best_q = q;
            best_fmt = fmt;
        }
    }

    if best_q > 0.0 {
        best_fmt
    } else {
        preferred
    }
}

fn build_etag_suffix(
    cfg_hash: u64,
    configured: Format,
    query_convert: Option<ImageFormat>,
    accept: Option<&str>,
) -> String {
    if let Some(conv) = query_convert {
        return format!("-kurp-{:016x}-{}", cfg_hash, format_ext_tag(conv));
    }

    if let Some(pref) = format_to_image_format(configured) {
        let negotiated = negotiate_accepted_format(pref, accept);
        if negotiated != pref {
            return format!("-kurp-{:016x}-{}", cfg_hash, format_ext_tag(negotiated));
        }
    }

    format!("-kurp-{:016x}", cfg_hash)
}

fn resolve_output_format(
    configured: Format,
    query_convert: Option<ImageFormat>,
    accept: Option<&str>,
    detected_format: ImageFormat,
) -> Format {
    if let Some(conv) = query_convert {
        return image_format_to_format(conv);
    }

    match configured {
        Format::Original => {
            let candidate = if detected_format == ImageFormat::Avif {
                ImageFormat::WebP
            } else {
                detected_format
            };
            let negotiated = negotiate_accepted_format(candidate, accept);
            if negotiated == candidate {
                Format::Original
            } else {
                image_format_to_format(negotiated)
            }
        }
        other => {
            let pref = format_to_image_format(other).unwrap_or(ImageFormat::WebP);
            let negotiated = negotiate_accepted_format(pref, accept);
            image_format_to_format(negotiated)
        }
    }
}

fn suffix_etag(raw_etag: &str, suffix: &str) -> String {
    let trimmed = raw_etag.trim();
    let (weak_prefix, rest) = if let Some(after) = trimmed.strip_prefix("W/") {
        ("W/", after.trim())
    } else {
        ("", trimmed)
    };

    if let Some(inner) = rest.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        if inner.ends_with(suffix) {
            format!("{weak_prefix}\"{inner}\"")
        } else {
            format!("{weak_prefix}\"{inner}{suffix}\"")
        }
    } else if rest.ends_with(suffix) {
        format!("{weak_prefix}\"{rest}\"")
    } else {
        format!("{weak_prefix}\"{rest}{suffix}\"")
    }
}

/// Inspects `If-None-Match` and translates ETags ending with `expected_suffix` back to
/// Komga's raw ETag. Strips non-matching ETags (and `If-Modified-Since` when unmatched)
/// so clients receive fresh upscaled responses after config changes or initial enablement.
fn rewrite_conditional_headers(headers: &mut HeaderMap<HeaderValue>, expected_suffix: &str) -> bool {
    let mut translated_tags = Vec::new();
    let mut matched = false;

    for val in headers.get_all(IF_NONE_MATCH) {
        let Ok(raw) = val.to_str() else {
            continue;
        };
        for item in split_etag_list(raw) {
            let trimmed = item.trim();
            if trimmed == "*" {
                translated_tags.push("*".to_string());
                matched = true;
                continue;
            }
            let (weak_prefix, rest) = if let Some(after) = trimmed.strip_prefix("W/") {
                ("W/", after.trim())
            } else {
                ("", trimmed)
            };
            if let Some(inner) = rest.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
                && let Some(orig_inner) = inner.strip_suffix(expected_suffix)
            {
                translated_tags.push(format!("{weak_prefix}\"{orig_inner}\""));
                matched = true;
            }
        }
    }

    headers.remove(IF_NONE_MATCH);

    if matched && !translated_tags.is_empty() {
        let joined = translated_tags.join(", ");
        if let Ok(hv) = HeaderValue::from_str(&joined) {
            headers.insert(IF_NONE_MATCH, hv);
        }
        true
    } else {
        headers.remove(IF_MODIFIED_SINCE);
        false
    }
}

fn split_etag_list(raw: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut in_quotes = false;
    let mut start = 0usize;
    let bytes = raw.as_bytes();
    let mut i = 0usize;

    while i < bytes.len() {
        match bytes[i] {
            b'\\' if in_quotes => {
                i += 2;
                continue;
            }
            b'"' => {
                in_quotes = !in_quotes;
            }
            b',' if !in_quotes => {
                let slice = raw[start..i].trim();
                if !slice.is_empty() {
                    out.push(slice);
                }
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }

    let tail = raw[start..].trim();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

fn merge_vary_accept(existing_headers: &[String]) -> Option<String> {
    let mut tokens: Vec<String> = Vec::new();
    let mut has_star = false;
    let mut has_accept = false;

    for raw in existing_headers {
        for part in raw.split(',') {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                continue;
            }
            if trimmed == "*" {
                has_star = true;
            } else if trimmed.eq_ignore_ascii_case("accept") {
                has_accept = true;
            }
            tokens.push(trimmed.to_string());
        }
    }

    if has_star {
        return Some("*".to_string());
    }
    if !has_accept {
        tokens.push("Accept".to_string());
    }
    Some(tokens.join(", "))
}

fn rewrite_content_disposition(val: &str, new_ext: &str) -> String {
    val.split(';')
        .map(|segment| {
            let trimmed = segment.trim();
            let Some((key, raw_val)) = trimmed.split_once('=') else {
                return trimmed.to_string();
            };
            let key_trim = key.trim();
            let val_trim = raw_val.trim();

            if key_trim.eq_ignore_ascii_case("filename*") {
                let (was_quoted, inner) = strip_outer_quotes(val_trim);
                if let Some((prefix, name)) = split_rfc5987_value(inner) {
                    let updated_name = replace_filename_extension(name, new_ext);
                    if was_quoted {
                        format!("{key_trim}=\"{prefix}{updated_name}\"")
                    } else {
                        format!("{key_trim}={prefix}{updated_name}")
                    }
                } else {
                    trimmed.to_string()
                }
            } else if key_trim.eq_ignore_ascii_case("filename") {
                let (was_quoted, inner) = strip_outer_quotes(val_trim);
                if inner.is_empty() {
                    trimmed.to_string()
                } else {
                    let updated = replace_filename_extension(inner, new_ext);
                    if was_quoted {
                        format!("{key_trim}=\"{updated}\"")
                    } else {
                        format!("{key_trim}={updated}")
                    }
                }
            } else {
                trimmed.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn strip_outer_quotes(s: &str) -> (bool, &str) {
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        (true, &s[1..s.len() - 1])
    } else {
        (false, s)
    }
}

fn split_rfc5987_value(s: &str) -> Option<(&str, &str)> {
    let first_quote = s.find('\'')?;
    let rest = &s[first_quote + 1..];
    let second_quote = rest.find('\'')?;
    let split_idx = first_quote + 1 + second_quote + 1;
    Some((&s[..split_idx], &s[split_idx..]))
}

fn replace_filename_extension(filename: &str, new_ext: &str) -> String {
    let sep_pos = filename.rfind(['/', '\\']);
    let (dir_prefix, base) = match sep_pos {
        Some(idx) => (&filename[..=idx], &filename[idx + 1..]),
        None => ("", filename),
    };

    if base.is_empty() {
        return filename.to_string();
    }

    let stem = match base.rfind('.') {
        Some(dot) if dot > 0 => &base[..dot],
        _ => base,
    };

    format!("{dir_prefix}{stem}.{new_ext}")
}
