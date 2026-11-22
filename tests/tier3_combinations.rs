//! Tier 3: Komga HTTP Auth Parity, Method Routing, Content Negotiation,
//! ETag/304 Caching, Metadata PATCH Invalidation, Proxy Headers & AppConfig Integration

mod common;
use common::*;

use std::collections::HashMap;
use std::fs;
use std::sync::Arc;

use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use image::{DynamicImage, Rgb, RgbImage};
use kurp::clients::komga_client::ForwardAuthHeaders;
use kurp::config::app_config::AppConfig;
use kurp::upscaler::upscaler::UpscaleEngine;
use realcugan_ncnn_vulkan_rs::RealCuganModelType;
use serde_json::json;

fn small_png(w: u32, h: u32, color: Rgb<u8>) -> bytes::Bytes {
    encode_to_png_bytes(&DynamicImage::ImageRgb8(RgbImage::from_pixel(w, h, color)))
}

#[tokio::test]
async fn test_tier3_komga_auth_parity_api_key_auth_token_bearer_basic_cookie_and_401() {
    let komga = MockKomgaServer::start().await;
    komga.set_require_auth(true);
    komga.set_book_tags("book-auth", "series-auth", vec!["upscale"]);
    let img_bytes = small_png(32, 32, Rgb([200, 200, 200]));
    komga.set_page("book-auth", 1, MockPageEntry::png(img_bytes));

    let mut cfg = create_cpu_upscale_config(&komga.base_url);
    cfg.upscale_tag = Some("upscale".to_string());
    let upscaler = Arc::new(UpscaleEngine::init(&cfg).await.expect("CPU upscaler init"));
    let kurp = TestKurpServer::start(cfg, upscaler.clone()).await;

    let client = reqwest::Client::new();
    let page_url = format!("{}/api/v1/books/book-auth/pages/1", kurp.base_url);

    // 1. Unauthenticated request must preserve upstream 401 (NEVER 502) and NOT poison BookProfileManager
    let unauth_resp = client.get(&page_url).send().await.unwrap();
    assert_eq!(unauth_resp.status(), StatusCode::UNAUTHORIZED);

    // 2. X-API-Key (Mihon / Tachiyomi)
    komga.clear_recorded();
    let r_api_key = client
        .get(&page_url)
        .header("X-API-Key", "komga-api-key-123")
        .send()
        .await
        .unwrap();
    assert_eq!(r_api_key.status(), StatusCode::OK);
    let upscaled_bytes = r_api_key.bytes().await.unwrap();
    let decoded = image::load_from_memory(&upscaled_bytes).unwrap();
    assert_eq!((decoded.width(), decoded.height()), (64, 64));
    assert!(
        komga
            .recorded_requests()
            .iter()
            .all(|r| r.headers.get("x-api-key").map(String::as_str) == Some("komga-api-key-123")),
        "X-API-Key must forward to /book, /pages, and page fetches"
    );

    // 3. X-Auth-Token (Komelia / mobile clients)
    komga.clear_recorded();
    let r_auth_tok = client
        .get(&page_url)
        .header("X-Auth-Token", "komga-session-tok-456")
        .send()
        .await
        .unwrap();
    assert_eq!(r_auth_tok.status(), StatusCode::OK);
    assert!(
        komga
            .recorded_requests()
            .iter()
            .any(|r| r.headers.get("x-auth-token").map(String::as_str) == Some("komga-session-tok-456"))
    );

    // 4. Authorization: Bearer (OIDC / proxy JWT — previously rejected with 400 by TypedHeader<Basic>!)
    komga.clear_recorded();
    let r_bearer = client
        .get(&page_url)
        .header(header::AUTHORIZATION, "Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig")
        .send()
        .await
        .unwrap();
    assert_eq!(r_bearer.status(), StatusCode::OK);
    assert!(
        komga.recorded_requests().iter().any(|r| {
            r.headers.get("authorization").map(String::as_str)
                == Some("Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig")
        })
    );

    // 5. Combined Basic + multiple Cookies must BOTH be forwarded cleanly
    let mut raw_headers = HeaderMap::new();
    raw_headers.insert(header::AUTHORIZATION, HeaderValue::from_static("Basic dXNlcjpwYXNz"));
    raw_headers.append(header::COOKIE, HeaderValue::from_static("KOMGA-SESSION=abc"));
    raw_headers.append(header::COOKIE, HeaderValue::from_static("komga-remember-me=xyz"));
    let parsed_auth = ForwardAuthHeaders::from_headers(&raw_headers);
    assert!(!parsed_auth.is_empty());
    assert_eq!(
        parsed_auth.cookie.as_ref().unwrap().to_str().unwrap(),
        "KOMGA-SESSION=abc; komga-remember-me=xyz"
    );

    komga.clear_recorded();
    let r_combo = client
        .get(&page_url)
        .header(header::AUTHORIZATION, "Basic dXNlcjpwYXNz")
        .header(header::COOKIE, "KOMGA-SESSION=abc; komga-remember-me=xyz")
        .send()
        .await
        .unwrap();
    assert_eq!(r_combo.status(), StatusCode::OK);
    assert!(komga.recorded_requests().iter().any(|r| {
        r.headers.get("authorization").map(String::as_str) == Some("Basic dXNlcjpwYXNz")
            && r.headers.get("cookie").map(String::as_str)
                == Some("KOMGA-SESSION=abc; komga-remember-me=xyz")
    }));

    upscaler.shutdown();
}

#[tokio::test]
async fn test_tier3_http_methods_head_options_and_opds_v1_2_zero_based() {
    let komga = MockKomgaServer::start().await;
    komga.set_book_tags("book-m", "series-m", vec!["upscale"]);
    let img_bytes = small_png(32, 32, Rgb([180, 180, 180]));
    let expected_len = img_bytes.len().to_string();
    komga.set_page("book-m", 1, MockPageEntry::png(img_bytes));

    let mut cfg = create_cpu_upscale_config(&komga.base_url);
    cfg.upscale_tag = Some("upscale".to_string());
    let upscaler = Arc::new(UpscaleEngine::init(&cfg).await.expect("CPU upscaler init"));
    let kurp = TestKurpServer::start(cfg, upscaler.clone()).await;

    let client = reqwest::Client::new();

    // 1. HEAD /api/v1/books/{id}/pages/1 MUST route to proxy_handler (keeps Content-Length, skips BookProfile)
    komga.clear_recorded();
    let head_resp = client
        .head(format!("{}/api/v1/books/book-m/pages/1", kurp.base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(head_resp.status(), StatusCode::OK);
    assert_eq!(
        head_resp
            .headers()
            .get(header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok()),
        Some(expected_len.as_str())
    );
    assert!(
        !komga
            .recorded_requests()
            .iter()
            .any(|r| r.path.ends_with("/pages")),
        "HEAD must NOT trigger BookProfile /pages fetch"
    );

    // 2. OPTIONS CORS preflight on intercepted routes MUST NOT return 405 Method Not Allowed
    for path in [
        "/api/v1/books/book-m/pages/1",
        "/opds/v1.2/books/book-m/pages/1",
        "/api/v1/books/book-m/metadata",
        "/api/v1/books/metadata",
    ] {
        let opt_resp = client
            .request(reqwest::Method::OPTIONS, format!("{}{}", kurp.base_url, path))
            .send()
            .await
            .unwrap();
        assert_eq!(
            opt_resp.status(),
            StatusCode::NO_CONTENT,
            "OPTIONS on {path} must proxy CORS preflight cleanly"
        );
    }

    // 3. OPDS v1.2 page route + ?zero_based=true intercepts & upscales cleanly
    let opds_resp = client
        .get(format!(
            "{}/opds/v1.2/books/book-m/pages/1?zero_based=true",
            kurp.base_url
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(opds_resp.status(), StatusCode::OK);
    assert_eq!(
        opds_resp.headers().get(header::CONTENT_TYPE).unwrap(),
        "image/webp"
    );
    let opds_img = image::load_from_memory(&opds_resp.bytes().await.unwrap()).unwrap();
    assert_eq!((opds_img.width(), opds_img.height()), (64, 64));

    upscaler.shutdown();
}

#[tokio::test]
async fn test_tier3_query_convert_accept_negotiation_vary_and_content_disposition() {
    let komga = MockKomgaServer::start().await;
    komga.set_book_tags("book-fmt", "series-fmt", vec!["upscale"]);
    let img_bytes = small_png(32, 32, Rgb([120, 160, 200]));
    komga.set_page(
        "book-fmt",
        1,
        MockPageEntry {
            bytes: img_bytes,
            content_type: "image/png; charset=utf-8".to_string(),
            etag: Some("\"raw-fmt-1\"".to_string()),
            content_disposition: Some(
                "inline;filename=\"page 01.png\"; filename*=UTF-8'en'page%2001.png".to_string(),
            ),
        },
    );

    let cfg = create_cpu_upscale_config(&komga.base_url);
    let upscaler = Arc::new(UpscaleEngine::init(&cfg).await.expect("CPU upscaler init"));
    let kurp = TestKurpServer::start(cfg, upscaler.clone()).await;

    let client = reqwest::Client::new();
    let base_page = format!("{}/api/v1/books/book-fmt/pages/1", kurp.base_url);

    // 1. Default return_format: WebP rewrites parameterized Content-Type, Content-Disposition (incl. RFC 5987), and Vary: Accept
    let r_webp = client.get(&base_page).send().await.unwrap();
    assert_eq!(r_webp.status(), StatusCode::OK);
    assert_eq!(r_webp.headers().get(header::CONTENT_TYPE).unwrap(), "image/webp");
    assert_eq!(r_webp.headers().get(header::VARY).unwrap(), "Accept");
    let cd = r_webp
        .headers()
        .get(header::CONTENT_DISPOSITION)
        .unwrap()
        .to_str()
        .unwrap();
    assert!(cd.contains("filename=\"page 01.webp\""), "got {cd}");
    assert!(cd.contains("filename*=UTF-8'en'page%2001.webp"), "got {cd}");

    // 2. ?convert=png overrides return_format: WebP
    let r_png = client
        .get(format!("{base_page}?convert=png"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_png.status(), StatusCode::OK);
    assert_eq!(r_png.headers().get(header::CONTENT_TYPE).unwrap(), "image/png");
    assert!(r_png.bytes().await.unwrap().starts_with(b"\x89PNG\r\n\x1a\n"));

    // 3. ?convert=jpeg overrides return_format: WebP
    let r_jpeg = client
        .get(format!("{base_page}?convert=jpeg"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_jpeg.status(), StatusCode::OK);
    assert_eq!(r_jpeg.headers().get(header::CONTENT_TYPE).unwrap(), "image/jpeg");
    assert!(r_jpeg.bytes().await.unwrap().starts_with(&[0xFF, 0xD8, 0xFF]));

    // 4. Accept header rejecting image/webp (`image/webp;q=0, image/png;q=0.9`) negotiates PNG
    let r_accept = client
        .get(&base_page)
        .header(header::ACCEPT, "image/png;q=0.9, image/webp;q=0")
        .send()
        .await
        .unwrap();
    assert_eq!(r_accept.status(), StatusCode::OK);
    assert_eq!(r_accept.headers().get(header::CONTENT_TYPE).unwrap(), "image/png");
    assert_eq!(r_accept.headers().get(header::VARY).unwrap(), "Accept");

    upscaler.shutdown();
}

#[tokio::test]
async fn test_tier3_206_partial_content_unknown_mime_and_truncated_body_error() {
    // 1. 206 Partial Content and unknown MIME (image/jxl) passthrough verbatim
    let komga = MockKomgaServer::start().await;
    komga.set_book_tags("book-pt", "series-pt", vec!["upscale"]);
    let raw_jxl = bytes::Bytes::from_static(b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0aFAKEJXLBYTES");
    komga.set_page(
        "book-pt",
        1,
        MockPageEntry {
            bytes: raw_jxl.clone(),
            content_type: "image/jxl".to_string(),
            etag: Some("\"jxl-1\"".to_string()),
            content_disposition: None,
        },
    );

    let mut cfg = create_cpu_upscale_config(&komga.base_url);
    cfg.auto_levels.enabled = false;
    let kurp = TestKurpServer::start(cfg, Arc::new(UpscaleEngine::new_empty())).await;
    let client = reqwest::Client::new();

    // Unknown MIME passthrough preserves exact Content-Type (NEVER rewritten to image/png!)
    let r_jxl = client
        .get(format!("{}/api/v1/books/book-pt/pages/1", kurp.base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(r_jxl.status(), StatusCode::OK);
    assert_eq!(r_jxl.headers().get(header::CONTENT_TYPE).unwrap(), "image/jxl");
    assert_eq!(r_jxl.bytes().await.unwrap(), raw_jxl);

    // 206 Partial Content passes through verbatim without image decode
    *komga.state.partial_206.write().unwrap() = true;
    let r_206 = client
        .get(format!("{}/api/v1/books/book-pt/pages/1", kurp.base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(r_206.status(), StatusCode::PARTIAL_CONTENT);
    assert!(r_206.headers().contains_key(header::CONTENT_RANGE));

    // 2. Truncated upstream body read error returns 502 Bad Gateway (NEVER empty 200 OK!)
    let trunc_upstream = spawn_truncated_body_upstream().await;
    let mut cfg_trunc = create_cpu_upscale_config(&trunc_upstream);
    cfg_trunc.auto_levels.enabled = false;
    let kurp_trunc = TestKurpServer::start(cfg_trunc, Arc::new(UpscaleEngine::new_empty())).await;
    let r_trunc = client
        .get(format!(
            "{}/api/v1/books/trunc-book/pages/1",
            kurp_trunc.base_url
        ))
        .send()
        .await
        .unwrap();
    assert_ne!(
        r_trunc.status(),
        StatusCode::OK,
        "Truncated upstream body must NEVER produce empty 200 OK"
    );
    assert_eq!(r_trunc.status(), StatusCode::BAD_GATEWAY);
}

#[tokio::test]
async fn test_tier3_etag_config_hash_304_translation_and_untagged_304() {
    let komga = MockKomgaServer::start().await;
    komga.set_book_tags("book-tagged", "s1", vec!["upscale"]);
    komga.set_book_tags("book-untagged", "s2", vec![]);

    let png_bytes = small_png(32, 32, Rgb([210, 210, 210]));
    komga.set_page(
        "book-tagged",
        1,
        MockPageEntry {
            bytes: png_bytes.clone(),
            content_type: "image/png".to_string(),
            etag: Some("W/\"orig-etag-1\"".to_string()),
            content_disposition: None,
        },
    );
    komga.set_page(
        "book-untagged",
        1,
        MockPageEntry {
            bytes: png_bytes,
            content_type: "image/png".to_string(),
            etag: Some("\"untagged-etag-9\"".to_string()),
            content_disposition: None,
        },
    );

    let mut cfg = create_cpu_upscale_config(&komga.base_url);
    cfg.upscale_tag = Some("upscale".to_string());
    let expected_suffix = format!("-kurp-{:016x}", cfg.config_hash());
    let upscaler = Arc::new(UpscaleEngine::init(&cfg).await.expect("CPU upscaler init"));
    let kurp = TestKurpServer::start(cfg, upscaler.clone()).await;

    let client = reqwest::Client::new();
    let tagged_url = format!("{}/api/v1/books/book-tagged/pages/1", kurp.base_url);

    // 1. Initial GET emits config-hashed weak ETag
    let r1 = client.get(&tagged_url).send().await.unwrap();
    assert_eq!(r1.status(), StatusCode::OK);
    let etag1 = r1
        .headers()
        .get(header::ETAG)
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    let expected_full = format!("W/\"orig-etag-1{expected_suffix}\"");
    assert_eq!(etag1, expected_full);

    // 2. Sending matching config-hashed If-None-Match translates back to Komga's raw ETag -> 304 Not Modified!
    let r_304 = client
        .get(&tagged_url)
        .header(header::IF_NONE_MATCH, &etag1)
        .send()
        .await
        .unwrap();
    assert_eq!(r_304.status(), StatusCode::NOT_MODIFIED);
    assert_eq!(
        r_304.headers().get(header::ETAG).unwrap().to_str().unwrap(),
        expected_full
    );
    assert_eq!(r_304.headers().get(header::VARY).unwrap(), "Accept");

    // 3. Sending stale/un-upscaled ETag on tagged book strips If-None-Match and returns fresh 200 upscaled bytes
    let r_stale = client
        .get(&tagged_url)
        .header(header::IF_NONE_MATCH, "W/\"orig-etag-1\"")
        .send()
        .await
        .unwrap();
    assert_eq!(r_stale.status(), StatusCode::OK);

    // 4. Untagged book retains raw If-None-Match -> 304 Not Modified (100% cache preservation!)
    let untagged_url = format!("{}/api/v1/books/book-untagged/pages/1", kurp.base_url);
    let r_untagged_304 = client
        .get(&untagged_url)
        .header(header::IF_NONE_MATCH, "\"untagged-etag-9\"")
        .send()
        .await
        .unwrap();
    assert_eq!(r_untagged_304.status(), StatusCode::NOT_MODIFIED);

    upscaler.shutdown();
}

#[tokio::test]
async fn test_tier3_patch_tags_null_bulk_metadata_and_2xx_only_invalidation() {
    let komga = MockKomgaServer::start().await;
    komga.set_book_tags("b1", "s1", vec!["upscale"]);
    komga.set_book_tags("b2", "s2", vec!["upscale"]);
    komga.set_book_tags("b3", "s3", vec!["upscale"]);
    let img = small_png(32, 32, Rgb([200, 200, 200]));
    for b in ["b1", "b2", "b3"] {
        komga.set_page(b, 1, MockPageEntry::png(img.clone()));
    }

    let mut cfg = create_cpu_upscale_config(&komga.base_url);
    cfg.upscale_tag = Some("upscale".to_string());
    cfg.auto_levels.enabled = false;
    let kurp = TestKurpServer::start(cfg, Arc::new(UpscaleEngine::new_empty())).await;
    let auth = ForwardAuthHeaders::default();

    // Warm UpscaleTagChecker cache for b1, b2, b3
    assert!(kurp.state.upscale_tag_checker.komga_contains_upscale_tag("b1", &auth).await.unwrap());
    assert!(kurp.state.upscale_tag_checker.komga_contains_upscale_tag("b2", &auth).await.unwrap());
    assert!(kurp.state.upscale_tag_checker.komga_contains_upscale_tag("b3", &auth).await.unwrap());

    // Flip tags on Komga to empty; cached values still return true until invalidated
    komga.set_book_tags("b1", "s1", vec![]);
    komga.set_book_tags("b2", "s2", vec![]);
    komga.set_book_tags("b3", "s3", vec![]);
    assert!(kurp.state.upscale_tag_checker.komga_contains_upscale_tag("b1", &auth).await.unwrap());

    let client = reqwest::Client::new();

    // 1. Upstream 500 on PATCH /api/v1/books/b1/metadata must NOT invalidate b1 cache
    *komga.state.patch_status.write().unwrap() = Some(StatusCode::INTERNAL_SERVER_ERROR);
    let r_fail = client
        .patch(format!("{}/api/v1/books/b1/metadata", kurp.base_url))
        .json(&json!({"tags": null}))
        .send()
        .await
        .unwrap();
    assert_eq!(r_fail.status(), StatusCode::INTERNAL_SERVER_ERROR);
    assert!(
        kurp.state.upscale_tag_checker.komga_contains_upscale_tag("b1", &auth).await.unwrap(),
        "Failed 500 PATCH must NOT invalidate tag cache"
    );

    // 2. Upstream 204 with `"tags": null` MUST invalidate b1 cache
    *komga.state.patch_status.write().unwrap() = None;
    let r_ok = client
        .patch(format!("{}/api/v1/books/b1/metadata", kurp.base_url))
        .json(&json!({"tags": null}))
        .send()
        .await
        .unwrap();
    assert_eq!(r_ok.status(), StatusCode::NO_CONTENT);
    assert!(
        !kurp.state.upscale_tag_checker.komga_contains_upscale_tag("b1", &auth).await.unwrap(),
        "2xx PATCH with tags:null must invalidate b1"
    );

    // 3. Malformed PATCH body must NOT return local 500 (proxied cleanly to Komga)
    let r_malformed = client
        .patch(format!("{}/api/v1/books/b2/metadata", kurp.base_url))
        .body("{not valid json")
        .send()
        .await
        .unwrap();
    assert_eq!(r_malformed.status(), StatusCode::NO_CONTENT);

    // 4. Bulk PATCH /api/v1/books/metadata invalidates ONLY b2 (has "tags"), leaving b3 ("title" only) untouched
    let r_bulk = client
        .patch(format!("{}/api/v1/books/metadata", kurp.base_url))
        .json(&json!({
            "b2": { "tags": [] },
            "b3": { "title": "Renamed Only" }
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(r_bulk.status(), StatusCode::NO_CONTENT);
    assert!(!kurp.state.upscale_tag_checker.komga_contains_upscale_tag("b2", &auth).await.unwrap());
    assert!(kurp.state.upscale_tag_checker.komga_contains_upscale_tag("b3", &auth).await.unwrap());

    // 5. Series PATCH with tags invalidates all cached books
    let r_series = client
        .patch(format!("{}/api/v1/series/s3/metadata", kurp.base_url))
        .json(&json!({"tags": null}))
        .send()
        .await
        .unwrap();
    assert_eq!(r_series.status(), StatusCode::NO_CONTENT);
    assert!(!kurp.state.upscale_tag_checker.komga_contains_upscale_tag("b3", &auth).await.unwrap());
}

#[tokio::test]
async fn test_tier3_proxy_dynamic_connection_hop_x_forwarded_and_location_rewrite() {
    let komga = MockKomgaServer::start().await;
    let cfg = create_cpu_upscale_config(&komga.base_url);
    let kurp = TestKurpServer::start(cfg, Arc::new(UpscaleEngine::new_empty())).await;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();

    // 1. Dynamic Connection hop stripping + Proxy-Connection stripping + TE: trailers + X-Forwarded-*
    let echo_resp = client
        .get(format!("{}/proxy-echo", kurp.base_url))
        .header(header::CONNECTION, "X-Secret-Hop, X-Custom-2")
        .header("X-Secret-Hop", "must-be-stripped")
        .header("X-Custom-2", "must-be-stripped")
        .header("Proxy-Connection", "keep-alive")
        .header(header::TE, "trailers")
        .header("X-Keep-Me", "preserved")
        .send()
        .await
        .unwrap();
    assert_eq!(echo_resp.status(), StatusCode::OK);
    let echoed: serde_json::Value = echo_resp.json().await.unwrap();
    let obj = echoed.as_object().unwrap();
    assert!(!obj.contains_key("x-secret-hop"));
    assert!(!obj.contains_key("x-custom-2"));
    assert!(!obj.contains_key("proxy-connection"));
    assert_eq!(obj.get("te").and_then(|v| v.as_str()), Some("trailers"));
    assert_eq!(obj.get("x-keep-me").and_then(|v| v.as_str()), Some("preserved"));
    assert!(obj.contains_key("x-forwarded-host"));
    assert!(obj.contains_key("x-forwarded-proto"));
    assert!(obj.contains_key("x-forwarded-for"));

    // 2. Upstream Location header rewrite (prevents leaking upstream Komga host/port on 302 redirects)
    let redir_resp = client
        .get(format!("{}/redirect-test", kurp.base_url))
        .header(header::HOST, "kurp.example.local:3030")
        .send()
        .await
        .unwrap();
    assert_eq!(redir_resp.status(), StatusCode::FOUND);
    let loc = redir_resp
        .headers()
        .get(header::LOCATION)
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(loc, "http://kurp.example.local:3030/login?next=/dashboard");
}

#[tokio::test]
async fn test_tier3_config_validation_posix_env_and_serde_saphyr_roundtrip() {
    let komga = MockKomgaServer::start().await;
    let disabled_kurp = TestKurpServer::start(
        create_cpu_upscale_config(&komga.base_url),
        Arc::new(UpscaleEngine::new_empty()),
    )
    .await;
    let client = reqwest::Client::new();

    // 1. /kurp/config is unmounted (404) when allow_config_updates is false, and works when true
    let r_disabled = client
        .get(format!("{}/kurp/config", disabled_kurp.base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(r_disabled.status(), StatusCode::NOT_FOUND);

    let mut cfg = create_cpu_upscale_config(&komga.base_url);
    cfg.allow_config_updates = true;
    let kurp = TestKurpServer::start(cfg, Arc::new(UpscaleEngine::new_empty())).await;

    let r_get = client
        .get(format!("{}/kurp/config", kurp.base_url))
        .send()
        .await
        .unwrap();
    assert_eq!(r_get.status(), StatusCode::OK);

    // 2. Invalid POST /kurp/config fails validate() with 400 BEFORE touching disk
    let mut bad_cfg = AppConfig::default();
    bad_cfg.realcugan.model = RealCuganModelType::Pro;
    bad_cfg.realcugan.scale = 4; // unsupported in Layer 2!
    let r_bad = client
        .post(format!("{}/kurp/config", kurp.base_url))
        .json(&bad_cfg)
        .send()
        .await
        .unwrap();
    assert_eq!(r_bad.status(), StatusCode::BAD_REQUEST);

    // 3. Isolated temp_dir: write_config_to_dir + POSIX __ env overrides + KURP_CONF_DIR ignore
    let tmp_dir = std::env::temp_dir().join(format!(
        "kurp_tier3_cfg_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let mut saved = AppConfig {
        port: 3131,
        ..AppConfig::default()
    };
    saved.realcugan.concurrent_jobs = 2;
    AppConfig::write_config_to_dir(&saved, &tmp_dir).unwrap();

    let mut env_map = HashMap::new();
    env_map.insert("KURP_CONF_DIR".to_string(), tmp_dir.to_string_lossy().to_string());
    env_map.insert("KURP_RAYON_THREADS".to_string(), "4".to_string());
    env_map.insert("KURP_REALCUGAN__GPUID".to_string(), "-1".to_string());
    env_map.insert("KURP_AUTO_LEVELS__CORRECT_PAPER_CAST".to_string(), "false".to_string());

    let loaded = AppConfig::from_dir_and_env(&tmp_dir, env_map).unwrap();
    assert_eq!(loaded.port, 3131);
    assert_eq!(loaded.rayon_threads, 4);
    assert_eq!(loaded.realcugan.gpuid, -1);
    assert_eq!(loaded.realcugan.concurrent_jobs, 2);
    assert!(!loaded.auto_levels.correct_paper_cast);
    assert!(loaded.validate().is_ok());

    let _ = fs::remove_dir_all(&tmp_dir);
}
