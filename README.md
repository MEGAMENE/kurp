# Kurp — Real-Time Manga & Comic Page Restoration & Upscaling Proxy for Komga

**Kurp** is a high-performance Rust reverse proxy for **[Komga](https://komga.org/)** that transparently restores and upscales manga and comic pages on the fly. It combines a first-principles **D65 CIELAB page restoration & auto-leveling engine** with Vulkan-accelerated **Real-CUGAN** super-resolution (`realcugan-ncnn-vulkan-rs`) and lossless **WebP** delivery.

Clients (Komga Web UI, Mihon/Tachiyomi, Panels, Paperback, OPDS readers) connect directly to Kurp on port `3030` exactly as if connecting to Komga. Kurp intercepts page image requests, optionally restricts processing to series or books carrying a specific Komga tag (`upscale_tag`), restores and upscales eligible pages in memory, and streams all other API and UI traffic transparently.

---

## Key Features

- **Transparent Komga REST & OPDS v1.2 Proxy**
  - Intercepts `GET /api/v1/books/{book_id}/pages/{page_number}` (1-based) and `GET /opds/v1.2/books/{book_id}/pages/{page_number}` (0-based OPDS index automatically normalized to 1-based for volume profiling).
  - Forwards `HEAD` requests directly to Komga without triggering image decoding, GPU inference, or `BookProfile` sampling (preserving upstream `Content-Length` and headers).
  - Streams `OPTIONS` preflight requests, HTTP `206 Partial Content` range responses, non-image conversions (`?convert=pdf`), and all other Komga routes unchanged with proper hop-by-hop `Connection` header stripping, `X-Forwarded-For` / `X-Forwarded-Proto` / `X-Forwarded-Host` injection, and upstream `Location` redirect rewriting.
- **Full Komga Authentication Parity**
  - Forwards `X-API-Key`, `X-Auth-Token`, `Authorization: Bearer <token>`, `Authorization: Basic <credentials>`, and `Cookie` session headers when querying Komga metadata or sampling pages for `BookProfile`.
- **Smart ETag Suffixing & Tag Cache Coherency**
  - Appends a deterministic configuration digest (`-kurp-<16_hex_hash>`) to upstream `ETag` headers so changing `config.yml` automatically invalidates browser/client caches, and strips the suffix on incoming `If-None-Match` headers so upstream `304 Not Modified` revalidation works seamlessly.
  - Emits `Vary: Accept` on page responses and rewrites `Content-Disposition` filenames to match the encoded output format (`.webp`, `.png`, `.jpg`).
  - Automatically invalidates cached book/series tags upon `2xx` responses to `PATCH /api/v1/series/{series_id}/metadata`, `PATCH /api/v1/books/{book_id}/metadata`, and bulk `PATCH /api/v1/books/metadata` (including `tags: null` reset semantics).
- **Zero-Harm D65 CIELAB Auto-Leveling & Page Restoration**
  - **Pre-Flight Veto Gates (`< 1.5 ms` on 6MP)**: Classifies pages in D65 CIELAB into `Color`, `Monochrome`, or `Mixed`. Unconditionally returns full-color pages, intentional dual-zone sepia artwork, high-key snow/fog scenes, low-key night scenes, small thumbnails (`< 400x400`), and already-optimal digital pages with **zero full-resolution copies** ($\Delta E = 0.0$).
  - **Unified `[2..=85]` Shadow Restoration & `BookProfile` Sanity Guard**: Detects faded ink floors across the `[2..=85]` shadow spectrum with 5-tap binomial dual-valley validation and a strict `3.5%` digital-black cap. Synchronously samples 9–15 evenly-spaced pages (default `12`) across the volume in parallel with Page 1's upstream body download to clamp anomalous single-page black-point drift.
  - **Inversion-Proof Fritsch-Carlson Monotonic Cubic Spline LUT**: Smooth tonal expansion with adaptive soft toe/shoulder knots and a gamma-aware midpoint.
  - **Arbitrary-Hue Localized Paper-Cast Neutralization**: Cleans aged yellow, blue, green, and pink/magenta paper casts as well as low-chroma muddy shadow ink. On `Mixed` pages (monochrome manga with color accents or pastel highlights), applies C1 smoothstep directional cast subtraction across a $4 \times 4$ bilinear cell grid and outer `10%..14%` margin ring while protecting a $3 \times 3$ morphologically dilated high-chroma accent mask ($C^*_{\text{res}} \ge 20.0$) and preserving interior cream skin tones.
  - **3-Tier Pre-Flight Tile Quality Guard (`< 15.0 ms` total 6MP restoration)**: Validates candidate corrections on a contiguous $32 \times 32$ tile grid across Tier 1 (dynamic range & positive-slope affine residual PSNR), Tier 2 (structure-only SSIM & 2D prime-lattice CIELAB $\Delta E_{\text{chroma}}$), and Tier 3 (tile-seam-safe strided Sobel gradient NCC & Michelson contrast). Automatically retries at 50% attenuation on Tier 2/3 rejection and returns the untouched full-resolution image if still rejected.
  - **Post-Upscale Monochrome `Luma8` Folding**: Folds restored monochrome pages to 1-channel grayscale (`Luma8`, white-compositing `Rgba8` alpha cleanly) after Real-CUGAN upscaling when `output_grayscale_for_monochrome: true` (skipped automatically when `correct_paper_cast: false` preserves an intentional paper tint).
- **Modern Pure-Rust AVIF & Lossless `libwebp` Codec Pipeline**
  - Decodes AVIF via pure-Rust `zenavif` with 64-bit ISOBMFF `largesize` overflow protection, primary-item `pitm`/`ipma` color property scoping (`rICC`/`prof` precedence over `nclx`), HDR PQ/HLG preservation, and automatic Display P3 / Rec.2020 ICC profile synthesis.
  - Encodes `WebP` via native `libwebp` (`webpx`) in **true lossless** mode with ICC profile preservation, falling back automatically to lossless PNG whenever an upscaled dimension exceeds WebP's `16383 px` hardware limit.
- **Lock-Free Hot-Reload & Bounded Concurrency**
  - Uses `Arc<RealCugan>` + `ArcSwap` for lock-free configuration reads and hot-reloads via `/kurp/config`, bounded by `max_concurrent_pipelines` and `realcugan.concurrent_jobs` semaphores, with graceful `Ctrl+C` / `SIGTERM` / Windows console close shutdown.

---

## Quick Start

### 1. Prebuilt Windows Release (`x86-64-v3`)

Download `kurp-windows-x86_64-v3.zip` from [GitHub Releases](../../releases):
- **Stable releases**: tagged `v*` / `windows-latest`
- **Preview builds**: tagged `preview-latest` (built from `development`)

Each archive contains `kurp.exe`, `config.yml`, and the bundled Real-CUGAN model directories (`models/models-se`, `models/models-pro`, `models/models-nose`) — **no external NCNN DLLs, Vulkan SDK, or `dav1d.dll` required**.

| Requirement | Minimum Supported |
| :--- | :--- |
| **OS** | Windows 10 / 11 (64-bit) or Windows Server 2019+ |
| **CPU ISA** | `x86-64-v3` (**AVX2**, BMI1/BMI2, F16C, FMA, LZCNT) — Intel 4th-Gen Haswell (2013+) or AMD Excavator / Ryzen 1000–9000+ |
| **GPU** | Vulkan 1.1+ capable GPU (NVIDIA, AMD, Intel Arc/iGPU), or CPU mode (`realcugan.gpuid: -1`) |

1. Extract `kurp-windows-x86_64-v3.zip`.
2. Edit `config.yml` and set `upstream_url` to your Komga server (e.g. `http://localhost:8080`).
3. Run `kurp.exe` (or set `KURP_CONF_DIR` to the directory containing `config.yml`).

---

### 2. Docker & Docker Compose

The `Dockerfile` builds a self-contained `ubuntu:noble` runtime image containing `/app/kurp`, `/app/models`, `libvulkan1`, `mesa-vulkan-drivers`, and a pre-created writable `/config` directory compatible with non-root `user: "1000:1000"`.

#### Build the Docker Image

```bash
docker build -t kurp:latest .
```

#### `docker-compose.yml` Example

```yaml
services:
  kurp:
    image: kurp:latest
    container_name: kurp
    user: "1000:1000"
    restart: unless-stopped
    ports:
      - "3030:3030"
    devices:
      - /dev/dri:/dev/dri
    volumes:
      - ./config:/config
    environment:
      KURP_CONF_DIR: "/config"
      KURP_UPSTREAM_URL: "http://komga:25600"
      KURP_RETURN_FORMAT: "WebP"
      KURP_MAX_CONCURRENT_PIPELINES: "6"
      KURP_REALCUGAN__GPUID: "0"
      KURP_REALCUGAN__SCALE: "2"
      KURP_REALCUGAN__MODEL: "Se"
      KURP_REALCUGAN__MODELS_PATH: "/app/models"
      KURP_AUTO_LEVELS__ENABLED: "true"
      RUST_LOG: "info"
```

> **Note on GPU Passthrough**:
> - **Intel / AMD GPUs (Linux)**: Pass `/dev/dri:/dev/dri` as shown above (supported directly by `mesa-vulkan-drivers` inside the container). If `/dev/dri/renderD128` requires group access on your host, add `group_add: ["render", "video"]` (or the numeric GIDs from `stat -c '%g' /dev/dri/render*`).
> - **NVIDIA GPUs (Linux)**: Install the NVIDIA Container Toolkit with Vulkan support and add `gpus: all` + `NVIDIA_DRIVER_CAPABILITIES: compute,utility,graphics`.
> - **CPU Fallback**: Set `KURP_REALCUGAN__GPUID: "-1"` if running without a Vulkan device.

---

### 3. Building from Source

Kurp compiles `ncnn` and `glslang` statically from its vendored `realcugan-ncnn-vulkan-rs` submodule via CMake and decodes AVIF in pure Rust via `zenavif`. **You do not need to install the Vulkan SDK, prebuilt NCNN libraries, or `libdav1d`.**

#### Prerequisites

- **Rust Toolchain**: Stable Rust (`1.93+` recommended for `zenavif 0.1.6`; crate MSRV `1.88`, Rust 2024 edition)
- **CMake**: `3.18+`
- **Build Generator & C++17 Compiler**:
  - **Linux**: `g++` (or `clang++`), `ninja-build`, `make`, `pkg-config`
  - **Windows**: Visual Studio 2022 Build Tools (`Desktop development with C++` / MSVC `cl.exe`) and CMake

#### Clone, Build & Test

```bash
# 1. Clone repository with nested Real-CUGAN / NCNN / glslang submodules
git clone --recurse-submodules https://github.com/MEGAMENE/kurp.git
cd kurp

# 2. Build optimized release binary (LTO enabled)
cargo build --release --locked

# 3. Copy models next to your config.yml (or set realcugan.models_path)
# Models are located in:
# realcugan-ncnn-vulkan-rs/src/realcugan-ncnn-vulkan/models/

# 4. Run linter and full release test suite
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked --release
```

> **Optional Native CPU Tuning**:
> - For portable `x86-64-v3` (AVX2) builds on Windows MSVC, set `CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUSTFLAGS="-C target-cpu=x86-64-v3"` and `REALCUGAN_NATIVE=1`.
> - For single-machine host tuning on Linux/macOS/Windows, setting `REALCUGAN_NATIVE=1` enables `-march=native` (GCC/Clang) or `/arch:AVX2` (MSVC x64) inside the C++ `realcugan-ncnn-vulkan` build. Leave `REALCUGAN_NATIVE` unset when building portable container images.

---

## Configuration Reference

Kurp loads `<KURP_CONF_DIR>/config.yml` (defaulting to `./config.yml` in the working directory when `KURP_CONF_DIR` is unset) and overlays any `KURP_*` environment variables. Unknown YAML keys or unknown `KURP_*` variables are rejected at startup (`#[serde(deny_unknown_fields)]`) to catch typos early.

### Default `config.yml`

```yaml
port: 3030
upstream_url: "http://localhost:8080"
allow_config_updates: false
# config_api_token: "change-me-if-config-updates-enabled"
upscale: true
size_threshold_enabled: true
size_threshold: 500
size_threshold_png: 1000
max_upscale_dimension: 4000
jpeg_quality: 90
max_concurrent_pipelines: 6
# rayon_threads: 8
return_format: WebP
# upscale_tag: "upscale"

realcugan:
  gpuid: 0
  scale: 2
  noise: -1
  model: Se
  tile_size: 0
  sync_gap: 2
  tta_level: 1
  use_fp16_arithmetic: false
  # num_threads: 4
  concurrent_jobs: 1
  models_path: "./models"

auto_levels:
  enabled: true
  output_grayscale_for_monochrome: true
  book_profile:
    cache_size: 100
    cache_ttl_seconds: 3600
    sample_pages: 12
    parallel_page_fetches: 8
    timeout_seconds: 45
  detection:
    black_clip_percent: 0.1
    white_clip_percent: 0.1
    surge_ratio_threshold: 2.8
    protect_sepia_artwork: true
  correction:
    max_black_shift: 85
    min_white_threshold: 200
    correct_paper_cast: true
    gamma: 1.0
  quality_guards:
    tier1:
      check_tail_erosion: true
      min_dynamic_range: 64
    tier2:
      ssim_threshold: 0.92
      chroma_delta_p95: 2.5
      chroma_delta_median: 1.5
    tier3:
      gradient_ncc: 0.90
      min_contrast_ratio: 0.50
      stride: 4
```

### POSIX Environment Variable Overrides (`__` Hierarchy Separator)

All configuration fields can be overridden with `KURP_`-prefixed POSIX environment variables using a **double underscore (`__`)** to separate nested sections (and single underscores within field names):

- `KURP_CONF_DIR=/path/to/config/dir` — Directory where Kurp reads and writes `config.yml`.
- Top-level fields: `KURP_<FIELD_NAME>` (e.g. `KURP_PORT=3030`, `KURP_UPSTREAM_URL=http://komga:25600`, `KURP_UPSCALE_TAG=upscale`).
- Nested `realcugan.*` fields: `KURP_REALCUGAN__<FIELD_NAME>` (e.g. `KURP_REALCUGAN__GPUID=0`, `KURP_REALCUGAN__MODELS_PATH=/app/models`).
- Nested `auto_levels.*` fields: `KURP_AUTO_LEVELS__<SECTION>__<FIELD_NAME>` (e.g. `KURP_AUTO_LEVELS__ENABLED=true`, `KURP_AUTO_LEVELS__BOOK_PROFILE__SAMPLE_PAGES=12`, `KURP_AUTO_LEVELS__QUALITY_GUARDS__TIER2__SSIM_THRESHOLD=0.92`).

### Complete Options Table

#### Top-Level Proxy & Pipeline Options

| YAML Key | Env Var | Default | Valid Range / Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| `port` | `KURP_PORT` | `3030` | `1..=65535` | TCP port on `0.0.0.0` where Kurp listens. |
| `upstream_url` | `KURP_UPSTREAM_URL` | `"http://localhost:8080"` | Valid `http://` or `https://` URL | Base URL of your upstream Komga instance. |
| `upscale` | `KURP_UPSCALE` | `true` | `true`, `false` | Master toggle for Kurp's page restoration and upscaling pipeline. When `false`, page requests bypass directly to Komga via raw proxy passthrough. |
| `return_format` | `KURP_RETURN_FORMAT` | `WebP` | `WebP`, `Png`, `Jpeg`, `Original` | Output image format when no client `?convert=` query or `Accept: image/webp` override applies. `WebP` uses lossless `libwebp` (with automatic PNG fallback above `16383 px`). `Original` preserves the source format (converting AVIF inputs to lossless WebP). |
| `size_threshold_enabled` | `KURP_SIZE_THRESHOLD_ENABLED` | `true` | `true`, `false` | When `true`, pages whose upstream file size already exceeds `size_threshold` (or `size_threshold_png` for PNGs) bypass Kurp processing and return the upstream response directly. |
| `size_threshold` | `KURP_SIZE_THRESHOLD` | `500` | `>= 0` (KiB) | Maximum file size in KiB for non-PNG images to enter the restoration/upscaling pipeline when `size_threshold_enabled: true`. |
| `size_threshold_png` | `KURP_SIZE_THRESHOLD_PNG` | `1000` | `>= 0` (KiB) | Maximum file size in KiB for PNG images to enter the restoration/upscaling pipeline when `size_threshold_enabled: true`. |
| `max_upscale_dimension` | `KURP_MAX_UPSCALE_DIMENSION` | `4000` | `>= 0` (px, `0` disables) | When `> 0`, pages whose source `width > max_upscale_dimension` or `height > max_upscale_dimension` bypass Kurp processing and return the upstream response directly. |
| `jpeg_quality` | `KURP_JPEG_QUALITY` | `90` | `1..=100` | JPEG encoder quality used when the resolved output format is `Jpeg`. |
| `max_concurrent_pipelines` | `KURP_MAX_CONCURRENT_PIPELINES` | `6` | `>= 1` | Maximum in-flight page decode/restore/upscale/encode pipelines allowed concurrently (prevents OOM under burst page loads). |
| `rayon_threads` | `KURP_RAYON_THREADS` | unset (all logical cores) | `>= 1` | Optional thread count for the global Rayon CPU pool used by `auto_levels`. |
| `upscale_tag` | `KURP_UPSCALE_TAG` | unset | Any non-empty string | When set (e.g. `"upscale"`), only books or series carrying that tag in Komga are processed; untagged books bypass straight to Komga. When unset, all eligible pages are processed. |
| `allow_config_updates` | `KURP_ALLOW_CONFIG_UPDATES` | `false` | `true`, `false` | Exposes `GET /kurp/config` and `POST /kurp/config` for runtime configuration inspection and hot-reload. |
| `config_api_token` | `KURP_CONFIG_API_TOKEN` | unset | Non-empty string | Optional shared secret protecting `/kurp/config`. When set, requests to `/kurp/config` must supply `Authorization: Bearer <token>` or `X-Kurp-Token: <token>`. |

#### `realcugan` Section

| YAML Key | Env Var | Default | Valid Range / Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| `realcugan.gpuid` | `KURP_REALCUGAN__GPUID` | `0` | `>= -1` | Vulkan device index (`0` = primary GPU, `-1` = CPU inference via NCNN). |
| `realcugan.scale` | `KURP_REALCUGAN__SCALE` | `2` | `2`, `3`, `4` | Upscaling factor (`Nose` supports `2` only; `Pro` supports `2`, `3`; `Se` supports `2`, `3`, `4`). |
| `realcugan.noise` | `KURP_REALCUGAN__NOISE` | `-1` | `-1`, `0`, `1`, `2`, `3` | Denoise strength (`-1` = conservative/no denoise; `Nose` supports `0` only; `Pro` supports `-1`, `0`, `3`; `Se` supports `-1`, `0`, `1`, `2`, `3`, with `3x`/`4x` supporting `-1`, `0`, `3`). |
| `realcugan.model` | `KURP_REALCUGAN__MODEL` | `Se` | `Se`, `Pro`, `Nose` | Real-CUGAN model family (`models-se`, `models-pro`, `models-nose`). |
| `realcugan.tile_size` | `KURP_REALCUGAN__TILE_SIZE` | `0` | `0` or `>= 32` | Tile size in pixels for GPU inference (`0` = automatic tiling based on GPU VRAM). |
| `realcugan.sync_gap` | `KURP_REALCUGAN__SYNC_GAP` | `2` | `0`, `1`, `2`, `3` | Vulkan queue synchronization interval (`0` = no sync, `1` = accurate, `2` = rough, `3` = very rough). |
| `realcugan.tta_level` | `KURP_REALCUGAN__TTA_LEVEL` | `1` | `1`, `2`, `4`, `8` | Test-Time Augmentation passes (`1` = disabled/fastest, `8` = 8x slower highest quality). |
| `realcugan.use_fp16_arithmetic` | `KURP_REALCUGAN__USE_FP16_ARITHMETIC` | `false` | `true`, `false` | Enables FP16 storage/packed/arithmetic inside NCNN. Defaults to `false` (FP32) to prevent black-tile artifacts on large pages; enable only if verified stable on your GPU. |
| `realcugan.num_threads` | `KURP_REALCUGAN__NUM_THREADS` | unset | `>= 1` | CPU thread count for NCNN when `gpuid: -1` or for CPU stages. |
| `realcugan.concurrent_jobs` | `KURP_REALCUGAN__CONCURRENT_JOBS` | `1` | `>= 1` | Maximum concurrent GPU inference calls dispatched to `RealCugan`. Keep at `1` or `2` on consumer GPUs to avoid VRAM exhaustion. |
| `realcugan.models_path` | `KURP_REALCUGAN__MODELS_PATH` | `"./models"` (or `<KURP_CONF_DIR>/models`) | Directory path | Path to the parent directory containing `models-se`, `models-pro`, and `models-nose`. |

#### `auto_levels` Section

| YAML Key | Env Var | Default | Valid Range / Values | Description |
| :--- | :--- | :--- | :--- | :--- |
| `auto_levels.enabled` | `KURP_AUTO_LEVELS__ENABLED` | `true` | `true`, `false` | Master toggle for the CIELAB page restoration & auto-leveling pipeline. |
| `auto_levels.output_grayscale_for_monochrome` | `KURP_AUTO_LEVELS__OUTPUT_GRAYSCALE_FOR_MONOCHROME` | `true` | `true`, `false` | Folds verified monochrome pages to 1-channel `Luma8` after upscaling (cutting lossless WebP/PNG size by 50–70%). |
| `auto_levels.book_profile.cache_size` | `KURP_AUTO_LEVELS__BOOK_PROFILE__CACHE_SIZE` | `100` | `>= 1` | Maximum number of books cached in the in-memory `BookProfile` LRU cache. |
| `auto_levels.book_profile.cache_ttl_seconds` | `KURP_AUTO_LEVELS__BOOK_PROFILE__CACHE_TTL_SECONDS` | `3600` | `>= 1` | Time-to-live in seconds for cached `BookProfile` entries. |
| `auto_levels.book_profile.sample_pages` | `KURP_AUTO_LEVELS__BOOK_PROFILE__SAMPLE_PAGES` | `12` | `>= 1` | Number of evenly-spaced pages sampled across a book to establish the volume's median black-floor outlier guard. |
| `auto_levels.book_profile.parallel_page_fetches` | `KURP_AUTO_LEVELS__BOOK_PROFILE__PARALLEL_PAGE_FETCHES` | `8` | `>= 1` | Global concurrency cap on background upstream page fetches during `BookProfile` sampling. |
| `auto_levels.book_profile.timeout_seconds` | `KURP_AUTO_LEVELS__BOOK_PROFILE__TIMEOUT_SECONDS` | `45` | `>= 1` | Timeout in seconds before a `BookProfile` sampling task is cancelled (`JoinSet` aborts remaining fetches cleanly). |
| `auto_levels.detection.black_clip_percent` | `KURP_AUTO_LEVELS__DETECTION__BLACK_CLIP_PERCENT` | `0.1` | `0.0..=25.0` | Lower luminance histogram clip percentile. |
| `auto_levels.detection.white_clip_percent` | `KURP_AUTO_LEVELS__DETECTION__WHITE_CLIP_PERCENT` | `0.1` | `0.0..=25.0` | Upper luminance histogram clip percentile. |
| `auto_levels.detection.surge_ratio_threshold` | `KURP_AUTO_LEVELS__DETECTION__SURGE_RATIO_THRESHOLD` | `2.8` | `> 0.0` | Histogram density surge ratio used for elevated shadow ink shelf detection. |
| `auto_levels.detection.protect_sepia_artwork` | `KURP_AUTO_LEVELS__DETECTION__PROTECT_SEPIA_ARTWORK` | `true` | `true`, `false` | Enables the D65 CIELAB dual-zone sepia guard ($b^*_{\text{paper}} \ge 6.0, b^*_{\text{shadow}} \ge 5.0, a^*_{\text{shadow}} \ge 0.5, \Delta b^*_{\text{grid}} < 12.0$) to prevent bleaching intentional sepia art. |
| `auto_levels.correction.max_black_shift` | `KURP_AUTO_LEVELS__CORRECTION__MAX_BLACK_SHIFT` | `85` | `0..=120` | Maximum shadow luminance level (`[2..=85]`) that may be anchored to black ink. |
| `auto_levels.correction.min_white_threshold` | `KURP_AUTO_LEVELS__CORRECTION__MIN_WHITE_THRESHOLD` | `200` | `128..=255` | Minimum highlight luminance level eligible to be stretched to paper white. |
| `auto_levels.correction.correct_paper_cast` | `KURP_AUTO_LEVELS__CORRECTION__CORRECT_PAPER_CAST` | `true` | `true`, `false` | Neutralizes aged paper color casts (yellow, blue, green, pink) and muddy shadow ink. When `false`, applies luminance-only stretching while preserving original paper tint (`paper_cast_preserved = true`). |
| `auto_levels.correction.gamma` | `KURP_AUTO_LEVELS__CORRECTION__GAMMA` | `1.0` | `0.2..=5.0` | Midtone gamma shaping parameter for the Fritsch-Carlson spline midpoint knot $P_2$. |
| `auto_levels.quality_guards.tier1.check_tail_erosion` | `KURP_AUTO_LEVELS__QUALITY_GUARDS__TIER1__CHECK_TAIL_EROSION` | `true` | `true`, `false` | Enables the Tier 1 midtone-crush (`<= 0.25`) check alongside the clamped positive-slope affine residual check (`MSE <= 220.0` / `PSNR >= 22.0 dB`). |
| `auto_levels.quality_guards.tier1.min_dynamic_range` | `KURP_AUTO_LEVELS__QUALITY_GUARDS__TIER1__MIN_DYNAMIC_RANGE` | `64` | `0..=255` | Minimum output dynamic range ($p_{99} - p_1$) required by Tier 1 on non-narrow-input pages. |
| `auto_levels.quality_guards.tier2.ssim_threshold` | `KURP_AUTO_LEVELS__QUALITY_GUARDS__TIER2__SSIM_THRESHOLD` | `0.92` | `0.0..=1.0` | Minimum structure-only SSIM required across active $8 \times 8$ blocks in Tier 2. |
| `auto_levels.quality_guards.tier2.chroma_delta_p95` | `KURP_AUTO_LEVELS__QUALITY_GUARDS__TIER2__CHROMA_DELTA_P95` | `2.5` | `> 0.0` | Maximum 95th-percentile CIELAB $\Delta E_{\text{chroma}}$ drift allowed on non-cast pixels in Tier 2. |
| `auto_levels.quality_guards.tier2.chroma_delta_median` | `KURP_AUTO_LEVELS__QUALITY_GUARDS__TIER2__CHROMA_DELTA_MEDIAN` | `1.5` | `> 0.0` | Maximum median CIELAB $\Delta E_{\text{chroma}}$ drift allowed on non-cast pixels in Tier 2. |
| `auto_levels.quality_guards.tier3.gradient_ncc` | `KURP_AUTO_LEVELS__QUALITY_GUARDS__TIER3__GRADIENT_NCC` | `0.90` | `0.0..=1.0` | Minimum strided Sobel gradient Normalized Cross-Correlation required in Tier 3. |
| `auto_levels.quality_guards.tier3.min_contrast_ratio` | `KURP_AUTO_LEVELS__QUALITY_GUARDS__TIER3__MIN_CONTRAST_RATIO` | `0.50` | `0.0..=1.0` | Minimum 1st-percentile ($p_{01}$) $32 \times 32$ block Michelson contrast retention ratio in Tier 3. |
| `auto_levels.quality_guards.tier3.stride` | `KURP_AUTO_LEVELS__QUALITY_GUARDS__TIER3__STRIDE` | `4` | `>= 1` | Pixel sampling stride for the Tier 3 tile-seam-safe Sobel gradient check. |

---

## Selective Upscaling with Komga Tags (`upscale_tag`)

By default (`upscale_tag` commented out), Kurp processes every eligible book page requested from Komga.

If you only want specific series or books processed:
1. Set `upscale_tag: "upscale"` in `config.yml` (or `KURP_UPSCALE_TAG=upscale`).
2. In Komga, add the `upscale` tag to any **Series** or individual **Book** you want restored/upscaled.
3. Kurp caches each book's tag membership in an LRU cache (checking both book tags and parent series tags using the caller's Komga credentials) and automatically invalidates the cache whenever you edit book or series metadata through Kurp (`PATCH /api/v1/series/{series_id}/metadata`, `PATCH /api/v1/books/{book_id}/metadata`, or bulk `PATCH /api/v1/books/metadata`).

---

## Runtime Configuration API (`/kurp/config`)

When `allow_config_updates: true` is set in `config.yml`, Kurp mounts:
- `GET /kurp/config` — Returns the active `AppConfig` as JSON.
- `POST /kurp/config` — Validates a JSON `AppConfig` payload, atomically swaps the in-memory `ArcSwap<AppConfig>` (rebuilding the `RealCugan` engine if `upscale` or `realcugan.*` parameters changed), persists the updated configuration to `<KURP_CONF_DIR>/config.yml`, and automatically restarts the HTTP listener if `port` changed.

When `config_api_token` is configured, all requests to `/kurp/config` must include either:
- `Authorization: Bearer <config_api_token>`
- `X-Kurp-Token: <config_api_token>`

---

## License

MIT License — see [LICENSE](LICENSE).
