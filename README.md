# Komga and Kavita upscaling reverse proxy

Reverse proxy that intercepts image requests and applies upscaling. Other requests are transparently proxied
without noticable delay

## Building

required dependencies:

- cmake
- g++
- vulkan loader library
- ncnn 
- glslang

1. run `git submodule update --init --recursive` to download subprojects required for build
2. set GLSLANG_TARGET_DIR environment variable (ubuntu: `/usr/lib/x86_64-linux-gnu/cmake/` arch linux: `/usr/lib/cmake`)
3. run `GLSLANG_TARGET_DIR=/usr/lib/cmake cargo build --release`


## Config

Config file must be named `config.yml`
By default config file location is set to the current directory of execution. You can override config location
with `KURP_CONF_DIR` environment variable. If no config file is found then default values will be used.

### Default config

```yaml
port: 3030 # listen port
upstream_url: "http://localhost:8080" # Komga or Kavita url
allow_config_updates: false # exposes config get and update enpoints that allow runtime config updates
upscale: true # enable upscaling
upscale_tag: # if present will only upscale if book or series contains specified tag. Komga only
size_threshold_enabled: true # enables content size check
size_threshold: 500 # in KB. will not upscale if image size is bigger than specified size
size_threshold_png: 1000 # in KB. will not upscale if image size is bigger than specified size. PNG only

# return format of the upscaled image. WebP is lossless.
# available options are "WebP", "Jpeg", "Png" and "Original"
return_format: WebP

realcugan:
  gpuid: 0 # gpu device to use (-1 = cpu). if you have single gpu then this should usually be 0
  scale: 2 # upscale ratio (2/3/4)
  noise: -1 # denoise level (-1/0/1/2/3)
  model: Se # realcugan model (Se, Pro, Nose)
  tile_size: 0 # tile size (>=32/0=auto)
  sync_gap: 3 # sync gap mode (0/1/2/3)
  tta_mode: false # enable tta mode
  num_threads: 2 # thread count for upscaling
  models_path: "./models" # path to directory with models

auto_levels:
  enabled: true # enable intelligent black/white level correction and tone adjustment
  mode: "All" # processing mode: "All" (manga + comics), "Manga" (B&W only), "Color" (color only), "Off"
  black_clip_percent: 0.1 # percentile for ink black point (0.1% preserves subtle shadow artwork)
  white_clip_percent: 0.1 # percentile for paper white point (0.1% protects soft highlights)
  max_black_shift: 30 # maximum black point shift (safeguard against crushing halftones)
  min_white_threshold: 200 # minimum white point (safeguard against blowing out dark scenes)
  correct_paper_cast: false # correct paper yellowing on vintage scans (disabled by default to preserve tinted art)
  gamma: 1.0 # midtone gamma adjustment (1.0 = linear)
  output_grayscale_for_monochrome: true # encode pure monochrome manga as 1-channel grayscale (saves 20-40% size)
```

## Docker Compose

```yml
version: "3.7"
services:
  kurp:
    container_name: kurp
    image: sndxr/kurp:latest
    user: "1000:1000"
    # optional env configuration
    environment: 
      - RUST_LOG=info
      - KURP_UPSTREAM_URL=http://kavita:5000
      - KURP_REALCUGAN.GPUID=0
    volumes:
      - ./kurp:/config
    ports:
      - 3030:3030
    devices:
      - /dev/dri/renderD128:/dev/dri/renderD128
      - /dev/dri/card0:/dev/dri/card0
    restart: unless-stopped
```
