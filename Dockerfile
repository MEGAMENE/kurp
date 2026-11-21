FROM ubuntu:noble AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    CMAKE_GENERATOR=Ninja

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        g++ \
        make \
        ninja-build \
        pkg-config \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable --profile minimal \
    && rustc --version \
    && cargo --version \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY realcugan-ncnn-vulkan-rs/Cargo.toml realcugan-ncnn-vulkan-rs/build.rs ./realcugan-ncnn-vulkan-rs/
COPY realcugan-ncnn-vulkan-rs/src ./realcugan-ncnn-vulkan-rs/src

# Build portable binary without host-specific -march=native / REALCUGAN_NATIVE=ON
RUN cargo build --release --locked --bin kurp

FROM ubuntu:noble AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    KURP_CONF_DIR=/config \
    KURP_REALCUGAN__MODELS_PATH=/app/models

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        libvulkan1 \
        mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /config /app/models \
    && chown -R 1000:1000 /config \
    && chmod 0777 /config

WORKDIR /app

COPY --from=builder /build/target/release/kurp /app/kurp
COPY --from=builder /build/realcugan-ncnn-vulkan-rs/src/realcugan-ncnn-vulkan/models /app/models

RUN chmod 0755 /app/kurp \
    && chmod -R a+rX /app/models

EXPOSE 3030

ENTRYPOINT ["/app/kurp"]
