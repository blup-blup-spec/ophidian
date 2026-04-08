# ==============================================================================
# 🐍 Snake Detector — Docker Multi-Stage Build (OPTIMIZED)
# Cross-compiles on your PC (x86_64) for Raspberry Pi 4 (ARM64)
# NCNN build stripped down to bare minimum to reduce compile time
# ==============================================================================

# ==============================================================================
# STAGE 1: BUILD — Compile NCNN + Snake Detector for ARM64
# ==============================================================================
FROM debian:bookworm-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    make \
    git \
    ca-certificates \
    pkg-config \
    libopencv-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Build NCNN from source — MINIMAL build (skip tests, tools, benchmarks, examples)
# Using -j2 instead of -j$(nproc) to prevent QEMU from OOM-killing
WORKDIR /opt
RUN git clone --depth 1 --branch 20240410 https://github.com/Tencent/ncnn.git && \
    cd ncnn && \
    git submodule update --init && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/opt/ncnn/install \
          -DNCNN_VULKAN=OFF \
          -DNCNN_BUILD_EXAMPLES=OFF \
          -DNCNN_BUILD_TOOLS=OFF \
          -DNCNN_BUILD_BENCHMARK=OFF \
          -DNCNN_BUILD_TESTS=OFF \
          -DNCNN_OPENMP=ON \
          -DNCNN_SIMPLEOCV=OFF \
          -DNCNN_SYSTEM_GLSLANG=OFF \
          -DNCNN_STDIO=ON \
          -DNCNN_STRING=ON \
          -DNCNN_PIXEL=ON \
          -DNCNN_PIXEL_ROTATE=OFF \
          -DNCNN_PIXEL_AFFINE=OFF \
          -DNCNN_PIXEL_DRAWING=OFF \
          .. && \
    make -j2 && \
    make install

# Copy and build the snake detector application
WORKDIR /app
COPY inference/ ./

RUN mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -Dncnn_DIR=/opt/ncnn/install/lib/cmake/ncnn \
          .. && \
    make -j2 && \
    echo "✅ Snake detector compiled successfully"

# ==============================================================================
# STAGE 2: RUNTIME — Minimal image for Raspberry Pi
# ==============================================================================
FROM debian:bookworm-slim AS runtime

# Install only runtime dependencies (no compilers, no dev headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core406 \
    libopencv-imgproc406 \
    libopencv-highgui406 \
    libopencv-videoio406 \
    libopencv-imgcodecs406 \
    libgomp1 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the compiled binary
COPY --from=builder /app/build/snake_detector /app/snake_detector

# Copy model files (these will be provided at build time or mounted at runtime)
COPY models/ /app/models/

# Environment tuning for Raspberry Pi 4
ENV OMP_NUM_THREADS=4
ENV NCNN_THREADS=4
ENV OMP_WAIT_POLICY=PASSIVE
ENV OPENBLAS_NUM_THREADS=1

# Default command: run headless (no display) for maximum FPS
ENTRYPOINT ["/app/snake_detector"]
CMD ["--model-dir", "/app/models", "--headless", "--camera", "0", "--size", "320"]
