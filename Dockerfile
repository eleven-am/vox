# ---- Base stage: system deps + vox core ----
FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg libopus0 libsoxr0 git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy and install core vox package
COPY pyproject.toml .
COPY src/ src/
RUN uv pip install --system --no-cache .

# Create vox home directory
RUN mkdir -p /root/.vox/adapters

EXPOSE 11435
VOLUME /root/.vox

CMD ["vox", "serve", "--host", "0.0.0.0"]

# ---- CPU target ----
FROM base AS cpu

# Pre-install common ML runtimes for CPU
RUN uv pip install --system --no-cache \
    torch --index-url https://download.pytorch.org/whl/cpu \
    onnxruntime \
    transformers \
    huggingface-hub

# ---- GPU target ----
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS gpu-base

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip \
    libsndfile1 ffmpeg libopus0 libsoxr0 git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf python3.12 /usr/bin/python3 \
    && ln -sf python3.12 /usr/bin/python

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY pyproject.toml .
COPY src/ src/
RUN uv pip install --system --no-cache .

# GPU ML runtimes
RUN uv pip install --system --no-cache \
    torch \
    onnxruntime-gpu \
    transformers \
    huggingface-hub

RUN mkdir -p /root/.vox/adapters

EXPOSE 11435
VOLUME /root/.vox

CMD ["vox", "serve", "--host", "0.0.0.0"]
