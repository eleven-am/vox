ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ARG VOX_UID=10000
ARG VOX_GID=10000
ARG VOX_ACCELERATOR=gpu
ARG VOX_DEFAULT_DEVICE=auto
ARG TORCH_VERSION=2.10.0
ARG TORCHAUDIO_VERSION=2.10.0
ARG TRANSFORMERS_VERSION=4.57.6

FROM ghcr.io/astral-sh/uv:0.7.20 AS uv

FROM ${BASE_IMAGE} AS builder

ARG VOX_UID
ARG VOX_GID
ARG TARGETARCH
ARG VOX_ACCELERATOR
ARG TORCH_VERSION
ARG TORCHAUDIO_VERSION
ARG TRANSFORMERS_VERSION

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    python3 \
    python3-dev \
    python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV HOME=/home/vox \
    PATH=/home/vox/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_HTTP_TIMEOUT=300

WORKDIR $HOME/app

COPY --from=uv /uv /bin/uv

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --compile-bytecode --no-install-project --python-preference only-system --python 3.12

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=src,target=src \
    uv sync --frozen --compile-bytecode --no-editable --python-preference only-system --python 3.12 && \
    .venv/bin/python -m ensurepip --upgrade

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$VOX_ACCELERATOR" = "cpu" ]; then \
        uv pip install --python .venv/bin/python \
            "torch==${TORCH_VERSION}" \
            "torchaudio==${TORCHAUDIO_VERSION}"; \
    elif [ "$TARGETARCH" = "amd64" ]; then \
        uv pip install --python .venv/bin/python --reinstall \
            "torch==${TORCH_VERSION}+cu128" \
            "torchaudio==${TORCHAUDIO_VERSION}+cu128" \
            --index-url https://download.pytorch.org/whl/cu128; \
    else \
        uv pip install --python .venv/bin/python \
            "torch==${TORCH_VERSION}" \
            "torchaudio==${TORCHAUDIO_VERSION}"; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$VOX_ACCELERATOR" = "cpu" ]; then \
        uv pip install --python .venv/bin/python \
            onnxruntime \
            "transformers==${TRANSFORMERS_VERSION}" \
            huggingface-hub; \
    elif [ "$TARGETARCH" = "amd64" ]; then \
        uv pip install --python .venv/bin/python \
            onnxruntime-gpu \
            "transformers==${TRANSFORMERS_VERSION}" \
            huggingface-hub; \
    else \
        uv pip install --python .venv/bin/python \
            onnxruntime \
            "transformers==${TRANSFORMERS_VERSION}" \
            huggingface-hub; \
    fi && \
    uv pip install --python .venv/bin/python \
        accelerate \
        datasets \
        librosa \
        sentencepiece \
        colorlog \
        espeakng-loader \
        phonemizer-fork && \
    uv pip install --python .venv/bin/python --no-deps \
        kokoro-onnx==0.4.9 \
        onnx-asr[hub]==0.11.0

FROM ${BASE_IMAGE} AS runtime

LABEL org.opencontainers.image.source="https://github.com/eleven-am/vox"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ARG VOX_UID
ARG VOX_GID
ARG VOX_DEFAULT_DEVICE

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    gosu \
    git \
    libopus0 \
    libsndfile1 \
    libsoxr0 \
    python3 \
    python3-dev \
    python3-venv \
    sox \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN groupadd --gid "$VOX_GID" vox && \
    useradd --create-home --shell /bin/bash --uid "$VOX_UID" --gid "$VOX_GID" vox

ENV HOME=/home/vox \
    PATH=/home/vox/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

COPY --from=uv /uv /bin/uv
COPY --from=builder --chown=vox:vox /home/vox/app/.venv /home/vox/app/.venv
COPY --chmod=755 docker/vox-entrypoint.sh /usr/local/bin/vox-entrypoint.sh

RUN install -d -o vox -g vox \
        $HOME/.vox/adapters \
        $HOME/.cache \
        $HOME/.cache/huggingface \
        $HOME/.cache/huggingface/hub \
        $HOME/.cache/torch \
        $HOME/.cache/torch/hub \
        /tmp/uvcache

ENV PATH="$HOME/app/.venv/bin:$PATH" \
    VOX_HOME=$HOME/.vox \
    VOX_DISABLE_BUNDLED_ADAPTERS=1 \
    VOX_DEVICE=${VOX_DEFAULT_DEVICE} \
    UV_CACHE_DIR=/tmp/uvcache \
    UV_LINK_MODE=copy \
    HF_HOME=$HOME/.cache/huggingface \
    HUGGINGFACE_HUB_CACHE=$HOME/.cache/huggingface/hub \
    HF_XET_CACHE=$HOME/.cache/huggingface/xet \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    HF_HUB_DISABLE_XET=1 \
    DO_NOT_TRACK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TORCH_CPP_LOG_LEVEL=ERROR

EXPOSE 11435
EXPOSE 9090
VOLUME $HOME/.vox

HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:11435/v1/health || exit 1

ENTRYPOINT ["/usr/local/bin/vox-entrypoint.sh"]
CMD ["vox", "serve", "--host", "0.0.0.0"]
