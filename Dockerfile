ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/eleven-am/vox"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
    gosu \
    libsndfile1 \
    libopus0 \
    libsoxr0 \
    git \
    python3 \
    python3-dev \
    python3-venv \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ARG VOX_UID=10000
ARG VOX_GID=10000

RUN groupadd --gid $VOX_GID vox && \
    useradd --create-home --shell /bin/bash --uid $VOX_UID --gid $VOX_GID vox

ENV HOME=/home/vox \
    PATH=/home/vox/.local/bin:$PATH

WORKDIR $HOME/app

COPY --from=ghcr.io/astral-sh/uv:0.7.20 /uv /bin/uv

ENV UV_HTTP_TIMEOUT=300

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --compile-bytecode --no-install-project --python-preference only-system --python 3.12

COPY --chmod=755 docker/vox-entrypoint.sh /usr/local/bin/vox-entrypoint.sh

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    --mount=type=bind,source=src,target=src \
    uv sync --frozen --compile-bytecode --no-editable --python-preference only-system --python 3.12 && \
    install -d -o vox -g vox \
        $HOME/.vox/adapters \
        $HOME/.cache \
        $HOME/.cache/huggingface \
        $HOME/.cache/huggingface/hub \
        $HOME/.cache/torch \
        $HOME/.cache/torch/hub \
        /tmp/uvcache

ARG TARGETARCH

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETARCH" = "amd64" ]; then \
        uv pip install --python .venv/bin/python --reinstall --pre torch torchaudio \
            --index-url https://download.pytorch.org/whl/nightly/cu128; \
    else \
        uv pip install --python .venv/bin/python torch torchaudio; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETARCH" = "amd64" ]; then \
        uv pip install --python .venv/bin/python onnxruntime-gpu "transformers==4.57.1" huggingface-hub; \
    else \
        uv pip install --python .venv/bin/python onnxruntime "transformers==4.57.1" huggingface-hub; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
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

ENV PATH="$HOME/app/.venv/bin:$PATH" \
    VOX_HOME=$HOME/.vox \
    VOX_DISABLE_BUNDLED_ADAPTERS=1 \
    VOX_DEVICE=auto \
    UV_CACHE_DIR=/tmp/uvcache \
    UV_LINK_MODE=copy \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    DO_NOT_TRACK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TORCH_CPP_LOG_LEVEL=ERROR

EXPOSE 11435
EXPOSE 9090
VOLUME $HOME/.vox

ENTRYPOINT ["/usr/local/bin/vox-entrypoint.sh"]
CMD ["vox", "serve", "--host", "0.0.0.0"]
