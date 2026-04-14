ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.source="https://github.com/eleven-am/vox"
LABEL org.opencontainers.image.licenses="Apache-2.0"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    ffmpeg \
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

COPY --chown=vox:vox . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --compile-bytecode --python-preference only-system --python 3.12 && \
    mkdir -p $HOME/.vox/adapters && \
    mkdir -p $HOME/.cache/huggingface/hub && \
    mkdir -p $HOME/.cache/torch/hub && \
    chown -R vox:vox $HOME

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
        uv pip install --python .venv/bin/python onnxruntime-gpu transformers huggingface-hub; \
    else \
        uv pip install --python .venv/bin/python onnxruntime transformers huggingface-hub; \
    fi && \
    chown -R vox:vox $HOME

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python .venv/bin/python \
        ./adapters/vox-kokoro \
        ./adapters/vox-microsoft \
        ./adapters/vox-parakeet \
        ./adapters/vox-qwen \
        ./adapters/vox-voxtral && \
    chown -R vox:vox $HOME

USER vox

ENV PATH="$HOME/app/.venv/bin:$PATH" \
    VOX_HOME=$HOME/.vox \
    VOX_BUNDLED_ADAPTERS=$HOME/app/adapters \
    VOX_DEVICE=auto \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    DO_NOT_TRACK=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TORCH_CPP_LOG_LEVEL=ERROR

EXPOSE 11435
EXPOSE 9090
VOLUME $HOME/.vox

CMD ["vox", "serve", "--host", "0.0.0.0"]
