ARG BASE_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ARG ORT_BUILDER_IMAGE=nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ARG ORT_GIT_REF=v1.24.0

FROM ${BASE_IMAGE} AS vox-base

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

# Install dependencies first (cached layer)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --compile-bytecode --no-install-project --python-preference only-system --python 3.12

COPY --chown=vox:vox . .

# Install vox itself + ML runtimes
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --compile-bytecode --python-preference only-system --python 3.12 && \
    mkdir -p $HOME/.vox/adapters && \
    mkdir -p $HOME/.cache/huggingface/hub && \
    mkdir -p $HOME/.cache/torch/hub && \
    chown -R vox:vox $HOME

ARG TARGETARCH

# Install PyTorch — CUDA wheels on amd64, CPU on arm64
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETARCH" = "amd64" ]; then \
        uv pip install --python .venv/bin/python --reinstall --pre torch torchaudio \
            --index-url https://download.pytorch.org/whl/nightly/cu128; \
    else \
        uv pip install --python .venv/bin/python torch torchaudio; \
    fi

FROM ${ORT_BUILDER_IMAGE} AS vox-arm64-onnx-builder

ARG TARGETARCH
ARG ORT_GIT_REF

ENV DEBIAN_FRONTEND=noninteractive

RUN if [ "$TARGETARCH" != "arm64" ]; then \
        mkdir -p /opt/ort-wheels; \
        exit 0; \
    fi && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ccache \
        cmake \
        git \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-venv \
        python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/onnxruntime

RUN if [ "$TARGETARCH" != "arm64" ]; then \
        exit 0; \
    fi && \
    git clone --branch "$ORT_GIT_REF" --depth 1 https://github.com/microsoft/onnxruntime.git . && \
    python3 -m venv /opt/ort-venv && \
    /opt/ort-venv/bin/pip install --upgrade pip setuptools wheel packaging numpy psutil && \
    pip3 install --break-system-packages numpy && \
    ./build.sh \
        --config Release \
        --update \
        --build \
        --parallel 4 \
        --build_wheel \
        --use_cuda \
        --skip_tests \
        --allow_running_as_root \
        --cmake_generator Ninja \
        --cuda_home /usr/local/cuda \
        --cudnn_home /usr/lib/aarch64-linux-gnu && \
    mkdir -p /opt/ort-wheels && \
    cp build/Linux/Release/dist/*.whl /opt/ort-wheels/

FROM vox-base AS vox-runtime

ARG TARGETARCH
ARG ORT_ARM64_PACKAGE=onnxruntime

# Install ONNX Runtime (GPU on amd64, CPU on arm64) + transformers
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETARCH" = "amd64" ]; then \
        uv pip install --python .venv/bin/python onnxruntime-gpu transformers huggingface-hub; \
    else \
        uv pip install --python .venv/bin/python "$ORT_ARM64_PACKAGE" transformers huggingface-hub; \
    fi && \
    chown -R vox:vox $HOME

# Install bundled adapters into the runtime environment so `vox pull` works out of the box.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETARCH" = "amd64" ]; then \
        uv pip install --python .venv/bin/python \
            ./adapters/vox-kokoro \
            ./adapters/vox-microsoft \
            ./adapters/vox-parakeet \
            ./adapters/vox-qwen \
            ./adapters/vox-voxtral; \
    else \
        uv pip install --python .venv/bin/python \
            ./adapters/vox-kokoro \
            ./adapters/vox-microsoft \
            ./adapters/vox-qwen \
            ./adapters/vox-voxtral && \
        uv pip install --python .venv/bin/python ./adapters/vox-parakeet; \
    fi && \
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

FROM vox-runtime AS vox

ARG TARGETARCH

USER root

COPY --from=vox-arm64-onnx-builder /opt/ort-wheels /tmp/ort-wheels

# Replace the arm64 package runtime with the custom CUDA wheel for the default GPU image.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETARCH" = "arm64" ]; then \
        ort_wheel="$(find /tmp/ort-wheels -maxdepth 1 -name '*.whl' | head -n1)" && \
        test -n "$ort_wheel" && \
        uv pip uninstall --python .venv/bin/python -y onnxruntime && \
        uv pip install --python .venv/bin/python "$ort_wheel"; \
    fi && \
    chown -R vox:vox $HOME

USER vox

FROM vox-runtime AS vox-spark

ARG TARGETARCH
ARG SPARK_ORT_PACKAGE=onnxruntime-gpu
ARG SPARK_ORT_INDEX_URL=
ARG SPARK_ORT_EXTRA_INDEX_URL=
ARG SPARK_ORT_WHEEL=

USER root

COPY --from=vox-arm64-onnx-builder /opt/ort-wheels /tmp/ort-wheels

# Spark-targeted arm64 image: prefer a prebuilt ONNX Runtime source from a wheel
# or custom package index when configured, and fall back to the locally built wheel.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$TARGETARCH" = "arm64" ]; then \
        uv pip uninstall --python .venv/bin/python -y onnxruntime || true; \
        uv pip uninstall --python .venv/bin/python -y onnxruntime-gpu || true; \
        if [ -n "$SPARK_ORT_WHEEL" ]; then \
            uv pip install --python .venv/bin/python "$SPARK_ORT_WHEEL"; \
        elif [ -n "$SPARK_ORT_INDEX_URL" ] || [ -n "$SPARK_ORT_EXTRA_INDEX_URL" ]; then \
            index_args=""; \
            if [ -n "$SPARK_ORT_INDEX_URL" ]; then \
                index_args="$index_args --index-url $SPARK_ORT_INDEX_URL"; \
            fi; \
            if [ -n "$SPARK_ORT_EXTRA_INDEX_URL" ]; then \
                index_args="$index_args --extra-index-url $SPARK_ORT_EXTRA_INDEX_URL"; \
            fi; \
            sh -c "uv pip install --python .venv/bin/python $index_args '$SPARK_ORT_PACKAGE'"; \
        else \
            ort_wheel="$(find /tmp/ort-wheels -maxdepth 1 -name '*.whl' | head -n1)" && \
            test -n "$ort_wheel" && \
            uv pip install --python .venv/bin/python "$ort_wheel"; \
        fi; \
    fi && \
    chown -R vox:vox $HOME

USER vox
