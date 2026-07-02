# Docker images

Vox ships four image variants. All are built from `Dockerfile` except Spark,
which uses `Dockerfile.spark`.

| Tag | Arch | Compute | Torch | Build | Use |
|---|---|---|---|---|---|
| `:latest` / `:vX.Y.Z` | amd64 | CUDA | ✅ | `make build` | NVIDIA GPU (x86), all models |
| `:lean` | amd64 + arm64 | CPU | ❌ | `make build-lean` | Linux CPU + Apple-Silicon Docker; CT2/ONNX models + streaming |
| `:cpu` | amd64 + arm64 | CPU | ✅ | `make build-cpu` | torch models on CPU (slow) |
| `:spark` | arm64 | CUDA | ✅ | `make build-spark` | NVIDIA arm (Jetson/SBSA) |

`:latest` is amd64/CUDA only — CUDA torch wheels are x86-only, so a generic
arm64 CUDA image isn't possible (that's what `:spark` is for). On arm64 hosts
(including Apple-Silicon Docker) use `:lean` or `:cpu`; for arm NVIDIA use
`:spark`.

Models and dynamically installed adapters persist in a Docker volume across
container restarts. No image rebuild is needed to add new models.

## Lean (torch-free) image

The `:lean` image drops the ~2GB torch stack. VAD runs on onnxruntime, so the
streaming/conversation path still works, and `vox pull` refuses torch-based
models up front. Use it if you only serve CTranslate2/ONNX families
(`whisper-stt-ct2`, `kokoro-tts-onnx`, `parakeet-stt-onnx`, `piper-tts-onnx`).

```bash
docker build --build-arg VOX_ACCELERATOR=cpu --build-arg VOX_INCLUDE_TORCH=0 -t vox:lean .
```

In a lean image, `vox pull` checks the environment's actual capabilities (torch,
onnxruntime, CUDA) against what each model needs and fails fast with a clear
message rather than downloading a model that can't load:

- `whisper-stt-ct2`, `kokoro-tts-onnx`, `parakeet-stt-onnx`, `piper-tts-onnx` — pull fine.
- torch models (Qwen, Voxtral, Sesame, Dia, …) — blocked; vLLM models also need a CUDA GPU.

Set `VOX_ALLOW_INCOMPATIBLE=1` to bypass the check and pull anyway. The full
image (default `VOX_INCLUDE_TORCH=1`) supports every model.

## GPU image (generic amd64)

The default GPU image is amd64/CUDA:
- `amd64` uses `onnxruntime-gpu`; the CPU `onnxruntime` is swapped out.

```bash
# Local image
make build-local

# Multi-arch publish build (amd64 only for gpu)
make build
```

## Spark image (arm64 NVIDIA)

For arm64 NVIDIA hardware (Jetson/SBSA) with a vendor-provided torch and ONNX
Runtime, use the dedicated Spark build:

```bash
# Local Spark build
make build-local-spark

# Published Spark build
make build-spark
```

Notes:
- `build-spark` is `linux/arm64` only.
- By default, `Dockerfile.spark` uses:
  - `nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04`
  - `torch==2.8.0` / `torchaudio==2.8.0` (the `make build-spark` path pulls them
    from `https://download.pytorch.org/whl/cu129` via `SPARK_TORCH_INDEX_URL`)
  - `onnxruntime==1.23.0` (`SPARK_ORT_PACKAGE`)
- `Dockerfile.spark` refuses to publish if it would install a CPU-only `torch`
  build. Provide a CUDA-capable torch source with either:
  - `SPARK_TORCH_WHEEL` and `SPARK_TORCHAUDIO_WHEEL`, or
  - `SPARK_TORCH_INDEX_URL` / `SPARK_TORCH_EXTRA_INDEX_URL`
- Override the ONNX Runtime source with:
  - `SPARK_ORT_WHEEL=/path/or/url/to/wheel`, or
  - `SPARK_ORT_INDEX_URL` / `SPARK_ORT_EXTRA_INDEX_URL`
- The generic `make build` path is unchanged and still produces the amd64 image.

## CI

A tag push (`v*`) triggers `publish-ghcr.yml`, which builds and pushes all four
tags on native runners (amd64 on `ubuntu-24.04`, arm64 on `ubuntu-24.04-arm`).
`docker-build.yml` builds every variant without pushing on pull requests and
main pushes that touch Docker-relevant files.
