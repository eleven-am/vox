# Vox Adapter Contract

This document defines the packaging boundary for Vox, adapter packages, adapter
runtime dependencies, and Docker images. The goal is to keep the base Vox
runtime small, generic, and model-agnostic while still allowing each model
family to bring the dependencies it needs.

For dependency pinning, `--upgrade`, verification, and repair rules inside
`$VOX_HOME/runtime/<runtime-name>`, see
[the adapter runtime dependency policy](adapter-runtime-dependency-policy.md).

## Terms

- **`vox-runtime`**: the core Vox package published as `vox-runtime`.
- **Adapter package**: a separately published Python package such as
  `vox-kokoro`, `vox-parakeet`, or `vox-qwen`.
- **Adapter package install directory**:
  `$VOX_HOME/adapters/<adapter-package>`.
- **Adapter runtime directory**:
  `$VOX_HOME/runtime/<runtime-name>`.
- **Model artifact**: model weights, manifests, tokenizer assets, voice bundles,
  and other downloaded model data.

## What Belongs In `vox-runtime`

`vox-runtime` owns the model-agnostic runtime surface:

- CLI commands and server startup.
- HTTP, WebSocket, PondSocket, and gRPC transports.
- OpenAI-compatible route shapes.
- Core adapter interfaces: `STTAdapter`, `TTSAdapter`, common event and result
  types, scheduler integration, model store, registry resolution, voice storage,
  audio codecs, resampling, and conversation orchestration.
- Generic infrastructure for installing and isolating adapter packages.
- Generic infrastructure for installing and isolating adapter runtime
  dependencies, including `vox.core.adapter_runtime`.
- Common platform dependencies needed by Vox itself, such as FastAPI, gRPC,
  WebRTC transport support, audio decoding/encoding helpers, and generic
  numerical/audio libraries.
- Small compatibility helpers that are not specific to one model family.

`vox-runtime` must not need to know how to load one specific model family. It
may know that an adapter package exists through registry metadata, but the
adapter owns the model-specific implementation.

## What Belongs In Each Adapter Package

An adapter package owns one model family or one closely related backend family.
It must include:

- One or more `vox.adapters` entry points.
- Adapter classes implementing `STTAdapter` or `TTSAdapter`.
- Model-family load, unload, inference, voice, timestamp, and language handling.
- Small metadata needed to expose capabilities through `AdapterInfo`.
- Lightweight Python dependencies needed to import the adapter package itself.
- Runtime bootstrap code for heavyweight or conflict-prone dependencies, using
  `vox.core.adapter_runtime` where possible.
- Adapter-specific compatibility patches that cannot reasonably live in the
  generic runtime.
- Adapter tests proving import safety, load behavior, runtime bootstrap behavior,
  and expected adapter capabilities.

Adapter packages should be import-light. Importing `vox-kokoro` or `vox-qwen`
should not require the full model backend to already be installed. Heavy
runtime imports should happen inside `load()` or a clearly isolated runtime
loader path.

Adapter package dependencies may include generic libraries that are needed for
the package to import and expose metadata. Heavy model-family dependencies
should not be placed in the package dependency list unless they are small,
stable, and not likely to conflict with other adapters.

## What Belongs In `$VOX_HOME/adapters/<adapter-package>`

`$VOX_HOME/adapters/<adapter-package>` is where Vox installs adapter packages
on demand.

This directory contains the adapter package wheel contents and package metadata:

- adapter Python modules
- `.dist-info` metadata
- entry point metadata
- small package assets

This directory should not be used for large model weights. It should also not
be used as the general location for heavyweight backend dependency stacks once
those dependencies need isolation from other adapters.

Vox activates one adapter package install directory only while loading that
adapter's entry point. Other adapter install directories are kept off
`sys.path` so their dependencies do not shadow each other.

## What Belongs In `$VOX_HOME/runtime/<runtime-name>`

`$VOX_HOME/runtime/<runtime-name>` is for heavyweight, optional, or
conflict-prone runtime dependencies used by an adapter backend.

Examples:

- a specific `transformers` build needed by Dia
- `nemo-toolkit[asr]` for Parakeet NeMo
- `faster-whisper` and CTranslate2 for Whisper
- Qwen ASR/TTS runtime packages
- Kokoro Torch runtime support packages
- VibeVoice runtime packages

Runtime directories are adapter-controlled and may be repaired or replaced by
the adapter. They are not part of the base Vox install. They should be safe to
delete and recreate if dependency resolution needs to be repaired.

Adapters should use `vox.core.adapter_runtime` for target-runtime installs when
the backend can run from a `--target` directory. If a backend needs a stronger
isolation boundary, such as a full virtual environment, the adapter may use a
custom runtime layout, but that exception should be deliberate and tested.

When activating one runtime directory, Vox should keep sibling runtime
directories off `sys.path` to avoid dependency leakage between adapters.

## What Belongs In Model Storage

Model artifacts belong in Vox model storage, not in the adapter package or the
base image.

This includes:

- model weights
- tokenizer and processor assets
- manifests and layer blobs
- downloaded reference voice files
- custom voice bundles

The registry tells Vox which adapter package to install and which model
artifacts to pull. The adapter package tells Vox how to load and run those
artifacts.

## What Must Never Be Bundled In The Base Vox Image

The base Vox image must remain model-agnostic. It must not bundle:

- model weights
- pre-pulled model artifacts
- adapter packages such as `vox-kokoro`, `vox-parakeet`, `vox-qwen`, etc.
- model-family runtime packages such as `kokoro-onnx`, `onnx-asr`,
  `faster-whisper`, `nemo-toolkit`, `qwen-tts`, VibeVoice runtime packages, or
  Dia-specific Transformers snapshots
- adapter-specific source trees copied into the image as active adapter packages

The base image may include generic system and Python dependencies needed for Vox
to run and to install packages at runtime:

- Python
- `uv` and `pip`
- compiler and system libraries needed to install adapter packages
- `ffmpeg`, `sox`, and generic audio shared libraries
- generic GPU/accelerator runtime libraries where the image variant is
  explicitly for that platform
- generic PyTorch/ONNX runtime support only when the image variant is explicitly
  defined as a compute-runtime image, not as a bundled model image

If a deployment wants to pre-warm adapters or runtime dependencies for faster
startup, it should do so in a persistent volume or a deliberately named
deployment-specific derivative image. That derivative image is not the generic
Vox base image and should not be treated as the default publish target.

## Current Compliance Notes

The target architecture is:

```text
vox-runtime
  core runtime, APIs, transports, scheduler, generic install/isolation helpers

$VOX_HOME/adapters/<adapter-package>
  adapter package wheel installed on demand

$VOX_HOME/runtime/<runtime-name>
  heavyweight backend runtime dependencies installed on demand

model store
  model artifacts and voice data pulled on demand
```

At the time this contract was written, some Docker packaging still needs a
follow-up compliance pass:

- the default Dockerfile still installs some model-family packages
- the Spark Dockerfile still copies adapter source trees into the image

Those are compatibility leftovers, not the desired long-term contract.

## Rules For New Adapters

New adapters should follow these rules:

1. Publish a separate adapter package.
2. Depend on `vox-runtime`, not the other way around.
3. Keep imports light so adapter discovery does not require model backends.
4. Put heavyweight backend dependencies in `$VOX_HOME/runtime/<runtime-name>`.
5. Put model weights and model assets in model storage.
6. Do not require changes to the base Vox image for a new model family.
7. Add tests for package metadata, import safety, runtime bootstrap, and adapter
   capability metadata.
