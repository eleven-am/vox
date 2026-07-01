# Vox Adapter Package Template

This template defines the standard shape for adapter packages in `adapters/`.
It extends the boundary in [the adapter contract](adapter-contract.md) and the
runtime install rules in
[the adapter runtime dependency policy](adapter-runtime-dependency-policy.md).

The goal is that a new adapter package is predictable: Vox discovers it through
entry points, imports it lightly, installs heavyweight backend dependencies in
`$VOX_HOME/runtime/<runtime-name>` when needed, and keeps model artifacts in
model storage.

## Directory Layout

Use this layout for new packages:

```text
adapters/vox-example/
  README.md
  pyproject.toml
  src/vox_example/
    __init__.py
    adapter.py
  tests live in ../../tests/test_example_adapter.py
```

Split `adapter.py` into smaller modules when an adapter has multiple backends,
worker subprocesses, or runtime bootstrap logic. Keep package import side
effects small enough that adapter discovery can run without model backends
already installed.

## `pyproject.toml`

Every adapter package must include:

```toml
[project]
name = "vox-example"
version = "0.1.0"
description = "Example STT/TTS adapter for Vox"
readme = { file = "README.md", content-type = "text/markdown" }
requires-python = ">=3.11"
dependencies = [
    "vox-runtime>=0.2.2",
    "numpy>=1.26.0,<2.4",
]

[tool.vox.adapter]
import-package = "vox_example"
runtime-policy = "target-runtime"
runtime-names = ["example"]
adapter-types = ["tts"]

[project.entry-points."vox.adapters"]
example-tts-torch = "vox_example.adapter:ExampleTTSAdapter"
example = "vox_example.adapter:ExampleTTSAdapter"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vox_example"]
```

Allowed `runtime-policy` values:

- `target-runtime`: heavyweight backend dependencies are installed into one or
  more `$VOX_HOME/runtime/<runtime-name>` target directories.
- `package-runtime`: backend dependencies are currently installed with the
  adapter package because they are small enough, stable enough, or not yet
  separated.
- `mixed`: some backends use package dependencies and some use target runtime
  directories.
- `venv-exception`: at least one backend deliberately uses a full venv under
  `$VOX_HOME/runtime/<runtime-name>`.

`runtime-names` must list every runtime directory owned by the adapter. Use an
empty list only for `package-runtime` adapters.

`adapter-types` must contain `stt`, `tts`, or both.

If `runtime-policy = "venv-exception"`, add:

```toml
venv-exceptions = ["example-tts"]
```

and document why a target directory is not enough.

## README Sections

Every adapter README must include:

- a title matching the package name
- included adapter entry point names
- `## Install`
- `## Runtime Dependencies`
- `## Use with Vox`

The runtime section must say whether backend dependencies live in the adapter
package, in `$VOX_HOME/runtime/<runtime-name>`, or in a deliberate venv
exception.

## Runtime Bootstrap Pattern

Use `vox.core.adapter_runtime` for target-directory runtimes:

```python
from vox.core.adapter_runtime import (
    activate_runtime_path,
    install_target_runtime_requirements,
    runtime_root,
)


def _runtime_root():
    return runtime_root() / "example"


def _ensure_runtime_path() -> str:
    runtime_dir = _runtime_root()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return activate_runtime_path(runtime_dir, root=runtime_dir.parent)
```

Install heavyweight dependencies only from load/bootstrap paths, not at module
import time. Verify imports, package paths, versions, or backend symbols after
installing. A successful installer exit code is not sufficient.

## Tests

Every adapter should have adversarial tests for the parts it owns:

- package metadata and README shape through `tests/test_adapter_package_metadata.py`
- adapter discovery entry point shape
- import safety without heavyweight backend dependencies installed
- runtime install command shape
- stale or broken runtime repair behavior
- backend capability metadata
- error clarity when required runtime packages cannot be installed

Prefer targeted adapter tests for backend-specific behavior and shared helper
tests in `tests/test_adapter_runtime.py` for behavior in
`vox.core.adapter_runtime`.

## Current Adapter Runtime Matrix

| Package | Runtime policy | Runtime names | Notes |
| --- | --- | --- | --- |
| `vox-chatterbox` | `target-runtime` | `chatterbox` | Chatterbox backend installs into target runtime. |
| `vox-cosyvoice` | `target-runtime` | `cosyvoice` | CosyVoice backend installs into target runtime. |
| `vox-dia` | `target-runtime` | `dia` | Isolated Transformers/Dia runtime. |
| `vox-indextts` | `target-runtime` | `indextts` | IndexTTS backend installs into target runtime. |
| `vox-kokoro` | `package-runtime` | none | Backend packages currently install with the adapter package. |
| `vox-microsoft` | `mixed` | `vibevoice` | SpeechT5 uses package deps; VibeVoice uses a target runtime. |
| `vox-openvoice` | `target-runtime` | `openvoice` | Upstream OpenVoice repo installs into target runtime. |
| `vox-orpheus` | `target-runtime` | `orpheus` | Orpheus/vLLM backend installs into target runtime. |
| `vox-parakeet` | `mixed` | `parakeet-nemo` | ONNX backend uses package deps; NeMo uses target runtime. |
| `vox-piper` | `target-runtime` | `piper` | Piper backend installs into target runtime. |
| `vox-qwen` | `target-runtime` | `qwen-asr`, `qwen-tts` | ASR/TTS use separate target runtimes. |
| `vox-sesame` | `target-runtime` | `sesame` | Transformers CSM runtime installs into target runtime. |
| `vox-voxtral` | `venv-exception` | `voxtral-stt`, `voxtral-tts` | TTS uses a deliberate venv exception. |
| `vox-whisper` | `target-runtime` | `whisper` | faster-whisper/CTranslate2 target runtime. |
| `vox-xtts` | `target-runtime` | `xtts` | Coqui XTTS target runtime. |

## Review Checklist

Before merging a new adapter package:

1. The base Vox image does not install or copy the adapter package.
2. Model artifacts stay in model storage.
3. Heavy backend dependencies are either in `$VOX_HOME/runtime/<runtime-name>`
   or documented as package-runtime/venv exceptions.
4. The adapter package declares `[tool.vox.adapter]` metadata.
5. The README explains runtime dependency placement.
6. Tests cover metadata, import safety, runtime bootstrap, and failure cases.
