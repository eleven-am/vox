# Vox Hardware-Aware Model Resolution (Design)

Status: proposal, not yet implemented. This is the authoritative spec merging the
initial variant-resolution design with the Codex review. Build from this.

## Goal

`vox pull` should understand the real runtime environment, explain whether a
model can run, choose the best backend when several viable paths exist, and fail
clearly when the machine cannot support the requested model. Compatibility must
stop being a loose `format == pytorch` check.

Direct runtime probing is authoritative. Environment variables are explicit
override and debug escape hatches only, never the default source of truth.

## Current ground truth

- `src/vox/core/registry.py`: the remote registry at
  `https://raw.githubusercontent.com/eleven-am/vox-registry/main` is the single
  source of truth. Fetched entries are cached in-memory for the process.
  Registry entries may be concrete backends or logical models with `variants`.
- `src/vox/operations/models.py::pull_model`: resolves the alias, looks up a
  catalog entry, runs the compatibility gate, installs the adapter package,
  downloads HF files, writes a manifest. The gate runs before the adapter is
  installed or imported, so pull-time compatibility must come from core Vox and
  catalog metadata, not adapter code, unless we deliberately change that order.
- `src/vox/core/capabilities.py`: current gate. Infers runtime needs from
  `format` and adapter naming. Lets `VOX_HAS_TORCH` / `VOX_HAS_ONNXRUNTIME`
  silently override reality (to be fixed).
- `src/vox/core/runtime.py`: `detect_runtime_capabilities()` already probes
  `system`, `machine`, `torch_cuda`, `onnx_cuda`, `onnx_coreml`, `mps`, and
  `nvidia_device`. It does not yet probe versions, device count, compute
  capability, VRAM, RAM, or driver CUDA version.
- `src/vox/core/adapter.py`: adapter metadata is only available after the adapter
  package is imported, which is after pull-time gating.
- `src/vox/core/store.py`: manifests persist the concrete pulled model config.

## Design principles

Model three separate questions rather than collapsing them:

1. Can this machine pull this model without wasting disk and network?
2. Can this machine run this model at all?
3. Which backend should this model use on this machine?

There are two distinct selection mechanisms, and the complete design needs both.
They layer; they are not alternatives.

- Pull-time variant selection: the download itself differs by hardware (different
  weights, format, or adapter package). Example: Kokoro ONNX on CPU vs Kokoro
  Torch on CUDA. Decided at pull time.
- Load-time backend selection: the download is the same, but a faster runtime may
  or may not be available. Example: faster-qwen3-tts (CUDA graphs) vs the standard
  qwen backend, both loading the same weights. Decided by the adapter at load
  time, re-checked every load because the container can change after pull.

## Runtime snapshot

Extend `src/vox/core/runtime.py` into one richer, directly probed snapshot:

```
system: linux | darwin | windows
machine: x86_64 | arm64 | aarch64 | ...
python_version
torch_installed, torch_version
torch_cuda_available, torch_cuda_version
torch_device_count, torch_device_names
torch_compute_capability      # e.g. 80 for sm_80, UNKNOWN if undetectable
torch_mps_available
onnxruntime_installed, onnxruntime_version, onnxruntime_providers
vram_gb                       # UNKNOWN if undetectable
ram_gb
nvidia_device_visible, nvidia_smi_available
nvidia_driver_version, driver_cuda_version
```

Detection is layered and best effort: torch when present, then `nvidia-smi` /
`pynvml`, then `UNKNOWN`. RAM via `psutil` or `/proc`. `UNKNOWN` is a first-class
value. The lean torch-free image genuinely cannot always report VRAM or compute
capability, and the resolver must handle that.

Env policy: probing is authoritative. `VOX_HAS_TORCH` / `VOX_HAS_ONNXRUNTIME`
apply only when `VOX_RUNTIME_OVERRIDE=1` is set. `VOX_ALLOW_INCOMPATIBLE=1` still
bypasses hard pull blocking, and the warning names exactly what is missing.

## Requirement model

```python
@dataclass(frozen=True)
class RuntimeRequirement:
    python_modules: tuple[str, ...] = ()
    min_versions: dict[str, str] = field(default_factory=dict)   # {"torch": "2.5.1"}
    accelerators: tuple[str, ...] = ()   # "cuda" | "mps" | "onnx_cuda" | "cpu"
    systems: tuple[str, ...] = ()        # "linux" | "darwin" | "windows"
    machines: tuple[str, ...] = ()       # "x86_64" | "arm64" | "aarch64"
    min_compute_capability: int | None = None   # 80 = sm_80
    min_cuda_version: str | None = None
    min_vram_gb: float | None = None
    min_ram_gb: float | None = None
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CapabilityCheck:
    runnable: bool
    preferred_backend: str | None
    missing: tuple[str, ...]
    warnings: tuple[str, ...]
```

Matching semantics:

- `python_modules`: all present.
- `min_versions`: each installed version at or above the minimum.
- `accelerators`: at least one required accelerator present (note `cuda` for
  torch CUDA is distinct from `onnx_cuda` for the ONNX CUDA provider).
- `systems` / `machines`: membership.
- ordered fields (`min_compute_capability`, `min_cuda_version`, `min_vram_gb`,
  `min_ram_gb`): detected value at or above the minimum.
- `UNKNOWN` fails an ordered constraint by default, with the reason logged, so the
  resolver falls back to a safer variant instead of guessing. A forced
  `variant` request or `VOX_ALLOW_INCOMPATIBLE=1` can force past it.

## Catalog schema

Keep existing concrete entries working. Add optional logical entries with
`variants` (pull-time), and let a variant declare `backends` (load-time).

Pull-time variants (Kokoro, different download per hardware):

```json
{
  "name": "kokoro-tts",
  "type": "tts",
  "variants": [
    { "id": "cuda", "priority": 100,
      "requires": { "accelerators": ["cuda"], "python_modules": ["torch"], "min_compute_capability": 70 },
      "adapter": "kokoro-tts-torch", "adapter_package": "vox-kokoro",
      "format": "pytorch", "source": "hexgrad/Kokoro-82M" },
    { "id": "mps", "priority": 90,
      "requires": { "systems": ["darwin"], "machines": ["arm64"] },
      "adapter": "kokoro-tts-mlx", "adapter_package": "vox-kokoro",
      "format": "mlx", "source": "mlx-community/Kokoro-82M-bf16" },
    { "id": "cpu", "priority": 0, "fallback": true,
      "requires": { "python_modules": ["onnxruntime"] },
      "adapter": "kokoro-tts-onnx", "adapter_package": "vox-kokoro",
      "format": "onnx", "source": "onnx-community/Kokoro-82M-ONNX" }
  ]
}
```

Load-time backends inside a variant (Qwen, same download, faster runtime when
possible). Public name stays `qwen3-tts`:

```json
{
  "name": "qwen3-tts",
  "type": "tts",
  "variants": [
    { "id": "torch", "priority": 0, "fallback": true,
      "requires": { "python_modules": ["torch"] },
      "adapter": "qwen3-tts-torch", "adapter_package": "vox-qwen",
      "format": "pytorch", "source": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
      "parameters": { "sample_rate": 24000, "default_voice": "Ryan" },
      "backends": {
        "preferred": [
          { "name": "faster-qwen3-tts",
            "requires": { "python_modules": ["torch", "faster_qwen3_tts"],
                          "min_versions": {"torch": "2.5.1"}, "accelerators": ["cuda"] },
            "reason": "CUDA-graph streaming generation" }
        ],
        "fallback": { "name": "qwen-tts", "requires": { "python_modules": ["torch"] } }
      }
    }
  ]
}
```

A concrete entry today (for example `kokoro-tts-onnx`) is treated internally as a
single-variant logical model, so backward compatibility is automatic.

## Resolver and naming

Grammar: `<name>[:<tag>]`, with the optional forced variant carried separately
as the CLI `--variant` flag or the HTTP/gRPC `variant` field.

- `vox pull kokoro-tts` auto-resolves the pull-time variant.
- `vox pull kokoro-tts --variant torch` forces a variant.
- `POST /v1/models/pull` and gRPC `PullRequest` pass the same value as
  `variant`.
- `vox pull kokoro-tts-onnx:v1.0` pulls the existing concrete backend directly.

Pull-time resolver:

```
resolve(entry, snapshot, forced=None):
  if forced: pick that variant, then run the capability gate on it
  candidates = [v for v in entry.variants if all v.requires match snapshot]
  if none: raise ModelIncompatibleError naming what each variant required
  return max(candidates, key=(priority, declaration_order))
```

The capability gate becomes the fallback path, not the primary mechanism. On a
torch-free box `vox pull kokoro-tts` simply resolves to the ONNX variant instead
of failing.

## Pull behavior

1. Resolve alias to a concrete catalog entry.
2. Build the runtime snapshot from direct probes.
3. Evaluate catalog runtime requirements (variant resolution).
4. Hard requirement missing: raise `ModelIncompatibleError` with exact missing
   facts and mention `VOX_ALLOW_INCOMPATIBLE=1`.
5. Only a preferred backend missing: allow the pull, emit a warning `PullEvent`,
   record which backend will likely be used.
6. Install adapter package.
7. Download model files.
8. Run adapter `prepare_runtime()` when available.
9. Write the manifest only after runtime preparation succeeds.

Do not install or import adapter packages before the gate. Pull-time
compatibility stays catalog-driven.

## Manifest

Keep the concrete source and adapter fields. Also record the pull-time runtime
evaluation as clearly diagnostic, non-authoritative metadata:

```json
"runtime": {
  "checked_at_pull": true,
  "resolved_variant": "cuda",
  "preferred_backend": "faster-qwen3-tts",
  "detected": { "system": "linux", "machine": "x86_64",
                "torch_installed": true, "torch_cuda_available": true, "torch_version": "2.8.0" },
  "warnings": []
}
```

Load time re-checks reality because the container or environment may change after
pull.

## Load-time behavior

Adapters own final backend selection and re-probe on every load. For `vox-qwen`:

1. On load, inspect the actual runtime again.
2. If CUDA, a compatible torch (>= 2.5.1), and `faster_qwen3_tts` are present, use
   `FasterQwen3TTS` with true streaming.
3. Otherwise use the current `qwen_tts` backend.
4. If neither works, raise a clear load error.

Both checks are needed: pull-time avoids bad downloads, load-time prevents stale
manifests or changed containers from lying.

## faster-qwen3-tts integration

Do not create a new public adapter package. Put the fast backend inside
`vox-qwen`. Public API is unchanged (`qwen3-tts:0.6b`, `:1.7b`,
`:0.6b-clone`, `:1.7b-clone`). Backend selection is internal:

- `faster-qwen3-tts` backend: requires torch, torch >= 2.5.1, CUDA; uses
  `FasterQwen3TTS` streaming APIs. MIT licensed, pip-installable.
- `qwen-tts` fallback backend: requires torch, runs on CPU/MPS, slower, existing
  behavior.

Do not require CUDA for all `qwen3-tts` pulls. Report that fast mode is
unavailable and fall back. (Confirm the current `qwen-tts` backend runs on CPU
before finalizing.)

## Adapter contract

Do not overload `BaseAdapter.info()` as the pull-time requirement source; it is
only importable after install. A future improvement can add optional declarative
runtime metadata to `[tool.vox.adapter]` in the adapter `pyproject.toml`, but core
pull behavior stays catalog-driven unless the pull order changes.

## CLI diagnostics (follow-up phase)

Build the runtime snapshot object now (pull needs it). Ship a thin command later:

```
vox doctor                     # prints the runtime snapshot
vox doctor qwen3-tts:0.6b      # explains runnable / preferred / fallback and why
```

## Backward compatibility

Concrete names keep working untouched and pull that exact backend. Logical entries
are additive. Existing manifests are never broken; the new `runtime` manifest
block is optional and diagnostic.

## Phases

1. Runtime snapshot + layered detection (+ env override policy). Tests.
2. `RuntimeRequirement` / `CapabilityCheck` + matching + pull-time resolver.
   Pure, unit-tested.
3. Registry: logical entries with `variants` and `backends`, forced `variant`
   pass-through, concrete-name compatibility.
4. `pull_model` integration: resolve, gate, warn, download, record variant and
   diagnostic runtime in the manifest. CLI `--variant`, HTTP and gRPC
   pass-through.
5. Adapter load-time backend selection (start with `vox-qwen` + faster-qwen3-tts;
   Kokoro variant wiring).
6. `vox-registry` schema + JSON Schema + CI validation + example logical models
   (`kokoro-tts`, `qwen3-tts`), plus `vox doctor` and docs.

Phases 1 to 4 land in this repo before any registry-repo change.

## Testing

- Runtime snapshot construction with mocked modules.
- Env overrides do not apply unless `VOX_RUNTIME_OVERRIDE=1`.
- `pytorch` model blocks when torch is truly missing.
- `onnx` model blocks when onnxruntime is missing.
- vLLM model blocks without CUDA.
- Qwen pull succeeds with a warning when torch exists but CUDA is missing.
- Qwen pull reports the faster backend when CUDA and torch version are available.
- Kokoro pull resolves to the ONNX variant on a torch-free box and the Torch
  variant on CUDA.
- `min_versions` blocks a preferred backend when torch is present but too old.
- `UNKNOWN` VRAM/compute fails a gated variant and falls back.
- Manifest records the diagnostic runtime metadata.
- Load-time `vox-qwen` chooses the faster backend only when available and falls
  back to `qwen_tts` otherwise.
- `VOX_ALLOW_INCOMPATIBLE=1` bypasses hard pull blocking but preserves the warning.

## Non-goals

- Move heavyweight backend dependencies into the base Vox image.
- Bundle adapter runtimes in Docker images.
- Make env vars the default truth source.
- Import every adapter package just to run `vox pull`.
- Create a new public Qwen model name.
- Break existing pulled manifests.
- Cross-target resolution (resolving for a machine other than the local one).

## Resolved decisions

- Pull UX: logical auto-resolve, `--variant` override, concrete
  names still work.
- Requirements: full set from day one, including compute capability, VRAM, RAM,
  plus `min_versions`.
- Scope: resolve for the local machine only.
- Env vars: probing authoritative; overrides only under `VOX_RUNTIME_OVERRIDE=1`.
- Manifest: store resolved variant and preferred backend as diagnostic only;
  load-time is authoritative.
- Qwen on CPU/MPS: allowed as a slow fallback, report fast mode unavailable.
- `vox doctor`: follow-up phase; build the snapshot object now.
- Multi-backend mechanism: general now, populate Qwen and Kokoro first.

## Open for final review

- Best cross-platform way to detect `compute_capability`, `vram_gb`,
  `driver_cuda_version` without a hard torch dependency (pynvml optional dep vs
  parsing `nvidia-smi`).
- Additional dimensions worth modeling: ROCm/AMD, Vulkan, Intel XPU, AVX-512,
  disk space, multi-GPU, minimum driver version.
- Whether "torch present but `cuda.is_available()` is false" should collapse to
  `accelerator: cpu` or surface as a distinct driver-mismatch diagnostic.
