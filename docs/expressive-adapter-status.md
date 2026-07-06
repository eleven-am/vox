# Expressive Adapter Status

This document tracks the current production-readiness state for the expressive
TTS adapters named in the ongoing adapter hardening goal. It is an audit aid,
not a replacement for the smoke validation runbook.

Use [the expressive adapter smoke runbook](expressive-adapter-smoke.md) before
marking any unproven GPU-heavy adapter as production-ready.

## Status Matrix

| Model | Adapter package | Packaging/runtime isolation | Runtime metadata | Smoke status |
| --- | --- | --- | --- | --- |
| `cosyvoice2-tts:0.5b` | `vox-cosyvoice==0.1.5` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/cosyvoice` | Linux x86_64 CUDA/Torch; CPU/ONNX and Spark/ARM NVIDIA not production-supported | Previously cluster-smoked successfully, but slow; retain as known baseline |
| `dia-tts:1.6b` | `vox-dia==0.2.12` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/dia` | Linux x86_64 CUDA/Torch; CPU/ONNX and Spark/ARM NVIDIA not production-supported | Pending isolated GPU smoke |
| `orpheus-tts:medium-3b` | `vox-orpheus==0.1.6` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/orpheus` | Linux x86_64 CUDA/Torch; CPU and Spark/ARM NVIDIA not packaged | Pending isolated GPU smoke |
| `indextts-tts:2` | `vox-indextts==0.1.5` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/indextts` | Linux x86_64 CUDA/Torch; CPU/ONNX and Spark/ARM NVIDIA not production-supported | Pending isolated GPU smoke |

## Evidence Already In The Repo

- Adapter package metadata, README shape, entry points, and runtime isolation
  policy are covered by `tests/test_adapter_package_metadata.py`.
- CosyVoice pull-time preparation is covered by
  `tests/test_cosyvoice_adapter.py`; the test proves `prepare_runtime()` can
  bootstrap the isolated runtime without loading model weights.
- Dia pull-time preparation is covered by `tests/test_dia_adapter.py`; the test
  proves the isolated Transformers runtime can be bootstrapped without loading
  processors or model weights. Dia runtime verification also rejects
  Dia-capable Transformers modules loaded from the Vox app environment instead
  of `$VOX_HOME/runtime/dia`.
- Orpheus stale-runtime repair is covered by `tests/test_orpheus_adapter.py`;
  the tests prove a stale `orpheus_tts` module missing `OrpheusModel` and a
  broken runtime import probe are repaired instead of accepted as valid.
  Orpheus runtime verification also rejects `orpheus_tts` modules loaded from
  outside `$VOX_HOME/runtime/orpheus`.
- IndexTTS stale-runtime repair is covered by `tests/test_indextts_adapter.py`;
  the tests prove a stale `indextts.infer_v2` module missing `IndexTTS2` and a
  broken runtime import probe are repaired instead of accepted as valid.
- Pull-time runtime metadata for these entries is covered in
  `tests/test_model_resolution.py` and `tests/test_registry.py`.
- Pull atomicity across adapter runtime preparation is covered by
  `tests/test_operations_models.py`; the test proves Vox does not save a model
  manifest when `prepare_runtime()` fails after model files have downloaded.
  This prevents a model from appearing in `/v1/models` when its isolated
  runtime is missing or broken.
- The registry repository has a dedicated expressive runtime metadata check in
  `tests/test_registry_metadata.py`.
- The smoke safety boundary and required evidence list are covered by
  `tests/test_expressive_adapter_smoke_docs.py`.

## Current Dia Cluster Finding

The production `vox` deployment was inspected read-only while running
`ghcr.io/eleven-am/vox:v0.2.86`. It had `dia-tts:1.6b` model artifacts and
`vox-dia==0.2.11` installed, but `/home/vox/.vox/runtime/dia` only contained
the Vox fallback `.pth` file. Dia synthesis did not reach model/runtime load:
the scheduler rejected the request because the deployment was started with
`--max-vram 10GiB --vram-headroom 1GiB`, while Dia is budgeted as a 10GB model
plus headroom.

That finding is not a successful smoke test. It is evidence that the current
production deployment is too tightly budgeted for Dia and that older manifests
may exist from before the pull-atomicity fix. A valid Dia pass still requires a
fresh pull and synthesis in the disposable smoke namespace with a VRAM budget
that satisfies the registry metadata.

## Remaining Work

The remaining production-readiness gap is runtime smoke evidence for:

- `dia-tts:1.6b`
- `orpheus-tts:medium-3b`
- `indextts-tts:2`

Do not run these against the production `vox` namespace or `vox-data` PVC.
Use a disposable namespace and PVC as described in
[the smoke runbook](expressive-adapter-smoke.md). The required proof is:

1. `vox pull` succeeds without `VOX_ALLOW_INCOMPATIBLE`.
2. Runtime dependencies are installed under `$VOX_HOME/runtime/<adapter>`.
3. Model files are stored in the model store.
4. Short and long synthesis return non-empty WAV files.
5. Output durations are plausible.
6. Peak RAM and VRAM fit the documented limits.
7. Audio is manually judged usable.
8. Any failure is classified as Vox, adapter, dependency, upstream, or hardware.
