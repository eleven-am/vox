# Expressive Adapter Status

This document tracks the current production-readiness state for the expressive
TTS adapters named in the ongoing adapter hardening goal. It is an audit aid,
not a replacement for the smoke validation runbook.

Use [the expressive adapter smoke runbook](expressive-adapter-smoke.md) before
marking any unproven GPU-heavy adapter as production-ready.

## Status Matrix

| Model | Adapter package | Packaging/runtime isolation | Runtime metadata | Smoke status |
| --- | --- | --- | --- | --- |
| `cosyvoice2-tts:0.5b` | `vox-cosyvoice==0.1.6` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/cosyvoice` | Linux x86_64 CUDA/Torch; registry requires `min_vram_gb=8`; CPU/ONNX and Spark/ARM NVIDIA not production-supported | Previously cluster-smoked successfully, but slow; retain as known baseline |
| `dia-tts:1.6b` | `vox-dia==0.2.13` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/dia`; wires Dia audio-prompt voice cloning when Vox supplies both `reference_audio` and `reference_text` | Linux x86_64 CUDA/Torch; registry requires `min_vram_gb=12`; CPU/ONNX and Spark/ARM NVIDIA not production-supported | Pending GPU smoke |
| `orpheus-tts:medium-3b` | `vox-orpheus==0.1.7` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/orpheus`; exposes Orpheus generation controls and validates preset voices before synthesis | Linux x86_64 CUDA/Torch; registry requires `min_vram_gb=10`; CPU and Spark/ARM NVIDIA not packaged | Pending isolated GPU smoke |
| `indextts-tts:2` | `vox-indextts==0.1.19` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/indextts`; keeps process NumPy stable with the TensorBoard `np.bool8` compatibility alias; installs NumPy-2-compatible runtime wheels; patches upstream `torchaudio.save` to write file outputs through soundfile; purges sibling-runtime TensorBoard/Transformers modules, stale Torch/CUDA runtime packages, and stale NumPy/Matplotlib artifacts before import probes; selects constructor signatures without swallowing internal model-load failures; exposes IndexTTS2 emotion text/vector/audio-prompt controls through Vox synthesis params | Linux x86_64 CUDA/Torch; registry requires `min_vram_gb=10`; CPU/ONNX and Spark/ARM NVIDIA not production-supported | Existing `vox` namespace smoke passed with Samantha voice: cold load/generate ~48s, warm generate ~6.8s for ~7.9s audio |

## Evidence Already In The Repo

- Adapter package metadata, README shape, entry points, and runtime isolation
  policy are covered by `tests/test_adapter_package_metadata.py`.
- CosyVoice pull-time preparation is covered by
  `tests/test_cosyvoice_adapter.py`; the test proves `prepare_runtime()` can
  bootstrap the isolated runtime without loading model weights. CosyVoice
  runtime verification also rejects `cosyvoice.cli.cosyvoice` modules loaded
  from outside `$VOX_HOME/runtime/cosyvoice`.
- Dia pull-time preparation is covered by `tests/test_dia_adapter.py`; the test
  proves the isolated Transformers runtime can be bootstrapped without loading
  processors or model weights. Dia runtime verification also rejects
  Dia-capable Transformers modules loaded from the Vox app environment instead
  of `$VOX_HOME/runtime/dia`. Dia clone-path tests prove incomplete clone
  requests are rejected and complete reference-audio/reference-text requests use
  Dia's `audio_prompt_len` decode path.
- Orpheus stale-runtime repair is covered by `tests/test_orpheus_adapter.py`;
  the tests prove a stale `orpheus_tts` module missing `OrpheusModel` and a
  broken runtime import probe are repaired instead of accepted as valid.
  Orpheus runtime verification also rejects `orpheus_tts` modules loaded from
  outside `$VOX_HOME/runtime/orpheus`.
- IndexTTS stale-runtime repair is covered by `tests/test_indextts_adapter.py`;
  the tests prove a stale `indextts.infer_v2` module missing `IndexTTS2` and a
  broken runtime import probe are repaired instead of accepted as valid.
  IndexTTS runtime verification also rejects `indextts.infer_v2` modules loaded
  from outside `$VOX_HOME/runtime/indextts`.
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
plus headroom. The remote registry now records this constraint directly as
`min_vram_gb=12`.

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

Existing-server smoke is available for the currently running Vox endpoint via
`scripts/expressive-adapter-served-smoke.py`. That path is the default for
checking the existing Vox service. It records short/long synthesis, timings,
WAV metadata, SHA-256 digests, silence checks, the requested model detail from
`GET /v1/models/{model}`, and `/v1/models/loaded` before and after synthesis
without creating namespaces, PVCs, or running `vox pull`.
Use `--inspect-only` when the goal is read-only inspection; it skips
`/v1/audio/speech` and records no synthesis cases. Full served smoke can still
change in-memory loaded model state and VRAM while a request is in flight, so
heavy synthesis against production should be intentional.

Existing-server smoke is not sufficient to mark a model production-ready because
it cannot prove a clean model pull or clean runtime install.

Full production-readiness still requires clean-pull smoke in an explicitly
approved test environment. Do not create a new namespace or PVC just because a
model needs testing. The required proof is:

1. `vox pull` succeeds without `VOX_ALLOW_INCOMPATIBLE`.
2. The pre-pull clean-state probe proves the target manifest, model link, and
   adapter runtime were not already present.
3. Runtime dependencies are installed under `$VOX_HOME/runtime/<adapter>`.
4. Model files are stored in the model store and storage usage is recorded.
5. Adapter package, runtime, manifest, and blob storage usage is recorded.
6. Short and long synthesis return non-empty WAV files.
7. Output durations are plausible.
8. Audio stream metadata has a valid codec, sample rate, and channel count.
9. Audio signal stats prove the output is not silent.
10. Peak RAM and VRAM fit the documented limits.
11. Audio is manually judged usable.
12. Any failure is classified as Vox, adapter, dependency, upstream, or hardware.
