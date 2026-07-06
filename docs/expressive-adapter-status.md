# Expressive Adapter Status

This document tracks the current production-readiness state for the expressive
TTS adapters named in the ongoing adapter hardening goal. It is an audit aid,
not a replacement for the smoke validation runbook.

Use [the expressive adapter smoke runbook](expressive-adapter-smoke.md) before
marking any unproven GPU-heavy adapter as production-ready.

## Status Matrix

| Model | Adapter package | Packaging/runtime isolation | Runtime metadata | Smoke status |
| --- | --- | --- | --- | --- |
| `cosyvoice2-tts:0.5b` | `vox-cosyvoice==0.1.4` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/cosyvoice` | Linux x86_64 CUDA/Torch; CPU/ONNX and Spark/ARM NVIDIA not production-supported | Previously cluster-smoked successfully, but slow; retain as known baseline |
| `dia-tts:1.6b` | `vox-dia==0.2.11` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/dia` | Linux x86_64 CUDA/Torch; CPU/ONNX and Spark/ARM NVIDIA not production-supported | Pending isolated GPU smoke |
| `orpheus-tts:medium-3b` | `vox-orpheus==0.1.3` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/orpheus` | Linux x86_64 CUDA/Torch; CPU and Spark/ARM NVIDIA not packaged | Pending isolated GPU smoke |
| `indextts-tts:2` | `vox-indextts==0.1.3` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/indextts` | Linux x86_64 CUDA/Torch; CPU/ONNX and Spark/ARM NVIDIA not production-supported | Pending isolated GPU smoke |

## Evidence Already In The Repo

- Adapter package metadata, README shape, entry points, and runtime isolation
  policy are covered by `tests/test_adapter_package_metadata.py`.
- CosyVoice, Dia, Orpheus, and IndexTTS adapter behavior and runtime bootstrap
  paths are covered by their adapter-specific tests.
- Pull-time runtime metadata for these entries is covered in
  `tests/test_model_resolution.py` and `tests/test_registry.py`.
- The registry repository has a dedicated expressive runtime metadata check in
  `tests/test_registry_metadata.py`.
- The smoke safety boundary and required evidence list are covered by
  `tests/test_expressive_adapter_smoke_docs.py`.

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
