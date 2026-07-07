# Expressive Adapter Status

This document tracks the current production-readiness state for the expressive
TTS adapters named in the ongoing adapter hardening goal. It is an audit aid,
not a replacement for the smoke validation runbook.

Use [the expressive adapter smoke runbook](expressive-adapter-smoke.md) before
marking any unproven GPU-heavy adapter as production-ready.

## Status Matrix

| Model | Adapter package | Packaging/runtime isolation | Runtime metadata | Smoke status |
| --- | --- | --- | --- | --- |
| `cosyvoice2-tts:0.5b` | `vox-cosyvoice==0.1.10` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/cosyvoice` | Linux x86_64 CUDA/Torch; registry requires `min_vram_gb=8`; CPU/ONNX and Spark/ARM NVIDIA not production-supported | Previously cluster-smoked successfully, but slow; current live `vox` pod no longer has the model installed |
| `dia-tts:1.6b` | `vox-dia==0.2.15` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/dia`; uses released `transformers==4.57.6`; wires Dia audio-prompt voice cloning when Vox supplies both `reference_audio` and `reference_text` | Linux x86_64 CUDA/Torch; registry requires `min_vram_gb=12`; CPU/ONNX and Spark/ARM NVIDIA not production-supported | Pending GPU smoke |
| `orpheus-tts:medium-3b` | `vox-orpheus==0.1.7` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/orpheus`; exposes Orpheus generation controls and validates preset voices before synthesis | Linux x86_64 CUDA/Torch; registry requires `min_vram_gb=10`; CPU and Spark/ARM NVIDIA not packaged | Not proven; current live `vox` pod does not have the model installed |
| `indextts-tts:2` | `vox-indextts==0.1.21` | Covered by adapter tests; runtime under `$VOX_HOME/runtime/indextts`; keeps process NumPy stable with the TensorBoard `np.bool8` compatibility alias; installs NumPy-2-compatible runtime wheels; patches upstream `torchaudio.save` to write file outputs through soundfile; purges sibling-runtime TensorBoard/Transformers modules, stale Torch/CUDA runtime packages, and stale NumPy/Matplotlib artifacts before import probes; selects constructor signatures without swallowing internal model-load failures; exposes IndexTTS2 emotion, audio-prompt, and advanced generation controls through Vox synthesis params; parses documented boolean param forms and rejects ambiguous boolean values | Linux x86_64 CUDA/Torch; registry requires `min_vram_gb=10`; CPU/ONNX and Spark/ARM NVIDIA not production-supported | Current live `vox` pod serves Samantha voice successfully; not a clean-pull proof |

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

## Current CosyVoice Finding

`cosyvoice2-tts:0.5b` is the known expressive baseline: it has previously
cluster-smoked successfully but was slow. The adapter package is
`vox-cosyvoice==0.1.10`, the runtime is `$VOX_HOME/runtime/cosyvoice`, and the
registry records Linux x86_64 CUDA/Torch with `min_vram_gb=8`. CPU/ONNX and
Spark/ARM NVIDIA are not production-supported for this entry.

The live production `vox` deployment was inspected on 2026-07-06. `GET
/v1/models/cosyvoice2-tts:0.5b` returned HTTP 404 with `Model
'cosyvoice2-tts:0.5b' not found`, so the current pod cannot re-prove CosyVoice
without a deliberate `vox pull` / PVC mutation. Do not treat the previous
successful cluster smoke as current served evidence.

The proof-target preset path was exercised on Roy's Mac without Docker, `vox
pull`, or model downloads:

```bash
uv run python scripts/expressive-adapter-local-smoke.py \
  --proof-target cosyvoice \
  --scratch-root /tmp/vox-adapter-lab \
  --estimate-only \
  --max-download-gb 20 \
  --cleanup
```

It resolved `cosyvoice2-tts:0.5b` to `FunAudioLLM/CosyVoice2-0.5B` and
automatically applied the expected clean-pull state: `vox-cosyvoice==0.1.10`,
runtime `cosyvoice`, model link `cosyvoice2-tts`, and image
`ghcr.io/eleven-am/vox:latest`. The guard skipped pull on the Mac because
Torch/CUDA/Linux/x86_64/VRAM requirements were missing and all 18 Hugging Face
files reported unknown sizes.
That estimate-only evidence is expected to report `proof_ready: false` with
`proof_blockers` explaining that it is not a synthesis proof and that the Mac
runtime is incompatible.

## Current Dia Finding

Upstream Dia 1.6B documentation states that the model has only been tested on
GPUs with PyTorch/CUDA and that CPU support is future work:
https://github.com/nari-labs/dia#hardware-and-inference-speed. Hugging Face's
model page also says the full version requires around 10GB of VRAM. The Vox
adapter therefore only claims Linux x86_64 CUDA/Torch today. CPU, ONNX, and
Spark/ARM NVIDIA remain unsupported unless a real upstream-compatible backend
is identified and smoked.

The production `vox` deployment was inspected read-only while running
`ghcr.io/eleven-am/vox:v0.2.86`. It had `dia-tts:1.6b` model artifacts and
`vox-dia==0.2.11` installed, but `/home/vox/.vox/runtime/dia` only contained
the Vox fallback `.pth` file. Dia synthesis did not reach model/runtime load:
the scheduler rejected the request because the deployment was started with
`--max-vram 10GiB --vram-headroom 1GiB`, while Dia is budgeted as a 10GB model
plus headroom. The remote registry now records this constraint directly as
`min_vram_gb=12`.

The existing-server smoke was rerun against the same production endpoint on
2026-07-06 and wrote evidence to `/tmp/vox-served-smoke/evidence.json`. Health,
model listing, model detail, loaded-model state, and `/v1/system/memory` all
responded. Short and long Dia synthesis both returned HTTP 500 in under 200 ms,
and server logs showed the concrete scheduler failure:
`Cannot satisfy VRAM budget: projected 12500000000 bytes plus headroom
1073741824 exceeds max 10737418240 bytes`. The loaded model state before and
after the run still contained only `parakeet-stt:tdt-0.6b-v3`; Dia was not
loaded and no Dia audio was generated. This classifies the current served
failure as deployment hardware/budget, while also showing the deployed image
still hides that budget failure behind a generic `Internal synthesis error`.
The current local Vox source has tests expecting this path to map to HTTP 507,
so that should be verified after the next image deploy.

The local estimate-only clean-pull preflight was run on Roy's Mac with:

```bash
uv run python scripts/expressive-adapter-local-smoke.py \
  --model dia-tts:1.6b \
  --scratch-root /tmp/vox-adapter-lab \
  --estimate-only \
  --max-download-gb 20 \
  --cleanup
```

That preflight did not run Docker, did not run `vox pull`, and did not download
model files. It proved that the remote registry resolves
`dia-tts:1.6b` to `nari-labs/Dia-1.6B-0626` and that Hugging Face metadata is
reachable. It also confirmed that the local Mac is not a valid Dia target:
missing Torch, missing CUDA, Darwin/arm64 host, and unknown VRAM. Hugging Face
reported 12 selected files but no file sizes, so the local clean-pull helper
would skip a real pull unless `--allow-large-download` is deliberately passed
after choosing a disposable scratch location with enough free space.

The newer proof-target preset path was also exercised without Docker, `vox
pull`, or model downloads:

```bash
uv run python scripts/expressive-adapter-local-smoke.py \
  --proof-target dia \
  --scratch-root /tmp/vox-adapter-lab \
  --estimate-only \
  --max-download-gb 20 \
  --cleanup
```

It resolved the same model and automatically applied the expected clean-pull
state: `vox-dia==0.2.15`, runtime `dia`, model link `dia-tts`, and image
`ghcr.io/eleven-am/vox:latest`. The guard still skipped pull on the Mac because
Torch/CUDA/Linux/x86_64/VRAM requirements were missing and all 12 Hugging Face
files reported unknown sizes.
That estimate-only evidence is expected to report `proof_ready: false` with
`proof_blockers`; it is not acceptable completion evidence for Dia.

That finding is not a successful smoke test. It is evidence that the current
production deployment is too tightly budgeted for Dia and that older manifests
may exist from before the pull-atomicity fix. A valid Dia pass still requires a
fresh pull and synthesis in an approved non-production clean-pull environment
with a VRAM budget that satisfies the registry metadata. Do not create a new
namespace or PVC in the live cluster to get that proof.

## Current Orpheus Finding

Upstream Orpheus documentation describes a Llama-3B-backed TTS model with
guided emotion tags, low-latency streaming, and zero-shot voice cloning:
https://github.com/canopyai/Orpheus-TTS. The current upstream streaming
example installs `orpheus-speech`, which uses vLLM under the hood, and the
Hugging Face model page requires accepting gated model access before files can
be downloaded. The registry source `canopylabs/orpheus-tts-0.1-finetune-prod`
currently resolves to the same Hugging Face revision as
`canopylabs/orpheus-3b-0.1-ft`, but it remains a gated model source.

The Vox adapter therefore only claims Linux x86_64 CUDA/Torch today. CPU,
ONNX, and Spark/ARM NVIDIA are not production-supported because this adapter
does not have a portable non-vLLM backend. Although upstream Orpheus advertises
zero-shot voice cloning, `vox-orpheus==0.1.7` only wires preset voices and
rejects Vox `reference_audio` / `reference_text` clearly. A valid Orpheus pass
still requires fresh clean-pull smoke and synthesis in an approved
non-production GPU environment.

The live production `vox` deployment was inspected on 2026-07-06. `GET
/v1/models/orpheus-tts:medium-3b` returned HTTP 404 with `Model
'orpheus-tts:medium-3b' not found`, so the current pod provides no Orpheus load
or synthesis evidence.

The local estimate-only clean-pull preflight was run on Roy's Mac with:

```bash
uv run python scripts/expressive-adapter-local-smoke.py \
  --model orpheus-tts:medium-3b \
  --scratch-root /tmp/vox-adapter-lab \
  --estimate-only \
  --max-download-gb 20 \
  --cleanup
```

That preflight did not run Docker, did not run `vox pull`, and did not download
model files. It proved that the remote registry resolves
`orpheus-tts:medium-3b` to `canopylabs/orpheus-tts-0.1-finetune-prod` and that
Hugging Face metadata is reachable. It also confirmed that the local Mac is not
a valid Orpheus target: missing Torch, missing CUDA, Darwin/arm64 host, and
unknown VRAM. Hugging Face reported 20 selected files but no file sizes, so the
local clean-pull helper would skip a real pull. `--allow-large-download` could
only bypass the unknown-size guard; it would not bypass the missing runtime
requirements.

The newer proof-target preset path was also exercised without Docker, `vox
pull`, or model downloads:

```bash
uv run python scripts/expressive-adapter-local-smoke.py \
  --proof-target orpheus \
  --scratch-root /tmp/vox-adapter-lab \
  --estimate-only \
  --max-download-gb 20 \
  --cleanup
```

It resolved the same model and automatically applied the expected clean-pull
state: `vox-orpheus==0.1.7`, runtime `orpheus`, model link `orpheus-tts`, and
image `ghcr.io/eleven-am/vox:latest`. The guard still skipped pull on the Mac
because Torch/CUDA/Linux/x86_64/VRAM requirements were missing and all 20
Hugging Face files reported unknown sizes.
That estimate-only evidence is expected to report `proof_ready: false` with
`proof_blockers`; it is not acceptable completion evidence for Orpheus.

## Current IndexTTS Finding

Upstream IndexTTS2 documents zero-shot speaker cloning, separated emotion and
timbre control, multiple emotion-control inputs, and duration-control research:
https://github.com/index-tts/index-tts. The upstream release notes say precise
duration control is not enabled in the current release, so Vox does not expose
a duration target parameter. Vox now exposes the upstream advanced generation
controls that are available in the IndexTTS2 WebUI, including `do_sample`,
`temperature`, `top_p`, `top_k`, `num_beams`, `repetition_penalty`,
`length_penalty`, `max_mel_tokens`, and `max_text_tokens_per_segment`, in
addition to the emotion text/vector/audio-prompt controls.

The existing `vox` namespace smoke result is useful evidence that IndexTTS can
load and generate on Roy's current CUDA hardware with the Samantha voice, but
it is not a clean-pull proof. A production-grade pass still requires fresh
clean-pull smoke in an approved non-production environment, with adapter
runtime preparation, model store state, short/long synthesis, memory sampling,
and manual audio usability recorded.

The live production `vox` deployment was rechecked on 2026-07-06 using the
actual cloned voice id `44a66a38` (`Samantha (Her)`), not the display name
`samantha`. The first invalid check with `voice=samantha` loaded IndexTTS in
about 16.7s but returned HTTP 400: `IndexTTS requires reference_audio or a voice
path for speaker cloning`. The corrected direct HTTP checks succeeded:

- Short text: HTTP 200, 37 input characters, 215596-byte WAV, 4.888s decoded
  duration, 5.42s wall time. Server log:
  `synthesize_streamed indextts-tts:2 chars=37 audio_ms=4887
  processing_ms=5325 format=wav`.
- Long text: HTTP 200, 241 input characters, 905772-byte WAV, 20.538s decoded
  duration, 19.45s wall time. Server log:
  `synthesize_streamed indextts-tts:2 chars=241 audio_ms=20538
  processing_ms=19358 format=wav`.

The same run showed `parakeet-stt:tdt-0.6b-v3` plus `indextts-tts:2` loaded
under the 10GiB VRAM policy, with estimated loaded VRAM at 8.5GB. After the
requests completed, both models reported `ref_count=0` and were evictable. This
is positive served evidence for the existing live install, but it still does not
prove a clean pull or fresh isolated runtime install.

One smoke-tooling caveat was found during this verification: Python
`urllib.request` in `scripts/expressive-adapter-served-smoke.py` hung while
reading the chunked IndexTTS WAV response after the server had generated audio.
Bounded `curl` against the same endpoint completed normally. That is a smoke
harness/client compatibility issue to fix separately before using the Python
served-smoke harness as authoritative for IndexTTS.

The local estimate-only clean-pull preflight was run on Roy's Mac with:

```bash
uv run python scripts/expressive-adapter-local-smoke.py \
  --model indextts-tts:2 \
  --scratch-root /tmp/vox-adapter-lab \
  --estimate-only \
  --max-download-gb 20 \
  --cleanup
```

That preflight did not run Docker, did not run `vox pull`, and did not download
model files. It proved that the remote registry resolves `indextts-tts:2` to
`IndexTeam/IndexTTS-2` and that Hugging Face metadata is reachable. It also
confirmed that the local Mac is not a valid IndexTTS target: missing Torch,
missing CUDA, Darwin/arm64 host, and unknown VRAM. Hugging Face reported 21
selected files but no file sizes, so the local clean-pull helper would skip a
real pull. `--allow-large-download` could only bypass the unknown-size guard;
it would not bypass the missing runtime requirements.

The newer proof-target preset path was also exercised without Docker, `vox
pull`, or model downloads:

```bash
uv run python scripts/expressive-adapter-local-smoke.py \
  --proof-target indextts \
  --scratch-root /tmp/vox-adapter-lab \
  --estimate-only \
  --max-download-gb 20 \
  --cleanup
```

It resolved the same model and automatically applied the expected clean-pull
state: `vox-indextts==0.1.21`, runtime `indextts`, model link `indextts-tts`,
default voice `samantha`, and image `ghcr.io/eleven-am/vox:latest`. The guard
still skipped pull on the Mac because Torch/CUDA/Linux/x86_64/VRAM requirements
were missing and all 21 Hugging Face files reported unknown sizes.
That estimate-only evidence is expected to report `proof_ready: false` with
`proof_blockers`; it is not acceptable completion evidence for IndexTTS.

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
plus `/v1/system/memory` before and after synthesis without creating
namespaces, PVCs, or running `vox pull`. Full synthesis runs also record
per-request `/v1/system/memory` samples under each synthesis case.
Use `--inspect-only` when the goal is read-only inspection; it skips
`/v1/audio/speech` and records no synthesis cases. Full served smoke can still
change in-memory loaded model state and VRAM while a request is in flight, so
heavy synthesis against production should be intentional.

Existing-server smoke is not sufficient to mark a model production-ready because
it cannot prove a clean model pull or clean runtime install.

Full production-readiness still requires clean-pull smoke in an explicitly
approved non-production environment, preferably local/Docker with disposable
scratch/cache directories. Do not create a new namespace or PVC in the live
cluster just because a model needs testing. The exact clean-pull command queue
for the remaining targets is in
[Approved Clean-Pull Proof Queue](expressive-adapter-smoke.md#approved-clean-pull-proof-queue).
The required proof is:

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
10. RAM/VRAM evidence is recorded. Existing-server smoke records
    `/v1/system/memory` before and after synthesis plus per-request memory
    sample summaries during short and long synthesis; local Docker clean-pull
    smoke records container RAM and optional `nvidia-smi` VRAM snapshots before
    and after smoke, plus continuous sample summaries with peak observed RAM
    and GPU memory for pull, short synthesis, and long synthesis.
11. Audio is manually judged usable.
12. Any failure is classified as Vox, adapter, dependency, upstream, or hardware
    and includes a concrete failure note with the likely cause or next fix.
