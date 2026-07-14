# Expressive Adapter Smoke Validation

This runbook is the verification gate for expressive TTS adapters that need
GPU-backed runtime dependencies. The default operational path is to test the
already-running Vox HTTP endpoint. Do not create a new namespace or PVC for
cluster testing. If cluster testing is needed, use the existing Vox namespace,
service, and HTTP port that are already running.

## Implementation Layout

The two CLI entry points keep environment-specific orchestration separate:

- `scripts/expressive-adapter-local-smoke.py` owns disposable Docker clean-pull runs.
- `scripts/expressive-adapter-served-smoke.py` owns read-only checks against an existing Vox server.

Shared behavior lives under `src/vox/smoke/`: `audio.py` inspects generated WAVs,
`evidence.py` serializes and classifies evidence, and `local_proof.py` evaluates the
clean-pull proof contract. Changes to evidence semantics belong in those shared modules
so local and served smoke runs do not drift.

## Scope

Use this runbook for:

- `cosyvoice2-tts:0.5b`
- `dia-tts:1.6b`
- `orpheus-tts:medium-3b`
- `indextts-tts:2`

The adapter package and registry metadata tests prove packaging boundaries and
pull-time compatibility checks. They do not prove that a model can load,
synthesize useful audio, or fit a specific GPU. A model is production-grade only
after this smoke validation has evidence for `vox pull`, short synthesis, long
synthesis, latency, output duration, RAM/VRAM behavior, and audio usability.

## Published Adapter Baseline

Smoke validation must resolve adapter packages from PyPI, not from a local
source tree or a patched live cluster directory. Verify the installed versions
before marking a model as smoke-tested:

| Model | Expected adapter package |
| --- | --- |
| `cosyvoice2-tts:0.5b` | `vox-cosyvoice==0.1.10` |
| `dia-tts:1.6b` | `vox-dia==0.2.15` |
| `orpheus-tts:medium-3b` | `vox-orpheus==0.1.7` |
| `indextts-tts:2` | `vox-indextts==0.1.21` |

If the registry points at a newer package, record that newer version in the
smoke evidence and update this table with the same change.

## Existing Server Smoke

When the goal is to test the Vox server that is already running, do not create
a new namespace or PVC. Use the served-smoke helper against the existing HTTP
endpoint and the existing Vox deployment:

For read-only inspection, use `--inspect-only`. This records server/model
state and stops before synthesis, so it should not load a TTS model or consume
additional VRAM:

```bash
python scripts/expressive-adapter-served-smoke.py \
  --base-url http://127.0.0.1:8000 \
  --model dia-tts:1.6b \
  --inspect-only

make smoke-expressive-served MODEL=dia-tts:1.6b SMOKE_BASE_URL=http://127.0.0.1:8000 SMOKE_INSPECT_ONLY=1
```

For synthesis validation, run without `--inspect-only` and manually mark audio
usability after listening to the generated files:

```bash
python scripts/expressive-adapter-served-smoke.py \
  --base-url http://127.0.0.1:8000 \
  --model dia-tts:1.6b \
  --memory-sample-interval 1.0 \
  --audio-usable yes

make smoke-expressive-served MODEL=dia-tts:1.6b SMOKE_BASE_URL=http://127.0.0.1:8000 SMOKE_MEMORY_SAMPLE_INTERVAL=1.0 SMOKE_AUDIO_USABLE=yes
```

If a served smoke run fails and the failure is being recorded instead of
immediately rerun, classify the likely owner explicitly:

```bash
python scripts/expressive-adapter-served-smoke.py \
  --base-url http://127.0.0.1:8000 \
  --model dia-tts:1.6b \
  --audio-usable no \
  --failure-class adapter \
  --failure-note "runtime verification failed after install"
```

If the existing server requires `VOX_API_KEY`, pass it through the environment
or `SMOKE_API_KEY`. The served-smoke evidence records only whether a key was
provided, never the key value:

```bash
VOX_API_KEY=... python scripts/expressive-adapter-served-smoke.py \
  --base-url https://vox.example \
  --model dia-tts:1.6b \
  --audio-usable yes

make smoke-expressive-served MODEL=dia-tts:1.6b SMOKE_BASE_URL=https://vox.example SMOKE_API_KEY=... SMOKE_AUDIO_USABLE=yes
```

For adapters that require a voice or reference, pass the value the running
server already knows how to resolve:

```bash
python scripts/expressive-adapter-served-smoke.py \
  --base-url http://127.0.0.1:8000 \
  --model indextts-tts:2 \
  --voice samantha \
  --audio-usable yes
```

This path intentionally does not call `kubectl`, `vox pull`, or mutate adapter,
runtime, model, or PVC contents. It records whether an API key was provided,
`/v1/health`, `/v1/models`, `GET /v1/models/{model}` for the requested model,
`/v1/models/loaded` before synthesis, `/v1/models/loaded` after synthesis,
`/v1/system/memory` before synthesis, `/v1/system/memory` after synthesis,
per-synthesis `/v1/system/memory` samples under `memory_samples`, short
synthesis, long synthesis, wall times, WAV metadata, SHA-256 digests, silence
checks, and explicit `failure_reasons` under
`/tmp/vox-served-smoke` by default.
Because this path uses an already-running server, the evidence also records
`clean_pull_proof: false` and `clean_pull_blockers`. A passing served smoke run
can prove the current endpoint synthesized usable audio, but it cannot prove a
clean `vox pull`, a clean adapter package install, or a clean adapter runtime
install.

Per-synthesis memory sampling defaults to every `1.0` second and records one
immediate sample before the synthesis request starts. Use
`--memory-sample-interval 0` or `SMOKE_MEMORY_SAMPLE_INTERVAL=0` to disable
per-request sampling.

Full existing-server smoke can change in-memory scheduler state and VRAM usage
while requests are in flight because synthesis can load models. It is
storage-safe, not side-effect-free. Use `--inspect-only` for read-only checks,
and do not run heavy synthesis against a live deployment unless that is the
requested test.

Existing-server smoke is not enough to mark an adapter fully production-ready
because it cannot prove a clean pull, clean runtime install, or clean model
store. It is useful evidence that the currently running server can synthesize
with the model without taking down or rebuilding the deployment.

## Local Docker Clean-Pull Smoke

Use this path when clean-pull evidence is needed without touching the live
cluster. The helper runs a Vox Docker image with all mutable state mounted
under one disposable scratch directory:

- `$VOX_HOME` -> `<scratch>/<model>/vox-home`
- `HF_HOME` / `HUGGINGFACE_HUB_CACHE` -> `<scratch>/<model>/hf-cache`
- `XDG_CACHE_HOME` -> `<scratch>/<model>/xdg-cache`
- `TMPDIR` -> `<scratch>/<model>/tmp`

The helper refuses to run `docker` or `vox pull` unless `--allow-download` is
passed. That guard exists because model pulls can be large. Before enabling it,
check available disk and choose a scratch location that is safe to delete:

```bash
df -h /tmp
```

Before allowing any Docker or pull work, run the host-side estimate-only path.
It resolves the same registry entry and pull variant, queries Hugging Face file
metadata, writes `local-smoke-evidence.json`, and exits before Docker can pull
an image or Vox can download model files:

```bash
python scripts/expressive-adapter-local-smoke.py \
  --model dia-tts:1.6b \
  --scratch-root /tmp/vox-adapter-lab \
  --estimate-only \
  --max-download-gb 20

make smoke-expressive-local MODEL=dia-tts:1.6b SMOKE_ESTIMATE_ONLY=1 SMOKE_MAX_DOWNLOAD_GB=20
```

`--estimate-only` is not a smoke pass. It only proves that the registry,
variant resolution, and Hugging Face metadata are reachable enough to decide
whether a later clean-pull run is safe for the selected scratch filesystem.
If variant resolution reports missing runtime requirements, such as CUDA,
Torch, Linux, x86_64, or enough VRAM, the helper records those requirements in
`download_guard_failures` and treats the later pull as unsafe for that
runtime.

After the runtime snapshot and before `vox pull`, the helper queries the
resolved Hugging Face repository and records an estimated download size in
`download_estimate`. By default it refuses to pull if the known size is above
`--max-download-gb 20`, if any selected files do not report a size, or if the
known download size would leave less than `--min-free-gb` free on the scratch
filesystem. Review the estimate first; then pass `--allow-large-download` only
when that scratch location is safe to fill. This is separate from
`--allow-download`: the first flag permits a pull attempt at all, while the
second acknowledges a large, unknown, unavailable, or reserve-breaching
estimate. `--allow-large-download` does not bypass missing runtime
requirements; use a compatible image/host instead.

For CUDA-only adapters such as Dia, Orpheus, and IndexTTS, use a
CUDA-capable Vox image on a Linux Docker host with NVIDIA GPU access. Do not
use the `:cpu` or `:lean` images for these GPU-only clean-pull checks.

```bash
python scripts/expressive-adapter-local-smoke.py \
  --model dia-tts:1.6b \
  --image ghcr.io/eleven-am/vox:latest \
  --scratch-root /tmp/vox-adapter-lab \
  --expect-adapter vox-dia \
  --expect-adapter-package vox-dia==0.2.15 \
  --expect-runtime dia \
  --expect-model-link dia-tts \
  --resource-sample-interval 1.0 \
  --max-download-gb 20 \
  --allow-download \
  --allow-large-download \
  --cleanup \
  --audio-usable yes

make smoke-expressive-local MODEL=dia-tts:1.6b SMOKE_IMAGE=ghcr.io/eleven-am/vox:latest SMOKE_EXPECT_ADAPTER=vox-dia SMOKE_EXPECT_ADAPTER_PACKAGE=vox-dia==0.2.15 SMOKE_EXPECT_RUNTIME=dia SMOKE_EXPECT_MODEL_LINK=dia-tts SMOKE_RESOURCE_SAMPLE_INTERVAL=1.0 SMOKE_MAX_DOWNLOAD_GB=20 SMOKE_ALLOW_DOWNLOAD=1 SMOKE_ALLOW_LARGE_DOWNLOAD=1 SMOKE_AUDIO_USABLE=yes SMOKE_CLEANUP=1
```

For CPU/ONNX validation, request the same pull variant used by `vox pull`:

```bash
python scripts/expressive-adapter-local-smoke.py \
  --model chatterbox-tts-turbo:0.1.7 \
  --variant onnx \
  --image ghcr.io/eleven-am/vox:lean \
  --scratch-root /tmp/vox-adapter-lab \
  --expect-adapter vox-chatterbox \
  --expect-adapter-package vox-chatterbox==0.1.2 \
  --expect-runtime chatterbox \
  --expect-model-link chatterbox-tts-turbo \
  --allow-download \
  --audio-usable yes
```

The helper records `evidence_schema_version`, runtime snapshot, resource snapshots before and after smoke
where measurable, Docker image inspect metadata including image ID and repo
digests when available, pre-pull download estimate, pre-pull clean state, pull
command, short/long synthesis commands, copied audio stats including sample
width, peak, RMS, and silence detection,
post-pull adapter/runtime/manifest/model-store state, disk before/after, and
`failure_reasons`, `proof_ready`, and `proof_blockers` in
`local-smoke-evidence.json` under the scratch directory. Local clean-pull
evidence currently uses `evidence_schema_version: 2`; schema 2 adds the
required `goal_checklist` proof map. Resource snapshots
include Linux RAM totals/available bytes from `/proc/meminfo` and optional
VRAM usage from `nvidia-smi` when the selected Docker image exposes it.
The local helper also samples RAM and `nvidia-smi` VRAM during `vox pull`,
short synthesis, and long synthesis by default, then records per-step sample
counts plus peak observed RAM and GPU memory under `resource_samples`. It also
records the configured interval as `resource_sample_interval_s`. Tune this with
`--resource-sample-interval`; use `--resource-sample-interval 0` to disable
continuous sampling.
The evidence also includes `goal_checklist`, a machine-readable map from the
adapter production-readiness goal to the concrete proof fields. Each check has
`status`, `evidence`, and `blockers` keys. Passing evidence requires all checks
to be `passed`; blocked checks explain the missing command, state, artifact,
resource, usability, or failure-classification proof.
Use `--expect-adapter`, `--expect-adapter-package`, `--expect-runtime`, and
`--expect-model-link` to make the helper enforce clean-pull state: expected
entries and adapter package metadata must be absent before pull and present
after pull. Missing post-pull state or reused scratch state fails the smoke run
instead of relying on manual inspection of the evidence.
Use `--expect-adapter-package NAME==VERSION` to prove that the adapter package
installed from PyPI matches the expected runbook baseline; mismatches are
recorded in `state_failures`.
When `--voice` looks like a file path, the local helper checks that the path
exists inside the disposable container before `vox pull`; a missing reference
file is recorded in `voice_reference_failures` and pull/synthesis are skipped.
Voice IDs such as `samantha` are passed through without a file-existence check.
Failing local smoke runs must set `--failure-class` to one of `Vox`,
`adapter`, `dependency`, `upstream`, or `hardware`; passing runs must leave the
class as `none`. This keeps evidence usable for deciding whether the fix
belongs in core Vox, an adapter package, dependency pins/runtime preparation,
upstream model packaging, or the target hardware. Classified failures must also
set `--failure-note` with the concrete cause or next fix, such as
`--failure-note "upstream wheel is unavailable for linux/arm64"`.
If the runtime snapshot or pre-pull clean-state check fails, the helper skips
`vox pull` so a broken image or unproven scratch state cannot start a large
download. If the download estimate fails, is not parseable, exceeds
`--max-download-gb`, contains unknown-sized files, or would breach the
`--min-free-gb` reserve, the helper also skips `vox pull` unless
`--allow-large-download` is passed. If variant resolution reports missing
runtime requirements, the helper skips `vox pull` regardless of
`--allow-large-download`; the fix is to use a compatible image/host or update
registry metadata if the requirement is wrong. These download, disk, and
runtime-compatibility blocks are recorded as `pre-pull guard failed` in
`skipped_commands`. If `vox pull` fails, the helper skips short/long synthesis.
Silent generated WAVs fail the run even if they are non-empty. Skipped steps
are recorded in `skipped_commands` instead of spending more time on requests
that cannot succeed.
The helper also evaluates the full clean-pull proof contract before writing
evidence. `proof_ready: true` means the run had a fresh pull, clean pre-pull
manifest/runtime/model state, expected post-pull adapter/runtime/manifest/model
state, successful short and long synthesis, non-silent copied WAVs,
`--audio-usable yes`, and RAM/VRAM
sample summaries. `proof_ready: false` is paired with `proof_blockers` so an
estimate-only run, missing expected state, disabled resource sampling, skipped
command, failed command, or unreviewed audio cannot be mistaken for a model
proof.
It copies generated WAV files into the `artifacts` directory before optional
cleanup. With `--cleanup`, it removes `vox-home`, Hugging Face cache, XDG cache,
and temp directories after writing evidence while preserving
`local-smoke-evidence.json` and copied WAV artifacts. Delete the full
model-specific scratch directory after collecting evidence unless you
intentionally want to keep the artifacts:

```bash
rm -rf /tmp/vox-adapter-lab/dia-tts-1.6b
```

## Approved Clean-Pull Proof Queue

The models below are the remaining expressive-adapter proof targets. These
commands are for an approved non-production Linux x86_64 CUDA host with enough
VRAM and disposable scratch storage. They are not for Roy's live Vox PVC, and
they are not for the local Mac when the estimate-only guard reports missing
Torch, CUDA, Linux, x86_64, or VRAM requirements.

Before each run, verify local scratch capacity on the proof host:

```bash
df -h /tmp
```

Run the estimate-only command first. If it reports missing runtime
requirements, do not pass `--allow-download`; move to a compatible proof host
or fix registry metadata if the requirement is wrong. If it reports unknown or
large Hugging Face file sizes but the runtime is otherwise compatible, use
`--allow-large-download` only after confirming the scratch filesystem is safe
to fill.

The local smoke helper has guarded proof presets for all four goal targets. The
presets fill the model name, CUDA image, expected adapter package, runtime,
model link, and the default IndexTTS voice id. They do not bypass
`--allow-download`, `--allow-large-download`, `--audio-usable`, or cleanup
requirements.

Estimate-only preset examples:

```bash
make smoke-expressive-local SMOKE_PROOF_TARGET=cosyvoice SMOKE_ESTIMATE_ONLY=1 SMOKE_MAX_DOWNLOAD_GB=20
make smoke-expressive-local SMOKE_PROOF_TARGET=dia SMOKE_ESTIMATE_ONLY=1 SMOKE_MAX_DOWNLOAD_GB=20
make smoke-expressive-local SMOKE_PROOF_TARGET=orpheus SMOKE_ESTIMATE_ONLY=1 SMOKE_MAX_DOWNLOAD_GB=20
make smoke-expressive-local SMOKE_PROOF_TARGET=indextts SMOKE_ESTIMATE_ONLY=1 SMOKE_MAX_DOWNLOAD_GB=20
```

CosyVoice baseline clean-pull proof:

```bash
make smoke-expressive-local SMOKE_PROOF_TARGET=cosyvoice SMOKE_RESOURCE_SAMPLE_INTERVAL=1.0 SMOKE_MAX_DOWNLOAD_GB=20 SMOKE_ALLOW_DOWNLOAD=1 SMOKE_ALLOW_LARGE_DOWNLOAD=1 SMOKE_CLEANUP=1 SMOKE_AUDIO_USABLE=yes
```

Dia clean-pull proof:

```bash
make smoke-expressive-local SMOKE_PROOF_TARGET=dia SMOKE_RESOURCE_SAMPLE_INTERVAL=1.0 SMOKE_MAX_DOWNLOAD_GB=20 SMOKE_ALLOW_DOWNLOAD=1 SMOKE_ALLOW_LARGE_DOWNLOAD=1 SMOKE_CLEANUP=1 SMOKE_AUDIO_USABLE=yes
```

Orpheus clean-pull proof:

```bash
make smoke-expressive-local SMOKE_PROOF_TARGET=orpheus SMOKE_RESOURCE_SAMPLE_INTERVAL=1.0 SMOKE_MAX_DOWNLOAD_GB=20 SMOKE_ALLOW_DOWNLOAD=1 SMOKE_ALLOW_LARGE_DOWNLOAD=1 SMOKE_CLEANUP=1 SMOKE_AUDIO_USABLE=yes
```

Orpheus uses gated Hugging Face assets. A failed download caused by missing
model access or missing Hugging Face credentials should be recorded as
`--failure-class upstream` with a concrete `--failure-note`; it is not evidence
that the Vox adapter loaded and failed.

IndexTTS clean-pull proof requires a voice id that the disposable runtime can
resolve, or a test-only reference WAV path that exists inside the container.
Do not use or delete production voice data under `$VOX_HOME/voices`.

```bash
make smoke-expressive-local SMOKE_PROOF_TARGET=indextts SMOKE_RESOURCE_SAMPLE_INTERVAL=1.0 SMOKE_MAX_DOWNLOAD_GB=20 SMOKE_ALLOW_DOWNLOAD=1 SMOKE_ALLOW_LARGE_DOWNLOAD=1 SMOKE_CLEANUP=1 SMOKE_AUDIO_USABLE=yes
```

The expanded equivalent is:

```bash
python scripts/expressive-adapter-local-smoke.py \
  --proof-target dia \
  --resource-sample-interval 1.0 \
  --max-download-gb 20 \
  --allow-download \
  --allow-large-download \
  --cleanup \
  --audio-usable yes
```

If a proof run fails, keep the evidence and classify the owner:

- `--failure-class Vox` for core resolver, pull, scheduler, API, or model-store
  behavior.
- `--failure-class adapter` for adapter import, load, parameter mapping,
  synthesis, or voice-handling bugs after dependencies are present.
- `--failure-class dependency` for pip/runtime install, version conflicts,
  broken wheels, or stale runtime repair failures.
- `--failure-class upstream` for gated repositories, missing upstream files,
  missing model cards/artifacts, or upstream package regressions.
- `--failure-class hardware` for VRAM, CUDA, architecture, or unsupported
  platform constraints.

Passing proof requires `proof_ready: true`: fresh `vox pull`, clean pre-pull
state, expected PyPI adapter package metadata, short and long non-silent WAVs,
plausible durations, RAM/VRAM samples, copied artifacts, and manual
`--audio-usable yes`. If `proof_ready` is false, use `proof_blockers` as the
authoritative list of what still needs to be fixed or rerun.

After a clean-pull run, verify the evidence file mechanically before treating
it as proof:

```bash
make verify-expressive-local-evidence EVIDENCE=/tmp/vox-adapter-lab/dia-tts-1.6b/artifacts/local-smoke-evidence.json VERIFY_PROOF_TARGET=dia VERIFY_MODEL=dia-tts:1.6b
```

This calls `scripts/verify-expressive-adapter-evidence.py`, checks
`evidence_schema_version`, requires `proof_ready: true`, and independently
checks the requested model/proof target, required command results, expected
adapter/runtime/model declarations, clean pre-pull manifest/runtime/model state,
actual post-pull adapter/runtime/manifest/model state, positive model blob storage,
short/long WAV stats,
copied artifact file presence, artifact byte counts, artifact SHA-256 digests,
resource sample summaries, `goal_checklist`, failure fields, and recorded
`proof_blockers`. It fails when the file is for the wrong model, estimate-only,
partial, failed, stale, reused scratch state, missing expected post-pull state,
missing its copied WAV artifacts, manually edited into an inconsistent state,
has blocked goal checks, or otherwise not production-readiness evidence.

## Legacy Disposable Kubernetes Smoke

This path is disabled by default and must not be used for the normal Vox
cluster. It exists only as an archived lab runner for a separate non-production
environment that was explicitly created for destructive clean-pull validation.
It is not the workflow for Roy's current Vox deployment.

For the live cluster, use [Existing Server Smoke](#existing-server-smoke)
against the current Vox endpoint. That path uses the running service and does
not create namespaces, PVCs, pods, or model/runtime storage.

If clean-pull evidence is needed, prefer a local/Docker disposable environment
with explicit scratch/cache directories. Do not use the live cluster as a clean
pull lab unless the user explicitly asks for live PVC mutation.

The legacy Kubernetes runner fails before any `kubectl` call unless
`VOX_ENABLE_DISPOSABLE_K8S_SMOKE=1` is set. That environment variable is an
intentional guardrail, not part of the normal workflow.

The historical disposable defaults are:

```bash
export VOX_SMOKE_NS=vox-adapter-smoke
export VOX_SMOKE_PVC=vox-adapter-smoke-data
export VOX_SMOKE_IMAGE=ghcr.io/eleven-am/vox:latest
```

If the approved lab namespace or PVC is missing, stop. Do not create one as a
shortcut, and do not mutate, clean, reinstall, restart, or scale the live Vox
deployment as part of adapter smoke testing.

Before running any smoke command, do a read-only preflight against the exact
disposable names:

```bash
kubectl get namespace "$VOX_SMOKE_NS"
kubectl -n "$VOX_SMOKE_NS" get pvc "$VOX_SMOKE_PVC"
kubectl -n "$VOX_SMOKE_NS" get pod vox-adapter-smoke
```

If the namespace or PVC is missing, stop. Do not invent new namespace/PVC names,
and do not mutate production storage to get a quick result. The preflight is
allowed to fail with `NotFound`; that means the legacy clean-pull smoke
environment does not exist.

The guarded legacy scripted path is:

```bash
VOX_ENABLE_DISPOSABLE_K8S_SMOKE=1 bash scripts/expressive-adapter-smoke.sh --model dia-tts:1.6b --audio-usable yes
```

To force a pull-time hardware/backend variant for models that publish multiple
variants, pass the same variant flag that `vox pull` accepts:

```bash
VOX_ENABLE_DISPOSABLE_K8S_SMOKE=1 bash scripts/expressive-adapter-smoke.sh --model chatterbox-tts-turbo:0.1.7 --variant onnx --audio-usable yes
```

For CPU/ONNX validation, explicitly create the disposable pod without a GPU
request so the result proves that the model can run without CUDA:

```bash
VOX_ENABLE_DISPOSABLE_K8S_SMOKE=1 bash scripts/expressive-adapter-smoke.sh --model chatterbox-tts-turbo:0.1.7 --variant onnx --cpu-only --audio-usable yes
```

The equivalent Makefile entrypoint is also guarded and should not be used for
the running cluster:

```bash
VOX_ENABLE_DISPOSABLE_K8S_SMOKE=1 make smoke-expressive MODEL=dia-tts:1.6b SMOKE_AUDIO_USABLE=yes
VOX_ENABLE_DISPOSABLE_K8S_SMOKE=1 make smoke-expressive MODEL=chatterbox-tts-turbo:0.1.7 SMOKE_VARIANT=onnx SMOKE_AUDIO_USABLE=yes
VOX_ENABLE_DISPOSABLE_K8S_SMOKE=1 make smoke-expressive MODEL=chatterbox-tts-turbo:0.1.7 SMOKE_VARIANT=onnx SMOKE_CPU_ONLY=1 SMOKE_AUDIO_USABLE=yes
```

For cloning adapters, copy a small test-only reference WAV into the disposable
PVC and pass it as the Vox voice path. Do not use production voice data:

```bash
VOX_ENABLE_DISPOSABLE_K8S_SMOKE=1 bash scripts/expressive-adapter-smoke.sh --model indextts-tts:2 --voice /home/vox/.vox/smoke-voices/reference.wav --audio-usable yes
VOX_ENABLE_DISPOSABLE_K8S_SMOKE=1 make smoke-expressive MODEL=indextts-tts:2 SMOKE_VOICE=/home/vox/.vox/smoke-voices/reference.wav SMOKE_AUDIO_USABLE=yes
```

The script refuses the production `vox` namespace and `vox-data` PVC. It also
refuses to create the disposable namespace, PVC, or pod unless the guard
environment variable is set and `--create` is passed explicitly:

```bash
VOX_ENABLE_DISPOSABLE_K8S_SMOKE=1 bash scripts/expressive-adapter-smoke.sh --model dia-tts:1.6b --create --audio-usable yes
VOX_ENABLE_DISPOSABLE_K8S_SMOKE=1 make smoke-expressive MODEL=dia-tts:1.6b SMOKE_CREATE=1 SMOKE_AUDIO_USABLE=yes
```

Evidence and copied WAV files are written under `/tmp/vox-adapter-smoke` by
default.

If the disposable pod already exists, the script verifies that it was created
with the requested image, PVC, and accelerator mode. Delete and recreate the
disposable pod when switching image tags, PVCs, or switching between GPU and
`--cpu-only` validation; otherwise the runner exits before writing misleading
evidence.

## Required Evidence

For each model, capture:

- image tag and digest
- adapter package version resolved from PyPI
- target model's expected adapter package baseline from this runbook
- registry entry used
- requested variant or `auto`
- accelerator request (`gpu` or `cpu-only`)
- voice id, voice path, or `none`
- manual audio usability verdict (`--audio-usable yes` for a passing run)
- voice path existence inside the disposable pod when `--voice` is a file path
- runtime capability snapshot from the pod
- pre-pull clean-state probe proving the target manifest, model link, and adapter runtime are not already present
- `vox pull <model>` output
- machine-readable `real` durations for pull, short synthesis, and long synthesis
- adapter, runtime, model, manifest, and blob storage usage after pull
- short synthesis wall time
- long synthesis wall time
- generated audio duration
- generated audio stream metadata, including codec, sample rate, and channels
- generated audio signal stats proving the WAV is not silent
- invalid, non-positive, silent, or shorter-than-expected long output fails the smoke
- pod memory and GPU memory snapshots after pull, short synthesis, and long synthesis
- output WAV artifact
- copied WAV byte size and SHA-256 digest
- smoke result and failed-step summary
- whether the audio is usable
- exact failure output if any step fails

Use a short text around one sentence and a longer text around one paragraph.
For cloning adapters, provide a small reference WAV copied into the disposable
PVC or mounted as test-only data, then pass it with `--voice`.
When `--voice` looks like a file path, the scripted smoke runner records a
voice-reference evidence section and fails the run if that path does not exist
inside the disposable pod. Voice IDs are recorded without a file-existence
check.
`indextts-tts:*` smoke validation requires `--voice` and fails before touching
Kubernetes resources when no disposable reference WAV or voice id is provided.
The smoke runner writes copied WAV artifacts even on failure. Listen to those
files and rerun with `--audio-usable yes` only when both short and long outputs
are usable; `--audio-usable no` records a failed usability verdict.
Failing smoke runs must be rerun or recorded with a concrete failure class:
`--failure-class Vox`, `adapter`, `dependency`, `upstream`, or `hardware`.
Passing runs must use the default `--failure-class none`.

## Commands

Create the disposable namespace and PVC only after approval:

```bash
kubectl create namespace "$VOX_SMOKE_NS"
kubectl -n "$VOX_SMOKE_NS" apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${VOX_SMOKE_PVC}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 40Gi
EOF
```

Run a one-shot test pod against the disposable PVC:

```bash
kubectl -n "$VOX_SMOKE_NS" run vox-adapter-smoke \
  --image="$VOX_SMOKE_IMAGE" \
  --restart=Never \
  --overrides='{
    "spec": {
      "containers": [{
        "name": "vox-adapter-smoke",
        "image": "'"$VOX_SMOKE_IMAGE"'",
        "command": ["sleep", "infinity"],
        "resources": {
          "limits": {"nvidia.com/gpu": "1"}
        },
        "volumeMounts": [{
          "name": "vox-home",
          "mountPath": "/home/vox/.vox"
        }]
      }],
      "volumes": [{
        "name": "vox-home",
        "persistentVolumeClaim": {"claimName": "'"$VOX_SMOKE_PVC"'"}
      }]
    }
  }'
```

Wait for the pod:

```bash
kubectl -n "$VOX_SMOKE_NS" wait --for=condition=Ready pod/vox-adapter-smoke --timeout=300s
```

Inside the pod, collect runtime facts:

```bash
kubectl -n "$VOX_SMOKE_NS" exec vox-adapter-smoke -- \
  sh -lc 'python - <<PY
from vox.core.runtime import detect_runtime_capabilities
print(detect_runtime_capabilities())
PY'
```

Pull and smoke one model:

```bash
export MODEL='dia-tts:1.6b'

kubectl -n "$VOX_SMOKE_NS" exec vox-adapter-smoke -- \
  sh -lc "time vox pull '$MODEL'"

kubectl -n "$VOX_SMOKE_NS" exec vox-adapter-smoke -- \
  sh -lc "python - <<'PY'
from importlib.metadata import version
for package in ('vox-cosyvoice', 'vox-dia', 'vox-orpheus', 'vox-indextts'):
    try:
        print(f'{package}=={version(package)}')
    except Exception as exc:
        print(f'{package}: not installed ({exc})')
PY"

kubectl -n "$VOX_SMOKE_NS" exec vox-adapter-smoke -- \
  sh -lc "time vox run '$MODEL' 'This is a short expressive smoke test.' --output /tmp/short.wav"

kubectl -n "$VOX_SMOKE_NS" exec vox-adapter-smoke -- \
  sh -lc "time vox run '$MODEL' 'This is a longer smoke test. It should produce stable speech, preserve the requested voice behavior, and finish without leaking memory or exhausting the GPU.' --output /tmp/long.wav"

kubectl -n "$VOX_SMOKE_NS" exec vox-adapter-smoke -- \
  sh -lc '
    for item in short:/tmp/short.wav long:/tmp/long.wav; do
      label="${item%%:*}"
      path="${item#*:}"
      if [ -f "$path" ]; then
        duration="$(ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$path" 2>&1)" || duration="error: $duration"
        printf "%s=%s\n" "$label" "$duration"
      else
        printf "%s=missing\n" "$label"
      fi
    done
  '
```

Copy artifacts out before cleanup:

```bash
mkdir -p /tmp/vox-adapter-smoke
kubectl -n "$VOX_SMOKE_NS" cp vox-adapter-smoke:/tmp/short.wav /tmp/vox-adapter-smoke/${MODEL//[:\/]/-}-short.wav
kubectl -n "$VOX_SMOKE_NS" cp vox-adapter-smoke:/tmp/long.wav /tmp/vox-adapter-smoke/${MODEL//[:\/]/-}-long.wav
```

Record GPU and pod memory while each synthesis command runs:

```bash
kubectl top pod -n "$VOX_SMOKE_NS" vox-adapter-smoke
kubectl -n "$VOX_SMOKE_NS" exec vox-adapter-smoke -- nvidia-smi
```

For `indextts-tts:2`, pass reference audio through the API or use an existing
test voice WAV inside the disposable PVC. The adapter is expected to reject
requests without reference audio or a voice-path prompt.

The scripted path can pass that same disposable voice WAV through `vox run`:

```bash
bash scripts/expressive-adapter-smoke.sh --model indextts-tts:2 --voice /home/vox/.vox/smoke-voices/reference.wav
```

## Evidence Record

Store one evidence file per model next to the copied WAV artifacts. Do not keep
the only record in chat logs or terminal scrollback.

```bash
cat > /tmp/vox-adapter-smoke/${MODEL//[:\/]/-}-evidence.md <<'EOF'
# Expressive Adapter Smoke Evidence

Model:
Variant:
Accelerator request:
Voice:
Manual audio usable:
Image tag:
Image digest:
Adapter package:
Expected adapter package:
Registry entry:
Runtime capability snapshot:

## Voice Reference

Voice value:
Voice path check:
Voice path exists:

## Model Resolution

Registry entry:
Resolved variant:
Preferred backend:
Manifest path:
Model store path:
Runtime root:
Manifest exists:

## Adapter Packages

Resolved packages:

## Pre-Pull Clean State

Manifest path:
Manifest exists:
Model store path:
Model store path exists:
Runtime paths:
Dirty:

## Pull

Command:
Exit status:
Duration:
Output summary:
Used VOX_ALLOW_INCOMPATIBLE: no

## Short Synthesis

Command:
Exit status:
Wall time:
Output path:
Output bytes:
Audio duration:
Audio usable: yes/no

## Long Synthesis

Command:
Exit status:
Wall time:
Output path:
Output bytes:
Audio duration:
Audio usable: yes/no

## Resource Snapshot After Pull

Pod memory:
GPU memory:

## Storage Snapshot After Pull

Filesystem:
Adapter package storage:
Runtime storage:
Model storage:
Manifest storage:
Blob storage:

## Resource Snapshot After Short Synthesis

Pod memory:
GPU memory:

## Resource Snapshot After Long Synthesis

Pod memory:
GPU memory:

## Audio Durations

Short:
Long:

## Audio Stream Metadata

Short:
Long:

## Audio Signal Stats

Short:
Long:

## Copied Artifact Stats

Short WAV bytes:
Short WAV sha256:
Long WAV bytes:
Long WAV sha256:

## Smoke Result

Result:
Failed steps:

## Classification

Result: pass/fail
Failure class: Vox / adapter / dependency / upstream / hardware / none
Exact error:
Notes:
EOF
```

If a model fails, fill in the same record instead of deleting it. The failure
record is the evidence needed to decide whether the next fix belongs in Vox,
the adapter package, the remote registry, upstream code, or the hardware
classification.

## Cleanup

Only delete the disposable smoke resources:

```bash
kubectl -n "$VOX_SMOKE_NS" delete pod vox-adapter-smoke --ignore-not-found
kubectl delete namespace "$VOX_SMOKE_NS"
```

Do not delete or modify production voice data under `$VOX_HOME/voices`.

## Completion Criteria

For each target model, mark the model as verified only when all of these are
true:

1. `vox pull` succeeds in the isolated clean-pull environment without `VOX_ALLOW_INCOMPATIBLE`.
2. Runtime dependencies are installed under `$VOX_HOME/runtime/<adapter>`.
3. Model files are stored in the model store, not in the adapter package or base image.
4. Short and long synthesis both return non-empty WAV files.
5. Output durations are readable, positive, and plausible for the text length;
   the smoke helpers fail a long output that is shorter than the short output.
6. RAM/VRAM evidence is recorded. Existing-server smoke records
   `/v1/system/memory` before and after synthesis plus per-request memory
   sample summaries during short and long synthesis; local Docker clean-pull
   smoke records container RAM and optional `nvidia-smi` VRAM snapshots before
   and after smoke, plus continuous sample summaries with peak observed RAM
   and GPU memory for pull, short synthesis, and long synthesis.
7. Audio is manually judged usable.
8. Failures, if any, are classified as Vox, adapter, dependency, upstream, or hardware
   and include a concrete failure note with the likely cause or next fix.
