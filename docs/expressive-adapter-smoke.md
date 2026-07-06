# Expressive Adapter Smoke Validation

This runbook is the final verification gate for expressive TTS adapters that
need GPU-backed runtime dependencies. It is intentionally separate from the
live Vox deployment.

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

The isolated smoke pod must resolve adapter packages from PyPI, not from a
local source tree or a patched live cluster directory. Verify the installed
versions before marking a model as smoke-tested:

| Model | Expected adapter package |
| --- | --- |
| `cosyvoice2-tts:0.5b` | `vox-cosyvoice==0.1.6` |
| `dia-tts:1.6b` | `vox-dia==0.2.12` |
| `orpheus-tts:medium-3b` | `vox-orpheus==0.1.6` |
| `indextts-tts:2` | `vox-indextts==0.1.14` |

If the registry points at a newer package, record that newer version in the
smoke evidence and update this table with the same change.

## Safety Rules

Do not use the production `vox` namespace or production `vox-data` PVC for this
work.

Create a separate namespace and disposable PVC:

```bash
export VOX_SMOKE_NS=vox-adapter-smoke
export VOX_SMOKE_PVC=vox-adapter-smoke-data
export VOX_SMOKE_IMAGE=ghcr.io/eleven-am/vox:latest
```

If the cluster does not already have an isolated namespace/PVC, stop and get
approval before creating one. Do not mutate, clean, reinstall, restart, or scale
the live deployment as part of adapter smoke testing.

Before running any smoke command, do a read-only preflight against the exact
disposable names:

```bash
kubectl get namespace "$VOX_SMOKE_NS"
kubectl -n "$VOX_SMOKE_NS" get pvc "$VOX_SMOKE_PVC"
kubectl -n "$VOX_SMOKE_NS" get pod vox-adapter-smoke
```

If the namespace or PVC is missing, stop and ask before creating it. Never
substitute `vox`, `vox-data`, or any production pod/PVC to "just test quickly".
The preflight is allowed to fail with `NotFound`; that means the isolated smoke
environment does not exist yet.

The scripted path is:

```bash
bash scripts/expressive-adapter-smoke.sh --model dia-tts:1.6b --audio-usable yes
```

To force a pull-time hardware/backend variant for models that publish multiple
variants, pass the same variant flag that `vox pull` accepts:

```bash
bash scripts/expressive-adapter-smoke.sh --model chatterbox-tts-turbo:0.1.7 --variant onnx --audio-usable yes
```

For CPU/ONNX validation, explicitly create the disposable pod without a GPU
request so the result proves that the model can run without CUDA:

```bash
bash scripts/expressive-adapter-smoke.sh --model chatterbox-tts-turbo:0.1.7 --variant onnx --cpu-only --audio-usable yes
```

The equivalent Makefile entrypoint is:

```bash
make smoke-expressive MODEL=dia-tts:1.6b SMOKE_AUDIO_USABLE=yes
make smoke-expressive MODEL=chatterbox-tts-turbo:0.1.7 SMOKE_VARIANT=onnx SMOKE_AUDIO_USABLE=yes
make smoke-expressive MODEL=chatterbox-tts-turbo:0.1.7 SMOKE_VARIANT=onnx SMOKE_CPU_ONLY=1 SMOKE_AUDIO_USABLE=yes
```

For cloning adapters, copy a small test-only reference WAV into the disposable
PVC and pass it as the Vox voice path. Do not use production voice data:

```bash
bash scripts/expressive-adapter-smoke.sh --model indextts-tts:2 --voice /home/vox/.vox/smoke-voices/reference.wav --audio-usable yes
make smoke-expressive MODEL=indextts-tts:2 SMOKE_VOICE=/home/vox/.vox/smoke-voices/reference.wav SMOKE_AUDIO_USABLE=yes
```

The script refuses the production `vox` namespace and `vox-data` PVC. It also
refuses to create the disposable namespace, PVC, or pod unless `--create` is
passed explicitly after approval:

```bash
bash scripts/expressive-adapter-smoke.sh --model dia-tts:1.6b --create --audio-usable yes
make smoke-expressive MODEL=dia-tts:1.6b SMOKE_CREATE=1 SMOKE_AUDIO_USABLE=yes
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
- `vox pull <model>` output
- machine-readable `real` durations for pull, short synthesis, and long synthesis
- adapter, runtime, model, manifest, and blob storage usage after pull
- short synthesis wall time
- long synthesis wall time
- generated audio duration
- generated audio stream metadata, including codec, sample rate, and channels
- generated audio signal stats proving the WAV is not silent
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

1. `vox pull` succeeds in the isolated pod without `VOX_ALLOW_INCOMPATIBLE`.
2. Runtime dependencies are installed under `$VOX_HOME/runtime/<adapter>`.
3. Model files are stored in the model store, not in the adapter package or base image.
4. Short and long synthesis both return non-empty WAV files.
5. Output durations are plausible for the text length.
6. Peak memory and VRAM fit the documented registry limits.
7. Audio is manually judged usable.
8. Failures, if any, are classified as Vox, adapter, dependency, upstream, or hardware.
