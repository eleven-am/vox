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
| `indextts-tts:2` | `vox-indextts==0.1.7` |

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

## Required Evidence

For each model, capture:

- image tag and digest
- adapter package version resolved from PyPI
- registry entry used
- runtime capability snapshot from the pod
- `vox pull <model>` output
- short synthesis wall time
- long synthesis wall time
- generated audio duration
- peak pod memory
- peak GPU memory while loaded and while synthesizing
- output WAV artifact
- whether the audio is usable
- exact failure output if any step fails

Use a short text around one sentence and a longer text around one paragraph.
For cloning adapters, provide a small reference WAV copied into the disposable
PVC or mounted as test-only data.

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
  sh -lc "ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 /tmp/short.wav /tmp/long.wav"
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

## Evidence Record

Store one evidence file per model next to the copied WAV artifacts. Do not keep
the only record in chat logs or terminal scrollback.

```bash
cat > /tmp/vox-adapter-smoke/${MODEL//[:\/]/-}-evidence.md <<'EOF'
# Expressive Adapter Smoke Evidence

Model:
Image tag:
Image digest:
Adapter package:
Registry entry:
Runtime capability snapshot:

## Pull

Command:
Exit status:
Duration:
Output summary:
Used VOX_ALLOW_INCOMPATIBLE: no
Runtime directory:
Model store path:

## Short Synthesis

Command:
Exit status:
Wall time:
Output path:
Output bytes:
Audio duration:
Peak pod memory:
Peak GPU memory:
Audio usable: yes/no

## Long Synthesis

Command:
Exit status:
Wall time:
Output path:
Output bytes:
Audio duration:
Peak pod memory:
Peak GPU memory:
Audio usable: yes/no

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
