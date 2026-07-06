#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  expressive-adapter-smoke.sh --model MODEL [options]

Required:
  --model MODEL              Model reference to smoke, e.g. dia-tts:1.6b

Options:
  --create                   Create the disposable namespace/PVC if missing
  --namespace NAME           Disposable namespace (default: vox-adapter-smoke)
  --pvc NAME                 Disposable PVC (default: vox-adapter-smoke-data)
  --image IMAGE              Vox image (default: ghcr.io/eleven-am/vox:latest)
  --storage SIZE             PVC storage request when --create is used (default: 40Gi)
  --output-dir DIR           Local evidence/artifact directory (default: /tmp/vox-adapter-smoke)
  --short TEXT               Short synthesis text
  --long TEXT                Long synthesis text
  --help                     Show this help

Safety:
  This script refuses the production namespace/PVC names: vox and vox-data.
  It only creates Kubernetes resources when --create is passed explicitly.
EOF
}

MODEL=""
CREATE=0
NS="${VOX_SMOKE_NS:-vox-adapter-smoke}"
PVC="${VOX_SMOKE_PVC:-vox-adapter-smoke-data}"
IMAGE="${VOX_SMOKE_IMAGE:-ghcr.io/eleven-am/vox:latest}"
STORAGE="${VOX_SMOKE_STORAGE:-40Gi}"
OUTPUT_DIR="${VOX_SMOKE_OUTPUT_DIR:-/tmp/vox-adapter-smoke}"
SHORT_TEXT="This is a short expressive smoke test."
LONG_TEXT="This is a longer smoke test. It should produce stable speech, preserve the requested voice behavior, and finish without leaking memory or exhausting the GPU."
POD="vox-adapter-smoke"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --create)
      CREATE=1
      shift
      ;;
    --namespace)
      NS="${2:-}"
      shift 2
      ;;
    --pvc)
      PVC="${2:-}"
      shift 2
      ;;
    --image)
      IMAGE="${2:-}"
      shift 2
      ;;
    --storage)
      STORAGE="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --short)
      SHORT_TEXT="${2:-}"
      shift 2
      ;;
    --long)
      LONG_TEXT="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL" ]]; then
  echo "--model is required" >&2
  usage >&2
  exit 2
fi

if [[ "$NS" == "vox" || "$PVC" == "vox-data" ]]; then
  echo "refusing to use production Vox namespace/PVC: namespace=$NS pvc=$PVC" >&2
  exit 3
fi

safe_model="${MODEL//[:\/]/-}"
mkdir -p "$OUTPUT_DIR"
evidence="$OUTPUT_DIR/${safe_model}-evidence.md"
short_wav="$OUTPUT_DIR/${safe_model}-short.wav"
long_wav="$OUTPUT_DIR/${safe_model}-long.wav"

if ! kubectl get namespace "$NS" >/dev/null 2>&1; then
  if [[ "$CREATE" != "1" ]]; then
    echo "namespace $NS does not exist; rerun with --create after approving disposable resources" >&2
    exit 4
  fi
  kubectl create namespace "$NS"
fi

if ! kubectl -n "$NS" get pvc "$PVC" >/dev/null 2>&1; then
  if [[ "$CREATE" != "1" ]]; then
    echo "pvc $NS/$PVC does not exist; rerun with --create after approving disposable resources" >&2
    exit 4
  fi
  kubectl -n "$NS" apply -f - <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ${PVC}
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: ${STORAGE}
EOF
fi

if ! kubectl -n "$NS" get pod "$POD" >/dev/null 2>&1; then
  if [[ "$CREATE" != "1" ]]; then
    echo "pod $NS/$POD does not exist; rerun with --create after approving disposable resources" >&2
    exit 4
  fi
  kubectl -n "$NS" run "$POD" \
    --image="$IMAGE" \
    --restart=Never \
    --overrides='{
      "spec": {
        "containers": [{
          "name": "vox-adapter-smoke",
          "image": "'"$IMAGE"'",
          "command": ["sleep", "infinity"],
          "resources": {"limits": {"nvidia.com/gpu": "1"}},
          "volumeMounts": [{"name": "vox-home", "mountPath": "/home/vox/.vox"}]
        }],
        "volumes": [{"name": "vox-home", "persistentVolumeClaim": {"claimName": "'"$PVC"'"}}]
      }
    }'
fi

kubectl -n "$NS" wait --for=condition=Ready "pod/$POD" --timeout=300s

image_id="$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.containerStatuses[0].imageID}')"
capabilities="$(kubectl -n "$NS" exec "$POD" -- sh -lc 'python - <<PY
from vox.core.runtime import detect_runtime_capabilities
print(detect_runtime_capabilities())
PY')"

cat > "$evidence" <<EOF
# Expressive Adapter Smoke Evidence

Model: $MODEL
Image tag: $IMAGE
Image digest: $image_id
Adapter package:
Registry entry:
Runtime capability snapshot:
$capabilities

## Pull

Command: vox pull $MODEL
Exit status:
Duration:
Output summary:
Used VOX_ALLOW_INCOMPATIBLE: no
Runtime directory:
Model store path:

## Short Synthesis

Command: vox run $MODEL "$SHORT_TEXT" --output /tmp/short.wav
Exit status:
Wall time:
Output path: $short_wav
Output bytes:
Audio duration:
Peak pod memory:
Peak GPU memory:
Audio usable: yes/no

## Long Synthesis

Command: vox run $MODEL "$LONG_TEXT" --output /tmp/long.wav
Exit status:
Wall time:
Output path: $long_wav
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

append_section() {
  local title="$1"
  local body="$2"
  {
    echo
    echo "## $title"
    echo
    echo '```text'
    printf '%s\n' "$body"
    echo '```'
  } >> "$evidence"
}

run_timed() {
  local label="$1"
  shift
  set +e
  local output
  output="$({ /usr/bin/time -p "$@"; } 2>&1)"
  local status=$?
  set -e
  append_section "$label" "exit=$status"$'\n'"$output"
  return "$status"
}

run_timed "Pull Output" kubectl -n "$NS" exec "$POD" -- sh -lc "vox pull '$MODEL'"

packages="$(kubectl -n "$NS" exec "$POD" -- sh -lc "python - <<'PY'
from importlib.metadata import version
for package in ('vox-cosyvoice', 'vox-dia', 'vox-orpheus', 'vox-indextts'):
    try:
        print(f'{package}=={version(package)}')
    except Exception as exc:
        print(f'{package}: not installed ({exc})')
PY")"
append_section "Adapter Packages" "$packages"

run_timed "Short Synthesis Output" \
  kubectl -n "$NS" exec "$POD" -- sh -lc "vox run '$MODEL' '$SHORT_TEXT' --output /tmp/short.wav"
run_timed "Long Synthesis Output" \
  kubectl -n "$NS" exec "$POD" -- sh -lc "vox run '$MODEL' '$LONG_TEXT' --output /tmp/long.wav"

durations="$(kubectl -n "$NS" exec "$POD" -- sh -lc 'ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 /tmp/short.wav /tmp/long.wav' 2>&1 || true)"
append_section "Audio Durations" "$durations"

resources="$(kubectl top pod -n "$NS" "$POD" 2>&1 || true; kubectl -n "$NS" exec "$POD" -- nvidia-smi 2>&1 || true)"
append_section "Resource Snapshot" "$resources"

kubectl -n "$NS" cp "$POD:/tmp/short.wav" "$short_wav" >/dev/null 2>&1 || true
kubectl -n "$NS" cp "$POD:/tmp/long.wav" "$long_wav" >/dev/null 2>&1 || true

echo "wrote evidence: $evidence"
echo "copied artifacts when available:"
echo "  $short_wav"
echo "  $long_wav"
