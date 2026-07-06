#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  expressive-adapter-smoke.sh --model MODEL [options]

Required:
  --model MODEL              Model reference to smoke, e.g. dia-tts:1.6b

Options:
  --variant VARIANT          Force a pull-time variant such as onnx or cuda
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
VARIANT=""
CREATE=0
NS="${VOX_SMOKE_NS:-vox-adapter-smoke}"
PVC="${VOX_SMOKE_PVC:-vox-adapter-smoke-data}"
IMAGE="${VOX_SMOKE_IMAGE:-ghcr.io/eleven-am/vox:latest}"
STORAGE="${VOX_SMOKE_STORAGE:-40Gi}"
OUTPUT_DIR="${VOX_SMOKE_OUTPUT_DIR:-/tmp/vox-adapter-smoke}"
SHORT_TEXT="This is a short expressive smoke test."
LONG_TEXT="This is a longer smoke test. It should produce stable speech, preserve the requested voice behavior, and finish without leaking memory or exhausting the GPU."
POD="vox-adapter-smoke"
FAILED=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --variant)
      VARIANT="${2:-}"
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
if [[ -n "$VARIANT" ]]; then
  safe_model="${safe_model}-${VARIANT//[:\/]/-}"
fi
mkdir -p "$OUTPUT_DIR"
evidence="$OUTPUT_DIR/${safe_model}-evidence.md"
short_wav="$OUTPUT_DIR/${safe_model}-short.wav"
long_wav="$OUTPUT_DIR/${safe_model}-long.wav"
pull_command="vox pull $MODEL"
if [[ -n "$VARIANT" ]]; then
  pull_command="$pull_command --variant $VARIANT"
fi

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
Variant: ${VARIANT:-auto}
Image tag: $IMAGE
Image digest: $image_id
Adapter package:
Registry entry:
Runtime capability snapshot:
$capabilities

## Pull

Command: $pull_command
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

record_timed() {
  local label="$1"
  shift
  if ! run_timed "$label" "$@"; then
    FAILED=1
  fi
}

record_timed "Pull Output" kubectl -n "$NS" exec "$POD" -- \
  env MODEL_REF="$MODEL" VARIANT_REF="$VARIANT" sh -lc '
    if [ -n "$VARIANT_REF" ]; then
      vox pull "$MODEL_REF" --variant "$VARIANT_REF"
    else
      vox pull "$MODEL_REF"
    fi
  '

model_resolution="$(kubectl -n "$NS" exec "$POD" -- env MODEL_REF="$MODEL" VARIANT_REF="$VARIANT" sh -lc "python - <<'PY'
import json
import os
from pathlib import Path

from vox.core.model_resolution import resolve_catalog_entry
from vox.core.registry import ModelRegistry
from vox.core.store import BlobStore
from vox.operations.models import (
    ModelReferenceRequest,
    resolve_model_reference,
)

store = BlobStore(Path(os.environ.get('VOX_HOME', str(Path.home() / '.vox'))))
registry = ModelRegistry(store)
request = ModelReferenceRequest(
    name=os.environ['MODEL_REF'],
    variant=os.environ.get('VARIANT_REF') or None,
)
payload = {'requested': request.name, 'requested_variant': request.variant}

try:
    resolved = resolve_model_reference(registry=registry, request=request)
    payload['resolved_reference'] = {
        'parsed_name': resolved.parsed_name,
        'parsed_tag': resolved.parsed_tag,
        'resolved_name': resolved.resolved_name,
        'resolved_tag': resolved.resolved_tag,
        'explicit_tag': resolved.explicit_tag,
        'requested_variant': resolved.requested_variant,
    }

    entry = registry.lookup(
        resolved.parsed_name,
        resolved.parsed_tag,
        explicit_tag=resolved.explicit_tag,
    )
    payload['registry_entry'] = entry
    if entry:
        variant = resolve_catalog_entry(entry, forced_variant=resolved.requested_variant)
        payload['resolved_variant'] = variant.variant_id
        payload['preferred_backend'] = variant.preferred_backend
        payload['variant_missing'] = list(variant.missing)
        payload['variant_warnings'] = list(variant.warnings)
        payload['concrete_entry'] = variant.entry

    manifest = store.resolve_model(resolved.resolved_name, resolved.resolved_tag)
    payload['manifest_path'] = str(
        store.manifests_dir / resolved.resolved_name / resolved.resolved_tag
    )
    payload['model_link_path'] = str(
        store.root / 'models' / 'links' / resolved.resolved_name / resolved.resolved_tag
    )
    payload['runtime_root'] = str(store.root / 'runtime')
    payload['manifest_exists'] = manifest is not None
    if manifest:
        payload['manifest_config'] = manifest.config
        payload['manifest_layers'] = [
            {
                'filename': layer.filename,
                'size': layer.size,
                'media_type': layer.media_type,
                'digest': layer.digest,
            }
            for layer in manifest.layers
        ]
except Exception as exc:
    payload['error'] = f'{type(exc).__name__}: {exc}'

print(json.dumps(payload, indent=2, sort_keys=True))
PY" 2>&1 || true)"
append_section "Model Resolution" "$model_resolution"

packages="$(kubectl -n "$NS" exec "$POD" -- sh -lc "python - <<'PY'
from importlib.metadata import version
for package in ('vox-cosyvoice', 'vox-dia', 'vox-orpheus', 'vox-indextts'):
    try:
        print(f'{package}=={version(package)}')
    except Exception as exc:
        print(f'{package}: not installed ({exc})')
PY")"
append_section "Adapter Packages" "$packages"

record_timed "Short Synthesis Output" \
  kubectl -n "$NS" exec "$POD" -- sh -lc "vox run '$MODEL' '$SHORT_TEXT' --output /tmp/short.wav"
record_timed "Long Synthesis Output" \
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

if [[ "$FAILED" != "0" ]]; then
  echo "one or more smoke steps failed; inspect evidence: $evidence" >&2
  exit 5
fi
