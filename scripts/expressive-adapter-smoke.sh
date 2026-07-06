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
  --cpu-only                 Create the disposable pod without requesting a GPU
  --create                   Create the disposable namespace/PVC if missing
  --namespace NAME           Disposable namespace (default: vox-adapter-smoke)
  --pvc NAME                 Disposable PVC (default: vox-adapter-smoke-data)
  --image IMAGE              Vox image (default: ghcr.io/eleven-am/vox:latest)
  --storage SIZE             PVC storage request when --create is used (default: 40Gi)
  --output-dir DIR           Local evidence/artifact directory (default: /tmp/vox-adapter-smoke)
  --short TEXT               Short synthesis text
  --long TEXT                Long synthesis text
  --voice VOICE              Voice id or WAV path to pass to vox run
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
GPU="${VOX_SMOKE_GPU:-1}"
STORAGE="${VOX_SMOKE_STORAGE:-40Gi}"
OUTPUT_DIR="${VOX_SMOKE_OUTPUT_DIR:-/tmp/vox-adapter-smoke}"
SHORT_TEXT="This is a short expressive smoke test."
LONG_TEXT="This is a longer smoke test. It should produce stable speech, preserve the requested voice behavior, and finish without leaking memory or exhausting the GPU."
VOICE=""
POD="vox-adapter-smoke"
FAILED=0
FAILED_STEPS=()

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
    --cpu-only)
      GPU=0
      shift
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
    --voice)
      VOICE="${2:-}"
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
accelerator_request="gpu"
resources_json='{"limits": {"nvidia.com/gpu": "1"}}'
if [[ "$GPU" == "0" ]]; then
  accelerator_request="cpu-only"
  resources_json='{}'
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
          "resources": '"$resources_json"',
          "volumeMounts": [{"name": "vox-home", "mountPath": "/home/vox/.vox"}]
        }],
        "volumes": [{"name": "vox-home", "persistentVolumeClaim": {"claimName": "'"$PVC"'"}}]
      }
    }'
fi

existing_image="$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.spec.containers[0].image}')"
if [[ "$existing_image" != "$IMAGE" ]]; then
  echo "pod $NS/$POD already exists with image $existing_image, expected $IMAGE; delete the disposable pod or rerun with the matching image" >&2
  exit 6
fi

existing_pvc="$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.spec.volumes[?(@.name=="vox-home")].persistentVolumeClaim.claimName}')"
if [[ "$existing_pvc" != "$PVC" ]]; then
  echo "pod $NS/$POD already exists with PVC $existing_pvc, expected $PVC; delete the disposable pod or rerun with the matching PVC" >&2
  exit 6
fi

existing_limits="$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.spec.containers[0].resources.limits}' 2>/dev/null || true)"
existing_has_gpu=0
if [[ "$existing_limits" == *"nvidia.com/gpu"* ]]; then
  existing_has_gpu=1
fi
if [[ "$GPU" == "0" && "$existing_has_gpu" == "1" ]]; then
  echo "pod $NS/$POD already exists with a GPU limit, but this run requested --cpu-only; delete the disposable pod or rerun without --cpu-only" >&2
  exit 6
fi
if [[ "$GPU" != "0" && "$existing_has_gpu" != "1" ]]; then
  echo "pod $NS/$POD already exists without a GPU limit, but this run requested GPU validation; delete the disposable pod or rerun with --cpu-only" >&2
  exit 6
fi

kubectl -n "$NS" wait --for=condition=Ready "pod/$POD" --timeout=300s

image_id="$(kubectl -n "$NS" get pod "$POD" -o jsonpath='{.status.containerStatuses[0].imageID}')"
capabilities="$(kubectl -n "$NS" exec "$POD" -- sh -lc 'python - <<PY
from vox.core.runtime import detect_runtime_capabilities
print(detect_runtime_capabilities())
PY')"

allow_incompatible="$(kubectl -n "$NS" exec "$POD" -- sh -lc 'printenv VOX_ALLOW_INCOMPATIBLE || true' 2>/dev/null || true)"
allow_incompatible_normalized="$(printf '%s' "$allow_incompatible" | tr '[:upper:]' '[:lower:]')"
case "$allow_incompatible_normalized" in
  1|true|yes|on)
    FAILED=1
    FAILED_STEPS+=("VOX_ALLOW_INCOMPATIBLE is enabled in the smoke pod")
    ;;
esac
allow_incompatible_evidence="${allow_incompatible:-no}"

voice_path_check="not-applicable"
voice_path_exists="not-applicable"
if [[ -n "$VOICE" ]]; then
  case "$VOICE" in
    /*|./*|../*|*/*)
      voice_path_check="file"
      if kubectl -n "$NS" exec "$POD" -- test -f "$VOICE" >/dev/null 2>&1; then
        voice_path_exists="yes"
      else
        voice_path_exists="no"
        FAILED=1
        FAILED_STEPS+=("Voice reference path missing: $VOICE")
      fi
      ;;
    *)
      voice_path_check="voice-id"
      ;;
  esac
fi

cat > "$evidence" <<EOF
# Expressive Adapter Smoke Evidence

Model: $MODEL
Variant: ${VARIANT:-auto}
Accelerator request: $accelerator_request
Voice: ${VOICE:-none}
Image tag: $IMAGE
Image digest: $image_id
Adapter package:
Registry entry:
Runtime capability snapshot:
$capabilities

## Voice Reference

Voice value: ${VOICE:-none}
Voice path check: $voice_path_check
Voice path exists: $voice_path_exists

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

Command: $pull_command
Exit status:
Duration:
Output summary:
Used VOX_ALLOW_INCOMPATIBLE: $allow_incompatible_evidence

## Short Synthesis

Command: vox run $MODEL "$SHORT_TEXT" ${VOICE:+--voice $VOICE }--output /tmp/short.wav
Exit status:
Wall time:
Output path: $short_wav
Output bytes:
Audio duration:
Audio usable: yes/no

## Long Synthesis

Command: vox run $MODEL "$LONG_TEXT" ${VOICE:+--voice $VOICE }--output /tmp/long.wav
Exit status:
Wall time:
Output path: $long_wav
Output bytes:
Audio duration:
Audio usable: yes/no

## Resource Snapshot After Pull

Pod memory:
GPU memory:

## Resource Snapshot After Short Synthesis

Pod memory:
GPU memory:

## Resource Snapshot After Long Synthesis

Pod memory:
GPU memory:

## Audio Durations

Short:
Long:

## Copied Artifact Stats

Short WAV bytes:
Long WAV bytes:

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

record_resources() {
  local label="$1"
  local resources
  resources="$(kubectl top pod -n "$NS" "$POD" 2>&1 || true; kubectl -n "$NS" exec "$POD" -- nvidia-smi 2>&1 || true)"
  append_section "$label" "$resources"
}

record_artifact_stats() {
  local label="$1"
  shift
  local body=""
  local path
  for path in "$@"; do
    if [[ -f "$path" ]]; then
      body+="$path bytes=$(stat -f%z "$path" 2>/dev/null || stat -c%s "$path" 2>/dev/null || echo unknown)"$'\n'
    else
      body+="$path missing"$'\n'
    fi
  done
  append_section "$label" "$body"
}

validate_copied_artifacts() {
  local path
  for path in "$@"; do
    if [[ ! -s "$path" ]]; then
      FAILED=1
      FAILED_STEPS+=("Missing or empty copied artifact: $path")
    fi
  done
}

record_audio_durations() {
  local body
  body="$(kubectl -n "$NS" exec "$POD" -- sh -lc '
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
  ' 2>&1 || true)"
  append_section "Audio Durations" "$body"
  validate_audio_durations "$body"
}

validate_audio_durations() {
  local body="$1"
  local label duration
  while IFS='=' read -r label duration; do
    case "$label" in
      short|long)
        if [[ "$duration" == missing || "$duration" == error:* ]]; then
          FAILED=1
          FAILED_STEPS+=("Missing or invalid $label audio duration: $duration")
          continue
        fi
        if [[ ! "$duration" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk "BEGIN { exit !($duration > 0) }"; then
          FAILED=1
          FAILED_STEPS+=("Non-positive $label audio duration: $duration")
        fi
        ;;
    esac
  done <<< "$body"
}

validate_model_resolution() {
  local body="$1"
  if grep -q '"error":' <<< "$body"; then
    FAILED=1
    FAILED_STEPS+=("Model resolution reported an error")
  fi
  if ! grep -q '"manifest_exists": true' <<< "$body"; then
    FAILED=1
    FAILED_STEPS+=("Model manifest missing after pull")
  fi
  if ! grep -q '"manifest_layers":' <<< "$body"; then
    FAILED=1
    FAILED_STEPS+=("Model manifest layers missing after pull")
  fi
  if ! grep -q '"adapter_package": "[^"]' <<< "$body"; then
    FAILED=1
    FAILED_STEPS+=("Model adapter package missing from resolved entry")
  fi
  if grep -q '"adapter_package_installed": false' <<< "$body"; then
    FAILED=1
    FAILED_STEPS+=("Resolved adapter package is not installed")
  fi
  if grep -q '"adapter_runtime_missing": true' <<< "$body"; then
    FAILED=1
    FAILED_STEPS+=("Expected adapter runtime path is missing after pull")
  fi
  if grep -q '"adapter_runtime_empty": true' <<< "$body"; then
    FAILED=1
    FAILED_STEPS+=("Expected adapter runtime path has no adapter-owned contents")
  fi
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
    FAILED_STEPS+=("$label")
  fi
}

record_smoke_result() {
  local body
  if [[ "$FAILED" == "0" ]]; then
    body="result=pass"$'\n'"failed_steps=none"
  else
    body="result=fail"$'\n'"failed_steps:"$'\n'"$(printf '%s\n' "${FAILED_STEPS[@]}")"
  fi
  append_section "Smoke Result" "$body"
}

record_timed "Pull Output" kubectl -n "$NS" exec "$POD" -- \
  env MODEL_REF="$MODEL" VARIANT_REF="$VARIANT" sh -lc '
    if [ -n "$VARIANT_REF" ]; then
      vox pull "$MODEL_REF" --variant "$VARIANT_REF"
    else
      vox pull "$MODEL_REF"
    fi
  '
record_resources "Resource Snapshot After Pull"

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
        concrete_entry = variant.entry or {}
        adapter_package = concrete_entry.get('adapter_package') or ''
        adapter_package_version = (
            registry.adapter_resolver.installed_version(adapter_package)
            if adapter_package
            else None
        )
        payload['adapter_package'] = adapter_package
        payload['adapter_package_version'] = adapter_package_version
        payload['adapter_package_installed'] = bool(
            not adapter_package or adapter_package_version
        )
        expected_runtime_names = {
            'vox-cosyvoice': ['cosyvoice'],
            'vox-dia': ['dia'],
            'vox-orpheus': ['orpheus'],
            'vox-indextts': ['indextts'],
        }.get(adapter_package, [])
        runtime_checks = []
        for runtime_name in expected_runtime_names:
            runtime_path = store.root / 'runtime' / runtime_name
            meaningful_entries = [
                path.name
                for path in runtime_path.iterdir()
                if path.name != '_vox_runtime_fallback_paths.pth'
            ] if runtime_path.is_dir() else []
            runtime_checks.append({
                'name': runtime_name,
                'path': str(runtime_path),
                'exists': runtime_path.is_dir(),
                'entry_count': (
                    len(list(runtime_path.iterdir()))
                    if runtime_path.is_dir()
                    else 0
                ),
                'meaningful_entry_count': len(meaningful_entries),
            })
        payload['adapter_runtime_paths'] = runtime_checks
        payload['adapter_runtime_missing'] = any(
            not check['exists'] for check in runtime_checks
        )
        payload['adapter_runtime_empty'] = any(
            check['exists'] and check['meaningful_entry_count'] == 0
            for check in runtime_checks
        )

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
validate_model_resolution "$model_resolution"

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
  kubectl -n "$NS" exec "$POD" -- \
    env MODEL_REF="$MODEL" TEXT_REF="$SHORT_TEXT" VOICE_REF="$VOICE" sh -lc '
      if [ -n "$VOICE_REF" ]; then
        vox run "$MODEL_REF" "$TEXT_REF" --voice "$VOICE_REF" --output /tmp/short.wav
      else
        vox run "$MODEL_REF" "$TEXT_REF" --output /tmp/short.wav
      fi
    '
record_resources "Resource Snapshot After Short Synthesis"
record_timed "Long Synthesis Output" \
  kubectl -n "$NS" exec "$POD" -- \
    env MODEL_REF="$MODEL" TEXT_REF="$LONG_TEXT" VOICE_REF="$VOICE" sh -lc '
      if [ -n "$VOICE_REF" ]; then
        vox run "$MODEL_REF" "$TEXT_REF" --voice "$VOICE_REF" --output /tmp/long.wav
      else
        vox run "$MODEL_REF" "$TEXT_REF" --output /tmp/long.wav
      fi
    '
record_resources "Resource Snapshot After Long Synthesis"

record_audio_durations

kubectl -n "$NS" cp "$POD:/tmp/short.wav" "$short_wav" >/dev/null 2>&1 || true
kubectl -n "$NS" cp "$POD:/tmp/long.wav" "$long_wav" >/dev/null 2>&1 || true
record_artifact_stats "Copied Artifact Stats" "$short_wav" "$long_wav"
validate_copied_artifacts "$short_wav" "$long_wav"
record_smoke_result

echo "wrote evidence: $evidence"
echo "copied artifacts when available:"
echo "  $short_wav"
echo "  $long_wav"

if [[ "$FAILED" != "0" ]]; then
  echo "one or more smoke steps failed; inspect evidence: $evidence" >&2
  exit 5
fi
