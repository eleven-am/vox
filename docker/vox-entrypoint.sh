#!/bin/sh
set -eu

VOX_HOME_DIR="${VOX_HOME:-/home/vox/.vox}"
VOX_CACHE_DIR="${XDG_CACHE_HOME:-$VOX_HOME_DIR/cache}"
UV_CACHE_PATH="${UV_CACHE_DIR:-$VOX_CACHE_DIR/uv}"
HF_HOME_DIR="${HF_HOME:-$VOX_CACHE_DIR/huggingface}"
HF_HUB_CACHE_DIR="${HUGGINGFACE_HUB_CACHE:-$HF_HOME_DIR/hub}"
HF_XET_CACHE_DIR="${HF_XET_CACHE:-$HF_HOME_DIR/xet}"
PIP_CACHE_PATH="${PIP_CACHE_DIR:-$VOX_CACHE_DIR/pip}"
TMP_PATH="${TMPDIR:-$VOX_HOME_DIR/tmp}"

export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE_DIR"
export HF_XET_CACHE="$HF_XET_CACHE_DIR"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export XDG_CACHE_HOME="$VOX_CACHE_DIR"
export UV_CACHE_DIR="$UV_CACHE_PATH"
export PIP_CACHE_DIR="$PIP_CACHE_PATH"
export TMPDIR="$TMP_PATH"

install -d \
  "$VOX_HOME_DIR" \
  "$VOX_HOME_DIR/adapters" \
  "$VOX_HOME_DIR/models" \
  "$VOX_HOME_DIR/models/blobs" \
  "$VOX_HOME_DIR/models/manifests" \
  "$VOX_HOME_DIR/models/manifests/library" \
  "$VOX_HOME_DIR/tmp" \
  "$VOX_CACHE_DIR" \
  "$PIP_CACHE_PATH" \
  "$HF_HOME_DIR" \
  "$HF_HUB_CACHE_DIR" \
  "$HF_XET_CACHE_DIR" \
  "$HF_XET_CACHE_DIR/logs" \
  "$HF_XET_CACHE_DIR/chunk-cache" \
  "$HF_XET_CACHE_DIR/shard-cache" \
  "$VOX_CACHE_DIR/torch" \
  "$VOX_CACHE_DIR/torch/hub" \
  "$UV_CACHE_PATH" \
  "$TMP_PATH"

chown vox:vox \
  "$VOX_HOME_DIR" \
  "$VOX_HOME_DIR/adapters" \
  "$VOX_HOME_DIR/models" \
  "$VOX_HOME_DIR/models/blobs" \
  "$VOX_HOME_DIR/models/manifests" \
  "$VOX_HOME_DIR/models/manifests/library" \
  "$VOX_HOME_DIR/tmp" \
  "$VOX_CACHE_DIR" \
  "$PIP_CACHE_PATH" \
  "$HF_HOME_DIR" \
  "$HF_HUB_CACHE_DIR" \
  "$HF_XET_CACHE_DIR" \
  "$HF_XET_CACHE_DIR/logs" \
  "$HF_XET_CACHE_DIR/chunk-cache" \
  "$HF_XET_CACHE_DIR/shard-cache" \
  "$VOX_CACHE_DIR/torch" \
  "$VOX_CACHE_DIR/torch/hub" \
  "$UV_CACHE_PATH" \
  "$TMP_PATH"

exec gosu vox "$@"
