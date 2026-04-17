#!/bin/sh
set -eu

VOX_HOME_DIR="${VOX_HOME:-/home/vox/.vox}"
VOX_CACHE_DIR="${HOME:-/home/vox}/.cache"
UV_CACHE_PATH="${UV_CACHE_DIR:-/tmp/uvcache}"
HF_HOME_DIR="${HF_HOME:-$VOX_CACHE_DIR/huggingface}"
HF_HUB_CACHE_DIR="${HUGGINGFACE_HUB_CACHE:-$HF_HOME_DIR/hub}"
HF_XET_CACHE_DIR="${HF_XET_CACHE:-$HF_HOME_DIR/xet}"

export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE_DIR"
export HF_XET_CACHE="$HF_XET_CACHE_DIR"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

install -d \
  "$VOX_HOME_DIR" \
  "$VOX_HOME_DIR/adapters" \
  "$VOX_HOME_DIR/models" \
  "$VOX_HOME_DIR/models/blobs" \
  "$VOX_HOME_DIR/models/manifests" \
  "$VOX_HOME_DIR/models/manifests/library" \
  "$VOX_CACHE_DIR" \
  "$HF_HOME_DIR" \
  "$HF_HUB_CACHE_DIR" \
  "$HF_XET_CACHE_DIR" \
  "$HF_XET_CACHE_DIR/logs" \
  "$HF_XET_CACHE_DIR/chunk-cache" \
  "$HF_XET_CACHE_DIR/shard-cache" \
  "$VOX_CACHE_DIR/torch" \
  "$VOX_CACHE_DIR/torch/hub" \
  "$UV_CACHE_PATH"

chown vox:vox \
  "$VOX_HOME_DIR" \
  "$VOX_HOME_DIR/adapters" \
  "$VOX_HOME_DIR/models" \
  "$VOX_HOME_DIR/models/blobs" \
  "$VOX_HOME_DIR/models/manifests" \
  "$VOX_HOME_DIR/models/manifests/library" \
  "$VOX_CACHE_DIR" \
  "$HF_HOME_DIR" \
  "$HF_HUB_CACHE_DIR" \
  "$HF_XET_CACHE_DIR" \
  "$HF_XET_CACHE_DIR/logs" \
  "$HF_XET_CACHE_DIR/chunk-cache" \
  "$HF_XET_CACHE_DIR/shard-cache" \
  "$VOX_CACHE_DIR/torch" \
  "$VOX_CACHE_DIR/torch/hub" \
  "$UV_CACHE_PATH"

exec gosu vox "$@"
