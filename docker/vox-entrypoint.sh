#!/bin/sh
set -eu

VOX_HOME_DIR="${VOX_HOME:-/home/vox/.vox}"
VOX_CACHE_DIR="${HOME:-/home/vox}/.cache"
UV_CACHE_PATH="${UV_CACHE_DIR:-/tmp/uvcache}"

install -d \
  "$VOX_HOME_DIR" \
  "$VOX_HOME_DIR/adapters" \
  "$VOX_HOME_DIR/models" \
  "$VOX_HOME_DIR/models/blobs" \
  "$VOX_HOME_DIR/models/manifests" \
  "$VOX_HOME_DIR/models/manifests/library" \
  "$VOX_CACHE_DIR" \
  "$VOX_CACHE_DIR/huggingface" \
  "$VOX_CACHE_DIR/huggingface/hub" \
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
  "$VOX_CACHE_DIR/huggingface" \
  "$VOX_CACHE_DIR/huggingface/hub" \
  "$VOX_CACHE_DIR/torch" \
  "$VOX_CACHE_DIR/torch/hub" \
  "$UV_CACHE_PATH"

exec gosu vox "$@"
