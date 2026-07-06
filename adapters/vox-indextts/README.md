# vox-indextts

`vox-indextts` provides a Vox TTS adapter for IndexTTS2.

Adapters:

- `indextts-tts-torch` - IndexTTS2 voice cloning backend

## Install

```bash
pip install vox-indextts
```

## Runtime Dependencies

The adapter package is intentionally light. The upstream IndexTTS runtime is
installed on demand from GitHub into the isolated target runtime
`$VOX_HOME/runtime/indextts`.

During `vox pull`, the adapter verifies or installs the IndexTTS runtime without
loading model weights. The upstream package source is installed without its
transitive dependencies so it does not pull a second Torch/CUDA stack into the
adapter runtime. Curated non-Torch runtime dependencies live under
`$VOX_HOME/runtime/indextts`; model weights remain in the normal Vox model store.

## Use with Vox

```bash
vox pull indextts-tts-torch:2
vox run indextts-tts-torch:2 "Hello from IndexTTS"
```

IndexTTS is a voice-cloning backend. Pass `reference_audio` through the Vox API
or use a voice value that points to a local WAV file.

The current Vox adapter is classified for Linux x86_64 CUDA/Torch runtimes.
CPU, ONNX, and Spark/ARM NVIDIA paths are not currently production-supported by
this adapter. Upstream documents CPU-style constructor switches, but Vox has
not validated those paths with acceptable latency or portable dependency
resolution.
