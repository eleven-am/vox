# vox-kokoro

Kokoro adapter package for Vox.

## Included adapters

- `kokoro-tts-onnx` — ONNX backend
- `kokoro-tts-torch` — native runtime backend

## Install

```bash
pip install vox-kokoro
```

## Runtime Dependencies

Kokoro currently installs its ONNX and Torch backend packages with the adapter
package. Model weights and voices still live in Vox model storage, not in the
base Vox image.

## Use with Vox

```bash
vox pull kokoro-tts-onnx:v1.0
vox pull kokoro-tts-torch:v1.0
```
