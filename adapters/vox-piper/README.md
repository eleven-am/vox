# vox-piper

Piper TTS adapter package for Vox.

## Included adapter

- `piper-tts-onnx` — Piper ONNX backend

## Install

```bash
pip install vox-piper
```

## Runtime Dependencies

Piper bootstraps `piper-tts` into the isolated target runtime
`$VOX_HOME/runtime/piper`.

## Use with Vox

```bash
vox pull piper-tts-onnx:en-us-lessac-medium
```
