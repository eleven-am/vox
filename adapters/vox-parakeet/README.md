# vox-parakeet

Parakeet STT adapter package for Vox.

## Included adapters

- `parakeet-stt-onnx` — ONNX backend
- `parakeet-stt-nemo` — native CUDA/NeMo backend

## Install

```bash
pip install vox-parakeet
```

## Runtime Dependencies

The ONNX backend is installed with the adapter package. The NeMo backend
bootstraps into `$VOX_HOME/runtime/parakeet-nemo`.

## Use with Vox

```bash
vox pull parakeet-stt-onnx:tdt-0.6b-v3
vox pull parakeet-stt-nemo:tdt-0.6b-v3
```
