# vox-dia

Dia TTS adapter package for Vox.

## Included adapter

- `dia-tts-torch` — Dia 1.6B text-to-speech backend

## Install

```bash
pip install vox-dia
```

## Runtime Dependencies

Dia uses an isolated target runtime at `$VOX_HOME/runtime/dia` for backend
packages that must not leak into the base Vox app environment.

The adapter package itself is intentionally lightweight and does not install
Torch. The current Dia backend uses Hugging Face Transformers and requires a
Vox runtime/image that already provides PyTorch with CUDA. CPU execution is not
supported by the official Dia Transformers runtime used by this adapter.

The current Vox adapter is classified for Linux x86_64 CUDA/Torch runtimes.
CPU, ONNX, and Spark/ARM NVIDIA paths are not currently production-supported by
this adapter because Vox has not validated a portable CPU/ONNX backend or a
clean Spark/ARM NVIDIA dependency stack for the Dia Transformers runtime.

During `vox pull`, the adapter verifies or installs the Dia-capable
Transformers runtime into `$VOX_HOME/runtime/dia`. Model weights remain in the
normal Vox model store.

## Use with Vox

```bash
vox pull dia-tts-torch:1.6b
```

Dia prompt control is text-driven. Use speaker tags and non-verbal markers in
the input text, for example:

```text
[S1] This is Dia speaking. (laughs) [S2] Keep the tags sparse or artifacts can appear.
```
