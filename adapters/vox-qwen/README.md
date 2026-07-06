# vox-qwen

Qwen speech adapter package for Vox.

## Included adapters

- `qwen3-stt-torch`
- `qwen3-tts-torch`

## Install

```bash
pip install vox-qwen
```

## Runtime Dependencies

Qwen ASR and TTS backend packages bootstrap into isolated target runtimes under
`$VOX_HOME/runtime/qwen-asr` and `$VOX_HOME/runtime/qwen-tts`.

Qwen TTS prefers the optional `faster-qwen3-tts` runtime when CUDA and
PyTorch 2.5.1 or newer are available. If that package cannot be installed or
loaded, the adapter falls back to the standard `qwen-tts` runtime without
changing the public model name.

## Use with Vox

```bash
vox pull qwen3-stt:0.6b
vox pull qwen3-tts:0.6b
```
