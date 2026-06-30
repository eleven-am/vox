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

## Use with Vox

```bash
vox pull qwen3-stt-torch:0.6b
vox pull qwen3-tts-torch:0.6b
```
