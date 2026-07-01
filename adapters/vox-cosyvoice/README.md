# vox-cosyvoice

`vox-cosyvoice` provides a Vox TTS adapter for CosyVoice 2.

Adapters:

- `cosyvoice2-tts-torch` - CosyVoice 2 zero-shot streaming backend

## Install

```bash
pip install vox-cosyvoice
```

## Runtime Dependencies

The adapter package is intentionally light. The official CosyVoice runtime is
installed on demand from GitHub into the isolated target runtime
`$VOX_HOME/runtime/cosyvoice`.

## Use with Vox

```bash
vox pull cosyvoice2-tts-torch:0.5b
vox run cosyvoice2-tts-torch:0.5b "Hello from CosyVoice 2"
```

CosyVoice 2 is best used with a reference voice. Pass `reference_audio` and
`reference_text` through the Vox API, or use a voice value that points to a
local WAV file.
