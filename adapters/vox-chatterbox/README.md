# vox-chatterbox

`vox-chatterbox` provides Vox TTS adapters for Resemble AI Chatterbox.

Adapters:

- `chatterbox-tts-turbo` - Chatterbox Turbo backend
- `chatterbox-tts` - Chatterbox backend
- `chatterbox-tts-multilingual` - Chatterbox multilingual backend

## Install

```bash
pip install vox-chatterbox
```

## Runtime Dependencies

The adapter package is intentionally light. The Chatterbox backend package is
installed on demand into the isolated target runtime
`$VOX_HOME/runtime/chatterbox`.

## Use with Vox

```bash
vox pull chatterbox-tts-turbo:0.1.7
vox run chatterbox-tts-turbo:0.1.7 "Hello from Chatterbox"
```

For voice cloning, pass a reference audio sample through the Vox API or use a
voice value that points to a local WAV file.
