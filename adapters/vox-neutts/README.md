# vox-neutts

`vox-neutts` provides a Vox TTS adapter for NeuTTS Air.

Adapters:

- `neutts-air-tts-torch` - NeuTTS Air PyTorch backend

## Install

```bash
pip install vox-neutts
```

## Runtime Dependencies

The adapter package is intentionally light. The `neutts` backend package is
installed on demand into the isolated target runtime `$VOX_HOME/runtime/neutts`.

## Use with Vox

```bash
vox pull neutts-air-tts-torch:air
vox run neutts-air-tts-torch:air "Hello from NeuTTS Air"
```

NeuTTS Air needs reference audio. Pass `reference_audio` and `reference_text`
through the Vox API, or use a voice value that points to a local WAV file.
