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

## Use with Vox

```bash
vox pull indextts-tts-torch:2
vox run indextts-tts-torch:2 "Hello from IndexTTS"
```

IndexTTS is a voice-cloning backend. Pass `reference_audio` through the Vox API
or use a voice value that points to a local WAV file.
