# vox-microsoft

Microsoft adapter package for Vox.

## Included adapters

- `speecht5-stt-torch`
- `speecht5-tts-torch`
- `vibevoice-tts-torch`

## Install

```bash
pip install vox-microsoft
```

## Runtime Dependencies

SpeechT5 currently uses adapter package dependencies. VibeVoice bootstraps its
heavier backend into `$VOX_HOME/runtime/vibevoice`.

## Use with Vox

```bash
vox pull speecht5-stt-torch:base
vox pull speecht5-tts-torch:base
vox pull vibevoice-tts-torch:realtime-0.5b
```
