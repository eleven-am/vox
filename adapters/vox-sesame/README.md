# vox-sesame

Sesame CSM TTS adapter package for Vox.

## Included adapter

- `sesame-tts-torch` — CSM 1B backend

## Install

```bash
pip install vox-sesame
```

## Runtime Dependencies

Sesame bootstraps its Transformers backend into the isolated target runtime
`$VOX_HOME/runtime/sesame`. Model weights still live in Vox model storage, not
in the base Vox image.

## Use with Vox

```bash
vox pull sesame-tts-torch:csm-1b
```
