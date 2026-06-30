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

## Use with Vox

```bash
vox pull dia-tts-torch:1.6b
```
