# vox-openvoice

OpenVoice TTS adapter package for Vox.

## Included adapter

- `openvoice-tts-torch` — OpenVoice V1 backend

## Install

```bash
pip install vox-openvoice
```

## Runtime Dependencies

OpenVoice installs the upstream backend into the isolated target runtime
`$VOX_HOME/runtime/openvoice`.

## Use with Vox

```bash
vox pull openvoice-tts-torch:v1
```
