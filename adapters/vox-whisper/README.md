# vox-whisper

Whisper STT adapter package for Vox.

## Included adapter

- `whisper-stt-ct2` — faster-whisper / CTranslate2 backend

## Install

```bash
pip install vox-whisper
```

## Runtime Dependencies

Whisper bootstraps faster-whisper and CTranslate2 into the isolated target
runtime `$VOX_HOME/runtime/whisper`.

## Use with Vox

```bash
vox pull whisper-stt-ct2:base.en
vox pull whisper-stt-ct2:large-v3
```
