# vox-voxtral

Voxtral STT and TTS adapter package for Vox.

## Included adapters

- `voxtral-stt-torch`
- `voxtral-tts-vllm`

## Install

```bash
pip install vox-voxtral
```

## Runtime Dependencies

Voxtral STT uses `$VOX_HOME/runtime/voxtral-stt`. Voxtral TTS deliberately uses
a full venv under `$VOX_HOME/runtime/voxtral-tts` because vLLM-Omni and GPU
PyTorch loading need stronger isolation than a simple target directory.

## Use with Vox

```bash
vox pull voxtral-stt-torch:mini-3b
vox pull voxtral-tts-vllm:4b
```
