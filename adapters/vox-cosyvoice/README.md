# vox-cosyvoice

`vox-cosyvoice` provides a Vox TTS adapter for CosyVoice 2.

Adapters:

- `cosyvoice2-tts-torch` - CosyVoice 2 zero-shot streaming backend

## Install

```bash
pip install vox-cosyvoice
```

## Runtime Dependencies

The adapter package is intentionally light. The official CosyVoice source is
checked out on demand from GitHub into `$VOX_HOME/runtime/cosyvoice/CosyVoice`,
and CosyVoice-specific Python dependencies are installed into the isolated
target runtime `$VOX_HOME/runtime/cosyvoice`.

The shared Vox GPU stack remains owned by the base Vox environment; the adapter
does not install its own Torch/CUDA/server runtime packages.

CosyVoice imports `whisper.log_mel_spectrogram` and
`whisper.tokenizer.Tokenizer` for frontend features. Vox provides those narrow
compatibility surfaces inside the isolated runtime instead of installing the
full `openai-whisper` package and its duplicate GPU stack.
Matcha-TTS also imports `matplotlib` through training/plotting utilities during
module import; Vox provides an inference-only compatibility package instead of
installing the full plotting stack.

## Use with Vox

```bash
vox pull cosyvoice2-tts-torch:0.5b
vox run cosyvoice2-tts-torch:0.5b "Hello from CosyVoice 2"
```

CosyVoice 2 is best used with a reference voice. Pass `reference_audio` and
`reference_text` through the Vox API, or use a voice value that points to a
local WAV file.
