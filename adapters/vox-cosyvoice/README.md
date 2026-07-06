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

The current Vox adapter is classified for Linux x86_64 CUDA/Torch runtimes.
CPU, ONNX, and Spark/ARM NVIDIA paths are not currently production-supported by
this adapter. Upstream CosyVoice 2 does not provide a Vox-validated ONNX backend
for this package, and the Spark/ARM NVIDIA dependency stack has not been proven
cleanly installable with the adapter runtime isolation contract.
Plan for at least 8GiB of usable VRAM budget before deployment headroom.

CosyVoice imports `whisper.log_mel_spectrogram` and
`whisper.tokenizer.Tokenizer` for frontend features. Vox provides those narrow
compatibility surfaces inside the isolated runtime instead of installing the
full `openai-whisper` package and its duplicate GPU stack.
Matcha-TTS also imports `matplotlib` through training/plotting utilities during
module import; Vox provides an inference-only compatibility package instead of
installing the full plotting stack.

## Use with Vox

```bash
vox pull cosyvoice2-tts:0.5b
vox run cosyvoice2-tts:0.5b "Hello from CosyVoice 2"
vox run cosyvoice2-tts:0.5b "Hello with a reference voice" --voice /path/to/reference.wav
```

CosyVoice 2 is best used with a reference voice. Pass `reference_audio` and
`reference_text` through the Vox API, or use a voice value that points to a
local WAV file.
