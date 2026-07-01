# vox-orpheus

`vox-orpheus` provides a Vox TTS adapter for Orpheus.

Adapters:

- `orpheus-tts-vllm` - Orpheus medium 3B backend through vLLM

## Install

```bash
pip install vox-orpheus
```

## Runtime Dependencies

The adapter package is intentionally light. The `orpheus-speech` backend and
its vLLM/SNAC runtime dependencies are installed on demand into the isolated
target runtime `$VOX_HOME/runtime/orpheus`.

## Use with Vox

```bash
vox pull orpheus-tts-vllm:medium-3b
vox run orpheus-tts-vllm:medium-3b "Hello from Orpheus" --voice tara
```

Orpheus is GPU-oriented. Expect this adapter to require a CUDA-capable runtime
for practical latency.
