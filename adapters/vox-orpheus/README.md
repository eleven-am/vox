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

During `vox pull`, the adapter verifies or installs the Orpheus runtime without
loading model weights. Model weights remain in the normal Vox model store.

## Use with Vox

```bash
vox pull orpheus-tts:medium-3b
vox run orpheus-tts:medium-3b "Hello from Orpheus" --voice tara
vox run orpheus-tts:medium-3b "<happy>Hello from Orpheus.<laugh>" --voice tara
```

Preset voices:

- `tara`
- `leah`
- `jess`
- `leo`
- `dan`
- `mia`
- `zoe`
- `zac`

Orpheus supports expressive prompt markup in the text itself. Use tags
sparingly; dense markup can create artifacts or unstable delivery.

Non-verbal tags:

- `<laugh>`
- `<chuckle>`
- `<sigh>`
- `<cough>`
- `<sniffle>`
- `<groan>`
- `<yawn>`
- `<gasp>`

Phrase-level emotion tags:

- `<happy>`
- `<sad>`
- `<angry>`
- `<excited>`
- `<fearful>`
- `<surprised>`
- `<disgusted>`
- `<neutral>`

This adapter supports Orpheus preset voices, not reference-audio voice cloning.
Requests with `reference_audio` or `reference_text` are rejected clearly rather
than silently ignored.

The current Orpheus backend is vLLM-based and is classified for Linux x86_64
CUDA runtimes. CPU and Spark/ARM NVIDIA execution are not currently supported
by this adapter because the `orpheus-speech`/vLLM runtime is not packaged as a
portable CPU or ARM NVIDIA backend.
