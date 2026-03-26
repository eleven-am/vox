# Vox

Local runtime for speech-to-text and text-to-speech models. Pull a model, start the server, hit an API.

## What it does

Vox manages STT and TTS models through a REST API. Models are downloaded from HuggingFace, and each model family (Whisper, Kokoro, Parakeet, etc.) is handled by a plugin adapter that's installed automatically on first pull.

```
vox pull kokoro:v1.0       # downloads model + installs adapter
vox pull whisper:large-v3   # same for STT
vox serve                   # starts REST API on :11435
```

## Install

```bash
pip install vox
# or
uv pip install vox
```

## Usage

### Server

```bash
vox serve --port 11435 --device auto
```

### Pull a model

```bash
vox pull kokoro:v1.0
vox pull parakeet:tdt-0.6b-v3
vox list
```

### Transcribe (STT)

```bash
# CLI
vox run parakeet:tdt-0.6b-v3 recording.wav

# API
curl -F file=@recording.wav http://localhost:11435/api/transcribe

# OpenAI-compatible
curl -F file=@recording.wav http://localhost:11435/v1/audio/transcriptions
```

### Synthesize (TTS)

```bash
# CLI
vox run kokoro:v1.0 "Hello, how are you?" -o output.wav

# API
curl -X POST http://localhost:11435/api/synthesize \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro:v1.0","input":"Hello, how are you?"}' \
  -o output.wav

# OpenAI-compatible
curl -X POST http://localhost:11435/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro:v1.0","input":"Hello"}' \
  -o output.wav
```

### Search available models

```bash
vox search
vox search --type tts
vox search --type stt
```

### Other commands

```bash
vox list          # downloaded models
vox ps            # loaded models
vox show kokoro:v1.0
vox rm kokoro:v1.0
vox voices kokoro:v1.0
```

## Docker

```bash
# GPU (default)
docker compose up -d
vox pull kokoro:v1.0  # auto-installs adapter inside container

# CPU
docker compose --profile cpu up -d
```

Models and adapters persist in a Docker volume across container restarts. No image rebuild needed to add new models.

## Available models

| Model | Type | Description |
|-------|------|-------------|
| `parakeet:tdt-0.6b` | STT | NVIDIA Parakeet TDT 0.6B |
| `parakeet:tdt-0.6b-v3` | STT | Parakeet TDT 0.6B v3, 25 languages |
| `whisper:large-v3` | STT | OpenAI Whisper Large V3 via CTranslate2 |
| `whisper:large-v3-turbo` | STT | Whisper Large V3 Turbo |
| `whisper:base.en` | STT | Whisper Base English |
| `kokoro:v1.0` | TTS | Kokoro 82M ONNX, preset voices |
| `piper:en-us-lessac-medium` | TTS | Piper English US Lessac |
| `fish-speech:v1.4` | TTS | Fish Speech 1.4, multilingual, voice cloning |
| `orpheus:3b` | TTS | Orpheus 3B, emotional speech |
| `dia:1.6b` | TTS | Dia 1.6B, multi-speaker dialogue |
| `sesame:csm-1b` | TTS | Sesame CSM 1B, conversational speech |

More models at [vox-registry](https://github.com/eleven-am/vox-registry). Add a model by submitting a PR with a JSON file.

## API

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/transcribe` | POST | Audio to text |
| `/api/synthesize` | POST | Text to audio |
| `/api/pull` | POST | Download a model |
| `/api/list` | GET | List downloaded models |
| `/api/show` | POST | Model details |
| `/api/delete` | DELETE | Remove a model |
| `/api/ps` | GET | Currently loaded models |
| `/api/voices` | GET | List voices for a TTS model |
| `/api/health` | GET | Health check |
| `/v1/audio/transcriptions` | POST | OpenAI-compatible STT |
| `/v1/audio/speech` | POST | OpenAI-compatible TTS |

## Adding a model

Write an adapter package that implements `STTAdapter` or `TTSAdapter`:

```python
from vox.core.adapter import TTSAdapter

class MyAdapter(TTSAdapter):
    def info(self): ...
    def load(self, model_path, device, **kwargs): ...
    def unload(self): ...
    @property
    def is_loaded(self): ...
    async def synthesize(self, text, *, voice=None, speed=1.0, **kwargs):
        yield SynthesizeChunk(audio=audio_bytes, sample_rate=24000)
```

Register it via entry point:

```toml
[project.entry-points."vox.adapters"]
my-model = "my_package.adapter:MyAdapter"
```

Add a JSON file to [vox-registry](https://github.com/eleven-am/vox-registry) so `vox pull` can find it.

## Project structure

```
src/vox/
  core/          # types, adapter ABCs, scheduler, store, registry
  audio/         # codec, resampling, pipeline
  server/        # FastAPI routes
  cli.py         # Click CLI
adapters/
  vox-parakeet/  # NVIDIA Parakeet STT
  vox-kokoro/    # Kokoro TTS
```

## License

Apache-2.0
