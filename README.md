# Vox

Vox is a local runtime for speech models.

It gives speech-to-text and text-to-speech models one operational surface: pull a model, serve one API, and run local speech workloads without hand-wiring each model family yourself. Vox is built around speech-native concerns like streaming audio, voice selection, backend-specific adapters, and one consistent interface across STT and TTS models.

## Why Vox

- One runtime for both speech-to-text and text-to-speech
- One CLI and one API surface across many model families
- Pull-on-demand model and adapter installation
- Multiple backends behind the same runtime: ONNX, Torch, NeMo, CTranslate2, and vLLM
- Stored custom voices for clone-capable TTS models
- REST, WebSocket, gRPC, and OpenAI-compatible endpoints
- Local-first deployment with Docker images that start empty and install only what you use

## Quickstart

```bash
pip install vox-runtime

vox pull kokoro-tts-onnx:v1.0
vox pull whisper-stt-ct2:large-v3
vox serve
```

Then hit the local API:

```bash
curl -X POST http://localhost:11435/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro-tts-onnx:v1.0","input":"Hello from Vox"}' \
  -o output.wav
```

gRPC starts with `vox serve` too and listens on `:9090` by default.

## What it does

Vox manages STT and TTS models through a consistent runtime API. Models are downloaded from Hugging Face, and each model family is handled by an adapter that is installed automatically on first pull.

The Docker images intentionally start without any models or adapter packages installed. Pulling a model installs the matching adapter on demand.

```bash
vox pull kokoro-tts-onnx:v1.0
vox pull whisper-stt-ct2:large-v3
vox serve
```

## Install

```bash
pip install vox-runtime
# or
uv pip install vox-runtime
```

## Usage

### Server

```bash
vox serve --port 11435 --device auto
```

### Pull a model

```bash
vox pull kokoro-tts-onnx:v1.0
vox pull parakeet-stt-onnx:tdt-0.6b-v3
vox list
```

### Transcribe (STT)

```bash
# CLI
vox run parakeet-stt-onnx:tdt-0.6b-v3 recording.wav
vox stream-transcribe parakeet-stt-onnx:tdt-0.6b-v3 meeting.mp3

# OpenAI-compatible: thin response (just {"text": ...})
curl -F file=@recording.wav http://localhost:11435/v1/audio/transcriptions

# Rich response with segments, word timestamps, entities, topics
curl -F file=@recording.wav -F response_format=verbose_json \
  http://localhost:11435/v1/audio/transcriptions
```

### Synthesize (TTS)

```bash
# CLI
vox run kokoro-tts-onnx:v1.0 "Hello, how are you?" -o output.wav
vox stream-synthesize kokoro-tts-onnx:v1.0 "Hello, how are you?" -o output.wav

# OpenAI-compatible
curl -X POST http://localhost:11435/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro-tts-onnx:v1.0","input":"Hello"}' \
  -o output.wav
```

### Create and use custom voices

Vox can store cloned voices and reuse them across HTTP and gRPC. This is only available for TTS adapters that declare voice-cloning support. Preset-only models still list their built-in voices, but they will reject stored cloned voices at synthesis time.

```bash
# create a cloned voice from a reference sample
curl -X POST http://localhost:11435/v1/audio/voices \
  -F audio_sample=@sample.wav \
  -F name="Roy" \
  -F language=en \
  -F reference_text="Hello there from my custom voice"

# list voices, including cloned voices for clone-capable models
curl "http://localhost:11435/v1/audio/voices?model=openvoice-tts-torch:v1"

# download the stored reference audio
curl -o reference.wav http://localhost:11435/v1/audio/voices/voice1234/reference

# synthesize with the stored voice id returned at creation time
curl -X POST http://localhost:11435/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"openvoice-tts-torch:v1","input":"Hello from Vox","voice":"voice1234"}' \
  -o output.wav

# delete a stored cloned voice
curl -X DELETE http://localhost:11435/v1/audio/voices/voice1234
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
vox show kokoro-tts-onnx:v1.0
vox rm kokoro-tts-onnx:v1.0
vox voices kokoro-tts-onnx:v1.0
```

## Streaming APIs

Use the unary HTTP endpoints for short bounded requests.

Use the WebSocket APIs for:
- long recordings
- browser or pipeline streaming
- live uploads where the client stays connected until the final result arrives

These streaming sessions are intentionally short-lived:
- no job store
- no durable result retention
- disconnect cancels the session

### Long-form STT over WebSocket

Endpoint:

```text
ws://localhost:11435/v1/audio/transcriptions/stream
```

Protocol:

1. Client sends a JSON config message.
2. Client sends binary audio chunks.
3. Client sends `{"type":"end"}`.
4. Server emits progress events and one final `done` event with the full transcript.

Example config:

```json
{
  "type": "config",
  "model": "parakeet-stt-onnx:tdt-0.6b-v3",
  "input_format": "pcm16",
  "sample_rate": 16000,
  "language": "en",
  "word_timestamps": true,
  "chunk_ms": 30000,
  "overlap_ms": 1000
}
```

Server events:

```json
{"type":"ready","model":"parakeet-stt-onnx:tdt-0.6b-v3","input_format":"pcm16","sample_rate":16000}
{"type":"progress","uploaded_ms":60000,"processed_ms":30000,"chunks_completed":1}
{"type":"done","text":"full transcript","duration_ms":120000,"processing_ms":8420,"segments":[]}
```

Notes:
- `pcm16` is the simplest long-form transport. The CLI helper uses it by default.
- `wav`, `flac`, `mp3`, `ogg`, and `webm` are also accepted as `input_format`, but each binary frame must be a self-contained decodable blob, such as a `MediaRecorder` chunk. Arbitrary byte slices of one compressed file are not supported.

### Long-form TTS over WebSocket

Endpoint:

```text
ws://localhost:11435/v1/audio/speech/stream
```

Protocol:

1. Client sends a JSON config message.
2. Client sends one or more `{"type":"text","text":"..."}` messages.
3. Client sends `{"type":"end"}`.
4. Server emits:
   - `ready`
   - `audio_start`
   - `progress`
   - binary audio chunks
   - final `done`

Example config:

```json
{
  "type": "config",
  "model": "kokoro-tts-onnx:v1.0",
  "voice": "af_heart",
  "speed": 1.0,
  "response_format": "pcm16"
}
```

Server events:

```json
{"type":"ready","model":"kokoro-tts-onnx:v1.0","response_format":"pcm16"}
{"type":"audio_start","sample_rate":24000,"response_format":"pcm16"}
{"type":"progress","completed_chars":120,"total_chars":480,"chunks_completed":1,"chunks_total":4}
{"type":"done","response_format":"pcm16","audio_duration_ms":2450,"processing_ms":891}
```

Binary frames between `audio_start` and `done` carry the synthesized audio payload. `pcm16` and `opus` are currently supported for the raw stream; the CLI helper writes `pcm16` into a WAV file.

### Streaming CLI helpers

These commands sit on top of the WebSocket APIs:

```bash
vox stream-transcribe parakeet-stt-onnx:tdt-0.6b-v3 meeting.mp3
vox stream-transcribe parakeet-stt-onnx:tdt-0.6b-v3 meeting.wav --json-output
vox stream-synthesize kokoro-tts-onnx:v1.0 script.txt -o script.wav
```

`vox stream-transcribe` transcodes the local input to streamed mono `pcm16` on the client side, then uploads chunk-by-chunk over the WebSocket session. For compressed inputs this uses `ffmpeg`; install it if you want the helper to handle formats that `soundfile` cannot stream directly.

## Docker

```bash
# GPU (default)
docker compose up -d
vox pull kokoro-tts-onnx:v1.0  # auto-installs adapter inside container

# CPU
docker compose --profile cpu up -d
```

Models and dynamically installed adapters persist in a Docker volume across container restarts. No image rebuild needed to add new models.

### Spark ONNX GPU build

The default GPU multi-arch image is generic:
- `amd64` uses `onnxruntime-gpu`
- `arm64` uses CPU `onnxruntime`

```bash
# Local image
make build-local

# Multi-arch publish build
make build
```

### Spark image

The default image stays generic. If you want a Spark-specific arm64 image with a NVIDIA-provided ONNX Runtime source, use the dedicated Spark build:

```bash
# Local Spark build
make build-local-spark

# Published Spark build
make build-spark
```

Notes:
- `build-spark` is `linux/arm64` only.
- By default, `Dockerfile.spark` uses the tested `cp312 linux_aarch64` NVIDIA Jetson AI Lab wheel:
  - `onnxruntime_gpu-1.23.0-cp312-cp312-linux_aarch64.whl`
- You can still override it with:
  - `SPARK_ORT_WHEEL=/path/or/url/to/wheel`
  - or `SPARK_ORT_INDEX_URL` / `SPARK_ORT_EXTRA_INDEX_URL`
- The generic `make build` path is unchanged and still produces the normal multi-arch image.

## Representative models

| Model | Type | Description |
|-------|------|-------------|
| `parakeet-stt-onnx:tdt-0.6b-v3` | STT | NVIDIA Parakeet TDT 0.6B v3 via ONNX |
| `parakeet-stt-nemo:tdt-0.6b-v3` | STT | NVIDIA Parakeet TDT 0.6B v3 via NeMo |
| `whisper-stt-ct2:large-v3` | STT | OpenAI Whisper Large V3 via CTranslate2 |
| `whisper-stt-ct2:base.en` | STT | Whisper Base English |
| `qwen3-stt-torch:0.6b` | STT | Qwen3 ASR 0.6B |
| `voxtral-stt-torch:mini-3b` | STT | Voxtral Mini 3B speech-to-text |
| `kokoro-tts-onnx:v1.0` | TTS | Kokoro 82M ONNX with preset voices |
| `kokoro-tts-torch:v1.0` | TTS | Kokoro native runtime backend |
| `qwen3-tts-torch:0.6b` | TTS | Qwen3 TTS 0.6B |
| `voxtral-tts-vllm:4b` | TTS | Voxtral 4B TTS via vLLM-Omni |
| `openvoice-tts-torch:v1` | TTS | OpenVoice voice-cloning backend |
| `piper-tts-onnx:en-us-lessac-medium` | TTS | Piper English US Lessac |
| `dia-tts-torch:1.6b` | TTS | Dia 1.6B multi-speaker dialogue |
| `sesame-tts-torch:csm-1b` | TTS | Sesame CSM 1B conversational speech |

More models at [vox-registry](https://github.com/eleven-am/vox-registry). Add a model by submitting a PR with a JSON file.

## API

All HTTP endpoints live under `/v1/`. STT/TTS endpoints are OpenAI-compatible by default; pass `response_format=verbose_json` on `/v1/audio/transcriptions` for the rich payload (segments, word timestamps, entities, topics).

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/health` | GET | Health check |
| `/v1/models` | GET | List downloaded models |
| `/v1/models/{name}` | GET | Model details |
| `/v1/models/{name}` | DELETE | Remove a model |
| `/v1/models/pull` | POST | Download a model |
| `/v1/models/loaded` | GET | Currently loaded models |
| `/v1/audio/transcriptions` | POST | Transcribe audio (OpenAI-compatible; `verbose_json` for rich payload) |
| `/v1/audio/speech` | POST | Synthesize speech (OpenAI-compatible; supports `stream`) |
| `/v1/audio/voices` | GET | List voices for a TTS model |
| `/v1/audio/voices` | POST | Create a stored cloned voice |
| `/v1/audio/voices/{id}` | DELETE | Delete a stored cloned voice |
| `/v1/audio/voices/{id}/reference` | GET | Download the stored reference audio |
| `/v1/audio/transcriptions/stream` | WS | Long-form streaming STT |
| `/v1/audio/speech/stream` | WS | Long-form streaming TTS |

## gRPC

`vox serve` starts the gRPC server automatically unless you disable it with `--grpc-port 0`.

- default gRPC port: `9090`
- health and model lifecycle:
  - `HealthService.Health`
  - `HealthService.ListLoaded`
  - `ModelService.Pull`
  - `ModelService.List`
  - `ModelService.Show`
  - `ModelService.Delete`
- speech:
  - `TranscriptionService.Transcribe`
  - `SynthesisService.Synthesize`
  - `SynthesisService.ListVoices`
  - `SynthesisService.CreateVoice`
  - `SynthesisService.DeleteVoice`
  - `StreamingService.StreamTranscribe`

The gRPC voice APIs use the same stored voice data as HTTP. Creating or deleting a cloned voice over one transport is immediately visible through the other.

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
