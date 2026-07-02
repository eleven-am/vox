# Vox

**The local voice layer for realtime speech agents.** Vox is a self-hosted runtime that gives speech-to-text and text-to-speech models one operational surface: you pull a model like Ollama and serve one OpenAI-compatible API. On top of that it adds a full realtime conversation stack: VAD, streaming STT, end-of-utterance turn detection, TTS, and true barge-in over WebRTC. **You bring the LLM; Vox is the ears and the mouth.**

## Why Vox

- **Realtime voice, self-hosted.** A local, OpenAI-Realtime-style conversation API over WebRTC, WebSocket, or gRPC, covering VAD, streaming transcription, semantic turn-taking, and real barge-in, with any LLM in the middle. Vox owns the audio; you own the text generation.
- **One runtime for STT and TTS.** `pull` a model, `serve` one API, no per-model Python wiring.
- **Many backends, one interface.** ONNX, CTranslate2, Torch, NeMo, and vLLM model families behind the same API.
- **OpenAI-compatible.** Drop-in `/v1/audio/speech` and `/v1/audio/transcriptions`, plus REST, WebSocket, and gRPC.
- **Pull-on-demand.** Models and their adapters install on first pull, from a community registry.
- **Custom voices.** Store cloned voices for clone-capable TTS models, shared across HTTP and gRPC.
- **Local-first and lean.** Docker images start empty and install only what you use; a torch-free lean image runs the CT2/ONNX and streaming paths without the ~2GB torch stack.

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

## Realtime voice conversations

Vox ships a self-hosted realtime voice stack, the local answer to the OpenAI
Realtime API. **Vox owns VAD, streaming STT, end-of-utterance (EOU) turn
detection, TTS, and interruption handling; you own the LLM.** User speech comes
in, transcripts come out, you generate a reply with any LLM, stream the text
back, and Vox speaks it with real barge-in.

Three transports carry a conversation session:

- **WebRTC:** Vox hosts the browser media connection (mic in, assistant audio
  out) while your backend drives a control stream. A dependency-free client is in
  [`examples/rtc-browser-client.html`](examples/rtc-browser-client.html).
- **WebSocket** (PondSocket) and **gRPC:** you own microphone capture and
  playback, and audio is PCM16 over the stream.

What Vox handles for you:

- **VAD:** Silero on onnxruntime (no torch, no runtime model download).
- **Turn-taking:** a semantic EOU detector shortens the endpointing delay when
  the user clearly finished and waits when they didn't.
- **Barge-in:** two-stage interruption with self-echo and backchannel
  rejection, so the assistant doesn't cancel itself on its own audio or a stray
  "mhm".
- **Turn profiles:** `headset`, `browser_default`, `speakerphone`, and
  `noisy_room` acoustic presets.
- **Browser-native events:** captions, turn state, and barge-in signals
  forwarded straight to a WebRTC data channel, with no backend relay required.

Full protocol and event reference: [docs/conversation-events.md](docs/conversation-events.md).

## What it does

Vox manages STT and TTS models through a consistent runtime API. Models are
downloaded from Hugging Face, and each model family is handled by an adapter that
installs automatically on first pull. Docker images start without any models or
adapters; pulling a model installs the matching adapter on demand.

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

# CPU (lean, torch-free)
docker compose --profile cpu up -d
```

Models and dynamically installed adapters persist in a Docker volume across
restarts, so no image rebuild is needed to add new models.

Four image variants are published: `:latest` (amd64 CUDA), `:lean` and `:cpu`
(multi-arch CPU), and `:spark` (arm64 NVIDIA). The `:lean` image drops the ~2GB
torch stack and still runs the CT2/ONNX and streaming paths; `vox pull` then
refuses torch-based models up front instead of failing at load time. See
[docs/docker.md](docs/docker.md) for the full matrix and build options.

## Representative models

Vox ships a bundled catalog of 25 model entries (39 tags) spanning ~18 model
families across 5 backends (ONNX, CTranslate2, Torch, NeMo, and vLLM), with more
available from the community registry. A representative slice:

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

## Security

Vox is local-first and runs open by default. For deployments that expose the
port beyond localhost, set these environment variables:

- `VOX_API_KEY`: when set, every HTTP route, audio websocket, and gRPC call
  requires the key (sent as `Authorization: Bearer <key>`, an `x-api-key`
  header, or an `api_key` query param). Health/probe paths and gRPC health and
  reflection stay open so orchestrator liveness checks keep working. When unset,
  the server is fully open (unchanged behavior).
- `VOX_ALLOW_UNVERIFIED_ADAPTERS=1`: pulling a model installs its adapter as a
  pip package; by default only the built-in catalog's `vox-*` packages are
  allowed. Set this only if you intentionally pull adapters outside that set.

gRPC still uses an insecure port and reflection; terminate TLS at a proxy for
untrusted networks.

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
| `/v1/rtc/sessions` | POST | Create a realtime WebRTC voice session |
| `/v1/rtc/sessions/{id}/offer` | POST | Submit the browser SDP offer, get the answer |
| `/v1/rtc/sessions/{id}/candidates` | POST | Trickle browser ICE candidates |
| `/v1/rtc/sessions/{id}/events` | GET (SSE) | Vox-side ICE / connection-state events |

The realtime conversation control stream runs over PondSocket
(`/v1/socket` channel `/conversation/{id}` or `/rtc/{id}`) or gRPC; see
[docs/conversation-events.md](docs/conversation-events.md).

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
- realtime conversation (bidi streams):
  - `ConversationService.Converse`
  - `RtcService.Control`

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

For the package/runtime boundary, see [the adapter contract](docs/adapter-contract.md).

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
