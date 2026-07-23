# Speech context

Vox can enrich a finalized transcript with compact evidence about how the
speaker sounded and what else was audible. It does not replace Parakeet and it
does not interpret the evidence for the application.

The service runs two isolated CPU workers concurrently:

- SenseVoiceSmall classifies speaker emotion and human vocal events.
- YAMNet classifies general acoustic sounds.

The workers receive the same canonical mono 16 kHz PCM16 WAV. Their runtimes
and models live outside the Vox application environment:

```text
$VOX_HOME/runtime/speech-context-speaker
$VOX_HOME/runtime/speech-context-audio-events
```

Neither runtime is bundled into a Vox image. The service uses Vox's protected
worker protocol, keeps model imports out of the application process, and
continues with a partial result when one worker is unavailable.

## Contract

The model-facing result is schema version 2:

```json
{
  "schema_version": 2,
  "status": "complete",
  "emotions": [
    {"label": "surprised", "start_ms": 0, "end_ms": 2500}
  ],
  "vocal": [
    {"label": "laughter", "start_ms": 7000, "end_ms": 10500}
  ],
  "sounds": [
    {"label": "dog", "start_ms": 4300, "end_ms": 5200}
  ]
}
```

`emotions` and `vocal` are owned by SenseVoice. `sounds` is owned by YAMNet.
YAMNet's human-voice and respiratory branches are removed structurally through
the AudioSet ontology so they cannot duplicate the speaker track. Other
classes remain dynamic; the reducer does not maintain an allowlist of expected
environmental sounds.

Unknown emotion, speech, and silence labels are omitted. Overlapping windows
with the same label are merged. Scores, model names, provider metadata,
AudioSet identifiers, ancestry, transcripts, and raw vectors remain internal.

`status` is `complete`, `partial`, or `failed`. When a requested worker fails,
`unavailable` contains `speaker`, `sounds`, or both. A successful track appears
as an empty array when it detects nothing.

## Installation

Install both isolated runtimes:

```bash
vox speech-context install
vox speech-context status
```

The SenseVoice runtime uses the official sherpa-onnx int8 model. Its archive is
pinned and checksum-verified before Vox extracts only the ONNX model and token
file. YAMNet and its AudioSet metadata are also checksum-verified.

Licenses:

- sherpa-onnx: Apache-2.0
- SenseVoice model: FunASR Model License; retain model attribution
- YAMNet: Apache-2.0
- AudioSet ontology: CC BY-SA 4.0

openSMILE is not part of this service or its runtime.

## Service harness

Record a browser microphone sample:

```bash
make speech-context-recorder
```

Open `http://127.0.0.1:11436/speech-context-recorder.html`, record the sample,
and save the WAV. The page releases the microphone immediately after stopping
and uploads nothing.

Run the production `SpeechContextService` directly, without starting Vox and
without making an STT request:

```bash
make speech-context-service \
  AUDIO=/path/to/speech.wav \
  EVIDENCE=/path/to/result.json
```

The output includes the canonical input identity, wall time, runtime sizes, and
the exact public `speech_context` result. This is the clean service smoke test.

For deeper diagnostics, run the evidence harness:

```bash
make speech-context-evidence \
  AUDIO=/path/to/speech.wav \
  EVIDENCE=/path/to/evidence.json \
  VOX_URL=http://127.0.0.1:11435
```

That command runs Parakeet, SenseVoice, and YAMNet concurrently and retains raw
worker output under `results`. The raw representation is diagnostic evidence,
not a public contract.
