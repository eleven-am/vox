# vox-spark

`vox-spark` provides a Vox TTS adapter for Spark-TTS.

Adapters:

- `spark-tts-torch` - Spark-TTS 0.5B backend

## Install

```bash
pip install vox-spark
```

## Runtime Dependencies

The adapter package is intentionally light. Spark-TTS is installed on demand
from the official GitHub repository into the isolated target runtime
`$VOX_HOME/runtime/spark`.

PyTorch is expected to be available in the Vox compute environment.

## Use with Vox

```bash
vox pull spark-tts-torch:0.5b
vox run spark-tts-torch:0.5b "Hello from Spark-TTS"
```

Spark-TTS supports prompt-audio voice cloning. Pass `reference_audio` and
`reference_text` through the Vox API, or use a voice value that points to a
local WAV file.
