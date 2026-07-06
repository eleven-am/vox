# vox-indextts

`vox-indextts` provides a Vox TTS adapter for IndexTTS2.

Adapters:

- `indextts-tts-torch` - IndexTTS2 voice cloning backend

## Install

```bash
pip install vox-indextts
```

## Runtime Dependencies

The adapter package is intentionally light. The upstream IndexTTS runtime is
installed on demand from GitHub into the isolated target runtime
`$VOX_HOME/runtime/indextts`.

During `vox pull`, the adapter verifies or installs the IndexTTS runtime without
loading model weights. The upstream package source is installed without its
transitive dependencies so it does not pull a second Torch/CUDA stack into the
adapter runtime. Curated non-Torch runtime dependencies live under
`$VOX_HOME/runtime/indextts`; model weights remain in the normal Vox model store.

## Use with Vox

```bash
vox pull indextts-tts:2
vox run indextts-tts:2 "Hello from IndexTTS" --voice /path/to/reference.wav
```

IndexTTS is a voice-cloning backend. Pass `reference_audio` through the Vox API
or use a voice value that points to a local WAV file.

IndexTTS2 emotion controls are exposed through Vox synthesis `params`:

- `emo_alpha` (number, default backend value, range `0..1`) controls emotion
  conditioning strength.
- `use_emo_text` (boolean, default `false`) asks IndexTTS2 to infer emotion
  from the synthesis text.
- `emo_text` (string, optional) supplies separate emotion-description text and
  implies `use_emo_text=true`.
- `emo_audio_prompt` (string, optional) supplies a server-side audio file path
  used as the IndexTTS2 emotional reference prompt.
- `use_random` (boolean, default `false`) enables stochastic emotion sampling;
  upstream notes that this may reduce voice-cloning fidelity.
- `emotion_happy`, `emotion_angry`, `emotion_sad`, `emotion_afraid`,
  `emotion_disgusted`, `emotion_melancholic`, `emotion_surprised`, and
  `emotion_calm` (numbers, default `0`, range `0..1`) map to the IndexTTS2
  eight-float `emo_vector` order. The adapter rejects vectors whose values sum
  above `1.5`.

Upstream IndexTTS2 describes duration control, but the current release notes say
that functionality is not enabled. Vox therefore does not expose a duration
parameter for this adapter yet.

The current Vox adapter is classified for Linux x86_64 CUDA/Torch runtimes.
CPU, ONNX, and Spark/ARM NVIDIA paths are not currently production-supported by
this adapter. Upstream documents CPU-style constructor switches, but Vox has
not validated those paths with acceptable latency or portable dependency
resolution.
Plan for at least 10GiB of usable VRAM budget before deployment headroom.
