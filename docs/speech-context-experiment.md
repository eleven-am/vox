# Speech-context evidence experiment

This branch evaluates speech context without replacing Parakeet, delaying its
transcript, or adding a public Vox API. One local runner canonicalizes an audio
file once and starts three independent analyses concurrently:

- the existing OpenAI-compatible transcription endpoint, using Parakeet by
  default;
- eGeMAPSv02 low-level descriptors and functionals;
- YAMNet scores, embeddings, and log-mel spectrogram frames.

The JSON file is evidence for schema and product decisions. It is not a public
contract. No model infers `angry`, `sad`, or any other combined emotion label.
The complete analyzer output remains under `results`; a deterministic reducer
also writes a lossy, compact `speech_context` projection for comparison.

## Ownership and isolation

The experiment owns audio canonicalization, concurrent execution, failure
isolation, timing, and the shared timestamp origin. Parakeet remains owned by
the running Vox server. The two experimental analyzers run in separate Python
3.12 worker environments under:

```text
$VOX_HOME/runtime/speech-context-prosody
$VOX_HOME/runtime/speech-context-audio-events
```

Neither dependency set is added to `pyproject.toml`, the Vox image, adapter
paths, or the application process. The prosody worker loads openSMILE's native
API directly instead of importing its pandas/pyarrow wrapper. The worker
protocol is carried over Vox's existing protected worker socket, so analyzer
stdout cannot corrupt evidence.

The runner sends the same canonical mono 16 kHz PCM16 WAV to every analysis.
Every timestamp in the JSON is milliseconds from the start of that WAV.
openSMILE low-level descriptors carry their own start/end intervals. YAMNet
scores and embeddings use 960 ms windows with 480 ms hops; spectrogram rows use
25 ms windows with 10 ms hops. Parakeet's complete verbose response is retained
unchanged.

## Licensing gate

YAMNet's TensorFlow Model Garden implementation and the selected full-output
TFLite model are Apache-2.0. The model URL is versioned and its SHA-256 is
verified before use. AudioSet's v1 ontology is installed beside the model from
a versioned, checksum-verified source under CC BY-SA 4.0. It is used internally
to remove redundant ancestor labels from compact event candidates:

- [YAMNet Model Garden documentation](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet)
- [YAMNet model card](https://www.kaggle.com/models/google/yamnet)
- [AudioSet ontology](https://github.com/audioset/ontology)

openSMILE 2.6.0 is **not** open source under Apache, MIT, or GPL. Its bundled
audEERING Research License permits non-commercial research, education, and
personal experimentation while prohibiting commercial product use without a
separate license. Vox therefore does not bundle it, publish it, install it as a
normal dependency, or imply GPL compatibility. This experiment installs it
only after the operator explicitly acknowledges those terms:

- [openSMILE license](https://github.com/audeering/opensmile/blob/master/LICENSE)
- [openSMILE Python documentation](https://audeering.github.io/opensmile-python/)

This licensing gate must be resolved before any production or public API work.

## Install and run

Record a browser microphone sample as mono 16 kHz PCM16 WAV:

```bash
make speech-context-recorder
```

Then open `http://127.0.0.1:11436/speech-context-recorder.html`, record the
sample, and save it. The recorder releases the microphone immediately after
stopping and does not upload the audio anywhere.

The install is deliberately separate from analysis:

```bash
uv run python scripts/speech-context-evidence.py install \
  --accept-opensmile-research-license

uv run python scripts/speech-context-evidence.py analyze ./speech.wav \
  --vox-url http://127.0.0.1:11435 \
  --output ./speech-context.json
```

Set `VOX_API_KEY` instead of placing the key on the command line when the Vox
server requires authentication. `--model` can override the default
`parakeet-stt:tdt-0.6b-v3`.

The command writes evidence even if one analyzer fails and exits with status 3
in that case. This preserves successful evidence while making partial results
impossible to mistake for a complete run. Temporary canonical audio and the
installer's private download caches are removed automatically.

## Evidence shape

The file records:

- original and canonical audio hashes, byte counts, duration, channels, and
  sample rate;
- wall time for each analysis and the concurrent run;
- process CPU time and peak RSS for local workers;
- explicit unavailable values for Parakeet server CPU, RAM, VRAM, and model
  size, because an HTTP client cannot measure those honestly;
- zero GPU use for the two CPU-only local analyzers;
- isolated runtime and model sizes;
- complete verbose Parakeet output;
- every eGeMAPSv02 low-level and functional value;
- every YAMNet class, score, embedding value, and spectrogram value.

The compact `speech_context` projection removes representation detail rather
than limiting which real-world events are allowed. Every class in YAMNet's
catalog is eligible. Event scores below `0.05` are removed. A surviving class
must appear in at least two overlapping score windows unless one window scores
at least `0.2`. Only the three strongest eligible classes in each score window
are considered. Consecutive windows for the same class become one bounded event
candidate whose score is the maximum supporting score. An AudioSet ancestor is
removed when a more specific descendant explains at least 80 percent of its
time range. Events are emitted as candidates because YAMNet scores are not
calibrated confidence values. Repeated occurrences of the same AudioSet class
share one candidate with compact `[start_ms, end_ms, score]` spans. AudioSet
class IDs and ancestry remain internal evidence and are not copied into the
model-facing projection.

The prosody projection removes low-level descriptor frames and all but 14
conversation-relevant eGeMAPSv02 functionals. The retained groups describe
pitch, energy, voice quality, spectral variation, and delivery. Durations are
converted to milliseconds and finite values are rounded to three decimal
places. Audio-event scores are rounded to two decimal places; embeddings and
log-mel spectrograms are absent from the compact projection. Malformed class
catalogs, ontology data, score vectors, timestamps, or required functional
vectors fail reduction explicitly while the complete analyzer evidence remains
in the file.

An experimental result therefore has two distinct representations:

```text
results.*.raw       complete analyzer evidence for evaluation and debugging
speech_context      compact lossy evidence intended to test application value
```

The reducer is stateless. It does not maintain speaker baselines, infer age or
gender, select event labels by name, or decide how an application should alter
its response.

The prospective public vocabulary is deliberately generic: transcription,
prosody, and audio events. Provider names, dependency names, source metadata,
and VAD internals are not part of a public schema.

## Why VAD is absent

VAD already owns speech activity in Vox, but there is not yet measured evidence
that a required context signal is missing from the selected prosody and event
tracks. Adding VAD-derived pause counts now would duplicate ownership and
prejudge the experiment. No VAD object or field is emitted. After real
controlled audio has been reviewed, VAD should be added only for a
specific missing signal with a named regression test.
