# Endpointing benchmark

The endpointing corpus links policy cases to five authoritative recorded WAV
files. The recordings are external test fixtures and are not stored in the
repository.
[`benchmarks/endpointing_recorded.json`](../benchmarks/endpointing_recorded.json)
pins each filename, SHA-256 digest, channel count, sample rate, sample width,
frame count, and duration.

## Evidence levels

The benchmark has three separate evidence levels.

### Recording verification

```console
uv run python scripts/benchmark_endpointing.py verify-recordings \
  --recordings-dir /path/to/recordings
```

This command is live evidence. It requires every authoritative WAV, verifies
the complete-file SHA-256 digest, reads the PCM container, compares all pinned
audio properties, consumes all frames, and passes the bytes through Vox's
current `prepare_for_stt` decode, mono, resample, normalize, and float32
conversion path. Missing, changed, malformed, or property-mismatched recordings
fail the command.

The unit suite exercises the same verifier with deterministic generated WAV
fixtures. Authoritative recording verification remains an explicit evidence
gate and fails when any requested fixture is missing.

### Runtime extraction

```console
VOX_API_KEY=... uv run python scripts/benchmark_endpointing.py runtime \
  --recordings-dir /path/to/recordings \
  --vox-url https://vox.example.test \
  --model parakeet-stt:tdt-0.6b-v3 \
  --verify \
  --output endpointing-runtime.json
```

This command is live model-backed evidence. It first performs recording
verification. It then sends every full WAV through the running Vox
`POST /v1/audio/transcriptions` operation with verbose segment and word
timestamps, and rescores every cumulative continuation and terminal transcript
with Vox's current LiveKit `v1.2.2-en` EOU implementation. The generated report
contains current terminal transcripts, STT processing times, EOU scores, and
deltas from the recorded corpus.

`--verify` exits unsuccessfully unless every current full-file transcription
matches the recorded terminal transcript after whitespace and case
normalization and every EOU score remains within `--eou-tolerance`, which
defaults to `0.01`. Without `--verify`, the command generates an inspection
report without treating model drift as a command failure.

This command requires a running Vox server with the requested STT model
installed. It is intentionally outside the ordinary unit suite because real
Parakeet execution is hardware and runtime dependent. LiveKit EOU remains a CPU
model but still requires its pinned model assets.

### Policy arithmetic

```console
uv run python scripts/benchmark_endpointing.py policy
```

This command is lightweight arithmetic over the checked-in corpus. It does not
open a recording and does not run STT, VAD, or EOU. Its purpose is to compare
delay policies after recording identity and model observations have been
established by the other commands.

## Recorded observations

The corpus contains eight interior continuation pauses from 560 to 2400
milliseconds and five terminal utterances. The transcript prefixes, pause
durations, EOU probabilities, and Parakeet processing times are recorded
observations from the original streaming capture. They are not live evidence
merely because their source WAV hashes are valid.

The full WAVs do not contain the original streaming event clock or preceding
conversation history. Therefore the runtime command can re-extract full-file
STT and rescore the stored cumulative transcript prefixes, but it cannot
independently reconstruct the original `pause_ms` observations from the WAV
container alone. The policy command treats those pause durations as recorded
annotations. It does not present them as newly measured values.

Prefix-specific Parakeet processing times were not captured. Continuation
calculations use zero processing time rather than borrowing the full-file
measurement and making the policy appear safer than it is. Terminal latency
adds the recorded full-file Parakeet processing time to the post-transcript
delay.

The current policy is evaluated with production dynamic endpointing enabled.
Continuation pauses are added to each source's history in chronological corpus
order. A false endpoint does not enter pause history. Pause history can extend
an incomplete turn, but a complete turn does not inherit that extension.

## Policy results

| Policy | False endpoints | Rate | Mean terminal latency | Mean complete-turn latency |
| --- | ---: | ---: | ---: | ---: |
| `v0.2.94` | 3/8 | 37.5% | 1200.2 ms | 1088.3 ms |
| `v0.2.123` | 8/8 | 100.0% | 780.6 ms | 698.3 ms |
| Current candidate | 3/8 | 37.5% | 1081.8 ms | 903.3 ms |

The current policy preserves the `v0.2.123` delay exactly for EOU probabilities
at or above 0.85. Below 0.50 it scales from 1000 to 1200 milliseconds. Between
0.50 and 0.85 it interpolates from the uncertain allowance to the existing
high-confidence curve. Dynamic pause history can lengthen the allowance only
while EOU remains below the completion threshold.

On the recorded observations, the current policy matches `v0.2.94` on false
endpoints. Its overall mean terminal latency is 118.4 milliseconds lower.
Among samples scored complete, mean terminal latency is 185.0 milliseconds
lower than `v0.2.94` and 205.0 milliseconds higher than `v0.2.123`. That latter
cost is deliberate: recorded continuation prefixes scoring from 0.61 to 0.75
include natural pauses, so threshold-complete does not mean confidently
complete. Moving the high-confidence boundary to 0.70 increases recorded false
endpoints from three to five. Scores at 0.85 and above retain the `v0.2.123`
latency exactly.

The three remaining recorded false endpoints are either pauses longer than two
seconds or transcript prefixes that LiveKit scored above the completion
threshold. Extending every high-confidence turn enough to cover those cases
would restore a fixed latency penalty to confidently complete turns, so the
policy deliberately leaves those cases unchanged.

`EndpointPauseHistory` records resumed-speech pauses while a turn remains open.
STT processing time is not pause history. It contributes to measured terminal
latency but does not increase a subsequent post-transcript delay.

## Final runtime verification

The final stabilization run on 2026-07-26 verified all five authoritative WAVs
against the running Parakeet model and LiveKit `v1.2.2-en`. Every normalized
terminal transcript matched. Parakeet processing times were 307, 219, 278, 278,
and 294 milliseconds. All 13 EOU observations matched the pinned corpus with a
maximum absolute delta of `0.000000464382171605493`.
