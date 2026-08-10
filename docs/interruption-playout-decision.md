# Interruption playout decision

## Problem

Transcript-free acoustic confirmation made Vox feel immediate but allowed echo,
room noise, and other VAD false starts to destroy active responses. Removing
that confirmation in v0.2.128 fixed destructive false positives but allowed the
assistant to keep speaking until partial STT arrived, about 1.56 seconds in the
observed browser call.

Server-side TTS holding alone does not make the call quiet. The RTC output track
may already contain up to two seconds of audio, and the browser has its own RTP
and jitter buffers. A response can therefore stop producing new chunks while
old speech remains audible.

## Decision

VAD onset creates one interruption candidate and one candidate-owned reversible
playout suspension:

- Conversation output holds future TTS chunks.
- RTC playout emits paced silence without consuming its pending frame or queue.
- Browser playback mutes the existing media element without pausing, loading,
  replacing, or discarding its stream.
- `response.audio.suspend` and `response.audio.resume` carry the candidate ID.
  A stale resume cannot release a newer candidate's suspension.
- Empty STT, no-transcript speech stop, detector failure, timer failure, and the
  evidence deadline release the suspension without cancelling the response.
- Only transcript-supported interruption confirmation emits
  `response.audio.clear`, cancels the response, and destroys queued audio.

The turn state remains `speaking` while a reversible suspension is active. The
interruption detector owns candidate truth; the response output owns future
audio holding; the RTC track owns server playout; and the browser SDK owns local
media muting.

## Latency work

The partial-ASR buffer is seeded with retained audio preceding the input chunk
that triggered VAD. Speech-context workers preload at server startup, and
speech-context analysis begins from the retained onset audio so it overlaps the
utterance and endpoint silence instead of starting after final STT.

The live Silero path consumes each new 512-sample frame once and preserves its
recurrent state and 64-sample context for the lifetime of the VAD processor.
The stateless batch timestamp API remains separate for offline callers.

Vox records two server-domain measurements for live calls:

- `vad_detection_latency_ms` measures the VAD-reported segment start to the
  server audio position where Vox emits the VAD event. The segment start may
  include configured VAD padding, so it is not a physical microphone-onset
  measurement.
- `suspend_to_silence_ms` measures RTC suspension request to the first silent
  output frame.

These measurements do not include microphone capture, network transit, browser
rendering, device buffering, or the physical speaker. A physical
speech-to-quiet number must come from synchronized client or loopback capture;
the server measurements must not be presented as that end-to-end result.

## Lifecycle ownership

A new response cannot reset output sequencing while a suspension owner or held
audio survives. The invariant is checked before response-stream creation. A
session close or RTC track stop terminally clears the owner and retained audio;
a browser disconnect restores the media element's pre-suspension mute state.

## Rejected alternatives

- Acoustic-only hard cancellation: fast but cannot distinguish the user's
  voice from speech-like echo when waveform correlation misses.
- Server-side hold without RTC/browser suspension: preserves generated audio
  but does not stop already-buffered speech.
- Volume ducking as the primary contract: reduces loudness but does not satisfy
  time-to-quietness.
- Restoring the legacy `paused` turn state: conflates turn semantics with three
  independent playout buffers and previously allowed timeout paths to strand
  output.
- Shortening the 800 ms endpoint silence alone: affects turn completion, not
  assistant time-to-quietness, and raises truncation risk without solving
  buffered playout.

## Required regressions

- A candidate suspends all three playout layers without clearing audio.
- A false candidate resumes the original response and preserved queue.
- Candidate identity rejects stale resumes.
- An in-flight RTC frame cannot escape after suspension and is not dropped.
- Empty STT never clears or cancels the response.
- A transcript-supported partial still clears and cancels promptly.
- Pre-onset audio contributes to the first partial-ASR window.
- Speech-context startup and analysis do not begin on the final-transcript
  critical path.
- Response completion, session close, RTC stop, and browser disconnect cannot
  strand an owned suspension.
