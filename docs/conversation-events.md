# Vox Conversation Event Guide

This guide describes how a client or agent process should use Vox conversation
streaming over PondSocket, gRPC, or Vox-hosted WebRTC.

The conversation API is agent-facing. The caller owns LLM text generation. Vox
owns VAD, STT, EOU scoring, turn state, TTS, and interruption signaling. With
the PondSocket and gRPC APIs the caller also owns microphone capture and playback.
With the RTC API, Vox owns the browser media connection and the caller owns the
control stream.

## Acoustic Echo Cancellation

If assistant audio is played through loudspeakers, the client or agent process
must run acoustic echo cancellation before sending microphone audio to Vox.
Otherwise the microphone can capture Vox's own TTS, causing VAD to report user
speech during assistant playback.

Recommended client-side capture settings:

- Enable acoustic echo cancellation.
- Enable noise suppression where available.
- Enable automatic gain control only if it improves your microphone path; AGC can
  sometimes amplify room echo.
- Send Vox the post-AEC microphone signal, not raw microphone audio.

For browser/WebRTC clients, request these constraints on the microphone track:

```js
const stream = await navigator.mediaDevices.getUserMedia({
  audio: {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true
  }
});
```

Vox also has a server-side self-echo guard: when a possible interruption
transcribes as text that closely matches the active assistant response, Vox emits
`interruption.false_positive` and resumes instead of cancelling itself. This is a
safety net, not a replacement for AEC. True AEC needs the far-end playback
reference before the speaker audio leaks into the microphone.

While assistant audio is active, Vox uses stricter interruption evidence by
default:

- `speaking_interrupt_min_duration_ms`: minimum confirm window during assistant
  playback.
- `speaking_interrupt_min_words`: minimum non-keyword user words needed before a
  mid-playback interruption can cancel TTS.
- `self_echo_min_words` and `self_echo_min_overlap`: how much partial transcript
  text must match active assistant text before Vox rejects it as likely self-echo.

## Transports

### RTC Session Control

Vox can create short-lived RTC session bindings for browser WebRTC media plus a
developer-owned control channel. The browser sends microphone audio to Vox over
WebRTC and receives assistant audio over WebRTC. The developer backend attaches
to a control stream to configure the conversation and stream assistant text into
Vox.

RTC sessions can also carry arbitrary JSON application events over a WebRTC data
channel. Vox relays those JSON payloads between the browser data channel and the
developer control stream without imposing an application schema.

Create a session:

```http
POST /v1/rtc/sessions
```

Response:

```json
{
  "session_id": "rtc_...",
  "client_token": "rtc_client_...",
  "expires_at": "2026-05-08T12:34:56+00:00",
  "join_token_ttl_seconds": 120,
  "ice_servers": [
    {
      "urls": ["stun:turn.example.com:3478"]
    },
    {
      "urls": ["turn:turn.example.com:3478?transport=udp", "turns:turn.example.com:5349"],
      "username": "1760000000",
      "credential": "<short-lived credential>"
    }
  ]
}
```

The `client_token` is for the browser/media side and is single-use. Its expiry
only controls how long the browser has to join; once joined, token expiry should
not end an active call.

Browser joins with an SDP offer:

```http
POST /v1/rtc/sessions/{session_id}/offer
Authorization: Bearer rtc_client_...
Content-Type: application/json
```

```json
{
  "type": "offer",
  "sdp": "v=0\r\n..."
}
```

The token may also be sent as `client_token` in the JSON body. The response is:

```json
{
  "session_id": "rtc_...",
  "media_token": "rtc_media_...",
  "events_url": "/v1/rtc/sessions/rtc_.../events?token=rtc_media_...",
  "type": "answer",
  "sdp": "v=0\r\n..."
}
```

The `media_token` is for media-side follow-up calls only. It is different from
the single-use `client_token`.

Browser receives Vox-side trickle ICE over Server-Sent Events:

```http
GET /v1/rtc/sessions/{session_id}/events?token=rtc_media_...
Accept: text/event-stream
```

Each SSE message has an `event` matching the payload `type`. Important events:

- `rtc.ice_candidate`: add `candidate` to the browser peer connection. If
  `candidate` is `null`, call `addIceCandidate(null)` or otherwise mark end of
  candidates in the browser stack.
- `rtc.connection_state`: Vox peer connection state changed.
- `rtc.ice_connection_state`: ICE state changed.
- `rtc.ice_gathering_state`: Vox ICE gathering state changed.

Browser sends its trickle ICE candidates back to Vox:

```http
POST /v1/rtc/sessions/{session_id}/candidates
Authorization: Bearer rtc_media_...
Content-Type: application/json
```

```json
{
  "candidate": {
    "candidate": "candidate:...",
    "sdpMid": "0",
    "sdpMLineIndex": 0
  }
}
```

At end-of-candidates, send:

```json
{
  "candidate": null
}
```

Developer backend can attach to the control-only stream over PondSocket:

```text
/v1/socket channel /rtc/{session_id}
```

This stream accepts `session.update`, `response.start`, `response.delta`,
`response.commit`, `response.cancel`, and `client.event`. It emits the same
conversation events as the conversation control channel, plus `browser.event` for browser data
channel payloads. It does not emit `response.audio.delta`; assistant audio
belongs on the WebRTC media path for this session.

### Browser-native events

Vox also forwards a curated subset of conversation events directly to the
browser data channel, so a browser client can render captions, state
indicators, and barge-in feedback without a hand-built backend relay. Each is
delivered as a data-channel message `{"event": "<wire type>", "payload": {...}}`
where `<wire type>` and payload match the control-stream event minus `type`:

- `turn.state_changed`
- `input_audio_buffer.speech_started` / `input_audio_buffer.speech_stopped`
- `conversation.item.input_audio_transcription.delta` and `.completed`
- `interruption.detected` / `interruption.false_positive`
- `response.created` / `response.done` / `response.cancelled`
- `response.audio.clear` — mute or duck local playback immediately; in-flight
  RTP plus the jitter buffer can otherwise play 100-300 ms of stale assistant
  audio after a barge-in

`response.audio.delta` is never forwarded; assistant audio stays on the media
path. Forwarding only happens while the data channel is open (missed events are
not buffered). Disable it by creating the session with
`POST /v1/rtc/sessions {"browser_events": false}`.

Developer backends can also attach over gRPC:

```text
RtcService.Control(stream RtcControlClientMessage) returns (stream ConverseServerMessage)
```

Expected gRPC control startup:

1. Open the bidi gRPC stream.
2. Send `RtcControlAttach { session_id }` as the first message.
3. Wait for `rtc_session_attached`.
4. Send `session_update`.
5. Drive `response_start`, `response_delta`, `response_commit`, and
   `response_cancel` the same way as the PondSocket control stream.
6. Use `RtcClientEvent.event` plus `RtcClientEvent.payload_json` for
   application events that should be relayed to the browser data channel.

The RTC gRPC control stream is backend-facing. It is not intended to replace
browser WebRTC media, and it is not a browser-native transport.

The expected RTC startup order is:

1. Developer backend creates `POST /v1/rtc/sessions`.
2. Developer backend gives `session_id`, `client_token`, and `ice_servers` to the
   browser through its own application protocol.
3. Browser creates a `RTCPeerConnection`, adds its microphone track with AEC
   enabled, and posts the SDP offer to `/offer`.
4. Browser applies Vox's SDP answer.
5. Browser listens to `events_url` for Vox-side trickle ICE and posts browser ICE
   candidates to `/candidates`.
6. Developer backend joins the PondSocket `/rtc/{session_id}` channel or attaches to
   `RtcService.Control` over gRPC, sends `session.update`, waits for
   `session.created`, and then drives assistant text with `response.delta` and
   `response.commit`.
7. Vox sends user transcripts and interruption events over the control stream; Vox sends
   assistant audio over WebRTC.

Optional data channel:

- The browser may create a reliable ordered WebRTC data channel such as
  `vox-events`.
- Backend to browser:
  - PondSocket control: send
    `{ "type": "client.event", "event": "<name>", "payload": <any JSON> }`
  - gRPC control: send
    `RtcClientEvent { event: "<name>", payload_json: "<valid JSON>" }`
- Browser to backend:
  - Browser sends JSON text shaped as
    `{ "event": "<name>", "payload": <any JSON> }` on the WebRTC data channel.
  - Vox relays that message to the control stream as `browser.event`
    (PondSocket) or `RtcClientEvent` (gRPC).
- Vox does not define the meaning of `event` names or payload contents. It only
  requires the transport envelope.

RTC diagnostics:

- Browser clients may send `rtc.stats` over the WebRTC data channel using the
  normal browser-to-backend data event envelope. Vox relays it as `browser.event`
  so application backends can correlate browser WebRTC health with Vox-side turn
  timing. Vox treats `rtc.stats` as application telemetry and does not use it for
  VAD, EOU, interruption, or TTS scheduling.

ICE servers are configured with environment variables:

```text
VOX_RTC_STUN_URLS=stun:turn.horus:3478
VOX_RTC_TURN_URLS=turn:turn.horus:3478?transport=udp,turns:turn.horus:5349
VOX_RTC_TURN_SECRET=<coturn static-auth-secret>
VOX_RTC_TURN_CREDENTIAL_TTL_SECONDS=3600
```

For local-only use, leave these unset and Vox returns `ice_servers: []`.

If a browser app is served from a different origin than Vox, enable CORS
explicitly:

```text
VOX_CORS_ORIGINS=http://localhost:8000,https://your-app.example.com
```

A minimal native browser client is available at
`examples/rtc-browser-client.html`. It uses browser WebRTC APIs directly; no npm
RTC client package is required.

### PondSocket

Connect to the Vox PondSocket gateway and join:

```text
/v1/socket channel /conversation/{session_id}
```

Client events use the same conversation event names. Audio payloads are
base64-encoded PCM16.

### gRPC

Use bidi streaming RPC:

```text
ConversationService.Converse(stream ConverseClientMessage) returns (stream ConverseServerMessage)
```

The gRPC fields mirror the conversation event names. For example,
`response.audio.delta` maps to gRPC `audio_delta`.

## Startup

1. Join the PondSocket channel or open the gRPC stream.
2. Send `session.update` / `session_update`.
3. Wait for `session.created` / `session_created`.
4. Start sending user audio with `input_audio_buffer.append` / `audio_append`.

Example session update:

```json
{
  "type": "session.update",
  "session": {
    "stt_model": "parakeet-stt:tdt-0.6b-v3",
    "tts_model": "kokoro-tts:v1.0",
    "voice": "af_heart",
    "language": "en",
    "sample_rate": 16000,
    "vad_backend": "silero",
    "turn_detector": "livekit",
    "include_word_timestamps": false,
    "turn_policy": {
      "allow_interrupt_while_speaking": true,
      "min_interrupt_duration_ms": 250,
      "max_endpointing_delay_ms": 3000,
      "stable_speaking_min_ms": 150,
      "false_interruption_timeout_ms": 2000,
      "min_interrupt_words": 0,
      "partial_interrupts": true,
      "dynamic_endpointing": true,
      "min_endpointing_delay_ms": 400,
      "speaking_interrupt_min_duration_ms": 500,
      "speaking_interrupt_min_words": 2,
      "self_echo_min_words": 3,
      "self_echo_min_overlap": 0.7,
      "vad_min_silence_ms": 1000
    }
  }
}
```

`include_word_timestamps` (default `false`) asks the STT adapter for word-level
timings on final transcripts; when enabled, `words` appears on
`conversation.item.input_audio_transcription.completed`.

`vad_min_silence_ms` controls how much trailing silence the VAD requires before
declaring end of speech. It is the dominant fixed contributor to endpointing
latency; profiles tune it (`headset` 600, `browser_default` 800, `noisy_room`
1200, otherwise 1000).

Endpointing is EOU-aware: when the semantic turn detector reports a probability
at or above its threshold, the transcript commit delay shrinks from the
continuation wait toward `min_endpointing_delay_ms` in proportion to the model's
confidence. Low-probability (incomplete) turns keep the full continuation wait,
bounded by `max_endpointing_delay_ms`.

`vad_backend` defaults to `silero`, which runs the Silero VAD model on
onnxruntime — no PyTorch dependency and no runtime model download (the model
ships with the package). `silero-torch` selects the legacy PyTorch loader
(`torch.hub`) and is kept only as a fallback; it needs `torch` installed.
`ten-vad` is an experimental alternative that needs the optional `ten-vad`
package. Both non-default backends should be tested against your own audio
before use.

`turn_detector` defaults to `livekit`, Vox's lightweight semantic EOU detector.
Experimental values such as `ten-turn` are intended for benchmarking heavier
semantic models, not for low-resource default deployments.

### Session Update Compatibility Fields

The canonical JSON fields for `session.update` are `stt_model`, `tts_model`,
`turn_profile`, `vad_backend`, and `turn_detector`.

For OpenAI Realtime-style clients and older Vox clients, the JSON parser also
accepts these explicit compatibility fields:

- `input_audio_transcription.model` for `stt_model`
- `output_audio_generation.model` for `tts_model`
- `profile` for `turn_profile`
- `vad` for `vad_backend`
- `eou_model` for `turn_detector`

When both canonical and compatibility fields are present, the canonical field
wins. gRPC uses the canonical protobuf fields only.

## Turn Profiles

Vox now supports server-owned turn profiles so callers can choose an acoustic
mode without copying threshold bundles into every client.

Supported profile names:

- `default`
- `browser_default`
- `headset`
- `speakerphone`
- `noisy_room`

Aliases accepted by the API:

- `browser` -> `browser_default`
- `web` -> `browser_default`
- `headphones` -> `headset`
- `speaker` -> `speakerphone`
- `loudspeaker` -> `speakerphone`
- `noisy` -> `noisy_room`

Use `turn_profile` on `session.update` / `session_update`:

```json
{
  "type": "session.update",
  "session": {
    "stt_model": "parakeet-stt:tdt-0.6b-v3",
    "tts_model": "kokoro-tts:v1.0",
    "voice": "af_heart",
    "turn_profile": "speakerphone"
  }
}
```

The returned `session.created` / `session_created` event includes both the
resolved `turn_profile` and the fully resolved `turn_policy`.

Recommended usage:

- choose a profile first
- only send `turn_policy` overrides when you have a concrete reason to diverge
- prefer `browser_default` for generic browser/WebRTC clients
- prefer `headset` when speaker leakage is minimal
- prefer `speakerphone` for loudspeaker tests and open-air playback
- prefer `noisy_room` when false starts are more expensive than a little extra
  interruption latency

Example audio append:

```json
{
  "type": "input_audio_buffer.append",
  "audio": "<base64 pcm16>",
  "sample_rate": 16000
}
```

## Producing Assistant Speech

When the client decides the assistant should speak:

1. Send `response.start` if you want an explicit response boundary.
2. Send one or more `response.delta` messages with text.
3. Send `response.commit` when no more text is coming.
4. Play `response.audio.delta` chunks as they arrive.
5. Treat `response.done` as the end of the TTS response.

Minimal PondSocket event sequence:

```json
{"type":"response.start"}
{"type":"response.delta","delta":"Hello. How can I help?"}
{"type":"response.commit"}
```

Vox may begin TTS before `response.commit` when text reaches a good chunk boundary. Do not assume audio only starts after commit.

## Client Playback Rules

Maintain an output playback queue for `response.audio.delta`.

When `response.audio.delta` / `audio_delta` arrives:

1. Decode the PCM16 audio.
2. Enqueue it for playback.
3. Use the provided `sample_rate`.

When `response.audio.clear` / `audio_clear` arrives:

1. Stop current assistant audio immediately.
2. Drop all queued assistant audio.
3. Do not replay previously received audio for the cancelled response.

When `response.cancelled` / `response_cancelled` arrives:

1. Mark the current assistant response as cancelled.
2. Stop any LLM text generation still feeding Vox.
3. Expect no more useful audio for that response.

`response.audio.clear` is the immediate playback instruction. `response.cancelled` is the response lifecycle instruction. Handle both.

## Speaking Interruption Contract

When assistant TTS is active, treat microphone activity as a two-stage
interruption check, not as an immediate cancel.

### 1. Candidate interruption

When Vox is in `speaking` and receives `input_audio_buffer.speech_started`, that
means "possible interruption", not "confirmed interruption".

Client rules:

- Do not stop playback on `input_audio_buffer.speech_started` alone.
- Do not drop queued audio on `input_audio_buffer.speech_stopped` alone.
- Do not treat `turn.state_changed: paused` as a playback-clear command.
- Keep playing until Vox either rejects the candidate or sends
  `response.audio.clear`.

### 2. Rejected candidate / false positive

If Vox decides the speech candidate was echo, leakage from loudspeakers, a short
backchannel, or other non-interrupting audio, it emits
`interruption.false_positive`.

Client rules:

- Do not cancel the active response.
- Keep or resume normal playback.
- Do not start a new assistant turn from this event.

### 3. Confirmed interruption

`response.audio.clear` is the authoritative playback-stop command for a real
barge-in.

Client rules:

- Stop current assistant playback immediately.
- Drop all queued audio for that `response_id`.
- Ignore any stale audio chunks for that cleared response.
- Treat `interruption.detected`, `response.cancelled`, and later state changes as
  lifecycle/state updates. They are not a substitute for `response.audio.clear`.

The ordering of `response.audio.clear`, `interruption.detected`,
`response.cancelled`, and nearby `turn.state_changed` events is not the client
contract. The contract is that `response.audio.clear` is the playback stop
signal.

### 4. Post-interrupt handoff

After a confirmed interruption, wait for the user transcript and
`turn.state_changed: thinking` before starting the next assistant response.

## Server Events

### `session.created`

The session is configured and ready. Start sending audio only after this event.

### `input_audio_buffer.speech_started`

VAD detected user speech.

Client behavior:

- Show listening/user-speaking UI.
- If assistant audio is playing, treat this as a candidate interruption only.
- Keep playing until Vox either resumes or sends `response.audio.clear`.
- Do not locally cancel assistant audio on this event alone. Vox may classify it as a cough or backchannel.

### `input_audio_buffer.speech_stopped`

VAD detected that user speech stopped.

Client behavior:

- Show that user speech ended.
- Keep waiting for transcript and turn state events.

### `conversation.item.input_audio_transcription.delta`

An interim transcript fragment for in-progress user speech. Emitted while the
user is still speaking, roughly every partial stride (~700 ms), carrying only
newly confirmed words.

Payload includes:

- `delta`: the newly confirmed text fragment
- `start_ms`, `end_ms`

Client behavior:

- Append `delta` fragments to build a live caption of the current user turn.
- Reset the caption on the next `conversation.item.input_audio_transcription.completed`,
  which remains the authoritative final text.
- Partials require `partial_interrupts` (on by default) or an interrupt keyword
  classifier; with both disabled no delta events are produced.

### `conversation.item.input_audio_transcription.completed`

STT completed for a user speech segment.

Payload includes:

- `transcript`
- `language`
- `start_ms`
- `end_ms`
- optional `eou_probability`
- optional `entities`, `topics`, `words`

Client behavior:

- Display or store the final user transcript.
- Use `eou_probability` as metadata only. Vox already uses it internally for turn timing.

### `turn.eou.predicted`

The semantic turn detector scored whether the user appears done.

Payload includes:

- `probability`
- `threshold`
- `decision`: `complete` or `incomplete`
- `action`: `commit` or `wait`
- `delay_ms`
- `turn_detector`
- `start_ms`, `end_ms`

Client behavior:

- Treat this as observability for turn timing, not as a separate command.
- Use it for debugging/benchmarks when Vox commits too early or waits too long.
- Start assistant generation from `turn.state_changed: thinking`, not from this event alone.

### `turn.state_changed`

The turn state changed.

States:

- `idle`: no active user speech or assistant speech.
- `listening`: user is speaking or Vox is waiting for possible continuation.
- `thinking`: user turn ended; client/agent should produce assistant text.
- `speaking`: TTS audio is being emitted.
- `paused`: user speech started during assistant speech; Vox is checking whether this is a real interruption.
- `interrupted`: barge-in confirmed; assistant speech was cancelled.

Client behavior:

- On `thinking`: start or continue LLM generation and send response text to Vox.
- On `speaking`: show assistant-speaking UI.
- On `paused`: do not clear playback locally; Vox may resume or clear soon.
- On `interrupted`: treat the assistant response as interrupted and prepare for a new user turn.

### `response.created`

Vox accepted a new assistant response stream.

Client behavior:

- Associate later audio and lifecycle events with this response.
- Store the `response_id`. Newer Vox events include this ID on response audio, clear, commit, cancel, and done events.

### `response.committed`

Vox received the response commit marker.

Client behavior:

- Stop sending text deltas for this response.
- Keep playing audio until `response.done`, `response.cancelled`, or `response.audio.clear`.

### `response.audio.delta`

Assistant audio chunk.

Payload:

- PondSocket: base64 `audio`, `sample_rate`, `audio_format`
- gRPC: raw PCM bytes and `sample_rate`
- Newer Vox versions also include `response_id` and monotonic per-response `sequence`.

Client behavior:

- Enqueue for playback unless the response was already cancelled.
- Drop duplicate or stale chunks if their `response_id` no longer matches the active response.
- Preserve chunk order by `sequence` when your playback layer can reorder frames.

### `response.audio.clear`

Confirmed interruption playback-clear signal.

Client behavior:

- Stop current playback immediately.
- Clear all queued assistant audio.
- Treat any already-buffered audio for the current response as invalid.
- Apply this to the matching `response_id` when present.

### `interruption.detected`

Vox confirmed that user speech during assistant speech is a real barge-in.

Payload includes:

- `response_id`
- `vad_active_ms`
- optional `partial_transcript`

Client behavior:

- Expect `response.audio.clear` and `response.cancelled` around this event.
- Do not rely on event ordering here; `response.audio.clear` remains the
  playback-stop command.
- Stop upstream LLM generation for that response if it is still producing deltas.

### `interruption.false_positive`

Vox rejected a possible barge-in as a false interruption or backchannel.

Payload includes:

- `response_id`
- `vad_active_ms`
- optional `partial_transcript`
- optional `reason`: why the candidate was rejected — one of `backchannel`,
  `output_echo`, `self_echo_transcript`, `self_echo_transcript_window`,
  `insufficient_interrupt_evidence`

Client behavior:

- Do not cancel the assistant response.
- Continue or resume playback as normal.
- Treat this as a rejected interruption candidate, not as an error.

### `response.cancelled`

The assistant response was cancelled, usually because of a confirmed barge-in or explicit client cancel.

Client behavior:

- Mark the response cancelled.
- Stop feeding more response text.
- Pair this with `response.audio.clear` for playback cleanup when present.

### `response.done`

TTS completed normally.

Client behavior:

- Mark the response complete.
- Return to idle/ready UI unless another state event says otherwise.

### `client.event`

Generic application JSON relayed between the backend control stream and the RTC
browser data channel.

Payload includes:

- PondSocket control:
  - `event`: string event name
  - `payload`: any valid JSON value
- gRPC control:
  - `RtcClientEvent.event`: string event name
  - `RtcClientEvent.payload_json`: valid JSON document

Client behavior:

- Do not feed this into the conversation state machine.
- Treat it as an application/UI event lane.
- Use it for URLs, images, cards, citations, progress messages, tool results, or
  any other structured app payload.
- Do not use it for assistant audio transport.

### `error`

The session hit an error.

Client behavior:

- Log/display the message.
- Depending on severity, close and recreate the conversation session.

## Recommended Client State Machine

Use these client-side states:

- `disconnected`
- `connecting`
- `ready`
- `user_speaking`
- `agent_thinking`
- `agent_speaking`
- `agent_paused_for_possible_barge_in`
- `interrupted`
- `error`

Recommended transitions:

| Incoming event | Client action |
| --- | --- |
| `session.created` | Move to `ready`; begin sending audio. |
| `input_audio_buffer.speech_started` | Move to `user_speaking` unless assistant is speaking; if assistant is speaking, wait for `paused`, `audio.clear`, or resume. |
| `turn.state_changed: listening` | Show user/listening state. |
| `turn.eou.predicted` | Record semantic EOU decision for observability. |
| `turn.state_changed: thinking` | Start LLM generation; send `response.start`, `response.delta`, `response.commit`. |
| `response.created` | Create local response record. |
| `response.audio.delta` | Enqueue assistant audio. |
| `turn.state_changed: speaking` | Move to `agent_speaking`. |
| `turn.state_changed: paused` | Move to `agent_paused_for_possible_barge_in`; keep current audio handling until further instruction. |
| `interruption.detected` | Stop upstream generation for the active response and wait for clear/cancel lifecycle events. |
| `interruption.false_positive` | Resume normal assistant playback state. |
| `response.audio.clear` | Stop playback and drop queued assistant audio immediately. |
| `response.cancelled` | Mark response cancelled; stop sending deltas for it. |
| `turn.state_changed: interrupted` | Move to `interrupted`; wait for the user's transcript or next listening/thinking state. |
| `response.done` | Mark response complete. |
| `error` | Move to `error`; usually close and reconnect. |

## Important Rules

- Do not treat every `speech_started` during TTS as a confirmed interruption. Vox may reject it as a backchannel and resume TTS.
- Only `response.audio.clear` tells the client to stop playback. `speech_started`,
  `speech_stopped`, and `turn.state_changed: paused` do not.
- Run client-side AEC when using loudspeaker playback; Vox should receive the post-AEC mic stream.
- Do stop playback immediately on `response.audio.clear`.
- Do stop sending response text on `response.cancelled`.
- Do not assume `response.done` always arrives after cancellation.
- Do not send a second response while one is still active unless the previous one is done or cancelled.
- Use `turn.state_changed: thinking` as the strongest signal that the user turn is ready for an assistant response.
- Keep audio chunks ordered by `sequence` when present.
- Use `response_id` to ignore stale audio after interruption or cancellation.
- Treat non-default `vad_backend` and `turn_detector` values as experiment flags until you have measured latency and false interruption rate on your own calls.
- Treat `turn.eou.predicted` as a debug/tuning event; the state machine remains authoritative.
