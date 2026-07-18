# RTC response-lifecycle stabilization plan

Source: structural audit findings F1-F8. Sequencing: phased, quick wins first.
The four uncommitted files (THINKING-state fix + tests) are part of Phase 0 and
must be preserved by all work.

## Phase 0 — contained criticals (now)

- **F1 Immediate output invalidation.** A confirmed interruption (and every
  cancellation that clears output) must emit `response.audio.clear` and flush the
  output queue BEFORE awaiting TTS worker teardown. `reap_task`'s 5s timeout may
  never sit between a confirmed barge-in and the clear reaching the client.
- **F7 Uniform cancellation semantics.** Cancelling a response flushes/clears
  queued output identically regardless of turn state (THINKING vs SPEAKING vs
  PAUSED). No path may leave queued RTC audio behind after a cancel.
- **F4 Lifecycle event delivery is observable.** Broadcast failures are logged
  with event type + session; callers check the result; lifecycle-critical events
  (`response.cancelled`, `response.audio.clear`, `interruption.*`, `error`) get
  one retry, and persistent failure logs at error level with the session marked.
  No silent `suppress(Exception) -> False -> ignored`.

## Phase 1 — single response owner (after Phase 0 lands)

- **F2** One serialized owner for response/generation state. All mutations
  (control-plane start/delta/commit/cancel, TTS task completion, state-machine
  actions) are applied on the session event loop; `ConversationResponseLifecycle`
  becomes the single source of truth; the orchestrator holds correlation ids only.
- **F3** State transitions commit only with their side effects: an action failure
  either rolls the transition back or drives a terminal recovery path (never
  log-and-continue divergence between reported state and actual pipeline state).

## Phase 2 — wire contract (design + implement; publish decided separately)

### F5 Typed errors — contract

Wire `error` event gains two fields (additive, backward compatible):

```json
{ "type": "error", "message": "...", "code": "<stable-slug>", "recoverable": true }
```

proto: `ConversationError { string message = 1; string code = 2; bool recoverable = 3; }`

Error codes (initial set; codes are stable API, messages are not):
- `response_rejected_turn_state` (recoverable) — response.start in a state that
  cannot accept it; retry after the next turn/state event.
- `response_rejected_user_speech` (recoverable) — start during active user
  speech; retry on `interruption.false_positive` or `turn.state_changed`.
- `response_stale_generation` (recoverable) — delta/commit for a generation that
  is no longer active; stop pumping THIS generation, session remains healthy.
- `response_already_active` (recoverable) — start while another generation runs.
- `command_invalid` (recoverable) — malformed payload for one command (also
  covers audio-ingest failures; session stays healthy).
- `response_failed` (recoverable) — TTS synthesis/adapter failure for one
  response; the session survives, start a new response.
- `session_failed` (fatal) — unrecoverable session error; client should close.
- Delta/commit with no started generation maps to `response_stale_generation`
  (that is what an SDK sees after a rejected or finished start).
- Missing/empty `code` (old servers) => clients must treat as recoverable unless
  the transport itself closed. Docs guidance in conversation-events.md changes
  from "move client to error state" to: only `recoverable: false` (or transport
  close) ends the call UI; recoverable errors are per-command failures.

### F6 Generation correlation — contract

- `generation_id` added to gRPC command messages (Start/Append/Commit/Cancel)
  mirroring the canonical commands (conversation_commands.py already carries it).
- Response lifecycle events (`response.created|committed|done|cancelled`,
  `response.audio.clear`, `interruption.*`) carry `generation_id` alongside
  `response_id` when known.
- **Start acknowledgement:** `response.created` is the positive ack and now
  echoes the caller's `generation_id`; a rejected start emits the typed error
  above with the same `generation_id`. SDK guidance: after sending
  `response.start`, gate delta-pumping on `response.created` (or abort on the
  correlated recoverable error) instead of fire-and-forget.

### Rollout

vox implements both (server accepts commands with or without generation_id;
events always carry what they know). Then the four RTC server SDKs add: typed
error surface (code/recoverable), start-ack awaiting, and generation threading.
Publish decided separately at the end.

## Phase 3 — adversarial lifecycle suite (grows with each phase)

Scenarios required across BOTH PondSocket and gRPC transports: stubborn TTS
cancellation (slow adapter vs immediate clear), failed event delivery, VAD vs
first-audio orderings, recoverable-error handling, full generation correlation,
interruption confirm/reject under echo.
