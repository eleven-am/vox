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

- **F5 Typed errors.** Canonical error event gains `code` (stable string) and
  `recoverable` (bool); `ConversationError` in vox.proto gains the same. Stale
  delta/start rejections are recoverable; session-fatal errors are not. Docs
  updated (conversation-events.md error guidance) so clients stop treating every
  error frame as fatal.
- **F6 Generation correlation.** `generation_id` on gRPC command messages and on
  response lifecycle events; `response.start` gets a correlated acknowledgement
  (accepted/rejected + generation_id) so SDKs need not fire-and-forget deltas.
- SDK updates for all four RTC server SDKs after the vox side lands.

## Phase 3 — adversarial lifecycle suite (grows with each phase)

Scenarios required across BOTH PondSocket and gRPC transports: stubborn TTS
cancellation (slow adapter vs immediate clear), failed event delivery, VAD vs
first-audio orderings, recoverable-error handling, full generation correlation,
interruption confirm/reject under echo.
