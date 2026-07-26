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

## Lifecycle hardening evidence — 2026-07-26

The stabilization audit against `5268e0f5b5e32dd0f44cdb8ac41f16829c7bf1a3`
closed the remaining ownership gaps:

- RTC negotiations bind every peer callback, local-description task,
  candidate, and completion marker to a negotiation generation. Replacement is
  transactional and established sessions no longer inherit the bootstrap
  attachment deadline.
- Answer publication uses an attempt-scoped barrier. A detached flush from an
  earlier attempt cannot consume or clear candidates for a replacement
  attempt, and future-generation remote candidates remain bound to the pending
  peer until that generation commits.
- RTC peer ownership is explicit across active, pending, and retired
  attachments. Closure failure never removes the last owner, failed cleanup
  closes the session, teardown retries retained resources, and another restart
  is rejected while its predecessor is still retiring.
- The shared RTC output track serializes reads across peer handoff. PCM is
  consumed only after pacing and epoch validation, so sender cancellation
  cannot drop a reserved frame and `audio.clear` invalidates audio already
  waiting for playout.
- Interruption classifier results revalidate candidate identity after awaits.
  Partial self-echo rejection remains provisional until final evidence, and
  response starts serialize on the session runner.
- Response text and RTC audio queues have close/epoch semantics that wake
  blocked producers and prevent pre-clear audio from appearing after
  `response.audio.clear`.
- TTS, VAD, speech-context, scheduler maintenance, and RTC teardown retain
  physical task ownership through cancellation and share a bounded shutdown
  deadline. A model load that physically finishes after that deadline unloads
  its adapter instead of publishing into the stopped scheduler, and new
  acquire/preload calls are rejected after shutdown starts.
- Adapter/runtime installation stages outside active directories and publishes
  with an atomic swap. Blob deduplication remains leased through manifest
  publication. Failure or cancellation aborts the lease and immediately
  collects only its unpublished candidate blobs.
- Pull publication has a durable transaction journal written before each
  directory swap. Startup resolves journals before constructing the registry:
  preparing pulls reverse their ordered swaps and restore the previous
  manifest, committed pulls finish cleanup, and superseded journals cannot
  replace a newer successful pull. Repeated swaps of one target are reversed
  in order. Commit and rollback attempt every runtime, adapter, journal, and
  blob owner even when an earlier owner fails.
- Pull and delete operations retain writer ownership through repeated
  cancellation. A writer acquired after cancellation is closed before the
  operation exits, and adapter/runtime mutations that finish staging after
  cancellation are rolled back before the writer lease is released.
- The canonical manifest is the pull commit point. If its atomic replacement
  succeeds but the following directory sync reports failure, the transaction
  rolls runtime and adapter state forward, emits a durability warning, and
  remains a successful publication rather than restoring mixed state.
- Manifest temporary files live outside the canonical manifest tree, cannot
  appear as models or retain blobs, and are pruned before model readers are
  constructed.
- Pull journals durably retain the complete candidate manifest before
  publication. Committed recovery re-publishes and syncs that manifest before
  removing the journal, while startup garbage collection preserves manifest
  layer roots independently of model metadata construction and fails closed
  on unreadable or symlinked manifest storage.
- An atomic swap that has already published remains successful if NFS prevents
  immediate backup cleanup. The retained backup is left for scoped startup
  cleanup rather than turning a committed publication into a false failure.
- Pre-offer RTC candidates are capped at 256. Pending browser application
  events are capped at 128 events and 262144 bytes. Speech-context admission is
  capped at two analyses and 256 MiB of source audio.
- Startup stale-directory cleanup targets `VOX_TEMP_ROOT`, default
  `/tmp/vox`; ambient operating-system `TMPDIR` contents are never treated as
  Vox-owned.

The deterministic local soak in `tests/test_stabilization_soak.py` uses
`RtcRuntime`, `ConversationOrchestrator`, and `RtcAudioOutputTrack` for the RTC
path. Each RTC cycle proves a completed response waits for playout drain, an
interrupted response clears the real output track, an ICE restart rejects the
old peer callback, and teardown releases runtime/media ownership. It reported:

| Soak | Cycles | RSS start | RSS end | Python bytes start/end | FDs | Child processes | Vox-owned threads | Async tasks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RTC connect/respond/interrupt/restart/disconnect | 100 | 110854144 | 104153088 | 62588 / 89732 | 12 / 12 | 0 / 0 | 0 / 0 | 1 / 1 |
| Worker-backed model load/request/cancel/trim/unload | 50 | 117817344 | 117620736 | 597893 / 606659 | 12 / 12 | 0 / 0 | 0 / 0 | 1 / 1 |

The local Mac has no CUDA runtime, so accelerator allocation is reported as
unavailable rather than inferred from simulated counters. Adapter allocation
accounting, worker death, and GPU-cache ownership remain covered by focused
tests; hardware VRAM measurement requires a later authorized image deployment
and is not represented as local proof.

Endpointing evidence and the selected confidence-shaped continuation policy are
recorded in `docs/endpointing-benchmark.md`.

The final audit correction pass also pinned:

- TTS task ownership through cancellation-resistant physical teardown.
- Peer activation before synchronous `setRemoteDescription` track/data-channel
  events and transactional restoration after failed negotiation.
- Offer-generation correlation on PondSocket and gRPC signaling errors.
- Attempt-scoped local-candidate flushing and generation-aware buffering for
  candidates received before their offer.
- Cancellation-safe answer publication and restart commit, including
  simultaneous sender deactivation and RTP-owner cancellation.
- Identity-safe peer retirement, bounded restart admission while closure is in
  flight, and retained cleanup ownership across offer, local-description,
  commit, rollback, and registry-teardown failures.
- Failed RTC teardown remains registry-owned under bounded-backoff supervision
  until physical cleanup succeeds; a later attached close restarts the same
  retained record rather than losing it after session removal.
- RTC media accepts attempt-scoped callbacks only from the committed
  negotiation or the pending answer barrier. Discarded attempts cannot become
  valid again through bounded-history eviction.
- Scheduler-retired adapters remain explicitly owned after worker death until
  their final logical and physical work completes, then unload and close their
  execution lane. Shutdown drains teardown tasks created by the final release.
- Qwen fallback synthesis sends private text and reference transcripts through
  stdin JSON rather than process arguments.
- Direct stale-file cleanup under the explicit Vox scratch root without
  following symlinks.
- Bounded final waits after worker `SIGKILL`; cancellation-resistant application,
  conversation-runtime, and session shutdown tasks retain a strong owner until
  physical completion.
- Test-owned NER loader threads are joined before fixture teardown, preventing
  background runtime installation from leaking into subsequent tests.
- Query credential redaction recognizes percent-encoded credential names
  without rewriting unrelated query keys.
- Endpointing unit tests use generated PCM fixtures and pass with an empty
  home directory. Authoritative recordings remain an explicit hash-verified
  evidence gate rather than an undeclared dependency of the default suite.

The final local verification run reported `2675 passed, 3 skipped`. The
race-sensitive RTC, interruption, response-ownership, shutdown, and pull
transaction subset passed five consecutive runs of 418 tests. Ruff and all
124 changed Python files pass formatting. Targeted Pyright across the changed
RTC, worker, pull transaction, store, model operation, logging, benchmark, and
application startup boundaries reports zero errors and zero warnings. The
source distribution and wheel build successfully.
