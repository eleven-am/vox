# Vox simplification program — working spec

Goal: v0.2.94's responsiveness, v0.2.109's robustness, in a smaller codebase
than either. Every mechanism justifies itself by naming the concrete failure it
prevents and the test pinning it; anything that cannot is deleted or collapsed.
Sources: six subsystem audits (2026-07-19), the interruption benchmark
(`test_interruption_benchmark.py`), and design review. No release-cadence
pressure; batches are sequential and each soaks before the next starts.

## Standing invariants (pinned by tests, enforced forever)

- I1 **One terminal state.** Every started response reaches exactly one of
  done | cancelled | failed. Never silence, never two.
- I2 **No forgetting.** No orchestration state may forget a response while that
  response remains live (orchestrator/session agreement is a property test,
  not a convention).
- I3 **KEEP-ledger law.** No new gate, window, flag, queue, or mirror lands
  without a test naming the failure it prevents.
- I4 **No state mirrors.** A fact has one owner; consumers ask, they do not
  cache.

## Batch sequence

R0 (incident) -> R1 (deletions) -> R2a (ownership) -> R2b (interruption) ->
R2c (detector API) -> R3 (NeMo subprocess) -> R4 (transport cleanup).
R2a and R2b share `conversation/session.py` and are strictly sequential.

---

## R0 — incident closure (contains the current production bug)

1. The whitespace-delta fix (already local, verified): blank text is content
   for an active expected generation, an outcome only when there is no
   expected generation.
2. Append/commit return a structured reason instead of bool, **born in the
   R2a shape** so R2a does not rewrite it:

   ```python
   class AppendResult(StrEnum):
       ACCEPTED = "accepted"
       SESSION_CLOSED = "session_closed"
       NO_ACTIVE_RESPONSE = "no_active_response"
       RESPONSE_MISMATCH = "response_mismatch"
       RESPONSE_COMMITTED = "response_committed"
       STREAM_ENDED = "stream_ended"
   ```

   The orchestrator maps each reason to its own wire code/message; only
   SESSION_CLOSED and STREAM_ENDED may clear orchestrator bookkeeping.
   RESPONSE_MISMATCH must not clear state for a response that may be live (the
   split-brain source).
3. Orphan guard: a rejected delta can never leave a response stuck in
   `thinking` — on any non-ACCEPTED delta for the active response, the session
   either keeps the response fully live (mismatch/blank cases) or terminalizes
   it loudly (closed/ended cases). Regression test drives the exact 18:14
   production sequence on both transports.

## R1 — pure deletions (approved; zero behavior change by construction)

- torch Silero VAD backend + its type-only test (~55)
- `rtc_timeline.py` + the per-event observe calls: emitted-but-unread (~130)
- dead pondsocket RTC broadcast helpers, `close_rtc_session`,
  `create_rtc_orchestrator_with` seam (~70)
- 5 write-only `InterruptionCandidate` fields (~15)
- duplicated `pcm16_to_float32` + one-shot resample wrapper (~20)
- requires-schema fields `min_compute_capability`, `min_cuda_version`,
  `min_ram_gb` — VERIFIED 2026-07-20: all 38 live registry entries scanned,
  zero users. Approved.
- spark/default alias profiles (all 30 in-repo pairs identical; the registry
  carries variant aliases, not profile splits). Approved.
- env-override snapshot layer (`VOX_RUNTIME_OVERRIDE`, `VOX_HAS_*`) —
  VERIFIED: referenced only in docs/model-resolution-design.md, absent from
  all deploy surfaces; owner confirms unused. Delete layer + update that doc.
- scheduler VRAM-budget subsystem — **DEFERRED 2026-07-20 (implementation
  STOP)**: the subsystem has live consumers the audit missed — registered
  endpoints `POST /v1/system/enforce-memory-budget` and `POST /v1/system/trim`,
  the `GET /v1/system/memory` policy payload, CLI flags `--idle-trim-ttl` and
  `--memory-over-budget` beyond the two approved, plus doc/test pins in
  vox-dia. Deleting registered HTTP surface exceeds the recorded approval.
  OWNER DECISION NEEDED: (a) delete the whole budget vertical including its
  endpoint, flags, and memory-payload policy block, keeping `/v1/system/trim`
  as an independently useful op; (b) delete both verticals including trim; or
  (c) keep as-is. Executed in a later batch once decided.

Gate: full suite + lint, nothing else changes.

---

## R2a — ownership: the stream is the response

### The model

The lifecycle owns exactly two values; everything else derives.

```python
@dataclass(frozen=True)
class TerminalRecord:
    response_id: str
    generation_id: str | None
    reason: Literal["done", "cancelled", "failed"]

class ConversationResponseLifecycle:
    stream: ResponseStream | None      # the live response, or None
    terminal: TerminalRecord | None    # last teardown, until the next start
```

- `ResponseStream` gains `generation_id: str | None`, stamped at construction
  from the client's start command, immutable. Response ids remain the existing
  monotonic counter.
- Aliveness IS `stream is not None and not stream.closed`. Committed IS
  `stream.committed`. `ResponsePhase` is deleted.
- **A rejected start is not a terminal response.** It has no response_id and
  produces a `StartRejection(generation_id, reason)` (the existing
  `ResponseStartRejection`, extended with generation_id). `TerminalRecord.reason`
  deliberately has no "rejected" member.

### The only door: `terminalize`

```python
def terminalize(self, stream: ResponseStream, reason: TerminalReason) -> TerminalRecord | None
```

- The **sole** closer of streams in the codebase. All scattered
  `stream.close()` / `clear_active_response` / `remember_cancelled_response` /
  `fail_stream_if_current` call sites route through it; those methods are
  deleted.
- **Identity-checked**: no-ops (returns None) unless `stream is self.stream`.
  A delayed callback for resp_1 cannot terminalize resp_2. IDs are never used
  for teardown decisions.
- **Atomic on the runner**: writes `terminal`, closes the stream, clears
  `stream`, returns the frozen record — one call, same runner turn. Teardown
  emissions (`response.cancelled`/`done`, `audio.clear`, error frames) take the
  returned record as their argument. Emit constructors accept a
  `ResponseStream | TerminalRecord`, never raw ids — a mismatched
  response_id/generation_id pair is unrepresentable.
- Emission idempotence lives here too: `terminalize` returning None means
  "already terminal, emit nothing" — replacing `_cancel_emitted_response_id`.
- `terminal` is cleared only by the next **successful** start. Because records
  are frozen and emissions happen in the same runner turn as the write, a
  later clear cannot corrupt an in-flight emission.

### The orchestrator becomes a translator

Deleted outright: `_control_generation_id`, `_client_generation_id`,
`_pending_start_generation_id`, `_correlate_generation`,
`_validate_response_generation`, `active_generation_id`,
`last_cancelled_response_id`. The orchestrator asks the session
(`response_active`, the stream's ids) and maps `AppendResult` to wire codes.
Events carry `generation_id` because the stream/record they are built from
carries it — the correlation join disappears rather than moving.

### Gates

Full suite; race subset x5; the I1/I2 property tests; adversarial review of
the diff (mandatory — this touches the Phase 1 hot zone); transport parity
suite unchanged.

---

## R2b — interruption: evidence-gated ablation, one window store, honest timing

Sequential after R2a has soaked.

### 1. Acoustic gate — ablation decides, not opinion

Protocol, per clause in {`min_rms`, `min_active_frame_ratio`,
`max_crest_factor`}:

1. Run `benchmark_interruptions` with only that clause removed.
2. Delete the clause only if no category's false-positive rate worsens beyond
   epsilon and the overall bar (>=50% FP reduction vs legacy, <=1pt recall
   loss, <=50ms median latency) holds.
3. Publish the ablation table in the PR; surviving clauses enter the KEEP
   ledger with their category named.

Prior flag: `active_frame_ratio` is expected to survive — `voiced_frame_ratio`
and flatness are computed over **active frames only** (`interrupt.py:122-136`),
so it is the only clause that sees sparsity (periodic-impulse-in-silence, e.g.
phone vibration). If the corpus lacks such a category, add one before ablating.

### 2. Distrust window — one store, named contributions, two predicates

```python
class SpeechGuard:
    # contributions: tts_start_warmup, tts_tail, resume_stability
    def suppresses_interrupt_evidence(self, now) -> bool   # feeds the detector
    def suppresses_transcript_trust(self, now) -> bool     # feeds delta emission / self-echo checks
```

- One timestamp store (max of contributions), **two** consumer-facing
  predicates — not one boolean (a merged boolean would suppress a user's
  legitimate next turn for up to 1.5s after playback ends), and not a
  trust-level taxonomy (heavier than the two real consumers justify).
- Current consumers map: flutter-deferral of SPEECH_STARTED and AEC-warmup
  deferral -> predicate 1; self-echo transcript window -> predicate 2.
- Honest knob count: 8 -> 4 (confirm window, warmup contribution, tail
  contribution, evidence timeout). `retry_after_ms` is deleted (see 3).

### 3. Timeout is terminal

`evaluate_timeout` returns CONFIRM or REJECT — never DEFER. DEFER exists only
in event-driven observes, where it means "wait for the next event"; the
confirm timer is armed once per candidate and its expiry always resolves the
candidate. The defer-retry rearm loop and `retry_after_ms` are deleted. No
candidate can strand by construction: every candidate ends by event or by its
one timer.

### 4. Confirm-window floor = the acoustic minimum

- The 500ms `speaking_interrupt_min_duration_ms` confirm-window floor is
  removed; the window becomes EOU-modulated with a floor of the **structural
  acoustic minimum** (`min_real_interrupt_ms`, 180ms) so the timer can never
  fire before enough audio exists for the gate to evaluate (a sub-180ms fire
  plus terminal-timeout would reject legitimate speech that merely lacks
  evidence).
- `speaking_interrupt_min_duration_ms` survives only inside
  `_strong_transcript` as the acoustic duration gate.
- Expected effect: worst-case time-to-interrupt while speaking drops from
  >=500ms toward ~180-310ms.

### Gates

Full benchmark bar + the per-clause ablation tables; before/after
time-to-interrupt measurements (median + p95) published in the PR; FP rates on
`tts_playback`, `aec_warmup`, `backchannel`, `handling_noise` categories must
not regress; soak on the homelab with R0's typed error codes watching.

## R2c — detector API cleanup (separate, after R2b soaks)

Transition-only decision emission (delete `newly_decided` replay suppression),
fold the three near-parallel `observe_final` support blocks into one
predicate, single self-echo call site. Pure restructure; benchmark must be
bit-identical on decisions.

## R3 — NeMo out of process (amended after design review, 2026-07-20)

Objective unchanged: the subprocess boundary fixes Parakeet's dependency
contamination and makes VRAM release a `kill`. Design amended: Voxtral's
current worker is REQUIREMENTS INPUT, not the implementation to extract — it
merges stderr into the protocol stream (`stderr=subprocess.STDOUT`), skips
arbitrary non-JSON lines, has no startup/request timeouts or request ids, and
can block forever on `readline()` (verified `backends.py:127-145`).

### R3a — `WorkerHost` contract (in core, proven against a synthetic noisy worker)

Design principle: **the worker is cattle — any anomaly means it is dead.**
One failure rule replaces per-anomaly handling: on request timeout, garbage
frame, EOF, or nonzero exit, the host kills the process group, fails the
current request loudly, and marks the adapter unloaded. Because a timed-out
worker is killed, a stale late response is impossible by construction, so the
protocol needs no request ids or correlation. Because a dead worker is just
"not loaded," respawn rides the EXISTING scheduler load path and a
permanently-broken model behaves exactly like a failed in-process load today —
no circuit-breaker subsystem. No replay ever, by the same rule.

The contract, in full:

- Spawn: process group, clean environment (no inherited sibling runtime
  paths), protocol pipe dup'ed to a dedicated fd, child fd1 redirected to
  stderr (native writes cannot corrupt frames), stderr drained continuously
  into parent logs.
- First frame is `ready` (or an error), bounded by a startup timeout —
  `load()` blocks on it. No version handshake: host and worker ship in the
  same wheel.
- One JSON-line request at a time (callers are already serialized), one
  JSON-line response, bounded by a per-request timeout.
- Shutdown ladder on kill: graceful -> terminate -> kill, waiting at each
  stage. stdin-EOF is a backstop only (a child blocked in inference does not
  read stdin).
- Scheduler integration (the one core change): acquire checks adapter health;
  a dead worker reads as unloaded and takes the normal load path, so it can
  never be served from cache.

Target size: ~150-200 lines including the child-side harness. Anything beyond
this contract must name its failure per the profile.

### R3b — Parakeet port

- The parent NEVER activates or imports the NeMo runtime. Install +
  verification run inside a sanitized subprocess (today
  `_install_nemo_runtime` mutates parent `sys.path`, import-checks
  `nemo.collections.asr`, and purges parent `sys.modules` — verified
  `nemo_adapter.py:91-235`; all of that moves into the worker or dies).
- Worker owns: runtime probing, NeMo imports, model load, CUDA-graph fallback,
  transcription, timestamp extraction.
- Audio transport: the EXISTING temp-WAV boundary (`nemo_adapter.py:509`
  already writes one) — JSON carries the file path, the response carries
  transcript metadata. No base64 (5-minute STT chunks = ~19MB raw / ~26MB
  encoded), no binary framing (unjustified).
- Deleted: shadow globs, `_clear_nemo_modules`, `_prime_lightning_imports`,
  parent-side runtime activation (~200 lines of surgery).
- KEEP (unchanged, ledgered): core RLock, `--target` installs, CFFI preload,
  and `_loaded_native_runtime_paths` — other in-process runtimes still use
  `activate_runtime_path`. Chatterbox stays in-process absent a named failure.

### R3c — optional Voxtral migration

Only after Parakeet proves the host. Keeps a working adapter out of the risky
first pass.

### Gates

Protocol-integrity test with a deliberately noisy worker (native fd1 writes
mid-stream); crash-mid-request fails loudly with no replay; circuit-breaker
test; process-group orphan test; per-call latency measured on realistic
partial windows before/after.

## R4 — transport cleanup

Merge fragment files (`rtc_media_events`, `rtc_sessions`, `rtc_cleanup`,
`rtc_tasks`, and `rtc_conversation`+`rtc_client_events` -> `rtc_session_io`);
collapse the gRPC serialize-then-reparse round-trip (typed event -> pb
directly); single shared lifecycle-retry helper. 11 rtc/pond files -> 6, one
event-taxonomy enumeration per direction.

---

## Process rules

- Every implementer (human or agent) reads
  `~/Desktop/talos/CLAUDE/ROY_CODING_PROFILE.md` before touching architecture.
  The invariants above are that profile operationalized; when in doubt, the
  profile decides.
- One batch in flight at a time; soak between batches.
- Structural batches (R2a, R2b, R3) get an independent adversarial review of
  the diff before commit.
- Behavior-adjacent batches (R2b) are gated on the benchmark + published
  measurements, not judgment.
- Sub-agents implement against this spec; verification (suite, lint, race
  repeats, gates) is re-run independently before any commit.
- This document is the source of truth for the program; amendments land here
  first.

## Program status — 2026-07-20

R0 through R4 implemented, adversarially verified, and merged (R0, R1, R2a,
R2b, R2c, R3a, R3b, R3c, R4). Deferred by owner decision: R1 item 9 (VRAM
budget vertical — see the item for the decision needed). Follow-up noted from
review: a dedicated retry test for the gRPC typed-event handler branch.
