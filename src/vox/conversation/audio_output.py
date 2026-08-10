from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class PendingAudio:
    audio: bytes
    sample_rate: int
    sequence: int


class ResponseAudioOutput:
    def __init__(self, *, pace_to_playout: bool = False) -> None:
        self._pace_to_playout = pace_to_playout
        self._pending: list[PendingAudio] = []
        self._sequence = 0
        self._playout_end_at = 0.0
        self._paused = False
        self._pause_owner: int | None = None

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def pause_owner(self) -> int | None:
        return self._pause_owner

    def reset_for_response(self) -> None:
        if self._paused:
            raise RuntimeError(f"cannot reset response audio with suspension owner={self._pause_owner}")
        if self._pending:
            raise RuntimeError(f"cannot reset response audio with pending audio={len(self._pending)}")
        self._sequence = 0
        self._playout_end_at = 0.0

    def reset_playout(self) -> None:
        self._playout_end_at = 0.0

    def clear_pending(self) -> None:
        self._pending = []

    def clear_all(self) -> None:
        self.clear_pending()
        self.reset_playout()

    def pause(self, owner: int | None = None) -> bool:
        if self._paused and self._pause_owner == owner:
            return False
        self._paused = True
        self._pause_owner = owner
        self.reset_playout()
        return True

    def finish_resume(self, owner: int | None = None) -> bool:
        if not self._paused:
            return False
        if owner is not None and self._pause_owner != owner:
            return False
        self._paused = False
        self._pause_owner = None
        return True

    def flush(self) -> None:
        self.release_all()

    def release_all(self) -> None:
        self.clear_pending()
        self._paused = False
        self._pause_owner = None
        self.reset_playout()

    def next_sequence(self) -> int:
        self._sequence += 1
        return self._sequence

    def hold(self, audio: bytes, sample_rate: int, sequence: int) -> None:
        self._pending.append(PendingAudio(audio=audio, sample_rate=sample_rate, sequence=sequence))

    def hold_if_paused(self, audio: bytes, sample_rate: int, sequence: int) -> bool:
        if not self._paused:
            return False
        self.hold(audio, sample_rate, sequence)
        return True

    def pop_pending_batch(self) -> list[PendingAudio]:
        pending = self._pending
        self._pending = []
        return pending

    def pending_resume_batches(self):
        while self.pending_count:
            yield self.pop_pending_batch()

    def mark_playout(self, pcm16_audio: bytes, sample_rate: int) -> None:
        if not self._pace_to_playout:
            return
        if sample_rate <= 0 or not pcm16_audio:
            return
        duration_s = len(pcm16_audio) / float(sample_rate * 2)
        now = time.monotonic()
        self._playout_end_at = max(now, self._playout_end_at) + duration_s

    def playout_delay_s(self) -> float:
        if not self._pace_to_playout:
            return 0.0
        return max(0.0, self._playout_end_at - time.monotonic())
