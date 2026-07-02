"""RTC turn timing diagnostics derived from wire events."""

from __future__ import annotations

import time
from typing import Any


class RtcTurnTimeline:
    """Track per-turn RTC timing without making routes own diagnostic state."""

    def __init__(self, *, session_id: str) -> None:
        self._session_id = session_id
        self._turn_index = 0
        self._speech_started_at: float | None = None
        self._speech_stopped_at: float | None = None
        self._transcript_at: float | None = None
        self._response_created_at: float | None = None
        self._response_committed_at: float | None = None

    def observe(self, wire: dict, *, audio_stats: dict | None = None) -> dict | None:
        event_type = wire.get("type")
        if not isinstance(event_type, str):
            return None
        now = time.perf_counter()

        if event_type == "input_audio_buffer.speech_started":
            self._turn_index += 1
            self._speech_started_at = now
            self._speech_stopped_at = None
            self._transcript_at = None
            self._response_created_at = None
            self._response_committed_at = None
            return self._event(event_type, now, wire, audio_stats)

        if event_type == "input_audio_buffer.speech_stopped":
            self._speech_stopped_at = now
            return self._event(event_type, now, wire, audio_stats)

        if event_type in {
            "turn.eou.predicted",
            "conversation.item.input_audio_transcription.completed",
            "response.created",
            "response.committed",
            "response.done",
            "response.cancelled",
            "response.audio.clear",
            "interruption.detected",
            "interruption.false_positive",
        }:
            if event_type == "conversation.item.input_audio_transcription.completed":
                self._transcript_at = now
            elif event_type == "response.created":
                self._response_created_at = now
            elif event_type == "response.committed":
                self._response_committed_at = now
            return self._event(event_type, now, wire, audio_stats)

        return None

    def _event(self, source_event: str, now: float, wire: dict, audio_stats: dict | None) -> dict:
        data = {
            "source_event": source_event,
            "turn_index": self._turn_index,
            "response_id": _wire_data(wire).get("response_id"),
            "ms_since_speech_started": _elapsed_ms(now, self._speech_started_at),
            "ms_since_speech_stopped": _elapsed_ms(now, self._speech_stopped_at),
            "ms_since_transcript": _elapsed_ms(now, self._transcript_at),
            "ms_since_response_created": _elapsed_ms(now, self._response_created_at),
            "ms_since_response_committed": _elapsed_ms(now, self._response_committed_at),
        }
        if audio_stats is not None:
            data["rtc_audio"] = audio_stats
        return {
            "type": "rtc.turn_timing",
            "session_id": self._session_id,
            "data": {key: value for key, value in data.items() if value is not None},
        }


def rtc_audio_stats(record: Any) -> dict | None:
    track = getattr(record, "audio_output_track", None)
    if track is None or not hasattr(track, "stats"):
        return None
    return track.stats()


def _elapsed_ms(now: float, started_at: float | None) -> int | None:
    if started_at is None:
        return None
    return max(0, int((now - started_at) * 1000))


def _wire_data(wire: dict) -> dict:
    data = wire.get("data")
    return data if isinstance(data, dict) else wire
