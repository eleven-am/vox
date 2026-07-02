from __future__ import annotations

from types import SimpleNamespace

from vox.server import rtc_timeline
from vox.server.rtc_timeline import RtcTurnTimeline, rtc_audio_stats


def test_rtc_turn_timeline_emits_derived_timing_events(monkeypatch):
    times = iter([10.0, 10.125, 10.25, 10.5])
    monkeypatch.setattr(rtc_timeline.time, "perf_counter", lambda: next(times))
    timeline = RtcTurnTimeline(session_id="rtc_test")

    started = timeline.observe(
        {"type": "input_audio_buffer.speech_started", "data": {}},
        audio_stats={"buffered_audio_ms": 0.0},
    )
    assert started == {
        "type": "rtc.turn_timing",
        "session_id": "rtc_test",
        "data": {
            "source_event": "input_audio_buffer.speech_started",
            "turn_index": 1,
            "ms_since_speech_started": 0,
            "rtc_audio": {"buffered_audio_ms": 0.0},
        },
    }

    stopped = timeline.observe({"type": "input_audio_buffer.speech_stopped", "data": {}})
    assert stopped is not None
    assert stopped["data"] == {
        "source_event": "input_audio_buffer.speech_stopped",
        "turn_index": 1,
        "ms_since_speech_started": 125,
        "ms_since_speech_stopped": 0,
    }

    transcript = timeline.observe(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "data": {"transcript": "hello"},
        }
    )
    assert transcript is not None
    assert transcript["data"] == {
        "source_event": "conversation.item.input_audio_transcription.completed",
        "turn_index": 1,
        "ms_since_speech_started": 250,
        "ms_since_speech_stopped": 125,
        "ms_since_transcript": 0,
    }

    created = timeline.observe({"type": "response.created", "data": {"response_id": "resp_1"}})
    assert created is not None
    assert created["data"] == {
        "source_event": "response.created",
        "turn_index": 1,
        "response_id": "resp_1",
        "ms_since_speech_started": 500,
        "ms_since_speech_stopped": 375,
        "ms_since_transcript": 250,
        "ms_since_response_created": 0,
    }


def test_rtc_turn_timeline_resets_per_turn_state(monkeypatch):
    times = iter([1.0, 1.25, 2.0])
    monkeypatch.setattr(rtc_timeline.time, "perf_counter", lambda: next(times))
    timeline = RtcTurnTimeline(session_id="rtc_test")

    timeline.observe({"type": "input_audio_buffer.speech_started", "data": {}})
    timeline.observe({"type": "conversation.item.input_audio_transcription.completed", "data": {}})
    restarted = timeline.observe({"type": "input_audio_buffer.speech_started", "data": {}})

    assert restarted is not None
    assert restarted["data"] == {
        "source_event": "input_audio_buffer.speech_started",
        "turn_index": 2,
        "ms_since_speech_started": 0,
    }


def test_rtc_turn_timeline_ignores_untracked_events():
    timeline = RtcTurnTimeline(session_id="rtc_test")

    assert timeline.observe({"type": "session.created", "data": {}}) is None
    assert timeline.observe({"data": {}}) is None


def test_rtc_audio_stats_reports_optional_track_stats():
    assert rtc_audio_stats(SimpleNamespace(audio_output_track=None)) is None
    assert rtc_audio_stats(SimpleNamespace()) is None
    assert rtc_audio_stats(SimpleNamespace(audio_output_track=object())) is None

    track = SimpleNamespace(stats=lambda: {"buffered_audio_ms": 120.0})

    assert rtc_audio_stats(SimpleNamespace(audio_output_track=track)) == {"buffered_audio_ms": 120.0}
