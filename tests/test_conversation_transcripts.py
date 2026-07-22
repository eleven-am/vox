from __future__ import annotations

import logging

import numpy as np

from vox.conversation.transcripts import (
    TRANSCRIPT_CONTINUATION_COMMIT_MS,
    WIRE_TRANSCRIPT_DONE,
    WIRE_TURN_EOU_PREDICTED,
    EndpointCommitDelayPolicy,
    EndpointPauseHistory,
    PendingTranscriptFinalizer,
    coalesce_transcript_payload,
    final_transcript_decision,
    is_transcript_revision,
    should_wait_for_pending_final_transcript,
    transcript_done_payload,
)
from vox.conversation.types import TimerKey, TurnEvent, TurnEventType, TurnPolicy
from vox.speech_context.types import SpeechContext
from vox.streaming.types import StreamTranscript


def test_transcript_done_payload_preserves_optional_fields_only_when_present():
    payload = transcript_done_payload(
        StreamTranscript(
            text="hello",
            start_ms=10,
            end_ms=400,
            eou_probability=0.7,
            topics=["greeting"],
            entities=[{"type": "person", "text": "Roy"}],
            words=[{"word": "hello", "start_ms": 10, "end_ms": 400}],
        ),
        language="en",
    )

    assert payload == {
        "type": WIRE_TRANSCRIPT_DONE,
        "transcript": "hello",
        "language": "en",
        "start_ms": 10,
        "end_ms": 400,
        "eou_probability": 0.7,
        "topics": ["greeting"],
        "entities": [{"type": "person", "text": "Roy"}],
        "words": [{"word": "hello", "start_ms": 10, "end_ms": 400}],
    }

    minimal = transcript_done_payload(StreamTranscript(text="hello"), language="en")
    assert "eou_probability" not in minimal
    assert "topics" not in minimal
    assert "entities" not in minimal
    assert "words" not in minimal


def test_transcript_revision_detection_treats_larger_overlap_as_replacement():
    assert is_transcript_revision("Can you tell me?", "What can you tell me?")
    assert is_transcript_revision("hello there", "hello there now")
    assert not is_transcript_revision("first part", "second part")


def test_coalesce_transcript_payload_keeps_best_revision_instead_of_appending():
    previous = {"type": WIRE_TRANSCRIPT_DONE, "transcript": "Can you tell me?", "start_ms": 0, "end_ms": 1200}
    current = {
        "type": WIRE_TRANSCRIPT_DONE,
        "transcript": "What can you tell me?",
        "start_ms": 0,
        "end_ms": 1800,
    }

    assert coalesce_transcript_payload(previous, current) == current
    assert coalesce_transcript_payload(current, previous) == current


def test_coalesce_transcript_payload_appends_continuations_and_metadata():
    previous = {
        "type": WIRE_TRANSCRIPT_DONE,
        "transcript": "first part",
        "language": "en",
        "start_ms": 100,
        "end_ms": 1000,
        "eou_probability": 0.6,
        "topics": ["first"],
        "entities": [{"type": "topic", "text": "first"}],
        "words": [{"word": "first", "start_ms": 100, "end_ms": 300}],
    }
    current = {
        "type": WIRE_TRANSCRIPT_DONE,
        "transcript": "second part",
        "language": "en",
        "start_ms": 1300,
        "end_ms": 2200,
        "topics": ["first", "second"],
        "entities": [{"type": "topic", "text": "second"}],
        "words": [{"word": "second", "start_ms": 1300, "end_ms": 1600}],
    }

    merged = coalesce_transcript_payload(previous, current)

    assert merged["transcript"] == "first part second part"
    assert merged["start_ms"] == 100
    assert merged["end_ms"] == 2200
    assert merged["eou_probability"] == 0.6
    assert merged["topics"] == ["first", "second"]
    assert merged["entities"] == [
        {"type": "topic", "text": "first"},
        {"type": "topic", "text": "second"},
    ]
    assert merged["words"] == [
        {"word": "first", "start_ms": 100, "end_ms": 300},
        {"word": "second", "start_ms": 1300, "end_ms": 1600},
    ]


def test_pending_transcript_finalizer_remembers_pops_clears_and_logs(caplog):
    finalizer = PendingTranscriptFinalizer(language="en")
    finalizer.remember(StreamTranscript(text="first", start_ms=0, end_ms=100))
    finalizer.remember(StreamTranscript(text="second", start_ms=150, end_ms=300))

    assert finalizer.pending_text() == "first second"
    payload = finalizer.pop()
    assert payload is not None
    assert payload["transcript"] == "first second"
    assert finalizer.pop() is None

    finalizer.remember(StreamTranscript(text="discard me"))
    finalizer.clear()
    assert finalizer.pending_text("fallback") == "fallback"

    caplog.set_level(logging.INFO, logger="vox.conversation.session")
    finalizer.log({"transcript": "logged", "start_ms": 1, "end_ms": 2})
    assert any(
        record.name == "vox.conversation.session"
        and "conversation final transcript emitted" in record.message
        and "logged" in record.message
        for record in caplog.records
    )


def test_pending_transcript_finalizer_reanalyzes_continuations_as_one_timeline():
    finalizer = PendingTranscriptFinalizer(language="en")
    context = SpeechContext(status="failed", unavailable=("prosody", "audio_events"))
    first_audio = np.full(1_600, 0.1, dtype=np.float32)
    second_audio = np.full(3_200, 0.2, dtype=np.float32)

    finalizer.remember(
        StreamTranscript(
            text="first phrase",
            start_ms=100,
            end_ms=200,
            speech_context=context,
            audio=first_audio,
        )
    )
    finalizer.remember(
        StreamTranscript(
            text="second phrase",
            start_ms=600,
            end_ms=800,
            speech_context=context,
            audio=second_audio,
        )
    )

    payload, chunks = finalizer.pop_with_audio()

    assert payload is not None
    assert payload["transcript"] == "first phrase second phrase"
    assert "speech_context" not in payload
    assert [(chunk.offset_ms, chunk.duration_ms) for chunk in chunks] == [
        (100, 100),
        (600, 200),
    ]


def test_pending_transcript_finalizer_replaces_revision_audio_and_context():
    finalizer = PendingTranscriptFinalizer(language="en")
    first = np.full(1_600, 0.1, dtype=np.float32)
    revision = np.full(3_200, 0.2, dtype=np.float32)

    finalizer.remember(StreamTranscript(text="hello", audio=first))
    finalizer.remember(
        StreamTranscript(
            text="hello there",
            end_ms=200,
            audio=revision,
            speech_context=SpeechContext(
                status="failed",
                unavailable=("prosody", "audio_events"),
            ),
        )
    )

    payload, chunks = finalizer.pop_with_audio()

    assert payload is not None
    assert payload["transcript"] == "hello there"
    assert payload["speech_context"]["status"] == "failed"
    assert len(chunks) == 1
    assert chunks[0].duration_ms == 200


def test_endpoint_commit_delay_uses_recent_pause_history_when_dynamic():
    policy = EndpointCommitDelayPolicy.from_turn_policy(
        TurnPolicy(
            max_endpointing_delay_ms=3000,
            min_endpointing_delay_ms=400,
            dynamic_endpointing=True,
        )
    )

    assert policy.commit_delay_ms(recent_pause_ms=[800, 1000, 1200]) == 1250
    assert policy.commit_delay_ms(recent_pause_ms=[100]) == 1200


def test_endpoint_pause_history_records_clamped_recent_pauses():
    history = EndpointPauseHistory(max_items=3)

    history.record_since(None, now=10.0)
    assert history.values() == ()

    history.record_since(10.0, now=11.0)
    history.record_since(10.0, now=12.0)
    history.record_since(10.0, now=13.0)
    history.record_since(15.0, now=14.0)

    assert history.values() == (2000, 3000, 0)


def test_endpoint_pause_history_keeps_at_least_one_item():
    history = EndpointPauseHistory(max_items=0)

    history.record_since(10.0, now=11.0)
    history.record_since(10.0, now=12.0)

    assert history.values() == (2000,)


def test_endpoint_commit_delay_ignores_pause_history_when_dynamic_disabled():
    policy = EndpointCommitDelayPolicy.from_turn_policy(
        TurnPolicy(
            max_endpointing_delay_ms=3000,
            min_endpointing_delay_ms=400,
            dynamic_endpointing=False,
        )
    )

    assert policy.commit_delay_ms(recent_pause_ms=[1600, 1800]) == 1200


def test_endpoint_commit_delay_shrinks_with_eou_confidence():
    policy = EndpointCommitDelayPolicy.from_turn_policy(
        TurnPolicy(
            max_endpointing_delay_ms=3000,
            min_endpointing_delay_ms=400,
            dynamic_endpointing=False,
        )
    )

    base_ms = policy.commit_delay_ms()

    assert base_ms == 1200
    assert policy.commit_delay_ms(eou_probability=1.0, eou_threshold=0.5) == 400
    mid_ms = policy.commit_delay_ms(eou_probability=0.75, eou_threshold=0.5)
    assert 400 < mid_ms < base_ms


def test_endpoint_commit_delay_extends_low_eou_but_keeps_missing_eou_fallback():
    policy = EndpointCommitDelayPolicy.from_turn_policy(
        TurnPolicy(
            max_endpointing_delay_ms=3000,
            min_endpointing_delay_ms=400,
            dynamic_endpointing=False,
        )
    )

    low_confidence_ms = policy.commit_delay_ms(eou_probability=0.3, eou_threshold=0.5)

    assert 1200 < low_confidence_ms < 3000
    assert policy.commit_delay_ms(eou_probability=0.0, eou_threshold=0.5) == 3000
    assert policy.commit_delay_ms(eou_probability=None, eou_threshold=0.5) == 1200


def test_endpoint_commit_delay_is_monotonic_across_incomplete_confidence():
    policy = EndpointCommitDelayPolicy.from_turn_policy(
        TurnPolicy(
            max_endpointing_delay_ms=3000,
            min_endpointing_delay_ms=400,
            dynamic_endpointing=False,
        )
    )

    barely_incomplete = policy.commit_delay_ms(eou_probability=0.49, eou_threshold=0.5)
    clearly_incomplete = policy.commit_delay_ms(eou_probability=0.25, eou_threshold=0.5)
    certainly_incomplete = policy.commit_delay_ms(eou_probability=0.0, eou_threshold=0.5)

    assert 1200 < barely_incomplete < clearly_incomplete < certainly_incomplete
    assert certainly_incomplete == 3000


def test_final_transcript_decision_emits_commit_eou_event_without_endpoint_timer():
    policy = EndpointCommitDelayPolicy.from_turn_policy(
        TurnPolicy(
            max_endpointing_delay_ms=3000,
            min_endpointing_delay_ms=400,
            dynamic_endpointing=False,
        )
    )

    decision = final_transcript_decision(
        StreamTranscript(text="done", eou_probability=0.9, start_ms=100, end_ms=600),
        endpoint_timer_active=False,
        commit_delay_policy=policy,
        recent_pause_ms=[],
        eou_threshold=0.5,
        turn_detector="livekit",
    )

    assert decision.commit_delay_ms == 0
    assert not decision.defer_commit
    assert decision.eou_complete
    assert decision.eou_event == {
        "type": WIRE_TURN_EOU_PREDICTED,
        "probability": 0.9,
        "threshold": 0.5,
        "decision": "complete",
        "action": "commit",
        "delay_ms": 0,
        "turn_detector": "livekit",
        "start_ms": 100,
        "end_ms": 600,
    }


def test_final_transcript_decision_defers_while_endpoint_timer_is_active():
    policy = EndpointCommitDelayPolicy.from_turn_policy(
        TurnPolicy(
            max_endpointing_delay_ms=3000,
            min_endpointing_delay_ms=400,
            dynamic_endpointing=False,
        )
    )

    decision = final_transcript_decision(
        StreamTranscript(text="maybe done", eou_probability=0.3, start_ms=0, end_ms=1200),
        endpoint_timer_active=True,
        commit_delay_policy=policy,
        recent_pause_ms=[],
        eou_threshold=0.5,
        turn_detector="livekit",
    )

    assert 1200 < decision.commit_delay_ms < 3000
    assert decision.defer_commit
    assert not decision.eou_complete
    assert decision.eou_event is not None
    assert decision.eou_event["decision"] == "incomplete"
    assert decision.eou_event["action"] == "wait"
    assert decision.eou_event["delay_ms"] == decision.commit_delay_ms


def test_near_zero_eou_uses_profile_maximum_for_every_detector_backend():
    policy = EndpointCommitDelayPolicy.from_turn_policy(
        TurnPolicy(
            max_endpointing_delay_ms=3000,
            min_endpointing_delay_ms=400,
            dynamic_endpointing=False,
        )
    )

    for turn_detector in ("livekit", "smart-turn:v3.2"):
        decision = final_transcript_decision(
            StreamTranscript(
                text="I am calling out because I wanna",
                eou_probability=0.000028,
                start_ms=0,
                end_ms=1800,
            ),
            endpoint_timer_active=True,
            commit_delay_policy=policy,
            recent_pause_ms=[],
            eou_threshold=0.5,
            turn_detector=turn_detector,
        )

        assert decision.defer_commit
        assert not decision.eou_complete
        assert decision.commit_delay_ms == 3000
        assert decision.eou_event is not None
        assert decision.eou_event["turn_detector"] == turn_detector
        assert decision.eou_event["action"] == "wait"


def test_final_transcript_decision_defers_missing_eou_without_emitting_eou_event():
    policy = EndpointCommitDelayPolicy.from_turn_policy(
        TurnPolicy(
            max_endpointing_delay_ms=3000,
            min_endpointing_delay_ms=400,
            dynamic_endpointing=False,
        )
    )

    decision = final_transcript_decision(
        StreamTranscript(text="no eou", start_ms=0, end_ms=800),
        endpoint_timer_active=True,
        commit_delay_policy=policy,
        recent_pause_ms=[],
        eou_threshold=0.5,
        turn_detector="livekit",
    )

    assert decision.commit_delay_ms == 1200
    assert decision.defer_commit
    assert not decision.eou_complete
    assert decision.eou_event is None


def test_pending_final_transcript_wait_ignores_unrelated_events():
    assert not should_wait_for_pending_final_transcript(
        TurnEvent(type=TurnEventType.SPEECH_STOPPED),
        awaiting_final_transcript=True,
        awaiting_started_at=10.0,
        max_endpointing_delay_ms=3000,
        now=10.5,
    )
    assert not should_wait_for_pending_final_transcript(
        TurnEvent(type=TurnEventType.TIMER_ELAPSED, payload={"key": TimerKey.CONFIRM_INTERRUPT.value}),
        awaiting_final_transcript=True,
        awaiting_started_at=10.0,
        max_endpointing_delay_ms=3000,
        now=10.5,
    )


def test_pending_final_transcript_wait_requires_pending_transcript():
    assert not should_wait_for_pending_final_transcript(
        TurnEvent(type=TurnEventType.TIMER_ELAPSED, payload={"key": TimerKey.ENDPOINTING.value}),
        awaiting_final_transcript=False,
        awaiting_started_at=10.0,
        max_endpointing_delay_ms=3000,
        now=10.5,
    )


def test_pending_final_transcript_wait_rechecks_until_max_delay_expires():
    event = TurnEvent(type=TurnEventType.TIMER_ELAPSED, payload={"key": TimerKey.ENDPOINTING.value})

    assert should_wait_for_pending_final_transcript(
        event,
        awaiting_final_transcript=True,
        awaiting_started_at=0.0,
        max_endpointing_delay_ms=3000,
        now=10.0,
    )
    assert should_wait_for_pending_final_transcript(
        event,
        awaiting_final_transcript=True,
        awaiting_started_at=10.0,
        max_endpointing_delay_ms=3000,
        now=12.9,
    )
    assert not should_wait_for_pending_final_transcript(
        event,
        awaiting_final_transcript=True,
        awaiting_started_at=10.0,
        max_endpointing_delay_ms=3000,
        now=13.0,
    )


def test_pending_final_transcript_wait_uses_continuation_floor():
    event = TurnEvent(type=TurnEventType.TIMER_ELAPSED, payload={"key": TimerKey.ENDPOINTING.value})
    started_at = 10.0

    assert should_wait_for_pending_final_transcript(
        event,
        awaiting_final_transcript=True,
        awaiting_started_at=started_at,
        max_endpointing_delay_ms=100,
        now=started_at + ((TRANSCRIPT_CONTINUATION_COMMIT_MS - 1) / 1000),
    )
    assert not should_wait_for_pending_final_transcript(
        event,
        awaiting_final_transcript=True,
        awaiting_started_at=started_at,
        max_endpointing_delay_ms=100,
        now=started_at + ((TRANSCRIPT_CONTINUATION_COMMIT_MS + 1) / 1000),
    )
