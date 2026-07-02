"""Tests for AEC warmup discard (L1) and per-utterance allow_interruptions (L2)."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import numpy as np
import pytest

from vox.conversation import TurnPolicy, TurnState
from vox.conversation.session import ConversationConfig, ConversationSession
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.streaming.types import SpeechStarted


class LongTTS(TTSAdapter):
    def __init__(self) -> None:
        self.cancelled = False

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="long",
            type=ModelType.TTS,
            architectures=("x",),
            default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *_a, **_k) -> None: ...
    def unload(self) -> None: ...

    @property
    def is_loaded(self) -> bool:
        return True

    def list_voices(self):
        return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text, **_):
        try:
            for _i in range(120):
                yield SynthesizeChunk(
                    audio=np.full(1024, 0.01, dtype=np.float32).tobytes(),
                    sample_rate=24_000,
                    is_final=False,
                )
                await asyncio.sleep(0.02)
            yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class Scheduler:
    def __init__(self, adapter) -> None:
        self._a = adapter

    @asynccontextmanager
    async def acquire(self, _):
        yield self._a


class Collector:
    def __init__(self) -> None:
        self.events: list[dict] = []

    async def __call__(self, e):
        self.events.append(e)

    def by_type(self, t):
        return [e for e in self.events if e.get("type") == t]


def _build(**policy_kwargs):
    tts = LongTTS()
    coll = Collector()
    policy = TurnPolicy(
        **{
            "min_interrupt_duration_ms": 80,
            "max_endpointing_delay_ms": 500,
            "stable_speaking_min_ms": 50,
            "speaking_interrupt_min_duration_ms": 80,
            "aec_warmup_ms": 250,
            **policy_kwargs,
        }
    )
    cfg = ConversationConfig(
        stt_model="x:1",
        tts_model="y:1",
        voice="default",
        language="en",
        policy=policy,
    )
    session = ConversationSession(scheduler=Scheduler(tts), config=cfg, on_event=coll)
    return session, coll, tts


def _install_pipeline_spy(session) -> list[int]:
    """Wrap ``session._pipeline.process_audio`` so we can verify it isn't
    invoked while AEC warmup is active. Returns a list of incoming audio sizes,
    one entry per call.
    """
    call_log: list[int] = []
    original = session._pipeline.process_audio

    async def spy(audio):
        call_log.append(audio.size)
        async for ev in original(audio):
            yield ev

    session._pipeline.process_audio = spy
    return call_log


class TestAECWarmupDiscard:
    @pytest.mark.asyncio
    async def test_pipeline_is_skipped_during_warmup_after_tts_starts(self):
        session, _, _ = _build(aec_warmup_ms=200)
        await session.start()
        call_log = _install_pipeline_spy(session)

        await session.submit_response_text("hello")
        await asyncio.sleep(0.1)
        assert session.state == TurnState.SPEAKING

        baseline_calls = len(call_log)
        audio = np.full(int(0.04 * 16_000), 0.1, dtype=np.float32).astype(np.float32).tobytes()
        await session.ingest_audio(audio, sample_rate=16_000)
        await asyncio.sleep(0.0)

        assert len(call_log) == baseline_calls

        await session.close()

    @pytest.mark.asyncio
    async def test_pipeline_resumes_after_warmup_elapses(self):
        session, _, _ = _build(aec_warmup_ms=40)
        await session.start()
        call_log = _install_pipeline_spy(session)

        await session.submit_response_text("hello")
        await asyncio.sleep(0.1)
        assert session.state == TurnState.SPEAKING

        await asyncio.sleep(0.08)

        baseline_calls = len(call_log)
        audio = np.full(int(0.04 * 16_000), 0.1, dtype=np.float32).astype(np.float32).tobytes()
        await session.ingest_audio(audio, sample_rate=16_000)
        await asyncio.sleep(0.0)

        assert len(call_log) == baseline_calls + 1

        await session.close()

    @pytest.mark.asyncio
    async def test_warmup_does_not_rearm_on_pause_resume(self):
        session, _, _ = _build(aec_warmup_ms=200, speaking_interrupt_min_duration_ms=80)
        await session.start()

        await session.submit_response_text("hello")
        await asyncio.sleep(0.1)
        assert session.state == TurnState.SPEAKING

        await asyncio.sleep(0.22)
        assert not session._aec_warmup_active()

        from vox.conversation.types import TurnEvent, TurnEventType

        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.PAUSED
        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STOPPED))
        await asyncio.sleep(0.01)

        assert session.state == TurnState.SPEAKING
        assert not session._aec_warmup_active()

        await session.close()

    @pytest.mark.asyncio
    async def test_audio_ring_still_fed_during_warmup(self):
        session, _, _ = _build(aec_warmup_ms=400)
        await session.start()

        await session.submit_response_text("hello")
        await asyncio.sleep(0.1)
        assert session.state == TurnState.SPEAKING
        assert session._aec_warmup_active()

        ring_size_before = session._audio_history.mic_size
        audio = np.full(int(0.04 * 16_000), 0.1, dtype=np.float32).astype(np.float32).tobytes()
        await session.ingest_audio(audio, sample_rate=16_000)

        assert session._audio_history.mic_size > ring_size_before

        await session.close()

    @pytest.mark.asyncio
    async def test_zero_warmup_ms_disables_feature(self):
        session, _, _ = _build(aec_warmup_ms=0)
        await session.start()
        call_log = _install_pipeline_spy(session)

        await session.submit_response_text("hello")
        await asyncio.sleep(0.1)
        assert session.state == TurnState.SPEAKING
        assert not session._aec_warmup_active()

        baseline_calls = len(call_log)
        audio = np.full(int(0.04 * 16_000), 0.1, dtype=np.float32).astype(np.float32).tobytes()
        await session.ingest_audio(audio, sample_rate=16_000)
        await asyncio.sleep(0.0)

        assert len(call_log) == baseline_calls + 1

        await session.close()


class TestWarmupArmingAlignedToPlayout:
    @pytest.mark.asyncio
    async def test_warmup_does_not_arm_before_first_audio_emit(self):
        session, _, _ = _build(aec_warmup_ms=200)
        await session.start()

        from vox.conversation.types import TurnEvent, TurnEventType

        await session._event_queue.put(TurnEvent(type=TurnEventType.TTS_AUDIO_STARTED))
        await asyncio.sleep(0.01)

        assert session.state == TurnState.SPEAKING
        assert not session._aec_warmup_active()
        assert session._aec_warmup_until == 0.0

        await session.close()

    @pytest.mark.asyncio
    async def test_warmup_arms_on_first_audio_emit(self):
        session, _, _ = _build(aec_warmup_ms=200)
        await session.start()

        await session.submit_response_text("hello")
        await asyncio.sleep(0.1)

        assert session.state == TurnState.SPEAKING
        assert session._aec_warmup_active()

        await session.close()


class TestPostSpeechTranscriptFilter:
    @pytest.mark.asyncio
    async def test_final_transcript_during_speaking_is_dropped(self):
        from vox.streaming.types import StreamTranscript

        session, coll, _ = _build(aec_warmup_ms=0, backchannel_end_cooldown_ms=1500)
        await session.start()

        await session.submit_response_text("hello")
        await asyncio.sleep(0.1)
        assert session.state == TurnState.SPEAKING
        assert session._agent_speech_active

        await session._forward_stream_event(StreamTranscript(
            text="echo of myself",
            start_ms=10,
            end_ms=500,
        ))
        await asyncio.sleep(0.0)

        assert not coll.by_type("conversation.item.input_audio_transcription.completed")
        echoes = [
            e for e in coll.by_type("interruption.false_positive")
            if e.get("reason") == "self_echo_transcript_window"
        ]
        assert echoes

        await session.close()

    @pytest.mark.asyncio
    async def test_final_transcript_within_cooldown_after_speech_is_dropped(self):
        from vox.streaming.types import StreamTranscript

        session, coll, _ = _build(aec_warmup_ms=0, backchannel_end_cooldown_ms=500)
        await session.start()

        session._mark_agent_speech_started()
        session._mark_agent_speech_ended()
        assert not session._agent_speech_active

        session._sm._state = TurnState.SPEAKING
        await session._forward_stream_event(StreamTranscript(
            text="leaked transcript",
            start_ms=10,
            end_ms=500,
        ))
        await asyncio.sleep(0.0)

        echoes = [
            e for e in coll.by_type("interruption.false_positive")
            if e.get("reason") == "self_echo_transcript_window"
        ]
        assert echoes
        assert not coll.by_type("conversation.item.input_audio_transcription.completed")

        await session.close()

    @pytest.mark.asyncio
    async def test_final_transcript_outside_cooldown_is_kept(self):
        from vox.streaming.types import StreamTranscript

        session, coll, _ = _build(aec_warmup_ms=0, backchannel_end_cooldown_ms=20)
        await session.start()

        session._mark_agent_speech_started()
        session._mark_agent_speech_ended()
        await asyncio.sleep(0.05)
        session._sm._state = TurnState.LISTENING

        await session._forward_stream_event(StreamTranscript(
            text="real user utterance",
            start_ms=10,
            end_ms=500,
        ))
        await asyncio.sleep(0.0)

        assert coll.by_type("conversation.item.input_audio_transcription.completed")
        echoes = [
            e for e in coll.by_type("interruption.false_positive")
            if e.get("reason") == "self_echo_transcript_window"
        ]
        assert not echoes

        await session.close()

    @pytest.mark.asyncio
    async def test_transcript_in_interrupted_state_is_kept(self):
        from vox.streaming.types import StreamTranscript

        session, coll, _ = _build(aec_warmup_ms=0, backchannel_end_cooldown_ms=2000)
        await session.start()

        session._mark_agent_speech_started()
        session._sm._state = TurnState.INTERRUPTED

        await session._forward_stream_event(StreamTranscript(
            text="confirmed barge-in content",
            start_ms=10,
            end_ms=500,
        ))
        await asyncio.sleep(0.0)

        assert coll.by_type("conversation.item.input_audio_transcription.completed")

        await session.close()


class TestUninterruptibleResponse:
    @pytest.mark.asyncio
    async def test_speech_started_during_uninterruptible_stays_speaking(self):
        session, coll, _ = _build(aec_warmup_ms=0)
        await session.start()

        await session.submit_response_text("uninterruptible message", allow_interruptions=False)
        await asyncio.sleep(0.1)
        assert session.state == TurnState.SPEAKING

        await session._forward_stream_event(SpeechStarted(timestamp_ms=1000))
        await asyncio.sleep(0.05)

        assert session.state == TurnState.SPEAKING
        assert not coll.by_type("response.audio.clear")

        await session.close()

    @pytest.mark.asyncio
    async def test_uninterruptible_drops_audio_at_ingest(self):
        session, _, _ = _build(aec_warmup_ms=0)
        await session.start()
        call_log = _install_pipeline_spy(session)

        await session.submit_response_text("uninterruptible message", allow_interruptions=False)
        await asyncio.sleep(0.1)
        assert session.state == TurnState.SPEAKING

        baseline = len(call_log)
        ring_before = session._audio_history.mic_size
        audio = np.full(int(0.04 * 16_000), 0.1, dtype=np.float32).astype(np.float32).tobytes()
        await session.ingest_audio(audio, sample_rate=16_000)

        assert len(call_log) == baseline
        assert session._audio_history.mic_size == ring_before

        await session.close()

    @pytest.mark.asyncio
    async def test_default_response_is_interruptible(self):
        from vox.streaming.types import StreamTranscript

        session, coll, _ = _build(aec_warmup_ms=0)
        await session.start()

        await session.submit_response_text("normal message")
        await asyncio.sleep(0.1)
        assert session.state == TurnState.SPEAKING

        await session._forward_stream_event(SpeechStarted(timestamp_ms=1000))
        await asyncio.sleep(0.05)

        assert session.state == TurnState.SPEAKING
        assert not coll.by_type("response.audio.clear")

        await session._forward_stream_event(StreamTranscript(
            text="stop please",
            is_partial=True,
        ))
        await asyncio.sleep(0.05)

        assert session.state == TurnState.INTERRUPTED
        assert coll.by_type("response.audio.clear")

        await session.close()


class TestWireParseAllowInterruptions:
    def test_parses_top_level_field(self):
        from vox.operations.conversation import parse_allow_interruptions

        assert parse_allow_interruptions({"allow_interruptions": False}) is False
        assert parse_allow_interruptions({"allow_interruptions": True}) is True

    def test_parses_nested_response_field(self):
        from vox.operations.conversation import parse_allow_interruptions

        msg = {"response": {"allow_interruptions": False}}
        assert parse_allow_interruptions(msg) is False

    def test_defaults_to_true_when_omitted(self):
        from vox.operations.conversation import parse_allow_interruptions

        assert parse_allow_interruptions({}) is True
        assert parse_allow_interruptions({"response": {}}) is True
