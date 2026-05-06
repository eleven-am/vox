"""Integration tests for v1 extras: anti-flutter cooldown + EOU-modulated confirm window."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import numpy as np
import pytest

from vox.conversation import TurnEvent, TurnEventType, TurnPolicy, TurnState
from vox.conversation.interrupt import HeuristicInterruptClassifier
from vox.conversation.session import (
    WIRE_AUDIO_DELTA,
    WIRE_RESPONSE_CANCELLED,
    ConversationConfig,
    ConversationSession,
)
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo


class ScriptedTTS(TTSAdapter):
    def __init__(self, chunks: int = 30, inter_chunk_delay: float = 0.02) -> None:
        self._chunks = chunks
        self._delay = inter_chunk_delay
        self.cancelled = False

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="scripted", type=ModelType.TTS,
            architectures=("x",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )
    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True
    def list_voices(self): return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text, **_):
        try:
            for i in range(self._chunks):
                yield SynthesizeChunk(
                    audio=np.full(1024, 0.01, dtype=np.float32).tobytes(),
                    sample_rate=24_000, is_final=False,
                )
                await asyncio.sleep(self._delay)
            yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class Scheduler:
    def __init__(self, adapter): self._a = adapter
    @asynccontextmanager
    async def acquire(self, _): yield self._a


class Collector:
    def __init__(self): self.events = []
    async def __call__(self, e): self.events.append(e)
    def types(self): return [e["type"] for e in self.events]


class _AcceptAllClassifier:
    """Test classifier that mimics the pre-backchannel-filter behaviour:
    window duration modulated by EOU as usual, but every timer fire confirms.
    These tests exercise state-machine / timing logic, not content filtering.
    """

    def __init__(self, high=0.7, low=0.3, hi_mult=0.5, lo_mult=1.5, floor=100):
        self.high = high
        self.low = low
        self.hi_mult = hi_mult
        self.lo_mult = lo_mult
        self.floor = floor

    def confirm_window_ms(self, base_ms, last_eou_probability):
        if last_eou_probability is None:
            return base_ms
        if last_eou_probability >= self.high:
            return max(int(base_ms * self.hi_mult), self.floor)
        if last_eou_probability < self.low:
            return max(int(base_ms * self.lo_mult), self.floor)
        return base_ms

    def wants_short_circuit(self):
        return False

    def should_short_circuit(self, partial_transcript):
        return False

    async def is_real_interrupt(self, audio, partial_transcript, eou, duration_ms, sample_rate):
        return True


def _build(**policy_kwargs):
    tts = ScriptedTTS()
    coll = Collector()
    policy = TurnPolicy(**{
        "min_interrupt_duration_ms": 100,
        "max_endpointing_delay_ms": 500,
        "stable_speaking_min_ms": 150,
        **policy_kwargs,
    })
    cfg = ConversationConfig(
        stt_model="x:1", tts_model="y:1", voice="default",
        language="en", policy=policy,
        interrupt_classifier=_AcceptAllClassifier(),
    )
    session = ConversationSession(scheduler=Scheduler(tts), config=cfg, on_event=coll)
    return session, coll, tts


class TestAntiFlutterCooldown:
    @pytest.mark.asyncio
    async def test_second_speech_during_cooldown_is_suppressed(self):
        session, coll, tts = _build(min_interrupt_duration_ms=80, stable_speaking_min_ms=200)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await asyncio.sleep(0.02)
        await session.submit_response_text("reply")
        await asyncio.sleep(0.05)
        assert session.state == TurnState.SPEAKING


        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.PAUSED


        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STOPPED))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.SPEAKING



        from vox.streaming.types import SpeechStarted
        await session._forward_stream_event(SpeechStarted(timestamp_ms=50))
        await asyncio.sleep(0.01)

        assert session.state == TurnState.SPEAKING
        assert tts.cancelled is False

        await session.close()

    @pytest.mark.asyncio
    async def test_speech_after_cooldown_elapses_is_honored(self):
        session, coll, tts = _build(min_interrupt_duration_ms=80, stable_speaking_min_ms=50)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await asyncio.sleep(0.02)
        await session.submit_response_text("reply")
        await asyncio.sleep(0.05)


        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STOPPED))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.SPEAKING


        await asyncio.sleep(0.1)


        from vox.streaming.types import SpeechStarted
        await session._forward_stream_event(SpeechStarted(timestamp_ms=100))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.PAUSED

        await session.close()

    @pytest.mark.asyncio
    async def test_cooldown_still_emits_wire_event_for_ui(self):
        """Even during cooldown, the speech_started wire event should go out."""
        session, coll, tts = _build(min_interrupt_duration_ms=50, stable_speaking_min_ms=300)
        await session.start()

        await session._event_queue.put(TurnEvent(type=TurnEventType.USER_TRANSCRIPT_FINAL))
        await asyncio.sleep(0.02)
        await session.submit_response_text("reply")
        await asyncio.sleep(0.03)


        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STARTED))
        await asyncio.sleep(0.01)
        await session._event_queue.put(TurnEvent(type=TurnEventType.SPEECH_STOPPED))
        await asyncio.sleep(0.01)

        wire_events_before = len([e for e in coll.events if e.get("type") == "input_audio_buffer.speech_started"])


        from vox.streaming.types import SpeechStarted
        await session._forward_stream_event(SpeechStarted(timestamp_ms=50))
        await asyncio.sleep(0.01)

        wire_events_after = len([e for e in coll.events if e.get("type") == "input_audio_buffer.speech_started"])
        assert wire_events_after == wire_events_before + 1,\
            "speech_started wire event should still be emitted for UI feedback"

        await session.close()


class TestEouModulatedConfirmWindow:
    @pytest.mark.asyncio
    async def test_high_eou_shortens_confirm_window(self):
        """When last turn had high EOU, barge-in should confirm faster."""
        session, coll, tts = _build(min_interrupt_duration_ms=300, stable_speaking_min_ms=50)
        await session.start()


        from vox.streaming.types import StreamTranscript
        await session._forward_stream_event(StreamTranscript(
            text="hello",
            eou_probability=0.9,
            start_ms=0,
            end_ms=500,
        ))
        await asyncio.sleep(0.01)


        await session.submit_response_text("reply")
        await asyncio.sleep(0.03)
        assert session.state == TurnState.SPEAKING


        from vox.streaming.types import SpeechStarted
        await session._forward_stream_event(SpeechStarted(timestamp_ms=100))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.PAUSED



        await asyncio.sleep(0.2)
        assert session.state == TurnState.INTERRUPTED

        await session.close()

    @pytest.mark.asyncio
    async def test_low_eou_lengthens_confirm_window(self):
        """When last turn had low EOU, skepticism wins — longer confirm window."""
        session, coll, tts = _build(min_interrupt_duration_ms=100, stable_speaking_min_ms=50)
        await session.start()

        from vox.streaming.types import StreamTranscript
        await session._forward_stream_event(StreamTranscript(
            text="um",
            eou_probability=0.1,
            start_ms=0, end_ms=200,
        ))
        await asyncio.sleep(0.01)

        await session.submit_response_text("reply")
        await asyncio.sleep(0.03)

        from vox.streaming.types import SpeechStarted
        await session._forward_stream_event(SpeechStarted(timestamp_ms=100))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.PAUSED


        await asyncio.sleep(0.1)
        assert session.state == TurnState.PAUSED


        await asyncio.sleep(0.1)
        assert session.state == TurnState.INTERRUPTED

        await session.close()

    @pytest.mark.asyncio
    async def test_no_prior_turn_falls_back_to_policy_default(self):
        """If no user turn has been committed yet, use policy.min_interrupt_duration_ms."""
        session, coll, tts = _build(min_interrupt_duration_ms=120, stable_speaking_min_ms=50)
        await session.start()


        await session.submit_response_text("reply")
        await asyncio.sleep(0.03)
        assert session.state == TurnState.SPEAKING

        from vox.streaming.types import SpeechStarted
        await session._forward_stream_event(SpeechStarted(timestamp_ms=100))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.PAUSED


        await asyncio.sleep(0.15)
        assert session.state == TurnState.INTERRUPTED

        await session.close()

    @pytest.mark.asyncio
    async def test_custom_classifier_controls_window(self):
        """A user-supplied classifier overrides the default heuristic."""

        class StrictClassifier:
            def confirm_window_ms(self, base_ms, last_eou_probability):
                return 500
            def wants_short_circuit(self):
                return False
            def should_short_circuit(self, partial_transcript):
                return False
            async def is_real_interrupt(self, audio, partial_transcript, eou, dur, sample_rate):
                return True

        tts = ScriptedTTS()
        coll = Collector()
        cfg = ConversationConfig(
            stt_model="x:1", tts_model="y:1", voice="default", language="en",
            policy=TurnPolicy(min_interrupt_duration_ms=50, stable_speaking_min_ms=50),
            interrupt_classifier=StrictClassifier(),
        )
        session = ConversationSession(scheduler=Scheduler(tts), config=cfg, on_event=coll)
        await session.start()

        await session.submit_response_text("reply")
        await asyncio.sleep(0.03)

        from vox.streaming.types import SpeechStarted
        await session._forward_stream_event(SpeechStarted(timestamp_ms=100))
        await asyncio.sleep(0.01)
        assert session.state == TurnState.PAUSED


        await asyncio.sleep(0.1)
        assert session.state == TurnState.PAUSED

        await session.close()
