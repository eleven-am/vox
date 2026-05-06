"""End-to-end backchannel rejection: user says "mhmm" while bot is speaking.

Exercises the full path:
  session.ingest_audio(pcm)
    → rolling audio ring buffer fills
    → VAD fires SpeechStarted
    → state goes PAUSED + confirm timer starts
    → timer fires → session._evaluate_interrupt_candidate
    → classifier inspects tail RMS → decides backchannel vs real
    → pushes TIMER_ELAPSED (real) OR synthetic SPEECH_STOPPED (backchannel)

Unlike the classifier unit tests, these drive the session directly without
running real Silero VAD — we push synthetic state-machine events and seed the
session's audio ring + _vad_started_at manually to control what the classifier
sees at timer-fire time.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager

import numpy as np
import pytest

from vox.conversation import TurnEvent, TurnEventType, TurnPolicy, TurnState
from vox.conversation.session import ConversationConfig, ConversationSession
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo


class LongTTS(TTSAdapter):
    """Emits ~1s of audio so there's a real TTS stream in flight during barge-in."""

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="long-tts", type=ModelType.TTS,
            architectures=("x",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True
    def list_voices(self): return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text, **_):
        for _ in range(50):
            yield SynthesizeChunk(
                audio=np.full(480, 0.01, dtype=np.float32).tobytes(),
                sample_rate=24_000, is_final=False,
            )
            await asyncio.sleep(0.02)
        yield SynthesizeChunk(audio=b"", sample_rate=24_000, is_final=True)


class Scheduler:
    def __init__(self, adapter): self._a = adapter
    @asynccontextmanager
    async def acquire(self, _): yield self._a


class Collector:
    def __init__(self): self.events = []
    async def __call__(self, e): self.events.append(e)
    def by_type(self, t): return [e for e in self.events if e.get("type") == t]


def _build():
    tts = LongTTS()
    coll = Collector()
    cfg = ConversationConfig(
        stt_model="x:1", tts_model="y:1", voice="default",
        language="en",
        policy=TurnPolicy(
            min_interrupt_duration_ms=80,
            max_endpointing_delay_ms=500,
            stable_speaking_min_ms=50,
        ),
    )
    session = ConversationSession(scheduler=Scheduler(tts), config=cfg, on_event=coll)
    return session, coll, tts


def _voice_signal(duration_s: float, amp: float = 0.1, sr: int = 16_000) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


class TestBackchannelRejection:
    @pytest.mark.asyncio
    async def test_mhmm_does_not_interrupt(self):
        """User says something like 'mhmm' — short voice burst followed by
        silence. The audio ring shows a quiet tail when the confirm timer
        fires → classifier says backchannel → TTS resumes.
        """
        session, coll, tts = _build()
        await session.start()


        await session.submit_response_text("hello world this is the assistant speaking")
        await asyncio.sleep(0.15)
        assert session.state == TurnState.SPEAKING


        voice = _voice_signal(0.25)
        silence = np.zeros(int(0.30 * 16_000), dtype=np.float32)
        session._audio_ring = np.concatenate([voice, silence])
        session._vad_started_at = time.monotonic() - 0.55

        await session._event_queue.put(TurnEvent(
            type=TurnEventType.SPEECH_STARTED,
            payload={"confirm_window_ms": 80},
        ))


        await asyncio.sleep(0.25)


        assert session.state != TurnState.INTERRUPTED
        assert not coll.by_type("response.cancelled")

        await session.close()

    @pytest.mark.asyncio
    async def test_sustained_interrupt_is_confirmed(self):
        """Sustained voice through the confirm window → real interrupt,
        TTS cancelled.
        """
        session, coll, tts = _build()
        await session.start()

        await session.submit_response_text("hello world this is the assistant speaking")
        await asyncio.sleep(0.15)
        assert session.state == TurnState.SPEAKING


        voice = _voice_signal(0.60, amp=0.15)
        session._audio_ring = voice
        session._vad_started_at = time.monotonic() - 0.60
        await session._event_queue.put(TurnEvent(
            type=TurnEventType.SPEECH_STARTED,
            payload={"confirm_window_ms": 80},
        ))


        await asyncio.sleep(0.25)

        assert session.state == TurnState.INTERRUPTED
        assert coll.by_type("response.cancelled")

        await session.close()


class TestMhmmAtRealisticWindow:
    """The scenario the user asked about: 'mhmm' during TTS, default-ish
    confirm window. 250 ms of voice then silence. With confirm_window=300 ms,
    the tail we inspect (last ~100 ms) is ~50 ms voice + 50 ms silence —
    enough for RMS to sink below the backchannel threshold.
    """

    @pytest.mark.asyncio
    async def test_mhmm_with_300ms_window_does_not_interrupt(self):
        tts = LongTTS()
        coll = Collector()
        cfg = ConversationConfig(
            stt_model="x:1", tts_model="y:1", voice="default",
            policy=TurnPolicy(
                min_interrupt_duration_ms=300,
                max_endpointing_delay_ms=500,
                stable_speaking_min_ms=50,
            ),
        )
        session = ConversationSession(scheduler=Scheduler(tts), config=cfg, on_event=coll)
        await session.start()

        await session.submit_response_text("hello this is the bot")
        await asyncio.sleep(0.15)
        assert session.state == TurnState.SPEAKING



        voice = _voice_signal(0.25)
        silence = np.zeros(int(0.15 * 16_000), dtype=np.float32)
        session._audio_ring = np.concatenate([voice, silence])


        session._vad_started_at = time.monotonic() - 0.40
        await session._event_queue.put(TurnEvent(
            type=TurnEventType.SPEECH_STARTED,
            payload={"confirm_window_ms": 300},
        ))

        await asyncio.sleep(0.40)

        assert session.state != TurnState.INTERRUPTED
        assert not coll.by_type("response.cancelled"),\
            "'mhmm' should NOT cancel TTS"

        await session.close()

    @pytest.mark.asyncio
    async def test_real_stop_with_300ms_window_interrupts(self):
        """'Stop!' — sustained voice right up to the confirm-timer fire →
        interrupt confirmed.
        """
        tts = LongTTS()
        coll = Collector()
        cfg = ConversationConfig(
            stt_model="x:1", tts_model="y:1", voice="default",
            policy=TurnPolicy(
                min_interrupt_duration_ms=300,
                max_endpointing_delay_ms=500,
                stable_speaking_min_ms=50,
            ),
        )
        session = ConversationSession(scheduler=Scheduler(tts), config=cfg, on_event=coll)
        await session.start()

        await session.submit_response_text("hello this is the bot")
        await asyncio.sleep(0.15)
        assert session.state == TurnState.SPEAKING


        voice = _voice_signal(0.40, amp=0.15)
        session._audio_ring = voice
        session._vad_started_at = time.monotonic() - 0.40
        await session._event_queue.put(TurnEvent(
            type=TurnEventType.SPEECH_STARTED,
            payload={"confirm_window_ms": 300},
        ))

        await asyncio.sleep(0.40)

        assert session.state == TurnState.INTERRUPTED
        assert coll.by_type("response.cancelled")

        await session.close()


class TestAudioRingBuffer:
    @pytest.mark.asyncio
    async def test_ring_buffer_caps_at_max_samples(self):
        """Long-running sessions don't grow the audio ring unbounded."""
        session, _, _ = _build()
        await session.start()


        five_seconds_pcm = (np.zeros(16_000 * 5, dtype=np.int16)).tobytes()
        await session.ingest_audio(five_seconds_pcm, sample_rate=16_000)


        assert session._audio_ring.size <= session._audio_ring_max_samples

        await session.close()


class TestClassifierFailureFallsBackToInterrupt:
    @pytest.mark.asyncio
    async def test_exception_in_classifier_defaults_to_real_interrupt(self):
        """If the classifier raises, the session should default to confirming
        the interrupt (fail-safe: worst case we interrupt when we shouldn't,
        better than leaving an impossible PAUSED state stuck).
        """

        class BrokenClassifier:
            def confirm_window_ms(self, base_ms, eou):
                return base_ms

            def wants_short_circuit(self):
                return False

            def should_short_circuit(self, partial_transcript):
                return False

            async def is_real_interrupt(self, audio, partial_transcript, eou, duration_ms, sample_rate):
                raise RuntimeError("classifier exploded")

        tts = LongTTS()
        coll = Collector()
        cfg = ConversationConfig(
            stt_model="x:1", tts_model="y:1", voice="default",
            policy=TurnPolicy(min_interrupt_duration_ms=80, stable_speaking_min_ms=50),
            interrupt_classifier=BrokenClassifier(),
        )
        session = ConversationSession(scheduler=Scheduler(tts), config=cfg, on_event=coll)
        await session.start()

        await session.submit_response_text("reply")
        await asyncio.sleep(0.15)
        assert session.state == TurnState.SPEAKING

        session._vad_started_at = time.monotonic()
        await session._event_queue.put(TurnEvent(
            type=TurnEventType.SPEECH_STARTED,
            payload={"confirm_window_ms": 80},
        ))
        await asyncio.sleep(0.25)


        assert session.state == TurnState.INTERRUPTED

        await session.close()
