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

from vox.conversation import TurnAction, TurnActionType, TurnEvent, TurnEventType, TurnPolicy, TurnState
from vox.conversation.session import ConversationConfig, ConversationSession
from vox.core.adapter import TTSAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType, SynthesizeChunk, VoiceInfo
from vox.streaming.types import SpeechStarted, SpeechStopped, StreamTranscript


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


class BriefTTS(TTSAdapter):
    """Finishes quickly enough to complete while the session is PAUSED."""

    def info(self) -> AdapterInfo:
        return AdapterInfo(
            name="brief-tts", type=ModelType.TTS,
            architectures=("x",), default_sample_rate=24_000,
            supported_formats=(ModelFormat.ONNX,),
        )

    def load(self, *a, **k): ...
    def unload(self): ...
    @property
    def is_loaded(self): return True
    def list_voices(self): return [VoiceInfo(id="default", name="Default")]

    async def synthesize(self, text, **_):
        for _ in range(4):
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
            speaking_interrupt_min_duration_ms=80,
        ),
    )
    session = ConversationSession(scheduler=Scheduler(tts), config=cfg, on_event=coll)
    return session, coll, tts


def _voice_signal(duration_s: float, amp: float = 0.1, sr: int = 16_000, freq: float = 220) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


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
        assert coll.by_type("interruption.false_positive")

        await session.close()

    @pytest.mark.asyncio
    async def test_assistant_self_echo_does_not_interrupt(self):
        """Loudspeaker echo can make STT hear the assistant's own words.
        Without true client-side AEC, Vox should still reject transcripts that
        clearly match the active TTS response instead of cancelling itself.
        """
        session, coll, _ = _build()
        await session.start()

        await session.submit_response_text("The appointment is tomorrow at noon.")
        await asyncio.sleep(0.15)
        assert session.state == TurnState.SPEAKING

        session._latest_partial = StreamTranscript(
            text="the appointment is tomorrow",
            is_partial=True,
        )
        session._audio_ring = _voice_signal(0.60, amp=0.15)
        session._vad_started_at = time.monotonic() - 0.60
        await session._event_queue.put(TurnEvent(
            type=TurnEventType.SPEECH_STARTED,
            payload={"confirm_window_ms": 80},
        ))

        await asyncio.sleep(0.25)

        assert session.state != TurnState.INTERRUPTED
        assert not coll.by_type("response.cancelled")
        assert coll.by_type("interruption.false_positive")

        await session.close()

    @pytest.mark.asyncio
    async def test_acoustic_output_echo_is_rejected_before_audio_clear(self):
        """When mic audio strongly matches recent TTS output, suppress the
        candidate before PAUSE_OUTPUT clears the audible response.
        """
        session, coll, _ = _build()
        await session.start()

        await session.submit_response_text("The assistant is speaking.")
        await asyncio.sleep(0.15)
        assert session.state == TurnState.SPEAKING

        echo = _voice_signal(0.50, amp=0.08, freq=330)
        session._output_audio_ring = echo.copy()
        session._audio_ring = echo.copy()

        await session._forward_stream_event(SpeechStarted(timestamp_ms=1000))
        await asyncio.sleep(0.14)

        assert session.state == TurnState.SPEAKING
        assert coll.by_type("interruption.false_positive")
        assert not coll.by_type("response.audio.clear")
        assert not coll.by_type("response.cancelled")

        await session.close()

    @pytest.mark.asyncio
    async def test_non_echo_speech_still_pauses_quickly_during_tts(self):
        session, coll, _ = _build()
        await session.start()

        await session.submit_response_text("The assistant is speaking.")
        await asyncio.sleep(0.15)
        assert session.state == TurnState.SPEAKING

        session._output_audio_ring = _voice_signal(0.50, amp=0.08, freq=330)
        session._audio_ring = _voice_signal(0.50, amp=0.15, freq=660)

        await session._forward_stream_event(SpeechStarted(timestamp_ms=1000))
        await asyncio.sleep(0.02)

        assert session.state == TurnState.SPEAKING
        assert not coll.by_type("response.audio.clear")

        await asyncio.sleep(0.20)

        assert session.state in {TurnState.PAUSED, TurnState.INTERRUPTED, TurnState.IDLE}
        assert coll.by_type("response.audio.clear")

        await session.close()

    @pytest.mark.asyncio
    async def test_backchannel_candidate_does_not_clear_before_false_positive_gate(self):
        session, coll, _ = _build()
        await session.start()

        await session.submit_response_text("The assistant is speaking.")
        await asyncio.sleep(0.15)
        assert session.state == TurnState.SPEAKING

        voice = _voice_signal(0.10, amp=0.08, freq=330)
        silence = np.zeros(int(0.30 * 16_000), dtype=np.float32)
        session._output_audio_ring = _voice_signal(0.50, amp=0.08, freq=440)
        session._audio_ring = np.concatenate([voice, silence])

        await session._forward_stream_event(SpeechStarted(timestamp_ms=1000))
        await asyncio.sleep(0.20)

        assert session.state == TurnState.SPEAKING
        assert coll.by_type("interruption.false_positive")
        assert not coll.by_type("response.audio.clear")
        assert not coll.by_type("response.cancelled")

        await session.close()

    @pytest.mark.asyncio
    async def test_paused_output_buffers_audio_until_false_positive_resume(self):
        session, coll, _ = _build()
        session._active_response_id = "resp_1"
        session._paused = True

        audio = np.full(480, 0.01, dtype=np.float32).tobytes()
        await session._handle_tts_chunk(audio, 24_000)

        assert session._pending_audio
        assert not coll.by_type("response.audio.delta")

        await session._execute(TurnAction(TurnActionType.RESUME_OUTPUT))

        assert not session._pending_audio
        assert coll.by_type("response.audio.delta")

        await session.close()

    @pytest.mark.asyncio
    async def test_resume_preserves_audio_order_when_new_chunks_arrive_mid_flush(self):
        session, coll, _ = _build()
        session._active_response_id = "resp_1"
        session._paused = True

        chunk_a = np.full(480, 0.01, dtype=np.float32).tobytes()
        chunk_b = np.full(480, 0.02, dtype=np.float32).tobytes()
        chunk_c = np.full(480, 0.03, dtype=np.float32).tobytes()
        chunk_d = np.full(480, 0.04, dtype=np.float32).tobytes()

        await session._handle_tts_chunk(chunk_a, 24_000)
        await session._handle_tts_chunk(chunk_b, 24_000)
        await session._handle_tts_chunk(chunk_c, 24_000)

        original_emit = session._on_event
        late_chunk_inserted = False

        async def emit_then_inject_late_chunk(event):
            nonlocal late_chunk_inserted
            await original_emit(event)
            if (
                not late_chunk_inserted
                and event.get("type") == "response.audio.delta"
                and event.get("sequence") == 1
            ):
                late_chunk_inserted = True
                await session._handle_tts_chunk(chunk_d, 24_000)

        session._on_event = emit_then_inject_late_chunk

        await session._execute(TurnAction(TurnActionType.RESUME_OUTPUT))

        assert late_chunk_inserted
        deltas = coll.by_type("response.audio.delta")
        sequences = [event["sequence"] for event in deltas]
        assert sequences == sorted(sequences)
        assert len(sequences) == 4
        assert session._paused is False
        assert not session._pending_audio

        await session.close()

    @pytest.mark.asyncio
    async def test_flush_output_clears_echo_rings(self):
        session, _, _ = _build()

        session._output_audio_ring = _voice_signal(0.50, amp=0.05, freq=330)
        session._audio_ring = _voice_signal(0.50, amp=0.05, freq=440)

        await session._execute(TurnAction(TurnActionType.FLUSH_OUTPUT))

        assert session._output_audio_ring.size == 0
        assert session._audio_ring.size == 0

        await session.close()

    @pytest.mark.asyncio
    async def test_short_non_keyword_partial_during_tts_does_not_interrupt(self):
        session, coll, _ = _build()
        await session.start()

        await session.submit_response_text("The appointment is tomorrow at noon.")
        await asyncio.sleep(0.15)
        assert session.state == TurnState.SPEAKING

        session._latest_partial = StreamTranscript(text="actually", is_partial=True)
        session._audio_ring = _voice_signal(0.60, amp=0.15)
        session._vad_started_at = time.monotonic() - 0.60
        await session._event_queue.put(TurnEvent(
            type=TurnEventType.SPEECH_STARTED,
            payload={"confirm_window_ms": 80},
        ))

        await asyncio.sleep(0.25)

        assert session.state != TurnState.INTERRUPTED
        assert not coll.by_type("response.cancelled")
        assert coll.by_type("interruption.false_positive")

        await session.close()

    @pytest.mark.asyncio
    async def test_keyword_partial_during_tts_still_interrupts(self):
        session, coll, _ = _build()
        await session.start()

        await session.submit_response_text("The appointment is tomorrow at noon.")
        await asyncio.sleep(0.15)
        assert session.state == TurnState.SPEAKING

        session._latest_partial = StreamTranscript(text="stop", is_partial=True)
        session._audio_ring = _voice_signal(0.60, amp=0.15)
        session._vad_started_at = time.monotonic() - 0.60
        await session._event_queue.put(TurnEvent(
            type=TurnEventType.SPEECH_STARTED,
            payload={"confirm_window_ms": 80},
        ))

        await asyncio.sleep(0.25)

        assert session.state == TurnState.INTERRUPTED
        assert coll.by_type("response.cancelled")

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
        session._latest_partial = StreamTranscript(text="I need", is_partial=True)
        session._vad_started_at = time.monotonic() - 0.60
        await session._event_queue.put(TurnEvent(
            type=TurnEventType.SPEECH_STARTED,
            payload={"confirm_window_ms": 80},
        ))


        await asyncio.sleep(0.25)

        assert session.state == TurnState.INTERRUPTED
        assert coll.by_type("response.cancelled")

        await session.close()

    @pytest.mark.asyncio
    async def test_meaningful_partial_overrides_quiet_tail_during_tts(self):
        session, coll, _ = _build()
        await session.start()

        await session.submit_response_text("The assistant is speaking.")
        await asyncio.sleep(0.15)
        assert session.state == TurnState.SPEAKING

        voice = _voice_signal(0.22, amp=0.12, freq=300)
        silence = np.zeros(int(0.30 * 16_000), dtype=np.float32)
        session._audio_ring = np.concatenate([voice, silence])
        session._latest_partial = StreamTranscript(
            text="Can you hear me?",
            is_partial=True,
            start_ms=0,
            end_ms=920,
            audio_duration_ms=920,
        )
        session._vad_started_at = time.monotonic() - 0.55

        await session._event_queue.put(TurnEvent(
            type=TurnEventType.SPEECH_STARTED,
            payload={"confirm_window_ms": 80},
        ))

        await asyncio.sleep(0.25)

        assert session.state == TurnState.INTERRUPTED
        assert coll.by_type("interruption.detected")
        assert coll.by_type("response.cancelled")

        await session.close()

    @pytest.mark.asyncio
    async def test_late_meaningful_partial_can_retro_confirm_interrupt(self):
        session, coll, _ = _build()
        await session.start()

        await session.submit_response_text("The assistant is speaking.")
        await asyncio.sleep(0.15)
        assert session.state == TurnState.SPEAKING

        voice = _voice_signal(0.10, amp=0.08, freq=330)
        silence = np.zeros(int(0.30 * 16_000), dtype=np.float32)
        session._audio_ring = np.concatenate([voice, silence])

        await session._forward_stream_event(SpeechStarted(timestamp_ms=1000))
        await asyncio.sleep(0.20)

        assert session.state == TurnState.SPEAKING
        assert coll.by_type("interruption.false_positive")

        await session._forward_stream_event(SpeechStopped(timestamp_ms=1320))

        await session._forward_stream_event(StreamTranscript(
            text="I'm talking, I'm talking.",
            is_partial=True,
            start_ms=0,
            end_ms=1260,
            audio_duration_ms=1260,
        ))
        await asyncio.sleep(0.05)

        assert session.state == TurnState.INTERRUPTED
        assert coll.by_type("interruption.detected")
        assert coll.by_type("response.cancelled")

        await session.close()

    @pytest.mark.asyncio
    async def test_response_done_is_deferred_until_paused_interrupt_is_decided(self):
        tts = BriefTTS()
        coll = Collector()
        cfg = ConversationConfig(
            stt_model="x:1", tts_model="y:1", voice="default",
            language="en",
            policy=TurnPolicy(
                min_interrupt_duration_ms=300,
                max_endpointing_delay_ms=500,
                stable_speaking_min_ms=50,
                speaking_interrupt_min_duration_ms=300,
            ),
        )
        session = ConversationSession(scheduler=Scheduler(tts), config=cfg, on_event=coll)
        await session.start()

        await session.submit_response_text("The assistant is still speaking.")
        for _ in range(50):
            await asyncio.sleep(0.01)
            if session.state == TurnState.SPEAKING:
                break
        assert session.state == TurnState.SPEAKING

        session._audio_ring = _voice_signal(0.60, amp=0.15)
        session._latest_partial = StreamTranscript(text="I am interrupting you", is_partial=True)
        session._vad_started_at = time.monotonic() - 0.60
        await session._event_queue.put(TurnEvent(
            type=TurnEventType.SPEECH_STARTED,
            payload={"confirm_window_ms": 300},
        ))
        await asyncio.sleep(0.15)
        assert session.state == TurnState.PAUSED
        assert coll.by_type("response.audio.clear")
        assert not coll.by_type("response.done")

        await asyncio.sleep(0.30)
        assert session.state == TurnState.INTERRUPTED
        assert coll.by_type("interruption.detected")
        assert coll.by_type("response.cancelled")
        assert not coll.by_type("response.done")

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
        session._latest_partial = StreamTranscript(text="stop", is_partial=True)
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


class TestClassifierFailureFallsBackToBackchannel:
    @pytest.mark.asyncio
    async def test_exception_in_classifier_defaults_to_backchannel(self):

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
        session._latest_partial = StreamTranscript(text="actually", is_partial=True)
        await session._event_queue.put(TurnEvent(
            type=TurnEventType.SPEECH_STARTED,
            payload={"confirm_window_ms": 80},
        ))
        await asyncio.sleep(0.25)


        assert session.state == TurnState.SPEAKING
        assert coll.by_type("interruption.false_positive")

        await session.close()
