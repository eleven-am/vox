from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from vox.streaming.buffer import AudioRingBuffer
from vox.streaming.codecs import pcm16_to_float32, float32_to_pcm16, resample_audio
from vox.streaming.partials import PartialTranscriptService, deduplicate_words
from vox.streaming.session import SpeechSession, MAX_SESSION_BUFFER_SAMPLES
from vox.streaming.types import (
    TARGET_SAMPLE_RATE,
    StreamSessionConfig,
    SpeechStarted,
    SpeechStopped,
    StreamTranscript,
    StreamError,
    samples_to_ms,
    MS_PER_SAMPLE,
)
from vox.streaming.vad import (
    VADConfig,
    VADProcessor,
    VADState,
    SpeechSegment,
    AudioRingBuffer as VADRingBuffer,
    MAX_BUFFER_SAMPLES,
    VAD_WINDOW_SIZE_SAMPLES,
)
from vox.streaming.eou import EOUModel, EOUConfig, ConversationTurn, MAX_HISTORY_TURNS


class TestAudioRingBufferEdgeCases:
    def test_single_sample(self):
        buf = AudioRingBuffer(10)
        buf.append(np.array([0.5], dtype=np.float32))
        assert len(buf) == 1
        np.testing.assert_array_equal(buf.get_all(), [0.5])

    def test_exact_capacity_fill(self):
        buf = AudioRingBuffer(100)
        buf.append(np.ones(100, dtype=np.float32))
        assert len(buf) == 100
        np.testing.assert_array_equal(buf.get_all(), np.ones(100, dtype=np.float32))

    def test_multiple_wraps(self):
        buf = AudioRingBuffer(10)
        for i in range(50):
            buf.append(np.array([float(i)], dtype=np.float32))
        assert len(buf) == 10
        result = buf.get_all()
        np.testing.assert_array_equal(result, np.arange(40, 50, dtype=np.float32))

    def test_get_slice_at_wrap_boundary(self):
        buf = AudioRingBuffer(10)
        buf.append(np.arange(8, dtype=np.float32))
        buf.append(np.arange(8, 14, dtype=np.float32))
        result = buf.get_slice(2, 8)
        assert len(result) == 6

    def test_get_slice_out_of_bounds_clamped(self):
        buf = AudioRingBuffer(100)
        buf.append(np.arange(50, dtype=np.float32))
        result = buf.get_slice(0, 1000)
        assert len(result) == 50

    def test_get_slice_reversed_returns_empty(self):
        buf = AudioRingBuffer(100)
        buf.append(np.arange(50, dtype=np.float32))
        result = buf.get_slice(30, 10)
        assert len(result) == 0

    def test_get_last_n_zero(self):
        buf = AudioRingBuffer(100)
        buf.append(np.ones(50, dtype=np.float32))
        result = buf.get_last_n(0)
        assert len(result) == 0

    def test_concurrent_append_and_read(self):
        buf = AudioRingBuffer(10000)
        errors = []

        def writer():
            for _ in range(100):
                buf.append(np.ones(100, dtype=np.float32))

        def reader():
            for _ in range(100):
                try:
                    buf.get_all()
                    buf.get_last_n(50)
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_repeated_clear_and_fill(self):
        buf = AudioRingBuffer(100)
        for _ in range(10):
            buf.append(np.ones(80, dtype=np.float32))
            buf.clear()
            assert len(buf) == 0
            buf.append(np.ones(50, dtype=np.float32))
            assert len(buf) == 50


class TestCodecsEdgeCases:
    def test_pcm16_empty_bytes(self):
        result = pcm16_to_float32(b"")
        assert len(result) == 0

    def test_pcm16_single_sample(self):
        pcm = np.array([16384], dtype=np.int16).tobytes()
        result = pcm16_to_float32(pcm)
        assert abs(result[0] - 0.5) < 0.001

    def test_float32_to_pcm16_empty(self):
        result = float32_to_pcm16(np.array([], dtype=np.float32))
        assert result == b""

    def test_float32_to_pcm16_very_small_values(self):
        audio = np.array([0.0001, -0.0001], dtype=np.float32)
        pcm = float32_to_pcm16(audio)
        assert len(pcm) == 4

    def test_resample_short_audio(self):
        audio = np.ones(10, dtype=np.float32)
        result = resample_audio(audio, 16000, 48000)
        assert len(result) > 10

    def test_resample_preserves_dtype(self):
        audio = np.random.randn(1000).astype(np.float32)
        result = resample_audio(audio, 16000, 44100)
        assert result.dtype == np.float32

    def test_pcm16_min_max_values(self):
        pcm = np.array([-32768, 32767], dtype=np.int16).tobytes()
        result = pcm16_to_float32(pcm)
        assert result[0] <= -0.99
        assert result[1] >= 0.99


class TestSpeechSessionEdgeCases:
    def test_double_start(self):
        session = SpeechSession()
        session.start_speech()
        session.append_audio(np.ones(500, dtype=np.float32))
        session.start_speech()
        assert session.get_buffer_length() == 0

    def test_stop_without_start(self):
        session = SpeechSession()
        session.stop_speech()
        assert session.is_active() is False

    def test_append_after_stop(self):
        session = SpeechSession()
        session.start_speech()
        session.append_audio(np.ones(500, dtype=np.float32))
        session.stop_speech()
        session.append_audio(np.ones(500, dtype=np.float32))
        assert session.get_buffer_length() == 500

    def test_buffer_max_capacity(self):
        session = SpeechSession()
        session.start_speech()
        huge_audio = np.ones(MAX_SESSION_BUFFER_SAMPLES + 1000, dtype=np.float32)
        session.append_audio(huge_audio)
        assert session.get_buffer_length() == MAX_SESSION_BUFFER_SAMPLES

    def test_concurrent_start_stop(self):
        session = SpeechSession()
        errors = []

        def toggle():
            try:
                for _ in range(100):
                    session.start_speech()
                    session.append_audio(np.ones(10, dtype=np.float32))
                    session.stop_speech()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggle) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_partial_state_update_concurrent(self):
        session = SpeechSession()
        errors = []

        def update():
            try:
                for i in range(100):
                    session.update_partial(i * 100, [f"word{i}"])
                    session.get_partial_state()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_get_buffer_audio_copies_data(self):
        session = SpeechSession()
        session.start_speech()
        session.append_audio(np.ones(100, dtype=np.float32))
        audio1 = session.get_buffer_audio()
        audio2 = session.get_buffer_audio()
        assert audio1 is not audio2
        np.testing.assert_array_equal(audio1, audio2)


class TestDeduplicateWordsEdgeCases:
    def test_single_word_overlap(self):
        new_text, words = deduplicate_words("hello world", ["hello"])
        assert new_text == "world"
        assert words == ["hello", "world"]

    def test_case_insensitive_overlap(self):
        new_text, words = deduplicate_words("Hello WORLD", ["hello", "world"])
        assert new_text == ""

    def test_whitespace_handling(self):
        new_text, words = deduplicate_words("  hello   world  ", [])
        assert new_text == "hello world"

    def test_long_overlap(self):
        confirmed = ["the", "quick", "brown", "fox"]
        new_text, words = deduplicate_words("brown fox jumps over", confirmed)
        assert new_text == "jumps over"

    def test_no_common_words(self):
        new_text, words = deduplicate_words("foo bar", ["baz", "qux"])
        assert new_text == "foo bar"

    def test_confirmed_words_grow(self):
        _, words = deduplicate_words("a b c", [])
        assert words == ["a", "b", "c"]
        _, words = deduplicate_words("c d e", words)
        assert words == ["a", "b", "c", "d", "e"]

    def test_single_word_text(self):
        new_text, words = deduplicate_words("hello", ["hello"])
        assert new_text == ""

    def test_repeated_words(self):
        new_text, words = deduplicate_words("yes yes yes", ["yes"])
        assert "yes" in new_text


class TestVADConfigDefaults:
    def test_default_values(self):
        config = VADConfig()
        assert config.start_threshold == 0.6
        assert config.continue_threshold == 0.4
        assert config.min_silence_duration_ms == 1000
        assert config.speech_pad_ms == 100
        assert config.min_speech_duration_ms == 250
        assert config.min_audio_duration_ms == 500
        assert config.max_utterance_ms == 15000

    def test_custom_values(self):
        config = VADConfig(start_threshold=0.8, max_utterance_ms=30000)
        assert config.start_threshold == 0.8
        assert config.max_utterance_ms == 30000


class TestVADState:
    def test_initial_state(self):
        state = VADState()
        assert state.audio_start_ms is None
        assert state.audio_end_ms is None
        assert state.active is False


class TestSpeechSegment:
    def test_empty_segment(self):
        segment = SpeechSegment(audio=np.array([], dtype=np.float32), start_ms=0, end_ms=0)
        assert len(segment.audio) == 0
        assert segment.end_ms - segment.start_ms == 0

    def test_segment_duration(self):
        audio = np.ones(16000, dtype=np.float32)
        segment = SpeechSegment(audio=audio, start_ms=0, end_ms=1000)
        assert segment.end_ms - segment.start_ms == 1000
        assert len(segment.audio) == 16000


class TestEOUConfig:
    def test_defaults(self):
        config = EOUConfig()
        assert config.threshold == 0.5
        assert config.max_context_turns == MAX_HISTORY_TURNS

    def test_custom_threshold(self):
        config = EOUConfig(threshold=0.8)
        assert config.threshold == 0.8


class TestConversationTurn:
    def test_user_turn(self):
        turn = ConversationTurn(role="user", content="hello")
        assert turn.role == "user"
        assert turn.content == "hello"

    def test_assistant_turn(self):
        turn = ConversationTurn(role="assistant", content="hi there")
        assert turn.role == "assistant"


class TestPartialTranscriptService:
    @pytest.mark.asyncio
    async def test_generate_partial_empty_buffer(self):
        mock_fn = AsyncMock(return_value=StreamTranscript(text="hello"))
        service = PartialTranscriptService(transcribe_async_fn=mock_fn)
        session = SpeechSession()
        config = StreamSessionConfig(partials=True)

        result = await service.generate_partial_async(session, config)
        assert result is None
        mock_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_generate_partial_below_stride(self):
        mock_fn = AsyncMock(return_value=StreamTranscript(text="hello"))
        service = PartialTranscriptService(transcribe_async_fn=mock_fn)
        session = SpeechSession()
        session.start_speech()
        session.append_audio(np.ones(8000, dtype=np.float32))
        config = StreamSessionConfig(partials=True, partial_window_ms=1500, partial_stride_ms=700)
        session.update_partial(400, [])

        result = await service.generate_partial_async(session, config)
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_partial_emits_new_words(self):
        mock_fn = AsyncMock(return_value=StreamTranscript(text="hello world foo"))
        service = PartialTranscriptService(transcribe_async_fn=mock_fn)
        session = SpeechSession()
        session.start_speech()
        session.append_audio(np.ones(32000, dtype=np.float32))
        config = StreamSessionConfig(partials=True, partial_window_ms=1500, partial_stride_ms=700)

        result = await service.generate_partial_async(session, config)
        assert result is not None
        assert result.text == "hello world foo"
        assert result.is_partial is True

    @pytest.mark.asyncio
    async def test_generate_partial_deduplicates(self):
        mock_fn = AsyncMock(return_value=StreamTranscript(text="hello world new"))
        service = PartialTranscriptService(transcribe_async_fn=mock_fn)
        session = SpeechSession()
        session.start_speech()
        session.append_audio(np.ones(32000, dtype=np.float32))
        session.update_partial(0, ["hello", "world"])
        config = StreamSessionConfig(partials=True, partial_window_ms=1500, partial_stride_ms=700)

        result = await service.generate_partial_async(session, config)
        assert result is not None
        assert result.text == "new"

    @pytest.mark.asyncio
    async def test_generate_partial_all_duplicate_returns_none(self):
        mock_fn = AsyncMock(return_value=StreamTranscript(text="hello world"))
        service = PartialTranscriptService(transcribe_async_fn=mock_fn)
        session = SpeechSession()
        session.start_speech()
        session.append_audio(np.ones(32000, dtype=np.float32))
        session.update_partial(0, ["hello", "world"])
        config = StreamSessionConfig(partials=True, partial_window_ms=1500, partial_stride_ms=700)

        result = await service.generate_partial_async(session, config)
        assert result is None

    def test_flush_remaining_audio_empty(self):
        mock_fn = AsyncMock()
        service = PartialTranscriptService(transcribe_async_fn=mock_fn)
        session = SpeechSession()
        result = service.flush_remaining_audio(session)
        assert result is None

    def test_flush_remaining_audio_has_data(self):
        mock_fn = AsyncMock()
        service = PartialTranscriptService(transcribe_async_fn=mock_fn)
        session = SpeechSession()
        session.start_speech()
        session.append_audio(np.ones(100, dtype=np.float32))
        result = service.flush_remaining_audio(session)
        assert result is not None
        assert len(result) == 100


class TestStreamPipelineConfig:
    def test_defaults(self):
        from vox.streaming.pipeline import StreamPipelineConfig
        config = StreamPipelineConfig()
        assert config.stt_workers == 4
        assert isinstance(config.vad_config, VADConfig)
        assert isinstance(config.eou_config, EOUConfig)


class TestStreamingGRPCServicerEdgeCases:
    @pytest.mark.asyncio
    async def test_servicer_unconfigured_audio(self):
        from vox.grpc.streaming_servicer import StreamingServiceServicer
        from vox.grpc import vox_pb2

        store = MagicMock()
        registry = MagicMock()
        scheduler = MagicMock()

        servicer = StreamingServiceServicer(store, registry, scheduler)

        audio_bytes = np.zeros(100, dtype=np.int16).tobytes()

        async def request_iter():
            yield vox_pb2.StreamInput(
                audio=vox_pb2.AudioFrame(pcm16=audio_bytes, sample_rate=16000)
            )

        ctx = MagicMock()
        ctx.cancelled.return_value = False

        messages = []
        async for msg in servicer.StreamTranscribe(request_iter(), ctx):
            messages.append(msg)

        assert len(messages) == 1
        assert messages[0].WhichOneof("msg") == "error"
        assert "not configured" in messages[0].error.message.lower()

    @pytest.mark.asyncio
    async def test_servicer_double_config(self):
        from vox.grpc.streaming_servicer import StreamingServiceServicer
        from vox.grpc import vox_pb2

        store = MagicMock()
        store.list_models.return_value = []
        registry = MagicMock()
        registry.available_models.return_value = {"whisper": {"large-v3": {"type": "stt"}}}
        scheduler = MagicMock()

        servicer = StreamingServiceServicer(store, registry, scheduler)

        async def request_iter():
            yield vox_pb2.StreamInput(config=vox_pb2.StreamConfig(language="en", model="whisper:large-v3"))
            yield vox_pb2.StreamInput(config=vox_pb2.StreamConfig(language="fr", model="whisper:large-v3"))

        ctx = MagicMock()
        ctx.cancelled.return_value = False

        messages = []
        async for msg in servicer.StreamTranscribe(request_iter(), ctx):
            messages.append(msg)

        assert messages[0].WhichOneof("msg") == "ready"
        assert messages[1].WhichOneof("msg") == "error"
        assert "already configured" in messages[1].error.message.lower()

    @pytest.mark.asyncio
    async def test_servicer_end_of_stream(self):
        from vox.grpc.streaming_servicer import StreamingServiceServicer
        from vox.grpc import vox_pb2

        store = MagicMock()
        store.list_models.return_value = []
        registry = MagicMock()
        registry.available_models.return_value = {"whisper": {"large-v3": {"type": "stt"}}}
        scheduler = MagicMock()

        servicer = StreamingServiceServicer(store, registry, scheduler)

        async def request_iter():
            yield vox_pb2.StreamInput(config=vox_pb2.StreamConfig(language="en", model="whisper:large-v3"))
            yield vox_pb2.StreamInput(end_of_stream=vox_pb2.EndOfStream())

        ctx = MagicMock()
        ctx.cancelled.return_value = False

        messages = []
        async for msg in servicer.StreamTranscribe(request_iter(), ctx):
            messages.append(msg)

        assert messages[0].WhichOneof("msg") == "ready"
