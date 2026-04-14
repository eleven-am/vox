from __future__ import annotations

import numpy as np
import pytest

from vox.streaming.buffer import AudioRingBuffer
from vox.streaming.codecs import pcm16_to_float32, float32_to_pcm16, resample_audio
from vox.streaming.partials import deduplicate_words
from vox.streaming.session import SpeechSession
from vox.streaming.types import (
    TARGET_SAMPLE_RATE,
    StreamSessionConfig,
    SpeechStarted,
    SpeechStopped,
    StreamTranscript,
    samples_to_ms,
)


class TestAudioRingBuffer:
    def test_empty_buffer(self):
        buf = AudioRingBuffer(1000)
        assert len(buf) == 0
        assert buf.get_all().size == 0

    def test_append_within_capacity(self):
        buf = AudioRingBuffer(1000)
        audio = np.ones(500, dtype=np.float32)
        buf.append(audio)
        assert len(buf) == 500
        np.testing.assert_array_equal(buf.get_all(), audio)

    def test_append_wraps_around(self):
        buf = AudioRingBuffer(100)
        buf.append(np.ones(80, dtype=np.float32))
        buf.append(np.ones(40, dtype=np.float32) * 2.0)
        assert len(buf) == 100
        result = buf.get_all()
        assert result[-40:].tolist() == [2.0] * 40

    def test_append_larger_than_buffer(self):
        buf = AudioRingBuffer(50)
        audio = np.arange(100, dtype=np.float32)
        buf.append(audio)
        assert len(buf) == 50
        np.testing.assert_array_equal(buf.get_all(), audio[-50:])

    def test_get_last_n(self):
        buf = AudioRingBuffer(100)
        buf.append(np.arange(50, dtype=np.float32))
        last_10 = buf.get_last_n(10)
        np.testing.assert_array_equal(last_10, np.arange(40, 50, dtype=np.float32))

    def test_get_last_n_more_than_length(self):
        buf = AudioRingBuffer(100)
        buf.append(np.arange(30, dtype=np.float32))
        result = buf.get_last_n(50)
        assert len(result) == 30

    def test_get_slice(self):
        buf = AudioRingBuffer(100)
        buf.append(np.arange(50, dtype=np.float32))
        result = buf.get_slice(10, 20)
        np.testing.assert_array_equal(result, np.arange(10, 20, dtype=np.float32))

    def test_clear(self):
        buf = AudioRingBuffer(100)
        buf.append(np.ones(50, dtype=np.float32))
        buf.clear()
        assert len(buf) == 0
        assert buf.get_all().size == 0

    def test_append_empty(self):
        buf = AudioRingBuffer(100)
        buf.append(np.array([], dtype=np.float32))
        assert len(buf) == 0


class TestCodecs:
    def test_pcm16_to_float32_silence(self):
        pcm = np.zeros(100, dtype=np.int16).tobytes()
        result = pcm16_to_float32(pcm)
        assert result.dtype == np.float32
        np.testing.assert_array_equal(result, np.zeros(100, dtype=np.float32))

    def test_pcm16_to_float32_max_value(self):
        pcm = np.array([32767], dtype=np.int16).tobytes()
        result = pcm16_to_float32(pcm)
        assert abs(result[0] - 1.0) < 0.001

    def test_float32_to_pcm16_roundtrip(self):
        original = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
        pcm_bytes = float32_to_pcm16(original)
        recovered = pcm16_to_float32(pcm_bytes)
        np.testing.assert_allclose(recovered, original, atol=0.001)

    def test_float32_to_pcm16_clamps(self):
        audio = np.array([2.0, -2.0], dtype=np.float32)
        pcm_bytes = float32_to_pcm16(audio)
        recovered = pcm16_to_float32(pcm_bytes)
        assert abs(recovered[0] - 1.0) < 0.001
        assert abs(recovered[1] - (-1.0)) < 0.001

    def test_resample_same_rate(self):
        audio = np.ones(100, dtype=np.float32)
        result = resample_audio(audio, 16000, 16000)
        assert result is audio

    def test_resample_changes_length(self):
        audio = np.ones(16000, dtype=np.float32)
        result = resample_audio(audio, 16000, 48000)
        assert abs(len(result) - 48000) < 100


class TestSpeechSession:
    def test_initial_state(self):
        session = SpeechSession()
        assert session.is_active() is False
        assert session.get_buffer_length() == 0

    def test_start_stop_speech(self):
        session = SpeechSession()
        session.start_speech()
        assert session.is_active() is True
        session.stop_speech()
        assert session.is_active() is False

    def test_append_audio_when_active(self):
        session = SpeechSession()
        session.start_speech()
        audio = np.ones(1000, dtype=np.float32)
        session.append_audio(audio)
        assert session.get_buffer_length() == 1000

    def test_append_audio_when_inactive(self):
        session = SpeechSession()
        audio = np.ones(1000, dtype=np.float32)
        session.append_audio(audio)
        assert session.get_buffer_length() == 0

    def test_get_buffer_tail(self):
        session = SpeechSession()
        session.start_speech()
        session.append_audio(np.arange(1000, dtype=np.float32))
        tail = session.get_buffer_tail(100)
        assert len(tail) == 100

    def test_partial_state(self):
        session = SpeechSession()
        session.update_partial(500, ["hello", "world"])
        ms, words = session.get_partial_state()
        assert ms == 500
        assert words == ["hello", "world"]

    def test_start_speech_clears_buffer(self):
        session = SpeechSession()
        session.start_speech()
        session.append_audio(np.ones(1000, dtype=np.float32))
        session.stop_speech()
        session.start_speech()
        assert session.get_buffer_length() == 0


class TestDeduplicateWords:
    def test_no_overlap(self):
        new_text, words = deduplicate_words("hello world", [])
        assert new_text == "hello world"
        assert words == ["hello", "world"]

    def test_full_overlap(self):
        new_text, words = deduplicate_words("hello world", ["hello", "world"])
        assert new_text == ""

    def test_partial_overlap(self):
        new_text, words = deduplicate_words("world foo bar", ["hello", "world"])
        assert new_text == "foo bar"
        assert words == ["hello", "world", "foo", "bar"]

    def test_empty_text(self):
        new_text, words = deduplicate_words("", ["hello"])
        assert new_text == ""


class TestStreamTypes:
    def test_samples_to_ms(self):
        assert samples_to_ms(16000) == 1000
        assert samples_to_ms(8000) == 500
        assert samples_to_ms(0) == 0

    def test_session_config_defaults(self):
        config = StreamSessionConfig()
        assert config.language == "en"
        assert config.sample_rate == 16000
        assert config.partials is False
        assert config.partial_window_ms == 1500

    def test_speech_started(self):
        event = SpeechStarted(timestamp_ms=100)
        assert event.timestamp_ms == 100

    def test_stream_transcript(self):
        t = StreamTranscript(text="hello", model="whisper:large-v3", eou_probability=0.9)
        assert t.text == "hello"
        assert t.eou_probability == 0.9


class TestStreamProtoMessages:
    def test_stream_config(self):
        from vox.grpc import vox_pb2

        cfg = vox_pb2.StreamConfig(
            language="en",
            model="whisper:large-v3",
            partials=True,
            partial_window_ms=1500,
        )
        assert cfg.language == "en"
        assert cfg.partials is True

    def test_stream_input_config(self):
        from vox.grpc import vox_pb2

        msg = vox_pb2.StreamInput(
            config=vox_pb2.StreamConfig(language="fr", model="voxtral:mini-3b")
        )
        assert msg.WhichOneof("msg") == "config"
        assert msg.config.language == "fr"

    def test_stream_input_audio(self):
        from vox.grpc import vox_pb2

        audio_bytes = np.zeros(100, dtype=np.int16).tobytes()
        msg = vox_pb2.StreamInput(
            audio=vox_pb2.AudioFrame(pcm16=audio_bytes, sample_rate=16000)
        )
        assert msg.WhichOneof("msg") == "audio"
        assert len(msg.audio.pcm16) == 200

    def test_stream_input_opus(self):
        from vox.grpc import vox_pb2

        msg = vox_pb2.StreamInput(
            opus_frame=vox_pb2.OpusFrame(data=b"\x00\x01\x02", sample_rate=48000, channels=1)
        )
        assert msg.WhichOneof("msg") == "opus_frame"

    def test_stream_output_ready(self):
        from vox.grpc import vox_pb2

        msg = vox_pb2.StreamOutput(ready=vox_pb2.StreamReady())
        assert msg.WhichOneof("msg") == "ready"

    def test_stream_output_speech_started(self):
        from vox.grpc import vox_pb2

        msg = vox_pb2.StreamOutput(
            speech_started=vox_pb2.StreamSpeechStarted(timestamp_ms=123)
        )
        assert msg.speech_started.timestamp_ms == 123

    def test_stream_output_transcript(self):
        from vox.grpc import vox_pb2

        msg = vox_pb2.StreamOutput(
            transcript=vox_pb2.StreamTranscriptResult(
                text="hello world",
                is_partial=False,
                eou_probability=0.95,
            )
        )
        assert msg.transcript.text == "hello world"
        assert msg.transcript.eou_probability == pytest.approx(0.95)

    def test_stream_output_error(self):
        from vox.grpc import vox_pb2

        msg = vox_pb2.StreamOutput(
            error=vox_pb2.StreamErrorMessage(message="something went wrong")
        )
        assert msg.error.message == "something went wrong"
