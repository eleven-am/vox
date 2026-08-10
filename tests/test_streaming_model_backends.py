from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vox.operations.conversation import parse_session_update, serialize_session_config
from vox.streaming.eou import (
    ConversationTurn,
    EOUConfig,
    TenTurnDetector,
    create_turn_detector,
)
from vox.streaming.pipeline import StreamPipeline, StreamPipelineConfig
from vox.streaming.vad import (
    SILERO_ONNX_CONTEXT_SAMPLES,
    SILERO_ONNX_WINDOW_SAMPLES,
    SileroOnnxVAD,
    TenVAD,
    VADConfig,
    _frames_to_timestamps,
    create_vad_backend,
)


class TestVADBackends:
    def test_unknown_vad_backend_fails_fast(self):
        with pytest.raises(ValueError, match="unknown VAD backend"):
            create_vad_backend("missing")

    def test_default_backend_is_silero_onnx(self):
        assert isinstance(create_vad_backend("silero"), SileroOnnxVAD)
        assert isinstance(create_vad_backend(""), SileroOnnxVAD)
        assert isinstance(create_vad_backend("silero-onnx"), SileroOnnxVAD)

    def test_silero_onnx_uses_16khz_window_and_context(self):
        assert SILERO_ONNX_WINDOW_SAMPLES == 512
        assert SILERO_ONNX_CONTEXT_SAMPLES == 64

    def test_silero_onnx_returns_no_speech_on_silence(self):
        vad = SileroOnnxVAD()
        silence = np.zeros(16_000, dtype=np.float32)
        assert all(probability < 0.5 for _, _, probability in vad.process_frames(silence))
        assert vad.process_frames(np.array([], dtype=np.float32)) == []

    def test_silero_onnx_runs_windowed_inference_without_error(self):
        vad = SileroOnnxVAD()
        rng = np.random.default_rng(0)
        # Multiple 512-sample windows so the context-prepend + state carry path
        # is exercised end to end; result must be well-formed timestamps.
        signal = rng.standard_normal(16_000).astype(np.float32) * 0.2
        frames = vad.process_frames(signal)
        assert frames
        for start, end, probability in frames:
            assert 0 <= start < end <= len(signal)
            assert 0.0 <= probability <= 1.0

    def test_silero_streaming_frames_consume_audio_once_and_preserve_recurrent_state(self, monkeypatch):
        states: list[float] = []

        class FakeSession:
            def run(self, _outputs, inputs):
                state = inputs["state"]
                states.append(float(state.reshape(-1)[0]))
                return np.array([[0.9]], dtype=np.float32), state + 1

        monkeypatch.setattr(SileroOnnxVAD, "_session", FakeSession())
        vad = SileroOnnxVAD()

        assert vad.process_frames(np.ones(320, dtype=np.float32)) == []
        first = vad.process_frames(np.ones(704, dtype=np.float32))
        second = vad.process_frames(np.ones(512, dtype=np.float32))

        assert [(start, end) for start, end, _ in first + second] == [
            (0, 512),
            (512, 1024),
            (1024, 1536),
        ]
        assert states == [0.0, 1.0, 2.0]

    def test_frames_to_timestamps_merges_and_pads(self):
        # two speech spans separated by a gap larger than min_silence -> two segments
        frames = [(0, 256), (256, 512), (4000, 4256)]
        ts = _frames_to_timestamps(
            frames,
            total_samples=8000,
            min_silence_duration_ms=100,   # 1600 samples; gap 4000-512 exceeds it
            speech_pad_ms=0,
            min_speech_duration_ms=1,
        )
        assert ts == [{"start": 0, "end": 512}, {"start": 4000, "end": 4256}]

    def test_frames_to_timestamps_drops_short_segments(self):
        ts = _frames_to_timestamps(
            [(0, 256)],
            total_samples=8000,
            min_silence_duration_ms=100,
            speech_pad_ms=0,
            min_speech_duration_ms=100,   # 1600 samples; the 256-sample span is too short
        )
        assert ts == []

    def test_frames_to_timestamps_enforces_min_speech_before_padding(self):
        short = _frames_to_timestamps(
            [(4000, 4512), (4512, 5024)],
            total_samples=16_000,
            min_silence_duration_ms=100,
            speech_pad_ms=100,
            min_speech_duration_ms=250,
        )
        assert short == []

        sustained = _frames_to_timestamps(
            [(4000 + (512 * i), 4000 + (512 * (i + 1))) for i in range(8)],
            total_samples=16_000,
            min_silence_duration_ms=100,
            speech_pad_ms=100,
            min_speech_duration_ms=250,
        )
        assert sustained == [{"start": 2400, "end": 9696}]

    def test_ten_vad_requires_optional_dependency(self, monkeypatch):
        monkeypatch.setattr("builtins.__import__", _missing_ten_vad_import)
        with pytest.raises(RuntimeError, match="TEN VAD backend requires"):
            TenVAD().process_frames(np.ones(160, dtype=np.float32))

    def test_ten_vad_streams_hop_frames_with_absolute_offsets(self, monkeypatch):
        class FakeTenVad:
            def __init__(self, hop_size, threshold):
                self.hop_size = hop_size
                self.threshold = threshold

            def process(self, frame):
                probability = 0.9 if np.any(frame) else 0.1
                return probability, int(probability >= self.threshold)

        monkeypatch.setitem(
            __import__("sys").modules,
            "ten_vad",
            SimpleNamespace(TenVad=FakeTenVad),
        )

        vad = TenVAD(hop_size=160)
        loud = np.ones(160, dtype=np.float32) * 0.25
        quiet = np.zeros(160, dtype=np.float32)

        assert vad.process_frames(loud[:80]) == []
        first = vad.process_frames(np.concatenate([loud[80:], quiet]))

        assert [(start, end) for start, end, _ in first] == [(0, 160), (160, 320)]
        assert first[0][2] == 1.0
        assert first[1][2] == pytest.approx(0.1)

    def test_silero_stream_frames_are_invariant_to_caller_chunking(self):
        rng = np.random.default_rng(1234)
        audio = rng.standard_normal(SILERO_ONNX_WINDOW_SAMPLES * 12).astype(np.float32) * 0.2

        single = SileroOnnxVAD().process_frames(audio)

        chunked_vad = SileroOnnxVAD()
        chunked: list[tuple[int, int, float]] = []
        for start in range(0, len(audio), 100):
            chunked.extend(chunked_vad.process_frames(audio[start:start + 100]))

        assert [(start, end) for start, end, _ in chunked] == [(start, end) for start, end, _ in single]
        assert [probability for _, _, probability in chunked] == pytest.approx(
            [probability for _, _, probability in single]
        )
        assert len(single) == 12


class TestTurnDetectorBackends:
    def test_unknown_turn_detector_fails_fast(self):
        with pytest.raises(ValueError, match="requires a scheduler"):
            create_turn_detector("missing")

    def test_ten_turn_detector_requires_optional_dependencies(self, monkeypatch):
        monkeypatch.setattr("builtins.__import__", _missing_transformers_import)
        detector = TenTurnDetector()
        with pytest.raises(RuntimeError, match="TEN turn detector requires"):
            detector.predict([ConversationTurn(role="user", content="hello")])

    def test_ten_turn_detector_maps_labels_to_probability(self, monkeypatch):
        detector = TenTurnDetector()
        monkeypatch.setattr(detector, "classify", lambda *_args, **_kwargs: "finished")
        assert detector.predict([ConversationTurn(role="user", content="hello")]) == 1.0

        monkeypatch.setattr(detector, "classify", lambda *_args, **_kwargs: "wait")
        assert detector.predict([ConversationTurn(role="user", content="wait")]) == 0.0


class TestPipelineBackendConfig:
    def test_pipeline_uses_configured_vad_and_turn_detector(self, monkeypatch):
        fake_detector = SimpleNamespace(
            predict=lambda *_args, **_kwargs: 1.0,
            token_count=lambda _text: 1,
        )
        monkeypatch.setattr(
            "vox.streaming.pipeline.create_turn_detector",
            lambda name, **_kwargs: fake_detector if name == "fake-turn" else None,
        )

        pipeline = StreamPipeline(
            scheduler=SimpleNamespace(),
            config=StreamPipelineConfig(
                vad_config=VADConfig(backend="silero"),
                eou_config=EOUConfig(model="fake-turn"),
            ),
        )

        assert pipeline._eou_model is fake_detector
        assert pipeline._vad.config.backend == "silero"


class TestConversationBackendConfig:
    def test_session_update_accepts_backend_selection(self):
        config = parse_session_update({
            "session": {
                "stt_model": "stt",
                "tts_model": "tts",
                "vad_backend": "ten-vad",
                "turn_detector": "ten-turn",
            }
        })

        assert config.vad_backend == "ten-vad"
        assert config.turn_detector == "ten-turn"

        serialized = serialize_session_config(config)
        assert serialized["vad_backend"] == "ten-vad"
        assert serialized["turn_detector"] == "ten-turn"


def _missing_ten_vad_import(name, *args, **kwargs):
    if name == "ten_vad":
        raise ImportError("missing ten_vad")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


def _missing_transformers_import(name, *args, **kwargs):
    if name in {"torch", "transformers"}:
        raise ImportError(f"missing {name}")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


_ORIGINAL_IMPORT = __import__
