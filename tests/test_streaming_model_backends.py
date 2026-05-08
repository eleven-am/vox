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
from vox.streaming.vad import TenVAD, VADConfig, create_vad_backend


class TestVADBackends:
    def test_unknown_vad_backend_fails_fast(self):
        with pytest.raises(ValueError, match="unknown VAD backend"):
            create_vad_backend("missing")

    def test_ten_vad_requires_optional_dependency(self, monkeypatch):
        monkeypatch.setattr("builtins.__import__", _missing_ten_vad_import)
        with pytest.raises(RuntimeError, match="TEN VAD backend requires"):
            TenVAD().get_speech_timestamps(np.ones(160, dtype=np.float32))

    def test_ten_vad_converts_frame_flags_to_timestamps(self, monkeypatch):
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
        audio = np.concatenate([
            np.ones(160, dtype=np.float32) * 0.25,
            np.zeros(160, dtype=np.float32),
        ])

        timestamps = vad.get_speech_timestamps(
            audio,
            threshold=0.5,
            min_silence_duration_ms=1,
            speech_pad_ms=0,
            min_speech_duration_ms=1,
        )

        assert timestamps == [{"start": 0, "end": 160}]


class TestTurnDetectorBackends:
    def test_unknown_turn_detector_fails_fast(self):
        with pytest.raises(ValueError, match="unknown turn detector"):
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
            lambda name: fake_detector if name == "fake-turn" else None,
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
