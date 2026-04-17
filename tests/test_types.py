"""Tests for vox.core.types — model references, validation, and frozen dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from vox.core.types import (
    ModelFormat,
    ModelInfo,
    ModelRef,
    ModelType,
    SynthesizeChunk,
    parse_model_name,
)







class TestModelRef:
    def test_model_ref_parse_with_tag(self):
        ref = ModelRef.parse("whisper:large-v3")
        assert ref.name == "whisper"
        assert ref.tag == "large-v3"

    def test_model_ref_parse_without_tag_defaults_latest(self):
        ref = ModelRef.parse("whisper")
        assert ref.name == "whisper"
        assert ref.tag == "latest"

    def test_model_ref_str_roundtrip(self):
        original = "whisper:large-v3"
        ref = ModelRef.parse(original)
        assert str(ref) == original


        ref2 = ModelRef.parse("whisper")
        assert str(ref2) == "whisper:latest"


def test_parse_model_name_delegates_to_model_ref():
    name, tag = parse_model_name("kokoro:v1")
    assert name == "kokoro"
    assert tag == "v1"

    name2, tag2 = parse_model_name("kokoro")
    assert name2 == "kokoro"
    assert tag2 == "latest"







class TestSynthesizeChunk:
    def test_synthesize_chunk_rejects_zero_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            SynthesizeChunk(audio=b"\x00\x00", sample_rate=0)

    def test_synthesize_chunk_rejects_negative_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            SynthesizeChunk(audio=b"\x00\x00", sample_rate=-16000)

    def test_synthesize_chunk_accepts_valid(self):
        chunk = SynthesizeChunk(audio=b"\x00\x00\x00\x00", sample_rate=24000)
        assert chunk.sample_rate == 24000
        assert chunk.audio == b"\x00\x00\x00\x00"
        assert chunk.is_final is False







class TestModelInfo:
    """Tests for ModelInfo construction, validation, and factory method."""

    _DEFAULTS = dict(
        type=ModelType.STT,
        format=ModelFormat.ONNX,
        architecture="whisper",
        adapter="whisper",
    )

    def _make(self, **overrides):
        kwargs = {"name": "whisper", "tag": "large-v3", **self._DEFAULTS, **overrides}
        return ModelInfo(**kwargs)

    def test_model_info_rejects_empty_name(self):
        with pytest.raises(ValueError, match="name must be non-empty"):
            self._make(name="")

    def test_model_info_rejects_empty_tag(self):
        with pytest.raises(ValueError, match="tag must be non-empty"):
            self._make(tag="")

    def test_model_info_rejects_empty_adapter(self):
        with pytest.raises(ValueError, match="adapter must be non-empty"):
            self._make(adapter="")

    def test_model_info_full_name(self):
        info = self._make()
        assert info.full_name == "whisper:large-v3"

    def test_model_info_from_manifest_config_valid(self):
        config = {
            "type": "stt",
            "format": "onnx",
            "architecture": "whisper",
            "adapter": "whisper",
            "description": "A test model",
        }
        info = ModelInfo.from_manifest_config("whisper", "large-v3", config, size_bytes=1024)
        assert info.name == "whisper"
        assert info.tag == "large-v3"
        assert info.type is ModelType.STT
        assert info.format is ModelFormat.ONNX
        assert info.adapter == "whisper"
        assert info.size_bytes == 1024
        assert info.description == "A test model"

    def test_model_info_from_manifest_config_missing_type_raises(self):
        config = {"format": "onnx", "adapter": "whisper"}
        with pytest.raises(ValueError, match="missing required key.*'type'"):
            ModelInfo.from_manifest_config("whisper", "v1", config)

    def test_model_info_from_manifest_config_missing_adapter_raises(self):
        config = {"type": "stt", "format": "onnx"}
        with pytest.raises(ValueError, match="missing required key.*'adapter'"):
            ModelInfo.from_manifest_config("whisper", "v1", config)







def test_frozen_dataclasses_are_immutable():
    ref = ModelRef.parse("whisper:large-v3")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.name = "other"

    chunk = SynthesizeChunk(audio=b"\x00", sample_rate=16000)
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.sample_rate = 44100

    info = ModelInfo(
        name="m",
        tag="t",
        type=ModelType.STT,
        format=ModelFormat.ONNX,
        architecture="w",
        adapter="w",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.name = "changed"
