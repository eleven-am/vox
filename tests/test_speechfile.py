"""Tests for vox.core.speechfile — Speechfile parser."""

from __future__ import annotations

import pytest

from vox.core.speechfile import SpeechfileParseError, parse_speechfile
from vox.core.types import ModelFormat, ModelType, VoiceInfo


# ---------------------------------------------------------------------------
# Minimal / full parse
# ---------------------------------------------------------------------------

class TestBasicParsing:
    def test_parse_minimal_only_from(self):
        sf = parse_speechfile("FROM openai/whisper-large-v3")
        assert sf.source == "openai/whisper-large-v3"
        # Defaults
        assert sf.type == ModelType.STT
        assert sf.format == ModelFormat.ONNX
        assert sf.architecture == ""
        assert sf.adapter == ""
        assert sf.parameters == {}
        assert sf.voices == ()
        assert sf.license == ""
        assert sf.description == ""
        assert sf.files == ()

    def test_parse_full_speechfile(self):
        content = """\
FROM hexgrad/Kokoro-82M-v1.0-ONNX
ARCHITECTURE kokoro
TYPE tts
ADAPTER kokoro
FORMAT onnx
PARAMETER sample_rate 24000
PARAMETER speed 1.5
PARAMETER default_voice af_heart
VOICE af_heart "American Female - Heart"
VOICE bf_emma "British Female - Emma"
LICENSE Apache-2.0
DESCRIPTION "Kokoro 82M ONNX TTS"
FILES model.onnx, voices.bin, config.json
"""
        sf = parse_speechfile(content)
        assert sf.source == "hexgrad/Kokoro-82M-v1.0-ONNX"
        assert sf.architecture == "kokoro"
        assert sf.type == ModelType.TTS
        assert sf.adapter == "kokoro"
        assert sf.format == ModelFormat.ONNX
        assert sf.parameters["sample_rate"] == 24000
        assert sf.parameters["speed"] == 1.5
        assert sf.parameters["default_voice"] == "af_heart"
        assert len(sf.voices) == 2
        assert sf.voices[0] == VoiceInfo(id="af_heart", name="American Female - Heart")
        assert sf.voices[1] == VoiceInfo(id="bf_emma", name="British Female - Emma")
        assert sf.license == "Apache-2.0"
        assert sf.description == "Kokoro 82M ONNX TTS"
        assert sf.files == ("model.onnx", "voices.bin", "config.json")


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestParseErrors:
    def test_parse_missing_from_raises(self):
        with pytest.raises(SpeechfileParseError, match="missing a required FROM"):
            parse_speechfile("TYPE tts\nFORMAT onnx")

    def test_parse_invalid_type_raises(self):
        with pytest.raises(SpeechfileParseError, match="Invalid TYPE"):
            parse_speechfile("FROM repo\nTYPE video")

    def test_parse_invalid_format_raises(self):
        with pytest.raises(SpeechfileParseError, match="Invalid FORMAT"):
            parse_speechfile("FROM repo\nFORMAT safetensors")


# ---------------------------------------------------------------------------
# Parameter coercion
# ---------------------------------------------------------------------------

class TestParameterCoercion:
    def test_parse_parameter_int_coercion(self):
        sf = parse_speechfile("FROM repo\nPARAMETER sample_rate 16000")
        assert sf.parameters["sample_rate"] == 16000
        assert isinstance(sf.parameters["sample_rate"], int)

    def test_parse_parameter_float_coercion(self):
        sf = parse_speechfile("FROM repo\nPARAMETER temperature 0.7")
        assert sf.parameters["temperature"] == pytest.approx(0.7)
        assert isinstance(sf.parameters["temperature"], float)

    def test_parse_parameter_string_value(self):
        sf = parse_speechfile('FROM repo\nPARAMETER voice "af_heart"')
        assert sf.parameters["voice"] == "af_heart"
        assert isinstance(sf.parameters["voice"], str)


# ---------------------------------------------------------------------------
# VOICE directive
# ---------------------------------------------------------------------------

class TestVoiceParsing:
    def test_parse_voice_with_quoted_name(self):
        sf = parse_speechfile('FROM repo\nVOICE af_heart "American Female - Heart"')
        assert len(sf.voices) == 1
        assert sf.voices[0].id == "af_heart"
        assert sf.voices[0].name == "American Female - Heart"


# ---------------------------------------------------------------------------
# FILES directive
# ---------------------------------------------------------------------------

class TestFilesParsing:
    def test_parse_files_comma_separated(self):
        sf = parse_speechfile("FROM repo\nFILES model.onnx , voices.bin , config.json")
        assert sf.files == ("model.onnx", "voices.bin", "config.json")


# ---------------------------------------------------------------------------
# Comments, blank lines, unknown directives
# ---------------------------------------------------------------------------

class TestWhitespaceAndComments:
    def test_parse_comments_ignored(self):
        content = "# This is a comment\nFROM repo\n# another comment"
        sf = parse_speechfile(content)
        assert sf.source == "repo"

    def test_parse_blank_lines_ignored(self):
        content = "\n\nFROM repo\n\n\n"
        sf = parse_speechfile(content)
        assert sf.source == "repo"

    def test_parse_unknown_directives_ignored(self):
        content = "FROM repo\nFOOBAR something\nBAZQUX another"
        sf = parse_speechfile(content)
        assert sf.source == "repo"


# ---------------------------------------------------------------------------
# Isolation between calls
# ---------------------------------------------------------------------------

class TestParserIsolation:
    def test_parse_called_twice_no_shared_state(self):
        sf1 = parse_speechfile("FROM repo1\nPARAMETER k1 100\nVOICE v1 name1")
        sf2 = parse_speechfile("FROM repo2")

        # sf2 must not inherit state from sf1
        assert sf2.source == "repo2"
        assert sf2.parameters == {}
        assert sf2.voices == ()

        # sf1 must remain unchanged
        assert sf1.source == "repo1"
        assert sf1.parameters["k1"] == 100
        assert len(sf1.voices) == 1
