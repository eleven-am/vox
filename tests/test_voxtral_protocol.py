from __future__ import annotations

import base64
import json

import numpy as np
import pytest

from vox_voxtral.protocol import (
    OP_SHUTDOWN,
    OP_SYNTHESIZE,
    STATUS_ERROR,
    STATUS_OK,
    STATUS_READY,
    ErrorResponse,
    OkResponse,
    ReadyResponse,
    ShutdownRequest,
    SynthesizeRequest,
    accumulate_chunk,
    decode_response,
    extract_audio_chunk,
    is_error,
    is_ok,
    is_ready,
)


class TestSynthesizeRequest:
    def test_make_sets_op_and_fields(self):
        req = SynthesizeRequest.make(text="hello", voice="neutral_female")
        assert req.op == OP_SYNTHESIZE
        assert req.text == "hello"
        assert req.voice == "neutral_female"

    def test_encode_produces_valid_json(self):
        req = SynthesizeRequest.make(text="hello", voice="cheerful_female")
        payload = json.loads(req.encode())
        assert payload["op"] == OP_SYNTHESIZE
        assert payload["text"] == "hello"
        assert payload["voice"] == "cheerful_female"

    def test_encode_roundtrip(self):
        req = SynthesizeRequest.make(text="test text", voice="casual_male")
        raw = req.encode()
        parsed = json.loads(raw)
        assert parsed["text"] == "test text"
        assert parsed["voice"] == "casual_male"


class TestShutdownRequest:
    def test_make_sets_op(self):
        req = ShutdownRequest.make()
        assert req.op == OP_SHUTDOWN

    def test_encode_produces_valid_json(self):
        req = ShutdownRequest.make()
        payload = json.loads(req.encode())
        assert payload["op"] == OP_SHUTDOWN


class TestReadyResponse:
    def test_decode_from_payload(self):
        resp = ReadyResponse.decode({"status": STATUS_READY})
        assert resp.status == STATUS_READY

    def test_is_ready_helper(self):
        assert is_ready({"status": STATUS_READY}) is True
        assert is_ready({"status": STATUS_OK}) is False
        assert is_ready({"status": STATUS_ERROR}) is False


class TestOkResponse:
    def _make_b64(self, audio: bytes) -> str:
        return base64.b64encode(audio).decode("ascii")

    def test_decode_roundtrip(self):
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        b64 = self._make_b64(audio)
        resp = OkResponse.decode({"status": STATUS_OK, "sample_rate": 24000, "audio_b64": b64})
        assert resp.status == STATUS_OK
        assert resp.sample_rate == 24000
        assert resp.audio_bytes() == audio

    def test_decode_uses_default_sample_rate(self):
        audio = b"\x00\x00\x80?"
        resp = OkResponse.decode({"status": STATUS_OK, "audio_b64": self._make_b64(audio)})
        assert resp.sample_rate == 24000

    def test_encode_audio_produces_valid_json(self):
        audio = b"\x01\x02\x03\x04"
        raw = OkResponse.encode_audio(audio, sample_rate=24000)
        payload = json.loads(raw)
        assert payload["status"] == STATUS_OK
        assert payload["sample_rate"] == 24000
        assert base64.b64decode(payload["audio_b64"]) == audio

    def test_is_ok_helper(self):
        assert is_ok({"status": STATUS_OK}) is True
        assert is_ok({"status": STATUS_READY}) is False
        assert is_ok({"status": STATUS_ERROR}) is False


class TestErrorResponse:
    def test_decode_from_payload(self):
        resp = ErrorResponse.decode({"status": STATUS_ERROR, "error": "boom"})
        assert resp.status == STATUS_ERROR
        assert resp.error == "boom"

    def test_decode_missing_error_field(self):
        resp = ErrorResponse.decode({"status": STATUS_ERROR})
        assert resp.error == "unknown error"

    def test_encode_error_produces_valid_json(self):
        raw = ErrorResponse.encode_error("something went wrong")
        payload = json.loads(raw)
        assert payload["status"] == STATUS_ERROR
        assert payload["error"] == "something went wrong"

    def test_is_error_helper(self):
        assert is_error({"status": STATUS_ERROR}) is True
        assert is_error({"status": STATUS_OK}) is False
        assert is_error({"status": STATUS_READY}) is False


class TestDecodeResponse:
    def test_decode_response_parses_json(self):
        payload = decode_response('{"status": "ok", "sample_rate": 24000}')
        assert payload["status"] == "ok"
        assert payload["sample_rate"] == 24000

    def test_decode_response_raises_on_invalid(self):
        import json as _json
        with pytest.raises(_json.JSONDecodeError):
            decode_response("not json")


class TestExtractAudioChunk:
    def test_tensor_like_detach(self):
        import unittest.mock as mock
        tensor = mock.MagicMock()
        tensor.detach.return_value = tensor
        tensor.float.return_value = tensor
        tensor.cpu.return_value = tensor
        tensor.numpy.return_value = np.array([1.0, 2.0], dtype=np.float32)

        result = extract_audio_chunk(tensor, chunk_idx=0)
        assert result.dtype == np.float32
        np.testing.assert_array_almost_equal(result, [1.0, 2.0])

    def test_numpy_array_passthrough(self):
        arr = np.array([0.5, 0.6], dtype=np.float32)
        result = extract_audio_chunk(arr, chunk_idx=0)
        np.testing.assert_array_equal(result, arr)

    def test_list_selects_by_chunk_idx(self):
        chunk0 = np.array([1.0], dtype=np.float32)
        chunk1 = np.array([2.0], dtype=np.float32)
        result = extract_audio_chunk([chunk0, chunk1], chunk_idx=1)
        np.testing.assert_array_equal(result, chunk1)

    def test_list_clamps_to_last_when_idx_out_of_range(self):
        chunk = np.array([9.0], dtype=np.float32)
        result = extract_audio_chunk([chunk], chunk_idx=5)
        np.testing.assert_array_equal(result, chunk)

    def test_empty_list_returns_empty(self):
        result = extract_audio_chunk([], chunk_idx=0)
        assert result.dtype == np.float32
        assert len(result) == 0


class TestAccumulateChunk:
    def test_accumulates_new_samples(self):
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        out, total = accumulate_chunk(arr, accumulated_sample=0, finished=False)
        np.testing.assert_array_equal(out, arr)
        assert total == 3

    def test_slices_on_finished(self):
        arr = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        out, total = accumulate_chunk(arr, accumulated_sample=2, finished=True)
        np.testing.assert_array_equal(out, arr[2:])
        assert total == 2 + 2

    def test_no_slice_when_not_finished(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        out, total = accumulate_chunk(arr, accumulated_sample=2, finished=False)
        np.testing.assert_array_equal(out, arr)
        assert total == 6

    def test_no_slice_when_array_not_longer_than_accumulated(self):
        arr = np.array([1.0, 2.0], dtype=np.float32)
        out, total = accumulate_chunk(arr, accumulated_sample=3, finished=True)
        np.testing.assert_array_equal(out, arr)
        assert total == 5
