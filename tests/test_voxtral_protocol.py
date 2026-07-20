from __future__ import annotations

import base64

import numpy as np
from vox_voxtral.protocol import (
    OP_SYNTHESIZE,
    VOXTRAL_TTS_SAMPLE_RATE,
    SynthesizeRequest,
    SynthesizeResponse,
    accumulate_chunk,
    extract_audio_chunk,
)


class TestSynthesizeRequest:
    def test_payload_sets_op_and_fields(self):
        payload = SynthesizeRequest(text="hello", voice="neutral_female").payload()
        assert payload == {"op": OP_SYNTHESIZE, "text": "hello", "voice": "neutral_female"}

    def test_decode_roundtrip(self):
        request = SynthesizeRequest(text="test text", voice="casual_male")
        decoded = SynthesizeRequest.decode(request.payload())
        assert decoded == request

    def test_decode_defaults_missing_fields_to_empty(self):
        decoded = SynthesizeRequest.decode({"op": OP_SYNTHESIZE})
        assert decoded.text == ""
        assert decoded.voice == ""


class TestSynthesizeResponse:
    def test_from_audio_payload_roundtrip(self):
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32).tobytes()
        response = SynthesizeResponse.from_audio(audio, sample_rate=24000)
        decoded = SynthesizeResponse.decode(response.payload())
        assert decoded.sample_rate == 24000
        assert decoded.audio_bytes() == audio

    def test_payload_carries_base64_audio(self):
        audio = b"\x01\x02\x03\x04"
        payload = SynthesizeResponse.from_audio(audio, sample_rate=24000).payload()
        assert payload["sample_rate"] == 24000
        assert base64.b64decode(payload["audio_b64"]) == audio

    def test_decode_uses_default_sample_rate(self):
        audio = b"\x00\x00\x80?"
        b64 = base64.b64encode(audio).decode("ascii")
        response = SynthesizeResponse.decode({"audio_b64": b64})
        assert response.sample_rate == VOXTRAL_TTS_SAMPLE_RATE
        assert response.audio_bytes() == audio


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
