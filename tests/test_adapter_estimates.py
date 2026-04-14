from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_kokoro_estimate_is_available_before_load():
    with patch.dict(
        "sys.modules",
        {
            "kokoro_onnx": MagicMock(),
            "onnxruntime": MagicMock(),
        },
    ):
        from vox_kokoro.adapter import KokoroAdapter

        adapter = KokoroAdapter()
        assert adapter.estimate_vram_bytes() == 330 * 1024 * 1024


def test_parakeet_estimate_uses_source_hint_before_load():
    with patch.dict(
        "sys.modules",
        {
            "onnx_asr": MagicMock(),
            "soundfile": MagicMock(),
        },
    ):
        from vox_parakeet.adapter import ParakeetAdapter

        adapter = ParakeetAdapter()
        assert adapter.estimate_vram_bytes(_source="nvidia/parakeet-tdt-0.6b-v3") == 1_300_000_000
