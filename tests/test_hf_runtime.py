from __future__ import annotations

import os
from pathlib import Path

from vox.core.hf_runtime import configure_hf_runtime


def test_configure_hf_runtime_sets_safe_defaults(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VOX_HOME", str(tmp_path / ".vox"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)

    configure_hf_runtime()

    expected = tmp_path / ".vox" / "cache" / "huggingface"
    assert expected.is_dir()
    assert (expected / "hub").is_dir()
    assert (expected / "xet").is_dir()
    assert (expected / "xet" / "logs").is_dir()
    assert (expected / "xet" / "chunk-cache").is_dir()
    assert (expected / "xet" / "shard-cache").is_dir()
    assert expected == Path(os.environ["HF_HOME"])
    assert "HF_HUB_DISABLE_XET" not in os.environ


def test_configure_hf_runtime_preserves_explicit_xet_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VOX_HOME", str(tmp_path / ".vox"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")

    configure_hf_runtime()

    assert Path(tmp_path / ".vox" / "cache" / "huggingface" / "xet").is_dir()
    assert Path(tmp_path / ".vox" / "cache" / "huggingface" / "xet" / "logs").is_dir()
    assert os.environ["HF_HUB_DISABLE_XET"] == "0"
