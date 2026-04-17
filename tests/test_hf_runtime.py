from __future__ import annotations

import os
from pathlib import Path

from vox.core.hf_runtime import configure_hf_runtime


def test_configure_hf_runtime_sets_safe_defaults(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)

    configure_hf_runtime()

    assert Path(tmp_path / ".cache" / "huggingface").is_dir()
    assert Path(tmp_path / ".cache" / "huggingface" / "hub").is_dir()
    assert Path(tmp_path / ".cache" / "huggingface" / "xet").is_dir()
    assert Path(tmp_path / ".cache" / "huggingface" / "xet" / "logs").is_dir()
    assert Path(tmp_path / ".cache" / "huggingface" / "xet" / "chunk-cache").is_dir()
    assert Path(tmp_path / ".cache" / "huggingface" / "xet" / "shard-cache").is_dir()
    assert Path(tmp_path / ".cache" / "huggingface") == Path(os.environ["HF_HOME"])
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"


def test_configure_hf_runtime_preserves_explicit_xet_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("HF_XET_CACHE", raising=False)
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")

    configure_hf_runtime()

    assert Path(tmp_path / ".cache" / "huggingface" / "xet").is_dir()
    assert Path(tmp_path / ".cache" / "huggingface" / "xet" / "logs").is_dir()
    assert os.environ["HF_HUB_DISABLE_XET"] == "0"
