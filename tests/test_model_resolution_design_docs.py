from __future__ import annotations

from pathlib import Path


def test_model_resolution_design_matches_current_variant_api():
    design = Path("docs/model-resolution-design.md").read_text(encoding="utf-8")

    assert "remote registry at" in design
    assert "single\n  source of truth" in design
    assert "CLI `--variant` flag" in design
    assert "HTTP/gRPC `variant` field" in design
    assert "vox pull kokoro-tts --variant torch" in design
    assert "CLI `--variant`, HTTP and gRPC\n   pass-through" in design

    assert "Grammar: `<name>[@<variant>][:<tag>]`" not in design
    assert "`@variant` / `--backend`" not in design
    assert "CLI `--backend` / `@variant`" not in design


def test_model_resolution_design_uses_public_qwen_model_name():
    design = Path("docs/model-resolution-design.md").read_text(encoding="utf-8")

    assert '"name": "qwen3-tts"' in design
    assert "qwen3-tts:0.6b" in design
    assert "qwen3-tts-torch:0.6b" not in design


def test_qwen_readme_uses_public_model_references():
    readme = Path("adapters/vox-qwen/README.md").read_text(encoding="utf-8")

    assert "vox pull qwen3-stt:0.6b" in readme
    assert "vox pull qwen3-tts:0.6b" in readme
    assert "vox pull qwen3-stt-torch:0.6b" not in readme
    assert "vox pull qwen3-tts-torch:0.6b" not in readme
