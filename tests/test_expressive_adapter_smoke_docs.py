from __future__ import annotations

from pathlib import Path


def test_expressive_adapter_smoke_runbook_keeps_production_safety_boundary():
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()

    assert "Do not use the production `vox` namespace or production `vox-data` PVC" in runbook
    assert "VOX_SMOKE_NS=vox-adapter-smoke" in runbook
    assert "VOX_SMOKE_PVC=vox-adapter-smoke-data" in runbook
    assert "separate namespace and disposable PVC" in runbook
    assert "Do not mutate, clean, reinstall, restart, or scale" in runbook


def test_expressive_adapter_smoke_runbook_lists_required_models_and_evidence():
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()

    for model in ("dia-tts:1.6b", "orpheus-tts:medium-3b", "indextts-tts:2"):
        assert model in runbook

    for evidence in (
        "`vox pull <model>` output",
        "short synthesis wall time",
        "long synthesis wall time",
        "generated audio duration",
        "peak GPU memory",
        "output WAV artifact",
        "audio is usable",
    ):
        assert evidence in runbook

