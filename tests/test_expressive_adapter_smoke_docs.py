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

    for model in ("cosyvoice2-tts:0.5b", "dia-tts:1.6b", "orpheus-tts:medium-3b", "indextts-tts:2"):
        assert model in runbook

    for evidence in (
        "image tag and digest",
        "adapter package version resolved from PyPI",
        "registry entry used",
        "runtime capability snapshot from the pod",
        "`vox pull <model>` output",
        "short synthesis wall time",
        "long synthesis wall time",
        "generated audio duration",
        "peak pod memory",
        "peak GPU memory",
        "output WAV artifact",
        "audio is usable",
        "exact failure output",
    ):
        assert evidence in runbook


def test_expressive_adapter_smoke_runbook_pins_published_adapter_baseline():
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()

    for package in (
        "vox-cosyvoice==0.1.3",
        "vox-dia==0.2.11",
        "vox-orpheus==0.1.3",
        "vox-indextts==0.1.3",
    ):
        assert package in runbook

    assert "resolve adapter packages from PyPI" in runbook
    assert "not from a\nlocal source tree or a patched live cluster directory" in runbook


def test_expressive_adapter_smoke_runbook_preserves_runtime_and_artifact_boundaries():
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()

    for requirement in (
        "Runtime dependencies are installed under `$VOX_HOME/runtime/<adapter>`",
        "Model files are stored in the model store, not in the adapter package or base image",
        "reference WAV copied into the disposable\nPVC or mounted as test-only data",
        "The adapter is expected to reject\nrequests without reference audio or a voice-path prompt",
        "Do not delete or modify production voice data under `$VOX_HOME/voices`",
        "Failures, if any, are classified as Vox, adapter, dependency, upstream, or hardware",
    ):
        assert requirement in runbook


def test_expressive_adapter_status_tracks_all_goal_targets_and_smoke_gap():
    status = Path("docs/expressive-adapter-status.md").read_text()

    for model in (
        "cosyvoice2-tts:0.5b",
        "dia-tts:1.6b",
        "orpheus-tts:medium-3b",
        "indextts-tts:2",
    ):
        assert model in status

    assert "Previously cluster-smoked successfully, but slow" in status
    assert "Pending isolated GPU smoke" in status
    assert "Do not run these against the production `vox` namespace or `vox-data` PVC" in status
    assert "vox pull` succeeds without `VOX_ALLOW_INCOMPATIBLE" in status
