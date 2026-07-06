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
        "vox-cosyvoice==0.1.5",
        "vox-dia==0.2.12",
        "vox-orpheus==0.1.6",
        "vox-indextts==0.1.6",
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


def test_expressive_adapter_status_names_local_regression_evidence():
    status = Path("docs/expressive-adapter-status.md").read_text()

    for evidence in (
        "`tests/test_cosyvoice_adapter.py`; the test proves `prepare_runtime()` can",
        "bootstrap the isolated runtime without loading model weights",
        "`tests/test_dia_adapter.py`; the test\n  proves the isolated Transformers runtime",
        "without loading\n  processors or model weights",
        "rejects\n  Dia-capable Transformers modules loaded from the Vox app environment",
        "Pull atomicity across adapter runtime preparation is covered by",
        "Vox does not save a model\n  manifest when `prepare_runtime()` fails",
        "`tests/test_orpheus_adapter.py`",
        "a stale `orpheus_tts` module missing `OrpheusModel` and a\n  broken runtime import probe are repaired",
        "rejects `orpheus_tts` modules loaded from\n  outside `$VOX_HOME/runtime/orpheus`",
        "`tests/test_indextts_adapter.py`",
        "a stale `indextts.infer_v2` module missing `IndexTTS2` and a\n  broken runtime import probe are repaired",
        "rejects `indextts.infer_v2` modules loaded\n  from outside `$VOX_HOME/runtime/indextts`",
    ):
        assert evidence in status


def test_expressive_adapter_status_records_dia_budget_finding():
    status = Path("docs/expressive-adapter-status.md").read_text()

    for evidence in (
        "Current Dia Cluster Finding",
        "`ghcr.io/eleven-am/vox:v0.2.86`",
        "`vox-dia==0.2.11`",
        "`/home/vox/.vox/runtime/dia` only contained",
        "`--max-vram 10GiB --vram-headroom 1GiB`",
        "not a successful smoke test",
        "fresh pull and synthesis in the disposable smoke namespace",
    ):
        assert evidence in status


def test_expressive_adapter_smoke_runbook_requires_durable_evidence_record():
    runbook = Path("docs/expressive-adapter-smoke.md").read_text()

    assert "Store one evidence file per model next to the copied WAV artifacts" in runbook
    assert "Do not keep\nthe only record in chat logs or terminal scrollback" in runbook
    assert "${MODEL//[:\\/]/-}-evidence.md" in runbook

    for field in (
        "Image digest:",
        "Adapter package:",
        "Runtime capability snapshot:",
        "Used VOX_ALLOW_INCOMPATIBLE: no",
        "Runtime directory:",
        "Model store path:",
        "Peak pod memory:",
        "Peak GPU memory:",
        "Failure class: Vox / adapter / dependency / upstream / hardware / none",
        "Exact error:",
    ):
        assert field in runbook
