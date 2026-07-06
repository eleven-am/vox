from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_served_smoke_module() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "expressive-adapter-served-smoke.py"
    spec = importlib.util.spec_from_file_location("expressive_adapter_served_smoke", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _http_result(module: ModuleType, url: str, *, status: int = 200):
    return module.HttpResult(
        status=status,
        headers={"Content-Type": "application/json"},
        body=json.dumps({"url": url}).encode("utf-8"),
        elapsed_s=0.01,
    )


def test_served_smoke_inspect_only_never_synthesizes(tmp_path, monkeypatch):
    module = _load_served_smoke_module()
    requested_urls: list[str] = []

    def fake_request_json(url, *, timeout, api_key):
        requested_urls.append(url)
        return _http_result(module, url)

    def fail_run_case(**kwargs):
        raise AssertionError("inspect-only must not call /v1/audio/speech")

    monkeypatch.setattr(module, "_request_json", fake_request_json)
    monkeypatch.setattr(module, "_run_case", fail_run_case)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-served-smoke.py",
            "--base-url",
            "http://vox.local",
            "--model",
            "dia-tts:1.6b",
            "--inspect-only",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert module.main() == 0

    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["inspect_only"] is True
    assert evidence["synthesis"] == []
    assert evidence["synthesis_skipped"] == "inspect_only"
    assert "loaded_after" not in evidence
    assert all("/v1/audio/speech" not in url for url in requested_urls)
    assert requested_urls == [
        "http://vox.local/v1/health",
        "http://vox.local/v1/models",
        "http://vox.local/v1/models/dia-tts%3A1.6b",
        "http://vox.local/v1/models/loaded",
    ]


def test_served_smoke_inspect_only_fails_on_read_endpoint_error(tmp_path, monkeypatch):
    module = _load_served_smoke_module()

    def fake_request_json(url, *, timeout, api_key):
        status = 404 if url.endswith("/v1/models/dia-tts%3A1.6b") else 200
        return _http_result(module, url, status=status)

    monkeypatch.setattr(module, "_request_json", fake_request_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-served-smoke.py",
            "--base-url",
            "http://vox.local",
            "--model",
            "dia-tts:1.6b",
            "--inspect-only",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["model_detail"]["status"] == 404
    assert evidence["synthesis"] == []


def test_served_smoke_full_mode_runs_two_synthesis_cases_and_loaded_after(tmp_path, monkeypatch):
    module = _load_served_smoke_module()
    requested_urls: list[str] = []
    case_names: list[str] = []

    def fake_request_json(url, *, timeout, api_key):
        requested_urls.append(url)
        return _http_result(module, url)

    def fake_run_case(**kwargs):
        case_names.append(kwargs["name"])
        return module.SynthesisEvidence(
            name=kwargs["name"],
            text_chars=len(kwargs["text"]),
            status=200,
            elapsed_s=0.2,
            content_type="audio/wav",
            output_path=str(tmp_path / f"{kwargs['name']}.wav"),
            audio=module.AudioStats(
                bytes=128,
                sha256="0" * 64,
                duration_s=0.5,
                sample_rate=24_000,
                channels=1,
                sample_width=2,
                peak=0.5,
                rms=0.2,
                silent=False,
            ),
        )

    monkeypatch.setattr(module, "_request_json", fake_request_json)
    monkeypatch.setattr(module, "_run_case", fake_run_case)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "expressive-adapter-served-smoke.py",
            "--base-url",
            "http://vox.local",
            "--model",
            "dia-tts:1.6b",
            "--audio-usable",
            "yes",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert module.main() == 0

    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["inspect_only"] is False
    assert case_names == ["short", "long"]
    assert [case["name"] for case in evidence["synthesis"]] == ["short", "long"]
    assert "loaded_after" in evidence
    assert requested_urls.count("http://vox.local/v1/models/loaded") == 2
