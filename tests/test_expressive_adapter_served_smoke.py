from __future__ import annotations

import importlib.util
import io
import json
import sys
import time
import wave
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


def _wav_bytes() -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24_000)
        wav.writeframes((1000).to_bytes(2, byteorder="little", signed=True) * 2400)
    return out.getvalue()


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
    assert evidence["evidence_schema_version"] == 1
    assert evidence["inspect_only"] is True
    assert evidence["clean_pull_proof"] is False
    assert evidence["clean_pull_blockers"] == [
        "existing-server smoke cannot prove a clean model pull or clean adapter runtime install",
    ]
    assert evidence["synthesis"] == []
    assert evidence["synthesis_skipped"] == "inspect_only"
    assert evidence["failure_class"] == "none"
    assert evidence["failure_reasons"] == []
    assert "loaded_after" not in evidence
    assert "memory_after" not in evidence
    assert "memory_before" in evidence
    assert all("/v1/audio/speech" not in url for url in requested_urls)
    assert requested_urls == [
        "http://vox.local/v1/health",
        "http://vox.local/v1/models",
        "http://vox.local/v1/models/dia-tts%3A1.6b",
        "http://vox.local/v1/models/loaded",
        "http://vox.local/v1/system/memory",
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
            "--failure-class",
            "Vox",
            "--failure-note",
            "model detail endpoint returned 404",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["model_detail"]["status"] == 404
    assert evidence["memory_before"]["status"] == 200
    assert evidence["synthesis"] == []
    assert evidence["failure_reasons"] == ["model_detail returned HTTP 404"]
    assert evidence["failure_class"] == "Vox"
    assert evidence["failure_note"] == "model detail endpoint returned 404"


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
    assert evidence["clean_pull_proof"] is False
    assert evidence["clean_pull_blockers"] == [
        "existing-server smoke cannot prove a clean model pull or clean adapter runtime install",
    ]
    assert case_names == ["short", "long"]
    assert [case["name"] for case in evidence["synthesis"]] == ["short", "long"]
    assert "loaded_after" in evidence
    assert "memory_before" in evidence
    assert "memory_after" in evidence
    assert evidence["failure_class"] == "none"
    assert evidence["failure_note"] == ""
    assert evidence["failure_reasons"] == []
    assert requested_urls.count("http://vox.local/v1/models/loaded") == 2
    assert requested_urls.count("http://vox.local/v1/system/memory") == 2


def test_run_case_records_memory_samples_during_synthesis(tmp_path, monkeypatch):
    module = _load_served_smoke_module()
    requested_urls: list[str] = []

    def fake_request_json(url, *, timeout, api_key):
        requested_urls.append(url)
        return module.HttpResult(
            status=200,
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "ram": {"used_bytes": 1024},
                    "gpu": [{"memory_used_mib": 256}],
                }
            ).encode("utf-8"),
            elapsed_s=0.01,
        )

    def fake_post_json(url, payload, *, timeout, api_key):
        time.sleep(0.03)
        return module.HttpResult(
            status=200,
            headers={"Content-Type": "audio/wav"},
            body=_wav_bytes(),
            elapsed_s=0.03,
        )

    monkeypatch.setattr(module, "_request_json", fake_request_json)
    monkeypatch.setattr(module, "_post_json", fake_post_json)

    evidence = module._run_case(
        name="short",
        base_url="http://vox.local",
        model="dia-tts:1.6b",
        text="hello",
        voice=None,
        response_format="wav",
        speed=1.0,
        params={},
        timeout=5.0,
        api_key=None,
        output_dir=tmp_path,
        memory_sample_interval=0.01,
    )

    assert requested_urls
    assert all(url == "http://vox.local/v1/system/memory" for url in requested_urls)
    assert evidence.memory_samples is not None
    assert evidence.memory_samples["interval_s"] == 0.01
    assert evidence.memory_samples["count"] >= 1
    assert evidence.memory_samples["peak_ram_used_bytes"] == 1024
    assert evidence.memory_samples["peak_gpu_memory_used_mib"] == 256
    assert evidence.audio.duration_s is not None
    assert evidence.audio.duration_s > 0


def test_run_case_can_disable_memory_sampling(tmp_path, monkeypatch):
    module = _load_served_smoke_module()

    def fail_request_json(url, *, timeout, api_key):
        raise AssertionError("memory sampling should be disabled")

    def fake_post_json(url, payload, *, timeout, api_key):
        return module.HttpResult(
            status=200,
            headers={"Content-Type": "audio/wav"},
            body=_wav_bytes(),
            elapsed_s=0.01,
        )

    monkeypatch.setattr(module, "_request_json", fail_request_json)
    monkeypatch.setattr(module, "_post_json", fake_post_json)

    evidence = module._run_case(
        name="short",
        base_url="http://vox.local",
        model="dia-tts:1.6b",
        text="hello",
        voice=None,
        response_format="wav",
        speed=1.0,
        params={},
        timeout=5.0,
        api_key=None,
        output_dir=tmp_path,
        memory_sample_interval=0,
    )

    assert evidence.memory_samples == {
        "interval_s": 0,
        "count": 0,
        "samples": [],
        "peak_ram_used_bytes": None,
        "peak_gpu_memory_used_mib": None,
    }
    assert evidence.audio.duration_s is not None
    assert evidence.audio.duration_s > 0


def test_served_smoke_full_mode_requires_manual_audio_usable_verdict(tmp_path, monkeypatch):
    module = _load_served_smoke_module()

    def fake_request_json(url, *, timeout, api_key):
        return _http_result(module, url)

    def fake_run_case(**kwargs):
        return module.SynthesisEvidence(
            name=kwargs["name"],
            text_chars=len(kwargs["text"]),
            status=200,
            elapsed_s=0.2,
            content_type="audio/wav",
            output_path=str(tmp_path / f"{kwargs['name']}.wav"),
            audio=module.AudioStats(
                bytes=128,
                sha256="1" * 64,
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
            "--output-dir",
            str(tmp_path),
            "--failure-class",
            "adapter",
            "--failure-note",
            "audio was not manually accepted",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["audio_usable"] == "unchecked"
    assert [case["audio"]["silent"] for case in evidence["synthesis"]] == [False, False]
    assert evidence["failure_reasons"] == ["manual audio usability verdict is unchecked"]
    assert evidence["failure_class"] == "adapter"
    assert evidence["failure_note"] == "audio was not manually accepted"


def test_served_smoke_full_mode_rejects_silent_audio_even_when_manually_accepted(tmp_path, monkeypatch):
    module = _load_served_smoke_module()

    def fake_request_json(url, *, timeout, api_key):
        return _http_result(module, url)

    def fake_run_case(**kwargs):
        return module.SynthesisEvidence(
            name=kwargs["name"],
            text_chars=len(kwargs["text"]),
            status=200,
            elapsed_s=0.2,
            content_type="audio/wav",
            output_path=str(tmp_path / f"{kwargs['name']}.wav"),
            audio=module.AudioStats(
                bytes=128,
                sha256="2" * 64,
                duration_s=0.5,
                sample_rate=24_000,
                channels=1,
                sample_width=2,
                peak=0.0,
                rms=0.0,
                silent=True,
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
            "--failure-class",
            "upstream",
            "--failure-note",
            "model returned silent audio",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["audio_usable"] == "yes"
    assert [case["audio"]["silent"] for case in evidence["synthesis"]] == [True, True]
    assert evidence["failure_reasons"] == [
        "short synthesis returned silent audio",
        "long synthesis returned silent audio",
    ]
    assert evidence["failure_class"] == "upstream"
    assert evidence["failure_note"] == "model returned silent audio"


def test_served_smoke_full_mode_rejects_long_audio_shorter_than_short_audio(tmp_path, monkeypatch):
    module = _load_served_smoke_module()

    def fake_request_json(url, *, timeout, api_key):
        return _http_result(module, url)

    def fake_run_case(**kwargs):
        duration = 0.2 if kwargs["name"] == "short" else 0.1
        return module.SynthesisEvidence(
            name=kwargs["name"],
            text_chars=len(kwargs["text"]),
            status=200,
            elapsed_s=0.2,
            content_type="audio/wav",
            output_path=str(tmp_path / f"{kwargs['name']}.wav"),
            audio=module.AudioStats(
                bytes=128,
                sha256="4" * 64,
                duration_s=duration,
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
            "--failure-class",
            "upstream",
            "--failure-note",
            "long synthesis was shorter than short synthesis",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["failure_reasons"] == [
        "long synthesis duration 0.100s is shorter than short synthesis duration 0.200s",
    ]
    assert evidence["failure_class"] == "upstream"
    assert evidence["failure_note"] == "long synthesis was shorter than short synthesis"


def test_served_smoke_full_mode_records_http_and_empty_audio_failures(tmp_path, monkeypatch):
    module = _load_served_smoke_module()

    def fake_request_json(url, *, timeout, api_key):
        status = 503 if url.endswith(("/v1/models/loaded", "/v1/system/memory")) else 200
        return _http_result(module, url, status=status)

    def fake_run_case(**kwargs):
        return module.SynthesisEvidence(
            name=kwargs["name"],
            text_chars=len(kwargs["text"]),
            status=500 if kwargs["name"] == "short" else 200,
            elapsed_s=0.2,
            content_type="application/json" if kwargs["name"] == "short" else "audio/wav",
            output_path=str(tmp_path / f"{kwargs['name']}.wav"),
            audio=module.AudioStats(
                bytes=0 if kwargs["name"] == "long" else 128,
                sha256="3" * 64,
                duration_s=None,
                sample_rate=None,
                channels=None,
                sample_width=None,
                peak=None,
                rms=None,
                silent=False,
            ),
            error="backend failed" if kwargs["name"] == "short" else None,
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
            "--failure-class",
            "dependency",
            "--failure-note",
            "backend returned HTTP 500 and empty long audio",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["failure_reasons"] == [
        "loaded_before returned HTTP 503",
        "memory_before returned HTTP 503",
        "loaded_after returned HTTP 503",
        "memory_after returned HTTP 503",
        "short synthesis returned HTTP 500",
        "short synthesis error: backend failed",
        "long synthesis returned empty audio",
    ]
    assert evidence["failure_class"] == "dependency"
    assert evidence["failure_note"] == "backend returned HTTP 500 and empty long audio"


def test_served_smoke_requires_failure_class_for_failing_evidence(tmp_path, monkeypatch):
    module = _load_served_smoke_module()

    def fake_request_json(url, *, timeout, api_key):
        return _http_result(module, url, status=404 if url.endswith("/v1/models/dia-tts%3A1.6b") else 200)

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
    assert evidence["failure_class"] == "none"
    assert evidence["failure_reasons"] == [
        "model_detail returned HTTP 404",
        "failing smoke run must set --failure-class to one of "
        "Vox, adapter, dependency, upstream, or hardware",
    ]


def test_served_smoke_rejects_passing_run_with_failure_class(tmp_path, monkeypatch):
    module = _load_served_smoke_module()

    def fake_request_json(url, *, timeout, api_key):
        return _http_result(module, url)

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
            "--failure-class",
            "hardware",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["failure_class"] == "hardware"
    assert evidence["failure_reasons"] == ["passing smoke run must use --failure-class none"]


def test_served_smoke_requires_failure_note_for_classified_failure(tmp_path, monkeypatch):
    module = _load_served_smoke_module()

    def fake_request_json(url, *, timeout, api_key):
        return _http_result(module, url, status=404 if url.endswith("/v1/models/dia-tts%3A1.6b") else 200)

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
            "--failure-class",
            "adapter",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["failure_class"] == "adapter"
    assert "classified failing smoke run must include --failure-note" in evidence["failure_reasons"]


def test_served_smoke_rejects_passing_run_with_failure_note(tmp_path, monkeypatch):
    module = _load_served_smoke_module()

    def fake_request_json(url, *, timeout, api_key):
        return _http_result(module, url)

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
            "--failure-note",
            "this should not be set on a pass",
        ],
    )

    assert module.main() == 1

    evidence = json.loads((tmp_path / "evidence.json").read_text(encoding="utf-8"))
    assert evidence["failure_class"] == "none"
    assert evidence["failure_note"] == "this should not be set on a pass"
    assert evidence["failure_reasons"] == ["passing smoke run must not set --failure-note"]
