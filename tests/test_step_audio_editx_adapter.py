from __future__ import annotations

import asyncio
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf
import vox_step_audio_editx.runtime as runtime_module
from vox_step_audio_editx.adapter import StepAudioEditXAdapter

from vox.core.errors import ModelLoadError
from vox.operations.errors import InvalidConfigError


def _model_dir(tmp_path: Path) -> Path:
    model = tmp_path / "model"
    (model / "CosyVoice-300M-25Hz").mkdir(parents=True)
    (model / "audio_tokenizer").mkdir()
    (model / "config.json").write_text("{}")
    (model / "audio_tokenizer" / "speech_tokenizer_v1.onnx").write_bytes(b"onnx")
    return model


def _land_source(path: Path) -> None:
    for relative in runtime_module.EXPECTED_SOURCE_PATHS:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    (path / ".vox-source-ref").write_text(f"{runtime_module.SOURCE_REF}\n")


class _FakeWorkerHost:
    instances: list[_FakeWorkerHost] = []

    def __init__(self, argv: list[str], *, env: dict[str, str], name: str, startup_timeout: float) -> None:
        self.argv = argv
        self.env = env
        self.name = name
        self.startup_timeout = startup_timeout
        self.requests: list[tuple[dict[str, Any], float]] = []
        self.closed = False
        self.samples = 12
        self.reported_samples: int | None = None
        type(self).instances.append(self)

    @property
    def alive(self) -> bool:
        return not self.closed

    def request(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        self.requests.append((dict(payload), timeout))
        reference, sample_rate = sf.read(payload["reference_path"], dtype="float32")
        assert len(reference) == 24
        assert sample_rate == 24_000
        sf.write(
            payload["output_path"],
            np.linspace(-0.25, 0.25, self.samples, dtype=np.float32),
            24_000,
            subtype="FLOAT",
        )
        return {
            "sample_rate": 24_000,
            "samples": self.reported_samples if self.reported_samples is not None else self.samples,
        }

    def close(self, grace: float = 5.0) -> None:
        self.closed = True


@pytest.fixture
def fake_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr("vox_step_audio_editx.adapter.ensure_runtime", lambda: runtime)
    monkeypatch.setattr("vox_step_audio_editx.adapter.worker_env", lambda path, device: {"DEVICE": device})
    monkeypatch.setattr("vox_step_audio_editx.adapter.WorkerHost", _FakeWorkerHost)
    _FakeWorkerHost.instances.clear()
    return runtime


def _loaded_adapter(tmp_path: Path, fake_runtime: Path) -> tuple[StepAudioEditXAdapter, _FakeWorkerHost]:
    adapter = StepAudioEditXAdapter()
    adapter.load(str(_model_dir(tmp_path)), "cuda")
    return adapter, _FakeWorkerHost.instances[-1]


def _collect(adapter: StepAudioEditXAdapter, **kwargs: Any):
    async def collect():
        return [chunk async for chunk in adapter.synthesize("Hello [Laughter]", **kwargs)]

    return asyncio.run(collect())


def test_package_import_does_not_import_heavy_runtime_modules():
    repo_root = Path(__file__).resolve().parents[1]
    package_src = repo_root / "adapters" / "vox-step-audio-editx" / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import vox_step_audio_editx as package; "
                "assert package.__all__ == ['StepAudioEditXAdapter']; "
                "assert not {'torch', 'vllm', 'model_loader', 'tokenizer', 'tts'} & set(sys.modules)"
            ),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(package_src), str(repo_root / "src"))),
        },
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_info_declares_clone_only_cuda_backend():
    info = StepAudioEditXAdapter().info()

    assert info.name == "step-audio-editx-tts-vllm"
    assert info.supports_voice_cloning is True
    assert info.supports_streaming is False
    assert info.supported_languages == ("en", "zh", "ja", "ko")
    assert info.default_sample_rate == 24_000


def test_load_rejects_cpu_without_installing_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ensure = MagicMock()
    monkeypatch.setattr("vox_step_audio_editx.adapter.ensure_runtime", ensure)

    with pytest.raises(ModelLoadError, match="requires a CUDA"):
        StepAudioEditXAdapter().load(str(tmp_path), "cpu")

    ensure.assert_not_called()


def test_load_rejects_incomplete_model_before_starting_worker(
    tmp_path: Path, fake_runtime: Path, monkeypatch: pytest.MonkeyPatch
):
    host = MagicMock()
    monkeypatch.setattr("vox_step_audio_editx.adapter.WorkerHost", host)

    with pytest.raises(ModelLoadError, match="artifacts are incomplete"):
        StepAudioEditXAdapter().load(str(tmp_path), "cuda")

    host.assert_not_called()


def test_load_starts_isolated_worker_and_unload_closes_it(tmp_path: Path, fake_runtime: Path):
    adapter, host = _loaded_adapter(tmp_path, fake_runtime)

    assert host.argv[:3] == [sys.executable, "-m", "vox_step_audio_editx.worker"]
    assert host.env == {"DEVICE": "cuda"}
    assert host.startup_timeout == 1800.0
    assert adapter.is_loaded is True

    adapter.unload()

    assert host.closed is True
    assert adapter.is_loaded is False


@pytest.mark.parametrize(
    ("reference_audio", "reference_text", "match"),
    [
        (None, "transcript", "requires reference_audio"),
        (np.array([], dtype=np.float32), "transcript", "requires reference_audio"),
        (np.ones(8, dtype=np.float32), None, "requires the reference transcript"),
        (np.ones(8, dtype=np.float32), " ", "requires the reference transcript"),
    ],
)
def test_clone_validation_rejects_missing_inputs(reference_audio, reference_text, match):
    with pytest.raises(InvalidConfigError, match=match):
        StepAudioEditXAdapter().validate_synthesis_request(
            reference_audio=reference_audio,
            reference_text=reference_text,
        )


def test_synthesize_forwards_real_parameters_and_returns_float_audio(tmp_path: Path, fake_runtime: Path):
    adapter, host = _loaded_adapter(tmp_path, fake_runtime)

    chunks = _collect(
        adapter,
        reference_audio=np.linspace(-0.5, 0.5, 24, dtype=np.float32),
        reference_text="This is the reference transcript.",
        language="en",
        params={"temperature": 0.45, "seed": 42},
    )

    payload, timeout = host.requests[0]
    assert payload["text"] == "Hello [Laughter]"
    assert payload["reference_text"] == "This is the reference transcript."
    assert payload["temperature"] == 0.45
    assert payload["seed"] == 42
    assert timeout == 1800.0
    assert not os.path.exists(payload["reference_path"])
    assert not os.path.exists(payload["output_path"])
    assert len(chunks) == 2
    assert np.frombuffer(chunks[0].audio, dtype=np.float32).size == host.samples
    assert chunks[0].sample_rate == 24_000
    assert chunks[-1].is_final is True


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"speed": 1.1}, "does not support speed"),
        ({"params": {"temperature": -0.1}}, "temperature"),
        ({"params": {"temperature": 2.1}}, "temperature"),
        ({"params": {"seed": True}}, "seed"),
        ({"params": {"seed": -1}}, "seed"),
    ],
)
def test_synthesize_rejects_unsupported_or_invalid_controls(
    tmp_path: Path, fake_runtime: Path, kwargs: dict[str, Any], match: str
):
    adapter, _host = _loaded_adapter(tmp_path, fake_runtime)

    with pytest.raises(InvalidConfigError, match=match):
        _collect(
            adapter,
            reference_audio=np.ones(24, dtype=np.float32),
            reference_text="Reference.",
            **kwargs,
        )


def test_synthesize_rejects_worker_metadata_mismatch(tmp_path: Path, fake_runtime: Path):
    adapter, host = _loaded_adapter(tmp_path, fake_runtime)
    host.reported_samples = host.samples + 1

    with pytest.raises(RuntimeError, match="inconsistent audio metadata"):
        _collect(
            adapter,
            reference_audio=np.ones(24, dtype=np.float32),
            reference_text="Reference.",
        )


def test_prepare_runtime_uses_locked_no_deps_install_and_pinned_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    runtime = tmp_path / "step-audio-editx"
    calls: list[dict[str, Any]] = []

    def install(target, requirements, **kwargs):
        calls.append({"target": target, "requirements": requirements, **kwargs})
        target.mkdir(parents=True, exist_ok=True)
        for relative in runtime_module.EXPECTED_RUNTIME_PATHS:
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        return True

    def extract(path: Path) -> None:
        _land_source(path)

    monkeypatch.setattr(runtime_module, "runtime_dir", lambda: runtime)
    monkeypatch.setattr(runtime_module, "install_target_runtime_requirements", install)
    monkeypatch.setattr(runtime_module, "_extract_source", extract)
    monkeypatch.setattr(runtime_module, "_probe_runtime", lambda path: True)

    result = runtime_module.ensure_runtime()

    assert result == runtime
    assert calls[0]["target"] != runtime
    assert calls[0]["target"].parent == runtime.parent
    assert calls[0]["requirements"] == runtime_module.RUNTIME_REQUIREMENTS
    assert calls[0]["no_deps"] is True
    assert calls[0]["upgrade"] is False
    assert calls[0]["installer_order"] == ("uv", "pip")
    names = {requirement.split("==", 1)[0].lower() for requirement in runtime_module.RUNTIME_REQUIREMENTS}
    assert "vllm" in names
    assert "torchvision" in names
    assert not names & {"torch", "torchaudio", "triton"}
    assert not any(name.startswith("nvidia-cuda-") for name in names)
    assert runtime_module.SOURCE_REF in runtime_module.SOURCE_URL
    assert (runtime / runtime_module.RUNTIME_SENTINEL).is_file()
    assert runtime_module._source_matches(runtime / "source")


def test_ready_runtime_is_verified_without_reinstall(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runtime = tmp_path / "step-audio-editx"
    runtime.mkdir()
    _land_source(runtime / "source")
    (runtime / runtime_module.RUNTIME_SENTINEL).touch()
    install = MagicMock()
    monkeypatch.setattr(runtime_module, "runtime_dir", lambda: runtime)
    monkeypatch.setattr(runtime_module, "_probe_runtime", lambda path: True)
    monkeypatch.setattr(runtime_module, "install_target_runtime_requirements", install)

    assert runtime_module.ensure_runtime() == runtime
    install.assert_not_called()


def test_source_extraction_ignores_links_and_rejects_escape_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    archive_path = tmp_path / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for relative in runtime_module.EXPECTED_SOURCE_PATHS:
            payload = b"source"
            member = tarfile.TarInfo(f"source/{relative}")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        link = tarfile.TarInfo("source/unused-link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/outside"
        archive.addfile(link)
        escape = tarfile.TarInfo("source/../escape")
        escape.size = 1
        archive.addfile(escape, io.BytesIO(b"x"))

    monkeypatch.setattr(
        runtime_module.urllib.request,
        "urlretrieve",
        lambda _url, target: Path(target).write_bytes(archive_path.read_bytes()),
    )

    with pytest.raises(RuntimeError, match="unsafe path"):
        runtime_module._extract_source(tmp_path / "target")

    assert not (tmp_path / "escape").exists()


def test_worker_generation_binds_seed_and_respects_runtime_context(monkeypatch: pytest.MonkeyPatch):
    from vox_step_audio_editx import worker

    captured: dict[str, Any] = {}

    class SamplingParams:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    fake_vllm = ModuleType("vllm")
    fake_vllm.SamplingParams = SamplingParams
    fake_torch = ModuleType("torch")
    fake_torch.long = "long"
    fake_torch.tensor = lambda values, dtype: ("tensor", values, dtype)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    engine = SimpleNamespace(
        llm=SimpleNamespace(
            generate=MagicMock(
                return_value=[SimpleNamespace(outputs=[SimpleNamespace(token_ids=[65536, 65537, 3])])]
            )
        )
    )

    result = worker._generate(engine, [1, 2, 3], 0.4, 77)

    assert captured == {
        "temperature": 0.4,
        "max_tokens": 3069,
        "skip_special_tokens": False,
        "seed": 77,
    }
    assert result == ("tensor", [65536, 65537], "long")


def test_worker_load_uses_checkpoint_quantization_metadata(monkeypatch: pytest.MonkeyPatch):
    from vox_step_audio_editx import worker

    captured: dict[str, Any] = {}

    class ModelSource:
        LOCAL = "local"

    class StepAudioTokenizer:
        def __init__(self, path: str, *, model_source: str) -> None:
            captured["tokenizer"] = (path, model_source)

    class StepAudioTTS:
        def __init__(self, model_path: str, tokenizer: Any, **kwargs: Any) -> None:
            captured["tts"] = (model_path, tokenizer, kwargs)
            tts.model_loader.load_model(model_path)

    model_loader = ModuleType("model_loader")
    model_loader.ModelSource = ModelSource
    tokenizer = ModuleType("tokenizer")
    tokenizer.StepAudioTokenizer = StepAudioTokenizer
    tts = ModuleType("tts")
    tts.StepAudioTTS = StepAudioTTS
    tts.model_loader = SimpleNamespace(
        load_model=lambda *args, **kwargs: captured.update({"load_model": (args, kwargs)})
    )
    monkeypatch.setitem(sys.modules, "model_loader", model_loader)
    monkeypatch.setitem(sys.modules, "tokenizer", tokenizer)
    monkeypatch.setitem(sys.modules, "tts", tts)

    engine = worker._load_engine("/models/step")

    assert isinstance(engine, StepAudioTTS)
    assert captured["tokenizer"] == ("/models/step/audio_tokenizer", "local")
    assert captured["tts"][2]["quantization"] is None
    assert captured["tts"][2]["max_model_len"] == 3072
    assert captured["tts"][2]["max_num_seqs"] == 1
    assert captured["tts"][2]["cosyvoice_cuda_graph"] is False
    assert captured["load_model"][1]["attention_config"] == {"backend": "TRITON_ATTN"}


def test_worker_generation_rejects_prompt_that_exhausts_context():
    from vox_step_audio_editx import worker

    with pytest.raises(RuntimeError, match="exceeds the 3072-token"):
        worker._generate(SimpleNamespace(), [1] * 3072, 0.7, None)
