from __future__ import annotations

import subprocess
import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

from vox.core import adapter_runtime


def test_target_runtime_uses_vox_home(monkeypatch, tmp_path):
    monkeypatch.setenv("VOX_HOME", str(tmp_path / "vox-home"))

    runtime = adapter_runtime.target_runtime("qwen-tts")

    assert runtime.name == "qwen-tts"
    assert runtime.path == tmp_path / "vox-home" / "runtime" / "qwen-tts"


def test_activate_runtime_path_prunes_sibling_runtime_paths(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtime"
    current = runtime_root / "qwen-tts"
    sibling = runtime_root / "kokoro-torch"
    external = tmp_path / "other"
    original_path = list(sys.path)
    monkeypatch.setattr(sys, "path", [str(sibling), str(external), *original_path])

    activated = adapter_runtime.activate_runtime_path(current, root=runtime_root)

    assert activated == str(current)
    assert sys.path[0] == str(current)
    assert str(sibling) not in sys.path
    assert str(external) in sys.path


def test_purge_runtime_modules_removes_configured_prefixes(monkeypatch):
    qwen_root = ModuleType("qwen_tts")
    qwen_child = ModuleType("qwen_tts.model")
    other = ModuleType("other_runtime")
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_root)
    monkeypatch.setitem(sys.modules, "qwen_tts.model", qwen_child)
    monkeypatch.setitem(sys.modules, "other_runtime", other)

    adapter_runtime.purge_runtime_modules(["qwen_tts"])

    assert "qwen_tts" not in sys.modules
    assert "qwen_tts.model" not in sys.modules
    assert sys.modules["other_runtime"] is other


def test_module_available_fails_closed_when_spec_probe_raises_import_error(monkeypatch):
    monkeypatch.setattr(adapter_runtime, "find_spec", lambda _name: (_ for _ in ()).throw(ImportError("broken")))

    assert adapter_runtime.module_available("broken_runtime") is False


def test_module_available_fails_closed_when_spec_probe_raises_value_error(monkeypatch):
    monkeypatch.setattr(adapter_runtime, "find_spec", lambda _name: (_ for _ in ()).throw(ValueError("bad spec")))

    assert adapter_runtime.module_available("broken_runtime") is False


def test_module_available_in_runtime_rejects_app_env_module_without_explicit_fallback(
    monkeypatch,
    tmp_path,
):
    runtime_path = tmp_path / "runtime" / "strict"
    app_path = tmp_path / "app" / "site-packages" / "heavy_runtime" / "__init__.py"
    spec = ModuleSpec("heavy_runtime", loader=None, origin=str(app_path))
    monkeypatch.setattr(adapter_runtime, "find_spec", lambda _name: spec)

    assert not adapter_runtime.module_available_in_runtime(
        "heavy_runtime",
        runtime_path,
        include_app_fallback=False,
    )
    assert adapter_runtime.module_available_in_runtime(
        "heavy_runtime",
        runtime_path,
        include_app_fallback=True,
    )


def test_module_available_in_runtime_accepts_target_runtime_package(monkeypatch, tmp_path):
    runtime_path = tmp_path / "runtime" / "strict"
    package_path = runtime_path / "heavy_runtime" / "__init__.py"
    spec = ModuleSpec("heavy_runtime", loader=None, origin=str(package_path))
    monkeypatch.setattr(adapter_runtime, "find_spec", lambda _name: spec)

    assert adapter_runtime.module_available_in_runtime(
        "heavy_runtime",
        runtime_path,
        include_app_fallback=False,
    )


def test_ensure_target_runtime_does_not_accept_app_env_module_as_strict_runtime(
    monkeypatch,
    tmp_path,
):
    calls: list[list[str]] = []
    runtime_root = tmp_path / "runtime"
    app_path = tmp_path / "app" / "site-packages" / "strict_runtime" / "__init__.py"

    def find_spec(import_name: str):
        assert import_name == "strict_runtime"
        if not calls:
            return ModuleSpec(import_name, loader=None, origin=str(app_path))
        target_path = runtime_root / "strict-runtime" / "strict_runtime" / "__init__.py"
        return ModuleSpec(import_name, loader=None, origin=str(target_path))

    def runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(adapter_runtime, "find_spec", find_spec)

    runtime_path = adapter_runtime.ensure_target_runtime(
        "strict-runtime",
        "strict-runtime==1.0.0",
        "strict_runtime",
        root=runtime_root,
        install_runner=runner,
        include_app_fallback=False,
    )

    assert runtime_path == runtime_root / "strict-runtime"
    assert calls
    assert not (runtime_path / "_vox_runtime_fallback_paths.pth").exists()


def test_ensure_target_runtime_prefers_uv_and_writes_app_fallback(tmp_path):
    calls: list[list[str]] = []

    def runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    probe_calls = 0

    def probe(import_name: str) -> bool:
        nonlocal probe_calls
        assert import_name == "qwen_tts"
        probe_calls += 1
        return probe_calls > 1

    runtime_path = adapter_runtime.ensure_target_runtime(
        "qwen-tts",
        "qwen-tts==1.0.0",
        "qwen_tts",
        include_app_fallback=True,
        extra_packages=["flash-attn==2.8.0"],
        no_deps=True,
        root=tmp_path / "runtime",
        install_runner=runner,
        module_probe=probe,
    )

    assert runtime_path == tmp_path / "runtime" / "qwen-tts"
    assert calls == [
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            str(runtime_path),
            "--upgrade",
            "--no-deps",
            "qwen-tts==1.0.0",
            "flash-attn==2.8.0",
        ]
    ]
    assert (runtime_path / "_vox_runtime_fallback_paths.pth").read_text(encoding="utf-8").strip()


def test_ensure_target_runtime_can_disable_app_fallback(tmp_path):
    probe_calls = 0

    def probe(import_name: str) -> bool:
        nonlocal probe_calls
        assert import_name == "strict_runtime"
        probe_calls += 1
        return probe_calls > 1

    runtime_path = adapter_runtime.ensure_target_runtime(
        "strict-runtime",
        "strict-runtime==1.0.0",
        "strict_runtime",
        root=tmp_path / "runtime",
        install_runner=lambda cmd, _timeout: subprocess.CompletedProcess(cmd, 0, "", ""),
        module_probe=probe,
        include_app_fallback=False,
    )

    assert runtime_path == tmp_path / "runtime" / "strict-runtime"
    assert not (runtime_path / "_vox_runtime_fallback_paths.pth").exists()


def test_ensure_target_runtime_falls_back_to_python_pip_when_uv_is_missing(tmp_path):
    calls: list[list[str]] = []

    def runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        if cmd[:2] == ["uv", "pip"]:
            raise FileNotFoundError("uv")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    probe_calls = 0

    def probe(import_name: str) -> bool:
        nonlocal probe_calls
        assert import_name == "parakeet"
        probe_calls += 1
        return probe_calls > 1

    runtime_path = adapter_runtime.ensure_target_runtime(
        "parakeet",
        "nemo-toolkit==2.4.0",
        "parakeet",
        include_app_fallback=True,
        root=tmp_path / "runtime",
        install_runner=runner,
        module_probe=probe,
    )

    assert runtime_path == tmp_path / "runtime" / "parakeet"
    assert calls[0][:2] == ["uv", "pip"]
    assert calls[1][:3] == [sys.executable, "-m", "pip"]
    assert calls[1][-1] == "nemo-toolkit==2.4.0"


def test_ensure_target_runtime_raises_when_installers_cannot_load_module(tmp_path):
    def runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, "", "")

    with pytest.raises(RuntimeError, match="runtime package is missing"):
        adapter_runtime.ensure_target_runtime(
            "broken",
            "broken-runtime",
            "broken_runtime",
            include_app_fallback=True,
            root=tmp_path / "runtime",
            install_runner=runner,
            module_probe=lambda _import_name: False,
        )


def test_install_target_runtime_requirements_can_disable_upgrade(tmp_path):
    calls: list[list[str]] = []

    def runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    assert adapter_runtime.install_target_runtime_requirements(
        tmp_path / "runtime",
        ["nemo-toolkit[asr]"],
        upgrade=False,
        install_runner=runner,
    )

    assert "--upgrade" not in calls[0]
    assert calls[0][-1] == "nemo-toolkit[asr]"


def test_install_target_runtime_requirements_rejects_success_without_expected_paths(tmp_path):
    calls: list[list[str]] = []
    missing_package = tmp_path / "runtime" / "transformers"

    def runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    assert not adapter_runtime.install_target_runtime_requirements(
        tmp_path / "runtime",
        ["transformers==4.57.6"],
        expected_paths=[missing_package],
        install_runner=runner,
    )

    assert calls[0][:2] == ["uv", "pip"]
    assert calls[1][:3] == [sys.executable, "-m", "pip"]


def test_install_target_runtime_requirements_accepts_success_with_expected_paths(tmp_path):
    calls: list[list[str]] = []
    installed_package = tmp_path / "runtime" / "transformers"

    def runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        installed_package.mkdir(parents=True)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    assert adapter_runtime.install_target_runtime_requirements(
        tmp_path / "runtime",
        ["transformers==4.57.6"],
        expected_paths=[installed_package],
        install_runner=runner,
    )

    assert calls[0][:2] == ["uv", "pip"]
    assert len(calls) == 1


def test_install_target_runtime_requirements_includes_extra_install_args(tmp_path):
    calls: list[list[str]] = []

    def runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    assert adapter_runtime.install_target_runtime_requirements(
        tmp_path / "runtime",
        ["git+https://example.invalid/runtime.git"],
        no_deps=True,
        upgrade=False,
        extra_install_args=["--no-build-isolation"],
        install_runner=runner,
    )

    assert "--no-build-isolation" in calls[0]
    assert "--no-deps" in calls[0]
    assert "--upgrade" not in calls[0]
