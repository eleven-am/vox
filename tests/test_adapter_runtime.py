from __future__ import annotations

import asyncio
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import pytest

from vox.core import adapter_runtime, atomic_install


def test_adapter_runtime_lock_serializes_dependency_graph_mutations():
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()

    def first_operation() -> None:
        first_entered.set()
        assert release_first.wait(timeout=2)

    def second_operation() -> None:
        second_entered.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(adapter_runtime.run_with_adapter_runtime_lock, first_operation)
        assert first_entered.wait(timeout=2)
        second = executor.submit(adapter_runtime.run_with_adapter_runtime_lock, second_operation)
        assert not second_entered.wait(timeout=0.05)
        release_first.set()
        first.result(timeout=2)
        second.result(timeout=2)

    assert second_entered.is_set()


def test_staged_runtime_mutation_blocks_competing_mutation_until_commit(tmp_path):
    runtime = tmp_path / "runtime" / "stable"
    runtime.mkdir(parents=True)
    (runtime / "stable.txt").write_text("stable")
    competing_entered = threading.Event()

    def staged_operation() -> None:
        with adapter_runtime.staged_target_runtime(runtime) as stage:
            (stage / "staged.txt").write_text("staged")

    def competing_operation() -> None:
        competing_entered.set()
        with adapter_runtime.staged_target_runtime(runtime) as stage:
            (stage / "competing.txt").write_text("competing")

    mutation = adapter_runtime.stage_adapter_runtime_mutation(staged_operation)
    with ThreadPoolExecutor(max_workers=1) as executor:
        competing = executor.submit(
            adapter_runtime.run_with_adapter_runtime_lock,
            competing_operation,
        )
        assert not competing_entered.wait(timeout=0.05)
        mutation.commit()
        competing.result(timeout=2)

    assert competing_entered.is_set()
    assert {path.name for path in runtime.iterdir()} == {"competing.txt"}


def test_adapter_runtime_lock_rolls_back_published_runtime_when_operation_fails(tmp_path):
    runtime = tmp_path / "runtime" / "stable"
    runtime.mkdir(parents=True)
    (runtime / "stable.txt").write_bytes(b"stable")

    def runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        target = Path(cmd[cmd.index("--target") + 1])
        (target / "replacement.txt").write_bytes(b"replacement")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def operation() -> None:
        assert adapter_runtime.install_target_runtime_requirements(
            runtime,
            ["replacement==2.0"],
            expected_paths=[runtime / "replacement.txt"],
            install_runner=runner,
        )
        raise RuntimeError("later runtime preparation failed")

    with pytest.raises(RuntimeError, match="later runtime preparation failed"):
        adapter_runtime.run_with_adapter_runtime_lock(operation)

    assert {path.relative_to(runtime): path.read_bytes() for path in runtime.rglob("*") if path.is_file()} == {
        Path("stable.txt"): b"stable"
    }


def test_runtime_rollback_restores_previous_directory_when_replacement_cleanup_is_busy(
    tmp_path,
    monkeypatch,
):
    runtime = tmp_path / "runtime" / "stable"
    runtime.mkdir(parents=True)
    (runtime / "stable.txt").write_text("stable")

    def operation() -> None:
        with adapter_runtime.staged_target_runtime(runtime) as stage:
            (stage / "replacement.txt").write_text("replacement")
        raise RuntimeError("later preparation failed")

    original_rmtree = atomic_install.shutil.rmtree

    def busy_replacement(path, *args, **kwargs):
        if Path(path) == runtime:
            raise OSError(16, "Device or resource busy")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(atomic_install.shutil, "rmtree", busy_replacement)

    with pytest.raises(RuntimeError, match="later preparation failed"):
        adapter_runtime.run_with_adapter_runtime_lock(operation)

    assert (runtime / "stable.txt").read_text() == "stable"
    assert not (runtime / "replacement.txt").exists()


def test_prune_stale_install_directories_removes_only_owned_artifacts(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    stale = [
        root / ".parakeet.installing-abc",
        root / ".parakeet.previous-def",
        root / ".parakeet.failed-ghi",
    ]
    for path in stale:
        path.mkdir()
        (path / "artifact.txt").write_text("stale")
    active = root / "parakeet"
    active.mkdir()
    unrelated = root / ".cache"
    unrelated.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink = root / ".parakeet.previous-link"
    symlink.symlink_to(outside, target_is_directory=True)

    assert atomic_install.prune_stale_install_directories(root) == 3

    assert all(not path.exists() for path in stale)
    assert active.is_dir()
    assert unrelated.is_dir()
    assert symlink.is_symlink()
    assert outside.is_dir()


def test_prune_restores_uncommitted_previous_over_replacement(tmp_path):
    root = tmp_path / "runtime"
    target = root / "parakeet"
    previous = root / ".parakeet.previous-transaction"
    target.mkdir(parents=True)
    previous.mkdir()
    (target / "replacement.txt").write_text("replacement")
    (previous / "stable.txt").write_text("stable")

    assert atomic_install.prune_stale_install_directories(root) == 1

    assert (target / "stable.txt").read_text() == "stable"
    assert not (target / "replacement.txt").exists()
    assert not previous.exists()


def test_prune_restores_uncommitted_previous_when_target_is_missing(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    target = root / "parakeet"
    previous = root / ".parakeet.previous-transaction"
    previous.mkdir()
    (previous / "stable.txt").write_text("stable")

    assert atomic_install.prune_stale_install_directories(root) == 1

    assert (target / "stable.txt").read_text() == "stable"
    assert not previous.exists()


def test_prune_removes_committed_backup_without_replacing_target(tmp_path):
    root = tmp_path / "runtime"
    target = root / "parakeet"
    committed = root / ".parakeet.committed-transaction"
    target.mkdir(parents=True)
    committed.mkdir()
    (target / "replacement.txt").write_text("replacement")
    (committed / "stable.txt").write_text("stable")

    assert atomic_install.prune_stale_install_directories(root) == 1

    assert (target / "replacement.txt").read_text() == "replacement"
    assert not (target / "stable.txt").exists()
    assert not committed.exists()


def test_adapter_runtime_lock_rolls_back_removals_and_multiple_publications(tmp_path):
    runtime = tmp_path / "runtime" / "stable"
    runtime.mkdir(parents=True)
    (runtime / "keep.txt").write_bytes(b"keep")
    (runtime / "remove.txt").write_bytes(b"remove")

    install_count = 0

    def runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        nonlocal install_count
        install_count += 1
        target = Path(cmd[cmd.index("--target") + 1])
        (target / f"installed-{install_count}.txt").write_bytes(str(install_count).encode())
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def operation() -> None:
        adapter_runtime.remove_target_runtime_paths(
            runtime,
            [runtime / "remove.txt"],
        )
        assert adapter_runtime.install_target_runtime_requirements(
            runtime,
            ["first==1.0"],
            expected_paths=[runtime / "installed-1.txt"],
            install_runner=runner,
        )
        assert adapter_runtime.install_target_runtime_requirements(
            runtime,
            ["second==1.0"],
            expected_paths=[runtime / "installed-2.txt"],
            install_runner=runner,
        )
        raise RuntimeError("later runtime preparation failed")

    with pytest.raises(RuntimeError, match="later runtime preparation failed"):
        adapter_runtime.run_with_adapter_runtime_lock(operation)

    assert {path.relative_to(runtime): path.read_bytes() for path in runtime.rglob("*") if path.is_file()} == {
        Path("keep.txt"): b"keep",
        Path("remove.txt"): b"remove",
    }


@pytest.mark.asyncio
async def test_bound_runtime_mutation_allows_same_transaction_reentry():
    mutation = adapter_runtime.stage_adapter_runtime_mutation(lambda: None)
    try:
        with adapter_runtime.bind_runtime_mutation(mutation):
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    adapter_runtime.run_with_adapter_runtime_lock,
                    lambda: "ready",
                ),
                timeout=0.2,
            )
    finally:
        mutation.rollback()

    assert result == "ready"


def test_staged_target_runtime_failure_preserves_previous_runtime(tmp_path):
    runtime = tmp_path / "runtime" / "stable"
    runtime.mkdir(parents=True)
    (runtime / "stable.txt").write_bytes(b"stable")

    with pytest.raises(RuntimeError, match="build failed"), adapter_runtime.staged_target_runtime(runtime) as stage:
        (stage / "partial.txt").write_bytes(b"partial")
        raise RuntimeError("build failed")

    assert {path.relative_to(runtime): path.read_bytes() for path in runtime.rglob("*") if path.is_file()} == {
        Path("stable.txt"): b"stable"
    }


def test_staged_target_runtime_publishes_complete_replacement(tmp_path):
    runtime = tmp_path / "runtime" / "stable"
    runtime.mkdir(parents=True)
    (runtime / "stable.txt").write_bytes(b"stable")

    with adapter_runtime.staged_target_runtime(runtime) as stage:
        (stage / "replacement.txt").write_bytes(b"replacement")

    assert {path.relative_to(runtime): path.read_bytes() for path in runtime.rglob("*") if path.is_file()} == {
        Path("replacement.txt"): b"replacement"
    }


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


def test_activate_runtime_path_preserves_sibling_with_loaded_native_extension(monkeypatch, tmp_path):
    runtime_root = tmp_path / "runtime"
    current = runtime_root / "parakeet-nemo"
    sibling = runtime_root / "kokoro"
    native_path = sibling / "spacy" / "tokens" / "doc.cpython-312-x86_64-linux-gnu.so"
    native_path.parent.mkdir(parents=True)
    native_path.touch()
    external = tmp_path / "other"

    native_module = ModuleType("spacy.tokens.doc")
    native_module.__file__ = str(native_path)
    monkeypatch.setitem(sys.modules, "spacy.tokens.doc", native_module)
    monkeypatch.setattr(sys, "path", [str(sibling), str(external), *sys.path])

    adapter_runtime.activate_runtime_path(current, root=runtime_root)

    assert sys.path[0] == str(current)
    assert str(sibling) in sys.path
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
    install_target = calls[0][calls[0].index("--target") + 1]
    assert Path(install_target).parent == runtime_path.parent
    assert Path(install_target).name.startswith(".qwen-tts.installing-")
    assert calls == [
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--target",
            install_target,
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


def test_failed_ensure_target_runtime_preserves_previous_runtime(tmp_path):
    runtime_root = tmp_path / "runtime"
    runtime = runtime_root / "stable"
    runtime.mkdir(parents=True)
    (runtime / "stable.txt").write_bytes(b"stable")

    with pytest.raises(RuntimeError, match="could not be bootstrapped"):
        adapter_runtime.ensure_target_runtime(
            "stable",
            "broken-runtime==2.0",
            "broken_runtime",
            include_app_fallback=True,
            root=runtime_root,
            install_runner=lambda cmd, _timeout: subprocess.CompletedProcess(
                cmd,
                1,
                "",
                "failed",
            ),
            module_probe=lambda _import_name: False,
        )

    assert {path.relative_to(runtime): path.read_bytes() for path in runtime.rglob("*") if path.is_file()} == {
        Path("stable.txt"): b"stable"
    }


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
        install_target = Path(cmd[cmd.index("--target") + 1])
        (install_target / "transformers").mkdir(parents=True)
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


def test_failed_runtime_install_preserves_previous_runtime_byte_for_byte(tmp_path):
    runtime = tmp_path / "runtime" / "stable"
    runtime.mkdir(parents=True)
    (runtime / "stable.txt").write_bytes(b"stable")

    def runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        target = Path(cmd[cmd.index("--target") + 1])
        (target / "partial.txt").write_bytes(b"partial")
        return subprocess.CompletedProcess(cmd, 1, "", "failed")

    assert not adapter_runtime.install_target_runtime_requirements(
        runtime,
        ["broken-runtime==2.0"],
        install_runner=runner,
    )

    assert {path.relative_to(runtime): path.read_bytes() for path in runtime.rglob("*") if path.is_file()} == {
        Path("stable.txt"): b"stable"
    }
    assert not any(path.name.startswith(".stable.installing-") for path in runtime.parent.iterdir())


def test_successful_runtime_install_swaps_verified_stage_and_preserves_unrelated_files(tmp_path):
    runtime = tmp_path / "runtime" / "stable"
    runtime.mkdir(parents=True)
    (runtime / "stable.txt").write_bytes(b"stable")
    install_targets = []

    def runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        target = Path(cmd[cmd.index("--target") + 1])
        install_targets.append(target)
        (target / "new_runtime").mkdir(parents=True)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    assert adapter_runtime.install_target_runtime_requirements(
        runtime,
        ["new-runtime==2.0"],
        expected_paths=[runtime / "new_runtime"],
        install_runner=runner,
    )

    assert all(target != runtime for target in install_targets)
    assert (runtime / "stable.txt").read_bytes() == b"stable"
    assert (runtime / "new_runtime").is_dir()
    assert not any(path.name.startswith(".stable.installing-") for path in runtime.parent.iterdir())


def test_committed_runtime_remains_successful_when_backup_cleanup_is_busy(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime" / "stable"
    runtime.mkdir(parents=True)
    (runtime / "stable.txt").write_bytes(b"stable")

    def runner(cmd: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
        target = Path(cmd[cmd.index("--target") + 1])
        (target / "new_runtime").mkdir(parents=True)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    original_rmtree = atomic_install.shutil.rmtree

    def busy_backup(path, *args, **kwargs):
        if Path(path).name.startswith(".stable.committed-"):
            raise OSError(16, "Device or resource busy")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(atomic_install.shutil, "rmtree", busy_backup)

    assert adapter_runtime.install_target_runtime_requirements(
        runtime,
        ["new-runtime==2.0"],
        expected_paths=[runtime / "new_runtime"],
        install_runner=runner,
    )

    backups = tuple(runtime.parent.glob(".stable.committed-*"))
    assert (runtime / "new_runtime").is_dir()
    assert (runtime / "stable.txt").read_bytes() == b"stable"
    assert len(backups) == 1
    assert (backups[0] / "stable.txt").read_bytes() == b"stable"
