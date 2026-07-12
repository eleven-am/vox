"""Tests for AdapterResolver: external interface only."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox.core.adapter_resolution import (
    ADAPTERS_DIR,
    ADAPTERS_NO_DEPS_ENV,
    AdapterInstallSpec,
    AdapterResolver,
)
from vox.core.errors import AdapterNotFoundError


class _FakeRunner:
    def __init__(self, returncode: int = 0, stderr: str = "") -> None:
        self.calls: list[list[str]] = []
        self.returncode = returncode
        self.stderr = stderr

    def __call__(self, cmd, timeout):
        self.calls.append(cmd)
        return MagicMock(returncode=self.returncode, stderr=self.stderr)


def _make_resolver(
    tmp_path: Path,
    *,
    adapters: dict | None = None,
    runner: _FakeRunner | None = None,
) -> AdapterResolver:
    with patch(
        "vox.core.adapter_resolution.entry_points",
        return_value=[
            _ep_mock(name, cls) for name, cls in (adapters or {}).items()
        ],
    ):
        return AdapterResolver(
            tmp_path,
            install_runner=runner or _FakeRunner(),
        )


def _ep_mock(name: str, cls: type) -> MagicMock:
    ep = MagicMock()
    ep.name = name
    ep.load.return_value = cls
    return ep


class TestResolve:
    def test_raises_when_missing(self, tmp_path: Path):
        resolver = _make_resolver(tmp_path, adapters={})
        with pytest.raises(AdapterNotFoundError):
            resolver.resolve("nonexistent")

    def test_returns_globally_discovered_class(self, tmp_path: Path):
        class FakeAdapter:
            pass

        resolver = _make_resolver(tmp_path, adapters={"fake": FakeAdapter})
        assert resolver.resolve("fake") is FakeAdapter

    def test_loads_isolated_adapter_on_demand(self, tmp_path: Path):
        class FakeAdapter:
            pass

        package_dir = tmp_path / ADAPTERS_DIR / "vox-fake"
        package_dir.mkdir(parents=True)

        resolver = _make_resolver(tmp_path, adapters={})
        entry_point = MagicMock()
        entry_point.load.return_value = FakeAdapter
        resolver._installed_specs = {
            "fake": AdapterInstallSpec(entry_point=entry_point, path=package_dir),
        }

        assert resolver.resolve("fake") is FakeAdapter
        assert str(package_dir) not in sys.path

    def test_drops_stale_isolated_adapter_spec_when_path_is_missing(self, tmp_path: Path):
        missing_dir = tmp_path / ADAPTERS_DIR / "vox-missing"
        resolver = _make_resolver(tmp_path, adapters={})
        resolver._installed_specs = {
            "fake": AdapterInstallSpec(entry_point=MagicMock(), path=missing_dir),
        }

        with pytest.raises(AdapterNotFoundError):
            resolver.resolve("fake")

        assert "fake" not in resolver._installed_specs

    def test_drops_broken_isolated_adapter_spec_when_import_fails(self, tmp_path: Path):
        package_dir = tmp_path / ADAPTERS_DIR / "vox-broken"
        package_dir.mkdir(parents=True)
        entry_point = MagicMock()
        entry_point.load.side_effect = ModuleNotFoundError("No module named 'vox_broken'")
        resolver = _make_resolver(tmp_path, adapters={})
        resolver._installed_specs = {
            "fake": AdapterInstallSpec(entry_point=entry_point, path=package_dir),
        }

        with pytest.raises(AdapterNotFoundError):
            resolver.resolve("fake")

        assert "fake" not in resolver._installed_specs

    def test_caches_resolved_class(self, tmp_path: Path):
        class FakeAdapter:
            pass

        resolver = _make_resolver(tmp_path, adapters={"fake": FakeAdapter})
        resolver.resolve("fake")
        with patch("vox.core.adapter_resolution.entry_points") as ep_call:
            assert resolver.resolve("fake") is FakeAdapter
            ep_call.assert_not_called()


class TestEnsure:
    def test_returns_true_when_globally_present(self, tmp_path: Path):
        class FakeAdapter:
            pass

        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={"fake": FakeAdapter}, runner=runner)
        assert resolver.ensure("fake", "vox-fake") is True
        assert runner.calls == []

    def test_rescans_specs_before_reinstalling(self, tmp_path: Path):
        package_dir = tmp_path / ADAPTERS_DIR / "vox-fake"
        package_dir.mkdir(parents=True)
        entry_point = MagicMock()
        spec = AdapterInstallSpec(entry_point=entry_point, path=package_dir)

        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)
        with patch.object(
            AdapterResolver, "_scan_install_specs", return_value={"fake": spec}
        ) as rescan_mock:
            assert resolver.ensure("fake", "vox-fake") is True
        assert rescan_mock.called
        assert runner.calls == []

    def test_reinstalls_when_cached_spec_path_was_removed(self, tmp_path: Path):
        missing_dir = tmp_path / ADAPTERS_DIR / "vox-fake"
        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)
        resolver._installed_specs = {
            "fake": AdapterInstallSpec(entry_point=MagicMock(), path=missing_dir),
        }

        with patch.object(AdapterResolver, "_scan_install_specs", return_value={}):
            assert resolver.ensure("fake", "vox-parakeet") is False

        assert runner.calls != []
        assert "fake" not in resolver._installed_specs

    def test_reinstalls_when_cached_adapter_install_dir_was_removed(self, tmp_path: Path):
        package_dir = tmp_path / ADAPTERS_DIR / "vox-fake"
        module_name = "_vox_fake_cached_adapter"
        module = types.ModuleType(module_name)
        module.__file__ = str(package_dir / "vox_fake" / "adapter.py")

        class FakeAdapter:
            pass

        FakeAdapter.__module__ = module_name

        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)
        resolver._adapters["fake"] = FakeAdapter

        original = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            with patch.object(AdapterResolver, "_scan_install_specs", return_value={}):
                assert resolver.ensure("fake", "vox-parakeet") is False
        finally:
            if original is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = original

        assert runner.calls != []
        assert "fake" not in resolver._adapters

    def test_reinstalls_when_cached_spec_import_is_broken(self, tmp_path: Path):
        class FakeAdapter:
            pass

        package_dir = tmp_path / ADAPTERS_DIR / "vox-fake"
        package_dir.mkdir(parents=True)
        broken_ep = MagicMock()
        broken_ep.load.side_effect = ModuleNotFoundError("No module named 'vox_fake'")
        repaired_ep = MagicMock()
        repaired_ep.load.return_value = FakeAdapter

        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)
        resolver._installed_specs = {
            "fake": AdapterInstallSpec(entry_point=broken_ep, path=package_dir),
        }

        scan_results = [
            {},
            {"fake": AdapterInstallSpec(entry_point=repaired_ep, path=package_dir)},
        ]
        with patch.object(AdapterResolver, "_scan_install_specs", side_effect=scan_results):
            assert resolver.ensure("fake", "vox-parakeet") is True

        assert runner.calls != []
        assert resolver.resolve("fake") is FakeAdapter

    def test_refuses_unverified_package_without_opt_in(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("VOX_ALLOW_UNVERIFIED_ADAPTERS", raising=False)
        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)

        assert resolver.ensure("evil", "totally-not-vox") is False
        assert runner.calls == []

    def test_allows_unverified_package_with_opt_in(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("VOX_ALLOW_UNVERIFIED_ADAPTERS", "1")
        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)

        resolver.ensure("evil", "totally-not-vox")
        assert runner.calls != []

    def test_allows_known_catalog_package(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("VOX_ALLOW_UNVERIFIED_ADAPTERS", raising=False)
        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)

        resolver.ensure("parakeet", "vox-parakeet")
        assert runner.calls != []


class TestDiscover:
    def test_lists_global_and_isolated_adapters(self, tmp_path: Path):
        class GlobalAdapter:
            pass

        package_dir = tmp_path / ADAPTERS_DIR / "vox-iso"
        package_dir.mkdir(parents=True)
        entry_point = MagicMock()
        spec = AdapterInstallSpec(entry_point=entry_point, path=package_dir)

        resolver = _make_resolver(tmp_path, adapters={"global": GlobalAdapter})
        resolver._installed_specs = {"isolated": spec}

        infos = {info.name: info for info in resolver.discover()}
        assert "global" in infos
        assert "isolated" in infos
        assert infos["global"].source == "entry_point"
        assert infos["isolated"].source == "isolated_install"
        assert infos["isolated"].path == package_dir

    def test_skips_broken_entry_points(self, tmp_path: Path):
        class GoodAdapter:
            pass

        good_ep = _ep_mock("good", GoodAdapter)
        bad_ep = MagicMock()
        bad_ep.name = "broken"
        bad_ep.load.side_effect = ImportError("missing dependency")

        with patch(
            "vox.core.adapter_resolution.entry_points",
            return_value=[good_ep, bad_ep],
        ):
            resolver = AdapterResolver(tmp_path)

        names = {info.name for info in resolver.discover()}
        assert "good" in names
        assert "broken" not in names


class TestInstalledVersion:
    def test_installed_version_none_when_dir_missing(self, tmp_path: Path):
        resolver = _make_resolver(tmp_path, adapters={})
        assert resolver.installed_version("vox-missing") is None


class TestInstallCommand:
    def test_skip_dependencies_for_curated_published_packages(self, tmp_path: Path):
        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)

        assert resolver._install_package("vox-kokoro") is True
        assert runner.calls == [
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(tmp_path / "adapters" / "vox-kokoro"),
                "--upgrade",
                "--refresh-package",
                "vox-kokoro",
                "--no-deps",
                "vox-kokoro",
            ]
        ]

    @pytest.mark.parametrize(
        "package_name",
        ["vox-chatterbox", "vox-sesame", "vox-whisper"],
    )
    def test_skip_dependencies_for_torch_backed_target_runtime_packages(
        self, tmp_path: Path, package_name: str
    ):
        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)

        assert resolver._install_package(package_name) is True
        assert runner.calls[0][-2:] == ["--no-deps", package_name]

    def test_includes_dependencies_for_non_curated_published_packages(
        self, tmp_path: Path
    ):
        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)

        assert resolver._install_package("vox-example") is True
        assert runner.calls == [
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(tmp_path / "adapters" / "vox-example"),
                "--upgrade",
                "--refresh-package",
                "vox-example",
                "vox-example",
            ]
        ]

    def test_includes_dependencies_for_parakeet_package(self, tmp_path: Path):
        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)

        assert resolver._install_package("vox-parakeet") is True
        assert runner.calls == [
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(tmp_path / "adapters" / "vox-parakeet"),
                "--upgrade",
                "--refresh-package",
                "vox-parakeet",
                "vox-parakeet",
            ]
        ]

    def test_skip_dependencies_via_env_for_published_packages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(ADAPTERS_NO_DEPS_ENV, "1")

        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)

        assert resolver._install_package("vox-whisper") is True
        assert runner.calls[0][-2:] == ["--no-deps", "vox-whisper"]


class TestSysPathHygiene:
    def test_sanitize_removes_root_and_package_dirs(self, tmp_path: Path):
        adapters_root = tmp_path / ADAPTERS_DIR
        package_dir = adapters_root / "vox-fake"
        package_dir.mkdir(parents=True)
        sys.path.insert(0, str(adapters_root))
        sys.path.insert(0, str(package_dir))
        try:
            _make_resolver(tmp_path, adapters={})
            assert str(adapters_root) not in sys.path
            assert str(package_dir) not in sys.path
        finally:
            sys.path[:] = [p for p in sys.path if p not in {str(adapters_root), str(package_dir)}]
