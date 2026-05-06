"""Tests for AdapterResolver: external interface only."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vox.core.adapter_resolution import (
    ADAPTERS_DIR,
    ADAPTERS_NO_DEPS_ENV,
    BUNDLED_ADAPTERS_ENV,
    BUNDLED_ADAPTERS_NO_DEPS_ENV,
    DISABLE_BUNDLED_ADAPTERS_ENV,
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
    bundled_root: Path | None = None,
) -> AdapterResolver:
    bundled = bundled_root or (tmp_path / "_no_bundled_dir")
    with patch(
        "vox.core.adapter_resolution.entry_points",
        return_value=[
            _ep_mock(name, cls) for name, cls in (adapters or {}).items()
        ],
    ):
        return AdapterResolver(
            tmp_path,
            bundled_adapters_root=bundled,
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

    def test_refreshes_outdated_bundled_install(self, tmp_path: Path):
        package_dir = tmp_path / ADAPTERS_DIR / "vox-fake"
        package_dir.mkdir(parents=True)
        entry_point = MagicMock()
        spec = AdapterInstallSpec(entry_point=entry_point, path=package_dir)

        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)
        resolver._installed_specs = {"fake": spec}

        with (
            patch.object(AdapterResolver, "bundled_version", return_value="0.2.31"),
            patch.object(
                AdapterResolver, "installed_version", side_effect=["0.2.30", "0.2.30"]
            ),
            patch.object(
                AdapterResolver,
                "_scan_install_specs",
                side_effect=[{"fake": spec}, {"fake": spec}],
            ) as rescan_mock,
        ):
            assert resolver.ensure("fake", "vox-fake") is True

        assert rescan_mock.call_count == 2
        assert len(runner.calls) == 1


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
            resolver = AdapterResolver(tmp_path, bundled_adapters_root=tmp_path / "_no_bundled")

        names = {info.name for info in resolver.discover()}
        assert "good" in names
        assert "broken" not in names


class TestInstalledAndBundledVersion:
    def test_installed_version_none_when_dir_missing(self, tmp_path: Path):
        resolver = _make_resolver(tmp_path, adapters={})
        assert resolver.installed_version("vox-missing") is None

    def test_bundled_version_from_pyproject(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        bundled_root = tmp_path / "bundled"
        adapter_dir = bundled_root / "vox-fake"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "pyproject.toml").write_text(
            "[project]\nname='vox-fake'\nversion='1.2.3'\n",
            encoding="utf-8",
        )
        resolver = _make_resolver(tmp_path, adapters={}, bundled_root=bundled_root)

        assert resolver.bundled_version("vox-fake") == "1.2.3"

    def test_bundled_version_disabled_via_env(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        bundled_root = tmp_path / "bundled"
        adapter_dir = bundled_root / "vox-fake"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "pyproject.toml").write_text(
            "[project]\nname='vox-fake'\nversion='1.2.3'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(DISABLE_BUNDLED_ADAPTERS_ENV, "1")
        resolver = _make_resolver(tmp_path, adapters={}, bundled_root=bundled_root)
        assert resolver.bundled_version("vox-fake") is None

    def test_bundled_version_env_override_takes_precedence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        bundled_root = tmp_path / "bundled"
        adapter_dir = bundled_root / "vox-kokoro"
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "pyproject.toml").write_text(
            "[project]\nname='vox-kokoro'\nversion='9.9.9'\n",
            encoding="utf-8",
        )
        monkeypatch.setenv(BUNDLED_ADAPTERS_ENV, str(bundled_root))
        resolver = _make_resolver(
            tmp_path,
            adapters={},
            bundled_root=tmp_path / "_unused_default",
        )
        assert resolver.bundled_version("vox-kokoro") == "9.9.9"


class TestInstallCommand:
    def test_prefers_bundled_source(self, tmp_path: Path):
        bundled_root = tmp_path / "bundled"
        bundled_adapter = bundled_root / "vox-kokoro"
        bundled_adapter.mkdir(parents=True)
        (bundled_adapter / "pyproject.toml").write_text(
            "[project]\nname='vox-kokoro'\n",
            encoding="utf-8",
        )

        runner = _FakeRunner()
        resolver = _make_resolver(
            tmp_path, adapters={}, runner=runner, bundled_root=bundled_root
        )

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
                str(bundled_adapter),
            ]
        ]

    def test_skip_dependencies_for_curated_published_packages(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv(DISABLE_BUNDLED_ADAPTERS_ENV, "1")
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

    def test_includes_dependencies_for_non_curated_published_packages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(DISABLE_BUNDLED_ADAPTERS_ENV, "1")
        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)

        assert resolver._install_package("vox-whisper") is True
        assert runner.calls == [
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--target",
                str(tmp_path / "adapters" / "vox-whisper"),
                "--upgrade",
                "--refresh-package",
                "vox-whisper",
                "vox-whisper",
            ]
        ]

    def test_skip_dependencies_via_env_for_published_packages(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(DISABLE_BUNDLED_ADAPTERS_ENV, "1")
        monkeypatch.setenv(ADAPTERS_NO_DEPS_ENV, "1")

        runner = _FakeRunner()
        resolver = _make_resolver(tmp_path, adapters={}, runner=runner)

        assert resolver._install_package("vox-whisper") is True
        assert runner.calls[0][-2:] == ["--no-deps", "vox-whisper"]

    def test_bundled_install_skips_deps_via_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        bundled_root = tmp_path / "bundled"
        bundled_adapter = bundled_root / "vox-whisper"
        bundled_adapter.mkdir(parents=True)
        (bundled_adapter / "pyproject.toml").write_text(
            "[project]\nname='vox-whisper'\n", encoding="utf-8"
        )
        monkeypatch.setenv(BUNDLED_ADAPTERS_NO_DEPS_ENV, "1")

        runner = _FakeRunner()
        resolver = _make_resolver(
            tmp_path, adapters={}, runner=runner, bundled_root=bundled_root
        )
        assert resolver._install_package("vox-whisper") is True
        assert runner.calls[0][-2:] == ["--no-deps", str(bundled_adapter)]


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
