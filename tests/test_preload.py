from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient

from vox.server.app import create_app
from vox.server.preload import (
    env_bool,
    merged_preload_models,
    parse_preload_list,
    preload_core_native_modules,
    preload_models,
    preload_ner,
    preload_turn_detector,
    preload_vad,
    should_preload_vad,
)


class TestPreloadCoreNativeModules:
    def test_imports_matching_cffi_wrapper_and_backend_in_order(self):
        cffi = MagicMock(__version__="2.0.0")
        backend = MagicMock(__version__="2.0.0")

        with patch(
            "vox.server.preload.importlib.import_module",
            side_effect=[cffi, backend],
        ) as import_module:
            preload_core_native_modules()

        assert import_module.call_args_list == [call("cffi"), call("_cffi_backend")]

    def test_rejects_mismatched_cffi_wrapper_and_backend(self):
        cffi = MagicMock(__version__="2.1.0")
        backend = MagicMock(__version__="2.0.0")

        with (
            patch(
                "vox.server.preload.importlib.import_module",
                side_effect=[cffi, backend],
            ),
            pytest.raises(
                RuntimeError,
                match=r"cffi=2\.1\.0, _cffi_backend=2\.0\.0",
            ),
        ):
            preload_core_native_modules()


class TestParsePreloadList:
    def test_none_returns_empty(self):
        assert parse_preload_list(None) == []

    def test_empty_string_returns_empty(self):
        assert parse_preload_list("") == []

    def test_single_ref(self):
        assert parse_preload_list("parakeet-stt-onnx:0.6b") == ["parakeet-stt-onnx:0.6b"]

    def test_comma_separated(self):
        assert parse_preload_list("a:1,b:2,c:3") == ["a:1", "b:2", "c:3"]

    def test_strips_whitespace(self):
        assert parse_preload_list(" a:1 , b:2 ") == ["a:1", "b:2"]

    def test_ignores_empty_segments(self):
        assert parse_preload_list("a:1,,b:2,") == ["a:1", "b:2"]


class TestEnvBool:
    def test_truthy_values(self):
        for v in ("1", "true", "TRUE", "yes", "YES", "on", "ON"):
            with patch.dict(os.environ, {"X": v}):
                assert env_bool("X") is True

    def test_falsy_values(self):
        for v in ("", "0", "false", "no", "off", "random"):
            with patch.dict(os.environ, {"X": v}):
                assert env_bool("X") is False

    def test_missing_is_false(self):
        with patch.dict(os.environ, {}, clear=True):
            assert env_bool("MISSING") is False


class TestMergedPreloadModels:
    def test_preserves_explicit_order_before_env_models(self):
        assert merged_preload_models(["explicit:1"], "env-a:1,env-b:2") == [
            "explicit:1",
            "env-a:1",
            "env-b:2",
        ]

    def test_dedups_explicit_and_env_models(self):
        assert merged_preload_models(["shared:1"], "shared:1,only-env:1") == [
            "shared:1",
            "only-env:1",
        ]


class TestShouldPreloadVAD:
    def test_explicit_true_wins_without_env(self):
        with patch.dict(os.environ, {}, clear=True):
            assert should_preload_vad(True) is True

    def test_env_can_enable_vad_preload(self):
        with patch.dict(os.environ, {"VOX_PRELOAD_VAD": "yes"}):
            assert should_preload_vad(False) is True

    def test_defaults_false(self):
        with patch.dict(os.environ, {}, clear=True):
            assert should_preload_vad(False) is False


class _FakeScheduler:
    def __init__(self, *, raise_on: set[str] | None = None):
        self._raise_on = raise_on or set()
        self.acquired: list[str] = []

    async def start(self): ...
    async def stop(self, *, deadline: float | None = None): ...

    @asynccontextmanager
    async def acquire(self, model: str):
        self.acquired.append(model)
        if model in self._raise_on:
            raise RuntimeError(f"cannot load {model}")
        yield MagicMock()


class TestPreloadModels:
    @pytest.mark.asyncio
    async def test_all_models_preloaded_in_order(self):
        scheduler = _FakeScheduler()
        await preload_models(scheduler, ["a:1", "b:2", "c:3"])
        assert scheduler.acquired == ["a:1", "b:2", "c:3"]

    @pytest.mark.asyncio
    async def test_failed_preload_logged_not_raised(self, caplog):
        scheduler = _FakeScheduler(raise_on={"b:2"})
        await preload_models(scheduler, ["a:1", "b:2", "c:3"])
        assert scheduler.acquired == ["a:1", "b:2", "c:3"]
        assert any("Failed to preload b:2" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_empty_list_is_noop(self):
        scheduler = _FakeScheduler()
        await preload_models(scheduler, [])
        assert scheduler.acquired == []


class TestPreloadNer:
    @pytest.mark.asyncio
    async def test_preloads_english_model_off_event_loop(self):
        with patch("vox.core.ner.preload_model", return_value=True) as preload:
            await preload_ner()

        preload.assert_called_once_with("en")

    @pytest.mark.asyncio
    async def test_failed_preload_is_logged_not_raised(self, caplog):
        with patch("vox.core.ner.preload_model", side_effect=RuntimeError("broken runtime")):
            await preload_ner()

        assert any("Failed to preload English NER model" in record.message for record in caplog.records)


class TestPreloadVAD:
    @pytest.mark.asyncio
    async def test_warm_calls_ensure_session(self):
        fake_cls = MagicMock()

        with patch("vox.streaming.vad.SileroOnnxVAD", fake_cls):
            await preload_vad()

        fake_cls._ensure_session.assert_called_once()

    @pytest.mark.asyncio
    async def test_warm_failure_is_logged_not_raised(self, caplog):
        fake_cls = MagicMock()
        fake_cls._ensure_session.side_effect = RuntimeError("no onnxruntime")

        with patch("vox.streaming.vad.SileroOnnxVAD", fake_cls):
            await preload_vad()

        assert any("Failed to preload VAD" in r.message for r in caplog.records)


class TestPreloadTurnDetector:
    @pytest.mark.asyncio
    async def test_preloads_selected_detector(self):
        detector = MagicMock()
        detector.preload = MagicMock(return_value=None)

        with patch("vox.streaming.eou.create_turn_detector", return_value=detector) as factory:
            await preload_turn_detector(MagicMock(), "livekit")

        factory.assert_called_once()
        detector.preload.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_failure_is_logged_not_raised(self, caplog):
        with patch(
            "vox.streaming.eou.create_turn_detector",
            side_effect=RuntimeError("unavailable"),
        ):
            await preload_turn_detector(MagicMock(), "livekit")

        assert any("Failed to preload turn detector livekit" in r.message for r in caplog.records)


class TestCreateAppWiring:
    def test_preload_kwargs_stored_on_app_state(self, tmp_path: Path):
        app = create_app(
            vox_home=tmp_path,
            preload_models=["a:1", "b:2"],
            preload_vad=True,
        )
        assert app.state.preload_models == ["a:1", "b:2"]
        assert app.state.preload_vad is True

    def test_preload_defaults_empty(self, tmp_path: Path):
        app = create_app(vox_home=tmp_path)
        assert app.state.preload_models == []
        assert app.state.preload_vad is False
        assert app.state.preload_turn_detector is None


class TestLifespanIntegration:
    def test_lifespan_runs_preload_from_app_state(self, tmp_path: Path):
        fake_sched = _FakeScheduler()
        app = create_app(vox_home=tmp_path, preload_models=["foo:1"])
        app.state.scheduler = fake_sched
        app.state.grpc_port = None

        with TestClient(app) as _:
            pass

        assert "foo:1" in fake_sched.acquired

    def test_lifespan_merges_env_with_explicit(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("VOX_PRELOAD", "env-a:1,env-b:2")
        fake_sched = _FakeScheduler()
        app = create_app(vox_home=tmp_path, preload_models=["explicit:1"])
        app.state.scheduler = fake_sched
        app.state.grpc_port = None

        with TestClient(app) as _:
            pass

        assert fake_sched.acquired[0] == "explicit:1"
        assert "env-a:1" in fake_sched.acquired
        assert "env-b:2" in fake_sched.acquired

    def test_lifespan_dedups_overlapping_refs(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("VOX_PRELOAD", "shared:1,only-env:1")
        fake_sched = _FakeScheduler()
        app = create_app(vox_home=tmp_path, preload_models=["shared:1"])
        app.state.scheduler = fake_sched
        app.state.grpc_port = None

        with TestClient(app) as _:
            pass

        assert fake_sched.acquired.count("shared:1") == 1

    def test_lifespan_triggers_vad_preload_from_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("VOX_PRELOAD_VAD", "1")
        fake_sched = _FakeScheduler()
        app = create_app(vox_home=tmp_path)
        app.state.scheduler = fake_sched
        app.state.grpc_port = None

        fake_cls = MagicMock()
        with patch("vox.streaming.vad.SileroOnnxVAD", fake_cls), TestClient(app) as _:
            pass

        fake_cls._ensure_session.assert_called_once()


class TestCLISignature:
    """Verify CLI exposes --preload and --preload-vad options."""

    def test_serve_has_preload_options(self):
        from vox.cli import serve

        opt_names = {p.name for p in serve.params}
        assert "preload_models" in opt_names
        assert "preload_vad" in opt_names
        assert "preload_turn_detector" in opt_names
