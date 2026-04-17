from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vox.server.app import (
    _env_bool,
    _parse_preload_list,
    _preload_models,
    _preload_vad,
    create_app,
)


class TestParsePreloadList:
    def test_none_returns_empty(self):
        assert _parse_preload_list(None) == []

    def test_empty_string_returns_empty(self):
        assert _parse_preload_list("") == []

    def test_single_ref(self):
        assert _parse_preload_list("parakeet-stt-onnx:0.6b") == ["parakeet-stt-onnx:0.6b"]

    def test_comma_separated(self):
        assert _parse_preload_list("a:1,b:2,c:3") == ["a:1", "b:2", "c:3"]

    def test_strips_whitespace(self):
        assert _parse_preload_list(" a:1 , b:2 ") == ["a:1", "b:2"]

    def test_ignores_empty_segments(self):
        assert _parse_preload_list("a:1,,b:2,") == ["a:1", "b:2"]


class TestEnvBool:
    def test_truthy_values(self):
        for v in ("1", "true", "TRUE", "yes", "YES", "on", "ON"):
            with patch.dict(os.environ, {"X": v}):
                assert _env_bool("X") is True

    def test_falsy_values(self):
        for v in ("", "0", "false", "no", "off", "random"):
            with patch.dict(os.environ, {"X": v}):
                assert _env_bool("X") is False

    def test_missing_is_false(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _env_bool("MISSING") is False


class _FakeScheduler:
    def __init__(self, *, raise_on: set[str] | None = None):
        self._raise_on = raise_on or set()
        self.acquired: list[str] = []

    async def start(self): ...
    async def stop(self): ...

    @asynccontextmanager
    async def acquire(self, model: str):
        self.acquired.append(model)
        if model in self._raise_on:
            raise RuntimeError(f"cannot load {model}")
        yield MagicMock()


class TestPreloadModels:
    @pytest.mark.asyncio
    async def test_all_models_preloaded_in_order(self):
        app = MagicMock()
        app.state.scheduler = _FakeScheduler()
        await _preload_models(app, ["a:1", "b:2", "c:3"])
        assert app.state.scheduler.acquired == ["a:1", "b:2", "c:3"]

    @pytest.mark.asyncio
    async def test_failed_preload_logged_not_raised(self, caplog):
        app = MagicMock()
        app.state.scheduler = _FakeScheduler(raise_on={"b:2"})
        await _preload_models(app, ["a:1", "b:2", "c:3"])
        assert app.state.scheduler.acquired == ["a:1", "b:2", "c:3"]
        assert any("Failed to preload b:2" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_empty_list_is_noop(self):
        app = MagicMock()
        app.state.scheduler = _FakeScheduler()
        await _preload_models(app, [])
        assert app.state.scheduler.acquired == []


class TestPreloadVAD:
    @pytest.mark.asyncio
    async def test_warm_calls_ensure_model(self):
        fake_vad = MagicMock()
        fake_cls = MagicMock(return_value=fake_vad)

        with patch("vox.streaming.vad.SileroVAD", fake_cls):
            await _preload_vad()

        fake_cls.assert_called_once()
        fake_vad._ensure_model.assert_called_once()

    @pytest.mark.asyncio
    async def test_warm_failure_is_logged_not_raised(self, caplog):
        fake_vad = MagicMock()
        fake_vad._ensure_model.side_effect = RuntimeError("no torch")

        with patch("vox.streaming.vad.SileroVAD", return_value=fake_vad):
            await _preload_vad()

        assert any("Failed to preload VAD" in r.message for r in caplog.records)


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

        fake_vad = MagicMock()
        with patch("vox.streaming.vad.SileroVAD", return_value=fake_vad):
            with TestClient(app) as _:
                pass

        fake_vad._ensure_model.assert_called_once()


class TestCLISignature:
    """Verify CLI exposes --preload and --preload-vad options."""

    def test_serve_has_preload_options(self):
        from vox.cli import serve
        opt_names = {p.name for p in serve.params}
        assert "preload_models" in opt_names
        assert "preload_vad" in opt_names
