from __future__ import annotations

import asyncio
import sys
import threading
from contextlib import asynccontextmanager
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from vox.core.adapter import TurnDetectorAdapter
from vox.core.types import AdapterInfo, ModelFormat, ModelType
from vox.streaming.eou import (
    EOU_MODEL_FILE,
    EOU_MODEL_ID,
    EOU_MODEL_REVISION,
    EOU_MODEL_SUBFOLDER,
    EOUModel,
    ModelTurnDetector,
)


def _transformers_stub(*, side_effect=None, return_value=None):
    module = ModuleType("transformers")
    auto_tokenizer = MagicMock()
    auto_tokenizer.from_pretrained = MagicMock(
        side_effect=side_effect,
        return_value=return_value,
    )
    module.AutoTokenizer = auto_tokenizer
    return module, auto_tokenizer.from_pretrained


@pytest.fixture(autouse=True)
def reset_eou_model():
    EOUModel._instance = None
    EOUModel._session = None
    EOUModel._tokenizer = None
    yield
    EOUModel._instance = None
    EOUModel._session = None
    EOUModel._tokenizer = None


def test_cached_livekit_assets_do_not_use_network():
    tokenizer = MagicMock()
    session = MagicMock()
    transformers, load_tokenizer = _transformers_stub(return_value=tokenizer)

    with (
        patch.dict(sys.modules, {"transformers": transformers}),
        patch("huggingface_hub.hf_hub_download", return_value="/cache/model_q8.onnx") as download,
        patch("onnxruntime.InferenceSession", return_value=session) as load_session,
    ):
        EOUModel()._ensure_loaded()

    download.assert_called_once_with(
        repo_id=EOU_MODEL_ID,
        filename=EOU_MODEL_FILE,
        subfolder=EOU_MODEL_SUBFOLDER,
        revision=EOU_MODEL_REVISION,
        local_files_only=True,
    )
    load_tokenizer.assert_called_once_with(
        EOU_MODEL_ID,
        revision=EOU_MODEL_REVISION,
        local_files_only=True,
    )
    load_session.assert_called_once_with(
        "/cache/model_q8.onnx",
        providers=["CPUExecutionProvider"],
    )
    assert EOUModel._tokenizer is tokenizer
    assert EOUModel._session is session


def test_missing_model_cache_falls_back_to_download():
    missing = LocalEntryNotFoundError("model is not cached")
    transformers, _ = _transformers_stub(return_value=MagicMock())

    with (
        patch.dict(sys.modules, {"transformers": transformers}),
        patch(
            "huggingface_hub.hf_hub_download",
            side_effect=[missing, "/downloaded/model_q8.onnx"],
        ) as download,
        patch("onnxruntime.InferenceSession", return_value=MagicMock()),
    ):
        EOUModel()._ensure_loaded()

    model_kwargs = {
        "repo_id": EOU_MODEL_ID,
        "filename": EOU_MODEL_FILE,
        "subfolder": EOU_MODEL_SUBFOLDER,
        "revision": EOU_MODEL_REVISION,
    }
    assert download.call_args_list == [
        call(**model_kwargs, local_files_only=True),
        call(**model_kwargs),
    ]


def test_missing_tokenizer_cache_falls_back_to_download():
    tokenizer = MagicMock()
    transformers, load_tokenizer = _transformers_stub(
        side_effect=[OSError("tokenizer is not cached"), tokenizer],
    )

    with (
        patch.dict(sys.modules, {"transformers": transformers}),
        patch("huggingface_hub.hf_hub_download", return_value="/cache/model_q8.onnx"),
        patch("onnxruntime.InferenceSession", return_value=MagicMock()),
    ):
        EOUModel()._ensure_loaded()

    assert load_tokenizer.call_args_list == [
        call(
            EOU_MODEL_ID,
            revision=EOU_MODEL_REVISION,
            local_files_only=True,
        ),
        call(EOU_MODEL_ID, revision=EOU_MODEL_REVISION),
    ]


@pytest.mark.asyncio
async def test_model_turn_detector_cancellation_keeps_physical_inference_owned():
    predict_started = threading.Event()
    release_predict = threading.Event()

    class BlockingTurnAdapter(TurnDetectorAdapter):
        def info(self) -> AdapterInfo:
            return AdapterInfo(
                name="blocking-turn",
                type=ModelType.TURN,
                architectures=("test",),
                default_sample_rate=16_000,
                supported_formats=(ModelFormat.ONNX,),
            )

        def load(self, model_path: str, device: str, **kwargs) -> None:
            pass

        def unload(self) -> None:
            pass

        @property
        def is_loaded(self) -> bool:
            return True

        def predict(self, audio, *, sample_rate: int) -> float:
            predict_started.set()
            release_predict.wait(5.0)
            return 0.75

    class SchedulerStub:
        def __init__(self, adapter: TurnDetectorAdapter) -> None:
            self.adapter = adapter

        @asynccontextmanager
        async def acquire(self, model: str):
            yield self.adapter

    adapter = BlockingTurnAdapter()
    detector = ModelTurnDetector(SchedulerStub(adapter), "smart-turn:test")
    predict_task = asyncio.create_task(
        detector.predict(
            [],
            audio=np.ones(1_600, dtype=np.float32),
            sample_rate=16_000,
        )
    )

    assert await asyncio.to_thread(predict_started.wait, 5.0)
    predict_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await predict_task

    assert adapter.physical_work_count == 1

    release_predict.set()
    await adapter.wait_execution_idle(timeout=5.0)
    assert adapter.physical_work_count == 0
