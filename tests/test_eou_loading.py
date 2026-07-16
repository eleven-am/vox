from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from vox.streaming.eou import (
    EOU_MODEL_FILE,
    EOU_MODEL_ID,
    EOU_MODEL_REVISION,
    EOU_MODEL_SUBFOLDER,
    EOUModel,
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
