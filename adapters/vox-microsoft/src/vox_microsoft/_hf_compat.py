from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from typing import Any


def _passthrough_value(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get("value")


def _wrap_validate_typed_dict(
    original: Callable[..., Any] | None,
) -> Callable[..., Any]:
    def _validate_typed_dict(*args: Any, **kwargs: Any) -> Any:
        if original is None:
            return _passthrough_value(args, kwargs)
        try:
            return original(*args, **kwargs)
        except (TypeError, ValueError) as exc:
            if "Unsupported type for field" in str(exc):
                return _passthrough_value(args, kwargs)
            raise

    return _validate_typed_dict


def _supports_kwarg(func: Callable[..., Any], keyword: str) -> bool:
    try:
        return keyword in inspect.signature(func).parameters
    except (TypeError, ValueError):
        return False


def _wrap_hf_hub_download(
    original: Callable[..., Any] | None,
) -> Callable[..., Any] | None:
    if original is None or _supports_kwarg(original, "tqdm_class"):
        return original

    def _hf_hub_download(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("tqdm_class", None)
        return original(*args, **kwargs)

    return _hf_hub_download


def ensure_huggingface_hub_compat() -> None:
    try:
        import huggingface_hub as huggingface_hub
    except ImportError:
        return

    if not hasattr(huggingface_hub, "is_offline_mode"):
        def _is_offline_mode() -> bool:
            value = os.environ.get("HF_HUB_OFFLINE", "")
            return value.strip().lower() in {"1", "true", "yes", "on"}

        huggingface_hub.is_offline_mode = _is_offline_mode

    wrapped_hf_hub_download = _wrap_hf_hub_download(getattr(huggingface_hub, "hf_hub_download", None))
    if wrapped_hf_hub_download is not None:
        huggingface_hub.hf_hub_download = wrapped_hf_hub_download

    try:
        from huggingface_hub import dataclasses as huggingface_hub_dataclasses
    except ImportError:
        return

    huggingface_hub_dataclasses.validate_typed_dict = _wrap_validate_typed_dict(
        getattr(huggingface_hub_dataclasses, "validate_typed_dict", None)
    )

    try:
        import huggingface_hub.file_download as huggingface_hub_file_download
    except ImportError:
        return

    wrapped_file_download = _wrap_hf_hub_download(getattr(huggingface_hub_file_download, "hf_hub_download", None))
    if wrapped_file_download is not None:
        huggingface_hub_file_download.hf_hub_download = wrapped_file_download
