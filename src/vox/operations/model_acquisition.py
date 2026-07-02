from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from vox.core.adapter import BaseAdapter
from vox.core.adapter_acquisition import (
    AdapterTypeMismatchError,
    EnteredAdapter,
)
from vox.core.adapter_acquisition import (
    enter_typed_adapter as enter_core_typed_adapter,
)
from vox.core.adapter_acquisition import (
    release_entered_adapter as release_entered_adapter,
)
from vox.core.adapter_acquisition import (
    release_entered_adapter_suppressing as release_entered_adapter_suppressing,
)
from vox.core.errors import ModelNotFoundError
from vox.operations.errors import StoredModelNotFoundError, WrongModelTypeError

AdapterT = TypeVar("AdapterT", bound=BaseAdapter)


async def enter_typed_adapter(
    scheduler: Any,
    *,
    model: str,
    adapter_type: type[AdapterT],
    expected_type: str,
) -> EnteredAdapter[AdapterT]:
    try:
        return await enter_core_typed_adapter(
            scheduler,
            model=model,
            adapter_type=adapter_type,
            expected_type=expected_type,
        )
    except ModelNotFoundError as exc:
        raise StoredModelNotFoundError(exc.model) from exc
    except AdapterTypeMismatchError as exc:
        raise WrongModelTypeError(model, expected_type) from exc


@asynccontextmanager
async def acquire_typed_adapter(
    scheduler: Any,
    *,
    model: str,
    adapter_type: type[AdapterT],
    expected_type: str,
) -> AsyncIterator[AdapterT]:
    entered = await enter_typed_adapter(
        scheduler,
        model=model,
        adapter_type=adapter_type,
        expected_type=expected_type,
    )

    try:
        yield entered.adapter
    except BaseException as exc:
        suppress_exception = await entered.manager.__aexit__(
            type(exc),
            exc,
            exc.__traceback__,
        )
        if not suppress_exception:
            raise
    else:
        await entered.manager.__aexit__(None, None, None)
