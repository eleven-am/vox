from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from vox.core.adapter import BaseAdapter
from vox.core.errors import VoxError

AdapterT = TypeVar("AdapterT", bound=BaseAdapter)


class AdapterTypeMismatchError(VoxError):
    def __init__(self, model: str, expected: str) -> None:
        self.model = model
        self.expected = expected
        super().__init__(f"model {model!r} is not a {expected} adapter")


@dataclass(frozen=True)
class EnteredAdapter(Generic[AdapterT]):
    manager: Any
    adapter: AdapterT


async def release_entered_adapter(entered: EnteredAdapter[Any]) -> None:
    await entered.manager.__aexit__(None, None, None)


async def release_entered_adapter_suppressing(entered: EnteredAdapter[Any]) -> None:
    with suppress(Exception):
        await release_entered_adapter(entered)


async def enter_typed_adapter(
    scheduler: Any,
    *,
    model: str,
    adapter_type: type[AdapterT],
    expected_type: str,
) -> EnteredAdapter[AdapterT]:
    manager = scheduler.acquire(model)
    adapter = await manager.__aenter__()

    if not isinstance(adapter, adapter_type):
        await manager.__aexit__(None, None, None)
        raise AdapterTypeMismatchError(model, expected_type)

    return EnteredAdapter(manager=manager, adapter=adapter)


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
