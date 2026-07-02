from __future__ import annotations

import asyncio
from typing import Any

from vox.core.tasks import drain_task


async def close_conversation_runtime_resources(
    *,
    orchestrator: Any,
    emit_task: asyncio.Task,
) -> None:
    await orchestrator.end_of_stream()
    await drain_task(emit_task)
    await orchestrator.close()
