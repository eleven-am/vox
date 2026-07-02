from __future__ import annotations

import asyncio

import pytest

from vox.conversation.timers import ConversationTimerRegistry


@pytest.mark.asyncio
async def test_timer_registry_emits_expiry_once():
    expired: list[str] = []
    registry = ConversationTimerRegistry(lambda key: _append_expired(expired, key))

    await registry.start("endpointing", 1)
    assert registry.has_active("endpointing")

    await asyncio.sleep(0.02)

    assert expired == ["endpointing"]
    assert not registry.has_active("endpointing")
    assert not registry.has_any_active()


@pytest.mark.asyncio
async def test_timer_registry_cancel_prevents_expiry():
    expired: list[str] = []
    registry = ConversationTimerRegistry(lambda key: _append_expired(expired, key))

    await registry.start("endpointing", 20)
    await registry.cancel("endpointing")
    await asyncio.sleep(0.03)

    assert expired == []
    assert not registry.has_active("endpointing")


@pytest.mark.asyncio
async def test_timer_registry_replaces_existing_timer_for_same_key():
    expired: list[str] = []
    registry = ConversationTimerRegistry(lambda key: _append_expired(expired, key))

    await registry.start("endpointing", 40)
    await registry.start("endpointing", 1)
    await asyncio.sleep(0.03)

    assert expired == ["endpointing"]
    assert not registry.has_active("endpointing")


@pytest.mark.asyncio
async def test_timer_registry_cancel_all_prevents_all_expiry():
    expired: list[str] = []
    registry = ConversationTimerRegistry(lambda key: _append_expired(expired, key))

    await registry.start("endpointing", 20)
    await registry.start("confirm_interrupt", 20)
    assert registry.has_any_active()

    registry.cancel_all()
    await asyncio.sleep(0.03)

    assert expired == []
    assert not registry.has_any_active()


async def _append_expired(expired: list[str], key: str) -> None:
    expired.append(key)
