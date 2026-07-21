from __future__ import annotations

import asyncio

import pytest

from vox.conversation.timers import ConversationTimerLease, ConversationTimerRegistry


@pytest.mark.asyncio
async def test_timer_registry_emits_expiry_once():
    expired: list[ConversationTimerLease] = []
    registry = ConversationTimerRegistry(lambda lease: _append_expired(expired, lease))

    await registry.start("endpointing", 1)
    assert registry.has_active("endpointing")

    await asyncio.sleep(0.02)

    assert [lease.key for lease in expired] == ["endpointing"]
    assert registry.consume(expired[0])
    assert not registry.consume(expired[0])
    assert not registry.has_active("endpointing")
    assert not registry.has_any_active()


@pytest.mark.asyncio
async def test_timer_registry_cancel_prevents_expiry():
    expired: list[ConversationTimerLease] = []
    registry = ConversationTimerRegistry(lambda lease: _append_expired(expired, lease))

    await registry.start("endpointing", 20)
    await registry.cancel("endpointing")
    await asyncio.sleep(0.03)

    assert expired == []
    assert not registry.has_active("endpointing")


@pytest.mark.asyncio
async def test_timer_registry_replaces_existing_timer_for_same_key():
    expired: list[ConversationTimerLease] = []
    registry = ConversationTimerRegistry(lambda lease: _append_expired(expired, lease))

    await registry.start("endpointing", 40)
    await registry.start("endpointing", 1)
    await asyncio.sleep(0.03)

    assert [lease.key for lease in expired] == ["endpointing"]
    assert registry.consume(expired[0])
    assert not registry.has_active("endpointing")


@pytest.mark.asyncio
async def test_timer_registry_cancel_all_prevents_all_expiry():
    expired: list[ConversationTimerLease] = []
    registry = ConversationTimerRegistry(lambda lease: _append_expired(expired, lease))

    await registry.start("endpointing", 20)
    await registry.start("confirm_interrupt", 20)
    assert registry.has_any_active()

    registry.cancel_all()
    await asyncio.sleep(0.03)

    assert expired == []
    assert not registry.has_any_active()


@pytest.mark.asyncio
async def test_expired_lease_cannot_consume_replacement_timer():
    expired: list[ConversationTimerLease] = []
    registry = ConversationTimerRegistry(lambda lease: _append_expired(expired, lease))

    await registry.start("endpointing", 1)
    await asyncio.sleep(0.02)
    stale = expired[0]

    replacement = await registry.start("endpointing", 100)

    assert replacement is not stale
    assert not registry.consume(stale)
    assert registry.has_active("endpointing")

    await registry.cancel("endpointing")


async def _append_expired(
    expired: list[ConversationTimerLease],
    lease: ConversationTimerLease,
) -> None:
    expired.append(lease)
