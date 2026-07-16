from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from vox.operations.conversation import ConversationSessionConfig
from vox.operations.errors import InvalidConfigError


@dataclass(frozen=True, slots=True)
class SessionUpdateCommand:
    config: ConversationSessionConfig


@dataclass(frozen=True, slots=True)
class AudioAppendCommand:
    pcm16: bytes
    sample_rate: int | None = None


@dataclass(frozen=True, slots=True)
class ResponseStartCommand:
    allow_interruptions: bool = True
    generation_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseDeltaCommand:
    text: str
    allow_interruptions: bool = True
    generation_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseCommitCommand:
    generation_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseCancelCommand:
    generation_id: str | None = None


@dataclass(frozen=True, slots=True)
class ResponseReplaceTextCommand:
    text: str
    allow_interruptions: bool = True


@dataclass(frozen=True, slots=True)
class ClientEventCommand:
    event: str
    payload: Any


@dataclass(frozen=True, slots=True)
class UnknownCommand:
    name: str


ConversationCommand = (
    SessionUpdateCommand
    | AudioAppendCommand
    | ResponseStartCommand
    | ResponseDeltaCommand
    | ResponseCommitCommand
    | ResponseCancelCommand
    | ResponseReplaceTextCommand
    | ClientEventCommand
    | UnknownCommand
)


def client_event_command(
    event_name: Any,
    payload: Any,
    *,
    non_empty_message: str = "client.event requires a non-empty string 'event'",
) -> ClientEventCommand:
    if not isinstance(event_name, str) or not event_name.strip():
        raise InvalidConfigError(non_empty_message)
    return ClientEventCommand(event=event_name.strip(), payload=payload)


def client_event_command_from_payload_json(
    event_name: Any,
    payload_json: str,
    *,
    invalid_json_message: str = "client.event requires valid payload JSON",
    non_empty_message: str = "client.event requires a non-empty string 'event'",
) -> ClientEventCommand:
    try:
        payload = json.loads(payload_json or "null")
    except json.JSONDecodeError as exc:
        raise InvalidConfigError(f"{invalid_json_message}: {exc}") from exc
    return client_event_command(
        event_name,
        payload,
        non_empty_message=non_empty_message,
    )
