from __future__ import annotations

import pytest

from vox.conversation.response_output import (
    ResponseOutputOptions,
    resolve_response_output,
)


def test_response_output_defaults_to_session_synthesis_config() -> None:
    resolved = resolve_response_output(
        None,
        model="kokoro-tts:v1.0",
        voice="af_heart",
        language="en",
    )

    assert resolved.to_payload() == {
        "model": "kokoro-tts:v1.0",
        "voice": "af_heart",
        "language": "en",
        "speed": 1.0,
        "params": {},
    }


def test_response_output_partial_override_resolves_field_by_field() -> None:
    options = ResponseOutputOptions(
        model="qwen3-tts:0.6b-clone",
        language="fr",
        speed=0.9,
        params={"temperature": 0.7, "sampling": {"top_p": 0.95}},
    )

    resolved = resolve_response_output(
        options,
        model="kokoro-tts:v1.0",
        voice="samantha",
        language="en",
    )

    assert resolved.to_payload() == {
        "model": "qwen3-tts:0.6b-clone",
        "voice": "samantha",
        "language": "fr",
        "speed": 0.9,
        "params": {"sampling": {"top_p": 0.95}, "temperature": 0.7},
    }


def test_response_output_snapshot_cannot_be_mutated_through_source_or_result() -> None:
    source = {"sequence": [1, 2]}
    options = ResponseOutputOptions(params=source)
    source["sequence"].append(3)

    resolved = resolve_response_output(
        options,
        model="kokoro-tts:v1.0",
        voice=None,
        language="en",
    )
    exposed = resolved.params_dict()
    exposed["sequence"].append(4)

    assert resolved.params_dict() == {"sequence": [1, 2]}


@pytest.mark.parametrize("speed", [0, -1, float("inf"), float("nan"), True])
def test_response_output_rejects_invalid_speed(speed) -> None:
    with pytest.raises(ValueError, match="speed"):
        ResponseOutputOptions(speed=speed)
