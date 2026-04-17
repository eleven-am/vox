"""WebSocket endpoint for agent-facing voice conversations.

Wire shape is OpenAI Realtime-compatible so existing SDKs can point at Vox with
a URL swap. See src/vox/conversation/session.py for the wire event names.

This route is NOT intended for direct browser use. Browsers / phones terminate
WebRTC (or SIP) at an *agent* process, which then speaks this WS on behalf of
the user. That keeps Vox's scope limited to speech inference + turn orchestration.
"""

from __future__ import annotations

import base64
import json
import logging
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from vox.conversation import TurnPolicy
from vox.conversation.session import ConversationConfig, ConversationSession
from vox.streaming.types import TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/v1/conversation")
async def conversation_ws(websocket: WebSocket) -> None:
    await websocket.accept()
    scheduler = websocket.app.state.scheduler

    session: ConversationSession | None = None

    async def emit(event: dict) -> None:

        with suppress(Exception):
            await websocket.send_json(event)

    try:
        while True:
            raw = await websocket.receive()
            if raw.get("type") == "websocket.disconnect":
                break
            if "text" not in raw or raw["text"] is None:
                await _send_error(websocket, "only JSON text frames are supported")
                continue

            try:
                msg = json.loads(raw["text"])
            except json.JSONDecodeError as exc:
                await _send_error(websocket, f"invalid JSON: {exc}")
                continue

            msg_type = msg.get("type")
            if not msg_type:
                await _send_error(websocket, "missing 'type' field")
                continue

            if msg_type == "session.update":
                if session is not None:
                    await _send_error(websocket, "session already configured")
                    continue
                try:
                    config = _parse_session_update(msg)
                except ValueError as exc:
                    await _send_error(websocket, str(exc))
                    continue
                session = ConversationSession(
                    scheduler=scheduler, config=config, on_event=emit,
                )
                await session.start()
                await websocket.send_json({
                    "type": "session.created",
                    "session": _serialize_session_config(config),
                })
                continue

            if session is None:
                await _send_error(websocket, "send session.update first")
                continue

            if msg_type == "input_audio_buffer.append":
                audio_b64 = msg.get("audio")
                if not audio_b64:
                    await _send_error(websocket, "audio field required")
                    continue
                try:
                    pcm = base64.b64decode(audio_b64)
                except Exception as exc:  # noqa: BLE001
                    await _send_error(websocket, f"invalid base64 audio: {exc}")
                    continue
                sample_rate = int(msg.get("sample_rate", 0)) or None
                await session.ingest_audio(pcm, sample_rate=sample_rate)

            elif msg_type == "response.start":
                await session.start_response_stream()

            elif msg_type == "response.delta":
                response = msg.get("response", {}) or {}
                text = response.get("delta") or msg.get("delta")
                if not text:
                    await _send_error(websocket, "response.delta requires 'delta' text")
                    continue
                await session.append_response_text(text)

            elif msg_type == "response.commit":
                await session.commit_response_stream()

            elif msg_type == "response.cancel":
                await session.cancel_response()

            else:
                await _send_error(websocket, f"unknown message type: {msg_type!r}")

    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("conversation WS error")
        with suppress(Exception):
            await _send_error(websocket, "internal error; closing")
    finally:
        if session is not None:
            await session.close()
        with suppress(Exception):
            await websocket.close()


def _parse_session_update(msg: dict) -> ConversationConfig:
    sess = msg.get("session") or msg
    stt_model = sess.get("stt_model") or sess.get("input_audio_transcription", {}).get("model")
    tts_model = sess.get("tts_model") or sess.get("output_audio_generation", {}).get("model")
    if not stt_model:
        raise ValueError("session.update requires 'stt_model'")
    if not tts_model:
        raise ValueError("session.update requires 'tts_model'")

    policy_in = sess.get("turn_policy") or sess.get("policy") or {}
    policy_kwargs = {}
    for field, key in (
        ("allow_interrupt_while_speaking", "allow_interrupt_while_speaking"),
        ("min_interrupt_duration_ms", "min_interrupt_duration_ms"),
        ("max_endpointing_delay_ms", "max_endpointing_delay_ms"),
        ("stable_speaking_min_ms", "stable_speaking_min_ms"),
    ):
        if key in policy_in:
            policy_kwargs[field] = policy_in[key]
    policy = TurnPolicy(**policy_kwargs) if policy_kwargs else TurnPolicy()

    return ConversationConfig(
        stt_model=stt_model,
        tts_model=tts_model,
        voice=sess.get("voice"),
        language=sess.get("language", "en"),
        sample_rate=int(sess.get("sample_rate") or TARGET_SAMPLE_RATE),
        policy=policy,
    )


def _serialize_session_config(config: ConversationConfig) -> dict:
    return {
        "stt_model": config.stt_model,
        "tts_model": config.tts_model,
        "voice": config.voice,
        "language": config.language,
        "sample_rate": config.sample_rate,
        "turn_policy": {
            "allow_interrupt_while_speaking": config.policy.allow_interrupt_while_speaking,
            "min_interrupt_duration_ms": config.policy.min_interrupt_duration_ms,
            "max_endpointing_delay_ms": config.policy.max_endpointing_delay_ms,
            "stable_speaking_min_ms": config.policy.stable_speaking_min_ms,
        },
    }


async def _send_error(websocket: WebSocket, message: str) -> None:
    with suppress(Exception):
        await websocket.send_json({"type": "error", "message": message})
