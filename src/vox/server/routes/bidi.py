from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket

from vox.operations.streaming_synthesis_longform import (
    LongformSynthesisSession,
    TtsAudioChunkEvent,
    TtsDoneEvent,
    TtsErrorEvent,
    longform_tts_event_payload,
    normalize_longform_tts_config,
)
from vox.operations.streaming_transcription_longform import (
    LongformDoneEvent,
    LongformErrorEvent,
    LongformTranscriptionSession,
    longform_transcription_event_payload,
    normalize_longform_config,
)
from vox.server.app_services import app_services
from vox.server.websocket import (
    WsBytesFrame,
    WsDisconnectFrame,
    WsTextFrame,
    emit_ws_session_events,
    receive_required_ws_config,
    receive_ws_frame,
    send_ws_error,
    send_ws_unknown_message_type,
    websocket_connection_scope,
    websocket_route_error_scope,
    websocket_session_event_scope,
    ws_operation_result_or_error,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/v1/audio/transcriptions/stream")
async def transcriptions_stream(websocket: WebSocket):
    async with websocket_connection_scope(websocket):
        logger.info("long-form STT ws connected")

        async with websocket_route_error_scope(
            websocket,
            logger=logger,
            disconnect_log_message="Long-form STT websocket disconnected",
            error_log_message="Long-form STT websocket error",
            closed_log_message="long-form STT ws closed",
        ):
            services = app_services(websocket)
            config_msg = await receive_required_ws_config(websocket, "audio")
            if config_msg is None:
                return
            config = await ws_operation_result_or_error(
                websocket,
                lambda: normalize_longform_config(
                    model=config_msg.get("model", "") or "",
                    sample_rate=int(config_msg.get("sample_rate") or 0) or None,
                    input_format=config_msg.get("input_format"),
                    language=config_msg.get("language"),
                    word_timestamps=bool(config_msg.get("word_timestamps", False)),
                    temperature=float(config_msg.get("temperature", 0.0) or 0.0),
                    chunk_ms=config_msg.get("chunk_ms"),
                    overlap_ms=config_msg.get("overlap_ms"),
                    registry=services.registry,
                    store=services.store,
                ),
            )
            if config is None:
                return

            session = LongformTranscriptionSession(
                scheduler=services.scheduler,
                registry=services.registry,
                store=services.store,
            )
            async with websocket_session_event_scope(
                session,
                lambda: _emit_longform_stt_events(websocket, session),
            ):
                if not await session.configure_or_report(config):
                    return

                while True:
                    frame = await receive_ws_frame(websocket)
                    if frame is None:
                        continue
                    if isinstance(frame, WsDisconnectFrame):
                        return
                    if isinstance(frame, WsTextFrame):
                        data = frame.message
                        msg_type = data.get("type", "")
                        if msg_type == "end":
                            break
                        await send_ws_unknown_message_type(websocket, msg_type)
                        continue
                    if isinstance(frame, WsBytesFrame) and not await session.submit_chunk_or_report(frame.data):
                        return

                await session.end_of_stream_or_report()


@router.websocket("/v1/audio/speech/stream")
async def speech_stream(websocket: WebSocket):
    async with websocket_connection_scope(websocket):
        logger.info("long-form TTS ws connected")

        async with websocket_route_error_scope(
            websocket,
            logger=logger,
            disconnect_log_message="Long-form TTS websocket disconnected",
            error_log_message="Long-form TTS websocket error",
            closed_log_message="long-form TTS ws closed",
        ):
            services = app_services(websocket)
            config_msg = await receive_required_ws_config(websocket, "text input")
            if config_msg is None:
                return
            config = await ws_operation_result_or_error(
                websocket,
                lambda: normalize_longform_tts_config(
                    model=config_msg.get("model", "") or "",
                    voice=config_msg.get("voice"),
                    speed=float(config_msg.get("speed", 1.0) or 1.0),
                    language=config_msg.get("language"),
                    response_format=config_msg.get("response_format"),
                    chunk_chars=config_msg.get("chunk_chars"),
                    registry=services.registry,
                    store=services.store,
                ),
            )
            if config is None:
                return

            session = LongformSynthesisSession(
                scheduler=services.scheduler,
                registry=services.registry,
                store=services.store,
            )
            async with websocket_session_event_scope(
                session,
                lambda: _emit_longform_tts_events(websocket, session),
                drain_timeout=10.0,
            ):
                if not await session.configure_or_report(config):
                    return

                while True:
                    frame = await receive_ws_frame(websocket)
                    if frame is None:
                        continue
                    if isinstance(frame, WsDisconnectFrame):
                        return
                    if not isinstance(frame, WsTextFrame):
                        await send_ws_error(websocket, "Binary messages are not supported for TTS input")
                        continue
                    data = frame.message
                    msg_type = data.get("type", "")
                    if msg_type == "text":
                        chunk = data.get("text", "")
                        if chunk:
                            session.append_text(chunk)
                        continue
                    if msg_type == "end":
                        break
                    await send_ws_unknown_message_type(websocket, msg_type)

                await session.end_of_stream_or_report()


async def _emit_longform_stt_events(websocket: WebSocket, session: LongformTranscriptionSession) -> None:
    await emit_ws_session_events(
        websocket,
        session.events(),
        json_payload=longform_transcription_event_payload,
        terminal_types=(LongformDoneEvent, LongformErrorEvent),
    )


async def _emit_longform_tts_events(websocket: WebSocket, session: LongformSynthesisSession) -> None:
    await emit_ws_session_events(
        websocket,
        session.events(),
        json_payload=longform_tts_event_payload,
        binary_payload=_longform_tts_binary_payload,
        terminal_types=(TtsDoneEvent, TtsErrorEvent),
    )


def _longform_tts_binary_payload(event) -> bytes | None:
    if isinstance(event, TtsAudioChunkEvent):
        return event.data
    return None
