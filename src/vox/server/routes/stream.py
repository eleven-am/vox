from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket

from vox.operations.errors import UnknownMessageTypeError
from vox.operations.streaming_transcription import (
    DoneEvent,
    StreamingTranscriptionSession,
    streaming_transcription_config_from_fields,
    streaming_transcription_event_payload,
)
from vox.server.app_services import app_services
from vox.server.websocket import (
    WsBytesFrame,
    WsDisconnectFrame,
    WsTextFrame,
    emit_ws_session_events,
    receive_ws_frame,
    websocket_connection_scope,
    websocket_route_error_scope,
    websocket_session_event_scope,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/v1/audio/stream")
async def audio_stream(websocket: WebSocket):
    async with websocket_connection_scope(websocket):
        logger.info("realtime STT ws connected")

        services = app_services(websocket)

        session = StreamingTranscriptionSession(
            scheduler=services.scheduler,
            registry=services.registry,
            store=services.store,
            speech_context_service=services.speech_context,
        )
        async def emit_events() -> None:
            await emit_ws_session_events(
                websocket,
                session.events(),
                json_payload=streaming_transcription_event_payload,
                terminal_types=(DoneEvent,),
                suppress_send_errors=True,
            )

        async with websocket_route_error_scope(
            websocket,
            logger=logger,
            disconnect_log_message="WS stream client disconnected",
            error_log_message="WS stream error",
            closed_log_message="realtime STT ws closed",
        ), websocket_session_event_scope(session, emit_events):
            while True:
                frame = await receive_ws_frame(websocket)
                if frame is None:
                    continue

                if isinstance(frame, WsDisconnectFrame):
                    return

                if isinstance(frame, WsTextFrame):
                    data = frame.message
                    msg_type = data.get("type", "")

                    if msg_type == "config":
                        await session.configure_or_report(
                            streaming_transcription_config_from_fields(
                                model=data.get("model", "") or "",
                                language=data.get("language", "en") or "en",
                                sample_rate=data.get("sample_rate") or 16_000,
                                partials=data.get("partials", False),
                                partial_window_ms=data.get("partial_window_ms") or 1500,
                                partial_stride_ms=data.get("partial_stride_ms") or 700,
                                include_word_timestamps=data.get(
                                    "include_word_timestamps",
                                    False,
                                ),
                                temperature=data.get("temperature", 0.0) or 0.0,
                                speech_context=data.get("speech_context", False),
                            )
                        )
                        continue

                    if msg_type == "end":
                        break

                    await session.report_error(str(UnknownMessageTypeError(msg_type)))
                    continue

                if isinstance(frame, WsBytesFrame):
                    await session.submit_pcm16_or_report(frame.data)

            await session.end_of_stream()
