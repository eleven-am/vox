"""Browser WebRTC signaling for RTC sessions."""

from __future__ import annotations

import asyncio
from typing import Any

from aiortc import RTCConfiguration, RTCIceServer, RTCSessionDescription

from vox.server.rtc_ice import (
    local_candidate_events,
    parse_browser_ice_candidate,
    rewrite_private_relay_candidates,
    server_ice_servers_from_env,
)
from vox.server.rtc_media import (
    RtcAudioOutputTrack,
    create_rtc_audio_queue,
    emit_media_event,
    pump_input_audio,
)
from vox.server.rtc_registry import RtcSessionRecord, RtcSessionRegistry, track_media_task
from vox.server.rtc_session_io import (
    emit_client_disconnected_to_control,
    flush_pending_client_events,
    handle_browser_data_channel_message,
    observe_rtc_audio_playout,
)
from vox.webrtc import TrickleRTCPeerConnection


async def create_browser_rtc_answer(
    *,
    registry: RtcSessionRegistry,
    record: RtcSessionRecord,
    offer: dict[str, Any],
) -> dict[str, Any]:
    session_id = record.session_id

    pc = TrickleRTCPeerConnection(configuration=rtc_configuration(server_ice_servers_from_env()))
    record.rtc_peer = pc
    if record.audio_output is None:
        record.audio_output = create_rtc_audio_queue()
    record.audio_output_track = RtcAudioOutputTrack(
        record.audio_output,
        on_playout=lambda pcm16, sample_rate: observe_rtc_audio_playout(
            record,
            pcm16,
            sample_rate,
        ),
    )
    pc.addTrack(record.audio_output_track)

    bind_peer_connection_handlers(record=record, session_id=session_id, registry=registry)

    await pc.setRemoteDescription(
        RTCSessionDescription(
            sdp=str(offer.get("sdp") or ""),
            type=str(offer.get("type") or "offer"),
        )
    )
    answer = await pc.createAnswer()
    track_media_task(record, _apply_local_description(record, registry, answer))
    await asyncio.sleep(0)
    answer_sdp = rewrite_private_relay_candidates(answer.sdp)

    return {
        "session_id": session_id,
        "type": answer.type,
        "sdp": answer_sdp,
    }


async def _apply_local_description(
    record: RtcSessionRecord,
    registry: RtcSessionRegistry,
    answer: RTCSessionDescription,
) -> None:
    try:
        await record.rtc_peer.setLocalDescription(answer)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        await emit_media_event(
            record,
            {
                "type": "rtc.signaling_error",
                "message": str(exc),
            },
        )
        current = asyncio.current_task()
        if current is not None:
            record.media_tasks.discard(current)
        registry.close(record.session_id)


async def add_browser_rtc_candidate(
    *,
    record: RtcSessionRecord,
    candidate: dict[str, Any],
) -> dict[str, bool]:
    if record.rtc_peer is None:
        raise RuntimeError("RTC browser is not attached")
    ice = parse_browser_ice_candidate(candidate)
    await record.rtc_peer.addIceCandidate(ice)
    return {"ok": True}


def bind_peer_connection_handlers(
    *,
    record: RtcSessionRecord,
    session_id: str,
    registry: RtcSessionRegistry,
) -> None:
    pc = record.rtc_peer
    if pc is None:
        return

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        await emit_media_event(
            record,
            {
                "type": "rtc.connection_state",
                "state": pc.connectionState,
            },
        )
        if pc.connectionState in {"closed", "failed"}:
            if pc.connectionState == "closed" and record.ice_restart_in_progress:
                return
            await emit_client_disconnected_to_control(
                record,
                session_id,
                reason=f"peer_connection_{pc.connectionState}",
                connection_state=pc.connectionState,
                ice_connection_state=pc.iceConnectionState,
                data_channel_state=getattr(record.data_channel, "readyState", None),
            )
            registry.close(session_id)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:
        await emit_media_event(
            record,
            {
                "type": "rtc.ice_connection_state",
                "state": pc.iceConnectionState,
            },
        )
        if pc.iceConnectionState == "failed":
            await emit_client_disconnected_to_control(
                record,
                session_id,
                reason="ice_connection_failed",
                connection_state=pc.connectionState,
                ice_connection_state=pc.iceConnectionState,
                data_channel_state=getattr(record.data_channel, "readyState", None),
            )

    @pc.on("icegatheringstatechange")
    async def on_icegatheringstatechange() -> None:
        await emit_media_event(
            record,
            {
                "type": "rtc.ice_gathering_state",
                "state": pc.iceGatheringState,
            },
        )

    @pc.on("icecandidate")
    async def on_icecandidate(candidate) -> None:
        if candidate is None:
            await emit_media_event(record, {"type": "rtc.ice_candidate", "candidate": None})
            return
        for event in local_candidate_events(candidate):
            await emit_media_event(record, event)

    @pc.on("track")
    def on_track(track) -> None:
        if track.kind == "audio":
            track_media_task(
                record,
                pump_input_audio(track, lambda pcm, sr: ingest_media_audio(record, pcm, sr)),
            )

    @pc.on("datachannel")
    def on_datachannel(channel) -> None:
        record.data_channel = channel

        @channel.on("open")
        def on_open() -> None:
            flush_pending_client_events(record)

        @channel.on("close")
        def on_close() -> None:
            if record.data_channel is channel:
                record.data_channel = None
            track_media_task(
                record,
                emit_client_disconnected_to_control(
                    record,
                    session_id,
                    reason="data_channel_closed",
                    connection_state=pc.connectionState,
                    ice_connection_state=pc.iceConnectionState,
                    data_channel_state=getattr(channel, "readyState", None),
                ),
            )

        @channel.on("message")
        def on_message(message) -> None:
            track_media_task(
                record,
                handle_browser_data_channel_message(record, session_id, message),
            )

        flush_pending_client_events(record)


def rtc_configuration(ice_servers: list[dict]) -> RTCConfiguration:
    return RTCConfiguration(
        iceServers=[
            RTCIceServer(
                urls=server["urls"],
                username=server.get("username"),
                credential=server.get("credential"),
            )
            for server in ice_servers
        ]
    )


async def ingest_media_audio(record: RtcSessionRecord, pcm16: bytes, sample_rate: int | None) -> None:
    orchestrator = record.orchestrator
    if orchestrator is not None and orchestrator.config is not None:
        await orchestrator.ingest_pcm16(pcm16, sample_rate=sample_rate)
