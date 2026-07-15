"""Browser WebRTC signaling for RTC sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription

from vox.server.rtc_client_events import (
    emit_client_disconnected_to_control,
    flush_pending_client_events,
    handle_browser_data_channel_message,
)
from vox.server.rtc_conversation import observe_rtc_audio_playout
from vox.server.rtc_ice import (
    InvalidIceCandidateError,
    candidate_events_from_sdp,
    parse_browser_ice_candidate,
    patch_aioice_turn_error_code_parser,
    rewrite_private_relay_candidates,
    server_ice_servers_from_env,
)
from vox.server.rtc_media import RtcAudioOutputTrack, create_rtc_audio_queue, pump_input_audio
from vox.server.rtc_media_events import emit_media_event
from vox.server.rtc_registry import RtcSessionRecord, RtcSessionRegistry
from vox.server.rtc_tasks import track_media_task


@dataclass(frozen=True)
class RtcSignalingError(Exception):
    status_code: int
    detail: str


async def create_browser_rtc_answer(
    *,
    registry: RtcSessionRegistry,
    session_id: str,
    client_token: str,
    offer: dict[str, Any],
) -> dict[str, Any]:
    attached = registry.attach_browser(client_token)
    if attached is None:
        raise RtcSignalingError(status_code=401, detail="invalid or expired client token")
    record, media_token = attached
    if record.session_id != session_id:
        registry.close(record.session_id)
        raise RtcSignalingError(status_code=404, detail="RTC session not found")

    patch_aioice_turn_error_code_parser()
    pc = RTCPeerConnection(configuration=rtc_configuration(server_ice_servers_from_env()))
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
    await pc.setLocalDescription(answer)
    answer_sdp = rewrite_private_relay_candidates(pc.localDescription.sdp)
    for event in candidate_events_from_sdp(answer_sdp):
        await emit_media_event(record, event)
    await emit_media_event(record, {"type": "rtc.ice_candidate", "candidate": None})

    return {
        "session_id": session_id,
        "media_token": media_token,
        "events_url": f"/v1/rtc/sessions/{session_id}/events?token={media_token}",
        "type": pc.localDescription.type,
        "sdp": answer_sdp,
    }


async def add_browser_rtc_candidate(
    *,
    registry: RtcSessionRegistry,
    session_id: str,
    media_token: str,
    candidate: dict[str, Any],
) -> dict[str, bool]:
    record = registry.validate_media_token(session_id, media_token)
    if record is None or record.rtc_peer is None:
        raise RtcSignalingError(status_code=401, detail="invalid RTC media token")

    try:
        ice = parse_browser_ice_candidate(candidate)
    except InvalidIceCandidateError as exc:
        raise RtcSignalingError(status_code=400, detail="invalid ICE candidate") from exc

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
