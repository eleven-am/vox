"""Browser WebRTC signaling for RTC sessions."""

from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from aiortc import RTCConfiguration, RTCIceServer, RTCSessionDescription

from vox.operations.errors import InvalidConfigError
from vox.server.rtc_ice import (
    local_candidate_events,
    parse_browser_ice_candidate,
    rewrite_private_relay_candidates,
    server_ice_servers_from_env,
)
from vox.server.rtc_media import (
    RtcAudioOutputTrack,
    RtcAudioSenderTrack,
    create_rtc_audio_queue,
    emit_media_event,
    pump_input_audio,
)
from vox.server.rtc_registry import RtcSessionRecord, RtcSessionRegistry, track_media_task
from vox.server.rtc_session_io import (
    emit_client_disconnected_to_control,
    flush_pending_client_events,
    handle_browser_data_channel_message_nowait,
    observe_rtc_audio_playout,
)
from vox.webrtc import TrickleRTCPeerConnection

logger = logging.getLogger(__name__)


@dataclass
class _PeerAttachment:
    peer: Any | None
    audio_output_track: Any | None
    input_audio_track: Any | None
    data_channel: Any | None
    browser_attached: bool
    generation: int | None
    attempt_id: str = "active"
    audio_sender_track: Any | None = None

    async def close(self) -> None:
        sender = self.audio_sender_track
        if sender is not None:
            sender.stop()
        peer = self.peer
        if peer is None:
            return
        await peer.close()


@dataclass
class PreparedRtcAnswer:
    session_id: str
    answer_type: str
    sdp: str
    record: RtcSessionRecord
    registry: RtcSessionRegistry
    attachment: _PeerAttachment
    previous: _PeerAttachment
    output_track: RtcAudioOutputTrack
    finished: bool = False

    async def commit(self) -> None:
        if self.finished:
            return
        if not _attachment_is_owned(self.record, self.attachment):
            await self.rollback()
            raise RuntimeError("RTC session changed before offer commit")
        if self.previous.peer is not None and not _attachment_is_active(self.record, self.previous):
            await self.rollback()
            raise RuntimeError("RTC session changed before offer commit")
        previous_sender = self.previous.audio_sender_track
        if isinstance(previous_sender, RtcAudioSenderTrack):
            previous_sender.deactivate()
        sender = self.attachment.audio_sender_track
        if isinstance(sender, RtcAudioSenderTrack):
            sender.activate()
        _activate_peer_attachment(self.record, self.attachment)
        self.output_track.set_playout_observer(
            lambda pcm16, sample_rate: _observe_current_playout(
                self.record,
                self.attachment.peer,
                self.attachment.generation,
                pcm16,
                sample_rate,
            )
        )
        channel = self.attachment.data_channel
        if channel is not None and getattr(channel, "readyState", None) == "open":
            flush_pending_client_events(self.record)
        self.finished = True
        if self.previous.peer is not None and self.previous.peer is not self.attachment.peer:
            _retire_attachment(self.record, self.previous)
            track_media_task(
                self.record,
                _close_retired_attachment(self.record, self.previous, self.registry),
            )
            try:
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()

    async def rollback(self) -> None:
        if self.finished:
            return
        if not _attachment_is_owned(self.record, self.attachment):
            self.finished = True
            return
        restored = _restore_peer_attachment(self.record, self.attachment, self.previous)
        sender = self.attachment.audio_sender_track
        if isinstance(sender, RtcAudioSenderTrack):
            sender.deactivate()
        if restored and self.previous.audio_output_track is self.output_track:
            previous_peer = self.previous.peer
            previous_generation = self.previous.generation
            self.output_track.set_playout_observer(
                lambda pcm16, sample_rate: _observe_current_playout(
                    self.record,
                    previous_peer,
                    previous_generation,
                    pcm16,
                    sample_rate,
                )
            )
        self.finished = True
        _retire_attachment(self.record, self.attachment)
        track_media_task(
            self.record,
            _close_retired_attachment(self.record, self.attachment, self.registry),
        )
        await asyncio.sleep(0)


def _retire_attachment(record: RtcSessionRecord, attachment: _PeerAttachment) -> None:
    if all(retired is not attachment for retired in record.retired_rtc_attachments):
        record.retired_rtc_attachments.append(attachment)


def _release_retired_attachment(record: RtcSessionRecord, attachment: _PeerAttachment) -> None:
    record.retired_rtc_attachments[:] = [
        retired for retired in record.retired_rtc_attachments if retired is not attachment
    ]


async def _close_retired_attachment(
    record: RtcSessionRecord,
    attachment: _PeerAttachment,
    registry: RtcSessionRegistry,
) -> None:
    try:
        await attachment.close()
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("RTC peer retirement failed for session %s", record.session_id)
        registry.close_record(record)
    else:
        _release_retired_attachment(record, attachment)


async def _cleanup_failed_attachment(
    record: RtcSessionRecord,
    attachment: _PeerAttachment,
    registry: RtcSessionRegistry,
) -> None:
    _retire_attachment(record, attachment)
    cleanup_task = asyncio.create_task(
        _close_retired_attachment(record, attachment, registry),
    )
    current = asyncio.current_task()
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            if current is not None:
                current.uncancel()
    cleanup_task.result()


async def create_browser_rtc_answer(
    *,
    registry: RtcSessionRegistry,
    record: RtcSessionRecord,
    offer: dict[str, Any],
    restart: bool = False,
    generation: int | None = None,
) -> PreparedRtcAnswer:
    session_id = record.session_id
    previous = _capture_peer_attachment(record)
    previous_peer = previous.peer
    if previous_peer is not None and not restart:
        raise RuntimeError("RTC browser is already attached")
    if restart and record.retired_rtc_attachments:
        raise RuntimeError("previous RTC peer is still closing")
    if restart and record.pending_rtc_attachment is not None:
        raise RuntimeError("RTC negotiation is already pending")
    pc = TrickleRTCPeerConnection(configuration=rtc_configuration(server_ice_servers_from_env()))
    if record.audio_output is None:
        record.audio_output = create_rtc_audio_queue()
    if restart and isinstance(previous.audio_output_track, RtcAudioOutputTrack):
        output_track = previous.audio_output_track
    else:
        output_track = RtcAudioOutputTrack(
            record.audio_output,
            on_playout=lambda pcm16, sample_rate: _observe_current_playout(
                record,
                pc,
                generation,
                pcm16,
                sample_rate,
            ),
        )
    sender_track = RtcAudioSenderTrack(output_track, active=not restart)
    attachment = _PeerAttachment(
        peer=pc,
        audio_output_track=output_track,
        input_audio_track=None,
        data_channel=None,
        browser_attached=True,
        generation=generation,
        attempt_id=secrets.token_urlsafe(12),
        audio_sender_track=sender_track,
    )
    if restart:
        record.pending_rtc_attachment = attachment
    else:
        _activate_peer_attachment(record, attachment)
    pc.addTrack(sender_track)
    bind_peer_connection_handlers(
        record=record,
        session_id=session_id,
        registry=registry,
        peer=pc,
        generation=generation,
        attachment=attachment,
    )

    try:
        await pc.setRemoteDescription(
            RTCSessionDescription(
                sdp=str(offer.get("sdp") or ""),
                type=str(offer.get("type") or "offer"),
            )
        )
        answer = await pc.createAnswer()
    except BaseException:
        _restore_peer_attachment(record, attachment, previous)
        await _cleanup_failed_attachment(record, attachment, registry)
        raise

    if not _attachment_is_owned(record, attachment):
        await _cleanup_failed_attachment(record, attachment, registry)
        raise RuntimeError("RTC session changed during offer exchange")

    answer_sdp = rewrite_private_relay_candidates(answer.sdp)
    prepared = PreparedRtcAnswer(
        session_id=session_id,
        answer_type=answer.type,
        sdp=answer_sdp,
        record=record,
        registry=registry,
        attachment=attachment,
        previous=previous,
        output_track=output_track,
    )
    description_task = track_media_task(
        record,
        _apply_local_description(record, registry, attachment, answer),
    )
    try:
        await asyncio.sleep(0)
        if description_task.done():
            description_task.result()
        return prepared
    except BaseException:
        cleanup_task = asyncio.create_task(
            _rollback_answer_publication(prepared, description_task),
        )
        current = asyncio.current_task()
        while not cleanup_task.done():
            try:
                await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                if current is not None:
                    current.uncancel()
        cleanup_task.result()
        raise


async def _rollback_answer_publication(
    prepared: PreparedRtcAnswer,
    description_task: asyncio.Task[Any],
) -> None:
    if not description_task.done():
        description_task.cancel()
    await asyncio.gather(description_task, return_exceptions=True)
    await prepared.rollback()


def _capture_peer_attachment(record: RtcSessionRecord) -> _PeerAttachment:
    return _PeerAttachment(
        peer=record.rtc_peer,
        audio_output_track=record.audio_output_track,
        input_audio_track=record.input_audio_track,
        data_channel=record.data_channel,
        browser_attached=record.browser_attached,
        generation=record.negotiation_generation,
        attempt_id="active",
        audio_sender_track=record.audio_sender_track,
    )


def _activate_peer_attachment(
    record: RtcSessionRecord,
    attachment: _PeerAttachment,
) -> None:
    record.rtc_peer = attachment.peer
    record.pending_rtc_attachment = None
    record.audio_output_track = attachment.audio_output_track
    record.audio_sender_track = attachment.audio_sender_track
    record.input_audio_track = attachment.input_audio_track
    record.data_channel = attachment.data_channel
    record.browser_attached = attachment.browser_attached
    record.negotiation_generation = attachment.generation


def _restore_peer_attachment(
    record: RtcSessionRecord,
    attachment: _PeerAttachment,
    previous: _PeerAttachment,
) -> bool:
    if record.pending_rtc_attachment is attachment:
        record.pending_rtc_attachment = None
        return _attachment_is_active(record, previous)
    if (
        record.closed
        or record.rtc_peer is not attachment.peer
        or record.negotiation_generation != attachment.generation
    ):
        return False
    _activate_peer_attachment(record, previous)
    return True


async def _apply_local_description(
    record: RtcSessionRecord,
    registry: RtcSessionRegistry,
    attachment: _PeerAttachment,
    answer: RTCSessionDescription,
) -> None:
    if not _attachment_is_owned(record, attachment):
        return
    peer = attachment.peer
    if peer is None:
        return
    try:
        await peer.setLocalDescription(answer)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        if not _attachment_is_owned(record, attachment):
            return
        await _emit_attachment_media_event(
            record,
            attachment,
            {
                "type": "rtc.signaling_error",
                "message": str(exc),
            },
        )
        current = asyncio.current_task()
        if current is not None:
            record.media_tasks.discard(current)
        if _attachment_is_active(record, attachment):
            registry.close(record.session_id)
        elif record.pending_rtc_attachment is attachment:
            record.pending_rtc_attachment = None
            _retire_attachment(record, attachment)
            await _close_retired_attachment(record, attachment, registry)


async def add_browser_rtc_candidate(
    *,
    record: RtcSessionRecord,
    candidate: dict[str, Any],
    peer: Any | None = None,
) -> dict[str, bool]:
    target = record.rtc_peer if peer is None else peer
    if target is None:
        raise RuntimeError("RTC browser is not attached")
    ice = parse_browser_ice_candidate(candidate)
    await target.addIceCandidate(ice)
    return {"ok": True}


def bind_peer_connection_handlers(
    *,
    record: RtcSessionRecord,
    session_id: str,
    registry: RtcSessionRegistry,
    peer: Any,
    generation: int | None,
    attachment: _PeerAttachment | None = None,
) -> None:
    pc = peer
    peer_attachment = attachment or _PeerAttachment(
        peer=peer,
        audio_output_track=record.audio_output_track,
        input_audio_track=record.input_audio_track,
        data_channel=record.data_channel,
        browser_attached=record.browser_attached,
        generation=generation,
        attempt_id="active",
        audio_sender_track=record.audio_sender_track,
    )

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        if not _attachment_is_owned(record, peer_attachment):
            return
        await _emit_attachment_media_event(
            record,
            peer_attachment,
            {
                "type": "rtc.connection_state",
                "state": pc.connectionState,
            },
        )
        if not _attachment_is_owned(record, peer_attachment):
            return
        if pc.connectionState in {"closed", "failed"}:
            if not _attachment_is_active(record, peer_attachment):
                return
            await emit_client_disconnected_to_control(
                record,
                session_id,
                reason=f"peer_connection_{pc.connectionState}",
                connection_state=pc.connectionState,
                ice_connection_state=pc.iceConnectionState,
                data_channel_state=getattr(record.data_channel, "readyState", None),
            )
            if _attachment_is_active(record, peer_attachment):
                registry.close(session_id)

    @pc.on("iceconnectionstatechange")
    async def on_iceconnectionstatechange() -> None:
        if not _attachment_is_owned(record, peer_attachment):
            return
        await _emit_attachment_media_event(
            record,
            peer_attachment,
            {
                "type": "rtc.ice_connection_state",
                "state": pc.iceConnectionState,
            },
        )
        if not _attachment_is_owned(record, peer_attachment):
            return
        if pc.iceConnectionState == "failed":
            if not _attachment_is_active(record, peer_attachment):
                return
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
        if not _attachment_is_owned(record, peer_attachment):
            return
        await _emit_attachment_media_event(
            record,
            peer_attachment,
            {
                "type": "rtc.ice_gathering_state",
                "state": pc.iceGatheringState,
            },
        )

    @pc.on("icecandidate")
    async def on_icecandidate(candidate) -> None:
        if not _attachment_is_owned(record, peer_attachment):
            return
        if candidate is None:
            await _emit_attachment_media_event(
                record,
                peer_attachment,
                {"type": "rtc.ice_candidate", "candidate": None},
            )
            return
        for event in local_candidate_events(candidate):
            if not _attachment_is_owned(record, peer_attachment):
                return
            await _emit_attachment_media_event(record, peer_attachment, event)

    @pc.on("track")
    def on_track(track) -> None:
        if not _attachment_is_owned(record, peer_attachment) or track.kind != "audio":
            with suppress(Exception):
                track.stop()
            return
        if peer_attachment.input_audio_track is not None and peer_attachment.input_audio_track is not track:
            with suppress(Exception):
                track.stop()
            return
        peer_attachment.input_audio_track = track
        if _attachment_is_active(record, peer_attachment):
            record.input_audio_track = track
        track_media_task(record, _pump_attachment_input_audio(record, peer_attachment, track))

    @pc.on("datachannel")
    def on_datachannel(channel) -> None:
        if not _attachment_is_owned(record, peer_attachment):
            with suppress(Exception):
                channel.close()
            return
        peer_attachment.data_channel = channel
        if _attachment_is_active(record, peer_attachment):
            record.data_channel = channel

        @channel.on("open")
        def on_open() -> None:
            if not _attachment_is_active(record, peer_attachment) or record.data_channel is not channel:
                return
            flush_pending_client_events(record)

        @channel.on("close")
        def on_close() -> None:
            if not _attachment_is_owned(record, peer_attachment) or peer_attachment.data_channel is not channel:
                return
            peer_attachment.data_channel = None
            if _attachment_is_active(record, peer_attachment):
                record.data_channel = None
            track_media_task(
                record,
                _emit_current_data_channel_disconnect(
                    record,
                    peer_attachment,
                    channel,
                    session_id,
                ),
            )

        @channel.on("message")
        def on_message(message) -> None:
            if not _attachment_is_active(record, peer_attachment) or record.data_channel is not channel:
                return
            try:
                handle_browser_data_channel_message_nowait(record, session_id, message)
            except InvalidConfigError:
                with suppress(Exception):
                    channel.close()

        if _attachment_is_active(record, peer_attachment):
            flush_pending_client_events(record)


def _peer_is_current(record: RtcSessionRecord, peer: Any, generation: int | None) -> bool:
    return not record.closed and record.rtc_peer is peer and record.negotiation_generation == generation


def _attachment_is_active(record: RtcSessionRecord, attachment: _PeerAttachment) -> bool:
    return attachment.peer is not None and _peer_is_current(record, attachment.peer, attachment.generation)


def _attachment_is_owned(record: RtcSessionRecord, attachment: _PeerAttachment) -> bool:
    return _attachment_is_active(record, attachment) or (
        not record.closed and record.pending_rtc_attachment is attachment
    )


def _observe_current_playout(
    record: RtcSessionRecord,
    peer: Any,
    generation: int | None,
    pcm16: bytes,
    sample_rate: int,
) -> None:
    if _peer_is_current(record, peer, generation):
        observe_rtc_audio_playout(record, pcm16, sample_rate)


async def _pump_attachment_input_audio(
    record: RtcSessionRecord,
    attachment: _PeerAttachment,
    track: Any,
) -> None:
    async def ingest(pcm16: bytes, sample_rate: int | None) -> None:
        if _attachment_is_active(record, attachment) and record.input_audio_track is track:
            await ingest_media_audio(record, pcm16, sample_rate)

    try:
        await pump_input_audio(track, ingest)
    finally:
        if _attachment_is_active(record, attachment) and record.input_audio_track is track:
            record.input_audio_track = None
        if attachment.input_audio_track is track:
            attachment.input_audio_track = None


async def _emit_current_data_channel_disconnect(
    record: RtcSessionRecord,
    attachment: _PeerAttachment,
    channel: Any,
    session_id: str,
) -> None:
    if not _attachment_is_active(record, attachment):
        return
    peer = attachment.peer
    if peer is None:
        return
    await emit_client_disconnected_to_control(
        record,
        session_id,
        reason="data_channel_closed",
        connection_state=peer.connectionState,
        ice_connection_state=peer.iceConnectionState,
        data_channel_state=getattr(channel, "readyState", None),
    )


async def _emit_attachment_media_event(
    record: RtcSessionRecord,
    attachment: _PeerAttachment,
    event: dict[str, Any],
) -> None:
    event["_rtc_attempt_id"] = attachment.attempt_id
    await emit_media_event(
        record,
        event,
        generation=attachment.generation,
    )


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
