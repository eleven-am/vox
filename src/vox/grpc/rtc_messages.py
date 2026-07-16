"""Protobuf mapping for RTC session and control operations."""

from __future__ import annotations

import json
from typing import Any

from vox.grpc import vox_pb2
from vox.grpc.conversation_events import conversation_wire_event_to_pb
from vox.operations.conversation import (
    control_event_as_client_event,
    control_event_client_payload_json,
    conversation_wire_event_payload,
)
from vox.operations.rtc_signaling import (
    RtcSessionBootstrap,
    RtcSessionBootstrapRequest,
)


def rtc_create_session_request_from_pb(
    request: vox_pb2.RtcCreateSessionRequest,
) -> RtcSessionBootstrapRequest:
    browser_events = request.browser_events if request.HasField("browser_events") else None
    return RtcSessionBootstrapRequest(
        control_transport="grpc",
        forward_browser_events=browser_events,
    )


def rtc_session_bootstrap_pb(result: RtcSessionBootstrap) -> vox_pb2.RtcSessionBootstrap:
    return vox_pb2.RtcSessionBootstrap(
        session_id=result.session_id,
        expires_at=result.expires_at,
        attach_ttl_seconds=result.attach_ttl_seconds,
        ice_servers=[_rtc_ice_server_pb(server) for server in result.ice_servers],
    )


def _rtc_ice_server_pb(server: dict[str, Any]) -> vox_pb2.RtcIceServer:
    urls = server.get("urls", [])
    if isinstance(urls, str):
        urls = [urls]
    return vox_pb2.RtcIceServer(
        urls=[str(url) for url in urls],
        username=str(server.get("username") or ""),
        credential=str(server.get("credential") or ""),
    )


def rtc_runtime_event_pb(event: dict[str, Any]) -> vox_pb2.RtcControlServerMessage:
    event_type = str(event.get("type") or "")
    if event_type == "rtc.session.attached":
        return vox_pb2.RtcControlServerMessage(
            attached=vox_pb2.RtcSessionAttached(
                session_id=str(event.get("session_id") or ""),
                provider=str(event.get("provider") or "grpc"),
            )
        )
    if event_type == "rtc.answer":
        answer = event.get("answer") if isinstance(event.get("answer"), dict) else {}
        return vox_pb2.RtcControlServerMessage(
            answer=vox_pb2.RtcControlAnswer(
                session_id=str(event.get("session_id") or ""),
                answer=vox_pb2.RtcSessionDescription(
                    type=str(answer.get("type") or "answer"),
                    sdp=str(answer.get("sdp") or ""),
                ),
            )
        )
    if event_type == "rtc.ice_candidate":
        candidate = event.get("candidate")
        if candidate is None:
            return vox_pb2.RtcControlServerMessage(candidates_complete=vox_pb2.RtcIceCandidatesComplete())
        return vox_pb2.RtcControlServerMessage(candidate=_rtc_candidate_pb(candidate))
    if event_type == "rtc.signaling_error":
        return rtc_error_pb(str(event.get("message") or "RTC signaling failed"))
    if event_type == "rtc.session.closed":
        return vox_pb2.RtcControlServerMessage(
            closed=vox_pb2.RtcControlClosed(
                session_id=str(event.get("session_id") or ""),
                reason=str(event.get("reason") or ""),
            )
        )
    if event_type == "browser.event":
        event_name, payload = control_event_as_client_event(event)
        return vox_pb2.RtcControlServerMessage(
            browser_event=vox_pb2.RtcClientEvent(
                event=event_name,
                payload_json=control_event_client_payload_json(payload),
            )
        )

    conversation = conversation_wire_event_to_pb(event)
    if conversation is not None:
        return vox_pb2.RtcControlServerMessage(conversation=conversation)
    event_name, payload = conversation_wire_event_payload(event)
    return vox_pb2.RtcControlServerMessage(
        event=vox_pb2.RtcWireEvent(
            type=event_name,
            payload_json=json.dumps(payload, separators=(",", ":")),
        )
    )


def rtc_error_pb(message: str) -> vox_pb2.RtcControlServerMessage:
    return vox_pb2.RtcControlServerMessage(error=vox_pb2.RtcSignalingError(message=message))


def _rtc_candidate_pb(candidate: Any) -> vox_pb2.RtcIceCandidate:
    if not isinstance(candidate, dict):
        candidate = {"candidate": str(candidate)}
    result = vox_pb2.RtcIceCandidate(candidate=str(candidate.get("candidate") or ""))
    sdp_mid = candidate.get("sdpMid", candidate.get("sdp_mid"))
    if sdp_mid is not None:
        result.sdp_mid = str(sdp_mid)
    sdp_m_line_index = candidate.get("sdpMLineIndex", candidate.get("sdp_m_line_index"))
    if sdp_m_line_index is not None:
        result.sdp_m_line_index = int(sdp_m_line_index)
    username_fragment = candidate.get("usernameFragment", candidate.get("username_fragment"))
    if username_fragment is not None:
        result.username_fragment = str(username_fragment)
    return result
