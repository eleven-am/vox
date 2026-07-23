from google.protobuf import struct_pb2 as _struct_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HealthRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: str
    def __init__(self, status: _Optional[str] = ...) -> None: ...

class ListLoadedRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListLoadedResponse(_message.Message):
    __slots__ = ("models",)
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[LoadedModel]
    def __init__(self, models: _Optional[_Iterable[_Union[LoadedModel, _Mapping]]] = ...) -> None: ...

class LoadedModel(_message.Message):
    __slots__ = ("name", "tag", "type", "device", "vram_bytes", "loaded_at", "last_used", "ref_count")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TAG_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    VRAM_BYTES_FIELD_NUMBER: _ClassVar[int]
    LOADED_AT_FIELD_NUMBER: _ClassVar[int]
    LAST_USED_FIELD_NUMBER: _ClassVar[int]
    REF_COUNT_FIELD_NUMBER: _ClassVar[int]
    name: str
    tag: str
    type: str
    device: str
    vram_bytes: int
    loaded_at: float
    last_used: float
    ref_count: int
    def __init__(self, name: _Optional[str] = ..., tag: _Optional[str] = ..., type: _Optional[str] = ..., device: _Optional[str] = ..., vram_bytes: _Optional[int] = ..., loaded_at: _Optional[float] = ..., last_used: _Optional[float] = ..., ref_count: _Optional[int] = ...) -> None: ...

class PullRequest(_message.Message):
    __slots__ = ("name", "variant")
    NAME_FIELD_NUMBER: _ClassVar[int]
    VARIANT_FIELD_NUMBER: _ClassVar[int]
    name: str
    variant: str
    def __init__(self, name: _Optional[str] = ..., variant: _Optional[str] = ...) -> None: ...

class PullProgress(_message.Message):
    __slots__ = ("status", "error", "completed", "total")
    STATUS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    COMPLETED_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    status: str
    error: str
    completed: int
    total: int
    def __init__(self, status: _Optional[str] = ..., error: _Optional[str] = ..., completed: _Optional[int] = ..., total: _Optional[int] = ...) -> None: ...

class ListModelsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListModelsResponse(_message.Message):
    __slots__ = ("models",)
    MODELS_FIELD_NUMBER: _ClassVar[int]
    models: _containers.RepeatedCompositeFieldContainer[ModelInfo]
    def __init__(self, models: _Optional[_Iterable[_Union[ModelInfo, _Mapping]]] = ...) -> None: ...

class ModelInfo(_message.Message):
    __slots__ = ("name", "type", "format", "architecture", "size_bytes", "description")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    ARCHITECTURE_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    format: str
    architecture: str
    size_bytes: int
    description: str
    def __init__(self, name: _Optional[str] = ..., type: _Optional[str] = ..., format: _Optional[str] = ..., architecture: _Optional[str] = ..., size_bytes: _Optional[int] = ..., description: _Optional[str] = ...) -> None: ...

class ShowRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class ShowResponse(_message.Message):
    __slots__ = ("name", "config", "layers")
    class ConfigEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    NAME_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    LAYERS_FIELD_NUMBER: _ClassVar[int]
    name: str
    config: _containers.ScalarMap[str, str]
    layers: _containers.RepeatedCompositeFieldContainer[LayerInfo]
    def __init__(self, name: _Optional[str] = ..., config: _Optional[_Mapping[str, str]] = ..., layers: _Optional[_Iterable[_Union[LayerInfo, _Mapping]]] = ...) -> None: ...

class LayerInfo(_message.Message):
    __slots__ = ("media_type", "digest", "size", "filename")
    MEDIA_TYPE_FIELD_NUMBER: _ClassVar[int]
    DIGEST_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    FILENAME_FIELD_NUMBER: _ClassVar[int]
    media_type: str
    digest: str
    size: int
    filename: str
    def __init__(self, media_type: _Optional[str] = ..., digest: _Optional[str] = ..., size: _Optional[int] = ..., filename: _Optional[str] = ...) -> None: ...

class DeleteRequest(_message.Message):
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class DeleteResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: str
    def __init__(self, status: _Optional[str] = ...) -> None: ...

class TranscribeRequest(_message.Message):
    __slots__ = ("audio", "model", "language", "word_timestamps", "temperature", "response_format", "format_hint", "speech_context")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    WORD_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    FORMAT_HINT_FIELD_NUMBER: _ClassVar[int]
    SPEECH_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    audio: bytes
    model: str
    language: str
    word_timestamps: bool
    temperature: float
    response_format: str
    format_hint: str
    speech_context: bool
    def __init__(self, audio: _Optional[bytes] = ..., model: _Optional[str] = ..., language: _Optional[str] = ..., word_timestamps: bool = ..., temperature: _Optional[float] = ..., response_format: _Optional[str] = ..., format_hint: _Optional[str] = ..., speech_context: bool = ...) -> None: ...

class TranscribeResponse(_message.Message):
    __slots__ = ("model", "text", "language", "duration_ms", "processing_ms", "segments", "entities", "topics", "speech_context")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_MS_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    TOPICS_FIELD_NUMBER: _ClassVar[int]
    SPEECH_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    model: str
    text: str
    language: str
    duration_ms: int
    processing_ms: int
    segments: _containers.RepeatedCompositeFieldContainer[TranscriptSegment]
    entities: _containers.RepeatedCompositeFieldContainer[Entity]
    topics: _containers.RepeatedScalarFieldContainer[str]
    speech_context: _struct_pb2.Struct
    def __init__(self, model: _Optional[str] = ..., text: _Optional[str] = ..., language: _Optional[str] = ..., duration_ms: _Optional[int] = ..., processing_ms: _Optional[int] = ..., segments: _Optional[_Iterable[_Union[TranscriptSegment, _Mapping]]] = ..., entities: _Optional[_Iterable[_Union[Entity, _Mapping]]] = ..., topics: _Optional[_Iterable[str]] = ..., speech_context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class Entity(_message.Message):
    __slots__ = ("type", "text", "start_char", "end_char")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    START_CHAR_FIELD_NUMBER: _ClassVar[int]
    END_CHAR_FIELD_NUMBER: _ClassVar[int]
    type: str
    text: str
    start_char: int
    end_char: int
    def __init__(self, type: _Optional[str] = ..., text: _Optional[str] = ..., start_char: _Optional[int] = ..., end_char: _Optional[int] = ...) -> None: ...

class AnnotateRequest(_message.Message):
    __slots__ = ("text", "language")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    text: str
    language: str
    def __init__(self, text: _Optional[str] = ..., language: _Optional[str] = ...) -> None: ...

class AnnotateResponse(_message.Message):
    __slots__ = ("entities", "topics")
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    TOPICS_FIELD_NUMBER: _ClassVar[int]
    entities: _containers.RepeatedCompositeFieldContainer[Entity]
    topics: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, entities: _Optional[_Iterable[_Union[Entity, _Mapping]]] = ..., topics: _Optional[_Iterable[str]] = ...) -> None: ...

class TranscriptSegment(_message.Message):
    __slots__ = ("text", "start_ms", "end_ms", "words")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    START_MS_FIELD_NUMBER: _ClassVar[int]
    END_MS_FIELD_NUMBER: _ClassVar[int]
    WORDS_FIELD_NUMBER: _ClassVar[int]
    text: str
    start_ms: int
    end_ms: int
    words: _containers.RepeatedCompositeFieldContainer[WordTimestamp]
    def __init__(self, text: _Optional[str] = ..., start_ms: _Optional[int] = ..., end_ms: _Optional[int] = ..., words: _Optional[_Iterable[_Union[WordTimestamp, _Mapping]]] = ...) -> None: ...

class WordTimestamp(_message.Message):
    __slots__ = ("word", "start_ms", "end_ms", "confidence")
    WORD_FIELD_NUMBER: _ClassVar[int]
    START_MS_FIELD_NUMBER: _ClassVar[int]
    END_MS_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    word: str
    start_ms: int
    end_ms: int
    confidence: float
    def __init__(self, word: _Optional[str] = ..., start_ms: _Optional[int] = ..., end_ms: _Optional[int] = ..., confidence: _Optional[float] = ...) -> None: ...

class SynthesizeRequest(_message.Message):
    __slots__ = ("model", "input", "voice", "speed", "language", "response_format", "params")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    VOICE_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    model: str
    input: str
    voice: str
    speed: float
    language: str
    response_format: str
    params: _struct_pb2.Struct
    def __init__(self, model: _Optional[str] = ..., input: _Optional[str] = ..., voice: _Optional[str] = ..., speed: _Optional[float] = ..., language: _Optional[str] = ..., response_format: _Optional[str] = ..., params: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class AudioChunk(_message.Message):
    __slots__ = ("audio", "sample_rate", "is_final")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    IS_FINAL_FIELD_NUMBER: _ClassVar[int]
    audio: bytes
    sample_rate: int
    is_final: bool
    def __init__(self, audio: _Optional[bytes] = ..., sample_rate: _Optional[int] = ..., is_final: bool = ...) -> None: ...

class StreamInput(_message.Message):
    __slots__ = ("config", "audio", "opus_frame", "encoded_audio", "end_of_stream")
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    OPUS_FRAME_FIELD_NUMBER: _ClassVar[int]
    ENCODED_AUDIO_FIELD_NUMBER: _ClassVar[int]
    END_OF_STREAM_FIELD_NUMBER: _ClassVar[int]
    config: StreamConfig
    audio: AudioFrame
    opus_frame: OpusFrame
    encoded_audio: EncodedAudioFrame
    end_of_stream: EndOfStream
    def __init__(self, config: _Optional[_Union[StreamConfig, _Mapping]] = ..., audio: _Optional[_Union[AudioFrame, _Mapping]] = ..., opus_frame: _Optional[_Union[OpusFrame, _Mapping]] = ..., encoded_audio: _Optional[_Union[EncodedAudioFrame, _Mapping]] = ..., end_of_stream: _Optional[_Union[EndOfStream, _Mapping]] = ...) -> None: ...

class EndOfStream(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StreamConfig(_message.Message):
    __slots__ = ("language", "sample_rate", "model", "partials", "partial_window_ms", "partial_stride_ms", "include_word_timestamps", "temperature", "speech_context")
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    PARTIALS_FIELD_NUMBER: _ClassVar[int]
    PARTIAL_WINDOW_MS_FIELD_NUMBER: _ClassVar[int]
    PARTIAL_STRIDE_MS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_WORD_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    SPEECH_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    language: str
    sample_rate: int
    model: str
    partials: bool
    partial_window_ms: int
    partial_stride_ms: int
    include_word_timestamps: bool
    temperature: float
    speech_context: bool
    def __init__(self, language: _Optional[str] = ..., sample_rate: _Optional[int] = ..., model: _Optional[str] = ..., partials: bool = ..., partial_window_ms: _Optional[int] = ..., partial_stride_ms: _Optional[int] = ..., include_word_timestamps: bool = ..., temperature: _Optional[float] = ..., speech_context: bool = ...) -> None: ...

class AudioFrame(_message.Message):
    __slots__ = ("pcm16", "sample_rate")
    PCM16_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    pcm16: bytes
    sample_rate: int
    def __init__(self, pcm16: _Optional[bytes] = ..., sample_rate: _Optional[int] = ...) -> None: ...

class OpusFrame(_message.Message):
    __slots__ = ("data", "sample_rate", "channels")
    DATA_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    sample_rate: int
    channels: int
    def __init__(self, data: _Optional[bytes] = ..., sample_rate: _Optional[int] = ..., channels: _Optional[int] = ...) -> None: ...

class EncodedAudioFrame(_message.Message):
    __slots__ = ("data", "format")
    DATA_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    format: str
    def __init__(self, data: _Optional[bytes] = ..., format: _Optional[str] = ...) -> None: ...

class StreamOutput(_message.Message):
    __slots__ = ("ready", "speech_started", "speech_stopped", "transcript", "error")
    READY_FIELD_NUMBER: _ClassVar[int]
    SPEECH_STARTED_FIELD_NUMBER: _ClassVar[int]
    SPEECH_STOPPED_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    ready: StreamReady
    speech_started: StreamSpeechStarted
    speech_stopped: StreamSpeechStopped
    transcript: StreamTranscriptResult
    error: StreamErrorMessage
    def __init__(self, ready: _Optional[_Union[StreamReady, _Mapping]] = ..., speech_started: _Optional[_Union[StreamSpeechStarted, _Mapping]] = ..., speech_stopped: _Optional[_Union[StreamSpeechStopped, _Mapping]] = ..., transcript: _Optional[_Union[StreamTranscriptResult, _Mapping]] = ..., error: _Optional[_Union[StreamErrorMessage, _Mapping]] = ...) -> None: ...

class StreamReady(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StreamSpeechStarted(_message.Message):
    __slots__ = ("timestamp_ms",)
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    timestamp_ms: int
    def __init__(self, timestamp_ms: _Optional[int] = ...) -> None: ...

class StreamSpeechStopped(_message.Message):
    __slots__ = ("timestamp_ms",)
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    timestamp_ms: int
    def __init__(self, timestamp_ms: _Optional[int] = ...) -> None: ...

class StreamTranscriptResult(_message.Message):
    __slots__ = ("text", "is_partial", "start_ms", "end_ms", "audio_duration_ms", "processing_duration_ms", "model", "eou_probability", "entities", "topics", "words", "segments", "speech_context")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    IS_PARTIAL_FIELD_NUMBER: _ClassVar[int]
    START_MS_FIELD_NUMBER: _ClassVar[int]
    END_MS_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    EOU_PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    TOPICS_FIELD_NUMBER: _ClassVar[int]
    WORDS_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    SPEECH_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    text: str
    is_partial: bool
    start_ms: int
    end_ms: int
    audio_duration_ms: int
    processing_duration_ms: int
    model: str
    eou_probability: float
    entities: _containers.RepeatedCompositeFieldContainer[Entity]
    topics: _containers.RepeatedScalarFieldContainer[str]
    words: _containers.RepeatedCompositeFieldContainer[WordTimestamp]
    segments: _containers.RepeatedCompositeFieldContainer[TranscriptSegment]
    speech_context: _struct_pb2.Struct
    def __init__(self, text: _Optional[str] = ..., is_partial: bool = ..., start_ms: _Optional[int] = ..., end_ms: _Optional[int] = ..., audio_duration_ms: _Optional[int] = ..., processing_duration_ms: _Optional[int] = ..., model: _Optional[str] = ..., eou_probability: _Optional[float] = ..., entities: _Optional[_Iterable[_Union[Entity, _Mapping]]] = ..., topics: _Optional[_Iterable[str]] = ..., words: _Optional[_Iterable[_Union[WordTimestamp, _Mapping]]] = ..., segments: _Optional[_Iterable[_Union[TranscriptSegment, _Mapping]]] = ..., speech_context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class StreamErrorMessage(_message.Message):
    __slots__ = ("message",)
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...

class ListVoicesRequest(_message.Message):
    __slots__ = ("model",)
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: str
    def __init__(self, model: _Optional[str] = ...) -> None: ...

class ListVoicesResponse(_message.Message):
    __slots__ = ("voices",)
    VOICES_FIELD_NUMBER: _ClassVar[int]
    voices: _containers.RepeatedCompositeFieldContainer[VoiceInfo]
    def __init__(self, voices: _Optional[_Iterable[_Union[VoiceInfo, _Mapping]]] = ...) -> None: ...

class CreateVoiceRequest(_message.Message):
    __slots__ = ("name", "audio", "language", "gender", "reference_text", "format_hint")
    NAME_FIELD_NUMBER: _ClassVar[int]
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_TEXT_FIELD_NUMBER: _ClassVar[int]
    FORMAT_HINT_FIELD_NUMBER: _ClassVar[int]
    name: str
    audio: bytes
    language: str
    gender: str
    reference_text: str
    format_hint: str
    def __init__(self, name: _Optional[str] = ..., audio: _Optional[bytes] = ..., language: _Optional[str] = ..., gender: _Optional[str] = ..., reference_text: _Optional[str] = ..., format_hint: _Optional[str] = ...) -> None: ...

class CreateVoiceResponse(_message.Message):
    __slots__ = ("voice", "created_at")
    VOICE_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    voice: VoiceInfo
    created_at: int
    def __init__(self, voice: _Optional[_Union[VoiceInfo, _Mapping]] = ..., created_at: _Optional[int] = ...) -> None: ...

class DeleteVoiceRequest(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: str
    def __init__(self, id: _Optional[str] = ...) -> None: ...

class DeleteVoiceResponse(_message.Message):
    __slots__ = ("id", "deleted")
    ID_FIELD_NUMBER: _ClassVar[int]
    DELETED_FIELD_NUMBER: _ClassVar[int]
    id: str
    deleted: bool
    def __init__(self, id: _Optional[str] = ..., deleted: bool = ...) -> None: ...

class VoiceInfo(_message.Message):
    __slots__ = ("id", "name", "language", "gender", "description", "is_cloned", "model")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    GENDER_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    IS_CLONED_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    language: str
    gender: str
    description: str
    is_cloned: bool
    model: str
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., language: _Optional[str] = ..., gender: _Optional[str] = ..., description: _Optional[str] = ..., is_cloned: bool = ..., model: _Optional[str] = ...) -> None: ...

class RtcCreateSessionRequest(_message.Message):
    __slots__ = ("browser_events",)
    BROWSER_EVENTS_FIELD_NUMBER: _ClassVar[int]
    browser_events: bool
    def __init__(self, browser_events: bool = ...) -> None: ...

class RtcIceServer(_message.Message):
    __slots__ = ("urls", "username", "credential")
    URLS_FIELD_NUMBER: _ClassVar[int]
    USERNAME_FIELD_NUMBER: _ClassVar[int]
    CREDENTIAL_FIELD_NUMBER: _ClassVar[int]
    urls: _containers.RepeatedScalarFieldContainer[str]
    username: str
    credential: str
    def __init__(self, urls: _Optional[_Iterable[str]] = ..., username: _Optional[str] = ..., credential: _Optional[str] = ...) -> None: ...

class RtcSessionBootstrap(_message.Message):
    __slots__ = ("session_id", "expires_at", "ice_servers", "attach_ttl_seconds")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    ICE_SERVERS_FIELD_NUMBER: _ClassVar[int]
    ATTACH_TTL_SECONDS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    expires_at: str
    ice_servers: _containers.RepeatedCompositeFieldContainer[RtcIceServer]
    attach_ttl_seconds: int
    def __init__(self, session_id: _Optional[str] = ..., expires_at: _Optional[str] = ..., ice_servers: _Optional[_Iterable[_Union[RtcIceServer, _Mapping]]] = ..., attach_ttl_seconds: _Optional[int] = ...) -> None: ...

class RtcSessionDescription(_message.Message):
    __slots__ = ("type", "sdp")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    SDP_FIELD_NUMBER: _ClassVar[int]
    type: str
    sdp: str
    def __init__(self, type: _Optional[str] = ..., sdp: _Optional[str] = ...) -> None: ...

class RtcControlAnswer(_message.Message):
    __slots__ = ("session_id", "answer")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    answer: RtcSessionDescription
    def __init__(self, session_id: _Optional[str] = ..., answer: _Optional[_Union[RtcSessionDescription, _Mapping]] = ...) -> None: ...

class RtcIceCandidate(_message.Message):
    __slots__ = ("candidate", "sdp_mid", "sdp_m_line_index", "username_fragment")
    CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    SDP_MID_FIELD_NUMBER: _ClassVar[int]
    SDP_M_LINE_INDEX_FIELD_NUMBER: _ClassVar[int]
    USERNAME_FRAGMENT_FIELD_NUMBER: _ClassVar[int]
    candidate: str
    sdp_mid: str
    sdp_m_line_index: int
    username_fragment: str
    def __init__(self, candidate: _Optional[str] = ..., sdp_mid: _Optional[str] = ..., sdp_m_line_index: _Optional[int] = ..., username_fragment: _Optional[str] = ...) -> None: ...

class RtcIceCandidatesComplete(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RtcControlOffer(_message.Message):
    __slots__ = ("offer", "restart")
    OFFER_FIELD_NUMBER: _ClassVar[int]
    RESTART_FIELD_NUMBER: _ClassVar[int]
    offer: RtcSessionDescription
    restart: bool
    def __init__(self, offer: _Optional[_Union[RtcSessionDescription, _Mapping]] = ..., restart: bool = ...) -> None: ...

class RtcControlClose(_message.Message):
    __slots__ = ("reason",)
    REASON_FIELD_NUMBER: _ClassVar[int]
    reason: str
    def __init__(self, reason: _Optional[str] = ...) -> None: ...

class RtcControlClosed(_message.Message):
    __slots__ = ("session_id", "reason")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    reason: str
    def __init__(self, session_id: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class RtcSignalingError(_message.Message):
    __slots__ = ("message", "code", "recoverable", "generation_id")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    RECOVERABLE_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    message: str
    code: str
    recoverable: bool
    generation_id: str
    def __init__(self, message: _Optional[str] = ..., code: _Optional[str] = ..., recoverable: bool = ..., generation_id: _Optional[str] = ...) -> None: ...

class RtcWireEvent(_message.Message):
    __slots__ = ("type", "payload_json")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_JSON_FIELD_NUMBER: _ClassVar[int]
    type: str
    payload_json: str
    def __init__(self, type: _Optional[str] = ..., payload_json: _Optional[str] = ...) -> None: ...

class ConverseClientMessage(_message.Message):
    __slots__ = ("session_update", "audio_append", "response_cancel", "response_start", "response_delta", "response_commit", "response_replace_text")
    SESSION_UPDATE_FIELD_NUMBER: _ClassVar[int]
    AUDIO_APPEND_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_CANCEL_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_START_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_DELTA_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_COMMIT_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_REPLACE_TEXT_FIELD_NUMBER: _ClassVar[int]
    session_update: ConversationSessionUpdate
    audio_append: ConversationAudioAppend
    response_cancel: ConversationResponseCancel
    response_start: ConversationResponseStart
    response_delta: ConversationResponseDelta
    response_commit: ConversationResponseCommit
    response_replace_text: ConversationResponseReplaceText
    def __init__(self, session_update: _Optional[_Union[ConversationSessionUpdate, _Mapping]] = ..., audio_append: _Optional[_Union[ConversationAudioAppend, _Mapping]] = ..., response_cancel: _Optional[_Union[ConversationResponseCancel, _Mapping]] = ..., response_start: _Optional[_Union[ConversationResponseStart, _Mapping]] = ..., response_delta: _Optional[_Union[ConversationResponseDelta, _Mapping]] = ..., response_commit: _Optional[_Union[ConversationResponseCommit, _Mapping]] = ..., response_replace_text: _Optional[_Union[ConversationResponseReplaceText, _Mapping]] = ...) -> None: ...

class RtcControlClientMessage(_message.Message):
    __slots__ = ("attach", "session_update", "response_cancel", "response_start", "response_delta", "response_commit", "client_event", "response_replace_text", "offer", "candidate", "candidates_complete", "close")
    ATTACH_FIELD_NUMBER: _ClassVar[int]
    SESSION_UPDATE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_CANCEL_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_START_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_DELTA_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_COMMIT_FIELD_NUMBER: _ClassVar[int]
    CLIENT_EVENT_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_REPLACE_TEXT_FIELD_NUMBER: _ClassVar[int]
    OFFER_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    CANDIDATES_COMPLETE_FIELD_NUMBER: _ClassVar[int]
    CLOSE_FIELD_NUMBER: _ClassVar[int]
    attach: RtcControlAttach
    session_update: ConversationSessionUpdate
    response_cancel: ConversationResponseCancel
    response_start: ConversationResponseStart
    response_delta: ConversationResponseDelta
    response_commit: ConversationResponseCommit
    client_event: RtcClientEvent
    response_replace_text: ConversationResponseReplaceText
    offer: RtcControlOffer
    candidate: RtcIceCandidate
    candidates_complete: RtcIceCandidatesComplete
    close: RtcControlClose
    def __init__(self, attach: _Optional[_Union[RtcControlAttach, _Mapping]] = ..., session_update: _Optional[_Union[ConversationSessionUpdate, _Mapping]] = ..., response_cancel: _Optional[_Union[ConversationResponseCancel, _Mapping]] = ..., response_start: _Optional[_Union[ConversationResponseStart, _Mapping]] = ..., response_delta: _Optional[_Union[ConversationResponseDelta, _Mapping]] = ..., response_commit: _Optional[_Union[ConversationResponseCommit, _Mapping]] = ..., client_event: _Optional[_Union[RtcClientEvent, _Mapping]] = ..., response_replace_text: _Optional[_Union[ConversationResponseReplaceText, _Mapping]] = ..., offer: _Optional[_Union[RtcControlOffer, _Mapping]] = ..., candidate: _Optional[_Union[RtcIceCandidate, _Mapping]] = ..., candidates_complete: _Optional[_Union[RtcIceCandidatesComplete, _Mapping]] = ..., close: _Optional[_Union[RtcControlClose, _Mapping]] = ...) -> None: ...

class RtcControlServerMessage(_message.Message):
    __slots__ = ("attached", "answer", "candidate", "candidates_complete", "conversation", "error", "closed", "browser_event", "event")
    ATTACHED_FIELD_NUMBER: _ClassVar[int]
    ANSWER_FIELD_NUMBER: _ClassVar[int]
    CANDIDATE_FIELD_NUMBER: _ClassVar[int]
    CANDIDATES_COMPLETE_FIELD_NUMBER: _ClassVar[int]
    CONVERSATION_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    CLOSED_FIELD_NUMBER: _ClassVar[int]
    BROWSER_EVENT_FIELD_NUMBER: _ClassVar[int]
    EVENT_FIELD_NUMBER: _ClassVar[int]
    attached: RtcSessionAttached
    answer: RtcControlAnswer
    candidate: RtcIceCandidate
    candidates_complete: RtcIceCandidatesComplete
    conversation: ConverseServerMessage
    error: RtcSignalingError
    closed: RtcControlClosed
    browser_event: RtcClientEvent
    event: RtcWireEvent
    def __init__(self, attached: _Optional[_Union[RtcSessionAttached, _Mapping]] = ..., answer: _Optional[_Union[RtcControlAnswer, _Mapping]] = ..., candidate: _Optional[_Union[RtcIceCandidate, _Mapping]] = ..., candidates_complete: _Optional[_Union[RtcIceCandidatesComplete, _Mapping]] = ..., conversation: _Optional[_Union[ConverseServerMessage, _Mapping]] = ..., error: _Optional[_Union[RtcSignalingError, _Mapping]] = ..., closed: _Optional[_Union[RtcControlClosed, _Mapping]] = ..., browser_event: _Optional[_Union[RtcClientEvent, _Mapping]] = ..., event: _Optional[_Union[RtcWireEvent, _Mapping]] = ...) -> None: ...

class RtcControlAttach(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class ConverseServerMessage(_message.Message):
    __slots__ = ("session_created", "speech_started", "speech_stopped", "transcript_done", "response_created", "audio_delta", "response_done", "response_cancelled", "state_changed", "error", "response_committed", "audio_clear", "interruption_detected", "interruption_false_positive", "turn_eou_predicted", "transcript_delta")
    SESSION_CREATED_FIELD_NUMBER: _ClassVar[int]
    SPEECH_STARTED_FIELD_NUMBER: _ClassVar[int]
    SPEECH_STOPPED_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPT_DONE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_CREATED_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DELTA_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_DONE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_CANCELLED_FIELD_NUMBER: _ClassVar[int]
    STATE_CHANGED_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_COMMITTED_FIELD_NUMBER: _ClassVar[int]
    AUDIO_CLEAR_FIELD_NUMBER: _ClassVar[int]
    INTERRUPTION_DETECTED_FIELD_NUMBER: _ClassVar[int]
    INTERRUPTION_FALSE_POSITIVE_FIELD_NUMBER: _ClassVar[int]
    TURN_EOU_PREDICTED_FIELD_NUMBER: _ClassVar[int]
    TRANSCRIPT_DELTA_FIELD_NUMBER: _ClassVar[int]
    session_created: ConversationSessionCreated
    speech_started: ConversationSpeechStarted
    speech_stopped: ConversationSpeechStopped
    transcript_done: ConversationTranscriptDone
    response_created: ConversationResponseCreated
    audio_delta: ConversationAudioDelta
    response_done: ConversationResponseDone
    response_cancelled: ConversationResponseCancelled
    state_changed: ConversationStateChanged
    error: ConversationError
    response_committed: ConversationResponseCommitted
    audio_clear: ConversationAudioClear
    interruption_detected: ConversationInterruptionDetected
    interruption_false_positive: ConversationInterruptionFalsePositive
    turn_eou_predicted: ConversationTurnEouPredicted
    transcript_delta: ConversationTranscriptDelta
    def __init__(self, session_created: _Optional[_Union[ConversationSessionCreated, _Mapping]] = ..., speech_started: _Optional[_Union[ConversationSpeechStarted, _Mapping]] = ..., speech_stopped: _Optional[_Union[ConversationSpeechStopped, _Mapping]] = ..., transcript_done: _Optional[_Union[ConversationTranscriptDone, _Mapping]] = ..., response_created: _Optional[_Union[ConversationResponseCreated, _Mapping]] = ..., audio_delta: _Optional[_Union[ConversationAudioDelta, _Mapping]] = ..., response_done: _Optional[_Union[ConversationResponseDone, _Mapping]] = ..., response_cancelled: _Optional[_Union[ConversationResponseCancelled, _Mapping]] = ..., state_changed: _Optional[_Union[ConversationStateChanged, _Mapping]] = ..., error: _Optional[_Union[ConversationError, _Mapping]] = ..., response_committed: _Optional[_Union[ConversationResponseCommitted, _Mapping]] = ..., audio_clear: _Optional[_Union[ConversationAudioClear, _Mapping]] = ..., interruption_detected: _Optional[_Union[ConversationInterruptionDetected, _Mapping]] = ..., interruption_false_positive: _Optional[_Union[ConversationInterruptionFalsePositive, _Mapping]] = ..., turn_eou_predicted: _Optional[_Union[ConversationTurnEouPredicted, _Mapping]] = ..., transcript_delta: _Optional[_Union[ConversationTranscriptDelta, _Mapping]] = ...) -> None: ...

class RtcSessionAttached(_message.Message):
    __slots__ = ("session_id", "provider")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    PROVIDER_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    provider: str
    def __init__(self, session_id: _Optional[str] = ..., provider: _Optional[str] = ...) -> None: ...

class RtcClientEvent(_message.Message):
    __slots__ = ("event", "payload_json")
    EVENT_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_JSON_FIELD_NUMBER: _ClassVar[int]
    event: str
    payload_json: str
    def __init__(self, event: _Optional[str] = ..., payload_json: _Optional[str] = ...) -> None: ...

class ConversationResponseCommitted(_message.Message):
    __slots__ = ("response_id", "generation_id")
    RESPONSE_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    response_id: str
    generation_id: str
    def __init__(self, response_id: _Optional[str] = ..., generation_id: _Optional[str] = ...) -> None: ...

class ConversationTurnPolicy(_message.Message):
    __slots__ = ("allow_interrupt_while_speaking", "min_interrupt_duration_ms", "max_endpointing_delay_ms", "false_interruption_timeout_ms", "min_interrupt_words", "partial_interrupts", "dynamic_endpointing", "min_endpointing_delay_ms", "speaking_interrupt_min_duration_ms", "speaking_interrupt_min_words", "self_echo_min_words", "self_echo_min_overlap", "aec_warmup_ms", "backchannel_end_cooldown_ms", "vad_min_silence_ms")
    ALLOW_INTERRUPT_WHILE_SPEAKING_FIELD_NUMBER: _ClassVar[int]
    MIN_INTERRUPT_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    MAX_ENDPOINTING_DELAY_MS_FIELD_NUMBER: _ClassVar[int]
    FALSE_INTERRUPTION_TIMEOUT_MS_FIELD_NUMBER: _ClassVar[int]
    MIN_INTERRUPT_WORDS_FIELD_NUMBER: _ClassVar[int]
    PARTIAL_INTERRUPTS_FIELD_NUMBER: _ClassVar[int]
    DYNAMIC_ENDPOINTING_FIELD_NUMBER: _ClassVar[int]
    MIN_ENDPOINTING_DELAY_MS_FIELD_NUMBER: _ClassVar[int]
    SPEAKING_INTERRUPT_MIN_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    SPEAKING_INTERRUPT_MIN_WORDS_FIELD_NUMBER: _ClassVar[int]
    SELF_ECHO_MIN_WORDS_FIELD_NUMBER: _ClassVar[int]
    SELF_ECHO_MIN_OVERLAP_FIELD_NUMBER: _ClassVar[int]
    AEC_WARMUP_MS_FIELD_NUMBER: _ClassVar[int]
    BACKCHANNEL_END_COOLDOWN_MS_FIELD_NUMBER: _ClassVar[int]
    VAD_MIN_SILENCE_MS_FIELD_NUMBER: _ClassVar[int]
    allow_interrupt_while_speaking: bool
    min_interrupt_duration_ms: int
    max_endpointing_delay_ms: int
    false_interruption_timeout_ms: int
    min_interrupt_words: int
    partial_interrupts: bool
    dynamic_endpointing: bool
    min_endpointing_delay_ms: int
    speaking_interrupt_min_duration_ms: int
    speaking_interrupt_min_words: int
    self_echo_min_words: int
    self_echo_min_overlap: float
    aec_warmup_ms: int
    backchannel_end_cooldown_ms: int
    vad_min_silence_ms: int
    def __init__(self, allow_interrupt_while_speaking: bool = ..., min_interrupt_duration_ms: _Optional[int] = ..., max_endpointing_delay_ms: _Optional[int] = ..., false_interruption_timeout_ms: _Optional[int] = ..., min_interrupt_words: _Optional[int] = ..., partial_interrupts: bool = ..., dynamic_endpointing: bool = ..., min_endpointing_delay_ms: _Optional[int] = ..., speaking_interrupt_min_duration_ms: _Optional[int] = ..., speaking_interrupt_min_words: _Optional[int] = ..., self_echo_min_words: _Optional[int] = ..., self_echo_min_overlap: _Optional[float] = ..., aec_warmup_ms: _Optional[int] = ..., backchannel_end_cooldown_ms: _Optional[int] = ..., vad_min_silence_ms: _Optional[int] = ...) -> None: ...

class ConversationSessionUpdate(_message.Message):
    __slots__ = ("stt_model", "tts_model", "voice", "language", "sample_rate", "policy", "vad_backend", "turn_detector", "turn_profile", "include_word_timestamps", "speech_context")
    STT_MODEL_FIELD_NUMBER: _ClassVar[int]
    TTS_MODEL_FIELD_NUMBER: _ClassVar[int]
    VOICE_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    POLICY_FIELD_NUMBER: _ClassVar[int]
    VAD_BACKEND_FIELD_NUMBER: _ClassVar[int]
    TURN_DETECTOR_FIELD_NUMBER: _ClassVar[int]
    TURN_PROFILE_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_WORD_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    SPEECH_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    stt_model: str
    tts_model: str
    voice: str
    language: str
    sample_rate: int
    policy: ConversationTurnPolicy
    vad_backend: str
    turn_detector: str
    turn_profile: str
    include_word_timestamps: bool
    speech_context: bool
    def __init__(self, stt_model: _Optional[str] = ..., tts_model: _Optional[str] = ..., voice: _Optional[str] = ..., language: _Optional[str] = ..., sample_rate: _Optional[int] = ..., policy: _Optional[_Union[ConversationTurnPolicy, _Mapping]] = ..., vad_backend: _Optional[str] = ..., turn_detector: _Optional[str] = ..., turn_profile: _Optional[str] = ..., include_word_timestamps: bool = ..., speech_context: bool = ...) -> None: ...

class ConversationAudioAppend(_message.Message):
    __slots__ = ("pcm16", "sample_rate")
    PCM16_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    pcm16: bytes
    sample_rate: int
    def __init__(self, pcm16: _Optional[bytes] = ..., sample_rate: _Optional[int] = ...) -> None: ...

class ConversationResponseOutput(_message.Message):
    __slots__ = ("model", "voice", "language", "speed", "params")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    VOICE_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    model: str
    voice: str
    language: str
    speed: float
    params: _struct_pb2.Struct
    def __init__(self, model: _Optional[str] = ..., voice: _Optional[str] = ..., language: _Optional[str] = ..., speed: _Optional[float] = ..., params: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ConversationResponseStart(_message.Message):
    __slots__ = ("allow_interruptions", "generation_id", "output")
    ALLOW_INTERRUPTIONS_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    allow_interruptions: bool
    generation_id: str
    output: ConversationResponseOutput
    def __init__(self, allow_interruptions: bool = ..., generation_id: _Optional[str] = ..., output: _Optional[_Union[ConversationResponseOutput, _Mapping]] = ...) -> None: ...

class ConversationResponseDelta(_message.Message):
    __slots__ = ("delta", "allow_interruptions", "generation_id")
    DELTA_FIELD_NUMBER: _ClassVar[int]
    ALLOW_INTERRUPTIONS_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    delta: str
    allow_interruptions: bool
    generation_id: str
    def __init__(self, delta: _Optional[str] = ..., allow_interruptions: bool = ..., generation_id: _Optional[str] = ...) -> None: ...

class ConversationResponseCommit(_message.Message):
    __slots__ = ("generation_id",)
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    generation_id: str
    def __init__(self, generation_id: _Optional[str] = ...) -> None: ...

class ConversationResponseCancel(_message.Message):
    __slots__ = ("generation_id",)
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    generation_id: str
    def __init__(self, generation_id: _Optional[str] = ...) -> None: ...

class ConversationResponseReplaceText(_message.Message):
    __slots__ = ("text", "allow_interruptions")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    ALLOW_INTERRUPTIONS_FIELD_NUMBER: _ClassVar[int]
    text: str
    allow_interruptions: bool
    def __init__(self, text: _Optional[str] = ..., allow_interruptions: bool = ...) -> None: ...

class ConversationSessionCreated(_message.Message):
    __slots__ = ("turn_profile", "policy")
    TURN_PROFILE_FIELD_NUMBER: _ClassVar[int]
    POLICY_FIELD_NUMBER: _ClassVar[int]
    turn_profile: str
    policy: ConversationTurnPolicy
    def __init__(self, turn_profile: _Optional[str] = ..., policy: _Optional[_Union[ConversationTurnPolicy, _Mapping]] = ...) -> None: ...

class ConversationSpeechStarted(_message.Message):
    __slots__ = ("timestamp_ms",)
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    timestamp_ms: int
    def __init__(self, timestamp_ms: _Optional[int] = ...) -> None: ...

class ConversationSpeechStopped(_message.Message):
    __slots__ = ("timestamp_ms",)
    TIMESTAMP_MS_FIELD_NUMBER: _ClassVar[int]
    timestamp_ms: int
    def __init__(self, timestamp_ms: _Optional[int] = ...) -> None: ...

class ConversationTranscriptDelta(_message.Message):
    __slots__ = ("delta", "start_ms", "end_ms")
    DELTA_FIELD_NUMBER: _ClassVar[int]
    START_MS_FIELD_NUMBER: _ClassVar[int]
    END_MS_FIELD_NUMBER: _ClassVar[int]
    delta: str
    start_ms: int
    end_ms: int
    def __init__(self, delta: _Optional[str] = ..., start_ms: _Optional[int] = ..., end_ms: _Optional[int] = ...) -> None: ...

class ConversationTranscriptDone(_message.Message):
    __slots__ = ("transcript", "language", "start_ms", "end_ms", "eou_probability", "entities", "topics", "words", "speech_context")
    TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    START_MS_FIELD_NUMBER: _ClassVar[int]
    END_MS_FIELD_NUMBER: _ClassVar[int]
    EOU_PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    ENTITIES_FIELD_NUMBER: _ClassVar[int]
    TOPICS_FIELD_NUMBER: _ClassVar[int]
    WORDS_FIELD_NUMBER: _ClassVar[int]
    SPEECH_CONTEXT_FIELD_NUMBER: _ClassVar[int]
    transcript: str
    language: str
    start_ms: int
    end_ms: int
    eou_probability: float
    entities: _containers.RepeatedCompositeFieldContainer[Entity]
    topics: _containers.RepeatedScalarFieldContainer[str]
    words: _containers.RepeatedCompositeFieldContainer[WordTimestamp]
    speech_context: _struct_pb2.Struct
    def __init__(self, transcript: _Optional[str] = ..., language: _Optional[str] = ..., start_ms: _Optional[int] = ..., end_ms: _Optional[int] = ..., eou_probability: _Optional[float] = ..., entities: _Optional[_Iterable[_Union[Entity, _Mapping]]] = ..., topics: _Optional[_Iterable[str]] = ..., words: _Optional[_Iterable[_Union[WordTimestamp, _Mapping]]] = ..., speech_context: _Optional[_Union[_struct_pb2.Struct, _Mapping]] = ...) -> None: ...

class ConversationResponseCreated(_message.Message):
    __slots__ = ("response_id", "generation_id", "output")
    RESPONSE_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    response_id: str
    generation_id: str
    output: ConversationResponseOutput
    def __init__(self, response_id: _Optional[str] = ..., generation_id: _Optional[str] = ..., output: _Optional[_Union[ConversationResponseOutput, _Mapping]] = ...) -> None: ...

class ConversationAudioDelta(_message.Message):
    __slots__ = ("audio", "sample_rate", "response_id", "sequence")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_ID_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_FIELD_NUMBER: _ClassVar[int]
    audio: bytes
    sample_rate: int
    response_id: str
    sequence: int
    def __init__(self, audio: _Optional[bytes] = ..., sample_rate: _Optional[int] = ..., response_id: _Optional[str] = ..., sequence: _Optional[int] = ...) -> None: ...

class ConversationAudioClear(_message.Message):
    __slots__ = ("response_id", "generation_id")
    RESPONSE_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    response_id: str
    generation_id: str
    def __init__(self, response_id: _Optional[str] = ..., generation_id: _Optional[str] = ...) -> None: ...

class ConversationInterruptionDetected(_message.Message):
    __slots__ = ("response_id", "vad_active_ms", "partial_transcript", "generation_id")
    RESPONSE_ID_FIELD_NUMBER: _ClassVar[int]
    VAD_ACTIVE_MS_FIELD_NUMBER: _ClassVar[int]
    PARTIAL_TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    response_id: str
    vad_active_ms: int
    partial_transcript: str
    generation_id: str
    def __init__(self, response_id: _Optional[str] = ..., vad_active_ms: _Optional[int] = ..., partial_transcript: _Optional[str] = ..., generation_id: _Optional[str] = ...) -> None: ...

class ConversationInterruptionFalsePositive(_message.Message):
    __slots__ = ("response_id", "vad_active_ms", "partial_transcript", "reason", "generation_id")
    RESPONSE_ID_FIELD_NUMBER: _ClassVar[int]
    VAD_ACTIVE_MS_FIELD_NUMBER: _ClassVar[int]
    PARTIAL_TRANSCRIPT_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    response_id: str
    vad_active_ms: int
    partial_transcript: str
    reason: str
    generation_id: str
    def __init__(self, response_id: _Optional[str] = ..., vad_active_ms: _Optional[int] = ..., partial_transcript: _Optional[str] = ..., reason: _Optional[str] = ..., generation_id: _Optional[str] = ...) -> None: ...

class ConversationTurnEouPredicted(_message.Message):
    __slots__ = ("probability", "threshold", "decision", "action", "delay_ms", "turn_detector", "start_ms", "end_ms")
    PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    DECISION_FIELD_NUMBER: _ClassVar[int]
    ACTION_FIELD_NUMBER: _ClassVar[int]
    DELAY_MS_FIELD_NUMBER: _ClassVar[int]
    TURN_DETECTOR_FIELD_NUMBER: _ClassVar[int]
    START_MS_FIELD_NUMBER: _ClassVar[int]
    END_MS_FIELD_NUMBER: _ClassVar[int]
    probability: float
    threshold: float
    decision: str
    action: str
    delay_ms: int
    turn_detector: str
    start_ms: int
    end_ms: int
    def __init__(self, probability: _Optional[float] = ..., threshold: _Optional[float] = ..., decision: _Optional[str] = ..., action: _Optional[str] = ..., delay_ms: _Optional[int] = ..., turn_detector: _Optional[str] = ..., start_ms: _Optional[int] = ..., end_ms: _Optional[int] = ...) -> None: ...

class ConversationResponseDone(_message.Message):
    __slots__ = ("response_id", "generation_id")
    RESPONSE_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    response_id: str
    generation_id: str
    def __init__(self, response_id: _Optional[str] = ..., generation_id: _Optional[str] = ...) -> None: ...

class ConversationResponseCancelled(_message.Message):
    __slots__ = ("response_id", "generation_id")
    RESPONSE_ID_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    response_id: str
    generation_id: str
    def __init__(self, response_id: _Optional[str] = ..., generation_id: _Optional[str] = ...) -> None: ...

class ConversationStateChanged(_message.Message):
    __slots__ = ("state", "previous_state")
    STATE_FIELD_NUMBER: _ClassVar[int]
    PREVIOUS_STATE_FIELD_NUMBER: _ClassVar[int]
    state: str
    previous_state: str
    def __init__(self, state: _Optional[str] = ..., previous_state: _Optional[str] = ...) -> None: ...

class ConversationError(_message.Message):
    __slots__ = ("message", "code", "recoverable", "generation_id")
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    CODE_FIELD_NUMBER: _ClassVar[int]
    RECOVERABLE_FIELD_NUMBER: _ClassVar[int]
    GENERATION_ID_FIELD_NUMBER: _ClassVar[int]
    message: str
    code: str
    recoverable: bool
    generation_id: str
    def __init__(self, message: _Optional[str] = ..., code: _Optional[str] = ..., recoverable: bool = ..., generation_id: _Optional[str] = ...) -> None: ...
