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
    __slots__ = ("name",)
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("audio", "model", "language", "word_timestamps", "temperature", "response_format", "format_hint")
    AUDIO_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    WORD_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    FORMAT_HINT_FIELD_NUMBER: _ClassVar[int]
    audio: bytes
    model: str
    language: str
    word_timestamps: bool
    temperature: float
    response_format: str
    format_hint: str
    def __init__(self, audio: _Optional[bytes] = ..., model: _Optional[str] = ..., language: _Optional[str] = ..., word_timestamps: bool = ..., temperature: _Optional[float] = ..., response_format: _Optional[str] = ..., format_hint: _Optional[str] = ...) -> None: ...

class TranscribeResponse(_message.Message):
    __slots__ = ("model", "text", "language", "duration_ms", "processing_ms", "segments")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_MS_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    model: str
    text: str
    language: str
    duration_ms: int
    processing_ms: int
    segments: _containers.RepeatedCompositeFieldContainer[TranscriptSegment]
    def __init__(self, model: _Optional[str] = ..., text: _Optional[str] = ..., language: _Optional[str] = ..., duration_ms: _Optional[int] = ..., processing_ms: _Optional[int] = ..., segments: _Optional[_Iterable[_Union[TranscriptSegment, _Mapping]]] = ...) -> None: ...

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
    __slots__ = ("model", "input", "voice", "speed", "language", "response_format")
    MODEL_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    VOICE_FIELD_NUMBER: _ClassVar[int]
    SPEED_FIELD_NUMBER: _ClassVar[int]
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    RESPONSE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    model: str
    input: str
    voice: str
    speed: float
    language: str
    response_format: str
    def __init__(self, model: _Optional[str] = ..., input: _Optional[str] = ..., voice: _Optional[str] = ..., speed: _Optional[float] = ..., language: _Optional[str] = ..., response_format: _Optional[str] = ...) -> None: ...

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
    __slots__ = ("language", "sample_rate", "model", "partials", "partial_window_ms", "partial_stride_ms", "include_word_timestamps", "temperature")
    LANGUAGE_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    PARTIALS_FIELD_NUMBER: _ClassVar[int]
    PARTIAL_WINDOW_MS_FIELD_NUMBER: _ClassVar[int]
    PARTIAL_STRIDE_MS_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_WORD_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    TEMPERATURE_FIELD_NUMBER: _ClassVar[int]
    language: str
    sample_rate: int
    model: str
    partials: bool
    partial_window_ms: int
    partial_stride_ms: int
    include_word_timestamps: bool
    temperature: float
    def __init__(self, language: _Optional[str] = ..., sample_rate: _Optional[int] = ..., model: _Optional[str] = ..., partials: bool = ..., partial_window_ms: _Optional[int] = ..., partial_stride_ms: _Optional[int] = ..., include_word_timestamps: bool = ..., temperature: _Optional[float] = ...) -> None: ...

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
    __slots__ = ("text", "is_partial", "start_ms", "end_ms", "audio_duration_ms", "processing_duration_ms", "model", "eou_probability")
    TEXT_FIELD_NUMBER: _ClassVar[int]
    IS_PARTIAL_FIELD_NUMBER: _ClassVar[int]
    START_MS_FIELD_NUMBER: _ClassVar[int]
    END_MS_FIELD_NUMBER: _ClassVar[int]
    AUDIO_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_DURATION_MS_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    EOU_PROBABILITY_FIELD_NUMBER: _ClassVar[int]
    text: str
    is_partial: bool
    start_ms: int
    end_ms: int
    audio_duration_ms: int
    processing_duration_ms: int
    model: str
    eou_probability: float
    def __init__(self, text: _Optional[str] = ..., is_partial: bool = ..., start_ms: _Optional[int] = ..., end_ms: _Optional[int] = ..., audio_duration_ms: _Optional[int] = ..., processing_duration_ms: _Optional[int] = ..., model: _Optional[str] = ..., eou_probability: _Optional[float] = ...) -> None: ...

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
