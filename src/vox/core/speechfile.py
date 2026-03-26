"""Parser for the Speechfile format — a declarative model spec akin to Ollama's Modelfile.

Example Speechfile::

    FROM hexgrad/Kokoro-82M-v1.0-ONNX
    ARCHITECTURE kokoro
    TYPE tts
    ADAPTER kokoro
    FORMAT onnx
    PARAMETER sample_rate 24000
    PARAMETER default_voice af_heart
    VOICE af_heart "American Female - Heart"
    LICENSE Apache-2.0
    DESCRIPTION "Kokoro 82M ONNX TTS"
"""

from __future__ import annotations

import shlex

from vox.core.errors import VoxError
from vox.core.types import ModelFormat, ModelType, Speechfile, VoiceInfo


class SpeechfileParseError(VoxError):
    """Raised when a Speechfile cannot be parsed."""


def _unquote(value: str) -> str:
    """Strip surrounding quotes from a string value if present."""
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    return value


def parse_speechfile(content: str) -> Speechfile:
    """Parse Speechfile text into a :class:`Speechfile` dataclass.

    Directives (case-insensitive):

    - ``FROM`` (required) — HuggingFace repo, URL, or local path
    - ``ARCHITECTURE`` — model architecture name
    - ``TYPE`` — ``stt`` or ``tts``
    - ``ADAPTER`` — entry point name
    - ``FORMAT`` — ``onnx``, ``ct2``, ``pytorch``, ``gguf``
    - ``PARAMETER`` key value — arbitrary key/value parameter
    - ``VOICE`` id "display name" — voice entry
    - ``LICENSE`` — license identifier
    - ``DESCRIPTION`` — human-readable description (may be quoted)
    - ``FILES`` — comma-separated list of filenames to download
    - Lines starting with ``#`` are comments.
    - Unknown directives are silently ignored.
    """

    source: str | None = None
    architecture: str = ""
    model_type: str = ""
    adapter: str = ""
    fmt: str = ""
    parameters: dict[str, object] = {}
    voices: list[VoiceInfo] = []
    license_: str = ""
    description: str = ""
    files: list[str] = []

    for lineno, raw_line in enumerate(content.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Split into directive + rest
        parts = line.split(None, 1)
        directive = parts[0].upper()
        rest = parts[1].strip() if len(parts) > 1 else ""

        if directive == "FROM":
            if not rest:
                raise SpeechfileParseError(f"line {lineno}: FROM requires a value")
            source = rest

        elif directive == "ARCHITECTURE":
            architecture = rest

        elif directive == "TYPE":
            model_type = rest.lower()

        elif directive == "ADAPTER":
            adapter = rest

        elif directive == "FORMAT":
            fmt = rest.lower()

        elif directive == "PARAMETER":
            key_value = rest.split(None, 1)
            if len(key_value) != 2:
                raise SpeechfileParseError(
                    f"line {lineno}: PARAMETER requires a key and value"
                )
            key, raw_val = key_value
            # Try to coerce numeric values.
            val: object
            try:
                val = int(raw_val)
            except ValueError:
                try:
                    val = float(raw_val)
                except ValueError:
                    val = _unquote(raw_val)
            parameters[key] = val

        elif directive == "VOICE":
            # VOICE id "display name"
            try:
                tokens = shlex.split(rest)
            except ValueError:
                tokens = rest.split(None, 1)
            if not tokens:
                raise SpeechfileParseError(f"line {lineno}: VOICE requires an id")
            voice_id = tokens[0]
            voice_name = tokens[1] if len(tokens) > 1 else voice_id
            voices.append(VoiceInfo(id=voice_id, name=voice_name))

        elif directive == "LICENSE":
            license_ = _unquote(rest)

        elif directive == "DESCRIPTION":
            description = _unquote(rest)

        elif directive == "FILES":
            files = [f.strip() for f in rest.split(",") if f.strip()]

        # Unknown directives are silently ignored.

    if source is None:
        raise SpeechfileParseError("Speechfile is missing a required FROM directive")

    try:
        type_enum = ModelType(model_type) if model_type else ModelType.STT
    except ValueError:
        raise SpeechfileParseError(f"Invalid TYPE value: {model_type!r}. Must be one of: {', '.join(t.value for t in ModelType)}")

    try:
        format_enum = ModelFormat(fmt) if fmt else ModelFormat.ONNX
    except ValueError:
        raise SpeechfileParseError(f"Invalid FORMAT value: {fmt!r}. Must be one of: {', '.join(f.value for f in ModelFormat)}")

    return Speechfile(
        source=source,
        architecture=architecture,
        type=type_enum,
        adapter=adapter,
        format=format_enum,
        parameters=parameters,
        voices=tuple(voices),
        license=license_,
        description=description,
        files=tuple(files),
    )
