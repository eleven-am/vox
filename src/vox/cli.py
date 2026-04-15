from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import wave
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import click
import httpx
import numpy as np
import soundfile as sf
from websockets.sync.client import ClientConnection, connect

from vox.streaming.codecs import float32_to_pcm16, resample_audio
from vox.streaming.types import TARGET_SAMPLE_RATE

DEFAULT_HOST = "http://localhost:11435"
DEFAULT_STREAM_CHUNK_MS = 5_000
DEFAULT_STREAM_TEXT_CHARS = 2_000


def _handle_request_error(e: Exception, host: str) -> None:
    """Print a user-friendly error for HTTP request failures."""
    if isinstance(e, httpx.ConnectError):
        click.echo(f"Error: cannot connect to Vox server at {host}", err=True)
        click.echo("Start the server with: vox serve", err=True)
    elif isinstance(e, httpx.TimeoutException):
        click.echo("Error: request to Vox server timed out", err=True)
    elif isinstance(e, httpx.HTTPStatusError):
        click.echo(f"Error: server returned {e.response.status_code}", err=True)
    else:
        click.echo(f"Error: {e}", err=True)
    sys.exit(1)


def _check_response(resp: httpx.Response) -> None:
    """Check HTTP status and raise on error."""
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        click.echo(f"Error: {detail}", err=True)
        sys.exit(1)


def _to_websocket_url(host: str, path: str) -> str:
    parsed = urlparse(host)
    if parsed.scheme not in {"http", "https", "ws", "wss"}:
        raise click.ClickException(f"Unsupported VOX host scheme: {parsed.scheme!r}")

    scheme = {
        "http": "ws",
        "https": "wss",
    }.get(parsed.scheme, parsed.scheme)
    base_path = parsed.path.rstrip("/")
    full_path = urljoin(f"{base_path or '/'}/", path.lstrip("/"))
    return urlunparse((scheme, parsed.netloc, full_path, "", "", ""))


def _send_ws_json(ws: ClientConnection, payload: dict[str, object]) -> None:
    ws.send(json.dumps(payload))


def _receive_ws_message(ws: ClientConnection, timeout: float | None = None) -> dict[str, object] | bytes:
    message = ws.recv(timeout=timeout)
    if isinstance(message, bytes):
        return message

    try:
        payload = json.loads(message)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Server returned invalid JSON over WebSocket: {exc}") from exc

    if payload.get("type") == "error":
        raise click.ClickException(str(payload.get("message", "unknown streaming error")))
    return payload


def _receive_ready_event(ws: ClientConnection, timeout: float = 30.0) -> dict[str, object]:
    message = _receive_ws_message(ws, timeout=timeout)
    if isinstance(message, bytes):
        raise click.ClickException("Server returned binary data before the ready event")
    if message.get("type") != "ready":
        raise click.ClickException(f"Expected ready event, got: {message}")
    return message


def _print_transcribe_progress(event: dict[str, object]) -> None:
    uploaded_ms = int(event.get("uploaded_ms", 0))
    processed_ms = int(event.get("processed_ms", 0))
    chunks_completed = int(event.get("chunks_completed", 0))
    click.echo(
        (
            f"[progress] uploaded={uploaded_ms / 1000:.1f}s "
            f"processed={processed_ms / 1000:.1f}s "
            f"chunks={chunks_completed}"
        ),
        err=True,
    )


def _print_tts_progress(event: dict[str, object]) -> None:
    completed_chars = int(event.get("completed_chars", 0))
    total_chars = int(event.get("total_chars", 0))
    chunks_completed = int(event.get("chunks_completed", 0))
    chunks_total = int(event.get("chunks_total", 0))
    click.echo(
        (
            f"[progress] chars={completed_chars}/{total_chars} "
            f"chunks={chunks_completed}/{chunks_total}"
        ),
        err=True,
    )


def _drain_transcribe_events(
    ws: ClientConnection,
    *,
    wait_for_done: bool,
) -> dict[str, object] | None:
    while True:
        try:
            message = _receive_ws_message(ws, timeout=None if wait_for_done else 0)
        except TimeoutError:
            return None

        if isinstance(message, bytes):
            raise click.ClickException("Transcription stream returned unexpected binary audio")

        msg_type = message.get("type")
        if msg_type == "progress":
            _print_transcribe_progress(message)
            continue
        if msg_type == "done":
            return message
        if msg_type == "ready":
            continue

        raise click.ClickException(f"Unexpected transcription event: {message}")


def _handle_tts_event(
    message: dict[str, object] | bytes,
    *,
    output_path: Path,
    writer: wave.Wave_write | None,
) -> tuple[wave.Wave_write | None, dict[str, object] | None]:
    if isinstance(message, bytes):
        if writer is None:
            raise click.ClickException("Received audio before audio_start metadata")
        writer.writeframes(message)
        return writer, None

    msg_type = message.get("type")
    if msg_type == "audio_start":
        sample_rate = int(message.get("sample_rate") or 0)
        response_format = str(message.get("response_format", ""))
        if response_format != "pcm16":
            raise click.ClickException(
                f"CLI streaming synthesis only supports pcm16 responses, got {response_format!r}"
            )
        if sample_rate <= 0:
            raise click.ClickException(f"Invalid sample rate in audio_start event: {message}")
        if writer is not None:
            return writer, None

        writer = wave.open(str(output_path), "wb")  # noqa: SIM115
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        return writer, None

    if msg_type == "progress":
        _print_tts_progress(message)
        return writer, None

    if msg_type == "done":
        return writer, message

    if msg_type == "ready":
        return writer, None

    raise click.ClickException(f"Unexpected synthesis event: {message}")


def _drain_tts_events(
    ws: ClientConnection,
    *,
    output_path: Path,
    writer: wave.Wave_write | None,
    wait_for_done: bool,
) -> tuple[wave.Wave_write | None, dict[str, object] | None]:
    while True:
        try:
            message = _receive_ws_message(ws, timeout=None if wait_for_done else 0)
        except TimeoutError:
            return writer, None
        writer, done = _handle_tts_event(message, output_path=output_path, writer=writer)
        if done is not None:
            return writer, done


def _iter_soundfile_pcm16_chunks(
    input_path: Path,
    *,
    target_rate: int,
    chunk_ms: int,
) -> Iterator[bytes]:
    with sf.SoundFile(str(input_path)) as audio_file:
        blocksize = max(1, round(audio_file.samplerate * chunk_ms / 1000))
        for block in audio_file.blocks(blocksize=blocksize, dtype="float32", always_2d=True):
            mono = block.mean(axis=1).astype(np.float32)
            if audio_file.samplerate != target_rate:
                mono = resample_audio(mono, audio_file.samplerate, target_rate)
            yield float32_to_pcm16(mono)


def _iter_ffmpeg_pcm16_chunks(
    input_path: Path,
    *,
    target_rate: int,
    chunk_ms: int,
) -> Iterator[bytes]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise click.ClickException(
            "ffmpeg is required to stream compressed audio files. "
            "Install ffmpeg or convert the input to WAV/FLAC first."
        )

    chunk_bytes = max(1, round(target_rate * chunk_ms / 1000) * 2)
    cmd = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-ar",
        str(target_rate),
        "pipe:1",
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(chunk_bytes)
            if not chunk:
                break
            yield chunk
        stderr = proc.stderr.read().decode("utf-8", errors="replace").strip() if proc.stderr else ""
        returncode = proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    if returncode != 0:
        raise click.ClickException(stderr or f"ffmpeg exited with status {returncode}")


def _iter_pcm16_audio_chunks(
    input_path: Path,
    *,
    target_rate: int = TARGET_SAMPLE_RATE,
    chunk_ms: int = DEFAULT_STREAM_CHUNK_MS,
) -> Iterator[bytes]:
    try:
        yield from _iter_soundfile_pcm16_chunks(input_path, target_rate=target_rate, chunk_ms=chunk_ms)
    except Exception:
        yield from _iter_ffmpeg_pcm16_chunks(input_path, target_rate=target_rate, chunk_ms=chunk_ms)


def _chunk_text_for_stream(text: str, max_chars: int = DEFAULT_STREAM_TEXT_CHARS) -> list[str]:
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if not sentences:
        return [cleaned[i:i + max_chars] for i in range(0, len(cleaned), max_chars)]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(sentence) <= max_chars:
            current = sentence
            continue
        for idx in range(0, len(sentence), max_chars):
            chunks.append(sentence[idx:idx + max_chars])
        current = ""

    if current:
        chunks.append(current)
    return chunks


@click.group()
@click.option("--host", default=DEFAULT_HOST, envvar="VOX_HOST", help="Vox server URL")
@click.pass_context
def cli(ctx, host: str):
    """Vox — Universal local runtime for STT and TTS models."""
    ctx.ensure_object(dict)
    ctx.obj["host"] = host.rstrip("/")


@cli.command()
@click.option("--port", default=11435, help="HTTP port to listen on")
@click.option("--grpc-port", default=9090, help="gRPC port to listen on (0 to disable)")
@click.option("--host", "bind_host", default="0.0.0.0", help="Host to bind to")
@click.option("--device", default="auto", envvar="VOX_DEVICE", help="Device: auto, cuda, cpu, mps")
@click.option("--max-loaded", default=3, help="Max models loaded simultaneously")
@click.option("--ttl", default=300, help="Idle model TTL in seconds")
def serve(port: int, grpc_port: int, bind_host: str, device: str, max_loaded: int, ttl: int):
    """Start the Vox server."""
    import uvicorn

    from vox.server.app import create_app

    app = create_app(
        default_device=device,
        max_loaded=max_loaded,
        ttl_seconds=ttl,
        grpc_port=grpc_port if grpc_port > 0 else None,
    )
    uvicorn.run(app, host=bind_host, port=port, log_level="info")


@cli.command()
@click.argument("model")
@click.pass_context
def pull(ctx, model: str):
    """Download a model."""
    host = ctx.obj["host"]
    had_error = False
    try:
        with httpx.stream("POST", f"{host}/api/pull", json={"name": model}, timeout=None) as resp:
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    if status == "error":
                        click.echo(f"  Error: {data.get('error', 'unknown error')}", err=True)
                        had_error = True
                    elif "completed" in data and "total" in data:
                        click.echo(f"  {status} [{data['completed']}/{data['total']}]")
                    else:
                        click.echo(f"  {status}")
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        _handle_request_error(e, host)

    if had_error:
        sys.exit(1)


@cli.command("list")
@click.pass_context
def list_models(ctx):
    """List downloaded models."""
    host = ctx.obj["host"]
    try:
        resp = httpx.get(f"{host}/api/list", timeout=10)
        _check_response(resp)
        models = resp.json().get("models", [])
        if not models:
            click.echo("No models downloaded. Pull one with: vox pull <model>")
            return
        click.echo(f"{'NAME':<30} {'TYPE':<6} {'FORMAT':<8} {'SIZE':<12} DESCRIPTION")
        for m in models:
            size = _format_size(m.get("size_bytes", 0))
            click.echo(
                f"{m['name']:<30} {m['type']:<6} "
                f"{m.get('format', ''):<8} {size:<12} {m.get('description', '')[:50]}"
            )
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        _handle_request_error(e, host)


@cli.command()
@click.argument("model")
@click.pass_context
def show(ctx, model: str):
    """Show model details."""
    host = ctx.obj["host"]
    try:
        resp = httpx.post(f"{host}/api/show", json={"name": model}, timeout=10)
        _check_response(resp)
        click.echo(json.dumps(resp.json(), indent=2))
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        _handle_request_error(e, host)


@cli.command()
@click.argument("model")
@click.pass_context
def rm(ctx, model: str):
    """Remove a downloaded model."""
    host = ctx.obj["host"]
    try:
        resp = httpx.request("DELETE", f"{host}/api/delete", json={"name": model}, timeout=10)
        _check_response(resp)
        click.echo(f"Deleted {model}")
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        _handle_request_error(e, host)


@cli.command()
@click.pass_context
def ps(ctx):
    """List currently loaded/running models."""
    host = ctx.obj["host"]
    try:
        resp = httpx.get(f"{host}/api/ps", timeout=10)
        _check_response(resp)
        models = resp.json().get("models", [])
        if not models:
            click.echo("No models currently loaded")
            return
        click.echo(f"{'NAME':<30} {'TYPE':<6} {'DEVICE':<8} {'REFS':<6}")
        for m in models:
            name = f"{m['name']}:{m['tag']}"
            click.echo(f"{name:<30} {m['type']:<6} {m['device']:<8} {m['ref_count']:<6}")
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        _handle_request_error(e, host)


@cli.command()
@click.argument("model")
@click.argument("input_arg")
@click.option("-o", "--output", default=None, help="Output file for TTS")
@click.option("--language", default=None)
@click.option("--voice", default=None)
@click.pass_context
def run(ctx, model: str, input_arg: str, output: str | None, language: str | None, voice: str | None):
    """Run a one-shot transcription or synthesis.

    For STT: vox run whisper:large-v3 audio.wav
    For TTS: vox run kokoro:v1.0 "Hello world" -o output.wav
    """
    host = ctx.obj["host"]
    input_path = Path(input_arg)

    try:
        if input_path.is_file():
            with open(input_path, "rb") as f:
                resp = httpx.post(
                    f"{host}/api/transcribe",
                    files={"file": (input_path.name, f)},
                    data={"model": model, "language": language or ""},
                    timeout=None,
                )
            _check_response(resp)
            click.echo(resp.json().get("text", ""))
        else:
            resp = httpx.post(
                f"{host}/api/synthesize",
                json={"model": model, "input": input_arg, "voice": voice, "response_format": "wav"},
                timeout=None,
            )
            _check_response(resp)
            out_path = output or "output.wav"
            with open(out_path, "wb") as f:
                f.write(resp.content)
            click.echo(f"Audio saved to {out_path}")
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        _handle_request_error(e, host)


@cli.command("stream-transcribe")
@click.argument("model")
@click.argument("audio_file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--language", default=None)
@click.option("--word-timestamps", is_flag=True, default=False)
@click.option(
    "--send-chunk-ms",
    default=DEFAULT_STREAM_CHUNK_MS,
    show_default=True,
    help="Client-side PCM upload chunk size in milliseconds",
)
@click.option("--json-output", is_flag=True, default=False, help="Print the final done payload as JSON")
@click.pass_context
def stream_transcribe(
    ctx,
    model: str,
    audio_file: Path,
    language: str | None,
    word_timestamps: bool,
    send_chunk_ms: int,
    json_output: bool,
):
    """Stream a local audio file over the long-form WS transcription API."""
    host = ctx.obj["host"]
    ws_url = _to_websocket_url(host, "/v1/audio/transcriptions/stream")

    try:
        chunks = _iter_pcm16_audio_chunks(audio_file, target_rate=TARGET_SAMPLE_RATE, chunk_ms=send_chunk_ms)
        with connect(ws_url, compression=None, max_queue=None, max_size=None, open_timeout=30) as ws:
            _send_ws_json(ws, {
                "type": "config",
                "model": model,
                "input_format": "pcm16",
                "sample_rate": TARGET_SAMPLE_RATE,
                "language": language,
                "word_timestamps": word_timestamps,
            })
            _receive_ready_event(ws)

            for chunk in chunks:
                ws.send(chunk)
                _drain_transcribe_events(ws, wait_for_done=False)

            _send_ws_json(ws, {"type": "end"})
            done = _drain_transcribe_events(ws, wait_for_done=True)
            if done is None:
                raise click.ClickException("Transcription stream ended before the final result arrived")
    except click.ClickException:
        raise
    except Exception as exc:
        raise click.ClickException(f"Streaming transcription failed: {exc}") from exc

    click.echo(
        (
            f"[done] duration={int(done.get('duration_ms', 0)) / 1000:.1f}s "
            f"processing={int(done.get('processing_ms', 0)) / 1000:.1f}s"
        ),
        err=True,
    )
    if json_output:
        click.echo(json.dumps(done, indent=2))
    else:
        click.echo(str(done.get("text", "")))


@cli.command("stream-synthesize")
@click.argument("model")
@click.argument("input_arg")
@click.option("-o", "--output", default="output.wav", show_default=True, type=click.Path(path_type=Path))
@click.option("--language", default=None)
@click.option("--voice", default=None)
@click.option("--speed", default=1.0, show_default=True, type=float)
@click.pass_context
def stream_synthesize(
    ctx,
    model: str,
    input_arg: str,
    output: Path,
    language: str | None,
    voice: str | None,
    speed: float,
):
    """Stream TTS audio over the long-form WS synthesis API and write a WAV file."""
    host = ctx.obj["host"]
    ws_url = _to_websocket_url(host, "/v1/audio/speech/stream")
    input_path = Path(input_arg)
    text = input_path.read_text(encoding="utf-8") if input_path.is_file() else input_arg
    text_chunks = _chunk_text_for_stream(text)
    if not text_chunks:
        raise click.ClickException("No input text provided")

    writer: wave.Wave_write | None = None
    cleanup_output = False
    try:
        with connect(ws_url, compression=None, max_queue=None, max_size=None, open_timeout=30) as ws:
            _send_ws_json(ws, {
                "type": "config",
                "model": model,
                "voice": voice,
                "language": language,
                "speed": speed,
                "response_format": "pcm16",
            })
            _receive_ready_event(ws)

            for chunk in text_chunks:
                _send_ws_json(ws, {"type": "text", "text": chunk})
                writer, _ = _drain_tts_events(
                    ws,
                    output_path=output,
                    writer=writer,
                    wait_for_done=False,
                )

            _send_ws_json(ws, {"type": "end"})
            writer, done = _drain_tts_events(
                ws,
                output_path=output,
                writer=writer,
                wait_for_done=True,
            )
            if done is None:
                raise click.ClickException("Synthesis stream ended before the final result arrived")
            if writer is None:
                raise click.ClickException("Synthesis completed without returning any audio")
    except click.ClickException:
        cleanup_output = True
        raise
    except Exception as exc:
        cleanup_output = True
        raise click.ClickException(f"Streaming synthesis failed: {exc}") from exc
    finally:
        if writer is not None:
            writer.close()
        if cleanup_output and output.exists():
            output.unlink()

    click.echo(
        (
            f"[done] audio={int(done.get('audio_duration_ms', 0)) / 1000:.1f}s "
            f"processing={int(done.get('processing_ms', 0)) / 1000:.1f}s"
        ),
        err=True,
    )
    click.echo(f"Audio saved to {output}")


@cli.command()
@click.argument("model")
@click.pass_context
def voices(ctx, model: str):
    """List voices for a TTS model."""
    host = ctx.obj["host"]
    try:
        resp = httpx.get(f"{host}/api/voices", params={"model": model}, timeout=10)
        _check_response(resp)
        voices_list = resp.json().get("voices", [])
        if not voices_list:
            click.echo(f"No voices found for {model}")
            return
        click.echo(f"{'ID':<20} {'NAME':<30} {'LANGUAGE':<10}")
        for v in voices_list:
            click.echo(f"{v['id']:<20} {v['name']:<30} {v.get('language') or '':<10}")
    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as e:
        _handle_request_error(e, host)


@cli.command()
@click.option("--type", "model_type", default=None, help="Filter by type: stt or tts")
def search(model_type: str | None):
    """Search available models from the registry."""
    from vox.core.registry import fetch_registry_index

    index = fetch_registry_index()
    if not index:
        click.echo("Error: could not reach the model registry", err=True)
        sys.exit(1)

    if model_type:
        index = [m for m in index if m.get("type") == model_type]

    if not index:
        click.echo("No models found")
        return

    click.echo(f"{'NAME':<30} {'TYPE':<6} DESCRIPTION")
    for m in index:
        name = f"{m['name']}:{m['tag']}"
        click.echo(f"{name:<30} {m.get('type', ''):<6} {m.get('description', '')[:50]}")


def _format_size(size_bytes: int) -> str:
    if size_bytes == 0:
        return "-"
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
