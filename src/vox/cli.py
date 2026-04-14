from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import httpx


DEFAULT_HOST = "http://localhost:11435"


def _handle_request_error(e: Exception, host: str) -> None:
    """Print a user-friendly error for HTTP request failures."""
    if isinstance(e, httpx.ConnectError):
        click.echo(f"Error: cannot connect to Vox server at {host}", err=True)
        click.echo("Start the server with: vox serve", err=True)
    elif isinstance(e, httpx.TimeoutException):
        click.echo(f"Error: request to Vox server timed out", err=True)
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
@click.option("--device", default="auto", help="Device: auto, cuda, cpu, mps")
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
            click.echo(f"{m['name']:<30} {m['type']:<6} {m.get('format', ''):<8} {size:<12} {m.get('description', '')[:50]}")
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
