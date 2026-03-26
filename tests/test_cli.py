"""Tests for the Vox CLI (vox.cli)."""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, Mock, patch, mock_open

import httpx
import pytest
from click.testing import CliRunner

from vox.cli import (
    _check_response,
    _format_size,
    _handle_request_error,
    cli,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# _format_size
# ---------------------------------------------------------------------------

class TestFormatSize:
    def test_zero(self):
        assert _format_size(0) == "-"

    def test_bytes(self):
        assert _format_size(500) == "500.0 B"

    def test_kilobytes(self):
        assert _format_size(2048) == "2.0 KB"

    def test_megabytes(self):
        assert _format_size(5 * 1024 * 1024) == "5.0 MB"

    def test_gigabytes(self):
        assert _format_size(3 * 1024 ** 3) == "3.0 GB"

    def test_terabytes(self):
        assert _format_size(2 * 1024 ** 4) == "2.0 TB"


# ---------------------------------------------------------------------------
# _handle_request_error
# ---------------------------------------------------------------------------

class TestHandleRequestError:
    def test_connect_error(self, capsys):
        with pytest.raises(SystemExit, match="1"):
            _handle_request_error(
                httpx.ConnectError("conn refused"),
                "http://localhost:11435",
            )
        err = capsys.readouterr().err
        assert "cannot connect" in err
        assert "vox serve" in err

    def test_timeout_error(self, capsys):
        with pytest.raises(SystemExit, match="1"):
            _handle_request_error(
                httpx.TimeoutException("timed out"),
                "http://localhost:11435",
            )
        err = capsys.readouterr().err
        assert "timed out" in err

    def test_http_status_error(self, capsys):
        response = Mock()
        response.status_code = 500
        request = Mock()
        exc = httpx.HTTPStatusError(
            "server error", request=request, response=response,
        )
        with pytest.raises(SystemExit, match="1"):
            _handle_request_error(exc, "http://localhost:11435")
        err = capsys.readouterr().err
        assert "500" in err

    def test_generic_error(self, capsys):
        with pytest.raises(SystemExit, match="1"):
            _handle_request_error(RuntimeError("boom"), "http://localhost:11435")
        err = capsys.readouterr().err
        assert "boom" in err


# ---------------------------------------------------------------------------
# _check_response
# ---------------------------------------------------------------------------

class TestCheckResponse:
    def test_ok_status(self):
        resp = Mock()
        resp.status_code = 200
        _check_response(resp)  # should not exit

    def test_error_status_with_json_detail(self, capsys):
        resp = Mock()
        resp.status_code = 404
        resp.json.return_value = {"detail": "model not found"}
        with pytest.raises(SystemExit, match="1"):
            _check_response(resp)
        assert "model not found" in capsys.readouterr().err

    def test_error_status_json_no_detail(self, capsys):
        resp = Mock()
        resp.status_code = 500
        resp.json.return_value = {}
        resp.text = "internal server error"
        with pytest.raises(SystemExit, match="1"):
            _check_response(resp)
        assert "internal server error" in capsys.readouterr().err

    def test_error_status_json_parse_fails(self, capsys):
        resp = Mock()
        resp.status_code = 502
        resp.json.side_effect = ValueError("bad json")
        resp.text = "bad gateway"
        with pytest.raises(SystemExit, match="1"):
            _check_response(resp)
        assert "bad gateway" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------

class TestServe:
    @patch("vox.cli.uvicorn", create=True)
    @patch("vox.server.app.create_app")
    def test_serve_calls_uvicorn(self, mock_create_app, mock_uvicorn, runner):
        """Verify serve calls create_app then uvicorn.run."""
        mock_app = MagicMock()
        mock_create_app.return_value = mock_app

        # We need to patch the imports inside the serve function
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            mock_uvicorn.run = MagicMock()
            result = runner.invoke(cli, ["serve", "--port", "9999", "--host", "127.0.0.1"])

        mock_create_app.assert_called_once()
        mock_uvicorn.run.assert_called_once()
        call_kwargs = mock_uvicorn.run.call_args
        assert call_kwargs[1]["port"] == 9999 or call_kwargs[0][0] is mock_app


# ---------------------------------------------------------------------------
# pull
# ---------------------------------------------------------------------------

class TestPull:
    @patch("httpx.stream")
    def test_pull_success_with_progress(self, mock_stream, runner):
        lines = [
            json.dumps({"status": "downloading", "completed": 50, "total": 100}),
            json.dumps({"status": "done"}),
        ]
        ctx = MagicMock()
        ctx.__enter__ = Mock(return_value=ctx)
        ctx.__exit__ = Mock(return_value=False)
        ctx.iter_lines.return_value = iter(lines)
        mock_stream.return_value = ctx

        result = runner.invoke(cli, ["pull", "whisper:large-v3"])
        assert result.exit_code == 0
        assert "downloading [50/100]" in result.output
        assert "done" in result.output

    @patch("httpx.stream")
    def test_pull_error_status_in_stream(self, mock_stream, runner):
        lines = [
            json.dumps({"status": "error", "error": "model not found"}),
        ]
        ctx = MagicMock()
        ctx.__enter__ = Mock(return_value=ctx)
        ctx.__exit__ = Mock(return_value=False)
        ctx.iter_lines.return_value = iter(lines)
        mock_stream.return_value = ctx

        result = runner.invoke(cli, ["pull", "bad-model"])
        assert result.exit_code != 0

    @patch("httpx.stream")
    def test_pull_connect_error(self, mock_stream, runner):
        mock_stream.side_effect = httpx.ConnectError("refused")
        result = runner.invoke(cli, ["pull", "whisper:large-v3"])
        assert result.exit_code != 0
        assert "cannot connect" in result.output or "cannot connect" in (result.output + (result.output or ""))


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

class TestList:
    @patch("httpx.get")
    def test_list_empty(self, mock_get, runner):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"models": []}
        mock_get.return_value = resp

        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "No models downloaded" in result.output

    @patch("httpx.get")
    def test_list_populated(self, mock_get, runner):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "models": [
                {
                    "name": "whisper:large-v3",
                    "type": "stt",
                    "format": "ggml",
                    "size_bytes": 3_000_000_000,
                    "description": "OpenAI Whisper large v3",
                },
            ]
        }
        mock_get.return_value = resp

        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "whisper:large-v3" in result.output
        assert "stt" in result.output

    @patch("httpx.get")
    def test_list_connect_error(self, mock_get, runner):
        mock_get.side_effect = httpx.ConnectError("refused")
        result = runner.invoke(cli, ["list"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------

class TestShow:
    @patch("httpx.post")
    def test_show_success(self, mock_post, runner):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"name": "whisper:large-v3", "type": "stt", "parameters": {}}
        mock_post.return_value = resp

        result = runner.invoke(cli, ["show", "whisper:large-v3"])
        assert result.exit_code == 0
        assert "whisper:large-v3" in result.output

    @patch("httpx.post")
    def test_show_404(self, mock_post, runner):
        resp = Mock()
        resp.status_code = 404
        resp.json.return_value = {"detail": "model not found"}
        mock_post.return_value = resp

        result = runner.invoke(cli, ["show", "nonexistent"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# rm
# ---------------------------------------------------------------------------

class TestRm:
    @patch("httpx.request")
    def test_rm_success(self, mock_request, runner):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {}
        mock_request.return_value = resp

        result = runner.invoke(cli, ["rm", "whisper:large-v3"])
        assert result.exit_code == 0
        assert "Deleted whisper:large-v3" in result.output

    @patch("httpx.request")
    def test_rm_404(self, mock_request, runner):
        resp = Mock()
        resp.status_code = 404
        resp.json.return_value = {"detail": "model not found"}
        mock_request.return_value = resp

        result = runner.invoke(cli, ["rm", "nonexistent"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# ps
# ---------------------------------------------------------------------------

class TestPs:
    @patch("httpx.get")
    def test_ps_empty(self, mock_get, runner):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"models": []}
        mock_get.return_value = resp

        result = runner.invoke(cli, ["ps"])
        assert result.exit_code == 0
        assert "No models currently loaded" in result.output

    @patch("httpx.get")
    def test_ps_populated(self, mock_get, runner):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "models": [
                {
                    "name": "whisper",
                    "tag": "large-v3",
                    "type": "stt",
                    "device": "cuda",
                    "ref_count": 1,
                },
            ]
        }
        mock_get.return_value = resp

        result = runner.invoke(cli, ["ps"])
        assert result.exit_code == 0
        assert "whisper:large-v3" in result.output
        assert "cuda" in result.output

    @patch("httpx.get")
    def test_ps_connect_error(self, mock_get, runner):
        mock_get.side_effect = httpx.ConnectError("refused")
        result = runner.invoke(cli, ["ps"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# run — STT (file input)
# ---------------------------------------------------------------------------

class TestRunSTT:
    @patch("httpx.post")
    def test_run_stt_transcription(self, mock_post, runner, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"text": "hello world"}
        mock_post.return_value = resp

        result = runner.invoke(cli, ["run", "whisper:large-v3", str(audio_file)])
        assert result.exit_code == 0
        assert "hello world" in result.output

    @patch("httpx.post")
    def test_run_stt_error(self, mock_post, runner, tmp_path):
        audio_file = tmp_path / "test.wav"
        audio_file.write_bytes(b"RIFF" + b"\x00" * 100)

        resp = Mock()
        resp.status_code = 500
        resp.json.return_value = {"detail": "inference failed"}
        mock_post.return_value = resp

        result = runner.invoke(cli, ["run", "whisper:large-v3", str(audio_file)])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# run — TTS (text input)
# ---------------------------------------------------------------------------

class TestRunTTS:
    @patch("httpx.post")
    def test_run_tts_default_output(self, mock_post, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        resp = Mock()
        resp.status_code = 200
        resp.content = b"\x00\x01\x02\x03"
        mock_post.return_value = resp

        result = runner.invoke(cli, ["run", "kokoro:v1.0", "Hello world"])
        assert result.exit_code == 0
        assert "Audio saved to output.wav" in result.output
        assert (tmp_path / "output.wav").read_bytes() == b"\x00\x01\x02\x03"

    @patch("httpx.post")
    def test_run_tts_custom_output(self, mock_post, runner, tmp_path):
        out_file = tmp_path / "speech.wav"

        resp = Mock()
        resp.status_code = 200
        resp.content = b"audio-data"
        mock_post.return_value = resp

        result = runner.invoke(
            cli, ["run", "kokoro:v1.0", "Hello world", "-o", str(out_file)]
        )
        assert result.exit_code == 0
        assert f"Audio saved to {out_file}" in result.output
        assert out_file.read_bytes() == b"audio-data"

    @patch("httpx.post")
    def test_run_tts_with_voice(self, mock_post, runner, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        resp = Mock()
        resp.status_code = 200
        resp.content = b"audio"
        mock_post.return_value = resp

        result = runner.invoke(
            cli, ["run", "kokoro:v1.0", "Hi", "--voice", "af_heart"]
        )
        assert result.exit_code == 0
        # Verify the voice was passed in the request body
        call_kwargs = mock_post.call_args
        body = call_kwargs[1].get("json") or call_kwargs.kwargs.get("json")
        assert body["voice"] == "af_heart"

    @patch("httpx.post")
    def test_run_tts_connect_error(self, mock_post, runner):
        mock_post.side_effect = httpx.ConnectError("refused")
        result = runner.invoke(cli, ["run", "kokoro:v1.0", "Hello"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# voices
# ---------------------------------------------------------------------------

class TestVoices:
    @patch("httpx.get")
    def test_voices_empty(self, mock_get, runner):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"voices": []}
        mock_get.return_value = resp

        result = runner.invoke(cli, ["voices", "kokoro:v1.0"])
        assert result.exit_code == 0
        assert "No voices found" in result.output

    @patch("httpx.get")
    def test_voices_populated(self, mock_get, runner):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "voices": [
                {"id": "af_heart", "name": "Heart", "language": "en"},
                {"id": "af_sky", "name": "Sky", "language": "en"},
            ]
        }
        mock_get.return_value = resp

        result = runner.invoke(cli, ["voices", "kokoro:v1.0"])
        assert result.exit_code == 0
        assert "af_heart" in result.output
        assert "Heart" in result.output
        assert "af_sky" in result.output

    @patch("httpx.get")
    def test_voices_with_no_language(self, mock_get, runner):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {
            "voices": [
                {"id": "v1", "name": "Voice One"},
            ]
        }
        mock_get.return_value = resp

        result = runner.invoke(cli, ["voices", "kokoro:v1.0"])
        assert result.exit_code == 0
        assert "v1" in result.output

    @patch("httpx.get")
    def test_voices_connect_error(self, mock_get, runner):
        mock_get.side_effect = httpx.ConnectError("refused")
        result = runner.invoke(cli, ["voices", "kokoro:v1.0"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# --host option / env var
# ---------------------------------------------------------------------------

class TestHostOption:
    @patch("httpx.get")
    def test_custom_host_flag(self, mock_get, runner):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"models": []}
        mock_get.return_value = resp

        result = runner.invoke(cli, ["--host", "http://myserver:8080", "list"])
        assert result.exit_code == 0
        mock_get.assert_called_once()
        url = mock_get.call_args[0][0]
        assert url.startswith("http://myserver:8080")

    @patch("httpx.get")
    def test_host_trailing_slash_stripped(self, mock_get, runner):
        resp = Mock()
        resp.status_code = 200
        resp.json.return_value = {"models": []}
        mock_get.return_value = resp

        result = runner.invoke(cli, ["--host", "http://myserver:8080/", "list"])
        url = mock_get.call_args[0][0]
        assert not url.startswith("http://myserver:8080//")


# ---------------------------------------------------------------------------
# Timeout error path
# ---------------------------------------------------------------------------

class TestTimeoutPath:
    @patch("httpx.get")
    def test_list_timeout(self, mock_get, runner):
        mock_get.side_effect = httpx.TimeoutException("timed out")
        result = runner.invoke(cli, ["list"])
        assert result.exit_code != 0

    @patch("httpx.post")
    def test_show_timeout(self, mock_post, runner):
        mock_post.side_effect = httpx.TimeoutException("timed out")
        result = runner.invoke(cli, ["show", "model"])
        assert result.exit_code != 0
