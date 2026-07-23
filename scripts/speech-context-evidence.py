#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from vox.speech_context.runner import (
    DEFAULT_MODEL,
    collect_speech_context_evidence,
    collect_speech_context_service_evidence,
)
from vox.speech_context.runtime import (
    SpeechContextError,
    install_speech_context_runtimes,
    runtime_inventory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Install and exercise Vox's isolated speech-context service.")
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="Override VOX_HOME for isolated experiment runtimes.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("install", help="Install isolated SenseVoice and YAMNet runtimes.")

    commands.add_parser("inventory", help="Show isolated analyzer runtime and model sizes.")

    analyze = commands.add_parser("analyze", help="Analyze an audio file and write complete JSON evidence.")
    analyze.add_argument("audio_file", type=Path)
    analyze.add_argument("--output", type=Path, default=None)
    analyze.add_argument("--vox-url", default=os.environ.get("VOX_URL", "http://127.0.0.1:11435"))
    analyze.add_argument("--api-key", default=os.environ.get("VOX_API_KEY"))
    analyze.add_argument("--model", default=DEFAULT_MODEL)
    analyze.add_argument("--timeout", type=float, default=300.0)

    service = commands.add_parser(
        "analyze-service",
        help="Run only the production SpeechContextService; no Vox server or STT required.",
    )
    service.add_argument("audio_file", type=Path)
    service.add_argument("--output", type=Path, default=None)
    service.add_argument("--timeout", type=float, default=300.0)
    return parser


def _write_json(payload: dict[str, object], destination: Path | None = None) -> None:
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    if destination is None:
        sys.stdout.write(rendered)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    print(destination)


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "install":
            _write_json(install_speech_context_runtimes(home=args.home))
            return 0
        if args.command == "inventory":
            _write_json(runtime_inventory(home=args.home))
            return 0

        output = args.output or args.audio_file.with_suffix(".speech-context.json")
        if args.command == "analyze-service":
            evidence = asyncio.run(
                collect_speech_context_service_evidence(
                    args.audio_file,
                    timeout=args.timeout,
                    home=args.home,
                )
            )
            _write_json(evidence, output)
            return 0 if evidence["speech_context"]["status"] == "complete" else 3

        evidence = asyncio.run(
            collect_speech_context_evidence(
                args.audio_file,
                base_url=args.vox_url,
                api_key=args.api_key,
                model=args.model,
                timeout=args.timeout,
                home=args.home,
            )
        )
        _write_json(evidence, output)
        complete = (
            all(result["status"] == "ok" for result in evidence["results"].values())
            and evidence["speech_context"]["status"] == "complete"
        )
        return 0 if complete else 3
    except SpeechContextError as error:
        print(f"speech-context evidence failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
