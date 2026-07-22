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
    SpeechContextError,
    collect_speech_context_evidence,
    install_experimental_runtimes,
    runtime_inventory,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect internal Parakeet, prosody, and audio-event evidence for one audio file."
    )
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="Override VOX_HOME for isolated experiment runtimes.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    install = commands.add_parser("install", help="Install isolated experimental analyzer runtimes.")
    install.add_argument(
        "--accept-opensmile-research-license",
        action="store_true",
        help="Acknowledge that openSMILE is research/non-commercial software and is not bundled by Vox.",
    )

    commands.add_parser("inventory", help="Show isolated analyzer runtime and model sizes.")

    analyze = commands.add_parser("analyze", help="Analyze an audio file and write complete JSON evidence.")
    analyze.add_argument("audio_file", type=Path)
    analyze.add_argument("--output", type=Path, default=None)
    analyze.add_argument("--vox-url", default=os.environ.get("VOX_URL", "http://127.0.0.1:11435"))
    analyze.add_argument("--api-key", default=os.environ.get("VOX_API_KEY"))
    analyze.add_argument("--model", default=DEFAULT_MODEL)
    analyze.add_argument("--timeout", type=float, default=300.0)
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
            _write_json(
                install_experimental_runtimes(
                    accept_opensmile_research_license=args.accept_opensmile_research_license,
                    home=args.home,
                )
            )
            return 0
        if args.command == "inventory":
            _write_json(runtime_inventory(home=args.home))
            return 0

        output = args.output or args.audio_file.with_suffix(".speech-context.json")
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
        return 0 if all(result["status"] == "ok" for result in evidence["results"].values()) else 3
    except SpeechContextError as error:
        print(f"speech-context evidence failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
