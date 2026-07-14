from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

FAILURE_CLASSES = ("none", "Vox", "adapter", "dependency", "upstream", "hardware")


def failure_classification_reasons(
    *,
    failure_reasons: list[str],
    failure_class: str,
    failure_note: str,
) -> list[str]:
    if failure_reasons and failure_class == "none":
        return [
            "failing smoke run must set --failure-class to one of "
            "Vox, adapter, dependency, upstream, or hardware"
        ]
    if failure_reasons and failure_class != "none" and not failure_note.strip():
        return ["classified failing smoke run must include --failure-note"]
    if not failure_reasons and failure_class != "none":
        return ["passing smoke run must use --failure-class none"]
    if not failure_reasons and failure_note.strip():
        return ["passing smoke run must not set --failure-note"]
    return []


def write_evidence(
    evidence: dict[str, Any],
    output_dir: Path,
    *,
    filename: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    rendered = json.dumps(evidence, indent=2, sort_keys=True)
    path.write_text(rendered, encoding="utf-8")
    print(rendered)
    print(f"Evidence written to {path}", file=sys.stderr)
    return path
