from __future__ import annotations

import os
import subprocess
from pathlib import Path

ENTRYPOINT = Path(__file__).resolve().parents[1] / "docker" / "vox-entrypoint.sh"


def _run_entrypoint(tmp_path: Path, *, tmpdir: Path | None = None) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("install", "chown"):
        script = bin_dir / name
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)

    gosu = bin_dir / "gosu"
    gosu.write_text('#!/bin/sh\nshift\nexec "$@"\n')
    gosu.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "VOX_HOME": str(tmp_path / "vox-home"),
    }
    if tmpdir is None:
        env.pop("TMPDIR", None)
    else:
        env["TMPDIR"] = str(tmpdir)

    result = subprocess.run(
        [
            "/bin/sh",
            str(ENTRYPOINT),
            "/usr/bin/env",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    return values["TMPDIR"]


def test_entrypoint_defaults_transient_files_to_container_local_storage(tmp_path: Path) -> None:
    assert _run_entrypoint(tmp_path) == "/tmp/vox"


def test_entrypoint_preserves_explicit_tmpdir(tmp_path: Path) -> None:
    configured = tmp_path / "operator-tmp"

    assert _run_entrypoint(tmp_path, tmpdir=configured) == str(configured)
