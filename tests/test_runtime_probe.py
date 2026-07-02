from __future__ import annotations

from unittest.mock import patch

import vox.core.runtime as rt


class _FakeResult:
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


def test_nvidia_smi_probe_parses_driver_vram_and_cuda_from_header():
    # cuda_version is not a --query-gpu field; querying it errors the whole call.
    # The probe must query only valid fields and read CUDA version from the header.
    def fake_run(cmd, **kwargs):
        joined = " ".join(cmd)
        if "--query-gpu=driver_version,memory.total" in joined:
            return _FakeResult(0, "550.54.15, 24576\n")
        return _FakeResult(
            0,
            "| NVIDIA-SMI 550.54  Driver Version: 550.54  CUDA Version: 12.4 |\n",
        )

    with (
        patch("vox.core.runtime.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("vox.core.runtime.subprocess.run", side_effect=fake_run),
    ):
        probe = rt._nvidia_smi_probe()

    assert probe.available is True
    assert probe.driver_version == "550.54.15"
    assert probe.cuda_version == "12.4"
    assert probe.vram_gb is not None
    assert 23.0 < probe.vram_gb < 25.0
