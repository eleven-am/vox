from __future__ import annotations

import os
from pathlib import Path


def configure_hf_runtime() -> None:
    """Set stable Hugging Face cache/runtime defaults for Vox.

    Vox mostly runs inside long-lived service containers with writable PVC-backed
    caches. Disabling Xet by default avoids flaky ranged-download behavior we
    have seen with large model artifacts on cluster storage. Users can still
    override this explicitly via environment.
    """

    hf_home = Path(os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    hub_cache = Path(os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub")))
    xet_cache = Path(os.environ.setdefault("HF_XET_CACHE", str(hf_home / "xet")))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    for path in (
        hf_home,
        hub_cache,
        xet_cache,
        xet_cache / "logs",
        xet_cache / "chunk-cache",
        xet_cache / "shard-cache",
    ):
        path.mkdir(parents=True, exist_ok=True)
