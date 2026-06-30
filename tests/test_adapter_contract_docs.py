from __future__ import annotations

from pathlib import Path


def test_adapter_contract_documents_packaging_boundaries():
    contract = Path("docs/adapter-contract.md").read_text(encoding="utf-8")

    required_sections = (
        "## What Belongs In `vox-runtime`",
        "## What Belongs In Each Adapter Package",
        "## What Belongs In `$VOX_HOME/runtime/<runtime-name>`",
        "## What Must Never Be Bundled In The Base Vox Image",
    )
    for section in required_sections:
        assert section in contract

    assert "$VOX_HOME/adapters/<adapter-package>" in contract
    assert "model weights" in contract
    assert "adapter packages such as" in contract
