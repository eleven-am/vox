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
    assert "adapter-package-template.md" in contract


def test_adapter_package_template_documents_standard_shape():
    template = Path("docs/adapter-package-template.md").read_text(encoding="utf-8")

    required_sections = (
        "## Directory Layout",
        "## `pyproject.toml`",
        "## README Sections",
        "## Runtime Bootstrap Pattern",
        "## Tests",
        "## Current Adapter Runtime Matrix",
        "## Review Checklist",
    )
    for section in required_sections:
        assert section in template

    assert "[tool.vox.adapter]" in template
    assert "runtime-policy" in template
    assert "$VOX_HOME/runtime/<runtime-name>" in template
    assert "venv-exception" in template
    assert "vox-voxtral" in template
