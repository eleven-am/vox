from __future__ import annotations

from pathlib import Path


def test_adapter_runtime_dependency_policy_documents_resolution_and_repair_rules():
    policy = Path("docs/adapter-runtime-dependency-policy.md").read_text(encoding="utf-8")

    required_sections = (
        "## Dependency Classes",
        "## `--upgrade` Policy",
        "## Verification After Install",
        "## Repairing Stale Or Broken Runtime Directories",
    )
    for section in required_sections:
        assert section in policy

    assert "Exact Pins" in policy
    assert "Bounded Ranges" in policy
    assert "Broad Ranges" in policy
    assert "successful `pip` or `uv`" in policy
    assert "exit code is not enough" in policy
    assert "sentinel must not bypass" in policy
