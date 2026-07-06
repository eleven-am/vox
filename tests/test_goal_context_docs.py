from __future__ import annotations

from pathlib import Path


def test_goal_required_adapter_runtime_contract_path_exists_and_points_to_current_docs():
    contract = Path("docs/adapter-runtime-contract.md")

    assert contract.is_file()

    content = contract.read_text()
    assert "adapter-contract.md" in content
    assert "adapter-runtime-dependency-policy.md" in content
