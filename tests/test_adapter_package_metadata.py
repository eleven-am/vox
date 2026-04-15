from __future__ import annotations

import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTERS_ROOT = REPO_ROOT / "adapters"


def test_all_adapter_packages_have_readme_metadata_and_file():
    pyprojects = sorted(ADAPTERS_ROOT.glob("*/pyproject.toml"))
    assert pyprojects, "expected adapter packages under adapters/"

    for pyproject_path in pyprojects:
        data = tomllib.loads(pyproject_path.read_text())
        project = data["project"]
        readme = project.get("readme")
        assert readme, f"{pyproject_path} is missing [project].readme"
        assert isinstance(readme, dict), f"{pyproject_path} should use table-form readme metadata"
        assert readme.get("content-type") == "text/markdown"

        readme_file = pyproject_path.parent / str(readme["file"])
        assert readme_file.is_file(), f"{readme_file} does not exist"
        content = readme_file.read_text().strip()
        assert content, f"{readme_file} is empty"
        assert content.startswith(f"# {project['name']}"), (
            f"{readme_file} should start with a markdown title matching the package name"
        )
        assert project["name"] in content, f"{readme_file} should mention the package name"
        assert "## Install" in content, f"{readme_file} should include an install section"
        assert f"pip install {project['name']}" in content, (
            f"{readme_file} should include the package install command"
        )


def test_all_adapter_packages_have_short_description():
    pyprojects = sorted(ADAPTERS_ROOT.glob("*/pyproject.toml"))
    assert pyprojects, "expected adapter packages under adapters/"

    for pyproject_path in pyprojects:
        data = tomllib.loads(pyproject_path.read_text())
        description = data["project"].get("description", "").strip()
        assert description, f"{pyproject_path} is missing a project description"
