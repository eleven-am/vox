from __future__ import annotations

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ADAPTERS_ROOT = REPO_ROOT / "adapters"
VALID_RUNTIME_POLICIES = {"target-runtime", "package-runtime", "mixed", "venv-exception"}
VALID_ADAPTER_TYPES = {"stt", "tts"}


def _adapter_pyprojects() -> list[Path]:
    pyprojects = sorted(ADAPTERS_ROOT.glob("*/pyproject.toml"))
    assert pyprojects, "expected adapter packages under adapters/"
    return pyprojects


def _load_pyproject(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def test_all_adapter_packages_have_readme_metadata_and_file():
    for pyproject_path in _adapter_pyprojects():
        data = _load_pyproject(pyproject_path)
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
        assert "## Runtime Dependencies" in content, (
            f"{readme_file} should describe adapter runtime dependency placement"
        )
        assert "## Use with Vox" in content, f"{readme_file} should include a Vox usage section"


def test_all_adapter_packages_have_short_description():
    for pyproject_path in _adapter_pyprojects():
        data = _load_pyproject(pyproject_path)
        description = data["project"].get("description", "").strip()
        assert description, f"{pyproject_path} is missing a project description"


def test_all_adapter_packages_depend_on_vox_runtime_and_use_hatchling():
    for pyproject_path in _adapter_pyprojects():
        data = _load_pyproject(pyproject_path)
        project = data["project"]
        dependencies = project.get("dependencies", [])
        assert any(dep.startswith("vox-runtime") for dep in dependencies), (
            f"{pyproject_path} should depend on vox-runtime"
        )

        assert data["build-system"]["build-backend"] == "hatchling.build"
        assert data["build-system"]["requires"] == ["hatchling"]


def test_all_adapter_packages_declare_vox_adapter_metadata():
    for pyproject_path in _adapter_pyprojects():
        data = _load_pyproject(pyproject_path)
        metadata = data.get("tool", {}).get("vox", {}).get("adapter")
        assert metadata, f"{pyproject_path} is missing [tool.vox.adapter]"

        import_package = metadata.get("import-package")
        assert isinstance(import_package, str) and import_package, (
            f"{pyproject_path} should declare tool.vox.adapter.import-package"
        )
        assert (pyproject_path.parent / "src" / import_package / "__init__.py").is_file(), (
            f"{pyproject_path} import package {import_package!r} does not exist under src/"
        )

        runtime_policy = metadata.get("runtime-policy")
        assert runtime_policy in VALID_RUNTIME_POLICIES, (
            f"{pyproject_path} has invalid runtime-policy {runtime_policy!r}"
        )

        runtime_names = metadata.get("runtime-names")
        assert isinstance(runtime_names, list), f"{pyproject_path} should declare runtime-names as a list"
        assert all(isinstance(name, str) and name for name in runtime_names), (
            f"{pyproject_path} runtime-names should contain non-empty strings"
        )
        if runtime_policy in {"target-runtime", "mixed", "venv-exception"}:
            assert runtime_names, f"{pyproject_path} should list owned runtime names"

        adapter_types = metadata.get("adapter-types")
        assert isinstance(adapter_types, list) and adapter_types, (
            f"{pyproject_path} should declare adapter-types"
        )
        assert set(adapter_types) <= VALID_ADAPTER_TYPES, (
            f"{pyproject_path} adapter-types must be stt, tts, or both"
        )

        if runtime_policy == "venv-exception":
            exceptions = metadata.get("venv-exceptions")
            assert isinstance(exceptions, list) and exceptions, (
                f"{pyproject_path} should list venv-exceptions"
            )
            assert set(exceptions) <= set(runtime_names), (
                f"{pyproject_path} venv-exceptions must be listed in runtime-names"
            )


def test_all_adapter_packages_have_valid_vox_entry_points_and_wheel_packages():
    entry_point_pattern = re.compile(r"^[a-z0-9][a-z0-9-]*$")
    target_pattern = re.compile(r"^[a-zA-Z_][\w.]*:[a-zA-Z_]\w*$")

    for pyproject_path in _adapter_pyprojects():
        data = _load_pyproject(pyproject_path)
        metadata = data["tool"]["vox"]["adapter"]
        import_package = metadata["import-package"]

        entry_points = data["project"]["entry-points"]["vox.adapters"]
        assert entry_points, f"{pyproject_path} should expose at least one vox.adapters entry point"
        for name, target in entry_points.items():
            assert entry_point_pattern.match(name), f"{pyproject_path} has invalid entry point {name!r}"
            assert target_pattern.match(target), f"{pyproject_path} has invalid target {target!r}"
            assert target.startswith(f"{import_package}."), (
                f"{pyproject_path} entry point {name!r} should target {import_package}"
            )

        wheel_packages = data["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        assert wheel_packages == [f"src/{import_package}"], (
            f"{pyproject_path} wheel package should match import-package"
        )


def test_externalized_adapter_runtime_dependencies_are_not_package_dependencies():
    expected_absent = {
        "vox-dia": ("transformers", "sentencepiece"),
        "vox-neutts": ("neutts", "torch", "transformers"),
        "vox-piper": ("piper-tts",),
        "vox-sesame": ("transformers", "sentencepiece"),
        "vox-spark": ("transformers", "torch", "einops"),
        "vox-xtts": ("coqui-tts", "torchaudio"),
    }

    for pyproject_path in _adapter_pyprojects():
        data = _load_pyproject(pyproject_path)
        project = data["project"]
        dependencies = project.get("dependencies", [])
        forbidden = expected_absent.get(project["name"], ())
        for package in forbidden:
            assert not any(dep.split(";", 1)[0].strip().startswith(package) for dep in dependencies), (
                f"{pyproject_path} should bootstrap {package} into its runtime directory"
            )
