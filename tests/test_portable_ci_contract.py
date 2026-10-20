"""Contract for the portable base-CI selection.

Verifies that tools/run_portable_checks.py classifies every test module in
this repository, that the selected modules cannot import optional ecosystem
packages at module import time, and that the GitHub base CI workflow invokes
the same selection.  This keeps CI selection changes explicit and inspectable
instead of implicit collection behavior.
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO_ROOT / "tools" / "run_portable_checks.py"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
DOCS_PATH = REPO_ROOT / "docs" / "PORTABLE_CI.md"


def _load_tool():
    spec = importlib.util.spec_from_file_location("run_portable_checks", TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


def _test_module_names():
    return sorted(p.stem for p in (REPO_ROOT / "tests").glob("test_*.py"))


def _module_level_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_every_test_module_is_classified(tool):
    classified = set(tool.PORTABLE_TEST_MODULES) | set(tool.EXCLUDED_TEST_MODULES)
    assert classified == set(_test_module_names())
    assert not set(tool.PORTABLE_TEST_MODULES) & set(tool.EXCLUDED_TEST_MODULES)


def test_selected_modules_exist_and_are_selected_exactly_once(tool):
    for name in tool.PORTABLE_TEST_MODULES:
        assert (REPO_ROOT / "tests" / f"{name}.py").is_file(), name
    assert len(set(tool.PORTABLE_TEST_MODULES)) == len(tool.PORTABLE_TEST_MODULES)


def test_portable_modules_never_import_optional_packages_at_module_level(tool):
    for name in tool.PORTABLE_TEST_MODULES:
        roots = _module_level_import_roots(REPO_ROOT / "tests" / f"{name}.py")
        assert not roots & set(tool.BLOCKED_OPTIONAL_PACKAGES), (
            f"{name} imports optional packages at module level: "
            f"{sorted(roots & set(tool.BLOCKED_OPTIONAL_PACKAGES))}"
        )


def test_portable_modules_never_import_excluded_test_modules(tool):
    for name in tool.PORTABLE_TEST_MODULES:
        roots = _module_level_import_roots(REPO_ROOT / "tests" / f"{name}.py")
        assert not roots & set(tool.EXCLUDED_TEST_MODULES), (
            f"{name} imports an excluded test module: {sorted(roots & set(tool.EXCLUDED_TEST_MODULES))}"
        )


def test_exclusions_have_concrete_documented_reasons(tool):
    for name, reason in tool.EXCLUDED_TEST_MODULES.items():
        assert (REPO_ROOT / "tests" / f"{name}.py").is_file(), name
        assert isinstance(reason, str) and len(reason) > 40, name
        assert "Remaining gate:" in reason, name


def test_docs_document_the_selection_and_every_exclusion(tool):
    docs = DOCS_PATH.read_text(encoding="utf-8")
    for name in tool.EXCLUDED_TEST_MODULES:
        assert name in docs, f"docs/PORTABLE_CI.md does not document {name}"
    for name in tool.PORTABLE_TEST_MODULES:
        assert name in docs, f"docs/PORTABLE_CI.md does not document {name}"


def test_ci_workflow_invokes_the_portable_selection():
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "tools/run_portable_checks.py" in workflow
    assert "pytest tests" not in workflow, (
        "base CI must not collect the whole tests/ tree; excluded modules have "
        "documented optional dependency or local fixture reasons"
    )


def test_slice_selection_covers_the_allowlist_exactly(tool):
    total = 3
    union: list[str] = []
    for index in range(1, total + 1):
        union.extend(tool.select_modules(f"{index}/{total}"))
    assert union == list(tool.PORTABLE_TEST_MODULES)


def test_licensed_orcfxapi_is_blocked_in_the_portable_runner(tool):
    """The runner must never execute licensed OrcFxAPI tests, even on a developer machine that has it installed."""
    assert "OrcFxAPI" in tool.BLOCKED_OPTIONAL_PACKAGES
    code = "\n".join([
        "import importlib.util, importlib.metadata as md",
        f"spec = importlib.util.spec_from_file_location('run_portable_checks', {str(TOOL_PATH)!r})",
        "tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)",
        "tool.install_optional_dependency_blockers()",
        "try:",
        "    import OrcFxAPI",
        "except ModuleNotFoundError:",
        "    pass",
        "else:",
        "    raise SystemExit('OrcFxAPI imported')",
        "assert importlib.util.find_spec('OrcFxAPI') is None",
        "for name in ('OrcFxAPI', 'orcfxapi'):",
        "    try:",
        "        md.version(name)",
        "    except md.PackageNotFoundError:",
        "        pass",
        "    else:",
        "        raise SystemExit('distribution visible: ' + name)",
    ])
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_orcaflex_licensed_tests_skip_through_importorskip():
    """The licensed tests of the export module reach OrcFxAPI only via importorskip, so the blocker turns them into skips."""
    path = REPO_ROOT / "tests" / "test_orcaflex_export.py"
    assert 'pytest.importorskip("OrcFxAPI")' in path.read_text(encoding="utf-8")
    assert "OrcFxAPI" not in _module_level_import_roots(path)
