"""Base-import isolation tests: no optional Capytaine or Qt at import time."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_base_import_excludes_optional_dependencies():
    code = (
        "import sys; import anyhydro; "
        "mods = [m for m in sys.modules if m.split('.')[0] in "
        "('capytaine', 'PySide6', 'PyQt5', 'PyQt6')]; "
        "assert not mods, f'optional modules imported: {mods}'; "
        "assert anyhydro.SCHEMA_VERSION; "
        "print('base-import-clean')"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "base-import-clean" in result.stdout


def test_public_api_surface():
    import anyhydro

    for name in (
        "HydroPanelMesh",
        "HydroBody",
        "HydroRequest",
        "validate_hydro_dataset",
        "validate_for_time_domain",
        "assemble_hydro_dataset",
        "save_hydro_dataset",
        "load_hydro_dataset",
        "shift_dataset_reference_point",
        "reorder_dataset",
        "ImportedHydroDatasetBackend",
        "dataset_content_hash",
        "coefficient_frame_axes",
        "require_global_dof_axes",
        "rotate_axes",
        "reframe_dataset",
        "CoefficientFrame",
        "HydrostaticContract",
        "attach_hydrostatic_contract",
        "read_hydrostatic_contract",
        "hydrostatic_composition_eligibility",
        "require_hydrostatic_composition",
        "HydrostaticContractError",
    ):
        assert hasattr(anyhydro, name), name
