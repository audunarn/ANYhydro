# SPDX-License-Identifier: MPL-2.0
"""ADR-015 L6-4 H-d: vessel source-study recipe (identity, round trip) and one synthetic end-to-end level."""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

capytaine = pytest.importorskip("capytaine", reason="optional dependency not installed")

from anyhydro import CapytaineBackend                                       # noqa: E402
from anyhydro.capytaine_backend import CapytaineExecutionLimits            # noqa: E402
from anyhydro.vessel_study import (KernelRecipe, VesselMeshLevel, VesselStudyError, VesselStudyRecipe,  # noqa: E402
                                   run_vessel_level)

from test_capytaine_backend import _box_body                               # noqa: E402

FULL_BLOCK = dict(omega_min=0.8, omega_max=2.5, block_rel_tol=0.05, modal_rel_tol=0.05,
                  energetic_fraction=1e-3, cluster_fraction=0.01)


def _recipe(**overrides):
    body = _box_body("hull", (0.3, -0.2, -0.3))
    data = dict(title="synthetic box", body_id="hull", levels=(VesselMeshLevel("coarse", body.mesh, body.lid_mesh),),
                rotation_center=(0.3, -0.2, -0.3), center_of_mass=(0.3, -0.2, -0.3), rho=1025.0, gravity=9.81,
                water_depth=math.inf, omegas=tuple(np.linspace(0.3, 3.0, 12)), headings=(0.0, math.pi / 2),
                retained_dofs=("surge", "sway", "heave", "roll", "pitch", "yaw"), length_scale=2.0,
                kernel=KernelRecipe(dt=0.1, n_time=401, bartlett_window=20.0, symmetry_tol=1e-6, passivity_tol=1e-6),
                full_block=FULL_BLOCK)
    data.update(overrides)
    return VesselStudyRecipe(**data)


def test_recipe_identity_round_trip_and_hash_coverage():
    recipe = _recipe()
    back = VesselStudyRecipe.from_dict(recipe.to_dict(), recipe.levels)
    assert back.content_hash() == recipe.content_hash()
    assert _recipe(length_scale=3.0).content_hash() != recipe.content_hash()
    assert recipe.full_block_criteria().dof_scales == (1.0, 1.0, 1.0, 0.5, 0.5, 0.5)
    other = _box_body("hull", (0.0, 0.0, -0.3))
    with pytest.raises(VesselStudyError, match="mesh identities"):
        VesselStudyRecipe.from_dict(recipe.to_dict(), (VesselMeshLevel("coarse", other.mesh, None),))
    with pytest.raises(VesselStudyError, match="canonical order"):
        _recipe(retained_dofs=("heave", "surge"))
    with pytest.raises(VesselStudyError, match="Bartlett window"):
        KernelRecipe(dt=0.1, n_time=11, bartlett_window=5.0, symmetry_tol=1e-6, passivity_tol=1e-6)


def test_synthetic_level_end_to_end():
    """The pipeline produces and binds the evidence. Whether a source passes the normalised gate and passivity is a
    property of the mesh, band and taper (the coarse 100-panel test box does not: its damping samples are not all
    PSD and the 20 s Bartlett taper smooths its peaked spectrum) - judged in the physical campaign after L6-F2."""
    recipe = _recipe()
    backend = CapytaineBackend(execution_limits=CapytaineExecutionLimits(
        max_total_panels=200, max_finite_omegas=len(recipe.omegas), max_wave_directions=1))
    result = run_vessel_level(recipe, "coarse", backend)
    summary = result.summary()
    assert summary["headings"] == 2 and summary["recipe_hash"] == recipe.content_hash()
    assert result.contract_admissible, result.contract_reasons
    assert summary["kernel_passivity_violations"] == result.kernel.diagnostics.passivity_violations
    assert np.isfinite(result.symbol_min_eigenvalue)
    assert result.report.criteria.dof_scales == recipe.dof_scales()
    assert all(ds.attrs["hydrostatic_contract"] for ds in result.datasets)
    assert set(summary) >= {"full_block_pass", "max_impedance_rel_error"}
    with pytest.raises(VesselStudyError, match="unknown level"):
        run_vessel_level(recipe, "fine", backend)
