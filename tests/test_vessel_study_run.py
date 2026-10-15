# SPDX-License-Identifier: MPL-2.0
"""ADR-015 L6-4 H-d: one synthetic vessel-study level end to end (Capytaine) and the response checks."""
from __future__ import annotations

import math

import numpy as np
import pytest

capytaine = pytest.importorskip("capytaine", reason="optional dependency not installed")

from anyhydro import CapytaineBackend                                       # noqa: E402
from anyhydro.capytaine_backend import CAPYTAINE_ADAPTER_VERSION, CapytaineExecutionLimits   # noqa: E402
from anyhydro.vessel_study import (KernelRecipe, VesselMeshLevel, VesselStudyError, VesselStudyRecipe,  # noqa: E402
                                   run_vessel_level)

from test_capytaine_backend import _box_body                               # noqa: E402

FULL_BLOCK = dict(omega_min=0.8, omega_max=2.5, block_rel_tol=0.05, modal_rel_tol=0.05,
                  energetic_fraction=1e-3, cluster_fraction=0.01)


def _recipe():
    body = _box_body("hull", (0.3, -0.2, -0.3))
    return VesselStudyRecipe(
        title="synthetic box", body_id="hull", levels=(VesselMeshLevel("coarse", body.mesh, body.lid_mesh),),
        rotation_center=(0.3, -0.2, -0.3), center_of_mass=(0.3, -0.2, -0.3), rho=1025.0, gravity=9.81,
        water_depth=math.inf, omegas=tuple(np.linspace(0.3, 3.0, 12)), headings=(0.0, math.pi / 2), length_scale=2.0,
        kernel=KernelRecipe(dt=0.1, n_time=401, bartlett_window=20.0, symmetry_tol=1e-6, passivity_tol=1e-6,
                            max_time_samples=401),
        full_block=FULL_BLOCK,
        producer=dict(backend="capytaine", backend_version=capytaine.__version__,
                      adapter_version=CAPYTAINE_ADAPTER_VERSION))


def _backend(recipe):
    return CapytaineBackend(execution_limits=CapytaineExecutionLimits(
        max_total_panels=200, max_finite_omegas=len(recipe.omegas), max_wave_directions=1))


def test_synthetic_level_end_to_end_binds_the_evidence():
    """Whether a source passes the gate and passivity depends on mesh, band and taper (this coarse box does not:
    its damping samples are not all PSD and a 20 s taper smooths its peaked spectrum) - judged in the physical
    campaign after L6-F2. Here: the pipeline produces and binds the evidence."""
    recipe = _recipe()
    result = run_vessel_level(recipe, "coarse", _backend(recipe))
    summary = result.summary()
    assert summary["headings"] == 2 and summary["recipe_hash"] == recipe.content_hash()
    assert summary["kernel_source_dataset_hash"] == summary["dataset_hashes"][0]
    assert summary["kernel_hash"] == result.kernel.content_hash()
    assert result.contract_admissible, result.contract_reasons
    assert summary["kernel_passivity_violations"] == result.kernel.diagnostics.passivity_violations
    assert np.isfinite(result.symbol_min_eigenvalue)
    assert result.report.criteria.dof_scales == recipe.dof_scales()


class _Tamper:
    """Wraps the real backend and tampers with responses (replay, altered coordinates, foreign producer)."""

    def __init__(self, inner, mode):
        self.inner, self.mode, self.first = inner, mode, None

    def solve(self, bodies, request):
        ds = self.inner.solve(bodies, request)
        if self.mode == "replay":
            if self.first is None:
                self.first = ds
            return self.first
        if self.mode == "omega":
            return ds.assign_coords(omega=ds["omega"].values * 2.0)
        if self.mode == "producer":
            out = ds.copy()
            out.attrs["backend_version"] = "0.0.0"
            return out
        return ds


@pytest.mark.parametrize("mode, message", [("replay", "heading"), ("omega", "frequencies"), ("producer", "producer")])
def test_responses_are_verified_against_their_requests(mode, message):
    recipe = _recipe()
    with pytest.raises(VesselStudyError, match=message):
        run_vessel_level(recipe, "coarse", _Tamper(_backend(recipe), mode))
    with pytest.raises(VesselStudyError, match="unknown level"):
        run_vessel_level(recipe, "fine", _backend(recipe))
