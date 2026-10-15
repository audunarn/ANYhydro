# SPDX-License-Identifier: MPL-2.0
"""ADR-015 L6-4 H-d: vessel source-study recipe identity, immutability and strictness (portable, no BEM)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from anyhydro.records import HydroPanelMesh
from anyhydro.vessel_study import KernelRecipe, VesselMeshLevel, VesselStudyError, VesselStudyRecipe

FULL_BLOCK = dict(omega_min=0.8, omega_max=2.5, block_rel_tol=0.05, modal_rel_tol=0.05,
                  energetic_fraction=1e-3, cluster_fraction=0.01)
PRODUCER = dict(backend="capytaine", backend_version="3.0.0", adapter_version="0.6.0")


def _panel(dz=0.0):
    verts = np.array([[0, 0, -1 - dz], [1, 0, -1 - dz], [1, 1, -1 - dz], [0, 1, -1 - dz]], float)
    return HydroPanelMesh(vertices=verts, faces=np.array([[0, 1, 2, 3]]), face_sizes=np.array([4], np.uint8),
                          body_id="hull", outward_normals_verified=False, source_geometry_refs=("test",), mesh_hash="")


def _kernel(**overrides):
    data = dict(dt=0.1, n_time=401, bartlett_window=20.0, symmetry_tol=1e-6, passivity_tol=1e-6, max_time_samples=401)
    data.update(overrides)
    return KernelRecipe(**data)


def _recipe(**overrides):
    data = dict(title="synthetic", body_id="hull", levels=(VesselMeshLevel("coarse", _panel()),),
                rotation_center=(0.3, -0.2, -0.3), center_of_mass=(0.3, -0.2, -0.3), rho=1025.0, gravity=9.81,
                water_depth=math.inf, omegas=tuple(np.linspace(0.3, 3.0, 12)), headings=(0.0, math.pi / 2),
                length_scale=2.0, kernel=_kernel(), full_block=FULL_BLOCK, producer=PRODUCER)
    data.update(overrides)
    return VesselStudyRecipe(**data)


def test_round_trip_and_hash_coverage():
    recipe = _recipe()
    back = VesselStudyRecipe.from_dict(recipe.to_dict(), recipe.levels)
    assert back.content_hash() == recipe.content_hash()
    for change in (dict(length_scale=3.0), dict(producer=dict(PRODUCER, backend_version="3.0.1")),
                   dict(kernel=_kernel(max_time_samples=1001)), dict(method="indirect"),
                   dict(levels=(VesselMeshLevel("coarse", _panel(0.1)),))):
        assert _recipe(**change).content_hash() != recipe.content_hash(), change
    assert recipe.full_block_criteria().dof_scales == (1.0, 1.0, 1.0, 0.5, 0.5, 0.5)
    assert recipe.retained_dofs == ("surge", "sway", "heave", "roll", "pitch", "yaw")


def test_criteria_and_producer_are_immutable():
    recipe = _recipe()
    with pytest.raises(TypeError):
        recipe.full_block["block_rel_tol"] = 1e9
    with pytest.raises(TypeError):
        recipe.producer["backend_version"] = "x"


def test_from_dict_is_strict():
    recipe = _recipe()
    data = recipe.to_dict()
    for bad in (dict(data, rho=True), {k: v for k, v in data.items() if k != "method"}, dict(data, extra=1),
                dict(data, kernel=dict(data["kernel"], unknown=1)), dict(data, retained_dofs=["heave"])):
        with pytest.raises(VesselStudyError):
            VesselStudyRecipe.from_dict(bad, recipe.levels)
    with pytest.raises(VesselStudyError, match="mesh identities"):
        VesselStudyRecipe.from_dict(data, (VesselMeshLevel("coarse", _panel(0.2)),))


@pytest.mark.parametrize("change, message", [
    (dict(omegas=(0.5, 1.0)), "omegas"),
    (dict(rho=math.inf), "rho must be finite"),
    (dict(gravity=True), "gravity must be a number"),
    (dict(producer=dict(backend="capytaine")), "producer"),
])
def test_invalid_recipes_are_refused_before_any_solve(change, message):
    with pytest.raises(VesselStudyError, match=message):
        _recipe(**change)


def test_kernel_recipe_validation():
    with pytest.raises(VesselStudyError, match="Bartlett window"):
        _kernel(n_time=11, max_time_samples=11, bartlett_window=5.0)
    with pytest.raises(VesselStudyError, match="max_time_samples"):
        _kernel(n_time=2001, max_time_samples=1001)
    with pytest.raises(VesselStudyError, match="not admissible"):
        _kernel(n_time=30001, max_time_samples=30001, bartlett_window=20.0)



def test_malformed_containers_and_empty_windows_are_typed_refusals():
    recipe = _recipe()
    data = recipe.to_dict()
    for key in ("retained_dofs", "levels", "omegas", "kernel", "producer"):
        with pytest.raises(VesselStudyError):
            VesselStudyRecipe.from_dict(dict(data, **{key: None}), recipe.levels)
    with pytest.raises(VesselStudyError, match="evaluation window"):
        _recipe(full_block=dict(FULL_BLOCK, omega_min=10.0, omega_max=20.0))
