"""A1: the radiation-kernel tolerance basis is expressible in, and recorded by, the vessel-study recipe."""
import copy
import json

import pytest

from anyhydro.vessel_study import KernelRecipe, VesselStudyError, VesselStudyRecipe
from test_vessel_study import _kernel, _recipe, _panel, VesselMeshLevel

REL = "relative_to_sample_norm"


def test_relative_basis_is_expressible_and_reaches_the_settings():
    k = _kernel(tolerance_basis=REL)
    assert k.settings().tolerance_basis == REL
    assert k.to_dict()["tolerance_basis"] == REL


def test_default_basis_is_absolute_and_omitted():
    k = _kernel()
    assert k.settings().tolerance_basis == "absolute"
    assert "tolerance_basis" not in k.to_dict()


def test_invalid_basis_refused():
    with pytest.raises(VesselStudyError, match="tolerance_basis"):
        _kernel(tolerance_basis="relative")
    with pytest.raises(VesselStudyError, match="tolerance_basis"):
        _kernel(tolerance_basis=1)


def test_relative_recipe_round_trip_and_hash_differs():
    rel, ab = _recipe(kernel=_kernel(tolerance_basis=REL)), _recipe()
    assert rel.content_hash() != ab.content_hash()
    data = json.loads(json.dumps(rel.to_dict()))
    back = VesselStudyRecipe.from_dict(data, rel.levels)
    assert back.content_hash() == rel.content_hash() and back.kernel.tolerance_basis == REL


def test_explicit_absolute_in_a_record_is_refused():
    ab = _recipe()
    data = copy.deepcopy(ab.to_dict())
    data["kernel"]["tolerance_basis"] = "absolute"
    with pytest.raises(VesselStudyError):
        VesselStudyRecipe.from_dict(data, ab.levels)


def test_unknown_kernel_key_still_refused():
    ab = _recipe()
    data = copy.deepcopy(ab.to_dict())
    data["kernel"]["bogus"] = 1
    with pytest.raises(VesselStudyError):
        VesselStudyRecipe.from_dict(data, ab.levels)
    data = copy.deepcopy(ab.to_dict())
    del data["kernel"]["dt"]
    with pytest.raises(VesselStudyError):
        VesselStudyRecipe.from_dict(data, ab.levels)


def test_run_with_relative_basis_records_it_on_the_kernel():
    pytest.importorskip("capytaine")
    import dataclasses
    from test_vessel_study_run import _recipe as run_recipe, _backend
    from anyhydro.vessel_study import run_vessel_level
    base = run_recipe()
    recipe = dataclasses.replace(base, kernel=dataclasses.replace(base.kernel, symmetry_tol=1e-2, tolerance_basis=REL))
    result = run_vessel_level(recipe, "coarse", _backend(recipe))
    assert result.kernel.settings.tolerance_basis == REL
    assert result.kernel.diagnostics.max_damping_asymmetry_ratio is not None
    assert result.recipe_hash == recipe.content_hash() != base.content_hash()
