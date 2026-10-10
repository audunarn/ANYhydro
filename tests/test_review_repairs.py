"""Permanent regressions for the independent-review repairs.

Each test pins one repaired defect from the parent review
(hydro-independent-01/02): dataset leakage, incomplete body-identity
checking, false zero-frequency capability, non-finite physical constants,
mutable request settings, and body_identity consistency through transforms.
"""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

from anyhydro import (
    BodyIdentityMismatchError,
    HydroBody,
    HydroPanelMesh,
    ImportedHydroDatasetBackend,
    RecordValidationError,
    shift_dataset_reference_point,
)
from anyhydro.fixtures import (
    FIXTURE_BODY_IDS,
    make_fixture_bodies,
    make_fixture_request,
    make_two_body_fixture,
)


@pytest.fixture()
def backend():
    return ImportedHydroDatasetBackend(make_two_body_fixture())


class TestDatasetPropertyIsolation:
    def test_property_returns_isolated_copy(self, backend):
        exposed = backend.dataset
        expected = backend.dataset.added_mass.values.copy()
        exposed.added_mass.values[:] = 0
        np.testing.assert_array_equal(backend.dataset.added_mass.values, expected)

    def test_property_copy_is_not_the_internal_object(self, backend):
        assert backend.dataset is not backend.dataset


class TestFullBodyIdentityChecking:
    def test_rejects_changed_rotation_center(self, backend):
        bodies = make_fixture_bodies()
        changed = replace(bodies[0], rotation_center=(9.0, 2.0, 3.0))
        with pytest.raises(BodyIdentityMismatchError):
            backend.solve((changed, bodies[1]), make_fixture_request())

    def test_rejects_changed_center_of_mass(self, backend):
        bodies = make_fixture_bodies()
        changed = replace(bodies[0], center_of_mass=(5.0, 0.0, 0.4))
        with pytest.raises(BodyIdentityMismatchError):
            backend.solve((changed, bodies[1]), make_fixture_request())

    def test_rejects_changed_enabled_dofs(self, backend):
        bodies = make_fixture_bodies()
        changed = replace(bodies[0], enabled_dofs=bodies[0].enabled_dofs[:3])
        with pytest.raises(BodyIdentityMismatchError):
            backend.solve((changed, bodies[1]), make_fixture_request())

    def test_rejects_added_lid_mesh(self, backend):
        bodies = make_fixture_bodies()
        alpha = bodies[0]
        lid = HydroPanelMesh(
            vertices=alpha.mesh.vertices + np.array([0.0, 0.0, 0.9]),
            faces=alpha.mesh.faces,
            face_sizes=alpha.mesh.face_sizes,
            body_id=alpha.body_id,
            outward_normals_verified=True,
            source_geometry_refs=("test-lid",),
            mesh_hash="",
        )
        changed = replace(alpha, lid_mesh=lid)
        with pytest.raises(BodyIdentityMismatchError):
            backend.solve((changed, bodies[1]), make_fixture_request())

    def test_accepts_exact_identity(self, backend):
        result = backend.solve(make_fixture_bodies(), make_fixture_request())
        assert "added_mass" in result.data_vars


class TestZeroFrequencyCapability:
    def test_positive_frequency_data_does_not_advertise_zero_limit(self, backend):
        assert not backend.capabilities().supports_zero_frequency

    def test_zero_limit_note_is_present(self, backend):
        notes = " ".join(backend.capabilities().notes)
        assert "supports_zero_frequency is always False" in notes


class TestPhysicalConstants:
    @pytest.mark.parametrize("field", ["rho", "gravity"])
    def test_must_be_finite_positive(self, field):
        for bad in (float("inf"), float("-inf"), float("nan"), 0.0, -1.0):
            with pytest.raises(RecordValidationError):
                replace(make_fixture_request(), **{field: bad})

    def test_deep_water_infinite_depth_is_valid(self):
        request = replace(make_fixture_request(), water_depth=float("inf"))
        assert request.water_depth == float("inf")

    @pytest.mark.parametrize("bad", [0.0, -10.0, float("-inf"), float("nan")])
    def test_invalid_water_depth_rejected(self, bad):
        with pytest.raises(RecordValidationError):
            replace(make_fixture_request(), water_depth=bad)

    def test_deep_water_hash_differs_from_finite_depth(self):
        deep = replace(make_fixture_request(), water_depth=float("inf"))
        finite = replace(make_fixture_request(), water_depth=150.0)
        assert deep.content_hash() != finite.content_hash()


class TestBackendSettingsImmutability:
    def test_top_level_mutation_blocked(self):
        request = make_fixture_request()
        with pytest.raises(TypeError):
            request.backend_settings["rho"] = 1.0

    def test_nested_mutation_blocked(self):
        request = make_fixture_request()
        settings = {"outer": {"inner": [1, 2]}, "seq": [{"a": 1}]}
        request = replace(request, backend_settings=settings)
        with pytest.raises(TypeError):
            request.backend_settings["outer"]["inner"] = [9]
        # frozen sequences have no mutating methods at all
        with pytest.raises((TypeError, AttributeError)):
            request.backend_settings["outer"]["inner"].append(9)
        with pytest.raises(TypeError):
            request.backend_settings["seq"][0]["a"] = 9
        with pytest.raises((TypeError, AttributeError)):
            request.backend_settings["seq"].append(0)

    def test_nested_freeze_preserves_hash_identity(self):
        plain = replace(
            make_fixture_request(), backend_settings={"a": {"b": [1, 2]}}
        )
        again = replace(
            make_fixture_request(), backend_settings={"a": {"b": [1, 2]}}
        )
        assert plain.content_hash() == again.content_hash()

    def test_caller_cannot_reach_record_state_through_settings(self):
        source = {"nested": {"x": 1}}
        request = replace(make_fixture_request(), backend_settings=source)
        source["nested"]["x"] = 999
        assert request.backend_settings["nested"]["x"] == 1


class TestBodyIdentityThroughTransforms:
    def test_shift_keeps_body_identity_consistent(self):
        ds = make_two_body_fixture()
        shifted = shift_dataset_reference_point(
            ds, {FIXTURE_BODY_IDS[0]: (1.0, -0.5, 0.25),
                 FIXTURE_BODY_IDS[1]: (12.0, 0.0, -0.5)}
        )
        identity = json.loads(shifted.attrs["body_identity"])
        assert identity[FIXTURE_BODY_IDS[0]]["rotation_center"] == [1.0, -0.5, 0.25]
        # unchanged identity fields survive the shift
        assert identity[FIXTURE_BODY_IDS[0]]["center_of_mass"] == (
            json.loads(ds.attrs["body_identity"])[FIXTURE_BODY_IDS[0]]["center_of_mass"]
        )
        assert identity[FIXTURE_BODY_IDS[0]]["mesh_hash"] == (
            json.loads(ds.attrs["body_identity"])[FIXTURE_BODY_IDS[0]]["mesh_hash"]
        )

    def test_shifted_dataset_admits_only_shifted_identity(self):
        ds = make_two_body_fixture()
        shifted = shift_dataset_reference_point(
            ds, {FIXTURE_BODY_IDS[0]: (1.0, -0.5, 0.25),
                 FIXTURE_BODY_IDS[1]: (12.0, 0.0, -0.5)}
        )
        backend = ImportedHydroDatasetBackend(shifted)
        bodies = make_fixture_bodies()
        with pytest.raises(BodyIdentityMismatchError):
            backend.solve(bodies, make_fixture_request())
        moved = replace(bodies[0], rotation_center=(1.0, -0.5, 0.25))
        result = backend.solve((moved, bodies[1]), make_fixture_request())
        assert "added_mass" in result.data_vars
