"""Backend tests: imported-dataset replay, mismatch rejection, diagnostics."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from anyhydro import (
    BodyIdentityMismatchError,
    HydroError,
    ImportedHydroDatasetBackend,
    RequestMismatchError,
    validate_hydro_dataset,
)
from anyhydro.fixtures import make_fixture_bodies, make_fixture_request


@pytest.fixture()
def backend(two_body_dataset):
    return ImportedHydroDatasetBackend(two_body_dataset)


class TestSolve:
    def test_returns_validated_data_for_matching_inputs(self, backend, two_body_dataset):
        result = backend.solve(make_fixture_bodies(), make_fixture_request())
        validate_hydro_dataset(result)
        for var in ("added_mass", "excitation_force", "hydrostatic_stiffness"):
            assert np.array_equal(
                result[var].values, two_body_dataset[var].values
            ), var

    def test_rejects_request_mismatch(self, backend):
        request = make_fixture_request()
        request = type(request)(
            omegas=request.omegas * 1.01,
            wave_directions=request.wave_directions,
            water_depth=request.water_depth,
            rho=request.rho,
            gravity=request.gravity,
            include_zero_frequency=request.include_zero_frequency,
            include_infinite_frequency=request.include_infinite_frequency,
            backend_settings=request.backend_settings,
        )
        with pytest.raises(RequestMismatchError):
            backend.solve(make_fixture_bodies(), request)

    def test_rejects_body_identity_mismatch(self, backend):
        from anyhydro import HydroBody, HydroPanelMesh, body_dof_keys

        alpha = make_fixture_bodies()[0]
        other_mesh = HydroPanelMesh(
            vertices=alpha.mesh.vertices,
            faces=alpha.mesh.faces,
            face_sizes=alpha.mesh.face_sizes,
            body_id="some-other-body",
            outward_normals_verified=False,
            source_geometry_refs=(),
            mesh_hash="",
        )
        other = HydroBody(
            body_id="some-other-body",
            name=alpha.name,
            mesh=other_mesh,
            lid_mesh=None,
            rotation_center=alpha.rotation_center,
            center_of_mass=alpha.center_of_mass,
            enabled_dofs=body_dof_keys("some-other-body"),
        )
        bodies = (other,) + make_fixture_bodies()[1:]
        with pytest.raises(BodyIdentityMismatchError):
            backend.solve(bodies, make_fixture_request())

    def test_rejects_mesh_hash_mismatch(self, backend):
        bodies = make_fixture_bodies()
        alpha = bodies[0]
        different = type(alpha.mesh)(
            vertices=alpha.mesh.vertices * 1.5,
            faces=alpha.mesh.faces,
            face_sizes=alpha.mesh.face_sizes,
            body_id=alpha.mesh.body_id,
            outward_normals_verified=False,
            source_geometry_refs=(),
            mesh_hash="",
        )
        new_bodies = list(bodies)
        new_bodies[0] = type(alpha)(
            body_id=alpha.body_id,
            name=alpha.name,
            mesh=different,
            lid_mesh=None,
            rotation_center=alpha.rotation_center,
            center_of_mass=alpha.center_of_mass,
            enabled_dofs=alpha.enabled_dofs,
        )
        with pytest.raises(BodyIdentityMismatchError):
            backend.solve(new_bodies, make_fixture_request())

    def test_caller_inputs_unchanged(self, backend):
        bodies = make_fixture_bodies()
        request = make_fixture_request()
        bodies_snapshot = copy.deepcopy(
            [
                (b.body_id, b.mesh.mesh_hash, b.enabled_dofs, b.rotation_center)
                for b in bodies
            ]
        )
        request_hash = request.content_hash()
        backend.solve(bodies, request)
        assert [
            (b.body_id, b.mesh.mesh_hash, b.enabled_dofs, b.rotation_center)
            for b in bodies
        ] == bodies_snapshot
        assert request.content_hash() == request_hash

    def test_returned_copy_is_isolated_from_backend(self, backend):
        result = backend.solve(make_fixture_bodies(), make_fixture_request())
        result["added_mass"].values[0, 0, 0] = -1.0
        fresh = backend.solve(make_fixture_bodies(), make_fixture_request())
        assert fresh["added_mass"].values[0, 0, 0] != -1.0

    def test_rejects_non_record_inputs(self, backend):
        with pytest.raises(HydroError):
            backend.solve([{"not": "a body"}], make_fixture_request())
        with pytest.raises(HydroError):
            backend.solve(make_fixture_bodies(), {"not": "a request"})


class TestCapabilities:
    def test_capabilities_reflect_dataset(self, backend, two_body_dataset):
        caps = backend.capabilities()
        assert caps.backend_name == "synthetic-fixture"
        assert caps.provides_added_mass_infinite is True
        assert caps.supports_multibody is True
        assert "no BEM solve capability" in caps.notes[0]

    def test_capabilities_without_a_inf(self, two_body_dataset_no_a_inf):
        caps = ImportedHydroDatasetBackend(two_body_dataset_no_a_inf).capabilities()
        assert caps.provides_added_mass_infinite is False
        assert caps.supports_infinite_frequency is False


class TestDiagnose:
    def test_diagnose_is_honest(self, backend, two_body_dataset):
        diagnostics = backend.diagnose()
        codes = {d.code for d in diagnostics}
        assert "no_bem_solver" in codes
        assert "imported_dataset" in codes
        assert "development_fixture" in codes
        assert "capytaine_unavailable" in codes or "capytaine_available_not_used" in codes
        for d in diagnostics:
            assert d.severity in ("info", "warning", "error")
        # no diagnostic claims a solve capability
        for d in diagnostics:
            assert "cannot solve new requests" in d.message or "no BEM solver" in d.message \
                or "capytaine" in d.message or "development fixture" in d.message \
                or "stored validated dataset" in d.message

    def test_backend_rejects_invalid_dataset(self):
        import xarray as xr

        with pytest.raises(Exception):
            ImportedHydroDatasetBackend(xr.Dataset())
