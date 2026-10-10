"""Record validation tests: mesh, body, request."""

from __future__ import annotations

import numpy as np
import pytest

from anyhydro import (
    HydroBody,
    HydroPanelMesh,
    HydroRequest,
    RecordValidationError,
    body_dof_keys,
)
from anyhydro.fixtures import make_fixture_bodies, make_fixture_request


def _box_mesh(body_id="body-a"):
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    faces = np.array(
        [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [3, 2, 6, 7], [0, 3, 7, 4],
         [1, 2, 6, 5]],
        dtype=np.int32,
    )
    sizes = np.full(6, 4, dtype=np.uint8)
    return HydroPanelMesh(
        vertices=vertices,
        faces=faces,
        face_sizes=sizes,
        body_id=body_id,
        outward_normals_verified=False,
        source_geometry_refs=("test",),
        mesh_hash="",
    )


class TestHydroPanelMesh:
    def test_hash_is_deterministic_and_input_mutation_independent(self):
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        v2 = vertices.copy()
        m1 = HydroPanelMesh(vertices, np.array([[0, 1, 1]], np.int32),
                            np.array([3], np.uint8), "b", False, (), "")
        m2 = HydroPanelMesh(v2, np.array([[0, 1, 1]], np.int32),
                            np.array([3], np.uint8), "b", False, (), "")
        assert m1.mesh_hash == m2.mesh_hash
        vertices[0, 0] = 5.0  # caller mutates its own array afterwards
        v2[0, 0] = 5.0
        assert m1.vertices[0, 0] == 0.0  # record holds a frozen copy

    def test_mesh_arrays_are_read_only(self):
        mesh = _box_mesh()
        with pytest.raises(ValueError):
            mesh.vertices[0, 0] = 1.0
        with pytest.raises(ValueError):
            mesh.faces[0, 0] = 7

    def test_rejects_nonfinite_vertices(self):
        vertices = np.zeros((4, 3))
        vertices[1, 1] = np.nan
        with pytest.raises(RecordValidationError):
            HydroPanelMesh(vertices, np.array([[0, 1, 2]], np.int32),
                           np.array([3], np.uint8), "b", False, (), "")

    def test_rejects_out_of_range_face_indices(self):
        mesh = _box_mesh()
        bad_faces = mesh.faces.copy()
        bad_faces[0, 0] = 99
        with pytest.raises(RecordValidationError):
            HydroPanelMesh(mesh.vertices, bad_faces, mesh.face_sizes, "b",
                           False, (), mesh.mesh_hash)

    def test_rejects_bad_padding(self):
        mesh = _box_mesh()
        sizes = np.array([3, 4, 4, 4, 4, 4], dtype=np.uint8)
        faces = mesh.faces.copy()
        faces[0, 3] = 0  # padding entry must be -1 for a triangular face
        with pytest.raises(RecordValidationError):
            HydroPanelMesh(mesh.vertices, faces, sizes, "b", False, (), "")

    def test_rejects_mismatched_provided_hash(self):
        mesh = _box_mesh()
        with pytest.raises(RecordValidationError):
            HydroPanelMesh(mesh.vertices, mesh.faces, mesh.face_sizes, "b",
                           False, (), "deadbeef")

    def test_rejects_body_id_with_separator(self):
        with pytest.raises(RecordValidationError):
            _box_mesh(body_id="bad::id")

    def test_outward_normals_flag_is_retained_not_asserted(self):
        mesh = _box_mesh()
        assert mesh.outward_normals_verified is False  # honestly not verified


class TestHydroBody:
    def _body(self, **overrides):
        mesh = _box_mesh("body-a")
        defaults = dict(
            body_id="body-a",
            name="a",
            mesh=mesh,
            lid_mesh=None,
            rotation_center=(0.0, 0.0, 0.0),
            center_of_mass=(0.0, 0.0, 0.5),
            enabled_dofs=body_dof_keys("body-a"),
        )
        defaults.update(overrides)
        return HydroBody(**defaults)

    def test_valid_body(self):
        body = self._body()
        assert body.enabled_dofs[0] == "body-a::surge"
        assert body.enabled_dofs[-1] == "body-a::yaw"

    def test_rejects_duplicate_enabled_dofs(self):
        dofs = body_dof_keys("body-a")
        with pytest.raises(RecordValidationError):
            self._body(enabled_dofs=dofs + (dofs[0],))

    def test_rejects_foreign_body_dof(self):
        with pytest.raises(RecordValidationError):
            self._body(enabled_dofs=("body-b::surge",))

    def test_rejects_unqualified_dof(self):
        with pytest.raises(RecordValidationError):
            self._body(enabled_dofs=("surge",))

    def test_rejects_incomplete_but_unique_dofs_is_allowed_subset(self):
        body = self._body(enabled_dofs=("body-a::surge", "body-a::heave"))
        assert len(body.enabled_dofs) == 2  # subsets are legal at record level

    def test_rejects_mesh_body_id_mismatch(self):
        other = _box_mesh("body-z")
        with pytest.raises(RecordValidationError):
            self._body(mesh=other)

    def test_rejects_lid_mesh_body_id_mismatch(self):
        with pytest.raises(RecordValidationError):
            self._body(lid_mesh=_box_mesh("body-z"))

    def test_rejects_nonfinite_center(self):
        with pytest.raises(RecordValidationError):
            self._body(center_of_mass=(0.0, np.inf, 0.0))


class TestHydroRequest:
    def _request(self, **overrides):
        defaults = dict(
            omegas=np.array([0.5, 1.0, 2.0]),
            wave_directions=np.array([0.0, np.pi / 2]),
            water_depth=100.0,
            rho=1025.0,
            gravity=9.81,
            include_zero_frequency=False,
            include_infinite_frequency=False,
            backend_settings={"key": "value"},
        )
        defaults.update(overrides)
        return HydroRequest(**defaults)

    def test_valid_request_hash_is_stable(self):
        r1 = self._request()
        r2 = self._request()
        assert r1.content_hash() == r2.content_hash()

    def test_rejects_zero_omega(self):
        with pytest.raises(RecordValidationError):
            self._request(omegas=np.array([0.0, 1.0]))

    def test_rejects_infinite_omega(self):
        with pytest.raises(RecordValidationError):
            self._request(omegas=np.array([1.0, np.inf]))

    def test_rejects_non_increasing_omegas(self):
        with pytest.raises(RecordValidationError):
            self._request(omegas=np.array([1.0, 1.0, 2.0]))
        with pytest.raises(RecordValidationError):
            self._request(omegas=np.array([2.0, 1.0]))

    def test_rejects_nonfinite_directions(self):
        with pytest.raises(RecordValidationError):
            self._request(wave_directions=np.array([0.0, np.nan]))

    def test_rejects_bad_scalars(self):
        with pytest.raises(RecordValidationError):
            self._request(rho=0.0)
        with pytest.raises(RecordValidationError):
            self._request(gravity=-9.81)
        with pytest.raises(RecordValidationError):
            self._request(water_depth=0.0)

    def test_rejects_non_json_backend_settings(self):
        with pytest.raises(RecordValidationError):
            self._request(backend_settings={"bad": object()})

    def test_request_is_isolated_from_caller_mutation(self):
        omegas = np.array([0.5, 1.0])
        settings = {"nested": [1, 2]}
        req = self._request(omegas=omegas, backend_settings=settings)
        omegas[0] = 99.0
        settings["nested"].append(3)
        assert req.omegas[0] == 0.5
        # nested sequences are deep-frozen: content equal, type immutable
        assert list(req.backend_settings["nested"]) == [1, 2]
        assert isinstance(req.backend_settings["nested"], tuple)

    def test_fixture_request_roundtrip(self):
        req = make_fixture_request()
        assert np.all(np.diff(req.omegas) > 0)
        assert req.content_hash() == make_fixture_request().content_hash()


class TestFixtureBodies:
    def test_fixture_bodies_have_verified_normals_and_full_dofs(self):
        bodies = make_fixture_bodies()
        assert len(bodies) == 2
        for body in bodies:
            assert body.mesh.outward_normals_verified is True
            assert len(body.enabled_dofs) == 6
