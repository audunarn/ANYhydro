"""First-slice deterministic wetted-hull/lid fixture tests (no BEM run).

Analytic geometry, determinism, case constants and backend admission
(including the downward-lid orientation refusal) are checked with mocked
solves only.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from anyhydro import (
    FIRST_SLICE_BODY_IDS,
    FIRST_SLICE_DRY_MASS,
    FIRST_SLICE_GRAVITY,
    FIRST_SLICE_HEAVE_BUOYANCY_STIFFNESS,
    FIRST_SLICE_OMEGAS,
    FIRST_SLICE_REFERENCE_WEIGHT,
    FIRST_SLICE_RHO,
    FIRST_SLICE_ROTATION_CENTERS,
    FIRST_SLICE_WATER_DEPTH,
    CapytaineBackend,
    CapytaineExecutionLimits,
    GeometryValidationError,
    HydroPanelMesh,
    assemble_hydro_dataset,
    make_first_slice_bodies,
    make_first_slice_request,
    verify_first_slice_fixture,
)

#: Golden default-level (n=2) identities captured from the historical
#: fixture (evidence source-study-worker-01): the parametric refinement
#: must reproduce them bit-for-bit at the default level.
GOLDEN_N2_HULL_A = "0de8c0d0036a86b4e93c9402f8e209b1d04b31f3664368c948f603ab97015eba"
GOLDEN_N2_LID_A = "f276bb4bea54b329fb3eaffa7cd33b5e8fccd5657d27544bbd7b86236811f128"
GOLDEN_N2_HULL_B = "d7531d53835b1dade2d56ef839bc150675d2247e594c4c12f4c7f68cdd995887"
GOLDEN_N2_LID_B = "f8ba90008ea844f1e24a10d7bfccda0fa7d59527a4fa6ef029c128e8ed099ca3"
GOLDEN_N2_REQUEST = "61d2a6c200c5dd85d5cb81dd4249fbe5b53e00fb033912c7cd0e79760f1c9b67"


class TestFixtureGeometry:
    def test_body_ids_references_and_coms(self):
        a, b = make_first_slice_bodies()
        assert (a.body_id, b.body_id) == FIRST_SLICE_BODY_IDS
        assert a.rotation_center == FIRST_SLICE_ROTATION_CENTERS[0]
        assert b.rotation_center == FIRST_SLICE_ROTATION_CENTERS[1]
        assert a.center_of_mass == (-2.0, 0.0, -0.25)
        assert b.center_of_mass == (2.0, 0.0, -0.25)
        assert len(a.enabled_dofs) == 6 and len(b.enabled_dofs) == 6

    def test_panel_counts_96_total(self):
        a, b = make_first_slice_bodies()
        for body in (a, b):
            assert body.mesh.faces.shape[0] == 40
            assert body.lid_mesh.faces.shape[0] == 8
        total = sum(
            body.mesh.faces.shape[0] + body.lid_mesh.faces.shape[0]
            for body in (a, b)
        )
        assert total == 96

    def test_shared_vertices(self):
        a, _ = make_first_slice_bodies()
        assert a.mesh.vertices.shape[0] == 25
        assert a.lid_mesh.vertices.shape[0] == 9

    def test_no_dry_top_or_above_water_hull(self):
        a, b = make_first_slice_bodies()
        for body in (a, b):
            assert body.mesh.vertices[:, 2].max() == 0.0
            assert body.mesh.vertices[:, 2].min() == -0.5
            centers = body.mesh.vertices[body.mesh.faces].mean(axis=1)
            assert centers[:, 2].max() < 0.0
            assert np.all(body.lid_mesh.vertices[:, 2] == 0.0)

    def test_hull_normals_outward_lid_normals_down(self):
        a, b = make_first_slice_bodies()
        for body in (a, b):
            interior = np.array([body.rotation_center[0], 0.0, -0.25])
            v, f = body.mesh.vertices, body.mesh.faces
            normals = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
            centers = v[f].mean(axis=1)
            assert np.all(np.sum((centers - interior) * normals, axis=1) > 0.0)
            lv, lf = body.lid_mesh.vertices, body.lid_mesh.faces
            lid_normals = np.cross(lv[lf[:, 1]] - lv[lf[:, 0]], lv[lf[:, 2]] - lv[lf[:, 0]])
            assert np.all(lid_normals[:, 2] < 0.0)

    def test_lid_flag_is_honestly_not_outward(self):
        a, _ = make_first_slice_bodies()
        assert a.mesh.outward_normals_verified is True
        # lid normals are deliberately downward (pinned capytaine
        # convention), so the lid does not claim outward normals
        assert a.lid_mesh.outward_normals_verified is False

    def test_deterministic_construction(self):
        first = make_first_slice_bodies()
        second = make_first_slice_bodies()
        for x, y in zip(first, second):
            assert x.mesh.mesh_hash == y.mesh.mesh_hash
            assert x.lid_mesh.mesh_hash == y.lid_mesh.mesh_hash
            np.testing.assert_array_equal(x.mesh.vertices, y.mesh.vertices)
            np.testing.assert_array_equal(x.mesh.faces, y.mesh.faces)
            np.testing.assert_array_equal(x.lid_mesh.faces, y.lid_mesh.faces)

    def test_records_are_read_only(self):
        a, _ = make_first_slice_bodies()
        assert a.mesh.vertices.flags.writeable is False
        assert a.mesh.faces.flags.writeable is False


class TestAnalyticVerification:
    def test_verify_report_values(self):
        bodies = make_first_slice_bodies()
        report = verify_first_slice_fixture(bodies)
        assert report["totals"]["panels"] == 96
        for body_id, cx in (("first-slice-a", -2.0), ("first-slice-b", 2.0)):
            entry = report["bodies"][body_id]
            assert entry["hull_panels"] == 40
            assert entry["lid_panels"] == 8
            assert entry["hull_area_m2"] == pytest.approx(3.0, abs=1e-12)
            assert entry["lid_area_m2"] == pytest.approx(1.0, abs=1e-12)
            assert entry["submerged_volume_m3"] == pytest.approx(0.5, abs=1e-12)
            assert entry["center_of_buoyancy_m"] == pytest.approx(
                [cx, 0.0, -0.25], abs=1e-12
            )
            assert entry["waterplane_area_m2"] == pytest.approx(1.0, abs=1e-12)

    def test_verify_rejects_wrong_body_ids(self):
        a, b = make_first_slice_bodies()
        other_mesh = replace(a.mesh, body_id="other", mesh_hash="")
        other_lid = replace(a.lid_mesh, body_id="other", mesh_hash="")
        other = replace(
            a,
            body_id="other",
            name="other",
            mesh=other_mesh,
            lid_mesh=other_lid,
            enabled_dofs=tuple(f"other::{d}" for d in ("surge", "sway", "heave", "roll", "pitch", "yaw")),
        )
        with pytest.raises(GeometryValidationError, match="body ids"):
            verify_first_slice_fixture((other, b))

    def test_case_constants(self):
        assert FIRST_SLICE_RHO == 1025.0
        assert FIRST_SLICE_GRAVITY == 9.81
        assert FIRST_SLICE_WATER_DEPTH == 150.0
        assert FIRST_SLICE_DRY_MASS == pytest.approx(FIRST_SLICE_RHO * 0.5)
        assert FIRST_SLICE_REFERENCE_WEIGHT == pytest.approx(5027.625)
        assert FIRST_SLICE_HEAVE_BUOYANCY_STIFFNESS == pytest.approx(
            FIRST_SLICE_RHO * FIRST_SLICE_GRAVITY * 1.0
        )
        assert FIRST_SLICE_HEAVE_BUOYANCY_STIFFNESS == pytest.approx(10055.25)
        assert tuple(FIRST_SLICE_OMEGAS) == (2.0, 4.0, 6.0)

    def test_request(self):
        request = make_first_slice_request()
        np.testing.assert_array_equal(request.omegas, [2.0, 4.0, 6.0])
        np.testing.assert_array_equal(request.wave_directions, [0.0])
        assert request.water_depth == 150.0
        assert request.rho == 1025.0
        assert request.gravity == 9.81
        assert request.include_infinite_frequency is True
        assert request.include_zero_frequency is False
        assert request.content_hash() == make_first_slice_request().content_hash()


class TestBackendAdmission:
    def test_fixture_admitted_under_default_limits(self, monkeypatch):
        bodies = make_first_slice_bodies()
        request = make_first_slice_request()
        monkeypatch.setattr(
            CapytaineBackend,
            "_solve_fresh",
            lambda self, b, r, s: _stub_dataset(b, r),
        )
        dataset = CapytaineBackend(
            execution_limits=CapytaineExecutionLimits()
        ).solve(bodies, request)
        assert dataset.attrs["backend"] == "stub"

    def test_upward_lid_refused(self, monkeypatch):
        bodies = make_first_slice_bodies()
        request = make_first_slice_request()
        monkeypatch.setattr(
            CapytaineBackend,
            "_solve_fresh",
            lambda self, b, r, s: _stub_dataset(b, r),
        )
        a, b = bodies
        bad_lid = HydroPanelMesh(
            vertices=a.lid_mesh.vertices,
            faces=a.lid_mesh.faces[:, ::-1],  # reversed winding: normals UP
            face_sizes=a.lid_mesh.face_sizes,
            body_id=a.body_id,
            outward_normals_verified=False,
            source_geometry_refs=a.lid_mesh.source_geometry_refs,
            mesh_hash="",
        )
        with pytest.raises(GeometryValidationError, match="point DOWN"):
            CapytaineBackend().solve((replace(a, lid_mesh=bad_lid), b), request)


def _stub_dataset(bodies, request):
    n_dof = 6 * len(bodies)
    n_omega = int(request.omegas.shape[0])
    n_dir = int(request.wave_directions.shape[0])
    return assemble_hydro_dataset(
        bodies,
        request,
        added_mass=np.zeros((n_omega, n_dof, n_dof)),
        radiation_damping=np.zeros((n_omega, n_dof, n_dof)),
        excitation_force=np.zeros((n_omega, n_dir, n_dof), dtype=np.complex128),
        hydrostatic_stiffness=np.zeros((n_dof, n_dof)),
        rotation_center=np.array([x.rotation_center for x in bodies]),
        center_of_mass=np.array([x.center_of_mass for x in bodies]),
        center_of_buoyancy=np.array([x.center_of_mass for x in bodies]),
        backend="stub",
        backend_version="0",
        qualification_status="development_fixture",
    )


class TestParametricRefinement:
    """Focused parametric-subdivision checks (no BEM run)."""

    def test_default_golden_hashes_bit_identical(self):
        a, b = make_first_slice_bodies()
        assert a.mesh.mesh_hash == GOLDEN_N2_HULL_A
        assert a.lid_mesh.mesh_hash == GOLDEN_N2_LID_A
        assert b.mesh.mesh_hash == GOLDEN_N2_HULL_B
        assert b.lid_mesh.mesh_hash == GOLDEN_N2_LID_B
        assert make_first_slice_request().content_hash() == GOLDEN_N2_REQUEST
        # explicit default keyword reproduces the same identity
        a2, _ = make_first_slice_bodies(subdivisions=2)
        assert a2.mesh.mesh_hash == GOLDEN_N2_HULL_A

    def test_subdivisions_refused_before_allocation(self):
        for bad in (True, False, 2.0, float("nan"), float("inf"), 0, -1, 17,
                    "2", None):
            with pytest.raises(GeometryValidationError):
                make_first_slice_bodies(subdivisions=bad)
            with pytest.raises(GeometryValidationError):
                verify_first_slice_fixture(make_first_slice_bodies(),
                                            subdivisions=bad)

    def test_refined_counts_and_shared_topology(self):
        for n in (1, 3, 4, 5, 6):
            a, b = make_first_slice_bodies(subdivisions=n)
            for body in (a, b):
                assert body.mesh.faces.shape[0] == 10 * n * n
                assert body.lid_mesh.faces.shape[0] == 2 * n * n
                assert body.mesh.vertices.shape[0] == 5 * n * n + 2 * n + 1
                assert body.lid_mesh.vertices.shape[0] == (n + 1) ** 2
                # exact rational sharing: no duplicate coordinates exist, so
                # shared vertices were merged by exact keys, not by
                # rounding/snapping coordinates to invent topology
                assert np.unique(body.mesh.vertices, axis=0).shape[0] \
                    == body.mesh.vertices.shape[0]
            total = sum(x.mesh.faces.shape[0] + x.lid_mesh.faces.shape[0]
                        for x in (a, b))
            assert total == 24 * n * n

    def test_refined_analytic_verification(self):
        for n in (1, 3, 4, 6):
            bodies = make_first_slice_bodies(subdivisions=n)
            report = verify_first_slice_fixture(bodies, subdivisions=n)
            assert report["totals"]["panels"] == 24 * n * n
            assert report["subdivisions"] == n
            for body_id, cx in (("first-slice-a", -2.0),
                                ("first-slice-b", 2.0)):
                entry = report["bodies"][body_id]
                assert entry["hull_area_m2"] == pytest.approx(3.0, abs=1e-12)
                assert entry["lid_area_m2"] == pytest.approx(1.0, abs=1e-12)
                assert entry["submerged_volume_m3"] == pytest.approx(
                    0.5, abs=1e-12)
                assert entry["center_of_buoyancy_m"] == pytest.approx(
                    [cx, 0.0, -0.25], abs=1e-12)
                assert entry["waterplane_area_m2"] == pytest.approx(
                    1.0, abs=1e-12)

    def test_verify_rejects_level_mismatch(self):
        bodies = make_first_slice_bodies(subdivisions=4)
        with pytest.raises(GeometryValidationError, match="subdivisions=2"):
            verify_first_slice_fixture(bodies)

    def test_refined_deterministic_and_wound(self):
        for n in (3, 6):
            first = make_first_slice_bodies(subdivisions=n)
            second = make_first_slice_bodies(subdivisions=n)
            for x, y in zip(first, second):
                assert x.mesh.mesh_hash == y.mesh.mesh_hash
                np.testing.assert_array_equal(x.mesh.vertices, y.mesh.vertices)
                np.testing.assert_array_equal(x.mesh.faces, y.mesh.faces)
            for body in first:
                interior = np.array([body.rotation_center[0], 0.0, -0.25])
                v, f = body.mesh.vertices, body.mesh.faces
                normals = np.cross(v[f[:, 1]] - v[f[:, 0]],
                                   v[f[:, 2]] - v[f[:, 0]])
                centers = v[f].mean(axis=1)
                assert np.all(np.sum((centers - interior) * normals, axis=1)
                              > 0.0)
                lv, lf = body.lid_mesh.vertices, body.lid_mesh.faces
                lid_normals = np.cross(lv[lf[:, 1]] - lv[lf[:, 0]],
                                        lv[lf[:, 2]] - lv[lf[:, 0]])
                assert np.all(lid_normals[:, 2] < 0.0)

    def test_refined_mutation_refused(self):
        a, _ = make_first_slice_bodies(subdivisions=4)
        bad_lid = HydroPanelMesh(
            vertices=a.lid_mesh.vertices,
            faces=a.lid_mesh.faces[:, ::-1],  # reversed winding: normals UP
            face_sizes=a.lid_mesh.face_sizes,
            body_id=a.body_id,
            outward_normals_verified=False,
            source_geometry_refs=a.lid_mesh.source_geometry_refs,
            mesh_hash="",
        )
        bodies = (replace(a, lid_mesh=bad_lid),
                  make_first_slice_bodies(subdivisions=4)[1])
        with pytest.raises(GeometryValidationError, match="point DOWN"):
            verify_first_slice_fixture(bodies, subdivisions=4)
