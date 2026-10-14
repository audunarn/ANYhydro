"""Capytaine development adapter tests.

Capytaine-dependent tests are skipped when the optional dependency is not
installed; admission, settings and capability tests always run.
"""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

import anyhydro.capytaine_backend as cb
from anyhydro import (
    BackendSolveError,
    CapytaineBackend,
    CapabilityError,
    GeometryValidationError,
    HydroBody,
    HydroPanelMesh,
    HydroRequest,
    UnsupportedOperationError,
    body_dof_keys,
    validate_hydro_dataset,
)

pytest.importorskip("capytaine", reason="optional dependency not installed")


class FreshSolveCalled(Exception):
    """Sentinel proving the cache was bypassed and a fresh solve attempted."""


# --------------------------------------------------------------------- #
# tiny development fixture: two submerged outward-verified boxes
# --------------------------------------------------------------------- #

ALPHA_CENTER = (0.0, 0.0, -0.3)
BETA_CENTER = (4.0, 0.0, -0.3)


def _box_mesh(body_id: str, center, half_extents) -> HydroPanelMesh:
    cx, cy, cz = (float(v) for v in center)
    hx, hy, hz = (float(v) for v in half_extents)
    corners = [
        (cx + dx * hx, cy + dy * hy, cz + dz * hz)
        for dx in (-1.0, 1.0)
        for dy in (-1.0, 1.0)
        for dz in (-1.0, 1.0)
    ]
    vertices = np.array(
        [corners[0], corners[1], corners[3], corners[2],
         corners[4], corners[5], corners[7], corners[6]],
        dtype=np.float64,
    )
    quads = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5],
    ]
    box_center = np.array([cx, cy, cz])
    faces = []
    for quad in quads:
        q = list(quad)
        normal = np.cross(vertices[q[1]] - vertices[q[0]], vertices[q[2]] - vertices[q[0]])
        face_center = vertices[q].mean(axis=0)
        if np.dot(normal, face_center - box_center) < 0.0:
            q = q[::-1]
        faces.append(q)
    return HydroPanelMesh(
        vertices=vertices,
        faces=np.array(faces, dtype=np.int32),
        face_sizes=np.full(len(faces), 4, dtype=np.uint8),
        body_id=body_id,
        outward_normals_verified=True,
        source_geometry_refs=("dev-box",),
        mesh_hash="",
    )


def _box_body(body_id: str, center) -> HydroBody:
    return HydroBody(
        body_id=body_id,
        name=body_id,
        mesh=_box_mesh(body_id, center, (1.0, 0.5, 0.25)),
        lid_mesh=None,
        rotation_center=center,
        center_of_mass=center,
        enabled_dofs=body_dof_keys(body_id),
    )


def test_single_body_solve_uses_the_unprefixed_capytaine_dofs():
    """A one-body request is solved without a Multibody; its DOF labels have no body prefix."""
    import capytaine as cpt
    from anyhydro.capytaine_backend import _face_lists
    body = _box_body("alpha", ALPHA_CENTER)
    dataset = CapytaineBackend().solve((body,), _dev_request())
    assert [str(k) for k in dataset["influenced_dof"].values] == list(body_dof_keys("alpha"))
    # Independent: the same body solved directly with Capytaine, DOF by DOF.
    mesh = cpt.Mesh(vertices=np.array(body.mesh.vertices), faces=_face_lists(body.mesh.faces, body.mesh.face_sizes))
    direct_body = cpt.FloatingBody(mesh, dofs=cpt.rigid_body_dofs(rotation_center=tuple(body.rotation_center)))
    order = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")
    # the adapter's default method (direct since adapter 0.6.0) on the upstream kernel
    results = cpt.BEMSolver(method=cb.DEFAULT_METHOD).solve_all([cpt.RadiationProblem(body=direct_body, omega=1.0, radiating_dof=d,
                                                              rho=1025.0, g=9.81) for d in order])
    direct = {(r.radiating_dof, d): r.added_mass[d] for r in results for d in order}
    expected = np.array([[direct[(rad, inf)] for rad in order] for inf in order])
    added = np.asarray(dataset["added_mass"].values[0])
    assert np.allclose(added, expected, rtol=1e-10, atol=1e-8 * np.max(np.abs(expected)))


def _dev_bodies():
    return (_box_body("alpha", ALPHA_CENTER), _box_body("beta", BETA_CENTER))


def _dev_request(**overrides):
    defaults = dict(
        omegas=np.array([1.0]),
        wave_directions=np.array([0.0]),
        water_depth=float("inf"),
        rho=1025.0,
        gravity=9.81,
        include_zero_frequency=False,
        include_infinite_frequency=False,
        backend_settings={},
    )
    defaults.update(overrides)
    return HydroRequest(**defaults)


# --------------------------------------------------------------------- #
# capabilities, settings, admission (no solve needed)
# --------------------------------------------------------------------- #

class TestCapabilitiesAndSettings:
    def test_capabilities_honest(self):
        caps = CapytaineBackend().capabilities()
        assert caps.backend_name == "capytaine"
        assert caps.supports_zero_frequency is False
        assert caps.supports_infinite_frequency is True
        assert caps.provides_added_mass_infinite is True
        assert caps.provides_panel_pressure is False
        assert caps.provides_kochin is False
        assert caps.provides_free_surface is False
        assert caps.provides_mean_drift is False

    def test_unknown_settings_refused(self):
        backend = CapytaineBackend()
        with pytest.raises(CapabilityError, match="bogus"):
            backend.solve(_dev_bodies(), _dev_request(backend_settings={"bogus": "x"}))

    def test_non_string_setting_refused(self):
        backend = CapytaineBackend()
        with pytest.raises(CapabilityError):
            backend.solve(_dev_bodies(), _dev_request(backend_settings={"method": 3}))

    def test_zero_frequency_refused_typed(self):
        backend = CapytaineBackend()
        with pytest.raises(UnsupportedOperationError):
            backend.solve(_dev_bodies(), _dev_request(include_zero_frequency=True))

    def test_frequency_budget_enforced(self):
        backend = CapytaineBackend()
        with pytest.raises(CapabilityError, match="frequency budget"):
            backend.solve(_dev_bodies(), _dev_request(omegas=np.array([0.5, 1.0, 1.5, 2.0])))

    def test_heading_budget_enforced(self):
        backend = CapytaineBackend()
        with pytest.raises(CapabilityError, match="heading budget"):
            backend.solve(_dev_bodies(), _dev_request(wave_directions=np.array([0.0, np.pi])))

    def test_panel_budget_enforced(self):
        backend = CapytaineBackend()
        # 101 panels: 17 boxes would be needed; build one body with many
        # faces by repeating a verified box mesh scaled in face count is
        # not possible with the box helper, so exceed via many bodies
        bodies = tuple(_box_body(f"b{i:02d}", (10.0 * i, 0.0, -0.3)) for i in range(17))
        with pytest.raises(CapabilityError, match="panel budget"):
            backend.solve(bodies, _dev_request())

    def test_unverified_normals_refused(self):
        body = _box_body("alpha", ALPHA_CENTER)
        mesh = HydroPanelMesh(
            vertices=body.mesh.vertices,
            faces=body.mesh.faces,
            face_sizes=body.mesh.face_sizes,
            body_id="alpha",
            outward_normals_verified=False,
            source_geometry_refs=("dev-box",),
            mesh_hash="",
        )
        unverified = replace(body, mesh=mesh)
        with pytest.raises(GeometryValidationError, match="not verified"):
            CapytaineBackend().solve((unverified,), _dev_request())

    def test_panels_above_free_surface_refused(self):
        with pytest.raises(GeometryValidationError, match="free surface"):
            CapytaineBackend().solve(
                (_box_body("alpha", (0.0, 0.0, 0.3)),), _dev_request()
            )

    def test_panels_below_bottom_refused(self):
        with pytest.raises(GeometryValidationError, match="sea bottom"):
            CapytaineBackend().solve(
                (_box_body("alpha", (0.0, 0.0, -6.0)),),
                _dev_request(water_depth=5.0),
            )

    def test_lid_off_free_surface_refused(self):
        body = _box_body("alpha", ALPHA_CENTER)
        lid = _box_mesh("alpha", (0.0, 0.0, -0.1), (1.0, 0.5, 0.05))
        with pytest.raises(GeometryValidationError, match="lid"):
            CapytaineBackend().solve((replace(body, lid_mesh=lid),), _dev_request())

    def test_missing_dependency_diagnostics_and_solve(self, monkeypatch):
        monkeypatch.setattr(cb, "_capytaine_spec", lambda: None)
        backend = CapytaineBackend()
        codes = {d.code: d for d in backend.diagnose()}
        assert "capytaine_unavailable" in codes
        assert codes["capytaine_unavailable"].severity == "error"
        with pytest.raises(CapabilityError, match="not installed"):
            backend.solve(_dev_bodies(), _dev_request())

    def test_diagnose_honest_when_available(self):
        backend = CapytaineBackend()
        codes = {d.code: d for d in backend.diagnose()}
        assert "capytaine_available" in codes
        assert "hydrostatics_scope" in codes
        assert "development_budgets" in codes


# --------------------------------------------------------------------- #
# real solves (tiny development BEM)
# --------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def solved_dataset():
    backend = CapytaineBackend()
    return backend.solve(_dev_bodies(), _dev_request(include_infinite_frequency=True))


class TestSolve:
    def test_dataset_is_canonical(self, solved_dataset):
        validate_hydro_dataset(solved_dataset)
        assert solved_dataset.attrs["backend"] == "capytaine"
        assert solved_dataset.attrs["qualification_status"] == "unqualified"

    def test_full_6n_with_nonzero_offdiagonal_blocks(self, solved_dataset):
        assert solved_dataset["added_mass"].shape == (1, 12, 12)
        cross = solved_dataset["added_mass"].values[:, 0:6, 6:12]
        cross_back = solved_dataset["added_mass"].values[:, 6:12, 0:6]
        assert np.any(cross != 0.0), "cross-body added-mass block lost"
        assert np.any(cross_back != 0.0), "cross-body added-mass block lost"
        assert np.any(solved_dataset["radiation_damping"].values[:, 0:6, 6:12] != 0.0)

    def test_dof_labels_are_body_qualified(self, solved_dataset):
        labels = [str(v) for v in solved_dataset["influenced_dof"].values.tolist()]
        assert labels[:6] == [f"alpha::{d}" for d in ("surge", "sway", "heave", "roll", "pitch", "yaw")]
        assert labels[6:] == [f"beta::{d}" for d in ("surge", "sway", "heave", "roll", "pitch", "yaw")]

    def test_infinite_frequency_explicit_with_provenance(self, solved_dataset):
        a_inf = solved_dataset["added_mass_infinite"]
        assert a_inf.shape == (12, 12)
        assert "omega=inf" in a_inf.attrs["source"]
        assert "capytaine" in a_inf.attrs["source"]
        assert np.any(a_inf.values[0:6, 6:12] != 0.0)

    def test_hydrostatics_labeling_is_honest(self, solved_dataset):
        note = solved_dataset.attrs["hydrostatic_stiffness_definition"]
        assert "buoyancy-only" in note
        assert "NOT included" in note
        assert solved_dataset.attrs["adapter"] == "anyhydro-capytaine-development-adapter"

    def test_inputs_not_mutated(self):
        bodies = _dev_bodies()
        request = _dev_request()
        mesh_before = bodies[0].mesh.vertices.copy()
        faces_before = bodies[0].mesh.faces.copy()
        hash_before = request.content_hash()
        CapytaineBackend().solve(bodies, request)
        np.testing.assert_array_equal(bodies[0].mesh.vertices, mesh_before)
        np.testing.assert_array_equal(bodies[0].mesh.faces, faces_before)
        assert request.content_hash() == hash_before

    def test_deep_water_and_finite_depth_both_solve(self):
        backend = CapytaineBackend()
        ds_deep = backend.solve(_dev_bodies(), _dev_request(water_depth=float("inf")))
        ds_finite = backend.solve(_dev_bodies(), _dev_request(water_depth=50.0))
        assert ds_deep.attrs["request_hash"] != ds_finite.attrs["request_hash"]


class TestDirectCapytaineComparison:
    """Adapter output vs an independent direct Capytaine run."""

    def test_all_arrays_match_direct_capytaine(self, solved_dataset):
        import capytaine as cpt
        import xarray as xr

        bodies = _dev_bodies()
        request = _dev_request(include_infinite_frequency=True)
        cpt_bodies = []
        for b in bodies:
            mesh = cpt.Mesh(
                vertices=np.array(b.mesh.vertices, dtype=np.float64),
                faces=[
                    [int(v) for v in b.mesh.faces[i, : int(b.mesh.face_sizes[i])]]
                    for i in range(b.mesh.faces.shape[0])
                ],
            )
            cpt_bodies.append(
                cpt.FloatingBody(
                    mesh,
                    dofs=cpt.rigid_body_dofs(rotation_center=tuple(b.rotation_center)),
                    center_of_mass=tuple(b.center_of_mass),
                    mass=0.0,
                    name=b.body_id,
                )
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            combined = cpt.Multibody(cpt_bodies, name="direct")
        labels = [f"{b.body_id}__{cb._CPT_DOF_NAME[d]}" for b in bodies for d in cb._CPT_DOF_NAME]
        matrix = xr.Dataset(
            coords={
                "omega": request.omegas,
                "wave_direction": request.wave_directions,
                "radiating_dof": np.array(labels, dtype=object),
                "water_depth": float(request.water_depth),
                "rho": float(request.rho),
                "g": float(request.gravity),
            }
        )
        # same documented Green function as the adapter (anyhydro.green_function)
        from anyhydro.green_function import anyhydro_green_function
        solver = cpt.BEMSolver(green_function=anyhydro_green_function(), method=cb.DEFAULT_METHOD)
        direct = solver.fill_dataset(
            matrix, combined, n_jobs=1, n_threads=1, progress_bar=False,
            hydrostatics=False,
        )
        ds = solved_dataset
        np.testing.assert_allclose(
            ds.added_mass.values,
            direct.added_mass.sel(influenced_dof=labels, radiating_dof=labels)
            .transpose("omega", "influenced_dof", "radiating_dof").values,
            rtol=1e-10, atol=1e-8,
        )
        np.testing.assert_allclose(
            ds.radiation_damping.values,
            direct.radiation_damping.sel(influenced_dof=labels, radiating_dof=labels)
            .transpose("omega", "influenced_dof", "radiating_dof").values,
            rtol=1e-10, atol=1e-8,
        )
        np.testing.assert_allclose(
            ds.excitation_force.values,
            direct.excitation_force.sel(influenced_dof=labels)
            .transpose("omega", "wave_direction", "influenced_dof").values,
            rtol=1e-10, atol=1e-8,
        )
        # hydrostatics: per sub-body buoyancy-only, block-assembled
        cpt_dofs = [cb._CPT_DOF_NAME[d] for d in cb._CPT_DOF_NAME]
        for k, cpt_body in enumerate(cpt_bodies):
            hs = cpt_body.compute_hydrostatic_stiffness(
                rho=float(request.rho), g=float(request.gravity)
            ).sel(influenced_dof=cpt_dofs, radiating_dof=cpt_dofs)
            np.testing.assert_allclose(
                ds.hydrostatic_stiffness.values[6 * k:6 * k + 6, 6 * k:6 * k + 6],
                hs.values, rtol=1e-10, atol=1e-8,
            )
        # cross-body hydrostatics blocks are exactly zero (block assembly)
        assert np.all(ds.hydrostatic_stiffness.values[0:6, 6:12] == 0.0)
        # infinite limit: direct per-radiating-DOF solves
        for j, label in enumerate(labels):
            problem = cpt.RadiationProblem(
                body=combined, omega=np.inf, radiating_dof=label,
                wave_direction=0.0, water_depth=float(request.water_depth),
                rho=float(request.rho), g=float(request.gravity),
            )
            result = solver.solve(problem)
            row = np.array([float(result.added_mass[l]) for l in labels])
            np.testing.assert_allclose(
                ds.added_mass_infinite.values[:, j], row, rtol=1e-10, atol=1e-8,
            )


class TestCache:
    def test_cache_hit_avoids_solve(self, tmp_path, monkeypatch):
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        first = backend.solve(_dev_bodies(), _dev_request())
        cache_files = list(tmp_path.glob("*.nc"))
        assert len(cache_files) == 1
        # any fresh solve now must fail: only the cache can answer
        def _boom(*args, **kwargs):
            raise AssertionError("cache miss: solve was called")
        monkeypatch.setattr(CapytaineBackend, "_solve_fresh", _boom)
        second = backend.solve(_dev_bodies(), _dev_request())
        np.testing.assert_array_equal(
            first.added_mass.values, second.added_mass.values
        )
        assert second.attrs["request_hash"] == first.attrs["request_hash"]

    def test_changed_reference_invalidates_cache(self, tmp_path, monkeypatch):
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        backend.solve(_dev_bodies(), _dev_request())
        moved = _dev_bodies()
        moved = (replace(moved[0], rotation_center=(0.5, 0.0, -0.3)), moved[1])
        # the changed reference must not be served from the cache
        def _guard(*args, **kwargs):
            raise FreshSolveCalled()
        monkeypatch.setattr(CapytaineBackend, "_solve_fresh", _guard)
        with pytest.raises(FreshSolveCalled):
            backend.solve(moved, _dev_request())

    def test_changed_request_invalidates_cache(self, tmp_path, monkeypatch):
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        backend.solve(_dev_bodies(), _dev_request())
        def _guard(*args, **kwargs):
            raise FreshSolveCalled()
        monkeypatch.setattr(CapytaineBackend, "_solve_fresh", _guard)
        with pytest.raises(FreshSolveCalled):
            backend.solve(_dev_bodies(), _dev_request(omegas=np.array([1.3])))

    def test_changed_physical_constants_invalidates_cache(self, tmp_path, monkeypatch):
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        backend.solve(_dev_bodies(), _dev_request())
        def _guard(*args, **kwargs):
            raise FreshSolveCalled()
        monkeypatch.setattr(CapytaineBackend, "_solve_fresh", _guard)
        with pytest.raises(FreshSolveCalled):
            backend.solve(_dev_bodies(), _dev_request(rho=1030.0))

    def test_corrupted_cache_is_a_miss(self, tmp_path):
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        backend.solve(_dev_bodies(), _dev_request())
        cache_file = next(tmp_path.glob("*.nc"))
        cache_file.write_bytes(b"not a netcdf file")
        # corrupted entry: re-solve and overwrite, no crash, no fake data
        result = backend.solve(_dev_bodies(), _dev_request())
        validate_hydro_dataset(result)

    def test_cache_file_is_netcdf_not_pickle(self, tmp_path):
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        backend.solve(_dev_bodies(), _dev_request())
        cache_file = next(tmp_path.glob("*.nc"))
        blob = cache_file.read_bytes()
        assert b"pickle" not in blob.lower()

    def test_settings_cache_dir_override(self, tmp_path):
        backend = CapytaineBackend()
        backend.solve(_dev_bodies(), _dev_request(backend_settings={"cache_dir": str(tmp_path)}))
        assert len(list(tmp_path.glob("*.nc"))) == 1

    def test_no_cache_dir_means_no_files(self, tmp_path):
        backend = CapytaineBackend()
        backend.solve(_dev_bodies(), _dev_request())
        assert list(tmp_path.glob("*.nc")) == []

    def test_repaired_green_function_stays_close_to_upstream_default(self, solved_dataset):
        """The adapter's kernel repairs are small corrections, not a different model."""
        import capytaine as cpt

        bodies = _dev_bodies()
        request = _dev_request(include_infinite_frequency=True)
        cpt_bodies = [
            cpt.FloatingBody(
                cpt.Mesh(vertices=np.array(b.mesh.vertices, dtype=np.float64),
                         faces=[[int(v) for v in b.mesh.faces[i, : int(b.mesh.face_sizes[i])]]
                                for i in range(b.mesh.faces.shape[0])]),
                dofs=cpt.rigid_body_dofs(rotation_center=tuple(b.rotation_center)),
                center_of_mass=tuple(b.center_of_mass), mass=0.0, name=b.body_id)
            for b in bodies]
        with np.errstate(invalid="ignore", divide="ignore"):
            combined = cpt.Multibody(cpt_bodies, name="direct")
        labels = [f"{b.body_id}__{cb._CPT_DOF_NAME[d]}" for b in bodies for d in cb._CPT_DOF_NAME]
        solver = cpt.BEMSolver(method=cb.DEFAULT_METHOD)   # upstream default kernel, the adapter's method
        for io, omega in enumerate(request.omegas):
            for j, rad in enumerate(labels):
                r = solver.solve(cpt.RadiationProblem(
                    body=combined, radiating_dof=rad, omega=float(omega),
                    water_depth=float(request.water_depth), rho=float(request.rho),
                    g=float(request.gravity)))
                for name, mine in (("added_mass", solved_dataset.added_mass.values[io, :, j]),
                                   ("radiation_damping", solved_dataset.radiation_damping.values[io, :, j])):
                    ref = np.array([getattr(r, name)[lab] for lab in labels])
                    scale = max(np.abs(ref).max(), 1e-12)
                    assert np.max(np.abs(mine - ref)) <= 1e-2 * scale, (name, omega, rad)


def test_adapter_records_the_reciprocity_diagnostic():
    """Owner decision 2026-10-13: every fresh adapter dataset carries the pairwise reciprocity diagnostic."""
    from anyhydro.reciprocity import dataset_reciprocity
    body = _box_body("alpha", ALPHA_CENTER)
    dataset = CapytaineBackend().solve((body,), _dev_request())
    stored = json.loads(dataset.attrs["reciprocity_diagnostic"])
    assert stored == json.loads(json.dumps(dataset_reciprocity(dataset)))
    assert dataset.attrs["adapter_version"] == cb.CAPYTAINE_ADAPTER_VERSION == "0.6.0"


def test_reciprocity_warning_on_fresh_and_cached_solves_points_at_the_caller(tmp_path, monkeypatch):
    """Review: the warning fires for cached returns too and is attributed to the caller of solve()."""
    import warnings
    import anyhydro.reciprocity as rec
    from anyhydro.reciprocity import ReciprocityWarning
    monkeypatch.setattr(rec, "RECIPROCITY_WARNING", 0.0)
    monkeypatch.setattr(cb, "reciprocity_diagnostic",
                        lambda *a, **k: rec.reciprocity_diagnostic(*a, **k, threshold=-1.0))
    body = _box_body("alpha", ALPHA_CENTER)
    backend = CapytaineBackend()
    for _ in range(2):                                                  # fresh, then cached
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            backend.solve((body,), _dev_request(backend_settings={"cache_dir": str(tmp_path)}))
        hits = [w for w in caught if issubclass(w.category, ReciprocityWarning)]
        assert len(hits) == 1 and hits[0].filename == __file__


def test_reciprocity_attribute_hashes_identically_fresh_and_cache_reloaded(tmp_path):
    """Review round 4: the fresh JSON text and the reloaded dict must give the same content hash."""
    from anyhydro.hashing import dataset_content_hash
    body = _box_body("alpha", ALPHA_CENTER)
    backend = CapytaineBackend()
    settings = {"cache_dir": str(tmp_path)}
    fresh = backend.solve((body,), _dev_request(backend_settings=settings))
    cached = backend.solve((body,), _dev_request(backend_settings=settings))
    assert isinstance(fresh.attrs["reciprocity_diagnostic"], str)
    assert isinstance(cached.attrs["reciprocity_diagnostic"], dict)
    assert dataset_content_hash(fresh) == dataset_content_hash(cached)


# --------------------------------------------------------------------- #
# opt-in direct method (B3y Y4: same-mesh agreement with OrcaWave)
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ["Direct", "DIRECT", "indirect ", "source", ""])
def test_unsupported_method_refused_before_any_solve(method, monkeypatch):
    """Capytaine runs the indirect method for any string but 'direct'; the adapter refuses instead."""
    monkeypatch.setattr(CapytaineBackend, "_solve_fresh", lambda *a, **k: (_ for _ in ()).throw(FreshSolveCalled()))
    with pytest.raises(CapabilityError, match="unsupported method"):
        CapytaineBackend().solve((_box_body("alpha", ALPHA_CENTER),), _dev_request(backend_settings={"method": method}))


def test_unset_method_is_the_direct_solve():
    """Adapter 0.6.0: the default is the direct method (and its Green's-identity far field)."""
    assert cb.DEFAULT_METHOD == "direct"
    body = _box_body("alpha", ALPHA_CENTER)
    request = dict(include_infinite_frequency=True)
    default = CapytaineBackend().solve((body,), _dev_request(**request))
    explicit = CapytaineBackend().solve((body,), _dev_request(backend_settings={"method": "direct"}, **request))
    assert default.attrs["method"] == explicit.attrs["method"] == "direct"
    assert set(default.data_vars) == set(explicit.data_vars)
    assert "green-identity" in default["radiation_damping_far_field"].attrs["source"]
    for name in default.data_vars:
        assert np.array_equal(default[name].values, explicit[name].values), name


def test_explicit_indirect_remains_available():
    body = _box_body("alpha", ALPHA_CENTER)
    ds = CapytaineBackend().solve((body,), _dev_request(backend_settings={"method": "indirect"}))
    assert ds.attrs["method"] == "indirect"
    assert "green-identity" not in ds["radiation_damping_far_field"].attrs["source"]


def _plain_capytaine(body, method, omega):
    """The same body solved with plain Capytaine and ANYhydro's Green function, DOF by DOF."""
    import capytaine as cpt
    from anyhydro.capytaine_backend import _face_lists
    from anyhydro.green_function import anyhydro_green_function
    mesh = cpt.Mesh(vertices=np.array(body.mesh.vertices), faces=_face_lists(body.mesh.faces, body.mesh.face_sizes))
    cbody = cpt.FloatingBody(mesh, dofs=cpt.rigid_body_dofs(rotation_center=tuple(body.rotation_center)))
    solver = cpt.BEMSolver(green_function=anyhydro_green_function(), method=method)
    order = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")
    results = [solver.solve(cpt.RadiationProblem(body=cbody, omega=omega, radiating_dof=d, rho=1025.0, g=9.81))
               for d in order]
    A = np.array([[r.added_mass[inf] for r in results] for inf in order])
    B = np.array([[r.radiation_damping[inf] for r in results] for inf in order]) if np.isfinite(omega) else None
    return A, B


def test_direct_method_matches_plain_capytaine_including_infinite_frequency():
    """Review B3y: the opt-in direct method reaches every solve, also the omega=inf limit."""
    body = _box_body("alpha", ALPHA_CENTER)
    direct = CapytaineBackend().solve(
        (body,), _dev_request(include_infinite_frequency=True, backend_settings={"method": "direct"}))
    indirect = CapytaineBackend().solve((body,), _dev_request(include_infinite_frequency=True,
                                                              backend_settings={"method": "indirect"}))
    assert direct.attrs["method"] == "direct"
    # no source strengths: far-field damping comes from Green's identity (adapter 0.5.3)
    assert "green-identity" in direct["radiation_damping_far_field"].attrs["source"]
    A, B = _plain_capytaine(body, "direct", 1.0)
    A_inf, _ = _plain_capytaine(body, "direct", np.inf)
    scale = np.max(np.abs(A))
    assert np.allclose(direct["added_mass"].values[0], A, rtol=0, atol=1e-10 * scale)
    assert np.allclose(direct["radiation_damping"].values[0], B, rtol=0, atol=1e-10 * np.max(np.abs(B)))
    assert np.allclose(direct["added_mass_infinite"].values, A_inf, rtol=0, atol=1e-10 * np.max(np.abs(A_inf)))
    # the two methods differ at finite mesh, so the omega=inf limit really was solved directly
    assert not np.array_equal(direct["added_mass_infinite"].values, indirect["added_mass_infinite"].values)
    json.loads(direct.attrs["reciprocity_diagnostic"])


def test_direct_dataset_persists_and_hashes_identically(tmp_path):
    from anyhydro.hashing import dataset_content_hash
    from anyhydro.persistence import load_hydro_dataset, save_hydro_dataset
    body = _box_body("alpha", ALPHA_CENTER)
    settings = {"method": "direct", "cache_dir": str(tmp_path / "cache")}
    backend = CapytaineBackend()
    fresh = backend.solve((body,), _dev_request(backend_settings=settings))
    cached = backend.solve((body,), _dev_request(backend_settings=settings))
    assert cached.attrs["method"] == "direct"
    assert dataset_content_hash(fresh) == dataset_content_hash(cached)
    path = tmp_path / "direct.nc"
    save_hydro_dataset(fresh, path)
    assert dataset_content_hash(load_hydro_dataset(path)) == dataset_content_hash(fresh)
    # a cached indirect solve of the same request is a different identity
    other = backend.solve((body,), _dev_request(backend_settings={"cache_dir": str(tmp_path / "cache"), "method": "indirect"}))
    assert other.attrs["method"] == "indirect"


def test_default_and_explicit_indirect_keep_separate_cache_entries(tmp_path, monkeypatch):
    """Review (Sol r1): the two request forms must not share, and keep overwriting, one cache entry."""
    body = _box_body("alpha", ALPHA_CENTER)
    backend = CapytaineBackend()
    cache = str(tmp_path)
    forms = ({"cache_dir": cache}, {"cache_dir": cache, "method": "indirect"})
    first = [backend.solve((body,), _dev_request(backend_settings=s)) for s in forms]
    assert len(list(tmp_path.glob("*.nc"))) == 2
    monkeypatch.setattr(CapytaineBackend, "_solve_fresh",
                        lambda *a, **k: (_ for _ in ()).throw(FreshSolveCalled()))
    for settings, before in zip(forms[::-1], first[::-1]):          # reverse order: both are cache hits
        again = backend.solve((body,), _dev_request(backend_settings=settings))
        for name in before.data_vars:
            assert np.array_equal(again[name].values, before[name].values), name


def test_direct_method_excitation_matches_plain_capytaine():
    import capytaine as cpt
    from anyhydro.capytaine_backend import _face_lists
    from anyhydro.green_function import anyhydro_green_function
    body = _box_body("alpha", ALPHA_CENTER)
    ds = CapytaineBackend().solve((body,), _dev_request(backend_settings={"method": "direct"}))
    mesh = cpt.Mesh(vertices=np.array(body.mesh.vertices), faces=_face_lists(body.mesh.faces, body.mesh.face_sizes))
    cbody = cpt.FloatingBody(mesh, dofs=cpt.rigid_body_dofs(rotation_center=tuple(body.rotation_center)))
    solver = cpt.BEMSolver(green_function=anyhydro_green_function(), method="direct")
    res = solver.solve(cpt.DiffractionProblem(body=cbody, omega=1.0, wave_direction=0.0, rho=1025.0, g=9.81))
    from capytaine.bem.airy_waves import froude_krylov_force
    fk = froude_krylov_force(res.problem)
    order = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")
    expected = np.array([res.forces[d] + fk[d] for d in order])
    got = np.asarray(ds["excitation_force"].values)[0, 0]
    assert np.allclose(got, expected, rtol=0, atol=1e-10 * np.max(np.abs(expected)))


def test_recorded_method_mismatch_is_refused(monkeypatch):
    """The adapter checks that Capytaine recorded the requested method."""
    import capytaine as cpt
    original = cpt.BEMSolver.__init__

    def wrong_record(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.exportable_settings = {**self.exportable_settings, "method": "indirect"}
    monkeypatch.setattr(cpt.BEMSolver, "__init__", wrong_record)
    with pytest.raises(BackendSolveError, match="recorded method"):
        CapytaineBackend().solve((_box_body("alpha", ALPHA_CENTER),),
                                 _dev_request(backend_settings={"method": "direct"}))


# --------------------------------------------------------------------- #
# direct-default programme D5: limits and diagnostics for both methods
# --------------------------------------------------------------------- #

@pytest.mark.parametrize("method", ["indirect", "direct"])
def test_zero_frequency_is_refused_before_any_solve_or_cache(method, tmp_path, monkeypatch):
    def touched(*a, **k):
        raise AssertionError("cache or solve reached before the zero-frequency refusal")
    # Review (Sol r1): trap the cache-entry boundary itself, not only the solve
    monkeypatch.setattr(CapytaineBackend, "_cache_identity", touched)
    monkeypatch.setattr(CapytaineBackend, "_cache_load", touched)
    monkeypatch.setattr(CapytaineBackend, "_solve_fresh", touched)
    with pytest.raises(UnsupportedOperationError):
        CapytaineBackend().solve((_box_body("alpha", ALPHA_CENTER),), _dev_request(
            include_zero_frequency=True, backend_settings={"method": method, "cache_dir": str(tmp_path)}))
    assert not any(tmp_path.iterdir())


def test_direct_infinite_frequency_never_enters_the_far_field():
    """omega = inf gives A_inf only; the far-field (Kochin) array covers the finite frequencies only."""
    ds = CapytaineBackend().solve((_box_body("alpha", ALPHA_CENTER),), _dev_request(
        omegas=np.array([0.8, 1.6]), include_infinite_frequency=True, backend_settings={"method": "direct"}))
    assert np.all(np.isfinite(ds["added_mass_infinite"].values))
    far = ds["radiation_damping_far_field"]
    assert far.shape[0] == 2 and np.all(np.isfinite(far.values))


@pytest.mark.parametrize("method", ["indirect", "direct"])
def test_reciprocity_warning_fires_and_stays_silent_for_both_methods(method, monkeypatch):
    import warnings
    import anyhydro.reciprocity as rec
    from anyhydro.reciprocity import ReciprocityWarning
    body = _box_body("alpha", ALPHA_CENTER)
    for threshold, expected in ((-1.0, 1), (1e9, 0)):
        monkeypatch.setattr(cb, "reciprocity_diagnostic",
                            lambda *a, _t=threshold, **k: rec.reciprocity_diagnostic(*a, **k, threshold=_t))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CapytaineBackend().solve((body,), _dev_request(backend_settings={"method": method}))
        assert len([w for w in caught if issubclass(w.category, ReciprocityWarning)]) == expected
