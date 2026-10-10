"""Regression: cache stamp must not leak into public dataset identity.

The first-slice preparation review found that ``CapytaineBackend.solve``
returned the cache-loaded dataset still carrying the private
``cache_identity`` attribute, while a fresh solve returned an unstamped
dataset. Identical physical requests therefore produced different public
``dataset_content_hash`` values and leaked cache bookkeeping into
application input identity.

The repair removes only the private cache stamp from the warm public
return, after ``_cache_load`` has validated it against the full solve
identity. The on-disk cache file keeps its stamp for validation.

All solves here run the public ``solve()`` production path with
``_solve_fresh`` mocked to assemble a synthetic canonical dataset: no
Capytaine import, no BEM solve, no scientific qualification.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace

import numpy as np
import pytest

import anyhydro.capytaine_backend as cb
from anyhydro import (
    CapytaineBackend,
    HydroBody,
    HydroPanelMesh,
    HydroRequest,
    body_dof_keys,
    validate_hydro_dataset,
)
from anyhydro.capytaine_backend import CapytaineExecutionLimits
from anyhydro.hashing import JSON_DATASET_ATTRS, dataset_content_hash, mesh_hash
from anyhydro.persistence import load_hydro_dataset, save_hydro_dataset
from anyhydro.schema import assemble_hydro_dataset


class FreshSolveForbidden(Exception):
    """Sentinel: a second fresh solve must never happen on a cache hit."""


# --------------------------------------------------------------------- #
# tiny development fixture: one submerged outward-verified box
# --------------------------------------------------------------------- #

CENTER = (0.0, 0.0, -0.3)


def _box_mesh(body_id: str) -> HydroPanelMesh:
    cx, cy, cz = CENTER
    hx, hy, hz = 1.0, 0.5, 0.25
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
    faces = np.array(faces, dtype=np.int32)
    face_sizes = np.full(len(faces), 4, dtype=np.uint8)
    return HydroPanelMesh(
        vertices=vertices,
        faces=faces,
        face_sizes=face_sizes,
        body_id=body_id,
        outward_normals_verified=True,
        source_geometry_refs=("dev-box",),
        mesh_hash=mesh_hash(body_id, vertices, faces, face_sizes),
    )


def _dev_body(body_id: str = "alpha") -> HydroBody:
    return HydroBody(
        body_id=body_id,
        name=body_id,
        mesh=_box_mesh(body_id),
        lid_mesh=None,
        rotation_center=CENTER,
        center_of_mass=CENTER,
        enabled_dofs=body_dof_keys(body_id),
    )


def _dev_bodies():
    return (_dev_body(),)


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
# mocked production-path solve: public solve() with _solve_fresh replaced
# --------------------------------------------------------------------- #

def _mocked_solve_fresh(self, bodies, request, settings):
    """Synthetic assembly exactly shaped like the real _solve_fresh return."""
    n_dof = 6 * len(bodies)
    n_omega = int(request.omegas.shape[0])
    n_dir = int(request.wave_directions.shape[0])
    added_mass = (np.arange(n_omega * n_dof * n_dof, dtype=np.float64) * 0.25).reshape(
        n_omega, n_dof, n_dof
    )
    damping = (np.arange(n_omega * n_dof * n_dof, dtype=np.float64) * 0.125).reshape(
        n_omega, n_dof, n_dof
    )
    excitation = (
        np.arange(n_omega * n_dir * n_dof, dtype=np.float64) * 0.5
        + 1j * np.arange(n_omega * n_dir * n_dof, dtype=np.float64) * 0.0625
    ).reshape(n_omega, n_dir, n_dof)
    stiffness = (np.arange(n_dof * n_dof, dtype=np.float64) * 0.75).reshape(n_dof, n_dof)
    return assemble_hydro_dataset(
        bodies,
        request,
        added_mass=added_mass,
        radiation_damping=damping,
        excitation_force=excitation,
        hydrostatic_stiffness=stiffness,
        rotation_center=np.array([b.rotation_center for b in bodies], dtype=np.float64),
        center_of_mass=np.array([b.center_of_mass for b in bodies], dtype=np.float64),
        center_of_buoyancy=np.array([b.center_of_mass for b in bodies], dtype=np.float64),
        backend="capytaine",
        backend_version=cb._capytaine_version(),
        backend_commit="unknown",
        qualification_status="unqualified",
        extra_attrs={
            "adapter": "anyhydro-capytaine-development-adapter",
            "adapter_version": cb.CAPYTAINE_ADAPTER_VERSION,
            "development_limits": self._limits.describe(),
            "execution_limits": self._limits.as_dict(),
        },
    )


def _forbidden_fresh(self, bodies, request, settings):
    raise FreshSolveForbidden("cache miss: a second fresh solve was attempted")


@pytest.fixture
def no_capytaine(monkeypatch):
    """Guard: any real Capytaine import inside these tests is a failure."""
    def _boom():
        raise AssertionError("real capytaine import attempted in a mocked test")
    monkeypatch.setattr(cb, "_require_capytaine", _boom)


def _install_mock(monkeypatch):
    monkeypatch.setattr(CapytaineBackend, "_solve_fresh", _mocked_solve_fresh)


def _install_forbidden(monkeypatch):
    monkeypatch.setattr(CapytaineBackend, "_solve_fresh", _forbidden_fresh)


def _as_obj(value):
    """Attr value as a plain object: persistence may decode JSON strings."""
    return json.loads(value) if isinstance(value, str) else value


def _canonical_attr(value):
    if isinstance(value, (str, dict)):
        try:
            return json.dumps(_as_obj(value), sort_keys=True)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _assert_same_public_attrs(fresh, warm):
    assert set(fresh.attrs) == set(warm.attrs)
    for key in fresh.attrs:
        if key == "created_utc":
            continue
        assert _canonical_attr(warm.attrs[key]) == _canonical_attr(fresh.attrs[key]), key


def _assert_same_arrays(fresh, warm):
    assert set(fresh.data_vars) == set(warm.data_vars)
    for name in fresh.data_vars:
        np.testing.assert_array_equal(fresh[name].values, warm[name].values, name)


# --------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------- #

class TestCacheStampNotPublicIdentity:
    def test_fresh_and_warm_share_public_identity(self, tmp_path, monkeypatch, no_capytaine):
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        _install_mock(monkeypatch)
        fresh = backend.solve(_dev_bodies(), _dev_request())
        assert "cache_identity" not in fresh.attrs
        validate_hydro_dataset(fresh)
        fresh_hash = dataset_content_hash(fresh)

        # a second fresh solve is now forbidden: only the cache may answer
        _install_forbidden(monkeypatch)
        warm = backend.solve(_dev_bodies(), _dev_request())
        assert "cache_identity" not in warm.attrs
        validate_hydro_dataset(warm)
        assert dataset_content_hash(warm) == fresh_hash
        _assert_same_public_attrs(fresh, warm)
        _assert_same_arrays(fresh, warm)
        # execution limits stay part of the public identity, not dropped
        assert "execution_limits" in warm.attrs
        assert _as_obj(warm.attrs["execution_limits"]) == backend._limits.as_dict()

    def test_warm_public_dataset_re_save_is_stable(self, tmp_path, monkeypatch, no_capytaine):
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        _install_mock(monkeypatch)
        fresh = backend.solve(_dev_bodies(), _dev_request())
        _install_forbidden(monkeypatch)
        warm = backend.solve(_dev_bodies(), _dev_request())
        target = tmp_path / "re-saved-public.nc"
        save_hydro_dataset(warm, target)
        reloaded = load_hydro_dataset(target)
        assert "cache_identity" not in reloaded.attrs
        assert dataset_content_hash(reloaded) == dataset_content_hash(fresh)
        validate_hydro_dataset(reloaded)

    def test_raw_cache_stamp_hashing_is_canonical(self, tmp_path, monkeypatch, no_capytaine):
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        _install_mock(monkeypatch)
        backend.solve(_dev_bodies(), _dev_request())
        raw = load_hydro_dataset(next(tmp_path.glob("*.nc")))
        assert "cache_identity" in raw.attrs
        # the decoded (dict) stamp and its canonical JSON-string form must
        # hash identically: deliberate raw cache-file hashing is canonical
        string_form = raw.copy()
        string_form.attrs["cache_identity"] = json.dumps(
            _as_obj(raw.attrs["cache_identity"]), sort_keys=True
        )
        assert dataset_content_hash(string_form) == dataset_content_hash(raw)

    def test_on_disk_cache_keeps_stamp_and_still_validates(self, tmp_path, monkeypatch, no_capytaine):
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        _install_mock(monkeypatch)
        request = _dev_request()
        backend.solve(_dev_bodies(), request)
        cache_file = next(tmp_path.glob("*.nc"))
        raw = load_hydro_dataset(cache_file)
        # the private stamp stays on disk for validation
        assert "cache_identity" in raw.attrs
        identity = _as_obj(raw.attrs["cache_identity"])
        assert identity["execution_limits"] == backend._limits.as_dict()
        assert identity["request"]["rho"] == float(request.rho)
        assert raw.attrs["request_hash"] == request.content_hash()
        # and the stamped entry still validates through the public path
        _install_forbidden(monkeypatch)
        warm = backend.solve(_dev_bodies(), request)
        assert "cache_identity" not in warm.attrs


class TestCacheMismatchRefusal:
    def test_changed_request_cannot_be_served(self, tmp_path, monkeypatch, no_capytaine):
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        _install_mock(monkeypatch)
        backend.solve(_dev_bodies(), _dev_request())
        _install_forbidden(monkeypatch)
        with pytest.raises(FreshSolveForbidden):
            backend.solve(_dev_bodies(), _dev_request(omegas=np.array([1.3])))

    def test_changed_physical_constants_cannot_be_served(self, tmp_path, monkeypatch, no_capytaine):
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        _install_mock(monkeypatch)
        backend.solve(_dev_bodies(), _dev_request())
        _install_forbidden(monkeypatch)
        with pytest.raises(FreshSolveForbidden):
            backend.solve(_dev_bodies(), _dev_request(rho=1030.0))

    def test_changed_body_reference_cannot_be_served(self, tmp_path, monkeypatch, no_capytaine):
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        _install_mock(monkeypatch)
        backend.solve(_dev_bodies(), _dev_request())
        moved = (replace(_dev_body(), rotation_center=(0.5, 0.0, -0.3)),)
        _install_forbidden(monkeypatch)
        with pytest.raises(FreshSolveForbidden):
            backend.solve(moved, _dev_request())

    def test_changed_execution_limits_cannot_be_served(self, tmp_path, monkeypatch, no_capytaine):
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        _install_mock(monkeypatch)
        backend.solve(_dev_bodies(), _dev_request())
        other = CapytaineBackend(
            cache_dir=str(tmp_path),
            execution_limits=CapytaineExecutionLimits(max_finite_omegas=2),
        )
        _install_forbidden(monkeypatch)
        with pytest.raises(FreshSolveForbidden):
            other.solve(_dev_bodies(), _dev_request())


def test_no_capytaine_imported_by_module_or_tests():
    # mocked production-path tests never import the optional backend
    assert "capytaine" not in sys.modules
