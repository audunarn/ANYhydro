# SPDX-License-Identifier: MPL-2.0
"""Far-field damping for the direct BEM method: Green's-identity Kochin functions (D1)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("capytaine", reason="optional dependency not installed")

import capytaine as cpt                                        # noqa: E402
from capytaine.post_pro.kochin import compute_kochin          # noqa: E402

from anyhydro import CapytaineBackend                          # noqa: E402
from anyhydro.green_function import (_depth_mode, anyhydro_green_function, far_field_damping,  # noqa: E402
                                     green_identity_kochin)

from test_capytaine_backend import ALPHA_CENTER, BETA_CENTER, _box_body, _dev_request  # noqa: E402

THETA = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
DOFS = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")


def _hull_kochin(result, theta, water_depth):
    body = result.problem.body
    hull = np.asarray(body.hull_mask)
    mesh = body.mesh_including_lid
    return green_identity_kochin(mesh.faces_centers[hull], mesh.faces_normals[hull], mesh.faces_areas[hull],
                                 np.asarray(result.problem.boundary_condition)[hull],
                                 np.asarray(result.potential)[hull], theta,
                                 k=float(result.wavenumber), water_depth=water_depth)


def _open_box(res, lid=True):
    hull = cpt.mesh_parallelepiped(size=(6.0, 3.0, 2.0), center=(0.0, 0.0, -1.0),
                                   resolution=(2 * res, res, res), missing_sides={"top"})
    return cpt.FloatingBody(hull, dofs=cpt.rigid_body_dofs(rotation_center=(0.0, 0.0, 0.0)),
                            lid_mesh=hull.generate_lid(z=0.0) if lid else None)


@pytest.mark.parametrize("depth", [np.inf, 20.0])
def test_green_identity_reproduces_a_manufactured_outgoing_field(depth):
    """Sources inside a closed submerged box: their field and Kochin function are known exactly.

    Catches sign, phase, normalisation and finite-depth mode errors that a damping Gram
    matrix (|H|^2) cannot see.
    """
    gf = anyhydro_green_function()
    k = 0.6
    src = cpt.mesh_sphere(radius=0.3, center=(0.4, -0.2, -2.1), resolution=(6, 6))
    sigma = np.random.default_rng(1).normal(size=src.nb_faces) + 1j * np.random.default_rng(2).normal(size=src.nb_faces)
    Z, _ = _depth_mode(src.faces_centers[:, 2], k, depth)
    phase = np.exp(-1j * k * (np.outer(np.cos(THETA), src.faces_centers[:, 0])
                              + np.outer(np.sin(THETA), src.faces_centers[:, 1])))
    expected = (phase * Z) @ (sigma * src.faces_areas) / (4.0 * np.pi)
    errors = []
    for res in (8, 16):
        box = cpt.mesh_parallelepiped(size=(4.0, 2.0, 1.5), center=(0.0, 0.0, -2.0), resolution=(res, res, res))
        S, K = gf.evaluate(box.faces_centers, src, free_surface=0.0, water_depth=depth, wavenumber=k,
                           adjoint_double_layer=True, early_dot_product=False, diagonal_term_in_double_layer=False)
        phi = S @ sigma
        dphi_dn = np.sum(np.einsum("dnm,m->nd", K, sigma) * box.faces_normals, axis=1)
        H = green_identity_kochin(box.faces_centers, box.faces_normals, box.faces_areas, dphi_dn, phi, THETA,
                                  k=k, water_depth=depth)
        errors.append(np.max(np.abs(H - expected)) / np.max(np.abs(expected)))
    # a sign, phase or normalisation error is O(1); the centroid-rule error is ~1e-3 and decreasing
    assert errors[1] < 2e-3 and errors[1] < errors[0], errors


def test_green_identity_matches_the_source_kochin_on_indirect_solutions():
    """Indirect solutions carry both sources and potential: the two Kochin routes must converge together.

    Errors are taken on the matrix scale (max over DOFs of |H|): weak modes (pitch of this short
    box) are judged against the energetic ones, not against their own tiny peak.
    """
    solver = cpt.BEMSolver(green_function=anyhydro_green_function(), method="indirect")
    worst = []
    for res in (4, 8):
        body = _open_box(res)
        diffs, peaks = [], []
        for d in DOFS:
            r = solver.solve(cpt.RadiationProblem(body=body, radiating_dof=d, omega=2 * np.pi / 6.0), keep_details=True)
            Hs = compute_kochin(r, THETA)
            diffs.append(np.max(np.abs(_hull_kochin(r, THETA, np.inf) - Hs)))
            peaks.append(np.max(np.abs(Hs)))
        worst.append(max(diffs) / max(peaks))
    assert worst[1] < 0.03 and worst[1] < worst[0], worst


def _direct_far_and_near(body, T, depth=np.inf):
    solver = cpt.BEMSolver(green_function=anyhydro_green_function(), method="direct")
    theta = np.linspace(0.0, 2.0 * np.pi, 720, endpoint=False)
    rs = [solver.solve(cpt.RadiationProblem(body=body, radiating_dof=d, omega=2 * np.pi / T, water_depth=depth),
                       keep_details=True) for d in DOFS]
    H = np.array([_hull_kochin(r, theta, depth) for r in rs])
    far = far_field_damping(H[None], np.array([2 * np.pi / T]), np.array([rs[0].wavenumber]), depth, 1000.0)[0]
    near = np.array([[r.radiation_damping[i] for r in rs] for i in DOFS])
    return far, 0.5 * (near + near.T)


def test_direct_far_field_damping_matches_near_field_damping():
    """Energy consistency of the direct solution; converges with the mesh."""
    errs = []
    for res in (6, 12):
        far, near = _direct_far_and_near(_open_box(res), 8.0)
        scale = np.max(np.diag(near))
        errs.append(np.max(np.abs(far - near)) / scale)
        energetic = np.diag(near) > 0.05 * scale
        rel = np.abs(np.diag(far) - np.diag(near))[energetic] / np.diag(near)[energetic]
        assert np.all(rel < 0.025), rel
        assert np.linalg.eigvalsh(far)[0] >= -1e-12 * scale
    assert errs[1] < 0.5 * errs[0], errs


def test_lid_does_not_change_the_direct_far_field_away_from_irregular_frequencies():
    far_lid, _ = _direct_far_and_near(_open_box(6, lid=True), 8.0)
    far_none, _ = _direct_far_and_near(_open_box(6, lid=False), 8.0)
    assert np.max(np.abs(far_lid - far_none)) < 0.005 * np.max(np.diag(far_none))


def _direct(bodies, **request):
    return CapytaineBackend().solve(bodies, _dev_request(backend_settings={"method": "direct"}, **request))


def test_adapter_direct_far_field_is_the_green_identity_route():
    body = _box_body("alpha", ALPHA_CENTER)
    ds = _direct((body,))
    source = ds["radiation_damping_far_field"].attrs["source"]
    assert "green-identity" in source and "Green's-identity" in source
    assert ds.attrs["method"] == "direct"
    far = ds["radiation_damping_far_field"].values[0]
    assert np.linalg.eigvalsh(far)[0] >= -1e-12 * np.max(np.abs(far))
    # indirect default keeps its own format and source text
    default = CapytaineBackend().solve((body,), _dev_request())
    assert "green-identity" not in default["radiation_damping_far_field"].attrs["source"]


def test_direct_far_field_is_translation_and_body_order_invariant():
    shift = np.array([5.0, 3.0, 0.0])
    alpha, beta = _box_body("alpha", ALPHA_CENTER), _box_body("beta", BETA_CENTER)
    base = _direct((alpha, beta))["radiation_damping_far_field"].values[0]

    def moved(b):
        from anyhydro.hashing import mesh_hash
        v = np.asarray(b.mesh.vertices) + shift
        mesh = replace(b.mesh, vertices=v, mesh_hash=mesh_hash(b.body_id, v, b.mesh.faces, b.mesh.face_sizes))
        return replace(b, mesh=mesh, rotation_center=tuple(np.asarray(b.rotation_center) + shift),
                       center_of_mass=tuple(np.asarray(b.center_of_mass) + shift))
    translated = _direct((moved(alpha), moved(beta)))["radiation_damping_far_field"].values[0]
    scale = np.max(np.abs(base))
    assert np.max(np.abs(translated - base)) < 1e-8 * scale
    swapped = _direct((beta, alpha))["radiation_damping_far_field"].values[0]
    perm = list(range(6, 12)) + list(range(6))
    assert np.max(np.abs(swapped - base[np.ix_(perm, perm)])) < 1e-8 * scale
