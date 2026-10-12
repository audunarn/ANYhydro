# SPDX-License-Identifier: MPL-2.0
"""ANYhydro production Green function (anyhydro.green_function)."""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO / 'src'), str(REPO / 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

from anyhydro import green_function as gfm  # noqa: E402

K_CASE = 0.2293577981651376


def test_base_import_does_not_import_capytaine():
    code = ('import sys; sys.path.insert(0, %r); import anyhydro.green_function; '
            'print("capytaine" in sys.modules)' % str(REPO / 'src'))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == 'False'


def test_propagating_imaginary_matches_john_series_imaginary_part():
    import check_finite_depth_john_oracle as oracle
    kern = oracle.FiniteDepthKernel(0.3, 10.0, n_modes=5)
    x = np.array([[1.0, 2.0, -0.5], [7.0, -3.0, -4.0]])
    y = np.array([[0.2, 0.1, -1.0], [-0.4, 0.3, -6.0]])
    S, _ = gfm.propagating_imaginary(x, y, np.ones(2), k=0.3, water_depth=10.0)
    gi, _ = kern.wave_imag(x, y)
    # the oracle returns the J0 coefficient c0 Z Z J0 of Gt = -4 pi G; Capytaine's
    # outgoing convention has Im Gt = -c0 Z Z J0 (sign confirmed against Capytaine below)
    np.testing.assert_allclose(S, gi / (4 * math.pi), rtol=1e-12)


@pytest.mark.parametrize('depth', [10.0, math.inf])
@pytest.mark.parametrize('which', ['receiver', 'source'])
def test_gradients_match_finite_differences(depth, which):
    x = np.array([[1.0, 2.0, -0.5]])
    y = np.array([[0.2, 0.1, -1.0], [3.0, -1.0, -0.2]])
    w = np.array([0.7, 1.3])
    S, G = gfm.propagating_imaginary(x, y, w, k=0.4, water_depth=depth, gradient=which)
    h = 1e-6
    for d in range(3):
        if which == 'receiver':
            xp, xm = x.copy(), x.copy()
            xp[0, d] += h
            xm[0, d] -= h
            fd = (gfm.propagating_imaginary(xp, y, w, k=0.4, water_depth=depth)[0]
                  - gfm.propagating_imaginary(xm, y, w, k=0.4, water_depth=depth)[0]) / (2 * h)
        else:
            fd = np.zeros_like(S)
            for j in range(len(y)):
                yp, ym = y.copy(), y.copy()
                yp[j, d] += h
                ym[j, d] -= h
                diff = (gfm.propagating_imaginary(x, yp, w, k=0.4, water_depth=depth)[0]
                        - gfm.propagating_imaginary(x, ym, w, k=0.4, water_depth=depth)[0])
                fd[:, j] = diff[:, j] / (2 * h)
        np.testing.assert_allclose(G[d], fd, rtol=1e-6, atol=1e-12)


def _meshes(cpt):
    a = cpt.mesh_parallelepiped(size=(1.0, 1.0, 1.0), center=(-2.0, 0, 0),
                                resolution=(4, 4, 4)).immersed_part()
    b = cpt.mesh_parallelepiped(size=(1.0, 1.0, 1.0), center=(2.0, 0, 0),
                                resolution=(4, 4, 4)).immersed_part()
    return a, b


@pytest.mark.parametrize('depth', [150.0, 12.0, math.inf])
@pytest.mark.parametrize('adjoint,early', [(True, True), (False, True), (True, False)])
def test_imaginary_part_equals_untabulated_capytaine(depth, adjoint, early):
    cpt = pytest.importorskip('capytaine')
    a, b = _meshes(cpt)
    k = K_CASE if math.isinf(depth) else 0.3
    plain = gfm.anyhydro_green_function(tabulation_nr=0, tabulation_nz=0)
    base = type(plain).__mro__[1]   # validated-Prony Delhommeau without the imaginary repair
    S0, K0 = base.evaluate(plain, a, b, water_depth=depth, wavenumber=k,
                           adjoint_double_layer=adjoint, early_dot_product=early)
    S1, K1 = gfm.anyhydro_green_function().evaluate(a, b, water_depth=depth, wavenumber=k,
                                                     adjoint_double_layer=adjoint,
                                                     early_dot_product=early)
    assert S1.shape == S0.shape and K1.shape == K0.shape
    # Capytaine integrates this term numerically (1001 nodes, ~1e-8 relative)
    np.testing.assert_allclose(S1.imag, S0.imag, rtol=0, atol=1e-6 * np.abs(S0.imag).max())
    np.testing.assert_allclose(K1.imag, K0.imag, rtol=0, atol=1e-6 * np.abs(K0.imag).max())


def test_points_array_receivers_and_passthrough_limits():
    cpt = pytest.importorskip('capytaine')
    a, b = _meshes(cpt)
    gf = gfm.anyhydro_green_function()
    pts = np.array([[0.0, 3.0, -0.3], [5.0, 1.0, -2.0]])
    S, K = gf.evaluate(pts, b, water_depth=150.0, wavenumber=0.3, early_dot_product=False)
    Si, _ = gfm.propagating_imaginary(pts, *b.quadrature_points, k=0.3, water_depth=150.0)
    np.testing.assert_allclose(S.imag, Si, rtol=1e-12)
    base = type(gf).__mro__[1]
    S1, _ = gf.evaluate(a, b, water_depth=150.0, wavenumber=np.inf)
    S0, _ = base.evaluate(gf, a, b, water_depth=150.0, wavenumber=np.inf)
    np.testing.assert_array_equal(S1, S0)             # no propagating mode: unchanged
    from anyhydro.errors import CapabilityError
    with pytest.raises(CapabilityError):               # finite-depth zero frequency refused
        gf.evaluate(a, b, water_depth=150.0, wavenumber=0.0)


def test_two_box_near_null_damping_is_positive_and_far_consistent():
    """W5 03c configuration: the tabulated default gives a negative eigenvalue."""
    cpt = pytest.importorskip('capytaine')
    from capytaine.post_pro.kochin import compute_kochin
    from capytaine.green_functions.delhommeau import Delhommeau
    omega, h, rho = 1.5, 150.0, 1025.0
    bodies = []
    for name, xc in (('a', -2.0), ('b', 2.0)):
        m = cpt.mesh_parallelepiped(size=(1.0, 1.0, 1.0), center=(xc, 0, 0),
                                    resolution=(8, 8, 8)).immersed_part()
        bodies.append(cpt.FloatingBody(m, dofs=cpt.rigid_body_dofs(rotation_center=(xc, 0, 0)),
                                       center_of_mass=(xc, 0, 0), name=name))
    both = cpt.Multibody(bodies)
    labels = ['a__Surge', 'a__Heave', 'b__Surge', 'b__Heave']
    theta = np.linspace(0, 2 * np.pi, 721)[:-1]

    def damping(gf):
        solver = cpt.BEMSolver(green_function=gf)
        B = np.zeros((4, 4))
        H = []
        k = None
        for j, rad in enumerate(labels):
            r = solver.solve(cpt.RadiationProblem(body=both, radiating_dof=rad, omega=omega,
                                                  water_depth=h, rho=rho), keep_details=True)
            for i, inf in enumerate(labels):
                B[i, j] = r.radiation_damping[inf]
            H.append(compute_kochin(r, theta))
            k = r.wavenumber
        H = np.array(H)
        c = 4 * math.pi * rho * k / (omega * gfm.dispersion_factor(k, h))
        return 0.5 * (B + B.T), c * (H.conj() @ H.T).real * (2 * np.pi / len(theta))

    B_tab, _ = damping(Delhommeau())
    B_new, B_far = damping(gfm.anyhydro_green_function())
    assert np.linalg.eigvalsh(B_tab)[0] < 0          # the defect being repaired
    lam, vec = np.linalg.eigh(B_new)
    assert lam[0] > 0
    for i in range(4):          # near/far on every mode (coarse 8x8 mesh: ~2 % far bias)
        u = vec[:, i]
        assert abs(u @ B_new @ u - u @ B_far @ u) <= 0.10 * abs(u @ B_far @ u)


def test_identity_records_both_repairs():
    pytest.importorskip('capytaine')
    gf = gfm.anyhydro_green_function()
    assert gf.exportable_settings['imaginary_part'] == gfm.IMAGINARY_PART_FORMAT
    assert gf.exportable_settings['finite_depth_prony_decomposition']


@pytest.mark.parametrize('which', ['receiver', 'source'])
def test_blockwise_normal_projection_equals_full_gradient(which, monkeypatch):
    monkeypatch.setattr(gfm, '_ROW_BLOCK', 3)        # force several row blocks
    rng = np.random.default_rng(4)
    x = rng.normal(size=(7, 3)) * [3, 3, 0.5] - [0, 0, 1]
    y = rng.normal(size=(5, 2, 3)) * [3, 3, 0.5] - [0, 0, 1]
    w = rng.uniform(0.1, 1.0, size=(5, 2))
    nrm = rng.normal(size=(7 if which == 'receiver' else 5, 3))
    S, G = gfm.propagating_imaginary(x, y, w, k=0.3, water_depth=40.0, gradient=which)
    S2, Kn = gfm.propagating_imaginary(x, y, w, k=0.3, water_depth=40.0, gradient=which, normals=nrm)
    expected = (np.einsum('dnm,nd->nm', G, nrm) if which == 'receiver'
                else np.einsum('dnm,md->nm', G, nrm))
    np.testing.assert_allclose(S2, S, rtol=1e-14)
    np.testing.assert_allclose(Kn, expected, rtol=1e-12, atol=1e-15)
    with pytest.raises(ValueError):
        gfm.propagating_imaginary(x, y, w, k=0.3, water_depth=40.0, gradient=which,
                                  normals=np.zeros((2, 3)))


def test_free_surface_self_panel_imaginary_part_is_continuous():
    """Upstream flips the sign of Im S for z=0 self panels; the exact term is smooth."""
    cpt = pytest.importorskip('capytaine')
    gf = gfm.anyhydro_green_function()
    vals = []
    for zc in (0.0, -1e-3, -1e-2):
        m = cpt.mesh_rectangle(size=(1.0, 1.0), center=(0, 0, zc), resolution=(1, 1), normal=(0, 0, 1))
        S, _ = gf.evaluate(m, m, water_depth=np.inf, wavenumber=0.4)
        vals.append(float(S.imag.ravel()[0]))
    assert vals[0] == pytest.approx(-0.2, rel=1e-12)                 # -k/2 * area * J0(0)
    assert vals[1] == pytest.approx(-0.2 * math.exp(2 * 0.4 * -1e-3), rel=1e-12)
    assert abs(vals[0] - vals[1]) < 1e-3 and abs(vals[1] - vals[2]) < 1e-2


def test_waterline_lid_damping_matches_far_field_as_well_as_upstream():
    cpt = pytest.importorskip('capytaine')
    from capytaine.post_pro.kochin import compute_kochin
    from capytaine.green_functions.delhommeau import Delhommeau
    hull = cpt.mesh_parallelepiped(size=(2.0, 2.0, 2.0), center=(0, 0, 0), resolution=(4, 4, 4)).immersed_part()
    lid = cpt.mesh_rectangle(size=(2.0, 2.0), center=(0, 0, 0.0), resolution=(2, 2), normal=(0, 0, 1))
    body = cpt.FloatingBody(hull, lid_mesh=lid, dofs=cpt.rigid_body_dofs(rotation_center=(0, 0, 0)),
                            center_of_mass=(0, 0, 0))
    theta = np.linspace(0, 2 * np.pi, 721)[:-1]
    gaps = {}
    for name, gf in (('upstream', Delhommeau()), ('anyhydro', gfm.anyhydro_green_function())):
        r = cpt.BEMSolver(green_function=gf).solve(
            cpt.RadiationProblem(body=body, radiating_dof='Heave', omega=1.5, water_depth=20.0, rho=1025.0),
            keep_details=True)
        H = compute_kochin(r, theta)
        k = r.wavenumber
        far = (4 * math.pi * 1025.0 * k / (1.5 * gfm.dispersion_factor(k, 20.0))
               * np.sum(np.abs(H) ** 2) * (2 * np.pi / len(theta)))
        gaps[name] = abs(r.radiation_damping['Heave'] / far - 1)
    assert gaps['anyhydro'] <= gaps['upstream'] + 1e-3
