"""TS1: thin-sheet kernels on Capytaine's convention (docs/NATIVE_THIN_SHEET_PLAN.md)."""
import math

import numpy as np
import pytest

from anyhydro.thin_sheet_kernels import doublet_panel_velocity, hypersingular_rankine

# Dunavant degree-5, 7 points (barycentric a, b, c and weights summing to 1).
_A1, _B1 = 0.059715871789770, 0.470142064105115
_A2, _B2 = 0.797426985353087, 0.101286507323456
_BARY = np.array([[1 / 3, 1 / 3, 1 / 3], [_A1, _B1, _B1], [_B1, _A1, _B1], [_B1, _B1, _A1],
                  [_A2, _B2, _B2], [_B2, _A2, _B2], [_B2, _B2, _A2]])
_W = np.array([0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3)


def double_layer(x, polygon, n_sub=24):
    """Independent Phi(x) = int dG/dn_xi dS, G = -1/(4 pi r), over the centroid fan of a polygon."""
    c = polygon.mean(axis=0)
    total = 0.0
    for i in range(len(polygon)):
        a, b = polygon[i], polygon[(i + 1) % len(polygon)]
        grid = [(s, t) for s in range(n_sub) for t in range(n_sub - s)]
        tris = []
        for s, t in grid:
            tris.append([(s, t), (s + 1, t), (s, t + 1)])
            if s + t < n_sub - 1:
                tris.append([(s + 1, t), (s + 1, t + 1), (s, t + 1)])
        for tri in tris:
            p = np.array([c + (a - c) * u / n_sub + (b - c) * v / n_sub for u, v in tri])
            area_normal = 0.5 * np.cross(p[1] - p[0], p[2] - p[0])
            xi = _BARY @ p
            d = x - xi
            r = np.linalg.norm(d, axis=1)
            total += np.sum(_W * -(d @ area_normal) / (4 * np.pi * r ** 3))
    return total


QUAD = np.array([[0.0, 0.0, 0.0], [1.3, 0.1, 0.05], [1.1, 0.9, 0.1], [-0.1, 1.0, 0.0]])   # non-planar
TRI = np.array([[0.0, 0.0, -3.0], [1.0, 0.1, -3.0], [0.2, 0.9, -3.0]])


@pytest.mark.parametrize('polygon, faces, sizes, x', [
    (QUAD, [[0, 1, 2, 3]], [4], np.array([0.4, 0.7, 0.9])),
    (QUAD, [[0, 1, 2, 3]], [4], np.array([0.6, 0.4, -0.35])),
    (TRI, [[0, 1, 2]], [3], np.array([0.4, 0.3, -2.6])),
    (TRI, [[0, 1, 2]], [3], np.array([2.0, -1.0, -4.0])),
])
def test_ring_velocity_is_the_gradient_of_the_double_layer(polygon, faces, sizes, x):
    h = 1e-4
    grad = np.array([(double_layer(x + h * e, polygon) - double_layer(x - h * e, polygon)) / (2 * h)
                     for e in np.eye(3)])
    ring = doublet_panel_velocity(x, polygon, np.array(faces), np.array(sizes))[0, 0]
    assert np.allclose(ring, grad, rtol=0, atol=1e-7 * np.max(np.abs(grad)))


@pytest.mark.parametrize('n_sides', [3, 8, 64])
def test_regular_polygon_centre_velocity_is_exact(n_sides):
    """Unit jump, circumradius a, normal +z: u(centre) = +N tan(pi/N) / (2 pi a) z (Capytaine sign:
    Phi = -1/2 (1 - z/sqrt(z^2 + a^2)) just above a disc, so dPhi/dz = +1/(2a) at its centre)."""
    a = 0.7
    t = 2 * np.pi * np.arange(n_sides) / n_sides
    polygon = np.column_stack([a * np.cos(t), a * np.sin(t), np.zeros(n_sides)])
    faces = np.array([list(range(n_sides))])
    u = doublet_panel_velocity(np.zeros(3), polygon, faces, np.array([n_sides]))[0, 0]
    expected = n_sides * math.tan(math.pi / n_sides) / (2 * math.pi * a)
    assert u[:2] == pytest.approx([0.0, 0.0], abs=1e-14)
    assert u[2] == pytest.approx(expected, rel=1e-13)
    assert hypersingular_rankine(np.zeros((1, 3)), [[0, 0, 1]], polygon, faces, [n_sides])[0, 0] == \
        pytest.approx(expected, rel=1e-13)


def test_capytaine_convention_is_pinned():
    """Rankine G = -1/(4 pi r); our Phi equals Capytaine's direct double layer, and point receivers
    need diagonal_term_in_double_layer=False (otherwise the 1/2 jump is added to entry (i, i))."""
    cpt = pytest.importorskip('capytaine')
    h = 0.01
    tiny = cpt.Mesh(vertices=[[-h, -h, -5], [h, -h, -5], [h, h, -5], [-h, h, -5]], faces=[[0, 1, 2, 3]])
    gf = cpt.Delhommeau()
    point = np.array([[3.0, 0.0, -5.0]])
    S, _ = gf.evaluate_rankine_only(point, tiny, early_dot_product=False)
    assert S[0, 0] / (2 * h) ** 2 == pytest.approx(-1 / (4 * np.pi * 3.0), rel=1e-4)
    mesh = cpt.Mesh(vertices=TRI, faces=[[0, 1, 2, 2]])
    x = np.array([[0.4, 0.3, -2.2]])
    _, K = gf.evaluate_rankine_only(x, mesh, adjoint_double_layer=False, early_dot_product=False,
                                    diagonal_term_in_double_layer=False)
    # Capytaine returns the full gradient as (3, n_receivers, n_panels).
    capytaine_phi = np.asarray(K)[:, 0, 0] @ np.asarray(mesh.faces_normals)[0]
    assert capytaine_phi == pytest.approx(double_layer(x[0], TRI), rel=1e-7)
    _, K_diag = gf.evaluate_rankine_only(x, mesh, adjoint_double_layer=False, early_dot_product=False)
    assert np.asarray(K_diag)[:, 0, 0] @ np.asarray(mesh.faces_normals)[0] == pytest.approx(capytaine_phi + 0.5)


# ---------------------------------------------------------------- wave part (TS1)
def _sheet():
    xs, ys = np.linspace(0, 1.5, 4), np.linspace(0, 1.0, 3)
    vertices = np.array([[x, y, -2.0 + 0.2 * x - 0.1 * y] for y in ys for x in xs])
    faces = np.array([[j * 4 + i, j * 4 + i + 1, (j + 1) * 4 + i + 1, (j + 1) * 4 + i] for j in range(2) for i in range(3)])
    return vertices, faces, np.full(len(faces), 4)


def test_capytaine_wave_term_identity_and_gradient_are_pinned():
    """(dg/dZ - k g) r1 = 2k exactly; nablag is grad_x g (vs central differences at moderate steps)."""
    cpt = pytest.importorskip('capytaine')
    gf = cpt.Delhommeau(tabulation_nr=0, tabulation_nz=0)
    fc, tab = gf.fortran_core, gf.all_tabulation_parameters
    sing = gf.gf_singularities_fortran_enum['low_freq']
    wave = lambda x, xi, k: fc.green_wave.wave_part_infinite_depth(np.asarray(x, float), np.asarray(xi, float), k, *tab, sing)
    rng = np.random.default_rng(3)
    for k in (0.3, 0.8, 2.5):
        xi = np.array([0.3, -0.2, -1.5])
        x = np.array([rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(-4, -0.5)])
        g, grad = wave(x, xi, k)
        r1 = np.linalg.norm(x - xi * [1, 1, -1])
        assert (grad[2] - k * g) * r1 == pytest.approx(2 * k, rel=1e-12)
        h = 2e-3
        fd = np.array([(wave(x + h * e, xi, k)[0] - wave(x - h * e, xi, k)[0]) / (2 * h) for e in np.eye(3)])
        # limited by Capytaine's own wave-term integration accuracy (~1e-4 relative)
        assert np.max(np.abs(grad - fd)) <= 5e-4 * np.max(np.abs(fd))


def test_wave_double_layer_reproduces_capytaine_panel_matrices():
    """Own assembly from the point-pair wave term equals Capytaine's D(k) - D(0) (value level)."""
    cpt = pytest.importorskip('capytaine')
    from anyhydro import thin_sheet_capytaine as tsc
    V, F, S = _sheet()
    gf = cpt.Delhommeau(tabulation_nr=0, tabulation_nz=0)
    mesh = tsc._capytaine_mesh(V, F, S)
    normals = np.asarray(mesh.faces_normals)
    x = np.array([V[f].mean(0) for f in F]) + 0.3 * normals
    for k in (0.5, 2.0):
        own = tsc.wave_double_layer(gf, x, V, F, S, wavenumber=k)
        cap = (tsc._double_layer(gf, x, mesh, normals, wavenumber=k)
               - tsc._double_layer(gf, x, mesh, normals, wavenumber=0.0))
        assert np.max(np.abs(own - cap)) <= 1e-7 * np.max(np.abs(cap))


def test_analytic_wave_hypersingular_matches_differences_of_the_exact_gradient():
    """Point level: n_i^T Hess(g) (M n_j) from the identity vs central differences of grad g."""
    cpt = pytest.importorskip('capytaine')
    gf = cpt.Delhommeau(tabulation_nr=0, tabulation_nz=0)
    from anyhydro import thin_sheet_capytaine as tsc
    # a single tiny panel makes the panel integral a point evaluation times its area
    h = 1e-3
    V = np.array([[0.4 - h, -0.1 - h, -1.7], [0.4 + h, -0.1 - h, -1.7], [0.4 + h, -0.1 + h, -1.7], [0.4 - h, -0.1 + h, -1.7]])
    F, S = np.array([[0, 1, 2, 3]]), np.array([4])
    xi = V.mean(0)
    area = (2 * h) ** 2
    fc, tab = gf.fortran_core, gf.all_tabulation_parameters
    sing = gf.gf_singularities_fortran_enum['low_freq']
    grad_at = lambda x, k: np.asarray(fc.green_wave.wave_part_infinite_depth(x, xi, k, *tab, sing)[1])
    b = np.array([-1.0, -1.0, 1.0]) * np.array([0.0, 0.0, 1.0])   # M n_j, grad_xi = diag(-1, -1, 1) grad_x
    rng = np.random.default_rng(5)
    for k in (0.5, 2.0):
        for _ in range(3):
            x = np.array([rng.uniform(-2, 2), rng.uniform(-2, 2), rng.uniform(-3.5, -0.6)])
            a = rng.normal(size=3)
            a /= np.linalg.norm(a)
            step = 2e-3
            hess_b = np.array([(grad_at(x + step * e, k) - grad_at(x - step * e, k)) / (2 * step) for e in np.eye(3)]).T @ b
            expected = -area * (a @ hess_b) / (4 * np.pi)          # G_wave = -g/(4 pi)
            got = tsc.wave_hypersingular(gf, x[None], a[None], V, F, S, wavenumber=k)[0, 0]
            # oracle limit: differences of a gradient that Capytaine integrates to ~1e-4 (measured
            # scatter up to ~6e-4); the analytic form itself is exact given g and grad g
            assert got == pytest.approx(expected, rel=2e-3)


def test_image_part_is_the_exact_mirrored_ring():
    """Image double layer of -1/(4 pi |x - R xi|): minus the ring of the mirrored panel (independent quadrature)."""
    from anyhydro import thin_sheet_capytaine as tsc
    V, F, S = _sheet()
    x = np.array([0.7, 0.4, -1.1])
    poly = V[F[2]]

    def image_phi(p):
        # int n . grad_xi[-1/(4 pi |x - R xi|)] dS = -Phi_rankine(x; mirrored panel, its own winding)
        return -double_layer(p, poly * [1, 1, -1])
    h = 1e-4
    normal = np.array([0.3, -0.2, 0.93]) / np.linalg.norm([0.3, -0.2, 0.93])
    fd = (image_phi(x + h * normal) - image_phi(x - h * normal)) / (2 * h)
    got = tsc.image_hypersingular(x[None], normal[None], V, F[[2]], S[[2]])[0, 0]
    assert got == pytest.approx(fd, rel=1e-6)
