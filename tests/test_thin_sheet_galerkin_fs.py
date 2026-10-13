"""TS2 second half: free-surface linear Galerkin sheets (docs/NATIVE_THIN_SHEET_PLAN.md).

The OrcaWave plate comparison (mesh-extrapolated, A33 +0.24 %, B33 +0.9 %, F3 +0.24 %) is evidence
in anymarine-evidence/thin-sheet-ts2-orcawave-20261013; these tests pin the internal identities.
"""
import numpy as np
import pytest

from anyhydro.thin_sheet_galerkin import isolated_sheet_added_mass, load_vector, triangulate
from anyhydro.thin_sheet_galerkin_fs import sheet_hydrodynamics

RHO, G = 1025.0, 9.80665
needs_capytaine = pytest.mark.skipif(
    __import__('importlib').util.find_spec('capytaine') is None, reason='wave Green function needs the capytaine extra')


def plate(L, d, n):
    xs = np.linspace(-L / 2, L / 2, n + 1)
    V = np.array([[x, y, -d] for y in xs for x in xs])
    idx = lambda i, j: j * (n + 1) + i
    F = np.array([[idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)] for j in range(n) for i in range(n)])
    return V, F, np.full(len(F), 4)


def test_load_vector_keeps_complex_values():
    """Review of the first run: a float accumulator dropped the imaginary part of the diffraction load."""
    vertices, triangles = triangulate(*plate(1.0, 1.0, 2))
    out = load_vector(vertices, triangles, lambda p, n: np.full((len(p), 1), 1j))
    assert np.iscomplexobj(out) and np.sum(out).imag == pytest.approx(1.0) and np.sum(out).real == 0.0


@needs_capytaine
@pytest.mark.parametrize('period', [2.0, 3.0])
def test_haskind_relation_heave_and_pitch(period):
    """Deep water: B_jj = k w / (4 pi rho g^2) int |F_j(beta)|^2 d beta (diffraction vs radiation)."""
    w = 2 * np.pi / period
    k = w * w / G
    beta = np.arange(0.0, 360.0, 10.0)
    _, B, F = sheet_hydrodynamics(*plate(2.0, 1.0, 6), omega=w, rho=RHO, gravity=G, rotation_center=(0, 0, -1.0),
                                  headings_deg=beta)
    for j in (2, 4):
        haskind = k * w / (4 * np.pi * RHO * G * G) * np.sum(np.abs(F[:, j]) ** 2) * np.radians(10.0)
        assert haskind == pytest.approx(B[j, j], rel=1e-3)


@needs_capytaine
def test_deep_sheet_tends_to_infinite_fluid():
    """Image and wave terms vanish with depth: A33 at 30 plate lengths is the Rankine value."""
    mesh = plate(1.0, 30.0, 6)
    A_inf, _ = isolated_sheet_added_mass(*mesh, rotation_center=(0, 0, -30.0), rho=RHO)
    A, B, _ = sheet_hydrodynamics(*mesh, omega=2.0, rho=RHO, gravity=G, rotation_center=(0, 0, -30.0),
                                  headings_deg=[0.0])
    assert A[2, 2] == pytest.approx(A_inf[2, 2], rel=1e-4)
    assert abs(B[2, 2]) < 1e-6 * A_inf[2, 2]


@needs_capytaine
def test_long_waves_raise_heave_added_mass_short_waves_damp():
    """Low frequency: the surface acts as a rigid lid (positive image), A33 above infinite fluid."""
    mesh = plate(2.0, 1.5, 6)
    A_inf, _ = isolated_sheet_added_mass(*mesh, rotation_center=(0, 0, 0), rho=RHO)
    A, B, _ = sheet_hydrodynamics(*mesh, omega=2 * np.pi / 10.0, rho=RHO, gravity=G, rotation_center=(0, 0, 0),
                                  headings_deg=[0.0])
    assert A[2, 2] > A_inf[2, 2] and B[2, 2] > 0.0


def tilted_plate(L, depth, n, angle_deg):
    """Square plate rotated about the x axis by ``angle_deg``, top edge at ``depth``."""
    V, F, S = plate(L, 0.0, n)
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    V = V @ np.array([[1, 0, 0], [0, c, s], [0, -s, c]]).T
    V[:, 2] -= depth + V[:, 2].max()
    return V, F, S


@needs_capytaine
def test_tilted_sheet_reciprocity_and_haskind_all_modes():
    """Review: horizontal normals exercise the radial and mixed Hessian terms."""
    w, beta = 2 * np.pi / 2.5, np.arange(0.0, 360.0, 10.0)
    k = w * w / G
    A, B, F = sheet_hydrodynamics(*tilted_plate(2.0, 0.8, 6, 60.0), omega=w, rho=RHO, gravity=G,
                                  rotation_center=(0, 0, -1.0), headings_deg=beta)
    assert np.allclose(A, A.T, rtol=0, atol=1e-6 * np.abs(A).max())
    assert np.allclose(B, B.T, rtol=0, atol=1e-6 * np.abs(B).max())
    assert np.linalg.eigvalsh(0.5 * (B + B.T)).min() > -1e-6 * np.abs(B).max()
    for j in range(6):
        if B[j, j] > 1e-3 * np.abs(B).max():
            haskind = k * w / (4 * np.pi * RHO * G * G) * np.sum(np.abs(F[:, j]) ** 2) * np.radians(10.0)
            assert haskind == pytest.approx(B[j, j], rel=1e-3)


def _second_difference(f, a, na, b, nb, h):
    return (f(a + h * na, b + h * nb) - f(a + h * na, b - h * nb) - f(a - h * na, b + h * nb)
            + f(a - h * na, b - h * nb)) / (4 * h * h)


def test_image_kernel_matches_finite_differences():
    from anyhydro.thin_sheet_galerkin_fs import image_kernel
    rng = np.random.default_rng(3)
    for _ in range(5):
        a, b = rng.normal(size=3) - [0, 0, 2], rng.normal(size=3) - [0, 0, 2]
        na, nb = (v / np.linalg.norm(v) for v in rng.normal(size=(2, 3)))
        g = lambda x, xi: -1.0 / (4 * np.pi * np.linalg.norm(x - xi * [1, 1, -1]))
        expected = _second_difference(g, a, na, b, nb, 1e-4)
        assert image_kernel(a[None], na[None], b[None], nb[None])[0, 0] == pytest.approx(expected, rel=1e-5)


def _pv(f, k):
    from scipy.integrate import quad
    a, _ = quad(f, 0, 2 * k, weight='cauchy', wvar=k, limit=400, epsabs=0, epsrel=1e-12)
    b, _ = quad(lambda m: f(m) / (m - k), 2 * k, np.inf, limit=800, epsabs=0, epsrel=1e-12)
    return a + b


def _exact_wave(x, xi, k, order):
    """Deep-water wave part g_w = 2k PV int e^{mZ} J0(mR)/(m-k) dm + 2 pi i k e^{kZ} J0(kR) (Capytaine's
    low_freq convention, checked against its untabulated value to 1e-4): gradient (order 1) or Hessian
    (order 2) with respect to the receiver x."""
    from scipy.special import j0, j1
    dh = x[:2] - xi[:2]
    R, Z = np.hypot(*dh), x[2] + xi[2]
    e = dh / R if R > 0 else np.array([1.0, 0.0])
    jr = lambda u: j1(u) / u if u > 0 else 0.5
    term = lambda f, p: 2 * k * _pv(lambda m: m ** p * np.exp(m * Z) * f(m * R), k)         + 2j * np.pi * k ** (p + 1) * np.exp(k * Z) * f(k * R)
    if order == 1:
        return np.concatenate([term(lambda u: -j1(u), 1) * e, [term(j0, 1)]])
    E = np.outer(e, e)
    H = np.zeros((3, 3), complex)
    H[:2, :2] = term(lambda u: -j0(u) + jr(u), 2) * E + term(lambda u: -jr(u), 2) * (np.eye(2) - E)
    H[:2, 2] = H[2, :2] = term(lambda u: -j1(u), 2) * e
    H[2, 2] = term(j0, 2)
    return H


class _ExactGradient:
    """Green-function stand-in whose wave gradient is the exact integral (isolates our Hessian formula)."""
    all_tabulation_parameters = ()
    gf_singularities_fortran_enum = {'low_freq': 0}

    class fortran_core:
        class interface:
            @staticmethod
            def vectorized_wave_part_infinite_depth(xi, src, k, singularities):
                return None, np.array([_exact_wave(a, b, k, 1) for a, b in zip(xi, src)])


def _pairs():
    rng = np.random.default_rng(5)
    cases = [(rng.normal(size=3) - [0, 0, 1.5], rng.normal(size=3) - [0, 0, 1.5]) for _ in range(4)]
    cases.append((np.array([0.3, -0.2, -1.0]), np.array([0.3, -0.2, -1.7])))     # co-vertical pair (R = 0)
    return [(a, b, *(v / np.linalg.norm(v) for v in rng.normal(size=(2, 3)))) for a, b in cases]


def _exact_kernel(a, b, na, nb, k):
    return -(na @ _exact_wave(a, b, k, 2) @ (nb * [-1, -1, 1])) / (4 * np.pi), np.abs(_exact_wave(a, b, k, 2)).max() / (4 * np.pi)


@pytest.mark.parametrize('wavenumber', [0.3, 2.0])
def test_wave_kernel_hessian_formula_is_exact_given_the_gradient(wavenumber):
    """Review: tilted normals and R = 0 exercise every Hessian term; exact Hessian by Bessel integrals."""
    from anyhydro.thin_sheet_galerkin_fs import wave_kernel
    for a, b, na, nb in _pairs():
        expected, scale = _exact_kernel(a, b, na, nb, wavenumber)
        got = wave_kernel(_ExactGradient(), wavenumber, a[None], na[None], b[None], nb[None])[0, 0]
        assert abs(got - expected) <= 1e-8 * scale


@needs_capytaine
@pytest.mark.parametrize('wavenumber', [0.3, 2.0])
def test_wave_kernel_with_capytaine_within_its_gradient_accuracy(wavenumber):
    """Capytaine's own wave gradient is accurate to ~7e-4 (measured against the exact integrals at
    kR 2.5..5.5, kZ -6..-9); the Hessian built from it stays within 5e-3 of the Hessian scale."""
    from anyhydro.green_function import anyhydro_green_function
    from anyhydro.thin_sheet_galerkin_fs import wave_kernel
    gf = anyhydro_green_function()
    for a, b, na, nb in _pairs():
        expected, scale = _exact_kernel(a, b, na, nb, wavenumber)
        got = wave_kernel(gf, wavenumber, a[None], na[None], b[None], nb[None])[0, 0]
        assert abs(got - expected) <= 5e-3 * scale


def test_blocked_assembly_equals_one_block():
    from anyhydro.thin_sheet_galerkin_fs import _assemble, image_kernel
    vertices, triangles = triangulate(*tilted_plate(1.0, 0.7, 3, 30.0))
    one = _assemble(image_kernel, vertices, triangles, max_entries=10 ** 9)
    many = _assemble(image_kernel, vertices, triangles, max_entries=1)
    assert np.allclose(one, many, rtol=1e-13, atol=0)


@pytest.mark.parametrize('cap', [1, 49, 50, 49 * 7, 49 * 18 * 18])
def test_assembly_respects_the_pair_entry_cap(cap):
    """Review round 2: one row triangle against every source exceeded the cap; sources are blocked too."""
    from anyhydro.thin_sheet_galerkin_fs import _assemble, image_kernel
    vertices, triangles = triangulate(*tilted_plate(1.0, 0.7, 3, 30.0))
    sizes = []

    def recording(xa, na, xb, nb):
        sizes.append(len(xa) * len(xb))
        return image_kernel(xa, na, xb, nb)
    blocked = _assemble(recording, vertices, triangles, max_entries=cap)
    assert max(sizes) <= max(cap, 49) and sum(sizes) == (7 * len(triangles)) ** 2
    assert np.allclose(blocked, _assemble(image_kernel, vertices, triangles, max_entries=10 ** 9), rtol=1e-13, atol=0)


@pytest.mark.parametrize('depth', [0.0, 0.2])
def test_sheets_near_the_free_surface_are_refused(depth):
    """Review: z = 0 divided by zero; shallow sheets lose accuracy silently (domain bound, TS4)."""
    with pytest.raises(ValueError, match='TS4'):
        sheet_hydrodynamics(*plate(2.0, depth, 4), omega=2.0, rho=RHO, gravity=G, rotation_center=(0, 0, 0),
                            headings_deg=[0.0], gf=object())


@needs_capytaine
def test_quadrature_converged_at_the_depth_bound():
    """At the bound (depth = longest edge) the 7-point rule matches a 448-point rule to 1e-4 (probe: 3e-6)."""
    from anyhydro.green_function import anyhydro_green_function
    from anyhydro.thin_sheet_galerkin import free_edge_nodes, rankine_galerkin
    from anyhydro.thin_sheet_galerkin_fs import (GAUSS_BARY, GAUSS_W, MIN_DEPTH_PER_EDGE, image_galerkin,
                                                 subdivided_rule, wave_galerkin)
    gf = anyhydro_green_function()
    n, L = 4, 2.0
    vertices, triangles = triangulate(*plate(L, MIN_DEPTH_PER_EDGE * L / n * np.sqrt(2), n))
    free = np.array([v for v in range(len(vertices)) if v not in set(free_edge_nodes(triangles))])
    b = load_vector(vertices, triangles, lambda p, nrm: nrm[:, 2:3])[free, 0]
    values = []
    for rule in ((GAUSS_BARY, GAUSS_W), subdivided_rule(3)):
        K = rankine_galerkin(vertices, triangles) + image_galerkin(vertices, triangles, rule) \
            + wave_galerkin(gf, vertices, triangles, 1.0, rule)
        values.append(b @ np.linalg.solve(K[np.ix_(free, free)], b))
    assert values[0] == pytest.approx(values[1], rel=1e-4)


def test_no_headings_gives_an_empty_six_column_excitation(monkeypatch):
    import anyhydro.thin_sheet_galerkin_fs as fs
    monkeypatch.setattr(fs, 'wave_galerkin', lambda gf, v, t, k, rule=None: np.zeros((len(v), len(v))))
    _, _, F = fs.sheet_hydrodynamics(*plate(2.0, 1.5, 4), omega=2.0, rho=RHO, gravity=G, rotation_center=(0, 0, 0),
                                     headings_deg=[], gf=object())
    assert F.shape == (0, 6)
