"""TS2: linear Galerkin thin-sheet panels (docs/NATIVE_THIN_SHEET_PLAN.md)."""
import numpy as np
import pytest

from anyhydro.thin_sheet_galerkin import isolated_sheet_added_mass, triangle_single_layer, free_edge_nodes, triangulate
from test_thin_sheet_kernels import _BARY, _W

TRI = np.array([[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.3, 0.9, 0.0]])


def _quadrature_single_layer(x, n=60):
    total = 0.0
    for s in range(n):
        for t in range(n - s):
            cells = [[(s, t), (s + 1, t), (s, t + 1)]] + ([[(s + 1, t), (s + 1, t + 1), (s, t + 1)]] if s + t < n - 1 else [])
            for cell in cells:
                q = np.array([TRI[0] + (TRI[1] - TRI[0]) * u / n + (TRI[2] - TRI[0]) * v / n for u, v in cell])
                area = 0.5 * np.linalg.norm(np.cross(q[1] - q[0], q[2] - q[0]))
                total += area * np.sum(_W / np.linalg.norm(x - _BARY @ q, axis=1))
    return total


@pytest.mark.parametrize('x', [[0.5, 0.3, 0.4], [2.0, -1.0, 0.7], [0.5, 0.3, -0.05]])
def test_triangle_single_layer_matches_quadrature_off_the_plane(x):
    x = np.array(x)
    assert triangle_single_layer(x, TRI)[0] == pytest.approx(_quadrature_single_layer(x), rel=1e-10)


def test_triangle_single_layer_in_the_plane_matches_polar_integration():
    """int 1/R dS from an interior point = int r(theta) dtheta (polar coordinates about the point)."""
    x = np.array([0.45, 0.3, 0.0])

    def reach(t):
        d = np.array([np.cos(t), np.sin(t)])
        best = np.inf
        for i in range(3):
            a, b = TRI[i, :2], TRI[(i + 1) % 3, :2]
            s, u = np.linalg.solve(np.column_stack([d, a - b]), a - x[:2])
            if s > 0 and -1e-12 <= u <= 1 + 1e-12:
                best = min(best, s)
        return best
    theta = np.linspace(0.0, 2 * np.pi, 20001)[:-1]
    expected = np.mean([reach(t) for t in theta]) * 2 * np.pi
    assert triangle_single_layer(x, TRI)[0] == pytest.approx(expected, rel=1e-7)


def disc(a, nr, nt):
    r = a * np.sin(0.5 * np.pi * np.arange(nr + 1) / nr)
    t = 2 * np.pi * np.arange(nt) / nt
    vertices = [[0.0, 0.0, 0.0]] + [[ri * np.cos(tj), ri * np.sin(tj), 0.0] for ri in r[1:] for tj in t]
    idx = lambda i, j: 1 + (i - 1) * nt + (j % nt)
    faces, sizes = [], []
    for j in range(nt):
        faces.append([0, idx(1, j), idx(1, j + 1), -1])
        sizes.append(3)
    for i in range(1, nr):
        for j in range(nt):
            faces.append([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1)])
            sizes.append(4)
    return np.array(vertices), np.array(faces), np.array(sizes)


def test_free_edge_nodes_are_the_rim():
    V, F, S = disc(1.0, 3, 12)
    vertices, triangles = triangulate(V, F, S)
    rim = free_edge_nodes(triangles)
    assert np.allclose(np.linalg.norm(vertices[rim, :2], axis=1), 1.0)
    assert len(rim) == 12


def test_ts2_thin_disc_heave_added_mass_converges_to_8_over_3():
    """Declared before running (plan TS2): finest mesh within 1 % of 8/3 rho a^3, error decreasing."""
    rho, a = 1025.0, 1.0
    errors = []
    for nr, nt in ((4, 16), (6, 24), (9, 36), (12, 48)):
        A, _ = isolated_sheet_added_mass(*disc(a, nr, nt), rotation_center=(0.0, 0.0, 0.0), rho=rho)
        errors.append(A[2, 2] / (8 / 3 * rho * a ** 3) - 1.0)
    assert all(abs(e2) < abs(e1) for e1, e2 in zip(errors, errors[1:]))
    assert abs(errors[-1]) <= 0.01
