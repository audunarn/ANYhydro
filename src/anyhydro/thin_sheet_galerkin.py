# SPDX-License-Identifier: MPL-2.0
"""Linear thin-sheet panels, Galerkin hypersingular operator (plan TS2, owner decision 2026-10-13).

The double-layer density ``mu`` (jump ``m = -mu``) is piecewise linear on flat triangles, zero at
free-edge nodes. The Rankine hypersingular part is tested by Galerkin through Maue's identity
(open surface, ``mu = psi = 0`` on the free boundary):

    <psi, H mu> = int int G0(x, xi) curl_S psi(x) . curl_S mu(xi) dS dS',   G0 = 1/(4 pi r),

so only triangle-pair integrals of ``1/r`` remain; they use the analytic integral of ``1/R`` over
a flat triangle (inner) and Dunavant points (outer). Qt-free, Capytaine-free.
"""
from __future__ import annotations

import numpy as np

# Dunavant degree 5, 7 points: barycentric coordinates and weights (sum 1).
_A1, _B1 = 0.059715871789770, 0.470142064105115
_A2, _B2 = 0.797426985353087, 0.101286507323456
GAUSS_BARY = np.array([[1 / 3, 1 / 3, 1 / 3], [_A1, _B1, _B1], [_B1, _A1, _B1], [_B1, _B1, _A1],
                       [_A2, _B2, _B2], [_B2, _A2, _B2], [_B2, _B2, _A2]])
GAUSS_W = np.array([0.225] + [0.132394152788506] * 3 + [0.125939180544827] * 3)


def triangle_single_layer(points, tri):
    """``int_T 1/|x - xi| dS(xi)`` for points (n, 3) and one flat triangle (3, 3), analytic (Wilton)."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    tri = np.asarray(tri, dtype=float)
    normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
    normal /= np.linalg.norm(normal)
    d = (points - tri[0]) @ normal
    rho = points - d[:, None] * normal
    ad = np.abs(d)
    total = np.zeros(points.shape[0])
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        length = np.linalg.norm(b - a)
        l_hat = (b - a) / length
        u_hat = np.cross(l_hat, normal)                       # outward in-plane edge normal
        p0 = (a - rho) @ u_hat
        l_plus = (b - rho) @ l_hat
        l_minus = (a - rho) @ l_hat
        r0_sq = p0 * p0 + d * d
        r_plus = np.sqrt(r0_sq + l_plus * l_plus)
        r_minus = np.sqrt(r0_sq + l_minus * l_minus)
        on_line = np.abs(p0) <= 1e-14 * length

        def r_plus_l(r, l):
            # R + l without cancellation: for l < 0 use the exact R + l = R0^2 / (R - l).
            return np.where(l >= 0.0, r + l, r0_sq / np.where(l >= 0.0, 1.0, r - l))
        num, den = r_plus_l(r_plus, l_plus), r_plus_l(r_minus, l_minus)
        log_term = np.where(on_line, 0.0, p0 * np.log(np.where(on_line, 1.0, num) / np.where(on_line, 1.0, den)))
        angle = (np.arctan2(p0 * l_plus, r0_sq + ad * r_plus) - np.arctan2(p0 * l_minus, r0_sq + ad * r_minus))
        total += log_term - ad * angle
    return total


def triangulate(vertices, faces, face_sizes):
    """Flat triangles (index triples) from polygon faces (quads split along their first diagonal)."""
    tris = []
    for face, size in zip(np.asarray(faces), np.asarray(face_sizes)):
        f = [int(v) for v in face[:size]]
        tris += [[f[0], f[k], f[k + 1]] for k in range(1, size - 1)]
    return np.asarray(vertices, dtype=float), np.array(tris)


def free_edge_nodes(triangles):
    """Nodes on edges used by a single triangle (the sheet's free boundary)."""
    count = {}
    for t in triangles:
        for k in range(3):
            e = tuple(sorted((int(t[k]), int(t[(k + 1) % 3]))))
            count[e] = count.get(e, 0) + 1
    return sorted({v for e, c in count.items() if c == 1 for v in e})


def triangle_geometry(vertices, triangles):
    """Areas, unit normals and the constant surface curls ``n x grad N_a`` (n_tri, 3 nodes, 3)."""
    p = vertices[triangles]                                    # (n, 3, 3)
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    normals = cross / twice_area[:, None]
    curls = np.zeros_like(p)
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        grad = np.cross(normals, p[:, c] - p[:, b]) / twice_area[:, None]       # grad N_a
        curls[:, a] = np.cross(normals, grad)
    return 0.5 * twice_area, normals, curls


def rankine_pair_integrals(vertices, triangles):
    """``I[T, T'] = int_T int_T' 1/(4 pi r)`` (symmetrised), outer Dunavant x inner analytic."""
    p = vertices[triangles]
    areas, _, _ = triangle_geometry(vertices, triangles)
    outer = np.einsum('qa,tak->tqk', GAUSS_BARY, p)            # (n_tri, 7, 3)
    flat = outer.reshape(-1, 3)
    n = len(triangles)
    pair = np.zeros((n, n))
    for j in range(n):
        sl = triangle_single_layer(flat, p[j]).reshape(n, len(GAUSS_W))
        pair[:, j] = areas * (sl @ GAUSS_W)
    pair /= 4.0 * np.pi
    return 0.5 * (pair + pair.T)


def rankine_galerkin(vertices, triangles):
    """Nodal Galerkin matrix ``K[v, w] = <N_v, H_rankine N_w>`` over all nodes (n_nodes, n_nodes)."""
    _, _, curls = triangle_geometry(vertices, triangles)
    pair = rankine_pair_integrals(vertices, triangles)
    n_nodes = len(vertices)
    K = np.zeros((n_nodes, n_nodes))
    local = np.einsum('tak,sbk->tasb', curls, curls) * pair[:, None, :, None]
    for a in range(3):
        for b in range(3):
            np.add.at(K, (triangles[:, a][:, None], triangles[:, b][None, :]), local[:, a, :, b])
    return K


def load_vector(vertices, triangles, values_at_points):
    """``b[v] = int N_v f dS`` with ``f`` evaluated at the Dunavant points (callable on (n, 3) points)."""
    p = vertices[triangles]
    areas, normals, _ = triangle_geometry(vertices, triangles)
    points = np.einsum('qa,tak->tqk', GAUSS_BARY, p)
    f = values_at_points(points.reshape(-1, 3), np.repeat(normals, len(GAUSS_W), axis=0))
    f = f.reshape(len(triangles), len(GAUSS_W), -1)
    out = np.zeros((len(vertices), f.shape[-1]), dtype=np.result_type(f.dtype, float))   # complex f stays complex
    for a in range(3):
        contribution = np.einsum('q,tqm->tm', GAUSS_W * GAUSS_BARY[:, a], f) * areas[:, None]
        np.add.at(out, triangles[:, a], contribution)
    return out


def isolated_sheet_added_mass(vertices, faces, face_sizes, *, rotation_center, rho):
    """Infinite-fluid added mass of an isolated sheet with linear Galerkin panels (Rankine only)."""
    vertices, triangles = triangulate(vertices, faces, face_sizes)
    fixed = set(free_edge_nodes(triangles))
    free = np.array([v for v in range(len(vertices)) if v not in fixed], dtype=int)
    K = rankine_galerkin(vertices, triangles)
    centre = np.asarray(rotation_center, dtype=float)

    def modes(points, normals):
        return np.column_stack([normals, np.cross(points - centre, normals)])
    b = load_vector(vertices, triangles, modes)                # (n_nodes, 6)
    mu = np.zeros((len(vertices), 6))
    if free.size:                       # a sheet with no interior node carries no linear jump
        mu[free] = np.linalg.solve(K[np.ix_(free, free)], b[free])
    added_mass = rho * b.T @ mu                                # A = rho int mu (V . n) dS (m = -mu)
    return 0.5 * (added_mass + added_mass.T), mu
