# SPDX-License-Identifier: MPL-2.0
"""Free-surface thin sheets with linear Galerkin panels, infinite depth (plan TS2, second half).

Adds to the Rankine Maue operator of ``thin_sheet_galerkin`` the two regular parts of Capytaine's
infinite-depth ``G = -(1/r + 1/r1 + g_w)/(4 pi)``, Galerkin-tested with a Gauss product rule:
- image: ``n_x^T grad_x grad_eta G0'(x, eta) (R n_xi)`` with ``eta = R xi``, ``R = diag(1, 1, -1)``
  and ``grad_x grad_eta (-1/(4 pi r)) = -(I/r^3 - 3 r r^T/r^5)/(4 pi)``;
- wave: TS1's analytic Hessian of Capytaine's point-pair wave term.
Radiation per unit mode velocity gives ``A = rho Re(b^T mu)``, ``B = w rho Im(b^T mu)``; the
diffraction force on a sheet is ``F = i w rho b^T mu_D`` (the incident pressure is continuous
across a sheet, so it exerts no Froude-Krylov force), with Capytaine's incident potential
``phi_I = -i g/w exp(k z) exp(i k (x cos b + y sin b))`` (x = Re{X exp(-i w t)}).
Sheets closer to the free surface than ``MIN_DEPTH_PER_EDGE`` longest edges (including surface-piercing
sheets) are refused; they need near-singular integration (TS4).
"""
from __future__ import annotations

import numpy as np

from .thin_sheet_galerkin import (GAUSS_BARY, GAUSS_W, free_edge_nodes, load_vector, rankine_galerkin,
                                  triangle_geometry, triangulate)

_R = np.array([1.0, 1.0, -1.0])
_M = np.array([-1.0, -1.0, 1.0])
# Supported domain: every vertex at least this many longest-triangle-edges below the free surface.
# The image and wave kernels are smooth there and the 7-point product rule resolves them; closer
# to z = 0 they are near-singular and need dedicated integration (TS4). Bound measured with the
# subdivided-rule probe (test_quadrature_converged_at_the_depth_bound).
MIN_DEPTH_PER_EDGE = 1.0
# Pair-kernel entries held at once during assembly (blocked over row and source triangles;
# a single triangle pair, nq**2 entries, is the floor).
MAX_PAIR_ENTRIES = 2_000_000


def subdivided_rule(levels):
    """The Dunavant rule on each of ``4**levels`` congruent sub-triangles (barycentric, weights sum 1)."""
    tris = [np.eye(3)]
    for _ in range(levels):
        nxt = []
        for t in tris:
            m01, m12, m20 = (t[0] + t[1]) / 2, (t[1] + t[2]) / 2, (t[2] + t[0]) / 2
            nxt += [np.array(q) for q in ((t[0], m01, m20), (m01, t[1], m12), (m20, m12, t[2]), (m12, m20, m01))]
        tris = nxt
    bary = np.concatenate([GAUSS_BARY @ t for t in tris])
    return bary, np.tile(GAUSS_W, len(tris)) / len(tris)


def check_supported_depth(vertices, triangles):
    """Refuse sheets closer to the free surface than ``MIN_DEPTH_PER_EDGE`` longest edges."""
    p = vertices[triangles]
    edge = max(np.linalg.norm(p[:, a] - p[:, (a + 1) % 3], axis=1).max() for a in range(3))
    depth = -float(vertices[:, 2].max())
    if not depth >= MIN_DEPTH_PER_EDGE * edge:
        raise ValueError(f"sheet top at depth {depth:.6g} m is closer to the free surface than "
                         f"{MIN_DEPTH_PER_EDGE} x the longest triangle edge {edge:.6g} m; near-surface "
                         "and surface-piercing sheets need near-singular integration (plan TS4)")


def image_kernel(xa, na, xb, nb):
    """``n_a . grad_x grad_xi (-1/(4 pi |x - R xi|)) . n_b`` for receivers a (rows), sources b (columns)."""
    eta, n_eta = xb * _R, nb * _R
    d = xa[:, None, :] - eta[None, :, :]
    r = np.sqrt(np.einsum('ijk,ijk->ij', d, d))
    a_dot_d = np.einsum('ik,ijk->ij', na, d)
    b_dot_d = np.einsum('jk,ijk->ij', n_eta, d)
    return -((na @ n_eta.T) / r ** 3 - 3.0 * a_dot_d * b_dot_d / r ** 5) / (4.0 * np.pi)


def wave_kernel(gf, wavenumber, xa, na, xb, nb):
    """``n_a . grad_x grad_xi G_w . n_b`` of Capytaine's wave part, ``G_w = -g_w/(4 pi)`` (TS1 Hessian)."""
    k = float(wavenumber)
    na_, nb_ = xa.shape[0], xb.shape[0]
    xi = np.repeat(xa, nb_, axis=0)
    src = np.tile(xb, (na_, 1))
    _, grad = gf.fortran_core.interface.vectorized_wave_part_infinite_depth(
        np.ascontiguousarray(xi), np.ascontiguousarray(src), k, *gf.all_tabulation_parameters,
        gf.gf_singularities_fortran_enum['low_freq'])
    grad = np.asarray(grad).reshape(na_, nb_, 3)
    a = np.broadcast_to(na[:, None, :], grad.shape)
    b = np.broadcast_to((nb * _M)[None, :, :], grad.shape)
    dh = xa[:, None, :2] - xb[None, :, :2]
    R = np.linalg.norm(dh, axis=-1)
    Z = xa[:, None, 2] + xb[None, :, 2]
    r1 = np.sqrt(R * R + Z * Z)
    small = R < 1e-12 * np.maximum(r1, 1e-300)
    e = np.where(small[..., None], 0.0, dh / np.where(small, 1.0, R)[..., None])
    g_R = np.einsum('...k,...k->...', grad[..., :2], e)
    g_Z = grad[..., 2]
    g_ZZ = k * g_Z - 2.0 * k * Z / r1 ** 3
    g_RZ = k * g_R - 2.0 * k * R / r1 ** 3
    g_R_over_R = np.where(small, -0.5 * g_ZZ, g_R / np.where(small, 1.0, R))
    g_RR = np.where(small, -0.5 * g_ZZ, -g_R_over_R - g_ZZ)
    ae = np.einsum('...k,...k->...', a[..., :2], e)
    be = np.einsum('...k,...k->...', b[..., :2], e)
    ab = np.einsum('...k,...k->...', a[..., :2], b[..., :2])
    return -(g_RR * ae * be + g_R_over_R * (ab - ae * be) + g_RZ * (ae * b[..., 2] + a[..., 2] * be)
             + g_ZZ * a[..., 2] * b[..., 2]) / (4.0 * np.pi)


def _assemble(kernel, vertices, triangles, rule=(GAUSS_BARY, GAUSS_W), max_entries=MAX_PAIR_ENTRIES):
    """``K[v, w] = sum_{T,T'} sum_{q,p} w_q N_v(q) w_p N_w(p) kernel(q, p)``, blocked over row and source
    triangles so one kernel call holds at most ``max(max_entries, nq**2)`` pair entries."""
    bary, w = rule
    areas, normals, _ = triangle_geometry(vertices, triangles)
    points = np.einsum('qa,tak->tqk', bary, vertices[triangles])
    n_tri, nq = points.shape[:2]
    basis = (areas[:, None] * w[None, :])[:, :, None] * bary[None, :, :]       # (n_tri, nq, 3)
    pairs = max(1, max_entries // (nq * nq))                                   # triangle pairs per call
    cols = min(n_tri, pairs)
    rows = max(1, pairs // cols)
    out = None
    for r0 in range(0, n_tri, rows):
        r = slice(r0, min(n_tri, r0 + rows))
        xa, na = points[r].reshape(-1, 3), np.repeat(normals[r], nq, axis=0)
        for c0 in range(0, n_tri, cols):
            c = slice(c0, min(n_tri, c0 + cols))
            kern = kernel(xa, na, points[c].reshape(-1, 3), np.repeat(normals[c], nq, axis=0))
            kern = kern.reshape(-1, nq, c.stop - c.start, nq)
            local = np.einsum('tqa,tqsp,spb->tasb', basis[r], kern, basis[c], optimize=True)
            if out is None:
                out = np.zeros((len(vertices), len(vertices)), dtype=local.dtype)
            for a in range(3):
                for b in range(3):
                    np.add.at(out, (triangles[r, a][:, None], triangles[None, c, b]), local[:, a, :, b])
    return out


def image_galerkin(vertices, triangles, rule=(GAUSS_BARY, GAUSS_W)):
    return _assemble(image_kernel, vertices, triangles, rule)


def wave_galerkin(gf, vertices, triangles, wavenumber, rule=(GAUSS_BARY, GAUSS_W)):
    """Galerkin wave part from TS1's analytic Hessian at Gauss point pairs (row-blocked)."""
    return _assemble(lambda xa, na, xb, nb: wave_kernel(gf, wavenumber, xa, na, xb, nb), vertices, triangles, rule)


def sheet_hydrodynamics(vertices, faces, face_sizes, *, omega, rho, gravity, rotation_center, headings_deg,
                        gf=None):
    """Added mass, damping (6 x 6) and diffraction excitation (headings x 6) of an isolated sheet."""
    if gf is None:
        from .green_function import anyhydro_green_function
        gf = anyhydro_green_function()
    vertices, triangles = triangulate(vertices, faces, face_sizes)
    check_supported_depth(vertices, triangles)
    fixed = set(free_edge_nodes(triangles))
    free = np.array([v for v in range(len(vertices)) if v not in fixed], dtype=int)
    k = omega * omega / gravity
    K = (rankine_galerkin(vertices, triangles) + image_galerkin(vertices, triangles)
         + wave_galerkin(gf, vertices, triangles, k))
    centre = np.asarray(rotation_center, dtype=float)

    def modes(points, normals):
        return np.column_stack([normals, np.cross(points - centre, normals)])
    b = load_vector(vertices, triangles, modes)                         # (n_nodes, 6), real
    mu = np.zeros((len(vertices), 6), dtype=complex)
    solve = (lambda rhs: np.linalg.solve(K[np.ix_(free, free)], rhs)) if free.size else None
    if solve is not None:
        mu[free] = solve(b[free])
    generalized = b.T @ mu
    added_mass = rho * np.real(generalized)
    damping = omega * rho * np.imag(generalized)
    forces = []
    for beta in np.radians(np.asarray(headings_deg, dtype=float)):
        direction = np.array([np.cos(beta), np.sin(beta), 0.0])

        def minus_dphi_dn(points, normals, direction=direction):
            phase = np.exp(k * points[:, 2] + 1j * k * (points @ direction))
            grad = (-1j * gravity / omega) * phase[:, None] * np.column_stack(
                [1j * k * direction[0] * np.ones(len(points)), 1j * k * direction[1] * np.ones(len(points)),
                 k * np.ones(len(points))])
            return -np.einsum('nk,nk->n', grad, normals)[:, None]
        rhs = load_vector(vertices, triangles, minus_dphi_dn)[:, 0]
        mu_d = np.zeros(len(vertices), dtype=complex)
        if solve is not None:
            mu_d[free] = solve(rhs[free])
        forces.append(1j * omega * rho * (b.T @ mu_d))
    return added_mass, damping, np.array(forces, dtype=complex).reshape(len(forces), 6)
