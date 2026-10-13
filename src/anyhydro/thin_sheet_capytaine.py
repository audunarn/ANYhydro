# SPDX-License-Identifier: MPL-2.0
"""Sheet-row (hypersingular) operator on Capytaine's Green function (plan TS1, infinite depth).

``H[i, j] = n_i . grad_x Phi_j(x_i)``, ``Phi_j(x) = int_j dG/dn_xi dS`` with Capytaine's infinite-
depth ``G = -(1/r + 1/r1 + g_w)/(4 pi)`` (``gf_singularities='low_freq'``). Three parts:
- Rankine ``-1/(4 pi r)``: the exact vortex ring of the panel (``thin_sheet_kernels``).
- Image ``-1/(4 pi r1)``: the exact ring of the panel mirrored in z = 0, with the opposite sign
  (reflection reverses the handedness of the mirrored winding).
- Wave term: analytic second derivatives of Capytaine's point-pair wave term (see
  ``wave_hypersingular``) on the panel's own quadrature points; central differences of
  Capytaine's panel matrices ``D(k) - D(0)`` remain only as a test oracle (they are limited by the
  wave term's integration noise, about 1e-8 per evaluation).
Capytaine integrates its image term with its own panel rule (about 0.4 % from the exact image
for a 0.5 m panel at 4 m in TS1); this route uses the exact image instead. Surface-piercing sheets
(receivers at z = 0) are outside TS1 (TS4). Capytaine is imported lazily.
"""
from __future__ import annotations

import numpy as np

from .thin_sheet_kernels import doublet_panel_velocity, hypersingular_rankine

_MIRROR = np.array([1.0, 1.0, -1.0])
# The wave term depends on (x_h - xi_h, z + zeta): grad_xi = diag(-1, -1, +1) grad_x.
_XI_FROM_X = np.array([-1.0, -1.0, 1.0])


def _capytaine_mesh(vertices, faces, face_sizes):
    import capytaine as cpt
    rows = []
    for face, size in zip(np.asarray(faces), np.asarray(face_sizes)):
        row = [int(i) for i in face[:size]]
        rows.append(row + [row[-1]] * (4 - len(row)))
    return cpt.Mesh(vertices=np.asarray(vertices, dtype=float), faces=np.array(rows))


def _double_layer(gf, points, mesh, source_normals, *, wavenumber):
    """Capytaine's infinite-depth direct double layer ``int dG/dn_xi`` at plain points -> (n, m)."""
    _, K = gf.evaluate(points, mesh, free_surface=0.0, water_depth=np.inf, wavenumber=wavenumber,
                       adjoint_double_layer=False, early_dot_product=False, diagonal_term_in_double_layer=False)
    K = np.asarray(K)
    if K.shape[0] != 3:
        raise ValueError(f'unexpected Capytaine gradient layout {K.shape}')
    return np.einsum('knm,mk->nm', K, source_normals)


def image_hypersingular(points, normals, vertices, faces, face_sizes):
    """Exact image part ``n_i . grad Phi_image``: minus the ring of the mirrored panel."""
    velocity = doublet_panel_velocity(points, np.asarray(vertices, dtype=float) * _MIRROR, faces, face_sizes)
    return -np.einsum('nk,nmk->nm', np.asarray(normals, dtype=float), velocity)


def wave_double_layer(gf, points, vertices, faces, face_sizes, *, wavenumber):
    """Wave part of the double layer ``-(1/4 pi) int (M n_j) . grad_x g_w dS`` at points (n, m), M = diag(-1, -1, 1).

    Built from Capytaine's point-pair wave term on the panel's own quadrature points; used to
    check the analytic operator against Capytaine's panel matrices (D(k) - D(0)).
    """
    g, grad, _, _, weights, mirrored = _wave_terms(gf, points, vertices, faces, face_sizes, wavenumber)
    return -np.einsum('nmq,nmqk,mk->nm', weights, grad, mirrored) / (4.0 * np.pi)


def _wave_terms(gf, points, vertices, faces, face_sizes, wavenumber):
    if not (0.0 < float(wavenumber) < np.inf):
        raise ValueError('the wave part needs a finite positive wavenumber')
    points = np.asarray(points, dtype=float)
    mesh = _capytaine_mesh(vertices, faces, face_sizes)
    qp, qw = mesh.quadrature_points
    qp, qw = np.asarray(qp, dtype=float), np.asarray(qw, dtype=float)
    n, m, nq = points.shape[0], qp.shape[0], qp.shape[1]
    x = np.broadcast_to(points[:, None, None, :], (n, m, nq, 3)).reshape(-1, 3)
    xi = np.broadcast_to(qp[None, :, :, :], (n, m, nq, 3)).reshape(-1, 3)
    singularities = gf.gf_singularities_fortran_enum['low_freq']
    g, grad = gf.fortran_core.interface.vectorized_wave_part_infinite_depth(
        np.ascontiguousarray(x), np.ascontiguousarray(xi), float(wavenumber), *gf.all_tabulation_parameters,
        singularities)
    mirrored = np.asarray(mesh.faces_normals, dtype=float) * _XI_FROM_X    # M n_j: n_j . grad_xi = (M n_j) . grad_x
    weights = np.broadcast_to(qw[None, :, :], (n, m, nq))
    return (np.asarray(g).reshape(n, m, nq), np.asarray(grad).reshape(n, m, nq, 3), x.reshape(n, m, nq, 3),
            xi.reshape(n, m, nq, 3), weights, mirrored)


def wave_hypersingular(gf, points, normals, vertices, faces, face_sizes, *, wavenumber):
    """Wave part of ``H`` analytically: ``-(1/4 pi) int n_i^T Hess_x g_w (M n_j) dS``, M = diag(-1, -1, 1).

    Capytaine's infinite-depth wave term satisfies ``d g/dZ - k g = 2k / r1`` exactly (Z = z + zeta,
    r1 = |x - mirror(xi)|; pinned by tests), so with Laplace's equation every second derivative
    follows from ``g`` and ``grad g`` without differencing:
    ``g_ZZ = k g_Z - 2k Z/r1^3``, ``g_RZ = k g_R - 2k R/r1^3``, ``g_RR = -g_R/R - g_ZZ``.
    """
    k = float(wavenumber)
    g, grad, x, xi, weights, mirrored = _wave_terms(gf, points, vertices, faces, face_sizes, k)
    a = np.asarray(normals, dtype=float)[:, None, None, :]
    b = mirrored[None, :, None, :]
    dh = x[..., :2] - xi[..., :2]
    R = np.linalg.norm(dh, axis=-1)
    Z = x[..., 2] + xi[..., 2]
    r1 = np.sqrt(R * R + Z * Z)
    g_Z = grad[..., 2]
    small = R < 1e-12 * np.maximum(r1, 1e-300)
    e = np.where(small[..., None], 0.0, dh / np.where(small, 1.0, R)[..., None])
    g_R = np.einsum('...k,...k->...', grad[..., :2], e)
    g_ZZ = k * g_Z - 2.0 * k * Z / r1 ** 3
    g_RZ = k * g_R - 2.0 * k * R / r1 ** 3
    g_R_over_R = np.where(small, -0.5 * g_ZZ, g_R / np.where(small, 1.0, R))
    g_RR = np.where(small, -0.5 * g_ZZ, -g_R_over_R - g_ZZ)
    ae, be = np.einsum('...k,...k->...', a[..., :2], e), np.einsum('...k,...k->...', b[..., :2], e)
    ab = np.einsum('...k,...k->...', a[..., :2], b[..., :2])
    contraction = (g_RR * ae * be + g_R_over_R * (ab - ae * be)
                   + g_RZ * (ae * b[..., 2] + a[..., 2] * be) + g_ZZ * a[..., 2] * b[..., 2])
    return -np.einsum('nmq,nmq->nm', weights, contraction) / (4.0 * np.pi)


def wave_hypersingular_fd(gf, points, normals, vertices, faces, face_sizes, *, wavenumber, step):
    """Test oracle: central differences of Capytaine's panel matrices ``D(k) - D(0)``."""
    points = np.asarray(points, dtype=float)
    normals = np.asarray(normals, dtype=float)
    mesh = _capytaine_mesh(vertices, faces, face_sizes)
    source_normals = np.asarray(mesh.faces_normals, dtype=float)
    values = []
    for sign in (+1.0, -1.0):
        shifted = points + sign * step * normals
        values.append(_double_layer(gf, shifted, mesh, source_normals, wavenumber=wavenumber)
                      - _double_layer(gf, shifted, mesh, source_normals, wavenumber=0.0))
    return (values[0] - values[1]) / (2.0 * step)


def sheet_hypersingular(gf, points, normals, vertices, faces, face_sizes, *, wavenumber):
    """Full infinite-depth sheet-row operator: Rankine ring + image ring + wave part."""
    return (hypersingular_rankine(points, normals, vertices, faces, face_sizes)
            + image_hypersingular(points, normals, vertices, faces, face_sizes)
            + wave_hypersingular(gf, points, normals, vertices, faces, face_sizes, wavenumber=wavenumber))
