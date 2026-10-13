# SPDX-License-Identifier: MPL-2.0
"""Thin-sheet (dipole panel) kernels on Capytaine's Green-function convention (plan TS1).

Capytaine's Rankine kernel is ``G = -1/(4 pi r)`` (pinned by tests). A flat panel carrying a
constant potential jump ``m`` (normal from the vertex winding, right-handed) has the double-layer
potential ``Phi(x) = m * int dG/dn_xi dS`` whose gradient off the panel equals the velocity of a
vortex ring on the panel boundary with circulation ``m``. That ring velocity is also the
Hadamard finite part on the panel itself, i.e. the hypersingular Rankine operator of the sheet
equations. Qt-free, Capytaine-free (numpy only).
"""
from __future__ import annotations

import numpy as np

_FOUR_PI = 4.0 * np.pi
# Direction of the equivalent ring relative to the vertex winding, for Capytaine's
# G = -1/(4 pi r); fixed by the off-panel quadrature test, not by derivation alone.
RING_SIGN = 1.0


def _segment_velocity(points, a, b, eps):
    """Biot-Savart velocity at ``points`` (n, 3) of unit-circulation segments a->b (m, 3) -> (n, m, 3)."""
    r1 = points[:, None, :] - a[None, :, :]
    r2 = points[:, None, :] - b[None, :, :]
    r0 = (b - a)[None, :, :]
    cross = np.cross(r1, r2)
    cross2 = np.einsum('nmk,nmk->nm', cross, cross)
    n1 = np.linalg.norm(r1, axis=2)
    n2 = np.linalg.norm(r2, axis=2)
    scale = np.einsum('nmk,nmk->nm', r0, r1 / np.maximum(n1, eps)[..., None] - r2 / np.maximum(n2, eps)[..., None])
    length2 = np.einsum('nmk,nmk->nm', r0, r0)
    # A point on the segment's line (|r1 x r2| ~ 0) gets no contribution (collocation points are
    # panel centroids, never on edges).
    on_line = cross2 <= (eps * eps) * np.maximum(length2, eps) ** 2
    factor = np.where(on_line, 0.0, scale / np.where(on_line, 1.0, cross2)) / _FOUR_PI
    return cross * factor[..., None]


def doublet_panel_velocity(points, vertices, faces, face_sizes, *, eps=1e-12):
    """Velocity ``grad Phi`` at ``points`` of unit-jump doublet panels -> (n_points, n_panels, 3).

    ``faces`` is the padded (n_panels, width) index array (-1 padding), ``face_sizes`` the vertex
    counts; the panel normal follows the vertex order (right-handed). Capytaine sign convention:
    ``Phi = int dG/dn_xi dS`` with ``G = -1/(4 pi r)``.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces)
    sizes = np.asarray(face_sizes)
    out = np.zeros((points.shape[0], faces.shape[0], 3))
    for k in range(int(sizes.max())):
        active = sizes > k
        rows = np.flatnonzero(active)
        start = vertices[faces[rows, k]]
        end = vertices[faces[rows, np.where(k + 1 < sizes[rows], k + 1, 0)]]
        out[:, rows, :] += RING_SIGN * _segment_velocity(points, start, end, eps)
    return out


def hypersingular_rankine(points, normals, vertices, faces, face_sizes):
    """``n_i . grad Phi_j(x_i)`` (n_points, n_panels): the Rankine sheet-row operator."""
    velocity = doublet_panel_velocity(points, vertices, faces, face_sizes)
    return np.einsum('nk,nmk->nm', np.asarray(normals, dtype=float), velocity)
