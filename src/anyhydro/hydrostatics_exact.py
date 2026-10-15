"""Exact immersed-volume moments and buoyancy baseline wrench of a panel mesh (ADR-015 section 3, slice L6-4 H-a).

The wet region is bounded by the immersed hull surface and the free-surface plane z = 0. With the divergence
theorem and integrands that vanish on z = 0 (so the waterplane lid contributes nothing):

    V           = int_S z n_z dS
    int x dV    = int_S x z n_z dS,   int y dV = int_S y z n_z dS,   int z dV = int_S (z^2 / 2) n_z dS

Every integrand is at most quadratic, so the edge-midpoint rule on each (planar) triangle is exact: the moments are
exact for the mesh geometry (no panel quadrature error). The mesh must be the immersed part (no vertex above z = 0).
"""
from __future__ import annotations

import numpy as np

from .errors import HydroError

#: tolerance for "no vertex above the free surface" (m)
ABOVE_SURFACE_TOLERANCE = 1.0e-9


class HydrostaticsExactError(HydroError):
    """The mesh cannot provide exact immersed-volume moments."""


def _triangles(vertices: np.ndarray, faces: np.ndarray):
    for face in faces:
        idx = [int(i) for i in face if int(i) >= 0]
        if len(idx) == 4 and idx[2] == idx[3]:
            idx = idx[:3]
        if len(idx) == 3:
            yield vertices[idx]
        elif len(idx) == 4:
            yield vertices[[idx[0], idx[1], idx[2]]]
            yield vertices[[idx[0], idx[2], idx[3]]]
        else:
            raise HydrostaticsExactError(f"faces must be triangles or quadrilaterals, got {len(idx)} vertices")


def immersed_volume_moments(vertices, faces) -> tuple[float, np.ndarray]:
    """(V, [int x dV, int y dV, int z dV]) of the region below z = 0 bounded by the outward-oriented wet surface."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or not np.all(np.isfinite(v)):
        raise HydrostaticsExactError("vertices must be finite (n, 3)")
    if np.max(v[:, 2]) > ABOVE_SURFACE_TOLERANCE:
        raise HydrostaticsExactError("the mesh has vertices above the free surface z = 0; pass the immersed part")
    volume = 0.0
    moments = np.zeros(3)
    for a, b, c in _triangles(v, f):
        nz_area = 0.5 * float(np.cross(b - a, c - a)[2])          # n_z dS for the whole triangle
        mids = np.array([(a + b) / 2.0, (b + c) / 2.0, (c + a) / 2.0])
        x, y, z = mids[:, 0], mids[:, 1], mids[:, 2]
        volume += nz_area * float(np.mean(z))
        moments += nz_area * np.array([np.mean(x * z), np.mean(y * z), np.mean(0.5 * z * z)])
    if not volume > 0.0:
        raise HydrostaticsExactError(f"the immersed volume is not positive ({volume:g}); check the normals")
    return volume, moments


def buoyancy_baseline_wrench(vertices, faces, rotation_center, rho: float, gravity: float) -> tuple[float, ...]:
    """Measured baseline buoyancy wrench (F_x, F_y, F_z, M_x, M_y, M_z) in world axes, moment about the rotation
    centre: F = rho g V e_z at the centre of buoyancy."""
    volume, moments = immersed_volume_moments(vertices, faces)
    centre_of_buoyancy = moments / volume
    force = np.array([0.0, 0.0, float(rho) * float(gravity) * volume])
    moment = np.cross(centre_of_buoyancy - np.asarray(rotation_center, dtype=np.float64), force)
    return tuple(float(x) for x in np.concatenate([force, moment]))
