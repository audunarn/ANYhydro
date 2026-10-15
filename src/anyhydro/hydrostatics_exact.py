"""Exact immersed-volume moments and buoyancy baseline wrench of a panel mesh (ADR-015 section 3, slice L6-4 H-a).

The wet region is bounded by the immersed hull surface and the free-surface plane z = 0. With the divergence
theorem and integrands that vanish on z = 0 (so the waterplane lid contributes nothing):

    V           = int_S z n_z dS
    int x dV    = int_S x z n_z dS,   int y dV = int_S y z n_z dS,   int z dV = int_S (z^2 / 2) n_z dS

Every integrand is at most quadratic, so the edge-midpoint rule on each (planar) triangle is exact: the moments are
exact for the mesh geometry (no panel quadrature error). The mesh must be the immersed part (no vertex above z = 0),
closed by the free surface (validated by the closure identities int x n_x = int y n_y = int z n_z = V and zero
horizontal vector area; no face in z = 0; planar quadrilaterals).
"""
from __future__ import annotations

import numpy as np

from .errors import HydroError

#: tolerance for "no vertex above the free surface" (m) and for "on the free surface"
ABOVE_SURFACE_TOLERANCE = 1.0e-9
#: quadrilaterals must be planar to this relative tolerance (the moments are exact for the triangulated surface)
PLANARITY_TOLERANCE = 1.0e-9


class HydrostaticsExactError(HydroError):
    """The mesh cannot provide exact immersed-volume moments."""


def _polygons(faces):
    for face in faces:
        idx = [int(i) for i in face if int(i) >= 0]
        if len(idx) == 4 and idx[2] == idx[3]:
            idx = idx[:3]
        if len(idx) not in (3, 4):
            raise HydrostaticsExactError(f"faces must be triangles or quadrilaterals, got {len(idx)} vertices")
        yield idx


def _extent(vertices: np.ndarray) -> float:
    """Translation-invariant geometry length scale: the largest bounding-box side."""
    return max(float(np.max(np.ptp(vertices, axis=0))), 1e-300)


def _validate_boundary(vertices: np.ndarray, faces) -> None:
    """Face-level checks: no face may lie in the free surface (a waterplane cap is an auxiliary closure, not wetted
    hull) and quadrilaterals must be planar (relative to the geometry extent, translation invariant)."""
    length = _extent(vertices)
    for idx in _polygons(faces):
        poly = vertices[idx]
        if np.all(np.abs(poly[:, 2]) <= ABOVE_SURFACE_TOLERANCE * max(length, 1.0)):
            raise HydrostaticsExactError("a face lies in the free surface z = 0 (waterplane closure/cap); the "
                                         "baseline needs the wetted hull only")
        if len(idx) == 4:
            normal = np.cross(poly[2] - poly[0], poly[3] - poly[1])
            norm = float(np.linalg.norm(normal))
            if norm > 0 and abs(float(np.dot(normal / norm, poly[1] - poly[0]))) > PLANARITY_TOLERANCE * length:
                raise HydrostaticsExactError("a quadrilateral is not planar; split it into triangles")


#: relative tolerance of the closure identities and of the pressure-vs-volume agreement
CLOSURE_TOLERANCE = 1.0e-9
# degree-3 exact triangle rule (Strang-Fix): barycentric points and weights (sum 1)
_BARY = np.array([[1 / 3, 1 / 3, 1 / 3], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2], [0.2, 0.2, 0.6]])
_WEIGHTS = np.array([-27 / 48, 25 / 48, 25 / 48, 25 / 48])


def pressure_wrench_unit(vertices, faces, reference) -> np.ndarray:
    """Hydrostatic pressure wrench on the submitted panels per unit rho g, about ``reference``:
    F = int z n dS, M = int z (x - r) x n dS (outward normals, p = -rho g z). The integrands are at most cubic, so
    the degree-3 triangle rule is exact. This is what the wetted panels actually feel, whatever the topology."""
    v = np.asarray(vertices, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    out = np.zeros(6)
    for a, b, c in _triangles(v, np.asarray(faces)):
        n_area = 0.5 * np.cross(b - a, c - a)
        points = _BARY @ np.array([a, b, c])
        z = points[:, 2]
        force = float(np.dot(_WEIGHTS, z)) * n_area
        moment = np.cross(np.einsum("q,qi->i", _WEIGHTS * z, points - r), n_area)
        out += np.concatenate([force, moment])
    return out


def _triangles(vertices: np.ndarray, faces: np.ndarray):
    for idx in _polygons(faces):
        if len(idx) == 3:
            yield vertices[idx]
        else:
            yield vertices[[idx[0], idx[1], idx[2]]]
            yield vertices[[idx[0], idx[2], idx[3]]]


def immersed_volume_moments(vertices, faces) -> tuple[float, np.ndarray]:
    """(V, [int x dV, int y dV, int z dV]) of the region below z = 0 bounded by the outward-oriented wet surface."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3 or not np.all(np.isfinite(v)):
        raise HydrostaticsExactError("vertices must be finite (n, 3)")
    if np.max(v[:, 2]) > ABOVE_SURFACE_TOLERANCE:
        raise HydrostaticsExactError("the mesh has vertices above the free surface z = 0; pass the immersed part")
    _validate_boundary(v, f)
    volume = volume_x = volume_y = area_x = area_y = total_area = 0.0
    moments = np.zeros(3)
    for a, b, c in _triangles(v, f):
        n_area = 0.5 * np.cross(b - a, c - a)                    # n dS for the whole triangle
        nz_area = float(n_area[2])
        mids = np.array([(a + b) / 2.0, (b + c) / 2.0, (c + a) / 2.0])
        x, y, z = mids[:, 0], mids[:, 1], mids[:, 2]
        volume += nz_area * float(np.mean(z))
        volume_x += float(n_area[0]) * float(np.mean(x))
        volume_y += float(n_area[1]) * float(np.mean(y))
        area_x += float(n_area[0])
        area_y += float(n_area[1])
        total_area += float(np.linalg.norm(n_area))
        moments += nz_area * np.array([np.mean(x * z), np.mean(y * z), np.mean(0.5 * z * z)])
    length = _extent(v)
    tol = CLOSURE_TOLERANCE * max(total_area * length, 1e-300)
    if (abs(volume_x - volume) > tol or abs(volume_y - volume) > tol
            or abs(area_x) > CLOSURE_TOLERANCE * total_area or abs(area_y) > CLOSURE_TOLERANCE * total_area):
        raise HydrostaticsExactError(
            "the wet surface is not closed by the free surface (an opening below z = 0 or inconsistent face "
            f"orientation): volume by x/y/z {volume_x:.12g} / {volume_y:.12g} / {volume:.12g}, horizontal vector "
            f"area ({area_x:.3g}, {area_y:.3g})")
    # sufficient for the baseline: the pressure actually acting on the submitted panels must equal the volume form
    # rho g V e_z at the centre of buoyancy (any opening/defect that changes the panels' load is refused)
    if volume > 0:
        centroid = moments / volume
        pressure = pressure_wrench_unit(v, f, centroid)
        if (np.max(np.abs(pressure[:2])) > CLOSURE_TOLERANCE * total_area * length
                or abs(pressure[2] - volume) > CLOSURE_TOLERANCE * total_area * length
                or np.max(np.abs(pressure[3:])) > CLOSURE_TOLERANCE * total_area * length * length):
            raise HydrostaticsExactError(
                "the pressure load on the submitted panels is not the buoyancy of the enclosed volume "
                f"(force/rho g {pressure[:3].tolist()}, moment about the centroid {pressure[3:].tolist()}, "
                f"volume {volume:.12g}): the wet surface is not closed by the free surface")
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
