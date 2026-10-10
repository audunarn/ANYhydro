"""Deterministic first-slice two-body wetted-hull/lid development fixture.

This is a DEVELOPMENT fixture for the bounded first-slice BEM preparation:
it confers no scientific qualification. It is physically specified so the
analytical checks below are exact in floating point:

- Two upright boxes, 1 m x 1 m waterplane, 0.5 m draft, hull = bottom +
  four sides over z in [-0.5, 0] (no dry top, no above-water hull).
- Each of the five rectangular wetted faces is subdivided n x n into
  quads (public keyword ``subdivisions=n``, default 2), each quad into 2
  triangles: 10n^2 hull panels per body; the waterplane lid covers the
  1 m square at z=0 with 2n^2 panels per body, normals DOWN (the pinned
  Capytaine 3.0.0 convention: ``FloatingBody.__init__`` forces
  ``with_normal_vector_going_down`` and ``Mesh.generate_lid`` uses
  ``normal=(0, 0, -1)``). Total: 24n^2 panels for the two bodies
  (96 at the default n=2, which reproduces the historical fixture
  bit-for-bit). Vertices are shared across faces by exact rational
  structured-grid keys, so non-binary levels (e.g. n=6) have the exact
  known topology without any coordinate rounding.
- Submerged volume 0.5 m^3 per body (divergence theorem is exact because
  the missing top lies at z=0 where its contribution is zero), center of
  buoyancy at (cx, 0, -0.25), waterplane area exactly 1.0 m^2.

Case constants (fixture/evidence metadata only — dry physics such as the
mechanical surge link and dry inertia are owned separately by the
mechanical case contract; this module adds no gravity or inertia to any
backend): rho 1025 kg/m^3, g 9.81 m/s^2, water depth 150 m, dry mass
512.5 kg per body (equilibrium: weight = buoyancy = 5027.625 N), analytical
heave buoyancy stiffness rho*g*A_wp = 10055.25 N/m per body. The first
slice constrains surge/heave only (matched FD/TD projections are defined
downstream); the fixture itself carries all 6 DOFs per body so the full
12x12 coupled space is solved and retained unmodified.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, Tuple

import numpy as np

from .dof import body_dof_keys
from .errors import GeometryValidationError
from .records import HydroBody, HydroPanelMesh, HydroRequest

__all__ = [
    "FIRST_SLICE_BODY_IDS",
    "FIRST_SLICE_ROTATION_CENTERS",
    "FIRST_SLICE_RHO",
    "FIRST_SLICE_GRAVITY",
    "FIRST_SLICE_WATER_DEPTH",
    "FIRST_SLICE_DRY_MASS",
    "FIRST_SLICE_REFERENCE_WEIGHT",
    "FIRST_SLICE_HEAVE_BUOYANCY_STIFFNESS",
    "FIRST_SLICE_OMEGAS",
    "FIRST_SLICE_WAVE_DIRECTIONS",
    "FIRST_SLICE_PROJECTED_DOFS",
    "FIRST_SLICE_MIN_SUBDIVISIONS",
    "FIRST_SLICE_MAX_SUBDIVISIONS",
    "make_first_slice_bodies",
    "make_first_slice_request",
    "verify_first_slice_fixture",
    "first_slice_case_metadata",
]

FIRST_SLICE_BODY_IDS = ("first-slice-a", "first-slice-b")
FIRST_SLICE_ROTATION_CENTERS = ((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0))
#: Center of mass relative to the body reference point: (0, 0, -0.25).
FIRST_SLICE_COM_RELATIVE = (0.0, 0.0, -0.25)

# Geometry (metres): upright 1 m x 1 m box, 0.5 m draft.
_BOX_HALF_WIDTH = 0.5
_DRAFT = 0.5
_SUBDIVISIONS = 2  # default 2x2 quads per rectangular face
#: Bounded deterministic refinement levels (public keyword ``subdivisions``).
FIRST_SLICE_MIN_SUBDIVISIONS = 1
FIRST_SLICE_MAX_SUBDIVISIONS = 16

# Case constants.
FIRST_SLICE_RHO = 1025.0
FIRST_SLICE_GRAVITY = 9.81
FIRST_SLICE_WATER_DEPTH = 150.0
FIRST_SLICE_DRY_MASS = 512.5  # kg per body: rho * submerged volume
FIRST_SLICE_REFERENCE_WEIGHT = FIRST_SLICE_DRY_MASS * FIRST_SLICE_GRAVITY
#: rho * g * waterplane area (1 m^2) = 1025 * 9.81 = 10055.25 N/m.
FIRST_SLICE_HEAVE_BUOYANCY_STIFFNESS = (
    FIRST_SLICE_RHO * FIRST_SLICE_GRAVITY * (2.0 * _BOX_HALF_WIDTH) ** 2
)
FIRST_SLICE_OMEGAS = (2.0, 4.0, 6.0)
FIRST_SLICE_WAVE_DIRECTIONS = (0.0,)
#: First-slice constrained projection (metadata only; enforced downstream).
FIRST_SLICE_PROJECTED_DOFS = (
    "first-slice-a::surge",
    "first-slice-a::heave",
    "first-slice-b::surge",
    "first-slice-b::heave",
)

_SUBMERGED_VOLUME = 0.5
_COB_Z = -0.25
_HULL_AREA = 3.0  # bottom 1 m^2 + four sides 0.5 m^2
_LID_AREA = 1.0

FIRST_SLICE_FREE_SURFACE_LID_POLICY = "free_surface_v1"
FIRST_SLICE_SUBMERGED_LID_POLICY = "submerged_0p1m_v1"

def first_slice_lid_elevation(policy: str) -> float:
    """Fixed auxiliary numerical recipe in metres; no physical sheet or mass."""
    if policy == FIRST_SLICE_FREE_SURFACE_LID_POLICY:
        return 0.0
    if policy == FIRST_SLICE_SUBMERGED_LID_POLICY:
        return -0.1
    raise GeometryValidationError(f"unsupported auxiliary lid policy {policy!r}")

_GEOMETRY_TOLERANCE = 1e-12


def _validate_subdivisions(value) -> int:
    """Typed admission of a refinement level, before any allocation.

    Refuses bools, non-integers, fractional floats, non-finite values and
    levels outside ``[FIRST_SLICE_MIN_SUBDIVISIONS,
    FIRST_SLICE_MAX_SUBDIVISIONS]``.
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise GeometryValidationError(
            f"subdivisions must be an integer in "
            f"[{FIRST_SLICE_MIN_SUBDIVISIONS}, "
            f"{FIRST_SLICE_MAX_SUBDIVISIONS}] (got {type(value).__name__}: "
            f"{value!r}); bools, floats and non-finite levels are refused"
        )
    n = int(value)
    if n < FIRST_SLICE_MIN_SUBDIVISIONS or n > FIRST_SLICE_MAX_SUBDIVISIONS:
        raise GeometryValidationError(
            f"subdivisions must be an integer in "
            f"[{FIRST_SLICE_MIN_SUBDIVISIONS}, "
            f"{FIRST_SLICE_MAX_SUBDIVISIONS}] (got {n})"
        )
    return n


def _expected_counts(n: int) -> Dict[str, int]:
    """Exact panel/vertex counts of the refined fixture at level ``n``."""
    return {
        "hull_panels_per_body": 10 * n * n,
        "lid_panels_per_body": 2 * n * n,
        "hull_vertices_per_body": 5 * n * n + 2 * n + 1,
        "lid_vertices_per_body": (n + 1) * (n + 1),
        "total_panels": 24 * n * n,
    }


def _face_specs(cx: float):
    """The five wetted hull faces as (origin, u, v) grids with u x v = n.

    ``u`` and ``v`` are the FULL edge vectors of each rectangular face, so
    the grid p(i, j) = origin + (i/2) u + (j/2) v spans the face exactly.
    Fixed deterministic order: bottom, x-min, x-max, y-min, y-max. The
    expected outward normal is u x v by construction.
    """
    h = _BOX_HALF_WIDTH
    d = _DRAFT
    return (
        # bottom (z = -0.5), outward normal (0, 0, -1)
        ((cx - h, h, -d), (2.0 * h, 0.0, 0.0), (0.0, -2.0 * h, 0.0)),
        # x-min side, outward normal (-1, 0, 0)
        ((cx - h, -h, 0.0), (0.0, 2.0 * h, 0.0), (0.0, 0.0, -d)),
        # x-max side, outward normal (+1, 0, 0)
        ((cx + h, -h, -d), (0.0, 2.0 * h, 0.0), (0.0, 0.0, d)),
        # y-min side, outward normal (0, -1, 0)
        ((cx - h, -h, 0.0), (0.0, 0.0, -d), (2.0 * h, 0.0, 0.0)),
        # y-max side, outward normal (0, +1, 0)
        ((cx - h, h, 0.0), (2.0 * h, 0.0, 0.0), (0.0, 0.0, -d)),
    )


def _lid_spec(cx: float, elevation: float = 0.0):
    """The waterplane lid grid: 1 m square at z=0, normals DOWN."""
    h = _BOX_HALF_WIDTH
    # u x v = (1,0,0) x (0,-1,0) = (0, 0, -1): downward by construction.
    return ((cx - h, h, elevation), (2.0 * h, 0.0, 0.0), (0.0, -2.0 * h, 0.0))


def _build_mesh(body_id: str, specs, *, mesh_role: str, subdivisions: int):
    """Deterministic triangulated mesh from (origin, u, v) face specs.

    Vertices are shared where the structured grids coincide: each grid
    point is keyed by its EXACT rational coordinates
    ``origin + (i/n) u + (j/n) v`` (numerator/denominator integer keys,
    no coordinate rounding or snapping), so faces share vertices at
    non-binary levels (e.g. n=6) exactly as the known structured-grid
    topology dictates. Each face grid is subdivided n x n into quads;
    each quad becomes 2 triangles wound so the cross-product normal is
    u x v. The default n=2 reproduces the historical arrays bit-for-bit.
    """
    pool: Dict[Tuple[Fraction, Fraction, Fraction], int] = {}
    vertices: list = []
    faces: list = []
    n = subdivisions
    for origin, u, v in specs:
        ox, oy, oz = origin
        ux, uy, uz = u
        vx, vy, vz = v
        # Exact rational anchors: Fraction(float) is the exact binary
        # value, Fraction(i, n) the exact subdivision fraction.
        fox, foy, foz = Fraction(ox), Fraction(oy), Fraction(oz)
        fux, fuy, fuz = Fraction(ux), Fraction(uy), Fraction(uz)
        fvx, fvy, fvz = Fraction(vx), Fraction(vy), Fraction(vz)
        grid = [
            [
                (
                    ox + (i / n) * ux + (j / n) * vx,
                    oy + (i / n) * uy + (j / n) * vy,
                    oz + (i / n) * uz + (j / n) * vz,
                )
                for j in range(n + 1)
            ]
            for i in range(n + 1)
        ]
        keys = [
            [
                (
                    fox + Fraction(i, n) * fux + Fraction(j, n) * fvx,
                    foy + Fraction(i, n) * fuy + Fraction(j, n) * fvy,
                    foz + Fraction(i, n) * fuz + Fraction(j, n) * fvz,
                )
                for j in range(n + 1)
            ]
            for i in range(n + 1)
        ]
        for i in range(n):
            for j in range(n):
                c00 = (grid[i][j], keys[i][j])
                c01 = (grid[i][j + 1], keys[i][j + 1])
                c11 = (grid[i + 1][j + 1], keys[i + 1][j + 1])
                c10 = (grid[i + 1][j], keys[i + 1][j])
                # wound so the cross-product normal is u x v: traverse the
                # u edge first in both triangles of the quad
                for tri in ((c00, c10, c11), (c00, c11, c01)):
                    idx = []
                    for point, key in tri:
                        if key not in pool:
                            pool[key] = len(vertices)
                            vertices.append(
                                (
                                    float(point[0]),
                                    float(point[1]),
                                    float(point[2]),
                                )
                            )
                        idx.append(pool[key])
                    faces.append(idx)
    vertices_arr = np.array(vertices, dtype=np.float64)
    faces_arr = np.array(faces, dtype=np.int32)
    sizes = np.full(len(faces), 3, dtype=np.uint8)
    return HydroPanelMesh(
        vertices=vertices_arr,
        faces=faces_arr,
        face_sizes=sizes,
        body_id=body_id,
        # Hull normals are analytically outward (verified below). Lid
        # normals are deliberately DOWNWARD per the pinned capytaine lid
        # convention, i.e. NOT outward from the enclosed volume, so the
        # lid honestly reports outward_normals_verified=False; its
        # downward orientation is certified by verify_first_slice_fixture
        # and enforced geometrically by the backend at admission.
        outward_normals_verified=(mesh_role == "hull"),
        source_geometry_refs=(
            f"first-slice-wetted-box-{mesh_role}",
            "first-slice-preparation-fixture-v1",
        ),
        mesh_hash="",
    )


def make_first_slice_bodies(
    *, subdivisions: int = _SUBDIVISIONS,
    lid_policy: str = FIRST_SLICE_FREE_SURFACE_LID_POLICY
) -> Tuple[HydroBody, HydroBody]:
    """The deterministic two-body fixture (hull + downward lid each).

    ``subdivisions=n`` refines every rectangular face into an n x n quad
    grid (hull 10n^2 panels per body, lid 2n^2, 24n^2 total for the two
    bodies). The default ``n=2`` reproduces the historical 96-panel
    fixture bit-for-bit. The level is validated before any allocation.
    """
    n = _validate_subdivisions(subdivisions)
    elevation = first_slice_lid_elevation(lid_policy)
    bodies = []
    for body_id, ref in zip(FIRST_SLICE_BODY_IDS, FIRST_SLICE_ROTATION_CENTERS):
        cx = float(ref[0])
        hull = _build_mesh(body_id, _face_specs(cx), mesh_role="hull", subdivisions=n)
        lid = _build_mesh(body_id, (_lid_spec(cx, elevation),), mesh_role="lid", subdivisions=n)
        com = (
            ref[0] + FIRST_SLICE_COM_RELATIVE[0],
            ref[1] + FIRST_SLICE_COM_RELATIVE[1],
            ref[2] + FIRST_SLICE_COM_RELATIVE[2],
        )
        bodies.append(
            HydroBody(
                body_id=body_id,
                name=f"first-slice wetted box {body_id}",
                mesh=hull,
                lid_mesh=lid,
                rotation_center=tuple(float(v) for v in ref),
                center_of_mass=com,
                enabled_dofs=body_dof_keys(body_id),
            )
        )
    return (bodies[0], bodies[1])


def make_first_slice_request(
    *, include_infinite_frequency: bool = True
) -> HydroRequest:
    """The first-slice pilot request: [2, 4, 6] rad/s, heading 0."""
    return HydroRequest(
        omegas=np.array(FIRST_SLICE_OMEGAS, dtype=np.float64),
        wave_directions=np.array(FIRST_SLICE_WAVE_DIRECTIONS, dtype=np.float64),
        water_depth=FIRST_SLICE_WATER_DEPTH,
        rho=FIRST_SLICE_RHO,
        gravity=FIRST_SLICE_GRAVITY,
        include_zero_frequency=False,
        include_infinite_frequency=include_infinite_frequency,
        backend_settings={},
    )


def _mesh_metrics(mesh: HydroPanelMesh) -> Dict[str, Any]:
    """Per-face areas, unit normals and analytic divergence-theorem integrals."""
    v = mesh.vertices
    f = mesh.faces
    n_faces = f.shape[0]
    areas = np.empty(n_faces)
    unit_normals = np.empty((n_faces, 3))
    for i in range(n_faces):
        p0, p1, p2 = v[f[i, 0]], v[f[i, 1]], v[f[i, 2]]
        cross = np.cross(p1 - p0, p2 - p0)
        twice = float(np.linalg.norm(cross))
        areas[i] = 0.5 * twice
        unit_normals[i] = cross / twice
    # Divergence-theorem volume/COB integrals over the (open) hull: the
    # missing top at z=0 contributes exactly zero to every integral.
    volume = float(np.mean([np.sum(v[f[:, k], k] * unit_normals[:, k] * areas) for k in range(3)]))
    # COB is only meaningful for a closed-ish surface with nonzero volume
    # (a flat lid has zero volume and no COB).
    cob = (
        [
            float(np.sum(unit_normals[:, k] * v[f[:, k], k] ** 2 * areas))
            / (2.0 * volume)
            for k in range(3)
        ]
        if volume != 0.0
        else None
    )
    waterplane_area = float(-np.sum(unit_normals[:, 2] * areas))
    return {
        "areas": areas,
        "unit_normals": unit_normals,
        "total_area": float(np.sum(areas)),
        "volume": volume,
        "center_of_buoyancy": cob,
        "waterplane_area": waterplane_area,
    }


def verify_first_slice_fixture(
    bodies, *, subdivisions: int = _SUBDIVISIONS,
    lid_policy: str = FIRST_SLICE_FREE_SURFACE_LID_POLICY
) -> Dict[str, Any]:
    """Analytic verification of the fixture; raises on any mismatch.

    Validates the exact physical fixture at refinement level
    ``subdivisions``: body identity, face roles, outward hull normals,
    downward lid normals, panel/vertex counts, areas, volume, COB and
    waterplane area — not just counts. Returns a JSON-serializable report
    (per-body and total). Every check is analytic: axis-aligned faces make
    areas, volume, COB and waterplane area exact in floating point.
    """
    n = _validate_subdivisions(subdivisions)
    elevation = first_slice_lid_elevation(lid_policy)
    expected = _expected_counts(n)
    if tuple(b.body_id for b in bodies) != FIRST_SLICE_BODY_IDS:
        raise GeometryValidationError(
            f"fixture body ids must be {FIRST_SLICE_BODY_IDS}, got "
            f"{tuple(b.body_id for b in bodies)}"
        )
    report: Dict[str, Any] = {
        "bodies": {},
        "totals": {},
        "subdivisions": n,
        "case_constants": first_slice_case_metadata(),
    }
    total_panels = 0
    for b in bodies:
        cx = float(b.rotation_center[0])
        interior_center = np.array([cx, 0.0, -0.25])
        hull = _mesh_metrics(b.mesh)
        lid = _mesh_metrics(b.lid_mesh)

        if b.mesh.vertices.shape[0] != expected["hull_vertices_per_body"]:
            raise GeometryValidationError(
                f"body {b.body_id!r}: expected "
                f"{expected['hull_vertices_per_body']} shared hull vertices "
                f"at subdivisions={n}, got "
                f"{b.mesh.vertices.shape[0]}"
            )
        if b.mesh.faces.shape[0] != expected["hull_panels_per_body"]:
            raise GeometryValidationError(
                f"body {b.body_id!r}: expected "
                f"{expected['hull_panels_per_body']} hull panels at "
                f"subdivisions={n}, got {b.mesh.faces.shape[0]}"
            )
        if b.lid_mesh is None or b.lid_mesh.faces.shape[0] != expected["lid_panels_per_body"]:
            raise GeometryValidationError(
                f"body {b.body_id!r}: expected "
                f"{expected['lid_panels_per_body']} lid panels at "
                f"subdivisions={n}"
            )
        if b.lid_mesh.vertices.shape[0] != expected["lid_vertices_per_body"]:
            raise GeometryValidationError(
                f"body {b.body_id!r}: expected "
                f"{expected['lid_vertices_per_body']} lid vertices at "
                f"subdivisions={n}, got "
                f"{b.lid_mesh.vertices.shape[0]}"
            )
        if not b.mesh.outward_normals_verified:
            raise GeometryValidationError(
                f"body {b.body_id!r}: hull outward normals must be verified"
            )
        if b.lid_mesh.outward_normals_verified:
            raise GeometryValidationError(
                f"body {b.body_id!r}: lid normals are deliberately downward "
                "(pinned capytaine convention), not outward; "
                "outward_normals_verified must be False for the lid"
            )

        # Hull normals outward: every face normal points away from the
        # body interior reference point.
        centers = b.mesh.vertices[b.mesh.faces].mean(axis=1)
        outward = np.sum(
            (centers - interior_center[None, :]) * hull["unit_normals"], axis=1
        )
        if np.any(outward <= _GEOMETRY_TOLERANCE):
            raise GeometryValidationError(
                f"body {b.body_id!r}: hull panel normals are not outward"
            )
        # Lid normals strictly DOWN.
        if np.any(lid["unit_normals"][:, 2] >= -_GEOMETRY_TOLERANCE):
            raise GeometryValidationError(
                f"body {b.body_id!r}: lid panel normals do not point DOWN"
            )
        # No dry top / above-water hull: all hull vertices at z <= 0, all
        # hull panel centroids strictly below the free surface.
        if np.any(b.mesh.vertices[:, 2] > _GEOMETRY_TOLERANCE):
            raise GeometryValidationError(
                f"body {b.body_id!r}: hull has vertices above the free surface"
            )
        if np.any(centers[:, 2] >= -_GEOMETRY_TOLERANCE):
            raise GeometryValidationError(
                f"body {b.body_id!r}: hull panel centroid at/above z=0"
            )
        if np.any(np.abs(b.lid_mesh.vertices[:, 2] - elevation) > _GEOMETRY_TOLERANCE):
            raise GeometryValidationError(
                f"body {b.body_id!r}: lid vertices must lie at z={elevation} m"
            )

        # Areas, volume, COB, waterplane area: analytic, exact.
        if abs(hull["total_area"] - _HULL_AREA) > 1e-9:
            raise GeometryValidationError(
                f"body {b.body_id!r}: hull area {hull['total_area']} != "
                f"{_HULL_AREA}"
            )
        if abs(lid["total_area"] - _LID_AREA) > 1e-9:
            raise GeometryValidationError(
                f"body {b.body_id!r}: lid area {lid['total_area']} != "
                f"{_LID_AREA}"
            )
        if abs(hull["volume"] - _SUBMERGED_VOLUME) > 1e-9:
            raise GeometryValidationError(
                f"body {b.body_id!r}: submerged volume "
                f"{hull['volume']} != {_SUBMERGED_VOLUME}"
            )
        expected_cob = (cx, 0.0, _COB_Z)
        if not np.allclose(
            hull["center_of_buoyancy"], expected_cob, rtol=0.0, atol=1e-9
        ):
            raise GeometryValidationError(
                f"body {b.body_id!r}: COB "
                f"{hull['center_of_buoyancy']} != {expected_cob}"
            )
        if abs(hull["waterplane_area"] - 1.0) > 1e-9:
            raise GeometryValidationError(
                f"body {b.body_id!r}: waterplane area "
                f"{hull['waterplane_area']} != 1.0"
            )

        total_panels += int(b.mesh.faces.shape[0] + b.lid_mesh.faces.shape[0])
        report["bodies"][b.body_id] = {
            "rotation_center": [float(x) for x in b.rotation_center],
            "center_of_mass": [float(x) for x in b.center_of_mass],
            "hull_mesh_hash": b.mesh.mesh_hash,
            "lid_mesh_hash": b.lid_mesh.mesh_hash,
            "hull_panels": int(b.mesh.faces.shape[0]),
            "lid_panels": int(b.lid_mesh.faces.shape[0]),
            "hull_vertices": int(b.mesh.vertices.shape[0]),
            "lid_vertices": int(b.lid_mesh.vertices.shape[0]),
            "hull_area_m2": hull["total_area"],
            "lid_area_m2": lid["total_area"],
            "submerged_volume_m3": hull["volume"],
            "center_of_buoyancy_m": hull["center_of_buoyancy"],
            "waterplane_area_m2": hull["waterplane_area"],
            "hull_normals": "outward (verified)",
            "lid_normals": "down (verified, pinned capytaine convention)",
            "enabled_dofs": list(b.enabled_dofs),
        }
    if total_panels != expected["total_panels"]:
        raise GeometryValidationError(
            f"fixture total must be {expected['total_panels']} panels at "
            f"subdivisions={n}, got {total_panels}"
        )
    report["totals"] = {
        "panels": total_panels,
        "subdivisions": n,
        "hull_panels_per_body": expected["hull_panels_per_body"],
        "lid_panels_per_body": expected["lid_panels_per_body"],
        "hull_vertices_per_body": expected["hull_vertices_per_body"],
        "lid_vertices_per_body": expected["lid_vertices_per_body"],
    }
    if lid_policy != FIRST_SLICE_FREE_SURFACE_LID_POLICY:
        report["auxiliary_lid"] = dict(policy=lid_policy, elevation_m=elevation,
                                       role="numerical_irregular_frequency_boundary")
    return report


def first_slice_case_metadata() -> Dict[str, Any]:
    """JSON-serializable case constants (metadata only, no dry physics)."""
    return {
        "rho_kg_m3": FIRST_SLICE_RHO,
        "gravity_m_s2": FIRST_SLICE_GRAVITY,
        "water_depth_m": FIRST_SLICE_WATER_DEPTH,
        "dry_mass_kg_per_body": FIRST_SLICE_DRY_MASS,
        "reference_weight_n_per_body": FIRST_SLICE_REFERENCE_WEIGHT,
        "heave_buoyancy_stiffness_n_per_m": FIRST_SLICE_HEAVE_BUOYANCY_STIFFNESS,
        "omegas_rad_s": list(FIRST_SLICE_OMEGAS),
        "wave_directions_rad": list(FIRST_SLICE_WAVE_DIRECTIONS),
        "projected_dofs": list(FIRST_SLICE_PROJECTED_DOFS),
        "note": (
            "case constants and fixture/evidence metadata only; the "
            "mechanical surge link and dry inertia are owned separately "
            "by the mechanical case contract and no gravity or inertia is "
            "added in any backend"
        ),
    }
