"""Deterministic synthetic two-body development fixtures.

These are development fixtures: they exercise the full 12x12 coupled DOF
space with nonzero cross-body blocks and complex excitation. They confer no
scientific qualification and are marked ``development_fixture``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .dof import DOF_NAMES, body_dof_keys, qualify_dof
from .records import HydroBody, HydroPanelMesh, HydroRequest
from .schema import assemble_hydro_dataset

FIXTURE_BODY_IDS = ("fixture-body-alpha", "fixture-body-beta")

FIXTURE_OMEGAS = np.array([0.3, 0.7, 1.1, 1.6, 2.2], dtype=np.float64)
FIXTURE_WAVE_DIRECTIONS = np.array(
    [0.0, np.pi / 4.0, np.pi / 2.0], dtype=np.float64
)
FIXTURE_WATER_DEPTH = 150.0
FIXTURE_RHO = 1025.0
FIXTURE_GRAVITY = 9.81

ALPHA_CENTER = (0.0, 0.0, 0.0)
BETA_CENTER = (12.0, 0.0, -0.5)


def _box_mesh(body_id: str, center, half_extents):
    """Axis-aligned box panel mesh with honestly verified outward normals."""
    cx, cy, cz = (float(v) for v in center)
    hx, hy, hz = (float(v) for v in half_extents)
    corners = []
    for dx in (-1.0, 1.0):
        for dy in (-1.0, 1.0):
            for dz in (-1.0, 1.0):
                corners.append((cx + dx * hx, cy + dy * hy, cz + dz * hz))
    # vertex order: index = 4*dx + 2*dy + dz with signs encoded
    vertices = np.array(
        [
            corners[0], corners[1], corners[3], corners[2],
            corners[4], corners[5], corners[7], corners[6],
        ],
        dtype=np.float64,
    )
    quads = [
        [0, 1, 2, 3],  # z-min
        [4, 5, 6, 7],  # z-max
        [0, 1, 5, 4],  # y-min
        [3, 2, 6, 7],  # y-max
        [0, 3, 7, 4],  # x-min
        [1, 2, 6, 5],  # x-max
    ]
    box_center = np.array([cx, cy, cz])
    faces = []
    verified = True
    for quad in quads:
        p0, p1, p2 = vertices[quad[0]], vertices[quad[1]], vertices[quad[2]]
        normal = np.cross(p1 - p0, p2 - p0)
        face_center = vertices[quad].mean(axis=0)
        if np.dot(normal, face_center - box_center) < 0.0:
            quad = quad[::-1]
            normal = -normal
        if np.dot(normal, face_center - box_center) <= 0.0:
            verified = False
        faces.append(quad)
    faces_arr = np.array(faces, dtype=np.int32)
    sizes = np.full(len(faces), 4, dtype=np.uint8)
    return HydroPanelMesh(
        vertices=vertices,
        faces=faces_arr,
        face_sizes=sizes,
        body_id=body_id,
        outward_normals_verified=verified,
        source_geometry_refs=("synthetic-box-fixture",),
        mesh_hash="",
    )


def make_fixture_bodies() -> tuple:
    """The two synthetic fixture bodies with full enabled DOF sets."""
    bodies = []
    for body_id, center in (
        (FIXTURE_BODY_IDS[0], ALPHA_CENTER),
        (FIXTURE_BODY_IDS[1], BETA_CENTER),
    ):
        mesh = _box_mesh(body_id, center, (1.0, 0.6, 0.8))
        bodies.append(
            HydroBody(
                body_id=body_id,
                name=f"synthetic fixture body {body_id}",
                mesh=mesh,
                lid_mesh=None,
                rotation_center=center,
                center_of_mass=(center[0], center[1], center[2] + 0.4),
                enabled_dofs=body_dof_keys(body_id),
            )
        )
    return tuple(bodies)


def make_fixture_request() -> HydroRequest:
    """The synthetic fixture request."""
    return HydroRequest(
        omegas=FIXTURE_OMEGAS.copy(),
        wave_directions=FIXTURE_WAVE_DIRECTIONS.copy(),
        water_depth=FIXTURE_WATER_DEPTH,
        rho=FIXTURE_RHO,
        gravity=FIXTURE_GRAVITY,
        include_zero_frequency=False,
        include_infinite_frequency=False,
        backend_settings={"fixture": "two-body-synthetic"},
    )


def _fixture_arrays(n_omega: int, n_dir: int, n_dof: int):
    rng_free = np.arange(n_omega)
    added_mass = np.empty((n_omega, n_dof, n_dof))
    damping = np.empty((n_omega, n_dof, n_dof))
    for o in range(n_omega):
        for i in range(n_dof):
            for j in range(n_dof):
                same_body = (i // 6) == (j // 6)
                if i == j:
                    a = 1000.0 + 100.0 * (i % 6) + 10.0 * o
                    b = 50.0 + 5.0 * (i % 6) + 2.0 * o
                elif same_body:
                    a = 25.0 + 5.0 * ((i * 7 + j * 3) % 11) + 1.5 * o
                    b = 5.0 + 1.0 * ((i * 5 + j * 2) % 9) + 0.3 * o
                else:
                    # cross-body coupling: nonzero by construction
                    a = 40.0 + 8.0 * ((i * 5 + j * 11) % 7) + 2.0 * o + 0.5 * rng_free[o]
                    b = 8.0 + 2.0 * ((i * 3 + j * 7) % 5) + 0.5 * o
                added_mass[o, i, j] = a
                damping[o, i, j] = b

    excitation = np.empty((n_omega, n_dir, n_dof), dtype=np.complex128)
    for o in range(n_omega):
        for w in range(n_dir):
            for i in range(n_dof):
                magnitude = 500.0 + 60.0 * (i % 6) + 40.0 * o + 10.0 * w
                phase = (
                    FIXTURE_WAVE_DIRECTIONS[w] + 0.3 * i + 0.1 * o + 0.05 * w
                )
                excitation[o, w, i] = magnitude * (
                    np.cos(phase) + 1j * np.sin(phase)
                )

    stiffness = np.zeros((n_dof, n_dof))
    for b in range(n_dof // 6):
        heave = 2 + 6 * b
        roll = 3 + 6 * b
        pitch = 4 + 6 * b
        stiffness[heave, heave] = 2.0e4
        stiffness[roll, roll] = 5.0e3
        stiffness[pitch, pitch] = 6.0e3
        stiffness[heave, pitch] = 1.5e3
        stiffness[pitch, heave] = 1.5e3
        stiffness[roll, pitch] = 2.0e2
        stiffness[pitch, roll] = 2.0e2

    return added_mass, damping, excitation, stiffness


def make_two_body_fixture(
    *, include_added_mass_infinite: bool = True
) -> "xr.Dataset":
    """Deterministic synthetic two-body 12x12 development fixture."""
    import xarray as xr  # local to keep module import light

    bodies = make_fixture_bodies()
    request = make_fixture_request()
    n_omega = request.omegas.shape[0]
    n_dir = request.wave_directions.shape[0]
    n_dof = 6 * len(bodies)

    added_mass, damping, excitation, stiffness = _fixture_arrays(
        n_omega, n_dir, n_dof
    )

    rotation_center = np.array([ALPHA_CENTER, BETA_CENTER], dtype=np.float64)
    center_of_mass = np.array(
        [(ALPHA_CENTER[0], ALPHA_CENTER[1], ALPHA_CENTER[2] + 0.4),
         (BETA_CENTER[0], BETA_CENTER[1], BETA_CENTER[2] + 0.4)],
        dtype=np.float64,
    )
    center_of_buoyancy = np.array(
        [(ALPHA_CENTER[0], ALPHA_CENTER[1], ALPHA_CENTER[2] - 0.2),
         (BETA_CENTER[0], BETA_CENTER[1], BETA_CENTER[2] - 0.1)],
        dtype=np.float64,
    )

    added_mass_infinite = None
    a_inf_source = None
    if include_added_mass_infinite:
        # explicitly distinct from the last finite-frequency sample
        added_mass_infinite = 0.5 * added_mass[-1] + 15.0
        a_inf_source = (
            "synthetic-development-fixture: analytic construction, not a "
            "backend limit solve"
        )

    return assemble_hydro_dataset(
        bodies,
        request,
        added_mass=added_mass,
        radiation_damping=damping,
        excitation_force=excitation,
        hydrostatic_stiffness=stiffness,
        rotation_center=rotation_center,
        center_of_mass=center_of_mass,
        center_of_buoyancy=center_of_buoyancy,
        added_mass_infinite=added_mass_infinite,
        added_mass_infinite_source=a_inf_source,
        backend="synthetic-fixture",
        backend_version="1.0.0",
        backend_commit="unknown",
        qualification_status="development_fixture",
    )
