"""Headless example: two-body BEM coefficients via the Capytaine adapter.

Builds two tiny submerged boxes (outward-verified normals, 6 panels each),
solves them as one combined multibody with the development Capytaine
adapter, prints the coupled coefficients (including the cross-body blocks),
then saves the canonical dataset to NetCDF, reloads it and verifies the
round trip.

Requirements: the optional ``capytaine`` dependency (>= 3.0, < 3.1).
Development evidence only — no scientific qualification is implied.

Run (headless, from the repository root):

    python examples/headless_two_body.py --out two_body.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

from anyhydro import (
    CapytaineBackend,
    HydroBody,
    HydroPanelMesh,
    HydroRequest,
    body_dof_keys,
    dataset_content_hash,
    load_hydro_dataset,
    save_hydro_dataset,
    validate_hydro_dataset,
)


def box_body(body_id: str, center) -> HydroBody:
    """A tiny submerged box with honestly verified outward normals."""
    cx, cy, cz = (float(v) for v in center)
    hx, hy, hz = 1.0, 0.5, 0.25
    corners = [
        (cx + dx * hx, cy + dy * hy, cz + dz * hz)
        for dx in (-1.0, 1.0)
        for dy in (-1.0, 1.0)
        for dz in (-1.0, 1.0)
    ]
    vertices = np.array(
        [corners[0], corners[1], corners[3], corners[2],
         corners[4], corners[5], corners[7], corners[6]],
        dtype=np.float64,
    )
    quads = [
        [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
        [3, 2, 6, 7], [0, 3, 7, 4], [1, 2, 6, 5],
    ]
    box_center = np.array([cx, cy, cz])
    faces = []
    for quad in quads:
        q = list(quad)
        normal = np.cross(vertices[q[1]] - vertices[q[0]], vertices[q[2]] - vertices[q[0]])
        face_center = vertices[q].mean(axis=0)
        if np.dot(normal, face_center - box_center) < 0.0:
            q = q[::-1]
        faces.append(q)
    mesh = HydroPanelMesh(
        vertices=vertices,
        faces=np.array(faces, dtype=np.int32),
        face_sizes=np.full(len(faces), 4, dtype=np.uint8),
        body_id=body_id,
        outward_normals_verified=True,
        source_geometry_refs=("headless-example-box",),
        mesh_hash="",
    )
    return HydroBody(
        body_id=body_id,
        name=body_id,
        mesh=mesh,
        lid_mesh=None,
        rotation_center=center,
        center_of_mass=center,
        enabled_dofs=body_dof_keys(body_id),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="two_body.nc", help="output NetCDF path")
    parser.add_argument(
        "--cache-dir", default=None, help="optional content-addressed cache directory"
    )
    args = parser.parse_args()

    bodies = (
        box_body("alpha", (0.0, 0.0, -0.3)),
        box_body("beta", (4.0, 0.0, -0.3)),
    )
    request = HydroRequest(
        omegas=np.array([1.0]),
        wave_directions=np.array([0.0]),
        water_depth=float("inf"),  # deep water
        rho=1025.0,
        gravity=9.81,
        include_zero_frequency=False,
        include_infinite_frequency=True,
        backend_settings={} if args.cache_dir is None else {"cache_dir": args.cache_dir},
    )

    backend = CapytaineBackend()
    dataset = backend.solve(bodies, request)
    validate_hydro_dataset(dataset)

    print("bodies:", [str(b) for b in dataset["body"].values.tolist()])
    print("DOF axis size:", dataset["influenced_dof"].size)
    print("added_mass shape:", dataset["added_mass"].shape)
    cross = dataset["added_mass"].values[0, 0:6, 6:12]
    print("cross-body added-mass block nonzero:", bool(np.any(cross != 0.0)))
    print(
        "alpha heave / beta heave added mass:",
        float(dataset["added_mass"].sel(
            influenced_dof="alpha::heave", radiating_dof="beta::heave").values[0]),
    )
    print(
        "alpha heave excitation (1.0 rad/s, heading 0):",
        complex(dataset["excitation_force"].sel(influenced_dof="alpha::heave").values[0, 0]),
    )
    print(
        "added_mass_infinite source:",
        dataset["added_mass_infinite"].attrs["source"],
    )
    print(
        "hydrostatics definition:",
        dataset.attrs["hydrostatic_stiffness_definition"],
    )

    out = Path(args.out)
    save_hydro_dataset(dataset, out)
    reloaded = load_hydro_dataset(out)
    validate_hydro_dataset(reloaded)
    assert dataset_content_hash(reloaded) == dataset_content_hash(dataset)
    print(f"saved and reloaded {out} (content hash stable)")

    # the replay backend admits exactly the same body identities and request
    from anyhydro import ImportedHydroDatasetBackend

    replay = ImportedHydroDatasetBackend(reloaded)
    replayed = replay.solve(bodies, request)
    print("replay backend round trip OK:", replayed["added_mass"].shape)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # typed failures, honest exit
        print(f"failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
