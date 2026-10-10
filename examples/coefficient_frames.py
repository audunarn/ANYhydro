"""Coefficient frame example: load/save/read transformed arrays through
public APIs.

DEVELOPMENT / UNQUALIFIED: this uses a synthetic development fixture and
confers no scientific qualification. The frame map demonstrated here is the
fixed EQUILIBRIUM-LINEAR coefficient coordinate change - it is NOT an exact
finite-pose reference/chart shift (ADR-004).

Run:  python examples/coefficient_frames.py [output_dir]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

import anyhydro
from anyhydro import (
    HydrostaticContract,
    attach_hydrostatic_contract,
    coefficient_frame_axes,
    dataset_content_hash,
    load_hydro_dataset,
    read_hydrostatic_contract,
    reframe_dataset,
    require_global_dof_axes,
    rotate_axes,
    save_hydro_dataset,
)
from anyhydro.fixtures import FIXTURE_BODY_IDS, make_two_body_fixture

ALPHA, BETA = FIXTURE_BODY_IDS


def _rotation(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def main() -> int:
    out_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    os.makedirs(out_dir, exist_ok=True)

    ds = make_two_body_fixture()
    print("DEVELOPMENT / UNQUALIFIED synthetic fixture")
    print("legacy content hash:", dataset_content_hash(ds))

    # legacy datasets are GLOBAL by published convention
    require_global_dof_axes(ds)
    print("legacy frames are GLOBAL (identity bases)")

    # rotate alpha's coefficient axes 30 deg about z; keep beta global
    new_axes = {ALPHA: _rotation((0.0, 0.0, 1.0), np.pi / 6.0), BETA: np.eye(3)}
    rotated = rotate_axes(ds, new_axes)
    recorded = coefficient_frame_axes(rotated)
    print("recorded alpha basis (columns in WORLD axes):")
    print(np.array2string(recorded[ALPHA], precision=6))

    # general reframe: axes and reference centers together
    reframed = reframe_dataset(
        rotated,
        new_axes={ALPHA: np.eye(3), BETA: _rotation((1.0, 0.0, 0.0), 0.2)},
        new_reference_centers={ALPHA: (0.5, 0.0, 0.0), BETA: (12.0, 0.0, -0.5)},
    )

    # attach an explicit honest hydrostatic contract (transport only).
    # The baseline wrench is the fixture's declared measured zero wrench:
    # an actual declaration, never inferred from absence (absent means
    # UNMEASURED and leaves composition unadmitted).
    contract = HydrostaticContract(
        role="BUOYANCY_ONLY",
        force_basis="GENERALIZED_DOF",
        stiffness_derivative="INCREMENTAL_ABOUT_EQUILIBRIUM",
        bodies=(ALPHA, BETA),
        baseline_wrench={ALPHA: (0.0,) * 6, BETA: (0.0,) * 6},
        provenance={"source": "development example: synthetic fixture"},
        full_source=None,
    )
    contracted = attach_hydrostatic_contract(reframed, contract)
    attached_contract = read_hydrostatic_contract(contracted)
    print("attached contract role:", attached_contract.role)
    # attach binds the declarations to the EXACT dataset content; the
    # versioned noncircular binding excludes only the contract payload.
    print("contract bound to exact dataset content:",
          anyhydro.hydrostatic_contract_binding_matches(contracted))
    print("binding version:", attached_contract.source_binding["binding_version"])
    # the bound, complete GENERALIZED_DOF declaration is metadata-admissible
    eligibility = anyhydro.require_hydrostatic_composition(contracted)
    print("composition eligibility (metadata level):", eligibility.admissible)

    # save / read back through the public persistence API
    path = os.path.join(out_dir, "coefficient_frames_example.nc")
    save_hydro_dataset(contracted, path)
    loaded = load_hydro_dataset(path)
    print("round-trip content hash stable:",
          dataset_content_hash(loaded) == dataset_content_hash(contracted))

    # authoritative arrays read back exactly
    for var in ("added_mass", "radiation_damping", "hydrostatic_stiffness",
                "added_mass_infinite", "excitation_force"):
        assert np.array_equal(loaded[var].values, contracted[var].values), var
    print("transformed arrays read back bit-for-bit from disk")

    # provenance is reproducible: old frames (the inverse) are recorded
    raw_transforms = loaded.attrs["reference_transformations"]
    transforms = json.loads(raw_transforms) if isinstance(raw_transforms, str) else raw_transforms
    print("provenance records:", len(transforms), "last kind:", transforms[-1]["kind"])

    # a non-global dataset is honestly refused for GLOBAL-only consumers
    try:
        require_global_dof_axes(loaded)
    except anyhydro.TransformError as exc:
        print("GLOBAL-only consumer refusal (expected):", str(exc)[:70], "...")

    print("output:", path)
    print("DEVELOPMENT / UNQUALIFIED; fixed equilibrium-linear frame semantics only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
