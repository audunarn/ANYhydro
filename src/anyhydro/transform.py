"""Reference-point translation maps for forces and matrices.

Adopted contract: for a shift ``r = new_origin - old_origin`` with axes
unchanged, per body

    T = [[I, skew(r)],   q_old = T q_new
         [0,     I  ]]
    F_new = T^T F_old
    A_new = T^T A_old T        (likewise B, C, A_inf)

The global map is block diagonal over bodies, so cross-body blocks transform
as ``T_i^T A_ij T_j`` and are preserved. With recorded coefficient frames
(``frames.coefficient_frames``) the shift acts in the body's coefficient
basis; legacy GLOBAL datasets keep the exact world-shift semantics and
hashes. General coefficient frame changes (axes and centers) live in
``frames``; ``rotate_axes`` is re-exported from there and its argument-free
legacy form remains a typed refusal.
"""

from __future__ import annotations

import json
from typing import Mapping, Sequence

import numpy as np
import xarray as xr

from .dof import DOF_NAMES, body_dof_keys, parse_dof_key
from .errors import TransformError
from .frames import COEFFICIENT_FRAMES_ATTR, coefficient_frames
from .frames import rotate_axes  # re-exported: the argument-free legacy form stays a typed refusal
from .schema import validate_hydro_dataset


def skew(r) -> np.ndarray:
    """3x3 skew matrix with ``skew(r) @ theta = r x theta``."""
    rx, ry, rz = (float(v) for v in r)
    return np.array([[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]])


def shift_matrix(r) -> np.ndarray:
    """6x6 per-body translation map T = [[I, skew(r)], [0, I]]."""
    T = np.zeros((6, 6))
    T[0:3, 0:3] = np.eye(3)
    T[0:3, 3:6] = skew(r)
    T[3:6, 3:6] = np.eye(3)
    return T


def _global_shift_map(dof_keys, old_centers, new_centers, old_frames=None) -> np.ndarray:
    """Block-diagonal global map over the given DOF axis order.

    Robust to arbitrary DOF axis ordering: each body's 6x6 block is placed
    with an index cross-product, so cross-body blocks are preserved when the
    full matrix is mapped with this map on both axes.

    With recorded coefficient frames the shift acts in the body's coefficient
    basis: ``d_old = E_old.T (O_new - O_old)``. For legacy GLOBAL datasets
    ``E_old`` is the identity, so the map is bit-for-bit the world shift.
    """
    n = len(dof_keys)
    T = np.zeros((n, n))
    position = {key: idx for idx, key in enumerate(dof_keys)}
    blocks = {}
    for body_id in {parse_dof_key(k)[0] for k in dof_keys}:
        r = np.asarray(new_centers[body_id], dtype=np.float64) - np.asarray(
            old_centers[body_id], dtype=np.float64
        )
        if old_frames is not None:
            r = old_frames[body_id].axes.T @ r
        blocks[body_id] = shift_matrix(r)
    for body_id, block in blocks.items():
        positions = [position[key] for key in body_dof_keys(body_id)]
        T[np.ix_(positions, positions)] = block
    return T


def map_matrix(matrix, T_influenced: np.ndarray, T_radiating: np.ndarray) -> np.ndarray:
    """Map a matrix: ``M_new = Ti^T M_old Tr`` (leading omega axis optional)."""
    M = np.asarray(matrix)
    Ti, Tr = np.asarray(T_influenced), np.asarray(T_radiating)
    if M.ndim == 2:
        return Ti.T @ M @ Tr
    if M.ndim == 3:
        return np.einsum("ai,oab,bj->oij", Ti, M, Tr)
    raise TransformError(f"matrix must be 2-D or 3-D (omega-leading), got {M.ndim}-D")


def map_force(force, T_influenced: np.ndarray) -> np.ndarray:
    """Map a force/moment vector: ``F_new = T^T F_old``."""
    F = np.asarray(force)
    Ti = np.asarray(T_influenced)
    if F.ndim == 1:
        return Ti.T @ F
    if F.ndim == 2:
        # (omega, dof) or (direction, dof)
        return np.einsum("ai,oa->oi", Ti, F)
    if F.ndim == 3:
        # (omega, direction, dof)
        return np.einsum("ai,owa->owi", Ti, F)
    raise TransformError(f"force must be 1-D, 2-D or 3-D, got {F.ndim}-D")


def _centers_from_dataset(ds: xr.Dataset) -> dict:
    bodies = [str(b) for b in ds["body"].values.tolist()]
    rc = ds["rotation_center"].values
    return {b: tuple(float(v) for v in rc[i]) for i, b in enumerate(bodies)}


def shift_dataset_reference_point(
    ds: xr.Dataset, new_rotation_centers: Mapping[str, Sequence[float]]
) -> xr.Dataset:
    """Shift each body's reference point to the given rotation centers.

    Returns a new dataset; the input is not mutated. The stored
    ``rotation_center`` is updated and a provenance record is appended to the
    ``reference_transformations`` attribute.
    """
    validate_hydro_dataset(ds)

    bodies = [str(b) for b in ds["body"].values.tolist()]
    if set(new_rotation_centers) != set(bodies):
        raise TransformError(
            f"new_rotation_centers must cover exactly the dataset bodies {bodies}, "
            f"got {sorted(new_rotation_centers)}"
        )

    old_centers = _centers_from_dataset(ds)
    new_centers = {}
    for b in bodies:
        arr = np.asarray(new_rotation_centers[b], dtype=np.float64)
        if arr.shape != (3,) or not np.all(np.isfinite(arr)):
            raise TransformError(f"new rotation center for body {b!r} must be a finite 3-vector")
        new_centers[b] = tuple(float(v) for v in arr)

    # recorded coefficient frames (identity for legacy GLOBAL datasets)
    old_frames = coefficient_frames(ds)
    framed = COEFFICIENT_FRAMES_ATTR in ds.attrs

    influenced_keys = [str(k) for k in ds["influenced_dof"].values.tolist()]
    radiating_keys = [str(k) for k in ds["radiating_dof"].values.tolist()]
    Ti = _global_shift_map(influenced_keys, old_centers, new_centers, old_frames)
    Tr = _global_shift_map(radiating_keys, old_centers, new_centers, old_frames)

    out = ds.copy(deep=True)

    out["added_mass"].values = map_matrix(ds["added_mass"].values, Ti, Tr)
    out["radiation_damping"].values = map_matrix(ds["radiation_damping"].values, Ti, Tr)
    out["hydrostatic_stiffness"].values = map_matrix(
        ds["hydrostatic_stiffness"].values, Ti, Tr
    )
    if "added_mass_infinite" in out.data_vars:
        out["added_mass_infinite"].values = map_matrix(
            ds["added_mass_infinite"].values, Ti, Tr
        )
    out["excitation_force"].values = map_force(ds["excitation_force"].values, Ti)

    rc = np.array([new_centers[b] for b in bodies], dtype=np.float64)
    out["rotation_center"].values = rc

    # keep the body_identity attribute consistent with the new reference
    # frame: rotation centers moved, mesh/COM/DOF identity did not
    raw_identity = out.attrs.get("body_identity")
    identity = json.loads(raw_identity) if isinstance(raw_identity, str) else dict(raw_identity)
    for b in bodies:
        identity[b]["rotation_center"] = [float(v) for v in new_centers[b]]
    out.attrs["body_identity"] = json.dumps(identity, sort_keys=True)

    transforms = []
    raw = out.attrs.get("reference_transformations")
    if raw:
        transforms = json.loads(raw)
    for b in bodies:
        record = {
            "kind": "translation_shift",
            "body_id": b,
            "old_rotation_center": list(old_centers[b]),
            "new_rotation_center": list(new_centers[b]),
            "axes": "unchanged",
            "map": "q_old = T q_new, T = [[I, skew(r)], [0, I]]",
        }
        if framed:
            # the shift acts in the body's recorded coefficient basis; the
            # frame metadata reference center moves with the variable
            record["d_coordinates"] = "old_coefficient_basis"
        transforms.append(record)
    out.attrs["reference_transformations"] = json.dumps(transforms)

    if framed:
        raw_frames = out.attrs[COEFFICIENT_FRAMES_ATTR]
        frames_payload = (
            json.loads(raw_frames) if isinstance(raw_frames, str) else dict(raw_frames)
        )
        for b in bodies:
            frames_payload[b]["reference_center"] = [float(v) for v in new_centers[b]]
        out.attrs[COEFFICIENT_FRAMES_ATTR] = json.dumps(frames_payload, sort_keys=True)

    return validate_hydro_dataset(out)


def reorder_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Reorder body and DOF axes to canonical order by stable identity.

    All matrix variables and the excitation are reordered consistently. The
    content hash is invariant under this operation.
    """
    validate_hydro_dataset(ds)
    bodies = sorted(str(b) for b in ds["body"].values.tolist())
    dof_keys = []
    for b in bodies:
        dof_keys.extend(body_dof_keys(b))
    return ds.sel(
        body=bodies,
        influenced_dof=dof_keys,
        radiating_dof=dof_keys,
    ).copy(deep=True)
