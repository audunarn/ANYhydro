"""Proper orthonormal per-body coefficient frames and the fixed
EQUILIBRIUM-LINEAR coefficient frame map.

Adopted contract (generalizes the translation-only map in ``transform``):
per body the coefficient frame is a proper orthonormal basis ``E`` (a 3x3
matrix whose COLUMNS are the coefficient-basis directions expressed in WORLD
axes) together with a world reference center ``O``. The canonical six
body-qualified DOF names are stable identity; their coordinate values are
components in the recorded coefficient basis about the recorded reference
center. For a frame change ``(E_old, O_old) -> (E_new, O_new)``:

    Q     = E_old.T @ E_new
    d_old = E_old.T @ (O_new - O_old)
    T     = [[Q, skew(d_old) @ Q],     q_old = T @ q_new
             [0,        Q         ]]

    M_new = Ti.T @ M_old @ Tr     (A, B, C, A_inf; leading omega retained)
    F_new = Ti.T @ F_old          (excitation and recognized force splits)

CRITICAL semantics: these are CONSTANT equilibrium-linear coefficient
coordinate changes. C congruence and ``T.T`` baseline transport do NOT by
themselves implement an exact finite-pose reference/chart shift; baseline and
connection terms and a matching consumer pose map are separate ADR-004
obligations. Geometry points (mesh, COM, CB, reference centers), wave
headings and the phasor convention stay in WORLD axes; only DOF coordinate
bases change.

Datasets without the ``coefficient_frames`` attribute follow the existing
published GLOBAL convention (identity axes, reference = ``rotation_center``);
their semantics and hashes are unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from .dof import body_dof_keys, parse_dof_key
from .errors import SchemaError, TransformError, UnsupportedOperationError
from .hashing import dataset_content_hash

COEFFICIENT_FRAMES_ATTR = "coefficient_frames"
SO3_TOLERANCE = 1e-12
FRAME_MAP_ID = "equilibrium_linear_v1"
GLOBAL_FRAME_LABEL = "GLOBAL"

FRAME_MAP_FORMULA = (
    "Q=E_old.T E_new; d_old=E_old.T(O_new-O_old); "
    "T=[[Q,skew(d_old)Q],[0,Q]]; q_old=T q_new"
)

# Optional variables whose exact mapping under a coefficient frame change is
# NOT defined in this slice: their presence is a typed refusal before any
# mutation. No silent dropping or relabeling.
UNSUPPORTED_UNDER_REFRAME = (
    "panel_pressure",
    "field_velocity",
    "free_surface_elevation",
    "kochin_radiation",
    "kochin_diffraction",
)

# Optional force splits that ARE recognized generalized forces in the DOF
# axes; they map exactly like the excitation force.
RECOGNIZED_FORCE_SPLITS = (
    "froude_krylov_force",
    "diffraction_force",
    "mean_drift_force",
)


def _skew(r) -> np.ndarray:
    rx, ry, rz = (float(v) for v in r)
    return np.array([[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]])


def validate_rotation_matrix(axes, name: str = "axes") -> np.ndarray:
    """Validate a proper orthonormal 3x3 basis (columns in WORLD axes).

    Orthogonality and determinant are checked at :data:`SO3_TOLERANCE`.
    Reflections (det = -1), shears and non-finite input fail typed; there is
    no silent orthonormalization.
    """
    arr = np.asarray(axes, dtype=np.float64)
    if arr.shape != (3, 3):
        raise TransformError(f"{name} must be a 3x3 matrix, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise TransformError(f"{name} contains non-finite values")
    gram = arr.T @ arr
    if not np.allclose(gram, np.eye(3), atol=SO3_TOLERANCE, rtol=0.0):
        raise TransformError(
            f"{name} is not orthonormal within {SO3_TOLERANCE} "
            f"(max |E.T E - I| = {float(np.max(np.abs(gram - np.eye(3)))):.3e}); "
            "no silent orthonormalization is performed"
        )
    det = float(np.linalg.det(arr))
    if abs(det - 1.0) > SO3_TOLERANCE:
        raise TransformError(
            f"{name} has determinant {det!r}, not +1 within {SO3_TOLERANCE}; "
            "reflections and improper bases are refused"
        )
    return arr


@dataclass(frozen=True)
class CoefficientFrame:
    """Immutable per-body coefficient frame: world-basis columns + center."""

    axes: np.ndarray
    reference_center: Tuple[float, float, float]

    def __post_init__(self) -> None:
        axes = validate_rotation_matrix(self.axes, "coefficient axes")
        axes = np.array(axes, dtype=np.float64, copy=True)
        axes.setflags(write=False)
        object.__setattr__(self, "axes", axes)
        center = np.asarray(self.reference_center, dtype=np.float64)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise TransformError(
                "reference_center must be a finite 3-tuple of world coordinates"
            )
        object.__setattr__(
            self, "reference_center", tuple(float(v) for v in center)
        )

    def to_json(self) -> dict:
        return {
            "axes": [[float(v) for v in row] for row in self.axes],
            "reference_center": list(self.reference_center),
        }

    @property
    def is_global(self) -> bool:
        return bool(
            np.allclose(self.axes, np.eye(3), atol=SO3_TOLERANCE, rtol=0.0)
        )


def _decode_frames_payload(raw, attr_name: str) -> dict:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaError(
                f"attribute {attr_name!r} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(decoded, dict):
            raise SchemaError(f"attribute {attr_name!r} must be a JSON object")
        return decoded
    raise SchemaError(f"attribute {attr_name!r} must be a JSON object")


def validate_coefficient_frames_attr(ds: xr.Dataset) -> None:
    """Schema-level validation of the ``coefficient_frames`` attribute.

    Malformed metadata fails; it is never ignored. Raises :class:`SchemaError`.
    """
    payload = _decode_frames_payload(ds.attrs[COEFFICIENT_FRAMES_ATTR], COEFFICIENT_FRAMES_ATTR)
    bodies = [str(b) for b in ds["body"].values.tolist()]
    if set(payload) != set(bodies):
        raise SchemaError(
            f"attribute {COEFFICIENT_FRAMES_ATTR!r} must cover exactly the "
            f"dataset bodies {bodies}, got {sorted(payload)}"
        )
    rotation_center = ds["rotation_center"].values
    for i, body in enumerate(bodies):
        entry = payload[body]
        where = f"attribute {COEFFICIENT_FRAMES_ATTR!r}[{body!r}]"
        if not isinstance(entry, dict) or set(entry) != {"axes", "reference_center"}:
            raise SchemaError(
                f"{where} must be a JSON object with exactly the keys "
                "['axes', 'reference_center']"
            )
        axes = entry["axes"]
        ok_axes = (
            isinstance(axes, list)
            and len(axes) == 3
            and all(
                isinstance(row, list)
                and len(row) == 3
                and all(
                    isinstance(v, (int, float)) and not isinstance(v, bool) for v in row
                )
                for row in axes
            )
        )
        if not ok_axes:
            raise SchemaError(f"{where} axes must be a 3x3 nested list of floats")
        try:
            validate_rotation_matrix(np.array(axes, dtype=np.float64), f"{where} axes")
        except TransformError as exc:
            raise SchemaError(str(exc)) from exc
        center = entry["reference_center"]
        ok_center = (
            isinstance(center, list)
            and len(center) == 3
            and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in center
            )
            and all(np.isfinite(v) for v in center)
        )
        if not ok_center:
            raise SchemaError(
                f"{where} reference_center must be a list of 3 finite floats"
            )
        if [float(v) for v in center] != [float(v) for v in rotation_center[i]]:
            raise SchemaError(
                f"{where} reference_center must equal the 'rotation_center' "
                "variable (world geometry does not move under frame changes)"
            )


def coefficient_frames(ds: xr.Dataset) -> Mapping[str, CoefficientFrame]:
    """Validated body_id -> :class:`CoefficientFrame` mapping.

    A dataset without the ``coefficient_frames`` attribute follows the
    existing published GLOBAL convention: identity axes with the stored
    ``rotation_center`` as reference center.
    """
    if COEFFICIENT_FRAMES_ATTR not in ds.attrs:
        bodies = [str(b) for b in ds["body"].values.tolist()]
        rc = ds["rotation_center"].values
        return {
            b: CoefficientFrame(
                axes=np.eye(3), reference_center=tuple(float(v) for v in rc[i])
            )
            for i, b in enumerate(bodies)
        }
    validate_coefficient_frames_attr(ds)
    payload = _decode_frames_payload(ds.attrs[COEFFICIENT_FRAMES_ATTR], COEFFICIENT_FRAMES_ATTR)
    return {
        body: CoefficientFrame(
            axes=np.array(entry["axes"], dtype=np.float64),
            reference_center=tuple(float(v) for v in entry["reference_center"]),
        )
        for body, entry in payload.items()
    }


def coefficient_frame_axes(ds: xr.Dataset) -> Mapping[str, np.ndarray]:
    """Public integration helper: validated body_id -> world-basis 3x3 mapping.

    Columns of each matrix are the body's coefficient-basis directions in
    WORLD axes. Legacy datasets without recorded frames return identity
    matrices (the published GLOBAL convention). The returned arrays are
    read-only.
    """
    frames = coefficient_frames(ds)
    return {body: frame.axes for body, frame in frames.items()}


def require_global_dof_axes(ds: xr.Dataset) -> xr.Dataset:
    """Typed gate for consumers that only admit GLOBAL DOF axes.

    Raises :class:`TransformError` if any body has a non-global recorded
    coefficient basis. Succeeds for legacy datasets without recorded axes:
    GLOBAL is their existing published convention. Returns the dataset.
    """
    frames = coefficient_frames(ds)
    for body, frame in frames.items():
        if not frame.is_global:
            raise TransformError(
                f"body {body!r} has non-global coefficient axes recorded in "
                f"attribute {COEFFICIENT_FRAMES_ATTR!r}; this consumer "
                "requires GLOBAL DOF axes"
            )
    return ds


def frame_change_matrix(old: CoefficientFrame, new: CoefficientFrame) -> np.ndarray:
    """6x6 per-body EQUILIBRIUM-LINEAR map T with ``q_old = T q_new``."""
    Q = old.axes.T @ new.axes
    d_old = old.axes.T @ (
        np.asarray(new.reference_center, dtype=np.float64)
        - np.asarray(old.reference_center, dtype=np.float64)
    )
    T = np.zeros((6, 6))
    T[0:3, 0:3] = Q
    T[0:3, 3:6] = _skew(d_old) @ Q
    T[3:6, 3:6] = Q
    return T


def _global_frame_map(dof_keys, old_frames, new_frames) -> np.ndarray:
    """Block-diagonal global map over an arbitrary DOF axis order."""
    n = len(dof_keys)
    T = np.zeros((n, n))
    position = {key: idx for idx, key in enumerate(dof_keys)}
    for body in {parse_dof_key(k)[0] for k in dof_keys}:
        block = frame_change_matrix(old_frames[body], new_frames[body])
        positions = [position[key] for key in body_dof_keys(body)]
        T[np.ix_(positions, positions)] = block
    return T


def _refuse_unsupported_variables(ds: xr.Dataset) -> None:
    present = [name for name in UNSUPPORTED_UNDER_REFRAME if name in ds.data_vars]
    if present:
        raise TransformError(
            "coefficient frame change refused: the exact mapping of optional "
            f"variable(s) {present} under a frame change is not defined in "
            "this slice; no silent dropping or relabeling is performed"
        )


def _refuse_unmappable_contract(ds: xr.Dataset) -> None:
    from .hydrostatic_contract import (
        FORCE_BASIS_GENERALIZED_DOF,
        HYDROSTATIC_CONTRACT_ATTR,
        read_hydrostatic_contract,
    )

    if HYDROSTATIC_CONTRACT_ATTR not in ds.attrs:
        return
    contract = read_hydrostatic_contract(ds)
    if contract.baseline_wrench is None:
        return
    if contract.force_basis != FORCE_BASIS_GENERALIZED_DOF:
        raise TransformError(
            "coefficient frame change refused: an actual baseline wrench is "
            f"attached with force basis {contract.force_basis!r}; only "
            f"{FORCE_BASIS_GENERALIZED_DOF!r} wrench coordinates have an "
            "exact mapping under the equilibrium-linear frame map"
        )


def _refuse_stale_bound_contract(ds: xr.Dataset) -> None:
    """Refuse to reframe a contract bound to different dataset content.

    Rebiding a foreign/stale bound record silently is not allowed; the
    caller must resolve the binding explicitly first.
    """
    from .hydrostatic_contract import (
        HYDROSTATIC_CONTRACT_ATTR,
        HydrostaticContractError,
        contract_source_binding,
        read_hydrostatic_contract,
    )

    if HYDROSTATIC_CONTRACT_ATTR not in ds.attrs:
        return
    contract = read_hydrostatic_contract(ds)
    if contract.source_binding is None:
        return
    if dict(contract.source_binding) != dict(contract_source_binding(ds)):
        raise HydrostaticContractError(
            "coefficient frame change refused: the attached contract is "
            "bound to different dataset content (bound "
            f"{contract.source_binding['dataset_content_hash']!r}, actual "
            f"{contract_source_binding(ds)['dataset_content_hash']!r}); "
            "rebiding a foreign or stale record requires an explicit "
            "resolution, not a silent frame-change rebind"
        )


def _rebind_contract(ds_out: xr.Dataset) -> None:
    """Rebind an attached contract to the reframed dataset's exact content.

    Runs AFTER all array, axes/centers and history changes, preserving the
    original source identity and provenance; only the noncircular source
    binding is recomputed. The binding hash excludes the contract payload
    itself, so bound reference settings stay unambiguous.
    """
    from .hydrostatic_contract import (
        HYDROSTATIC_CONTRACT_ATTR,
        rebind_hydrostatic_contract,
    )

    if HYDROSTATIC_CONTRACT_ATTR not in ds_out.attrs:
        return
    ds_out.attrs[HYDROSTATIC_CONTRACT_ATTR] = rebind_hydrostatic_contract(ds_out)


def _transport_contract_wrench(ds: xr.Dataset, out: xr.Dataset, old_frames, new_frames) -> None:
    """Transport an attached generalized-DOF baseline wrench with ``T.T``."""
    from .hydrostatic_contract import (
        HYDROSTATIC_CONTRACT_ATTR,
        read_hydrostatic_contract,
        replace_baseline_wrench,
    )

    if HYDROSTATIC_CONTRACT_ATTR not in ds.attrs:
        return
    contract = read_hydrostatic_contract(ds)
    if contract.baseline_wrench is None:
        return
    transported = {}
    for body, wrench in contract.baseline_wrench.items():
        T = frame_change_matrix(old_frames[body], new_frames[body])
        transported[body] = tuple(float(v) for v in (T.T @ np.asarray(wrench)))
    out.attrs[HYDROSTATIC_CONTRACT_ATTR] = replace_baseline_wrench(
        contract, transported
    )


def _apply_frame_change(
    ds: xr.Dataset,
    old_frames: Mapping[str, CoefficientFrame],
    new_frames: Mapping[str, CoefficientFrame],
    kind: str,
) -> xr.Dataset:
    from .schema import validate_hydro_dataset
    from .transform import map_force, map_matrix

    validate_hydro_dataset(ds)
    _refuse_unsupported_variables(ds)
    _refuse_unmappable_contract(ds)
    _refuse_stale_bound_contract(ds)

    bodies = [str(b) for b in ds["body"].values.tolist()]
    influenced_keys = [str(k) for k in ds["influenced_dof"].values.tolist()]
    radiating_keys = [str(k) for k in ds["radiating_dof"].values.tolist()]
    Ti = _global_frame_map(influenced_keys, old_frames, new_frames)
    Tr = _global_frame_map(radiating_keys, old_frames, new_frames)

    source_content_hash = dataset_content_hash(ds)

    out = ds.copy(deep=True)
    out["added_mass"].values = map_matrix(ds["added_mass"].values, Ti, Tr)
    out["radiation_damping"].values = map_matrix(
        ds["radiation_damping"].values, Ti, Tr
    )
    out["hydrostatic_stiffness"].values = map_matrix(
        ds["hydrostatic_stiffness"].values, Ti, Tr
    )
    if "added_mass_infinite" in out.data_vars:
        out["added_mass_infinite"].values = map_matrix(
            ds["added_mass_infinite"].values, Ti, Tr
        )
    out["excitation_force"].values = map_force(ds["excitation_force"].values, Ti)
    for name in RECOGNIZED_FORCE_SPLITS:
        if name in out.data_vars:
            out[name].values = map_force(ds[name].values, Ti)

    centers_changed = any(
        old_frames[b].reference_center != new_frames[b].reference_center
        for b in bodies
    )
    if centers_changed:
        rc = np.array(
            [new_frames[b].reference_center for b in bodies], dtype=np.float64
        )
        out["rotation_center"].values = rc
        raw_identity = out.attrs.get("body_identity")
        identity = (
            json.loads(raw_identity) if isinstance(raw_identity, str) else dict(raw_identity)
        )
        for b in bodies:
            identity[b]["rotation_center"] = [
                float(v) for v in new_frames[b].reference_center
            ]
        out.attrs["body_identity"] = json.dumps(identity, sort_keys=True)

    out.attrs[COEFFICIENT_FRAMES_ATTR] = json.dumps(
        {b: new_frames[b].to_json() for b in bodies}, sort_keys=True
    )

    _transport_contract_wrench(ds, out, old_frames, new_frames)

    transforms = []
    raw = out.attrs.get("reference_transformations")
    if raw:
        transforms = json.loads(raw) if isinstance(raw, str) else list(raw)
    for b in bodies:
        transforms.append(
            {
                "kind": kind,
                "body_id": b,
                "map_id": FRAME_MAP_ID,
                "map": FRAME_MAP_FORMULA,
                "old_axes": [[float(v) for v in row] for row in old_frames[b].axes],
                "new_axes": [[float(v) for v in row] for row in new_frames[b].axes],
                "old_reference_center": list(old_frames[b].reference_center),
                "new_reference_center": list(new_frames[b].reference_center),
                "inverse": {
                    "axes": [[float(v) for v in row] for row in old_frames[b].axes],
                    "reference_center": list(old_frames[b].reference_center),
                },
                "settings": {"so3_tolerance": SO3_TOLERANCE},
                "source_content_hash": source_content_hash,
                "semantics": "constant equilibrium-linear coefficient change; "
                "not an exact finite-pose reference/chart shift",
            }
        )
    out.attrs["reference_transformations"] = json.dumps(transforms)

    # Explicit transport/rebind of the attached contract AFTER all array,
    # axes/centers and history changes; preserves the original source
    # identity and provenance.
    _rebind_contract(out)

    return validate_hydro_dataset(out)


def _validate_new_axes(new_axes, bodies) -> dict:
    if not isinstance(new_axes, Mapping):
        raise TransformError(
            "new_axes must be a mapping of body_id -> 3x3 world-basis matrices"
        )
    if set(new_axes) != set(bodies):
        raise TransformError(
            f"new_axes must cover exactly the dataset bodies {bodies}, "
            f"got {sorted(new_axes)}"
        )
    frames = {}
    for b in bodies:
        frames[b] = validate_rotation_matrix(new_axes[b], f"new axes for body {b!r}")
    return frames


def _validate_new_centers(new_centers, bodies) -> dict:
    if not isinstance(new_centers, Mapping):
        raise TransformError(
            "new_reference_centers must be a mapping of body_id -> 3-vector"
        )
    if set(new_centers) != set(bodies):
        raise TransformError(
            f"new_reference_centers must cover exactly the dataset bodies "
            f"{bodies}, got {sorted(new_centers)}"
        )
    centers = {}
    for b in bodies:
        arr = np.asarray(new_centers[b], dtype=np.float64)
        if arr.shape != (3,) or not np.all(np.isfinite(arr)):
            raise TransformError(
                f"new reference center for body {b!r} must be a finite 3-vector"
            )
        centers[b] = tuple(float(v) for v in arr)
    return centers


def rotate_axes(ds: xr.Dataset, new_axes=None) -> xr.Dataset:
    """Rotate each body's coefficient axes to the given world bases.

    ``new_axes`` maps body_id -> 3x3 matrix whose columns are the new
    coefficient-basis directions in WORLD axes. Reference centers are
    unchanged. The argument-free legacy form remains a typed refusal.
    """
    if new_axes is None:
        raise UnsupportedOperationError(
            "rotation of coordinate axes requires explicit new bases: call "
            "rotate_axes(ds, new_axes); the argument-free form remains "
            "explicitly unsupported"
        )
    from .schema import validate_hydro_dataset

    validate_hydro_dataset(ds)
    bodies = [str(b) for b in ds["body"].values.tolist()]
    axes = _validate_new_axes(new_axes, bodies)
    old_frames = coefficient_frames(ds)
    new_frames = {
        b: CoefficientFrame(axes=axes[b], reference_center=old_frames[b].reference_center)
        for b in bodies
    }
    return _apply_frame_change(ds, old_frames, new_frames, kind="coefficient_frame_change")


def reframe_dataset(
    ds: xr.Dataset,
    new_axes=None,
    new_reference_centers=None,
) -> xr.Dataset:
    """General coefficient frame change: axes and/or reference centers.

    Applies the fixed EQUILIBRIUM-LINEAR map. At least one of ``new_axes``
    and ``new_reference_centers`` must be given; each covers exactly the
    dataset bodies. Geometry, wave headings and phasor convention stay in
    WORLD axes; only DOF coordinate bases change.
    """
    if new_axes is None and new_reference_centers is None:
        raise TransformError(
            "reframe_dataset requires new_axes and/or new_reference_centers; "
            "nothing to reframe"
        )
    from .schema import validate_hydro_dataset

    validate_hydro_dataset(ds)
    bodies = [str(b) for b in ds["body"].values.tolist()]
    old_frames = coefficient_frames(ds)
    axes = (
        _validate_new_axes(new_axes, bodies)
        if new_axes is not None
        else {b: old_frames[b].axes for b in bodies}
    )
    centers = (
        _validate_new_centers(new_reference_centers, bodies)
        if new_reference_centers is not None
        else {b: old_frames[b].reference_center for b in bodies}
    )
    new_frames = {
        b: CoefficientFrame(axes=axes[b], reference_center=centers[b]) for b in bodies
    }
    return _apply_frame_change(ds, old_frames, new_frames, kind="coefficient_frame_change")
