"""Canonical hydrodynamic dataset schema: constants, assembly, validation."""

from __future__ import annotations

import datetime as _dt
import json
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import xarray as xr

from .dof import DOF_NAMES, body_dof_keys, parse_dof_key
from .errors import SchemaError, TimeDomainAdmissionError
from .frames import COEFFICIENT_FRAMES_ATTR, validate_coefficient_frames_attr
from .hashing import JSON_DATASET_ATTRS
from .hydrostatic_contract import (
    HYDROSTATIC_CONTRACT_ATTR,
    validate_hydrostatic_contract_attr,
)
from .records import HydroBody, HydroRequest

from .surfaces import SURFACE_IDENTITY_ATTR, SURFACE_DATASET_SCHEMA, surface_identity, validate_surface_dataset

SCHEMA_VERSION = "1.1.0"
#: emitted only by datasets carrying the opt-in Kochin export or far-field mean drift (docs/MEAN_DRIFT_CONTRACT.md)
KOCHIN_SCHEMA_VERSION = "1.3.0"
KOCHIN_MIN_THETA = 720            # far_field_theta_count minimum (docs/MEAN_DRIFT_CONTRACT.md section 2)
MEAN_DRIFT_UNITS = "N/m^2 (surge, sway), N.m/m^2 (yaw) per unit wave amplitude squared"
KOCHIN_EXTENSION_VARIABLES = ("kochin_radiation", "kochin_diffraction", "wavenumber", "mean_drift_far_field",
                              "water_depth", "rho", "g")

PHASOR_CONVENTION = "x(t) = Re{X * exp(-i*omega*t)}"
HEADING_CONVENTION = "wave propagation direction from +X toward +Y, radians"
NORMAL_CONVENTION = "outward"
LENGTH_UNIT = "m"
FORCE_UNIT = "N"
MOMENT_UNIT = "N.m"

QUALIFICATION_STATUSES = ("development_fixture", "unqualified", "qualified")

MANDATORY_COORDS = ("omega", "wave_direction", "radiating_dof", "influenced_dof", "body", "xyz")
XYZ_VALUES = ("x", "y", "z")

# name -> (dims, dtype kind) for mandatory variables
MANDATORY_VARIABLES: Mapping[str, tuple] = {
    "added_mass": (("omega", "influenced_dof", "radiating_dof"), "float"),
    "radiation_damping": (("omega", "influenced_dof", "radiating_dof"), "float"),
    "excitation_force": (("omega", "wave_direction", "influenced_dof"), "complex"),
    "hydrostatic_stiffness": (("influenced_dof", "radiating_dof"), "float"),
    "rotation_center": (("body", "xyz"), "float"),
    "center_of_mass": (("body", "xyz"), "float"),
    "center_of_buoyancy": (("body", "xyz"), "float"),
}

OPTIONAL_VARIABLES: Mapping[str, tuple] = {
    "added_mass_infinite": (("influenced_dof", "radiating_dof"), "float"),
    # far-field (Kochin energy-flux) damping; positive semidefinite per frequency
    "radiation_damping_far_field": (("omega", "influenced_dof", "radiating_dof"), "float"),
    "froude_krylov_force": (("omega", "wave_direction", "influenced_dof"), "complex"),
    "diffraction_force": (("omega", "wave_direction", "influenced_dof"), "complex"),
    "mean_drift_force": (("omega", "wave_direction", "influenced_dof"), "complex"),
    # schema 1.3.0 Kochin export / far-field mean drift (validated by validate_kochin_extension)
    "wavenumber": (("omega",), "float"),
    "mean_drift_far_field": (("omega", "wave_direction", "drift_dof"), "float"),
    "rho": ((), "float"),
    "g": ((), "float"),
    "water_depth": None,          # may be +inf (deep water); checked by validate_kochin_extension
    # dims not fixed by this schema version; only finiteness is checked
    "panel_pressure": None,
    "pressure_jump": None,
    "field_velocity": None,
    "free_surface_elevation": None,
    "kochin_radiation": None,
    "kochin_diffraction": None,
}

MANDATORY_ATTRS = (
    "schema_version",
    "backend",
    "backend_version",
    "backend_commit",
    "phasor_convention",
    "heading_convention",
    "normal_convention",
    "length_unit",
    "force_unit",
    "moment_unit",
    "mesh_hashes",
    "body_identity",
    "request_hash",
    "created_utc",
    "qualification_status",
)


def _is_string_array(values: np.ndarray) -> bool:
    return values.dtype.kind in ("U", "S") or (
        values.dtype == object and all(isinstance(v, str) for v in values.tolist())
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _check_finite(da: xr.DataArray, name: str) -> None:
    values = da.values
    if np.iscomplexobj(values):
        ok = np.all(np.isfinite(values.real)) and np.all(np.isfinite(values.imag))
    else:
        ok = bool(np.all(np.isfinite(values)))
    _require(ok, f"variable {name!r} contains non-finite values")


def _validate_dof_axis(ds: xr.Dataset, name: str) -> list:
    _require(name in ds.coords, f"missing coordinate {name!r}")
    values = ds[name].values
    _require(
        _is_string_array(values) and values.ndim == 1,
        f"coordinate {name!r} must be a 1-D string array",
    )
    keys = [str(v) for v in values.tolist()]
    _require(len(set(keys)) == len(keys), f"coordinate {name!r} contains duplicate DOF keys")
    for key in keys:
        try:
            parse_dof_key(key)
        except ValueError as exc:
            raise SchemaError(f"coordinate {name!r}: {exc}") from exc
    return keys


def _decode_json_attr(attrs: Mapping[str, Any], key: str) -> Any:
    raw = attrs[key]
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"attribute {key!r} is not valid JSON: {exc}") from exc


BODY_IDENTITY_KEYS = (
    "mesh_hash",
    "lid_mesh_hash",
    "rotation_center",
    "center_of_mass",
    "enabled_dofs",
)


def _validate_body_identity(
    ds: xr.Dataset, bodies: list, mesh_hashes: dict, body_identity: Any
) -> None:
    """Validate the per-body identity attribute used for importer matching.

    Records exactly what the producing side knew: hull and lid mesh hashes,
    reference centers and the enabled DOF set. It deliberately contains no
    dry mass or mechanical data, and claims nothing beyond what the stored
    dataset actually provides.
    """
    _require(
        isinstance(body_identity, dict) and set(body_identity) == set(bodies),
        "attribute 'body_identity' must be a JSON object covering exactly "
        "the dataset bodies",
    )
    rotation_center = ds["rotation_center"].values
    center_of_mass = ds["center_of_mass"].values
    for i, body in enumerate(bodies):
        entry = body_identity[body]
        where = f"attribute 'body_identity[{body!r}]'"
        _require(isinstance(entry, dict), f"{where} must be a JSON object")
        _require(
            set(entry) == set(BODY_IDENTITY_KEYS),
            f"{where} must contain exactly {sorted(BODY_IDENTITY_KEYS)}, "
            f"got {sorted(entry)}",
        )
        _require(
            isinstance(entry["mesh_hash"], str) and entry["mesh_hash"] == mesh_hashes[body],
            f"{where} mesh_hash must match attribute 'mesh_hashes'",
        )
        _require(
            entry["lid_mesh_hash"] is None or isinstance(entry["lid_mesh_hash"], str),
            f"{where} lid_mesh_hash must be a string or null",
        )
        for field_name, stored in (
            ("rotation_center", rotation_center),
            ("center_of_mass", center_of_mass),
        ):
            value = entry[field_name]
            ok = (
                isinstance(value, list)
                and len(value) == 3
                and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
                and all(np.isfinite(v) for v in value)
            )
            _require(ok, f"{where} {field_name} must be a list of 3 finite floats")
            _require(
                [float(v) for v in value] == [float(v) for v in stored[i]],
                f"{where} {field_name} must equal the '{field_name}' variable",
            )
        dofs = entry["enabled_dofs"]
        _require(
            isinstance(dofs, list) and len(dofs) >= 1,
            f"{where} enabled_dofs must be a non-empty list",
        )
        _require(
            len(set(dofs)) == len(dofs),
            f"{where} enabled_dofs contains duplicates",
        )
        for key in dofs:
            try:
                key_body, _ = parse_dof_key(str(key))
            except ValueError as exc:
                raise SchemaError(f"{where} enabled_dofs: {exc}") from exc
            _require(
                key_body == body,
                f"{where} enabled DOF {key!r} does not belong to body {body!r}",
            )


def _json_attr(ds: xr.Dataset, name: str):
    value = ds.attrs.get(name)
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise SchemaError(f"attribute {name!r} is not valid JSON: {exc}") from exc
    return value


def validate_kochin_extension(ds: xr.Dataset) -> None:
    """Schema 1.3.0: opt-in Kochin export and far-field mean drift (docs/MEAN_DRIFT_CONTRACT.md)."""
    # kochin_radiation/kochin_diffraction were already optional (free dims) before 1.3.0: legacy datasets keep them
    present = [name for name in KOCHIN_EXTENSION_VARIABLES
               if name in ds.data_vars and name not in ("kochin_radiation", "kochin_diffraction")]
    if ds.attrs.get("schema_version") != KOCHIN_SCHEMA_VERSION:
        _require(not present, f"variables {present} require schema_version {KOCHIN_SCHEMA_VERSION!r}")
        return
    _require("surface_identity" not in ds.attrs,
             "the Kochin export is not available for surface datasets (schema 1.2.0) yet")
    for name in ("kochin_radiation", "kochin_diffraction", "wavenumber"):
        _require(name in ds.data_vars, f"schema {KOCHIN_SCHEMA_VERSION} requires {name!r}")
    _require("theta" in ds.coords, "schema 1.3.0 requires the coordinate 'theta'")
    theta = np.asarray(ds["theta"].values)
    n = theta.size
    _require(ds["theta"].dtype == np.float64 and theta.ndim == 1 and n >= KOCHIN_MIN_THETA,
             f"coordinate 'theta' must be a 1-D float64 grid of at least {KOCHIN_MIN_THETA} samples")
    _require(bool(np.allclose(theta, 2.0 * np.pi * np.arange(n) / n, rtol=0.0, atol=1e-12)),
             "coordinate 'theta' must be the uniform periodic grid 2 pi j / N")
    for name, dims in (("kochin_radiation", ("omega", "radiating_dof", "theta")),
                       ("kochin_diffraction", ("omega", "wave_direction", "theta"))):
        _require(tuple(ds[name].dims) == dims and ds[name].dtype == np.complex128,
                 f"variable {name!r} must be complex128 with dims {dims}")
        _check_finite(ds[name], name)
    _require(tuple(ds["wavenumber"].dims) == ("omega",) and ds["wavenumber"].dtype == np.float64,
             "variable 'wavenumber' must be float64 with dims ('omega',)")
    _check_finite(ds["wavenumber"], "wavenumber")
    env = _json_attr(ds, "wave_environment")
    _require(isinstance(env, dict) and set(env) == {"water_depth", "rho", "g"},
             "attribute 'wave_environment' must hold water_depth, rho and g")
    depth = np.inf if env["water_depth"] == "inf" else float(env["water_depth"])
    g = float(env["g"])
    _require(depth > 0 and float(env["rho"]) > 0 and g > 0, "wave_environment values must be positive")
    for name, value in (("water_depth", depth), ("rho", float(env["rho"])), ("g", g)):
        _require(name in ds.data_vars and ds[name].dims == () and ds[name].dtype == np.float64
                 and float(ds[name].values) == value,
                 f"scalar {name!r} must be float64 and equal the wave_environment value")
    k = ds["wavenumber"].values
    omega = ds["omega"].values
    rhs = g * k * (np.tanh(k * depth) if np.isfinite(depth) else 1.0)
    _require(bool(np.all(k > 0)) and bool(np.allclose(omega ** 2, rhs, rtol=1e-12, atol=0.0)),
             "wavenumber violates the dispersion relation w^2 = g k tanh(kh)")
    export = _json_attr(ds, "kochin_export")
    _require(isinstance(export, dict) and {"route", "reference_point", "theta_count"} <= set(export)
             and export["reference_point"] == [0.0, 0.0] and int(export["theta_count"]) == n,
             "attribute 'kochin_export' must record route, world-origin reference_point and theta_count")
    if "mean_drift_far_field" in ds.data_vars:
        _require(tuple(ds["mean_drift_far_field"].dims) == ("omega", "wave_direction", "drift_dof")
                 and ds["mean_drift_far_field"].dtype == np.float64,
                 "variable 'mean_drift_far_field' must be float64 with dims (omega, wave_direction, drift_dof)")
        _check_finite(ds["mean_drift_far_field"], "mean_drift_far_field")
        dofs = [str(v) for v in ds["drift_dof"].values]
        bodies = [str(b) for b in ds["body"].values]
        _require(len(bodies) == 1 and dofs == [f"{bodies[0]}::{d}" for d in ("surge", "sway", "yaw")],
                 "coordinate 'drift_dof' must be the dataset body's qualified (surge, sway, yaw)")
        meta = _json_attr(ds, "mean_drift_far_field")
        _require(isinstance(meta, dict) and meta.get("method") == "far_field_momentum_conservation"
                 and meta.get("units") == MEAN_DRIFT_UNITS
                 and isinstance(meta.get("motions"), dict) and meta["motions"].get("motion_hash"),
                 "attribute 'mean_drift_far_field' must record the method, the units and the motion provenance")


def validate_hydro_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Validate a dataset against the canonical schema.

    Returns the dataset unchanged on success; raises :class:`SchemaError`
    with a specific message otherwise.
    """
    _require(isinstance(ds, xr.Dataset), "expected an xarray.Dataset")

    for name in MANDATORY_COORDS:
        _require(name in ds.coords, f"missing coordinate {name!r}")

    omega = ds["omega"].values
    _require(
        ds["omega"].dtype == np.float64 and omega.ndim == 1 and omega.size >= 1,
        "coordinate 'omega' must be a non-empty 1-D float64 array",
    )
    _require(
        bool(np.all(np.isfinite(omega))) and bool(np.all(omega > 0.0)),
        "coordinate 'omega' must be finite and strictly positive "
        "(zero/infinite limits are handled separately)",
    )
    _require(
        omega.size == 1 or bool(np.all(np.diff(omega) > 0.0)),
        "coordinate 'omega' must be strictly increasing",
    )

    directions = ds["wave_direction"].values
    _require(
        ds["wave_direction"].dtype == np.float64
        and directions.ndim == 1
        and directions.size >= 1
        and bool(np.all(np.isfinite(directions))),
        "coordinate 'wave_direction' must be a non-empty finite 1-D float64 array",
    )

    influenced = _validate_dof_axis(ds, "influenced_dof")
    radiating = _validate_dof_axis(ds, "radiating_dof")
    _require(
        sorted(influenced) == sorted(radiating),
        "influenced_dof and radiating_dof must contain the same DOF set",
    )

    body_values = ds["body"].values
    _require(
        _is_string_array(body_values) and body_values.ndim == 1,
        "coordinate 'body' must be a 1-D string array",
    )
    bodies = [str(b) for b in body_values.tolist()]
    _require(len(set(bodies)) == len(bodies), "coordinate 'body' contains duplicate ids")
    _require(all(b.strip() for b in bodies), "coordinate 'body' contains an empty id")

    dof_bodies = {parse_dof_key(k)[0] for k in influenced}
    _require(
        set(bodies) == dof_bodies,
        "body coordinate must match exactly the bodies implied by the DOF axes",
    )
    for body in bodies:
        expected = set(body_dof_keys(body))
        present = {k for k in influenced if parse_dof_key(k)[0] == body}
        _require(
            present == expected,
            f"body {body!r} has incomplete DOFs: missing {sorted(expected - present)}",
        )

    xyz = [str(v) for v in ds["xyz"].values.tolist()]
    _require(tuple(xyz) == XYZ_VALUES, f"coordinate 'xyz' must be {XYZ_VALUES}, got {xyz}")

    for name, spec in MANDATORY_VARIABLES.items():
        _require(name in ds.data_vars, f"missing mandatory variable {name!r}")
        dims, kind = spec
        _require(
            tuple(ds[name].dims) == dims,
            f"variable {name!r} must have dims {dims}, got {tuple(ds[name].dims)}",
        )
        if kind == "float":
            _require(
                ds[name].dtype == np.float64,
                f"variable {name!r} must be float64, got {ds[name].dtype}",
            )
        else:
            _require(
                ds[name].dtype == np.complex128,
                f"variable {name!r} must be complex128, got {ds[name].dtype}",
            )
        _check_finite(ds[name], name)

    for name in ds.data_vars:
        _require(
            name in MANDATORY_VARIABLES or name in OPTIONAL_VARIABLES,
            f"unknown variable {name!r} is not allowed by the schema "
            "(dry mass and mechanical authority are excluded)",
        )

    for name, spec in OPTIONAL_VARIABLES.items():
        if name not in ds.data_vars:
            continue
        if name in ("added_mass_infinite", "radiation_damping_far_field"):
            _require(
                tuple(ds[name].dims) == spec[0] and ds[name].dtype == np.float64,
                f"variable {name!r} must be float64 with dims {spec[0]}",
            )
            source = ds[name].attrs.get("source")
            _require(
                isinstance(source, str) and source.strip(),
                f"variable {name!r} requires a non-empty 'source' provenance attribute",
            )
            _check_finite(ds[name], name)
        elif spec is not None:
            _require(
                tuple(ds[name].dims) == spec[0],
                f"variable {name!r} must have dims {spec[0]}, got {tuple(ds[name].dims)}",
            )
            _check_finite(ds[name], name)
        elif name != "water_depth":       # +inf is deep water; validate_kochin_extension checks it
            _check_finite(ds[name], name)

    for name in MANDATORY_ATTRS:
        _require(name in ds.attrs, f"missing mandatory dataset attribute {name!r}")

    _require(
        ds.attrs["schema_version"] in (SCHEMA_VERSION, SURFACE_DATASET_SCHEMA, KOCHIN_SCHEMA_VERSION),
        f"schema_version must be {SCHEMA_VERSION!r}, {SURFACE_DATASET_SCHEMA!r} or {KOCHIN_SCHEMA_VERSION!r}, "
        f"got {ds.attrs['schema_version']!r}",
    )
    validate_kochin_extension(ds)
    for name in ("backend", "backend_version", "backend_commit"):
        _require(
            isinstance(ds.attrs[name], str) and ds.attrs[name].strip(),
            f"attribute {name!r} must be a non-empty string ('unknown' is allowed)",
        )
    _require(ds.attrs["phasor_convention"] == PHASOR_CONVENTION, "phasor_convention mismatch")
    _require(ds.attrs["heading_convention"] == HEADING_CONVENTION, "heading_convention mismatch")
    _require(ds.attrs["normal_convention"] == NORMAL_CONVENTION, "normal_convention mismatch")
    _require(ds.attrs["length_unit"] == LENGTH_UNIT, "length_unit must be 'm'")
    _require(ds.attrs["force_unit"] == FORCE_UNIT, "force_unit must be 'N'")
    _require(ds.attrs["moment_unit"] == MOMENT_UNIT, "moment_unit must be 'N.m'")

    mesh_hashes = _decode_json_attr(ds.attrs, "mesh_hashes")
    _require(
        isinstance(mesh_hashes, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in mesh_hashes.items()),
        "attribute 'mesh_hashes' must be a JSON object of body_id -> hash",
    )
    _require(
        set(mesh_hashes) == set(bodies),
        "attribute 'mesh_hashes' must cover exactly the dataset bodies",
    )

    body_identity = _decode_json_attr(ds.attrs, "body_identity")
    _validate_body_identity(ds, bodies, mesh_hashes, body_identity)

    _require(
        isinstance(ds.attrs["request_hash"], str) and ds.attrs["request_hash"].strip(),
        "attribute 'request_hash' must be a non-empty string",
    )
    try:
        _dt.datetime.fromisoformat(str(ds.attrs["created_utc"]))
    except ValueError as exc:
        raise SchemaError(f"attribute 'created_utc' must be ISO 8601: {exc}") from exc
    _require(
        ds.attrs["qualification_status"] in QUALIFICATION_STATUSES,
        f"qualification_status must be one of {QUALIFICATION_STATUSES}",
    )

    if "reference_transformations" in ds.attrs:
        transforms = _decode_json_attr(ds.attrs, "reference_transformations")
        _require(isinstance(transforms, list), "attribute 'reference_transformations' must be a JSON array")

    # optional 1.1.0 extensions: malformed recognized metadata fails, it is
    # never silently ignored. Absence is the legacy GLOBAL convention.
    if COEFFICIENT_FRAMES_ATTR in ds.attrs:
        validate_coefficient_frames_attr(ds)
    if HYDROSTATIC_CONTRACT_ATTR in ds.attrs:
        validate_hydrostatic_contract_attr(ds)

    if "auxiliary_lid_policy" in ds.attrs:
        _require(isinstance(ds.attrs["auxiliary_lid_policy"], str) and
                 ds.attrs["auxiliary_lid_policy"] == "submerged_0p1m_v1",
                 "unsupported auxiliary_lid_policy attribute")
        _require(float(np.max(ds["omega"].values)) <= 8.,
                 "submerged_0p1m_v1 band exceeds 8 rad/s")
    validate_surface_dataset(ds)
    return ds


def validate_for_time_domain(ds: xr.Dataset) -> xr.Dataset:
    """Admission check for the time-domain route.

    Requires the explicit ``added_mass_infinite`` limit with provenance. A
    frequency-only dataset is rejected; the last finite-frequency sample is
    never silently substituted.
    """
    validate_hydro_dataset(ds)
    if "added_mass_infinite" not in ds.data_vars:
        raise TimeDomainAdmissionError(
            "dataset is not admissible to the time-domain route: "
            "added_mass_infinite is missing; the last finite-frequency sample "
            "is not a valid substitute"
        )
    return ds


def assemble_hydro_dataset(
    bodies: Sequence[HydroBody],
    request: HydroRequest,
    *,
    added_mass,
    radiation_damping,
    excitation_force,
    hydrostatic_stiffness,
    rotation_center,
    center_of_mass,
    center_of_buoyancy,
    added_mass_infinite=None,
    added_mass_infinite_source: Optional[str] = None,
    radiation_damping_far_field=None,
    radiation_damping_far_field_source: Optional[str] = None,
    backend: str = "unknown",
    backend_version: str = "unknown",
    backend_commit: str = "unknown",
    qualification_status: str = "unqualified",
    created_utc: Optional[str] = None,
    extra_attrs: Optional[Mapping[str, Any]] = None,
) -> xr.Dataset:
    """Build a canonical hydrodynamic dataset from validated pieces.

    The full 6N-DOF space is always assembled, regardless of each body's
    enabled DOFs: complete coupled matrices are the production model.
    """
    _require(len(bodies) >= 1, "at least one body is required")
    body_ids = [b.body_id for b in bodies]
    _require(len(set(body_ids)) == len(body_ids), "duplicate body ids in bodies")
    for b in bodies:
        _require(isinstance(b, HydroBody), f"body {b!r} is not a HydroBody")
    _require(isinstance(request, HydroRequest), "request must be a HydroRequest")

    dof_axis = []
    for b in bodies:
        dof_axis.extend(body_dof_keys(b.body_id))
    n_dof = len(dof_axis)
    n_omega = request.omegas.shape[0]
    n_dir = request.wave_directions.shape[0]

    def _float64(name, value, shape):
        arr = np.array(value, dtype=np.float64, copy=True)
        _require(arr.shape == shape, f"{name} must have shape {shape}, got {arr.shape}")
        _require(bool(np.all(np.isfinite(arr))), f"{name} contains non-finite values")
        return arr

    added_mass = _float64("added_mass", added_mass, (n_omega, n_dof, n_dof))
    damping = _float64("radiation_damping", radiation_damping, (n_omega, n_dof, n_dof))
    excitation = np.array(excitation_force, dtype=np.complex128, copy=True)
    _require(
        excitation.shape == (n_omega, n_dir, n_dof),
        f"excitation_force must have shape {(n_omega, n_dir, n_dof)}, got {excitation.shape}",
    )
    _require(
        bool(np.all(np.isfinite(excitation.real)) and np.all(np.isfinite(excitation.imag))),
        "excitation_force contains non-finite values",
    )
    stiffness = _float64("hydrostatic_stiffness", hydrostatic_stiffness, (n_dof, n_dof))

    centers = {}
    for name, values in (
        ("rotation_center", rotation_center),
        ("center_of_mass", center_of_mass),
        ("center_of_buoyancy", center_of_buoyancy),
    ):
        arr = _float64(name, values, (len(bodies), 3))
        centers[name] = arr

    data_vars = {
        "added_mass": (("omega", "influenced_dof", "radiating_dof"), added_mass),
        "radiation_damping": (("omega", "influenced_dof", "radiating_dof"), damping),
        "excitation_force": (("omega", "wave_direction", "influenced_dof"), excitation),
        "hydrostatic_stiffness": (("influenced_dof", "radiating_dof"), stiffness),
        "rotation_center": (("body", "xyz"), centers["rotation_center"]),
        "center_of_mass": (("body", "xyz"), centers["center_of_mass"]),
        "center_of_buoyancy": (("body", "xyz"), centers["center_of_buoyancy"]),
    }

    if added_mass_infinite is not None:
        _require(
            isinstance(added_mass_infinite_source, str) and added_mass_infinite_source.strip(),
            "added_mass_infinite requires a non-empty added_mass_infinite_source",
        )
        a_inf = _float64("added_mass_infinite", added_mass_infinite, (n_dof, n_dof))
        data_vars["added_mass_infinite"] = (("influenced_dof", "radiating_dof"), a_inf)

    if radiation_damping_far_field is not None:
        _require(
            isinstance(radiation_damping_far_field_source, str)
            and radiation_damping_far_field_source.strip(),
            "radiation_damping_far_field requires a non-empty source description",
        )
        b_far = _float64("radiation_damping_far_field", radiation_damping_far_field,
                         (n_omega, n_dof, n_dof))
        data_vars["radiation_damping_far_field"] = (
            ("omega", "influenced_dof", "radiating_dof"), b_far)

    if created_utc is None:
        created_utc = _dt.datetime.now(_dt.timezone.utc).isoformat()

    attrs = {
        "schema_version": SCHEMA_VERSION,
        "backend": str(backend),
        "backend_version": str(backend_version),
        "backend_commit": str(backend_commit),
        "phasor_convention": PHASOR_CONVENTION,
        "heading_convention": HEADING_CONVENTION,
        "normal_convention": NORMAL_CONVENTION,
        "length_unit": LENGTH_UNIT,
        "force_unit": FORCE_UNIT,
        "moment_unit": MOMENT_UNIT,
        "mesh_hashes": json.dumps(
            {b.body_id: b.mesh.mesh_hash for b in bodies}, sort_keys=True
        ),
        "body_identity": json.dumps(
            {
                b.body_id: {
                    "mesh_hash": b.mesh.mesh_hash,
                    "lid_mesh_hash": None if b.lid_mesh is None else b.lid_mesh.mesh_hash,
                    "rotation_center": [float(v) for v in b.rotation_center],
                    "center_of_mass": [float(v) for v in b.center_of_mass],
                    "enabled_dofs": list(b.enabled_dofs),
                }
                for b in bodies
            },
            sort_keys=True,
        ),
        "request_hash": request.content_hash(),
        "created_utc": str(created_utc),
        "qualification_status": qualification_status,
    }
    if "auxiliary_lid_policy" in request.backend_settings:
        policy = request.backend_settings["auxiliary_lid_policy"]
        _require(policy == "submerged_0p1m_v1", "unsupported auxiliary lid policy")
        attrs["auxiliary_lid_policy"] = policy
    explicit_surfaces = surface_identity(bodies)
    if explicit_surfaces is not None:
        attrs["schema_version"] = SURFACE_DATASET_SCHEMA
        attrs[SURFACE_IDENTITY_ATTR] = json.dumps(explicit_surfaces, sort_keys=True)
    if extra_attrs:
        for key, value in extra_attrs.items():
            if key in (SURFACE_IDENTITY_ATTR, "auxiliary_lid_policy"):
                raise SchemaError('surface_identity is built from the actual body records; cannot be overridden')
            if isinstance(value, (dict, list)):
                value = json.dumps(value, sort_keys=True)
            attrs[str(key)] = value

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "omega": np.asarray(request.omegas, dtype=np.float64),
            "wave_direction": np.asarray(request.wave_directions, dtype=np.float64),
            "influenced_dof": np.array(dof_axis, dtype=object),
            "radiating_dof": np.array(dof_axis, dtype=object),
            "body": np.array(body_ids, dtype=object),
            "xyz": np.array(XYZ_VALUES, dtype=object),
        },
        attrs=attrs,
    )

    if "added_mass_infinite" in ds.data_vars:
        ds["added_mass_infinite"] = ds["added_mass_infinite"].assign_attrs(
            source=str(added_mass_infinite_source)
        )
    if "radiation_damping_far_field" in ds.data_vars:
        ds["radiation_damping_far_field"] = ds["radiation_damping_far_field"].assign_attrs(
            source=str(radiation_damping_far_field_source)
        )

    return validate_hydro_dataset(ds)
