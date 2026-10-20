# SPDX-License-Identifier: MPL-2.0
"""Export of a validated single-body ANYhydro dataset to an OrcaFlex 11.6c vessel type (slice O1b).

Two layers, see docs/ORCAFLEX_EXPORT.md for the conventions and their evidence:

- :func:`vessel_type_data` is PURE. It validates the dataset and the declared mass properties, composes the hydrostatic
  stiffness (buoyancy from the attached hydrostatic contract through the ADR-004 chart conversion, plus the gravity block of
  the declared mass and centre of gravity), applies the declared symmetrisation, converts units, phases and headings and
  returns plain Python data with a provenance record. It never imports OrcFxAPI.
- :func:`write_vessel_type` is a thin writer that sets exactly those items through OrcFxAPI (imported lazily) and reads
  them back. :func:`export_vessel_type` combines both.

Nothing is converted silently: every unsupported or inconsistent input is a typed :class:`OrcaFlexExportError`.
Mass, centre of gravity and inertia are required declarations (the dataset carries none); a mass that is not the
displaced mass, a skew part above the declared bound or a non-diagonal inertia is refused.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .dof import DOF_NAMES, body_dof_keys
from .errors import HydroError
from .frames import require_global_dof_axes
from .hashing import dataset_content_hash
from .hydrostatic_contract import (ROLE_BUOYANCY_ONLY, ROLE_UNKNOWN, chart_stiffness, read_hydrostatic_contract,
                                   require_hydrostatic_composition)
from .schema import FORCE_UNIT, HEADING_CONVENTION, LENGTH_UNIT, MOMENT_UNIT, PHASOR_CONVENTION, validate_hydro_dataset

FORMAT = "anyhydro-orcaflex-vessel-type-v1"
#: tags of the written vessel type: the full provenance JSON and the dataset content hash
PROVENANCE_TAG = "anyhydro_provenance"
HASH_TAG = "anyhydro_dataset_content_hash"

#: ADR-015 section 8: the skew part of a symmetrised operator, in normalised coordinates, must not exceed this
SKEW_BOUND = 1.0e-2
#: |mass - rho V| <= this x rho V (owner design rule 2)
MASS_BALANCE_RELATIVE = 1.0e-3
#: equilibrium of the declared weight against the baseline buoyancy wrench (as ANYmarine hydro_rigid.EQUILIBRIUM_RELATIVE)
EQUILIBRIUM_RELATIVE = 1.0e-3
#: baseline buoyancy position vs the dataset centre of buoyancy, relative to the length scale
COB_CONSISTENCY_RELATIVE = 1.0e-3
#: OrcaFlex warns when two neighbouring RAO directions are further apart than this
HEADING_GAP_WARNING_DEG = 45.0
#: gravity of OrcaFlex (verified by the O1a probe D5b); only recorded, the exporter uses the declared dataset gravity
ORCAFLEX_GRAVITY = 9.80665
DENSITY_RELATIVE_TOLERANCE = 1.0e-6

UNIT_FACTORS = {
    "mass": 1.0e-3,                       # kg -> te
    "inertia": 1.0e-3,                    # kg m^2 -> te m^2
    "added_mass": 1.0e-3,                 # kg, kg m, kg m^2 -> te, te m, te m^2
    "force": 1.0e-3,                      # N -> kN
    "damping": 1.0e-3,                    # N s/m, N s, N m s/rad -> kN s/m, ...
    "stiffness": 1.0e-3,                  # N/m, N, N m/rad -> kN/m, kN, kN m/rad
    "load_rao": 1.0e-3,                   # N/m, N m/m -> kN/m, kN m/m
    "mean_drift": 1.0e-3,                 # N/m^2, N m/m^2 -> kN/m^2, kN m/m^2
    "displacement_rao_translation": 1.0,  # m/m
    "displacement_rao_rotation": 1.0,     # rad/m, written with RAOResponseUnits = radians
    "length": 1.0,
    "frequency": 1.0,                     # rad/s, written with WavesReferredToBy = frequency (rad/s)
    "phase": "degrees(arg X), no sign change",
    "heading": "degrees(wave_direction) mod 360, no sign change",
}

_TRANSLATIONS = ("surge", "sway", "heave")
_RAO_COLUMNS = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")
_MATRIX_COLUMNS = ("X", "Y", "Z", "Rx", "Ry", "Rz")
_DRIFT_DOFS = ("surge", "sway", "yaw")
_HRP = (2, 3, 4)                          # heave, roll, pitch: the only stiffness OrcaFlex's displacement method takes
_RAO_SETS = ("Wave load RAO", "Displacement RAO", "Wave drift QTF")
_DOF_WORDS = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")

#: A new vessel type holds Orcina's default tanker. These pre-filled load-coefficient blocks are not part of the
#: exported data, so they are set to zero (they act only if the Vessel object enables the effect) and read back.
ZEROED_LOAD_SCALARS = tuple(
    [f"OtherDamping{kind}Coeff{axis}" for kind in ("Linear", "Quadratic") for axis in ("x", "y", "z", "Rx", "Ry", "Rz")]
    + [f"{block}Coeff{word}Area" for block in ("Current", "Wind") for word in ("Surge", "Sway", "Heave")]
    + [f"{block}Coeff{word}AreaMoment" for block in ("Current", "Wind") for word in ("Roll", "Pitch", "Yaw")]
    + [f"CurrentYawRateDrag{word}Factor" for word in ("Surge", "Sway", "Yaw")])
#: coefficient tables: the direction column stays, every coefficient column is zeroed in every row
ZEROED_LOAD_TABLES = {block: [f"{block}{word}" for word in _DOF_WORDS] for block in ("CurrentCoeff", "WindCoeff")}


class OrcaFlexExportError(HydroError):
    """The dataset or declarations cannot be exported to an OrcaFlex vessel type; nothing is converted silently."""


def _fail(message: str):
    raise OrcaFlexExportError(message)


def _finite_float(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        _fail(f"{name} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        _fail(f"{name} must be a finite number")
    if positive and out <= 0.0:
        _fail(f"{name} must be a positive number")
    return out


@dataclass(frozen=True)
class DeclaredMassProperties:
    """Declared mass properties of the vessel (the dataset carries none): SI, world axes.

    ``mass`` kg, ``centre_of_gravity`` (x, y, z) m in world coordinates, ``inertia`` the 3 x 3 inertia tensor in
    kg m^2 about the centre of gravity in world axes. Only diagonal inertia is exportable (see docs/ORCAFLEX_EXPORT.md).
    """

    mass: float
    centre_of_gravity: tuple
    inertia: tuple

    def __post_init__(self) -> None:
        mass = _finite_float(self.mass, "mass", positive=True)
        try:
            cog = tuple(_finite_float(v, "centre_of_gravity") for v in self.centre_of_gravity)
        except TypeError:
            _fail("centre_of_gravity must be three finite numbers")
        if len(cog) != 3:
            _fail("centre_of_gravity must be three finite numbers")
        try:
            tensor = np.array(self.inertia, dtype=float)
        except (TypeError, ValueError):
            _fail("inertia must be a 3 x 3 matrix of finite numbers")
        if tensor.shape != (3, 3) or not np.all(np.isfinite(tensor)):
            _fail("inertia must be a 3 x 3 matrix of finite numbers (finite entries)")
        if not np.allclose(tensor, tensor.T, rtol=1e-9, atol=1e-12 * max(1.0, float(np.abs(tensor).max()))):
            _fail("inertia must be symmetric")
        if np.any(np.diag(tensor) <= 0.0):
            _fail("inertia must have positive diagonal moments (positive)")
        object.__setattr__(self, "mass", mass)
        object.__setattr__(self, "centre_of_gravity", cog)
        object.__setattr__(self, "inertia", tuple(tuple(float(v) for v in row) for row in tensor))

    def as_record(self) -> dict:
        return {"mass_kg": self.mass, "centre_of_gravity_m": list(self.centre_of_gravity),
                "inertia_kg_m2_about_cog_world_axes": [list(row) for row in self.inertia]}


# ---------------------------------------------------------------------------------------------------- pure layer
def _json_attr(dataset, name):
    """A metadata mapping stored as JSON text or a mapping; absent is None, present but undecodable is refused."""
    if name not in dataset.attrs:
        return None
    value = dataset.attrs[name]
    try:
        decoded = json.loads(value) if isinstance(value, (str, bytes, bytearray)) else value
    except ValueError as error:
        _fail(f"the dataset attribute {name!r} is present but is not valid JSON ({error}); it is not treated as absent")
    if not isinstance(decoded, Mapping):
        _fail(f"the dataset attribute {name!r} is present but is not a mapping (found {type(decoded).__name__}); "
              "it is not treated as absent")
    return dict(decoded)


def _skew3(v) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _symmetrise(name: str, matrix, scale: np.ndarray):
    """Declared symmetrisation (ADR-015 section 8): split S X S into symmetric and skew parts in normalised
    coordinates, refuse a skew part above SKEW_BOUND, return the SI symmetric part and the skew ratio."""
    X = np.asarray(matrix, dtype=float)
    Xn = X * np.outer(scale, scale)
    norm = float(np.linalg.norm(Xn))
    ratio = float(np.linalg.norm(0.5 * (Xn - Xn.T))) / norm if norm > 0.0 else 0.0
    if ratio > SKEW_BOUND:
        _fail(f"{name}: skew part {ratio:.3g} of the operator exceeds the declared bound {SKEW_BOUND:g} "
              "(OrcaFlex accepts only symmetric matrices; nothing is symmetrised silently)")
    return 0.5 * (X + X.T), ratio


def _worst(name: str, stack, scale):
    """Symmetrise every frequency sample; returns (symmetric stack, max ratio, index of the max)."""
    out = np.empty_like(stack)
    ratios = []
    for i, sample in enumerate(stack):
        out[i], ratio = _symmetrise(name(i), sample, scale)
        ratios.append(ratio)
    worst = int(np.argmax(ratios))
    return out, float(ratios[worst]), worst


def _resolve_scalar(label: str, found: Mapping[str, Any]):
    """All declarations of a quantity must be numbers and mutually equal; returns the common value."""
    values = {}
    for where, value in found.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)) or not math.isfinite(
                float(value)):
            _fail(f"the {label} declaration of the {where} is not a finite number: {value!r}")
        values[where] = float(value)
    if len(set(values.values())) > 1:
        _fail(f"the {label} declarations disagree: {values}")
    return next(iter(values.values()))


def _check_conventions(dataset) -> None:
    for key, expected in (("phasor_convention", PHASOR_CONVENTION), ("heading_convention", HEADING_CONVENTION),
                          ("length_unit", LENGTH_UNIT), ("force_unit", FORCE_UNIT), ("moment_unit", MOMENT_UNIT)):
        found = dataset.attrs.get(key)
        if found != expected:
            word = key.split("_")[0]
            _fail(f"unknown {word} convention: dataset attribute {key!r} is {found!r}, the exporter proves only {expected!r}")


def _circular_gap(a: float, b: float) -> float:
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _array(dataset, name, dims, keys=None):
    da = dataset[name]
    if keys is not None:
        for dim in ("influenced_dof", "radiating_dof"):
            if dim in da.dims:
                da = da.sel({dim: list(keys)})
    return np.asarray(da.transpose(*dims).values)


def _displacement_input(raos, dataset, keys, nw, nh) -> np.ndarray:
    """The optional displacement RAOs as (omega, heading, 6) complex in dataset order."""
    if hasattr(raos, "dims"):
        if tuple(raos.dims) != ("omega", "wave_direction", "radiating_dof"):
            _fail("displacement_raos as a DataArray needs dims (omega, wave_direction, radiating_dof)")
        if raos["omega"].size != nw or not np.array_equal(np.asarray(raos["omega"].values, float),
                                                          np.asarray(dataset["omega"].values, float)):
            _fail("displacement_raos coordinate omega must equal the dataset's exactly")
        ours = np.mod(np.asarray(dataset["wave_direction"].values, float), 2.0 * np.pi)
        theirs = np.mod(np.asarray(raos["wave_direction"].values, float), 2.0 * np.pi)
        if ours.shape != theirs.shape or not np.allclose(ours, theirs, rtol=0.0, atol=1e-12):
            _fail("displacement_raos coordinate wave_direction must equal the dataset's")
        if sorted(str(k) for k in raos["radiating_dof"].values) != sorted(keys):
            _fail("displacement_raos radiating_dof must be the dataset body's six qualified DOFs")
        values = np.asarray(raos.sel(radiating_dof=list(keys)).values)
    else:
        values = np.asarray(raos)
    try:
        values = values.astype(np.complex128)
    except (TypeError, ValueError):
        _fail("displacement_raos must be complex numbers")
    if values.shape != (nw, nh, 6):
        _fail(f"displacement_raos shape {values.shape} must be (omega, wave_direction, 6) = {(nw, nh, 6)}")
    if not (np.all(np.isfinite(values.real)) and np.all(np.isfinite(values.imag))):
        _fail("displacement_raos must be finite")
    return values


def _motion_hash(values: np.ndarray, omega, wave_direction, keys) -> str:
    """SHA-256 of the canonical displacement-RAO array: little-endian complex128, C order, -0.0 normalised to 0.0, with
    its omega and wave_direction coordinates, DOF keys and shape (pattern of anyhydro.mean_drift)."""
    h = hashlib.sha256()
    h.update(json.dumps({"omega": [repr(float(v)) for v in omega], "wave_direction": [repr(float(v)) for v in wave_direction],
                         "radiating_dof": [str(k) for k in keys], "shape": list(values.shape),
                         "dtype": "complex128 little-endian, C order"}, sort_keys=True).encode())
    h.update(np.ascontiguousarray((values + (0.0 + 0.0j)).astype("<c16")).tobytes())
    return h.hexdigest()


def vessel_type_data(dataset, mass_properties: DeclaredMassProperties, *, length_scale: float, gravity: float,
                     displacement_raos=None, motions_source: str | None = None, include_mean_drift: bool = True) -> dict:
    """Validate, compose and convert a single-body ANYhydro dataset into plain OrcaFlex vessel-type data.

    Returns ``{"format", "units_system", "scalars", "indexed", "matrices", "rao_sets", "added_mass_damping",
    "provenance"}`` in OrcaFlex SI units (te, kN, m, s; angles in degrees in the RAO tables). Pure: no OrcFxAPI.
    Raises :class:`OrcaFlexExportError` for everything that cannot be exported faithfully.

    ``displacement_raos`` (optional) needs ``motions_source``, a non-empty description of where the motions come from; the
    provenance records it together with a SHA-256 of the canonical complex motion array and its coordinates.
    """
    if displacement_raos is None:
        if motions_source is not None:
            _fail("motions_source was given without displacement_raos")
    elif not isinstance(motions_source, str) or not motions_source.strip():
        _fail("displacement_raos need motions_source: a non-empty string declaring where the motions come from")
    if not isinstance(mass_properties, DeclaredMassProperties):
        _fail("mass_properties must be a DeclaredMassProperties record (mass, centre of gravity and inertia are required)")
    length = _finite_float(length_scale, "length_scale", positive=True)
    g = _finite_float(gravity, "gravity", positive=True)

    import xarray as xr
    if not isinstance(dataset, xr.Dataset):
        _fail("dataset must be an xarray.Dataset in the ANYhydro schema")
    _check_conventions(dataset)
    try:
        validate_hydro_dataset(dataset)
    except (HydroError, ValueError, TypeError, KeyError) as error:      # the schema validator can raise raw ValueError
        _fail(f"the dataset is not a valid ANYhydro dataset: {error!s}")
    bodies = [str(b) for b in dataset["body"].values.tolist()]
    if len(bodies) != 1:
        _fail(f"single body only: the dataset has {len(bodies)} bodies (multibody export is not implemented)")
    body = bodies[0]
    try:
        require_global_dof_axes(dataset)
    except HydroError as error:
        _fail(f"the exporter needs GLOBAL DOF axes: {error}")
    keys = list(body_dof_keys(body))
    if sorted(str(k) for k in dataset["influenced_dof"].values) != sorted(keys):
        _fail("the dataset must carry the six rigid-body DOFs of its single body")
    if "added_mass_infinite" not in dataset.data_vars:
        _fail("the dataset has no added_mass_infinite: OrcaFlex needs A_inf (computed or band-consistent) and the last "
              "finite-frequency sample is not a substitute")

    # ---------------------------------------------------------------- hydrostatics: contract, g, rho, V, equilibrium
    try:
        eligibility = require_hydrostatic_composition(dataset)
        contract = read_hydrostatic_contract(dataset)
    except HydroError as error:
        _fail(f"no usable hydrostatic contract: {error}")
    if contract.role == ROLE_UNKNOWN or eligibility.role != contract.role:
        _fail("no usable hydrostatic contract (role UNKNOWN)")
    if contract.role != ROLE_BUOYANCY_ONLY:
        _fail(f"only a BUOYANCY_ONLY hydrostatic contract composes with the declared mass (role {contract.role!r}); "
              "the gravity terms are added here from the declared mass and centre of gravity")
    environment = _json_attr(dataset, "wave_environment") or {}
    gravity_found = {"hydrostatic contract provenance": contract.provenance.get("gravity")}
    if "gravity" not in contract.provenance:
        _fail("the hydrostatic contract provenance declares no gravity: one declared g must equal the dataset's gravity")
    if "g" in environment:
        gravity_found["wave_environment attribute"] = environment["g"]
    if "g" in dataset.data_vars:
        gravity_found["dataset scalar g"] = dataset["g"].values.item()
    dataset_g = _resolve_scalar("gravity", gravity_found)
    if dataset_g != g:
        _fail(f"one declared g: the declared gravity {g!r} must equal the dataset's gravity {dataset_g!r}")
    rho_found = {}
    if "rho" in contract.provenance:
        rho_found["hydrostatic contract provenance"] = contract.provenance["rho"]
    if "rho" in environment:
        rho_found["wave_environment attribute"] = environment["rho"]
    if "rho" in dataset.data_vars:
        rho_found["dataset scalar rho"] = dataset["rho"].values.item()
    if not rho_found:
        _fail("the dataset declares no water density (rho): neither the contract provenance, the wave_environment "
              "attribute nor a scalar rho")
    rho = _resolve_scalar("rho", rho_found)
    if rho <= 0.0:
        _fail("rho must be positive")

    rc = np.asarray(dataset["rotation_center"].sel(body=body).values, dtype=float)
    cob = np.asarray(dataset["center_of_buoyancy"].sel(body=body).values, dtype=float)
    cog = np.array(mass_properties.centre_of_gravity, dtype=float)
    baseline = np.asarray(contract.baseline_wrench[body], dtype=float)
    fz = float(baseline[2])
    if not fz > 0.0:
        _fail("the baseline buoyancy force of the hydrostatic contract must be positive")
    volume = fz / (rho * g)
    displaced_mass = rho * volume
    mass = mass_properties.mass
    imbalance = abs(mass - displaced_mass) / displaced_mass
    if imbalance > MASS_BALANCE_RELATIVE:
        _fail(f"mass {mass:.8g} kg differs from the displaced mass rho V = {displaced_mass:.8g} kg by {imbalance:.3g} "
              f"(limit {MASS_BALANCE_RELATIVE:g}); the exporter writes DisplacedVolume and CentreOfBuoyancy explicitly "
              "and refuses a vessel that is not in vertical balance")
    weight = mass * g
    c = cog - rc
    tau = -weight * np.cross(c, [0.0, 0.0, 1.0])
    force_residual = baseline[:3] + np.array([0.0, 0.0, -weight])
    moment_residual = baseline[3:] + tau
    if (np.linalg.norm(force_residual) > EQUILIBRIUM_RELATIVE * weight
            or np.linalg.norm(moment_residual) > EQUILIBRIUM_RELATIVE * weight * length):
        _fail(f"the vessel is not in equilibrium (declared weight against the baseline buoyancy wrench): |F| "
              f"{np.linalg.norm(force_residual):.6g} N, |M| {np.linalg.norm(moment_residual):.6g} N m (limits "
              f"{EQUILIBRIUM_RELATIVE:g} W and {EQUILIBRIUM_RELATIVE:g} W L)")
    cob_from_baseline = np.array([rc[0] - baseline[4] / fz, rc[1] + baseline[3] / fz])
    cob_discrepancy = float(np.linalg.norm(cob_from_baseline - cob[:2]))
    if cob_discrepancy > COB_CONSISTENCY_RELATIVE * length:
        _fail(f"the dataset centre of buoyancy (x, y) differs from the position implied by the contract baseline wrench "
              f"by {cob_discrepancy:.6g} m (limit {COB_CONSISTENCY_RELATIVE:g} L)")

    inertia = np.array(mass_properties.inertia, dtype=float)
    if np.abs(inertia - np.diag(np.diag(inertia))).max() > 1e-12 * float(np.abs(inertia).max()):
        _fail("non-diagonal inertia (products of inertia) is refused: the OrcaFlex sign convention of products of inertia "
              "is not probed yet")

    # ---------------------------------------------------------------- symmetrised operators
    scale = np.array([1.0 if d in _TRANSLATIONS else 1.0 / length for d in DOF_NAMES])
    omega = np.asarray(dataset["omega"].values, dtype=float)
    A = _array(dataset, "added_mass", ("omega", "influenced_dof", "radiating_dof"), keys)
    B = _array(dataset, "radiation_damping", ("omega", "influenced_dof", "radiating_dof"), keys)
    A_inf = _array(dataset, "added_mass_infinite", ("influenced_dof", "radiating_dof"), keys)
    K_source = _array(dataset, "hydrostatic_stiffness", ("influenced_dof", "radiating_dof"), keys)
    A_s, a_ratio, a_at = _worst(lambda i: f"added mass at omega={omega[i]:g}", A, scale)
    B_s, b_ratio, b_at = _worst(lambda i: f"radiation damping at omega={omega[i]:g}", B, scale)
    A_inf_s, ainf_ratio = _symmetrise("A_inf", A_inf, scale)
    try:
        K_chart = chart_stiffness(contract, body, K_source)
    except HydroError as error:
        _fail(f"the hydrostatic source stiffness cannot be pulled back to the solver chart: {error}")
    K_buoy, k_ratio = _symmetrise("K_buoyancy_chart", K_chart, scale)
    gravity_block = np.zeros((6, 6))
    gravity_block[3:, 3:] = -weight * (c[2] * np.eye(3) - np.outer(c, [0.0, 0.0, 1.0])) - 0.5 * _skew3(tau)
    K_total = K_buoy + gravity_block
    block = np.ix_(_HRP, _HRP)
    K_hrp = K_total[block]
    not_exported = K_total.copy()
    not_exported[block] = 0.0
    dropped_ratio = _dropped_ratio(K_total, not_exported, scale)

    # ---------------------------------------------------------------- headings and tables
    wave_direction = np.asarray(dataset["wave_direction"].values, dtype=float)
    nw, nh_raw = omega.size, wave_direction.size
    X = _array(dataset, "excitation_force", ("omega", "wave_direction", "influenced_dof"), keys).astype(np.complex128)
    drift = None
    drift_status = {"status": "not present in the dataset"}
    if not include_mean_drift:
        drift_status = {"status": "excluded by the caller"}
    elif "mean_drift_far_field" in dataset.data_vars:
        drift = np.asarray(dataset["mean_drift_far_field"].sel(drift_dof=[f"{body}::{d}" for d in _DRIFT_DOFS])
                           .transpose("omega", "wave_direction", "drift_dof").values, dtype=float)
        drift_status = {"status": "exported", "dofs": list(_DRIFT_DOFS), "method": "Newman's approximation (diagonal)",
                        "heave_roll_pitch": "zero (not provided by the far-field method)",
                        "units": "kN/m^2, kN.m/m^2 per unit wave amplitude squared; yaw about the rotation centre"}
    motions = None if displacement_raos is None else _displacement_input(displacement_raos, dataset, keys, nw, nh_raw)
    motion_hash = None if motions is None else _motion_hash(motions, omega, wave_direction, keys)

    degrees = np.degrees(wave_direction) % 360.0
    degrees = np.where(degrees > 360.0 - 1e-9, degrees - 360.0, degrees)
    degrees = np.maximum(degrees, 0.0)
    keep, dropped = [], []
    for k in range(nh_raw):
        twin = next((j for j in keep if _circular_gap(degrees[j], degrees[k]) < 1e-9), None)
        if twin is None:
            keep.append(k)
            continue
        same = True
        for arr in (X, drift, motions):
            if arr is not None:
                # per-quantity (per-DOF) scale: a large coefficient of another DOF cannot hide a disagreement
                for dof in range(arr.shape[-1]):
                    ref = float(np.abs(arr[..., dof]).max()) or 1.0
                    same = same and bool(np.allclose(arr[:, twin, dof], arr[:, k, dof], rtol=1e-9, atol=1e-9 * ref))
        if not same:
            _fail(f"duplicate headings {np.degrees(wave_direction[twin]):g} and {np.degrees(wave_direction[k]):g} degrees "
                  "are equal modulo 360 but their data differ: OrcaFlex rejects equal directions and nothing is merged "
                  "silently")
        dropped.append({"dropped_index": int(k), "kept_index": int(twin), "direction_deg": float(degrees[twin])})
    if len(keep) < 2:
        _fail("OrcaFlex needs at least two distinct headings (a single direction aborts the OrcaFlex run)")
    order = sorted(keep, key=lambda i: degrees[i])
    directions = np.array([degrees[i] for i in order])
    gaps = [(directions[(i + 1) % len(directions)] - directions[i]) % 360.0 or 360.0 for i in range(len(directions))]
    warnings = []
    if max(gaps) > HEADING_GAP_WARNING_DEG + 1e-6:
        warnings.append(f"largest heading gap {max(gaps):g} deg exceeds {HEADING_GAP_WARNING_DEG:g} deg: OrcaFlex warns "
                        "and interpolates complex RAOs linearly between directions")
    warnings.append(f"RAO frequency band {omega[0]:g}..{omega[-1]:g} rad/s: OrcaFlex holds the end value below the band and "
                    "extrapolates linearly in period above it (zero response at infinite frequency is written)")

    def rao_set(values, scale_factor, origin, phase_origin, with_phase=True):
        picked = values[:, order, :]
        amp = np.abs(picked).transpose(1, 0, 2) * scale_factor
        phase = np.degrees(np.angle(picked)).transpose(1, 0, 2) if with_phase else None
        return {"origin": [float(v) for v in origin],
                "phase_origin": None if phase_origin is None else [float(v) for v in phase_origin],
                "directions_deg": directions.copy(), "frequencies": omega.copy(), "amplitude": amp, "phase_deg": phase,
                "infinite_frequency_row": "zero response"}

    world_origin = (0.0, 0.0, 0.0)
    load = rao_set(X, UNIT_FACTORS["load_rao"], rc, world_origin)
    displacement_set = None if motions is None else rao_set(motions, 1.0, rc, world_origin)
    drift_set = None
    if drift is not None:
        full = np.zeros((nw, nh_raw, 6))
        full[:, :, 0], full[:, :, 1], full[:, :, 5] = drift[:, :, 0], drift[:, :, 1], drift[:, :, 2]
        drift_set = rao_set(full.astype(np.complex128), UNIT_FACTORS["mean_drift"], rc, None, with_phase=False)
        drift_set["amplitude"] = full[:, order, :].transpose(1, 0, 2) * UNIT_FACTORS["mean_drift"]   # signed values

    stiffness = K_hrp * UNIT_FACTORS["stiffness"]
    inertia_te = np.diag(inertia) * UNIT_FACTORS["inertia"]
    scalars = {
        "WavesReferredToBy": "frequency (rad/s)", "RAOPhaseConvention": "lags", "RAOPhaseUnitsConvention": "degrees",
        "RAOPhaseRelativeToConvention": "crest", "RAOResponseUnits": "radians", "RAOWaveUnit": "amplitude",
        "SurgePositive": "forward", "SwayPositive": "port", "HeavePositive": "up", "RollPositiveStarboard": "down",
        "PitchPositiveBow": "down", "YawPositiveBow": "port", "Symmetry": "None", "Length": length,
        "Mass": mass * UNIT_FACTORS["mass"],
        "CentreOfMassX": float(cog[0]), "CentreOfMassY": float(cog[1]), "CentreOfMassZ": float(cog[2]),
        "StiffnessInertiaDampingRefOriginx": float(rc[0]), "StiffnessInertiaDampingRefOriginy": float(rc[1]),
        "StiffnessInertiaDampingRefOriginz": float(rc[2]),
        "HydrostaticReferenceOriginDatumPositionz": float(rc[2]),
        "HydrostaticReferenceOriginDatumOrientationx": 0.0, "HydrostaticReferenceOriginDatumOrientationy": 0.0,
        "HydrostaticStiffnessMethod": "Displacement", "DisplacedVolume": float(volume),
        "CentreOfBuoyancyX": float(cob[0]), "CentreOfBuoyancyY": float(cob[1]), "CentreOfBuoyancyZ": float(cob[2]),
        "AMDMethod": "Frequency dependent", "WaveDriftQTFMethod": "Newman's approximation",
    }
    indexed = {
        "MomentOfInertiaTensorX": [float(inertia_te[0])],
        "MomentOfInertiaTensorY": [0.0, float(inertia_te[1])],
        "MomentOfInertiaTensorZ": [0.0, 0.0, float(inertia_te[2])],
        "HydrostaticStiffnessz": [float(stiffness[0, 0])],
        "HydrostaticStiffnessRx": [float(stiffness[0, 1]), float(stiffness[1, 1])],
        "HydrostaticStiffnessRy": [float(stiffness[0, 2]), float(stiffness[1, 2]), float(stiffness[2, 2])],
    }
    provenance = {
        "format": FORMAT,
        "dataset_content_hash": dataset_content_hash(dataset),
        "single_body": body,
        "conventions": {
            "phasor": PHASOR_CONVENTION, "heading": HEADING_CONVENTION,
            "axes": "GLOBAL; vessel axes coincide with the ANYhydro world axes (vessel at the world origin, heading 0)",
            "dataset_units": "m, N, N.m, kg, rad, rad/s",
            "orcaflex": {"units_system": "SI", "waves_referred_to_by": "frequency (rad/s)",
                         "phase": "lags, degrees, relative to the wave crest at the phase origin (world origin)",
                         "rao_response_units": "radians (displacement RAOs)", "symmetry": "None",
                         "heading": "degrees, wave propagation direction from the vessel x towards the vessel y"},
        },
        "unit_factors": dict(UNIT_FACTORS),
        "declared": {"gravity": g, "length_scale": length, "rho": rho, "displaced_volume_m3": float(volume),
                     "mass_properties": mass_properties.as_record(), "rotation_centre": [float(v) for v in rc],
                     "centre_of_buoyancy": [float(v) for v in cob]},
        "mass_balance": {"rho_V_kg": float(displaced_mass), "relative_imbalance": float(imbalance),
                         "limit": MASS_BALANCE_RELATIVE},
        "hydrostatic_composition": {
            "role": contract.role, "force_basis": contract.force_basis, "rate_map": contract.rate_map,
            "contract_source": str(contract.provenance.get("source")),
            "rule": "K = symmetric part of the chart-converted buoyancy stiffness + gravity block (ADR-004) of the "
                    "declared mass and centre of gravity; heave/roll/pitch block only",
            "gravity_roll_pitch_diagonal_kN": float(-weight * c[2] * UNIT_FACTORS["force"]),
            "equilibrium_residual_force_N": float(np.linalg.norm(force_residual)),
            "equilibrium_residual_moment_Nm": float(np.linalg.norm(moment_residual)),
            "centre_of_buoyancy_baseline_discrepancy_m": cob_discrepancy,
            "not_exported_entries_frobenius_ratio": dropped_ratio,
            "not_exported": "surge, sway and yaw rows and columns (moorings belong in the OrcaFlex model)",
        },
        "skew_ratios": {"bound": SKEW_BOUND, "K_buoyancy_chart": k_ratio, "A_max_over_frequency": a_ratio,
                        "A_argmax_omega": float(omega[a_at]), "B_max_over_frequency": b_ratio,
                        "B_argmax_omega": float(omega[b_at]), "A_inf": ainf_ratio,
                        "normalisation": "translations 1, rotations 1/length_scale"},
        "headings": {"wave_direction_rad_in": [float(v) for v in wave_direction],
                     "direction_deg": [float(v) for v in directions], "dropped_duplicates": dropped},
        "omega_band_rad_s": [float(omega[0]), float(omega[-1])],
        "mean_drift": drift_status,
        "displacement_raos": ("not supplied" if motions is None else {
            "units": "m/m translations, rad/m rotations about the rotation centre", "phase": "arg X, lags",
            "source": motions_source.strip(), "motion_hash": motion_hash, "hash_algorithm": "sha256",
            "hash_covers": "omega, wave_direction, radiating_dof, shape and the complex128 little-endian C-order array",
            "radiating_dof": list(keys)}),
        "load_coefficient_blocks": {
            "status": "zeroed, not exported",
            "reason": "a new vessel type holds Orcina's default tanker; no default data may act if a user enables the effect",
            "zeroed_scalars": list(ZEROED_LOAD_SCALARS),
            "zeroed_tables": {block: list(names) for block, names in ZEROED_LOAD_TABLES.items()}},
        "added_mass_infinite": str(dataset["added_mass_infinite"].attrs.get("source", "")) or "dataset added_mass_infinite",
        "not_exported": ["surge/sway/yaw hydrostatic stiffness", "full QTFs and sum-frequency QTFs",
                         "current, wind and other damping coefficients (zeroed, not exported)", "vessel-level (Vessel object) settings"],
        "warnings": warnings,
        "orcaflex_gravity_m_s2": ORCAFLEX_GRAVITY,
        "orcaflex_gravity_relative_difference": abs(g - ORCAFLEX_GRAVITY) / ORCAFLEX_GRAVITY,
    }
    return {
        "format": FORMAT, "units_system": "SI", "scalars": scalars, "indexed": indexed,
        "matrices": {"hydrostatic_stiffness": stiffness},
        "zeroed_load_blocks": {"scalars": {name: 0.0 for name in ZEROED_LOAD_SCALARS},
                               "tables": {block: list(names) for block, names in ZEROED_LOAD_TABLES.items()}},
        "rao_sets": {"Wave load RAO": load, "Displacement RAO": displacement_set, "Wave drift QTF": drift_set},
        "added_mass_damping": {
            "frequencies": omega.copy(), "added_mass": A_s * UNIT_FACTORS["added_mass"],
            "damping": B_s * UNIT_FACTORS["damping"], "added_mass_infinite": A_inf_s * UNIT_FACTORS["added_mass"],
            "infinite_frequency_row": "added_mass_infinite, damping zero"},
        "provenance": provenance,
    }


# ---------------------------------------------------------------------------------------------------- writer
def _orcfxapi():
    try:
        import OrcFxAPI
    except ImportError as error:
        raise OrcaFlexExportError("OrcFxAPI (licensed OrcaFlex 11.6c) is not available: writing a vessel type needs it") from error
    return OrcFxAPI


def _zero_set(with_phase: bool) -> dict:
    """The reset of an unused RAO set: OrcaFlex pre-fills a new vessel type with Orcina's default vessel data."""
    return {"origin": None, "phase_origin": None, "directions_deg": np.array([0.0, 180.0]), "frequencies": np.array([1.0]),
            "amplitude": np.zeros((2, 1, 6)), "phase_deg": np.zeros((2, 1, 6)) if with_phase else None}


def _set_upper(vt, prefix: str, matrix) -> None:
    """Only the upper triangle of symmetric matrices is editable (rows <= column)."""
    for j, column in enumerate(_MATRIX_COLUMNS):
        for i in range(j + 1):
            vt.SetData(prefix + column, i, float(matrix[i][j]))


def _effective_rao_sets(data: dict) -> dict:
    """The three RAO sets exactly as written: unused sets are the zero tables, origins are resolved; the drift QTF has no
    phase origin. The writer and the read-back both use this one description."""
    origin = data["rao_sets"]["Wave load RAO"]["origin"]
    out = {}
    for which in _RAO_SETS:
        spec = data["rao_sets"][which]
        if spec is None:
            spec = _zero_set(which != "Wave drift QTF")
        phase_origin = None
        if which != "Wave drift QTF":
            phase_origin = spec["phase_origin"] if spec["phase_origin"] is not None else (0.0, 0.0, 0.0)
        out[which] = dict(spec, origin=[float(v) for v in (spec["origin"] or origin)],
                          phase_origin=None if phase_origin is None else [float(v) for v in phase_origin])
    return out


def _write_rao_set(o, vt, which: str, spec: dict) -> None:
    vt.SelectedRAOs = which
    vt.RAOOriginX, vt.RAOOriginY, vt.RAOOriginZ = spec["origin"]
    if spec["phase_origin"] is not None:
        vt.RAOPhaseOriginX, vt.RAOPhaseOriginY, vt.RAOPhaseOriginZ = spec["phase_origin"]
    directions = [float(d) for d in spec["directions_deg"]]
    vt.NumberOfRAODirections = len(directions)
    for i in range(len(directions)):             # park on unique values first: OrcaFlex refuses duplicate directions
        vt.SelectedRAODirectionIndex = i
        vt.RAODirection = -0.0371 - 0.0113 * i
    for i, d in enumerate(directions):
        vt.SelectedRAODirectionIndex = i
        vt.RAODirection = d
    frequencies = [float(w) for w in spec["frequencies"]]
    with_phase = spec["phase_deg"] is not None
    for k in range(len(directions)):
        vt.SelectedRAODirectionIndex = k
        vt.SetDataRowCount("RAOPeriodOrFrequency", len(frequencies) + 1)
        for r, w in enumerate(frequencies):
            vt.SetData("RAOPeriodOrFrequency", r, w)
            for c, name in enumerate(_RAO_COLUMNS):
                vt.SetData(f"RAO{name}Amp", r, float(spec["amplitude"][k, r, c]))
                if with_phase:
                    vt.SetData(f"RAO{name}Phase", r, float(spec["phase_deg"][k, r, c]))
        last = len(frequencies)
        vt.SetData("RAOPeriodOrFrequency", last, o.OrcinaInfinity())         # infinite frequency: zero response
        for name in _RAO_COLUMNS:
            vt.SetData(f"RAO{name}Amp", last, 0.0)
            if with_phase:
                vt.SetData(f"RAO{name}Phase", last, 0.0)


def _dropped_ratio(K_total, not_exported, scale) -> float:
    """Normalised Frobenius share of the stiffness not written; a neutral system (zero total stiffness) drops
    nothing and records 0, never a NaN (review round 2)."""
    S = np.outer(scale, scale)
    total = float(np.linalg.norm(K_total * S))
    return float(np.linalg.norm(not_exported * S)) / total if total > 0.0 else 0.0


def _close(a, b) -> bool:
    # non-finite read-back values never match (inf - finite and NaN comparisons fail open otherwise; review round 2)
    return bool(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-9 * max(abs(a), abs(b)) + 1e-12)


def _same_value(a: float, b: float) -> bool:
    return a == b or _close(a, b)


def _match(expected, actual):
    """One-to-one match by value (OrcaFlex re-sorts tables): returns (index map, unmatched expected, unused actual)."""
    free = list(range(len(actual)))
    found, missing = {}, []
    for k, e in enumerate(expected):
        hit = next((i for i in free if _same_value(float(actual[i]), float(e))), None)
        if hit is None:
            missing.append(k)
        else:
            found[k] = hit
            free.remove(hit)
    return found, missing, free


def _verify_rao_set(vt, which: str, spec: dict, infinity: float, problems: list) -> None:
    vt.SelectedRAOs = which
    got_origin = [float(vt.RAOOriginX), float(vt.RAOOriginY), float(vt.RAOOriginZ)]
    if not all(_close(g, w) for g, w in zip(got_origin, spec["origin"])):
        problems.append(f"{which}: response origin wrote {spec['origin']}, read {got_origin}")
    if spec["phase_origin"] is not None:
        got_phase_origin = [float(vt.RAOPhaseOriginX), float(vt.RAOPhaseOriginY), float(vt.RAOPhaseOriginZ)]
        if not all(_close(g, w) for g, w in zip(got_phase_origin, spec["phase_origin"])):
            problems.append(f"{which}: phase origin wrote {spec['phase_origin']}, read {got_phase_origin}")
    directions = []
    for i in range(vt.NumberOfRAODirections):
        vt.SelectedRAODirectionIndex = i
        directions.append(float(vt.RAODirection))
    wanted = [float(d) for d in spec["directions_deg"]]
    if len(directions) != len(wanted):
        problems.append(f"{which}: directions wrote {len(wanted)}, read {len(directions)}")
    mapping, missing, extra = _match(wanted, directions)
    problems.extend(f"{which}: directions {wanted[k]:g} not found after writing" for k in missing)
    if extra and len(directions) == len(wanted):
        problems.append(f"{which}: directions read back contain unexpected values")
    wanted_frequencies = [float(w) for w in spec["frequencies"]] + [infinity]
    with_phase = spec["phase_deg"] is not None
    for k, index in mapping.items():
        d = wanted[k]
        vt.SelectedRAODirectionIndex = index
        rows = [float(vt.GetData("RAOPeriodOrFrequency", r)) for r in range(vt.GetDataRowCount("RAOPeriodOrFrequency"))]
        if len(rows) != len(wanted_frequencies):
            problems.append(f"{which} {d:g} deg: frequencies wrote {len(wanted_frequencies)} rows, read {len(rows)}")
        found, absent, _ = _match(wanted_frequencies, rows)
        problems.extend(f"{which} {d:g} deg: frequencies {wanted_frequencies[j]:g} not found after writing" for j in absent)
        for j, r in found.items():
            finite = j < len(spec["frequencies"])
            w = wanted_frequencies[j]
            for c, name in enumerate(_RAO_COLUMNS):
                want_amp = float(spec["amplitude"][k, j, c]) if finite else 0.0
                if not _close(float(vt.GetData(f"RAO{name}Amp", r)), want_amp):
                    problems.append(f"{which} {d:g} deg w={w:g} {name} amplitude")
                if with_phase and want_amp != 0.0:           # at zero amplitude the phase carries no information
                    want_z = np.exp(1j * math.radians(float(spec["phase_deg"][k, j, c])))
                    got_z = np.exp(1j * math.radians(float(vt.GetData(f"RAO{name}Phase", r))))
                    if not np.isfinite(got_z) or abs(got_z - want_z) > 1e-8:
                        problems.append(f"{which} {d:g} deg w={w:g} {name} phase")


def _verify_added_mass_damping(vt, amd: dict, infinity: float, problems: list) -> None:
    count = vt.NumberOfAddedMassAndDampingPeriodsOrFrequencies
    periods = [float(vt.GetData("AMDPeriodOrFrequency", i)) for i in range(count)]
    wanted = [infinity] + [float(w) for w in amd["frequencies"]]
    tables = [(amd["added_mass_infinite"], np.zeros((6, 6)))] + [
        (amd["added_mass"][j], amd["damping"][j]) for j in range(len(amd["frequencies"]))]
    if count != len(wanted):
        problems.append(f"added mass and damping frequencies wrote {len(wanted)}, read {count}")
    found, absent, _ = _match(wanted, periods)
    problems.extend(f"added mass and damping frequencies {wanted[j]:g} not found after writing" for j in absent)
    for j, i in found.items():
        vt.SelectedAddedMassAndDampingPeriodOrFrequencyIndex = i
        label = "at infinite frequency" if j == 0 else f"w={wanted[j]:g}"
        for kind, prefix, matrix in (("added mass", "AddedMassMatrix", tables[j][0]), ("damping", "Damping", tables[j][1])):
            for c, column in enumerate(_MATRIX_COLUMNS):
                for r in range(6):
                    if not _close(float(vt.GetData(prefix + column, r)), float(matrix[r][c])):
                        problems.append(f"{kind} {label} [{r},{c}]")


def _verify_written(o, vt, data: dict) -> None:
    """Read EVERYTHING written back through OrcFxAPI and compare: scalars, indexed items, the zeroed load blocks, every RAO
    set (origins, phase origin, directions, frequencies, amplitudes, phases as unit phasors; unused sets must be the zero
    tables) and every added-mass and damping row including the infinite-frequency row. Tables are matched by value because
    OrcaFlex re-sorts them; exact membership is required (no extra and no missing directions or frequencies)."""
    vt.SelectedDraught = vt.GetData("DraughtName", 0)
    problems = []
    for name, value in data["scalars"].items():
        got = vt.GetData(name, -1)
        ok = (str(got).strip().lower() == value.lower()) if isinstance(value, str) else _close(float(got), float(value))
        if not ok:
            problems.append(f"{name}: wrote {value!r}, read {got!r}")
    for name, rows in data["indexed"].items():
        for i, value in enumerate(rows):
            if not _close(float(vt.GetData(name, i)), float(value)):
                problems.append(f"{name}[{i}]: wrote {value!r}, read {vt.GetData(name, i)!r}")
    for name in data["zeroed_load_blocks"]["scalars"]:
        if float(vt.GetData(name, -1)) != 0.0:
            problems.append(f"zeroed load coefficient {name} reads {vt.GetData(name, -1)!r}")
    for block, names in data["zeroed_load_blocks"]["tables"].items():
        for column in names:
            for r in range(vt.GetDataRowCount(block + "Direction")):
                if float(vt.GetData(column, r)) != 0.0:
                    problems.append(f"zeroed load coefficient table {column}[{r}] reads {vt.GetData(column, r)!r}")
    # impulse-response integration step must be OrcaFlex's '~' (= the simulation time step); O1c measured 8 % / 6.6 deg
    # first-order errors with 0.2 s against a 0.05 s simulation step, 2.2 % / 1.8 deg with '~'
    amd_time_step = float(vt.GetData("AMDTimeStep", -1))
    if amd_time_step != float(o.OrcinaDefaultReal()):
        problems.append(f"AMDTimeStep: wrote '~' (the simulation time step), read {amd_time_step!r}")
    infinity = o.OrcinaInfinity()
    for which, spec in _effective_rao_sets(data).items():
        _verify_rao_set(vt, which, spec, infinity, problems)
    _verify_added_mass_damping(vt, data["added_mass_damping"], infinity, problems)
    if problems:
        _fail("read-back after writing differs from the written data: " + "; ".join(problems[:8]))


def write_vessel_type(model, data: dict, *, name: str):
    """Create a vessel type ``name`` in an OrcaFlex ``model`` from :func:`vessel_type_data` output; returns it.

    The model must use the OrcaFlex unit system SI, its water density must equal the dataset's, and ``name`` must be
    new. Every item is set (the new type holds Orcina's default vessel, so unused RAO sets are reset to zero tables),
    then read back. The provenance JSON is stored in the vessel type's tags ``anyhydro_provenance`` and the dataset hash
    in ``anyhydro_dataset_content_hash``. A failed write destroys the half-written object.
    """
    o = _orcfxapi()
    if not isinstance(name, str) or not name.strip():
        _fail("name must be a non-empty string")
    if data.get("format") != FORMAT:
        _fail(f"data is not {FORMAT} output of vessel_type_data")
    if str(model.general.UnitsSystem) != "SI":
        _fail(f"the model unit system is {model.general.UnitsSystem!r}; the export is written in OrcaFlex SI (te, kN, m, s)")
    rho = data["provenance"]["declared"]["rho"]
    density = float(model.environment.Density) * 1000.0
    if abs(density - rho) > DENSITY_RELATIVE_TOLERANCE * rho:
        _fail(f"the model water density {density:g} kg/m^3 differs from the dataset's rho {rho:g}: set "
              "environment.Density (te/m^3) first, the explicit DisplacedVolume balances the mass only at that density")
    try:
        model[name]
    except Exception:                                            # noqa: BLE001 - OrcFxAPI raises DLLError for unknown names
        pass
    else:
        _fail(f"an object named {name!r} already exists in the model (name must be new)")
    vt = None
    try:
        vt = model.CreateObject(o.ObjectType.VesselType, name)
        if vt.NumberOfDraughts != 1:
            vt.NumberOfDraughts = 1
        vt.SelectedDraught = vt.GetData("DraughtName", 0)
        for item, value in data["scalars"].items():
            setattr(vt, item, value)
        for item, rows in data["indexed"].items():
            for i, value in enumerate(rows):
                vt.SetData(item, i, float(value))
        for item, value in data["zeroed_load_blocks"]["scalars"].items():
            setattr(vt, item, float(value))
        for block, names in data["zeroed_load_blocks"]["tables"].items():
            for column in names:
                for r in range(vt.GetDataRowCount(block + "Direction")):
                    vt.SetData(column, r, 0.0)
        for which, spec in _effective_rao_sets(data).items():
            _write_rao_set(o, vt, which, spec)
        amd = data["added_mass_damping"]
        frequencies = [o.OrcinaInfinity()] + [float(w) for w in amd["frequencies"]]
        vt.NumberOfAddedMassAndDampingPeriodsOrFrequencies = len(frequencies)
        for i, w in enumerate(frequencies):
            vt.SetData("AMDPeriodOrFrequency", i, w)
        for i in range(len(frequencies)):
            vt.SelectedAddedMassAndDampingPeriodOrFrequencyIndex = i
            _set_upper(vt, "AddedMassMatrix", amd["added_mass_infinite"] if i == 0 else amd["added_mass"][i - 1])
            _set_upper(vt, "Damping", np.zeros((6, 6)) if i == 0 else amd["damping"][i - 1])
        vt.AMDTimeStep = o.OrcinaDefaultReal()                  # '~': the convolution uses the simulation time step
        _verify_written(o, vt, data)
        stored = dict(data["provenance"])
        stored["writer"] = {"OrcFxAPI_DLL_version": o.DLLVersion(), "vessel_type_name": name,
                            "provenance_tag": PROVENANCE_TAG, "hash_tag": HASH_TAG,
                            "model_density_te_per_m3": density * 1e-3,
                            "impulse_response": _impulse_response_record(o, vt)}
        vt.tags[PROVENANCE_TAG] = json.dumps(stored, sort_keys=True)
        vt.tags[HASH_TAG] = data["provenance"]["dataset_content_hash"]
    except OrcaFlexExportError:
        _discard(model, vt)
        raise
    except Exception as error:                                   # noqa: BLE001 - OrcFxAPI error types differ by call
        _discard(model, vt)
        raise OrcaFlexExportError(f"OrcFxAPI refused the vessel type data: {str(error).strip().splitlines()[-1][:240]}") from error
    return vt


def _impulse_response_record(o, vt) -> dict:
    """The convolution controls as written: the time step is always '~'; the cutoff time and tolerance are OrcaFlex's."""
    default = float(o.OrcinaDefaultReal())

    def shown(value):
        return "~" if float(value) == default else float(value)
    return {"AMDTimeStep": "~ (the simulation time step: implicit constant step or explicit outer step)",
            "AMDCutoffTime": shown(vt.GetData("AMDCutoffTime", -1)),
            "AMDCutoffTolerance": shown(vt.GetData("AMDCutoffTolerance", -1))}


def _discard(model, vt) -> None:
    if vt is not None:
        try:
            model.DestroyObject(vt)
        except Exception:                                        # noqa: BLE001 - best effort cleanup
            pass


def export_vessel_type(model, dataset, mass_properties: DeclaredMassProperties, *, name: str, **options):
    """:func:`vessel_type_data` followed by :func:`write_vessel_type`; options go to the pure function."""
    return write_vessel_type(model, vessel_type_data(dataset, mass_properties, **options), name=name)
