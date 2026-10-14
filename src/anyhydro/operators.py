"""Derived retained-coordinate hydrodynamic operators (development slice).

A :class:`DerivedHydroOperators` artifact projects a validated raw full-6N
dataset onto an explicitly ordered retained DOF subspace and records the
projection, the raw retained arrays, the derived arrays and the complete
criteria report. It is a separate unqualified artifact: it is never a raw
solver dataset relabelled, and it confers no scientific qualification.

Conventions (docs/FIRST_SLICE_OPERATOR_CONTRACT.md, adopted 2026-10-08):

- ``q_full = P q_retained`` and ``Q_retained = P^T Q_full`` with ``P`` an
  exact 0/1 selector built from source DOF labels (no numeric positions).
- Retained operators are ``P^T X P`` and retained excitation ``P^T F``;
  full coupled matrices are preserved, never truncated blocks of a
  reordered dataset.
- Normalization is strictly the congruence ``X_hat = S^T X S`` for a
  declared fixed positive diagonal scale ``S``; virtual work
  ``Q^T q = Q_hat^T q_hat`` is preserved.
- The correction policy is explicit: ``"none"`` keeps the raw retained
  operators; ``"reciprocal_average"`` replaces A, B and A_inf by
  ``(X + X^T)/2`` only; ``"reciprocal_average_far_field_weak_modes"`` does
  the same and then, per frequency in normalized coordinates, replaces the
  symmetric damping on its weak subspace (eigenvalues below
  ``WEAK_MODE_FRACTION`` times the largest) by the far-field (Kochin
  energy-flux) damping compressed onto that subspace. The
  far-field damping is positive semidefinite by construction; the weak
  modes are those the pressure-integrated damping cannot resolve (W5
  evidence w5-physical-20261012). Energetic modes, A and A_inf are the
  reciprocal average. Hydrostatics and excitation are never modified,
  nothing is clamped, tuned or fitted, and there is no default policy.
- Admission criteria are caller-declared, finite and nonnegative; there
  are no scientific defaults. Failed criteria retain the artifact and
  report for inspection but make radiation consumption a typed
  non-success.
- Every admission criterion operates in the declared normalized
  coordinates ``X_hat = S^T X S``: reciprocity, correction and symmetry
  residuals, and the retained B PSD eigenvalues with their negative
  modes, are all computed on normalized arrays, so each absolute
  threshold is interpreted in that operator's own normalized unit (A_hat
  and B_hat units differ; there is no mixing). A normalized-coordinate
  eigenvector ``v_hat`` maps to the physical retained direction
  ``q = S v_hat``. The mechanical stored arrays (raw and derived
  A/B/Ainf, hydrostatics, excitation) stay in SI retained coordinates.
- Raw-full diagnostics (excluded-DOF defects included) are recorded
  separately from retained diagnostics in unscaled mixed-SI coordinates;
  they are informational only, never feed the retained pass/fail and are
  never read as normalized severity measures.

Base import touches only numpy, xarray and the standard library (plus
anyhydro base modules) - never Qt, ANYsolver or Capytaine.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import xarray as xr

from .dof import DOF_NAMES, parse_dof_key
from .errors import HydroError
from .hashing import dataset_content_hash
from .schema import validate_for_time_domain

DERIVED_OPERATORS_MODULE_VERSION = "1.1.0"
DERIVED_OPERATORS_ARTIFACT_FORMAT = "anyhydro-derived-operators-v2"

# Development-slice hard boundaries, mirroring the radiation caps
# (radiation.MAX_DOFS / radiation.MAX_INPUT_FREQUENCIES). Duplicated here
# so this module never imports the radiation module (which imports this
# one). Rejection happens before any large allocation.
MAX_SOURCE_DOFS = 12
MAX_SOURCE_FREQUENCIES = 513

#: Coordinate label carried by the raw-full diagnostic section. Those
#: arrays are computed on the raw full source in unscaled mixed SI
#: coordinates (they lack the retained normalization S); they are
#: informational only and are never normalized admission or severity
#: measures.
RAW_FULL_DIAGNOSTIC_COORDINATES = "unscaled_mixed_si_informational"

CORRECTION_NONE = "none"
CORRECTION_RECIPROCAL_AVERAGE = "reciprocal_average"
CORRECTION_FAR_FIELD_WEAK_MODES = "reciprocal_average_far_field_weak_modes"
CORRECTION_POLICIES = (CORRECTION_NONE, CORRECTION_RECIPROCAL_AVERAGE,
                       CORRECTION_FAR_FIELD_WEAK_MODES)
#: Policies that consume the dataset's ``radiation_damping_far_field``.
FAR_FIELD_POLICIES = (CORRECTION_FAR_FIELD_WEAK_MODES,)
#: Weak damping modes: eigenvalue below this fraction of the largest, in the
#: declared normalized coordinates (same threshold as the adopted W5 criteria).
WEAK_MODE_FRACTION = 1e-3

#: Report sections per operator matrix (order is part of the artifact hash).
REPORT_MATRIX_KEYS = ("added_mass", "damping", "added_mass_infinite")


class DerivedOperatorError(HydroError):
    """A derived-operator input, artifact or report is invalid."""


class DerivedOperatorSettingsError(DerivedOperatorError):
    """Derived-operator settings are invalid or incomplete."""


class DerivedOperatorCriteriaError(DerivedOperatorError):
    """Caller-declared admission criteria are invalid."""


class DerivedOperatorLimitError(DerivedOperatorError):
    """A development-slice hard boundary (DOFs, frequencies) was exceeded."""


class DerivedOperatorSerializationError(DerivedOperatorError):
    """A derived-operator artifact could not be saved, loaded or verified."""


class DerivedOperatorAdmissionError(DerivedOperatorError):
    """Consumption of a non-admitted derived-operator artifact."""


def _require(condition: bool, error: Exception) -> None:
    if not condition:
        raise error


def _frozen_float_array(value, name: str, ndim: Optional[int] = None) -> np.ndarray:
    """Defensive copy as float64, finite-checked, truly read-only."""
    try:
        arr = np.array(value, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise DerivedOperatorSettingsError(f"{name} is not a numeric array: {exc}") from exc
    if ndim is not None and arr.ndim != ndim:
        raise DerivedOperatorSettingsError(
            f"{name} must have {ndim} dimension(s), got shape {arr.shape}")
    if arr.size and not np.all(np.isfinite(arr)):
        raise DerivedOperatorSettingsError(f"{name} contains non-finite values")
    if arr.size == 0:
        arr.setflags(write=False)
        return arr
    return np.frombuffer(arr.tobytes(), dtype=np.float64).reshape(arr.shape)


def _frozen_complex_array(value, name: str, ndim: Optional[int] = None) -> np.ndarray:
    """Defensive copy as complex128, finite-checked, truly read-only."""
    try:
        arr = np.array(value, dtype=np.complex128, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise DerivedOperatorSettingsError(f"{name} is not a numeric array: {exc}") from exc
    if ndim is not None and arr.ndim != ndim:
        raise DerivedOperatorSettingsError(
            f"{name} must have {ndim} dimension(s), got shape {arr.shape}")
    if arr.size and not (np.all(np.isfinite(arr.real)) and np.all(np.isfinite(arr.imag))):
        raise DerivedOperatorSettingsError(f"{name} contains non-finite values")
    if arr.size == 0:
        arr.setflags(write=False)
        return arr
    return np.frombuffer(arr.tobytes(), dtype=np.complex128).reshape(arr.shape)


def _deep_freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_dof_labels(labels, name: str) -> Tuple[str, ...]:
    """Ordered retained DOF labels: non-empty, unique, body-qualified."""
    if isinstance(labels, str) or not isinstance(labels, (list, tuple, np.ndarray)):
        raise DerivedOperatorSettingsError(
            f"{name} must be a sequence of body-qualified DOF labels "
            f"('body_id::dof'), got {type(labels).__name__}")
    out = []
    for label in labels:
        if not isinstance(label, str):
            raise DerivedOperatorSettingsError(
                f"{name} entries must be strings, got {type(label).__name__}")
        try:
            parse_dof_key(label)
        except ValueError as exc:
            raise DerivedOperatorSettingsError(f"{name}: {exc}") from exc
        out.append(label)
    _require(len(out) > 0, DerivedOperatorSettingsError(f"{name} must not be empty"))
    _require(len(set(out)) == len(out),
             DerivedOperatorSettingsError(f"{name} contains duplicate DOF labels"))
    return tuple(out)


@dataclass(frozen=True)
class NormalizationSpec:
    """Declared fixed positive diagonal scale ``S`` with a named convention.

    ``q = S q_hat`` and ``Q_hat = S^T Q``; the congruence applied to
    operators is strictly ``X_hat = S^T X S``. The convention name and unit
    convention are declared prose recorded in the artifact; the scales are
    per retained DOF and must be strictly positive and finite.
    """

    name: str
    unit_convention: str
    scales: np.ndarray

    def __post_init__(self) -> None:
        for field in ("name", "unit_convention"):
            value = getattr(self, field)
            _require(isinstance(value, str) and value.strip(),
                     DerivedOperatorSettingsError(
                         f"normalization {field} must be a non-empty string"))
            object.__setattr__(self, field, str(value))
        raw = self.scales
        if isinstance(raw, (str, bytes)) or raw is None:
            raise DerivedOperatorSettingsError(
                "normalization scales must be a sequence of positive floats")
        if not isinstance(raw, np.ndarray):
            try:
                items = list(raw)
            except TypeError:
                items = None
            if items is not None and any(
                    isinstance(v, (bool, np.bool_)) for v in items):
                raise DerivedOperatorSettingsError(
                    "normalization scales must be floats, not bools")
        try:
            probe = np.asarray(raw)
        except (TypeError, ValueError) as exc:
            raise DerivedOperatorSettingsError(
                f"normalization scales are not a numeric sequence: {exc}") from exc
        _require(probe.dtype.kind != "b", DerivedOperatorSettingsError(
            "normalization scales must be floats, not bools"))
        _require(not (probe.dtype == object and
                      any(isinstance(v, (bool, np.bool_)) for v in probe.ravel().tolist())),
                 DerivedOperatorSettingsError(
                     "normalization scales must be floats, not bools"))
        scales = _frozen_float_array(raw, "normalization scales", ndim=1)
        _require(scales.size > 0, DerivedOperatorSettingsError(
            "normalization scales must not be empty"))
        _require(bool(np.all(np.isfinite(scales))), DerivedOperatorSettingsError(
            "normalization scales must be strictly positive and finite"))
        _require(bool(np.all(scales > 0.0)), DerivedOperatorSettingsError(
            "normalization scales must be strictly positive and finite"))
        object.__setattr__(self, "scales", scales)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "unit_convention": self.unit_convention,
            "scales": [float(v) for v in self.scales.tolist()],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NormalizationSpec":
        if not isinstance(data, Mapping):
            raise DerivedOperatorSettingsError(
                "serialized normalization must be a mapping")
        unknown = set(data) - {"name", "unit_convention", "scales"}
        if unknown:
            raise DerivedOperatorSettingsError(
                f"serialized normalization contains unknown key(s): {sorted(unknown)}")
        try:
            return cls(name=data["name"], unit_convention=data["unit_convention"],
                       scales=data["scales"])
        except KeyError as exc:
            raise DerivedOperatorSettingsError(
                f"serialized normalization is missing field {exc}") from exc


@dataclass(frozen=True)
class DerivedOperatorCriteria:
    """Caller-declared admission criteria (all fields required, no defaults).

    All thresholds are finite and nonnegative; the near-zero floors are
    strictly positive. Every criterion operates in the declared
    normalized coordinates ``X_hat = S^T X S``: each absolute threshold
    is interpreted in that operator's own normalized unit (the A_hat and
    B_hat units differ; there is no cross-operator mixing). Thresholds
    combine an absolute term and a relative term against a declared
    per-matrix global normalization:

        threshold = abs_tol + rel_tol * max(global_matrix_scale, floor)

    where the global matrix scale is the largest normalized magnitude in
    the matrix concerned (its own normalized unit). No
    coefficient-by-coefficient division ever occurs.

    - ``reciprocity_*``: raw normalized reciprocity defects
      ``|X_hat - X_hat^T|`` of the raw retained A, B and A_inf, complete
      and frequency-indexed.
    - ``correction_*``: normalized correction magnitudes
      ``|X_hat_derived - X_hat_raw|`` (exactly zero for policy ``none``).
    - ``derived_symmetry_tol``: absolute symmetry tolerance of the
      derived retained operators in normalized coordinates.
    - ``damping_psd_tol``: eigenvalue tolerance of the derived retained
      B in normalized coordinates (``B_hat = S^T B S``). The stored
      mechanical arrays stay SI; only the admission arithmetic moves to
      the declared normalized coordinates, because a finite numeric
      threshold is not invariant under congruence even though the
      eigenvalue sign is.
    """

    reciprocity_abs_tol: float
    reciprocity_rel_tol: float
    reciprocity_floor: float
    correction_abs_tol: float
    correction_rel_tol: float
    correction_floor: float
    derived_symmetry_tol: float
    damping_psd_tol: float

    def __post_init__(self) -> None:
        for name in ("reciprocity_abs_tol", "reciprocity_rel_tol",
                     "correction_abs_tol", "correction_rel_tol",
                     "derived_symmetry_tol", "damping_psd_tol"):
            value = getattr(self, name)
            _require(not isinstance(value, (bool, np.bool_)),
                     DerivedOperatorCriteriaError(f"{name} must be a float, not a bool"))
            try:
                tol = float(value)
            except (TypeError, ValueError) as exc:
                raise DerivedOperatorCriteriaError(f"{name} must be a float: {exc}") from exc
            _require(np.isfinite(tol) and tol >= 0.0,
                     DerivedOperatorCriteriaError(
                         f"{name} must be finite and >= 0 (explicit threshold required)"))
            object.__setattr__(self, name, tol)
        for name in ("reciprocity_floor", "correction_floor"):
            value = getattr(self, name)
            _require(not isinstance(value, (bool, np.bool_)),
                     DerivedOperatorCriteriaError(f"{name} must be a float, not a bool"))
            try:
                floor = float(value)
            except (TypeError, ValueError) as exc:
                raise DerivedOperatorCriteriaError(f"{name} must be a float: {exc}") from exc
            _require(np.isfinite(floor) and floor > 0.0,
                     DerivedOperatorCriteriaError(
                         f"{name} must be finite and > 0 (explicit near-zero floor required)"))
            object.__setattr__(self, name, floor)

    def to_dict(self) -> dict:
        return {
            "reciprocity_abs_tol": self.reciprocity_abs_tol,
            "reciprocity_rel_tol": self.reciprocity_rel_tol,
            "reciprocity_floor": self.reciprocity_floor,
            "correction_abs_tol": self.correction_abs_tol,
            "correction_rel_tol": self.correction_rel_tol,
            "correction_floor": self.correction_floor,
            "derived_symmetry_tol": self.derived_symmetry_tol,
            "damping_psd_tol": self.damping_psd_tol,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DerivedOperatorCriteria":
        if not isinstance(data, Mapping):
            raise DerivedOperatorCriteriaError("serialized criteria must be a mapping")
        expected = set(cls.__dataclass_fields__)
        unknown = set(data) - expected
        if unknown:
            raise DerivedOperatorCriteriaError(
                f"serialized criteria contain unknown key(s): {sorted(unknown)}")
        missing = expected - set(data)
        if missing:
            raise DerivedOperatorCriteriaError(
                f"serialized criteria are missing field(s): {sorted(missing)}")
        return cls(**dict(data))


SETTINGS_SERIALIZED_KEYS = frozenset({
    "correction_policy", "retained_dofs", "normalization", "criteria",
})


@dataclass(frozen=True, init=False)
class DerivedOperatorSettings:
    """Immutable settings for one derived-operator construction.

    Every field is explicit; there is no default correction policy, no
    default retained subspace, no default normalization and no default
    criteria. Unknown settings are refused.
    """

    correction_policy: str
    retained_dofs: Tuple[str, ...]
    normalization: NormalizationSpec
    criteria: DerivedOperatorCriteria

    def __init__(
        self,
        *,
        correction_policy: str,
        retained_dofs,
        normalization: NormalizationSpec,
        criteria: DerivedOperatorCriteria,
        **unknown: Any,
    ) -> None:
        if unknown:
            raise DerivedOperatorSettingsError(
                f"unknown derived-operator setting(s) {sorted(unknown)}; "
                "DerivedOperatorSettings refuses silent unknowns")
        if not isinstance(correction_policy, str):
            raise DerivedOperatorSettingsError(
                f"correction_policy must be a string, got {type(correction_policy).__name__}")
        _require(correction_policy in CORRECTION_POLICIES,
                 DerivedOperatorSettingsError(
                     "correction_policy must explicitly be one of "
                     f"{CORRECTION_POLICIES}, got {correction_policy!r}; "
                     "there is no default policy"))
        if not isinstance(normalization, NormalizationSpec):
            raise DerivedOperatorSettingsError(
                "normalization must be a NormalizationSpec instance "
                f"(got {type(normalization).__name__})")
        if not isinstance(criteria, DerivedOperatorCriteria):
            raise DerivedOperatorSettingsError(
                "criteria must be a DerivedOperatorCriteria instance "
                f"(got {type(criteria).__name__})")
        object.__setattr__(self, "correction_policy", str(correction_policy))
        object.__setattr__(self, "retained_dofs",
                           _validate_dof_labels(retained_dofs, "retained_dofs"))
        object.__setattr__(self, "normalization", normalization)
        object.__setattr__(self, "criteria", criteria)

    def to_dict(self) -> dict:
        return {
            "correction_policy": self.correction_policy,
            "retained_dofs": list(self.retained_dofs),
            "normalization": self.normalization.to_dict(),
            "criteria": self.criteria.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DerivedOperatorSettings":
        if not isinstance(data, Mapping):
            raise DerivedOperatorSettingsError(
                f"serialized settings must be a mapping (got {type(data).__name__})")
        unknown = set(data) - SETTINGS_SERIALIZED_KEYS
        if unknown:
            raise DerivedOperatorSettingsError(
                f"serialized settings contain unknown top-level key(s): {sorted(unknown)}")
        try:
            return cls(
                correction_policy=data["correction_policy"],
                retained_dofs=data["retained_dofs"],
                normalization=NormalizationSpec.from_dict(data["normalization"]),
                criteria=DerivedOperatorCriteria.from_dict(data["criteria"]),
            )
        except KeyError as exc:
            raise DerivedOperatorSettingsError(
                f"serialized settings are missing field {exc}") from exc


def build_projection(source_dof_order, retained_dofs) -> np.ndarray:
    """Exact 0/1 selector ``P`` (n_full x n_ret) from labels only.

    ``P[i, j] = 1`` iff ``source_dof_order[i] == retained_dofs[j]``. No
    numeric DOF positions are assumed or hardcoded.
    """
    source = [str(k) for k in source_dof_order]
    retained = _validate_dof_labels(retained_dofs, "retained_dofs")
    _require(len(set(source)) == len(source) and source,
             DerivedOperatorSettingsError(
                 "source_dof_order must be non-empty with unique labels"))
    for label in retained:
        _require(label in source, DerivedOperatorSettingsError(
            f"retained DOF {label!r} is not present in the source DOF labels"))
    P = np.zeros((len(source), len(retained)), dtype=np.float64)
    for j, label in enumerate(retained):
        P[source.index(label), j] = 1.0
    return P


def _normalized(X: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Strict congruence ``S^T X S`` for diagonal positive ``S``."""
    if X.ndim == 2:
        return scales[:, None] * X * scales[None, :]
    return scales[None, :, None] * X * scales[None, None, :]


def _asymmetry_defect(X: np.ndarray) -> np.ndarray:
    """Complete elementwise ``|X - X^T|`` (frequency-indexed when X is)."""
    if X.ndim == 2:
        return np.abs(X - X.T)
    return np.abs(X - np.transpose(X, (0, 2, 1)))


def _threshold_array(abs_tol: float, rel_tol: float, scale: float,
                     floor: float, shape) -> np.ndarray:
    g = max(float(scale), float(floor))
    return np.full(shape, abs_tol + rel_tol * g, dtype=np.float64)


def _retained_report_arrays(
    criteria: DerivedOperatorCriteria,
    normalization: NormalizationSpec,
    raw_A: np.ndarray, raw_B: np.ndarray, raw_Ainf: np.ndarray,
    derived_A: np.ndarray, derived_B: np.ndarray, derived_Ainf: np.ndarray,
) -> dict:
    """Recomputable retained-side report arrays (never trusted from storage).

    Everything here is a deterministic function of the stored raw retained
    arrays, the derived arrays and the declared settings, so the artifact
    constructor and the loader recompute it and refuse mismatches. All
    admission arithmetic is in the declared normalized coordinates
    ``X_hat = S^T X S``; the stored mechanical arrays stay SI.
    """
    s = np.asarray(normalization.scales, dtype=np.float64)
    n = s.size
    out: dict = {}

    raw_norm = {
        "added_mass": _normalized(np.asarray(raw_A), s),
        "damping": _normalized(np.asarray(raw_B), s),
        "added_mass_infinite": _normalized(np.asarray(raw_Ainf), s),
    }
    derived = {
        "added_mass": np.asarray(derived_A),
        "damping": np.asarray(derived_B),
        "added_mass_infinite": np.asarray(derived_Ainf),
    }

    for key in REPORT_MATRIX_KEYS:
        Xh = raw_norm[key]
        defect = _asymmetry_defect(Xh)
        scale = float(np.max(np.abs(Xh))) if Xh.size else 0.0
        threshold = _threshold_array(criteria.reciprocity_abs_tol,
                                     criteria.reciprocity_rel_tol,
                                     scale, criteria.reciprocity_floor,
                                     defect.shape)
        out[f"reciprocity_defect__{key}"] = defect
        out[f"reciprocity_threshold__{key}"] = threshold
        out[f"reciprocity_pass__{key}"] = defect <= threshold

        Xh_der = _normalized(derived[key], s)
        diff = np.abs(Xh_der - Xh)
        cthreshold = _threshold_array(criteria.correction_abs_tol,
                                      criteria.correction_rel_tol,
                                      scale, criteria.correction_floor,
                                      diff.shape)
        out[f"correction_defect__{key}"] = diff
        out[f"correction_threshold__{key}"] = cthreshold
        out[f"correction_pass__{key}"] = diff <= cthreshold

        sym_defect = _asymmetry_defect(Xh_der)
        out[f"symmetry_defect__{key}"] = sym_defect
        out[f"symmetry_pass__{key}"] = bool(
            np.all(sym_defect <= criteria.derived_symmetry_tol))

    # Retained B PSD admission in normalized coordinates: complete
    # eigenbasis of sym(B_hat) = 0.5*(B_hat + B_hat^T) per frequency, where
    # B_hat = S^T B_derived S, plus every negative mode (eigenvalue below
    # -tol), never clamped. The eigensolver always sees the explicit
    # symmetric part, never silently one triangle of a nonsymmetric
    # B_hat (possible under the "none" policy). eigh returns eigenvectors
    # as COLUMNS of v; a normalized eigenvector v_hat maps to the physical
    # retained direction S v_hat.
    B_hat = _normalized(np.asarray(derived_B, dtype=np.float64), s)
    n_w = B_hat.shape[0]
    eigvals = np.empty((n_w, n), dtype=np.float64)
    eigvecs = np.empty((n_w, n, n), dtype=np.float64)
    for o in range(n_w):
        sym = 0.5 * (B_hat[o] + B_hat[o].T)
        w, v = np.linalg.eigh(sym)
        eigvals[o] = w
        eigvecs[o] = v
    out["damping_eigenvalues"] = eigvals
    out["damping_eigenvectors"] = eigvecs
    neg = eigvals < -criteria.damping_psd_tol
    freq_idx, basis_idx = np.nonzero(neg)
    out["damping_negative_mode_frequency_indices"] = freq_idx.astype(np.int64)
    out["damping_negative_mode_basis_indices"] = basis_idx.astype(np.int64)
    out["damping_negative_mode_eigenvalues"] = eigvals[neg].astype(np.float64)
    # each negative mode is the COLUMN eigvecs[freq_idx, :, basis_idx],
    # giving shape (n_modes, n); eigvecs[neg] would wrongly select rows
    out["damping_negative_mode_vectors"] = eigvecs[freq_idx, :, basis_idx].astype(
        np.float64)
    out["damping_psd_pass"] = bool(freq_idx.size == 0)
    return out


def _raw_full_diagnostics(
    full_A: np.ndarray, full_B: np.ndarray, P: np.ndarray,
    psd_tol: float,
) -> dict:
    """Informational raw-full diagnostics; never part of retained admission.

    These arrays are computed on the raw full source in unscaled mixed-SI
    coordinates (they lack the retained normalization ``S``). They are
    informational only: excluded-DOF PSD defects stay visible here with
    their negative-mode vectors and retained alignment, but they can never
    be read as a retained failure or pass, never as a normalized severity
    measure, and are never repaired.
    """
    n_w = full_B.shape[0]
    n_full = full_B.shape[1]
    min_eigs = np.empty(n_w, dtype=np.float64)
    neg_idx_l = []
    neg_vals_l = []
    neg_vecs_l = []
    align_l = []
    for o in range(n_w):
        sym = 0.5 * (full_B[o] + full_B[o].T)
        w, v = np.linalg.eigh(sym)
        min_eigs[o] = w[0]
        if w[0] < -psd_tol:
            vec = v[:, 0]
            retained_part = P.T @ vec
            align = float(np.linalg.norm(retained_part) / np.linalg.norm(vec))
            neg_idx_l.append(o)
            neg_vals_l.append(w[0])
            neg_vecs_l.append(vec)
            align_l.append(align)
    return {
        "raw_full_unscaled_damping_min_eigenvalues": min_eigs,
        "raw_full_unscaled_negative_mode_indices": np.asarray(neg_idx_l, dtype=np.int64),
        "raw_full_unscaled_negative_mode_eigenvalues": np.asarray(neg_vals_l, dtype=np.float64),
        "raw_full_unscaled_negative_mode_vectors": np.asarray(neg_vecs_l, dtype=np.float64).reshape(
            len(neg_vecs_l), n_full),
        "raw_full_unscaled_negative_mode_retained_alignment": np.asarray(align_l, dtype=np.float64),
        "raw_full_unscaled_max_damping_asymmetry": float(np.max(_asymmetry_defect(full_B)))
            if full_B.size else 0.0,
        "raw_full_unscaled_max_added_mass_asymmetry": float(np.max(_asymmetry_defect(full_A)))
            if full_A.size else 0.0,
    }


def _apply_policy(policy: str, X: np.ndarray) -> np.ndarray:
    """Policy application to one operator (omega-leading or single matrix)."""
    if policy == CORRECTION_NONE:
        return np.array(X, dtype=np.float64, copy=True)
    if policy == CORRECTION_RECIPROCAL_AVERAGE:
        if X.ndim == 2:
            return 0.5 * (X + X.T)
        return 0.5 * (X + np.transpose(X, (0, 2, 1)))
    raise DerivedOperatorSettingsError(f"unknown correction policy {policy!r}")


def _derive_damping(policy: str, raw_B: np.ndarray, raw_B_far: Optional[np.ndarray],
                    scales: np.ndarray) -> np.ndarray:
    """Derived retained damping (omega-leading) under ``policy``.

    For the far-field weak-mode policy, per frequency in normalized
    coordinates ``B_hat = S B S``: eigendecompose the reciprocal average and
    replace its block on the weak subspace (eigenvalues below
    ``WEAK_MODE_FRACTION`` times the largest, orthonormal basis ``V_w``) by
    the far-field damping compressed onto it,
    ``B_hat + V_w (V_w^T F_hat V_w - Lambda_w) V_w^T`` with
    ``F_hat = S B_far S``. If no eigenvalue is positive, every mode is weak.
    The result is symmetrized exactly and mapped back to SI coordinates.
    """
    if policy not in FAR_FIELD_POLICIES:
        return _apply_policy(policy, raw_B)
    _require(raw_B_far is not None,
             DerivedOperatorError(f"policy {policy!r} requires far-field damping"))
    sym = _apply_policy(CORRECTION_RECIPROCAL_AVERAGE, raw_B)
    s = np.asarray(scales, dtype=np.float64)
    outer = s[:, None] * s[None, :]
    out = np.empty_like(sym)
    for o in range(sym.shape[0]):
        Bh = sym[o] * outer
        Fh = 0.5 * (raw_B_far[o] + raw_B_far[o].T) * outer
        lam, V = np.linalg.eigh(Bh)
        weak = lam < WEAK_MODE_FRACTION * lam[-1] if lam[-1] > 0.0 else np.ones_like(lam, dtype=bool)
        Vw = V[:, weak]
        # weak subspace takes the far-field damping compressed onto it:
        # basis-independent (repeated weak eigenvalues) and PSD on that subspace
        new = Bh + Vw @ (Vw.T @ Fh @ Vw - np.diag(lam[weak])) @ Vw.T
        new = 0.5 * (new + new.T)
        out[o] = new / outer
    return out


@dataclass(frozen=True)
class DerivedOperatorReport:
    """Complete criteria results: residuals, thresholds, passes, modes.

    Retained-side arrays are recomputed by the artifact constructor and the
    loader; stored pass bits are never trusted. Reciprocity, correction and
    symmetry residuals are complete and frequency-indexed (shape
    ``(n_w, n, n)`` for the frequency-dependent A and B, ``(n, n)`` for
    A_inf) with the declared threshold arrays alongside; maxima appear in
    :meth:`summary` only. The retained B section carries the complete
    normalized eigenbasis (eigenvalues and column eigenvectors of the
    symmetric part ``0.5*(B_hat + B_hat^T)``, ``B_hat = S^T B S``, per
    frequency) plus every negative mode (all
    eigenvalues below ``-damping_psd_tol``, not only the worst per
    frequency); a normalized eigenvector ``v_hat`` maps to the physical
    retained direction ``S v_hat``. Raw-full arrays are informational
    unscaled mixed-SI diagnostics of the raw source and never decide
    retained admission or carry normalized severity.
    """

    criteria: DerivedOperatorCriteria
    normalization_name: str
    retained_dofs: Tuple[str, ...]
    omegas: np.ndarray
    reciprocity: Mapping[str, Mapping[str, np.ndarray]]
    correction: Mapping[str, Mapping[str, np.ndarray]]
    symmetry_defect: Mapping[str, np.ndarray]
    symmetry_pass: Mapping[str, bool]
    damping_eigenvalues: np.ndarray
    damping_eigenvectors: np.ndarray
    damping_negative_mode_frequency_indices: np.ndarray
    damping_negative_mode_basis_indices: np.ndarray
    damping_negative_mode_eigenvalues: np.ndarray
    damping_negative_mode_vectors: np.ndarray
    damping_psd_pass: bool
    raw_full_coordinate_label: str
    raw_full_unscaled_damping_min_eigenvalues: np.ndarray
    raw_full_unscaled_negative_mode_indices: np.ndarray
    raw_full_unscaled_negative_mode_eigenvalues: np.ndarray
    raw_full_unscaled_negative_mode_vectors: np.ndarray
    raw_full_unscaled_negative_mode_retained_alignment: np.ndarray
    raw_full_unscaled_max_damping_asymmetry: float
    raw_full_unscaled_max_added_mass_asymmetry: float

    def __post_init__(self) -> None:
        _require(isinstance(self.criteria, DerivedOperatorCriteria),
                 DerivedOperatorCriteriaError("criteria must be DerivedOperatorCriteria"))
        _require(isinstance(self.normalization_name, str) and self.normalization_name.strip(),
                 DerivedOperatorError("normalization_name must be a non-empty string"))
        labels = _validate_dof_labels(self.retained_dofs, "retained_dofs")
        object.__setattr__(self, "retained_dofs", labels)
        n = len(labels)

        omegas = _frozen_float_array(self.omegas, "report omegas", ndim=1)
        _require(omegas.size >= 1, DerivedOperatorError("report needs at least one frequency"))
        object.__setattr__(self, "omegas", omegas)
        n_w = omegas.size

        def _frozen_matrix_map(value, name, kind):
            _require(isinstance(value, Mapping), DerivedOperatorError(
                f"report {name} must be a mapping"))
            out = {}
            for key in REPORT_MATRIX_KEYS:
                _require(key in value, DerivedOperatorError(
                    f"report {name} is missing matrix {key!r}"))
                # frequency-indexed residuals for A and B, single matrix
                # for A_inf (it has no frequency axis)
                expected = (n_w, n, n) if key != "added_mass_infinite" else (n, n)
                if kind == "bool":
                    arr = np.asarray(value[key], dtype=bool)
                    _require(arr.shape == expected, DerivedOperatorError(
                        f"report {name}[{key}] must have shape {expected}"))
                    arr = arr.copy()
                    arr.setflags(write=False)
                elif kind == "scalar_bool":
                    arr = bool(value[key])
                else:
                    arr = _frozen_float_array(value[key], f"report {name}[{key}]", ndim=len(expected))
                    _require(arr.shape == expected, DerivedOperatorError(
                        f"report {name}[{key}] must have shape {expected}"))
                out[key] = arr
            return MappingProxyType(out)

        object.__setattr__(self, "reciprocity", MappingProxyType({
            "defect": _frozen_matrix_map(self.reciprocity["defect"], "reciprocity defect", "float"),
            "threshold": _frozen_matrix_map(self.reciprocity["threshold"], "reciprocity threshold", "float"),
            "pass": _frozen_matrix_map(self.reciprocity["pass"], "reciprocity pass", "bool"),
        }))
        object.__setattr__(self, "correction", MappingProxyType({
            "defect": _frozen_matrix_map(self.correction["defect"], "correction defect", "float"),
            "threshold": _frozen_matrix_map(self.correction["threshold"], "correction threshold", "float"),
            "pass": _frozen_matrix_map(self.correction["pass"], "correction pass", "bool"),
        }))
        object.__setattr__(self, "symmetry_defect",
                            _frozen_matrix_map(self.symmetry_defect, "symmetry defect", "float"))
        sym_pass = {}
        for key in REPORT_MATRIX_KEYS:
            _require(key in self.symmetry_pass, DerivedOperatorError(
                f"report symmetry_pass is missing matrix {key!r}"))
            sym_pass[key] = bool(self.symmetry_pass[key])
        object.__setattr__(self, "symmetry_pass", MappingProxyType(sym_pass))

        eig = _frozen_float_array(self.damping_eigenvalues, "damping eigenvalues", ndim=2)
        _require(eig.shape == (n_w, n), DerivedOperatorError(
            f"report damping_eigenvalues must have shape {(n_w, n)}"))
        object.__setattr__(self, "damping_eigenvalues", eig)
        vecs = _frozen_float_array(self.damping_eigenvectors, "damping eigenvectors", ndim=3)
        _require(vecs.shape == (n_w, n, n), DerivedOperatorError(
            f"report damping_eigenvectors must have shape {(n_w, n, n)}"))
        object.__setattr__(self, "damping_eigenvectors", vecs)
        fidx = np.asarray(self.damping_negative_mode_frequency_indices, dtype=np.int64).copy()
        fidx.setflags(write=False)
        object.__setattr__(self, "damping_negative_mode_frequency_indices", fidx)
        bidx = np.asarray(self.damping_negative_mode_basis_indices, dtype=np.int64).copy()
        bidx.setflags(write=False)
        object.__setattr__(self, "damping_negative_mode_basis_indices", bidx)
        vals = _frozen_float_array(self.damping_negative_mode_eigenvalues,
                                   "damping negative eigenvalues", ndim=1)
        nvecs = _frozen_float_array(self.damping_negative_mode_vectors,
                                    "damping negative vectors", ndim=2)
        _require(vals.size == fidx.size and bidx.shape == fidx.shape
                 and nvecs.shape == (fidx.size, n),
                 DerivedOperatorError("negative-mode arrays must be consistent"))
        _require(fidx.size == 0 or (bool(fidx.min() >= 0) and bool(fidx.max() < n_w)
                                    and bool(bidx.min() >= 0) and bool(bidx.max() < n)),
                 DerivedOperatorError("negative-mode indices out of range"))
        object.__setattr__(self, "damping_negative_mode_eigenvalues", vals)
        object.__setattr__(self, "damping_negative_mode_vectors", nvecs)
        _require(isinstance(self.damping_psd_pass, (bool, np.bool_)),
                 DerivedOperatorError("damping_psd_pass must be a bool"))
        object.__setattr__(self, "damping_psd_pass", bool(self.damping_psd_pass))

        _require(self.raw_full_coordinate_label == RAW_FULL_DIAGNOSTIC_COORDINATES,
                 DerivedOperatorError(
                     "raw_full_coordinate_label must be "
                     f"{RAW_FULL_DIAGNOSTIC_COORDINATES!r}"))
        object.__setattr__(self, "raw_full_coordinate_label",
                           RAW_FULL_DIAGNOSTIC_COORDINATES)
        rf_eig = _frozen_float_array(self.raw_full_unscaled_damping_min_eigenvalues,
                                     "raw full min eigenvalues", ndim=1)
        _require(rf_eig.shape == (n_w,), DerivedOperatorError(
            "raw_full_unscaled_damping_min_eigenvalues must have one entry per frequency"))
        object.__setattr__(self, "raw_full_unscaled_damping_min_eigenvalues", rf_eig)
        rf_idx = np.asarray(self.raw_full_unscaled_negative_mode_indices, dtype=np.int64).copy()
        rf_idx.setflags(write=False)
        object.__setattr__(self, "raw_full_unscaled_negative_mode_indices", rf_idx)
        rf_vals = _frozen_float_array(self.raw_full_unscaled_negative_mode_eigenvalues,
                                      "raw full negative eigenvalues", ndim=1)
        rf_vecs = np.asarray(self.raw_full_unscaled_negative_mode_vectors, dtype=np.float64)
        _require(rf_vecs.ndim == 2, DerivedOperatorError(
            "raw_full_unscaled_negative_mode_vectors must be 2-D"))
        n_full = rf_vecs.shape[1]
        rf_vecs = _frozen_float_array(rf_vecs, "raw full negative vectors", ndim=2)
        rf_align = _frozen_float_array(self.raw_full_unscaled_negative_mode_retained_alignment,
                                       "raw full negative alignment", ndim=1)
        _require(rf_vals.size == rf_idx.size and rf_vecs.shape == (rf_idx.size, n_full)
                 and rf_align.shape == (rf_idx.size,),
                 DerivedOperatorError("raw-full negative-mode arrays must be consistent"))
        object.__setattr__(self, "raw_full_unscaled_negative_mode_eigenvalues", rf_vals)
        object.__setattr__(self, "raw_full_unscaled_negative_mode_vectors", rf_vecs)
        object.__setattr__(self, "raw_full_unscaled_negative_mode_retained_alignment", rf_align)
        for name in ("raw_full_unscaled_max_damping_asymmetry",
                     "raw_full_unscaled_max_added_mass_asymmetry"):
            value = float(getattr(self, name))
            _require(np.isfinite(value), DerivedOperatorError(f"diagnostic {name} is not finite"))
            object.__setattr__(self, name, value)

    @property
    def raw_reciprocity_pass(self) -> bool:
        return all(bool(np.all(self.reciprocity["pass"][k])) for k in REPORT_MATRIX_KEYS)

    @property
    def correction_pass(self) -> bool:
        return all(bool(np.all(self.correction["pass"][k])) for k in REPORT_MATRIX_KEYS)

    @property
    def derived_symmetry_pass(self) -> bool:
        return all(self.symmetry_pass[k] for k in REPORT_MATRIX_KEYS)

    @property
    def admitted(self) -> bool:
        return bool(self.raw_reciprocity_pass and self.correction_pass
                    and self.derived_symmetry_pass and self.damping_psd_pass)

    def summary(self) -> dict:
        """Maxima and counts only; the complete frequency-indexed
        residual/threshold/pass arrays and the complete eigenbasis live in
        the report fields themselves."""
        return {
            "normalization_name": self.normalization_name,
            "retained_dofs": list(self.retained_dofs),
            "n_frequencies": int(self.omegas.size),
            "raw_reciprocity_pass": self.raw_reciprocity_pass,
            "correction_pass": self.correction_pass,
            "derived_symmetry_pass": self.derived_symmetry_pass,
            "damping_psd_pass": self.damping_psd_pass,
            "admitted": self.admitted,
            "max_reciprocity_defect": {
                k: float(np.max(self.reciprocity["defect"][k]))
                for k in REPORT_MATRIX_KEYS},
            "max_correction_defect": {
                k: float(np.max(self.correction["defect"][k]))
                for k in REPORT_MATRIX_KEYS},
            "max_symmetry_defect": {
                k: float(np.max(self.symmetry_defect[k]))
                for k in REPORT_MATRIX_KEYS},
            "min_damping_eigenvalue": float(np.min(self.damping_eigenvalues))
                if self.damping_eigenvalues.size else None,
            "n_damping_negative_modes": int(
                self.damping_negative_mode_frequency_indices.size),
            "raw_full_coordinate_label": self.raw_full_coordinate_label,
            "raw_full_unscaled_min_damping_eigenvalue":
                float(np.min(self.raw_full_unscaled_damping_min_eigenvalues))
                if self.raw_full_unscaled_damping_min_eigenvalues.size else None,
            "n_raw_full_unscaled_negative_modes":
                int(self.raw_full_unscaled_negative_mode_indices.size),
            "raw_full_unscaled_negative_mode_retained_alignment":
                self.raw_full_unscaled_negative_mode_retained_alignment.tolist(),
            "raw_full_unscaled_max_damping_asymmetry":
                self.raw_full_unscaled_max_damping_asymmetry,
            "raw_full_unscaled_max_added_mass_asymmetry":
                self.raw_full_unscaled_max_added_mass_asymmetry,
        }

    def _named_arrays(self):
        """Every report array, in a fixed order (part of the hash)."""
        arrays = []
        for key in REPORT_MATRIX_KEYS:
            arrays.extend([
                (f"reciprocity_defect__{key}", self.reciprocity["defect"][key]),
                (f"reciprocity_threshold__{key}", self.reciprocity["threshold"][key]),
                (f"reciprocity_pass__{key}", self.reciprocity["pass"][key]),
                (f"correction_defect__{key}", self.correction["defect"][key]),
                (f"correction_threshold__{key}", self.correction["threshold"][key]),
                (f"correction_pass__{key}", self.correction["pass"][key]),
                (f"symmetry_defect__{key}", self.symmetry_defect[key]),
            ])
        arrays.extend([
            ("damping_eigenvalues", self.damping_eigenvalues),
            ("damping_eigenvectors", self.damping_eigenvectors),
            ("damping_negative_mode_frequency_indices",
             self.damping_negative_mode_frequency_indices),
            ("damping_negative_mode_basis_indices",
             self.damping_negative_mode_basis_indices),
            ("damping_negative_mode_eigenvalues",
             self.damping_negative_mode_eigenvalues),
            ("damping_negative_mode_vectors", self.damping_negative_mode_vectors),
            ("raw_full_unscaled_damping_min_eigenvalues",
             self.raw_full_unscaled_damping_min_eigenvalues),
            ("raw_full_unscaled_negative_mode_indices",
             self.raw_full_unscaled_negative_mode_indices),
            ("raw_full_unscaled_negative_mode_eigenvalues",
             self.raw_full_unscaled_negative_mode_eigenvalues),
            ("raw_full_unscaled_negative_mode_vectors",
             self.raw_full_unscaled_negative_mode_vectors),
            ("raw_full_unscaled_negative_mode_retained_alignment",
             self.raw_full_unscaled_negative_mode_retained_alignment),
        ])
        return tuple(arrays)


@dataclass(frozen=True)
class DerivedHydroOperators:
    """Immutable derived retained-coordinate operator artifact.

    Stores the source dataset hash, the raw source DOF/body order, the
    exact selector ``P``, the raw retained A/B/A_inf, the derived A/B/A_inf,
    the projected unmodified hydrostatics/excitation and frequency/wave
    coordinates, full source provenance and the complete criteria report.
    The constructor recomputes the selector, the policy application and the
    retained-side report, refusing any inconsistency; ``admitted`` is always
    the recomputed report verdict, never a stored bit.
    """

    settings: DerivedOperatorSettings
    source_dof_order: Tuple[str, ...]
    source_body_order: Tuple[str, ...]
    projection: np.ndarray
    omegas: np.ndarray
    wave_directions: np.ndarray
    raw_added_mass: np.ndarray
    raw_damping: np.ndarray
    raw_added_mass_infinite: np.ndarray
    derived_added_mass: np.ndarray
    derived_damping: np.ndarray
    derived_added_mass_infinite: np.ndarray
    hydrostatic_stiffness: np.ndarray
    excitation_force: np.ndarray
    report: DerivedOperatorReport
    provenance: Mapping[str, Any]
    #: retained far-field damping (SI), present iff the policy consumes it
    raw_damping_far_field: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        _require(isinstance(self.settings, DerivedOperatorSettings),
                 DerivedOperatorError("settings must be DerivedOperatorSettings"))
        _require(isinstance(self.report, DerivedOperatorReport),
                 DerivedOperatorError("report must be a DerivedOperatorReport"))

        source_dofs = _validate_dof_labels(self.source_dof_order, "source_dof_order")
        object.__setattr__(self, "source_dof_order", source_dofs)
        bodies = tuple(str(b) for b in self.source_body_order)
        _require(len(set(bodies)) == len(bodies) and bodies,
                 DerivedOperatorError("source_body_order must be non-empty and unique"))
        object.__setattr__(self, "source_body_order", bodies)

        # hard boundaries and full-6N body/DOF-order consistency, enforced
        # in the constructor and reader before any effective array freezes
        n_full = len(source_dofs)
        _require(n_full <= MAX_SOURCE_DOFS,
                 DerivedOperatorLimitError(
                     f"{n_full} source DOFs exceed the development limit of "
                     f"{MAX_SOURCE_DOFS}"))
        try:
            n_omega_raw = int(np.asarray(self.omegas).size)
        except (TypeError, ValueError) as exc:
            raise DerivedOperatorError("omegas must be a numeric array") from exc
        _require(n_omega_raw <= MAX_SOURCE_FREQUENCIES,
                 DerivedOperatorLimitError(
                     f"{n_omega_raw} source frequencies exceed the development "
                     f"limit of {MAX_SOURCE_FREQUENCIES}"))
        per_body: dict = {}
        for label in source_dofs:
            body_id, dof = parse_dof_key(label)
            per_body.setdefault(body_id, []).append(dof)
        _require(set(per_body) == set(bodies) and n_full == 6 * len(bodies),
                 DerivedOperatorError(
                     "source_dof_order is not consistent with source_body_order: "
                     "every source body must carry exactly its six DOFs"))
        for body_id, dofs in per_body.items():
            _require(len(dofs) == 6 and set(dofs) == set(DOF_NAMES),
                     DerivedOperatorError(
                         f"source body {body_id!r} does not carry the complete "
                         "six-DOF set (surge, sway, heave, roll, pitch, yaw)"))

        retained = self.settings.retained_dofs
        n_ret = len(retained)
        _require(n_ret <= n_full, DerivedOperatorError(
            "retained DOF count must not exceed the source DOF count"))
        _require(len(self.settings.normalization.scales) == n_ret,
                 DerivedOperatorSettingsError(
                     "normalization scales must have one entry per retained DOF"))

        P = _frozen_float_array(self.projection, "projection", ndim=2)
        expected_P = build_projection(source_dofs, retained)
        _require(P.shape == expected_P.shape and bool(np.array_equal(P, expected_P)),
                 DerivedOperatorError(
                     "projection is not the exact selector implied by the source "
                     "and retained DOF labels"))
        object.__setattr__(self, "projection", P)

        omegas = _frozen_float_array(self.omegas, "omegas", ndim=1)
        _require(omegas.size >= 1 and bool(np.all(omegas > 0.0))
                 and bool(np.all(np.diff(omegas) > 0.0)),
                 DerivedOperatorError("omegas must be strictly increasing and > 0"))
        object.__setattr__(self, "omegas", omegas)
        n_w = omegas.size
        dirs = _frozen_float_array(self.wave_directions, "wave_directions", ndim=1)
        _require(dirs.size >= 1, DerivedOperatorError("wave_directions must be non-empty"))
        object.__setattr__(self, "wave_directions", dirs)

        for name in ("raw_added_mass", "raw_damping",
                     "derived_added_mass", "derived_damping"):
            arr = _frozen_float_array(getattr(self, name), name, ndim=3)
            _require(arr.shape == (n_w, n_ret, n_ret),
                     DerivedOperatorError(f"{name} must have shape {(n_w, n_ret, n_ret)}"))
            object.__setattr__(self, name, arr)
        for name in ("raw_added_mass_infinite", "derived_added_mass_infinite",
                     "hydrostatic_stiffness"):
            arr = _frozen_float_array(getattr(self, name), name, ndim=2)
            _require(arr.shape == (n_ret, n_ret),
                     DerivedOperatorError(f"{name} must have shape {(n_ret, n_ret)}"))
            object.__setattr__(self, name, arr)
        exc = _frozen_complex_array(self.excitation_force, "excitation_force", ndim=3)
        _require(exc.shape == (n_w, dirs.size, n_ret),
                 DerivedOperatorError(
                     f"excitation_force must have shape {(n_w, dirs.size, n_ret)}"))
        object.__setattr__(self, "excitation_force", exc)

        # invariant: derived arrays are exactly the policy applied to raw
        policy = self.settings.correction_policy
        if policy in FAR_FIELD_POLICIES:
            _require(self.raw_damping_far_field is not None,
                     DerivedOperatorError(
                         f"policy {policy!r} requires raw_damping_far_field"))
            far = _frozen_float_array(self.raw_damping_far_field,
                                      "raw_damping_far_field", ndim=3)
            _require(far.shape == (n_w, n_ret, n_ret),
                     DerivedOperatorError(
                         f"raw_damping_far_field must have shape {(n_w, n_ret, n_ret)}"))
            object.__setattr__(self, "raw_damping_far_field", far)
        else:
            _require(self.raw_damping_far_field is None,
                     DerivedOperatorError(
                         f"policy {policy!r} does not consume far-field damping; "
                         "raw_damping_far_field must be absent"))
        expected_B = _derive_damping(policy, self.raw_damping,
                                     self.raw_damping_far_field,
                                     np.asarray(self.settings.normalization.scales))
        _require(bool(np.array_equal(self.derived_damping, expected_B)),
                 DerivedOperatorError(
                     "derived_damping is not the declared policy applied to raw_damping"))
        for raw_name, der_name in (
            ("raw_added_mass", "derived_added_mass"),
            ("raw_added_mass_infinite", "derived_added_mass_infinite"),
        ):
            expected = _apply_policy(
                CORRECTION_RECIPROCAL_AVERAGE if policy in FAR_FIELD_POLICIES else policy,
                getattr(self, raw_name))
            _require(bool(np.array_equal(getattr(self, der_name), expected)),
                     DerivedOperatorError(
                         f"{der_name} is not the declared policy applied to {raw_name}"))

        # invariant: the retained-side report is exactly the recomputation
        recomputed = _retained_report_arrays(
            self.settings.criteria, self.settings.normalization,
            self.raw_added_mass, self.raw_damping, self.raw_added_mass_infinite,
            self.derived_added_mass, self.derived_damping,
            self.derived_added_mass_infinite)
        self._verify_report(recomputed)

        _require(isinstance(self.provenance, Mapping) and self.provenance,
                 DerivedOperatorError("provenance must be a non-empty mapping"))
        object.__setattr__(self, "provenance", _deep_freeze(dict(self.provenance)))

    def _verify_report(self, recomputed: Mapping[str, Any]) -> None:
        report = self.report
        for key in REPORT_MATRIX_KEYS:
            for group, stored_map in (
                ("reciprocity_defect", report.reciprocity["defect"]),
                ("reciprocity_threshold", report.reciprocity["threshold"]),
                ("reciprocity_pass", report.reciprocity["pass"]),
                ("correction_defect", report.correction["defect"]),
                ("correction_threshold", report.correction["threshold"]),
                ("correction_pass", report.correction["pass"]),
                ("symmetry_defect", report.symmetry_defect),
            ):
                stored = stored_map[key]
                expect = recomputed[f"{group}__{key}"]
                if group.endswith("_pass"):
                    _require(bool(np.array_equal(np.asarray(stored, dtype=bool),
                                                 np.asarray(expect, dtype=bool))),
                             DerivedOperatorError(
                                 f"stored report {group}[{key}] does not match the "
                                 "recomputed criteria results; refusing"))
                else:
                    _require(bool(np.array_equal(stored, expect)),
                             DerivedOperatorError(
                                 f"stored report {group}[{key}] does not match the "
                                 "recomputed criteria results; refusing"))
            _require(report.symmetry_pass[key] == recomputed[f"symmetry_pass__{key}"],
                     DerivedOperatorError(
                         f"stored symmetry pass for {key} does not match the "
                         "recomputed criteria results; refusing"))
        for name in ("damping_eigenvalues", "damping_eigenvectors",
                     "damping_negative_mode_frequency_indices",
                     "damping_negative_mode_basis_indices",
                     "damping_negative_mode_eigenvalues",
                     "damping_negative_mode_vectors"):
            stored = getattr(report, name)
            _require(bool(np.array_equal(stored, recomputed[name])),
                     DerivedOperatorError(
                         f"stored report {name} does not match the recomputed "
                         "criteria results; refusing"))
        _require(report.damping_psd_pass == recomputed["damping_psd_pass"],
                 DerivedOperatorError(
                     "stored damping PSD pass does not match the recomputed "
                     "criteria results; refusing"))

    @property
    def retained_dofs(self) -> Tuple[str, ...]:
        return self.settings.retained_dofs

    @property
    def n_retained(self) -> int:
        return len(self.settings.retained_dofs)

    @property
    def admitted(self) -> bool:
        return self.report.admitted

    @property
    def qualification_status(self) -> str:
        return "unqualified"

    def verify_raw_source(self, ds: xr.Dataset) -> None:
        """Full typed raw-source proof by rederivation, not by hash alone.

        Recomputes the complete public derivation from ``ds`` under the
        immutable stored settings: the label-built selector ``P``, every
        raw projected A/B/Ainf/H/F, the frequency and wave grids, the
        source DOF and body order, the derived arrays, the report and the
        producing provenance (backend settings and controls included).
        Every stored value is then compared exactly. A matching source
        content hash is necessary but never sufficient: an artifact whose
        stored data came from a different dataset is refused even if its
        recorded source hash was forged to match. Failed (non-admitted)
        artifacts verify exactly like admitted ones; source identity and
        failure evidence are preserved.
        """
        if not isinstance(ds, xr.Dataset):
            raise DerivedOperatorError(
                "raw source must be an xarray Dataset "
                f"(got {type(ds).__name__})")
        validate_for_time_domain(ds)
        _require(dataset_content_hash(ds) == self.provenance["source_dataset_hash"],
                 DerivedOperatorError(
                     "source dataset hash does not match the recorded source "
                     "identity; this artifact was not derived from this dataset"))
        rebuilt = derived_operators_from_dataset(ds, self.settings)
        _require(self.source_dof_order == rebuilt.source_dof_order,
                 DerivedOperatorError(
                     "source DOF order does not match the raw-source rederivation"))
        _require(self.source_body_order == rebuilt.source_body_order,
                 DerivedOperatorError(
                     "source body order does not match the raw-source rederivation"))
        for name in ("projection", "omegas", "wave_directions",
                     "raw_added_mass", "raw_damping", "raw_added_mass_infinite",
                     "derived_added_mass", "derived_damping",
                     "derived_added_mass_infinite", "hydrostatic_stiffness",
                     "excitation_force", "raw_damping_far_field"):
            mine, theirs = getattr(self, name), getattr(rebuilt, name)
            _require((mine is None) == (theirs is None),
                     DerivedOperatorError(
                         f"stored {name} presence does not match the raw-source rederivation"))
            if mine is None:
                continue
            _require(bool(np.array_equal(np.asarray(mine), np.asarray(theirs))),
                     DerivedOperatorError(
                         f"stored {name} does not match the raw-source "
                         "rederivation; the artifact data did not come from "
                         "this dataset"))
        for (name, mine), (_, theirs) in zip(self.report._named_arrays(),
                                             rebuilt.report._named_arrays()):
            _require(bool(np.array_equal(np.asarray(mine), np.asarray(theirs))),
                     DerivedOperatorError(
                         f"stored report {name} does not match the raw-source "
                         "rederivation"))
        for scalar in ("damping_psd_pass", "raw_full_coordinate_label",
                       "raw_full_unscaled_max_damping_asymmetry",
                       "raw_full_unscaled_max_added_mass_asymmetry"):
            _require(getattr(self.report, scalar) == getattr(rebuilt.report, scalar),
                     DerivedOperatorError(
                         f"stored report {scalar} does not match the "
                         "raw-source rederivation"))
        for key in REPORT_MATRIX_KEYS:
            _require(self.report.symmetry_pass[key] == rebuilt.report.symmetry_pass[key],
                     DerivedOperatorError(
                         f"stored symmetry pass for {key} does not match the "
                         "raw-source rederivation"))
        _require(_canonical_json(dict(self.provenance))
                 == _canonical_json(dict(rebuilt.provenance)),
                 DerivedOperatorError(
                     "stored provenance does not match the producing controls "
                     "recomputed from this dataset"))

    def matches_raw_source(self, ds: xr.Dataset) -> bool:
        """Predicate form of :meth:`verify_raw_source` (full rederivation).

        True only if the complete rederivation from ``ds`` under the
        stored settings reproduces every stored array, report result and
        producing provenance exactly. Never trust the recorded source
        hash alone.
        """
        try:
            self.verify_raw_source(ds)
        except (DerivedOperatorError, HydroError):
            return False
        return True

    def content_hash(self) -> str:
        """SHA-256 over all data, settings, criteria and report results."""
        h = hashlib.sha256()
        for payload in (
            DERIVED_OPERATORS_ARTIFACT_FORMAT.encode("utf-8"),
            DERIVED_OPERATORS_MODULE_VERSION.encode("utf-8"),
            _canonical_json(self.settings.to_dict()).encode("utf-8"),
            _canonical_json(list(self.source_dof_order)).encode("utf-8"),
            _canonical_json(list(self.source_body_order)).encode("utf-8"),
            _canonical_json(dict(self.provenance)).encode("utf-8"),
            _canonical_json({
                "symmetry_pass": {k: self.report.symmetry_pass[k]
                                  for k in REPORT_MATRIX_KEYS},
                "damping_psd_pass": self.report.damping_psd_pass,
                "raw_full_coordinate_label": self.report.raw_full_coordinate_label,
                "raw_full_unscaled_max_damping_asymmetry":
                    self.report.raw_full_unscaled_max_damping_asymmetry,
                "raw_full_unscaled_max_added_mass_asymmetry":
                    self.report.raw_full_unscaled_max_added_mass_asymmetry,
            }).encode("utf-8"),
        ):
            h.update(len(payload).to_bytes(8, "little"))
            h.update(payload)
        for name, arr in self._named_arrays():
            data = _array_bytes(arr)
            h.update(name.encode("utf-8"))
            h.update(len(data).to_bytes(8, "little"))
            h.update(data)
        return h.hexdigest()

    def _named_arrays(self):
        arrays = [
            ("projection", self.projection),
            ("omegas", self.omegas),
            ("wave_directions", self.wave_directions),
            ("raw_added_mass", self.raw_added_mass),
            ("raw_damping", self.raw_damping),
            ("raw_added_mass_infinite", self.raw_added_mass_infinite),
            ("derived_added_mass", self.derived_added_mass),
            ("derived_damping", self.derived_damping),
            ("derived_added_mass_infinite", self.derived_added_mass_infinite),
            ("hydrostatic_stiffness", self.hydrostatic_stiffness),
            ("excitation_force", self.excitation_force),
        ]
        if self.raw_damping_far_field is not None:
            arrays.append(("raw_damping_far_field", self.raw_damping_far_field))
        arrays.extend(self.report._named_arrays())
        return tuple(arrays)


def _array_bytes(arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr)
    if np.iscomplexobj(arr):
        return arr.astype("<c16").tobytes()
    if arr.dtype == np.bool_:
        return arr.astype(np.bool_).tobytes()
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype("<i8").tobytes()
    return arr.astype("<f8").tobytes()


#: Source variables whose actual attrs are preserved as producing evidence.
SOURCE_VARIABLES = ("added_mass", "radiation_damping", "added_mass_infinite",
                    "hydrostatic_stiffness", "excitation_force", "omega",
                    "wave_direction")


def _decoded_attr(value):
    """Decode JSON-string attrs; everything else is preserved verbatim."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _source_provenance(ds: xr.Dataset, include_far_field: bool = False) -> dict:
    """Complete producing provenance of the raw source, verbatim.

    Preserves the actual complete source attributes (backend settings,
    method, fluid rho/g/depth and unit controls as actually present -
    nothing is invented or defaulted) and the relevant variable attrs,
    alongside the explicit identity keys used by consumers.
    """
    a_inf_source = str(ds["added_mass_infinite"].attrs["source"])
    _require(a_inf_source.strip(), DerivedOperatorError(
        "added_mass_infinite has an empty source"))
    bodies = [str(v) for v in ds["body"].values.tolist()]
    rotation_center = ds["rotation_center"].values
    center_of_mass = ds["center_of_mass"].values
    center_of_buoyancy = ds["center_of_buoyancy"].values
    source_attrs = {str(k): _decoded_attr(v) for k, v in ds.attrs.items()}
    variable_attrs = {}
    names = SOURCE_VARIABLES + (("radiation_damping_far_field",) if include_far_field else ())
    for name in names:
        if name in ds.variables and len(ds[name].attrs):
            variable_attrs[name] = {str(k): _decoded_attr(v)
                                    for k, v in ds[name].attrs.items()}
    return {
        "source_kind": "raw_dataset",
        "source_dataset_hash": dataset_content_hash(ds),
        "source_request_hash": str(ds.attrs["request_hash"]),
        "source_mesh_hashes": _decoded_attr(ds.attrs["mesh_hashes"]),
        "source_body_identity": _decoded_attr(ds.attrs["body_identity"]),
        "source_reference_transformations":
            _decoded_attr(ds.attrs["reference_transformations"])
            if "reference_transformations" in ds.attrs else None,
        "a_inf_source": a_inf_source,
        "phasor_convention": str(ds.attrs["phasor_convention"]),
        "heading_convention": str(ds.attrs["heading_convention"]),
        "source_qualification_status": str(ds.attrs["qualification_status"]),
        "source_schema_version": str(ds.attrs["schema_version"]),
        "source_backend": str(ds.attrs["backend"]),
        "source_backend_version": str(ds.attrs["backend_version"]),
        "source_backend_commit": str(ds.attrs["backend_commit"]),
        "source_attrs": source_attrs,
        "source_variable_attrs": variable_attrs,
        "source_rotation_center": {
            b: [float(v) for v in rotation_center[i]]
            for i, b in enumerate(bodies)},
        "source_center_of_mass": {
            b: [float(v) for v in center_of_mass[i]]
            for i, b in enumerate(bodies)},
        "source_center_of_buoyancy": {
            b: [float(v) for v in center_of_buoyancy[i]]
            for i, b in enumerate(bodies)},
        "derived_operator_module_version": DERIVED_OPERATORS_MODULE_VERSION,
    }


def derived_operators_from_dataset(
    ds: xr.Dataset, settings: DerivedOperatorSettings
) -> DerivedHydroOperators:
    """Build the derived retained-coordinate artifact from a raw dataset.

    The dataset must pass :func:`anyhydro.validate_for_time_domain` and is
    never modified. The retained subspace, normalization, correction policy
    and criteria are all taken from the explicit ``settings``; there are no
    defaults. Hard boundaries (12 source DOFs, 513 frequencies) are
    enforced before any allocation.
    """
    if not isinstance(settings, DerivedOperatorSettings):
        raise DerivedOperatorSettingsError(
            "settings must be a DerivedOperatorSettings instance")

    validate_for_time_domain(ds)

    omegas = np.asarray(ds["omega"].values, dtype=np.float64)
    _require(omegas.size <= MAX_SOURCE_FREQUENCIES,
             DerivedOperatorLimitError(
                 f"{omegas.size} source frequencies exceed the development limit "
                 f"of {MAX_SOURCE_FREQUENCIES}"))
    influenced = [str(v) for v in ds["influenced_dof"].values.tolist()]
    radiating = [str(v) for v in ds["radiating_dof"].values.tolist()]
    _require(influenced == radiating,
             DerivedOperatorError(
                 "influenced_dof and radiating_dof must share the same order; "
                 "refusing to reorder silently"))
    n_full = len(influenced)
    _require(n_full <= MAX_SOURCE_DOFS,
             DerivedOperatorLimitError(
                 f"{n_full} source DOFs exceed the development limit of "
                 f"{MAX_SOURCE_DOFS}"))

    retained = settings.retained_dofs
    n_ret = len(retained)
    _require(len(settings.normalization.scales) == n_ret,
             DerivedOperatorSettingsError(
                 "normalization scales must have one entry per retained DOF"))
    P = build_projection(influenced, retained)

    A_full = np.asarray(ds["added_mass"].values, dtype=np.float64)
    B_full = np.asarray(ds["radiation_damping"].values, dtype=np.float64)
    Ainf_full = np.asarray(ds["added_mass_infinite"].values, dtype=np.float64)
    C_full = np.asarray(ds["hydrostatic_stiffness"].values, dtype=np.float64)
    F_full = np.asarray(ds["excitation_force"].values, dtype=np.complex128)

    raw_A = np.einsum("ia,oij,jb->oab", P, A_full, P)
    raw_B = np.einsum("ia,oij,jb->oab", P, B_full, P)
    raw_Ainf = P.T @ Ainf_full @ P
    hydro = P.T @ C_full @ P
    excitation = np.einsum("ia,owi->owa", P, F_full)

    policy = settings.correction_policy
    raw_B_far = None
    if policy in FAR_FIELD_POLICIES:
        _require("radiation_damping_far_field" in ds.data_vars,
                 DerivedOperatorError(
                     f"policy {policy!r} requires the dataset variable "
                     "radiation_damping_far_field (Capytaine adapter >= 0.5.0 indirect, >= 0.5.3 direct)"))
        B_far_full = np.asarray(ds["radiation_damping_far_field"].values, dtype=np.float64)
        raw_B_far = np.einsum("ia,oij,jb->oab", P, B_far_full, P)
    av_policy = CORRECTION_RECIPROCAL_AVERAGE if policy in FAR_FIELD_POLICIES else policy
    derived_A = _apply_policy(av_policy, raw_A)
    derived_B = _derive_damping(policy, raw_B, raw_B_far,
                                np.asarray(settings.normalization.scales))
    derived_Ainf = _apply_policy(av_policy, raw_Ainf)

    dirs = np.asarray(ds["wave_direction"].values, dtype=np.float64)
    bodies = [str(v) for v in ds["body"].values.tolist()]

    provenance = _source_provenance(ds, include_far_field=policy in FAR_FIELD_POLICIES)

    report = _build_report(settings, omegas, P, raw_A, raw_B, raw_Ainf,
                           derived_A, derived_B, derived_Ainf,
                           A_full, B_full)

    return DerivedHydroOperators(
        settings=settings,
        source_dof_order=tuple(influenced),
        source_body_order=tuple(bodies),
        projection=P,
        omegas=omegas,
        wave_directions=dirs,
        raw_added_mass=raw_A,
        raw_damping=raw_B,
        raw_added_mass_infinite=raw_Ainf,
        derived_added_mass=derived_A,
        derived_damping=derived_B,
        derived_added_mass_infinite=derived_Ainf,
        hydrostatic_stiffness=hydro,
        excitation_force=excitation,
        report=report,
        provenance=provenance,
        raw_damping_far_field=raw_B_far,
    )


def _build_report(settings, omegas, P, raw_A, raw_B, raw_Ainf,
                  derived_A, derived_B, derived_Ainf,
                  full_A, full_B) -> DerivedOperatorReport:
    retained_arrays = _retained_report_arrays(
        settings.criteria, settings.normalization,
        raw_A, raw_B, raw_Ainf, derived_A, derived_B, derived_Ainf)
    raw_full = _raw_full_diagnostics(full_A, full_B, P,
                                     settings.criteria.damping_psd_tol)
    return DerivedOperatorReport(
        criteria=settings.criteria,
        normalization_name=settings.normalization.name,
        retained_dofs=settings.retained_dofs,
        omegas=omegas,
        reciprocity={
            "defect": {k: retained_arrays[f"reciprocity_defect__{k}"]
                       for k in REPORT_MATRIX_KEYS},
            "threshold": {k: retained_arrays[f"reciprocity_threshold__{k}"]
                          for k in REPORT_MATRIX_KEYS},
            "pass": {k: retained_arrays[f"reciprocity_pass__{k}"]
                     for k in REPORT_MATRIX_KEYS},
        },
        correction={
            "defect": {k: retained_arrays[f"correction_defect__{k}"]
                       for k in REPORT_MATRIX_KEYS},
            "threshold": {k: retained_arrays[f"correction_threshold__{k}"]
                          for k in REPORT_MATRIX_KEYS},
            "pass": {k: retained_arrays[f"correction_pass__{k}"]
                     for k in REPORT_MATRIX_KEYS},
        },
        symmetry_defect={k: retained_arrays[f"symmetry_defect__{k}"]
                         for k in REPORT_MATRIX_KEYS},
        symmetry_pass={k: retained_arrays[f"symmetry_pass__{k}"]
                       for k in REPORT_MATRIX_KEYS},
        damping_eigenvalues=retained_arrays["damping_eigenvalues"],
        damping_eigenvectors=retained_arrays["damping_eigenvectors"],
        damping_negative_mode_frequency_indices=retained_arrays[
            "damping_negative_mode_frequency_indices"],
        damping_negative_mode_basis_indices=retained_arrays[
            "damping_negative_mode_basis_indices"],
        damping_negative_mode_eigenvalues=retained_arrays[
            "damping_negative_mode_eigenvalues"],
        damping_negative_mode_vectors=retained_arrays[
            "damping_negative_mode_vectors"],
        damping_psd_pass=retained_arrays["damping_psd_pass"],
        raw_full_coordinate_label=RAW_FULL_DIAGNOSTIC_COORDINATES,
        raw_full_unscaled_damping_min_eigenvalues=raw_full[
            "raw_full_unscaled_damping_min_eigenvalues"],
        raw_full_unscaled_negative_mode_indices=raw_full[
            "raw_full_unscaled_negative_mode_indices"],
        raw_full_unscaled_negative_mode_eigenvalues=raw_full[
            "raw_full_unscaled_negative_mode_eigenvalues"],
        raw_full_unscaled_negative_mode_vectors=raw_full[
            "raw_full_unscaled_negative_mode_vectors"],
        raw_full_unscaled_negative_mode_retained_alignment=raw_full[
            "raw_full_unscaled_negative_mode_retained_alignment"],
        raw_full_unscaled_max_damping_asymmetry=raw_full[
            "raw_full_unscaled_max_damping_asymmetry"],
        raw_full_unscaled_max_added_mass_asymmetry=raw_full[
            "raw_full_unscaled_max_added_mass_asymmetry"],
    )


# ---------------------------------------------------------------------------
# persistence: strict-field npz with no pickle, invariant and hash checks

_META_KEYS = frozenset({
    "artifact_format", "module_version", "settings", "source_dof_order",
    "source_body_order", "provenance", "report_scalars", "content_hash",
})


def _expected_array_names() -> tuple:
    names = [
        "projection", "omegas", "wave_directions",
        "raw_added_mass", "raw_damping", "raw_added_mass_infinite",
        "derived_added_mass", "derived_damping", "derived_added_mass_infinite",
        "hydrostatic_stiffness", "excitation_force",
    ]
    for key in REPORT_MATRIX_KEYS:
        names.extend([
            f"reciprocity_defect__{key}", f"reciprocity_threshold__{key}",
            f"reciprocity_pass__{key}", f"correction_defect__{key}",
            f"correction_threshold__{key}", f"correction_pass__{key}",
            f"symmetry_defect__{key}",
        ])
    names.extend([
        "damping_eigenvalues", "damping_eigenvectors",
        "damping_negative_mode_frequency_indices",
        "damping_negative_mode_basis_indices",
        "damping_negative_mode_eigenvalues", "damping_negative_mode_vectors",
        "raw_full_unscaled_damping_min_eigenvalues",
        "raw_full_unscaled_negative_mode_indices",
        "raw_full_unscaled_negative_mode_eigenvalues",
        "raw_full_unscaled_negative_mode_vectors",
        "raw_full_unscaled_negative_mode_retained_alignment",
    ])
    return tuple(names)


def _artifact_arrays(artifact: DerivedHydroOperators) -> dict:
    return {name: arr for name, arr in artifact._named_arrays()}


def save_derived_operators(artifact: DerivedHydroOperators, path) -> None:
    """Atomically save a derived-operator artifact (npz, no pickle).

    The file content is verified on load by exact field sets, strict format
    version checks, invariant recomputation and the content hash.
    """
    from pathlib import Path

    if not isinstance(artifact, DerivedHydroOperators):
        raise DerivedOperatorSerializationError(
            "artifact must be a DerivedHydroOperators")
    target = Path(path)
    if target.parent and not target.parent.exists():
        raise DerivedOperatorSerializationError(
            f"target directory does not exist: {target.parent}")
    meta = {
        "artifact_format": DERIVED_OPERATORS_ARTIFACT_FORMAT,
        "module_version": DERIVED_OPERATORS_MODULE_VERSION,
        "settings": artifact.settings.to_dict(),
        "source_dof_order": list(artifact.source_dof_order),
        "source_body_order": list(artifact.source_body_order),
        "provenance": _jsonable(dict(artifact.provenance)),
        "report_scalars": {
            "symmetry_pass": {k: artifact.report.symmetry_pass[k]
                              for k in REPORT_MATRIX_KEYS},
            "damping_psd_pass": artifact.report.damping_psd_pass,
            "raw_full_coordinate_label": artifact.report.raw_full_coordinate_label,
            "raw_full_unscaled_max_damping_asymmetry":
                artifact.report.raw_full_unscaled_max_damping_asymmetry,
            "raw_full_unscaled_max_added_mass_asymmetry":
                artifact.report.raw_full_unscaled_max_added_mass_asymmetry,
        },
        "content_hash": artifact.content_hash(),
    }
    arrays = _artifact_arrays(artifact)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent),
                                    prefix=target.name + ".", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with open(tmp, "wb") as handle:
            np.savez(handle, meta=np.array(_canonical_json(meta)), **arrays)
        os.replace(tmp, target)
    except Exception as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise DerivedOperatorSerializationError(
            f"failed to save derived operators to {target}: {exc}") from exc


def _report_from_arrays(settings, raw, scalars) -> DerivedOperatorReport:
    def matrix_map(kind):
        return {k: raw[f"{kind}__{k}"] for k in REPORT_MATRIX_KEYS}

    return DerivedOperatorReport(
        criteria=settings.criteria,
        normalization_name=settings.normalization.name,
        retained_dofs=settings.retained_dofs,
        omegas=raw["omegas"],
        reciprocity={
            "defect": matrix_map("reciprocity_defect"),
            "threshold": matrix_map("reciprocity_threshold"),
            "pass": matrix_map("reciprocity_pass"),
        },
        correction={
            "defect": matrix_map("correction_defect"),
            "threshold": matrix_map("correction_threshold"),
            "pass": matrix_map("correction_pass"),
        },
        symmetry_defect=matrix_map("symmetry_defect"),
        symmetry_pass=scalars["symmetry_pass"],
        damping_eigenvalues=raw["damping_eigenvalues"],
        damping_eigenvectors=raw["damping_eigenvectors"],
        damping_negative_mode_frequency_indices=raw[
            "damping_negative_mode_frequency_indices"],
        damping_negative_mode_basis_indices=raw[
            "damping_negative_mode_basis_indices"],
        damping_negative_mode_eigenvalues=raw[
            "damping_negative_mode_eigenvalues"],
        damping_negative_mode_vectors=raw["damping_negative_mode_vectors"],
        damping_psd_pass=scalars["damping_psd_pass"],
        raw_full_coordinate_label=scalars["raw_full_coordinate_label"],
        raw_full_unscaled_damping_min_eigenvalues=raw[
            "raw_full_unscaled_damping_min_eigenvalues"],
        raw_full_unscaled_negative_mode_indices=raw[
            "raw_full_unscaled_negative_mode_indices"],
        raw_full_unscaled_negative_mode_eigenvalues=raw[
            "raw_full_unscaled_negative_mode_eigenvalues"],
        raw_full_unscaled_negative_mode_vectors=raw[
            "raw_full_unscaled_negative_mode_vectors"],
        raw_full_unscaled_negative_mode_retained_alignment=raw[
            "raw_full_unscaled_negative_mode_retained_alignment"],
        raw_full_unscaled_max_damping_asymmetry=scalars[
            "raw_full_unscaled_max_damping_asymmetry"],
        raw_full_unscaled_max_added_mass_asymmetry=scalars[
            "raw_full_unscaled_max_added_mass_asymmetry"],
    )


def load_derived_operators(path, source_dataset=None) -> DerivedHydroOperators:
    """Load and verify a derived-operator artifact.

    Refuses: non-npz files, pickle payloads, unknown or missing fields,
    format/version mismatches, tampered arrays or metadata (content hash),
    and any stored report result that does not match the recomputed
    invariants.

    ``source_dataset`` (optional): when the raw source dataset is
    available, pass it here to run the full raw-source proof
    (:meth:`DerivedHydroOperators.verify_raw_source` - complete
    rederivation, never the recorded hash alone). A failed proof is a
    typed :class:`DerivedOperatorSerializationError`; the loaded artifact
    is then not the derivation of the provided source.
    """
    from pathlib import Path

    target = Path(path)
    if not target.exists():
        raise DerivedOperatorSerializationError(f"file does not exist: {target}")
    # Materialise arrays while the archive is owned by this scope. Later
    # validation/source-proof errors can retain their traceback, so relying
    # on garbage collection leaves the NPZ open on Windows refusal paths.
    try:
        with np.load(target, allow_pickle=False) as archive:
            names = set(archive.files)
            expected = set(_expected_array_names()) | {"meta"}
            if "raw_damping_far_field" in names:
                expected = expected | {"raw_damping_far_field"}
            if "meta" not in names:
                raise DerivedOperatorSerializationError(
                    f"{target} is missing the meta record; not a "
                    f"{DERIVED_OPERATORS_ARTIFACT_FORMAT} artifact")
            unknown = names - expected
            missing = expected - names
            if unknown or missing:
                raise DerivedOperatorSerializationError(
                    f"{target} field mismatch (unknown: {sorted(unknown)}, "
                    f"missing: {sorted(missing)}); refusing")
            data = {name: archive[name] for name in names}
    except DerivedOperatorSerializationError:
        raise
    except Exception as exc:
        raise DerivedOperatorSerializationError(
            f"failed to load {target}: {exc}") from exc

    try:
        meta = json.loads(str(data["meta"][()]))
    except Exception as exc:
        raise DerivedOperatorSerializationError(
            f"{target} has a corrupt meta record: {exc}") from exc
    _require(isinstance(meta, Mapping), DerivedOperatorSerializationError(
        f"{target} meta record is not a JSON object"))
    meta_keys = set(meta)
    if meta_keys != _META_KEYS:
        raise DerivedOperatorSerializationError(
            f"{target} meta field mismatch (unknown: {sorted(meta_keys - _META_KEYS)}, "
            f"missing: {sorted(_META_KEYS - meta_keys)}); refusing")
    _require(meta["artifact_format"] == DERIVED_OPERATORS_ARTIFACT_FORMAT,
             DerivedOperatorSerializationError(
                 f"{target} is not a {DERIVED_OPERATORS_ARTIFACT_FORMAT} artifact "
                 f"(got {meta['artifact_format']!r})"))
    _require(meta["module_version"] == DERIVED_OPERATORS_MODULE_VERSION,
             DerivedOperatorSerializationError(
                 f"{target} module version mismatch: stored "
                 f"{meta['module_version']!r}, expected "
                 f"{DERIVED_OPERATORS_MODULE_VERSION!r}"))

    try:
        settings = DerivedOperatorSettings.from_dict(meta["settings"])
    except DerivedOperatorError as exc:
        raise DerivedOperatorSerializationError(
            f"{target} has invalid or tampered settings: {exc}") from exc

    raw = {name: data[name] for name in _expected_array_names()}
    far_present = "raw_damping_far_field" in data
    _require(far_present == (settings.correction_policy in FAR_FIELD_POLICIES),
             DerivedOperatorSerializationError(
                 f"{target} far-field damping presence does not match its policy"))
    report = _report_from_arrays(settings, raw, meta["report_scalars"])
    try:
        artifact = DerivedHydroOperators(
            settings=settings,
            source_dof_order=tuple(meta["source_dof_order"]),
            source_body_order=tuple(meta["source_body_order"]),
            projection=raw["projection"],
            omegas=raw["omegas"],
            wave_directions=raw["wave_directions"],
            raw_added_mass=raw["raw_added_mass"],
            raw_damping=raw["raw_damping"],
            raw_added_mass_infinite=raw["raw_added_mass_infinite"],
            derived_added_mass=raw["derived_added_mass"],
            derived_damping=raw["derived_damping"],
            derived_added_mass_infinite=raw["derived_added_mass_infinite"],
            hydrostatic_stiffness=raw["hydrostatic_stiffness"],
            excitation_force=raw["excitation_force"],
            report=report,
            provenance=meta["provenance"],
            raw_damping_far_field=data["raw_damping_far_field"] if far_present else None,
        )
    except DerivedOperatorError as exc:
        raise DerivedOperatorSerializationError(
            f"{target} failed artifact invariant verification: {exc}") from exc

    expected_hash = str(meta["content_hash"])
    _require(expected_hash == artifact.content_hash(),
             DerivedOperatorSerializationError(
                 f"{target} failed the content-hash integrity check "
                 f"(stored {expected_hash!r}, computed {artifact.content_hash()!r})"))
    if source_dataset is not None:
        try:
            artifact.verify_raw_source(source_dataset)
        except (DerivedOperatorError, HydroError) as exc:
            raise DerivedOperatorSerializationError(
                f"{target} failed the raw-source proof against the provided "
                f"source dataset: {exc}") from exc
    return artifact
