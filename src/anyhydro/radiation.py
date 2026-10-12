"""Radiation preprocessing: causal memory kernel from neutral datasets.

Development slice. Governing contract:
C:/ANYworkspaceWork/anymarine-evidence/radiation-contract.md

ANYhydro conventions: phasor ``x(t) = Re{X * exp(-i*omega*t)}`` and
influence-row / radiating-column matrices. The positive left-hand radiation
term is ``A_inf * a + K * v`` with the causal kernel

    K(t) = (2/pi) * integral_0^infinity B(omega) * cos(omega*t) d omega

and the reconstructions

    B_rec(w) = integral_0^infinity K(t) * cos(w*t) dt
    A_rec(w) = A_inf - integral_0^infinity K(t) * sin(w*t) dt / w   (w > 0)
    A_rec(0) = A_inf - integral_0^infinity t * K(t) dt     (explicit limit)

The sine sign follows from substituting the ``exp(-i*omega*t)`` phasor into
the causal convolution; it is never flipped to match a positive-exponent
convention. An independent exponential-kernel oracle with
``B = C*lambda/(lambda^2+w^2)`` and ``A = A_inf - C/(lambda^2+w^2)`` has the
closed-form kernel ``K(t) = C * exp(-lambda*t)`` and is used by the tests
and the analytic example.

This module is development-only: kernel artifacts never inherit a qualified
status from their source dataset, and nothing here confers scientific
qualification. Base import touches only numpy, xarray and the standard
library (plus anyhydro base modules) - never Qt, ANYsolver or Capytaine.
"""

from __future__ import annotations

import hashlib
import json
import numbers
import os
import tempfile
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

import numpy as np
import xarray as xr

from .dof import parse_dof_key
from .errors import HydroError
from .hashing import dataset_content_hash
from .operators import DerivedHydroOperators, DerivedOperatorAdmissionError
from .schema import validate_for_time_domain, validate_hydro_dataset

RADIATION_MODULE_VERSION = "1.0.0"
RADIATION_ARTIFACT_FORMAT = "anyhydro-radiation-kernel-v1"

# Development-slice limits (radiation-contract.md).
MAX_INPUT_FREQUENCIES = 513
MAX_TIME_SAMPLES = 1001
MAX_DOFS = 12
MIN_FREQUENCY_SAMPLES = 3

# Explicit long-kernel admission (docs/LONG_KERNEL_CHECKPOINT.md): the
# default development cap stays MAX_TIME_SAMPLES = 1001. A caller may
# explicitly declare a larger per-construction cap through
# RadiationExecutionLimits, hard-bounded by MAX_TIME_SAMPLES_CEILING;
# there is no global cap increase and no silent admission.
MIN_TIME_SAMPLES = 2
MAX_TIME_SAMPLES_CEILING = 20001

LOW_TAIL_EXPLICIT_ZERO = "explicit_zero_sample"
LOW_TAIL_LINEAR_TO_ZERO = "linear_to_zero_extension"
LOW_TAIL_POLICIES = (LOW_TAIL_EXPLICIT_ZERO, LOW_TAIL_LINEAR_TO_ZERO)

QUADRATURE_FREQUENCY_TRAPEZOID = "frequency_trapezoid"
QUADRATURE_PIECEWISE_LINEAR = "piecewise_linear_exact"
QUADRATURE_METHODS = (QUADRATURE_FREQUENCY_TRAPEZOID, QUADRATURE_PIECEWISE_LINEAR)

HIGH_TAIL_TRUNCATE = "truncate"
HIGH_TAIL_POWER_LAW = "power_law_extension"
HIGH_TAIL_POLICIES = (HIGH_TAIL_TRUNCATE, HIGH_TAIL_POWER_LAW)
#: geometric ratio between consecutive power-law tail samples (linear
#: interpolation error of nu^-n is about n(n+1)(r-1)^2/8 relative)
HIGH_TAIL_RATIO = 1.01
#: allocation bound on the number of geometric tail samples
MAX_HIGH_TAIL_SAMPLES = 20000
HIGH_TAIL_SCOPE_FULL = "full"
HIGH_TAIL_SCOPE_BODY_DIAGONAL = "body_diagonal"
HIGH_TAIL_SCOPES = (HIGH_TAIL_SCOPE_FULL, HIGH_TAIL_SCOPE_BODY_DIAGONAL)
A_INF_COMPUTED = "computed"
A_INF_BAND_CONSISTENT = "band_consistent"
A_INF_POLICIES = (A_INF_COMPUTED, A_INF_BAND_CONSISTENT)

ARTIFACT_QUALIFICATION_STATUS = "unqualified"


class RadiationError(HydroError):
    """A radiation preprocessing input or artifact is invalid."""


class RadiationSettingsError(RadiationError):
    """Radiation settings are invalid or incomplete."""


class RadiationLimitError(RadiationError):
    """A development-slice limit (frequencies, time samples, DOFs) was exceeded."""


class ReconstructionCriteriaError(RadiationError):
    """Caller reconstruction criteria are invalid for the requested comparison."""


class RadiationSerializationError(RadiationError):
    """A radiation kernel artifact could not be saved, loaded or verified."""


def _require(condition: bool, error: Exception) -> None:
    if not condition:
        raise error


def _validate_time_limit(name: str, value) -> int:
    """Typed admission of one radiation execution-limit control.

    Rejects bools, fractional/non-integral values and non-finite values
    before any allocation; the value must lie within
    ``[MIN_TIME_SAMPLES, MAX_TIME_SAMPLES_CEILING]``.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise RadiationSettingsError(
            f"execution limit {name!r} must be an integer (got "
            f"{type(value).__name__}: {value!r}); bools, fractional and "
            "non-finite controls are refused"
        )
    ivalue = int(value)
    if ivalue < MIN_TIME_SAMPLES:
        raise RadiationLimitError(
            f"execution limit {name!r} must be at least {MIN_TIME_SAMPLES} (got {ivalue})"
        )
    if ivalue > MAX_TIME_SAMPLES_CEILING:
        raise RadiationLimitError(
            f"execution limit {name!r} exceeds its hard ceiling "
            f"{MAX_TIME_SAMPLES_CEILING} (got {ivalue})"
        )
    return ivalue


@dataclass(frozen=True, init=False)
class RadiationExecutionLimits:
    """Frozen public execution limits for one kernel construction.

    Defaults preserve the development cap: ``max_time_samples=1001``.
    Explicit limits may admit longer causal kernels up to the hard
    ceiling ``MAX_TIME_SAMPLES_CEILING = 20001``. Every control is
    validated at construction: bool, fractional, non-finite, unknown and
    out-of-range controls raise typed errors before any allocation.
    Limits are execution bounds only and confer no scientific
    qualification; they never loosen any other validation.
    """

    max_time_samples: int

    def __init__(
        self,
        *,
        max_time_samples: int = MAX_TIME_SAMPLES,
        **unknown: Any,
    ) -> None:
        if unknown:
            raise RadiationSettingsError(
                f"unknown execution-limit control(s) {sorted(unknown)}; "
                "RadiationExecutionLimits supports only ['max_time_samples'] "
                "and refuses silent unknowns"
            )
        object.__setattr__(
            self, "max_time_samples",
            _validate_time_limit("max_time_samples", max_time_samples),
        )

    def as_dict(self) -> dict:
        """The effective limits as a plain JSON-serializable dict."""
        return {"max_time_samples": int(self.max_time_samples)}

    def describe(self) -> str:
        """Compact human-readable description for provenance records."""
        return f"max_time_samples={self.max_time_samples}"


def _frozen_float_array(value, name: str, ndim: Optional[int] = None) -> np.ndarray:
    """Defensive copy as float64, finite-checked, truly read-only.

    The copy is backed by an immutable bytes buffer, so neither the array
    nor any view of it can re-enable writes via ``setflags(write=True)``:
    the read-only state is structural, not a mutable flag.
    """
    try:
        arr = np.array(value, dtype=np.float64, copy=True, order="C")
    except (TypeError, ValueError) as exc:
        raise RadiationSettingsError(f"{name} is not a numeric array: {exc}") from exc
    if ndim is not None and arr.ndim != ndim:
        raise RadiationSettingsError(f"{name} must have {ndim} dimension(s), got shape {arr.shape}")
    if arr.size and not np.all(np.isfinite(arr)):
        raise RadiationSettingsError(f"{name} contains non-finite values")
    if arr.size == 0:
        arr.setflags(write=False)
        return arr
    return np.frombuffer(arr.tobytes(), dtype=np.float64).reshape(arr.shape)


def _deep_freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Deterministic trapezoidal quadrature weights for a 1-D grid (>= 2 points)."""
    if x.ndim != 1 or x.size < 2:
        raise RadiationError("quadrature grid must be 1-D with at least 2 points")
    w = np.empty_like(x)
    w[0] = (x[1] - x[0]) / 2.0
    w[-1] = (x[-1] - x[-2]) / 2.0
    if x.size > 2:
        w[1:-1] = (x[2:] - x[:-2]) / 2.0
    return w


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# Exact top-level keys a serialized settings mapping may contain. The
# reader refuses anything else (fail-closed compatibility with the old
# RadiationSettings(**settings_dict) constructor, which rejected unknown
# top-level keys).
SETTINGS_SERIALIZED_KEYS = frozenset({
    "dt", "n_time", "symmetry_tol", "passivity_tol", "low_tail_policy",
    "zero_frequency_damping", "extension_name", "time_window",
    "execution_limits", "quadrature_method",
    "high_tail_policy", "high_tail_exponent", "high_tail_upper", "high_tail_name",
    "high_tail_scope", "a_inf_policy", "a_inf_calibration_min", "a_inf_calibration_max",
})


@dataclass(frozen=True)
class RadiationSettings:
    """Immutable settings for one kernel construction.

    Fields (all explicit; there is no blanket scientific default):

    - ``quadrature_method``: legacy ``frequency_trapezoid`` (omitted from old
      serialized settings) or explicit ``piecewise_linear_exact``. The latter
      integrates the finite-band componentwise linear damping interpolant exactly
      against cos(omega*t); it does not prove interpolation/tail adequacy.
    - ``dt`` / ``n_time``: uniform time grid ``t_k = k*dt``, ``k = 0..n_time-1``,
      starting at zero with ``dt > 0``.
    - ``time_window``: optional explicit kernel time window; when given, the
      kernel is zero for ``t > time_window``. Default ``None`` (no window).
    - ``low_tail_policy``: ``"explicit_zero_sample"`` (default) requires
      ``zero_frequency_damping`` (a full coupled matrix, the explicit
      ``B(0)``); ``"linear_to_zero_extension"`` requires ``extension_name``
      and extends B linearly from the first grid frequency to ``B(0) = 0``.
    - ``symmetry_tol`` / ``passivity_tol``: explicit validation tolerances
      used for diagnostics. Violations are reported and preserved, never
      clamped or silently repaired.
    - ``execution_limits``: optional explicit :class:`RadiationExecutionLimits`
      admitting a longer kernel (up to the hard ceiling). Default ``None``
      keeps the development cap ``MAX_TIME_SAMPLES`` and preserves all old
      serialized settings and hashes; explicit limits participate in the
      settings dict, the artifact content hash, provenance and persistence.
    """

    dt: float
    n_time: int
    symmetry_tol: float
    passivity_tol: float
    low_tail_policy: str = LOW_TAIL_EXPLICIT_ZERO
    zero_frequency_damping: Optional[np.ndarray] = None
    extension_name: Optional[str] = None
    time_window: Optional[float] = None
    execution_limits: Optional[RadiationExecutionLimits] = None
    quadrature_method: str = QUADRATURE_FREQUENCY_TRAPEZOID
    high_tail_policy: str = HIGH_TAIL_TRUNCATE
    high_tail_exponent: Optional[float] = None
    high_tail_upper: Optional[float] = None
    high_tail_name: Optional[str] = None
    high_tail_scope: str = HIGH_TAIL_SCOPE_FULL
    a_inf_policy: str = A_INF_COMPUTED
    a_inf_calibration_min: Optional[float] = None
    a_inf_calibration_max: Optional[float] = None

    def __post_init__(self) -> None:
        try:
            dt = float(self.dt)
        except (TypeError, ValueError) as exc:
            raise RadiationSettingsError(f"dt must be a float: {exc}") from exc
        _require(np.isfinite(dt) and dt > 0.0, RadiationSettingsError("dt must be finite and > 0"))
        object.__setattr__(self, "dt", dt)

        _require(isinstance(self.quadrature_method, str)
                 and self.quadrature_method in QUADRATURE_METHODS,
                 RadiationSettingsError(f"quadrature_method must be one of {QUADRATURE_METHODS}"))

        limits = self.execution_limits
        if limits is not None and not isinstance(limits, RadiationExecutionLimits):
            raise RadiationSettingsError(
                "execution_limits must be a RadiationExecutionLimits instance or "
                f"None (got {type(limits).__name__}); plain values are refused"
            )
        object.__setattr__(self, "execution_limits", limits)

        if isinstance(self.n_time, bool) or not isinstance(self.n_time, (int, np.integer)):
            raise RadiationSettingsError("n_time must be an integer")
        n_time = int(self.n_time)
        effective_max = MAX_TIME_SAMPLES if limits is None else int(limits.max_time_samples)
        _require(2 <= n_time <= effective_max,
                 RadiationLimitError(f"n_time must be within [2, {effective_max}], got {n_time}"))
        object.__setattr__(self, "n_time", n_time)

        duration = (n_time - 1) * dt
        _require(np.isfinite(duration),
                 RadiationSettingsError(
                     f"settings time-grid duration (n_time-1)*dt must be finite, got {duration}"))

        for name in ("symmetry_tol", "passivity_tol"):
            try:
                tol = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise RadiationSettingsError(f"{name} must be a float: {exc}") from exc
            _require(np.isfinite(tol) and tol >= 0.0,
                     RadiationSettingsError(f"{name} must be finite and >= 0 (explicit tolerance required)"))
            object.__setattr__(self, name, tol)

        policy = str(self.low_tail_policy)
        _require(policy in LOW_TAIL_POLICIES,
                 RadiationSettingsError(f"low_tail_policy must be one of {LOW_TAIL_POLICIES}, got {policy!r}"))
        object.__setattr__(self, "low_tail_policy", policy)

        if policy == LOW_TAIL_EXPLICIT_ZERO:
            _require(self.zero_frequency_damping is not None,
                     RadiationSettingsError(
                         "default low-tail policy 'explicit_zero_sample' requires an explicit "
                         "zero_frequency_damping matrix (B(0)); the final finite-frequency sample "
                         "is never substituted"))
            _require(self.extension_name is None,
                     RadiationSettingsError("extension_name is only valid with the linear-to-zero extension policy"))
            b0 = _frozen_float_array(self.zero_frequency_damping, "zero_frequency_damping", ndim=2)
            object.__setattr__(self, "zero_frequency_damping", b0)
        else:
            _require(self.zero_frequency_damping is None,
                     RadiationSettingsError(
                         "the linear-to-zero extension defines B(0)=0 itself; "
                         "zero_frequency_damping must not be supplied"))
            name = self.extension_name
            _require(isinstance(name, str) and name.strip(),
                     RadiationSettingsError(
                         "the linear-to-zero B extension must be separately named "
                         "via extension_name (recorded as an assumption)"))
            object.__setattr__(self, "extension_name", str(name))

        high = str(self.high_tail_policy)
        _require(high in HIGH_TAIL_POLICIES,
                 RadiationSettingsError(
                     f"high_tail_policy must be one of {HIGH_TAIL_POLICIES}, got {high!r}"))
        object.__setattr__(self, "high_tail_policy", high)
        if high == HIGH_TAIL_TRUNCATE:
            _require(self.high_tail_exponent is None and self.high_tail_upper is None
                     and self.high_tail_name is None,
                     RadiationSettingsError(
                         "high_tail_exponent/upper/name are only valid with the "
                         "power-law high-tail extension"))
        else:
            _require(self.quadrature_method == QUADRATURE_PIECEWISE_LINEAR,
                     RadiationSettingsError(
                         "the power-law high tail requires quadrature_method="
                         f"{QUADRATURE_PIECEWISE_LINEAR!r} (exact on the oscillatory tail)"))
            try:
                exponent = float(self.high_tail_exponent)
                upper = float(self.high_tail_upper)
            except (TypeError, ValueError) as exc:
                raise RadiationSettingsError(
                    f"power-law high tail needs numeric exponent and upper: {exc}") from exc
            _require(np.isfinite(exponent) and exponent > 1.0,
                     RadiationSettingsError("high_tail_exponent must be finite and > 1"))
            _require(np.isfinite(upper) and upper > 0.0,
                     RadiationSettingsError("high_tail_upper must be finite and > 0"))
            name = self.high_tail_name
            _require(isinstance(name, str) and name.strip(),
                     RadiationSettingsError(
                         "the power-law high tail must be named via high_tail_name "
                         "(recorded as an assumption)"))
            object.__setattr__(self, "high_tail_exponent", exponent)
            object.__setattr__(self, "high_tail_upper", upper)
            object.__setattr__(self, "high_tail_name", str(name))
        scope = str(self.high_tail_scope)
        _require(scope in HIGH_TAIL_SCOPES,
                 RadiationSettingsError(f"high_tail_scope must be one of {HIGH_TAIL_SCOPES}"))
        _require(scope == HIGH_TAIL_SCOPE_FULL or high == HIGH_TAIL_POWER_LAW,
                 RadiationSettingsError("high_tail_scope applies to the power-law tail only"))
        object.__setattr__(self, "high_tail_scope", scope)

        a_policy = str(self.a_inf_policy)
        _require(a_policy in A_INF_POLICIES,
                 RadiationSettingsError(f"a_inf_policy must be one of {A_INF_POLICIES}"))
        object.__setattr__(self, "a_inf_policy", a_policy)
        if a_policy == A_INF_COMPUTED:
            _require(self.a_inf_calibration_min is None and self.a_inf_calibration_max is None,
                     RadiationSettingsError(
                         "a_inf calibration band is only valid with a_inf_policy='band_consistent'"))
        else:
            try:
                lo = float(self.a_inf_calibration_min)
                hi = float(self.a_inf_calibration_max)
            except (TypeError, ValueError) as exc:
                raise RadiationSettingsError(
                    f"band-consistent A_inf needs a numeric calibration band: {exc}") from exc
            _require(np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi,
                     RadiationSettingsError("calibration band must satisfy 0 < min < max"))
            object.__setattr__(self, "a_inf_calibration_min", lo)
            object.__setattr__(self, "a_inf_calibration_max", hi)

        if self.time_window is not None:
            try:
                window = float(self.time_window)
            except (TypeError, ValueError) as exc:
                raise RadiationSettingsError(f"time_window must be a float: {exc}") from exc
            duration = (n_time - 1) * dt
            _require(np.isfinite(window) and 0.0 < window <= duration,
                     RadiationSettingsError(
                         f"time_window must be finite in (0, {duration}] (the kernel time grid duration)"))
            object.__setattr__(self, "time_window", window)

    @property
    def kernel_duration(self) -> float:
        return (self.n_time - 1) * self.dt

    def to_dict(self) -> dict:
        data = {
            "dt": float(self.dt),
            "n_time": int(self.n_time),
            "symmetry_tol": float(self.symmetry_tol),
            "passivity_tol": float(self.passivity_tol),
            "low_tail_policy": self.low_tail_policy,
            "zero_frequency_damping": None if self.zero_frequency_damping is None
            else np.asarray(self.zero_frequency_damping).tolist(),
            "extension_name": self.extension_name,
            "time_window": self.time_window,
        }
        # None keeps the exact old serialized form (no new key, old hashes
        # preserved); explicit limits are recorded and hashed.
        if self.execution_limits is not None:
            data["execution_limits"] = self.execution_limits.as_dict()
        # Legacy default preserves the exact serialized keys and artifact hashes.
        if self.quadrature_method != QUADRATURE_FREQUENCY_TRAPEZOID:
            data["quadrature_method"] = self.quadrature_method
        # Default truncation keeps the exact old serialized form and hashes.
        if self.high_tail_policy != HIGH_TAIL_TRUNCATE:
            data["high_tail_policy"] = self.high_tail_policy
            data["high_tail_exponent"] = float(self.high_tail_exponent)
            data["high_tail_upper"] = float(self.high_tail_upper)
            data["high_tail_name"] = self.high_tail_name
        if self.high_tail_scope != HIGH_TAIL_SCOPE_FULL:
            data["high_tail_scope"] = self.high_tail_scope
        if self.a_inf_policy != A_INF_COMPUTED:
            data["a_inf_policy"] = self.a_inf_policy
            data["a_inf_calibration_min"] = float(self.a_inf_calibration_min)
            data["a_inf_calibration_max"] = float(self.a_inf_calibration_max)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RadiationSettings":
        """Faithfully reconstruct settings from a serialized dict.

        Explicit execution limits are reconstructed and enforced (unknown
        limit controls are refused); a serialized artifact whose declared
        controls were deleted refuses here through the ordinary
        ``n_time`` bound instead of silently admitting a longer kernel.
        Unknown top-level keys are refused before any settings or kernel
        allocation: only the exact serialized fields are admitted.
        """
        if not isinstance(data, Mapping):
            raise RadiationSettingsError(
                f"serialized settings must be a mapping (got {type(data).__name__})")
        unknown = set(data) - SETTINGS_SERIALIZED_KEYS
        if unknown:
            raise RadiationSettingsError(
                "serialized settings contain unknown top-level key(s): "
                f"{sorted(unknown)}")
        limits_raw = data.get("execution_limits")
        execution_limits = None
        if limits_raw is not None:
            if not isinstance(limits_raw, Mapping):
                raise RadiationSettingsError(
                    "serialized execution_limits must be a mapping of limit controls")
            execution_limits = RadiationExecutionLimits(**dict(limits_raw))
        zero_damping = None
        try:
            if data["low_tail_policy"] == LOW_TAIL_EXPLICIT_ZERO:
                zero_damping = np.asarray(data["zero_frequency_damping"], dtype=np.float64)
            return cls(
                dt=data["dt"],
                n_time=data["n_time"],
                symmetry_tol=data["symmetry_tol"],
                passivity_tol=data["passivity_tol"],
                low_tail_policy=data["low_tail_policy"],
                zero_frequency_damping=zero_damping,
                extension_name=data["extension_name"],
                time_window=data["time_window"],
                execution_limits=execution_limits,
                quadrature_method=data.get("quadrature_method", QUADRATURE_FREQUENCY_TRAPEZOID),
                high_tail_policy=data.get("high_tail_policy", HIGH_TAIL_TRUNCATE),
                high_tail_exponent=data.get("high_tail_exponent"),
                high_tail_upper=data.get("high_tail_upper"),
                high_tail_name=data.get("high_tail_name"),
                high_tail_scope=data.get("high_tail_scope", HIGH_TAIL_SCOPE_FULL),
                a_inf_policy=data.get("a_inf_policy", A_INF_COMPUTED),
                a_inf_calibration_min=data.get("a_inf_calibration_min"),
                a_inf_calibration_max=data.get("a_inf_calibration_max"),
            )
        except KeyError as exc:
            raise RadiationSettingsError(
                f"serialized settings are missing field {exc}") from exc


@dataclass(frozen=True)
class ReconstructionCriteria:
    """Caller-declared reconstruction admission criteria (all fields required).

    Comparison happens on the reference grid frequencies inside the finite
    window ``[omega_min, omega_max]``. Per matrix coefficient (i, j):

        max_error_ij <= abs_tol + rel_tol * max_over_window(|reference_ij|)

    Added mass and damping have separate absolute tolerances (different
    units). Near-zero reference entries are covered by the absolute
    allowance. There is no blanket scientific default and no admission on
    damping alone: both matrices must pass.
    """

    omega_min: float
    omega_max: float
    added_mass_abs_tol: float
    added_mass_rel_tol: float
    damping_abs_tol: float
    damping_rel_tol: float

    def __post_init__(self) -> None:
        for name in ("omega_min", "omega_max"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ReconstructionCriteriaError(f"{name} must be a float: {exc}") from exc
            _require(np.isfinite(value) and value > 0.0,
                     ReconstructionCriteriaError(f"{name} must be finite and > 0"))
            object.__setattr__(self, name, value)
        _require(self.omega_min <= self.omega_max,
                 ReconstructionCriteriaError("omega_min must be <= omega_max"))
        for name in ("added_mass_abs_tol", "added_mass_rel_tol",
                     "damping_abs_tol", "damping_rel_tol"):
            try:
                tol = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ReconstructionCriteriaError(f"{name} must be a float: {exc}") from exc
            _require(np.isfinite(tol) and tol >= 0.0,
                     ReconstructionCriteriaError(f"{name} must be finite and >= 0"))
            object.__setattr__(self, name, tol)

    def to_dict(self) -> dict:
        return {
            "omega_min": self.omega_min,
            "omega_max": self.omega_max,
            "added_mass_abs_tol": self.added_mass_abs_tol,
            "added_mass_rel_tol": self.added_mass_rel_tol,
            "damping_abs_tol": self.damping_abs_tol,
            "damping_rel_tol": self.damping_rel_tol,
        }


@dataclass(frozen=True)
class FullBlockReconstructionCriteria:
    """W5 K1-K2 reconstruction criteria (adopted 2026-10-12), per window frequency.

    K1: ||Z_rec - Z||_F, ||A_rec - A||_F, ||B_rec - B||_F each within
    ``block_rel_tol`` of the reference norm (Z = B + i omega A).
    K2: for A and B, energetic modes of S(X) (eigenvalue >= energetic_fraction
    times the largest; clusters closer than cluster_fraction use the spectral
    norm of V^T (S(X_rec) - S(X)) V over the cluster minimum) within
    ``modal_rel_tol``. Per-coefficient errors are reported only (K3).
    """

    omega_min: float
    omega_max: float
    block_rel_tol: float
    modal_rel_tol: float
    energetic_fraction: float
    cluster_fraction: float

    FORM = "full_block"

    def __post_init__(self) -> None:
        for name in ("omega_min", "omega_max", "block_rel_tol", "modal_rel_tol",
                     "energetic_fraction", "cluster_fraction"):
            try:
                value = float(getattr(self, name))
            except (TypeError, ValueError) as exc:
                raise ReconstructionCriteriaError(f"{name} must be a float: {exc}") from exc
            _require(np.isfinite(value) and value > 0.0,
                     ReconstructionCriteriaError(f"{name} must be finite and > 0"))
            object.__setattr__(self, name, value)
        _require(self.omega_min <= self.omega_max,
                 ReconstructionCriteriaError("omega_min must be <= omega_max"))
        _require(self.energetic_fraction < 1.0 and self.cluster_fraction < 1.0,
                 ReconstructionCriteriaError("fractions must be below 1"))

    def to_dict(self) -> dict:
        return {"form": self.FORM, "omega_min": self.omega_min, "omega_max": self.omega_max,
                "block_rel_tol": self.block_rel_tol, "modal_rel_tol": self.modal_rel_tol,
                "energetic_fraction": self.energetic_fraction,
                "cluster_fraction": self.cluster_fraction}


def reconstruction_criteria_from_dict(data: Mapping[str, Any]):
    """Per-coefficient criteria (legacy form) or ``form='full_block'`` criteria."""
    if not isinstance(data, Mapping):
        raise ReconstructionCriteriaError("criteria must be a mapping")
    fields = dict(data)
    form = fields.pop("form", None)
    try:
        if form == FullBlockReconstructionCriteria.FORM:
            return FullBlockReconstructionCriteria(**fields)
        _require(form is None, ReconstructionCriteriaError(f"unknown criteria form {form!r}"))
        return ReconstructionCriteria(**fields)
    except TypeError as exc:
        raise ReconstructionCriteriaError(f"invalid reconstruction criteria: {exc}") from exc


def _energetic_groups(lam, fraction, cluster):
    groups, current = [], []
    for k in np.flatnonzero(lam >= fraction * lam[0]):
        if current and (lam[current[-1]] - lam[k]) >= cluster * lam[current[-1]]:
            groups.append(current)
            current = []
        current.append(int(k))
    if current:
        groups.append(current)
    return groups


def _rel_norm(diff, ref):
    """||diff||/||ref||; 0 when both vanish, inf when only ref vanishes."""
    d, r = float(np.linalg.norm(diff)), float(np.linalg.norm(ref))
    if r == 0.0:
        return 0.0 if d == 0.0 else np.inf
    return d / r


def _weak_modes(rec, ref, fraction):
    """K3 report: weak eigenvalues of S(ref) and the reconstructed Rayleigh values."""
    sr = 0.5 * (ref + ref.T)
    sc = 0.5 * (rec + rec.T)
    lam, vec = np.linalg.eigh(sr)
    top = float(np.max(lam))
    out = []
    for k in range(lam.size):
        if top > 0.0 and lam[k] >= fraction * top:
            continue
        out.append([float(lam[k]), float(vec[:, k] @ sc @ vec[:, k])])
    return out


def _modal_error(rec, ref, fraction, cluster):
    """Largest K2 relative error over the energetic modes of S(ref).

    A reference without a positive eigenvalue has no energetic mode (0).
    """
    sr = 0.5 * (ref + ref.T)
    sc = 0.5 * (rec + rec.T)
    lam, vec = np.linalg.eigh(sr)
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    if not lam[0] > 0.0:
        return 0.0
    worst = 0.0
    for g in _energetic_groups(lam, fraction, cluster):
        V = vec[:, g]
        if len(g) == 1:
            err = abs(float(V[:, 0] @ sc @ V[:, 0]) - lam[g[0]]) / lam[g[0]]
        else:
            err = float(np.linalg.norm(V.T @ (sc - sr) @ V, 2) / np.min(lam[g]))
        worst = max(worst, err)
    return worst


@dataclass(frozen=True)
class FullBlockReconstructionReport:
    """K1/K2 results per window frequency plus K3 reported quantities."""

    criteria: FullBlockReconstructionCriteria
    window_omegas: np.ndarray
    impedance_rel_error: np.ndarray
    added_mass_rel_error: np.ndarray
    damping_rel_error: np.ndarray
    added_mass_modal_error: np.ndarray
    damping_modal_error: np.ndarray
    added_mass_coefficient_error: np.ndarray
    damping_coefficient_error: np.ndarray
    a_inf_difference: Optional[np.ndarray] = None
    #: K3: per window frequency, [reference weak eigenvalue, reconstructed Rayleigh]
    added_mass_weak_modes: tuple = ()
    damping_weak_modes: tuple = ()

    @property
    def block_pass(self) -> bool:
        tol = self.criteria.block_rel_tol
        return bool(np.all(self.impedance_rel_error <= tol)
                    and np.all(self.added_mass_rel_error <= tol)
                    and np.all(self.damping_rel_error <= tol))

    @property
    def modal_pass(self) -> bool:
        tol = self.criteria.modal_rel_tol
        return bool(np.all(self.added_mass_modal_error <= tol)
                    and np.all(self.damping_modal_error <= tol))

    @property
    def overall_pass(self) -> bool:
        return self.block_pass and self.modal_pass

    def summary(self) -> dict:
        return {
            "criteria": self.criteria.to_dict(),
            "n_window_frequencies": int(self.window_omegas.size),
            "window_omega_range": [float(self.window_omegas[0]), float(self.window_omegas[-1])],
            "overall_pass": self.overall_pass,
            "block_pass": self.block_pass,
            "modal_pass": self.modal_pass,
            "max_impedance_rel_error": float(np.max(self.impedance_rel_error)),
            "max_added_mass_error": float(np.max(self.added_mass_rel_error)),
            "max_damping_error": float(np.max(self.damping_rel_error)),
            "max_added_mass_modal_error": float(np.max(self.added_mass_modal_error)),
            "max_damping_modal_error": float(np.max(self.damping_modal_error)),
            "reported_max_added_mass_coefficient_error":
                float(np.max(self.added_mass_coefficient_error)),
            "reported_max_damping_coefficient_error":
                float(np.max(self.damping_coefficient_error)),
            "reported_a_inf_difference": None if self.a_inf_difference is None
            else np.asarray(self.a_inf_difference).tolist(),
            "reported_added_mass_weak_modes": [[list(x) for x in f] for f in self.added_mass_weak_modes],
            "reported_damping_weak_modes": [[list(x) for x in f] for f in self.damping_weak_modes],
        }


def _full_block_report(criteria, window, a_rec, b_rec, a_ref, b_ref, a_inf_difference):
    z_err, a_err, b_err, am, bm, aw, bw = [], [], [], [], [], [], []
    for i, w in enumerate(window):
        z_ref = b_ref[i] + 1j * w * a_ref[i]
        z_rec = b_rec[i] + 1j * w * a_rec[i]
        z_err.append(_rel_norm(z_rec - z_ref, z_ref))
        a_err.append(_rel_norm(a_rec[i] - a_ref[i], a_ref[i]))
        b_err.append(_rel_norm(b_rec[i] - b_ref[i], b_ref[i]))
        am.append(_modal_error(a_rec[i], a_ref[i], criteria.energetic_fraction, criteria.cluster_fraction))
        bm.append(_modal_error(b_rec[i], b_ref[i], criteria.energetic_fraction, criteria.cluster_fraction))
        aw.append(_weak_modes(a_rec[i], a_ref[i], criteria.energetic_fraction))
        bw.append(_weak_modes(b_rec[i], b_ref[i], criteria.energetic_fraction))
    return FullBlockReconstructionReport(
        criteria=criteria, window_omegas=np.asarray(window),
        impedance_rel_error=np.asarray(z_err), added_mass_rel_error=np.asarray(a_err),
        damping_rel_error=np.asarray(b_err), added_mass_modal_error=np.asarray(am),
        damping_modal_error=np.asarray(bm),
        added_mass_coefficient_error=np.max(np.abs(a_rec - a_ref), axis=0),
        damping_coefficient_error=np.max(np.abs(b_rec - b_ref), axis=0),
        a_inf_difference=a_inf_difference,
        added_mass_weak_modes=tuple(tuple(tuple(x) for x in f) for f in aw),
        damping_weak_modes=tuple(tuple(tuple(x) for x in f) for f in bw))


def verify_band_consistent_added_mass_infinite(kernel, computed_a_inf, omegas, added_mass,
                                               rel_tol: float = 1e-9) -> np.ndarray:
    """Prove a kernel's band-consistent A_inf; return the kernel A_inf.

    Requires ``a_inf_policy='band_consistent'``, the recorded computed A_inf
    equal to ``computed_a_inf`` exactly, and the kernel's own A_inf to make
    the mean of (A - A_rec) over the recorded calibration frequencies vanish
    (to ``rel_tol`` of max|A|), with those frequencies drawn from ``omegas``.
    """
    _require(kernel.settings.a_inf_policy == A_INF_BAND_CONSISTENT,
             RadiationError("kernel does not declare a band-consistent A_inf"))
    prov = kernel.provenance
    recorded = np.asarray(prov.get("a_inf_computed"), dtype=np.float64)
    _require(recorded.shape == np.shape(computed_a_inf)
             and bool(np.array_equal(recorded, np.asarray(computed_a_inf, dtype=np.float64))),
             RadiationError("recorded computed A_inf does not match the source A_inf"))
    cal = np.asarray(prov.get("a_inf_calibration_omegas"), dtype=np.float64)
    om = np.asarray(omegas, dtype=np.float64)
    st = kernel.settings
    selection = (om >= st.a_inf_calibration_min) & (om <= st.a_inf_calibration_max)
    _require(cal.ndim == 1 and cal.size >= 2 and bool(np.array_equal(cal, om[selection])),
             RadiationError("recorded calibration frequencies are not exactly the source "
                            "frequencies inside the declared calibration band"))
    idx = np.flatnonzero(selection)
    delta = np.asarray(prov.get("a_inf_band_consistent_delta"), dtype=np.float64)
    _require(delta.shape == recorded.shape
             and bool(np.array_equal(recorded + delta, np.asarray(kernel.added_mass_infinite))),
             RadiationError("recorded A_inf difference does not reproduce the kernel A_inf"))
    a_rec, _ = kernel.reconstruct(cal)
    residual = np.mean(np.asarray(added_mass)[idx] - a_rec, axis=0)
    scale = float(np.max(np.abs(np.asarray(added_mass)[idx])))
    _require(float(np.max(np.abs(residual))) <= rel_tol * scale,
             RadiationError("kernel A_inf is not band-consistent over its calibration band"))
    return np.asarray(kernel.added_mass_infinite)


@dataclass(frozen=True)
class RadiationDiagnostics:
    """Observed tail, symmetry and passivity diagnostics (errors preserved).

    Nothing here is clamped, symmetrized or repaired: violations are
    counted and reported so a caller can act on them.
    """

    frequency_grid: np.ndarray
    zero_frequency_damping: np.ndarray
    damping_at_omega_max: np.ndarray
    omega_max: float
    max_frequency_spacing: float
    omega_max_times_dt: float
    kernel_duration: float
    duration_over_spacing_period: float
    max_abs_damping_at_omega_max: float
    max_damping_asymmetry: float
    max_added_mass_asymmetry: float
    damping_symmetry_violations: int
    added_mass_symmetry_violations: int
    min_damping_eigenvalue: float
    passivity_violations: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "frequency_grid",
                           _frozen_float_array(self.frequency_grid, "frequency_grid", ndim=1))
        for name in ("zero_frequency_damping", "damping_at_omega_max"):
            object.__setattr__(self, name, _frozen_float_array(getattr(self, name), name, ndim=2))
        for name in ("omega_max", "max_frequency_spacing", "omega_max_times_dt",
                     "kernel_duration", "duration_over_spacing_period",
                     "max_abs_damping_at_omega_max", "max_damping_asymmetry",
                     "max_added_mass_asymmetry", "min_damping_eigenvalue"):
            value = float(getattr(self, name))
            _require(np.isfinite(value), RadiationError(f"diagnostic {name} is not finite"))
            object.__setattr__(self, name, value)
        for name in ("damping_symmetry_violations", "added_mass_symmetry_violations",
                     "passivity_violations"):
            value = int(getattr(self, name))
            _require(value >= 0, RadiationError(f"diagnostic {name} must be >= 0"))
            object.__setattr__(self, name, value)

    def scalars(self) -> dict:
        return {
            "omega_max": self.omega_max,
            "max_frequency_spacing": self.max_frequency_spacing,
            "omega_max_times_dt": self.omega_max_times_dt,
            "kernel_duration": self.kernel_duration,
            "duration_over_spacing_period": self.duration_over_spacing_period,
            "max_abs_damping_at_omega_max": self.max_abs_damping_at_omega_max,
            "max_damping_asymmetry": self.max_damping_asymmetry,
            "max_added_mass_asymmetry": self.max_added_mass_asymmetry,
            "damping_symmetry_violations": self.damping_symmetry_violations,
            "added_mass_symmetry_violations": self.added_mass_symmetry_violations,
            "min_damping_eigenvalue": self.min_damping_eigenvalue,
            "passivity_violations": self.passivity_violations,
        }


@dataclass(frozen=True)
class RadiationKernel:
    """Immutable causal radiation kernel artifact with full provenance.

    Arrays are defensive copies backed by immutable buffers (truly
    read-only, not merely flagged); the provenance mapping is
    deep-frozen. ``qualification_status`` is always ``"unqualified"``: a
    kernel artifact cannot inherit a qualified status from its source
    dataset.
    """

    settings: RadiationSettings
    time: np.ndarray
    dof_keys: Tuple[str, ...]
    body_ids: Tuple[str, ...]
    kernel: np.ndarray
    added_mass_infinite: np.ndarray
    diagnostics: RadiationDiagnostics
    provenance: Mapping[str, Any]
    assumptions: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.settings, RadiationSettings):
            raise RadiationError("settings must be a RadiationSettings")
        if not isinstance(self.diagnostics, RadiationDiagnostics):
            raise RadiationError("diagnostics must be a RadiationDiagnostics")

        time = _frozen_float_array(self.time, "time", ndim=1)
        _require(time.size == self.settings.n_time,
                 RadiationError("time grid size does not match settings.n_time"))
        _require(time.size >= 2 and time[0] == 0.0,
                 RadiationError("time grid must start at zero with at least 2 samples"))
        diffs = np.diff(time)
        _require(np.all(diffs > 0.0) and np.allclose(diffs, diffs[0]),
                 RadiationError("time grid must be uniform with dt > 0"))
        expected_grid = np.arange(time.size, dtype=np.float64) * self.settings.dt
        _require(bool(np.allclose(time, expected_grid, rtol=1.0e-9, atol=self.settings.dt * 1.0e-9)),
                 RadiationError("the complete time grid must match settings: "
                                "t_k = k*dt for k = 0..n_time-1"))
        object.__setattr__(self, "time", time)

        dof_keys = tuple(str(k) for k in self.dof_keys)
        _require(len(set(dof_keys)) == len(dof_keys) and dof_keys,
                 RadiationError("dof_keys must be non-empty and unique"))
        _require(len(dof_keys) <= MAX_DOFS,
                 RadiationLimitError(f"kernel has {len(dof_keys)} DOFs; development limit is {MAX_DOFS}"))
        object.__setattr__(self, "dof_keys", dof_keys)
        body_ids = tuple(str(b) for b in self.body_ids)
        _require(len(set(body_ids)) == len(body_ids) and body_ids,
                 RadiationError("body_ids must be non-empty and unique"))
        object.__setattr__(self, "body_ids", body_ids)

        n_dof = len(dof_keys)
        kernel = _frozen_float_array(self.kernel, "kernel", ndim=3)
        _require(kernel.shape == (time.size, n_dof, n_dof),
                 RadiationError(f"kernel must have shape {(time.size, n_dof, n_dof)}, got {kernel.shape}"))
        object.__setattr__(self, "kernel", kernel)

        a_inf = _frozen_float_array(self.added_mass_infinite, "added_mass_infinite", ndim=2)
        _require(a_inf.shape == (n_dof, n_dof),
                 RadiationError(f"added_mass_infinite must have shape {(n_dof, n_dof)}"))
        object.__setattr__(self, "added_mass_infinite", a_inf)

        _require(isinstance(self.provenance, Mapping) and self.provenance,
                 RadiationError("provenance must be a non-empty mapping"))
        object.__setattr__(self, "provenance", _deep_freeze(dict(self.provenance)))
        assumptions = tuple(str(a) for a in self.assumptions)
        object.__setattr__(self, "assumptions", assumptions)

    @property
    def n_dof(self) -> int:
        return len(self.dof_keys)

    @property
    def qualification_status(self) -> str:
        return ARTIFACT_QUALIFICATION_STATUS

    def reconstruct(self, omegas) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct added mass and damping at the given frequencies.

        Returns ``(A_rec, B_rec)``, each of shape ``(n_w, n_dof, n_dof)``.
        Frequencies must be finite and >= 0. At ``w = 0`` the explicit limit
        ``A_rec(0) = A_inf - integral t*K(t) dt`` is used; ``B_rec(0)`` is
        ``integral K(t) dt``. A non-finite phase product ``w*t`` (overflow)
        or any non-finite reconstructed output raises
        :class:`RadiationError`; caller data may be finite while intermediate
        arithmetic overflows.
        """
        w = np.atleast_1d(np.asarray(omegas, dtype=np.float64))
        if w.ndim != 1:
            raise RadiationError("omegas must be a 1-D array of frequencies")
        if w.size and (not np.all(np.isfinite(w)) or np.any(w < 0.0)):
            raise RadiationError("omegas must be finite and >= 0")

        t = self.time
        with np.errstate(over="ignore", invalid="ignore"):
            phase = np.outer(w, t)
        if not np.all(np.isfinite(phase)):
            raise RadiationError(
                "reconstruction phase product w*t is not finite (overflow); refusing")
        tw = _trapezoid_weights(t)
        n_t = t.size
        n_dof = self.n_dof
        kernel_flat = self.kernel.reshape(n_t, n_dof * n_dof)
        weighted = tw[:, None] * kernel_flat

        cos_wt = np.cos(phase)
        b_rec = (cos_wt @ weighted).reshape(w.size, n_dof, n_dof)

        a_rec = np.empty((w.size, n_dof, n_dof), dtype=np.float64)
        a_inf_flat = self.added_mass_infinite.reshape(n_dof * n_dof)
        positive = w > 0.0
        if np.any(positive):
            sin_wt = np.sin(phase[positive])
            a_int = sin_wt @ weighted
            a_rec[positive] = (a_inf_flat[None, :] - a_int / w[positive, None]).reshape(
                int(np.count_nonzero(positive)), n_dof, n_dof)
        if not np.all(positive):
            # explicit zero-frequency limit: A(0) = A_inf - integral t*K(t) dt;
            # the single limit matrix broadcasts across all zero query entries
            t_weighted = (t * tw) @ kernel_flat
            a_rec[~positive] = (a_inf_flat - t_weighted).reshape(n_dof, n_dof)
        if not np.all(np.isfinite(a_rec)) or not np.all(np.isfinite(b_rec)):
            raise RadiationError("reconstruction produced non-finite output; refusing")
        return a_rec, b_rec

    def content_hash(self) -> str:
        """Deterministic SHA-256 over settings, provenance, assumptions and data."""
        h = hashlib.sha256()
        for payload in (
            RADIATION_ARTIFACT_FORMAT.encode("utf-8"),
            RADIATION_MODULE_VERSION.encode("utf-8"),
            _canonical_json(self.settings.to_dict()).encode("utf-8"),
            _canonical_json(dict(self.provenance)).encode("utf-8"),
            _canonical_json(list(self.assumptions)).encode("utf-8"),
            _canonical_json(list(self.dof_keys)).encode("utf-8"),
            _canonical_json(list(self.body_ids)).encode("utf-8"),
            _canonical_json(self.diagnostics.scalars()).encode("utf-8"),
        ):
            h.update(len(payload).to_bytes(8, "little"))
            h.update(payload)
        for name, arr in (
            ("time", self.time),
            ("kernel", self.kernel),
            ("added_mass_infinite", self.added_mass_infinite),
            ("frequency_grid", self.diagnostics.frequency_grid),
            ("zero_frequency_damping", self.diagnostics.zero_frequency_damping),
            ("damping_at_omega_max", self.diagnostics.damping_at_omega_max),
        ):
            data = np.ascontiguousarray(arr, dtype="<f8").tobytes()
            h.update(name.encode("utf-8"))
            h.update(len(data).to_bytes(8, "little"))
            h.update(data)
        return h.hexdigest()


@dataclass(frozen=True)
class ReconstructionReport:
    """Criteria-based reconstruction comparison of BOTH added mass and damping.

    ``overall_pass`` requires every coefficient of both matrices to satisfy
    the caller criteria; there is no admission on damping alone.
    """

    criteria: ReconstructionCriteria
    window_omegas: np.ndarray
    added_mass_max_error: np.ndarray
    damping_max_error: np.ndarray
    added_mass_threshold: np.ndarray
    damping_threshold: np.ndarray
    added_mass_pass: np.ndarray
    damping_pass: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.criteria, ReconstructionCriteria):
            raise ReconstructionCriteriaError("criteria must be ReconstructionCriteria")
        for name in ("window_omegas", "added_mass_max_error", "damping_max_error",
                     "added_mass_threshold", "damping_threshold",
                     "added_mass_pass", "damping_pass"):
            arr = np.asarray(getattr(self, name))
            if arr.size == 0:
                arr = arr.copy()
                arr.setflags(write=False)
            else:
                # immutable backing buffer: views cannot re-enable writes
                buf = np.ascontiguousarray(arr).tobytes()
                arr = np.frombuffer(buf, dtype=arr.dtype).reshape(arr.shape)
            object.__setattr__(self, name, arr)
        _require(self.window_omegas.size >= 1,
                 ReconstructionCriteriaError("window must contain at least one frequency"))

    @property
    def overall_pass(self) -> bool:
        return bool(np.all(self.added_mass_pass) and np.all(self.damping_pass))

    @property
    def max_added_mass_error(self) -> float:
        return float(np.max(self.added_mass_max_error))

    @property
    def max_damping_error(self) -> float:
        return float(np.max(self.damping_max_error))

    def _worst_ratio(self, error: np.ndarray, threshold: np.ndarray) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(threshold > 0.0, error / threshold,
                             np.where(error > 0.0, np.inf, 0.0))
        return float(np.max(ratio))

    def summary(self) -> dict:
        return {
            "criteria": self.criteria.to_dict(),
            "n_window_frequencies": int(self.window_omegas.size),
            "window_omega_range": [float(self.window_omegas[0]), float(self.window_omegas[-1])],
            "overall_pass": self.overall_pass,
            "added_mass_pass": bool(np.all(self.added_mass_pass)),
            "damping_pass": bool(np.all(self.damping_pass)),
            "max_added_mass_error": self.max_added_mass_error,
            "max_damping_error": self.max_damping_error,
            "added_mass_worst_error_to_threshold_ratio":
                self._worst_ratio(self.added_mass_max_error, self.added_mass_threshold),
            "damping_worst_error_to_threshold_ratio":
                self._worst_ratio(self.damping_max_error, self.damping_threshold),
            "failing_added_mass_coefficients": [
                [int(i), int(j)]
                for i, j in zip(*np.where(~self.added_mass_pass))
            ],
            "failing_damping_coefficients": [
                [int(i), int(j)]
                for i, j in zip(*np.where(~self.damping_pass))
            ],
        }


def _decode_json_attr(attrs: Mapping[str, Any], key: str) -> Any:
    raw = attrs[key]
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def radiation_kernel_from_dataset(ds: xr.Dataset, settings: RadiationSettings) -> RadiationKernel:
    """Dataset bridge: build the causal kernel from a validated neutral dataset.

    The dataset must pass :func:`anyhydro.validate_for_time_domain`
    (explicit ``added_mass_infinite`` with non-empty source provenance; the
    last finite-frequency sample is never substituted). Full coupled DOF
    matrices and the dataset's body/DOF ordering are preserved as-is.
    """
    if not isinstance(settings, RadiationSettings):
        raise RadiationSettingsError("settings must be a RadiationSettings instance")

    validate_for_time_domain(ds)

    omegas = np.asarray(ds["omega"].values, dtype=np.float64)
    _require(omegas.size >= MIN_FREQUENCY_SAMPLES,
             RadiationError(f"frequency grid needs at least {MIN_FREQUENCY_SAMPLES} samples, got {omegas.size}"))
    _require(omegas.size <= MAX_INPUT_FREQUENCIES,
             RadiationLimitError(f"{omegas.size} input frequencies exceed the development limit of {MAX_INPUT_FREQUENCIES}"))

    influenced = [str(v) for v in ds["influenced_dof"].values.tolist()]
    radiating = [str(v) for v in ds["radiating_dof"].values.tolist()]
    _require(influenced == radiating,
             RadiationError("influenced_dof and radiating_dof must share the same order "
                            "for entrywise kernel transforms; refusing to reorder silently"))
    n_dof = len(influenced)
    _require(n_dof <= MAX_DOFS,
             RadiationLimitError(f"{n_dof} DOFs exceed the development limit of {MAX_DOFS}"))

    bodies = [str(v) for v in ds["body"].values.tolist()]
    damping = np.array(ds["radiation_damping"].values, dtype=np.float64, copy=True)
    added_mass = np.array(ds["added_mass"].values, dtype=np.float64, copy=True)
    a_inf = np.array(ds["added_mass_infinite"].values, dtype=np.float64, copy=True)
    a_inf_source = str(ds["added_mass_infinite"].attrs["source"])
    _require(a_inf_source.strip(), RadiationError("added_mass_infinite has an empty source"))

    provenance = {
        "source_dataset_hash": dataset_content_hash(ds),
        "source_request_hash": str(ds.attrs["request_hash"]),
        "source_mesh_hashes": _decode_json_attr(ds.attrs, "mesh_hashes"),
        "source_body_identity": _decode_json_attr(ds.attrs, "body_identity"),
        "source_reference_transformations": (
            _decode_json_attr(ds.attrs, "reference_transformations")
            if "reference_transformations" in ds.attrs else None
        ),
        "a_inf_source": a_inf_source,
        "phasor_convention": str(ds.attrs["phasor_convention"]),
        "source_qualification_status": str(ds.attrs["qualification_status"]),
        "source_schema_version": str(ds.attrs["schema_version"]),
        "source_backend": str(ds.attrs["backend"]),
    }
    if settings.execution_limits is not None:
        provenance["execution_limits"] = settings.execution_limits.as_dict()

    return _kernel_from_arrays(
        omegas=omegas,
        added_mass=added_mass,
        damping=damping,
        a_inf=a_inf,
        dof_keys=tuple(influenced),
        body_ids=tuple(bodies),
        settings=settings,
        provenance=provenance,
    )


def _piecewise_linear_cosine_integral(omega, damping, time):
    """Exact finite-band cosine integral of the declared linear interpolant.

    For each interval midpoint m, halfwidth h and endpoint matrices Ba,Bb,
    z=h*t, M=(Ba+Bb)/2 and D=(Bb-Ba)/2, the integral is
    2h*[M*cos(m*t)*sinc(z) + D*sin(m*t)*sinc_prime(z)].
    The small-z series avoids cancellation in (sin(z)-z*cos(z))/z**2;
    the t=0 limit is exactly the interval trapezoid of B itself. Blocks
    bound temporary memory independently of the admitted long time horizon.
    No damping sample, tail, added mass, symmetry or PSD value is changed.
    """
    width = np.diff(omega)
    half = width / 2.0
    middle = (omega[1:] + omega[:-1]) / 2.0
    average = ((damping[1:] + damping[:-1]) / 2.0).reshape(width.size, -1)
    difference = (damping[1:] - damping[:-1]).reshape(width.size, -1)
    result = np.empty((time.size, average.shape[1]), dtype=np.float64)
    for start in range(0, time.size, 512):
        times = time[start:start + 512, None]
        z = times * half[None, :]
        j1 = np.empty_like(z)
        small = np.abs(z) < 0.05
        zs = z[small]
        j1[small] = zs / 3.0 - zs**3 / 30.0 + zs**5 / 840.0 - zs**7 / 45360.0
        zl = z[~small]
        j1[~small] = (np.sin(zl) / zl - np.cos(zl)) / zl
        phase = times * middle[None, :]
        result[start:start + 512] = (
            (width[None, :] * np.cos(phase) * np.sinc(z / np.pi)) @ average
            - (half[None, :] * np.sin(phase) * j1) @ difference
        )
    return result.reshape((time.size,) + damping.shape[1:])


def _reconstruct_added_mass(time, kernel, a_inf, omegas):
    """A_rec(w) = A_inf - (1/w) int K(t) sin(w t) dt (trapezoid in t), w > 0."""
    w = np.asarray(omegas, dtype=np.float64)
    n_dof = a_inf.shape[0]
    weighted = _trapezoid_weights(time)[:, None] * kernel.reshape(time.size, n_dof * n_dof)
    a_int = np.sin(np.outer(w, time)) @ weighted
    return (a_inf.reshape(n_dof * n_dof)[None, :] - a_int / w[:, None]).reshape(w.size, n_dof, n_dof)


def _kernel_from_arrays(
    *,
    omegas,
    added_mass,
    damping,
    a_inf,
    dof_keys,
    body_ids,
    settings: RadiationSettings,
    provenance: Mapping[str, Any],
    extra_assumptions: Tuple[str, ...] = (),
) -> RadiationKernel:
    """Shared private kernel core for the dataset bridge and the derived path.

    Both public builders call this one numerical implementation: low tail,
    causal kernel quadrature, time window, diagnostics and base assumptions
    are identical code, so a retained-coordinate kernel is never a copied
    hydrodynamic truth. The dataset bridge passes exactly its historical
    inputs, preserving old artifacts and hashes bit-for-bit.
    """
    omegas = np.asarray(omegas, dtype=np.float64)
    added_mass = np.asarray(added_mass, dtype=np.float64)
    damping = np.asarray(damping, dtype=np.float64)
    a_inf = np.asarray(a_inf, dtype=np.float64)
    dof_keys = tuple(str(k) for k in dof_keys)
    body_ids = tuple(str(b) for b in body_ids)
    n_dof = len(dof_keys)

    # --- low tail: explicit zero-frequency sample or named linear-to-zero extension
    if settings.low_tail_policy == LOW_TAIL_EXPLICIT_ZERO:
        b0 = np.array(settings.zero_frequency_damping, dtype=np.float64, copy=True)
        _require(b0.shape == (n_dof, n_dof),
                 RadiationSettingsError(f"zero_frequency_damping must have shape {(n_dof, n_dof)}, got {b0.shape}"))
        low_tail_assumption = (
            "low_tail_policy=explicit_zero_sample: caller-supplied explicit "
            "zero-frequency damping sample B(0); no substitution from finite frequencies"
        )
    else:
        b0 = np.zeros((n_dof, n_dof), dtype=np.float64)
        low_tail_assumption = (
            f"low_tail_policy=linear_to_zero_extension: named extension "
            f"{settings.extension_name!r} extends B linearly from omega_1 to B(0)=0 "
            f"(recorded assumption, not solver data)"
        )

    omega_ext = np.concatenate(([0.0], omegas))
    b_ext = np.concatenate((b0[None, :, :], damping), axis=0)
    omega_max = float(omegas[-1])
    if settings.high_tail_policy == HIGH_TAIL_POWER_LAW:
        _require(settings.high_tail_upper > omega_max,
                 RadiationSettingsError(
                     f"high_tail_upper {settings.high_tail_upper} must exceed the last "
                     f"grid frequency {omega_max}"))
        # the kernel is sampled at dt: integrated bandwidth stays below Nyquist
        _require(settings.high_tail_upper * settings.dt <= np.pi,
                 RadiationSettingsError(
                     f"high_tail_upper*dt = {settings.high_tail_upper * settings.dt:.6g} "
                     "exceeds pi: the declared tail would integrate above the kernel's "
                     "Nyquist frequency"))
        span = np.log(settings.high_tail_upper) - np.log(omega_max)
        n_tail = int(np.ceil(span / np.log(HIGH_TAIL_RATIO)))
        _require(n_tail <= MAX_HIGH_TAIL_SAMPLES,
                 RadiationLimitError(
                     f"power-law tail needs {n_tail} samples > {MAX_HIGH_TAIL_SAMPLES}"))
        nu = np.geomspace(omega_max, settings.high_tail_upper, n_tail + 1)[1:]
        factor = (omega_max / nu) ** settings.high_tail_exponent
        tail_matrix = damping[-1]
        if settings.high_tail_scope == HIGH_TAIL_SCOPE_BODY_DIAGONAL:
            # each body's own block continues; inter-body coupling (oscillating
            # in omega, not a power law) is truncated. A principal block of a
            # PSD matrix is PSD, so passivity is inherited.
            owner = [parse_dof_key(key)[0] for key in dof_keys]
            same = np.array([[a == b for b in owner] for a in owner])
            tail_matrix = np.where(same, tail_matrix, 0.0)
        omega_ext = np.concatenate((omega_ext, nu))
        b_ext = np.concatenate((b_ext, factor[:, None, None] * tail_matrix[None, :, :]), axis=0)
    _require(bool(np.all(np.isfinite(b_ext))),
             RadiationError("extended damping data contains non-finite values; refusing"))

    # --- causal kernel: K(t) = (2/pi) * integral_0^omega_max B(omega) cos(omega t) d omega
    n_time = settings.n_time
    t = np.arange(n_time, dtype=np.float64) * settings.dt
    if settings.quadrature_method == QUADRATURE_PIECEWISE_LINEAR:
        kernel = (2.0 / np.pi) * _piecewise_linear_cosine_integral(omega_ext, b_ext, t)
    else:
        # Keep the legacy arithmetic and array order unchanged.
        weights = _trapezoid_weights(omega_ext)
        cos_t_w = np.cos(np.outer(t, omega_ext))
        kernel = (2.0 / np.pi) * (cos_t_w @ (weights[:, None] * b_ext.reshape(omega_ext.size, n_dof * n_dof)))
        kernel = kernel.reshape(n_time, n_dof, n_dof)
    _require(bool(np.all(np.isfinite(kernel))),
             RadiationError("computed kernel contains non-finite values; refusing"))

    if settings.time_window is not None:
        kernel[t > settings.time_window] = 0.0

    # --- diagnostics: high tail, symmetry, passivity (errors preserved, never clamped)
    damping_at_omega_max = damping[-1]
    max_spacing = float(np.max(np.diff(omega_ext)))
    kernel_duration = settings.kernel_duration

    asym_damping = np.abs(b_ext[:omegas.size + 1]
                          - np.transpose(b_ext[:omegas.size + 1], (0, 2, 1)))
    asym_added = np.abs(added_mass - np.transpose(added_mass, (0, 2, 1)))
    iu, ju = np.triu_indices(n_dof, k=1)
    damping_asym_per_pair = asym_damping[:, iu, ju].max(axis=0) if iu.size else np.zeros(0)
    added_asym_per_pair = asym_added[:, iu, ju].max(axis=0) if iu.size else np.zeros(0)
    damping_sym_violations = int(np.count_nonzero(damping_asym_per_pair > settings.symmetry_tol))
    added_sym_violations = int(np.count_nonzero(added_asym_per_pair > settings.symmetry_tol))

    # diagnostics over the source grid plus the declared low tail only (the
    # power-law tail is a positive multiple of B(omega_max))
    b_diag = b_ext[:omegas.size + 1]
    sym_damping = 0.5 * (b_diag + np.transpose(b_diag, (0, 2, 1)))
    min_eigs = np.array([np.linalg.eigvalsh(sym_damping[k]).min() for k in range(sym_damping.shape[0])])
    passivity_violations = int(np.count_nonzero(min_eigs < -settings.passivity_tol))

    diagnostics = RadiationDiagnostics(
        frequency_grid=omega_ext[:omegas.size + 1],
        zero_frequency_damping=b0,
        damping_at_omega_max=damping_at_omega_max,
        omega_max=omega_max,
        max_frequency_spacing=max_spacing,
        omega_max_times_dt=omega_max * settings.dt,
        kernel_duration=kernel_duration,
        duration_over_spacing_period=kernel_duration * max_spacing / (2.0 * np.pi),
        max_abs_damping_at_omega_max=float(np.max(np.abs(damping_at_omega_max))),
        max_damping_asymmetry=float(np.max(asym_damping)) if asym_damping.size else 0.0,
        max_added_mass_asymmetry=float(np.max(asym_added)) if asym_added.size else 0.0,
        damping_symmetry_violations=damping_sym_violations,
        added_mass_symmetry_violations=added_sym_violations,
        min_damping_eigenvalue=float(np.min(min_eigs)),
        passivity_violations=passivity_violations,
    )

    if settings.high_tail_policy == HIGH_TAIL_POWER_LAW:
        high_tail_assumption = (
            f"high_tail_policy=power_law_extension: named extension "
            f"{settings.high_tail_name!r} continues B(nu) = B(omega_max)*(omega_max/nu)"
            f"^{settings.high_tail_exponent:.17g} from omega_max={omega_max:.17g} to "
            f"{settings.high_tail_upper:.17g} rad/s (geometric samples, ratio "
            f"{HIGH_TAIL_RATIO}), scope {settings.high_tail_scope}, truncated beyond; integrated bandwidth "
            f"high_tail_upper*dt={settings.high_tail_upper * settings.dt:.17g} <= pi "
            f"(source omega_max*dt in the diagnostics); recorded assumption, not solver data")
    else:
        high_tail_assumption = (
            f"high_tail_truncated_at_omega_max={omega_max:.17g}: B is explicitly truncated "
            f"at the last grid frequency; no extrapolation, fitting or symmetrization")
    assumptions = [
        low_tail_assumption,
        high_tail_assumption,
        f"time_window={'none' if settings.time_window is None else repr(float(settings.time_window))}",
        "kernel_artifact_unqualified: development artifact; cannot inherit qualified "
        "status from the source dataset",
    ]
    if (settings.quadrature_method == QUADRATURE_PIECEWISE_LINEAR
            and settings.high_tail_policy == HIGH_TAIL_TRUNCATE):
        assumptions.append(
            "quadrature_method=piecewise_linear_exact: exact cosine integral of the "
            "componentwise linear damping interpolant, including the declared low tail; "
            "high tail remains explicitly truncated. Interpolation/source/tail adequacy "
            "and finite-time/discrete radiation passivity are not established. "
            "duration_over_spacing_period is a legacy spacing indicator, not an "
            "alias recurrence criterion for this method."
        )
    elif settings.quadrature_method == QUADRATURE_PIECEWISE_LINEAR:
        assumptions.append(
            "quadrature_method=piecewise_linear_exact: exact cosine integral of the "
            "componentwise linear damping interpolant, including the declared low tail "
            "and the declared power-law high tail up to high_tail_upper (truncated "
            "there). Interpolation/source/tail adequacy and finite-time/discrete "
            "radiation passivity are not established. duration_over_spacing_period "
            "is a legacy spacing indicator, not an alias recurrence criterion."
        )
    if settings.a_inf_policy == A_INF_BAND_CONSISTENT:
        cal = (omegas >= settings.a_inf_calibration_min) & (omegas <= settings.a_inf_calibration_max)
        _require(int(np.count_nonzero(cal)) >= 2,
                 RadiationSettingsError(
                     "the A_inf calibration band must contain at least two source frequencies"))
        a_rec_cal = _reconstruct_added_mass(t, kernel, a_inf, omegas[cal])
        delta = np.mean(added_mass[cal] - a_rec_cal, axis=0)
        a_inf_computed = a_inf
        a_inf = a_inf + delta
        provenance = dict(provenance)
        provenance["a_inf_computed"] = a_inf_computed.tolist()
        provenance["a_inf_band_consistent_delta"] = delta.tolist()
        provenance["a_inf_calibration_omegas"] = omegas[cal].tolist()
        assumptions.append(
            f"a_inf_policy=band_consistent: time-domain A_inf = computed A_inf + mean over "
            f"{int(np.count_nonzero(cal))} source frequencies in "
            f"[{settings.a_inf_calibration_min:.17g}, {settings.a_inf_calibration_max:.17g}] "
            "of (A - A_rec); computed A_inf and the difference are in the provenance "
            "(Ogilvie-consistent with this band-limited kernel)")
    assumptions.extend(str(a) for a in extra_assumptions)
    if settings.execution_limits is not None:
        assumptions.append(
            f"execution_limits={settings.execution_limits.describe()}: explicitly declared "
            "longer-kernel admission bound (execution bound only, no scientific "
            "qualification, no other validation loosened)"
        )

    return RadiationKernel(
        settings=settings,
        time=t,
        dof_keys=dof_keys,
        body_ids=body_ids,
        kernel=kernel,
        added_mass_infinite=a_inf,
        diagnostics=diagnostics,
        provenance=provenance,
        assumptions=tuple(assumptions),
    )


def radiation_kernel_from_derived_operators(
    derived: DerivedHydroOperators, settings: RadiationSettings,
    source_dataset: xr.Dataset = None,
) -> RadiationKernel:
    """Build a retained-coordinate kernel from an admitted derived artifact.

    Uses the same numerical kernel implementation as the dataset path
    (shared private core); no hydrodynamic truth is copied, no cap is
    relaxed and no schema is forged. ``settings`` is an ordinary
    :class:`RadiationSettings`; ``zero_frequency_damping`` (B(0)) is
    caller-supplied in retained coordinates. Non-admitted artifacts are a
    typed non-success: the artifact and its report remain inspectable, but
    no kernel is built.

    ``source_dataset`` (optional but mandatory in intent): consumers that
    hold the raw source dataset MUST pass it here so the full raw-source
    proof (complete rederivation under the stored settings, never the
    recorded hash alone) runs before consumption; a failed proof is a
    typed refusal.
    """
    if not isinstance(derived, DerivedHydroOperators):
        raise RadiationError(
            "derived must be a DerivedHydroOperators artifact "
            f"(got {type(derived).__name__})")
    if not isinstance(settings, RadiationSettings):
        raise RadiationSettingsError("settings must be a RadiationSettings instance")
    if source_dataset is not None:
        derived.verify_raw_source(source_dataset)
    if not derived.admitted:
        raise DerivedOperatorAdmissionError(
            "derived operator artifact is not admitted: at least one declared "
            "criterion failed (see the artifact report); radiation consumption "
            "is a typed non-success while the artifact remains inspectable")

    omegas = np.asarray(derived.omegas, dtype=np.float64)
    _require(omegas.size >= MIN_FREQUENCY_SAMPLES,
             RadiationError(f"frequency grid needs at least {MIN_FREQUENCY_SAMPLES} samples, got {omegas.size}"))
    _require(omegas.size <= MAX_INPUT_FREQUENCIES,
             RadiationLimitError(f"{omegas.size} input frequencies exceed the development limit of {MAX_INPUT_FREQUENCIES}"))
    n_ret = derived.n_retained
    _require(n_ret <= MAX_DOFS,
             RadiationLimitError(f"{n_ret} DOFs exceed the development limit of {MAX_DOFS}"))

    retained_bodies = tuple(dict.fromkeys(
        parse_dof_key(label)[0] for label in derived.retained_dofs))

    provenance = {
        "source_kind": "derived_operators",
        "derived_content_hash": derived.content_hash(),
        "source_dataset_hash": str(derived.provenance["source_dataset_hash"]),
        "derived_correction_policy": derived.settings.correction_policy,
        "derived_normalization_name": derived.settings.normalization.name,
        "derived_criteria": derived.settings.criteria.to_dict(),
        "retained_dofs": list(derived.retained_dofs),
        "a_inf_source": str(derived.provenance["a_inf_source"]),
        "phasor_convention": str(derived.provenance["phasor_convention"]),
        "source_qualification_status": str(derived.provenance["source_qualification_status"]),
    }
    if settings.execution_limits is not None:
        provenance["execution_limits"] = settings.execution_limits.as_dict()

    extra_assumptions = (
        "derived_operators_source: retained-coordinate kernel built from the "
        "admitted derived operator artifact (bound to its content hash and the "
        "raw source dataset hash); the raw full-6N source is not consumed here",
    )

    return _kernel_from_arrays(
        omegas=omegas,
        added_mass=derived.derived_added_mass,
        damping=derived.derived_damping,
        a_inf=derived.derived_added_mass_infinite,
        dof_keys=derived.retained_dofs,
        body_ids=retained_bodies,
        settings=settings,
        provenance=provenance,
        extra_assumptions=extra_assumptions,
    )


def _compare_window(rec, ref, abs_tol, rel_tol):
    """Shared per-coefficient comparison: error, threshold, pass arrays."""
    error = np.max(np.abs(rec - ref), axis=0)
    scale = np.max(np.abs(ref), axis=0)
    threshold = abs_tol + rel_tol * scale
    passed = error <= threshold
    return error, threshold, passed


def compare_reconstruction(
    kernel: RadiationKernel,
    reference_ds: xr.Dataset,
    criteria: ReconstructionCriteria,
) -> ReconstructionReport:
    """Compare reconstructed A and B against the reference in a caller window.

    The reference dataset must be the kernel's own source (content-hash
    bound). The window selects reference grid frequencies inside
    ``[criteria.omega_min, criteria.omega_max]``; each matrix coefficient must
    satisfy the prospective per-coefficient tolerance. Observed errors and
    failures are reported; nothing is admitted on damping alone.
    """
    if not isinstance(kernel, RadiationKernel):
        raise RadiationError("kernel must be a RadiationKernel")
    if not isinstance(criteria, ReconstructionCriteria):
        raise ReconstructionCriteriaError(
            "caller-declared ReconstructionCriteria are required; there is no default")

    validate_hydro_dataset(reference_ds)
    source_hash = kernel.provenance["source_dataset_hash"]
    reference_hash = dataset_content_hash(reference_ds)
    _require(reference_hash == source_hash,
             RadiationError(
                 "reference dataset content hash does not match the kernel source "
                 f"({reference_hash} != {source_hash}); refusing to compare against "
                 "a different dataset"))

    omegas = np.asarray(reference_ds["omega"].values, dtype=np.float64)
    selection = (omegas >= criteria.omega_min) & (omegas <= criteria.omega_max)
    _require(bool(np.any(selection)),
             ReconstructionCriteriaError(
                 f"window [{criteria.omega_min}, {criteria.omega_max}] contains no "
                 "reference grid frequencies"))
    window = omegas[selection]

    a_rec, b_rec = kernel.reconstruct(window)
    a_ref = np.asarray(reference_ds["added_mass"].values[selection], dtype=np.float64)
    b_ref = np.asarray(reference_ds["radiation_damping"].values[selection], dtype=np.float64)

    a_error, a_threshold, a_pass = _compare_window(
        a_rec, a_ref, criteria.added_mass_abs_tol, criteria.added_mass_rel_tol)
    b_error, b_threshold, b_pass = _compare_window(
        b_rec, b_ref, criteria.damping_abs_tol, criteria.damping_rel_tol)

    return ReconstructionReport(
        criteria=criteria,
        window_omegas=window,
        added_mass_max_error=a_error,
        damping_max_error=b_error,
        added_mass_threshold=a_threshold,
        damping_threshold=b_threshold,
        added_mass_pass=a_pass,
        damping_pass=b_pass,
    )


def compare_derived_reconstruction(
    kernel: RadiationKernel,
    derived: DerivedHydroOperators,
    criteria: ReconstructionCriteria,
    source_dataset: xr.Dataset = None,
) -> ReconstructionReport:
    """Compare a derived-source kernel's reconstructed A and B.

    The kernel must be the retained kernel built from exactly this derived
    artifact: the comparison is bound to the derived content hash AND the
    raw source dataset hash, and to the retained DOF order, correction
    policy, normalization and criteria identity. The reference is the
    derived A/B of the artifact itself - never the raw source relabelled.

    ``source_dataset`` (optional but mandatory in intent): consumers that
    hold the raw source dataset MUST pass it here so the full raw-source
    proof runs before the comparison; a failed proof is a typed refusal.
    """
    if not isinstance(kernel, RadiationKernel):
        raise RadiationError("kernel must be a RadiationKernel")
    if not isinstance(derived, DerivedHydroOperators):
        raise RadiationError(
            "derived must be a DerivedHydroOperators artifact "
            f"(got {type(derived).__name__})")
    if not isinstance(criteria, (ReconstructionCriteria, FullBlockReconstructionCriteria)):
        raise ReconstructionCriteriaError(
            "caller-declared ReconstructionCriteria are required; there is no default")
    if source_dataset is not None:
        derived.verify_raw_source(source_dataset)

    provenance = kernel.provenance
    _require(provenance.get("source_kind") == "derived_operators",
             RadiationError(
                 "kernel was not built from a derived operator artifact "
                 f"(source_kind={provenance.get('source_kind')!r})"))
    _require(provenance.get("derived_content_hash") == derived.content_hash(),
             RadiationError(
                 "kernel derived content hash does not match this artifact "
                 f"({provenance.get('derived_content_hash')} != "
                 f"{derived.content_hash()}); refusing to compare against a "
                 "different derived source"))
    _require(provenance.get("source_dataset_hash")
             == derived.provenance["source_dataset_hash"],
             RadiationError(
                 "kernel raw source dataset hash does not match the artifact "
                 "raw source; refusing to compare against a different source"))
    _require(tuple(kernel.dof_keys) == tuple(derived.retained_dofs),
             RadiationError(
                 "kernel DOF order does not match the artifact retained DOFs"))
    _require(provenance.get("derived_correction_policy")
             == derived.settings.correction_policy,
             RadiationError("kernel correction policy identity mismatch"))
    _require(provenance.get("derived_normalization_name")
             == derived.settings.normalization.name,
             RadiationError("kernel normalization identity mismatch"))
    _require(provenance.get("derived_criteria") == derived.settings.criteria.to_dict(),
             RadiationError("kernel admission criteria identity mismatch"))

    omegas = np.asarray(derived.omegas, dtype=np.float64)
    selection = (omegas >= criteria.omega_min) & (omegas <= criteria.omega_max)
    _require(bool(np.any(selection)),
             ReconstructionCriteriaError(
                 f"window [{criteria.omega_min}, {criteria.omega_max}] contains no "
                 "derived grid frequencies"))
    window = omegas[selection]

    a_rec, b_rec = kernel.reconstruct(window)
    a_ref = np.asarray(derived.derived_added_mass[selection], dtype=np.float64)
    b_ref = np.asarray(derived.derived_damping[selection], dtype=np.float64)
    if (isinstance(criteria, ReconstructionCriteria)
            and kernel.settings.a_inf_policy == A_INF_BAND_CONSISTENT):
        raise ReconstructionCriteriaError(
            "a band-consistent A_inf kernel is compared with the full-block criteria only")
    if isinstance(criteria, FullBlockReconstructionCriteria):
        a_inf_difference = None
        if kernel.settings.a_inf_policy == A_INF_BAND_CONSISTENT:
            _require(kernel.settings.a_inf_calibration_max < criteria.omega_min,
                     ReconstructionCriteriaError(
                         "the A_inf calibration band must lie below the evaluation window"))
            a_inf_kernel = verify_band_consistent_added_mass_infinite(
                kernel, derived.derived_added_mass_infinite, derived.omegas,
                derived.derived_added_mass)
            a_inf_difference = a_inf_kernel - np.asarray(derived.derived_added_mass_infinite)
        return _full_block_report(criteria, window, a_rec, b_rec, a_ref, b_ref, a_inf_difference)

    a_error, a_threshold, a_pass = _compare_window(
        a_rec, a_ref, criteria.added_mass_abs_tol, criteria.added_mass_rel_tol)
    b_error, b_threshold, b_pass = _compare_window(
        b_rec, b_ref, criteria.damping_abs_tol, criteria.damping_rel_tol)

    return ReconstructionReport(
        criteria=criteria,
        window_omegas=window,
        added_mass_max_error=a_error,
        damping_max_error=b_error,
        added_mass_threshold=a_threshold,
        damping_threshold=b_threshold,
        added_mass_pass=a_pass,
        damping_pass=b_pass,
    )


def _kernel_to_dataset(kernel: RadiationKernel) -> xr.Dataset:
    dof = np.array(kernel.dof_keys, dtype=object)
    return xr.Dataset(
        data_vars={
            "kernel": (("time", "influenced_dof", "radiating_dof"), kernel.kernel),
            "added_mass_infinite": (("influenced_dof", "radiating_dof"),
                                    kernel.added_mass_infinite),
            "zero_frequency_damping": (("influenced_dof", "radiating_dof"),
                                       kernel.diagnostics.zero_frequency_damping),
            "damping_at_omega_max": (("influenced_dof", "radiating_dof"),
                                     kernel.diagnostics.damping_at_omega_max),
        },
        coords={
            "time": kernel.time,
            "frequency": kernel.diagnostics.frequency_grid,
            "influenced_dof": dof,
            "radiating_dof": dof,
        },
        attrs={
            "artifact_format": RADIATION_ARTIFACT_FORMAT,
            "radiation_module_version": RADIATION_MODULE_VERSION,
            "qualification_status": kernel.qualification_status,
            "content_hash": kernel.content_hash(),
            "settings": _canonical_json(kernel.settings.to_dict()),
            "provenance": _canonical_json(dict(kernel.provenance)),
            "assumptions": _canonical_json(list(kernel.assumptions)),
            "body_ids": _canonical_json(list(kernel.body_ids)),
            "diagnostics_scalars": _canonical_json(kernel.diagnostics.scalars()),
        },
    )


def save_radiation_kernel(kernel: RadiationKernel, path) -> None:
    """Atomically save a kernel artifact to a deterministic NetCDF file.

    The file contains no timestamps: saving the same artifact twice
    produces byte-identical files (reproducible serialization).
    """
    if not isinstance(kernel, RadiationKernel):
        raise RadiationSerializationError("kernel must be a RadiationKernel")
    from pathlib import Path

    target = Path(path)
    if target.parent and not target.parent.exists():
        raise RadiationSerializationError(f"target directory does not exist: {target.parent}")

    storable = _kernel_to_dataset(kernel)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=target.name + ".", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        storable.to_netcdf(tmp, engine="scipy", format="NETCDF3_64BIT")
        os.replace(tmp, target)
    except Exception as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise RadiationSerializationError(f"failed to save radiation kernel to {target}: {exc}") from exc


def load_radiation_kernel(path) -> RadiationKernel:
    """Load and verify a radiation kernel artifact (deterministic NetCDF)."""
    from pathlib import Path

    target = Path(path)
    if not target.exists():
        raise RadiationSerializationError(f"file does not exist: {target}")
    try:
        raw = xr.load_dataset(target, engine="scipy")
    except Exception as exc:
        raise RadiationSerializationError(f"failed to load {target}: {exc}") from exc

    attrs = dict(raw.attrs)
    _require(attrs.get("artifact_format") == RADIATION_ARTIFACT_FORMAT,
             RadiationSerializationError(f"{target} is not a {RADIATION_ARTIFACT_FORMAT} artifact"))

    try:
        settings_dict = json.loads(attrs["settings"])
        provenance = json.loads(attrs["provenance"])
        assumptions = json.loads(attrs["assumptions"])
        body_ids = json.loads(attrs["body_ids"])
        scalars = json.loads(attrs["diagnostics_scalars"])
    except (KeyError, json.JSONDecodeError) as exc:
        raise RadiationSerializationError(f"{target} has corrupt metadata: {exc}") from exc

    try:
        settings = RadiationSettings.from_dict(settings_dict)
    except RadiationError as exc:
        raise RadiationSerializationError(
            f"{target} has invalid or tampered settings: {exc}") from exc

    diagnostics = RadiationDiagnostics(
        frequency_grid=np.asarray(raw["frequency"].values, dtype=np.float64),
        zero_frequency_damping=np.asarray(raw["zero_frequency_damping"].values, dtype=np.float64),
        damping_at_omega_max=np.asarray(raw["damping_at_omega_max"].values, dtype=np.float64),
        **scalars,
    )

    kernel = RadiationKernel(
        settings=settings,
        time=np.asarray(raw["time"].values, dtype=np.float64),
        dof_keys=tuple(str(v) for v in raw["influenced_dof"].values.tolist()),
        body_ids=tuple(str(v) for v in body_ids),
        kernel=np.asarray(raw["kernel"].values, dtype=np.float64),
        added_mass_infinite=np.asarray(raw["added_mass_infinite"].values, dtype=np.float64),
        diagnostics=diagnostics,
        provenance=provenance,
        assumptions=tuple(str(a) for a in assumptions),
    )

    expected = str(attrs.get("content_hash", ""))
    _require(expected == kernel.content_hash(),
             RadiationSerializationError(
                 f"{target} failed the content-hash integrity check "
                 f"(stored {expected!r}, computed {kernel.content_hash()!r})"))
    return kernel


def radiation_kernel_hash(kernel: RadiationKernel) -> str:
    """Deterministic content hash of a radiation kernel artifact."""
    return kernel.content_hash()
