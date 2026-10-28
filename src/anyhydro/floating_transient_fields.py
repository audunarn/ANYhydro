"""Finite-frequency native-family adapter and pointwise waterline clearance guard (development, UNQUALIFIED).

``build_family_radiation_field_kernel`` turns the retained floating field family (``FloatingFamilySnapshot``
sequence from :mod:`anyhydro.floating_radiation_fields`) plus the admitted infinite-frequency
:class:`~anyhydro.radiation_field_kernel.NativeFieldCoefficients` of the SAME posed family into a field kernel:

* panel-pressure transfer ``H_p(w)[p, j] = panel_pressure_radiation[j, p]`` (per unit displacement, ``p = i w rho phi``
  at the native face centres, exactly what the snapshot retained);
* waterline radiation-elevation transfer ``H_e(w)[i, j] = i w phi_j(x_i) / g`` from the RETAINED sources through the
  owner's recovery routine (``floating_radiation_fields._potentials``: replayed Green function, ``compute_potential``
  only - no solve, no Prony refit) at exactly the native query points;
* the infinite-frequency coefficients of the native object under its selected policy (``P_inf`` and the projection
  unchanged; ``E_inf`` effective, i.e. 0 only under an admitted ``exact_dirichlet_z0`` evaluation, raw values retained
  in the native artifact).

Before anything is built the adapter proves structural coherence from the ACTUAL data (radiation key, provider,
geometry, face labels, DOF order, query points, frequency grid of the supplied generalized ``RadiationKernel``, its
A_inf against the native A_inf and its damping at omega_max against the last snapshot); then the EXISTING unchanged
pressure / elevation reconstruction criteria and the 1e-8 signed-projection check of
:func:`~anyhydro.radiation_field_kernel.build_radiation_field_kernel` run.  The snapshot ids and the native content hash
are bound into the hashed kernel identity.  No coefficient is chosen, extrapolated, symmetrised or refitted.

``guarded_clearance`` evaluates, for every supplied sample, the envelope ``sum_j B_ij |a_j|`` of the pointwise
waterline consistency bound and the guarded margin; it reports explicit non-success when the envelope consumes the
margin.  It does NOT prove that any caller evaluates all integration samples (the ANYfem wiring is separate, held work).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .radiation import MIN_FREQUENCY_SAMPLES, RadiationKernel
from .radiation_field_kernel import (
    DEFAULT_WORK_BYTES,
    N_DOF,
    PROJECTION_REL_TOL,
    FieldReconstructionCriteria,
    FieldResourceLimits,
    NativeFieldCoefficients,
    RadiationFieldError,
    RadiationFieldKernel,
    RadiationFieldLimitError,
    _frozen_any,
    _real_array,
    _require,
    build_radiation_field_kernel,
    native_binding_record,
)

__all__ = [
    "ClearanceConsumedError",
    "ClearanceEnvelope",
    "FamilyAdapterError",
    "build_family_radiation_field_kernel",
    "guarded_clearance",
]


class FamilyAdapterError(RadiationFieldError):
    """The retained family, the native coefficients and the generalized kernel do not coherently match."""


class ClearanceConsumedError(RadiationFieldError):
    """The pointwise boundary envelope consumes the supplied clearance margin (explicit non-success)."""


# ----------------------------------------------------------------------------- clearance guard

@dataclass(frozen=True)
class ClearanceEnvelope:
    """Per-sample, per-point envelope ``sum_j B_ij |a_tj|`` and guarded margin ``margin - envelope``."""

    envelope: np.ndarray            # (n_t, n_q)
    guarded_margin: np.ndarray      # (n_t, n_q)
    consumed: np.ndarray            # (n_t, n_q) bool: guarded margin <= 0
    status: str                     # 'clear' | 'margin_consumed'
    first_consumed: Optional[tuple]  # (sample, point) of the first consumed entry in C order, or None
    n_samples: int

    @property
    def is_clear(self) -> bool:
        return self.status == "clear"

    def require_clear(self) -> "ClearanceEnvelope":
        if not self.is_clear:
            raise ClearanceConsumedError(
                f"waterline consistency envelope consumes the clearance margin at (sample, point) {self.first_consumed}; "
                "explicit non-success")
        return self


def guarded_clearance(pointwise_bound: Any, accelerations: Any, margins: Any, *,
                      max_work_bytes: int = DEFAULT_WORK_BYTES) -> ClearanceEnvelope:
    """Inflate clearance margins by the pointwise per-DOF bound for EVERY supplied sample.

    ``pointwise_bound`` (n_q, 6) nonnegative (e.g. ``NativeFieldCoefficients.pointwise_bound``); ``accelerations``
    (n_t, 6) rigid accelerations at the samples; ``margins`` (n_t, n_q) or (n_q,) clearance margins already computed by
    the caller (any length unit consistent with E_inf * acceleration).  Entries with ``margin - envelope <= 0`` are
    consumed; the status is then ``'margin_consumed'``.  Memory is admitted before allocation and the ceiling can only
    be tightened.  The helper is independent of any output decimation: it evaluates exactly the samples given.
    """
    _require(isinstance(max_work_bytes, (int, np.integer)) and not isinstance(max_work_bytes, bool)
             and 0 < int(max_work_bytes) <= DEFAULT_WORK_BYTES,
             f"max_work_bytes must lie in (0, {DEFAULT_WORK_BYTES}]", RadiationFieldLimitError)
    B = _real_array(pointwise_bound, "pointwise_bound")
    a = _real_array(accelerations, "accelerations")
    m = _real_array(margins, "margins")
    _require(B.ndim == 2 and B.shape[1] == N_DOF and B.shape[0] >= 1, "pointwise_bound must be (n_q >= 1, 6)")
    _require(a.ndim == 2 and a.shape[1] == N_DOF and a.shape[0] >= 1, "accelerations must be (n_t >= 1, 6)")
    n_t, n_q = a.shape[0], B.shape[0]
    _require(m.shape in ((n_t, n_q), (n_q,)), f"margins must be ({n_t}, {n_q}) or ({n_q},), got {m.shape}")
    for name, arr in (("pointwise_bound", B), ("accelerations", a), ("margins", m)):
        _require(bool(np.all(np.isfinite(arr))), f"{name} must be finite")
    _require(bool(np.all(B >= 0.0)), "pointwise_bound must be nonnegative")
    need = 4 * n_t * n_q * 8
    _require(need <= int(max_work_bytes), f"clearance envelope needs {need} B > working-memory limit {int(max_work_bytes)} B",
             RadiationFieldLimitError)
    env = np.abs(a) @ B.T
    guarded = (np.broadcast_to(m, (n_t, n_q)) - env)
    consumed = guarded <= 0.0
    first = None
    if bool(np.any(consumed)):
        k = int(np.argmax(consumed.reshape(-1)))
        first = (k // n_q, k % n_q)
    return ClearanceEnvelope(envelope=_frozen_any(env, np.float64, "envelope"),
                             guarded_margin=_frozen_any(guarded, np.float64, "guarded_margin"),
                             consumed=_frozen_any(consumed, np.bool_, "consumed"),
                             status="margin_consumed" if first is not None else "clear",
                             first_consumed=first, n_samples=n_t)


# ----------------------------------------------------------------------------- native-family adapter

def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise FamilyAdapterError(message)


def _canon(value: Any) -> str:
    import json

    def plain(v):
        if isinstance(v, Mapping):
            return {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v
    return json.dumps(plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


_PROVIDER_KEYS = ("backend", "version", "path", "method", "green_function")


def build_family_radiation_field_kernel(
    snapshots: Sequence[Any],
    native: NativeFieldCoefficients,
    query_points: Any,
    *,
    kernel: RadiationKernel,
    pressure_criteria: FieldReconstructionCriteria,
    elevation_criteria: FieldReconstructionCriteria,
    pressure_zero_frequency: Any = None,
    elevation_zero_frequency: Any = None,
    projection_rel_tol: float = PROJECTION_REL_TOL,
    limits: Optional[FieldResourceLimits] = None,
    chunk_size: int = 64,
    identity: Optional[Mapping[str, Any]] = None,
    evaluator: Optional[Callable[[Any, np.ndarray, int], Any]] = None,
    development: bool = False,
) -> RadiationFieldKernel:
    """Production adapter: actual retained family snapshots + admitted native coefficients -> bound field kernel.

    ``development=True`` (explicit development factory only) retains a FAILED transfer reconstruction in the kernel
    diagnostics (``unpassed_gates``) and marks the hashed identity ``development='unverified_hydrodynamics'``; every
    coherence, projection and resource check is unchanged.

    ``evaluator(snapshot, points, dof_index)`` defaults to the owner's ``floating_radiation_fields._potentials``
    (retained sources, replayed GF, compute_potential only).  Every coherence failure is a typed
    :class:`FamilyAdapterError`; a native object not admitted under its selected policy raises
    ``NativeBoundaryError``; reconstruction / projection failures are those of the unchanged generic builder.
    """
    from .floating_radiation_fields import DOF_NAMES, FloatingFamilySnapshot, _potentials, radiation_key
    from .wave_surface_fields import WaveFieldError

    lim = limits or FieldResourceLimits()
    _fail(isinstance(lim, FieldResourceLimits), "limits must be a FieldResourceLimits")
    _fail(isinstance(native, NativeFieldCoefficients), "native must be a NativeFieldCoefficients")
    native.require_effective_admitted()
    _fail(isinstance(kernel, RadiationKernel), "kernel must be the supplied generalized RadiationKernel")
    snaps = tuple(snapshots)
    _fail(len(snaps) >= MIN_FREQUENCY_SAMPLES, f"at least {MIN_FREQUENCY_SAMPLES} family snapshots are required")
    for s in snaps:
        _fail(isinstance(s, FloatingFamilySnapshot), "every snapshot must be a retained FloatingFamilySnapshot")
        try:
            s.validate()
        except WaveFieldError as exc:
            raise FamilyAdapterError(f"snapshot failed its identity validation: {exc}") from exc
        _fail(s.panel_pressure_radiation is not None, "snapshot lacks retained native panel pressures")

    # ---- one posed family: radiation basis, provider, geometry, labels, DOF order
    first = snaps[0]
    key0 = radiation_key(first.request)
    prov0 = _canon({k: first.provider.get(k) for k in _PROVIDER_KEYS})
    for s in snaps[1:]:
        _fail(radiation_key(s.request) == key0, "snapshots belong to different radiation bases (geometry/axes/environment)")
        _fail(_canon({k: s.provider.get(k) for k in _PROVIDER_KEYS}) == prov0, "snapshots come from different providers")
        _fail(np.array_equal(s.vertices, first.vertices) and np.array_equal(s.faces, first.faces)
              and tuple(s.surface_ids) == tuple(first.surface_ids), "snapshot geometry / face labels differ")
    omegas = np.array([float(s.omega) for s in snaps])
    _fail(bool(np.all(np.diff(omegas) > 0.0)), "snapshots must be in strictly increasing frequency order (no reordering)")
    _fail(native.request is not None and native.vertices is not None and native.faces is not None,
          "native coefficients do not retain their family request / geometry")
    _fail(radiation_key(_plain_request(native.request)) == key0,
          "native coefficients belong to a different radiation basis than the snapshots")
    _fail(np.array_equal(native.vertices, first.vertices) and np.array_equal(native.faces, first.faces)
          and tuple(native.surface_ids) == tuple(str(x) for x in first.surface_ids),
          "native geometry / face labels differ from the snapshots")
    _fail(native.provider is not None and
          _canon({k: native.provider.get(k) for k in _PROVIDER_KEYS}) == prov0,
          "native coefficients come from a different provider than the snapshots")
    _fail(tuple(native.dof_names) == tuple(DOF_NAMES), "native DOF order differs from the family DOF order")
    _fail(len(kernel.dof_keys) == N_DOF and all(str(k).split(".")[-1].lower() == d.lower()
                                                 for k, d in zip(kernel.dof_keys, DOF_NAMES)),
          f"generalized kernel DOF order {kernel.dof_keys} differs from the family order {DOF_NAMES}")
    n_panel = int(first.faces.shape[0])
    _fail(native.pressure_inf.shape[0] == n_panel and all(s.panel_pressure_radiation.shape == (N_DOF, n_panel)
                                                       for s in snaps), "panel counts differ between native and snapshots")
    pts = _real_array(query_points, "query_points")
    _fail(pts.shape == native.query_points.shape and np.array_equal(pts, native.query_points),
          "query points differ from the points of the native coefficients (actual points required)")

    # ---- the supplied generalized kernel must be the operator of the same family
    grid = np.asarray(kernel.diagnostics.frequency_grid)
    _fail(grid.size == omegas.size + 1 and grid[0] == 0.0 and np.array_equal(grid[1:], omegas),
          "the generalized kernel was built on a different frequency grid than the snapshots")
    a_k, a_n = np.asarray(kernel.added_mass_infinite), np.asarray(native.added_mass_infinite)
    _fail(float(np.max(np.abs(a_k - a_n))) <= projection_rel_tol * max(float(np.max(np.abs(a_n))), np.finfo(float).tiny),
          "the generalized kernel A_inf differs from the native A_inf of the same family")
    b_k, b_s = np.asarray(kernel.diagnostics.damping_at_omega_max), np.asarray(snaps[-1].damping)
    _fail(float(np.max(np.abs(b_k - b_s))) <= projection_rel_tol * max(float(np.max(np.abs(b_s))), np.finfo(float).tiny),
          "the generalized kernel damping at omega_max differs from the family snapshot damping")

    # ---- bounded transfer extraction (no solves / refits)
    n_w, n_q = omegas.size, pts.shape[0]
    need = 2 * n_w * (n_panel + n_q) * N_DOF * 16
    _require(need <= lim.max_work_bytes, f"family transfers need {need} B > working-memory limit {lim.max_work_bytes} B",
             RadiationFieldLimitError)
    gravity = float(_plain_request(native.request)["gravity"])
    ev = evaluator or (lambda snap, points, k: _potentials(snap, points, radiation_index=k))
    hp = np.empty((n_w, n_panel, N_DOF), dtype=np.complex128)
    he = np.empty((n_w, n_q, N_DOF), dtype=np.complex128)
    for w_i, s in enumerate(snaps):
        hp[w_i] = np.asarray(s.panel_pressure_radiation, dtype=np.complex128).T
        for k in range(N_DOF):
            phi = np.asarray(ev(s, pts.copy(), k))
            _fail(phi.shape == (n_q,) and phi.dtype.kind in "fc" and bool(np.all(np.isfinite(phi))),
                  f"recovered radiation potential for DOF {k} at omega {s.omega!r} has an invalid shape / value")
            he[w_i, :, k] = 1j * float(s.omega) * phi / gravity

    snapshot_ids = [str(s.snapshot_id) for s in snaps]
    binding = native_binding_record(native, "native_family_adapter", source_snapshot_ids=snapshot_ids,
                                    family_radiation_key=hashlib.sha256(key0.encode("utf-8")).hexdigest())
    return build_radiation_field_kernel(
        kernel=kernel, omegas=omegas, pressure_transfer=hp, elevation_transfer=he,
        pressure_inf=native.pressure_inf, elevation_inf=native.effective_elevation_inf, projection=native.projection,
        panel_ids=native.panel_ids, query_point_ids=native.query_point_ids,
        pressure_zero_frequency=pressure_zero_frequency, elevation_zero_frequency=elevation_zero_frequency,
        pressure_criteria=pressure_criteria, elevation_criteria=elevation_criteria,
        projection_rel_tol=projection_rel_tol, limits=lim, chunk_size=chunk_size, identity=dict(identity or {}),
        _native_binding=binding, _development=bool(development))


def _plain_request(request: Any) -> dict:
    import json
    return json.loads(_canon(request))
