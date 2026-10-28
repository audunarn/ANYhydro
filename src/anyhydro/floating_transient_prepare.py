"""Explicit DEVELOPMENT coarse-source factory for the floating transient workflow (UNVERIFIED hydrodynamics).

``build_development_coarse_source`` turns the retained finite-frequency floating family (``FloatingFamilySnapshot``
sequence, raw unsymmetrised A/B per frequency) plus the admitted native ``omega = inf`` coefficients of the SAME posed
family into the two operators the time-domain run needs:

* the generalized radiation kernel through the canonical transform (:func:`anyhydro.radiation._kernel_from_arrays`:
  same low tail, quadrature, window and diagnostics as every other kernel; raw A/B, the NATIVE A_inf, nothing
  symmetrised, clipped or refitted);
* the panel-pressure / waterline-elevation field kernel through the unchanged family adapter
  (:func:`anyhydro.floating_transient_fields.build_family_radiation_field_kernel`) with ``development=True``.

What differs from the accepted path is ONLY that failed quality reports are RETAINED instead of refusing:

* source damping/added-mass symmetry (``symmetry_tol <= 1e-3``) and damping passivity (``passivity_tol <= 1e-8``)
  violations of the canonical kernel diagnostics;
* failed panel-pressure / waterline-elevation transfer reconstruction.

Every failed report becomes an entry of ``unpassed_gates``; the result is always ``hydrodynamic_verification =
'unverified'`` and carries :data:`DEVELOPMENT_MARKER`.  No report is ever rewritten as passed.  The signed projection
check, coherence of the family, native admission, the 1001-sample kernel cap, the 513-frequency cap, the 600-source cap
and the field resource ceilings (256 MiB work / 2 GiB artifact defaults) refuse exactly as before.  No BEM is run.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .radiation import (
    MAX_INPUT_FREQUENCIES,
    MAX_TIME_SAMPLES,
    RadiationKernel,
    RadiationSettings,
    _kernel_from_arrays,
)
from .radiation_field_kernel import (
    N_DOF,
    FieldReconstructionCriteria,
    FieldResourceLimits,
    NativeFieldCoefficients,
    RadiationFieldError,
    RadiationFieldKernel,
    RadiationFieldLimitError,
    numeric_array_digest,
)

__all__ = [
    "DEVELOPMENT_MARKER",
    "MAX_COARSE_SOURCES",
    "SOURCE_PASSIVITY_TOL",
    "SOURCE_SYMMETRY_TOL",
    "DevelopmentCoarseSource",
    "DevelopmentSourceError",
    "build_development_coarse_source",
]

DEVELOPMENT_MARKER = "development_unverified_coarse_source"
#: canonical source-quality tolerances: a development build may only tighten them
SOURCE_SYMMETRY_TOL = 1.0e-3
SOURCE_PASSIVITY_TOL = 1.0e-8
#: full retained source vector cap of the coarse source (lid sources included)
MAX_COARSE_SOURCES = 600


GEOMETRY_DIGEST_VERSION = "coarse-family-geometry-v1"


def family_geometry_digest(vertices: Any, faces: Any, surface_ids: Any) -> str:
    """Digest of the ACTUAL posed source geometry (world frame of the posed family, exactly as solved): float64
    vertices, int64 faces and the face labels.  Used to prove a supplied source manifest IS the retained family."""
    import hashlib
    import json
    v = np.ascontiguousarray(np.asarray(vertices, dtype="<f8"))
    f = np.ascontiguousarray(np.asarray(faces, dtype="<i8"))
    h = hashlib.sha256(GEOMETRY_DIGEST_VERSION.encode())
    for a in (v, f):
        h.update(json.dumps(list(a.shape)).encode()); h.update(a.tobytes())
    h.update(json.dumps([str(s) for s in surface_ids]).encode())
    return h.hexdigest()


def query_points_digest(points: Any) -> str:
    import hashlib
    q = np.ascontiguousarray(np.asarray(points, dtype="<f8"))
    return hashlib.sha256(b"coarse-query-points-v1" + str(q.shape).encode() + q.tobytes()).hexdigest()


class DevelopmentSourceError(RadiationFieldError):
    """The development coarse source cannot be formed (structure, limits or an unchanged refusing check)."""


def _fail(condition: bool, message: str, error: type = DevelopmentSourceError) -> None:
    if not condition:
        raise error(message)


@dataclass(frozen=True)
class DevelopmentCoarseSource:
    """Raw development operators of one coarse posed family; ALWAYS unverified hydrodynamics."""

    kernel: RadiationKernel
    field_kernel: RadiationFieldKernel
    admission_omegas: np.ndarray              # the actual finite source frequencies
    source_quality: Mapping[str, Any]         # raw canonical diagnostics (violations counted, never clamped)
    unpassed_gates: tuple
    identity: Mapping[str, Any]
    hydrodynamic_verification: str = "unverified"
    marker: str = DEVELOPMENT_MARKER
    notes: tuple = field(default=("development coarse source: failed quality reports retained, not passed",))

    def __post_init__(self) -> None:
        _fail(self.hydrodynamic_verification == "unverified" and self.marker == DEVELOPMENT_MARKER,
              "a development coarse source is always unverified and marked")

    @property
    def added_mass_infinite(self) -> np.ndarray:
        return self.kernel.added_mass_infinite

    def identity_dict(self) -> dict:
        """Plain, fully JSON-serialisable identity (for job inputs / project provenance)."""
        return json_plain(dict(self.identity))

    def quality_dict(self) -> dict:
        return json_plain(dict(self.source_quality))

    def solver_binding(self) -> dict:
        """Fields the ANYfem input binds (digests of the exact arrays the solver will receive)."""
        return {"kernel_digest": numeric_array_digest(self.kernel.kernel),
                "ainf_digest": numeric_array_digest(self.kernel.added_mass_infinite),
                "field_kernel_hash": self.field_kernel.content_hash(),
                "kernel_hash": self.kernel.content_hash(),
                "native_hash": self.identity["native_hash"],
                "hydro_admission": "development_unverified",
                "hydrodynamic_verification": "unverified",
                "unpassed_gates": list(self.unpassed_gates)}


def build_development_coarse_source(
    snapshots: Sequence[Any],
    native: NativeFieldCoefficients,
    query_points: Any,
    *,
    settings: RadiationSettings,
    pressure_criteria: FieldReconstructionCriteria,
    elevation_criteria: FieldReconstructionCriteria,
    body_id: str = "body",
    pressure_zero_frequency: Any = None,
    elevation_zero_frequency: Any = None,
    limits: Optional[FieldResourceLimits] = None,
    chunk_size: int = 64,
    identity: Optional[Mapping[str, Any]] = None,
    evaluator: Optional[Callable[[Any, np.ndarray, int], Any]] = None,
) -> DevelopmentCoarseSource:
    """EXPLICIT development factory (see module docstring).  Never a production or qualified path."""
    from .floating_radiation_fields import DOF_NAMES, FloatingFamilySnapshot
    from .floating_transient_fields import build_family_radiation_field_kernel

    _fail(isinstance(settings, RadiationSettings), "settings must be a RadiationSettings")
    _fail(isinstance(native, NativeFieldCoefficients), "native must be a NativeFieldCoefficients")
    lim = limits or FieldResourceLimits()
    _fail(isinstance(lim, FieldResourceLimits), "limits must be a FieldResourceLimits")
    # ---- caps and tolerances BEFORE any array is copied
    snaps = tuple(snapshots)
    _fail(1 <= len(snaps) <= MAX_INPUT_FREQUENCIES,
          f"{len(snaps)} family frequencies outside [1, {MAX_INPUT_FREQUENCIES}]", RadiationFieldLimitError)
    _fail(int(settings.n_time) <= MAX_TIME_SAMPLES,
          f"kernel n_time {settings.n_time} exceeds the development cap {MAX_TIME_SAMPLES}", RadiationFieldLimitError)
    _fail(0.0 < float(settings.symmetry_tol) <= SOURCE_SYMMETRY_TOL
          and 0.0 < float(settings.passivity_tol) <= SOURCE_PASSIVITY_TOL,
          f"source quality tolerances may only tighten ({SOURCE_SYMMETRY_TOL:g} symmetry, "
          f"{SOURCE_PASSIVITY_TOL:g} passivity)")
    for s in snaps:
        _fail(isinstance(s, FloatingFamilySnapshot), "every snapshot must be a retained FloatingFamilySnapshot")
        n_src = int(s.radiation_sources.shape[-1])
        _fail(n_src <= MAX_COARSE_SOURCES, f"{n_src} retained sources exceed the coarse cap {MAX_COARSE_SOURCES}",
              RadiationFieldLimitError)
    need = 2 * len(snaps) * N_DOF * N_DOF * 8 + int(settings.n_time) * N_DOF * N_DOF * 8 * 3
    _fail(need <= lim.max_work_bytes, f"generalized kernel work {need} B exceeds {lim.max_work_bytes} B",
          RadiationFieldLimitError)
    omegas = np.array([float(s.omega) for s in snaps])
    _fail(bool(np.all(np.diff(omegas) > 0.0)), "snapshots must be in strictly increasing frequency order")

    # ---- canonical generalized kernel: raw A / B of the family, native A_inf (never the last finite sample)
    a_inf = np.array(native.added_mass_infinite, dtype=np.float64, copy=True)
    kernel = _kernel_from_arrays(
        omegas=omegas, added_mass=np.stack([s.added_mass for s in snaps]),
        damping=np.stack([s.damping for s in snaps]), a_inf=a_inf,
        dof_keys=tuple(f"{body_id}.{d.lower()}" for d in DOF_NAMES), body_ids=(str(body_id),), settings=settings,
        provenance={"source": DEVELOPMENT_MARKER, "native_hash": native.content_hash(),
                    "snapshot_ids": [str(s.snapshot_id) for s in snaps],
                    "a_inf_source": "native_symbolic_omega_inf"},
        extra_assumptions=("development coarse source: raw finite-frequency A/B, unverified hydrodynamics",))
    dg = kernel.diagnostics
    # a development source is never admitted, even when every retained report happens to pass
    unpassed = ["development_source_not_admitted"]
    if int(dg.damping_symmetry_violations):
        unpassed.append("source_damping_symmetry")
    if int(dg.added_mass_symmetry_violations):
        unpassed.append("source_added_mass_symmetry")
    if int(dg.passivity_violations):
        unpassed.append("source_damping_passivity")
    quality = json_plain(dict(dg.scalars()))

    # ---- field kernel through the unchanged adapter; reconstruction failures retained (projection still refuses)
    ident = dict(identity or {})
    ident["coarse_source"] = DEVELOPMENT_MARKER
    # hashed into the field-kernel identity so a direct run can verify the guard coordinates / family geometry
    ident["query_points_world"] = np.asarray(native.query_points, dtype=float).tolist()
    ident["family_geometry_digest"] = family_geometry_digest(snaps[0].vertices, snaps[0].faces, snaps[0].surface_ids)
    fk = build_family_radiation_field_kernel(
        snaps, native, query_points, kernel=kernel, pressure_criteria=pressure_criteria,
        elevation_criteria=elevation_criteria, pressure_zero_frequency=pressure_zero_frequency,
        elevation_zero_frequency=elevation_zero_frequency, limits=lim, chunk_size=chunk_size, identity=ident,
        evaluator=evaluator, development=True)
    unpassed += list(fk.diagnostics.get("unpassed_gates", ()))
    quality["field_reconstruction"] = json_plain(fk.diagnostics.get("reconstruction", {}))
    out_ident = {"marker": DEVELOPMENT_MARKER, "native_hash": native.content_hash(),
                 "snapshot_ids": [str(s.snapshot_id) for s in snaps], "kernel_hash": kernel.content_hash(),
                 "field_kernel_hash": fk.content_hash(), "panel_ids": list(fk.panel_ids),
                 "query_point_ids": list(fk.query_point_ids),
                 # actual provenance of the posed family (never a caller copy): the downstream preparation compares
                 # these with the solved operating pose and the guard query points
                 "family_Q0": [list(map(float, r)) for r in snaps[0].request["Q0"]],
                 "family_cg_world": [float(x) for x in snaps[0].request["cg_world"]],
                 "query_points_world": np.asarray(native.query_points, dtype=float).tolist(),
                 "query_points_digest": query_points_digest(native.query_points),
                 "family_geometry_digest": family_geometry_digest(snaps[0].vertices, snaps[0].faces,
                                                                  snaps[0].surface_ids)}
    w = np.array(omegas, dtype=np.float64)
    w.setflags(write=False)
    return DevelopmentCoarseSource(kernel=kernel, field_kernel=fk, admission_omegas=w,
                                   source_quality=MappingProxyType(json_plain(quality)), unpassed_gates=tuple(unpassed),
                                   identity=MappingProxyType(json_plain(out_ident)))


def json_plain(value: Any) -> Any:
    """Fully JSON-serialisable plain copy (mappings, sequences, numpy scalars / arrays; non-finite floats as strings)."""
    if isinstance(value, Mapping):
        return {str(k): json_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_plain(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value
