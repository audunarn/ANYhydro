"""Development Capytaine 3.0 adapter (linear potential-flow BEM).

Scope — this is a DEVELOPMENT adapter only:

- Radiation/diffraction via one combined ``cpt.Multibody`` ``fill_dataset``
  call: full 6N coupled matrices, cross-body off-diagonal blocks preserved.
- ``hydrostatic_stiffness`` is Capytaine's linear buoyancy-only hydrostatic
  stiffness computed per sub-body with ``mass=0`` and block-assembled. It is
  NOT the total dry-body restoring: the gravity restoring of the dry body is
  owned separately by the solver. This is stated in the dataset attributes.
- Zero-frequency limit: typed refusal (the canonical schema stores strictly
  positive finite frequencies only; ``supports_zero_frequency`` is False).
- Infinite-frequency limit: explicit per-radiating-DOF ``omega=inf``
  radiation solves assembled into ``added_mass_infinite`` with explicit
  source provenance.
- Optional pressure/field/Kochin/free-surface/mean-drift capabilities are
  disabled pending their own tests.
- Tiny development budgets are enforced (panels, frequencies, headings) so
  no accidental large solve can be launched through this adapter.

Capytaine is imported lazily inside methods only: the base ``anyhydro``
import never touches it. Conventions were verified against the installed
Capytaine 3.0.0 source: incident potential ``exp(+i k (x cos b + y sin b))``
with ``exp(-i w t)`` time factor, i.e. phasor ``exp(-i w t)``; heading
``beta = 0`` propagates along +X, ``beta = pi/2`` propagates along +Y, and
headings are measured counterclockwise from +X toward +Y — identical to the
anyhydro canonical conventions, so no sign flips are applied. Waterplane
lid normals point DOWN, the pinned Capytaine convention
(``FloatingBody.__init__`` forces ``with_normal_vector_going_down`` and
``Mesh.generate_lid`` uses ``normal=(0, 0, -1)``); this adapter requires
source lids to already have downward normals instead of relying on that
silent flip.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
import importlib.metadata
import importlib.util
import json
import numbers
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import xarray as xr

from .backends import HydroBackendCapabilities, HydrodynamicsBackend, Diagnostic
from .dof import DOF_NAMES, body_dof_keys
from .errors import (
    BackendSolveError,
    CapabilityError,
    GeometryValidationError,
    HydroError,
    UnsupportedOperationError,
)
from .persistence import load_hydro_dataset, save_hydro_dataset
from .reciprocity import reciprocity_diagnostic, warn_if_exceeded
from .green_function import (FAR_FIELD_DAMPING_FORMAT, FAR_FIELD_DAMPING_GREEN_IDENTITY_FORMAT,
                             IMAGINARY_PART_FORMAT, anyhydro_green_function, far_field_damping,
                             far_field_theta_count, green_identity_kochin)
from .prony_fit import PRONY_FIT_FORMAT
from .records import HydroBody, HydroRequest
from .schema import SCHEMA_VERSION, assemble_hydro_dataset

__all__ = [
    "CapytaineBackend",
    "CapytaineExecutionLimits",
    "CAPYTAINE_ADAPTER_VERSION",
    "SUPPORTED_SETTINGS",
]

CAPYTAINE_ADAPTER_VERSION = "0.6.0"   # 0.6.0: method "direct" is the default (direct-default programme D7)


def _warn_reciprocity(dataset):
    """Warn (from ``solve``, fresh or cached) when the stored reciprocity diagnostic exceeds its level.

    A fresh dataset holds the JSON text; a cache load decodes JSON attributes to dicts (persistence).
    """
    stored = dataset.attrs.get("reciprocity_diagnostic")
    if stored is not None:
        warn_if_exceeded(json.loads(stored) if isinstance(stored, str) else stored, stacklevel=4)


#: Largest heading count for the far-field damping (k R bound); refused above.
MAX_FAR_FIELD_THETA = 16384

#: The only backend setting keys this adapter understands; unknown keys are
#: refused instead of silently ignored.
SUPPORTED_SETTINGS = ("method", "cache_dir", "auxiliary_lid_policy")
#: BEM formulation used when the request names none.  "direct" since adapter 0.6.0: on the same panel
#: mesh it agrees with OrcaWave to <= 0.3 % (A/B) where "indirect" deviated 5-11 %; evidence and the
#: qualified scope in docs/CAPYTAINE_ADAPTER.md.  "indirect" remains available explicitly.
DEFAULT_METHOD = "direct"

# Tiny development budgets (this adapter is development-scoped). These are
# the DEFAULT execution limits; explicit CapytaineExecutionLimits may admit
# more total panels or finite frequencies (up to the first-slice ceilings
# below) for later registered cases, never executed by default.
MAX_TOTAL_PANELS = 100
MAX_FINITE_OMEGAS = 3
MAX_WAVE_DIRECTIONS = 1

# First-slice preparation ceilings for EXPLICIT limits: the absolute
# software panel ceiling is 6144 (raised 2026-10-12 from 2048 for the W5
# extended-band case, levels 10/15 = 2400/5400 panels; dense S/K plus LU
# stay near 2.5 GB); the heading ceiling stays at the development default;
# the finite-frequency ceiling admits registered dense cases. Raising a
# limit never confers run authority: cases above the 6144-panel capacity
# are refused honestly, with no autounbounded profile.
_CEILING_TOTAL_PANELS = 6144
_CEILING_FINITE_OMEGAS = 513
_CEILING_WAVE_DIRECTIONS = 1
_CEILING_JOBS = 1
_CEILING_THREADS = 1

_LIMIT_FIELDS = (
    "max_total_panels",
    "max_finite_omegas",
    "max_wave_directions",
    "n_jobs",
    "n_threads",
)
_LIMIT_CEILINGS = {
    "max_total_panels": _CEILING_TOTAL_PANELS,
    "max_finite_omegas": _CEILING_FINITE_OMEGAS,
    "max_wave_directions": _CEILING_WAVE_DIRECTIONS,
    "n_jobs": _CEILING_JOBS,
    "n_threads": _CEILING_THREADS,
}


def _validate_limit(name: str, value) -> int:
    """Typed admission of one execution-limit control.

    Rejects bools, fractional/non-integral values, non-finite values,
    negatives and zero before any backend import or cache lookup.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise CapabilityError(
            f"execution limit {name!r} must be an integer (got "
            f"{type(value).__name__}: {value!r}); bools, fractional and "
            "non-finite controls are refused"
        )
    ivalue = int(value)
    if ivalue < 1:
        raise CapabilityError(
            f"execution limit {name!r} must be positive (got {ivalue})"
        )
    ceiling = _LIMIT_CEILINGS[name]
    if ivalue > ceiling:
        raise CapabilityError(
            f"execution limit {name!r} exceeds its first-slice ceiling "
            f"{ceiling} (got {ivalue})"
        )
    return ivalue


@dataclass(frozen=True, init=False)
class CapytaineExecutionLimits:
    """Frozen public execution limits for the Capytaine development adapter.

    Defaults preserve the tiny development budgets: 100 total panels,
    3 finite frequencies, 1 heading, one job and one thread. Explicit
    limits may admit up to 6144 total panels (the absolute software
    ceiling; the registered 384/864-panel source-study meshes need
    explicit limits and are still refused by the 100-panel DEFAULT) and
    up to 513 finite frequencies for later registered cases; the
    heading ceiling stays at 1 in this first-slice preparation, and
    ``n_jobs``/``n_threads`` are pinned to 1 (no silent thread or
    environment overrides). Every control is validated at construction —
    bool, fractional, non-finite, negative, zero and unknown controls
    raise ``CapabilityError`` before any Capytaine import or cache
    lookup. A profile name confers no qualification: these are execution
    bounds only.
    """

    max_total_panels: int
    max_finite_omegas: int
    max_wave_directions: int
    n_jobs: int
    n_threads: int

    def __init__(
        self,
        *,
        max_total_panels: int = MAX_TOTAL_PANELS,
        max_finite_omegas: int = MAX_FINITE_OMEGAS,
        max_wave_directions: int = MAX_WAVE_DIRECTIONS,
        n_jobs: int = 1,
        n_threads: int = 1,
        **unknown: Any,
    ) -> None:
        if unknown:
            raise CapabilityError(
                f"unknown execution-limit control(s) {sorted(unknown)}; "
                f"this adapter supports only {list(_LIMIT_FIELDS)} and "
                "refuses silent unknowns"
            )
        for name, default in (
            ("max_total_panels", max_total_panels),
            ("max_finite_omegas", max_finite_omegas),
            ("max_wave_directions", max_wave_directions),
            ("n_jobs", n_jobs),
            ("n_threads", n_threads),
        ):
            object.__setattr__(self, name, _validate_limit(name, default))

    def as_dict(self) -> dict:
        """The effective limits as a plain JSON-serializable dict."""
        return {name: int(getattr(self, name)) for name in _LIMIT_FIELDS}

    def describe(self) -> str:
        """Compact human-readable description for provenance attrs."""
        return (
            f"max_total_panels={self.max_total_panels}, "
            f"max_finite_omegas={self.max_finite_omegas}, "
            f"max_wave_directions={self.max_wave_directions}, "
            f"n_jobs={self.n_jobs}, n_threads={self.n_threads}"
        )

# Capytaine's own free-surface/bottom admission tolerance.
_GEOMETRY_TOLERANCE = 1e-8

# anyhydro DOF name -> Capytaine rigid-body DOF name (verified capitalized).
_CPT_DOF_NAME = {
    "surge": "Surge",
    "sway": "Sway",
    "heave": "Heave",
    "roll": "Roll",
    "pitch": "Pitch",
    "yaw": "Yaw",
}

ADAPTER_NOTE = (
    "anyhydro capytaine development adapter: linear potential-flow BEM via "
    "capytaine; development evidence only, no scientific qualification"
)
HYDROSTATIC_NOTE = (
    "hydrostatic_stiffness is linear buoyancy-only (capytaine "
    "compute_hydrostatic_stiffness with mass=0, block-assembled per body); "
    "the gravity restoring of the dry body is NOT included and is owned "
    "separately by the solver; no claim of total dry-body restoring is made"
)


def _cpt_dof_label(body_id: str, dof: str) -> str:
    """The Multibody-renamed Capytaine DOF label: ``<body_id>__<Dof>``."""
    return f"{body_id}__{_CPT_DOF_NAME[dof]}"


def _face_lists(faces: np.ndarray, face_sizes: np.ndarray) -> list:
    """Convert a -1-padded index array to Capytaine's per-face index lists.

    The source record arrays are only read, never mutated.
    """
    return [
        [int(v) for v in faces[i, : int(face_sizes[i])]]
        for i in range(faces.shape[0])
    ]


def _face_centers(vertices: np.ndarray, faces: np.ndarray, face_sizes: np.ndarray):
    """Per-face centroid coordinates (read-only view of the record arrays)."""
    n = faces.shape[0]
    centers = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        idx = faces[i, : int(face_sizes[i])]
        centers[i] = vertices[idx].mean(axis=0)
    return centers


def _face_normals_and_areas(
    vertices: np.ndarray, faces: np.ndarray, face_sizes: np.ndarray
):
    """Per-face cross-product normals (unnormalized) and areas.

    Read-only view of the record arrays; used for geometric orientation
    checks at admission (the adapter never rewrites source geometry).
    """
    n = faces.shape[0]
    normals = np.zeros((n, 3), dtype=np.float64)
    areas = np.zeros(n, dtype=np.float64)
    for i in range(n):
        idx = faces[i, : int(face_sizes[i])]
        pts = vertices[idx]
        cross = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        normals[i] = cross
        areas[i] = 0.5 * float(np.linalg.norm(cross))
    return normals, areas


def _check_settings(settings: Mapping[str, Any]) -> dict:
    unknown = sorted(set(settings) - set(SUPPORTED_SETTINGS))
    if unknown:
        raise CapabilityError(
            f"unknown backend setting(s) {unknown}; this adapter supports "
            f"only {sorted(SUPPORTED_SETTINGS)} and refuses silent unknowns"
        )
    out = {}
    for key in SUPPORTED_SETTINGS:
        if key in settings:
            value = settings[key]
            if not isinstance(value, str):
                raise CapabilityError(f"backend setting {key!r} must be a string")
            out[key] = value
    if "auxiliary_lid_policy" in out and out["auxiliary_lid_policy"] != "submerged_0p1m_v1":
        raise CapabilityError("unsupported auxiliary_lid_policy; only explicit submerged_0p1m_v1 is supported")
    if "method" in out:
        # Capytaine silently runs the indirect method for any string other
        # than exactly "direct"; refuse instead of recording a method that
        # was never used.
        if out["method"] not in ("direct", "indirect"):
            raise CapabilityError(
                f"unsupported method {out['method']!r}; this adapter supports "
                "only 'direct' (the default) and 'indirect'")
        # An explicit 'indirect' is the default solve, but it stays in the
        # settings: the request (and so its hash, cache key and provenance)
        # differs from an unset method, and sharing one cache entry would
        # make the two request forms overwrite each other.
    return out


def _capytaine_version() -> str:
    try:
        return importlib.metadata.version("capytaine")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _capytaine_spec():
    """Probe capytaine availability without importing it."""
    return importlib.util.find_spec("capytaine")


def _require_capytaine():
    if _capytaine_spec() is None:
        raise CapabilityError(
            "optional dependency capytaine is not installed; it is never "
            "imported by anyhydro at base import time (install the "
            "'capytaine' optional extra to use this backend)"
        )
    import capytaine  # lazy: only inside solve/diagnose

    return capytaine


class CapytaineBackend(HydrodynamicsBackend):
    """Development BEM backend backed by Capytaine 3.0.

    No Capytaine object outlives a ``solve`` call: nothing backend-side is
    persisted, and the NetCDF cache stores validated canonical datasets
    only (never pickled backend objects).
    """

    def __init__(
        self,
        *,
        cache_dir: Optional[str] = None,
        execution_limits: Optional[CapytaineExecutionLimits] = None,
    ) -> None:
        if execution_limits is None:
            execution_limits = CapytaineExecutionLimits()
        if not isinstance(execution_limits, CapytaineExecutionLimits):
            raise CapabilityError(
                "execution_limits must be a CapytaineExecutionLimits "
                f"instance (got {type(execution_limits).__name__}); "
                "limits are validated at construction, before any "
                "capytaine import or cache lookup"
            )
        self._limits = execution_limits
        self._cache_dir = cache_dir

    # ------------------------------------------------------------------ #
    # capabilities / diagnostics (never import capytaine here)
    # ------------------------------------------------------------------ #

    def capabilities(self) -> HydroBackendCapabilities:
        return HydroBackendCapabilities(
            backend_name="capytaine",
            backend_version=_capytaine_version(),
            supports_multibody=True,
            supports_zero_frequency=False,
            supports_infinite_frequency=True,
            provides_added_mass_infinite=True,
            provides_panel_pressure=False,
            provides_kochin=False,
            provides_free_surface=False,
            provides_mean_drift=False,
            notes=(
                ADAPTER_NOTE,
                HYDROSTATIC_NOTE,
                "supports_zero_frequency is always False: the canonical "
                "schema stores strictly positive finite frequencies only",
                "added_mass_infinite comes from explicit capytaine omega=inf "
                "radiation limit solves with recorded provenance",
                "pressure/field/Kochin/free-surface/mean-drift capabilities "
                "are disabled pending their own tests",
                "execution limits are explicit, frozen and recorded in "
                f"diagnostics, cache identity and dataset provenance: "
                f"{self._limits.describe()}",
                "a limits profile name confers no qualification: these are "
                "execution bounds only",
            ),
        )

    def diagnose(self) -> tuple:
        diagnostics = [
            Diagnostic(
                code="capytaine_adapter",
                severity="info",
                message=ADAPTER_NOTE,
                details={"adapter_version": CAPYTAINE_ADAPTER_VERSION},
            ),
            Diagnostic(
                code="hydrostatics_scope",
                severity="warning",
                message=HYDROSTATIC_NOTE,
                details={},
            ),
            Diagnostic(
                code="development_budgets",
                severity="warning",
                message=(
                    "explicit execution limits are enforced (panels, "
                    "frequencies, headings, jobs/threads) before cache "
                    "lookup; this adapter is not qualified for production "
                    "solves"
                ),
                details={
                    "effective_limits": self._limits.as_dict(),
                    "default_limits": CapytaineExecutionLimits().as_dict(),
                },
            ),
        ]
        if _capytaine_spec() is None:
            diagnostics.append(
                Diagnostic(
                    code="capytaine_unavailable",
                    severity="error",
                    message=(
                        "optional dependency capytaine is not installed; "
                        "solve will raise a typed CapabilityError"
                    ),
                    details={},
                )
            )
        else:
            diagnostics.append(
                Diagnostic(
                    code="capytaine_available",
                    severity="info",
                    message=(
                        f"capytaine {_capytaine_version()} is installed and "
                        "imported lazily inside solve only"
                    ),
                    details={"version": _capytaine_version()},
                )
            )
        return tuple(diagnostics)

    # ------------------------------------------------------------------ #
    # cache identity
    # ------------------------------------------------------------------ #

    def _cache_identity(
        self,
        bodies: Sequence[HydroBody],
        request: HydroRequest,
        settings: Mapping[str, Any],
        cpt_version: str,
    ) -> dict:
        """Full solve identity: geometry, references, constants, settings.

        Includes the actual reference data (rotation centers, centers of
        mass, enabled DOFs, hull and lid mesh hashes) and the physical
        constants, so a cached entry can never be reused for changed input.
        """
        return {
            "adapter": "anyhydro-capytaine",
            "adapter_version": CAPYTAINE_ADAPTER_VERSION,
            "schema_version": SCHEMA_VERSION,
            "capytaine_version": cpt_version,
            "finite_depth_prony_decomposition": PRONY_FIT_FORMAT,
            "green_function_imaginary_part": IMAGINARY_PART_FORMAT,
            "far_field_damping": (FAR_FIELD_DAMPING_FORMAT
                                  if settings.get("method", DEFAULT_METHOD) == "indirect"
                                  else FAR_FIELD_DAMPING_GREEN_IDENTITY_FORMAT),
            "settings": dict(settings),
            "execution_limits": self._limits.as_dict(),
            "request": {
                "omegas": [float(v) for v in request.omegas],
                "wave_directions": [float(v) for v in request.wave_directions],
                # deep water (+inf) is encoded as a sentinel: JSON has no inf
                "water_depth": (
                    float(request.water_depth)
                    if np.isfinite(request.water_depth)
                    else "inf"
                ),
                "rho": float(request.rho),
                "gravity": float(request.gravity),
                "include_zero_frequency": bool(request.include_zero_frequency),
                "include_infinite_frequency": bool(
                    request.include_infinite_frequency
                ),
            },
            "bodies": {
                b.body_id: {
                    "mesh_hash": b.mesh.mesh_hash,
                    "lid_mesh_hash": None
                    if b.lid_mesh is None
                    else b.lid_mesh.mesh_hash,
                    "rotation_center": [float(v) for v in b.rotation_center],
                    "center_of_mass": [float(v) for v in b.center_of_mass],
                    "enabled_dofs": list(b.enabled_dofs),
                }
                for b in bodies
            },
        }

    @staticmethod
    def _cache_key(identity: dict) -> str:
        payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_load(self, path, identity: dict, request: HydroRequest):
        """Validated cache load; any mismatch or corruption is a miss."""
        try:
            ds = load_hydro_dataset(path)
        except (HydroError, OSError, ValueError):
            return None
        raw = ds.attrs.get("cache_identity")
        try:
            stored = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError:
            return None
        if stored != identity:
            return None
        if ds.attrs.get("request_hash") != request.content_hash():
            return None
        return ds

    # ------------------------------------------------------------------ #
    # geometry admission
    # ------------------------------------------------------------------ #

    def _admit_geometry(
        self, bodies: Sequence[HydroBody], water_depth: float, *, lid_policy="free_surface_v1"
    ) -> None:
        from .first_slice import first_slice_lid_elevation
        elevation = first_slice_lid_elevation(lid_policy)
        if lid_policy == "submerged_0p1m_v1":
            from math import isqrt
            from .first_slice import make_first_slice_bodies
            n = isqrt(len(bodies[0].mesh.faces) // 10) if bodies else 0
            expected = make_first_slice_bodies(subdivisions=n, lid_policy=lid_policy)
            if (len(bodies) != 2 or any(
                actual.body_id != target.body_id or actual.mesh.mesh_hash != target.mesh.mesh_hash or
                actual.lid_mesh is None or actual.lid_mesh.mesh_hash != target.lid_mesh.mesh_hash
                for actual, target in zip(bodies, expected))):
                raise GeometryValidationError("submerged_0p1m_v1 requires exact fixed first-slice hull/lid geometry")
        total_panels = 0
        for b in bodies:
            if lid_policy == "submerged_0p1m_v1":
                lo = b.mesh.vertices.min(axis=0); hi = b.mesh.vertices.max(axis=0)
                ref = np.asarray(b.rotation_center)
                if (b.lid_mesh is None or
                    not np.allclose(lo, ref + [-.5, -.5, -.5], atol=1e-12, rtol=0) or
                    not np.allclose(hi, ref + [.5, .5, 0.], atol=1e-12, rtol=0) or ref[2] != 0.):
                    raise GeometryValidationError("submerged_0p1m_v1 requires the fixed 1x1m, 0.5m draft boxes at z=0")
                if np.isfinite(water_depth) and elevation <= -water_depth:
                    raise GeometryValidationError("auxiliary lid must be above the sea bottom")
            if not b.mesh.outward_normals_verified:
                raise GeometryValidationError(
                    f"body {b.body_id!r}: outward normals are not verified; "
                    "this adapter refuses unverified geometry instead of "
                    "silently flipping normal vectors"
                )
            for mesh, kind in ((b.mesh, "hull"), (b.lid_mesh, "lid")):
                if mesh is None:
                    continue
                total_panels += int(mesh.face_sizes.shape[0])
                centers = _face_centers(mesh.vertices, mesh.faces, mesh.face_sizes)
                z = centers[:, 2]
                if kind == "hull":
                    above = z >= 0.0 + _GEOMETRY_TOLERANCE
                    if np.any(above):
                        raise GeometryValidationError(
                            f"body {b.body_id!r}: {int(np.sum(above))} hull "
                            "panel(s) are above the free surface; the BEM "
                            "geometry must already be equilibrium-wetted "
                            "(this adapter never clips geometry)"
                        )
                    if np.isfinite(water_depth):
                        below = z <= -water_depth
                        if np.any(below):
                            raise GeometryValidationError(
                                f"body {b.body_id!r}: {int(np.sum(below))} "
                                "hull panel(s) are below the sea bottom"
                            )
                else:
                    if np.any(np.abs(mesh.vertices[:, 2] - elevation) > _GEOMETRY_TOLERANCE):
                        raise GeometryValidationError(
                            f"body {b.body_id!r}: lid panels must lie in the "
                            f"declared auxiliary plane z={elevation} m"
                        )
                    # The pinned Capytaine lid convention is downward
                    # normals; Capytaine would silently flip wrong lids
                    # (FloatingBody forces with_normal_vector_going_down).
                    # This adapter refuses them instead, so source panel
                    # identity and orientation are never silently changed.
                    normals, areas = _face_normals_and_areas(
                        mesh.vertices, mesh.faces, mesh.face_sizes
                    )
                    degenerate = areas <= _GEOMETRY_TOLERANCE
                    if np.any(degenerate):
                        raise GeometryValidationError(
                            f"body {b.body_id!r}: {int(np.sum(degenerate))} "
                            "lid panel(s) are degenerate (zero area)"
                        )
                    unit_z = normals[:, 2] / (2.0 * areas)
                    upward = unit_z >= -_GEOMETRY_TOLERANCE
                    if np.any(upward):
                        raise GeometryValidationError(
                            f"body {b.body_id!r}: {int(np.sum(upward))} lid "
                            "panel(s) do not point DOWN; the pinned capytaine "
                            "lid convention requires downward normals and "
                            "this adapter never flips source geometry"
                        )
        if total_panels > self._limits.max_total_panels:
            raise CapabilityError(
                f"development adapter panel budget exceeded: {total_panels} "
                f"panels > {self._limits.max_total_panels} allowed"
            )

    # ------------------------------------------------------------------ #
    # solve
    # ------------------------------------------------------------------ #

    def solve(
        self,
        bodies: Sequence[HydroBody],
        request: HydroRequest,
    ) -> xr.Dataset:
        for b in bodies:
            if not isinstance(b, HydroBody):
                raise HydroError(f"body {b!r} is not a HydroBody")
        if not isinstance(request, HydroRequest):
            raise HydroError("request must be a HydroRequest")
        if len(bodies) < 1:
            raise HydroError("at least one body is required")

        for body in bodies:
            if body.attached_surfaces:
                raise UnsupportedOperationError(
                    'Capytaine adapter native_thin_sheet=unsupported: attached sheets require a genuine two-sided formulation; '
                    'use a verified coefficient import with exact surface provenance or explicitly model finite thickness. '
                    'No sheet is omitted, duplicated or given numerical thickness.')
            if body.mesh.surface is not None and body.mesh.surface.primary_hydrodynamics != 'potential_flow':
                raise UnsupportedOperationError('Capytaine requires potential_flow ownership for hull boundaries; Morison/none ownership cannot also be solved by BEM')

        settings = _check_settings(request.backend_settings)

        if request.include_zero_frequency:
            raise UnsupportedOperationError(
                "the zero-frequency limit is not supported: the canonical "
                "schema stores strictly positive finite frequencies only "
                "(supports_zero_frequency is False)"
            )

        n_omega = int(request.omegas.shape[0])
        if n_omega > self._limits.max_finite_omegas:
            raise CapabilityError(
                f"development adapter frequency budget exceeded: {n_omega} "
                f"finite frequencies > {self._limits.max_finite_omegas} "
                "allowed"
            )
        n_dir = int(request.wave_directions.shape[0])
        if n_dir > self._limits.max_wave_directions:
            raise CapabilityError(
                f"development adapter heading budget exceeded: {n_dir} "
                f"headings > {self._limits.max_wave_directions} allowed"
            )

        if settings.get("auxiliary_lid_policy") == "submerged_0p1m_v1" and (
            request.gravity != 9.81 or request.rho != 1025. or request.water_depth != 150.):
            raise CapabilityError("submerged_0p1m_v1 requires fixed g=9.81, rho=1025 and depth=150m")
        if settings.get("auxiliary_lid_policy") == "submerged_0p1m_v1" and max(request.omegas) > 8.:
            raise CapabilityError("submerged_0p1m_v1 admits only bands up to 8 rad/s")
        self._admit_geometry(bodies, request.water_depth,
                             lid_policy=settings.get("auxiliary_lid_policy", "free_surface_v1"))

        cache_dir = settings.get("cache_dir", self._cache_dir)
        identity = None
        cache_path = None
        if cache_dir:
            from pathlib import Path

            cpt_version = _capytaine_version()
            identity = self._cache_identity(bodies, request, settings, cpt_version)
            cache_path = Path(cache_dir) / f"{self._cache_key(identity)}.nc"
            if cache_path.exists():
                cached = self._cache_load(cache_path, identity, request)
                if cached is not None:
                    # The cache stamp is private validation bookkeeping
                    # kept on disk only; it must never leak into the
                    # public dataset, so fresh and warm returns of the
                    # same physical request hash identically.
                    cached.attrs.pop("cache_identity", None)
                    _warn_reciprocity(cached)
                    return cached

        dataset = self._solve_fresh(bodies, request, settings)
        # Version-scoped audit: do not infer conventions for an untested backend.
        source_inventory_supported = all(
            tuple(b.enabled_dofs) == tuple(f"{b.body_id}::{d}" for d in DOF_NAMES)
            for b in bodies
        )
        if dataset.attrs.get("backend_version") == "3.0.0" and source_inventory_supported:
            from .excitation_source import attach_excitation_source
            dataset = attach_excitation_source(
                dataset, request, bodies,
                convention_evidence="Capytaine 3.0.0 bem/airy_waves.py and theory; "
                    "zero-speed unit elevation at global origin; "
                    "ANYhydro EXCITATION_SOURCE_ADMISSION_CONTRACT revision 1",
            )

        if cache_path is not None:
            stamped = dataset.copy(deep=True)
            stamped.attrs["cache_identity"] = json.dumps(
                identity, sort_keys=True, allow_nan=False
            )
            try:
                save_hydro_dataset(stamped, cache_path)
            except HydroError:
                # a failed cache write never invalidates a correct solve
                pass
        _warn_reciprocity(dataset)
        return dataset

    def _solve_fresh(
        self,
        bodies: Sequence[HydroBody],
        request: HydroRequest,
        settings: Mapping[str, Any],
    ) -> xr.Dataset:
        cpt = _require_capytaine()

        # -- build Capytaine bodies without mutating the source records ----
        cpt_bodies = []
        for b in bodies:
            try:
                mesh = cpt.Mesh(
                    vertices=np.array(b.mesh.vertices, dtype=np.float64),
                    faces=_face_lists(b.mesh.faces, b.mesh.face_sizes),
                )
                lid_mesh = None
                if b.lid_mesh is not None:
                    lid_mesh = cpt.Mesh(
                        vertices=np.array(b.lid_mesh.vertices, dtype=np.float64),
                        faces=_face_lists(b.lid_mesh.faces, b.lid_mesh.face_sizes),
                    )
                body = cpt.FloatingBody(
                    mesh,
                    dofs=cpt.rigid_body_dofs(
                        rotation_center=tuple(b.rotation_center)
                    ),
                    lid_mesh=lid_mesh,
                    center_of_mass=tuple(b.center_of_mass),
                    mass=0.0,
                    name=b.body_id,
                )
            except GeometryValidationError:
                raise
            except Exception as exc:  # capytaine rejected the geometry
                raise GeometryValidationError(
                    f"capytaine rejected the geometry of body "
                    f"{b.body_id!r}: {exc}"
                ) from exc
            cpt_bodies.append(body)

        # mass=0 is deliberate (buoyancy-only hydrostatics; no dry mass
        # enters the dataset): capytaine's Multibody then computes a NaN
        # combined center of gravity from 0/0, which the BEM solve never
        # uses. The expected division warning is silenced; the NaN itself
        # never reaches the assembled dataset.
        with np.errstate(invalid="ignore", divide="ignore"):
            combined = (
                cpt_bodies[0]
                if len(cpt_bodies) == 1
                else cpt.Multibody(cpt_bodies, name="anyhydro-combined")
            )

        # -- canonical anyhydro DOF axis and the matching cpt labels -------
        our_keys = []
        cpt_labels = []
        for b in bodies:
            for dof in DOF_NAMES:
                our_keys.append(f"{b.body_id}::{dof}")
                # A single body is solved directly (no Multibody), so its Capytaine
                # DOF labels carry no ``<body_id>__`` prefix.
                cpt_labels.append(_CPT_DOF_NAME[dof] if len(bodies) == 1 else _cpt_dof_label(b.body_id, dof))
        n_dof = len(our_keys)

        # -- one combined solve over the full grid ---------------------------
        # Equivalent to BEMSolver.fill_dataset (problems_from_dataset +
        # solve_all + assemble_dataset). The details of the same solves give
        # the radiation Kochin functions and from them the far-field
        # (energy-flux) damping, positive semidefinite by construction
        # (anyhydro.green_function): from the source strengths with the
        # indirect method, from Green's identity on the hull potential with
        # the direct method (which has no source strengths).
        method = settings.get("method", DEFAULT_METHOD)
        far_field = True
        far_field_format = (FAR_FIELD_DAMPING_FORMAT if method == "indirect"
                            else FAR_FIELD_DAMPING_GREEN_IDENTITY_FORMAT)
        n_theta = 0
        if far_field:
            all_vertices = np.vstack(
                [np.asarray(b.mesh.vertices, dtype=np.float64) for b in bodies]
                + [np.asarray(b.lid_mesh.vertices, dtype=np.float64)
                   for b in bodies if b.lid_mesh is not None])
            # H_i^* H_j does not depend on the Kochin reference point, so the
            # bandwidth is set by the radius about the horizontal bounding centre
            centre = 0.5 * (all_vertices[:, :2].min(axis=0) + all_vertices[:, :2].max(axis=0))
            radius = float(np.max(np.hypot(all_vertices[:, 0] - centre[0],
                                           all_vertices[:, 1] - centre[1])))
            n_theta = far_field_theta_count(
                request.omegas, float(request.water_depth), float(request.gravity), radius)
            if n_theta > MAX_FAR_FIELD_THETA:
                raise CapabilityError(
                    f"far-field damping needs {n_theta} headings (k R too large) > "
                    f"{MAX_FAR_FIELD_THETA}; this adapter refuses rather than undersample")
        matrix = xr.Dataset(
            coords={
                "omega": np.asarray(request.omegas, dtype=np.float64),
                "wave_direction": np.asarray(
                    request.wave_directions, dtype=np.float64
                ),
                "radiating_dof": np.array(cpt_labels, dtype=object),
                "water_depth": float(request.water_depth),
                "rho": float(request.rho),
                "g": float(request.gravity),
            }
        )
        # Capytaine Delhommeau with the validated finite-depth Prony
        # decomposition and the exact propagating imaginary part (the
        # tabulated imaginary part gave indefinite damping); see
        # anyhydro.green_function and docs/FINITE_DEPTH_PRONY.md.
        # The solver owns the method, so the finite-frequency solves, the
        # omega=inf solves below and the recorded settings all use the same one.
        solver = cpt.BEMSolver(green_function=anyhydro_green_function(), method=method)
        from capytaine.io.xarray import assemble_dataset, problems_from_dataset
        problems = problems_from_dataset(matrix, combined)
        results = solver.solve_all(
            problems,
            keep_details=far_field,
            n_jobs=self._limits.n_jobs,
            n_threads=self._limits.n_threads,
            progress_bar=False,
        )
        attrs = {"start_of_computation": datetime.now().isoformat(),
                 **solver.exportable_settings}
        if attrs.get("method") != method:
            raise BackendSolveError(
                f"capytaine recorded method {attrs.get('method')!r}, requested {method!r}")
        result = assemble_dataset(results, attrs=attrs, hydrostatics=False)

        def _finite(name: str, values: np.ndarray) -> np.ndarray:
            if np.iscomplexobj(values):
                ok = np.all(np.isfinite(values.real)) and np.all(
                    np.isfinite(values.imag)
                )
            else:
                ok = bool(np.all(np.isfinite(values)))
            if not ok:
                raise BackendSolveError(
                    f"capytaine returned invalid (non-finite) {name} data; "
                    "this adapter never invents replacement coefficients"
                )
            return values

        added_mass = _finite(
            "added_mass",
            result["added_mass"]
            .sel(influenced_dof=cpt_labels, radiating_dof=cpt_labels)
            .transpose("omega", "influenced_dof", "radiating_dof")
            .values,
        )
        damping = _finite(
            "radiation_damping",
            result["radiation_damping"]
            .sel(influenced_dof=cpt_labels, radiating_dof=cpt_labels)
            .transpose("omega", "influenced_dof", "radiating_dof")
            .values,
        )
        excitation = _finite(
            "excitation_force",
            result["excitation_force"]
            .sel(influenced_dof=cpt_labels)
            .transpose("omega", "wave_direction", "influenced_dof")
            .values.astype(np.complex128),
        )
        damping_far = None
        if far_field:
            from capytaine.post_pro.kochin import compute_kochin
            theta = np.arange(n_theta, dtype=np.float64) * (2.0 * np.pi / n_theta)
            omegas_req = np.asarray(request.omegas, dtype=np.float64)
            kochin = np.full((omegas_req.size, n_dof, n_theta), np.nan, dtype=np.complex128)
            label_index = {label: i for i, label in enumerate(cpt_labels)}
            for res in results:
                dof = getattr(res.problem, "radiating_dof", None)
                if dof is None:
                    continue          # diffraction: no far-field damping needed
                o = int(np.flatnonzero(omegas_req == float(res.problem.omega))[0])
                if method == "indirect":
                    kochin[o, label_index[dof]] = compute_kochin(res, theta)
                else:
                    # Green's identity on the physical hull; auxiliary lid panels excluded
                    hull = np.asarray(res.problem.body.hull_mask)
                    mesh_all = res.problem.body.mesh_including_lid
                    kochin[o, label_index[dof]] = green_identity_kochin(
                        mesh_all.faces_centers[hull], mesh_all.faces_normals[hull],
                        mesh_all.faces_areas[hull], np.asarray(res.problem.boundary_condition)[hull],
                        np.asarray(res.potential)[hull], theta,
                        k=float(res.wavenumber), water_depth=float(request.water_depth))
            kochin = _finite("kochin_radiation", kochin)
            wavenumbers = np.asarray(result["wavenumber"].values, dtype=np.float64)
            damping_far = _finite(
                "radiation_damping_far_field",
                far_field_damping(kochin, np.asarray(request.omegas, dtype=np.float64),
                                  wavenumbers, float(request.water_depth), float(request.rho)),
            )

        # -- buoyancy-only hydrostatics per sub-body, block-assembled -------
        stiffness = np.zeros((n_dof, n_dof), dtype=np.float64)
        centers_of_buoyancy = np.zeros((len(bodies), 3), dtype=np.float64)
        cpt_dof_order = [_CPT_DOF_NAME[d] for d in DOF_NAMES]
        for k, (b, cpt_body) in enumerate(zip(bodies, cpt_bodies)):
            hs = cpt_body.compute_hydrostatic_stiffness(
                rho=float(request.rho), g=float(request.gravity)
            )
            hs = hs.sel(
                influenced_dof=cpt_dof_order, radiating_dof=cpt_dof_order
            )
            block = _finite(
                f"hydrostatic_stiffness({b.body_id})", hs.values
            )
            stiffness[6 * k : 6 * k + 6, 6 * k : 6 * k + 6] = block
            cob = np.asarray(cpt_body.center_of_buoyancy, dtype=np.float64)
            if not np.all(np.isfinite(cob)):
                raise BackendSolveError(
                    f"capytaine returned an invalid center of buoyancy for "
                    f"body {b.body_id!r}"
                )
            centers_of_buoyancy[k] = cob

        # -- optional explicit infinite-frequency limit --------------------
        added_mass_infinite = None
        a_inf_source = None
        if request.include_infinite_frequency:
            added_mass_infinite = np.zeros((n_dof, n_dof), dtype=np.float64)
            for j, label in enumerate(cpt_labels):
                problem = cpt.RadiationProblem(
                    body=combined,
                    omega=np.inf,
                    radiating_dof=label,
                    wave_direction=0.0,
                    water_depth=float(request.water_depth),
                    rho=float(request.rho),
                    g=float(request.gravity),
                )
                solved = solver.solve(problem)
                row = solved.added_mass
                for i, influenced in enumerate(cpt_labels):
                    if influenced not in row:
                        raise BackendSolveError(
                            f"capytaine omega=inf solve for radiating DOF "
                            f"{label!r} is missing influenced DOF "
                            f"{influenced!r}"
                        )
                    value = float(row[influenced])
                    if not np.isfinite(value):
                        raise BackendSolveError(
                            "capytaine returned invalid (non-finite) "
                            "added_mass_infinite data"
                        )
                    added_mass_infinite[i, j] = value
            a_inf_source = (
                f"capytaine {_capytaine_version()} explicit omega=inf "
                "symbolic radiation limit solves (one per radiating DOF)"
            )

        rotation_center = np.array(
            [b.rotation_center for b in bodies], dtype=np.float64
        )
        center_of_mass = np.array(
            [b.center_of_mass for b in bodies], dtype=np.float64
        )

        reciprocity = reciprocity_diagnostic(added_mass, damping, request.omegas, our_keys)

        return assemble_hydro_dataset(
            bodies,
            request,
            added_mass=added_mass,
            radiation_damping=damping,
            excitation_force=excitation,
            hydrostatic_stiffness=stiffness,
            rotation_center=rotation_center,
            center_of_mass=center_of_mass,
            center_of_buoyancy=centers_of_buoyancy,
            added_mass_infinite=added_mass_infinite,
            added_mass_infinite_source=a_inf_source,
            radiation_damping_far_field=damping_far,
            radiation_damping_far_field_source=(
                None if damping_far is None else
                f"{FAR_FIELD_DAMPING_FORMAT}: (4 pi rho k/(omega D)) Re int H_i^* H_j dtheta "
                f"from capytaine {_capytaine_version()} Kochin functions of the same "
                f"radiation solves, {n_theta} uniform theta samples"
                if method == "indirect" else
                f"{far_field_format}: (4 pi rho k/(omega D)) Re int H_i^* H_j dtheta "
                f"from Green's-identity Kochin functions (hull potential and normal "
                f"velocity of the same capytaine {_capytaine_version()} direct radiation "
                f"solves, centroid rule, auxiliary lid excluded), {n_theta} uniform theta samples"),
            backend="capytaine",
            backend_version=_capytaine_version(),
            backend_commit="unknown",
            qualification_status="unqualified",
            extra_attrs={
                "adapter": "anyhydro-capytaine-development-adapter",
                "adapter_version": CAPYTAINE_ADAPTER_VERSION,
                "method": method,          # resolved BEM formulation: 'indirect' (default) or 'direct'
                "hydrostatic_stiffness_definition": HYDROSTATIC_NOTE,
                "development_limits": self._limits.describe(),
                "execution_limits": self._limits.as_dict(),
                "reciprocity_diagnostic": reciprocity,
            },
        )
