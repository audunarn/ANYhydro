"""Backend protocol and the imported-dataset backend.

This slice contains no BEM solver implementation. The imported-dataset
backend consumes validated stored data and reports honest, typed
diagnostics for unsupported capabilities. Optional Capytaine is never
imported here; availability is probed without importing.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import xarray as xr

from .dof import DOF_NAMES  # noqa: F401  (re-exported for backend authors)
from .errors import (
    BodyIdentityMismatchError,
    HydroError,
    RequestMismatchError,
)
from .records import HydroBody, HydroRequest
from .schema import validate_hydro_dataset
from .surfaces import CAPABILITY_STATES, SURFACE_IDENTITY_ATTR, surface_identity

__all__ = [
    "HydroBackendCapabilities",
    "Diagnostic",
    "HydrodynamicsBackend",
    "ImportedHydroDatasetBackend",
]


@dataclass(frozen=True)
class HydroBackendCapabilities:
    """Declared capabilities of a hydrodynamic backend."""

    backend_name: str
    backend_version: str
    supports_multibody: bool
    supports_zero_frequency: bool
    supports_infinite_frequency: bool
    provides_added_mass_infinite: bool
    provides_panel_pressure: bool
    provides_kochin: bool
    provides_free_surface: bool
    provides_mean_drift: bool
    notes: Tuple[str, ...] = ()
    native_thin_sheet: str = 'unsupported'
    mixed_hull_sheet: str = 'unsupported'
    multibody_with_sheets: str = 'unsupported'
    sheet_pressure_jump: str = 'unsupported'

    def __post_init__(self):
        for name in ('native_thin_sheet', 'mixed_hull_sheet', 'multibody_with_sheets', 'sheet_pressure_jump'):
            if getattr(self, name) not in CAPABILITY_STATES:
                raise HydroError(f'{name} must use an explicit capability evidence state')


@dataclass(frozen=True)
class Diagnostic:
    """A typed backend diagnostic."""

    code: str
    severity: str  # "info" | "warning" | "error"
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)


class HydrodynamicsBackend:
    """Protocol-shaped base class: solve / capabilities / diagnose."""

    def solve(
        self,
        bodies: Sequence[HydroBody],
        request: HydroRequest,
    ) -> xr.Dataset:
        raise NotImplementedError

    def capabilities(self) -> HydroBackendCapabilities:
        raise NotImplementedError

    def diagnose(self) -> Tuple[Diagnostic, ...]:
        raise NotImplementedError


def _decode_mesh_hashes(ds: xr.Dataset) -> dict:
    raw = ds.attrs["mesh_hashes"]
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def _decode_body_identity(ds: xr.Dataset) -> dict:
    raw = ds.attrs["body_identity"]
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


class ImportedHydroDatasetBackend(HydrodynamicsBackend):
    """Replays a validated stored hydrodynamic dataset.

    ``solve`` returns the stored data only when the caller's body identities
    (including mesh hashes) and request hash match the stored dataset
    identity; otherwise it raises a typed mismatch error instead of silently
    returning data.
    """

    def __init__(self, dataset: xr.Dataset) -> None:
        validated = validate_hydro_dataset(dataset)
        self._ds = validated.copy(deep=True)

    @property
    def dataset(self) -> xr.Dataset:
        """An isolated deep copy of the stored dataset.

        The backend's internal dataset is never exposed: callers cannot
        corrupt backend state through the returned object.
        """
        return self._ds.copy(deep=True)

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

        stored_bodies = [str(v) for v in self._ds["body"].values.tolist()]
        caller_bodies = [b.body_id for b in bodies]
        if sorted(caller_bodies) != sorted(stored_bodies):
            raise BodyIdentityMismatchError(
                f"body identity mismatch: dataset has {sorted(stored_bodies)}, "
                f"caller passed {sorted(caller_bodies)}"
            )

        mesh_hashes = _decode_mesh_hashes(self._ds)
        body_identity = _decode_body_identity(self._ds)
        for b in bodies:
            stored_hash = mesh_hashes.get(b.body_id)
            if stored_hash != b.mesh.mesh_hash:
                raise BodyIdentityMismatchError(
                    f"mesh hash mismatch for body {b.body_id!r}: dataset has "
                    f"{stored_hash!r}, caller passed {b.mesh.mesh_hash!r}"
                )
            # full body identity: hull and lid mesh hashes, reference frame
            # (rotation center), center of mass and the enabled DOF set
            identity = body_identity[b.body_id]
            stored_lid = identity.get("lid_mesh_hash")
            caller_lid = None if b.lid_mesh is None else b.lid_mesh.mesh_hash
            if stored_lid != caller_lid:
                raise BodyIdentityMismatchError(
                    f"lid mesh hash mismatch for body {b.body_id!r}: dataset "
                    f"has {stored_lid!r}, caller passed {caller_lid!r}"
                )
            for field_name, caller_value in (
                ("rotation_center", b.rotation_center),
                ("center_of_mass", b.center_of_mass),
            ):
                stored_value = tuple(float(v) for v in identity[field_name])
                if stored_value != tuple(float(v) for v in caller_value):
                    raise BodyIdentityMismatchError(
                        f"{field_name} mismatch for body {b.body_id!r}: "
                        f"dataset has {stored_value!r}, caller passed "
                        f"{tuple(float(v) for v in caller_value)!r}"
                    )
            stored_dofs = tuple(str(k) for k in identity["enabled_dofs"])
            if sorted(stored_dofs) != sorted(b.enabled_dofs):
                raise BodyIdentityMismatchError(
                    f"enabled DOF set mismatch for body {b.body_id!r}: "
                    f"dataset has {sorted(stored_dofs)}, caller passed "
                    f"{sorted(b.enabled_dofs)}"
                )

        caller_surfaces = surface_identity(bodies)
        stored_surfaces = self._ds.attrs.get(SURFACE_IDENTITY_ATTR)
        if isinstance(stored_surfaces, str): stored_surfaces = json.loads(stored_surfaces)
        if caller_surfaces != stored_surfaces:
            raise BodyIdentityMismatchError('surface identity mismatch: hull/sheet/lid semantics, topology or ownership changed; stored coefficients cannot omit or relabel surfaces')

        request_hash = request.content_hash()
        if request_hash != self._ds.attrs["request_hash"]:
            raise RequestMismatchError(
                "request mismatch: dataset request_hash "
                f"{self._ds.attrs['request_hash']!r} != caller request hash "
                f"{request_hash!r}"
            )

        return self._ds.copy(deep=True)

    def capabilities(self) -> HydroBackendCapabilities:
        ds = self._ds
        return HydroBackendCapabilities(
            backend_name=str(ds.attrs["backend"]),
            backend_version=str(ds.attrs["backend_version"]),
            supports_multibody=True,
            # The canonical schema forbids omega=0, so no stored dataset can
            # ever serve the zero-frequency limit; this is False regardless
            # of how many positive frequencies are stored.
            supports_zero_frequency=False,
            supports_infinite_frequency="added_mass_infinite" in ds.data_vars,
            provides_added_mass_infinite="added_mass_infinite" in ds.data_vars,
            provides_panel_pressure="panel_pressure" in ds.data_vars,
            sheet_pressure_jump='implemented' if 'pressure_jump' in ds.data_vars else 'unsupported',
            provides_kochin="kochin_radiation" in ds.data_vars
            or "kochin_diffraction" in ds.data_vars,
            provides_free_surface="free_surface_elevation" in ds.data_vars,
            provides_mean_drift="mean_drift_force" in ds.data_vars,
            notes=(
                "replays stored validated data only; no BEM solve capability",
                "supports_zero_frequency is always False: the canonical "
                "schema stores strictly positive finite frequencies only",
                f"qualification_status={ds.attrs['qualification_status']}",
            ),
        )

    def diagnose(self) -> Tuple[Diagnostic, ...]:
        ds = self._ds
        diagnostics = [
            Diagnostic(
                code="imported_dataset",
                severity="info",
                message="backend serves a stored validated dataset; it cannot "
                "solve new requests",
                details={
                    "backend": str(ds.attrs["backend"]),
                    "bodies": [str(v) for v in ds["body"].values.tolist()],
                    "qualification_status": str(ds.attrs["qualification_status"]),
                },
            ),
            Diagnostic(
                code="no_bem_solver",
                severity="warning",
                message="no BEM solver is implemented in this slice; "
                "ImportedHydroDatasetBackend does not provide potential-flow "
                "solves",
                details={},
            ),
        ]
        spec = importlib.util.find_spec("capytaine")
        if spec is None:
            diagnostics.append(
                Diagnostic(
                    code="capytaine_unavailable",
                    severity="info",
                    message="optional dependency capytaine is not installed; "
                    "it is never imported by anyhydro at base import time",
                    details={},
                )
            )
        else:
            diagnostics.append(
                Diagnostic(
                    code="capytaine_available_not_used",
                    severity="info",
                    message="optional dependency capytaine is installed but not "
                    "used by this backend in this slice",
                    details={},
                )
            )
        if ds.attrs["qualification_status"] == "development_fixture":
            diagnostics.append(
                Diagnostic(
                    code="development_fixture",
                    severity="warning",
                    message="stored dataset is a development fixture and confers "
                    "no scientific qualification",
                    details={},
                )
            )
        return tuple(diagnostics)
