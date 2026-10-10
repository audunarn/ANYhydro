"""Immutable hydrodynamic records: mesh, body, request.

All records are frozen dataclasses. Arrays are copied on construction and
marked read-only, so callers cannot mutate a record through a shared
reference. Duplicate or incomplete identities are rejected at construction.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .dof import DOF_NAMES, body_dof_keys, parse_dof_key
from .errors import RecordValidationError
from .hashing import mesh_hash as _mesh_hash
from .hashing import request_hash as _request_hash

from .surfaces import SurfaceSemantics, sheet_geometry

JSONValue = Union[None, bool, int, float, str, list, dict]


def _deep_freeze(value):
    """Recursively freeze JSON-like settings: mappings and sequences."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _frozen_float_array(value, name: str, ndim: int) -> np.ndarray:
    arr = np.array(value, dtype=np.float64, copy=True, order="C")
    if arr.ndim != ndim:
        raise RecordValidationError(
            f"{name} must have {ndim} dimension(s), got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise RecordValidationError(f"{name} contains non-finite values")
    arr.setflags(write=False)
    return arr


def _check_body_id(body_id: str) -> str:
    if not isinstance(body_id, str) or not body_id.strip():
        raise RecordValidationError("body_id must be a non-empty string")
    if "::" in body_id:
        raise RecordValidationError(
            f"body_id {body_id!r} must not contain the DOF separator '::'"
        )
    return body_id


@dataclass(frozen=True)
class HydroPanelMesh:
    """Immutable panel mesh with deterministic content hash."""

    vertices: np.ndarray
    faces: np.ndarray
    face_sizes: np.ndarray
    body_id: str
    outward_normals_verified: bool
    source_geometry_refs: Tuple[str, ...]
    mesh_hash: str
    surface: Optional[SurfaceSemantics] = None

    def __post_init__(self) -> None:
        body_id = _check_body_id(self.body_id)
        object.__setattr__(self, "body_id", body_id)

        if not isinstance(self.outward_normals_verified, bool):
            raise RecordValidationError("outward_normals_verified must be bool")

        vertices = _frozen_float_array(self.vertices, "vertices", 2)
        if vertices.shape[1] != 3:
            raise RecordValidationError(
                f"vertices must be (n, 3), got shape {vertices.shape}"
            )
        if vertices.shape[0] < 1:
            raise RecordValidationError("vertices must contain at least one vertex")
        object.__setattr__(self, "vertices", vertices)

        if self.surface is not None and isinstance(self.surface, SurfaceSemantics) and self.surface.surface_kind == 'thin_sheet':
            for name, value in (('faces', self.faces), ('face_sizes', self.face_sizes)):
                raw = np.asarray(value)
                if raw.dtype.kind not in 'iu' or raw.dtype.kind == 'b':
                    raise RecordValidationError(f'thin_sheet {name} requires exact integer topology')
                if name == 'faces' and raw.size and (int(raw.min()) < -1 or int(raw.max()) >= len(vertices) or int(raw.max()) > np.iinfo(np.int32).max):
                    raise RecordValidationError('thin_sheet raw face index is outside the vertex inventory; integer narrowing is refused')
                if name == 'face_sizes' and (np.any(raw < 3) or np.any(raw > 255)):
                    raise RecordValidationError('thin_sheet face_sizes must be between 3 and 255')
        sizes = np.array(self.face_sizes, dtype=np.uint8, copy=True, order="C")
        if sizes.ndim != 1 or sizes.shape[0] < 1:
            raise RecordValidationError(
                "face_sizes must be a non-empty 1-D array of per-face vertex counts"
            )
        if np.any(sizes < 3):
            raise RecordValidationError("each face needs at least 3 vertices")
        sizes.setflags(write=False)
        object.__setattr__(self, "face_sizes", sizes)

        faces = np.array(self.faces, dtype=np.int32, copy=True, order="C")
        if faces.ndim != 2:
            raise RecordValidationError(
                f"faces must be a 2-D padded index array, got shape {faces.shape}"
            )
        if faces.shape[0] != sizes.shape[0]:
            raise RecordValidationError(
                f"faces has {faces.shape[0]} rows but face_sizes has {sizes.shape[0]}"
            )
        expected_width = int(np.max(sizes))
        if faces.shape[1] != expected_width:
            raise RecordValidationError(
                f"faces width {faces.shape[1]} does not match max(face_sizes) {expected_width}"
            )
        n_vertices = vertices.shape[0]
        for i, size in enumerate(sizes.tolist()):
            row = faces[i]
            used = row[:size]
            if np.any(used < 0) or np.any(used >= n_vertices):
                raise RecordValidationError(
                    f"face {i} has vertex indices outside [0, {n_vertices})"
                )
            if size < faces.shape[1] and np.any(row[size:] != -1):
                raise RecordValidationError(
                    f"face {i} padding entries must be -1"
                )
        faces.setflags(write=False)
        object.__setattr__(self, "faces", faces)

        refs = tuple(str(r) for r in self.source_geometry_refs)
        object.__setattr__(self, "source_geometry_refs", refs)

        if self.surface is not None:
            if not isinstance(self.surface, SurfaceSemantics):
                raise RecordValidationError('surface must be explicit SurfaceSemantics')
            if self.surface.owning_body_id != body_id:
                raise RecordValidationError('surface owning_body_id must equal mesh body_id')
            if self.surface.source_geometry_id not in refs:
                raise RecordValidationError('surface source_geometry_id must be retained in source_geometry_refs')
            if self.surface.surface_kind == 'thin_sheet':
                if self.outward_normals_verified:
                    raise RecordValidationError('thin_sheet uses oriented minus-to-plus winding, not closed-body outward normals')
                sheet_geometry(self)

        computed = _mesh_hash(body_id, vertices, faces, sizes, surface=self.surface)
        provided = self.mesh_hash
        if provided is None or provided == "":
            object.__setattr__(self, "mesh_hash", computed)
        elif provided != computed:
            raise RecordValidationError(
                f"mesh_hash mismatch for body {body_id!r}: provided {provided!r}, "
                f"computed {computed!r}"
            )


@dataclass(frozen=True)
class HydroBody:
    """Immutable hydrodynamic body with body-qualified enabled DOFs."""

    body_id: str
    name: str
    mesh: HydroPanelMesh
    lid_mesh: Optional[HydroPanelMesh]
    rotation_center: Tuple[float, float, float]
    center_of_mass: Tuple[float, float, float]
    enabled_dofs: Tuple[str, ...]
    attached_surfaces: Tuple[HydroPanelMesh, ...] = ()

    def __post_init__(self) -> None:
        body_id = _check_body_id(self.body_id)
        object.__setattr__(self, "body_id", body_id)

        if not isinstance(self.name, str):
            raise RecordValidationError("name must be a string")

        if not isinstance(self.mesh, HydroPanelMesh):
            raise RecordValidationError("mesh must be a HydroPanelMesh")
        if self.mesh.body_id != body_id:
            raise RecordValidationError(
                f"mesh body_id {self.mesh.body_id!r} does not match body {body_id!r}"
            )

        if self.lid_mesh is not None:
            if not isinstance(self.lid_mesh, HydroPanelMesh):
                raise RecordValidationError("lid_mesh must be a HydroPanelMesh or None")
            if self.lid_mesh.body_id != body_id:
                raise RecordValidationError(
                    f"lid_mesh body_id {self.lid_mesh.body_id!r} does not match "
                    f"body {body_id!r}"
                )

        if self.mesh.surface is not None and self.mesh.surface.surface_kind != 'body_boundary':
            raise RecordValidationError('body mesh is a body_boundary; attach thin sheets explicitly')
        if self.lid_mesh is not None and self.lid_mesh.surface is not None and self.lid_mesh.surface.surface_kind != 'numerical_lid':
            raise RecordValidationError('lid_mesh must be an explicit numerical_lid')
        attached = tuple(self.attached_surfaces)
        ids = [m.surface.surface_id for m in (self.mesh, self.lid_mesh) if m is not None and m.surface is not None]
        for mesh in attached:
            if not isinstance(mesh, HydroPanelMesh) or mesh.body_id != body_id or mesh.surface is None or mesh.surface.surface_kind != 'thin_sheet':
                raise RecordValidationError('attached_surfaces require explicitly owned thin_sheet panel meshes')
            ids.append(mesh.surface.surface_id)
        if len(ids) != len(set(ids)):
            raise RecordValidationError('duplicate surface_id in body')
        object.__setattr__(self, 'attached_surfaces', attached)

        for field_name in ("rotation_center", "center_of_mass"):
            value = getattr(self, field_name)
            arr = np.asarray(value, dtype=np.float64)
            if arr.shape != (3,) or not np.all(np.isfinite(arr)):
                raise RecordValidationError(
                    f"{field_name} must be a finite 3-tuple of floats"
                )
            object.__setattr__(self, field_name, tuple(float(v) for v in arr))

        dofs = tuple(self.enabled_dofs)
        if not dofs:
            raise RecordValidationError("enabled_dofs must not be empty")
        seen = set()
        for key in dofs:
            try:
                key_body, dof = parse_dof_key(key)
            except ValueError as exc:
                raise RecordValidationError(f"invalid enabled DOF key: {exc}") from exc
            if key_body != body_id:
                raise RecordValidationError(
                    f"enabled DOF {key!r} does not belong to body {body_id!r}"
                )
            if key in seen:
                raise RecordValidationError(f"duplicate enabled DOF {key!r}")
            seen.add(key)
        object.__setattr__(self, "enabled_dofs", dofs)

    def all_dof_keys(self) -> Tuple[str, ...]:
        """The full six body-qualified DOF keys of this body."""
        return body_dof_keys(self.body_id)


@dataclass(frozen=True)
class HydroRequest:
    """Immutable hydrodynamic solve request.

    ``omegas`` holds finite positive strictly increasing frequencies only. The
    zero and infinite frequency limits are separate boolean flags and never
    appear inside ``omegas``.
    """

    omegas: np.ndarray
    wave_directions: np.ndarray
    water_depth: float
    rho: float
    gravity: float
    include_zero_frequency: bool
    include_infinite_frequency: bool
    backend_settings: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        omegas = _frozen_float_array(self.omegas, "omegas", 1)
        if omegas.shape[0] < 1:
            raise RecordValidationError("omegas must not be empty")
        if np.any(omegas <= 0.0):
            raise RecordValidationError(
                "omegas must be strictly positive; the zero-frequency limit is "
                "requested via include_zero_frequency"
            )
        if omegas.shape[0] > 1 and not np.all(np.diff(omegas) > 0.0):
            raise RecordValidationError("omegas must be strictly increasing")
        object.__setattr__(self, "omegas", omegas)

        directions = _frozen_float_array(self.wave_directions, "wave_directions", 1)
        if directions.shape[0] < 1:
            raise RecordValidationError("wave_directions must not be empty")
        object.__setattr__(self, "wave_directions", directions)

        try:
            water_depth = float(self.water_depth)
        except (TypeError, ValueError) as exc:
            raise RecordValidationError("water_depth must be a float") from exc
        # +inf is the canonical deep-water depth; every finite value must be
        # positive. -inf and NaN are rejected.
        if np.isnan(water_depth) or water_depth == -np.inf or (
            np.isfinite(water_depth) and not water_depth > 0.0
        ):
            raise RecordValidationError(
                f"water_depth must be positive finite or +inf for deep water "
                f"(got {water_depth!r})"
            )
        object.__setattr__(self, "water_depth", water_depth)

        for name, value in (("rho", self.rho), ("gravity", self.gravity)):
            try:
                scalar = float(value)
            except (TypeError, ValueError) as exc:
                raise RecordValidationError(f"{name} must be a float") from exc
            if not np.isfinite(scalar) or not scalar > 0.0:
                raise RecordValidationError(
                    f"{name} must be finite and > 0 (got {scalar!r})"
                )
            object.__setattr__(self, name, scalar)

        for name in ("include_zero_frequency", "include_infinite_frequency"):
            if not isinstance(getattr(self, name), bool):
                raise RecordValidationError(f"{name} must be bool")

        settings = copy.deepcopy(dict(self.backend_settings))
        try:
            json.dumps(settings, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RecordValidationError(
                f"backend_settings must be JSON-serializable: {exc}"
            ) from exc
        # Deep-frozen read-only view: nested mappings and sequences are
        # frozen too, so callers cannot mutate the record's identity at any
        # depth.
        object.__setattr__(self, "backend_settings", _deep_freeze(settings))

    def content_hash(self) -> str:
        """Deterministic request hash used as dataset identity."""
        return _request_hash(
            self.omegas,
            self.wave_directions,
            self.water_depth,
            self.rho,
            self.gravity,
            self.include_zero_frequency,
            self.include_infinite_frequency,
            self.backend_settings,
        )
