"""Deterministic content hashing.

All hashes are SHA-256 over explicitly byte-encoded content with little-endian
numeric encoding. Python object addresses, ``id()`` values or in-memory
``hash()`` values are never used.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import xarray as xr

from .dof import DOF_NAMES, body_dof_keys

# Attributes stored as JSON strings in datasets; decoded for hashing.
# ``cache_identity`` is the private on-disk cache stamp: persistence decodes
# it to a dict on load while the stamped file stores a canonical JSON
# string, so hashing must canonicalize both forms identically. Public
# solve() returns never carry it, so public identity is unaffected.
JSON_DATASET_ATTRS = (
    "mesh_hashes",
    "reference_transformations",
    "backend_settings",
    "body_identity",
    "execution_limits",
    "cache_identity",
    "surface_identity",
    "sheet_source_provenance",
    "pressure_jump_contract",
    # optional 1.1.0 extensions: absent on legacy datasets, so legacy hashes
    # are unchanged; present they must hash as canonical JSON (tamper-proof)
    "coefficient_frames",
    "hydrostatic_contract",
    "excitation_source_contract",
    "wamit_import",
    "reciprocity_diagnostic",
)


def _feed(h: "hashlib._Hash", payload: bytes) -> None:
    h.update(len(payload).to_bytes(8, "little"))
    h.update(payload)


def _float_bytes(values) -> bytes:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return arr.astype("<f8").tobytes()


def _int_bytes(values) -> bytes:
    arr = np.ascontiguousarray(np.asarray(values, dtype=np.int64))
    return arr.astype("<i8").tobytes()


def _str_bytes(values: Iterable[str]) -> bytes:
    return "\x1f".join(values).encode("utf-8")


def mesh_hash(
    body_id: str,
    vertices,
    faces,
    face_sizes,
    *, surface=None,
) -> str:
    """Deterministic hash of a panel mesh's content."""
    h = hashlib.sha256()
    _feed(h, b"anyhydro-mesh-v1")
    _feed(h, body_id.encode("utf-8"))
    v = np.ascontiguousarray(np.asarray(vertices, dtype=np.float64))
    _feed(h, b"vertices")
    _feed(h, str(v.shape).encode("ascii"))
    _feed(h, v.astype("<f8").tobytes())
    f = np.ascontiguousarray(np.asarray(faces, dtype=np.int64))
    _feed(h, b"faces")
    _feed(h, str(f.shape).encode("ascii"))
    _feed(h, f.astype("<i8").tobytes())
    s = np.ascontiguousarray(np.asarray(face_sizes, dtype=np.int64))
    _feed(h, b"face_sizes")
    _feed(h, s.astype("<i8").tobytes())
    if surface is not None:
        _feed(h, b'anyhydro-surface-v1')
        _feed(h, json.dumps(surface.to_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8'))
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    """Normalize frozen (mappingproxy/tuple) structures to plain JSON types.

    Hash identity is invariant under freezing: a deep-frozen settings mapping
    and its plain source produce identical JSON.
    """
    if isinstance(value, Mapping):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def request_hash(
    omegas,
    wave_directions,
    water_depth: float,
    rho: float,
    gravity: float,
    include_zero_frequency: bool,
    include_infinite_frequency: bool,
    backend_settings: Mapping[str, Any],
) -> str:
    """Deterministic hash of a hydrodynamic request."""
    h = hashlib.sha256()
    _feed(h, b"anyhydro-request-v1")
    _feed(h, _float_bytes(omegas))
    _feed(h, _float_bytes(wave_directions))
    for name, value in (
        ("water_depth", water_depth),
        ("rho", rho),
        ("gravity", gravity),
    ):
        _feed(h, name.encode("ascii"))
        _feed(h, np.asarray(value, dtype="<f8").tobytes())
    _feed(h, b"include_zero_frequency")
    _feed(h, bytes([int(bool(include_zero_frequency))]))
    _feed(h, b"include_infinite_frequency")
    _feed(h, bytes([int(bool(include_infinite_frequency))]))
    _feed(h, b"backend_settings")
    _feed(
        h,
        json.dumps(
            _jsonable(backend_settings),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
    )
    return h.hexdigest()


def _canonical_attr_value(key: str, value: Any) -> bytes:
    if isinstance(value, (dict, list)):
        return (
            "json::"
            + json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        ).encode("utf-8")
    if key in JSON_DATASET_ATTRS and isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return f"str::{value}".encode("utf-8")
        return (
            "json::"
            + json.dumps(decoded, sort_keys=True, separators=(",", ":"), allow_nan=False)
        ).encode("utf-8")
    return f"{type(value).__name__}::{value}".encode("utf-8")


def _canonical_dof_order(ds: xr.Dataset) -> tuple:
    bodies = sorted(str(b) for b in ds["body"].values.tolist())
    dof_keys: list[str] = []
    for body in bodies:
        dof_keys.extend(body_dof_keys(body))
    return tuple(dof_keys)


def dataset_content_hash(
    ds: xr.Dataset, exclude_attrs: Sequence[str] = ()
) -> str:
    """Stable content hash of a validated hydrodynamic dataset.

    Excludes ``created_utc``. Considers coordinate labels, data, conventions
    and provenance attributes. DOF and body axes are canonicalized first, so
    the hash is invariant under consistent reordering.

    ``exclude_attrs`` names additional attributes to exclude, used for
    noncircular identity: the hydrostatic source binding hashes the dataset
    content with only its own contract/binding payload excluded. The default
    (empty) behavior and every legacy hash are unchanged.
    """
    excluded = set(exclude_attrs)
    dof_keys = _canonical_dof_order(ds)
    canon = ds.sel(body=list(sorted(str(b) for b in ds["body"].values.tolist())))
    canon = canon.sel(influenced_dof=list(dof_keys), radiating_dof=list(dof_keys))

    h = hashlib.sha256()
    _feed(h, b"anyhydro-dataset-v1")

    for key in sorted(ds.attrs):
        if key == "created_utc" or key in excluded:
            continue
        _feed(h, key.encode("utf-8"))
        _feed(h, _canonical_attr_value(key, ds.attrs[key]))

    for name in sorted(ds.coords):
        values = canon[name].values
        _feed(h, f"coord::{name}".encode("utf-8"))
        if np.issubdtype(canon[name].dtype, np.floating):
            _feed(h, _float_bytes(values))
        elif np.issubdtype(canon[name].dtype, np.integer):
            _feed(h, _int_bytes(values))
        else:
            _feed(h, _str_bytes([str(v) for v in values.tolist()]))

    for name in sorted(canon.data_vars):
        da = canon[name]
        _feed(h, f"var::{name}".encode("utf-8"))
        _feed(h, ",".join(da.dims).encode("utf-8"))
        _feed(h, str(da.dtype).encode("ascii"))
        _feed(h, str(da.shape).encode("ascii"))
        arr = np.ascontiguousarray(da.values)
        if np.iscomplexobj(arr):
            _feed(h, arr.astype("<c16").tobytes())
        elif np.issubdtype(arr.dtype, np.floating):
            _feed(h, arr.astype("<f8").tobytes())
        elif np.issubdtype(arr.dtype, np.integer):
            _feed(h, arr.astype("<i8").tobytes())
        else:
            _feed(h, _str_bytes([str(v) for v in arr.ravel().tolist()]))
        for akey in sorted(da.attrs):
            _feed(h, f"attr::{akey}".encode("utf-8"))
            _feed(h, _canonical_attr_value(akey, da.attrs[akey]))

    return h.hexdigest()
