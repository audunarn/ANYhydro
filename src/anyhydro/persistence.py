"""Lossless NetCDF persistence with explicit real/imag split.

Format: NetCDF (scipy engine, NETCDF3_64BIT). Complex variables are split
into ``<name>_real`` / ``<name>_imag`` float64 variables and recombined
exactly on load. Nested provenance attributes are JSON-encoded strings. No
backend objects or pickle payloads are stored. Writes are atomic.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .errors import PersistenceError
from .hashing import JSON_DATASET_ATTRS
from .schema import (
    MANDATORY_VARIABLES,
    OPTIONAL_VARIABLES,
    validate_hydro_dataset,
)

_NETCDF_FORMAT = "NETCDF3_64BIT"


def _split_complex_vars(ds: xr.Dataset) -> xr.Dataset:
    out = xr.Dataset(attrs=dict(ds.attrs))
    out = out.assign_coords({name: ds[name] for name in ds.coords})
    for name in ds.data_vars:
        values = ds[name].values
        if np.iscomplexobj(values):
            out[f"{name}_real"] = (ds[name].dims, np.ascontiguousarray(values.real))
            out[f"{name}_imag"] = (ds[name].dims, np.ascontiguousarray(values.imag))
        else:
            out[name] = ds[name].copy(deep=True)
    return out


def _encode_attrs(attrs: dict) -> dict:
    encoded = {}
    for key, value in attrs.items():
        if isinstance(value, (dict, list, tuple)):
            encoded[key] = json.dumps(value, sort_keys=True)
        else:
            encoded[key] = value
    return encoded


def _decode_attrs(attrs: dict) -> dict:
    decoded = {}
    for key, value in attrs.items():
        if isinstance(value, str) and (
            key in JSON_DATASET_ATTRS or value.startswith(("{", "["))
        ):
            try:
                decoded[key] = json.loads(value)
            except json.JSONDecodeError:
                decoded[key] = value
        else:
            decoded[key] = value
    return decoded


def _recombine_complex_vars(ds: xr.Dataset) -> xr.Dataset:
    out = xr.Dataset(attrs=dict(ds.attrs))
    out = out.assign_coords({name: ds[name] for name in ds.coords})
    consumed = set()
    for name in ds.data_vars:
        if name.endswith("_real"):
            base = name[: -len("_real")]
            imag_name = f"{base}_imag"
            if imag_name in ds.data_vars:
                out[base] = (
                    ds[name].dims,
                    ds[name].values + 1j * ds[imag_name].values,
                )
                consumed.update({name, imag_name})
    for name in ds.data_vars:
        if name not in consumed and not name.endswith("_imag"):
            out[name] = ds[name].copy(deep=True)
    return out


def save_hydro_dataset(ds: xr.Dataset, path) -> None:
    """Validate and atomically save a hydrodynamic dataset to NetCDF."""
    validate_hydro_dataset(ds)
    target = Path(path)
    if target.parent and not target.parent.exists():
        raise PersistenceError(f"target directory does not exist: {target.parent}")

    storable = _split_complex_vars(ds)
    storable.attrs = _encode_attrs(dict(ds.attrs))

    fd, tmp_name = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
    )
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        storable.to_netcdf(tmp, engine="scipy", format=_NETCDF_FORMAT)
        os.replace(tmp, target)
    except Exception as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise PersistenceError(f"failed to save dataset to {target}: {exc}") from exc


def load_hydro_dataset(path) -> xr.Dataset:
    """Load, recombine and validate a hydrodynamic dataset from NetCDF."""
    target = Path(path)
    if not target.exists():
        raise PersistenceError(f"file does not exist: {target}")
    try:
        raw = xr.load_dataset(target, engine="scipy")
    except Exception as exc:
        raise PersistenceError(f"failed to load {target}: {exc}") from exc

    raw.attrs = _decode_attrs(dict(raw.attrs))
    ds = _recombine_complex_vars(raw)
    return validate_hydro_dataset(ds)
