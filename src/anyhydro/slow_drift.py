# SPDX-License-Identifier: MPL-2.0
"""Newman difference-frequency QTF from the far-field mean drift (slow drift, slice N1).

Contract: docs/MEAN_DRIFT_CONTRACT.md section 8. Development only; nothing here qualifies slow-drift loads.

Given the mean drift D_k(w) of a dataset (``mean_drift_far_field``: surge, sway, yaw per unit amplitude squared, one
value per heading), Newman's approximation (Newman 1974) builds the difference-frequency QTF

    geometric (default):  T_k(w_i, w_j) = s * sqrt(|D_k(w_i) D_k(w_j)|),  s = sign D_k(w_i) if the signs agree, else 0
    arithmetic:           T_k(w_i, w_j) = (D_k(w_i) + D_k(w_j)) / 2

Both are real, symmetric and reproduce the mean drift exactly on the diagonal. The QTF exists only on the dataset's
frequency grid: values are looked up at grid points, never interpolated (an interpolated first-order basis gave
garbage second-order loads in the NEMOH evaluation, 2026-10-16). Limits: valid for small difference frequencies
relative to the band; surge, sway and yaw only (the far-field drift has no heave/roll/pitch); unidirectional (one
heading per QTF; no cross-heading terms).
"""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np
import xarray as xr

from .errors import HydroError

NEWMAN_QTF_FORMAT = "anyhydro-newman-difference-qtf-v1"
NEWMAN_METHODS = ("geometric", "arithmetic")
NEWMAN_QTF_UNITS = "N/m^2 (surge, sway), N.m/m^2 (yaw) per product of the two component amplitudes"


class SlowDriftError(HydroError):
    """Slow-drift inputs are malformed or not admitted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SlowDriftError(message)


def newman_values(D: np.ndarray, method: str = "geometric") -> np.ndarray:
    """T[i, j, ...] from the diagonal D[i, ...] (any trailing axes), by the declared Newman rule."""
    _require(method in NEWMAN_METHODS, f"unknown Newman method {method!r}; expected one of {NEWMAN_METHODS}")
    raw = np.asarray(D)
    _require(not np.iscomplexobj(raw), "mean drift must be real (a complex array is refused, never truncated)")
    D = raw.astype(np.float64)
    _require(D.ndim >= 1 and D.shape[0] >= 1 and bool(np.all(np.isfinite(D))), "mean drift must be finite")
    Di, Dj = D[:, None, ...], D[None, :, ...]
    if method == "arithmetic":
        return 0.5 * Di + 0.5 * Dj                       # no intermediate overflow; exact on the diagonal
    same = np.sign(Di) == np.sign(Dj)
    root_i, root_j = np.sqrt(np.abs(Di)), np.sqrt(np.abs(Dj))   # no product overflow/underflow
    T = np.where(same, np.sign(Di) * root_i * root_j, 0.0)
    n = D.shape[0]
    T[np.arange(n), np.arange(n)] = D                     # the diagonal is the mean drift exactly
    return T


def newman_qtf(dataset: xr.Dataset, method: str = "geometric") -> xr.Dataset:
    """Newman difference-frequency QTF of a dataset carrying ``mean_drift_far_field``.

    Returns a Dataset with ``newman_qtf(omega_1, omega_2, wave_direction, drift_dof)`` on the dataset's own frequency
    grid (both axes), the source mean-drift record and the declared method in ``attrs['newman_qtf']``.
    """
    _require(isinstance(dataset, xr.Dataset) and "mean_drift_far_field" in dataset.data_vars,
             "dataset lacks 'mean_drift_far_field' (compute the far-field mean drift first)")
    record = dataset.attrs.get("mean_drift_far_field")
    _require(isinstance(record, str), "dataset lacks the mean_drift_far_field record (attrs)")
    source = json.loads(record)
    from .mean_drift import MEAN_DRIFT_FORMAT
    from .schema import validate_mean_drift_block
    from .errors import SchemaError
    _require(isinstance(source, Mapping) and source.get("format") == MEAN_DRIFT_FORMAT,
             f"the mean drift record is not {MEAN_DRIFT_FORMAT!r}")
    try:
        validate_mean_drift_block(dataset)                # array, labels, method, units, motions, yaw reference
    except SchemaError as error:
        raise SlowDriftError(f"the mean-drift block is not admissible: {error}") from error
    drift = dataset["mean_drift_far_field"]
    headings = np.asarray(dataset["wave_direction"].values)
    _require(headings.ndim == 1 and headings.size >= 1 and not np.iscomplexobj(headings)
             and bool(np.all(np.isfinite(headings.astype(float)))), "headings must be a non-empty finite list")
    normalised = np.round(np.mod(headings.astype(float), 2 * np.pi), 12)
    normalised[normalised == round(2 * np.pi, 12)] = 0.0
    _require(len(set(normalised.tolist())) == normalised.size, "equivalent duplicate headings are refused")
    omega = np.asarray(dataset["omega"].values, dtype=np.float64)
    _require(omega.ndim == 1 and omega.size >= 1 and bool(np.all(np.isfinite(omega))) and bool(np.all(omega > 0))
             and bool(np.all(np.diff(omega) > 0)), "omega must be finite, positive and strictly increasing")
    T = newman_values(drift.values, method)
    out = xr.Dataset(
        {"newman_qtf": (("omega_1", "omega_2", "wave_direction", "drift_dof"), T)},
        coords={"omega_1": omega, "omega_2": omega, "wave_direction": dataset["wave_direction"].values,
                "drift_dof": dataset["drift_dof"].values})
    out.attrs["newman_qtf"] = json.dumps(dict(
        format=NEWMAN_QTF_FORMAT, method=method, units=NEWMAN_QTF_UNITS, source_mean_drift=source,
        drift_sha256=_drift_hash(dataset),
        binding=("drift_sha256 binds the QTF to the drift array it was built from; the upstream mean-drift record "
                 "carries no content hash of its own, so a record that does not describe its array cannot be "
                 "detected here"),
        rule=("sign-preserving geometric mean, opposite signs -> 0" if method == "geometric"
              else "arithmetic mean"),
        limits="small difference frequencies; surge/sway/yaw only; unidirectional; grid points only",
        qualification="unqualified"), sort_keys=True)
    return out


def _drift_hash(dataset: xr.Dataset) -> str:
    h = hashlib.sha256()
    for name in ("omega", "wave_direction", "drift_dof"):
        h.update(json.dumps([str(v) if name == "drift_dof" else repr(float(v)) for v in dataset[name].values]).encode())
    h.update(np.ascontiguousarray(dataset["mean_drift_far_field"].values.astype(np.float64)).tobytes())
    return h.hexdigest()


def qtf_at(qtf: xr.Dataset, omega_1: float, omega_2: float) -> np.ndarray:
    """T(omega_1, omega_2) for all headings and DOFs; each frequency must be an exact grid point of its own axis
    (no interpolation)."""
    _require(isinstance(qtf, xr.Dataset) and "newman_qtf" in qtf.data_vars, "not a Newman QTF dataset")
    index = {}
    for axis, w in (("omega_1", omega_1), ("omega_2", omega_2)):
        grid = np.asarray(qtf[axis].values, dtype=np.float64)
        hit = np.flatnonzero(grid == float(w))
        _require(hit.size == 1, f"frequency {w!r} is not a grid point of {axis} (no interpolation)")
        index[axis] = int(hit[0])
    return np.asarray(qtf["newman_qtf"].isel(**index).transpose("wave_direction", "drift_dof").values,
                      dtype=np.float64)
