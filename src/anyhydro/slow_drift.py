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
import math
import re
from dataclasses import dataclass
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
        T = 0.5 * Di + 0.5 * Dj                           # no intermediate overflow
    else:
        same = np.sign(Di) == np.sign(Dj)
        root_i, root_j = np.sqrt(np.abs(Di)), np.sqrt(np.abs(Dj))   # no product overflow/underflow
        T = np.where(same, np.sign(Di) * root_i * root_j, 0.0)
    n = D.shape[0]
    T[np.arange(n), np.arange(n)] = D                     # the diagonal is the mean drift exactly
    return T


def _validated(dataset: xr.Dataset):
    """(source record, drift DataArray, omega grid) of an admissible mean-drift dataset; raises SlowDriftError."""
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
    return source, drift, omega


def newman_qtf(dataset: xr.Dataset, method: str = "geometric") -> xr.Dataset:
    """Newman difference-frequency QTF of a dataset carrying ``mean_drift_far_field``.

    Returns a Dataset with ``newman_qtf(omega_1, omega_2, wave_direction, drift_dof)`` on the dataset's own frequency
    grid (both axes), the source mean-drift record and the declared method in ``attrs['newman_qtf']``.
    """
    source, drift, omega = _validated(dataset)
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


# ----------------------------------------------------------------------------------------------- N2: time series

DRIFT_SERIES_VERSION = "anyhydro-newman-drift-series-v1"


@dataclass(frozen=True, eq=False)
class DriftSeries:
    """Newman slow-drift force F2(t) = sum_i sum_j a_i a_j T(w_i, w_j) cos((w_i - w_j) t - (phi_i - phi_j)).

    Components follow ANYloads' elevation a cos(k.x - w t + phi) at the global origin, i.e. a cos(w t - phi), the
    convention of the first-order ExcitationSeries. ``drift`` holds D(w_i) per component and DOF (surge, sway, yaw),
    linearly interpolated from the dataset's mean drift inside its stored range (never extrapolated). Evaluation is
    exact without the O(N^2) pair sum: geometric = sum over sign groups s |sum a_i sqrt|D_i| e^{i theta_i}|^2,
    arithmetic = Re[(sum a_i D_i e^{i theta_i}) conj(sum a_j e^{i theta_j})], theta_i = w_i t - phi_i.
    """
    dof_keys: tuple
    omegas: np.ndarray
    amplitudes: np.ndarray
    phases: np.ndarray
    drift: np.ndarray
    heading_radians: float
    method: str
    wave_realization_hash: str
    hydro_content_hash: str
    drift_sha256: str

    def __post_init__(self):
        keys = tuple(str(k) for k in self.dof_keys)
        _require(len(keys) == 3 and len(set(keys)) == 3, "three distinct drift DOF keys (surge, sway, yaw) required")
        w = _real_array(self.omegas, "omega", 1)
        a = _real_array(self.amplitudes, "amplitude", 1)
        ph = _real_array(self.phases, "phase", 1)
        d = _real_array(self.drift, "drift", 2)
        _require(bool(np.all(w > 0)) and bool(np.all(a >= 0)) and a.shape == w.shape == ph.shape
                 and d.shape == (len(w), 3), "positive frequencies, nonnegative amplitudes and matching shapes required")
        _require(self.method in NEWMAN_METHODS, f"unknown Newman method {self.method!r}")
        _require(isinstance(self.heading_radians, float) and np.isfinite(self.heading_radians), "finite heading required")
        for key in ("wave_realization_hash", "hydro_content_hash", "drift_sha256"):
            value = getattr(self, key)
            _require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None,
                     f"{key} must be a SHA-256 hex digest")
        for key, value in (("dof_keys", keys), ("omegas", w), ("amplitudes", a), ("phases", ph), ("drift", d)):
            object.__setattr__(self, key, value)

    def evaluate(self, times) -> np.ndarray:
        """F2 at the given times, shape (len(times), 3)."""
        t = _real_array(times, "times", 1)
        out = np.empty((len(t), 3))
        for start in range(0, len(t), 2048):
            theta = t[start:start + 2048, None] * self.omegas[None, :] - self.phases[None, :]
            e = np.exp(1j * theta)                                              # (time, component)
            for k in range(3):
                D = self.drift[:, k]
                if self.method == "arithmetic":
                    out[start:start + 2048, k] = (e @ (self.amplitudes * D)
                                                  * np.conj(e @ self.amplitudes)).real
                else:
                    total = np.zeros(e.shape[0])
                    for sign in (1.0, -1.0):
                        group = np.sign(D) == sign
                        if np.any(group):
                            s = e[:, group] @ (self.amplitudes[group] * np.sqrt(np.abs(D[group])))
                            total += sign * np.abs(s) ** 2
                    out[start:start + 2048, k] = total
        _require(bool(np.all(np.isfinite(out))), "nonfinite synthesized drift force")
        return out

    def mean(self) -> np.ndarray:
        """Time-mean slow-drift force: sum_i a_i^2 D(w_i) (the diagonal), shape (3,)."""
        return (self.amplitudes[:, None] ** 2 * self.drift).sum(axis=0)

    def to_dict(self) -> dict:
        return dict(api_version=DRIFT_SERIES_VERSION, dof_keys=list(self.dof_keys), omegas=self.omegas.tolist(),
                    amplitudes=self.amplitudes.tolist(), phases=self.phases.tolist(), drift=self.drift.tolist(),
                    heading_radians=self.heading_radians, method=self.method,
                    wave_realization_hash=self.wave_realization_hash, hydro_content_hash=self.hydro_content_hash,
                    drift_sha256=self.drift_sha256)

    @classmethod
    def from_dict(cls, data) -> "DriftSeries":
        fields = {"api_version", "dof_keys", "omegas", "amplitudes", "phases", "drift", "heading_radians", "method",
                  "wave_realization_hash", "hydro_content_hash", "drift_sha256"}
        _require(type(data) is dict and set(data) == fields and data["api_version"] == DRIFT_SERIES_VERSION,
                 "drift series must match its exact named schema")
        return cls(**{k: v for k, v in data.items() if k != "api_version"})

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"),
                                         allow_nan=False).encode("utf-8")).hexdigest()


def _real_array(value, name, ndim) -> np.ndarray:
    raw = np.asarray(value)
    _require(not np.iscomplexobj(raw) and raw.dtype.kind in "iuf" and raw.ndim == ndim and raw.size > 0,
             f"{name} requires real data of rank {ndim}")
    out = raw.astype(np.float64)
    _require(bool(np.all(np.isfinite(out))), f"{name} must be finite")
    return np.frombuffer(out.tobytes(), dtype=np.float64).reshape(out.shape)


def build_drift_series(dataset: xr.Dataset, realization, method: str = "geometric") -> DriftSeries:
    """Newman slow-drift series of a unidirectional wave realization (ANYloads WaveRealization).

    All components must share one heading that equals a dataset heading (modulo 2 pi, 1e-12 rad, as the first-order
    excitation series); mixed headings are refused (Newman's approximation has no cross-heading terms). D(w) is
    interpolated linearly inside the dataset's frequency range per DOF; a component outside it is refused.
    """
    try:
        from anyloads import WaveRealization
    except ImportError as error:
        raise SlowDriftError("the ANYloads sea-state extra is required for drift synthesis") from error
    _require(type(realization) is WaveRealization, "an exact named WaveRealization is required")
    _require(method in NEWMAN_METHODS, f"unknown Newman method {method!r}; expected one of {NEWMAN_METHODS}")
    source, drift, omega = _validated(dataset)
    from .hashing import dataset_content_hash
    components = realization.components
    directions = {c.direction_degrees for c in components}
    _require(len(directions) == 1, "a unidirectional realization (one heading) is required for Newman slow drift")
    heading = math.radians(next(iter(directions)))
    headings = np.asarray(dataset["wave_direction"].values, dtype=float)
    distance = np.abs(np.remainder(headings - heading + math.pi, math.tau) - math.pi)
    match = np.flatnonzero(distance <= 1e-12)
    _require(match.size == 1, "the wave heading requires one exact dataset heading; no interpolation")
    D = np.asarray(drift.values[:, match[0], :], dtype=np.float64)                 # (omega, 3)
    w = np.array([c.omega for c in components])
    _require(bool(np.all((w >= omega[0]) & (w <= omega[-1]))), "wave frequency outside the stored mean-drift range")
    Dc = np.stack([np.interp(w, omega, D[:, k]) for k in range(3)], axis=1)
    return DriftSeries(dof_keys=tuple(str(v) for v in dataset["drift_dof"].values), omegas=w,
                       amplitudes=np.array([c.amplitude for c in components]),
                       phases=np.array([c.phase for c in components]), drift=Dc,
                       heading_radians=float(headings[match[0]]), method=method,
                       wave_realization_hash=realization.content_hash,
                       hydro_content_hash=dataset_content_hash(dataset), drift_sha256=_drift_hash(dataset))


def verify_drift_binding(series: DriftSeries, dataset: xr.Dataset, realization) -> bool:
    """Rebuild against the authoritative dataset and realization; a stored series alone proves no binding."""
    _require(type(series) is DriftSeries, "a named DriftSeries is required")
    expected = build_drift_series(dataset, realization, series.method)
    _require(series.content_hash == expected.content_hash,
             "drift series does not match the authoritative dataset and wave realization")
    return True

