"""Far-field mean wave drift (surge, sway, yaw) from exported Kochin functions.

Contract: docs/MEAN_DRIFT_CONTRACT.md (rev 3). Development only; nothing here qualifies drift loads.

With H = i (H_D + sum_j X_j H_R,j) the total Kochin function of the body (world-origin reference, unit incident
amplitude, ANYhydro phasor x = Re{X exp(-i w t)}), t = tanh(kh), D = t + kh sech^2(kh):

    (F_x, F_y) = -c1 Im H(beta) (cos beta, sin beta) - c2 int_0^2pi |H|^2 (cos theta, sin theta) dtheta
    M_z,0      = (c1 / k) Re H'(beta) - (c2 / k) Im int_0^2pi conj(H) H' dtheta
    c1 = 2 pi rho w / t,   c2 = 2 pi rho k^2 / D

H at the heading by trigonometric (DFT) interpolation, H' by spectral differentiation, integrals by the periodic
trapezoid rule. Yaw is transferred from the world origin to the dataset rotation centre c:
M_z,c = M_z,0 - c_x F_y + c_y F_x. Finite depth was admitted after validation V3 (MacCamy-Fuchs ratios within
0.09 %, MD2 evidence 2026-10-15); where tanh(kh) >= 1 - 1e-6 the deep-water coefficients are used and recorded.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np
import xarray as xr

from .errors import CapabilityError, HydroError

MEAN_DRIFT_FORMAT = "anyhydro-far-field-mean-drift-v1"
DRIFT_DOFS = ("surge", "sway", "yaw")
#: finite-depth admission gate; set after the MacCamy-Fuchs finite-depth validation V3 passed (owner, 2026-10-15)
FINITE_DEPTH_ADMITTED = True
DEEP_TANH_THRESHOLD = 1.0 - 1e-6


class MeanDriftError(HydroError):
    """Mean drift inputs are malformed or not admitted."""


def _require(condition: bool, message: str, error=MeanDriftError) -> None:
    if not condition:
        raise error(message)


def wave_environment(dataset: xr.Dataset) -> dict:
    raw = dataset.attrs.get("wave_environment")
    _require(raw is not None, "dataset carries no wave_environment (request kochin='export' from the backend)")
    env = json.loads(raw) if isinstance(raw, str) else dict(raw)
    depth = math.inf if env["water_depth"] == "inf" else float(env["water_depth"])
    return dict(water_depth=depth, rho=float(env["rho"]), g=float(env["g"]))


def fourier_coefficients(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FFT coefficients c_m and integer orders m of samples on theta_j = 2 pi j / N (last axis)."""
    n = values.shape[-1]
    coefficients = np.fft.fft(values, axis=-1) / n
    orders = np.fft.fftfreq(n, d=1.0 / n)
    if n % 2 == 0:                       # split the Nyquist term symmetrically (real-valued interpolant)
        half = n // 2
        coefficients = np.concatenate([coefficients, coefficients[..., half:half + 1] / 2.0], axis=-1)
        coefficients[..., half] /= 2.0
        orders = np.concatenate([orders, [float(half)]])
    return coefficients, orders


def evaluate(coefficients: np.ndarray, orders: np.ndarray, theta: float, derivative: bool = False) -> np.ndarray:
    phase = np.exp(1j * orders * theta)
    if derivative:
        phase = 1j * orders * phase
    return coefficients @ phase


def spectral_derivative(values: np.ndarray) -> np.ndarray:
    """d/dtheta of the trigonometric interpolant at the grid points (FFT, O(N log N)).

    With the symmetric Nyquist split the Nyquist term contributes cos(N theta / 2), whose derivative vanishes on the
    grid, so its order is zeroed.
    """
    n = values.shape[-1]
    orders = np.fft.fftfreq(n, d=1.0 / n)
    if n % 2 == 0:
        orders[n // 2] = 0.0
    return np.fft.ifft(np.fft.fft(values, axis=-1) * (1j * orders), axis=-1)


def coefficients(omega: float, k: float, depth: float, rho: float) -> tuple[float, float, str]:
    """(c1, c2, treatment) for one frequency."""
    if math.isinf(depth):
        return 2 * math.pi * rho * omega, 2 * math.pi * rho * k ** 2, "infinite"
    t = math.tanh(k * depth)
    if t >= DEEP_TANH_THRESHOLD:
        return 2 * math.pi * rho * omega, 2 * math.pi * rho * k ** 2, "deep_substituted"
    _require(FINITE_DEPTH_ADMITTED,
             f"finite-depth drift is not admitted yet (tanh(kh) = {t:.6g} < {DEEP_TANH_THRESHOLD}); "
             "MacCamy-Fuchs finite-depth validation V3 is pending", CapabilityError)
    D = t + k * depth / math.cosh(k * depth) ** 2
    return 2 * math.pi * rho * omega / t, 2 * math.pi * rho * k ** 2 / D, "finite"


def drift_from_kochin(H: np.ndarray, beta: float, omega: float, k: float, depth: float, rho: float) -> tuple:
    """(F_x, F_y, M_z0, treatment) for one total Kochin function H on the periodic theta grid."""
    n = H.shape[-1]
    theta = 2.0 * np.pi * np.arange(n) / n
    c1, c2, treatment = coefficients(omega, k, depth, rho)
    coeff, orders = fourier_coefficients(H)
    h_beta = evaluate(coeff, orders, beta)
    dh_beta = evaluate(coeff, orders, beta, derivative=True)
    dH = spectral_derivative(H)
    weight = 2.0 * np.pi / n
    power = np.abs(H) ** 2
    fx = -c1 * math.cos(beta) * h_beta.imag - c2 * weight * float(np.sum(power * np.cos(theta)))
    fy = -c1 * math.sin(beta) * h_beta.imag - c2 * weight * float(np.sum(power * np.sin(theta)))
    mz0 = (c1 / k) * dh_beta.real - (c2 / k) * weight * float(np.sum(np.conj(H) * dH).imag)
    return float(fx), float(fy), float(mz0), treatment


def _motion_hash(motions: xr.DataArray) -> str:
    h = hashlib.sha256()
    for name in ("omega", "wave_direction", "radiating_dof"):
        h.update(json.dumps([str(v) if name == "radiating_dof" else repr(float(v))
                             for v in motions[name].values]).encode())
    h.update(np.ascontiguousarray(motions.values.astype(np.complex128)).tobytes())
    return h.hexdigest()


def _check_motion_provenance(provenance: Mapping[str, Any], moving: bool) -> None:
    """Contract section 3: 'fixed' exactly for all-zero motions; otherwise the equation-of-motion source.

    Moving bodies need either ``source='prescribed'`` (motions given directly, e.g. validation) or a
    ``matrices_hash`` (non-empty string) and ``damping`` in {'raw', 'derived'}.
    """
    source = provenance["source"]
    if not moving:
        _require(source == "fixed", "all-zero motions must have provenance source 'fixed'")
        return
    _require(source != "fixed", "provenance source 'fixed' given with nonzero motions")
    if source == "prescribed":
        return
    _require(isinstance(provenance.get("matrices_hash"), str) and provenance["matrices_hash"].strip(),
             "moving-body provenance needs a non-empty 'matrices_hash' (or source 'prescribed')")
    _require(provenance.get("damping") in ("raw", "derived"),
             "moving-body provenance needs 'damping' = 'raw' or 'derived'")


def far_field_mean_drift(dataset: xr.Dataset, motions: xr.DataArray, provenance: Mapping[str, Any]) -> xr.Dataset:
    """Return a copy of ``dataset`` with ``mean_drift_far_field(omega, wave_direction, drift_dof)``.

    ``motions``: complex (omega, wave_direction, radiating_dof) per unit wave amplitude about the dataset rotation
    centre, coordinates exactly equal to the dataset's; all zeros for a fixed body. ``provenance``: a mapping with at
    least ``source`` (e.g. 'fixed', or a description of the equation of motion) - recorded with the motion hash.
    """
    from .schema import KOCHIN_SCHEMA_VERSION, MEAN_DRIFT_UNITS, _json_attr, validate_hydro_dataset
    for name in ("kochin_radiation", "kochin_diffraction", "wavenumber"):
        _require(name in dataset.data_vars, f"dataset lacks {name!r} (request kochin='export')")
    bodies = list(dataset["body"].values)
    _require(len(bodies) == 1, "mean drift is admitted for a single body only")
    _require(isinstance(provenance, Mapping) and isinstance(provenance.get("source"), str)
             and provenance["source"].strip(), "provenance needs a non-empty 'source'")
    _require(isinstance(motions, xr.DataArray) and tuple(motions.dims) == ("omega", "wave_direction", "radiating_dof"),
             "motions must be a DataArray with dims (omega, wave_direction, radiating_dof)")
    for name in ("omega", "wave_direction", "radiating_dof"):
        ours, theirs = dataset[name].values, motions[name].values
        if name == "wave_direction":          # equal after normalisation to [0, 2 pi) (contract section 3)
            ours, theirs = np.mod(ours.astype(float), 2 * np.pi), np.mod(theirs.astype(float), 2 * np.pi)
        _require(len(ours) == len(theirs) and all(a == b for a, b in zip(ours, theirs)),
                 f"motions coordinate {name!r} must equal the dataset's exactly")
    X = np.asarray(motions.values, dtype=np.complex128)
    _require(bool(np.all(np.isfinite(X))), "motions must be finite")
    _check_motion_provenance(provenance, moving=bool(np.any(X != 0)))
    env = wave_environment(dataset)
    headings = np.mod(np.asarray(dataset["wave_direction"].values, dtype=float), 2 * np.pi)
    _require(len(set(np.round(headings, 12))) == len(headings), "equivalent duplicate headings are refused")
    H_R = dataset["kochin_radiation"].transpose("omega", "radiating_dof", "theta").values
    H_D = dataset["kochin_diffraction"].transpose("omega", "wave_direction", "theta").values
    k = dataset["wavenumber"].values
    centre = np.asarray(dataset["rotation_center"].values, dtype=float).reshape(-1, 3)[0]
    body = bodies[0]
    out = np.zeros((len(dataset["omega"]), len(headings), 3))
    treatments = set()
    for i, omega in enumerate(dataset["omega"].values):
        for h, beta in enumerate(headings):
            H = 1j * (H_D[i, h] + X[i, h] @ H_R[i])
            fx, fy, mz0, treatment = drift_from_kochin(H, float(beta), float(omega), float(k[i]),
                                                       env["water_depth"], env["rho"])
            out[i, h] = (fx, fy, mz0 - centre[0] * fy + centre[1] * fx)
            treatments.add(treatment)
    result = dataset.copy()
    result = result.assign_coords(drift_dof=[f"{body}::{d}" for d in DRIFT_DOFS])
    result["mean_drift_far_field"] = (("omega", "wave_direction", "drift_dof"), out)
    result.attrs["schema_version"] = KOCHIN_SCHEMA_VERSION
    result.attrs["mean_drift_far_field"] = json.dumps(dict(
        format=MEAN_DRIFT_FORMAT, method="far_field_momentum_conservation",
        coefficients="c1 = 2 pi rho w / tanh(kh), c2 = 2 pi rho k^2 / D", depth_treatment=sorted(treatments),
        units=MEAN_DRIFT_UNITS,
        yaw_reference=[float(v) for v in centre], kochin=_json_attr(dataset, "kochin_export"),
        motions=dict(provenance, motion_hash=_motion_hash(motions)),
        qualification="unqualified"), sort_keys=True)
    return validate_hydro_dataset(result)
