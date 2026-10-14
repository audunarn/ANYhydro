# SPDX-License-Identifier: MPL-2.0
"""Pairwise reciprocity diagnostic of radiation matrices (owner decision 2026-10-13).

For each frequency and DOF pair ``i < j`` the defect is ``|M_ij - M_ji| / sqrt(|M_ii M_jj|)``.
Reciprocity makes it zero for the exact solution; for a panel method it shrinks with resolution,
so a large value marks an under-resolved coupling. It does not change any coefficient (the
derived-operator correction policies in ``operators`` stay separate) and it is no admission
criterion: the adapter records it in the dataset and warns above ``RECIPROCITY_WARNING``.

Pairs whose scale ``sqrt(|M_ii M_jj|)`` is below ``SCALE_FLOOR`` of that pair's largest scale
over frequency (e.g. damping at long periods) are skipped, as are pairs with a zero scale.
Evidence for the level: OrcaWave benchmark B3, barge sway/roll A42 vs A24 at T = 8 s gives 8.6 %
(1700 panels), 2.4 % (6800) and 1.4 % (27200); the extrapolated coefficients of the two codes agree
within 0.5 % (orcawave-benchmark-b-20261013/b3-uniform-refinement).
The level holds for both BEM methods (direct-default programme D5, 2026-10-14; worst A / B
defect): barge U0 deep indirect 4.4 % / 9.3 %, direct 3.8 % / 1.6 %; W5 two-box fixture n = 10
added mass indirect 2.5 %, direct 0.8 %. Weak modes inflate the ratio for both methods (W5 damping
heave/yaw: indirect 162 %, direct 105 %, yaw damping of a square box being near zero), so a
warning on a pair with a near-zero diagonal is a resolution hint, not an error measure.
"""
from __future__ import annotations

import warnings

import numpy as np

RECIPROCITY_WARNING = 0.02
SCALE_FLOOR = 1e-3


class ReciprocityWarning(UserWarning):
    """A radiation matrix breaks reciprocity by more than ``RECIPROCITY_WARNING`` (a mesh-resolution sign)."""


def pairwise_defect(matrices):
    """Worst pairwise defect of real ``(n_omega, n, n)`` matrices -> (value, omega index, (i, j)).

    Returns ``(0.0, None, None)`` when no pair has a usable scale.
    """
    M = np.asarray(matrices, dtype=np.float64)
    if M.ndim != 3 or M.shape[1] != M.shape[2]:
        raise ValueError(f"expected (n_omega, n, n) matrices, got shape {M.shape}")
    if not np.all(np.isfinite(M)):
        raise ValueError("reciprocity diagnostic needs finite matrices")
    diag = np.abs(np.diagonal(M, axis1=1, axis2=2))                       # (n_omega, n)
    with np.errstate(divide="ignore"):
        log_root = 0.5 * np.log(diag)                                      # -inf for a zero diagonal
    log_scale = log_root[:, :, None] + log_root[:, None, :]                # log sqrt(|M_ii M_jj|), no over/underflow
    log_peak = log_scale.max(axis=0, keepdims=True)
    # inclusive floor; the 1e-9 log allowance keeps a pair exactly at the floor despite log rounding
    finite = np.isfinite(log_scale)
    relative = np.full(log_scale.shape, -np.inf)
    np.subtract(log_scale, log_peak, out=relative, where=finite)    # a finite scale has a finite peak
    usable = finite & (relative >= np.log(SCALE_FLOOR) - 1e-9)
    usable &= np.triu(np.ones(M.shape[1:], dtype=bool), k=1)[None]
    # Each pair is normalised by its own magnitude m = max(|M_ij|, |M_ji|): |M_ij/m - M_ji/m| lies in
    # [0, 2] exactly representable for any finite (also subnormal) entries; the factor m / scale is
    # formed in the log domain and capped so the result stays a finite (JSON-safe) float.
    Mt = np.transpose(M, (0, 2, 1))
    m = np.maximum(np.abs(M), np.abs(Mt))
    ok = usable & (m > 0)
    unit = np.abs(M[ok] / m[ok] - Mt[ok] / m[ok])
    cap = np.log(np.finfo(np.float64).max) - np.log(4.0)          # unit <= 2: product <= max/2 despite rounding
    defect = np.zeros_like(M)
    defect[ok] = unit * np.exp(np.minimum(np.log(m[ok]) - log_scale[ok], cap))
    if not usable.any():
        return 0.0, None, None
    o, i, j = np.unravel_index(int(np.argmax(np.where(usable, defect, -1.0))), defect.shape)
    return float(defect[o, i, j]), int(o), (int(i), int(j))


def reciprocity_diagnostic(added_mass, damping, omegas, dof_keys, *, threshold=RECIPROCITY_WARNING):
    """JSON-ready diagnostic of A and B; ``exceeds`` is true when either worst defect > ``threshold``."""
    omegas = np.asarray(omegas, dtype=np.float64)
    out = {"definition": "max over omega and i<j of |M_ij - M_ji| / sqrt(|M_ii M_jj|)",
           "threshold": float(threshold), "scale_floor": SCALE_FLOOR}
    for name, M in (("added_mass", added_mass), ("radiation_damping", damping)):
        value, o, pair = pairwise_defect(M)
        out[name] = {"max": value,
                     "omega": None if o is None else float(omegas[o]),
                     "dofs": None if pair is None else [str(dof_keys[pair[0]]), str(dof_keys[pair[1]])]}
    out["exceeds"] = bool(out["added_mass"]["max"] > threshold or out["radiation_damping"]["max"] > threshold)
    return out


def warn_if_exceeded(diagnostic, *, stacklevel=2):
    """Emit a ``ReciprocityWarning`` naming the worst pair when the diagnostic exceeds its threshold."""
    if not diagnostic["exceeds"]:
        return
    parts = []
    for name in ("added_mass", "radiation_damping"):
        d = diagnostic[name]
        if d["max"] > diagnostic["threshold"]:
            parts.append(f"{name} {100 * d['max']:.1f} % ({d['dofs'][0]} / {d['dofs'][1]} at omega {d['omega']:.4g} rad/s)")
    warnings.warn("radiation reciprocity defect above "
                  f"{100 * diagnostic['threshold']:.0f} %: " + "; ".join(parts)
                  + " -- the panel mesh likely under-resolves this coupling", ReciprocityWarning, stacklevel=stacklevel)


def dataset_reciprocity(ds, *, threshold=RECIPROCITY_WARNING):
    """The diagnostic of any ANYhydro hydro dataset (adapter, HAMS or WAMIT import)."""
    dofs = list(ds["added_mass"].coords["influenced_dof"].values)
    if dofs != list(ds["added_mass"].coords["radiating_dof"].values):
        raise ValueError("added_mass influenced and radiating DOF axes differ; reciprocity needs one DOF order")
    return reciprocity_diagnostic(ds["added_mass"].transpose("omega", "influenced_dof", "radiating_dof").values,
                                  ds["radiation_damping"].transpose("omega", "influenced_dof", "radiating_dof").values,
                                  ds["omega"].values, dofs, threshold=threshold)
