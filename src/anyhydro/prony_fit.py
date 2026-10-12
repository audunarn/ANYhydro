# SPDX-License-Identifier: MPL-2.0
"""Deterministic, validated Prony decomposition for finite-depth Green functions.

Capytaine 3.0 approximates the finite-depth remainder
``ref(x) = F(x) - A(kh)/(x - kh) - 2`` (``x = kappa h``) by a sum of
exponentials ``sum a_i exp(lam_i x)`` fitted on ``[-0.1, 20]``.  Its
default ``python`` route accepts the first fit whose mean-square error is
below 1e-4 on a randomly perturbed grid.  That admits pointwise errors of
0.02-0.08 at every kh in the ANYmarine range, and the resulting kernel
violates the free-surface condition (W5 physical-field evidence
``w5-prony-attribution-20261012``; field errors of 1 % in the far field and
up to 31 % at deep receivers for kh = 34.4).

This module replaces that step with a deterministic fit that is accepted
only on a measured maximum pointwise error over a dense grid, and refuses
otherwise.  Capytaine is imported lazily (never at base import time).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from functools import lru_cache

import numpy as np

from .errors import CapabilityError

PRONY_FIT_FORMAT = 'anyhydro-validated-prony-v1'
X_MIN = -0.1
X_MAX = 20.0
TAIL_MAX = 60.0
N_EXP_RANGE = tuple(range(4, 31, 2))
TARGET_MAX_ABS_ERROR = 1e-4
ACCEPT_MAX_ABS_ERROR = 5e-4
VALIDATION_POINTS = 8001
POLE_WINDOW = 0.02     # decimal evaluation within this distance of x = kh
MIN_KH = 0.1          # Capytaine's own lower limit for the finite-depth route
MAX_KH = 1e5          # at and above this Capytaine uses its asymptotic form


@dataclass(frozen=True)
class PronyDecomposition:
    """Accepted decomposition with its measured validation errors."""
    kh: float
    lam: tuple
    amp: tuple
    n_samples: int
    max_abs_error: float
    rms_error: float
    tail_max_abs_error: float

    def as_array(self):
        """Capytaine layout: row 0 growth rates, row 1 amplitudes."""
        return np.array([self.lam, self.amp], dtype=np.float64)

    def record(self):
        return dict(format=PRONY_FIT_FORMAT, kh=self.kh,
                    route='asymptotic' if self.kh >= MAX_KH else 'finite_depth',
                    n_exp=len(self.lam),
                    n_samples=self.n_samples, max_abs_error=self.max_abs_error,
                    rms_error=self.rms_error,
                    tail_max_abs_error=self.tail_max_abs_error,
                    algebraic_tail_coefficient=tail_coefficient(self.kh),
                    target_max_abs_error=TARGET_MAX_ABS_ERROR,
                    accept_max_abs_error=ACCEPT_MAX_ABS_ERROR,
                    lam=list(self.lam), amp=list(self.amp))


def _ref_float(x, kh):
    # tanh form of F: e^x/(x sinh x - q cosh x) = (1 + tanh x)/(x tanh x - q);
    # overflow-free for every x (upstream uses the exponential form)
    t = math.tanh(kh)
    q = kh * t
    sing = (1 + t) ** 2 / (1 - t * t + t / kh)
    with np.errstate(divide='ignore', invalid='ignore'):
        return (x + q) * (1 + np.tanh(x)) / (x * np.tanh(x) - q) - sing / (x - kh) - 2


def _ref_decimal(x, kh, prec=60):
    """``ref`` in 60-digit decimal arithmetic (used only next to the pole)."""
    with localcontext() as ctx:
        ctx.prec = prec
        X = Decimal(float(x))
        KH = Decimal(float(kh))
        e2k = (2 * KH).exp()
        t = (e2k - 1) / (e2k + 1)
        q = KH * t
        sing = (1 + t) ** 2 / (1 - t * t + t / KH)
        if X == KH:   # removable pole: symmetric limit, error O(1e-40)
            d = Decimal(10) ** -20
            return float((_ref_decimal_exact(KH - d, q, sing, KH)
                          + _ref_decimal_exact(KH + d, q, sing, KH)) / 2)
        return float(_ref_decimal_exact(X, q, sing, KH))


def _ref_decimal_exact(X, q, sing, KH):
    e2x = (2 * X).exp()
    tx = (e2x - 1) / (e2x + 1)
    return (X + q) * (1 + tx) / (X * tx - q) - sing / (X - KH) - 2


def reference_function(x, kh):
    """``ref(x) = F(x) - A/(x - kh) - 2`` for ``x = kappa h``.

    Points within ``POLE_WINDOW`` of the removable pole ``x = kh`` are
    evaluated in 60-digit decimal arithmetic.  For ``kh >= MAX_KH``
    (including the infinite-frequency limit) this is Capytaine's
    asymptotic form ``2 - 2/(1 + exp(-2x))``.
    """
    x = np.asarray(x, dtype=np.float64)
    if kh >= MAX_KH:
        return 2.0 - 2.0 / (1.0 + np.exp(-2.0 * x))
    out = np.array(_ref_float(x, kh), dtype=np.float64)
    near = np.abs(x - kh) < POLE_WINDOW
    if np.any(near):
        flat = out.reshape(-1)
        for i in np.flatnonzero(near.reshape(-1)):
            flat[i] = _ref_decimal(x.reshape(-1)[i], kh)
        out = flat.reshape(x.shape)
    return out[()] if out.ndim == 0 else out


def tail_coefficient(kh):
    """``c`` in ``ref(x) ~ c/x`` for large ``x`` (``c = 4 kh tanh kh - A``).

    Exponential sums cannot represent this algebraic tail beyond the fit
    range; it is negligible for kh above about 8 and is a documented
    residual of both the upstream and the validated decomposition.
    """
    if kh >= MAX_KH:
        return 0.0
    t = math.tanh(kh)
    return 4 * kh * t - kh * (1 + t) ** 2 / (t + kh * (1 - t * t))


def prony_fit(x, f, m):
    """Prony's method on an equispaced grid: growth rates, then amplitudes.

    Returns only decaying modes ``-20 < lam < 0`` (as Capytaine), with
    least-squares amplitudes; deterministic for given samples.
    """
    x = np.asarray(x, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    n = len(x)
    if n < 2 * m + 1:
        raise ValueError('too few samples for the requested exponentials')
    # linear prediction: f[j+m] = sum_i p_i f[j+m-1-i]
    rows = np.array([f[j:j + m][::-1] for j in range(n - m)])
    p, *_ = np.linalg.lstsq(rows, f[m:], rcond=None)
    roots = np.roots(np.concatenate(([1.0], -p)))
    roots = roots[np.logical_or(roots.imag != 0.0, roots.real >= 0.0)]
    with np.errstate(divide='ignore'):
        lam = np.unique(np.real(np.log(roots.astype(complex)) / (x[1] - x[0])))
    lam = lam[np.isfinite(lam) & (lam > -20.0) & (lam < 0.0)]
    if lam.size == 0:
        raise ValueError('no decaying exponential found')
    basis = np.exp(np.outer(x, lam))
    amp, *_ = np.linalg.lstsq(basis, f, rcond=None)
    return lam, amp


def _errors(lam, amp, kh):
    xv = np.linspace(X_MIN, X_MAX, VALIDATION_POINTS)
    err = reference_function(xv, kh) - (amp[None, :] * np.exp(np.outer(xv, lam))).sum(1)
    xt = np.linspace(X_MAX, TAIL_MAX, 2001)
    tail = reference_function(xt, kh) - (amp[None, :] * np.exp(np.outer(xt, lam))).sum(1)
    return float(np.max(np.abs(err))), float(np.sqrt(np.mean(err ** 2))), \
        float(np.max(np.abs(tail)))


def validated_decomposition(kh):
    """Fewest-term fit meeting the target; else the best one within acceptance.

    ``kh >= MAX_KH`` (including ``inf``) shares one asymptotic fit, as in
    Capytaine.  Raises :class:`CapabilityError` for ``kh <= MIN_KH`` or NaN,
    or when no candidate is finite, decaying and within
    ``ACCEPT_MAX_ABS_ERROR`` on the dense grid and the tail.
    """
    kh = float(kh)
    if not (kh > MIN_KH):
        raise CapabilityError(
            f'validated finite-depth decomposition requires kh > {MIN_KH}, got {kh}')
    return _validated_decomposition(min(kh, MAX_KH))


@lru_cache(maxsize=256)
def _validated_decomposition(kh):
    best = None
    for m in N_EXP_RANGE:
        n = 4 * m + 1
        xs = np.linspace(X_MIN, X_MAX, n)
        try:
            lam, amp = prony_fit(xs, reference_function(xs, kh), m)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not (np.all(np.isfinite(lam)) and np.all(np.isfinite(amp))):
            continue
        mx, rms, tail = _errors(lam, amp, kh)
        cand = PronyDecomposition(kh, tuple(float(v) for v in lam),
                                  tuple(float(v) for v in amp), n, mx, rms, tail)
        worst = max(mx, tail)
        if worst <= TARGET_MAX_ABS_ERROR:
            return cand
        if best is None or worst < max(best.max_abs_error, best.tail_max_abs_error):
            best = cand
    if best is None or max(best.max_abs_error, best.tail_max_abs_error) > ACCEPT_MAX_ABS_ERROR:
        found = 'none' if best is None else f'{max(best.max_abs_error, best.tail_max_abs_error):.3e}'
        raise CapabilityError(
            f'no validated Prony decomposition for kh={kh}: best max error {found} '
            f'> {ACCEPT_MAX_ABS_ERROR}')
    return best


_VALIDATED_CLASS = None


def _validated_class():
    """Module-level subclass, created on first use (lazy Capytaine import).

    Defined at module scope (via ``__getattr__``) so instances pickle by
    reference for Capytaine's joblib ``n_jobs > 1`` route.
    """
    global _VALIDATED_CLASS
    if _VALIDATED_CLASS is None:
        from capytaine.green_functions.delhommeau import Delhommeau

        class ValidatedPronyDelhommeau(Delhommeau):
            """Delhommeau with the validated finite-depth decomposition."""

            def find_best_exponential_decomposition(self, dimensionless_wavenumber, *,
                                                    method=None):
                return validated_decomposition(float(dimensionless_wavenumber)).as_array()

        ValidatedPronyDelhommeau.__module__ = __name__
        ValidatedPronyDelhommeau.__qualname__ = 'ValidatedPronyDelhommeau'
        _VALIDATED_CLASS = ValidatedPronyDelhommeau
    return _VALIDATED_CLASS


def __getattr__(name):
    if name == 'ValidatedPronyDelhommeau':
        return _validated_class()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def validated_delhommeau(capytaine_module=None, **options):
    """Capytaine ``Delhommeau`` whose finite-depth decomposition is validated.

    ``options`` are passed to the constructor.  The decomposition method is
    recorded in ``exportable_settings`` so cached results and solver
    identities never mix with the upstream default route.
    ``capytaine_module`` is accepted for call-site symmetry only.
    """
    gf = _validated_class()(**options)
    gf.exportable_settings = dict(gf.exportable_settings,
                                  finite_depth_prony_decomposition=PRONY_FIT_FORMAT)
    gf._hash = hash(tuple(gf.exportable_settings.values()))
    return gf
