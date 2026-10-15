# SPDX-License-Identifier: MPL-2.0
"""Far-field mean drift (docs/MEAN_DRIFT_CONTRACT.md): formulas, admission, Kochin export, persistence."""
from __future__ import annotations

import math

import numpy as np
import pytest

from anyhydro import mean_drift as md
from anyhydro.errors import CapabilityError


def _series(theta, coeffs):
    return sum(a * np.exp(1j * m * theta) for m, a in coeffs.items())


COEFFS = {-3: 0.2 - 0.1j, -1: 0.7 + 0.3j, 0: -0.4 + 0.9j, 1: 1.1 - 0.2j, 2: 0.05 + 0.6j, 5: -0.3 + 0.25j}


def _reference(beta, omega, k, rho, n_dense=8192):
    """Same formulas, H(beta) and H'(beta) from the exact series, integrals on a dense grid."""
    theta = 2 * np.pi * np.arange(n_dense) / n_dense
    H = _series(theta, COEFFS)
    dH = sum(1j * m * a * np.exp(1j * m * theta) for m, a in COEFFS.items())
    hb = sum(a * np.exp(1j * m * beta) for m, a in COEFFS.items())
    dhb = sum(1j * m * a * np.exp(1j * m * beta) for m, a in COEFFS.items())
    c1, c2 = 2 * np.pi * rho * omega, 2 * np.pi * rho * k ** 2
    w = 2 * np.pi / n_dense
    fx = -c1 * np.cos(beta) * hb.imag - c2 * w * np.sum(np.abs(H) ** 2 * np.cos(theta))
    fy = -c1 * np.sin(beta) * hb.imag - c2 * w * np.sum(np.abs(H) ** 2 * np.sin(theta))
    mz = (c1 / k) * dhb.real - (c2 / k) * w * np.sum(np.conj(H) * dH).imag
    return fx, fy, mz


@pytest.mark.parametrize("n", [32, 33, 64])
@pytest.mark.parametrize("beta", [0.0, 0.37, math.pi, 5.9])
def test_spectral_evaluation_is_exact_for_resolved_series(n, beta):
    theta = 2 * np.pi * np.arange(n) / n
    H = _series(theta, COEFFS)
    omega, rho = 1.3, 1025.0
    k = omega ** 2 / 9.81
    got = md.drift_from_kochin(H, beta, omega, k, math.inf, rho)
    ref = _reference(beta, omega, k, rho)
    scale = max(abs(v) for v in ref)
    assert np.allclose(got[:3], ref, rtol=0, atol=1e-10 * scale)
    assert got[3] == "infinite"


def test_finite_depth_gate_and_deep_substitution(monkeypatch):
    theta = 2 * np.pi * np.arange(64) / 64
    H = _series(theta, COEFFS)
    assert md.FINITE_DEPTH_ADMITTED is True                          # admitted after V3 (MD2 evidence)
    assert md.drift_from_kochin(H, 0.3, 1.0, 0.2, 5.0, 1025.0)[3] == "finite"     # kh = 1
    monkeypatch.setattr(md, "FINITE_DEPTH_ADMITTED", False)
    with pytest.raises(CapabilityError, match="not admitted"):
        md.drift_from_kochin(H, 0.3, 1.0, 0.2, 5.0, 1025.0)
    deep = md.drift_from_kochin(H, 0.3, 1.0, 0.2, math.inf, 1025.0)
    nearly = md.drift_from_kochin(H, 0.3, 1.0, 0.2, 200.0, 1025.0)       # kh = 40
    assert nearly[3] == "deep_substituted" and np.allclose(nearly[:3], deep[:3], rtol=0, atol=0)


def test_finite_depth_coefficients_follow_the_contract():
    md_admitted = md.FINITE_DEPTH_ADMITTED
    try:
        md.FINITE_DEPTH_ADMITTED = True
        k, h, w, rho = 1.0, 1.0, 2.0, 1000.0
        c1, c2, treatment = md.coefficients(w, k, h, rho)
        t = math.tanh(k * h)
        D = t + k * h / math.cosh(k * h) ** 2
        assert treatment == "finite"
        assert c1 == pytest.approx(2 * math.pi * rho * w / t) and c2 == pytest.approx(2 * math.pi * rho * k ** 2 / D)
    finally:
        md.FINITE_DEPTH_ADMITTED = md_admitted


def test_drift_is_quadratic_in_the_total_kochin_function():
    theta = 2 * np.pi * np.arange(64) / 64
    H = _series(theta, COEFFS)
    a = md.drift_from_kochin(H, 0.8, 1.2, 0.15, math.inf, 1025.0)
    b = md.drift_from_kochin(3.0 * H, 0.8, 1.2, 0.15, math.inf, 1025.0)
    # the incident-interference term is linear in H, the radiated term quadratic: check the decomposition
    c1, c2, _ = md.coefficients(1.2, 0.15, math.inf, 1025.0)
    lin = np.array(md.drift_from_kochin(H, 0.8, 1.2, 0.15, math.inf, 1025.0)[:3])
    assert not np.allclose(np.array(b[:3]), 9 * lin)          # not purely quadratic (interference term)
    hb = md.evaluate(*md.fourier_coefficients(H), 0.8)
    interference_x = -c1 * math.cos(0.8) * hb.imag
    assert b[0] - 3 * interference_x == pytest.approx(9 * (a[0] - interference_x), rel=1e-12)


def test_symmetric_kochin_gives_zero_sway_and_yaw_in_head_seas():
    n = 128
    theta = 2 * np.pi * np.arange(n) / n
    H = (1.0 + 0.5j) + (0.3 - 0.2j) * np.cos(theta) + (0.1 + 0.05j) * np.cos(2 * theta)     # even in theta
    fx, fy, mz, _ = md.drift_from_kochin(H, 0.0, 1.0, 0.1, math.inf, 1025.0)
    assert abs(fy) <= 1e-8 * abs(fx) and abs(mz) <= 1e-8 * abs(fx) * 10


@pytest.mark.parametrize("n", [720, 721])
def test_fft_derivative_equals_the_interpolant_derivative(n):
    theta = 2 * np.pi * np.arange(n) / n
    H = _series(theta, COEFFS) + 0.3 * np.cos(n // 2 * theta)       # includes a Nyquist term for even n
    coeff, orders = md.fourier_coefficients(H)
    dense = np.array([md.evaluate(coeff, orders, t_, derivative=True) for t_ in theta])
    assert np.allclose(md.spectral_derivative(H), dense, rtol=0, atol=1e-10 * np.abs(dense).max())


# ------------------------------------------------ physics anchor: MacCamy-Fuchs cylinder (no BEM, scipy only)
G_, RHO_ = 9.81, 1000.0


def _havelock(ka):
    """Deep-water drift on a fixed vertical cylinder, F / (rho g A^2 a) (MD2 evidence: near field = far field)."""
    from scipy.special import h1vp
    n = np.arange(int(ka + 40 + 10 * ka ** (1 / 3)) + 1)
    hp0, hp1 = h1vp(n, ka), h1vp(n + 1, ka)
    return 4 / (np.pi ** 2 * ka ** 3) * np.sum((1 - n * (n + 1) / ka ** 2) ** 2 / (np.abs(hp0) ** 2 * np.abs(hp1) ** 2))


def _cylinder_kochin(ka, theta, beta):
    """Diffraction Kochin function of the MacCamy-Fuchs scattered wave, Capytaine compute_kochin normalisation."""
    from scipy.special import h1vp, jvp
    k, w = ka, math.sqrt(G_ * ka)
    n = np.arange(int(ka + 40 + 10 * ka ** (1 / 3)) + 1)
    eps = np.where(n == 0, 1.0, 2.0)
    amp = -(eps * jvp(n, ka) / h1vp(n, ka)) @ np.cos(np.outer(n, theta - beta))
    return -(-1j * G_ / w) * amp / (2 * np.pi * 1j * k)


@pytest.mark.parametrize("ka, expected", [(0.5, 0.285986), (1.0, 0.664926), (2.0, 0.627175)])
def test_cylinder_drift_matches_havelock(ka, expected):
    beta = 0.7
    theta = 2 * np.pi * np.arange(720) / 720
    H = 1j * _cylinder_kochin(ka, theta, beta)                     # contract: H = i (H_D + sum X_j H_R,j)
    fx, fy, mz, _ = md.drift_from_kochin(H, beta, math.sqrt(G_ * ka), ka, math.inf, RHO_)
    assert _havelock(ka) == pytest.approx(expected, abs=1e-6)
    assert math.hypot(fx, fy) / (RHO_ * G_) == pytest.approx(_havelock(ka), rel=1e-9)
    assert math.atan2(fy, fx) == pytest.approx(beta, abs=1e-9)    # along the heading
    assert abs(mz) <= 1e-9 * RHO_ * G_


def test_havelock_short_wave_limit():
    assert _havelock(1000.0) == pytest.approx(2 / 3, rel=1e-3)
