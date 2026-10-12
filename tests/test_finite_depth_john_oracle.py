# SPDX-License-Identifier: MPL-2.0
"""Tests for the independent finite-depth John-series oracle."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO / 'src'), str(REPO / 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

import check_finite_depth_john_oracle as oracle  # noqa: E402
import check_physical_green_field as cpgf  # noqa: E402

EVIDENCE = Path('C:/ANYworkspaceWork/anymarine-evidence')
RECORD = EVIDENCE / 'w5-free-surface-repair-20261011' / 'physical-e1-primitives-01.json'
REFITS = EVIDENCE / 'w5-prony-attribution-20261012' / 'coeffs.json'


@pytest.fixture(scope='module')
def moderate():
    # kh = 3: finite-depth effects are O(1), unlike the deep frozen case
    return oracle.FiniteDepthKernel(0.3, 10.0, n_modes=400)


def test_series_satisfies_surface_and_bottom_conditions(moderate):
    src = np.array([[0.0, 0.0, -1.3], [0.4, -0.2, -6.0]])
    surf = np.array([[2.0, 1.0, 0.0], [-3.5, 4.0, 0.0], [7.0, -1.0, 0.0]])
    bot = surf.copy()
    bot[:, 2] = -moderate.h
    g, gz, tail = moderate.green(surf, src)
    assert tail < 1e-30
    assert np.max(np.abs(gz - moderate.nu * g)) <= 1e-12 * np.max(np.abs(gz))
    _, gzb, _ = moderate.green(bot, src)
    assert np.max(np.abs(gzb)) <= 1e-12 * np.max(np.abs(gz))


def test_series_is_harmonic_and_gradient_consistent(moderate):
    src = np.array([[0.0, 0.0, -2.0]])
    x0 = np.array([1.7, -0.9, -4.0])
    step = 1e-3
    pts = [x0]
    for d in range(3):
        for s in (1, -1):
            p = x0.copy()
            p[d] += s * step
            pts.append(p)
    g, gz, _ = moderate.green(np.array(pts), src)
    g = g[:, 0]
    lap = (g[1:].sum() - 6 * g[0]) / step ** 2
    assert abs(lap) <= 1e-4 * abs(g[0]) / 1.0 ** 2
    fd_z = (g[5] - g[6]) / (2 * step)
    assert abs(fd_z - gz[0, 0]) <= 1e-6 * abs(gz[0, 0])


def test_far_field_matches_checker_asymptote(moderate):
    k, h = moderate.k, moderate.h
    src = np.array([[[0.1, -0.05, -1.0]]])
    w = np.array([[1.0]])
    rec = np.array([[200 / k * math.cos(0.4), 200 / k * math.sin(0.4), -2.5]])
    far, _ = cpgf.far_field_potential(rec, src, w, k=k, water_depth=h)
    g, _, _ = moderate.green(rec, src[0])
    gi, _ = moderate.wave_imag(rec, src[0])
    john = -1 / (4 * math.pi) * (g[0, 0] - 1j * gi[0, 0])
    # same propagating mode; the checker uses the large-kR Hankel asymptote
    assert abs(john.real - far[0].real) <= 2e-3 * abs(far[0])
    assert abs(abs(john) - abs(far[0])) <= 2e-3 * abs(far[0])


def test_prony_error_field_equals_closed_form_images(moderate):
    # an exactly exponential "fit error" must give the four Lipschitz images
    mu, b = -1.5, 0.37
    x = np.array([[1.2, 0.4, -0.3], [-2.0, 3.0, -7.5]])
    y = np.array([[0.0, 0.0, -0.8], [0.3, 0.1, -4.0]])
    m, mz = moderate.prony_error_field(x, y, None, eps_fn=lambda s: b * np.exp(mu * s))
    with_term = moderate.rankine_images(x, y, ([mu], [b]))
    base = moderate.rankine_images(x, y, ([], []))
    np.testing.assert_allclose(m, with_term[0] - base[0], rtol=1e-11, atol=0)
    np.testing.assert_allclose(mz, with_term[1] - base[1], rtol=1e-9, atol=1e-14)


def test_ref_removable_pole_is_smooth():
    kern = oracle.FiniteDepthKernel(0.5, 10.0, n_modes=10)   # kh = 5 inside fit range
    xs = kern.kh + np.linspace(-0.05, 0.05, 41)
    v = kern.ref(xs)
    assert np.all(np.isfinite(v))
    second = np.diff(v, 2)
    assert np.max(np.abs(second)) < 1e-4


def test_series_is_mode_converged():
    kern_a = oracle.FiniteDepthKernel(0.2293577981651376, 150.0, n_modes=900)
    kern_b = oracle.FiniteDepthKernel(0.2293577981651376, 150.0, n_modes=1500)
    x = np.array([[6.1, -5.2, 0.0], [5.5, 4.8, -22.5], [8.2, 7.7, -150.0]])
    y = np.array([[0.0, 0.03, -0.02]])
    ga, gza, _ = kern_a.green(x, y)
    gb, gzb, _ = kern_b.green(x, y)
    np.testing.assert_allclose(ga, gb, rtol=1e-13)
    np.testing.assert_allclose(gza, gzb, rtol=1e-11, atol=1e-25)


def test_error_field_refuses_algebraic_tail():
    # kh = 3: ref(x) ~ 0.29/x, so a cut-off at X_MAX would truncate the field
    kern = oracle.FiniteDepthKernel(0.02, 150.0, n_modes=10)
    with pytest.raises(ValueError, match='algebraic tail'):
        kern.prony_error_field(np.array([[5.0, 0.0, 0.0]]), np.array([[0.0, 0.0, -0.1]]),
                               ([-1.0], [0.1]))


def test_slow_candidate_exponential_extends_the_cutoff():
    # review case: a slow term 2e-4*exp(-0.01x) passes grid acceptance but its
    # field lies mostly beyond x = 30; the cut-off must follow it, and the
    # field must equal the closed-form Lipschitz images of that term
    kern = oracle.FiniteDepthKernel(0.2293577981651376, 150.0, n_modes=10)
    assert kern.integration_cutoff(([-1.7, -0.01], [0.5, 2e-4])) > 1000
    with pytest.raises(ValueError, match='candidate exponential tail'):
        kern.integration_cutoff(([-0.001], [1.0]))
    # review case: passes only at the cap itself (required cut-off ~1773)
    assert kern.integration_cutoff(([-0.01], [5e-4])) == oracle.X_CUT_CAP
    x = np.array([[5.0, 1.0, -0.2]])
    y = np.array([[0.0, 0.0, -0.1]])
    mu, b = -0.01, 2e-4
    xc = kern.integration_cutoff(([mu], [b]))
    xs, ws, _ = kern.eps_nodes(None, lambda s: b * np.exp(mu * s), x_cut=xc)
    with_term = kern.rankine_images(x, y, ([mu], [b]))
    base = kern.rankine_images(x, y, ([], []))
    h = kern.h
    r = np.hypot(5.0, 1.0)
    kap = xs / h
    from scipy.special import j0
    cc, _ = oracle.depth_products(xs, -0.2 / h, -0.1 / h)
    val = (2 / h) * np.sum(ws * b * np.exp(mu * xs) * cc * j0(kap * r))
    assert abs(val - (with_term[0] - base[0])[0, 0]) < 1e-9 * abs(with_term[0] - base[0])[0, 0]


def test_depth_products_match_direct_form():
    x = np.linspace(0, 30, 301)
    for a, b in ((-0.0, -0.001), (-0.3, -0.7), (-1.0, -0.2)):
        cc, sc = oracle.depth_products(x, a, b)
        np.testing.assert_allclose(cc, np.exp(-2 * x) * np.cosh(x * (1 + a)) * np.cosh(x * (1 + b)),
                                   rtol=1e-13, atol=1e-300)
        np.testing.assert_allclose(sc, np.exp(-2 * x) * np.sinh(x * (1 + a)) * np.cosh(x * (1 + b)),
                                   rtol=1e-12, atol=1e-15)
    cc, sc = oracle.depth_products(np.array([2000.0]), -0.001, -0.002)
    assert np.isfinite(cc).all() and np.isfinite(sc).all()


def test_constructor_refuses_invalid_input():
    with pytest.raises(ValueError):
        oracle.FiniteDepthKernel(-1.0, 10.0)
    with pytest.raises(ValueError):
        oracle.FiniteDepthKernel(0.3, 10.0).green(np.zeros((1, 3)), np.zeros((1, 3)))


@pytest.mark.skipif(not RECORD.is_file(), reason='actual evidence record not present')
def test_actual_record_residual_is_prony_plus_mixing():
    record = json.loads(RECORD.read_text(encoding='utf-8'))
    res = oracle.attribute_record(record)
    expected = {'hull': (2.6666, 2.9019), 'lid': (2.9248, 3.1494)}
    for label, case in res['cases'].items():
        assert case['series_tail'] < 1e-30
        assert abs(case['normalization_fit'] - 1) < 1e-3
        assert case['native_vs_model_max_rel'] < 1e-11
        assert case['native_kz_vs_model_max_rel'] < 1e-11   # off-surface gradients too
        for row, ratio in zip(case['surface'], expected[label]):
            assert abs(row['measured_over_predicted'] - 1) < 1e-9
            assert abs(row['composite_gate_ratio'] - ratio) < 1e-4
            assert abs(row['exact_constructed']) < 1e-6 * abs(row['measured_real_residual'])
            # without the Prony error the remaining mixing residual passes
            assert row['predicted_gate_ratio_without_prony'] < 0.2


@pytest.mark.skipif(not (RECORD.is_file() and REFITS.is_file()),
                    reason='actual evidence record/refits not present')
def test_actual_record_refit_prediction_passes_gate():
    record = json.loads(RECORD.read_text(encoding='utf-8'))
    refit = json.loads(REFITS.read_text(encoding='utf-8'))['refit30']
    res = oracle.attribute_record(record, coefficients=refit)
    assert res['prony_fit']['max_abs_eps_fit_range'] < 1e-4
    for case in res['cases'].values():
        # the measured native arrays still carry the frozen fit; only the
        # model prediction with the refit coefficients is checked here
        for row in case['surface']:
            assert row['predicted_gate_ratio'] < 0.2
