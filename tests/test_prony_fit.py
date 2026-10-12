# SPDX-License-Identifier: MPL-2.0
"""Validated finite-depth Prony decomposition (anyhydro.prony_fit)."""
from __future__ import annotations

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

from anyhydro import prony_fit as pf  # noqa: E402
from anyhydro.errors import CapabilityError  # noqa: E402

# W5 physical-field case (frozen k, h = 150 m)
K_CASE = 0.2293577981651376
H_CASE = 150.0
FROZEN = np.array([[-5.793225496607965, -3.757424621020802, -1.8872409998889776,
                    -1.7035922273520088],
                   [0.6855145366159685, -1.753020260925611, 1.977187800060,
                    0.02662042]])


def test_reference_function_matches_upstream_formula():
    kh = 6.0
    x = np.array([-0.1, 0.0, 0.7, 3.0, 12.0, 19.5])
    t = math.tanh(kh)
    sing = (1 + t) ** 2 / (1 - t * t + t / kh)
    expected = ((x + kh * t) * np.exp(x)) / (x * np.sinh(x) - kh * t * np.cosh(x)) \
        - sing / (x - kh) - 2
    np.testing.assert_allclose(pf.reference_function(x, kh), expected, rtol=1e-10)
    # asymptotic route used by Capytaine for kh >= 1e5 (and omega = inf)
    np.testing.assert_allclose(pf.reference_function(x, math.inf),
                               2 - 2 / (1 + np.exp(-2 * x)), rtol=1e-15)


def test_removable_pole_is_interpolated_smoothly():
    kh = 5.0
    x = kh + np.linspace(-0.05, 0.05, 101)
    v = pf.reference_function(x, kh)
    assert np.all(np.isfinite(v))
    assert np.max(np.abs(np.diff(v, 2))) < 1e-5


@pytest.mark.parametrize('kh', [0.101, 0.3, 5.0, 19.99])
def test_pole_evaluation_is_accurate_near_lower_limit(kh):
    # the 60-digit path agrees with the float formula just outside its window
    d = pf.POLE_WINDOW
    outside = np.array([kh - 1.5 * d, kh + 1.5 * d])
    np.testing.assert_allclose(pf.reference_function(outside, kh),
                               [pf._ref_decimal(v, kh) for v in outside], rtol=1e-8, atol=1e-12)
    # removable limit: value at the pole equals the symmetric limit from
    # points 1e-6 away (truncation O(1e-12), well inside the fit tolerance)
    limit = 0.5 * (pf._ref_decimal(kh - 1e-6, kh) + pf._ref_decimal(kh + 1e-6, kh))
    assert abs(float(pf.reference_function(np.array(kh), kh)) - limit) < 1e-9
    # no jump at the window edge
    edge = np.array([kh + d * (1 - 1e-9), kh + d * (1 + 1e-9)])
    v = pf.reference_function(edge, kh)
    assert abs(v[0] - v[1]) < 1e-8


def test_large_x_is_overflow_free_and_follows_algebraic_tail():
    kh = 3.0
    x = np.array([200.0, 1000.0, 5000.0])
    v = pf.reference_function(x, kh)
    assert np.all(np.isfinite(v))
    np.testing.assert_allclose(v * x, pf.tail_coefficient(kh), rtol=0.05)
    assert abs(pf.tail_coefficient(K_CASE * H_CASE)) < 1e-12


def test_prony_fit_recovers_exact_exponential_sum():
    lam = np.array([-0.4, -1.3, -3.1])
    amp = np.array([0.8, -0.25, 1.6])
    x = np.linspace(-0.1, 20.0, 41)
    got_lam, got_amp = pf.prony_fit(x, (amp * np.exp(np.outer(x, lam))).sum(1), 3)
    order = np.argsort(got_lam)
    np.testing.assert_allclose(got_lam[order], np.sort(lam), rtol=1e-8)
    np.testing.assert_allclose(got_amp[order], amp[np.argsort(lam)], rtol=1e-7)


@pytest.mark.parametrize('kh', [0.8, 1.5, 6.0, 19.9, K_CASE * H_CASE, 979.0, 2000.0, math.inf])
def test_validated_decomposition_meets_acceptance(kh):
    d = pf.validated_decomposition(kh)
    assert d.max_abs_error <= pf.ACCEPT_MAX_ABS_ERROR
    assert d.tail_max_abs_error <= pf.ACCEPT_MAX_ABS_ERROR
    assert all(-20 < v < 0 for v in d.lam)
    arr = d.as_array()
    assert arr.shape == (2, len(d.lam)) and np.all(np.isfinite(arr))
    # independent re-measurement on a different, finer grid
    x = np.linspace(pf.X_MIN, pf.X_MAX, 20011)
    err = pf.reference_function(x, min(kh, pf.MAX_KH)) - \
        (np.asarray(d.amp)[None, :] * np.exp(np.outer(x, d.lam))).sum(1)
    assert np.max(np.abs(err)) <= 1.05 * pf.ACCEPT_MAX_ABS_ERROR


def test_validated_decomposition_is_deterministic():
    kh = K_CASE * H_CASE
    a = pf.validated_decomposition(kh)
    pf._validated_decomposition.cache_clear()
    b = pf.validated_decomposition(kh)
    assert a.lam == b.lam and a.amp == b.amp
    assert a.record()['format'] == pf.PRONY_FIT_FORMAT


def test_frozen_case_fit_is_far_better_than_upstream_frozen_coefficients():
    kh = K_CASE * H_CASE
    x = np.linspace(pf.X_MIN, pf.X_MAX, 8001)
    ref = pf.reference_function(x, kh)
    frozen_err = np.max(np.abs(ref - (FROZEN[1][None, :] * np.exp(np.outer(x, FROZEN[0]))).sum(1)))
    assert frozen_err > 0.05
    assert pf.validated_decomposition(kh).max_abs_error < 1e-4


@pytest.mark.parametrize('kh', [0.0, 0.1, -1.0, math.nan])
def test_out_of_range_kh_is_refused(kh):
    with pytest.raises(CapabilityError):
        pf.validated_decomposition(kh)


def test_unreachable_accuracy_is_refused_not_degraded():
    with pytest.raises(CapabilityError, match='no validated Prony decomposition'):
        pf.validated_decomposition(0.15)


def test_base_import_does_not_import_capytaine():
    import subprocess
    code = ('import sys; sys.path.insert(0, %r); import anyhydro.prony_fit; '
            'print("capytaine" in sys.modules)' % str(REPO / 'src'))
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == 'False'


# ---- real Capytaine kernel (skipped when Capytaine is not installed) ----
def _surface_case():
    tri = np.array([[0.0, -0.1111111111111111, 0.0], [0.0, 0.055555555555555525, 0.0],
                    [0.0, 0.055555555555555525, -0.08333333333333333]])
    receivers = np.array([[6.1, -5.2, 0.0], [-7.3, 6.6, 0.0]])
    return tri, receivers


def test_capytaine_kernel_with_validated_fit_satisfies_free_surface_condition():
    cpt = pytest.importorskip('capytaine')
    from capytaine.green_functions.delhommeau import Delhommeau
    import check_finite_depth_john_oracle as oracle
    tri, receivers = _surface_case()
    mesh = cpt.Mesh(vertices=tri, faces=np.array([[0, 1, 2, 2]])).with_quadrature('Gauss-Legendre 2')
    opts = dict(tabulation_nr=0, tabulation_nz=0, tabulation_cache_dir=None,
                gf_singularities='low_freq', floating_point_precision='float64')
    nu = K_CASE * math.tanh(K_CASE * H_CASE)

    def robin_ratios(gf):
        S, K = gf.evaluate(receivers, mesh, free_surface=0.0, water_depth=H_CASE,
                           wavenumber=K_CASE, adjoint_double_layer=True,
                           early_dot_product=False, diagonal_term_in_double_layer=False)
        s, kz = np.real(S)[:, 0], np.real(K)[2, :, 0]
        res = np.abs(kz - nu * s)
        return res / (1e-9 + 1e-4 * np.maximum(np.abs(kz), np.abs(nu * s))), s

    class Frozen(Delhommeau):
        def find_best_exponential_decomposition(self, kh, method=None):
            return FROZEN

    frozen_ratio, _ = robin_ratios(Frozen(**opts))
    ratio, s = robin_ratios(pf.validated_delhommeau(cpt, **opts))
    assert np.all(frozen_ratio > 2.0)           # the recorded W5 failure
    assert np.all(ratio < 0.5)                  # unchanged criterion now met
    kern = oracle.FiniteDepthKernel(K_CASE, H_CASE)
    qp, qw = mesh.quadrature_points
    exact = -1 / (4 * math.pi) * (kern.green(receivers, qp[0])[0] * qw[0][None, :]).sum(1)
    assert np.max(np.abs(s - exact) / np.abs(exact)) < 1e-4


def test_validated_green_function_identity_is_distinct():
    cpt = pytest.importorskip('capytaine')
    from capytaine.green_functions.delhommeau import Delhommeau
    gf = pf.validated_delhommeau(cpt)
    assert gf.exportable_settings['finite_depth_prony_decomposition'] == pf.PRONY_FIT_FORMAT
    assert hash(gf) != hash(Delhommeau())
    np.testing.assert_array_equal(gf.find_best_exponential_decomposition(K_CASE * H_CASE),
                                  pf.validated_decomposition(K_CASE * H_CASE).as_array())


def test_validated_green_function_pickles_for_parallel_solves():
    # joblib (Capytaine n_jobs > 1) serializes with cloudpickle; the Fortran
    # core module attribute makes plain pickle fail for upstream Delhommeau too
    cpt = pytest.importorskip('capytaine')
    cloudpickle = pytest.importorskip('cloudpickle')
    import pickle
    gf = pf.validated_delhommeau(cpt)
    clone = pickle.loads(cloudpickle.dumps(gf))
    assert type(clone).__name__ == 'ValidatedPronyDelhommeau'
    assert clone.exportable_settings == gf.exportable_settings
    assert hash(clone) == hash(gf)
    np.testing.assert_array_equal(clone.find_best_exponential_decomposition(34.4),
                                  pf.validated_decomposition(34.4).as_array())
