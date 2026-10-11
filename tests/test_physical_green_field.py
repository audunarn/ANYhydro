# SPDX-License-Identifier: MPL-2.0
"""Real pytest suite for the physical Green field discriminator.

NO native Green function execution anywhere: the native provider is a
synthetic independent analytic oracle (exact outgoing Hankel wave field
plus a real BC-consistent evanescent mode standing in for the real
Rankine/image part).  The pole ALWAYS comes from the unchanged public
``anyhydro.panel_integration_candidate.imaginary_pole``.

Independent oracles (no mirror-implementation-only assertions):
- dispersion factor vs a finite difference of k*tanh(kh) in k,
- the candidate's WRONG dispersion formula is detected as different,
- finite-depth mode boundary conditions (Robin at z=0, no-flux at
  z=-h, Z''=k^2 Z) via independent finite differences,
- far-field oracle vs an exact scipy hankel1 field (plane wave mode),
- source phase shift exp(-i k d.y) under a rigid source translation,
- caller refusals (typed, raised before any provider call),
- CLI guards (fresh output, resolved-containment location, symlink,
  canonically recomputed case hash, snapshot-array identity, fixed
  caps, root-only native authorization, input immutability),
- a MOCK-POSITIVE main regression: with the public capytaine modules
  and ``ArtifactBoundPanelProfile`` replaced by mocks (reference bundle
  as a real DIRECTORY fixture), the actual-route CLI reaches
  ``gf.evaluate`` with 2 calls x 47 receivers, 376 source-quadrature
  interactions, canonical ``[[0, 1, 2, 2]]`` faces, the same
  Gauss-Legendre 2 weights, ONE shared constructor and an OBSERVED
  coefficient-override count.  No native Green code runs anywhere.
"""
import hashlib
import json
import math
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from scipy.special import hankel1, k0 as K0, k1 as K1

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO / 'src'), str(REPO / 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

import check_physical_green_field as tool  # noqa: E402
from anyhydro.errors import CapabilityError, GeometryValidationError  # noqa: E402
from anyhydro.panel_integration_candidate import (  # noqa: E402
    imaginary_pole, triangle_quadrature)

K = tool.FROZEN_K
DEPTH = tool.FROZEN_WATER_DEPTH
OMEGA = tool.FROZEN_OMEGA
GRAVITY = tool.FROZEN_GRAVITY


# ---- synthetic source geometry (deterministic, tiny, not the actual case)
def synthetic_source_triangles():
    """A vertical hull triangle and a horizontal lid at z=-0.1."""
    hull = np.array([[0.0, 0.0, -0.5], [0.0, 1.0, -0.5],
                     [0.0, 0.4, -1.5]])
    lid = np.array([[-0.6, -0.4, -0.1], [0.1, 0.8, -0.1],
                    [0.7, -0.2, -0.1]])  # winding gives normal (0,0,-1)
    return hull, lid


def quad(tri):
    pts, w, _n = triangle_quadrature(tri[None, :, :])
    return pts[0], w[0]


# ---- independent analytic oracle provider ------------------------------
def evanescent_mu(k, water_depth):
    """mu solving mu*tan(mu*h) = -nu (evanescent mode, mu*h in (pi/2,pi))."""
    nu = tool.robin_nu(k, water_depth)
    lo, hi = math.pi / 2 + 1e-9, math.pi - 1e-9
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * math.tan(mid) + nu * water_depth < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi) / water_depth


class SyntheticNativeProvider:
    """Independent oracle: native real part = Re(Phi_wave) + amp*Phi_ev.

    Phi_wave = -i k/(2D) Z(x.z) sum_q w_q Z(y_q.z) H0^(1)(k r_q)
    Phi_ev   = cos(mu(z+h))/cos(mu h) * K0(mu r)   (real, BC-consistent)

    The composite route then satisfies
        S_comp = native_S + 1j*pole = Phi_wave + amp*Phi_ev  EXACTLY
    because pole = Im(Phi_wave) by construction of the public pole.
    """

    def __init__(self, source_points, source_weights, k, water_depth,
                 amp=1.0):
        self.y = np.asarray(source_points, dtype=float)
        self.w = np.asarray(source_weights, dtype=float)
        self.k = k
        self.h = water_depth
        self.amp = amp
        self.mu = evanescent_mu(k, water_depth)
        self.calls = 0

    def _wave(self, x):
        k, h = self.k, self.h
        Zx, dZx = tool.depth_mode(x[:, 2], k, h)
        c = -1j * k / (2.0 * tool.dispersion_factor(k, h))
        nP = self.y.shape[0]
        acc = np.zeros((len(x), nP), dtype=complex)
        grad = np.zeros((3, len(x), nP), dtype=complex)
        for p in range(nP):
            for qi in range(self.y.shape[1]):
                yq = self.y[p, qi]
                Zyq = np.cosh(k * (yq[2] + h)) / math.cosh(k * h)
                base = c * self.w[p, qi] * Zyq
                dd = x[:, :2] - yq[:2]
                r = np.linalg.norm(dd, axis=1)
                H0 = hankel1(0, k * r)
                H1 = hankel1(1, k * r)
                acc[:, p] += base * Zx * H0
                for i in (0, 1):
                    grad[i, :, p] += base * Zx * (-k * H1) * dd[:, i] \
                        / np.maximum(r, 1e-300)
                grad[2, :, p] += base * dZx * H0
        return acc, grad

    def _ev(self, x):
        mu, h = self.mu, self.h
        z = x[:, 2]
        Zev = np.cos(mu * (z + h)) / np.cos(mu * h)
        dZev = -mu * np.sin(mu * (z + h)) / np.cos(mu * h)
        r = np.hypot(x[:, 0], x[:, 1])
        Kk = K0(mu * r)
        Kk1 = K1(mu * r)
        nP = self.y.shape[0]
        phi = np.repeat((self.amp * Zev * Kk)[:, None], nP, axis=1)
        grad = np.zeros((3, len(x), nP))
        for i in (0, 1):
            grad[i] = np.repeat((self.amp * Zev * (-mu * Kk1)
                                * x[:, i] / np.maximum(r, 1e-300)
                                )[:, None], nP, axis=1)
        grad[2] = np.repeat((self.amp * dZev * Kk)[:, None], nP, axis=1)
        return phi, grad

    def exact_total(self, x):
        """Independent oracle: the exact TOTAL composite field (summed)."""
        wv, gwv = self._wave(x)
        ev, gev = self._ev(x)
        return (wv.sum(axis=1) + ev.sum(axis=1),
                gwv.sum(axis=2) + gev.sum(axis=2))

    def __call__(self, x):
        self.calls += 1
        wv, gwv = self._wave(x)
        ev, gev = self._ev(x)
        return wv.real + ev, gwv.real + gev


def run_synthetic_case(tri, amp=1.0):
    pts, w = quad(tri)
    provider = SyntheticNativeProvider(pts[None], w[None], K, DEPTH,
                                       amp=amp)
    budget = tool.NativeBudget()
    out = tool.evaluate_diagnostics(provider, pts[None], w[None], k=K,
                                     water_depth=DEPTH, budget=budget,
                                     case_label='synthetic')
    return provider, budget, out, pts[None], w[None]


# The evanescent stand-in amplitude is an arbitrary fixture choice, not
# physics: the far/Sommerfeld gates compare the TOTAL field against the
# propagating-wave asymptote, so the fixture's non-wave component must
# be small at kR=100/200 (unlike the ACTUAL real part, whose leading 1/R
# term is cancelled by the free-surface image and decays algebraically;
# that assessment belongs to root - see
# docs/PHYSICAL_GREEN_FIELD_DISCRIMINATOR.md).
SYNTHETIC_EVANESENT_AMP = 0.001


# ---- dispersion factor --------------------------------------------------
@pytest.mark.parametrize('kh', [0.3, 1.0, 2.0, 5.0])
def test_dispersion_factor_matches_derivative_of_k_tanh(kh):
    """D must equal d/dk [k*tanh(kh)] with h fixed (moderate kh)."""
    h = DEPTH
    k = kh / h
    eps = 1e-6 * k
    f = lambda kk: kk * math.tanh(kk * h)  # noqa: E731
    fd = (f(k + eps) - f(k - eps)) / (2 * eps)
    assert tool.dispersion_factor(k, h) == pytest.approx(fd, rel=1e-5)


def test_dispersion_factor_rejects_candidate_wrong_formula():
    """The candidate bug th + kh*(1-th)**2 must be detected as wrong."""
    kh = 1.0
    th = math.tanh(kh)
    correct = th + kh * (1 - th * th)
    wrong = th + kh * (1 - th) ** 2
    assert tool.dispersion_factor(1.0, 1.0) == pytest.approx(correct)
    assert abs(tool.dispersion_factor(1.0, 1.0) - wrong) > 1e-2


def test_robin_nu_and_wavenumber_match_frozen_dispersion():
    """k*tanh(kh) = omega^2/g and the solved k is the frozen k."""
    assert tool.robin_nu(K, DEPTH) == pytest.approx(OMEGA ** 2 / GRAVITY,
                                                    rel=1e-12)
    k_solved = tool.solve_wavenumber(OMEGA, GRAVITY, DEPTH)
    assert k_solved == pytest.approx(K, rel=1e-12)
    assert abs(k_solved - K) <= 1e-15


def test_frozen_k_exact_hex_policy():
    """Exact frozen k only after independent dispersion consistency.

    The independent bisection solve may differ from the frozen literal
    by the known 1 ULP; the frozen k itself must satisfy the dispersion
    relation, and its hex must be EXACTLY the frozen hex (the bound
    constructor enforces the same against the reference manifest).
    """
    k_solved = tool.solve_wavenumber(OMEGA, GRAVITY, DEPTH)
    ulp = abs(k_solved - K) / math.ulp(K)
    assert ulp <= tool.K_SOLVE_MAX_ULP
    target = OMEGA ** 2 / GRAVITY
    residual = abs(K * math.tanh(K * DEPTH) - target)
    assert residual <= 16.0 * math.ulp(target)
    assert K.hex() == tool.FROZEN_K_HEX


# ---- finite-depth mode ---------------------------------------------------
def test_depth_mode_boundary_conditions():
    """dZ/dz(0) = nu*Z(0); dZ/dz(-h) = 0; Z'' = k^2 Z (independent FD)."""
    nu = tool.robin_nu(K, DEPTH)
    Z0, dZ0 = tool.depth_mode(np.array([0.0]), K, DEPTH)
    assert dZ0[0] == pytest.approx(nu * Z0[0], rel=1e-12)
    Zh, dZh = tool.depth_mode(np.array([-DEPTH]), K, DEPTH)
    assert dZh[0] == pytest.approx(0.0, abs=1e-12)
    z = np.array([-0.3 * DEPTH])
    eps = 1e-3
    Zp, _ = tool.depth_mode(z + eps, K, DEPTH)
    Zm, _ = tool.depth_mode(z - eps, K, DEPTH)
    Z0v, _ = tool.depth_mode(z, K, DEPTH)
    second = (Zp[0] - 2 * Z0v[0] + Zm[0]) / eps ** 2
    assert second == pytest.approx(K * K * Z0v[0], rel=1e-6)


# ---- far-field oracle ----------------------------------------------------
def test_far_field_matches_exact_hankel_plane_mode():
    """Far oracle vs an exact scipy hankel1 field (independent oracle)."""
    tri, _ = synthetic_source_triangles()
    pts, w = quad(tri)
    far = []
    for kr in (100.0, 200.0):
        for theta in (0.3, 1.9):
            far.append([kr / K * math.cos(theta),
                        kr / K * math.sin(theta), -0.05 * DEPTH])
    far = np.asarray(far)
    phi_far, grad_far = tool.far_field_potential(far, pts[None], w[None],
                                                 k=K, water_depth=DEPTH)
    # exact field, computed independently here
    Zx, dZx = tool.depth_mode(far[:, 2], K, DEPTH)
    c = -1j * K / (2.0 * tool.dispersion_factor(K, DEPTH))
    phi_exact = np.zeros(len(far), dtype=complex)
    for qi in range(4):
        Zyq = np.cosh(K * (pts[qi, 2] + DEPTH)) / math.cosh(K * DEPTH)
        r = np.hypot(far[:, 0] - pts[qi, 0], far[:, 1] - pts[qi, 1])
        phi_exact += c * w[qi] * Zyq * Zx * hankel1(0, K * r)
    rel = np.abs(phi_far - phi_exact) / np.abs(phi_exact)
    assert rel[2:].max() < 0.03      # kR = 200 (gate target)
    assert rel[:2].max() < 0.06     # kR = 100 (recorded, looser)
    # Sommerfeld character of the oracle gradient itself
    theta = np.arctan2(far[:, 1], far[:, 0])
    radial = grad_far[0] * np.cos(theta) + grad_far[1] * np.sin(theta)
    som = np.abs(radial - 1j * K * phi_far)
    scale = np.maximum(np.maximum(K * np.abs(phi_far), np.abs(radial)),
                       1e-12)
    assert (som / scale).max() < 0.03


def test_far_field_source_phase_shift():
    """Rigid source translation multiplies Phi_far by exp(-i k d.dx)."""
    tri, _ = synthetic_source_triangles()
    pts, w = quad(tri)
    far = np.array([[200.0 / K * math.cos(0.7),
                     200.0 / K * math.sin(0.7), -0.05 * DEPTH]])
    phi0, _ = tool.far_field_potential(far, pts[None], w[None], k=K,
                                       water_depth=DEPTH)
    delta = 0.37
    shifted = pts.copy()
    shifted[:, 0] += delta
    phi1, _ = tool.far_field_potential(far, shifted[None], w[None], k=K,
                                        water_depth=DEPTH)
    factor = np.exp(-1j * K * delta * math.cos(0.7))
    assert np.abs(phi1[0] - phi0[0] * factor) <= 1e-9 * np.abs(phi0[0])


# ---- full synthetic diagnostics -----------------------------------------
def test_synthetic_composite_matches_independent_oracle():
    tri, _ = synthetic_source_triangles()
    provider, budget, out, y, w = run_synthetic_case(
        tri, amp=SYNTHETIC_EVANESENT_AMP)
    rec = np.asarray(out['receivers_saved'])
    exact_S, exact_K = provider.exact_total(rec)
    pole_S, pole_K = imaginary_pole(rec, y, w, k=K, water_depth=DEPTH)
    S_comp, K_comp = tool.composite_assembly(
        np.asarray(provider(rec)[0]), np.asarray(provider(rec)[1]),
        pole_S, pole_K)
    # per-panel gradient shape (3, receivers, panels); total sums axis 2
    assert np.asarray(provider(rec)[1]).shape == (3, len(rec), 1)
    assert K_comp.shape == (3, len(rec), 1)
    assert np.abs(S_comp.sum(axis=1) - exact_S).max() < 1e-10
    assert np.abs(K_comp.sum(axis=2) - exact_K).max() < 1e-10
    # no doubled jump off-surface: composite total equals the exact field
    assert out['composite_far_gate_kr200'] is True
    assert out['composite_sommerfeld_gate'] is True
    assert all(out['composite_pde_gates'].values())
    # native raw diagnostics present and clearly named (not the route)
    assert 'native_raw_far_rel_error_by_kr' in out
    assert 'composite_far_rel_error_by_kr' in out


def test_synthetic_plan_counts_and_actual_calls():
    tri, _ = synthetic_source_triangles()
    provider, budget, out, _y, _w = run_synthetic_case(tri)
    plan = out['plan_counts']
    assert plan['n_receivers'] == 47
    assert plan['n_batches'] == 1
    assert plan['max_batch_size'] <= 96
    assert plan['source_quad_interactions'] == 47 * 4
    assert plan['constructor_calls'] == 1
    assert plan['coefficient_overrides'] == 1
    assert plan['probe_calls'] == 0
    # enforce ACTUAL counts, not merely planned
    assert provider.calls == plan['provider_calls_planned'] == 1
    assert out['actual_provider_calls'] == 1
    assert budget.actual_provider_calls == 1
    assert budget.per_case_actual == {'synthetic': 1}
    assert 'unmeasured' in budget.as_dict()['native_e1_primitive_calls']


def test_both_fd_steps_must_pass_no_picked_step():
    """A perturbation on the SECOND FD step only must fail the gate."""
    tri, _ = synthetic_source_triangles()
    pts, w = quad(tri)
    base = SyntheticNativeProvider(pts[None], w[None], K, DEPTH)
    budget0 = tool.NativeBudget()
    clean = tool.evaluate_diagnostics(base, pts[None], w[None], k=K,
                                      water_depth=DEPTH, budget=budget0,
                                      case_label='clean')
    assert clean['composite_pde_gates']['laplace'] is True

    layout = tool.build_receiver_set(K, DEPTH, pts[None])
    # receivers of FD step index 1 (si=1), direction 0, both signs
    bad_plus, _bad_minus = layout.fd_index[(0, 1, 0)]

    # perturb the exact gradient at the step-1 shifted receiver only
    class PerturbedGrad:
        def __init__(self):
            self.calls = 0

        def __call__(self, x):
            self.calls += 1
            S, Kk = base(x)
            S = np.array(S)
            Kk = np.array(Kk)
            Kk[0, bad_plus, 0] += 0.1 * abs(S[bad_plus, 0]) + 1e-6
            return S, Kk

    budget1 = tool.NativeBudget()
    out = tool.evaluate_diagnostics(PerturbedGrad(), pts[None], w[None],
                                    k=K, water_depth=DEPTH, budget=budget1,
                                    case_label='perturbed')
    lap = out['composite_laplace_residual']
    lsc = out['composite_laplace_scale']
    tol = lambda r, s: r <= tool.LAPLACE_REL_TOL * s + tool.LAPLACE_ABS_TOL  # noqa: E731
    # step 0 (entries 0,2,4) still passes; step 1 (entries 1,3,5) fails
    assert all(tol(lap[j * 2 + 0], lsc[j * 2 + 0]) for j in range(3))
    assert not all(tol(lap[j * 2 + 1], lsc[j * 2 + 1]) for j in range(3))
    assert out['composite_pde_gates']['laplace'] is False


def test_gradient_per_panel_shape_and_sum_two_panels():
    """Per-panel composite fields sum to the exact two-source total."""
    hull, lid = synthetic_source_triangles()
    pts = np.stack([quad(hull)[0], quad(lid)[0]])
    w = np.stack([quad(hull)[1], quad(lid)[1]])
    provider = SyntheticNativeProvider(pts, w, K, DEPTH)
    layout = tool.build_receiver_set(K, DEPTH, pts)
    x = layout.receivers
    S, Kk = provider(x)
    pole_S, pole_K = imaginary_pole(x, pts, w, k=K, water_depth=DEPTH)
    S_comp, K_comp = tool.composite_assembly(S, Kk, pole_S, pole_K)
    exact_S, exact_K = provider.exact_total(x)
    assert S_comp.shape == (len(x), 2)
    assert K_comp.shape == (3, len(x), 2)
    assert np.abs(S_comp.sum(axis=1) - exact_S).max() < 1e-10
    assert np.abs(K_comp.sum(axis=2) - exact_K).max() < 1e-10


# ---- caller refusals (typed, before any provider call) ------------------
def test_native_unavailable_is_typed_refusal():
    tri, _ = synthetic_source_triangles()
    pts, w = quad(tri)
    with pytest.raises(tool.NativeUnavailableError):
        tool.evaluate_diagnostics(None, pts[None], w[None], k=K,
                                  water_depth=DEPTH,
                                  budget=tool.NativeBudget(),
                                  case_label='refusal')
    assert issubclass(tool.NativeUnavailableError, CapabilityError)


class CountingProvider:
    def __init__(self):
        self.calls = 0

    def __call__(self, x):
        self.calls += 1
        n = len(x)
        return (np.zeros((n, 1)), np.zeros((3, n, 1)))


@pytest.mark.parametrize('kwargs', [
    dict(k=-1.0), dict(k=0.0), dict(k=float('nan')),
    dict(water_depth=-5.0), dict(water_depth=0.0),
    dict(water_depth=float('inf')),
])
def test_invalid_params_refused_before_provider_call(kwargs):
    tri, _ = synthetic_source_triangles()
    pts, w = quad(tri)
    provider = CountingProvider()
    args = dict(k=K, water_depth=DEPTH)
    args.update(kwargs)
    with pytest.raises(GeometryValidationError):
        tool.evaluate_diagnostics(provider, pts[None], w[None],
                                  budget=tool.NativeBudget(),
                                  case_label='invalid', **args)
    assert provider.calls == 0


@pytest.mark.parametrize('points,weights', [
    (np.zeros((1, 4, 4)), np.ones((1, 4))),
    (np.zeros((1, 4, 3)), np.ones((1, 3))),
    (np.full((1, 4, 3), np.nan), np.ones((1, 4))),
    (np.zeros((1, 4, 3)), np.full((1, 4), np.nan)),
    (np.zeros((1, 4, 3)), np.zeros((1, 4))),
    (np.zeros((2, 4, 3)), np.ones((2, 4))),  # >1 source panel per call
])
def test_invalid_source_arrays_refused(points, weights):
    provider = CountingProvider()
    with pytest.raises((GeometryValidationError, CapabilityError)):
        tool.evaluate_diagnostics(provider, points, weights, k=K,
                                  water_depth=DEPTH,
                                  budget=tool.NativeBudget(),
                                  case_label='invalid')
    assert provider.calls == 0


def test_nonfinite_provider_output_refused():
    tri, _ = synthetic_source_triangles()
    pts, w = quad(tri)

    class NanProvider:
        def __call__(self, x):
            n = len(x)
            S = np.zeros((n, 1))
            S[0, 0] = np.nan
            return S, np.zeros((3, n, 1))

    with pytest.raises(GeometryValidationError):
        tool.evaluate_diagnostics(NanProvider(), pts[None], w[None], k=K,
                                  water_depth=DEPTH,
                                  budget=tool.NativeBudget(),
                                  case_label='nan')


def test_provider_shape_mismatch_refused():
    tri, _ = synthetic_source_triangles()
    pts, w = quad(tri)

    class BadShape:
        def __call__(self, x):
            n = len(x)
            return np.zeros((n, 2)), np.zeros((3, n, 1))

    with pytest.raises(GeometryValidationError):
        tool.evaluate_diagnostics(BadShape(), pts[None], w[None], k=K,
                                  water_depth=DEPTH,
                                  budget=tool.NativeBudget(),
                                  case_label='shape')


def test_check_micro_probe_refused_in_plan():
    tri, _ = synthetic_source_triangles()
    pts, w = quad(tri)
    budget = tool.NativeBudget()
    with pytest.raises(CapabilityError):
        budget.plan(np.zeros((5, 3)), pts[None], w[None],
                    case_label='probes', probes=('check_micro',))


def test_batch_budget_refused():
    tri, _ = synthetic_source_triangles()
    pts, w = quad(tri)
    budget = tool.NativeBudget()
    with pytest.raises(CapabilityError):
        budget.plan(np.zeros((24 * 96 + 1, 3)), pts[None], w[None],
                    case_label='toobig')


def test_cumulative_batch_budget_refused_across_cases():
    tri, _ = synthetic_source_triangles()
    pts, w = quad(tri)
    budget = tool.NativeBudget(max_batches=1)
    budget.plan(np.zeros((47, 3)), pts[None], w[None], case_label='a')
    with pytest.raises(CapabilityError):
        budget.plan(np.zeros((47, 3)), pts[None], w[None], case_label='b')


def test_actual_calls_mismatch_refused():
    tri, _ = synthetic_source_triangles()
    pts, w = quad(tri)
    budget = tool.NativeBudget()
    plan = budget.plan(np.zeros((47, 3)), pts[None], w[None],
                       case_label='a')
    with pytest.raises(CapabilityError):
        budget.record_actual_calls('a', plan, actual=2)


# ---- filesystem guards --------------------------------------------------
def test_fresh_output_and_location_and_symlink_guards(tmp_path):
    out = tmp_path / 'out.json'
    out.write_text('x')
    with pytest.raises(tool.GuardRefused):
        tool.guard_fresh_output(out)
    with pytest.raises(tool.GuardRefused):
        tool.guard_output_location(REPO / 'src' / 'x.json')
    with pytest.raises(tool.GuardRefused):
        tool.guard_output_location(
            REPO / '.development-evidence' / 'other' / 'x.json')
    # task scope is allowed (resolved containment, both platform forms)
    for scope in tool._task_evidence_scopes():
        tool.guard_output_location(scope / 'out.json',
                                   task_evidence=tool._task_evidence_scopes())
    # symlink refusal (patched: OS symlink support is not assumed)
    real = tmp_path / 'real.json'
    real.write_text('x')
    orig = Path.is_symlink

    def fake_is_symlink(self):
        return self.name == 'real.json'

    Path.is_symlink = fake_is_symlink
    try:
        with pytest.raises(tool.GuardRefused):
            tool.guard_symlinks(real)
    finally:
        Path.is_symlink = orig


def test_guard_containment_is_resolved_not_string_prefix(tmp_path):
    """No str.startswith containment, no ``..`` escape, no prefix bug."""
    case_dir = tmp_path / 'case'
    case_dir.mkdir()
    # inside the forbidden root -> refused
    with pytest.raises(tool.GuardRefused):
        tool.guard_output_location(case_dir / 'sub' / 'x.json',
                                   forbidden_roots=(case_dir,))
    # ``..`` collapse back into the forbidden root -> refused
    with pytest.raises(tool.GuardRefused):
        tool.guard_output_location(
            tmp_path / 'case' / '..' / 'case' / 'x.json',
            forbidden_roots=(case_dir,))
    # a string-PREFIX sibling is NOT inside -> allowed (containment).
    # tmp_path can lie inside the harness development-evidence tree
    # (the metered harness points TEMP at the task attempts dir), so
    # the legitimate sibling is explicitly inside the allowed scope.
    tool.guard_output_location(
        tmp_path / (case_dir.name + '-evil') / 'x.json',
        forbidden_roots=(case_dir,), task_evidence=tmp_path)
    # prefix sibling of the allowed evidence scope -> refused
    scopes = tool._task_evidence_scopes()
    with pytest.raises(tool.GuardRefused):
        tool.guard_output_location(
            scopes[0].parent / (tool.TASK_NAME + '-evil') / 'x.json',
            task_evidence=scopes)
    # ``..`` escape out of the allowed scope into a sibling task -> refused
    with pytest.raises(tool.GuardRefused):
        tool.guard_output_location(
            scopes[0] / '..' / 'other-task' / 'x.json',
            task_evidence=scopes)


def test_reference_directory_fingerprint(tmp_path):
    """Reference bundle files are flattened to sha256 entries; no writes."""
    ref = tmp_path / 'reference'
    ref.mkdir()
    (ref / 'manifest.json').write_text('{"k_hex": "0x1p-3"}',
                                       encoding='utf-8')
    np.savez(ref / 'input.npz', dummy=np.zeros(3))
    sub = ref / 'nested'
    sub.mkdir()
    (sub / 'reference.npz').write_bytes(b'reference-bytes')
    before = tool.fingerprint_reference_directory(ref)
    assert set(before) == {'manifest.json', 'input.npz',
                          'nested/reference.npz'}
    # unchanged -> identical; a regular-file reference is refused
    assert tool.fingerprint_reference_directory(ref) == before
    notadir = tmp_path / 'notadir'
    notadir.write_text('{}', encoding='utf-8')
    with pytest.raises(tool.GuardRefused):
        tool.fingerprint_reference_directory(notadir)


def test_input_immutability_detection(tmp_path):
    f = tmp_path / 'input.bin'
    f.write_bytes(b'data')
    before = tool.input_fingerprint([f])
    assert tool.check_input_immutability(before, [f]) is True
    f.write_bytes(b'data-mutated')
    assert tool.check_input_immutability(before, [f]) is False


# ---- CLI ----------------------------------------------------------------
# mock NEW (rectangle-envelope) native float64 core identity for the
# mock-positive actual-route regression; admitted ONLY via an explicit
# envelope-table patch in that test (never by name/generic capability).
MOCK_NEW_CORE_SHA = 'e1' + '0' * 62


def make_case_dir(tmp_path, case_hash=None, stale_identity=False,
                  core_sha=None):
    """Synthetic case with a CANONICALLY consistent case.json.

    The case hash is computed with the producer's convention (sorted-key
    compact JSON of the identity) and ``snapshot_arrays`` pins the
    input.npz arrays, exactly like the actual frozen case.  With
    ``stale_identity`` the stored hash is kept while a field is changed,
    proving the canonical recompute catches tampering.  ``core_sha``
    pins the binding's float64 native core identity (default: the mock
    NEW rectangle-envelope core admitted by the mock-positive test).
    """
    if core_sha is None:
        core_sha = MOCK_NEW_CORE_SHA
    hull, lid = synthetic_source_triangles()
    vertices = np.vstack([hull, lid])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    points = np.stack([quad(hull)[0], quad(lid)[0]])
    weights = np.stack([quad(hull)[1], quad(lid)[1]])
    d = tmp_path / 'case'
    d.mkdir(parents=True)
    np.savez(d / 'input.npz', vertices=vertices, faces=faces,
             points=points, weights=weights)
    arrays = dict(vertices=vertices, faces=faces, points=points,
                  weights=weights)
    identity = dict(
        version='test', omega=OMEGA, water_depth=DEPTH, gravity=GRAVITY,
        rho=1025.0,
        snapshot_arrays={
            name: dict(shape=values.shape, dtype=str(values.dtype),
                       sha256=hashlib.sha256(
                           np.ascontiguousarray(values).tobytes()
                       ).hexdigest())
            for name, values in arrays.items()})
    stored = case_hash or tool.canonical_case_hash(identity)
    if stale_identity:
        identity['omega'] = 999.0     # stored hash must now be stale
    identity['case_hash'] = stored
    (d / 'case.json').write_text(json.dumps(identity, indent=2),
                                 encoding='utf-8')
    # reference bundle is a DIRECTORY (ArtifactBoundPanelProfile.load
    # contract), flattened to regular files like the real bundle
    ref = d / 'reference'
    ref.mkdir()
    (ref / 'manifest.json').write_text(
        json.dumps({'k_hex': tool.FROZEN_K_HEX}), encoding='utf-8')
    np.savez(ref / 'input.npz', dummy=np.zeros(3))
    np.savez(ref / 'reference.npz', dummy=np.zeros(3))
    (d / 'binding').write_text(
        json.dumps({'format': 'mock-binding',
                    'reference_manifest_sha256': '0' * 64,
                    'evaluator': {
                        'external_identity': 'mock',
                        'native_extension_sha256': {
                            'capytaine/green_functions/'
                            'Delhommeau_float64.cpython-312-'
                            'x86_64-linux-gnu.so': core_sha}}}),
        encoding='utf-8')
    (d / 'sidecar').write_text('{}', encoding='utf-8')
    return d


def stored_case_hash(d):
    """The synthetic case.json's OWN stored case hash.

    For the default fixture this is the canonical hash of the synthetic
    identity; the negative fixtures keep a wrong/stale stored hash and
    must STILL be refused by the canonical recompute inside
    ``load_case`` (the product guard is not weakened).
    """
    return json.loads(
        (d / 'case.json').read_text(encoding='utf-8'))['case_hash']


def cli_args(d, output, extra=()):
    # --expected-case-hash defaults to the ACTUAL frozen case hash; the
    # synthetic fixture must pin its OWN case.json hash instead.
    return ['--case-input', str(d / 'input.npz'),
            '--case-json', str(d / 'case.json'),
            '--reference', str(d / 'reference'),
            '--binding', str(d / 'binding'),
            '--sidecar', str(d / 'sidecar'),
            '--output', str(output),
            '--expected-case-hash', stored_case_hash(d),
            '--hull-face', '0', '--lid-face', '1'] + list(extra)


def test_cli_plan_only(tmp_path, capsys):
    d = make_case_dir(tmp_path)
    out = tmp_path / 'result.json'
    code = tool.main(cli_args(d, out, ['--plan-only']))
    assert code == 0
    record = json.loads(out.read_text(encoding='utf-8'))
    assert record['status'] == 'plan_only'
    assert record['qualification'] is False
    # exact frozen k policy (independent solve + dispersion consistency)
    assert record['case']['k_hex'] == tool.FROZEN_K_HEX
    assert record['case']['k_hex_exact'] is True
    assert record['case']['k_solved_ulp_difference'] \
        <= tool.K_SOLVE_MAX_ULP
    assert record['case']['params_match_frozen'] is True
    assert record['case']['case_hash_canonically_recomputed'] is True
    assert abs(record['case']['k'] - tool.FROZEN_K) <= 1e-15
    # input.npz verified against the case identity snapshot arrays
    assert set(record['input_array_hashes']) == \
        {'vertices', 'faces', 'points', 'weights'}
    # planned vs actual constructor contract (root repair 2026-10-11):
    # plan_only planned ONE shared gf constructor but OBSERVED none
    assert record['constructor_calls_planned'] == 1
    assert record['constructor_calls_actual'] == 0
    assert 'constructor_calls_attempted' not in record
    assert record['plans']['hull']['constructor_calls'] == 1
    assert record['plans']['lid']['constructor_calls'] == 0
    assert set(record['plans']) == {'hull', 'lid'}
    for plan in record['plans'].values():
        assert plan['n_receivers'] == 47
        assert plan['n_batches'] == 1
        assert plan['source_quad_interactions'] == 47 * 4
        assert plan['coefficient_overrides'] == 1
        assert plan['probe_calls'] == 0
    assert record['budget']['planned_batches'] == 2
    assert record['budget']['actual_provider_calls'] == 0
    assert 'unmeasured' in record['budget']['native_e1_primitive_calls']
    # runtime measured (not just env vars); trust scope stated
    assert 'cpu_count' in record['metadata']['runtime']
    assert stored_case_hash(d) in record['trust_scope']
    # progress/call counts printed BEFORE any evaluation
    printed = capsys.readouterr().out.splitlines()
    assert any('"status": "planned"' in line for line in printed)
    # sources verified deterministically; translation recorded honestly
    assert record['sources']['hull']['face_index'] == 0
    assert record['sources']['lid']['face_index'] == 1
    assert record['sources']['hull']['quadrature_receipt'] is True
    assert record['sources']['lid']['quadrature_receipt'] is True
    assert abs(record['sources']['lid']['z_mean'] - tool.LID_Z) < 1e-12
    assert len(record['sources']['hull']['translation_delta_xy']) == 2
    assert 'no global coupled-case claim' in \
        record['sources']['hull']['translation_note']


def test_cli_fresh_output_refusal(tmp_path):
    d = make_case_dir(tmp_path)
    out = tmp_path / 'result.json'
    assert tool.main(cli_args(d, out, ['--plan-only'])) == 0
    code = tool.main(cli_args(d, out, ['--plan-only']))
    assert code == tool.EXIT_GUARD_REFUSED
    assert json.loads(out.read_text(encoding='utf-8'))['status'] \
        == 'plan_only'


def test_cli_output_location_refusal(tmp_path):
    d = make_case_dir(tmp_path)
    out = REPO / 'src' / 'should_not_exist.json'
    code = tool.main(cli_args(d, out, ['--plan-only']))
    assert code == tool.EXIT_GUARD_REFUSED
    assert not out.exists()


def test_cli_case_hash_mismatch_refusal(tmp_path):
    d = make_case_dir(tmp_path, case_hash='deadbeef')
    out = tmp_path / 'result.json'
    code = tool.main(cli_args(d, out, ['--plan-only']))
    assert code == tool.EXIT_GUARD_REFUSED
    assert not out.exists()


def test_cli_native_execution_not_authorized(tmp_path):
    """Without the root-only flag the CLI refuses before any native use."""
    d = make_case_dir(tmp_path)
    out = tmp_path / 'result.json'
    code = tool.main(cli_args(d, out))  # no --plan-only, no allow flag
    assert code == tool.EXIT_NOT_AUTHORIZED
    record = json.loads(out.read_text(encoding='utf-8'))
    assert record['status'] == 'native_execution_not_authorized'
    assert record['qualification'] is False
    # no authorization: the planned constructor was never attempted,
    # so the OBSERVED actual count is 0 (not the old pre-set 1)
    assert record['constructor_calls_planned'] == 1
    assert record['constructor_calls_actual'] == 0
    assert 'constructor_calls_attempted' not in record
    assert 'capytaine' not in sys.modules


def test_cli_single_source_case(tmp_path):
    d = make_case_dir(tmp_path)
    out = tmp_path / 'result.json'
    assert tool.main(cli_args(d, out, ['--plan-only', '--source', 'lid'])) \
        == 0
    record = json.loads(out.read_text(encoding='utf-8'))
    assert set(record['plans']) == {'lid'}
    assert record['budget']['planned_batches'] == 1


def test_cli_reference_must_be_a_directory(tmp_path):
    """A regular-file reference is refused (root finding 1)."""
    d = make_case_dir(tmp_path)
    shutil.rmtree(d / 'reference')
    (d / 'reference').write_text('not a directory', encoding='utf-8')
    out = tmp_path / 'result.json'
    assert tool.main(cli_args(d, out, ['--plan-only'])) \
        == tool.EXIT_GUARD_REFUSED
    assert not out.exists()


def test_cli_output_inside_case_or_reference_refused(tmp_path):
    """The run-owned output may not live inside the case/reference."""
    d = make_case_dir(tmp_path)
    outputs = [d / 'out.json',
               d / 'reference' / 'out.json',
               d / '..' / 'case' / 'out.json']
    for out in outputs:
        assert tool.main(cli_args(d, out, ['--plan-only'])) \
            == tool.EXIT_GUARD_REFUSED
    assert not (d / 'out.json').exists()
    assert not (d / 'reference' / 'out.json').exists()


def test_cli_output_prefix_attack_and_dotdot_refused(tmp_path):
    """No string-prefix scope matching, no ``..`` escape (root finding 4)."""
    d = make_case_dir(tmp_path)
    evil_repo = (REPO / '.development-evidence'
                 / (tool.TASK_NAME + '-evil') / 'x.json')
    assert tool.main(cli_args(d, evil_repo, ['--plan-only'])) \
        == tool.EXIT_GUARD_REFUSED
    assert not evil_repo.exists()
    workspace = Path('C:/ANYworkspaceWork/anymarine-evidence')
    evil_scope = workspace / (tool.TASK_NAME + '-evil') / 'x.json'
    assert tool.main(cli_args(d, evil_scope, ['--plan-only'])) \
        == tool.EXIT_GUARD_REFUSED
    escape = (workspace / tool.TASK_NAME / '..' / 'other-task' / 'x.json')
    assert tool.main(cli_args(d, escape, ['--plan-only'])) \
        == tool.EXIT_GUARD_REFUSED


def test_cli_fixed_caps_cannot_be_increased(tmp_path):
    """--max-batches/--receivers-per-batch may tighten, never increase."""
    d = make_case_dir(tmp_path)
    out = tmp_path / 'result.json'
    assert tool.main(cli_args(d, out, ['--plan-only',
                                       '--max-batches', '25'])) \
        == tool.EXIT_GUARD_REFUSED
    assert tool.main(cli_args(d, out, ['--plan-only',
                                       '--receivers-per-batch', '97'])) \
        == tool.EXIT_GUARD_REFUSED
    assert not out.exists()
    # tightening is allowed
    assert tool.main(cli_args(d, out, ['--plan-only',
                                       '--max-batches', '4'])) == 0


def test_cli_bad_args_leave_inventory_unchanged(tmp_path):
    """Refused runs create no directories and write no bytes."""
    d = make_case_dir(tmp_path, stale_identity=True)
    out = tmp_path / 'nested' / 'deep' / 'result.json'
    assert tool.main(cli_args(d, out, ['--plan-only'])) \
        == tool.EXIT_GUARD_REFUSED
    assert not (tmp_path / 'nested').exists()
    # a regular-file reference refusal also leaves nothing behind
    d2 = make_case_dir(tmp_path / 'second')
    shutil.rmtree(d2 / 'reference')
    (d2 / 'reference').write_text('x', encoding='utf-8')
    out2 = tmp_path / 'second' / 'nested2' / 'result.json'
    assert tool.main(cli_args(d2, out2, ['--plan-only'])) \
        == tool.EXIT_GUARD_REFUSED
    assert not (tmp_path / 'second' / 'nested2').exists()


def test_cli_case_hash_canonical_recompute_catches_tampering(tmp_path):
    """A stale stored hash (field changed, hash kept) is refused."""
    d = make_case_dir(tmp_path, stale_identity=True)
    out = tmp_path / 'result.json'
    assert tool.main(cli_args(d, out, ['--plan-only'])) \
        == tool.EXIT_GUARD_REFUSED
    assert not out.exists()


def test_cli_input_array_mismatch_refused(tmp_path):
    """input.npz arrays must match the case identity snapshot hashes."""
    d = make_case_dir(tmp_path)
    with np.load(d / 'input.npz') as data:
        arrays = {name: data[name] for name in data.files}
    arrays['vertices'] = arrays['vertices'] + 0.5
    np.savez(d / 'input.npz', **arrays)
    out = tmp_path / 'result.json'
    assert tool.main(cli_args(d, out, ['--plan-only'])) \
        == tool.EXIT_INVALID_INPUT
    assert not out.exists()


# ---- mock-positive main (public native/provider mocked; no native code)
class MockMesh:
    """Records the canonical face/quadrature requests like Capytaine."""

    def __init__(self, vertices=None, faces=None):
        self.vertices = np.asarray(vertices, dtype=float)
        self.faces_recorded = np.asarray(faces).tolist()
        self.quadrature_method = None
        self.quadrature = None

    def with_quadrature(self, method):
        self.quadrature_method = method
        pts, w, _n = triangle_quadrature(self.vertices[None, :, :])
        self.quadrature = (pts, w)
        return self

    @property
    def quadrature_points(self):
        return self.quadrature


class MockGF:
    """Mock bound gf: counts evaluate calls and override consultations."""

    def __init__(self, k, water_depth):
        self.k = k
        self.water_depth = water_depth
        self.exportable_settings = dict(
            gf_singularities='low_freq', floating_point_precision='float64')
        self.frozen_constructor_wall_seconds = 0.0001
        self.evaluate_calls = []

    def find_best_exponential_decomposition(self, kh, method=None):
        return 'mock-pinned-coefficients'

    def evaluate(self, receivers, mesh, **kwargs):
        self.evaluate_calls.append(
            dict(n_receivers=len(receivers), mesh=mesh, kwargs=kwargs))
        # the pinned decomposition is consulted once per native call
        self.find_best_exponential_decomposition(
            self.k * self.water_depth)
        pts, weights = mesh.quadrature
        provider = SyntheticNativeProvider(
            pts, weights, self.k, self.water_depth,
            amp=SYNTHETIC_EVANESENT_AMP)
        return provider(np.asarray(receivers, dtype=float))


class MockProfile:
    load_calls = []
    gfs = []

    @classmethod
    def load(cls, reference_directory, binding_record_path,
             sidecar_path):
        cls.load_calls.append((str(reference_directory),
                               str(binding_record_path),
                               str(sidecar_path)))
        return cls()

    def green_function(self, base, capytaine, *, k, water_depth):
        assert float(k).hex() == tool.FROZEN_K_HEX
        gf = MockGF(k, water_depth)
        cls = type(self)
        cls.gfs.append(gf)
        return gf

    def evaluator_case_metadata(self, gf):
        return dict(format='mock',
                    effective_settings=dict(gf.exportable_settings))


@pytest.fixture
def mock_native_modules(tmp_path):
    """Inject mock capytaine/profile modules for the actual-route CLI."""
    cpt = types.ModuleType('capytaine')
    cpt.Mesh = MockMesh
    cpt.__file__ = str(tmp_path / 'fake-capytaine' / '__init__.py')
    gf_mod = types.ModuleType('capytaine.green_functions')
    dh = types.ModuleType('capytaine.green_functions.delhommeau')

    class Delhommeau:
        pass

    dh.Delhommeau = Delhommeau
    abpp = types.ModuleType('anyhydro.artifact_bound_panel_profile')
    abpp.ArtifactBoundPanelProfile = MockProfile
    names = ('capytaine', 'capytaine.green_functions',
             'capytaine.green_functions.delhommeau',
             'anyhydro.artifact_bound_panel_profile')
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules['capytaine'] = cpt
    sys.modules['capytaine.green_functions'] = gf_mod
    sys.modules['capytaine.green_functions.delhommeau'] = dh
    sys.modules['anyhydro.artifact_bound_panel_profile'] = abpp
    MockProfile.load_calls = []
    MockProfile.gfs = []
    yield cpt
    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


def test_cli_mock_native_positive_directory_reference(
        tmp_path, mock_native_modules, capsys, monkeypatch):
    """Mock-positive main: the actual route is exercised end to end.

    With the public capytaine modules and ArtifactBoundPanelProfile
    mocked (reference bundle as a real DIRECTORY fixture), the CLI with
    --allow-native-execution must actually reach gf.evaluate: 2 calls,
    47 receivers each, 376 source-quadrature interactions, canonical
    [[0, 1, 2, 2]] faces, the same Gauss-Legendre 2 weights, ONE
    shared constructor and an OBSERVED coefficient-override count.
    The mock binding pins the mock NEW core identity, admitted ONLY by
    an explicit envelope-table patch (A ∪ B ∪ rectangle); the typed
    E1 domain preflight must PASS on all 1504 planned image arguments
    BEFORE the native constructor.
    """
    monkeypatch.setitem(tool.E1_CORE_ENVELOPES, MOCK_NEW_CORE_SHA,
                        tool.E1_ENVELOPE_AB_RECT)
    d = make_case_dir(tmp_path)
    out = tmp_path / 'mock-result.json'
    code = tool.main(cli_args(d, out, ['--allow-native-execution']))
    assert code == 0
    record = json.loads(out.read_text(encoding='utf-8'))
    assert record['status'] == 'completed_unqualified'
    assert record['qualification'] is False
    # typed E1 domain preflight PASSED before the native constructor
    pre = record['domain_preflight']
    assert pre['status'] == 'checked'
    assert pre['core_sha256'] == MOCK_NEW_CORE_SHA
    assert pre['envelope'] == tool.E1_ENVELOPE_AB_RECT
    assert pre['planned_args'] == 1504      # 2 cases x 47 x 4 nodes x 4
    assert pre['unsupported'] == 0
    assert pre['first_offender'] is None
    assert pre['native_entry'] is False     # preflight itself: no native
    # the reference bundle was loaded as a DIRECTORY and fingerprinted
    assert len(MockProfile.load_calls) == 1
    assert Path(MockProfile.load_calls[0][0]).is_dir()
    assert set(record['reference_fingerprint']) == \
        {'manifest.json', 'input.npz', 'reference.npz'}
    assert record['reference_bundle']['format'] == 'mock-binding'
    # exactly ONE shared gf constructor for both source cases:
    # planned 1, and actual 1 only because the constructor was
    # OBSERVED to complete (mocked completed caller)
    assert record['constructor_calls_planned'] == 1
    assert record['constructor_calls_actual'] == 1
    assert record['constructor_calls_attempted'] is True
    assert len(MockProfile.gfs) == 1
    gf = MockProfile.gfs[0]
    # gf.evaluate actually reached: 2 calls, 47 receivers each
    assert len(gf.evaluate_calls) == 2
    assert all(c['n_receivers'] == 47 for c in gf.evaluate_calls)
    for call in gf.evaluate_calls:
        assert float(call['kwargs']['wavenumber']).hex() \
            == tool.FROZEN_K_HEX
        mesh = call['mesh']
        assert mesh.faces_recorded == [[0, 1, 2, 2]]
        assert mesh.quadrature_method == 'Gauss-Legendre 2'
    # 376 source-quadrature interactions across both cases
    total_quad = sum(p['source_quad_interactions']
                     for p in record['plans'].values())
    assert total_quad == 376
    assert record['evaluation_budget']['actual_provider_calls'] == 2
    assert record['evaluation_budget']['planned_batches'] == 2
    # mesh receipts: native 4-point quadrature matches the saved one
    assert set(record['mesh_receipts']) == {'hull', 'lid'}
    for receipt in record['mesh_receipts'].values():
        assert receipt['faces'] == [[0, 1, 2, 2]]
        assert receipt['quadrature_method'] == 'Gauss-Legendre 2'
        assert receipt['n_quad_points'] == 4
        assert receipt['quadrature_points_match'] is True
        assert receipt['quadrature_weights_match'] is True
    # same Gauss weights as the saved translated source quadrature
    hull_mesh = gf.evaluate_calls[0]['mesh']
    np.testing.assert_allclose(
        hull_mesh.quadrature[1][0],
        record['sources']['hull']['quad_weights'],
        rtol=1e-13, atol=1e-15)
    # OBSERVED coefficient-override use (not a planned-only claim)
    assert record['coefficient_override']['planned_per_case'] == 1
    assert record['coefficient_override']['observed'] == 2
    # inputs (including the reference directory) are immutable
    assert record['inputs_immutable'] is True
    assert record['input_fingerprint_after'] == \
        record['input_fingerprint_before']
    # diagnostics present for both cases; gates are diagnostics only
    assert set(record['diagnostics']) == {'hull', 'lid'}
    for label in ('hull', 'lid'):
        assert record['diagnostics'][label]['actual_provider_calls'] == 1
        assert record['diagnostics'][label]['qualification'] is False
    # progress was saved to the run-owned output BEFORE the native calls
    printed = capsys.readouterr().out.splitlines()
    assert any('"status": "planned"' in line for line in printed)
    # trust scope is limited to the actual known case
    assert stored_case_hash(d) in record['trust_scope']


# ---- typed E1 domain preflight (BEFORE native constructor/evaluate) ----
OLD_BOUNDED_CORE_SHA = ('6f05eb7534dd1d3fe0a0c3e067d050c4f5f218076'
                        '1175201eb3cbd335d2ff891')


def test_e1_classify_arg_envelopes():
    legacy = 'legacy_A_union_B'
    rect = tool.E1_ENVELOPE_AB_RECT
    # case A / case B admitted by BOTH envelopes
    assert tool.classify_e1_arg(5.0, -40.0, legacy) == 'A'
    assert tool.classify_e1_arg(1.0, 0.25, legacy) == 'B'
    # rectangle-only arguments: refused by the legacy envelope,
    # admitted as 'RECT' only with the exact NEW core envelope
    assert tool.classify_e1_arg(100.0, -1.7, legacy) is None
    assert tool.classify_e1_arg(100.0, -1.7, rect) == 'RECT'
    assert tool.classify_e1_arg(200.0, -137.0, rect) == 'RECT'
    # outside the rectangle too
    assert tool.classify_e1_arg(211.0, -1.0, rect) is None
    assert tool.classify_e1_arg(100.0, -151.0, rect) is None
    assert tool.classify_e1_arg(100.0, 0.5, rect) is None
    # singular z = 0 and nonfinite arguments are always refused
    assert tool.classify_e1_arg(1.0, 0.0, rect) is None
    assert tool.classify_e1_arg(1.0, float('nan'), rect) is None
    assert tool.classify_e1_arg(float('inf'), -1.0, rect) is None
    # case B |z| <= 8 boundary is conservative over the angular arc
    assert tool.classify_e1_arg(8.0, 0.5, legacy) is None
    assert tool.classify_e1_arg(7.9, 0.5, legacy) == 'B'


def test_e1_domain_preflight_unknown_core_typed_refusal():
    ok, rep = tool.e1_domain_preflight(
        tool.FROZEN_K, tool.FROZEN_WATER_DEPTH,
        np.zeros((1, 3)), np.zeros((1, 3)), 'unknown-core-sha')
    assert ok is False
    assert rep['status'] == 'unknown_core'
    assert 'unknown native core identity' in rep['reason']
    assert OLD_BOUNDED_CORE_SHA in rep['supported_cores']


def test_e1_domain_preflight_counts_bound_by_actual_arrays():
    receivers = np.array([[100.0, 0.0, -1.0], [0.0, 0.0, -1.0]])
    nodes = np.array([[0.0, 0.0, -0.1], [1.0, 0.0, -0.1],
                      [0.0, 1.0, -0.1]])
    # legacy envelope: the far receiver is unsupported
    ok, rep = tool.e1_domain_preflight(
        tool.FROZEN_K, tool.FROZEN_WATER_DEPTH, receivers, nodes,
        OLD_BOUNDED_CORE_SHA)
    assert rep['planned_args'] == 2 * 3 * 4      # all 4 images per pair
    assert ok is False
    assert rep['unsupported'] > 0
    assert rep['first_offender']['receiver'] == [100.0, 0.0, -1.0]
    # rectangle envelope (monkeypatched NEW core): everything supported
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(tool.E1_CORE_ENVELOPES, 'newcore',
                   tool.E1_ENVELOPE_AB_RECT)
        ok2, rep2 = tool.e1_domain_preflight(
            tool.FROZEN_K, tool.FROZEN_WATER_DEPTH, receivers, nodes,
            'newcore')
    assert ok2 is True
    assert rep2['unsupported'] == 0
    assert rep2['planned_args'] == 2 * 3 * 4


def test_cli_old_core_domain_refused_before_native(
        tmp_path, mock_native_modules, capsys):
    """Exact OLD bounded core: typed domain_refused BEFORE the native
    constructor/evaluate (no gf built, no evaluate call), with
    unsupported counts and the first offender."""
    d = make_case_dir(tmp_path, core_sha=OLD_BOUNDED_CORE_SHA)
    out = tmp_path / 'old-core-result.json'
    code = tool.main(cli_args(d, out, ['--allow-native-execution']))
    assert code == tool.EXIT_DOMAIN_REFUSED
    record = json.loads(out.read_text(encoding='utf-8'))
    assert record['status'] == 'domain_refused'
    assert record['qualification'] is False
    pre = record['domain_preflight']
    assert pre['core_sha256'] == OLD_BOUNDED_CORE_SHA
    assert pre['envelope'] == 'legacy_A_union_B'
    assert pre['planned_args'] == 1504
    assert pre['unsupported'] > 0
    assert pre['first_offender'] is not None
    assert pre['first_offender']['case'] in ('hull', 'lid')
    # NO native entry: the profile was loaded but no gf was constructed
    assert len(MockProfile.load_calls) == 1
    assert MockProfile.gfs == []
    # planned vs actual constructor contract: domain refusal happens
    # BEFORE the constructor, so actual is the OBSERVED 0
    assert record['constructor_calls_planned'] == 1
    assert record['constructor_calls_actual'] == 0
    assert 'constructor_calls_attempted' not in record
    # measured input/reference post-check on the typed refusal
    assert record['inputs_immutable_post_check'] is True
    assert record['input_fingerprint_after'] == \
        record['input_fingerprint_before']
    printed = capsys.readouterr().out.splitlines()
    assert any('"status": "domain_refused"' in line for line in printed)


def test_cli_unknown_core_typed_refusal(
        tmp_path, mock_native_modules):
    """UNKNOWN artifact core identity: typed refusal, no native entry."""
    d = make_case_dir(tmp_path, core_sha='f' * 64)
    out = tmp_path / 'unknown-core-result.json'
    code = tool.main(cli_args(d, out, ['--allow-native-execution']))
    assert code == tool.EXIT_DOMAIN_REFUSED
    record = json.loads(out.read_text(encoding='utf-8'))
    assert record['status'] == 'domain_refused'
    pre = record['domain_preflight']
    assert pre['status'] == 'unknown_core'
    assert pre['core_sha256'] == 'f' * 64
    assert MockProfile.gfs == []
    # unknown core: typed refusal BEFORE the constructor; actual is
    # the OBSERVED 0 and the input/reference post-check is measured
    assert record['constructor_calls_planned'] == 1
    assert record['constructor_calls_actual'] == 0
    assert 'constructor_calls_attempted' not in record
    assert record['inputs_immutable_post_check'] is True
