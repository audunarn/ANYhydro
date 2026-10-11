# SPDX-License-Identifier: MPL-2.0
#!/usr/bin/env python3
"""Independent direct-Rankine panel-pair discriminator for one retained case.

Bounded postprocessing of one frozen development case. It (1) verifies artifact
identity (file hashes, case-hash agreement, snapshot_arrays manifest hash/
shape/dtype comparison for every input array, structural shape/finite/positive
weights/positive triangle areas/unit aligned normals/face-index validation,
no input mutation, fresh output directory resolved outside the immutable case
and parent trees before mkdir), (2) sequentially reconstructs the summed
weighted operator M = SUM_q diag(w[:,q]) S_q and the recorded summed-asymmetry
metric, (3) ranks unordered off-diagonal panel pairs by the real antisymmetric
part of M and keeps at most 12, and (4) evaluates an independent direct-Rankine
reference primitive on the selected pairs: constant unit source over the
physical source triangle, kernel G_R(x,y) = -1/(4*pi*|x-y|), receiver gradient
grad_x G_R = (x-y)/(4*pi*|x-y|^3).

The reference primitive is a closed-form analytic triangle formula derived
from boundary edge integrals plus the signed solid angle:

  scalar   Phi(x) = -1/(4*pi) * INT_T 1/r dA with r = |x-y|,
  gradient grad_x Phi(x) = 1/(4*pi) * INT_T (x-y)/r^3 dA,

using, per edge k with in-plane outward normal m_k, signed line offset p_k,
arc parameters s_0 < s_1 and P_k = sqrt(p_k^2 + h^2) (h = signed height of x
over the source plane):

  INT_T 1/r dA   = SUM_k p_k * [I(s_1) - I(s_0)],
  I(s) = asinh(s/P) - (2|h|/|p|) * atan( |p| s / ((P+|h|)(sqrt(s^2+P^2)+P)) ),
  tangential grad = SUM_k m_k * [asinh(s_1/P) - asinh(s_0/P)],
  normal grad     = solid angle Omega(x) (van Oosterom-Strackee form with the
                    sign convention E.n = -2*atan2(num, denom)).

The analytic candidate is matched against an independently derived control at
every node: a projection-anchored radial-exact angular Gauss-Legendre rule at
the fixed control levels 16/32/64 per dimension, in which the radial integral
INT_0^R rho/sqrt(rho^2+h^2) drho = sqrt(R^2+h^2) - |h| is exact and only the
angular integral is numerical, so near-coplanar receiver points avoid
the internal ridge of source-centroid Duffy quadrature. Gradient controls
use its exact radial moments followed by angular Gauss quadrature of
normal and tangential components at 16/32/64. Neither
control calls the analytic candidate. The previous centroid-split Duffy
product rule is retained at 16/32/64 as labelled legacy
data; its failed convergence on near-coplanar nodes is reported, never
loosened or dropped.

Per pair it reports the stored forward/reverse Mij and skew, the mixed
exact-source/4-receiver-node Rankine skew (analytic reference), the
symmetric-by-construction 4x4 double-quadrature control, observed control
differences, a conservative sum of weighted absolute observed errors
(correlated nodes; an estimate, not a rigorous bound), and the unexplained
residual skew_full - skew_rankine_mixed. Quantitative attribution of a pair
requires the fixed prospective criteria |I64-I32| <= 1e-10 + 1e-7*|I64|
(scalar) and ||g64-g32|| <= 1e-8 + 1e-6*||g64|| (gradient) to pass on all
eight directed node integrals and the conservative uncertainty to satisfy
u_skew <= 0.1*max(|stored skew|, |reference skew|, 1e-12). Worst pairs are
kept and reported as partial/unresolved when controls fail; nothing is
dropped or loosened.

This is a diagnostic primitive only. The direct Rankine term is NOT the
total finite-water Green function; the residual is unexplained (wave-part
quadrature skew, kernel asymmetry or evaluator error are not separated).
No Green function beyond -1/(4*pi*r), no BEM assembly, no linear solves, no
HAMS/Capytaine/native calls, no np.linalg.solve. On posix a missing or
non-single measured threadpool is a refusal before any numerical reference.
Inputs are never modified; outputs go only to a new output directory outside
the immutable case and parent trees. Development fixtures confer no
scientific qualification; W5 NO_GO is unchanged.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

FORMAT = 'anyhydro-retained-panel-pair-discriminator-v2'
DEFAULT_LEVELS = (16, 32, 64)
MAX_PAIRS = 12
MAX_NODE_COMPARISONS = 96  # 12 pairs * 2 directions * 4 receiver nodes
MAX_ANALYTIC_EVALS = MAX_NODE_COMPARISONS  # analytic candidate only
MAX_RADIAL_EVALS = MAX_NODE_COMPARISONS * 3  # base + normal derivative at each level
MAX_DUFFY_EVALS = MAX_NODE_COMPARISONS * 3
DEFAULT_RLIMIT_MIB = 1536
RECORDED_ASYM_REL = 0.03446422663563036
RECORDED_ASYM_IMAG_REL = 6.414440566959449e-16
EXPECTED_CASE_HASH = '6d78c0fc3c6a4d5743777888e20cb0d7f1f278e40e6fae0ca42d98d36929556c'
SCALAR_ABS = 1e-10
SCALAR_REL = 1e-7
GRAD_ABS = 1e-8
GRAD_REL = 1e-6
UNCERTAINTY_FRACTION = 0.1
DEGENERATE_AREA = 1e-12
NEAR_TOL = 1e-9
UNIT_NORMAL_TOL = 1e-9
ORIENTATION_TOL = 1e-6
IMMUTABLE_ANCESTORS = 3  # case tree plus parent, grandparent, great-grandparent

HONEST_LIMITS = (
    'Diagnostic postprocessing of one retained development case; the direct '
    'Rankine -1/(4*pi*r) term is not the total finite-water Green function.',
    'The residual skew_full - skew_rankine_mixed is UNEXPLAINED: wave-part '
    'quadrature skew, kernel asymmetry and evaluator error are not separated.',
    'The analytic reference (boundary edge integrals plus solid angle) is '
    'matched by an independent radial-exact angular quadrature control; '
    'observed differences are convergence estimates, not rigorous bounds.',
    'Pair uncertainty is a conservative sum of weighted absolute observed '
    'errors treating the eight directed nodes as fully correlated; it is an '
    'estimate, not a rigorous bound.',
    'The centroid-split Duffy product rule is retained as labelled legacy '
    'data; its failed convergence on near-coplanar nodes is reported and '
    'never loosened, dropped or replaced by identical values.',
    'The 4x4 double-quadrature control is symmetric by construction; it '
    'validates assembly algebra, not kernel accuracy.',
    'Attribution is per selected pair only; no global Green-accuracy, '
    'production-admission or W5 qualification claim is made.',
    'Sequential operator loading and per-pair primitive evaluation only; '
    'fixed control levels 16/32/64, no '
    'adaptive refinement sweeps.',
    'RLIMIT_AS is a virtual address-space cap, not a resident-memory claim.',
    'On posix a missing or non-single measured threadpool refuses the run '
    'before any numerical reference is loaded.',
    'One development point cannot admit W5 or any capability; W5 NO_GO is '
    'unchanged.',
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _max_abs(array: np.ndarray) -> float:
    return float(np.max(np.abs(array)))


def _rel(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _gauss_legendre01(level: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(level)
    return 0.5 * (nodes + 1.0), 0.5 * weights


def triangle_area(tri: np.ndarray) -> float:
    """Physical area of a triangle (3,3); orientation independent."""
    return 0.5 * float(np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])))


def _source_scale(tri: np.ndarray) -> float:
    area = triangle_area(tri)
    return max(float(np.max(np.linalg.norm(tri - tri[0][None, :], axis=1))), area ** 0.5)


def _point_triangle_min_distance(x: np.ndarray, tri: np.ndarray) -> float:
    """Minimum distance from x to the closed triangle (no linear solves)."""
    best = float(np.min(np.linalg.norm(tri - x[None, :], axis=1)))
    for k in range(3):
        a, b = tri[k], tri[(k + 1) % 3]
        edge = b - a
        denom = float(edge @ edge)
        t = 0.0 if denom <= 0.0 else min(1.0, max(0.0, float((x - a) @ edge) / denom))
        best = min(best, float(np.linalg.norm(a + t * edge - x)))
    e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
    n = np.cross(e1, e2)
    n2 = float(n @ n)
    if n2 > 0.0:
        h = float((x - tri[0]) @ n) / math.sqrt(n2)
        # barycentric coordinates via scalar triple products (exact, no solve)
        lam1 = float(np.cross(x - tri[0], e2) @ n) / n2
        lam2 = float(np.cross(e1, x - tri[0]) @ n) / n2
        if lam1 >= -1e-12 and lam2 >= -1e-12 and lam1 + lam2 <= 1.0 + 1e-12:
            best = min(best, abs(h))
    return best


def _plane_frame(tri: np.ndarray):
    """Unit normal and right-handed in-plane basis (n, u1, u2) or None."""
    e1 = tri[1] - tri[0]
    e2 = tri[2] - tri[0]
    n = np.cross(e1, e2)
    nn = float(np.linalg.norm(n))
    n1 = float(np.linalg.norm(e1))
    if not math.isfinite(nn) or nn <= 0.0 or not math.isfinite(n1) or n1 <= 0.0:
        return None
    n_hat = n / nn
    u1 = e1 / n1
    u2 = np.cross(n_hat, u1)
    return n_hat, u1, u2


def _edge_antiderivative(s: float, big_p: float, abs_p: float, abs_h: float) -> float:
    """I(s) = INT ds / (sqrt(s^2 + P^2) + |h|), valid for abs_p > 0, P > |h|.

    I(s) = asinh(s/P) - (2|h|/|p|) * atan( |p| s / ((P+|h|)(sqrt(s^2+P^2)+P)) )
    """
    value = math.asinh(s / big_p)
    if abs_h > 0.0:
        arg = abs_p * s / ((big_p + abs_h) * (math.sqrt(s * s + big_p * big_p) + big_p))
        value -= (2.0 * abs_h / abs_p) * math.atan(arg)
    return value


def _primitive_refusal(x: np.ndarray, tri: np.ndarray):
    """Shared refusal checks; returns (scale, frame) or (None, reason)."""
    if not (np.all(np.isfinite(tri)) and np.all(np.isfinite(x))):
        return None, 'nonfinite'
    area = triangle_area(tri)
    if not math.isfinite(area) or area <= DEGENERATE_AREA:
        return None, 'degenerate_source_triangle'
    scale = _source_scale(tri)
    if _point_triangle_min_distance(x, tri) <= NEAR_TOL * max(scale, 1.0):
        return None, 'on_or_near_source_triangle'
    frame = _plane_frame(tri)
    if frame is None:
        return None, 'degenerate_source_triangle'
    return (scale, frame), 'ok'


def analytic_triangle_integrals(x: np.ndarray, tri: np.ndarray):
    """Closed-form lamina integrals (boundary edge integrals + solid angle).

    Returns ((phi1, grad), 'ok') with phi1 = INT_T 1/|x-y| dA and
    grad = INT_T (x-y)/|x-y|^3 dA, or (None, refusal_reason).
    """
    x = np.asarray(x, dtype=float)
    tri = np.asarray(tri, dtype=float)
    base, status = _primitive_refusal(x, tri)
    if base is None:
        return None, status
    scale, frame = base
    n_hat, u1, u2 = frame
    h = float((x - tri[0]) @ n_hat)
    x_perp = x - h * n_hat
    phi = 0.0
    tangential = np.zeros(3)
    for k in range(3):
        a, b = tri[k], tri[(k + 1) % 3]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if not math.isfinite(length) or length <= 0.0:
            return None, 'degenerate_source_triangle'
        u = edge / length
        m = np.cross(u, n_hat)  # outward in-plane unit normal of edge k
        p = float((a - x_perp) @ m)
        s0 = float((a - x_perp) @ u)
        s1 = float((b - x_perp) @ u)
        big_p = math.sqrt(p * p + h * h)
        if big_p <= NEAR_TOL * max(scale, 1.0):
            return None, 'receiver_on_edge_line'
        abs_p, abs_h = abs(p), abs(h)
        if abs_p > NEAR_TOL * max(scale, 1.0):
            phi += p * (_edge_antiderivative(s1, big_p, abs_p, abs_h)
                        - _edge_antiderivative(s0, big_p, abs_p, abs_h))
        # else: the edge contribution limit is exactly 0
        tangential += m * (math.asinh(s1 / big_p) - math.asinh(s0 / big_p))
    # normal component: signed solid angle, E.n = -2*atan2(num, denom)
    a, b, c = tri[0] - x, tri[1] - x, tri[2] - x
    r1 = float(np.linalg.norm(a))
    r2 = float(np.linalg.norm(b))
    r3 = float(np.linalg.norm(c))
    num = float(a @ np.cross(b, c))
    den = (r1 * r2 * r3 + float(a @ b) * r3 + float(a @ c) * r2 + float(b @ c) * r1)
    omega = -2.0 * math.atan2(num, den)
    grad = tangential + omega * n_hat
    if not (math.isfinite(phi) and np.all(np.isfinite(grad))):
        return None, 'nonfinite'
    return (phi, grad), 'ok'


def radial_triangle_scalar(x: np.ndarray, tri: np.ndarray, level: int,
                           with_normal: bool = False, with_tangential: bool = False):
    """Projection-anchored radial-exact angular Gauss-Legendre quadrature.

    Splits T into the three signed sub-triangles (x_perp, y_k, y_{k+1}).
    The radial integral INT_0^R rho/sqrt(rho^2+h^2) drho = sqrt(R^2+h^2)-|h|
    is exact, so only the angular integral is numerical (level-point
    Gauss-Legendre per sector) and near-coplanar receivers carry no
    unresolved internal ridge. Returns (value, normal, 'ok') with
    value = INT_T 1/r dA and, if requested, normal = d(value)/dh, or
    (None, None, refusal_reason). With with_tangential=True the third
    return is the integrated tangential part of (x-y)/r^3, and status
    is fourth; refusals then return three None values and their status.
    """
    x = np.asarray(x, dtype=float)
    tri = np.asarray(tri, dtype=float)
    base, status = _primitive_refusal(x, tri)
    if base is None:
        return (None, None, None, status) if with_tangential else (None, None, status)
    scale, frame = base
    n_hat, u1, u2 = frame
    h = float((x - tri[0]) @ n_hat)
    x_perp = x - h * n_hat
    loc = np.empty((3, 2))
    loc[:, 0] = (tri - x_perp) @ u1
    loc[:, 1] = (tri - x_perp) @ u2
    if float(np.min(np.linalg.norm(loc, axis=1))) <= NEAR_TOL * max(scale, 1.0):
        return (None, None, None, 'radial_projection_on_vertex') if with_tangential else (None, None, 'radial_projection_on_vertex')
    theta = np.arctan2(loc[:, 1], loc[:, 0])
    nodes, weights = _gauss_legendre01(level)
    total = 0.0
    normal = 0.0
    tangential = np.zeros(3)
    sign_h = 1.0 if h > 0.0 else (-1.0 if h < 0.0 else 0.0)
    for k in range(3):
        k2 = (k + 1) % 3
        dth = math.remainder(float(theta[k2] - theta[k]), 2.0 * math.pi)
        if abs(dth) <= 1e-15:
            continue  # degenerate sector contributes exactly zero
        edge = tri[k2] - tri[k]
        length = float(np.linalg.norm(edge))
        if not math.isfinite(length) or length <= 0.0:
            return (None, None, None, 'degenerate_source_triangle') if with_tangential else (None, None, 'degenerate_source_triangle')
        u = edge / length
        m = np.cross(u, n_hat)
        p = float((tri[k] - x_perp) @ m)
        if p == 0.0:
            # Signed sub-triangle has zero area, even when its endpoints
            # subtend pi. The off-plane integrand is finite; skip its
            # measure-zero sector rather than refusing a valid receiver.
            continue
        m_loc = np.array([float(m @ u1), float(m @ u2)])
        ang = theta[k] + nodes * dth
        dhat = np.stack([np.cos(ang), np.sin(ang)], axis=1)
        den = dhat @ m_loc
        if float(np.min(np.abs(den))) <= 1e-12 or not bool(np.all(den * p > 0.0)):
            return (None, None, None, 'radial_sector_degenerate') if with_tangential else (None, None, 'radial_sector_degenerate')
        radius = p / den
        if not np.all(np.isfinite(radius)) or float(np.min(radius)) <= 0.0:
            return (None, None, None, 'radial_sector_degenerate') if with_tangential else (None, None, 'radial_sector_degenerate')
        sq = np.sqrt(radius * radius + h * h)
        total += dth * float(np.sum(weights * (sq - abs(h))))
        if with_tangential:
            # Exact radial moment of (x-y)/r^3: -dhat *
            # [asinh(R/|h|)-R/sqrt(R^2+h^2)]. At h=0 outside
            # the triangle, the common divergent constant cancels over
            # signed sectors; a finite arbitrary reference scale suffices.
            if h != 0.0:
                moment = np.arcsinh(radius / abs(h)) - radius / sq
            else:
                moment = np.log(2.0 * radius / scale) - 1.0
            directions = dhat[:, :1] * u1 + dhat[:, 1:] * u2
            tangential -= dth * np.sum((weights * moment)[:, None] * directions, axis=0)
        if with_normal:
            normal += dth * float(np.sum(weights * (h / sq - sign_h)))
    if not (math.isfinite(total) and math.isfinite(normal)
            and np.all(np.isfinite(tangential))):
        return (None, None, None, 'nonfinite') if with_tangential else (None, None, 'nonfinite')
    if with_tangential:
        return total, normal, tangential, 'ok'
    return total, normal, 'ok'


def duffy_rule(tri: np.ndarray, level: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric 3-rotation radial centroid-split Duffy product rule (legacy).

    Splits the triangle into the three sub-triangles (centroid, v_k, v_{k+1})
    and maps each to the unit square with the Duffy transform
    y = c + s*(v_k - c) + s*t*(v_{k+1} - v_k), whose Jacobian measure is
    2*A_sub*s ds dt with A_sub = A/3. Returns quadrature points (3*level^2, 3)
    and weights (3*level^2,) including the factor s but not 2*A/3, which the
    caller applies (weights sum to 1/2 per sub-triangle, so the caller factor
    is 2*area/3 per sub-triangle and the three rotations sum to the area).
    """
    nodes, weights = _gauss_legendre01(level)
    centroid = tri.mean(axis=0)
    points = []
    wts = []
    for k in range(3):
        a = tri[k] - centroid
        b = tri[(k + 1) % 3] - tri[k]
        s_i = np.repeat(nodes, level)
        st = (nodes[:, None] * nodes[None, :]).ravel()
        wq = np.repeat(weights, level) * np.tile(weights, level) * s_i
        y = centroid[None, :] + s_i[:, None] * a[None, :] + st[:, None] * b[None, :]
        points.append(y)
        wts.append(wq)
    return np.vstack(points), np.concatenate(wts)


def duffy_node_integrals(x: np.ndarray, tri: np.ndarray, levels):
    """Legacy centroid-split Duffy scalar/gradient integrals per level.

    Returns (per-level results, status); status 'ok' or a refusal reason.
    """
    tri = np.asarray(tri, dtype=float)
    x = np.asarray(x, dtype=float)
    base, status = _primitive_refusal(x, tri)
    if base is None:
        return None, status
    scale, _frame = base
    area = triangle_area(tri)
    results = {}
    for level in levels:
        pts, wts = duffy_rule(tri, level)
        factor = 2.0 * area / 3.0
        d = x[None, :] - pts
        r = np.sqrt(np.sum(d * d, axis=1))
        if not np.all(np.isfinite(r)) or float(np.min(r)) <= 0.0:
            return None, 'nonfinite'
        scalar = factor * float(np.sum(wts * (-1.0 / (4.0 * math.pi * r))))
        grad = factor * (wts[:, None] * d / (4.0 * math.pi * r[:, None] ** 3)).sum(axis=0)
        if not (math.isfinite(scalar) and np.all(np.isfinite(grad))):
            return None, 'nonfinite'
        results[int(level)] = {'scalar': scalar, 'grad': grad}
    return results, 'ok'


def duffy_control_record(results: dict, levels) -> dict:
    """Legacy observed convergence differences and prospective pass flags."""
    record = {'label': 'legacy_centroid_split_duffy'}
    for level in levels:
        record[f'i{level}'] = results[level]['scalar']
    for level in levels:
        record[f'g{level}'] = [float(v) for v in results[level]['grad']]
    if len(levels) >= 2:
        lo, mid = levels[0], levels[1]
        record['d_mid_lo'] = abs(results[mid]['scalar'] - results[lo]['scalar'])
        record['ge_mid_lo'] = float(np.linalg.norm(results[mid]['grad'] - results[lo]['grad']))
    if len(levels) >= 3:
        top = levels[2]
        record['d_top_mid'] = abs(results[top]['scalar'] - results[levels[1]]['scalar'])
        record['ge_top_mid'] = float(np.linalg.norm(results[top]['grad'] - results[levels[1]]['grad']))
        i_top = results[top]['scalar']
        g_top = float(np.linalg.norm(results[top]['grad']))
        record['scalar_pass'] = bool(record['d_top_mid'] <= SCALAR_ABS + SCALAR_REL * abs(i_top))
        record['grad_pass'] = bool(record['ge_top_mid'] <= GRAD_ABS + GRAD_REL * g_top)
    return record


def node_reference_record(x: np.ndarray, tri: np.ndarray, levels, counters: dict):
    """All reference evaluations and controls for one receiver/source comparison.

    Returns (record, scalar_reference); scalar_reference is the analytic
    physical scalar -1/(4*pi) * INT_T 1/r dA, or None on refusal.
    """
    counters['node_comparisons'] += 1
    analytic, status = analytic_triangle_integrals(x, tri)
    if analytic is None:
        return {'status': status}, None
    counters['analytic_evaluations'] += 1
    phi1, grad_raw = analytic
    scalar_ref = -phi1 / (4.0 * math.pi)
    grad_ref = grad_raw / (4.0 * math.pi)
    record = {'status': 'ok',
              'scalar_reference': scalar_ref,
              'grad_reference': [float(v) for v in grad_ref]}

    # radial-exact angular control at the fixed levels
    radial = {}
    radial_normal = {}
    radial_tangential = {}
    radial_status = 'ok'
    for level in levels:
        counters['radial_evaluations'] += 1
        value, _normal, _tangential, st = radial_triangle_scalar(
            x, tri, level, with_normal=True, with_tangential=True)
        if value is None:
            radial_status = st
            break
        radial[int(level)] = -value / (4.0 * math.pi)
        radial_normal[int(level)] = -_normal / (4.0 * math.pi)
        radial_tangential[int(level)] = _tangential / (4.0 * math.pi)
    record['radial_status'] = radial_status
    if len(radial) == len(levels):
        lo, mid, top = levels
        record['radial'] = {str(l): radial[l] for l in levels}
        record['radial_d_mid_lo'] = abs(radial[mid] - radial[lo])
        record['radial_d_top_mid'] = abs(radial[top] - radial[mid])
        record['radial_scalar_pass'] = bool(
            record['radial_d_top_mid'] <= SCALAR_ABS + SCALAR_REL * abs(radial[top]))
        record['analytic_radial_diff'] = abs(scalar_ref - radial[top])
        record['analytic_match_pass'] = bool(
            record['analytic_radial_diff'] <= SCALAR_ABS + SCALAR_REL * abs(scalar_ref))
        record['scalar_pass'] = bool(record['radial_scalar_pass'] and record['analytic_match_pass'])
        record['observed_error_estimate'] = max(record['analytic_radial_diff'],
                                                record['radial_d_top_mid'],
                                                record['radial_d_mid_lo'])
    else:
        record['scalar_pass'] = False
        record['observed_error_estimate'] = None

    # Independent angular quadrature after exact radial integration of
    # potential and gradient; neither control calls the analytic candidate.
    if len(radial) == len(levels):
        n_hat = _plane_frame(np.asarray(tri, dtype=float))[0]
        controls = {level: radial_tangential[level] + radial_normal[level] * n_hat
                    for level in levels}
        lo, mid, top = levels
        d_mid_lo = float(np.linalg.norm(controls[mid] - controls[lo]))
        d_top_mid = float(np.linalg.norm(controls[top] - controls[mid]))
        match = float(np.linalg.norm(controls[top] - grad_ref))
        limit = GRAD_ABS + GRAD_REL * float(np.linalg.norm(grad_ref))
        record['grad_controls'] = {
            'method': 'independent_radial_angular_gradient',
            'values': {str(l): controls[l].tolist() for l in levels},
            'd_mid_lo': d_mid_lo, 'd_top_mid': d_top_mid,
            'analytic_match_diff': match, 'limit': limit}
        record['grad_pass'] = bool(d_top_mid <= limit and match <= limit)
    else:
        record['grad_pass'] = False

    # legacy labelled Duffy data (old reference; failures reported, not fixed)
    duffy, dstatus = duffy_node_integrals(x, tri, levels)
    if duffy is None:
        record['duffy_legacy'] = {'status': dstatus, 'label': 'legacy_centroid_split_duffy'}
    else:
        counters['duffy_evaluations'] += len(levels)
        record['duffy_legacy'] = duffy_control_record(duffy, levels)
    return record, scalar_ref


def rankine_kernel(x: np.ndarray, y: np.ndarray) -> float:
    r = float(np.linalg.norm(np.asarray(x, float) - np.asarray(y, float)))
    if r <= 0.0 or not math.isfinite(r):
        raise ValueError('singular or nonfinite Rankine evaluation')
    return -1.0 / (4.0 * math.pi * r)


def double_quadrature(nodes_a: np.ndarray, w_a: np.ndarray,
                      nodes_b: np.ndarray, w_b: np.ndarray) -> float:
    """4x4 receiver/source double quadrature of G_R (symmetric by construction)."""
    total = 0.0
    for q in range(nodes_a.shape[0]):
        for p in range(nodes_b.shape[0]):
            total += w_a[q] * w_b[p] * rankine_kernel(nodes_a[q], nodes_b[p])
    return total


def verify_single_thread_runtime(posix: bool) -> dict:
    """Measured threadpool verification. On posix a missing, empty or
    non-single threadpool is a refusal before any numerical reference."""
    info: dict = {'single_thread_required': posix}
    try:
        import threadpoolctl
    except ImportError:
        info['threadpoolctl'] = 'unavailable'
        if posix:
            raise RuntimeError('threadpoolctl unavailable on posix: '
                               'refusing before numerical reference')
        return info
    pools = threadpoolctl.threadpool_info()
    max_threads = max((int(p.get('num_threads', 0)) for p in pools), default=0)
    info['threadpoolctl'] = {'pools': len(pools), 'max_num_threads': max_threads}
    if posix and (len(pools) == 0 or max_threads != 1):
        raise RuntimeError('measured threadpool missing or num_threads != 1 on posix: '
                           'refusing before numerical reference')
    return info


def resolve_and_guard_output(case_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """Resolve both trees (following symlinks) and refuse any output inside the
    immutable case tree or its first IMMUTABLE_ANCESTORS ancestor trees,
    before any directory is created."""
    case_resolved = case_dir.resolve()
    out_resolved = output_dir.resolve()
    immutable = [case_resolved] + list(case_resolved.parents)[:IMMUTABLE_ANCESTORS]
    for root in immutable:
        if out_resolved == root or root in out_resolved.parents:
            raise ValueError(f'output directory {out_resolved} resolves inside '
                             f'immutable tree {root}')
    return case_resolved, out_resolved


def load_case(case_dir: Path) -> dict:
    files = ['input.npz', 'radiation.npz', 'operator-0.npz', 'operator-1.npz',
             'operator-2.npz', 'operator-3.npz', 'case.json', 'status.json']
    file_hashes = {}
    for name in files:
        path = case_dir / name
        if not path.is_file():
            raise FileNotFoundError(f'missing case file: {path}')
        file_hashes[name] = _sha256_file(path)
    case_json = json.loads((case_dir / 'case.json').read_text())
    case_hash = case_json.get('case_hash')
    if case_hash != EXPECTED_CASE_HASH:
        raise ValueError(f'case_hash mismatch: {case_hash!r} != {EXPECTED_CASE_HASH!r}')
    manifest = case_json.get('snapshot_arrays')
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('case.json snapshot_arrays manifest missing or empty')
    with np.load(case_dir / 'input.npz') as data:
        arrays = {key: data[key] for key in data.files}
    # manifest identity: every listed array must be present with exactly the
    # recorded shape, dtype and content hash (a copied case_hash alone does
    # not authenticate mutated arrays)
    if set(manifest.keys()) != set(arrays.keys()):
        raise ValueError(f'snapshot_arrays manifest keys {sorted(manifest.keys())} '
                         f'do not match input.npz arrays {sorted(arrays.keys())}')
    for name in sorted(manifest):
        spec = manifest[name]
        arr = arrays[name]
        if not isinstance(spec, dict) or 'shape' not in spec or 'dtype' not in spec \
                or 'sha256' not in spec:
            raise ValueError(f'manifest entry for {name!r} is incomplete')
        if list(spec['shape']) != list(arr.shape):
            raise ValueError(f'manifest shape mismatch for {name}: '
                             f'{spec["shape"]} != {list(arr.shape)}')
        if str(spec['dtype']) != arr.dtype.name:
            raise ValueError(f'manifest dtype mismatch for {name}: '
                             f'{spec["dtype"]} != {arr.dtype.name}')
        if _sha256_array(arr) != str(spec['sha256']):
            raise ValueError(f'manifest sha256 mismatch for input array {name}: '
                             'arrays do not match the case snapshot')
    required = ('vertices', 'faces', 'points', 'weights', 'normals', 'owner', 'hull_mask')
    for name in required:
        if name not in arrays:
            raise ValueError(f'input.npz is missing required array: {name}')
    validate_arrays(arrays, case_json)
    array_hashes = {key: _sha256_array(arrays[key]) for key in arrays}
    return {'file_hashes': file_hashes, 'case_json': case_json,
            'array_hashes': array_hashes, 'manifest_verified': True, **arrays}


def validate_arrays(arrays: dict, case_json: dict) -> None:
    """Structural validation of the input arrays before any reference use."""
    for name in arrays:
        if arrays[name].dtype == object:
            raise ValueError(f'object dtype in input array: {name}')
    verts = arrays['vertices']
    if verts.ndim != 2 or verts.shape[1] != 3 or not np.all(np.isfinite(verts)):
        raise ValueError(f'vertices must be (V,3) finite: shape {verts.shape}')
    faces = arrays['faces']
    if faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer):
        raise ValueError(f'faces must be (P,3) integer: shape {faces.shape} '
                          f'dtype {faces.dtype}')
    if int(faces.min()) < 0 or int(faces.max()) >= verts.shape[0]:
        raise ValueError('face index out of vertex range')
    panel_count = faces.shape[0]
    points = arrays['points']
    if points.shape != (panel_count, 4, 3) or not np.all(np.isfinite(points)):
        raise ValueError(f'points must be (P,4,3) finite: shape {points.shape}')
    weights = arrays['weights']
    if weights.shape != (panel_count, 4) or not np.all(np.isfinite(weights)) \
            or float(weights.min()) <= 0.0:
        raise ValueError('weights must be (P,4) finite and strictly positive')
    normals = arrays['normals']
    if normals.shape != (panel_count, 3) or not np.all(np.isfinite(normals)):
        raise ValueError(f'normals must be (P,3) finite: shape {normals.shape}')
    if float(np.max(np.abs(np.linalg.norm(normals, axis=1) - 1.0))) > UNIT_NORMAL_TOL:
        raise ValueError('normals must be unit length')
    owner = arrays['owner']
    body_ids = case_json.get('body_ids', [])
    if owner.shape != (panel_count,) or not np.issubdtype(owner.dtype, np.integer) \
            or int(owner.min()) < 0 or int(owner.max()) >= len(body_ids):
        raise ValueError('owner must be (P,) integer indexing body_ids')
    hull_mask = arrays['hull_mask']
    if hull_mask.shape != (panel_count,) or hull_mask.dtype != bool:
        raise ValueError(f'hull_mask must be (P,) bool: shape {hull_mask.shape} '
                          f'dtype {hull_mask.dtype}')
    tri = verts[faces]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    if not np.all(np.isfinite(areas)) or float(areas.min()) <= DEGENERATE_AREA:
        raise ValueError('all physical triangle areas must be finite and positive')
    geom = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    geom = geom / np.linalg.norm(geom, axis=1)[:, None]
    dots = np.sum(normals * geom, axis=1)
    if float(np.min(dots)) < 1.0 - ORIENTATION_TOL:
        raise ValueError('stored normals must align with the face orientation')


def reconstruct_summed_weighted_s(case_dir: Path, weights: np.ndarray, panel_count: int):
    """Sequential M = SUM_q diag(w[:,q]) S_q from the stored per-node operators."""
    summed = np.zeros((panel_count, panel_count), dtype=complex)
    for q in range(weights.shape[1]):
        with np.load(case_dir / f'operator-{q}.npz') as data:
            s = data['s']
        if s.shape != (panel_count, panel_count) or s.dtype != np.complex128:
            raise ValueError(f'operator-{q}.npz s has unexpected shape/dtype {s.shape}/{s.dtype}')
        if not np.all(np.isfinite(s)):
            raise ValueError(f'operator-{q}.npz s contains nonfinite entries')
        summed += weights[:, q][:, None] * s
        del s
    if not np.all(np.isfinite(summed)):
        raise ValueError('summed weighted S contains nonfinite entries')
    asym_rel = _rel(_max_abs(summed - summed.T), _max_abs(summed))
    imag_asym_rel = _rel(_max_abs(summed.imag - summed.imag.T), _max_abs(summed.imag))
    return summed, {'weighted_s_summed_asym_rel': asym_rel,
                    'weighted_s_summed_imag_asym_rel': imag_asym_rel}


def pair_geometry(i: int, j: int, arrays: dict, case_json: dict) -> dict:
    faces, vertices = arrays['faces'], arrays['vertices']
    body_ids = case_json.get('body_ids', [])
    shared = sorted(set(int(v) for v in faces[i]) & set(int(v) for v in faces[j]))
    adjacency = 'edge' if len(shared) >= 2 else ('vertex' if len(shared) == 1 else 'none')
    tri_i, tri_j = vertices[faces[i]], vertices[faces[j]]
    centroid_i, centroid_j = tri_i.mean(axis=0), tri_j.mean(axis=0)
    nodes_i, nodes_j = arrays['points'][i], arrays['points'][j]
    node_gap = float(np.min(np.linalg.norm(nodes_i[:, None, :] - nodes_j[None, :, :], axis=2)))
    area_i = triangle_area(tri_i)
    area_j = triangle_area(tri_j)
    geom_normal_i = np.cross(tri_i[1] - tri_i[0], tri_i[2] - tri_i[0])
    geom_normal_j = np.cross(tri_j[1] - tri_j[0], tri_j[2] - tri_j[0])
    record = {
        'panel_i': int(i), 'panel_j': int(j),
        'owner_i': int(arrays['owner'][i]), 'owner_j': int(arrays['owner'][j]),
        'body_id_i': body_ids[int(arrays['owner'][i])] if int(arrays['owner'][i]) < len(body_ids) else 'unknown',
        'body_id_j': body_ids[int(arrays['owner'][j])] if int(arrays['owner'][j]) < len(body_ids) else 'unknown',
        'role_i': 'hull' if bool(arrays['hull_mask'][i]) else 'lid',
        'role_j': 'hull' if bool(arrays['hull_mask'][j]) else 'lid',
        'shared_vertices': shared, 'adjacency': adjacency,
        'centroid_distance': float(np.linalg.norm(centroid_i - centroid_j)),
        'min_node_separation': node_gap,
        'area_i': area_i, 'area_j': area_j,
        'weights_row_sum_i': float(np.sum(arrays['weights'][i])),
        'weights_row_sum_j': float(np.sum(arrays['weights'][j])),
        'normal_i': [float(v) for v in arrays['normals'][i]],
        'normal_j': [float(v) for v in arrays['normals'][j]],
        'normal_orientation_dot_i': float(arrays['normals'][i] @ geom_normal_i /
                                           np.linalg.norm(geom_normal_i)) if np.linalg.norm(geom_normal_i) > 0 else None,
        'normal_orientation_dot_j': float(arrays['normals'][j] @ geom_normal_j /
                                           np.linalg.norm(geom_normal_j)) if np.linalg.norm(geom_normal_j) > 0 else None,
    }
    return record


def collect_stored_nodes(case_dir: Path, selected: list, node_count: int) -> dict:
    """One sequential operator pass collecting stored per-node S values for the
    selected pairs (both directions), instead of per-pair reloads."""
    stored = {(i, j): {q: None for q in range(node_count)} for i, j in selected}
    stored.update({(j, i): {q: None for q in range(node_count)} for i, j in selected})
    for q in range(node_count):
        with np.load(case_dir / f'operator-{q}.npz') as data:
            s = data['s']
        for (a, b) in stored:
            stored[(a, b)][q] = {'re': float(s[a, b].real), 'im': float(s[a, b].imag)}
        del s
    return stored


def evaluate_pair(i: int, j: int, summed: np.ndarray, arrays: dict, case_json: dict,
                  levels, stored_nodes: dict, counters: dict) -> dict:
    record = pair_geometry(i, j, arrays, case_json)
    m_ij = complex(summed[i, j])
    m_ji = complex(summed[j, i])
    record['m_ij'] = {'re': m_ij.real, 'im': m_ij.imag}
    record['m_ji'] = {'re': m_ji.real, 'im': m_ji.imag}
    record['skew_full'] = float((m_ij - m_ji).real)
    record['skew_full_imag'] = float((m_ij - m_ji).imag)
    record['stored_s_nodes'] = {
        'forward': [stored_nodes[(i, j)][q] for q in range(arrays['weights'].shape[1])],
        'reverse': [stored_nodes[(j, i)][q] for q in range(arrays['weights'].shape[1])],
    }

    directions = {}
    refusal = None
    for tag, (a, b) in (('forward', (i, j)), ('reverse', (j, i))):
        nodes = arrays['points'][a]
        wts = arrays['weights'][a]
        entries = []
        mixed = 0.0
        for q in range(nodes.shape[0]):
            counters['primitive_comparisons'] += 1
            entry, scalar_ref = node_reference_record(
                nodes[q], arrays['vertices'][arrays['faces'][b]], levels, counters)
            entry['node'] = q
            entries.append(entry)
            if scalar_ref is None:
                refusal = refusal or entry['status']
                continue
            mixed += float(wts[q]) * scalar_ref
        directions[tag] = {'nodes': entries,
                           'mixed_exact_source': mixed if refusal is None else None}
    record['rankine_directions'] = directions
    record['refusal'] = refusal

    # 4x4 double-quadrature control (symmetric by construction, not an oracle)
    nodes_i, nodes_j = arrays['points'][i], arrays['points'][j]
    w_i, w_j = arrays['weights'][i], arrays['weights'][j]
    double_ij = double_quadrature(nodes_i, w_i, nodes_j, w_j)
    double_ji = double_quadrature(nodes_j, w_j, nodes_i, w_i)
    record['double_quadrature'] = {'m_ij': double_ij, 'm_ji': double_ji,
                                   'skew': double_ij - double_ji}

    if refusal is None:
        skew_r = directions['forward']['mixed_exact_source'] - directions['reverse']['mixed_exact_source']
        record['skew_rankine_mixed'] = float(skew_r)
        record['residual_unexplained'] = float(record['skew_full'] - skew_r)
        # conservative sum of weighted absolute observed errors: the eight
        # directed node integrals are treated as fully correlated (no
        # unsupported RSS independence assumption)
        conservative = 0.0
        for tag in ('forward', 'reverse'):
            wts = arrays['weights'][i if tag == 'forward' else j]
            for entry in directions[tag]['nodes']:
                u = entry.get('observed_error_estimate')
                if u is None:
                    continue
                conservative += abs(float(wts[entry['node']])) * u
        record['skew_uncertainty_conservative'] = conservative
        controls_ok = all(e['status'] == 'ok' and e.get('scalar_pass') and e.get('grad_pass')
                          for tag in ('forward', 'reverse') for e in directions[tag]['nodes'])
        gate = UNCERTAINTY_FRACTION * max(abs(record['skew_full']), abs(skew_r), 1e-12)
        if controls_ok and conservative <= gate:
            record['attribution'] = 'quantitative'
        else:
            record['attribution'] = 'partial_controls'
        record['attribution_gate'] = gate
        record['attribution_fraction'] = (float(skew_r) / record['skew_full']) if abs(record['skew_full']) > 0.0 else None
    else:
        record['skew_rankine_mixed'] = None
        record['residual_unexplained'] = None
        record['attribution'] = 'unresolved_refusal'
    return record


def next_discriminator(pairs: list, aggregates: dict) -> dict:
    quantitative = [p for p in pairs if p['attribution'] == 'quantitative']
    sum_skew_full = sum(abs(p['skew_full']) for p in quantitative)
    sum_skew_r = sum(abs(p['skew_rankine_mixed']) for p in quantitative)
    sum_residual = sum(abs(p['residual_unexplained']) for p in quantitative)
    aggregates.update({'quantitative_sum_abs_skew_full': sum_skew_full,
                       'quantitative_sum_abs_skew_rankine': sum_skew_r,
                       'quantitative_sum_abs_residual': sum_residual})
    if not quantitative:
        proposal = ('prospective controls insufficient on every selected pair; next '
                    'discriminator: independent analytic triangle 1/r formula with '
                    'closed-form controls on the worst pairs before any wave-part claim')
    elif sum_residual <= 0.5 * sum_skew_r:
        proposal = ('mixed exact-source/4-node receiver quadrature of the direct '
                    'Rankine term dominates the quantitatively attributed selected '
                    'skew; next discriminator: wave-part (non-Rankine) quadrature skew '
                    'on these pairs under new explicit authority')
    else:
        proposal = ('unexplained residual dominates the quantitatively attributed '
                    'selected skew; next discriminator: independent wave-part Green '
                    'reference on the top-residual pairs under new explicit authority')
    aggregates['next_discriminator'] = proposal
    return aggregates


def build_report(args, case_dir: Path, identity: dict, reconstruction: dict,
                 pairs: list, aggregates: dict, counters: dict, runtime: dict) -> dict:
    return {
        'format': FORMAT,
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'case_dir': str(case_dir),
        'identity': identity,
        'reconstruction': {
            'weighted_s_summed_asym_rel': reconstruction['weighted_s_summed_asym_rel'],
            'weighted_s_summed_imag_asym_rel': reconstruction['weighted_s_summed_imag_asym_rel'],
            'recorded_asym_rel': RECORDED_ASYM_REL,
            'recorded_asym_imag_rel': RECORDED_ASYM_IMAG_REL,
            'abs_diff_rel': abs(reconstruction['weighted_s_summed_asym_rel'] - RECORDED_ASYM_REL),
            'abs_diff_imag_rel': abs(reconstruction['weighted_s_summed_imag_asym_rel'] - RECORDED_ASYM_IMAG_REL),
        },
        'pairs': pairs,
        'aggregates': aggregates,
        'counters': counters,
        'runtime': runtime,
        'honest_limits': list(HONEST_LIMITS),
    }


def write_markdown(path: Path, report: dict) -> None:
    lines = [
        '# Retained panel-pair discriminator (direct Rankine primitive)', '',
        f"Format: `{report['format']}`; case: `{report['case_dir']}`", '',
        '| pair | roles | adjacency | skew_full | skew_rankine_mixed | residual | u_skew | attribution |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for p in report['pairs']:
        lines.append(
            f"| {p['panel_i']}-{p['panel_j']} "
            f"| {p['role_i']}/{p['role_j']} "
            f"| {p['adjacency']} "
            f"| {p['skew_full']:.6e} "
            f"| {('%.6e' % p['skew_rankine_mixed']) if p['skew_rankine_mixed'] is not None else 'refused'} "
            f"| {('%.6e' % p['residual_unexplained']) if p['residual_unexplained'] is not None else 'n/a'} "
            f"| {('%.3e' % p.get('skew_uncertainty_conservative', float('nan'))) if p.get('skew_uncertainty_conservative') is not None else 'n/a'} "
            f"| {p['attribution']} |")
    lines += ['', f"Next discriminator: {report['aggregates']['next_discriminator']}", '']
    path.write_text('\n'.join(lines))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--case-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-pairs', type=int, default=MAX_PAIRS)
    parser.add_argument('--levels', type=int, nargs='+', default=list(DEFAULT_LEVELS))
    args = parser.parse_args(argv)
    if not (1 <= args.max_pairs <= MAX_PAIRS):
        parser.error(f'--max-pairs must be in 1..{MAX_PAIRS}')
    if list(args.levels) != list(DEFAULT_LEVELS):
        parser.error(f'--levels must be exactly {" ".join(str(l) for l in DEFAULT_LEVELS)} '
                     'for this case tool: no duplicate, reordered, subset or '
                     'out-of-scope control levels')
    started = time.perf_counter()
    if os.name == 'posix':
        import resource
        limit = DEFAULT_RLIMIT_MIB << 20
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    runtime = {'python': sys.version, 'platform': platform.platform(),
               'rlimit_as_mib': DEFAULT_RLIMIT_MIB if os.name == 'posix' else None}
    counters = {'primitive_comparisons': 0, 'node_comparisons': 0,
                'analytic_evaluations': 0, 'radial_evaluations': 0,
                'duffy_evaluations': 0, 'pairs_selected': 0,
                'control_levels': len(args.levels), 'max_pairs': args.max_pairs,
                'max_node_comparisons': MAX_NODE_COMPARISONS,
                'max_analytic_evaluations': MAX_ANALYTIC_EVALS,
                'max_radial_evaluations': MAX_RADIAL_EVALS,
                'max_duffy_evaluations': MAX_DUFFY_EVALS}
    try:
        runtime.update(verify_single_thread_runtime(os.name == 'posix'))
        case_dir, output_dir = resolve_and_guard_output(Path(args.case_dir),
                                                         Path(args.output_dir))
        arrays = load_case(case_dir)
        arrays['case_dir'] = case_dir
        try:
            output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f'REFUSAL: output directory already exists: {output_dir}', file=sys.stderr)
            return 2
        panel_count = arrays['points'].shape[0]
        summed, reconstruction = reconstruct_summed_weighted_s(case_dir, arrays['weights'], panel_count)
        antisym = (summed - summed.T).real
        iu = np.triu_indices(panel_count, k=1)
        skews = antisym[iu]
        order = np.argsort(-np.abs(skews))[:args.max_pairs]
        full_mass = float(np.sum(np.abs(skews)))
        selected_mass = float(np.sum(np.abs(skews[order])))
        roles = np.where(arrays['hull_mask'], 'hull', 'lid')
        same_body = arrays['owner'][iu[0]] == arrays['owner'][iu[1]]
        category_mass = {}
        for role_a in ('hull', 'lid'):
            for role_b in ('hull', 'lid'):
                for sb in (True, False):
                    mask = ((roles[iu[0]] == role_a) & (roles[iu[1]] == role_b) & (same_body == sb))
                    category_mass[f'{role_a}-{role_b}-{"same" if sb else "cross"}-body'] = {
                        'abs_skew_mass': float(np.sum(np.abs(skews[mask]))),
                        'pairs': int(np.count_nonzero(mask))}
        plateau = int(np.count_nonzero(np.abs(skews) >= 0.5 * float(np.max(np.abs(skews)))))
        stored = collect_stored_nodes(case_dir, [(int(iu[0][idx]), int(iu[1][idx])) for idx in order],
                                      arrays['weights'].shape[1])
        pairs = []
        for idx in order:
            i, j = int(iu[0][idx]), int(iu[1][idx])
            pairs.append(evaluate_pair(i, j, summed, arrays, arrays['case_json'], args.levels, stored, counters))
            counters['pairs_selected'] += 1
        del summed
        if counters['primitive_comparisons'] > counters['max_node_comparisons'] \
                or counters['node_comparisons'] > counters['max_node_comparisons'] \
                or counters['analytic_evaluations'] > counters['max_analytic_evaluations'] \
                or counters['radial_evaluations'] > counters['max_radial_evaluations'] \
                or counters['duffy_evaluations'] > counters['max_duffy_evaluations']:
            raise RuntimeError('primitive evaluation bound exceeded')
        aggregates = {'full_real_skew_mass': full_mass,
                      'selected_real_skew_mass': selected_mass,
                      'capture_fraction': _rel(selected_mass, full_mass),
                      'category_real_skew_mass': category_mass,
                      'pairs_above_half_max_abs_skew': plateau,
                      'selected_min_node_separation_range': [
                          float(min(p['min_node_separation'] for p in pairs)),
                          float(max(p['min_node_separation'] for p in pairs))]}
        groups: dict = {}
        for p in pairs:
            key = f"{p['role_i']}_{p['role_j']}_{p['body_id_i']}__{p['body_id_j']}"
            groups.setdefault(key, {'pairs': 0, 'abs_skew_full': 0.0})
            groups[key]['pairs'] += 1
            groups[key]['abs_skew_full'] += abs(p['skew_full'])
        aggregates['groups'] = groups
        aggregates['attribution_counts'] = {
            status: sum(1 for p in pairs if p['attribution'] == status)
            for status in ('quantitative', 'partial_controls', 'unresolved_refusal')}
        aggregates = next_discriminator(pairs, aggregates)
        if not all(_sha256_file(case_dir / name) == digest
                   for name, digest in arrays['file_hashes'].items()):
            raise RuntimeError('case input files changed during the run')
        identity = {'file_hashes': arrays['file_hashes'],
                    'array_hashes': arrays['array_hashes'],
                    'manifest_verified': arrays['manifest_verified'],
                    'case_hash': arrays['case_json'].get('case_hash'),
                    'case_hash_agrees': arrays['case_json'].get('case_hash') == EXPECTED_CASE_HASH,
                    'quadrature': arrays['case_json'].get('quadrature'),
                    'version': arrays['case_json'].get('version'),
                    'body_ids': arrays['case_json'].get('body_ids'),
                    'inventory': arrays['case_json'].get('inventory'),
                    'input_files_unchanged': True}
        runtime['wall_seconds'] = time.perf_counter() - started
        report = build_report(args, case_dir, identity, reconstruction, pairs, aggregates, counters, runtime)
        (output_dir / 'panel-pair-discriminator.json').write_text(json.dumps(report, indent=2))
        write_markdown(output_dir / 'panel-pair-discriminator.md', report)
        print(json.dumps({
            'asym_rel': report['reconstruction']['weighted_s_summed_asym_rel'],
            'abs_diff_rel': report['reconstruction']['abs_diff_rel'],
            'capture_fraction': aggregates['capture_fraction'],
            'attribution_counts': aggregates['attribution_counts'],
            'primitive_comparisons': counters['primitive_comparisons'],
            'next_discriminator': aggregates['next_discriminator'],
            'wall_seconds': runtime['wall_seconds'],
        }))
        return 0
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f'REFUSAL: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
