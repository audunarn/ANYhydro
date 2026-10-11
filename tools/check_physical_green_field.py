# SPDX-License-Identifier: MPL-2.0
"""Physical Green field discriminator for the retained composite route.

Owned NEW tool for W5 (w5-physical-green-discriminator-20261011).  It
ports the preserved candidate
``.development-evidence/w5-composite-farfield-20261011/composite_farfield.py``
into a coherent, guarded implementation and fixes the root findings:

1. ``dispersion_factor`` is ``tanh(kh) + kh*(1 - tanh(kh)**2)``
   (``= tanh(kh) + kh*sech(kh)**2``), NOT ``th + kh*(1-th)**2``.
2. The candidate's Laplace block had a missing ``div +=`` statement and
   a shifted-receiver index-map bug; the index map is now built
   explicitly during receiver-set construction.
3. One source panel per native call (the candidate passed two panels in
   a single provider call).
4. Invalid source arrays, non-positive parameters and non-finite
   provider output are rejected BEFORE any pass is recorded.
5. Actual provider call counts are enforced against the budget, not
   merely planned counts.
6. ``--reference`` is the reference bundle DIRECTORY required by
   ``ArtifactBoundPanelProfile.load``; its regular files
   (manifest/input/reference) are fingerprinted together with the
   binding and sidecar, and the binding's
   ``reference_manifest_sha256`` bundle hash is preserved in the
   record.  No writes to any of them.
7. The exact frozen wavenumber is used only after an independent
   dispersion solve agrees within the known 1 ULP and the frozen k
   itself satisfies ``k*tanh(kh) = omega^2/g``; ``k.hex()`` must equal
   the frozen hex exactly (the bound constructor enforces the same
   against the reference manifest).
8. ``case.json`` is verified by CANONICALLY RECOMPUTING its case hash
   (sorted-key compact JSON of the identity), never by trusting the
   stored field, and the ``input.npz`` arrays are verified against the
   case identity ``snapshot_arrays`` shape/dtype/sha256 entries.
9. The native source mesh uses the canonical Capytaine triangle face
   ``[[0, 1, 2, 2]]`` with ``.with_quadrature("Gauss-Legendre 2")``
   and the native 4-point source quadrature is asserted to match the
   saved translated quadrature (a centroid default is refused).
10. Output guards use resolved-path containment (no string prefixes,
    no ``..`` escape), refuse outputs inside the case/reference/
    historic-evidence trees, keep the output a fresh run-owned JSON
    file, leave the filesystem inventory unchanged on bad arguments,
    and save the plan/progress record BEFORE any native call.

Retained composite route (unchanged public pole, no doubled jump
off-surface; gradients with respect to receivers):

    S_comp    = native_S.real    + 1j * pole_S
    grad_comp = native_grad.real + 1j * pole_grad

with ``pole_S``/``pole_grad`` from the UNCHANGED public
``anyhydro.panel_integration_candidate.imaginary_pole``.  The raw native
complex values are reported separately and clearly named; they are NOT
the retained route.

Governing physics (exp(-i omega t)): Laplace away from the source,
free surface ``d_z Phi - (omega^2/g) Phi = 0`` at z=0, seabed
``d_z Phi = 0`` at z=-h, outgoing ``exp(+i k R)``.  Far-field oracle
(single actual source triangle at the global origin,
d = (cos theta, sin theta)):

    Z(z)     = cosh(k(z+h))/cosh(kh)
    D        = tanh(kh) + kh*sech(kh)^2
    A(theta) = sum_q w_q Z(y_q.z) exp(-i k d.y_q.xy)
    Phi_far  = -i k/(2D) * sqrt(2/(pi k R)) * exp(i(kR - pi/4))
               * Z(x.z) * A(theta)

This is the outgoing Hankel asymptotic consistent with
``Im G = -k/(2D) Z(x.z) Z(y.z) J0(kR)`` (current pole + power
normalization).  The comparison target is the TOTAL composite field
including the real Rankine/image part, with no amplitude or phase
fitting; it is an exact-asymptotic comparison at finite kR (100, 200),
not a finite-distance equality.

The native provider is INJECTED.  ``None`` is a typed
:class:`NativeUnavailableError` refusal (no fallback, no broadening).
Actual native Green execution is ROOT-ONLY and additionally requires
the explicit ``--allow-native-execution`` flag; this worker never
executes the actual Green function, Capytaine, HAMS, Prony or any
linear solve.  ``profile.check_micro`` hidden probes are refused by the
budget.  Qualification is ALWAYS false: a successful evaluation is
recorded as ``completed_unqualified``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_SRC = _REPO / 'src'
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from anyhydro.errors import CapabilityError, GeometryValidationError  # noqa: E402
from anyhydro.panel_integration_candidate import (  # noqa: E402
    imaginary_pole, triangle_quadrature)

# ---- frozen case identity (brief, 2026-10-11) --------------------------
FROZEN_CASE_HASH = ('6d78c0fc3c6a4d5743777888e20cb0d7f1f278e4'
                    '0e6fae0ca42d98d36929556c')
FROZEN_K_HEX = '0x1.d5b98a919d5b9p-3'
FROZEN_K = float.fromhex(FROZEN_K_HEX)
FROZEN_OMEGA = 1.5
FROZEN_WATER_DEPTH = 150.0
FROZEN_GRAVITY = 9.81
HULL_FACE_INDEX = 72     # first vertical physical hull triangle
LID_FACE_INDEX = 720     # first horizontal internal lid triangle (z=-0.1)
LID_Z = -0.1

# ---- budgets (prospective; root may reduce) ----------------------------
MAX_NATIVE_BATCHES = 24          # total across both source cases
MAX_RECEIVERS_PER_BATCH = 96
MAX_SOURCE_PANELS_PER_CALL = 1

# ---- native route identity (matches the existing panel route) ----------
NATIVE_QUADRATURE_METHOD = 'Gauss-Legendre 2'   # public QUADRATURE_METHOD
CANONICAL_MESH_FACE = (0, 1, 2, 2)              # Capytaine triangle form
K_SOLVE_MAX_ULP = 2.0                           # known bisection ULP gap
TASK_NAME = 'w5-physical-green-discriminator-20261011'
# Connected W5 follow-up scope (free-surface Robin defect repair): the
# checker may write fresh evidence into EITHER the original task scope
# or the connected repair scope; every other location stays refused.
CONNECTED_TASK_NAME = 'w5-free-surface-repair-20261011'

# ---- proposed gate tolerances (root adopts before actual execution) ----
FAR_KR_TARGETS = (100.0, 200.0)
FAR_REL_TOL_AT_KR200 = 3.0e-2
FAR_FLOOR = 1.0e-12
SOMMERFELD_REL_TOL = 3.0e-2
FD_STEPS_K = (0.01, 0.005)
LAPLACE_REL_TOL = 1.0e-3
LAPLACE_ABS_TOL = 1.0e-9
GRADFD_REL_TOL = 1.0e-4
GRADFD_ABS_TOL = 1.0e-9
ROBIN_REL_TOL = 1.0e-4
ROBIN_ABS_TOL = 1.0e-9
BOTTOM_REL_TOL = 1.0e-4
BOTTOM_ABS_TOL = 1.0e-9

# receiver layout sizes
N_FAR = len(FAR_KR_TARGETS) * 2
N_INTERIOR = 3
N_SURFACE = 2
N_BOTTOM = 2

EXIT_OK = 0
EXIT_INVALID_INPUT = 2
EXIT_NOT_AUTHORIZED = 3
EXIT_GUARD_REFUSED = 4
EXIT_NATIVE_UNAVAILABLE = 5
EXIT_INPUT_MUTATED = 6
EXIT_EVALUATION_FAILED = 7
EXIT_DOMAIN_REFUSED = 8


class NativeUnavailableError(CapabilityError):
    """Typed refusal: the native Green function is not available."""


class GuardRefused(CapabilityError):
    """Typed refusal: an output/input guard rejected the request."""


# ---- typed E1 domain preflight (BEFORE native constructor/evaluate) -----
# Envelope table keyed by the EXACT known native float64 core identity
# (sha256 of capytaine/green_functions/Delhommeau_float64*.so as pinned
# by the binding's evaluator.native_extension_sha256).  An UNKNOWN core
# is a typed refusal with NO native entry; admission is never by
# Green-function name or generic backend capability.
E1_CORE_ENVELOPES = {
    # old bounded artifact core (w5-native-accuracy / coupled-followup):
    # legacy union of case A (Re(z) <= -30, |Im| <= 10) and case B
    # (Re(z) > -0.5, |Im| <= 10, 0 < |z| <= 8) only.
    '6f05eb7534dd1d3fe0a0c3e067d050c4f5f2180761175201eb3cbd335d2ff891':
        'legacy_A_union_B',
    # The w5-e1-domain-extension-20261011 candidate core (union with the
    # NEW rectangle Re(z) in [-150, 0], |Im| <= 210, z /= 0) is admitted
    # ONLY with its exact core sha256, and only after the wheel build,
    # the independent 408-point mpmath gate and the identity evidence
    # are adopted by root.  Root HAS adopted it (decision file
    # .development-evidence/w5-e1-root-verification-20261011/
    # root-adoption.json: 408 point values PASS, 14 typed refusals
    # PASS): the exact experimental core below is registered for THIS
    # discriminator ONLY -- no uniform qualification and no general
    # production admission.
    '6648b6c5eb71816244c03d0b27a21cb8b05dfcbb9d94ba7f42f7d32a6257fa4a':
        'A_union_B_union_rect',
}
E1_ENV_A_RE_MAX = -30.0
E1_ENV_ABS_IM_MAX = 10.0
E1_ENV_B_RE_MIN = -0.5
E1_ENV_B_ABS_Z_MAX = 8.0
E1_RECT_RE_MIN = -150.0
E1_RECT_RE_MAX = 0.0
E1_RECT_ABS_IM_MAX = 210.0
E1_ENVELOPE_AB_RECT = 'A_union_B_union_rect'


def classify_e1_arg(r, z, envelope):
    """Conservative (r, z)-level envelope check covering EVERY angular
    quadrature node zeta = z + i*r*cos(theta) of the 1001-node rule:
    |Im zeta| <= r and |zeta| <= hypot(r, |z|).  Returns 'A', 'B',
    'RECT' or None (unsupported).  The legacy envelope admits A and B
    only; the rectangle extension is admitted only for the exact NEW
    core identity."""
    if not (math.isfinite(r) and math.isfinite(z)):
        return None
    if z <= E1_ENV_A_RE_MAX and abs(r) <= E1_ENV_ABS_IM_MAX:
        return 'A'
    if (z > E1_ENV_B_RE_MIN and abs(r) <= E1_ENV_ABS_IM_MAX
            and z != 0.0 and math.hypot(r, z) <= E1_ENV_B_ABS_Z_MAX):
        return 'B'
    if envelope == E1_ENVELOPE_AB_RECT:
        if (E1_RECT_RE_MIN <= z <= E1_RECT_RE_MAX and z != 0.0
                and abs(r) <= E1_RECT_ABS_IM_MAX):
            return 'RECT'
    return None


def e1_image_args(k, water_depth, z_receiver, z_source):
    """The four scaled image z-arguments of the finite-depth native
    wave part for one receiver/source-Gauss pair (same convention as
    the retained native_accuracy_adapter planned_args)."""
    return (k * (z_receiver + z_source),
            k * (-z_receiver - 2.0 * water_depth + z_source),
            k * (z_receiver - z_source - 2.0 * water_depth),
            k * (-z_receiver - z_source - 4.0 * water_depth))


def e1_domain_preflight(k, water_depth, receivers, source_points, core_sha):
    """Typed preflight over ALL four image arguments per receiver and
    per source Gauss node, bounded by the ACTUAL arrays (no broad
    by-Green-function-name admission).  Returns (ok, report); an
    unknown core identity is a typed refusal (ok False, status
    'unknown_core') with NO native entry."""
    envelope = E1_CORE_ENVELOPES.get(core_sha)
    if envelope is None:
        return False, {'status': 'unknown_core',
                       'core_sha256': core_sha,
                       'reason': ('unknown native core identity: typed '
                                  'refusal, no native entry'),
                       'supported_cores': sorted(E1_CORE_ENVELOPES)}
    pts = np.asarray(source_points, dtype=float).reshape(-1, 3)
    recs = np.asarray(receivers, dtype=float)
    total = 0
    unsupported = 0
    first = None
    classes = set()
    for m in recs:
        for xi in pts:
            r = k * math.sqrt(float(np.sum((m[:2] - xi[:2]) ** 2)))
            for zm in e1_image_args(k, water_depth, float(m[2]),
                                    float(xi[2])):
                total += 1
                cls = classify_e1_arg(r, zm, envelope)
                if cls is None:
                    unsupported += 1
                    if first is None:
                        first = {'r': r, 'z': zm,
                                 'receiver': [float(v) for v in m],
                                 'source_node': [float(v) for v in xi]}
                else:
                    classes.add(cls)
    report = {'status': 'checked', 'core_sha256': core_sha,
              'envelope': envelope, 'planned_args': total,
              'unsupported': unsupported, 'classes': sorted(classes),
              'first_offender': first}
    return unsupported == 0, report


# ---- physics helpers ----------------------------------------------------
def dispersion_factor(k, water_depth):
    """D = tanh(kh) + kh*sech(kh)^2 = tanh(kh) + kh*(1 - tanh(kh)^2).

    This is d/dk [k*tanh(kh)] and is NOT ``th + kh*(1-th)**2`` (the
    candidate bug, preserved in the old evidence, fixed here).
    """
    kh = k * water_depth
    th = math.tanh(kh)
    return th + kh * (1.0 - th * th)


def depth_mode(z, k, water_depth):
    """Z(z) = cosh(k(z+h))/cosh(kh) and dZ/dz (vectorized)."""
    z = np.asarray(z, dtype=float)
    kh = k * water_depth
    return (np.cosh(k * (z + water_depth)) / math.cosh(kh),
            k * np.sinh(k * (z + water_depth)) / math.cosh(kh))


def robin_nu(k, water_depth):
    """Free-surface Robin coefficient nu = k*tanh(kh) (= omega^2/g)."""
    return k * math.tanh(k * water_depth)


def solve_wavenumber(omega, gravity, water_depth):
    """Solve omega^2/g = k*tanh(kh) for k by bisection (independent)."""
    if not (math.isfinite(omega) and omega > 0 and math.isfinite(gravity)
            and gravity > 0 and math.isfinite(water_depth)
            and water_depth > 0):
        raise GeometryValidationError('invalid dispersion parameters')
    target = omega * omega / gravity
    lo, hi = 1e-12, target / max(math.tanh(1e-6), 1e-12)
    hi = max(hi, 1.0)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * math.tanh(mid * water_depth) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---- input validation (before any pass is recorded) --------------------
def validate_params(k, water_depth):
    if not math.isfinite(k) or k <= 0:
        raise GeometryValidationError('wavenumber must be finite positive')
    if not math.isfinite(water_depth) or water_depth <= 0:
        raise GeometryValidationError('water depth must be finite positive')


def validate_source_arrays(source_points, source_weights):
    y = np.asarray(source_points, dtype=float)
    w = np.asarray(source_weights, dtype=float)
    if y.ndim != 3 or y.shape[2] != 3 or y.shape[1] < 1:
        raise GeometryValidationError(
            'source points must be finite (panels,nodes,3)')
    if w.shape != y.shape[:2]:
        raise GeometryValidationError('source weight shape mismatch')
    if not (np.all(np.isfinite(y)) and np.all(np.isfinite(w))):
        raise GeometryValidationError('source arrays must be finite')
    if np.any(w <= 0):
        raise GeometryValidationError('source weights must be positive')
    if y.shape[0] > MAX_SOURCE_PANELS_PER_CALL:
        raise CapabilityError(
            f'at most {MAX_SOURCE_PANELS_PER_CALL} source panel per native '
            f'call, got {y.shape[0]}')
    return y, w


def _validate_provider_output(S, K, n_receivers, n_panels):
    S = np.asarray(S)
    K = np.asarray(K)
    if S.shape != (n_receivers, n_panels):
        raise GeometryValidationError(
            f'provider S shape {S.shape} != {(n_receivers, n_panels)}')
    if K.shape != (3, n_receivers, n_panels):
        raise GeometryValidationError(
            f'provider K shape {K.shape} != {(3, n_receivers, n_panels)}')
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(K))):
        raise GeometryValidationError('non-finite provider output refused')
    return S, K


# ---- far-field oracle ---------------------------------------------------
def far_field_potential(receivers, source_points, source_weights, *, k,
                        water_depth):
    """Analytic far-field oracle Phi_far and its receiver gradient.

    Phi_far = -i k/(2D) sqrt(2/(pi k R)) exp(i(kR - pi/4)) Z(x.z) A(theta)
    A(theta) = sum_q w_q Z(y_q.z) exp(-i k d.y_q.xy), d=(cos,sin theta).
    """
    x = np.asarray(receivers, dtype=float)
    y, w = validate_source_arrays(source_points, source_weights)
    if x.ndim != 2 or x.shape[1] != 3 or not np.all(np.isfinite(x)):
        raise GeometryValidationError('invalid far-field receiver shapes')
    h = water_depth
    D = dispersion_factor(k, h)
    R = np.hypot(x[:, 0], x[:, 1])
    if np.any(R <= 0):
        raise GeometryValidationError('far-field receiver at origin')
    theta = np.arctan2(x[:, 1], x[:, 0])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    Zx, dZx = depth_mode(x[:, 2], k, h)
    Zy, _ = depth_mode(y[:, :, 2], k, h)
    ddy = cos_t[:, None, None] * y[None, :, :, 0] \
        + sin_t[:, None, None] * y[None, :, :, 1]
    phase = np.exp(-1j * k * ddy)
    A = np.einsum('rpq,pq->r', phase, w * Zy)
    ddt = -sin_t[:, None, None] * y[None, :, :, 0] \
        + cos_t[:, None, None] * y[None, :, :, 1]
    dA_dtheta = np.einsum('rpq,pq->r', phase * (-1j * k * ddt), w * Zy)
    amp = -1j * k / (2.0 * D) * math.sqrt(2.0 / (math.pi * k))
    radial = np.exp(1j * (k * R - math.pi / 4.0)) / np.sqrt(R)
    phi = amp * radial * Zx * A
    grad = np.zeros((3, len(x)), dtype=complex)
    dradial = radial * (1j * k - 1.0 / (2.0 * R))
    for i in (0, 1):
        d_hat = cos_t if i == 0 else sin_t
        tang = -sin_t if i == 0 else cos_t
        grad[i] = amp * Zx * A * dradial * d_hat \
            + amp * radial * Zx * dA_dtheta / R * tang
    grad[2] = amp * radial * A * dZx
    return phi, grad


def composite_assembly(native_S, native_grad, pole_S, pole_grad):
    """S_comp = native_S.real + 1j*pole_S (same for the gradient)."""
    nS = np.asarray(native_S)
    nG = np.asarray(native_grad)
    pS = np.asarray(pole_S)
    pG = np.asarray(pole_grad)
    if nS.shape != pS.shape or nG.shape != pG.shape:
        raise GeometryValidationError('native/pole shape mismatch')
    return nS.real + 1j * pS, nG.real + 1j * pG


# ---- native call budget -------------------------------------------------
@dataclass(frozen=True)
class NativeCallPlan:
    """Pre-call accounting, produced BEFORE any native evaluation."""
    case_label: str
    batches: tuple
    n_receivers: int
    n_batches: int
    max_batch_size: int
    constructor_calls: int
    probe_calls: int
    coefficient_overrides: int
    source_quad_interactions: int
    provider_calls_planned: int

    def as_dict(self):
        return dict(case_label=self.case_label, n_receivers=self.n_receivers,
                    n_batches=self.n_batches,
                    max_batch_size=self.max_batch_size,
                    constructor_calls=self.constructor_calls,
                    probe_calls=self.probe_calls,
                    coefficient_overrides=self.coefficient_overrides,
                    source_quad_interactions=self.source_quad_interactions,
                    provider_calls_planned=self.provider_calls_planned)


class NativeBudget:
    """Cumulative budget across source cases; actual counts enforced.

    Public gf.evaluate calls and source-quadrature interactions are
    counted.  Native E1 primitive evaluations inside the kernel are NOT
    measurable here and are reported as unmeasured.
    """

    def __init__(self, max_batches=MAX_NATIVE_BATCHES,
                 max_per_batch=MAX_RECEIVERS_PER_BATCH):
        self.max_batches = int(max_batches)
        self.max_per_batch = int(max_per_batch)
        self.planned_batches = 0
        self.actual_provider_calls = 0
        self.per_case_actual = {}

    def plan(self, receivers, source_points, source_weights, *,
             case_label, probes=(), constructor_calls=1,
             coefficient_overrides=1):
        x = np.asarray(receivers, dtype=float)
        y, _w = validate_source_arrays(source_points, source_weights)
        if x.ndim != 2 or x.shape[1] != 3 or not np.all(np.isfinite(x)):
            raise GeometryValidationError('invalid plan receiver shapes')
        probes = tuple(probes)
        forbidden = tuple(p for p in probes if 'check_micro' in str(p))
        if forbidden:
            raise CapabilityError(
                'hidden profile.check_micro probes are forbidden: '
                + repr(forbidden))
        n = len(x)
        n_batches = max(1, math.ceil(n / self.max_per_batch))
        if n_batches > self.max_batches:
            raise CapabilityError(
                f'receiver count {n} needs {n_batches} batches > '
                f'{self.max_batches}')
        if self.planned_batches + n_batches > self.max_batches:
            raise CapabilityError(
                f'cumulative batch budget exceeded: '
                f'{self.planned_batches}+{n_batches} > {self.max_batches}')
        batches = tuple(tuple(range(s, min(s + self.max_per_batch, n)))
                        for s in range(0, n, self.max_per_batch))
        self.planned_batches += n_batches
        quad = n * y.shape[0] * y.shape[1]
        return NativeCallPlan(
            case_label=case_label, batches=batches, n_receivers=n,
            n_batches=n_batches, max_batch_size=max(len(b) for b in batches),
            constructor_calls=int(constructor_calls),
            probe_calls=len(probes),
            coefficient_overrides=int(coefficient_overrides),
            source_quad_interactions=quad,
            provider_calls_planned=n_batches)

    def record_actual_calls(self, case_label, plan, actual):
        if actual != plan.provider_calls_planned:
            raise CapabilityError(
                f'actual provider calls {actual} != planned '
                f'{plan.provider_calls_planned} for case {case_label}')
        self.actual_provider_calls += actual
        self.per_case_actual[case_label] = actual
        if self.actual_provider_calls > self.max_batches:
            raise CapabilityError(
                f'actual cumulative provider calls {self.actual_provider_calls}'
                f' > batch budget {self.max_batches}')

    def as_dict(self):
        return dict(max_batches=self.max_batches,
                    max_receivers_per_batch=self.max_per_batch,
                    planned_batches=self.planned_batches,
                    actual_provider_calls=self.actual_provider_calls,
                    per_case_actual=self.per_case_actual,
                    native_e1_primitive_calls='unmeasured (inside the '
                    'native kernel; not counted by this tool)')


# ---- receiver layout ----------------------------------------------------
@dataclass
class ReceiverLayout:
    receivers: np.ndarray
    n_base: int
    base_dist: np.ndarray
    far_index: tuple
    interior_index: tuple
    surface_index: tuple
    bottom_index: tuple
    fd_index: dict  # (jl, step_i, direction) -> (i_plus, i_minus)
    fd_steps: tuple


def build_receiver_set(k, water_depth, source_points, *,
                       far_kr_targets=FAR_KR_TARGETS,
                       steps_k=FD_STEPS_K):
    """Deterministic exterior receivers: z in [-h, 0], away from source.

    Layout: far (kR targets x 2 headings, z=-0.05h), 3 interior (FD PDE
    checks), 2 free-surface (Robin), 2 bottom (no-flux), plus central-
    difference shifts of the interior receivers for both FD steps.
    """
    h = water_depth
    src = np.asarray(source_points, dtype=float).reshape(-1, 3)
    far = []
    for kr in far_kr_targets:
        for theta in (0.3, 1.9):
            r = kr / k
            far.append([r * math.cos(theta), r * math.sin(theta),
                        -0.05 * h])
    interior = np.array([[3.7, -2.1, -0.4 * h], [-4.9, 3.3, -0.7 * h],
                        [5.5, 4.8, -0.15 * h]])
    surface = np.array([[6.1, -5.2, 0.0], [-7.3, 6.6, 0.0]])
    bottom = np.array([[8.2, 7.7, -h], [-9.1, -8.4, -h]])
    base = np.vstack([np.asarray(far), interior, surface, bottom])
    dist = np.linalg.norm(base[:, None, :] - src[None, :, :],
                          axis=2).min(axis=1)
    if np.any(dist <= 0):
        raise GeometryValidationError('receiver coincides with a source')
    steps = tuple(s / k for s in steps_k)
    # shifted blocks are ordered (step, direction, sign); each block holds
    # all interior receivers.  The explicit index map below removes any
    # arithmetic-order assumption (candidate index bug fixed).
    shifted = []
    for hstep in steps:
        for direction in (0, 1, 2):
            for sign in (+1.0, -1.0):
                pts = np.array(interior, dtype=float)
                pts[:, direction] += sign * hstep
                shifted.append(pts)
    all_receivers = np.vstack([base] + shifted)
    fd_index = {}
    for jl in range(N_INTERIOR):
        for si in range(len(steps)):
            for direction in (0, 1, 2):
                block = (si * 6 + direction * 2)
                i_plus = len(base) + (block + 0) * N_INTERIOR + jl
                i_minus = len(base) + (block + 1) * N_INTERIOR + jl
                fd_index[(jl, si, direction)] = (i_plus, i_minus)
    return ReceiverLayout(
        receivers=all_receivers, n_base=len(base), base_dist=dist,
        far_index=tuple(range(len(far))),
        interior_index=tuple(range(len(far), len(far) + N_INTERIOR)),
        surface_index=tuple(range(len(far) + N_INTERIOR,
                                  len(far) + N_INTERIOR + N_SURFACE)),
        bottom_index=tuple(range(len(far) + N_INTERIOR + N_SURFACE,
                                 len(far) + N_INTERIOR + N_SURFACE + N_BOTTOM)),
        fd_index=fd_index, fd_steps=steps)


# ---- diagnostics --------------------------------------------------------
def _gate(residuals, scales, rel_tol, abs_tol):
    return bool(all(r <= rel_tol * s + abs_tol
                    for r, s in zip(residuals, scales)))


def evaluate_diagnostics(provider, source_points, source_weights, *, k,
                         water_depth, budget, case_label,
                         steps_k=FD_STEPS_K, far_kr_targets=FAR_KR_TARGETS,
                         constructor_calls=1, coefficient_overrides=1):
    """One source case: native + composite diagnostics, clearly named.

    The provider is called once per planned batch (<=96 receivers,
    exactly one source panel).  The public pole is evaluated on the
    same receivers in the same call sequence.  All validation happens
    BEFORE any pass is recorded; a typed refusal never broadens.
    """
    validate_params(k, water_depth)
    y, w = validate_source_arrays(source_points, source_weights)
    if provider is None:
        raise NativeUnavailableError(
            'native Green function unavailable; typed refusal, report to '
            'root (no fallback, no broadening)')
    if budget is None:
        raise CapabilityError('a NativeBudget is required')

    layout = build_receiver_set(k, water_depth, y, steps_k=steps_k,
                                far_kr_targets=far_kr_targets)
    plan = budget.plan(layout.receivers, y, w, case_label=case_label,
                       constructor_calls=constructor_calls,
                       coefficient_overrides=coefficient_overrides)
    counts = plan.as_dict()  # recorded BEFORE any evaluation

    # public pole on the same receiver set (unchanged public helper)
    pole_S, pole_grad = imaginary_pole(layout.receivers, y, w, k=k,
                                       water_depth=water_depth)
    _validate_provider_output(pole_S, pole_grad, len(layout.receivers),
                              y.shape[0])

    S_parts, K_parts = [], []
    actual_calls = 0
    for batch in plan.batches:
        S_b, K_b = provider(layout.receivers[list(batch)])
        S_b, K_b = _validate_provider_output(S_b, K_b, len(batch), y.shape[0])
        S_parts.append(S_b)
        K_parts.append(K_b)
        actual_calls += 1
    budget.record_actual_calls(case_label, plan, actual_calls)
    native_S = np.concatenate(S_parts, axis=0)
    native_K = np.concatenate(K_parts, axis=1)

    # retained composite route (single jump stays in the native real part)
    S_comp, K_comp = composite_assembly(native_S, native_K, pole_S, pole_grad)
    # per-panel gradient shape is (3, receivers, panels); total sums axis 2
    S_comp_tot = S_comp.sum(axis=1)
    K_comp_tot = K_comp.sum(axis=2)
    S_nat_tot = native_S.astype(complex).sum(axis=1)
    K_nat_tot = native_K.astype(complex).sum(axis=2)

    out = dict(case_label=case_label, k=k, water_depth=water_depth,
               plan_counts=counts,
               actual_provider_calls=actual_calls,
               public_pole_calls=1,
               receivers_saved=layout.receivers.tolist(),
               source_points_saved=y.tolist(),
               source_weights_saved=w.tolist(),
               base_dist_to_source=layout.base_dist.tolist(),
               fd_steps=list(layout.fd_steps),
               far_kr_targets=list(far_kr_targets))

    # ---- authoritative primitive arrays (independent replay basis) ----
    # The derived metrics above are NOT sufficient for independent
    # adjudication: the raw complex potential/gradient arrays of every
    # retained component are saved with full component metadata.  The
    # arrays are per (receiver, source panel) for the potential and
    # (3, receiver, panel) for the gradient, exactly as validated above.
    out['primitive_potential_native'] = {
        'real': native_S.real.tolist(), 'imag': native_S.imag.tolist()}
    out['primitive_gradient_native'] = {
        'real': native_K.real.tolist(), 'imag': native_K.imag.tolist()}
    out['primitive_potential_pole'] = {
        'real': pole_S.real.tolist(), 'imag': pole_S.imag.tolist()}
    out['primitive_gradient_pole'] = {
        'real': pole_grad.real.tolist(), 'imag': pole_grad.imag.tolist()}
    out['primitive_potential_composite'] = {
        'real': S_comp.real.tolist(), 'imag': S_comp.imag.tolist()}
    out['primitive_gradient_composite'] = {
        'real': K_comp.real.tolist(), 'imag': K_comp.imag.tolist()}
    out['primitive_metadata'] = dict(
        dtype='float64',
        potential_shape=[int(native_S.shape[0]), int(native_S.shape[1])],
        gradient_shape=[3, int(native_S.shape[0]),
                        int(native_S.shape[1])],
        components=('native_raw', 'public_pole', 'composite'),
        composite_assembly=('S_comp = native_S.real + 1j*pole_S; '
                            'K_comp = native_K.real + 1j*pole_grad'),
        robin_nu=robin_nu(k, water_depth),
        surface_index=[int(j) for j in layout.surface_index],
        bottom_index=[int(j) for j in layout.bottom_index],
        interior_index=[int(j) for j in layout.interior_index],
        far_index=[int(j) for j in layout.far_index],
        note=('authoritative primitive complex arrays for independent '
              'replay/adjudication; derived metrics are recomputable '
              'from these arrays and the saved receivers/sources'))

    # ---- far-field comparison (total field, no amplitude/phase fit) ----
    far = list(layout.far_index)
    phi_far, _ = far_field_potential(layout.receivers[far], y, w, k=k,
                                     water_depth=water_depth)
    theta_far = np.arctan2(layout.receivers[far, 1],
                           layout.receivers[far, 0])
    kr_actual = (k * np.hypot(layout.receivers[far, 0],
                              layout.receivers[far, 1])).tolist()
    out['far_kr_actual'] = kr_actual
    for label, S, K in (('native_raw', S_nat_tot, K_nat_tot),
                        ('composite', S_comp_tot, K_comp_tot)):
        phi = S[far]
        rel = np.abs(phi - phi_far) / np.maximum(np.abs(phi_far), FAR_FLOOR)
        radial = K[0, far] * np.cos(theta_far) + K[1, far] * np.sin(theta_far)
        som = np.abs(radial - 1j * k * phi)
        som_scale = np.maximum(np.maximum(k * np.abs(phi), np.abs(radial)),
                               FAR_FLOOR)
        out[label + '_far_rel_error_by_kr'] = rel.tolist()
        out[label + '_sommerfeld_residual'] = (som / som_scale).tolist()
    out['composite_far_gate_kr200'] = bool(all(
        e <= FAR_REL_TOL_AT_KR200
        for e, r in zip(out['composite_far_rel_error_by_kr'], kr_actual)
        if abs(r - 200.0) < 1e-6))
    out['composite_sommerfeld_gate'] = bool(
        all(r <= SOMMERFELD_REL_TOL
            for r in out['composite_sommerfeld_residual']))

    # ---- PDE / derivative checks (BOTH FD steps must pass) -------------
    nu = robin_nu(k, water_depth)
    for label, S, K in (('native_raw', S_nat_tot, K_nat_tot),
                        ('composite', S_comp_tot, K_comp_tot)):
        laplace_res, laplace_scale = [], []
        gradfd_res, gradfd_scale = [], []
        for jl in range(N_INTERIOR):
            jg = layout.interior_index[jl]
            for si in range(len(layout.fd_steps)):
                hstep = layout.fd_steps[si]
                div = 0.0 + 0.0j
                gerr = 0.0
                for direction in (0, 1, 2):
                    ip, im = layout.fd_index[(jl, si, direction)]
                    div += (K[direction, ip] - K[direction, im]) \
                        / (2.0 * hstep)
                    fd = (S[ip] - S[im]) / (2.0 * hstep)
                    gerr = max(gerr, abs(K[direction, jg] - fd))
                phi = S[jg]
                dist = layout.base_dist[jg]
                lap_scale = max(k * k * abs(phi), abs(phi) / dist ** 2,
                                FAR_FLOOR)
                gfd_scale = max(np.abs(K[:, jg]).max(), abs(phi) / dist,
                                FAR_FLOOR)
                laplace_res.append(abs(div))
                laplace_scale.append(lap_scale)
                gradfd_res.append(gerr)
                gradfd_scale.append(gfd_scale)
        out[label + '_laplace_residual'] = laplace_res
        out[label + '_laplace_scale'] = laplace_scale
        out[label + '_gradfd_residual'] = gradfd_res
        out[label + '_gradfd_scale'] = gradfd_scale
        surf = list(layout.surface_index)
        rob = [abs(K[2, j] - nu * S[j]) for j in surf]
        rob_scale = [max(abs(K[2, j]), nu * abs(S[j]), FAR_FLOOR)
                     for j in surf]
        out[label + '_surface_robin_residual'] = rob
        out[label + '_surface_robin_scale'] = rob_scale
        bot = list(layout.bottom_index)
        out[label + '_bottom_residual'] = [abs(K[2, j]) for j in bot]
        out[label + '_bottom_scale'] = [
            max(np.abs(K[:, j]).max(), k * abs(S[j]), FAR_FLOOR)
            for j in bot]
    out['composite_pde_gates'] = dict(
        laplace=_gate(out['composite_laplace_residual'],
                      out['composite_laplace_scale'],
                      LAPLACE_REL_TOL, LAPLACE_ABS_TOL),
        gradfd=_gate(out['composite_gradfd_residual'],
                     out['composite_gradfd_scale'],
                     GRADFD_REL_TOL, GRADFD_ABS_TOL),
        surface_robin=_gate(out['composite_surface_robin_residual'],
                            out['composite_surface_robin_scale'],
                            ROBIN_REL_TOL, ROBIN_ABS_TOL),
        bottom=_gate(out['composite_bottom_residual'],
                     out['composite_bottom_scale'],
                     BOTTOM_REL_TOL, BOTTOM_ABS_TOL))
    out['status'] = 'evaluated_unqualified'
    out['qualification'] = False
    return out


# ---- filesystem guards --------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def input_fingerprint(paths):
    return {str(p): sha256_file(p) for p in paths}


def check_input_immutability(before, paths):
    """True iff every input hash is unchanged (fresh re-hash)."""
    after = input_fingerprint(paths)
    return all(after[str(p)] == v for p, v in before.items())


def refusal_post_check(record, fingerprint_before, file_inputs,
                       reference, reference_fingerprint):
    """Measured input/reference immutability post-check on a typed
    refusal (domain refusal happens BEFORE any native entry; the
    post-check is a fresh measurement, never an unmeasured
    no-mutation claim, and root can independently re-hash)."""
    record['input_fingerprint_after'] = input_fingerprint(file_inputs)
    immutable = check_input_immutability(fingerprint_before, file_inputs)
    if reference_fingerprint is not None:
        immutable = (immutable
                     and fingerprint_reference_directory(reference)
                     == reference_fingerprint)
    record['inputs_immutable_post_check'] = bool(immutable)


def fingerprint_reference_directory(reference_directory):
    """Flatten the reference bundle's regular files to sha256 entries.

    The reference is a DIRECTORY (``ArtifactBoundPanelProfile.load``
    contract).  Every regular file inside it is fingerprinted by
    relative path; symlinks and non-regular files are refused.  No
    writes; the binding's ``reference_manifest_sha256`` bundle hash is
    preserved separately by the caller.
    """
    root = Path(reference_directory)
    if not root.is_dir():
        raise GuardRefused(f'reference must be a directory: {root}')
    guard_symlinks(root)
    entries = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise GuardRefused(f'symlinked reference file refused: {p}')
        if p.is_dir():
            continue
        if not p.is_file():
            raise GuardRefused(f'non-regular reference file refused: {p}')
        entries[p.relative_to(root).as_posix()] = sha256_file(p)
    if not entries:
        raise GuardRefused(f'reference directory is empty: {root}')
    return entries


def guard_symlinks(path):
    """Refuse if the path or any existing ancestor is a symlink."""
    p = Path(path)
    chain = [p]
    chain.extend(p.absolute().parents)
    for part in chain:
        if part.is_symlink():
            raise GuardRefused(f'symlink component refused: {part}')


def _resolved(path):
    return Path(path).resolve(strict=False)


def _contains(root, path):
    """Resolved-path containment: no string prefixes, no ``..`` escape."""
    root = _resolved(root)
    p = _resolved(path)
    return p == root or root in p.parents


def _task_evidence_scopes(repo_root=_REPO):
    """The ONLY evidence scopes this task may write to.

    Both the Windows form (``C:/ANYworkspaceWork/...``) and the Linux
    WSL form (``/mnt/c/ANYworkspaceWork/...``) are listed so the guard
    is correct on the Ubuntu-24.04 root runtime as well.
    """
    return (Path('C:/ANYworkspaceWork/anymarine-evidence') / TASK_NAME,
            Path('/mnt/c/ANYworkspaceWork/anymarine-evidence') / TASK_NAME,
            Path(repo_root) / '.development-evidence' / TASK_NAME,
            Path('C:/ANYworkspaceWork/anymarine-evidence')
            / CONNECTED_TASK_NAME,
            Path('/mnt/c/ANYworkspaceWork/anymarine-evidence')
            / CONNECTED_TASK_NAME,
            Path(repo_root) / '.development-evidence' / CONNECTED_TASK_NAME)


def _workspace_evidence_roots():
    return (Path('C:/ANYworkspaceWork/anymarine-evidence'),
            Path('/mnt/c/ANYworkspaceWork/anymarine-evidence'))


def guard_fresh_output(output_path):
    if _resolved(output_path).exists() or Path(output_path).exists():
        raise GuardRefused(
            f'fresh output guard: {output_path} already exists')


def guard_output_location(output_path, repo_root=_REPO,
                           task_evidence=(), forbidden_roots=()):
    """Output must stay outside source trees and foreign evidence trees.

    Containment is evaluated on RESOLVED paths (``..`` collapses,
    symlinks are refused separately by :func:`guard_symlinks`), never
    on string prefixes.  ``forbidden_roots`` are extra protected roots
    (the actual case directory and the reference bundle directory).
    ``task_evidence`` lists the ONLY evidence scopes where this task
    may write fresh evidence; every other repo or workspace evidence
    subtree is refused.
    """
    p = _resolved(output_path)
    root = _resolved(repo_root)
    for sub in ('src', 'tests', 'tools', 'docs'):
        if _contains(root / sub, p):
            raise GuardRefused(
                f'output inside source tree refused: {p}')
    for fr in forbidden_roots:
        if _contains(fr, p):
            raise GuardRefused(
                'output inside a protected case/reference location '
                f'refused: {p}')
    if isinstance(task_evidence, (str, Path)):
        task_evidence = (task_evidence,)
    allowed = tuple(_resolved(t) for t in task_evidence)
    if _contains(root / '.development-evidence', p) and not any(
            _contains(a, p) for a in allowed):
        raise GuardRefused(
            'output inside a protected evidence tree refused '
            f'(only the task scope is allowed): {p}')
    for wt in _workspace_evidence_roots():
        if _contains(wt, p) and not any(_contains(a, p) for a in allowed):
            raise GuardRefused(
                'output inside the workspace evidence tree refused '
                f'(only the task scope is allowed): {p}')


# ---- case loading -------------------------------------------------------
def canonical_case_hash(case):
    """Recompute the case hash from the case.json identity.

    The producer's convention (``panel_integration_candidate``) is
    sha256 over sorted-key compact JSON of the identity, i.e. of every
    case.json field EXCEPT ``case_hash`` itself.  The stored field is
    never trusted on its own.
    """
    identity = {key: value for key, value in case.items()
                if key != 'case_hash'}
    payload = json.dumps(identity, sort_keys=True,
                         separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def load_case(case_json_path, expected_case_hash):
    with open(case_json_path, 'r', encoding='utf-8') as fh:
        case = json.load(fh)
    stored = case.get('case_hash')
    recomputed = canonical_case_hash(case)
    if stored != recomputed:
        raise GuardRefused(
            f'case_hash does not canonically match the case.json '
            f'contents: stored {stored} != recomputed {recomputed}')
    if stored != expected_case_hash:
        raise GuardRefused(
            f'case_hash mismatch: {stored} != expected '
            f'{expected_case_hash}')
    return case


def verify_case_arrays(input_npz_path, case):
    """Verify input.npz against the case identity ``snapshot_arrays``.

    Each entry pins shape, dtype and sha256 of the ascontiguousarray
    bytes.  This binds the input arrays to the known trusted case
    artifact; it is NOT a generic qualification.
    """
    snapshot = case.get('snapshot_arrays') or {}
    if not snapshot:
        raise GeometryValidationError(
            'case.json carries no snapshot_arrays identity')
    with np.load(input_npz_path) as data:
        available = set(data.files)
        hashes = {}
        for name, spec in sorted(snapshot.items()):
            if name not in available:
                raise GeometryValidationError(
                    f'input.npz is missing the case array {name!r}')
            arr = np.asarray(data[name])
            digest = hashlib.sha256(
                np.ascontiguousarray(arr).tobytes()).hexdigest()
            hashes[name] = digest
            if (list(arr.shape) != list(spec['shape'])
                    or str(arr.dtype) != str(spec['dtype'])
                    or digest != spec['sha256']):
                raise GeometryValidationError(
                    f'input.npz array {name!r} does not match the '
                    f'case identity snapshot')
    return hashes


def select_source_triangles(input_npz_path, hull_face, lid_face):
    """Deterministic source selection + property verification + receipt."""
    d = np.load(input_npz_path)
    V = np.asarray(d['vertices'], dtype=float)
    F = np.asarray(d['faces'], dtype=int)
    stored_points = np.asarray(d['points'], dtype=float)
    stored_weights = np.asarray(d['weights'], dtype=float)
    if V.ndim != 2 or V.shape[1] != 3 or F.ndim != 2 or F.shape[1] != 3:
        raise GeometryValidationError('invalid mesh arrays in input.npz')
    sources = {}
    for label, idx, kind in (('hull', hull_face, 'vertical'),
                             ('lid', lid_face, 'horizontal')):
        if not (0 <= idx < len(F)):
            raise GeometryValidationError(f'{label} face {idx} out of range')
        tri = V[F[idx]]
        pts, w, normals = triangle_quadrature(tri[None, :, :])
        # quadrature receipt against the stored Gauss2 nodes/weights
        receipt = bool(np.allclose(pts[0], stored_points[idx], rtol=0,
                                   atol=1e-12)
                       and np.allclose(w[0], stored_weights[idx], rtol=0,
                                       atol=1e-12))
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(n)
        if norm <= 0:
            raise GeometryValidationError(f'{label} face {idx} degenerate')
        n = n / norm
        if kind == 'vertical' and abs(n[2]) > 0.25:
            raise GeometryValidationError(
                f'hull face {idx} is not vertical (normal {n})')
        if kind == 'horizontal':
            if n[2] > -0.95:
                raise GeometryValidationError(
                    f'lid face {idx} is not a downward horizontal lid '
                    f'(normal {n})')
            if abs(tri[:, 2].mean() - LID_Z) > 1e-6:
                raise GeometryValidationError(
                    f'lid face {idx} z {tri[:, 2].mean()} != {LID_Z}')
        if not receipt:
            raise GeometryValidationError(
                f'quadrature receipt failed for {label} face {idx}')
        # translate the horizontal centroid to the global origin (z
        # preserved).  Adopted for THIS isolated source check only
        # (horizontal translation invariance of the homogeneous flat
        # fluid); it is NOT a global coupled-case claim.
        delta_xy = tri[:, :2].mean(axis=0)
        translated = tri.copy()
        translated[:, :2] -= delta_xy
        pts_t, w_t, _ = triangle_quadrature(translated[None, :, :])
        sources[label] = dict(
            face_index=int(idx), normal=n.tolist(),
            vertices_original=tri.tolist(),
            vertices_translated=translated.tolist(),
            translation_delta_xy=delta_xy.tolist(),
            translation_note=(
                'isolated-source horizontal translation to the global '
                'origin, adopted for this check only under homogeneous '
                'flat-fluid translation invariance; height, shape and '
                'quadrature are preserved; no global coupled-case claim'),
            quad_points=pts_t[0].tolist(), quad_weights=w_t[0].tolist(),
            quadrature_receipt=receipt,
            z_mean=float(tri[:, 2].mean()))
    return sources


# ---- native route helpers (ROOT-ONLY path) ------------------------------
def make_source_mesh(captaine, vertices_translated, saved_points,
                     saved_weights):
    """Canonical one-triangle Capytaine mesh with the public quadrature.

    The face is the canonical Capytaine triangle ``[[0, 1, 2, 2]]``
    (exactly the existing panel route's
    ``np.column_stack((triangles, triangles[:, -1]))`` form) and the
    quadrature is the public ``'Gauss-Legendre 2'`` scheme.  The native
    4-point source quadrature is then ASSERTED to match the saved
    translated quadrature points/weights; a centroid default (1 point)
    is refused.
    """
    tri = np.asarray(vertices_translated, dtype=float)
    mesh = captaine.Mesh(
        vertices=tri,
        faces=np.array([list(CANONICAL_MESH_FACE)], dtype=int))
    mesh = mesh.with_quadrature(NATIVE_QUADRATURE_METHOD)
    native_points, native_weights = mesh.quadrature_points
    native_points = np.asarray(native_points, dtype=float)
    native_weights = np.asarray(native_weights, dtype=float)
    if native_points.shape != (1, 4, 3):
        raise GeometryValidationError(
            f'native mesh quadrature is not the 4-point Gauss2 rule: '
            f'points shape {native_points.shape}')
    pts_match = bool(np.allclose(native_points[0], saved_points,
                                 rtol=1e-13, atol=1e-13))
    weights_match = bool(np.allclose(native_weights[0], saved_weights,
                                    rtol=1e-13, atol=1e-15))
    if not (pts_match and weights_match):
        raise GeometryValidationError(
            'native mesh quadrature does not match the saved translated '
            'Gauss2 source quadrature (a centroid default is refused)')
    receipt = dict(
        faces=[list(CANONICAL_MESH_FACE)],
        quadrature_method=NATIVE_QUADRATURE_METHOD,
        n_quad_points=int(native_points.shape[1]),
        quadrature_points_match=pts_match,
        quadrature_weights_match=weights_match)
    return mesh, receipt


def install_coefficient_override_counter(gf):
    """Count ACTUAL uses of the pinned Prony coefficient override.

    The bound constructor pins
    ``find_best_exponential_decomposition`` to the frozen coefficients.
    This installs a counting wrapper as an INSTANCE attribute that
    delegates to the pinned class-level function unchanged, so the
    record can state the OBSERVED override use count.  If the
    attribute is not observable on the constructed gf, ``None`` is
    returned and the caller must report the override as unmeasured -
    never as a false actual claim.
    """
    original = getattr(type(gf), 'find_best_exponential_decomposition',
                       None)
    if not callable(original):
        return None
    counter = {'calls': 0}

    def counting(*args, **kwargs):
        counter['calls'] += 1
        return original(gf, *args, **kwargs)

    try:
        gf.find_best_exponential_decomposition = counting
    except (AttributeError, TypeError):
        return None
    return counter


# ---- CLI ---------------------------------------------------------------
def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(repr(obj))


def _dump(obj, path):
    """Write the run-owned output file (parents created lazily here).

    Bad-argument refusals never reach this function, so they leave the
    filesystem inventory completely unchanged; the output parent is
    created only when a record is actually written to this run's own
    fresh output file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def _runtime_measurements():
    """Actual runtime measurements taken BEFORE any native call.

    Env caps are recorded, but the measured threadpool size, the
    process affinity, the RLIMIT_AS soft/hard limits and (when
    threadpoolctl is installed) the live threadpool info are measured,
    not assumed.  Anything unavailable on the platform is reported as
    such - never invented.
    """
    keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS')
    out = dict(env={k: os.environ.get(k) for k in keys},
              cpu_count=os.cpu_count())
    try:
        out['available_threadpool_size'] = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        out['available_threadpool_size'] = (
            'unavailable on this platform (no sched_getaffinity)')
    try:
        import resource  # noqa: PLC0415 (POSIX only)
        out['rlimit_as'] = ['unlimited' if v == -1 else v
                            for v in resource.getrlimit(
                                resource.RLIMIT_AS)]
    except (ImportError, AttributeError, ValueError, OSError):
        out['rlimit_as'] = 'unavailable on this platform (no resource)'
    try:
        from threadpoolctl import threadpool_info  # noqa: PLC0415
        out['threadpool_info'] = threadpool_info()
    except Exception:  # noqa: BLE001
        out['threadpoolctl'] = (
            'unavailable (not installed); env caps and affinity above '
            'are the measured fallback')
    return out


def build_parser():
    p = argparse.ArgumentParser(
        description='Physical Green field discriminator (retained '
                    'composite route). Actual native execution is '
                    'ROOT-ONLY and requires --allow-native-execution.')
    p.add_argument('--case-input', required=True,
                   help='input.npz of the actual case')
    p.add_argument('--case-json', required=True,
                   help='case.json of the actual case')
    p.add_argument('--reference', required=True,
                   help='reference bundle directory '
                        '(ArtifactBoundPanelProfile.load)')
    p.add_argument('--binding', required=True,
                   help='native binding record JSON path')
    p.add_argument('--sidecar', required=True,
                   help='native sidecar JSON path')
    p.add_argument('--output', required=True,
                   help='fresh output JSON path (must not exist)')
    p.add_argument('--expected-case-hash', default=FROZEN_CASE_HASH)
    p.add_argument('--source', choices=('hull', 'lid', 'both'),
                   default='both')
    p.add_argument('--hull-face', type=int, default=HULL_FACE_INDEX)
    p.add_argument('--lid-face', type=int, default=LID_FACE_INDEX)
    p.add_argument('--max-batches', type=int, default=MAX_NATIVE_BATCHES)
    p.add_argument('--receivers-per-batch', type=int,
                   default=MAX_RECEIVERS_PER_BATCH)
    p.add_argument('--plan-only', action='store_true',
                   help='write plan/counts/metadata only; no profile '
                        'load, no native call')
    p.add_argument('--allow-native-execution', action='store_true',
                   help='ROOT-ONLY: permit the actual native evaluation')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    # fixed caps: the CLI may tighten them but NEVER increase them
    if (args.max_batches > MAX_NATIVE_BATCHES
            or args.receivers_per_batch > MAX_RECEIVERS_PER_BATCH):
        print(json.dumps({
            'status': 'guard_refused',
            'reason': (f'fixed caps cannot be increased: max-batches '
                       f'<= {MAX_NATIVE_BATCHES}, receivers-per-batch '
                       f'<= {MAX_RECEIVERS_PER_BATCH}'),
            'qualification': False}))
        return EXIT_GUARD_REFUSED
    output = Path(args.output)
    case_input = Path(args.case_input)
    case_json = Path(args.case_json)
    reference = Path(args.reference)
    binding = Path(args.binding)
    sidecar = Path(args.sidecar)
    task_evidence = _task_evidence_scopes()
    try:
        guard_symlinks(output)
        guard_fresh_output(output)
        # case/binding/sidecar are regular files; the reference is the
        # bundle DIRECTORY required by ArtifactBoundPanelProfile.load
        # (never a regular file).
        file_inputs = [case_input, case_json]
        if not args.plan_only:
            file_inputs += [binding, sidecar]
        for p in file_inputs:
            if not p.is_file():
                raise GuardRefused(f'input missing: {p}')
            if p.is_symlink():
                raise GuardRefused(f'symlinked input refused: {p}')
        if not reference.is_dir():
            raise GuardRefused(
                f'reference must be a bundle DIRECTORY: {reference}')
        if reference.is_symlink():
            raise GuardRefused(
                f'symlinked reference refused: {reference}')
        guard_symlinks(reference)
        # resolved containment: the output may not live inside the
        # actual case directory, the reference bundle, any historic
        # evidence tree or any source tree (guard_output_location
        # handles the evidence and source trees).
        guard_output_location(
            output, task_evidence=task_evidence,
            forbidden_roots=(case_input.resolve(strict=False).parent,
                             case_json.resolve(strict=False).parent,
                             reference.resolve(strict=False)))
        fingerprint_before = input_fingerprint(file_inputs)
        reference_fingerprint = (
            None if args.plan_only
            else fingerprint_reference_directory(reference))
        # NOTE: no mkdir here - a refused or invalid run must leave the
        # filesystem inventory unchanged; _dump creates the parent of
        # this run's own fresh output file lazily on first write.
    except GuardRefused as exc:
        print(json.dumps({'status': 'guard_refused', 'reason': str(exc),
                          'qualification': False}))
        return EXIT_GUARD_REFUSED

    try:
        case = load_case(case_json, args.expected_case_hash)
        input_array_hashes = verify_case_arrays(case_input, case)
        omega = float(case['omega'])
        water_depth = float(case['water_depth'])
        gravity = float(case['gravity'])
        if (omega, water_depth, gravity) != (FROZEN_OMEGA,
                                             FROZEN_WATER_DEPTH,
                                             FROZEN_GRAVITY):
            raise GeometryValidationError(
                'case parameters differ from the frozen actual criteria '
                '(omega/water_depth/gravity)')
        # exact frozen k, adopted ONLY after independent dispersion
        # consistency: the independent bisection solve must agree with
        # the frozen literal within the known ULP gap, and the frozen k
        # itself must satisfy omega^2/g = k*tanh(kh).  The bound
        # constructor additionally enforces k.hex() == manifest k_hex.
        k_solved = solve_wavenumber(omega, gravity, water_depth)
        k_solved_hex = float(k_solved).hex()
        k_ulp = abs(k_solved - FROZEN_K) / math.ulp(FROZEN_K)
        target = omega * omega / gravity
        disp_residual = abs(
            FROZEN_K * math.tanh(FROZEN_K * water_depth) - target)
        disp_tol = 16.0 * math.ulp(target)
        if k_ulp > K_SOLVE_MAX_ULP:
            raise GeometryValidationError(
                f'independent dispersion solve differs from the frozen '
                f'k by {k_ulp} ULP (> {K_SOLVE_MAX_ULP}): '
                f'{k_solved_hex} vs {FROZEN_K_HEX}')
        if disp_residual > disp_tol:
            raise GeometryValidationError(
                'frozen k fails independent dispersion consistency: '
                f'residual {disp_residual} > tolerance {disp_tol}')
        k = FROZEN_K
        k_hex = k.hex()
        if k_hex != FROZEN_K_HEX:
            raise GeometryValidationError(
                f'exact frozen k hex required: {k_hex} != '
                f'{FROZEN_K_HEX}')
        sources = select_source_triangles(case_input,
                                           args.hull_face, args.lid_face)
    except GuardRefused as exc:
        print(json.dumps({'status': 'guard_refused', 'reason': str(exc),
                          'qualification': False}))
        return EXIT_GUARD_REFUSED
    except (GeometryValidationError, KeyError, ValueError, OSError) as exc:
        print(json.dumps({'status': 'invalid_input', 'reason': repr(exc),
                          'qualification': False}))
        return EXIT_INVALID_INPUT

    labels = (('hull', 'lid') if args.source == 'both'
              else (args.source,))
    # preview plans (progress/call counts BEFORE any evaluation); the
    # evaluation budget is a SEPARATE budget, so actual-mode planning
    # is not double-counted by evaluate_diagnostics.  The gf is
    # constructed ONCE and shared by all source cases: only the first
    # case plans the constructor call.
    preview_budget = NativeBudget(max_batches=args.max_batches,
                                  max_per_batch=args.receivers_per_batch)
    layouts = {}
    plans = {}
    for i, label in enumerate(labels):
        y = np.asarray(sources[label]['quad_points'])[None, :, :]
        w = np.asarray(sources[label]['quad_weights'])[None, :]
        layout = build_receiver_set(k, water_depth, y)
        plan = preview_budget.plan(
            layout.receivers, y, w, case_label=label,
            constructor_calls=1 if i == 0 else 0,
            coefficient_overrides=1)
        layouts[label] = layout
        plans[label] = plan.as_dict()

    record = dict(
        tool='check_physical_green_field',
        qualification=False,
        trust_scope=(
            'binding and verification are limited to the actual known '
            f'case {args.expected_case_hash} and its pinned artifacts; '
            'this is NOT a generic qualification of the backend, the '
            'route or the solver'),
        case=dict(case_hash=case.get('case_hash'),
                  expected_case_hash=args.expected_case_hash,
                  case_hash_canonically_recomputed=True,
                  version=case.get('version'),
                  omega=omega, water_depth=water_depth,
                  gravity=gravity, rho=case.get('rho'),
                  params_match_frozen=True,
                  k=k, k_hex=k_hex, k_hex_exact=True,
                  k_solved_independently_hex=k_solved_hex,
                  k_solved_ulp_difference=k_ulp,
                  k_dispersion_residual=disp_residual,
                  k_dispersion_tolerance=disp_tol),
        input_array_hashes=input_array_hashes,
        sources={label: sources[label] for label in labels},
        plans=plans,
        budget=preview_budget.as_dict(),
        # planned vs actual constructor contract (root repair
        # 2026-10-11): planned is the single shared gf constructor the
        # preview plans for the first source case; actual starts at 0
        # and is ONLY incremented after an OBSERVED successful
        # constructor.  plan_only / not-authorized / native-unavailable
        # / domain-refused records therefore carry actual=0 (the old
        # record pre-set actual=1 before any constructor existed).
        constructor_calls_planned=1,
        constructor_calls_actual=0,
        constructor_calls_contract=(
            'constructor_calls_planned is the planned shared gf '
            'constructor; constructor_calls_actual counts only '
            'OBSERVED successful constructors (0 for plan_only, '
            'not-authorized, native-unavailable and domain-refused; '
            '1 only after the constructor returned normally); '
            'constructor_calls_attempted=True marks an attempt whose '
            'outcome a hard crash could leave unobserved'),
        metadata=dict(runtime=_runtime_measurements(),
                      plan_only=bool(args.plan_only),
                      allow_native_execution=bool(
                          args.allow_native_execution)))
    record['input_fingerprint_before'] = fingerprint_before
    if reference_fingerprint is not None:
        record['reference_fingerprint'] = reference_fingerprint
        try:
            with open(binding, 'r', encoding='utf-8') as fh:
                binding_record = json.load(fh)
        except (ValueError, OSError) as exc:
            print(json.dumps({'status': 'invalid_input',
                              'reason': repr(exc),
                              'qualification': False}))
            return EXIT_INVALID_INPUT
        record['reference_bundle'] = dict(
            reference_manifest_sha256=binding_record.get(
                'reference_manifest_sha256'),
            format=binding_record.get('format'),
            external_identity=binding_record.get(
                'evaluator', {}).get('external_identity'))
    # progress/call counts BEFORE any evaluation
    print(json.dumps({'status': 'planned', 'cases': list(labels),
                      'plans': plans,
                      'budget': preview_budget.as_dict()}))

    if args.plan_only:
        record['status'] = 'plan_only'
        _dump(record, output)
        print(json.dumps({'status': 'plan_only', 'output': str(output)}))
        return EXIT_OK

    if not args.allow_native_execution:
        record['status'] = 'native_execution_not_authorized'
        _dump(record, output)
        print(json.dumps({'status': 'native_execution_not_authorized',
                          'qualification': False}))
        return EXIT_NOT_AUTHORIZED

    # progress/call counts preserved in the run-owned output BEFORE
    # any native call (a hard crash can then only lose later fields)
    record['status'] = 'planned_progress_saved'
    _dump(record, output)

    # ---- actual native route (ROOT-ONLY; lazy imports) -----------------
    try:
        import capytaine as cpt
        from capytaine.green_functions.delhommeau import Delhommeau
        from anyhydro.artifact_bound_panel_profile import (
            ArtifactBoundPanelProfile)
    except Exception as exc:  # noqa: BLE001
        record['status'] = 'native_unavailable'
        record['reason'] = repr(exc)
        _dump(record, output)
        print(json.dumps({'status': 'native_unavailable',
                          'reason': repr(exc)}))
        return EXIT_NATIVE_UNAVAILABLE
    try:
        eval_budget = NativeBudget(max_batches=args.max_batches,
                                   max_per_batch=args.receivers_per_batch)
        profile = ArtifactBoundPanelProfile.load(
            reference, binding, sidecar)
        # ---- typed E1 domain preflight: BEFORE the native constructor --
        # The exact float64 core identity pinned by the binding selects
        # the tested envelope table; an unknown core is a typed refusal
        # and the actual case arrays (all 4 images per receiver/source
        # Gauss node) are classified WITHOUT any native entry, so the
        # old bounded artifact returns domain_refused here instead of a
        # fatal native ERRORSTOP.
        core_ext = {key: value for key, value in binding_record.get(
            'evaluator', {}).get('native_extension_sha256',
                                 {}).items()
            if 'Delhommeau_float64' in key}
        core_sha = (list(core_ext.values())[0]
                    if len(core_ext) == 1 else None)
        preflight_total = 0
        preflight_unsupported = 0
        preflight_first = None
        preflight_classes = set()
        preflight_envelope = None
        for label in labels:
            ok, rep = e1_domain_preflight(
                k, water_depth, layouts[label].receivers,
                np.asarray(sources[label]['quad_points'], dtype=float),
                core_sha)
            if rep.get('status') == 'unknown_core':
                record['status'] = 'domain_refused'
                record['domain_preflight'] = rep
                refusal_post_check(record, fingerprint_before,
                                   file_inputs, reference,
                                   reference_fingerprint)
                _dump(record, output)
                print(json.dumps({'status': 'domain_refused',
                                  'reason': rep['reason'],
                                  'core_sha256': core_sha,
                                  'qualification': False}))
                return EXIT_DOMAIN_REFUSED
            preflight_total += rep['planned_args']
            preflight_unsupported += rep['unsupported']
            preflight_classes.update(rep['classes'])
            preflight_envelope = rep['envelope']
            if (preflight_first is None
                    and rep['first_offender'] is not None):
                preflight_first = dict(rep['first_offender'],
                                       case=label)
        record['domain_preflight'] = dict(
            status='checked', core_sha256=core_sha,
            envelope=preflight_envelope,
            planned_args=preflight_total,
            unsupported=preflight_unsupported,
            classes=sorted(preflight_classes),
            first_offender=preflight_first,
            native_entry=False)
        if preflight_unsupported:
            record['status'] = 'domain_refused'
            refusal_post_check(record, fingerprint_before,
                               file_inputs, reference,
                               reference_fingerprint)
            _dump(record, output)
            print(json.dumps({
                'status': 'domain_refused',
                'reason': (f'{preflight_unsupported} of '
                           f'{preflight_total} planned image arguments '
                           f'are outside the tested envelope of the '
                           f'bound native core; typed refusal BEFORE '
                           f'any native constructor/evaluate call'),
                'planned_args': preflight_total,
                'unsupported': preflight_unsupported,
                'first_offender': preflight_first,
                'qualification': False}))
            return EXIT_DOMAIN_REFUSED
        # constructor attempt begins: actual is incremented only after
        # an OBSERVED successful constructor; a hard crash inside the
        # constructor leaves attempted=True / actual=0 (honest
        # uncertainty, the pre-call dump already saved the record)
        record['constructor_calls_attempted'] = True
        _dump(record, output)
        # native core identity MEASURED before the constructor (the
        # binding pins it; this is the fresh before/after measurement
        # required for independent adjudication)
        core_file = None
        if core_sha is not None:
            for _key in binding_record.get('evaluator', {}).get(
                    'native_extension_sha256', {}):
                if 'Delhommeau_float64' in _key:
                    core_file = (Path(cpt.__file__).resolve().parent
                                 / 'green_functions' / Path(_key).name)
        if core_file is not None and core_file.is_file():
            record['native_core_identity'] = dict(
                pinned_sha256=core_sha,
                file=core_file.name,
                measured_before=sha256_file(core_file))
        else:
            record['native_core_identity'] = dict(
                pinned_sha256=core_sha,
                measured_before='unavailable (core file not found)')
        # frozen Prony coefficients, read-only from the reference
        # bundle; the identity is already enforced by the profile load
        with np.load(reference / 'input.npz') as _ref_arrays:
            _coeff = np.asarray(_ref_arrays['coefficients'], dtype=float)
        record['frozen_prony_coefficients'] = dict(
            shape=list(_coeff.shape),
            sha256=hashlib.sha256(
                np.ascontiguousarray(_coeff).tobytes()).hexdigest(),
            values=_coeff.tolist(),
            note=('read-only from the reference bundle input.npz; '
                  'identity enforced by ArtifactBoundPanelProfile.load; '
                  'no new fit, no relabeling'))
        _dump(record, output)
        gf = profile.green_function(Delhommeau, cpt, k=k,
                                    water_depth=water_depth)
        record['constructor_calls_actual'] = 1
        override_counter = install_coefficient_override_counter(gf)
        record['metadata']['gf'] = dict(
            green_function=type(gf).__name__,
            constructor_wall_seconds=float(
                getattr(gf, 'frozen_constructor_wall_seconds', float('nan'))),
            exportable_settings=dict(gf.exportable_settings))
        record['metadata']['evaluator_case_metadata'] = \
            profile.evaluator_case_metadata(gf)
        diagnostics = {}
        mesh_receipts = {}
        for i, label in enumerate(labels):
            # canonical Capytaine triangle [[0,1,2,2]] with the public
            # Gauss-Legendre 2 quadrature; the native 4-point source
            # quadrature is asserted to match the saved translated
            # quadrature (no centroid default).
            mesh, receipt = make_source_mesh(
                cpt, sources[label]['vertices_translated'],
                np.asarray(sources[label]['quad_points'],
                           dtype=float),
                np.asarray(sources[label]['quad_weights'],
                           dtype=float))
            mesh_receipts[label] = receipt

            def provider(receivers, _mesh=mesh):
                S, K = gf.evaluate(np.asarray(receivers, dtype=float),
                                   _mesh, free_surface=0.0,
                                   water_depth=water_depth, wavenumber=k,
                                   adjoint_double_layer=True,
                                   early_dot_product=False,
                                   diagonal_term_in_double_layer=False)
                return S, K

            diagnostics[label] = evaluate_diagnostics(
                provider,
                np.asarray(sources[label]['quad_points'])[None, :, :],
                np.asarray(sources[label]['quad_weights'])[None, :],
                k=k, water_depth=water_depth, budget=eval_budget,
                case_label=label,
                constructor_calls=1 if i == 0 else 0,
                coefficient_overrides=1)
        record['diagnostics'] = diagnostics
        record['mesh_receipts'] = mesh_receipts
        record['evaluation_budget'] = eval_budget.as_dict()
        record['coefficient_override'] = dict(
            planned_per_case=1,
            observed=(override_counter['calls']
                      if override_counter is not None else
                      'unmeasured (find_best_exponential_decomposition '
                      'is not observable on the constructed gf); the '
                      'planned count above is NOT an actual claim'))
        # native core identity MEASURED after the evaluation
        if isinstance(record.get('native_core_identity'), dict):
            if core_file is not None and core_file.is_file():
                record['native_core_identity']['measured_after'] = \
                    sha256_file(core_file)
            else:
                record['native_core_identity']['measured_after'] = \
                    'unavailable (core file not found)'
    except CapabilityError as exc:
        record['status'] = 'native_unavailable'
        record['reason'] = repr(exc)
        _dump(record, output)
        print(json.dumps({'status': 'native_unavailable',
                          'reason': repr(exc)}))
        return EXIT_NATIVE_UNAVAILABLE
    except (GeometryValidationError, ValueError, OSError) as exc:
        record['status'] = 'evaluation_failed'
        record['reason'] = repr(exc)
        _dump(record, output)
        print(json.dumps({'status': 'evaluation_failed',
                          'reason': repr(exc)}))
        return EXIT_EVALUATION_FAILED

    immutable = check_input_immutability(fingerprint_before, file_inputs)
    if reference_fingerprint is not None:
        immutable = (immutable
                     and fingerprint_reference_directory(reference)
                     == reference_fingerprint)
    if not immutable:
        record['status'] = 'input_mutated'
        _dump(record, output)
        print(json.dumps({'status': 'input_mutated',
                          'qualification': False}))
        return EXIT_INPUT_MUTATED
    record['input_fingerprint_after'] = input_fingerprint(file_inputs)
    record['inputs_immutable'] = True
    # Unqualified ALWAYS: gates are diagnostics, never qualification.
    record['status'] = 'completed_unqualified'
    _dump(record, output)
    print(json.dumps({'status': 'completed_unqualified',
                      'qualification': False,
                      'budget': eval_budget.as_dict()}))
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
