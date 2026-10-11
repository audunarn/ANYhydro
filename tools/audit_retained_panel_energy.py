# SPDX-License-Identifier: MPL-2.0
#!/usr/bin/env python3
"""Read-only stored-field energy-closure audit for one retained panel radiation case.

Loads the frozen solve artifacts of a single development energy point
(case/{input,radiation,operator-*,native-operator-*}.npz, case.json, status.json
and the parent summary.json, energy-comparison.json, retained-blocks.npz) and
verifies, by pure postprocessing only:

* artifact identity: file hashes, array hashes against the case snapshot,
  case-hash agreement between case.json and the parent summary;
* machine-algebra restoration of every recorded near/far/force-work and
  residual diagnostic for the frozen rate;
* sequential reconstruction of pressure, node residual, mean residual and the
  force-per-velocity matrix from the stored assembled operators;
* the real normal-trace assembly and the +0.5 self jump against the stored
  native integrals (the imaginary pole cannot be re-derived without a new
  kernel evaluation and is only cross-checked algebraically);
* per-coordinate (retained DOFs, unit rate) and frozen-rate energy closure,
  residual and reciprocity summaries, including lid contributions;
* a discrete work/adjoint diagnosis: the closure defect is decomposed
  exactly into within-panel pressure/normal-velocity covariance and a
  panel-mean persistent Green-relation defect of the saved trace, with
  projected retained work matrices, weighted-S reciprocity and the
  force-transpose adjoint identity;
* a closed-form far-field power built from the stored pole part of the
  operators against the frozen angular sums (64/128/256);
* cancellation and eigen diagnostics of the retained damping blocks and of
  the frozen HAMS reference blocks.

This is a diagnostic postprocessing tool. It performs no Green function
evaluations, no BEM assembly, no linear solves and no HAMS/Capytaine execution,
and it confers no scientific qualification: one development point cannot admit
W5 or any capability. Inputs are never modified; outputs go only to a new
output directory. Node/mean residual maxima are velocities in m/s; legacy
JSON keys ending in ``_w`` for those quantities are historical labels.
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

FORMAT = 'anyhydro-retained-panel-energy-audit-v3'
DEFAULT_RETAINED_DOFS = (0, 2, 6, 8)
DEFAULT_ANGULAR_COUNTS = (64, 128, 256)
DEFAULT_RLIMIT_MIB = 1536
AUDIT_JSON_NAME = 'retained-panel-energy-audit.json'
AUDIT_MD_NAME = 'retained-panel-energy-audit.md'

HONEST_LIMITS = (
    'Pure postprocessing of one retained development case; no Green function '
    'evaluations, no BEM assembly, no linear solves, no HAMS or Capytaine execution.',
    'The far source-pole construction is not an independent acceptance oracle; '
    'the far constant is frozen from independent analytic review.',
    'The closed-form far power and the finite angular sums share the same stored '
    'pole construction; their agreement validates algebra, not physics.',
    'The all-panel flux includes an artificial internal lid; it is not automatically '
    'the physical exterior-fluid boundary integral. Kernel accuracy versus '
    'discretization remains unresolved. No symmetry-forcing repair is adopted.',
    'A vanishing panel-mean residual does not prove pointwise boundary-condition '
    'satisfaction; node residual maxima are velocities in m/s.',
    'HAMS reference values are frozen recorded numbers; recomputing the reference '
    'power from the frozen reference blocks cross-checks algebra only.',
    'One development point cannot admit W5, thin_sheet or any capability; '
    'qualification is false and W5 NO_GO is unchanged.',
    'RLIMIT_AS is a virtual address-space cap for this analysis process, not a '
    'resident-memory claim.',
    'Angular counts are frozen at 64/128/256 for the retained case; no convergence '
    'sweep and no universal BEM convergence floor is claimed.',
    'The HAMS native omega 1.49974386 maps to 1.5 through the gravity similarity '
    '9.80665->9.81 preserving omega^2/g; it is not an undeclared offset.',
    'The work/adjoint decomposition is exact algebra on stored fields; its '
    'decision classifies where the discrete defect lives, not physical accuracy.',
    'Per-quadrature weighted-S asymmetry is stencil information (fixed target '
    'node, all-node source integral), not a reciprocity failure claim; the '
    'meaningful same-space measure is the quadrature sum Sum_q diag(w_q) S_q.',
    'A persistent panel-mean defect locates the discrepancy inside the stored '
    'discrete relation; without new independent evidence it does not '
    'distinguish kernel accuracy from discretization.',
)


class AuditRefusal(RuntimeError):
    """Raised when the audit refuses to proceed (identity, layout or safety)."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(values).tobytes())


def apply_virtual_limit(requested_mib: int) -> dict:
    """Best-effort RLIMIT_AS virtual address-space cap (Linux analysis only)."""
    record = {'requested_mib': int(requested_mib), 'applied': False,
              'platform': platform.system(),
              'note': 'virtual address-space cap for this process, not an RSS claim'}
    try:
        import resource
        if platform.system() != 'Linux':
            record['note'] += '; RLIMIT_AS not applied on this platform'
            return record
        limit = int(requested_mib) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        record.update(applied=True, soft_bytes=resource.getrlimit(resource.RLIMIT_AS)[0])
    except Exception as exc:  # pragma: no cover - platform dependent
        record['note'] += f'; setrlimit failed: {type(exc).__name__}: {exc}'
    return record


def load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise AuditRefusal(f'missing required {label}: {path}')
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise AuditRefusal(f'unreadable {label} at {path}: {exc}') from exc


def load_npz(path: Path, label: str):
    if not path.is_file():
        raise AuditRefusal(f'missing required {label}: {path}')
    try:
        return np.load(path, allow_pickle=False)
    except Exception as exc:
        raise AuditRefusal(f'refusing to load {label} from {path}: {exc}') from exc


def _require_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise AuditRefusal(f'nonfinite values in stored {name}')


def dispersion_D(k: float, water_depth: float) -> float:
    kh = k * water_depth
    t = math.tanh(kh)
    return t + kh * (1 - t * t)


def stable_Z(z: np.ndarray, k: float, water_depth: float) -> np.ndarray:
    """Stable finite-depth vertical mode cosh(k(z+h))/cosh(kh)."""
    return (np.exp(k * z) + np.exp(-k * (z + 2 * water_depth))) / (1 + math.exp(-2 * k * water_depth))


def near_power(force_per_velocity: np.ndarray, rate: np.ndarray) -> float:
    """P_mechanical = -.5 Re(v^H F v) with F per unit peak velocity."""
    return -.5 * float(np.real(np.vdot(rate, force_per_velocity @ rate)))


def boundary_work(weights: np.ndarray, hull: np.ndarray, pressure_v: np.ndarray,
                  trace_v: np.ndarray) -> float:
    """+.5 Re(sum w p conj(g)) on hull panels (frozen recorded convention)."""
    return .5 * float(np.real(np.sum(weights[hull] * pressure_v[hull] * np.conj(trace_v[hull]))))


def actual_flux(weights: np.ndarray, pressure_v: np.ndarray, trace_v: np.ndarray,
                residual_v: np.ndarray) -> float:
    """+.5 Re(sum w p conj(g + r)) over all panels; g is zero on lid panels."""
    return .5 * float(np.real(np.sum(weights * pressure_v * np.conj(trace_v + residual_v))))


def residual_diagnostics(weights: np.ndarray, hull: np.ndarray, pressure_v: np.ndarray,
                          residual_v: np.ndarray) -> dict:
    signed = .5 * np.real(weights * pressure_v * np.conj(residual_v))
    absolute = .5 * weights * np.abs(pressure_v) * np.abs(residual_v)
    return {
        'residual_work_all_w': float(np.sum(signed)),
        'residual_work_hull_w': float(np.sum(signed[hull])),
        'residual_work_lid_w': float(np.sum(signed[~hull])),
        'residual_work_abs_bound_all_w': float(np.sum(absolute)),
        'residual_work_abs_bound_hull_w': float(np.sum(absolute[hull])),
        'node_residual_max_velocity_m_per_s': float(np.max(np.abs(residual_v))),
    }


def panel_mean_fields(weights: np.ndarray, areas: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Area-weighted panel means of per-quadrature fields (count, nq, ...)."""
    return np.einsum('iq,iq...->i...', weights / areas[:, None], values)


def mean_covariance_split(weights: np.ndarray, areas: np.ndarray, pressure_v: np.ndarray,
                          velocity_v: np.ndarray) -> tuple[float, float]:
    """Exact within-panel split of +.5 Re(sum_iq w p conj(u)).

    The pointwise boundary power decomposes exactly into a panel-mean part
    .5 Re(sum_i A_i pbar_i conj(ubar_i)) and a within-panel covariance part
    .5 Re(sum_iq w_iq (p_iq - pbar_i) conj(u_iq - ubar_i)): the weighted-mean
    cross terms vanish per panel, so mean + cov reproduces the pointwise
    power to machine precision for arbitrary complex fields.
    """
    if pressure_v.shape != velocity_v.shape or pressure_v.shape != weights.shape:
        raise AuditRefusal('mean/covariance split requires matching (panel, quadrature) fields')
    _require_finite('mean/covariance pressure', pressure_v)
    _require_finite('mean/covariance velocity', velocity_v)
    normalized = weights / areas[:, None]
    pbar = np.einsum('iq,iq->i', normalized, pressure_v)
    ubar = np.einsum('iq,iq->i', normalized, velocity_v)
    mean_w = .5 * float(np.real(np.sum(areas * pbar * np.conj(ubar))))
    cov_w = .5 * float(np.real(np.sum(weights * (pressure_v - pbar[:, None])
                                      * np.conj(velocity_v - ubar[:, None]))))
    return mean_w, cov_w


def green_defect_decision(far_w: float, green_defect_w: float,
                          mean_persistent_w: float) -> str:
    """Classify the discrete Green energy defect from data, not presumption.

    ``green_defect_w`` is far minus pointwise flux; ``mean_persistent_w`` is
    far minus panel-mean flux. If the pointwise relation closes while the
    panel-mean one does not, within-panel covariance is what closes it; if
    removing the covariance does not even halve the defect, the defect
    persists in the saved-trace Green relation itself. The 1e-9 relative
    scale is a machine-algebra zero for fabricated exact fixtures, not a
    convergence floor or tolerance waiver.
    """
    scale = max(abs(far_w), abs(green_defect_w), abs(mean_persistent_w), 1e-300)
    if abs(green_defect_w) <= 1e-9 * scale:
        if abs(mean_persistent_w) <= 1e-9 * scale:
            return 'discrete_green_relation_closed'
        return 'within_panel_covariance_accounted'
    if abs(mean_persistent_w) < .5 * abs(green_defect_w):
        return 'largely_within_panel_covariance'
    return 'persists_in_saved_trace_green_relation'


def rate_work_adjoint_decomposition(weights: np.ndarray, areas: np.ndarray,
                                    hull: np.ndarray, pressure_v: np.ndarray,
                                    trace_v: np.ndarray, residual_v: np.ndarray,
                                    far_w: float) -> dict:
    """Discrete work/adjoint decomposition of one rate against its far power.

    Recorded-phasor identities (exp(-i omega t), p = i rho omega phi,
    u = g + r, quadrature weights w):
      flux = +.5 Re(sum w p conj(u)) over all panels,
      work = +.5 Re(sum w p conj(g)) over hull panels,
      flux = work + residual work                    (exact, u = g + r),
      flux = flux_panel_mean + flux_covariance        (exact within-panel split),
      far  = -(rho omega/2) Re(v^H M_pole v)          (stored pole; because
             sigma is panel-constant this is already the panel-mean quadrature
             sum_i A_i conj(sigma)(mean_Im(S) sigma)),
      far - flux = (far - flux_panel_mean) - flux_covariance,
      far - work = (far - work_panel_mean) - work_covariance.
    """
    if not (pressure_v.shape == trace_v.shape == residual_v.shape == weights.shape):
        raise AuditRefusal('work/adjoint decomposition requires matching pointwise fields')
    for name, values in (('pressure', pressure_v), ('trace', trace_v),
                         ('residual', residual_v)):
        _require_finite(f'work/adjoint {name}', values)
    velocity_v = trace_v + residual_v
    flux_pt = actual_flux(weights, pressure_v, trace_v, residual_v)
    hull_flux = .5 * float(np.real(np.sum(weights[hull] * pressure_v[hull] * np.conj(velocity_v[hull]))))
    lid_flux = flux_pt - hull_flux
    flux_mean, flux_cov = mean_covariance_split(weights, areas, pressure_v, velocity_v)
    work_pt = boundary_work(weights, hull, pressure_v, trace_v)
    work_mean, work_cov = mean_covariance_split(weights[hull], areas[hull],
                                                pressure_v[hull], trace_v[hull])
    green_defect = far_w - flux_pt
    mean_persistent = far_w - flux_mean
    closure_gap = far_w - work_pt
    closure_mean = far_w - work_mean
    scale = max(abs(far_w), abs(flux_pt), 1e-300)
    return {
        'far_w': float(far_w),
        'flux_pointwise_w': flux_pt,
        'flux_physical_hull_pointwise_w': hull_flux,
        'flux_artificial_lid_pointwise_w': lid_flux,
        'far_minus_physical_hull_flux_w': far_w - hull_flux,
        'physical_boundary_note': 'The computational all-panel functional includes an artificial internal lid and is not automatically the physical exterior-fluid boundary integral.',
        'flux_panel_mean_w': flux_mean,
        'flux_within_panel_covariance_w': flux_cov,
        'work_pointwise_w': work_pt,
        'work_panel_mean_w': work_mean,
        'work_within_panel_covariance_w': work_cov,
        'residual_work_pointwise_w': flux_pt - work_pt,
        'residual_work_panel_mean_w': flux_mean - work_mean,
        'residual_work_within_panel_covariance_w': flux_cov - work_cov,
        'flux_minus_far_w': flux_pt - far_w,
        'green_defect_far_minus_flux_w': green_defect,
        'mean_persistent_defect_far_minus_flux_mean_w': mean_persistent,
        'closure_gap_far_minus_work_w': closure_gap,
        'closure_mean_persistent_w': closure_mean,
        'green_defect_rel_far': _rel(abs(green_defect), scale),
        'mean_persistent_rel_green_defect': _rel(abs(mean_persistent), abs(green_defect)),
        'covariance_abs_ratio_green_defect': _rel(abs(flux_cov), abs(green_defect)),
        'split_flux_identity_max_abs_w': abs(flux_pt - flux_mean - flux_cov),
        'split_work_identity_max_abs_w': abs(work_pt - work_mean - work_cov),
        'green_decomposition_identity_max_abs_w': abs(green_defect - (mean_persistent - flux_cov)),
        'closure_decomposition_identity_max_abs_w': abs(closure_gap - (closure_mean - work_cov)),
        'decision': green_defect_decision(far_w, green_defect, mean_persistent),
    }


def far_mean_powers(points: np.ndarray, weights: np.ndarray, sigma: np.ndarray, *,
                    k: float, rho: float, omega: float, water_depth: float,
                    angular_counts) -> dict:
    """Frozen far formula: H = sum(sigma w Z exp(-ik theta.xy))/(4 pi);
    P_f = 2 pi rho omega k / D * integral_theta |H|^2."""
    kh = k * water_depth
    D = dispersion_D(k, water_depth)
    Z = stable_Z(points[:, :, 2], k, water_depth)
    coeff = sigma[:, None] * weights * Z
    powers = {}
    for count in angular_counts:
        theta = 2 * np.pi * np.arange(count) / count
        H = np.zeros(count, complex)
        for qi in range(points.shape[1]):
            phase = np.exp(-1j * k * (np.cos(theta)[:, None] * points[None, :, qi, 0]
                                      + np.sin(theta)[:, None] * points[None, :, qi, 1]))
            H += phase @ coeff[:, qi] / (4 * np.pi)
        powers[int(count)] = float(2 * np.pi * rho * omega * k / D
                                   * (2 * np.pi / count) * np.sum(np.abs(H) ** 2))
    return powers


def source_pole_power_matrix(far_form: np.ndarray, rho: float, omega: float) -> np.ndarray:
    """Stored Im(S) quadratic form; its real work has no extra factor of pi."""
    return -(rho * omega / 2) * far_form


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values))) if np.size(values) else 0.0


def _rel(diff: float, scale: float) -> float:
    return float(diff / max(scale, 1e-300))


def _restore(recorded, recomputed) -> dict:
    entry = {'recorded': None if recorded is None else float(recorded),
             'recomputed': float(recomputed)}
    if recorded is None:
        entry.update(abs_diff=None, rel_diff=None, note='not recorded in frozen inputs')
        return entry
    diff = abs(entry['recorded'] - entry['recomputed'])
    entry.update(abs_diff=float(diff), rel_diff=_rel(diff, abs(entry['recorded'])))
    return entry


def load_case(case_dir: Path, retained_dofs) -> dict:
    case_json = load_json(case_dir / 'case.json', 'case.json')
    status_json = load_json(case_dir / 'status.json', 'status.json')
    inputs = load_npz(case_dir / 'input.npz', 'input.npz')
    radiation = load_npz(case_dir / 'radiation.npz', 'radiation.npz')
    required_input = ('points', 'weights', 'normals', 'hull_mask', 'rigid_trace', 'vertices', 'faces', 'owner')
    required_radiation = ('sources', 'pressure', 'node_residual', 'mean_residual',
                           'force_per_velocity', 'added_mass', 'radiation_damping', 'dofs')
    for name in required_input + required_radiation:
        if name not in (inputs.files if name in required_input else radiation.files):
            raise AuditRefusal(f'missing array {name} in stored case')
    points = np.asarray(inputs['points'])
    weights = np.asarray(inputs['weights'])
    normals = np.asarray(inputs['normals'])
    hull = np.asarray(inputs['hull_mask'], dtype=bool)
    trace = np.asarray(inputs['rigid_trace'])
    sources = np.asarray(radiation['sources'])
    pressure = np.asarray(radiation['pressure'])
    node_residual = np.asarray(radiation['node_residual'])
    mean_residual = np.asarray(radiation['mean_residual'])
    force = np.asarray(radiation['force_per_velocity'])
    added_mass = np.asarray(radiation['added_mass'])
    damping = np.asarray(radiation['radiation_damping'])
    dofs = [str(d) for d in np.asarray(radiation['dofs']).tolist()]
    if points.ndim != 3 or points.shape[-1] != 3:
        raise AuditRefusal('stored points must have shape (panels, quadrature, 3)')
    if np.asarray(radiation['dofs']).ndim != 1 or len(set(dofs)) != len(dofs):
        raise AuditRefusal('stored DOFs must be a unique one-dimensional list')
    count, nq, _ = points.shape
    ndof = len(dofs)
    if (weights.shape != (count, nq) or normals.shape != (count, 3)
            or hull.shape != (count,) or trace.shape != (count, nq, ndof)
            or sources.shape != (count, ndof) or pressure.shape != (count, nq, ndof)
            or node_residual.shape != (count, nq, ndof)
            or mean_residual.shape != (count, ndof)
            or force.shape != (ndof, ndof) or added_mass.shape != (ndof, ndof)
            or damping.shape != (ndof, ndof) or nq < 1 or count < 1 or ndof < 1):
        raise AuditRefusal('stored case array shapes are inconsistent with each other')
    if np.any(weights <= 0):
        raise AuditRefusal('stored quadrature weights must be positive')
    for name, values in (('points', points), ('weights', weights), ('normals', normals),
                         ('rigid_trace', trace), ('sources', sources), ('pressure', pressure),
                         ('node_residual', node_residual), ('mean_residual', mean_residual),
                         ('force_per_velocity', force), ('added_mass', added_mass),
                         ('radiation_damping', damping)):
        _require_finite(name, values)
    retained = tuple(int(d) for d in retained_dofs)
    if not retained or len(set(retained)) != len(retained) or any(d < 0 or d >= ndof for d in retained):
        raise AuditRefusal(f'retained DOF indices {retained} invalid for {ndof} stored DOFs')
    snapshot = case_json.get('snapshot_arrays', {})
    snapshot_match = {}
    for name, values in (('vertices', inputs['vertices']), ('faces', inputs['faces']),
                         ('owner', inputs['owner']), ('hull_mask', inputs['hull_mask']),
                         ('points', points), ('weights', weights), ('normals', normals),
                         ('rigid_trace', trace)):
        recorded = snapshot.get(name, {}).get('sha256')
        snapshot_match[name] = None if recorded is None else recorded == sha256_array(values)
    if any(v is False for v in snapshot_match.values()):
        raise AuditRefusal('stored input snapshot identity mismatch')
    return {
        'case_json': case_json, 'status_json': status_json,
        'points': points, 'weights': weights, 'normals': normals, 'hull': hull,
        'trace': trace, 'sources': sources, 'pressure': pressure,
        'node_residual': node_residual, 'mean_residual': mean_residual,
        'force': force, 'added_mass': added_mass, 'damping': damping,
        'dofs': dofs, 'count': count, 'nq': nq, 'ndof': ndof,
        'retained': retained, 'snapshot_match': snapshot_match,
        'vertices': np.asarray(inputs['vertices']), 'faces': np.asarray(inputs['faces']),
        'owner': np.asarray(inputs['owner']),
    }


def parse_reference_blocks(energy_comparison: dict) -> dict:
    """Extract frozen 4x4 (retained) candidate/reference A and B blocks."""
    blocks = {}
    for block, key in (('added_mass', 'added_mass'), ('radiation_damping', 'radiation_damping')):
        coefficients = energy_comparison.get(key, {}).get('coefficients', [])
        if not coefficients:
            raise AuditRefusal(f'energy-comparison.json has no {key} coefficients')
        positions=[(c['row'],c['column']) for c in coefficients]
        if any(not isinstance(i,int) or isinstance(i,bool) or i < 0 for pos in positions for i in pos):
            raise AuditRefusal('reference matrix indices must be nonnegative integers')
        size = max(max(pos) for pos in positions) + 1
        if len(set(positions)) != len(positions) or set(positions) != {(i,j) for i in range(size) for j in range(size)}:
            raise AuditRefusal('reference matrix coefficients must be complete and unique')
        candidate = np.zeros((size, size))
        reference = np.zeros((size, size))
        for c in coefficients:
            candidate[c['row'], c['column']] = c['candidate']
            reference[c['row'], c['column']] = c['reference']
        _require_finite(f'{block} candidate',candidate)
        _require_finite(f'{block} reference',reference)
        blocks[block] = {'candidate': candidate, 'reference': reference}
    return blocks


def audit(case_dir: Path, parent_dir: Path, *, retained_dofs, angular_counts,
          verify_native: bool, rlimit_mib: int, candidate_source: Path | None) -> dict:
    started = time.perf_counter()
    if rlimit_mib <= 0 or rlimit_mib > DEFAULT_RLIMIT_MIB:
        raise AuditRefusal('virtual limit must be positive and at most 1536 MiB')
    rlimit = apply_virtual_limit(rlimit_mib)
    if platform.system() == 'Linux' and not rlimit['applied']:
        raise AuditRefusal('required Linux virtual address-space cap unavailable')
    case_dir = case_dir.resolve()
    parent_dir = parent_dir.resolve()
    if not case_dir.is_dir() or not parent_dir.is_dir():
        raise AuditRefusal('case directory and parent directory must exist')
    case = load_case(case_dir, retained_dofs)
    summary = load_json(parent_dir / 'summary.json', 'parent summary.json')
    comparison = load_json(parent_dir / 'energy-comparison.json', 'parent energy-comparison.json')
    blocks_path = parent_dir / 'retained-blocks.npz'
    blocks = load_npz(blocks_path, 'parent retained-blocks.npz') if blocks_path.is_file() else None

    points, weights, hull, trace = case['points'], case['weights'], case['hull'], case['trace']
    sources, pressure = case['sources'], case['pressure']
    node_residual, mean_residual = case['node_residual'], case['mean_residual']
    force, dofs, ndof = case['force'], case['dofs'], case['ndof']
    count, nq = case['count'], case['nq']
    if np.any(trace[~hull] != 0):
        raise AuditRefusal('stored rigid trace must be zero on artificial lid panels')
    retained = case['retained']
    areas = weights.sum(axis=1)

    candidate = summary.get('candidate', {})
    rate_list = candidate.get('rate')
    if not isinstance(rate_list, list) or len(rate_list) != ndof:
        raise AuditRefusal(f'parent summary rate must list {ndof} entries')
    rate = np.asarray(rate_list, dtype=float)
    if np.any(rate[[d for d in range(ndof) if d not in retained]] != 0) or not np.any(rate):
        raise AuditRefusal('frozen rate must be nonzero and use only retained DOFs')
    if not np.all(np.isfinite(rate)):
        raise AuditRefusal('parent summary rate must be finite')
    omega = float(candidate.get('omega', case['case_json'].get('omega', 0.0)))
    rho = float(candidate.get('rho', case['case_json'].get('rho', 0.0)))
    water_depth = float(candidate.get('water_depth', case['case_json'].get('water_depth', 0.0)))
    k = float(candidate.get('k', 0.0))
    gravity = float(case['case_json'].get('gravity', 9.81))
    if not all(math.isfinite(v) for v in (omega,rho,water_depth,k,gravity)) or gravity <= 0 or omega <= 0 or rho <= 0 or water_depth <= 0 or k <= 0:
        raise AuditRefusal('omega, rho, water_depth and k must be positive in frozen inputs')
    dispersion_residual = abs(k * math.tanh(k * water_depth) - omega ** 2 / gravity)

    # ---- file identity -------------------------------------------------
    file_inventory = {}
    for path in sorted([case_dir / 'input.npz', case_dir / 'radiation.npz',
                        case_dir / 'case.json', case_dir / 'status.json',
                        parent_dir / 'summary.json', parent_dir / 'energy-comparison.json']
                       + [case_dir / f'operator-{qi}.npz' for qi in range(nq)]
                       + ([case_dir / f'native-operator-{qi}.npz' for qi in range(nq)]
                          if verify_native else [])
                       + ([blocks_path] if blocks is not None else [])):
        rel = path.relative_to(parent_dir if path.is_relative_to(parent_dir) else case_dir)
        prefix = '' if path.is_relative_to(parent_dir) else 'case/'
        file_inventory[prefix + str(rel)] = {'bytes': path.stat().st_size,
                                             'sha256': sha256_file(path)}
    source_identity = {'path': None, 'sha256': None,
                       'case_module_sha256': case['case_json'].get('module_sha256'),
                       'match': None}
    if candidate_source is not None and Path(candidate_source).is_file():
        source_identity.update(path=str(candidate_source), sha256=sha256_file(Path(candidate_source)))
        source_identity['match'] = (source_identity['sha256'] == source_identity['case_module_sha256'])
    case_hash = case['case_json'].get('case_hash')
    summary_case_hash = candidate.get('case_hash')

    if case_hash is None or summary_case_hash is None or case_hash != summary_case_hash:
        raise AuditRefusal('case/summary identity missing or mismatched')
    if source_identity['match'] is False:
        raise AuditRefusal('candidate source identity mismatch')

    # ---- operator reconstruction (sequential loads) ---------------------
    mean_k = np.zeros((count, count), complex)
    far_form = np.zeros((ndof, ndof), complex)
    summed_weighted_s = np.zeros((count, count), complex)
    per_quadrature = []
    native_per_quadrature = []
    s_reciprocity = []
    for qi in range(nq):
        operator = load_npz(case_dir / f'operator-{qi}.npz', f'operator-{qi}.npz')
        s = np.asarray(operator['s'])
        kmat = np.asarray(operator['k'])
        if s.shape != (count, count) or kmat.shape != (count, count):
            raise AuditRefusal(f'operator-{qi}.npz shapes {s.shape}/{kmat.shape} '
                               f'inconsistent with {count} panels')
        _require_finite(f'operator-{qi} s', s)
        _require_finite(f'operator-{qi} k', kmat)
        pressure_recon = 1j * rho * omega * (s @ sources)
        residual_recon = kmat @ sources - trace[:, qi, :]
        per_quadrature.append({
            'qi': qi,
            'pressure_max_abs_diff': _max_abs(pressure_recon - pressure[:, qi, :]),
            'pressure_rel_diff': _rel(_max_abs(pressure_recon - pressure[:, qi, :]),
                                      _max_abs(pressure[:, qi, :])),
            'node_residual_max_abs_diff': _max_abs(residual_recon - node_residual[:, qi, :]),
            'node_residual_rel_diff': _rel(_max_abs(residual_recon - node_residual[:, qi, :]),
                                           _max_abs(node_residual[:, qi, :])),
        })
        mean_k += np.einsum('i,ij->ij', weights[:, qi] / areas, kmat)
        far_form += np.einsum('i,id,ie->de', weights[:, qi], sources.conj(),
                              s.imag @ sources)
        weighted_s = s * weights[:, qi][:, None]
        summed_weighted_s += weighted_s
        s_reciprocity.append({
            'qi': qi,
            'weighted_s_asym_rel': _rel(_max_abs(weighted_s - weighted_s.T),
                                        _max_abs(weighted_s)),
            'weighted_s_imag_asym_rel': _rel(_max_abs(weighted_s.imag - weighted_s.imag.T),
                                             _max_abs(weighted_s.imag)),
            'note': 'stencil information only: the target quadrature node is '
                    'fixed while the source integrates all nodes, so '
                    'per-quadrature asymmetry is not a reciprocity failure '
                    'claim; the meaningful same-space measure is the '
                    'quadrature sum Sum_q diag(w_q) S_q',
        })
        del weighted_s
        if verify_native:
            native = load_npz(case_dir / f'native-operator-{qi}.npz', f'native-operator-{qi}.npz')
            s_native = np.asarray(native['s'])
            grad = np.asarray(native['gradient'])
            if s_native.shape != (count, count) or grad.shape != (3, count, count):
                raise AuditRefusal(f'native-operator-{qi}.npz shapes inconsistent')
            k_real_recon = np.einsum('dim,id->im', grad.real, case['normals'])
            k_real_recon[np.arange(count), np.arange(count)] += .5
            native_per_quadrature.append({
                'qi': qi,
                's_real_max_abs_diff': _max_abs(s.real - s_native.real),
                'k_real_trace_max_abs_diff': _max_abs(kmat.real - k_real_recon),
                'k_real_trace_rel_diff': _rel(_max_abs(kmat.real - k_real_recon),
                                              _max_abs(kmat.real)),
                'note': 'imaginary pole not re-derivable without a new kernel evaluation',
            })
            del native, s_native, grad
        del operator, s, kmat, pressure_recon, residual_recon
    summed_s_reciprocity = {
        'weighted_s_summed_asym_rel':
            _rel(_max_abs(summed_weighted_s - summed_weighted_s.T),
                 _max_abs(summed_weighted_s)),
        'weighted_s_summed_imag_asym_rel':
            _rel(_max_abs(summed_weighted_s.imag - summed_weighted_s.imag.T),
                 _max_abs(summed_weighted_s.imag)),
        'note': 'meaningful same-space reciprocity measure Sum_q diag(w_q) S_q; '
                'per-quadrature entries are stencil info only',
    }
    del summed_weighted_s
    mean_rhs = np.einsum('iq,iqd->id', weights / areas[:, None], trace)
    mean_residual_recon = mean_k @ sources - mean_rhs
    force_recon = -np.einsum('iq,iqd,iqe->de', weights[hull], trace[hull], pressure[hull])
    operator_reconstruction = {
        'per_quadrature': per_quadrature,
        'mean_residual_max_abs_diff': _max_abs(mean_residual_recon - mean_residual),
        'mean_residual_rel_diff': _rel(_max_abs(mean_residual_recon - mean_residual),
                                      _max_abs(mean_rhs)),
        'force_max_abs_diff': _max_abs(force_recon - force),
        'force_rel_diff': _rel(_max_abs(force_recon - force), _max_abs(force)),
        'added_mass_max_abs_diff': _max_abs(force_recon.imag / omega - case['added_mass']),
        'radiation_damping_max_abs_diff': _max_abs(-force_recon.real - case['damping']),
        'mean_operator_note': 'mean operator accumulated from stored assembled traces; '
                              'no new solve performed',
    }
    if verify_native:
        operator_reconstruction['native'] = {
            'per_quadrature': native_per_quadrature,
            'note': 'real trace and +0.5 self jump verified against stored native integrals; '
                    'imaginary pole quadratic form cross-checked only through the shared angular far identity',
        }
    del mean_k

    # ---- frozen-rate and per-DOF energy --------------------------------
    pressure_v = pressure @ rate
    trace_v = trace @ rate
    residual_v = node_residual @ rate
    sigma_v = sources @ rate
    near = near_power(force, rate)
    work = boundary_work(weights, hull, pressure_v, trace_v)
    flux = actual_flux(weights, pressure_v, trace_v, residual_v)
    rate_diagnostics = residual_diagnostics(weights, hull, pressure_v, residual_v)
    rate_diagnostics['mean_residual_max_velocity_m_per_s'] = float(np.max(np.abs(mean_residual @ rate)))
    far_rate = far_mean_powers(points, weights, sigma_v, k=k, rho=rho, omega=omega,
                               water_depth=water_depth, angular_counts=angular_counts)
    far_closed_matrix = source_pole_power_matrix(far_form, rho, omega)
    far_closed_rate = float(np.real(np.vdot(rate, far_closed_matrix @ rate)))
    hull_only_far = far_mean_powers(points, weights, np.where(hull, sigma_v, 0.0), k=k,
                                    rho=rho, omega=omega, water_depth=water_depth,
                                    angular_counts=angular_counts)
    largest_count = max(angular_counts)
    per_dof = []
    near_work_identity = 0.0
    for d in retained:
        unit = np.zeros(ndof)
        unit[d] = 1.0
        p_d = pressure @ unit
        g_d = trace @ unit
        r_d = node_residual @ unit
        near_d = near_power(force, unit)
        work_d = boundary_work(weights, hull, p_d, g_d)
        near_work_identity = max(near_work_identity, abs(near_d - work_d))
        far_d = far_mean_powers(points, weights, sources[:, d], k=k, rho=rho, omega=omega,
                                water_depth=water_depth, angular_counts=(largest_count,))
        far_d_closed = float(np.real(far_closed_matrix[d, d]))
        diag = residual_diagnostics(weights, hull, p_d, r_d)
        hull_only_d = far_mean_powers(points, weights, np.where(hull, sources[:, d], 0.0),
                                      k=k, rho=rho, omega=omega, water_depth=water_depth,
                                      angular_counts=(largest_count,))
        per_dof.append({
            'dof_index': d, 'dof': dofs[d],
            'near_w': near_d, 'boundary_work_w': work_d,
            'far_w': {str(largest_count): far_d[largest_count]},
            'far_closed_form_w': far_d_closed,
            'residual_work_all_w': diag['residual_work_all_w'],
            'residual_work_lid_w': diag['residual_work_lid_w'],
            'node_residual_max_velocity_m_per_s': diag['node_residual_max_velocity_m_per_s'],
            'gap_far_minus_near_w': far_d[largest_count] - near_d,
            'hull_only_far_power_ratio': (float(hull_only_d[largest_count] / far_d[largest_count])
                                 if far_d[largest_count] > 0 else None),
        })
    energy = {
        'rate_vector': [float(v) for v in rate],
        'retained_rate': [float(rate[d]) for d in retained],
        'near_w': near,
        'boundary_work_w': work,
        'force_work_error_w': abs(near - work),
        'near_work_identity_max_abs_w': near_work_identity,
        'far_w': {str(c): far_rate[c] for c in angular_counts},
        'far_closed_form_w': far_closed_rate,
        'far_closed_vs_largest_rel_diff': _rel(abs(far_closed_rate - far_rate[largest_count]),
                                               abs(far_rate[largest_count])),
        'actual_flux_w': flux,
        'flux_minus_mech_minus_residual_work_w': flux - work
        - rate_diagnostics['residual_work_all_w'],
        'hull_only_far_w': {str(c): hull_only_far[c] for c in angular_counts},
        'hull_only_far_power_ratio': (float(hull_only_far[largest_count] / far_rate[largest_count])
                             if far_rate[largest_count] > 0 else None),
        'closure_far_minus_mech_w': far_rate[largest_count] - near,
        'closure_far_minus_flux_w': far_rate[largest_count] - flux,
        'per_dof': per_dof,
        **{f'rate_{key}': value for key, value in rate_diagnostics.items()},
    }

    # ---- reciprocity, cancellation, eigen diagnostics ------------------
    def asym_rel(matrix):
        return _rel(_max_abs(matrix - matrix.T), _max_abs(matrix))

    retained_idx = list(retained)
    B_ret = case['damping'][np.ix_(retained_idx, retained_idx)]
    A_ret = case['added_mass'][np.ix_(retained_idx, retained_idx)]
    F_ret = force[np.ix_(retained_idx, retained_idx)]
    v_ret = rate[retained_idx]
    blocks_parsed = parse_reference_blocks(comparison)
    A_ref = blocks_parsed['added_mass']['reference']
    B_ref = blocks_parsed['radiation_damping']['reference']
    B_cand_blocks = blocks_parsed['radiation_damping']['candidate']
    A_cand_blocks = blocks_parsed['added_mass']['candidate']
    if B_ref.shape != (len(retained_idx), len(retained_idx)):
        raise AuditRefusal('reference blocks do not match the retained DOF count')
    candidate_block_diff = _max_abs(B_cand_blocks - B_ret)
    for j, entry in enumerate(per_dof):
        reference = .5 * float(B_ref[j,j])
        entry['hams_reference_unit_power_w'] = reference
        entry['near_minus_hams_unit_rel_diff'] = _rel(abs(entry['near_w']-reference),abs(reference))
        entry['far_minus_near_rel_diff'] = _rel(abs(entry['gap_far_minus_near_w']),abs(entry['near_w']))
    energy['hull_only_far_power_note'] = 'Hull-only source deletion is a diagnostic counterfactual. Interference prevents an additive lid power fraction.'

    def cancellation(matrix, vector):
        gross = .5 * float(np.sum(np.abs(np.outer(vector, vector) * matrix)))
        net = .5 * float(vector @ matrix @ vector)
        return {'gross_abs_w': gross, 'net_w': net,
                'cancellation_ratio': float(gross / abs(net)) if net != 0 else None}

    eigen_candidate = np.linalg.eigvalsh(.5 * (B_ret + B_ret.T))
    eigen_reference = np.linalg.eigvalsh(.5 * (B_ref + B_ref.T))
    rayleigh = float(v_ret @ B_ret @ v_ret) / float(np.vdot(v_ret, v_ret))

    # ---- discrete work/adjoint diagnosis -------------------------------
    velocity_fields = trace + node_residual
    u_mean_fields = panel_mean_fields(weights, areas, velocity_fields)
    g_mean_fields = panel_mean_fields(weights, areas, trace)
    p_mean_fields = panel_mean_fields(weights, areas, pressure)
    mean_enforcement_max = _max_abs(u_mean_fields - g_mean_fields)
    hull_weights = np.where(hull[:, None], weights, 0.0)
    # Motion-coordinate convention: A[e,d] = sum_iq w conj(G[i,e]) P[i,d],
    # so mean power = .5 Re(v^H A v) for arbitrary complex velocity phasors.
    # The pressure-first stencil M[d,e] = sum_iq w P[i,d] conj(G[i,e]) is the
    # TRANSPOSE of A; exported matrices are motion-coordinate.
    a_flux = np.einsum('iq,iqe,iqd->ed', weights, np.conj(velocity_fields), pressure)
    a_work = np.einsum('iq,iqe,iqd->ed', hull_weights, np.conj(trace), pressure)
    a_res = a_flux - a_work
    a_flux_mean = np.einsum('ie,id->ed', np.conj(u_mean_fields),
                            areas[:, None] * p_mean_fields)
    a_work_mean = np.einsum('ie,id->ed', np.conj(g_mean_fields[hull]),
                            areas[hull][:, None] * p_mean_fields[hull])
    a_flux_cov = a_flux - a_flux_mean
    a_work_cov = a_work - a_work_mean
    # F = -G^T W P is the correctly oriented work matrix for the real stored
    # rigid trace: force + A_work vanishes (A_work equals M_work^T of the
    # pressure-first stencil).
    adjoint_identity = _max_abs(force + a_work)

    def _block(matrix):
        return matrix[np.ix_(retained_idx, retained_idx)]

    def _complex_block(matrix):
        matrix = np.asarray(matrix)
        return {'real': matrix.real.tolist(), 'imag': matrix.imag.tolist()}

    flux_ret = _block(a_flux)
    active_flux_ret = .5 * (flux_ret + flux_ret.conj().T)
    reactive_flux_ret = .5 * (flux_ret - flux_ret.conj().T)
    # Full mean-power Hermitian matrix: the .5 of .5 Re(v^H A v) enters before
    # Hermitianization, so these eigenvalues are mean-power scale (half the
    # work-form eigenvalues).
    mean_power_flux_ret = .5 * active_flux_ret
    pole_ret = _block(far_form)
    pole_power_ret = source_pole_power_matrix(pole_ret, rho, omega)
    # The pole form Sigma^H W Im(S) Sigma is already motion-coordinate and is
    # NOT transposed with the pressure-first flux matrices; -(rho omega/2)
    # already carries the mean-power .5.
    active_pole_power_ret = .5 * (pole_power_ret + pole_power_ret.conj().T)
    far_form_hermitian_asym = _rel(_max_abs(far_form - far_form.conj().T),
                                   _max_abs(far_form))
    far_unit = {entry['dof_index']: list(entry['far_w'].values())[0] for entry in per_dof}
    frozen_decomp = rate_work_adjoint_decomposition(
        weights, areas, hull, pressure_v, trace_v, residual_v, far_rate[largest_count])
    per_dof_decomp = []
    for j, d in enumerate(retained):
        decomp = rate_work_adjoint_decomposition(
            weights, areas, hull, pressure[:, :, d], trace[:, :, d],
            node_residual[:, :, d], far_unit[d])
        per_dof_decomp.append(decomp)
        for key in ('flux_pointwise_w', 'flux_panel_mean_w',
                    'flux_within_panel_covariance_w',
                    'work_within_panel_covariance_w',
                    'green_defect_far_minus_flux_w',
                    'mean_persistent_defect_far_minus_flux_mean_w'):
            per_dof[j][key] = decomp[key]
        per_dof[j]['work_adjoint_decision'] = decomp['decision']
    decisions = [frozen_decomp['decision']] + [e['decision'] for e in per_dof_decomp]
    persists_count = sum(1 for item in decisions
                         if item == 'persists_in_saved_trace_green_relation')
    if all(item == 'discrete_green_relation_closed' for item in decisions):
        overall_decision = 'discrete_green_relation_closed'
    elif persists_count == len(decisions):
        overall_decision = 'persists_in_saved_trace_green_relation'
    elif persists_count == 0:
        overall_decision = 'within_panel_covariance_accounted'
    else:
        overall_decision = 'mixed'
    discrete_work = {
        'identities': [
            'flux = +.5 Re(sum_iq w p conj(g+r)) over all panels; work = '
            '+.5 Re(sum_iq w p conj(g)) over hull; flux = work + residual work exactly',
            'flux = flux_panel_mean + flux_within_panel_covariance exactly; the '
            'weighted-mean cross terms vanish per panel',
            'motion-coordinate projected matrices A[e,d] = sum_iq w conj(G[i,e]) '
            'P[i,d] carry mean power .5 Re(v^H A v) for arbitrary complex '
            'velocity phasors; the pressure-first stencil sum_d P conj(G_e) '
            'is the transpose of A',
            'far = -(rho omega/2) Re(v^H M_pole v) from the stored Im(S) with '
            'pointwise target weights and panel-constant sigma; because sigma is '
            'panel-constant this is already the panel-mean quadrature '
            'sum_i A_i conj(sigma)(mean_Im(S) sigma); the pole form is already '
            'motion-coordinate and is not transposed',
            'far - flux = (far - flux_panel_mean) - flux_within_panel_covariance',
            'far - work = (far - work_panel_mean) - work_within_panel_covariance',
            'force adjoint structure: F = -A_work exactly for the real stored '
            'rigid trace (A_work the motion-coordinate work matrix; the '
            'unconjugated force pairing vs the conjugated power pairing)',
            'full mean-power Hermitian matrix .5 (A + A^H)/2: the .5 enters '
            'before Hermitianization, so its eigenvalues are mean-power scale',
        ],
        'mean_enforcement_max_abs_velocity_m_per_s': mean_enforcement_max,
        'adjoint_force_transpose_identity_max_abs': adjoint_identity,
        'weighted_s_stencil_asymmetry_per_quadrature': s_reciprocity,
        'weighted_s_summed_reciprocity': summed_s_reciprocity,
        'far_form_hermitian_asym_rel': far_form_hermitian_asym,
        'projected_retained': {
            'matrix_convention': 'motion-coordinate A[e,d] = sum_iq w conj(G[i,e]) '
                                 'P[i,d]; mean power = .5 Re(v^H A v); the pole '
                                 'form Sigma^H W Im(S) Sigma is already '
                                 'motion-coordinate and is not transposed',
            'flux_pointwise': _complex_block(flux_ret),
            'flux_panel_mean': _complex_block(_block(a_flux_mean)),
            'flux_within_panel_covariance': _complex_block(_block(a_flux_cov)),
            'work_pointwise': _complex_block(_block(a_work)),
            'work_panel_mean': _complex_block(_block(a_work_mean)),
            'work_within_panel_covariance': _complex_block(_block(a_work_cov)),
            'residual_pointwise': _complex_block(_block(a_res)),
            'pole_form': _complex_block(pole_ret),
            'flux_mean_power_matrix_hermitian_eigenvalues':
                [float(v) for v in np.linalg.eigvalsh(mean_power_flux_ret)],
            'reactive_power_matrix_max_abs': _max_abs(reactive_flux_ret),
            'pole_mean_power_matrix_hermitian_eigenvalues':
                [float(v) for v in np.linalg.eigvalsh(active_pole_power_ret)],
            'normalization_note': 'flux/work eigenvalues include the .5 of '
                                  '.5 Re(v^H A v) before Hermitianization '
                                  '(mean-power scale, W/(m/s)^2); pole '
                                  'eigenvalues are Hermitian parts of '
                                  '-(rho omega/2) Sigma^H W Im(S) Sigma, '
                                  'already mean-power scale',
            'note': 'retained 4x4 projected diagnostics only; not source qualification',
        },
        'frozen_rate': frozen_decomp,
        'per_dof': per_dof_decomp,
        'decision_summary': {
            'overall': overall_decision,
            'frozen_rate_decision': frozen_decomp['decision'],
            'per_dof_decisions': [e['decision'] for e in per_dof_decomp],
            'note': 'classification compares |far - flux_panel_mean| against '
                    '|far - flux_pointwise|; covariance-accounted means the '
                    'panel-mean persistent defect is below half the pointwise '
                    'defect, machine-zero cases excepted',
        },
    }
    del velocity_fields, u_mean_fields, g_mean_fields, p_mean_fields
    del a_flux, a_work, a_res, a_flux_mean, a_work_mean, a_flux_cov, a_work_cov

    hams_power = .5 * float(v_ret @ B_ref @ v_ret)
    mean_power = comparison.get('mean_power', {})
    diagnostics = comparison.get('diagnostics_only', {})
    recorded_far = diagnostics.get('far_source_pole_w', {})

    restorations = {
        'near_mean_power_w': _restore(candidate.get('near_mean_power_w'), near),
        'mechanical_node_work_w': _restore(diagnostics.get('mechanical_node_work_w'), work),
        'force_work_error_w': _restore(diagnostics.get('force_work_error_w'), abs(near - work)),
        'mean_power_candidate_w': _restore(mean_power.get('candidate_w'), near),
        'mean_power_reference_w': _restore(mean_power.get('reference_w'), hams_power),
        'node_residual_max_velocity_m_per_s': _restore(
            diagnostics.get('node_residual_max_w'),
            rate_diagnostics['node_residual_max_velocity_m_per_s']),
        'mean_residual_max_velocity_m_per_s': _restore(
            diagnostics.get('mean_residual_max_w'),
            rate_diagnostics['mean_residual_max_velocity_m_per_s']),
        'node_residual_pressure_work_w': _restore(
            diagnostics.get('node_residual_pressure_work_w'),
            rate_diagnostics['residual_work_all_w']),
        'hull_node_residual_pressure_work_w': _restore(
            diagnostics.get('hull_node_residual_pressure_work_w'),
            rate_diagnostics['residual_work_hull_w']),
        'node_residual_pressure_work_abs_bound_w': _restore(
            diagnostics.get('node_residual_pressure_work_abs_bound_w'),
            rate_diagnostics['residual_work_abs_bound_all_w']),
        'hull_node_residual_pressure_work_abs_bound_w': _restore(
            diagnostics.get('hull_node_residual_pressure_work_abs_bound_w'),
            rate_diagnostics['residual_work_abs_bound_hull_w']),
    }
    for c in angular_counts:
        restorations[f'far_source_pole_w_{c}'] = _restore(recorded_far.get(str(c)), far_rate[c])

    retained_blocks_record = None
    if blocks is not None:
        retained_blocks_record = {
            'rate_max_abs_diff': _max_abs(np.asarray(blocks['rate']) - rate),
            'damping_max_abs_diff': _max_abs(np.asarray(blocks['radiation_damping']) - B_ret),
            'added_mass_max_abs_diff': _max_abs(np.asarray(blocks['added_mass']) - A_ret),
            'force_max_abs_diff': _max_abs(np.asarray(blocks['force_per_velocity']) - F_ret),
        }

    results = {
        'format': FORMAT,
        'status': 'completed',
        'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'tool': {'sha256': _sha256_bytes(Path(__file__).read_bytes()),
                 'numpy_version': np.__version__, 'python': platform.python_version(),
                 'platform': platform.platform()},
        'limits': {'rlimit': rlimit,
                   'thread_env': {name: os.environ.get(name) for name in
                                  ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                   'NUMEXPR_NUM_THREADS')},
                   'no_green_evaluations': True, 'no_linear_solves': True,
                   'qualification': False},
        'inputs': {
            'case_dir': str(case_dir), 'parent_dir': str(parent_dir),
            'files': file_inventory,
            'case_hash': case_hash, 'summary_case_hash': summary_case_hash,
            'case_hash_match': None if case_hash is None or summary_case_hash is None
            else case_hash == summary_case_hash,
            'snapshot_array_hashes_match': case['snapshot_match'],
            'candidate_source': source_identity,
            'status_state': case['status_json'].get('state'),
            'dofs': dofs, 'retained_dofs': [dofs[d] for d in retained],
            'retained_dof_indices': [int(d) for d in retained],
            'retained_blocks': retained_blocks_record,
        },
        'conventions': {
            'phasor': 'exp(-i omega t)', 'velocity_amplitude': 'peak',
            'force_per_velocity': 'F = -B + i omega A',
            'mechanical_power': 'P = -.5 Re(v^H F v)',
            'boundary_work': '+.5 Re(sum w p conj(g)) on hull',
            'far_formula': 'H = sum(sigma w Z exp(-i k theta.xy))/(4 pi); '
                           'P_f = 2 pi rho omega k / D * integral |H|^2',
            'far_closed_form': 'P_f_inf = -(rho omega / 2) * REAL sum_iq w conj(sigma) '
                               '(s.imag @ sigma) from the stored pole',
            'omega': omega, 'k': k, 'rho': rho, 'water_depth': water_depth,
            'gravity': gravity, 'dispersion_residual': dispersion_residual,
            'angular_counts': [int(c) for c in angular_counts],
            'hams_omega_note': 'HAMS native omega 1.49974386 maps to 1.5 via gravity '
                               'similarity 9.80665->9.81 preserving omega^2/g',
        },
        'machine_restorations': restorations,
        'operator_reconstruction': operator_reconstruction,
        'energy': energy,
        'discrete_work_adjoint': discrete_work,
        'cancellation': {
            'candidate_retained': cancellation(B_ret, v_ret),
            'hams_reference_retained': cancellation(B_ref, v_ret),
        },
        'reciprocity': {
            'candidate_12dof': {'force_asym_rel': asym_rel(force),
                                'damping_asym_rel': asym_rel(case['damping']),
                                'added_mass_asym_rel': asym_rel(case['added_mass'])},
            'candidate_retained': {'force_asym_rel': asym_rel(F_ret),
                                   'damping_asym_rel': asym_rel(B_ret),
                                   'added_mass_asym_rel': asym_rel(A_ret)},
            'hams_reference_retained': {'damping_asym_rel': asym_rel(B_ref),
                                        'added_mass_asym_rel': asym_rel(A_ref)},
            'note': 'asymmetry is relative to the max absolute entry; real rate vectors '
                    'are insensitive to the antisymmetric part',
        },
        'eigen_diagnostic': {
            'candidate_retained_sym_eigenvalues': [float(v) for v in eigen_candidate],
            'hams_reference_retained_sym_eigenvalues': [float(v) for v in eigen_reference],
            'rayleigh_quotient_candidate': rayleigh,
            'min_eigenvalue_candidate': float(eigen_candidate[0]),
            'note': 'diagnostic only on the retained 4x4 blocks; not source qualification',
        },
        'hams_comparison': {
            'recorded_reference_w': mean_power.get('reference_w'),
            'recomputed_reference_w': hams_power,
            'recorded_candidate_w': mean_power.get('candidate_w'),
            'recomputed_candidate_w': near,
            'candidate_block_vs_radiation_max_abs_diff': candidate_block_diff,
            'candidate_decomposition': cancellation(B_ret, v_ret),
            'reference_decomposition': cancellation(B_ref, v_ret),
        },
        'honest_limits': list(HONEST_LIMITS),
    }
    results['interpretation'] = interpret(results)
    results['runtime'] = {'elapsed_seconds': time.perf_counter() - started}
    return results


def interpret(results: dict) -> dict:
    """Data-backed interpretation; no presupposed defect mechanism."""
    restorations = results['machine_restorations']
    worst_restore = max((entry['rel_diff'] for entry in restorations.values()
                         if entry['rel_diff'] is not None), default=float('inf'))
    recon = results['operator_reconstruction']
    assembly_rel = max([q['pressure_rel_diff'] for q in recon['per_quadrature']]
                       + [q['node_residual_rel_diff'] for q in recon['per_quadrature']]
                       + [recon['mean_residual_rel_diff'], recon['force_rel_diff']])
    native_rel = None
    if 'native' in recon:
        native_rel = max([q['k_real_trace_rel_diff'] for q in recon['native']['per_quadrature']]
                         + [q['s_real_max_abs_diff'] for q in recon['native']['per_quadrature']])
    energy = results['energy']
    cancellation = results['cancellation']
    candidate_ratio = cancellation['candidate_retained']['cancellation_ratio']
    reference_ratio = cancellation['hams_reference_retained']['cancellation_ratio']
    gaps = [(entry['dof'], entry['gap_far_minus_near_w']) for entry in energy['per_dof']]
    dominant_dof, dominant_gap = max(gaps, key=lambda item: abs(item[1]))
    closure = energy['closure_far_minus_mech_w']
    residual_share = (energy['rate_residual_work_all_w'] / closure if closure != 0 else None)
    findings = [
        f'machine restorations: worst relative difference {worst_restore:.3e} '
        f'(recorded near/far/force-work/residual diagnostics reproduced from stored fields)',
        f'operator reconstruction: worst relative difference {assembly_rel:.3e} '
        '(pressure, node residual, mean residual and force rebuilt from stored assembled '
        'operators without any new solve)',
    ]
    if native_rel is not None:
        findings.append(f'native real-trace/self-jump verification: worst relative difference '
                        f'{native_rel:.3e} (real normal-trace assembly and +0.5 jump consistent '
                        'with stored native integrals; imaginary pole not re-derived)')
    if candidate_ratio is not None and reference_ratio is not None:
        findings.append(
            f'frozen-rate power is cancellation-dominated: gross |v_d B_de v_e| terms are '
            f'{candidate_ratio:.1f}x (candidate) and {reference_ratio:.1f}x (HAMS reference) '
            'the net mechanical power')
    findings.append(
        f'far-minus-mechanical closure gap at the frozen rate is {closure:.6e} W; the signed '
        f'node-residual pressure work explains {residual_share * 100 if residual_share is not None else float("nan"):.1f}% '
        'of it, the remainder is discrete flux/far closure error')
    findings.append(
        f'largest per-DOF far-minus-mechanical gap: {dominant_dof} ({dominant_gap:.6e} W)')
    energetic_gaps = [entry['far_minus_near_rel_diff'] for entry in energy['per_dof']]
    min_eigenvalue = results['eigen_diagnostic']['min_eigenvalue_candidate']
    findings.append(f'coordinate near/far relative gaps span {min(energetic_gaps):.3%} '
                    f'to {max(energetic_gaps):.3%}; the closure defect is not confined '
                    'to the frozen near-null rate')
    findings.append(f'retained symmetric damping minimum eigenvalue {min_eigenvalue:.12e}; '
                    'a negative value is a passivity warning, not a new qualified result')
    if 'discrete_work_adjoint' in results:
        dw = results['discrete_work_adjoint']
        frozen = dw['frozen_rate']
        defect = frozen['green_defect_far_minus_flux_w']
        cov = frozen['flux_within_panel_covariance_w']
        persistent = frozen['mean_persistent_defect_far_minus_flux_mean_w']
        findings.append(
            f'discrete Green defect at the frozen rate: far - pointwise flux '
            f'{defect:.6e} W; within-panel pressure/velocity covariance '
            f'{cov:.6e} W; panel-mean persistent defect {persistent:.6e} W '
            f"({frozen['mean_persistent_rel_green_defect']:.2f} of the pointwise defect)")
        findings.append(
            'work/adjoint decision, frozen rate: ' + frozen['decision']
            + '; unit motions: '
            + ', '.join(f"{results['energy']['per_dof'][k]['dof']}: {e['decision']}"
                        for k, e in enumerate(dw['per_dof'])))
        findings.append(
            f"mean enforcement reproduces the panel-mean trace to "
            f"{dw['mean_enforcement_max_abs_velocity_m_per_s']:.3e} m/s; the force "
            f"equals the negative motion-coordinate work matrix (F = -A_work for "
            f"the real stored trace) to "
            f"{dw['adjoint_force_transpose_identity_max_abs']:.3e}")
        findings.append(
            'weighted-S summed reciprocity (complex/imag): '
            f"{dw['weighted_s_summed_reciprocity']['weighted_s_summed_asym_rel']:.3e}/"
            f"{dw['weighted_s_summed_reciprocity']['weighted_s_summed_imag_asym_rel']:.3e} "
            '(Sum_q diag(w_q) S_q; per-quadrature entries are stencil info, '
            'not reciprocity failure claims)')
    if assembly_rel >= 1e-9:
        verdict = 'stored-field reconstruction shows assembly/trace inconsistency'
    else:
        verdict = ('stored fields reproduce their producer, but energetic coordinate '
                   'motions also have closure defects; cancellation magnifies the '
                   'frozen-rate discrepancy and does not establish source accuracy')
    next_action = ('Investigate the discrete work/adjoint relation of the current '
                   'panel-mean Neumann formulation and its pointwise pressure recovery '
                   'using retained operators; distinguish discretization inconsistency '
                   'from evaluator error before selecting any prospective source repair. '
                   'Do not change normalization, thresholds or qualify this case.')
    if 'discrete_work_adjoint' in results:
        overall = results['discrete_work_adjoint']['decision_summary']['overall']
        if overall == 'persists_in_saved_trace_green_relation':
            verdict = ('within-panel pressure/velocity covariance does not account for '
                       'the closure defect; it persists in the panel-mean saved-trace '
                       'Green relation (stored pole power vs panel-mean boundary flux)')
            next_action = ('Localize the summed weighted-S asymmetry to source/target '
                           'panel pairs and compare an independently matched panel '
                           'integral/normal-trace reference before proposing a source '
                           'or discretization repair. The all-panel functional includes '
                           'an artificial internal lid; its difference from far power '
                           'alone does not prove a continuous kernel identity fails. '
                           'Weight symmetrization is not an established Galerkin '
                           'formulation or an energy-identity proof. Do not enforce '
                           'symmetry, change normalization or thresholds, or qualify '
                           'this case.')
        elif overall == 'within_panel_covariance_accounted':
            verdict = ('the closure defect is accounted for by within-panel '
                       'pressure/normal-velocity covariance; the panel-mean Green '
                       'relation closes only through the covariance correction')
            next_action = ('Repair should target the quadrature consistency of the '
                           'pressure/normal-velocity pairing (pointwise recovery vs '
                           'panel-mean enforcement) rather than the far construction. '
                           'Do not change normalization, thresholds or qualify this case.')
        elif overall == 'discrete_green_relation_closed':
            verdict = ('the discrete Green energy relation closes for this stored case; '
                       'no closure defect remains to attribute')
            next_action = ('No source repair is indicated by this diagnostic; keep the '
                           'formulation frozen and record the closed relation. Do not '
                           'change normalization, thresholds or qualify this case.')
        else:
            verdict = ('the closure-defect attribution is mixed across the frozen rate '
                       'and unit motions; per-rate decisions are reported separately')
            next_action = ('Use the per-rate decisions in discrete_work_adjoint to '
                           'localize which motions close through covariance; do not '
                           'change normalization, thresholds or qualify this case.')
    confidence = ('machine-algebra restorations are exact to '
                 f'{worst_restore:.1e} relative; interpretation is limited to stored-field '
                 'postprocessing of one development point and is not source qualification')
    return {'findings': findings, 'verdict': verdict, 'confidence': confidence,
            'next_action': next_action}


def build_report(results: dict) -> str:
    energy = results['energy']
    restorations = results['machine_restorations']
    lines = [
        '# Retained panel energy-closure audit',
        '',
        f"Format `{results['format']}`, generated {results['generated_utc']}, "
        f"status `{results['status']}`.",
        '',
        'Read-only stored-field postprocessing; not source qualification. '
        'Inputs were never modified.',
        '',
        '## Identity',
        '',
        f"- case hash `{results['inputs']['case_hash']}` "
        f"(summary match: {results['inputs']['case_hash_match']})",
        f"- status state `{results['inputs']['status_state']}`",
        f"- snapshot array hashes match: "
        f"{all(v is True for v in results['inputs']['snapshot_array_hashes_match'].values())}",
        f"- candidate source sha256 match with case.json module hash: "
        f"{results['inputs']['candidate_source']['match']}",
        f"- omega {results['conventions']['omega']}, k {results['conventions']['k']}, "
        f"rho {results['conventions']['rho']}, depth {results['conventions']['water_depth']}, "
        f"dispersion residual {results['conventions']['dispersion_residual']:.3e}",
        f"- retained DOFs {results['inputs']['retained_dofs']}",
        '',
        '## Machine restorations (frozen rate)',
        '',
        '| quantity | recorded | recomputed | rel diff |',
        '| --- | --- | --- | --- |',
    ]
    for name, entry in restorations.items():
        recorded = 'not recorded' if entry['recorded'] is None else f"{entry['recorded']:.12e}"
        rel = 'n/a' if entry['rel_diff'] is None else f"{entry['rel_diff']:.3e}"
        lines.append(f"| {name} | {recorded} | {entry['recomputed']:.12e} | {rel} |")
    recon = results['operator_reconstruction']
    lines += [
        '',
        '## Operator reconstruction (sequential loads, no new solve)',
        '',
        f"- worst pressure rel diff {max(q['pressure_rel_diff'] for q in recon['per_quadrature']):.3e}",
        f"- worst node-residual rel diff {max(q['node_residual_rel_diff'] for q in recon['per_quadrature']):.3e}",
        f"- mean-residual rel diff {recon['mean_residual_rel_diff']:.3e}",
        f"- force rel diff {recon['force_rel_diff']:.3e}",
    ]
    if 'native' in recon:
        native = recon['native']['per_quadrature']
        lines.append(f"- native s.real max abs diff {max(q['s_real_max_abs_diff'] for q in native):.3e}")
        lines.append(f"- native real-trace/jump rel diff {max(q['k_real_trace_rel_diff'] for q in native):.3e}")
    lines += [
        '',
        '## Energy closure',
        '',
        f"- near mechanical power {energy['near_w']:.12e} W",
        f"- hull boundary work {energy['boundary_work_w']:.12e} W "
        f"(identity |near-work| max {energy['near_work_identity_max_abs_w']:.3e} W)",
        f"- far power " + ', '.join(f"{c}: {v:.12e}" for c, v in energy['far_w'].items()),
        f"- far closed form (stored pole) {energy['far_closed_form_w']:.12e} W "
        f"(vs largest angular sum rel diff {energy['far_closed_vs_largest_rel_diff']:.3e})",
        f"- actual flux (all panels, g+r) {energy['actual_flux_w']:.12e} W; "
        f"flux-(mech+residual work) {energy['flux_minus_mech_minus_residual_work_w']:.3e} W",
        f"- signed node-residual pressure work all/hull/lid "
        f"{energy['rate_residual_work_all_w']:.6e}/{energy['rate_residual_work_hull_w']:.6e}/"
        f"{energy['rate_residual_work_lid_w']:.6e} W",
        f"- node residual max {energy['rate_node_residual_max_velocity_m_per_s']:.6e} m/s "
        "(velocity; legacy JSON label ends in _w)",
        f"- mean residual max {energy['rate_mean_residual_max_velocity_m_per_s']:.3e} m/s",
        f"- closure far-mech {energy['closure_far_minus_mech_w']:.6e} W, "
        f"far-flux {energy['closure_far_minus_flux_w']:.6e} W",
        f"- hull-only/full far power ratio (includes interference; not a lid fraction) {energy['hull_only_far_power_ratio']}",
        '',
        '| DOF | near W | far W (largest N) | far closed W | gap far-near W | residual work all W | hull-only/full far |',
        '| --- | --- | --- | --- | --- | --- | --- |',
    ]
    for entry in energy['per_dof']:
        largest = list(entry['far_w'].values())[0]
        lid = 'n/a' if entry['hull_only_far_power_ratio'] is None else f"{entry['hull_only_far_power_ratio']:.3f}"
        lines.append(
            f"| {entry['dof']} | {entry['near_w']:.6e} | {largest:.6e} | "
            f"{entry['far_closed_form_w']:.6e} | {entry['gap_far_minus_near_w']:.6e} | "
            f"{entry['residual_work_all_w']:.6e} | {lid} |")
    dw = results['discrete_work_adjoint']
    frozen_dw = dw['frozen_rate']
    lines += [
        '',
        '## Discrete work/adjoint diagnosis',
        '',
        f"- physical hull flux {frozen_dw['flux_physical_hull_pointwise_w']:.12e} W; artificial lid flux {frozen_dw['flux_artificial_lid_pointwise_w']:.12e} W",
        f"- far minus physical hull flux {frozen_dw['far_minus_physical_hull_flux_w']:.12e} W",
        '- All-panel flux is a computational functional including an artificial internal lid; it is not an independent physical kernel-identity oracle.',
        '',
        'Identities (recorded phasor exp(-i omega t), p = i rho omega phi, u = g + r):',
        '',
    ]
    for identity in dw['identities']:
        lines.append(f'- {identity}')
    lines += [
        '',
        '| rate | far W | flux pt W | flux mean W | cov W | defect far-flux W | '
        'persistent far-mean W | decision |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
        f"| frozen | {frozen_dw['far_w']:.6e} | {frozen_dw['flux_pointwise_w']:.6e} | "
        f"{frozen_dw['flux_panel_mean_w']:.6e} | "
        f"{frozen_dw['flux_within_panel_covariance_w']:.6e} | "
        f"{frozen_dw['green_defect_far_minus_flux_w']:.6e} | "
        f"{frozen_dw['mean_persistent_defect_far_minus_flux_mean_w']:.6e} | "
        f"{frozen_dw['decision']} |",
    ]
    for k, entry in enumerate(dw['per_dof']):
        dof = energy['per_dof'][k]['dof']
        lines.append(
            f"| {dof} | {entry['far_w']:.6e} | {entry['flux_pointwise_w']:.6e} | "
            f"{entry['flux_panel_mean_w']:.6e} | "
            f"{entry['flux_within_panel_covariance_w']:.6e} | "
            f"{entry['green_defect_far_minus_flux_w']:.6e} | "
            f"{entry['mean_persistent_defect_far_minus_flux_mean_w']:.6e} | "
            f"{entry['decision']} |")
    lines += [
        '',
        f"- split identities max abs: flux "
        f"{frozen_dw['split_flux_identity_max_abs_w']:.3e} W, work "
        f"{frozen_dw['split_work_identity_max_abs_w']:.3e} W, Green decomposition "
        f"{frozen_dw['green_decomposition_identity_max_abs_w']:.3e} W, closure "
        f"decomposition {frozen_dw['closure_decomposition_identity_max_abs_w']:.3e} W",
        f"- mean enforcement max |u_mean - g_mean| "
        f"{dw['mean_enforcement_max_abs_velocity_m_per_s']:.3e} m/s; force "
        f"adjoint identity F = -A_work (real trace) "
        f"{dw['adjoint_force_transpose_identity_max_abs']:.3e}",
        f"- work covariance (frozen) "
        f"{frozen_dw['work_within_panel_covariance_w']:.6e} W; closure mean persistent "
        f"{frozen_dw['closure_mean_persistent_w']:.6e} W",
        f"- weighted-S summed reciprocity (complex/imag) "
        f"{dw['weighted_s_summed_reciprocity']['weighted_s_summed_asym_rel']:.3e}/"
        f"{dw['weighted_s_summed_reciprocity']['weighted_s_summed_imag_asym_rel']:.3e} "
        '(Sum_q diag(w_q) S_q)',
        '- per-quadrature weighted-S stencil asymmetry (complex/imag per q; '
        'stencil info, not reciprocity failure claims): '
        + '; '.join(f"q{e['qi']} {e['weighted_s_asym_rel']:.3e}/"
                    f"{e['weighted_s_imag_asym_rel']:.3e}"
                    for e in dw['weighted_s_stencil_asymmetry_per_quadrature']),
        f"- far-form Hermitian asymmetry {dw['far_form_hermitian_asym_rel']:.3e}",
        '- retained flux mean-power Hermitian eigenvalues (include the .5 of '
        '.5 Re(v^H A v)): '
        + ', '.join(f"{v:.6e}" for v in
                    dw['projected_retained']['flux_mean_power_matrix_hermitian_eigenvalues']),
        '- retained pole mean-power Hermitian eigenvalues: '
        + ', '.join(f"{v:.6e}" for v in
                    dw['projected_retained']['pole_mean_power_matrix_hermitian_eigenvalues']),
        f"- overall decision: {dw['decision_summary']['overall']}",
    ]
    cancellation = results['cancellation']
    def _ratio(value):
        return 'n/a' if value is None else f'{value:.1f}'
    lines += [
        '',
        '## Cancellation and reciprocity',
        '',
        f"- candidate retained: gross {cancellation['candidate_retained']['gross_abs_w']:.6e} W, "
        f"net {cancellation['candidate_retained']['net_w']:.6e} W, "
        f"ratio {_ratio(cancellation['candidate_retained']['cancellation_ratio'])}",
        f"- HAMS reference retained: gross {cancellation['hams_reference_retained']['gross_abs_w']:.6e} W, "
        f"net {cancellation['hams_reference_retained']['net_w']:.6e} W, "
        f"ratio {_ratio(cancellation['hams_reference_retained']['cancellation_ratio'])}",
        f"- damping asymmetry (retained): candidate "
        f"{results['reciprocity']['candidate_retained']['damping_asym_rel']:.3e}, HAMS reference "
        f"{results['reciprocity']['hams_reference_retained']['damping_asym_rel']:.3e}",
        f"- retained symmetrized damping eigenvalues (candidate): "
        + ', '.join(f"{v:.6e}" for v in results['eigen_diagnostic']['candidate_retained_sym_eigenvalues']),
        f"- retained symmetrized damping eigenvalues (HAMS reference): "
        + ', '.join(f"{v:.6e}" for v in results['eigen_diagnostic']['hams_reference_retained_sym_eigenvalues']),
        '',
        '## HAMS comparison',
        '',
        f"- recorded reference power {results['hams_comparison']['recorded_reference_w']}, "
        f"recomputed from frozen reference blocks {results['hams_comparison']['recomputed_reference_w']:.12e} W",
        f"- candidate block vs radiation submatrix max abs diff "
        f"{results['hams_comparison']['candidate_block_vs_radiation_max_abs_diff']:.3e}",
        '',
        '## Interpretation',
        '',
    ]
    for finding in results['interpretation']['findings']:
        lines.append(f'- {finding}')
    lines += [
        '',
        f"Verdict: {results['interpretation']['verdict']}",
        '',
        f"Confidence: {results['interpretation']['confidence']}",
        '',
        f"Next action: {results['interpretation']['next_action']}",
        '',
        '## Honest limits',
        '',
    ]
    for limit in results['honest_limits']:
        lines.append(f'- {limit}')
    lines += [
        '',
        f"Runtime: {results['runtime']['elapsed_seconds']:.3f} s; "
        f"RLIMIT_AS applied: {results['limits']['rlimit']['applied']}.",
        '',
    ]
    return '\n'.join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--case-dir', required=True, type=Path)
    parser.add_argument('--parent-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--retained-dofs', default=','.join(str(d) for d in DEFAULT_RETAINED_DOFS))
    parser.add_argument('--angular-counts', default=','.join(str(c) for c in DEFAULT_ANGULAR_COUNTS))
    parser.add_argument('--rlimit-mib', type=int, default=DEFAULT_RLIMIT_MIB)
    parser.add_argument('--skip-native', action='store_true')
    parser.add_argument('--candidate-source', type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        retained = tuple(int(d) for d in str(args.retained_dofs).split(',') if d != '')
        counts = tuple(int(c) for c in str(args.angular_counts).split(',') if c != '')
        if not counts or any(c < 16 or c > 4096 for c in counts):
            raise AuditRefusal('angular counts must be integers in [16,4096]')
        output_dir = args.output_dir.resolve()
        case_dir = args.case_dir.resolve()
        parent_dir = args.parent_dir.resolve()
        if output_dir.exists():
            raise AuditRefusal(f'output directory already exists, refusing to overwrite: {output_dir}')
        if output_dir == case_dir or output_dir == parent_dir \
                or case_dir in output_dir.parents or parent_dir in output_dir.parents:
            raise AuditRefusal('output directory must be outside the read-only case and parent inputs')
        candidate_source = args.candidate_source
        if candidate_source is None:
            guessed = Path(__file__).resolve().parents[1] / 'src' / 'anyhydro' / 'panel_integration_candidate.py'
            candidate_source = guessed if guessed.is_file() else None
        results = audit(case_dir, parent_dir, retained_dofs=retained,
                        angular_counts=counts, verify_native=not args.skip_native,
                        rlimit_mib=args.rlimit_mib, candidate_source=candidate_source)
        output_dir.mkdir(parents=True)
        (output_dir / AUDIT_JSON_NAME).write_text(
            json.dumps(results, indent=1, sort_keys=True) + '\n', encoding='utf-8')
        (output_dir / AUDIT_MD_NAME).write_text(build_report(results), encoding='utf-8')
    except (AuditRefusal, ValueError, KeyError, TypeError, OSError, IndexError) as exc:
        print(json.dumps({'format': FORMAT, 'status': 'refused', 'reason': str(exc)}),
              file=sys.stderr)
        return 2
    worst = max((entry['rel_diff'] for entry in results['machine_restorations'].values()
                 if entry['rel_diff'] is not None), default=float('inf'))
    print(json.dumps({
        'status': 'completed',
        'output_dir': str(output_dir),
        'worst_restoration_rel_diff': worst,
        'closure_far_minus_mech_w': results['energy']['closure_far_minus_mech_w'],
        'verdict': results['interpretation']['verdict'],
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
