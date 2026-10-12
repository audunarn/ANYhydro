# SPDX-License-Identifier: MPL-2.0
"""W5 retained-block coefficient and energy criteria R1-R8.

Implements the criteria adopted by the owner on 2026-10-12
(ANYmarine docs/IMPLEMENTATION_PLAN.md, "W5 retained-block coefficient and
energy criteria").  Pure numpy; reads retained matrices only.  A criterion
without its evidence is ``unresolved``, never ``passed``; ``qualification``
is always false (this tool evaluates a gate input, it does not admit a route).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REL_TOL = 0.03          # R1, R2, R3, R6
SYM_TOL = 0.02          # R4
CONV_TOL = 0.01         # R8
ENERGETIC = 1e-3        # mode k energetic when lambda_k >= ENERGETIC * rho
COUPLING_FLOOR = 1e-3   # R3 applies when ||R_ab|| >= COUPLING_FLOOR * ||R||
CLUSTER = 0.01          # energetic eigenvalues closer than 1 % form one cluster
MIN_REFINEMENT = 1.5    # R8 linear panel-size ratio

PASS, FAIL, UNRESOLVED = 'passed', 'failed', 'unresolved'


def _sym(x):
    return 0.5 * (x + x.T)


def _fro(x):
    return float(np.linalg.norm(x))


def _rel(diff, ref):
    """||diff||/||ref||; 0 when both vanish, inf when only ref vanishes."""
    d, r = _fro(diff), _fro(ref)
    if r == 0.0:
        return 0.0 if d == 0.0 else math.inf
    return d / r


def _verdict(ok):
    return PASS if ok else FAIL


def _combine(verdicts):
    if FAIL in verdicts:
        return FAIL
    if UNRESOLVED in verdicts or not verdicts:
        return UNRESOLVED
    return PASS


def modes(reference):
    """Eigenpairs of S(reference), decreasing; energetic mask."""
    lam, vec = np.linalg.eigh(_sym(reference))
    order = np.argsort(lam)[::-1]
    lam, vec = lam[order], vec[:, order]
    rho = float(lam[0])
    if not rho > 0:
        raise ValueError('reference has no positive spectral radius')
    return lam, vec, lam >= ENERGETIC * rho


def r1_full_block(c, r, label):
    err = _rel(c - r, r)
    return dict(criterion='R1', matrix=label, relative_error=err, limit=REL_TOL,
                verdict=_verdict(err <= REL_TOL))


def clusters(lam, energetic):
    """Groups of energetic eigenvalue indices closer than CLUSTER (relative)."""
    groups, current = [], []
    for k in np.flatnonzero(energetic):
        if current and (lam[current[-1]] - lam[k]) >= CLUSTER * lam[current[-1]]:
            groups.append(current)
            current = []
        current.append(int(k))
    if current:
        groups.append(current)
    return groups


def modal_errors(c, r):
    """Basis-independent energetic-mode errors of S(c) against S(r).

    For a single mode the error is |v^T S(c) v - lambda| / lambda.  For a
    cluster of (near-)degenerate reference eigenvalues it is the spectral
    norm of V^T (S(c) - S(r)) V over the smallest eigenvalue of the cluster,
    which does not depend on the basis chosen inside the cluster.
    """
    lam, vec, energetic = modes(r)
    out = []
    for g in clusters(lam, energetic):
        V = vec[:, g]
        rq = [float(V[:, i] @ _sym(c) @ V[:, i]) for i in range(len(g))]
        if len(g) == 1:
            err = abs(rq[0] - lam[g[0]]) / lam[g[0]]
        else:
            M = V.T @ (_sym(c) - _sym(r)) @ V
            err = np.linalg.norm(M, 2) / np.min(lam[g])
        out.append(dict(modes=g, reference=[float(v) for v in lam[g]],
                        candidate_rayleigh=rq, relative_error=float(err)))
    return out, lam, vec, energetic


def r2_energetic_modes(c, r, label):
    rows, lam, _, energetic = modal_errors(c, r)
    for row in rows:
        row['verdict'] = _verdict(row['relative_error'] <= REL_TOL)
    return dict(criterion='R2', matrix=label, limit=REL_TOL, groups=rows,
                weak_reference_eigenvalues=[float(v) for v in lam[~energetic]],
                verdict=_combine([x['verdict'] for x in rows]))


def r3_coupling(c, r, label, bodies):
    names = sorted(set(bodies))
    out = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ia = [k for k, n in enumerate(bodies) if n == a]
            ib = [k for k, n in enumerate(bodies) if n == b]
            for rows, cols in ((ia, ib), (ib, ia)):
                ref = r[np.ix_(rows, cols)]
                cand = c[np.ix_(rows, cols)]
                size = _fro(ref) / _fro(r)
                item = dict(block=f'{bodies[rows[0]]}->{bodies[cols[0]]}',
                            reference_fraction=size)
                if size >= COUPLING_FLOOR:
                    item['relative_error'] = _rel(cand - ref, ref)
                    item['verdict'] = _verdict(item['relative_error'] <= REL_TOL)
                else:
                    item['verdict'] = 'reported_only'
                out.append(item)
    gated = [x['verdict'] for x in out if x['verdict'] != 'reported_only']
    return dict(criterion='R3', matrix=label, limit=REL_TOL, blocks=out,
                verdict=_combine(gated) if gated else PASS)


def r4_symmetry(c, r, label):
    asym = _rel(c - c.T, c)
    return dict(criterion='R4', matrix=label, candidate_asymmetry=asym,
                reference_asymmetry=_rel(r - r.T, r), limit=SYM_TOL,
                verdict=_verdict(asym <= SYM_TOL))


def r5_weak_modes(c, r, label, uncertainty=None):
    """Weak modes of S(r); ``uncertainty`` maps reference mode index -> U_k."""
    lam, vec, energetic = modes(r)
    rows = []
    for k in np.flatnonzero(~energetic):
        q = float(vec[:, k] @ _sym(c) @ vec[:, k])
        row = dict(mode=int(k), reference=float(lam[k]), candidate_rayleigh=q,
                   difference=q - float(lam[k]))
        u = (uncertainty or {}).get(int(k))
        if u is not None:
            row['uncertainty'] = float(u)
            row['verdict'] = _verdict(abs(row['difference']) <= row['uncertainty'])
        else:
            row['verdict'] = UNRESOLVED
        rows.append(row)
    return dict(criterion='R5', matrix=label, modes=rows,
                verdict=_combine([x['verdict'] for x in rows]) if rows else PASS)


def r6_near_far(b_near, b_far=None):
    if b_far is None:
        return dict(criterion='R6', verdict=UNRESOLVED,
                    reason='far-field (Kochin) damping of the candidate route not supplied')
    lam, vec, energetic = modes(b_near)
    rows = []
    for k in np.flatnonzero(energetic):
        u = vec[:, k]
        near = float(u @ b_near @ u)
        far = float(u @ b_far @ u)
        err = abs(near - far) / near
        rows.append(dict(mode=int(k), near=near, far=far, relative_error=err,
                         verdict=_verdict(err <= REL_TOL)))
    return dict(criterion='R6', limit=REL_TOL, modes=rows,
                verdict=_combine([x['verdict'] for x in rows]))


def r7_passivity(b_used=None):
    if b_used is None:
        return dict(criterion='R7', verdict=UNRESOLVED,
                    reason='time-domain damping route not yet selected; free decay not run')
    lam = np.linalg.eigvalsh(_sym(b_used))
    return dict(criterion='R7', min_eigenvalue=float(lam[0]),
                verdict=UNRESOLVED if lam[0] >= 0 else FAIL,
                reason='PSD is necessary; discrete free decay is still required')


def r8_convergence(coarse=None, fine=None):
    """Matched refinement pair: dicts with A, B and 'panels' (panel count).

    The linear panel-size ratio sqrt(panels_fine / panels_coarse) must be at
    least MIN_REFINEMENT.  Changes are measured on the full block and,
    directionally, on the fine solution's energetic modes (clusters handled
    basis-independently), so mode swaps between meshes are not hidden.
    """
    if coarse is None or fine is None:
        return dict(criterion='R8', verdict=UNRESOLVED,
                    reason='no matched refinement pair of the candidate route supplied')
    counts = (coarse.get('panels'), fine.get('panels'))
    if not all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in counts):
        return dict(criterion='R8', verdict=UNRESOLVED,
                    reason='refinement pair lacks positive integer panel counts')
    ratio = math.sqrt(counts[1] / counts[0])
    if ratio < MIN_REFINEMENT:
        return dict(criterion='R8', verdict=UNRESOLVED, linear_ratio=ratio,
                    reason=f'linear refinement ratio {ratio:.3f} < {MIN_REFINEMENT}')
    out = []
    for label in ('A', 'B'):
        c = np.asarray(coarse[label], dtype=float)
        f = np.asarray(fine[label], dtype=float)
        rows, _, _, _ = modal_errors(c, f)
        change = _rel(c - f, f)
        modal = max(row['relative_error'] for row in rows)
        out.append(dict(matrix=label, block_change=change, energetic_mode_change=modal,
                        verdict=_verdict(max(change, modal) <= CONV_TOL)))
    return dict(criterion='R8', limit=CONV_TOL, linear_ratio=ratio, items=out,
                verdict=_combine([x['verdict'] for x in out]))


def evaluate(A_c, B_c, A_r, B_r, omega, bodies, *, b_far=None, weak_uncertainty=None,
             refinement=None, b_time_domain=None):
    A_c, B_c, A_r, B_r = (np.asarray(x, dtype=float) for x in (A_c, B_c, A_r, B_r))
    if not (A_c.shape == B_c.shape == A_r.shape == B_r.shape and A_c.shape[0] == A_c.shape[1]
            and len(bodies) == A_c.shape[0]):
        raise ValueError('retained matrices must be square, equal-shaped and labelled by body')
    if not math.isfinite(omega) or omega <= 0:
        raise ValueError('omega must be finite and positive')
    Z_c = B_c + 1j * omega * A_c
    Z_r = B_r + 1j * omega * A_r
    results = [r1_full_block(A_c, A_r, 'A'), r1_full_block(B_c, B_r, 'B'),
               r1_full_block(Z_c, Z_r, 'Z')]
    results += [r2_energetic_modes(A_c, A_r, 'A'), r2_energetic_modes(B_c, B_r, 'B'),
                r3_coupling(A_c, A_r, 'A', bodies), r3_coupling(B_c, B_r, 'B', bodies),
                r4_symmetry(A_c, A_r, 'A'), r4_symmetry(B_c, B_r, 'B'),
                r5_weak_modes(A_c, A_r, 'A', (weak_uncertainty or {}).get('A')),
                r5_weak_modes(B_c, B_r, 'B', (weak_uncertainty or {}).get('B')),
                r6_near_far(B_c, b_far), r7_passivity(b_time_domain),
                r8_convergence(*(refinement or (None, None)))]
    return dict(tool='check_retained_block_criteria', qualification=False, omega=omega,
                bodies=list(bodies), results=results,
                verdict=_combine([r['verdict'] for r in results]))


def _from_energy_comparison(path):
    e = json.loads(Path(path).read_text(encoding='utf-8'))
    if e.get('format') != 'anyhydro-e1-energy-hams-comparison-v1':
        raise ValueError('unsupported energy-comparison format')
    mats = {}
    for key in ('added_mass', 'radiation_damping'):
        coeffs = e[key]['coefficients']
        n = 1 + max(max(c['row'], c['column']) for c in coeffs)
        cand, ref = np.zeros((n, n)), np.zeros((n, n))
        seen = np.zeros((n, n), dtype=bool)
        for c in coeffs:
            i, j = c['row'], c['column']
            if seen[i, j]:
                raise ValueError(f'{key}: duplicate coefficient ({i}, {j})')
            seen[i, j] = True
            cand[i, j] = c['candidate']
            ref[i, j] = c['reference']
        if not seen.all():
            raise ValueError(f'{key}: incomplete {n}x{n} coefficient set')
        if not (np.isfinite(cand).all() and np.isfinite(ref).all()):
            raise ValueError(f'{key}: non-finite coefficient')
        mats[key] = (cand, ref)
    return mats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--energy-comparison', required=True, type=Path)
    p.add_argument('--omega', required=True, type=float)
    p.add_argument('--row-bodies', required=True,
                   help='comma-separated body label of each retained row, e.g. a,a,b,b')
    p.add_argument('--output', type=Path)
    a = p.parse_args(argv)
    mats = _from_energy_comparison(a.energy_comparison)
    (A_c, A_r), (B_c, B_r) = mats['added_mass'], mats['radiation_damping']
    res = evaluate(A_c, B_c, A_r, B_r, a.omega, a.row_bodies.split(','))
    text = json.dumps(res, indent=1)
    if a.output:
        if a.output.exists():
            raise SystemExit(f'refusing to overwrite {a.output}')
        a.output.write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
