# SPDX-License-Identifier: MPL-2.0
"""W5 retained-block criteria R1-R8 (tools/check_retained_block_criteria.py)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for p in (str(REPO / 'src'), str(REPO / 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

import check_retained_block_criteria as crit  # noqa: E402

BODIES = ['a', 'a', 'b', 'b']
EVIDENCE = Path('C:/ANYworkspaceWork/anymarine-evidence/w5-route-comparison-20261012/'
                '03c-case/energy-comparison.json')


def _spd(eigs, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(4, 4)))
    return q @ np.diag(eigs) @ q.T


@pytest.fixture()
def reference():
    return _spd([420.0, 400.0, 360.0, 350.0], 1), _spd([200.0, 28.0, 0.33, 0.001], 2)


def _by(results, criterion, matrix=None):
    return [r for r in results['results'] if r['criterion'] == criterion
            and (matrix is None or r.get('matrix') == matrix)]


def test_identical_matrices_pass_r1_to_r4_and_leave_evidence_criteria_unresolved(reference):
    A, B = reference
    res = crit.evaluate(A, B, A, B, 1.5, BODIES)
    for c in ('R1', 'R2', 'R3', 'R4'):
        assert all(r['verdict'] == 'passed' for r in _by(res, c)), c
    for c in ('R6', 'R7', 'R8'):
        assert _by(res, c)[0]['verdict'] == 'unresolved', c
    assert _by(res, 'R5', 'B')[0]['verdict'] == 'unresolved'   # weak B mode 0.001
    assert res['verdict'] == 'unresolved' and res['qualification'] is False


def test_full_block_error_just_above_limit_fails(reference):
    A, B = reference
    E = np.zeros_like(B)
    E[0, 1] = E[1, 0] = 1.0
    scale = 0.0301 * np.linalg.norm(B) / np.linalg.norm(E)
    res = crit.evaluate(A, B + scale * E, A, B, 1.5, BODIES)
    assert _by(res, 'R1', 'B')[0]['verdict'] == 'failed'
    assert res['verdict'] == 'failed'


def test_weak_mode_error_does_not_fail_energetic_criterion_but_stays_unresolved(reference):
    A, B = reference
    lam, vec, energetic = crit.modes(B)
    weak = vec[:, ~energetic][:, 0]
    C = B - 0.01 * np.outer(weak, weak)       # flips the weak eigenvalue negative
    res = crit.evaluate(A, C, A, B, 1.5, BODIES)
    assert _by(res, 'R2', 'B')[0]['verdict'] == 'passed'
    assert _by(res, 'R5', 'B')[0]['verdict'] == 'unresolved'
    # with a documented uncertainty smaller than the error it fails
    res = crit.evaluate(A, C, A, B, 1.5, BODIES, weak_uncertainty={'B': {3: 0.001}})
    assert _by(res, 'R5', 'B')[0]['verdict'] == 'failed'


def test_coupling_error_is_caught_even_when_full_block_passes(reference):
    A, B = reference
    C = B.copy()
    # 10 % error confined to the a->b coupling block
    C[np.ix_([0, 1], [2, 3])] *= 1.10
    res = crit.evaluate(A, C, A, B, 1.5, BODIES)
    blocks = _by(res, 'R3', 'B')[0]['blocks']
    assert any(b['verdict'] == 'failed' for b in blocks)


def test_asymmetry_limit(reference):
    A, B = reference
    K = np.zeros_like(B)
    K[0, 2], K[2, 0] = 1.0, -1.0
    C = B + 0.0201 * np.linalg.norm(B) / np.linalg.norm(K) * K
    assert _by(crit.evaluate(A, C, A, B, 1.5, BODIES), 'R4', 'B')[0]['verdict'] == 'failed'


def test_near_far_and_convergence_and_passivity_routes(reference):
    A, B = reference
    res = crit.evaluate(A, B, A, B, 1.5, BODIES, b_far=B * 1.02,
                        refinement=(dict(A=A * 1.005, B=B * 1.005, panels=864), dict(A=A, B=B, panels=1944)),
                        b_time_domain=B)
    assert _by(res, 'R6')[0]['verdict'] == 'passed'
    assert _by(res, 'R8')[0]['verdict'] == 'passed'
    assert _by(res, 'R7')[0]['verdict'] == 'unresolved'   # PSD alone never passes
    res = crit.evaluate(A, B, A, B, 1.5, BODIES, b_far=B * 1.04,
                        refinement=(dict(A=A * 1.02, B=B, panels=864), dict(A=A, B=B, panels=1944)),
                        b_time_domain=B - 0.01 * np.eye(4))
    assert _by(res, 'R6')[0]['verdict'] == 'failed'
    assert _by(res, 'R8')[0]['verdict'] == 'failed'
    assert _by(res, 'R7')[0]['verdict'] == 'failed'


def test_invalid_inputs_are_refused(reference):
    A, B = reference
    with pytest.raises(ValueError):
        crit.evaluate(A, B, A, B, 1.5, ['a', 'b'])
    with pytest.raises(ValueError):
        crit.evaluate(A, B, A, B, -1.0, BODIES)
    with pytest.raises(ValueError):
        crit.modes(-np.eye(3))


@pytest.mark.skipif(not EVIDENCE.is_file(), reason='03c evidence not present')
def test_actual_03c_retained_block_verdicts(tmp_path):
    out = tmp_path / 'r.json'
    assert crit.main(['--energy-comparison', str(EVIDENCE), '--omega', '1.5',
                      '--row-bodies', 'a,a,b,b', '--output', str(out)]) == 0
    res = json.loads(out.read_text(encoding='utf-8'))
    for c in ('R1', 'R2', 'R3', 'R4'):
        assert all(r['verdict'] == 'passed' for r in _by(res, c)), c
    assert _by(res, 'R5', 'B')[0]['verdict'] == 'unresolved'
    assert res['verdict'] == 'unresolved'


# ---- review findings (codex-review-06) ----
def test_weak_added_mass_modes_are_checked(reference):
    _, B = reference
    A = _spd([400.0, 300.0, 200.0, 0.01], 3)
    res = crit.evaluate(A, B, A, B, 1.5, BODIES)
    assert _by(res, 'R5', 'A')[0]['verdict'] == 'unresolved'


def test_refinement_needs_panel_counts_and_ratio(reference):
    A, B = reference
    same = dict(A=A, B=B)
    assert _by(crit.evaluate(A, B, A, B, 1.5, BODIES, refinement=(same, same)),
               'R8')[0]['verdict'] == 'unresolved'
    small = (dict(A=A, B=B, panels=864), dict(A=A, B=B, panels=1200))
    assert _by(crit.evaluate(A, B, A, B, 1.5, BODIES, refinement=small),
               'R8')[0]['verdict'] == 'unresolved'


def test_refinement_mode_swap_is_detected():
    A = np.diag([400.0, 300.0, 200.0, 100.0])
    fine = np.diag([200.0, 28.0, 0.33, 0.001])
    coarse = np.diag([200.0, 28.0, 0.001, 0.33])    # same sorted spectrum
    res = crit.evaluate(A, fine, A, fine, 1.5, BODIES,
                        refinement=(dict(A=A, B=coarse, panels=864), dict(A=A, B=fine, panels=1944)))
    assert _by(res, 'R8')[0]['verdict'] == 'failed'


def test_missing_or_duplicate_coefficients_are_refused(tmp_path):
    coeffs = [dict(row=i, column=j, candidate=1.0 if i == j else 0.1, reference=1.0 if i == j else 0.1)
              for i in range(4) for j in range(4)]
    doc = {'format': 'anyhydro-e1-energy-hams-comparison-v1',
           'added_mass': {'coefficients': coeffs}, 'radiation_damping': {'coefficients': coeffs}}
    path = tmp_path / 'e.json'
    path.write_text(json.dumps(doc), encoding='utf-8')
    assert crit._from_energy_comparison(path)['added_mass'][0].shape == (4, 4)
    doc['radiation_damping'] = {'coefficients': [c for c in coeffs if (c['row'] < 2) == (c['column'] < 2)]}
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='incomplete'):
        crit._from_energy_comparison(path)
    doc['radiation_damping'] = {'coefficients': coeffs + coeffs[:1]}
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        crit._from_energy_comparison(path)


def test_zero_blocks_do_not_crash():
    A = np.diag([400.0, 300.0, 200.0, 100.0])
    B = np.diag([20.0, 10.0, 0.0, 0.0])            # zero inter-body blocks
    res = crit.evaluate(A, B, A, B, 1.5, BODIES)
    assert _by(res, 'R3', 'B')[0]['verdict'] == 'passed'
    assert crit.r4_symmetry(np.zeros((4, 4)), B, 'B')['verdict'] == 'passed'
    assert crit.r1_full_block(np.ones((2, 2)), np.zeros((2, 2)), 'X')['verdict'] == 'failed'


def test_degenerate_cluster_error_is_basis_independent():
    r = np.diag([100.0, 100.0, 50.0, 10.0])
    c = np.diag([104.0, 100.0, 50.0, 10.0])
    rot = np.eye(4)
    t = np.pi / 5
    rot[:2, :2] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    e1, _, _, _ = crit.modal_errors(c, r)
    e2, _, _, _ = crit.modal_errors(rot @ c @ rot.T, rot @ r @ rot.T)
    assert e1[0]['modes'] == [0, 1]
    assert e1[0]['relative_error'] == pytest.approx(0.04)
    assert e2[0]['relative_error'] == pytest.approx(0.04)


def test_invalid_panel_counts_leave_r8_unresolved(reference):
    A, B = reference
    for counts in ((864, float('nan')), (864, float('inf')), (-864, -1944), (864.0, 1944.0)):
        pair = (dict(A=A, B=B, panels=counts[0]), dict(A=A, B=B, panels=counts[1]))
        res = crit.evaluate(A, B, A, B, 1.5, BODIES, refinement=pair)
        assert _by(res, 'R8')[0]['verdict'] == 'unresolved', counts


def test_nan_followed_by_duplicate_is_refused(tmp_path):
    coeffs = [dict(row=i, column=j, candidate=1.0, reference=1.0) for i in range(2) for j in range(2)]
    bad = [dict(row=0, column=0, candidate=float('nan'), reference=1.0)] + coeffs
    doc = {'format': 'anyhydro-e1-energy-hams-comparison-v1',
           'added_mass': {'coefficients': coeffs}, 'radiation_damping': {'coefficients': bad}}
    path = tmp_path / 'e.json'
    path.write_text(json.dumps(doc), encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        crit._from_energy_comparison(path)


def test_eigenvalues_exactly_one_percent_apart_are_not_clustered():
    lam = np.array([10000.0, 100.0, 99.0])
    groups = crit.clusters(lam, np.ones(3, dtype=bool))
    assert groups == [[0], [1], [2]]
    assert crit.clusters(np.array([100.0, 99.5]), np.ones(2, dtype=bool)) == [[0, 1]]
