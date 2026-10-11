# SPDX-License-Identifier: MPL-2.0
"""Independent focused tests for the retained weak Neumann candidate tool.

Synthetic fixtures only (at most 8 panels and 8 DOFs); the frozen
development case is never touched by this suite. Hand-computed complex
fixtures check the assembled hull equations, the conjugated test side, the
original lid rows, the jump-not-doubled handling, original panel ordering
and full coupling; CLI-level checks cover synthetic solve/recovery,
refusals, fresh output, input immutability, singular systems and the
status-saved-before-solve path. No scientific qualification is conferred.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL_PATH = REPO / 'tools' / 'solve_retained_weak_neumann.py'


def _load_tool():
    spec = importlib.util.spec_from_file_location('solve_retained_weak_neumann',
                                                 TOOL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def tool():
    return _load_tool()


def _sha256_array(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# synthetic case construction
# --------------------------------------------------------------------------

def build_geometry(panel_count):
    """P disjoint CCW triangles in the z=0 plane; unit +z normals."""
    vertices = []
    faces = []
    for i in range(panel_count):
        x0 = 3.0 * i
        vertices += [[x0, 0.0, 0.0], [x0 + 1.0, 0.0, 0.0], [x0 + 0.5, 1.0, 0.0]]
        faces.append([3 * i, 3 * i + 1, 3 * i + 2])
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (panel_count, 1))
    bary = vertices[faces].mean(axis=1)
    offsets = np.array([[-0.1, -0.05, 0.0], [0.1, -0.05, 0.0],
                        [0.0, 0.05, 0.0], [0.0, 0.15, 0.0]])
    points = (bary[:, None, :] + offsets[None, :, :]).astype(np.float64)
    weights = np.full((panel_count, 4), 0.5 / 4.0, dtype=np.float64)  # area 0.5
    return vertices, faces, normals, points, weights


def build_operators(panel_count, rng, *, jump=0.5, scale=0.05, singular=False):
    """Well-conditioned complex S and K; K carries the +0.5 jump internally."""
    operators = []
    for _ in range(4):
        if singular:
            S = np.zeros((panel_count, panel_count), dtype=complex)
            K = np.zeros((panel_count, panel_count), dtype=complex)
        else:
            S = (2.0 * np.eye(panel_count)
                 + scale * (rng.standard_normal((panel_count, panel_count))
                            + 1j * rng.standard_normal((panel_count, panel_count))))
            K = ((1.0 + jump) * np.eye(panel_count)
                 + scale * (rng.standard_normal((panel_count, panel_count))
                            + 1j * rng.standard_normal((panel_count, panel_count))))
        operators.append((S.astype(np.complex128), K.astype(np.complex128)))
    return operators


def write_synthetic_case(case_dir, *, hull_mask, ndof, operators, seed=7,
                          case_hash='synthetic-weak-neumann-case',
                          trace=None, dofs=None, nonfinite_operator=False,
                          mutate_input=None):
    """Write a complete synthetic case layout with an exact manifest."""
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    panel_count = len(hull_mask)
    rng = np.random.default_rng(seed)
    hull_mask = np.asarray(hull_mask, dtype=bool)
    vertices, faces, normals, points, weights = build_geometry(panel_count)
    if trace is None:
        trace = rng.standard_normal((panel_count, 4, ndof))
    trace = np.asarray(trace, dtype=np.float64)
    trace[~hull_mask] = 0.0
    owner = np.zeros(panel_count, dtype=np.int64)
    owner[panel_count // 2:] = 1
    arrays = {'vertices': vertices, 'faces': faces, 'points': points,
              'weights': weights, 'normals': normals, 'owner': owner,
              'hull_mask': hull_mask, 'rigid_trace': trace}
    if mutate_input is not None:
        mutate_input(arrays)
    np.savez(case_dir / 'input.npz', **arrays)
    manifest = {name: {'shape': list(arr.shape), 'dtype': arr.dtype.name,
                       'sha256': _sha256_array(arr)} for name, arr in arrays.items()}
    case_json = {'case_hash': case_hash, 'body_ids': ['synthetic-a', 'synthetic-b'],
                 'gravity': 9.81, 'omega': 1.5, 'rho': 1025.0,
                 'water_depth': 150.0, 'snapshot_arrays': manifest}
    (case_dir / 'case.json').write_text(json.dumps(case_json), encoding='utf-8')
    (case_dir / 'status.json').write_text(json.dumps({'state': 'completed'}),
                                           encoding='utf-8')
    if dofs is None:
        dofs = [f'synthetic-a::dof{d}' for d in range(ndof // 2)] \
            + [f'synthetic-b::dof{d}' for d in range(ndof - ndof // 2)]
    sources = rng.standard_normal((panel_count, ndof)) * 0.1
    pressure = rng.standard_normal((panel_count, 4, ndof)) * 0.1
    node_residual = rng.standard_normal((panel_count, 4, ndof)) * 0.01
    mean_residual = rng.standard_normal((panel_count, ndof)) * 0.01
    force = rng.standard_normal((ndof, ndof)) * 0.1
    np.savez(case_dir / 'radiation.npz', sources=sources, pressure=pressure,
             node_residual=node_residual, mean_residual=mean_residual,
             force_per_velocity=force, added_mass=np.zeros((ndof, ndof)),
             radiation_damping=np.zeros((ndof, ndof)),
             dofs=np.asarray(dofs))
    for qi, (S, K) in enumerate(operators):
        if nonfinite_operator and qi == 0:
            K = K.copy()
            K[0, 0] = np.nan + 0j
        np.savez(case_dir / f'operator-{qi}.npz', s=S, k=K)


def write_synthetic_parent(parent_dir, *, case_hash, rate, ndof):
    parent_dir = Path(parent_dir)
    parent_dir.mkdir(parents=True)
    summary = {'candidate': {'case_hash': case_hash, 'rate': list(rate),
                             'omega': 1.5, 'rho': 1025.0, 'water_depth': 150.0,
                             'k': 0.2293577981651376}}
    (parent_dir / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')


def mutate_input_with_manifest(case_dir, fn):
    """Rewrite one input array and keep the manifest consistent, so refusals
    come from the targeted check and not from manifest identity."""
    case_dir = Path(case_dir)
    with np.load(case_dir / 'input.npz') as data:
        arrays = {k: data[k] for k in data.files}
    fn(arrays)
    np.savez(case_dir / 'input.npz', **arrays)
    case_json = json.loads((case_dir / 'case.json').read_text())
    case_json['snapshot_arrays'] = {
        name: {'shape': list(arr.shape), 'dtype': arr.dtype.name,
               'sha256': _sha256_array(arr)} for name, arr in arrays.items()}
    (case_dir / 'case.json').write_text(json.dumps(case_json), encoding='utf-8')


def synthetic_rate(ndof):
    rate = [0.0] * ndof
    rate[0] = -0.6
    rate[ndof // 2] = 0.3
    return rate


def make_link(target, link):
    """Directory symlink, or a Windows junction when symlinks are unavailable
    (no privilege needed); Path.resolve() follows both."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return 'symlink'
    except OSError:
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
        return 'junction'


def remove_link(link):
    try:
        os.unlink(link)
    except OSError:
        os.rmdir(link)


def snapshot_tree(root):
    """Byte-level directory inventory: entry kinds, file sha256, link targets."""
    root = Path(root)
    snapshot = {}
    for path in sorted(root.rglob('*')):
        relative = str(path.relative_to(root)).replace('\\', '/')
        if path.is_symlink():
            snapshot[relative] = ('symlink', os.readlink(path))
        elif path.is_dir():
            snapshot[relative] = ('dir', None)
        else:
            snapshot[relative] = ('file', hashlib.sha256(path.read_bytes()).hexdigest())
    return snapshot


@contextlib.contextmanager
def actual_controls(tool, rate, *, expected_case_hash=None, retained_dofs=None,
                    angular_counts=None, rlimit_mib=None):
    """Substitute test-only module defaults for the fixed actual CLI controls.

    The production CLI accepts only the registered actual controls; these
    small synthetic tests explicitly authorize synthetic values without
    relaxing the gate itself.
    """
    names = ('ACTUAL_CASE_HASH', 'ACTUAL_RETAINED_DOFS', 'ACTUAL_ANGULAR_COUNTS',
             'ACTUAL_RLIMIT_MIB', 'REGISTERED_FROZEN_RATE')
    saved = {name: getattr(tool, name) for name in names}
    try:
        if expected_case_hash is not None:
            tool.ACTUAL_CASE_HASH = str(expected_case_hash)
        if retained_dofs is not None:
            tool.ACTUAL_RETAINED_DOFS = tuple(int(d) for d in retained_dofs)
        if angular_counts is not None:
            tool.ACTUAL_ANGULAR_COUNTS = tuple(int(c) for c in angular_counts)
        if rlimit_mib is not None:
            tool.ACTUAL_RLIMIT_MIB = int(rlimit_mib)
        tool.REGISTERED_FROZEN_RATE = tuple(float(v) for v in rate)
        yield
    finally:
        for name, value in saved.items():
            setattr(tool, name, value)


def run_tool(tool, case_dir, parent_dir, output_dir, rate, authorized=None, **kwargs):
    """CLI-level synthetic run. By default the module's fixed-control
    defaults are substituted with exactly the values passed on the command
    line (test-only authorization); pass `authorized` to keep the gate
    active against deliberately deviating CLI values."""
    argv = ['--case', str(case_dir), '--parent', str(parent_dir),
            '--output', str(output_dir)]
    for key, value in kwargs.items():
        argv += [f'--{key.replace("_", "-")}']
        if isinstance(value, (list, tuple)):
            argv += [str(v) for v in value]
        else:
            argv.append(str(value))
    controls = dict(kwargs if authorized is None else authorized)
    with actual_controls(tool, rate=rate, **controls):
        return tool.main(argv)


def make_solvable_case(tmp_path, *, hull_mask=(True, True, False, False, True, False),
                       ndof=4, seed=7, case_hash='synthetic-weak-neumann-case',
                       **kwargs):
    # the case sits three levels below tmp_path so the tool's immutable-ancestor
    # guard (case plus three ancestors) does not cover the sibling output dirs
    rng = np.random.default_rng(seed)
    case_dir = tmp_path / 'g0' / 'g1' / 'g2' / 'case'
    operators = build_operators(len(hull_mask), rng)
    write_synthetic_case(case_dir, hull_mask=hull_mask, ndof=ndof,
                         operators=operators, seed=seed, case_hash=case_hash,
                         **kwargs)
    rate = synthetic_rate(ndof)
    parent_dir = tmp_path / 'g0' / 'parent'
    write_synthetic_parent(parent_dir, case_hash=case_hash, rate=rate, ndof=ndof)
    return case_dir, parent_dir


# --------------------------------------------------------------------------
# pure assembly checks (hand-computed complex fixtures)
# --------------------------------------------------------------------------

def test_hull_weak_rows_hand_computed(tool):
    # 3 panels: hull H = [0, 1], lid L = [2]; one receiver node, one DOF.
    S = np.array([[1.0 + 1j, 2.0 - 1j, 0.5],
                  [0.0, 1.5 + 0.5j, 1.0],
                  [0.25, -0.5, 2.0 + 2j]], dtype=complex)
    K = np.array([[1.0, 0.5j, 0.0],
                  [-0.5, 2.0 + 1j, 0.75],
                  [0.1, 0.0, 1.0 - 0.5j]], dtype=complex)
    w = np.array([0.2, 0.3, 0.25])
    g = np.array([[0.7], [-1.2], [0.0]])
    areas = np.array([0.5, 0.5, 0.5])
    H = np.array([0, 1])
    L = np.array([2])
    A = np.zeros((3, 3), dtype=complex)
    b = np.zeros((3, 1), dtype=complex)
    tool.accumulate_quadrature(A, b, S, K, w, g, H, L, areas)
    # independent hand computation: hull row j: sum_i conj(S[i, j]) w_i K[i, :]
    for j in H:
        expected_row = sum(np.conj(S[i, j]) * w[i] * K[i, :] for i in H)
        expected_rhs = sum(np.conj(S[i, j]) * w[i] * g[i] for i in H)
        assert np.allclose(A[j], expected_row, atol=1e-15)
        assert np.allclose(b[j], expected_rhs, atol=1e-15)
    # lid row: original panel mean (w/area) * K, RHS (w/area) * g
    assert np.allclose(A[2], (w[2] / areas[2]) * K[2, :], atol=1e-15)
    assert np.allclose(b[2], (w[2] / areas[2]) * g[2], atol=1e-15)


def test_conjugate_test_side(tool):
    # imaginary, asymmetric S: the hull test side must conjugate S.
    S = np.array([[0.0, 1j, 0.0],
                  [0.0, 0.0, 1.0],
                  [1.0, 0.0, 0.0]], dtype=complex)
    K = np.eye(3, dtype=complex)
    w = np.ones(3)
    g = np.zeros((3, 1))
    areas = np.ones(3)
    H = np.array([0, 1])
    L = np.array([2])
    A = np.zeros((3, 3), dtype=complex)
    b = np.zeros((3, 1), dtype=complex)
    tool.accumulate_quadrature(A, b, S, K, w, g, H, L, areas)
    # hull row 1 collects conj(S[0, 1]) * w_0 * K[0, :] = conj(1j) = -1j
    assert np.allclose(A[1, 0], -1j, atol=1e-15)
    # a non-conjugated transpose would give +1j here
    assert A[1, 0] != 1j


def test_lid_rows_original_means(tool):
    # two receiver nodes; lid rows must equal the original panel-mean rows
    P, ndof = 4, 2
    rng = np.random.default_rng(3)
    H = np.array([0, 2])
    L = np.array([1, 3])
    areas = np.full(P, 0.5)
    A = np.zeros((P, P), dtype=complex)
    b = np.zeros((P, ndof), dtype=complex)
    expected_A = np.zeros((P, P), dtype=complex)
    expected_b = np.zeros((P, ndof), dtype=complex)
    for q in range(2):
        S = rng.standard_normal((P, P)) + 1j * rng.standard_normal((P, P))
        K = rng.standard_normal((P, P)) + 1j * rng.standard_normal((P, P))
        w = rng.random(P) + 0.1
        g = rng.standard_normal((P, ndof))
        g[L] = 0.0
        tool.accumulate_quadrature(A, b, S, K, w, g, H, L, areas)
        for i in L:
            expected_A[i] += (w[i] / areas[i]) * K[i, :]
            expected_b[i] += (w[i] / areas[i]) * g[i]
    assert np.allclose(A[L], expected_A[L], atol=1e-14)
    assert np.allclose(b[L], expected_b[L], atol=1e-14)
    # the rigid trace is zero on the lid, so the lid RHS is exactly zero
    assert np.all(b[L] == 0)


def test_jump_not_doubled(tool):
    # K already contains the +0.5 jump on its diagonal; the assembly must use
    # K exactly as stored. Split K = Kbase + 0.5 I and verify the assembled
    # hull rows contain the jump exactly once.
    P = 3
    H = np.array([0, 1])
    L = np.array([2])
    rng = np.random.default_rng(11)
    S = rng.standard_normal((P, P)) + 1j * rng.standard_normal((P, P))
    Kbase = rng.standard_normal((P, P)) + 1j * rng.standard_normal((P, P))
    K = Kbase + 0.5 * np.eye(P)
    w = rng.random(P) + 0.1
    g = np.zeros((P, 1))
    areas = np.full(P, 0.5)
    A = np.zeros((P, P), dtype=complex)
    b = np.zeros((P, 1), dtype=complex)
    tool.accumulate_quadrature(A, b, S, K, w, g, H, L, areas)
    expected_hull = S[np.ix_(H, H)].conj().T @ (w[H, None] * Kbase[H, :]) \
        + 0.5 * S[np.ix_(H, H)].conj().T @ (w[H, None] * np.eye(P)[H, :])
    assert np.allclose(A[H], expected_hull, atol=1e-14)
    # a doubled jump would add another 0.5 * T^H W on the hull diagonal block
    doubled = A[H] + 0.5 * S[np.ix_(H, H)].conj().T @ (w[H, None] * np.eye(P)[H, :])
    assert not np.allclose(A[H], doubled, atol=1e-14)


def test_panel_ordering_and_full_coupling(tool):
    # interleaved hull/lid panels, 8 DOFs across two bodies: rows must land at
    # the original panel positions and every source column/DOF is retained.
    P, ndof = 4, 8
    hull_mask = np.array([True, False, True, False])
    H = np.flatnonzero(hull_mask)
    L = np.flatnonzero(~hull_mask)
    rng = np.random.default_rng(5)
    areas = np.full(P, 0.5)
    A = np.zeros((P, P), dtype=complex)
    b = np.zeros((P, ndof), dtype=complex)
    S = rng.standard_normal((P, P)) + 1j * rng.standard_normal((P, P))
    K = rng.standard_normal((P, P)) + 1j * rng.standard_normal((P, P))
    w = rng.random(P) + 0.1
    g = rng.standard_normal((P, ndof))
    g[L] = 0.0
    tool.accumulate_quadrature(A, b, S, K, w, g, H, L, areas)
    # hull rows sit exactly at the hull panel positions
    for row, i in enumerate(H):
        expected = sum(np.conj(S[i2, i]) * w[i2] * K[i2, :] for i2 in H)
        assert np.allclose(A[i], expected, atol=1e-13)
    # cross-body source coupling is retained: a lid column enters hull rows
    assert np.all(A[np.ix_(H, L)] != 0)
    # the full 8-column right-hand side is retained on hull rows
    assert b.shape == (P, ndof)
    assert np.all(b[H] != 0)
    assert np.all(b[L] == 0)


# --------------------------------------------------------------------------
# pure solve / verdict / force checks
# --------------------------------------------------------------------------

def test_solve_equilibrated_preserves_equations(tool):
    rng = np.random.default_rng(13)
    P, ndof = 5, 3
    A = rng.standard_normal((P, P)) + 1j * rng.standard_normal((P, P))
    A += 3.0 * np.eye(P)
    b = rng.standard_normal((P, ndof)) + 1j * rng.standard_normal((P, ndof))
    sigma, meta = tool.solve_equilibrated(A, b)
    assert np.allclose(A @ sigma, b, atol=1e-9)
    assert meta['equation_residual_max_abs'] < 1e-9
    assert all(r < 1e-9 for r in meta['equation_residual_rel_per_dof'])


def test_solve_equilibrated_singular_refused(tool):
    A = np.zeros((3, 3), dtype=complex)
    b = np.ones((3, 1))
    with pytest.raises(tool.CandidateSolveError):
        tool.solve_equilibrated(A, b)


def test_closure_verdict_normalization(tool):
    # unchanged original gate: rel 0.03, abs 1e-8, floor 1e-8,
    # scale = max(|far|, |near|)
    verdict = tool.closure_verdict(1.0, 1.02)
    assert verdict['verdict'] == 'pass'
    assert verdict['scale'] == 1.02
    assert verdict['limit'] == 1e-8 + 0.03 * 1.02
    verdict = tool.closure_verdict(1.0, 0.5)
    assert verdict['verdict'] == 'fail'
    # near-zero floor: tiny values are compared at the 1e-8 scale
    verdict = tool.closure_verdict(1e-12, 0.0)
    assert verdict['scale'] == 1e-8
    assert verdict['limit'] == 1e-8 + 0.03e-8
    assert verdict['verdict'] == 'pass'
    with pytest.raises(tool.CandidateRefusal):
        tool.closure_verdict(float('nan'), 1.0)


def test_physical_hull_force_convention(tool):
    # F[d, e] = -sum_hq w conj(g_d) p_e over hull panels, complex trace.
    P, nq, ndof = 4, 2, 2
    rng = np.random.default_rng(17)
    hull = np.array([True, True, False, False])
    w = rng.random((P, nq)) + 0.1
    g = rng.standard_normal((P, nq, ndof)) + 1j * rng.standard_normal((P, nq, ndof))
    p = rng.standard_normal((P, nq, ndof)) + 1j * rng.standard_normal((P, nq, ndof))
    F = tool.physical_hull_force(w, hull, g, p)
    expected = np.zeros((ndof, ndof), dtype=complex)
    for i in range(P):
        if hull[i]:
            for q in range(nq):
                for d in range(ndof):
                    for e in range(ndof):
                        expected[d, e] -= w[i, q] * np.conj(g[i, q, d]) * p[i, q, e]
    assert np.allclose(F, expected, atol=1e-13)


# --------------------------------------------------------------------------
# CLI-level checks on synthetic cases
# --------------------------------------------------------------------------

def test_synthetic_solve_and_recovery(tmp_path, tool):
    case_dir, parent_dir = make_solvable_case(tmp_path)
    output_dir = tmp_path / 'out'
    code = run_tool(tool, case_dir, parent_dir, output_dir, rate=synthetic_rate(4),
                    expected_case_hash='synthetic-weak-neumann-case',
                    retained_dofs=[0, 1], angular_counts=[16, 32])
    assert code == 0
    status = json.loads((output_dir / 'status.json').read_text())
    assert status['state'] == 'completed_unqualified'
    assert status['source_admission'] is False
    assert status['qualification'] is False
    assert status['solve_attempts_this_run'] == 1
    report = json.loads((output_dir / 'candidate-solve.json').read_text())
    assert report['status'] == 'completed_unqualified'
    assert report['source_admission'] is False
    assert report['qualification'] is False
    assert report['identity']['inputs_unchanged'] is True
    assert report['identity']['parent_summary_sha256_pre'] \
        == report['identity']['parent_summary_sha256_post']
    assert report['effective_controls']['retained_dofs'] == [0, 1]
    assert report['effective_controls']['angular_counts'] == [16, 32]
    assert report['effective_controls']['rlimit_mib'] == 1536
    assert report['effective_controls']['expected_case_hash'] \
        == 'synthetic-weak-neumann-case'
    assert report['case']['dispersion_relative_residual'] < 1e-9
    # recovery through the exact stored operators, recomputed independently
    saved = np.load(output_dir / 'candidate-solve.npz')
    sigma = saved['sigma']
    case = np.load(case_dir / 'input.npz')
    trace = case['rigid_trace']
    weights = case['weights']
    hull = case['hull_mask']
    rho, omega = 1025.0, 1.5
    for qi in range(4):
        op = np.load(case_dir / f'operator-{qi}.npz')
        S, K = op['s'], op['k']
        assert np.allclose(saved['phi'][:, qi, :], S @ sigma, atol=1e-10)
        assert np.allclose(saved['pressure'][:, qi, :],
                           1j * rho * omega * (S @ sigma), atol=1e-10)
        assert np.allclose(saved['node_residual'][:, qi, :],
                           K @ sigma - trace[:, qi, :], atol=1e-9)
    # force convention and near/work identity for the real rigid trace
    F = saved['force_per_velocity']
    expected_F = tool.physical_hull_force(weights, hull, trace, saved['pressure'])
    assert np.allclose(F, expected_F, atol=1e-10)
    rate = saved['rate']
    near = tool.near_power(F, rate)
    work = tool.boundary_work(weights, hull, saved['pressure'] @ rate,
                              trace @ rate)
    assert abs(near - work) < 1e-9 * max(1.0, abs(near))
    # lid rows of the assembled system are the original mean equations
    A, b = saved['A'], saved['b']
    weak = A @ sigma - b
    mean_residual = saved['mean_residual']
    assert np.allclose(weak[~hull], mean_residual[~hull], atol=1e-9)
    # full coupling preserved: 12-style DOF inventory intact in the report
    assert len(report['case']['dofs']) == 4
    assert report['case']['retained_dofs'] == [0, 1]
    assert report['coefficients']['damping_symmetric_eigenvalues_unclipped_full']


def test_refusals(tmp_path, tool):
    rate = synthetic_rate(4)
    # missing operator file
    case_dir, parent_dir = make_solvable_case(tmp_path / 'missing')
    (case_dir / 'operator-3.npz').unlink()
    code = run_tool(tool, case_dir, parent_dir, tmp_path / 'out-missing', rate=rate,
                    expected_case_hash='synthetic-weak-neumann-case',
                    retained_dofs=[0, 1], angular_counts=[16])
    assert code == 1
    assert json.loads((tmp_path / 'out-missing' / 'status.json')
                      .read_text())['state'] == 'refused'

    # case hash mismatch
    case_dir, parent_dir = make_solvable_case(tmp_path / 'hash')
    code = run_tool(tool, case_dir, parent_dir, tmp_path / 'out-hash', rate=rate,
                    expected_case_hash='wrong-hash')
    assert code == 1

    # manifest mismatch: mutate an input array after the manifest was written
    case_dir, parent_dir = make_solvable_case(tmp_path / 'manifest')
    with np.load(case_dir / 'input.npz') as data:
        arrays = {k: data[k] for k in data.files}
    arrays['weights'] = arrays['weights'] * 1.5
    np.savez(case_dir / 'input.npz', **arrays)
    code = run_tool(tool, case_dir, parent_dir, tmp_path / 'out-manifest', rate=rate,
                    expected_case_hash='synthetic-weak-neumann-case',
                    retained_dofs=[0, 1], angular_counts=[16])
    assert code == 1

    # nonfinite operator entries
    case_dir, parent_dir = make_solvable_case(tmp_path / 'nonfinite')
    rng = np.random.default_rng(7)
    operators = build_operators(6, rng)
    write_synthetic_case(case_dir, hull_mask=(True, True, False, False, True, False),
                         ndof=4, operators=operators,
                         nonfinite_operator=True)
    code = run_tool(tool, case_dir, parent_dir, tmp_path / 'out-nonfinite', rate=rate,
                    expected_case_hash='synthetic-weak-neumann-case',
                    retained_dofs=[0, 1], angular_counts=[16])
    assert code == 1

    # rigid trace nonzero on the lid (manifest kept consistent)
    case_dir, parent_dir = make_solvable_case(tmp_path / 'lidtrace')
    mutate_input_with_manifest(
        case_dir, lambda arrays: arrays['rigid_trace'].__setitem__((2, 0, 0), 0.5))
    code = run_tool(tool, case_dir, parent_dir, tmp_path / 'out-lidtrace', rate=rate,
                    expected_case_hash='synthetic-weak-neumann-case',
                    retained_dofs=[0, 1], angular_counts=[16])
    assert code == 1

    # output inside the immutable case tree
    case_dir, parent_dir = make_solvable_case(tmp_path / 'inside')
    code = run_tool(tool, case_dir, parent_dir, case_dir / 'escape', rate=rate,
                    expected_case_hash='synthetic-weak-neumann-case',
                    retained_dofs=[0, 1], angular_counts=[16])
    assert code == 1
    assert not (case_dir / 'escape').exists()

    # existing output directory is refused
    case_dir, parent_dir = make_solvable_case(tmp_path / 'existing')
    (tmp_path / 'out-existing').mkdir()
    code = run_tool(tool, case_dir, parent_dir, tmp_path / 'out-existing', rate=rate,
                    expected_case_hash='synthetic-weak-neumann-case',
                    retained_dofs=[0, 1], angular_counts=[16])
    assert code == 1


def test_no_input_mutation_fresh_output(tmp_path, tool):
    case_dir, parent_dir = make_solvable_case(tmp_path / 'immutable')
    files = sorted(p for p in case_dir.iterdir() if p.is_file())
    hashes = {p.name: _sha256_file(p) for p in files}
    parent_hash = _sha256_file(parent_dir / 'summary.json')
    output_dir = tmp_path / 'fresh-out'
    code = run_tool(tool, case_dir, parent_dir, output_dir, rate=synthetic_rate(4),
                    expected_case_hash='synthetic-weak-neumann-case',
                    retained_dofs=[0], angular_counts=[16])
    assert code == 0
    assert all(_sha256_file(p) == hashes[p.name] for p in files)
    assert _sha256_file(parent_dir / 'summary.json') == parent_hash
    assert (output_dir / 'candidate-solve.json').is_file()
    assert (output_dir / 'candidate-solve.npz').is_file()
    assert (output_dir / 'candidate-solve.md').is_file()
    report = json.loads((output_dir / 'candidate-solve.json').read_text())
    assert report['identity']['input_file_hashes_pre'] \
        == report['identity']['input_file_hashes_post']
    assert report['identity']['parent_summary_sha256_pre'] \
        == report['identity']['parent_summary_sha256_post']


def test_singular_solve_failure_persisted(tmp_path, tool):
    rng = np.random.default_rng(7)
    case_dir = tmp_path / 'g0' / 'g1' / 'g2' / 'case'
    operators = build_operators(6, rng, singular=True)
    write_synthetic_case(case_dir, hull_mask=(True, True, False, False, True, False),
                         ndof=4, operators=operators)
    parent_dir = tmp_path / 'g0' / 'parent'
    write_synthetic_parent(parent_dir, case_hash='synthetic-weak-neumann-case',
                           rate=[-0.6, 0.0, 0.3, 0.0], ndof=4)
    output_dir = tmp_path / 'out'
    code = run_tool(tool, case_dir, parent_dir, output_dir, rate=synthetic_rate(4),
                    expected_case_hash='synthetic-weak-neumann-case',
                    retained_dofs=[0, 1], angular_counts=[16])
    assert code == 2
    status = json.loads((output_dir / 'status.json').read_text())
    assert status['state'] == 'solve_failed'
    assert status['solve_attempts_this_run'] == 1
    assert not (output_dir / 'candidate-solve.json').exists()


def test_status_saved_before_solve(tmp_path, tool, monkeypatch):
    case_dir, parent_dir = make_solvable_case(tmp_path / 'before-solve')
    output_dir = tmp_path / 'out'
    observed = {}

    def fake_solve(A, b):
        status_path = output_dir / 'status.json'
        observed['exists'] = status_path.is_file()
        if status_path.is_file():
            status = json.loads(status_path.read_text())
            observed.update(state=status.get('state'),
                            attempts=status.get('solve_attempts_this_run'),
                            recorded=status.get('solve_recorded_before_solve'))
        raise RuntimeError('simulated cancellation during solve')

    monkeypatch.setattr(tool.np.linalg, 'solve', fake_solve)
    code = run_tool(tool, case_dir, parent_dir, output_dir, rate=synthetic_rate(4),
                    expected_case_hash='synthetic-weak-neumann-case',
                    retained_dofs=[0, 1], angular_counts=[16])
    assert code == 3
    assert observed['exists'] is True
    assert observed['state'] == 'pre_solve'
    assert observed['attempts'] == 1
    assert observed['recorded'] is True
    status = json.loads((output_dir / 'status.json').read_text())
    assert status['state'] == 'failed'
    assert status['solve_attempts_this_run'] == 1


def test_cli_requires_arguments(tool):
    with pytest.raises(SystemExit):
        tool.parse_args([])


# --------------------------------------------------------------------------
# production-path failure-write containment (main-level, byte+inventory)
# --------------------------------------------------------------------------

def test_main_output_equals_case_never_written(tmp_path, tool):
    # `--output CASE` must refuse without touching the input tree: the input
    # status.json and every other case byte must be identical pre/post.
    case_dir, parent_dir = make_solvable_case(tmp_path / 'eq')
    before = snapshot_tree(case_dir)
    code = run_tool(tool, case_dir, parent_dir, case_dir, rate=synthetic_rate(4),
                    expected_case_hash='synthetic-weak-neumann-case',
                    retained_dofs=[0, 1], angular_counts=[16])
    assert code == 1
    assert snapshot_tree(case_dir) == before


def test_main_existing_output_sentinels_untouched(tmp_path, tool):
    # refusal of a pre-existing output must not rewrite its old evidence
    case_dir, parent_dir = make_solvable_case(tmp_path / 'old')
    out = tmp_path / 'old-out'
    out.mkdir()
    (out / 'status.json').write_text(
        json.dumps({'state': 'completed', 'evidence': 'old run'}), encoding='utf-8')
    (out / 'sentinel.bin').write_bytes(b'\x00\x01old-evidence')
    (out / 'nested').mkdir()
    (out / 'nested' / 'keep.txt').write_text('keep', encoding='utf-8')
    before = snapshot_tree(out)
    code = run_tool(tool, case_dir, parent_dir, out, rate=synthetic_rate(4),
                    expected_case_hash='synthetic-weak-neumann-case',
                    retained_dofs=[0, 1], angular_counts=[16])
    assert code == 1
    assert snapshot_tree(out) == before


def test_main_output_link_to_case_refused_no_write(tmp_path, tool):
    # an output path that resolves through a symlink/junction into the case
    # tree is refused and nothing is written through the link
    case_dir, parent_dir = make_solvable_case(tmp_path / 'link')
    link = tmp_path / 'case-link'
    make_link(case_dir, link)
    try:
        case_before = snapshot_tree(case_dir)
        code = run_tool(tool, case_dir, parent_dir, link, rate=synthetic_rate(4),
                        expected_case_hash='synthetic-weak-neumann-case',
                        retained_dofs=[0, 1], angular_counts=[16])
        assert code == 1
        assert snapshot_tree(case_dir) == case_before
    finally:
        remove_link(link)


def test_main_missing_case_refused_no_output_created(tmp_path, tool):
    parent_dir = tmp_path / 'g0' / 'parent'
    parent_dir.mkdir(parents=True)
    missing_case = tmp_path / 'g0' / 'g1' / 'g2' / 'no-case'
    out = tmp_path / 'out'
    code = run_tool(tool, missing_case, parent_dir, out, rate=synthetic_rate(4),
                    expected_case_hash='synthetic-weak-neumann-case',
                    retained_dofs=[0, 1], angular_counts=[16])
    assert code == 1
    assert not out.exists()


def test_main_refusal_after_ownership_recorded_in_fresh_output(tmp_path, tool):
    # failure evidence is kept, but only inside the fresh directory this run
    # created and owns (here: unauthorized case hash after the output exists)
    case_dir, parent_dir = make_solvable_case(tmp_path / 'unauth')
    out = tmp_path / 'out-unauth'
    code = run_tool(tool, case_dir, parent_dir, out, rate=synthetic_rate(4),
                    expected_case_hash='wrong-hash')
    assert code == 1
    status = json.loads((out / 'status.json').read_text())
    assert status['state'] == 'refused'
    assert 'expected' in status['reason'] or 'hash' in status['reason']
    assert not (out / 'candidate-solve.json').exists()


# --------------------------------------------------------------------------
# parent summary authentication
# --------------------------------------------------------------------------

@pytest.mark.parametrize('mutation', ['omega', 'rho', 'water_depth', 'k',
                                      'rate', 'missing_k'])
def test_parent_summary_authentication_refusals(tmp_path, tool, mutation):
    case_dir, parent_dir = make_solvable_case(tmp_path / f'summary-{mutation}')
    summary = json.loads((parent_dir / 'summary.json').read_text())
    if mutation == 'rate':
        # same case_hash and vector length, different values: not authority
        summary['candidate']['rate'] = [v + 0.001 for v in summary['candidate']['rate']]
    elif mutation == 'missing_k':
        del summary['candidate']['k']
    else:
        summary['candidate'][mutation] = float(summary['candidate'][mutation]) * 1.001
    (parent_dir / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    out = tmp_path / 'out'
    code = run_tool(tool, case_dir, parent_dir, out, rate=synthetic_rate(4),
                    expected_case_hash='synthetic-weak-neumann-case',
                    retained_dofs=[0, 1], angular_counts=[16])
    assert code == 1
    assert not (out / 'candidate-solve.json').exists()
    status = json.loads((out / 'status.json').read_text())
    assert status['state'] == 'refused'


# --------------------------------------------------------------------------
# fixed actual CLI control gates
# --------------------------------------------------------------------------

def test_cli_rejects_changed_duplicate_reordered_subset_controls(tmp_path, tool):
    case_dir, parent_dir = make_solvable_case(tmp_path / 'gates')
    rate = synthetic_rate(4)
    authorized = {'expected_case_hash': 'synthetic-weak-neumann-case',
                  'retained_dofs': [0, 1], 'angular_counts': [16, 32],
                  'rlimit_mib': 1536}
    variants = [
        ('retained_dofs', [1, 0]),           # reordered
        ('retained_dofs', [0, 0]),           # duplicate
        ('retained_dofs', [0]),              # subset
        ('retained_dofs', [0, 1, 2]),        # extension
        ('angular_counts', [32, 16]),        # reordered
        ('angular_counts', [16, 16]),        # duplicate
        ('angular_counts', [16]),            # subset
        ('angular_counts', [64, 128, 256]),  # changed (actual values, not authorized)
        ('rlimit_mib', 2048),                # changed
        ('expected_case_hash', 'other-hash'),  # changed
    ]
    for index, (key, value) in enumerate(variants):
        kwargs = dict(authorized)
        kwargs[key] = value
        out = tmp_path / f'gate-out-{index}'
        code = run_tool(tool, case_dir, parent_dir, out, rate=rate,
                        authorized=authorized, **kwargs)
        assert code == 1, (key, value)
        # the gate refuses before any output directory is created
        assert not out.exists()


def test_actual_cli_authority_not_broadened(tmp_path):
    # a fresh module with the real registered actual controls: neither a
    # substituted --expected-case-hash nor the actual hash itself admits a
    # synthetic case through the actual CLI gates
    fresh = _load_tool()
    case_dir, parent_dir = make_solvable_case(tmp_path / 'authority')
    out = tmp_path / 'out-override'
    code = fresh.main(['--case', str(case_dir), '--parent', str(parent_dir),
                       '--output', str(out),
                       '--expected-case-hash', 'synthetic-weak-neumann-case'])
    assert code == 1
    assert not out.exists()
    out2 = tmp_path / 'out-actual-hash'
    code = fresh.main(['--case', str(case_dir), '--parent', str(parent_dir),
                       '--output', str(out2),
                       '--expected-case-hash', fresh.DEFAULT_CASE_HASH])
    assert code == 1
    # the gate passed (registered controls) but the synthetic case itself was
    # refused by the loader; the refusal is recorded only in the fresh output
    status = json.loads((out2 / 'status.json').read_text())
    assert status['state'] == 'refused'
    assert not (out2 / 'candidate-solve.json').exists()
