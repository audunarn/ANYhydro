# SPDX-License-Identifier: MPL-2.0
"""Tests for the retained panel-pair direct-Rankine discriminator.

Accuracy is checked against independent references only: closed-form solid
angle and lamina-potential fixtures, an exact in-plane equilateral limit, far
field closed forms, exact similarity scaling, exact rigid-transform
invariance, additivity, finite-difference receiver gradients, orientation
independence, and the independently derived radial-exact angular quadrature.
The end-to-end synthetic case checks mechanics (identity, manifest, refusals,
counters, reconstruction and attribution bookkeeping) with an independent
known antisymmetric perturbation, not the implementation as its own oracle.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audit_retained_panel_pairs as ppd  # noqa: E402

LEVELS = (16, 32, 64)


def _counters():
    return {'primitive_comparisons': 0, 'node_comparisons': 0,
            'analytic_evaluations': 0, 'radial_evaluations': 0,
            'duffy_evaluations': 0}


def _equilateral(side=0.3):
    h = side * math.sqrt(3.0) / 2.0
    return np.array([[0.0, 0.0, 0.0], [side, 0.0, 0.0], [side / 2.0, h, 0.0]])


def _analytic(x, tri):
    result, status = ppd.analytic_triangle_integrals(np.asarray(x, float), tri)
    assert status == 'ok', status
    return result


def _radial(x, tri, level=64, with_normal=False):
    value, normal, status = ppd.radial_triangle_scalar(
        np.asarray(x, float), tri, level, with_normal=with_normal)
    assert status == 'ok', status
    return value, normal


def _node_record(x, tri):
    record, scalar = ppd.node_reference_record(np.asarray(x, float), tri, LEVELS, _counters())
    return record, scalar


# ---------------------------------------------------------------- primitives

def test_edge_antiderivative_matches_quadrature():
    nodes, weights = np.polynomial.legendre.leggauss(2000)
    for p, h in [(0.2, 0.5), (0.05, 1e-4), (1.0, 2.0), (-0.3, 0.7), (0.4, 0.0)]:
        big_p = math.sqrt(p * p + h * h)
        s0, s1 = -0.7, 1.3
        mid = 0.5 * (s0 + s1)
        half = 0.5 * (s1 - s0)
        pts = mid + half * nodes
        quad = half * float(np.sum(weights / (np.sqrt(pts * pts + big_p * big_p) + abs(h))))
        closed = (ppd._edge_antiderivative(s1, big_p, abs(p), abs(h))
                  - ppd._edge_antiderivative(s0, big_p, abs(p), abs(h)))
        assert abs(closed - quad) <= 1e-12 * max(abs(quad), 1e-12), (p, h)


def test_far_field_closed_form_scalar_and_gradient():
    tri = _equilateral(0.3)
    d = 3.0
    x = tri.mean(axis=0) + np.array([0.0, 0.0, d])
    (phi1, grad) = _analytic(x, tri)
    area = ppd.triangle_area(tri)
    closed = area * (-1.0 / (4.0 * math.pi * d))
    assert abs(-phi1 / (4.0 * math.pi) - closed) <= (0.3 / d) ** 2 * abs(closed)
    grad_closed = area * (x - tri.mean(axis=0)) / (4.0 * math.pi * d ** 3)
    err = float(np.linalg.norm(grad / (4.0 * math.pi) - grad_closed))
    assert err <= (0.3 / d) ** 2 * float(np.linalg.norm(grad_closed))
    rad, _ = _radial(x, tri)
    assert abs(-rad / (4.0 * math.pi) - closed) <= (0.3 / d) ** 2 * abs(closed)


def test_solid_angle_closed_form_fixtures():
    # octant trihedral: E.n = -pi/2 along the outward tetrahedral normal
    tri = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    (phi1, grad) = _analytic(np.zeros(3), tri)
    n_hat = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    assert abs(float(grad @ n_hat) + math.pi / 2.0) <= 1e-12
    assert float(np.linalg.norm(grad - (grad @ n_hat) * n_hat)) <= 1e-12
    # regular tetrahedron face seen from the center: E.n = -pi
    tri = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0]]) / math.sqrt(3.0)
    (_, grad) = _analytic(np.zeros(3), tri)
    n_hat = np.array([1.0, 1.0, -1.0]) / math.sqrt(3.0)
    assert abs(float(grad @ n_hat) + math.pi) <= 1e-12


def test_solid_angle_against_quadrature():
    tri = _equilateral(0.3)
    x = tri.mean(axis=0) + np.array([0.05, -0.02, 0.4])
    (_, grad) = _analytic(x, tri)
    e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
    n_hat = np.cross(e1, e2)
    n_hat = n_hat / np.linalg.norm(n_hat)
    pts, wts = ppd.duffy_rule(tri, 64)
    factor = 2.0 * ppd.triangle_area(tri) / 3.0
    d = x[None, :] - pts
    r = np.sqrt(np.sum(d * d, axis=1))
    omega_quad = factor * float(np.sum(wts * ((d @ n_hat) / r ** 3)))
    assert abs(float(grad @ n_hat) - omega_quad) <= 1e-6 * max(abs(omega_quad), 1e-12)


def test_in_plane_equilateral_closed_form_limit():
    # Phi(h) + 2*pi*h -> a*sqrt(3)*ln(2+sqrt(3)) as h -> 0+ (projection inside)
    a = 0.3
    tri = _equilateral(a)
    closed = a * math.sqrt(3.0) * math.log(2.0 + math.sqrt(3.0))
    for h in (1e-6 * a, 1e-8 * a):
        x = tri.mean(axis=0) + np.array([0.0, 0.0, h])
        (phi1, _) = _analytic(x, tri)
        assert abs(phi1 + 2.0 * math.pi * h - closed) <= 1e-9 * closed, h
        rad, _ = _radial(x, tri)
        assert abs(rad + 2.0 * math.pi * h - closed) <= 1e-9 * closed, h


def test_similarity_scaling_exact():
    tri = _equilateral(0.3)
    x = tri.mean(axis=0) + np.array([0.05, -0.02, 0.4])
    (phi1, grad) = _analytic(x, tri)
    (phi1s, grads) = _analytic(2.0 * x, 2.0 * tri)
    assert abs(phi1s - 2.0 * phi1) <= 1e-12 * abs(phi1)
    # gradient integral is similarity invariant: s^2 dA cancels s/s^3
    assert float(np.linalg.norm(grads - grad)) <= 1e-12 * float(np.linalg.norm(grad))
    rad, _ = _radial(x, tri)
    rads, _ = _radial(2.0 * x, 2.0 * tri)
    assert abs(rads - 2.0 * rad) <= 1e-12 * abs(rad)


def test_rigid_transform_invariance():
    tri = _equilateral(0.3)
    x = tri.mean(axis=0) + np.array([0.05, -0.02, 0.4])
    ca, sa = math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)
    cb, sb = math.cos(math.pi / 4.0), math.sin(math.pi / 4.0)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
    ry = np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]])
    rot = ry @ rx
    shift = np.array([10.0, -5.0, 3.0])
    (phi1, grad) = _analytic(x, tri)
    (phi1r, gradr) = _analytic(rot @ x + shift, tri @ rot.T + shift)
    assert abs(phi1r - phi1) <= 1e-12 * abs(phi1)
    assert float(np.linalg.norm(gradr - rot @ grad)) <= 1e-12 * float(np.linalg.norm(grad))
    rad, _ = _radial(x, tri)
    radr, _ = _radial(rot @ x + shift, tri @ rot.T + shift)
    assert abs(radr - rad) <= 1e-12 * abs(rad)


def test_orientation_flip_invariance():
    tri = _equilateral(0.3)
    x = tri.mean(axis=0) + np.array([0.05, -0.02, 0.4])
    (phi1, grad) = _analytic(x, tri)
    (phi1f, gradf) = _analytic(x, tri[[0, 2, 1]])
    assert abs(phi1f - phi1) <= 1e-12 * abs(phi1)
    assert float(np.linalg.norm(gradf - grad)) <= 1e-12 * float(np.linalg.norm(grad))
    rad, _ = _radial(x, tri)
    radf, _ = _radial(x, tri[[0, 2, 1]])
    assert abs(radf - rad) <= 1e-12 * abs(rad)


def test_additivity_over_centroid_split():
    tri = _equilateral(0.3)
    c = tri.mean(axis=0)
    x = c + np.array([0.0, 0.0, 0.5])
    (total, _) = _analytic(x, tri)
    parts = 0.0
    for k in range(3):
        (part, _) = _analytic(x, np.array([c, tri[k], tri[(k + 1) % 3]]))
        parts += part
    assert abs(total - parts) <= 1e-11 * max(abs(total), 1e-12)


def test_receiver_gradient_finite_difference():
    tri = _equilateral(0.3)
    x = tri.mean(axis=0) + np.array([0.02, -0.01, 0.35])
    (phi1, grad) = _analytic(x, tri)
    h = 1e-4
    for k in range(3):
        step = np.zeros(3)
        step[k] = h
        (plus, _) = _analytic(x + step, tri)
        (minus, _) = _analytic(x - step, tri)
        fd = -(plus - minus) / (2.0 * h)  # grad = -d/dx Phi1
        assert abs(fd - grad[k]) <= 1e-6 * max(abs(grad[k]), 1e-6)


def test_radial_matches_analytic_varied_geometry():
    tri = _equilateral(0.3)
    c = tri.mean(axis=0)
    cases = {
        'far': c + np.array([0.0, 0.0, 3.0]),
        'oblique': c + np.array([0.05, -0.02, 0.4]),
        'near_coplanar': c + np.array([0.01, -0.03, 1e-6 * 0.3]),
        'in_plane_exterior': np.array([0.7, -0.1, 0.0]),
        'beyond_edge_offplane': np.array([0.15, -0.35, 0.02]),
    }
    for name, x in cases.items():
        (phi1, grad) = _analytic(x, tri)
        rad, normal_deriv = _radial(x, tri, with_normal=True)
        assert abs(rad - phi1) <= 1e-11 * max(abs(phi1), 1e-12), name
        frame = ppd._plane_frame(tri)
        n_hat = frame[0]
        # E.n = -dPhi1/dh
        assert abs(-normal_deriv - float(grad @ n_hat)) <= 1e-9 * max(abs(grad @ n_hat), 1e-9), name


def test_near_coplanar_controls_pass_and_legacy_duffy_fails():
    tri = _equilateral(0.3)
    x = 0.2 * tri[0] + 0.3 * tri[1] + 0.5 * tri[2] + np.array([0.0, 0.0, 1e-6 * 0.3])
    record, scalar = _node_record(x, tri)
    assert record['status'] == 'ok'
    assert record['radial_status'] == 'ok'
    assert record['radial_scalar_pass'] is True
    assert record['analytic_match_pass'] is True
    assert record['scalar_pass'] is True
    assert record['grad_pass'] is True
    assert record['observed_error_estimate'] <= ppd.SCALAR_ABS + ppd.SCALAR_REL * abs(scalar)
    # the old centroid-split Duffy reference still fails here; it stays labelled
    assert record['duffy_legacy']['scalar_pass'] is False
    assert record['duffy_legacy']['label'] == 'legacy_centroid_split_duffy'


def test_refusals_singular_degenerate_nonfinite_edge_line():
    tri = _equilateral(0.3)
    c = tri.mean(axis=0)
    def first_and_status(func, x, t):
        out = func(x, t)
        return out[0], out[-1]

    radial16 = lambda x, t: ppd.radial_triangle_scalar(x, t, 16)
    duffy16 = lambda x, t: ppd.duffy_node_integrals(x, t, (16,))
    for func in (ppd.analytic_triangle_integrals, radial16, duffy16):
        _v, status = first_and_status(func, c, tri)
        assert status == 'on_or_near_source_triangle', status
        _v, status = first_and_status(func, tri[1], tri)
        assert status == 'on_or_near_source_triangle', status
    degenerate = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    _v, status = ppd.analytic_triangle_integrals(np.array([0.0, 0.0, 1.0]), degenerate)
    assert status == 'degenerate_source_triangle'
    _v, status = ppd.analytic_triangle_integrals(np.array([np.nan, 0.0, 1.0]), tri)
    assert status == 'nonfinite'
    # in-plane receiver on the edge line, outside the segment
    on_line = np.array([0.5, 0.0, 0.0])
    _v, status = ppd.analytic_triangle_integrals(on_line, tri)
    assert status == 'receiver_on_edge_line'
    # the radial rule skips the degenerate sector (contribution exactly zero)
    # and evaluates the remaining two signed sectors exactly
    value, _n, status = ppd.radial_triangle_scalar(on_line, tri, 64)
    assert status == 'ok'
    assert math.isfinite(value)


def test_duffy_rule_partition_of_unity():
    tri = _equilateral()
    area = ppd.triangle_area(tri)
    for level in (2, 16, 64):
        pts, wts = ppd.duffy_rule(tri, level)
        assert pts.shape == (3 * level * level, 3)
        assert abs(float(np.sum(wts)) * 2.0 * area / 3.0 - area) <= 1e-12 * area
        e1, e2 = tri[1] - tri[0], tri[2] - tri[0]
        n = np.cross(e1, e2)
        n2 = float(n @ n)
        assert np.max(np.abs((pts - tri[0]) @ n)) <= 1e-12 * math.sqrt(n2)
        lam1 = (np.cross(pts - tri[0], e2) @ n) / n2
        lam2 = (np.cross(e1, pts - tri[0]) @ n) / n2
        assert lam1.min() >= -1e-9 and lam2.min() >= -1e-9
        assert (lam1 + lam2).max() <= 1.0 + 1e-9


def test_double_quadrature_symmetry():
    tri_a = _equilateral(0.3)
    tri_b = tri_a + np.array([0.4, 0.1, 0.05])
    nodes_a = np.array([tri_a.mean(axis=0) + 0.1 * (tri_a[k] - tri_a.mean(axis=0)) for k in range(3)] + [tri_a.mean(axis=0)])
    nodes_b = np.array([tri_b.mean(axis=0) + 0.1 * (tri_b[k] - tri_b.mean(axis=0)) for k in range(3)] + [tri_b.mean(axis=0)])
    w_a = np.full(4, ppd.triangle_area(tri_a) / 4.0)
    w_b = np.full(4, ppd.triangle_area(tri_b) / 4.0)
    v_ab = ppd.double_quadrature(nodes_a, w_a, nodes_b, w_b)
    v_ba = ppd.double_quadrature(nodes_b, w_b, nodes_a, w_a)
    assert abs(v_ab - v_ba) <= 1e-14 * max(abs(v_ab), 1e-300)


# -------------------------------------------------------------------- guards

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest(arrays):
    return {name: {'shape': list(arr.shape), 'dtype': arr.dtype.name,
                   'sha256': ppd._sha256_array(arr)}
            for name, arr in arrays.items()}


def _build_synthetic_case(case_dir):
    verts = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
        [5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [5.0, 1.0, 0.0],
        [5.0, 0.0, 1.0], [6.0, 0.0, 1.0], [5.0, 1.0, 1.0]])
    faces = np.array([[0, 1, 2], [1, 3, 2], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    count = faces.shape[0]
    points = np.zeros((count, 4, 3))
    weights = np.zeros((count, 4))
    for p in range(count):
        tri = verts[faces[p]]
        c = tri.mean(axis=0)
        nodes = np.array([c + 0.1 * (tri[k] - c) for k in range(3)] + [c])
        points[p] = nodes
        weights[p] = ppd.triangle_area(tri) / 4.0
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (count, 1))
    owner = np.array([0, 0, 1, 1])
    hull_mask = np.array([True, True, False, False])
    arrays = {'vertices': verts, 'faces': faces, 'points': points,
              'weights': weights, 'normals': normals, 'owner': owner,
              'hull_mask': hull_mask}
    np.savez(case_dir / 'input.npz', **arrays)
    np.savez(case_dir / 'radiation.npz', sources=np.zeros((count, 12)))
    # stored operators: exact analytic mixed Rankine values plus a known
    # antisymmetric perturbation 0.05*(j-i) per node
    eps = 0.05
    for q in range(4):
        s = np.zeros((count, count), dtype=np.complex128)
        for i in range(count):
            for j in range(count):
                if i == j:
                    s[i, j] = 0.0  # self term: never evaluated by the pair tool
                    continue
                (phi1, _), status = ppd.analytic_triangle_integrals(points[i, q], verts[faces[j]])
                assert status == 'ok', status
                s[i, j] = -phi1 / (4.0 * math.pi) + eps * (j - i)
        np.savez(case_dir / f'operator-{q}.npz', s=s, k=np.zeros((count, count), dtype=np.complex128))
    (case_dir / 'case.json').write_text(json.dumps({
        'case_hash': 'test-hash-000', 'body_ids': ['body-a', 'body-b'],
        'inventory': {'body-a': 'mesh', 'body-b': 'mesh'},
        'quadrature': 'Gauss-Legendre 2', 'version': 'test',
        'snapshot_arrays': _manifest(arrays)}))
    (case_dir / 'status.json').write_text(json.dumps({'state': 'finished'}))
    return eps


def _rewrite_input(case_dir, arrays):
    np.savez(case_dir / 'input.npz', **arrays)


def test_end_to_end_synthetic_case_identity_and_attribution():
    saved_hash = ppd.EXPECTED_CASE_HASH
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_dir = root / 'a' / 'b' / 'c' / 'case'
        out_dir = root / 'out'
        case_dir.mkdir(parents=True)
        eps = _build_synthetic_case(case_dir)
        ppd.EXPECTED_CASE_HASH = 'test-hash-000'
        try:
            before = {p.name: _sha(p) for p in sorted(case_dir.iterdir())}
            rc = ppd.main(['--case-dir', str(case_dir), '--output-dir', str(out_dir), '--max-pairs', '6'])
            assert rc == 0
            report = json.loads((out_dir / 'panel-pair-discriminator.json').read_text())
            assert report['identity']['case_hash_agrees'] is True
            assert report['identity']['manifest_verified'] is True
            assert report['identity']['input_files_unchanged'] is True
            assert report['counters']['pairs_selected'] == 6
            assert report['counters']['primitive_comparisons'] == 6 * 2 * 4
            assert report['counters']['node_comparisons'] <= report['counters']['max_node_comparisons']
            assert report['counters']['radial_evaluations'] <= report['counters']['max_radial_evaluations']
            assert report['counters']['duffy_evaluations'] <= report['counters']['max_duffy_evaluations']
            assert sum(report['aggregates']['attribution_counts'].values()) == 6
            assert report['aggregates']['attribution_counts']['quantitative'] == 6
            assert len(report['honest_limits']) >= 5
            assert report['aggregates']['next_discriminator']
            for pair in report['pairs']:
                i, j = pair['panel_i'], pair['panel_j']
                expected = eps * (j - i) * (pair['area_i'] + pair['area_j'])
                assert pair['attribution'] == 'quantitative'
                tol = (pair.get('skew_uncertainty_conservative') or 0.0) + 1e-9
                assert abs(pair['residual_unexplained'] - expected) <= tol
                assert pair['skew_uncertainty_conservative'] <= pair['attribution_gate']
                for direction in pair['rankine_directions'].values():
                    for entry in direction['nodes']:
                        assert entry['duffy_legacy']['label'] == 'legacy_centroid_split_duffy'
            after = {p.name: _sha(p) for p in sorted(case_dir.iterdir())}
            assert before == after
            # fresh-output-directory refusal
            rc = ppd.main(['--case-dir', str(case_dir), '--output-dir', str(out_dir)])
            assert rc == 2
        finally:
            ppd.EXPECTED_CASE_HASH = saved_hash


def test_manifest_hash_mismatch_refusal():
    saved_hash = ppd.EXPECTED_CASE_HASH
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_dir = root / 'a' / 'b' / 'c' / 'case'
        case_dir.mkdir(parents=True)
        _build_synthetic_case(case_dir)
        ppd.EXPECTED_CASE_HASH = 'test-hash-000'
        try:
            # mutate weights but keep case_hash: only the manifest can catch it
            with np.load(case_dir / 'input.npz') as data:
                arrays = {k: data[k] for k in data.files}
            arrays['weights'] = arrays['weights'] * 1.000001
            _rewrite_input(case_dir, arrays)
            rc = ppd.main(['--case-dir', str(case_dir), '--output-dir', str(root / 'o1')])
            assert rc == 2
        finally:
            ppd.EXPECTED_CASE_HASH = saved_hash


def test_manifest_shape_and_dtype_mismatch_refusal():
    saved_hash = ppd.EXPECTED_CASE_HASH
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_dir = root / 'a' / 'b' / 'c' / 'case'
        case_dir.mkdir(parents=True)
        _build_synthetic_case(case_dir)
        ppd.EXPECTED_CASE_HASH = 'test-hash-000'
        try:
            with np.load(case_dir / 'input.npz') as data:
                arrays = {k: data[k] for k in data.files}
            bad_shape = dict(arrays)
            bad_shape['points'] = arrays['points'][:, :3, :]
            _rewrite_input(case_dir, bad_shape)
            assert ppd.main(['--case-dir', str(case_dir), '--output-dir', str(root / 'o1')]) == 2
            _rewrite_input(case_dir, arrays)
            bad_dtype = dict(arrays)
            bad_dtype['weights'] = arrays['weights'].astype(np.float32)
            _rewrite_input(case_dir, bad_dtype)
            assert ppd.main(['--case-dir', str(case_dir), '--output-dir', str(root / 'o2')]) == 2
            _rewrite_input(case_dir, arrays)
            missing = {k: v for k, v in arrays.items() if k != 'owner'}
            _rewrite_input(case_dir, missing)
            assert ppd.main(['--case-dir', str(case_dir), '--output-dir', str(root / 'o3')]) == 2
        finally:
            ppd.EXPECTED_CASE_HASH = saved_hash


def test_structural_validation_refusals():
    saved_hash = ppd.EXPECTED_CASE_HASH
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_dir = root / 'a' / 'b' / 'c' / 'case'
        case_dir.mkdir(parents=True)
        _build_synthetic_case(case_dir)
        ppd.EXPECTED_CASE_HASH = 'test-hash-000'
        try:
            def run_with(mutation, out):
                with np.load(case_dir / 'input.npz') as data:
                    arrays = {k: data[k] for k in data.files}
                mutation(arrays)
                _rewrite_input(case_dir, arrays)
                case = json.loads((case_dir / 'case.json').read_text())
                case['snapshot_arrays'] = _manifest(arrays)
                (case_dir / 'case.json').write_text(json.dumps(case))
                return ppd.main(['--case-dir', str(case_dir), '--output-dir', str(root / out)])

            def bad_weights(arrays):
                arrays['weights'] = arrays['weights'].copy()
                arrays['weights'][0, 0] = -1.0
            assert run_with(bad_weights, 'o1') == 2

            def zero_area_face(arrays):
                arrays['faces'] = arrays['faces'].copy()
                arrays['faces'][3] = arrays['faces'][3][0]
            assert run_with(zero_area_face, 'o2') == 2

            def bad_face_index(arrays):
                arrays['faces'] = arrays['faces'].copy()
                arrays['faces'][3, 2] = 999
            assert run_with(bad_face_index, 'o3') == 2

            def non_unit_normal(arrays):
                arrays['normals'] = arrays['normals'] * 2.0
            assert run_with(non_unit_normal, 'o4') == 2

            def flipped_normal(arrays):
                arrays['normals'] = arrays['normals'].copy()
                arrays['normals'][0] *= -1.0
            assert run_with(flipped_normal, 'o5') == 2

            def nonfinite_vertex(arrays):
                arrays['vertices'] = arrays['vertices'].astype(float).copy()
                arrays['vertices'][0, 1] = np.nan
            assert run_with(nonfinite_vertex, 'o6') == 2
        finally:
            ppd.EXPECTED_CASE_HASH = saved_hash


def test_cli_contract_refusals():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case = root / 'tree' / 'case'
        base = ['--case-dir', str(case), '--output-dir', str(root / 'o')]
        for argv in (base + ['--max-pairs', '13'],
                     base + ['--max-pairs', '0'],
                     base + ['--levels', '16', '32'],
                     base + ['--levels', '32', '16', '64'],
                     base + ['--levels', '16', '32', '64', '64'],
                     base + ['--levels', '16', '32', '64', '128'],
                     base + ['--levels', '8', '16', '32'],
                     base + ['--levels', '16', '32', '64', '32']):
            try:
                ppd.main(argv)
                raised = False
            except SystemExit as exc:
                raised = exc.code == 2
            assert raised, argv


def test_threadpool_verification_posix_refusal():
    saved = sys.modules.get('threadpoolctl')
    sys.modules['threadpoolctl'] = None  # import raises ImportError
    try:
        try:
            ppd.verify_single_thread_runtime(True)
            raised = False
        except RuntimeError:
            raised = True
        assert raised
        info = ppd.verify_single_thread_runtime(False)
        assert info['threadpoolctl'] == 'unavailable'
    finally:
        if saved is None:
            sys.modules.pop('threadpoolctl', None)
        else:
            sys.modules['threadpoolctl'] = saved
    try:
        import threadpoolctl
    except ImportError:
        return  # honest Windows reporting already covered above
    original = threadpoolctl.threadpool_info
    threadpoolctl.threadpool_info = lambda: [{'num_threads': 2}]
    try:
        try:
            ppd.verify_single_thread_runtime(True)
            raised = False
        except RuntimeError:
            raised = True
        assert raised
        info = ppd.verify_single_thread_runtime(False)
        assert info['threadpoolctl']['max_num_threads'] == 2
    finally:
        threadpoolctl.threadpool_info = original
    threadpoolctl.threadpool_info = lambda: []
    try:
        try:
            ppd.verify_single_thread_runtime(True)
            raised = False
        except RuntimeError:
            raised = True
        assert raised
    finally:
        threadpoolctl.threadpool_info = original


def test_output_containment_refusals():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case = root / 'tree' / 'case'
        case.mkdir(parents=True)
        # inside the case tree
        assert ppd.main(['--case-dir', str(case), '--output-dir', str(case / 'sub')]) == 2
        # inside the case parent tree
        assert ppd.main(['--case-dir', str(case), '--output-dir', str(root / 'tree' / 'out')]) == 2
        # inside the grandparent tree
        assert ppd.main(['--case-dir', str(case), '--output-dir', str(root / 'out')]) == 2
        # symlink containment
        link = root / 'elsewhere' / 'link'
        link.parent.mkdir()
        try:
            os.symlink(case, link)
        except (OSError, NotImplementedError):
            return  # platform refuses directory symlinks; resolution guard
            # is still exercised by the resolved-path checks above
        assert ppd.main(['--case-dir', str(case), '--output-dir', str(link / 'o')]) == 2


def _run_all():
    failures = 0
    for name, func in sorted(globals().items()):
        if name.startswith('test_') and callable(func):
            try:
                func()
                print(f'PASS {name}')
            except AssertionError as exc:
                failures += 1
                print(f'FAIL {name}: {exc}')
    return failures


if __name__ == '__main__':
    sys.exit(1 if _run_all() else 0)


def test_gradient_control_does_not_reuse_analytic_scalar(monkeypatch):
    tri = _equilateral(0.3)
    x = tri.mean(axis=0) + np.array([0.02, 0.01, 0.15])
    original = ppd.analytic_triangle_integrals
    calls = []
    def perturbed(x, tri):
        calls.append(1)
        value, status = original(x, tri)
        return (value[0], value[1] + np.array([0.1, 0.0, 0.0])), status
    monkeypatch.setattr(ppd, 'analytic_triangle_integrals', perturbed)
    record, _ = _node_record(x, tri)
    assert len(calls) == 1
    assert record['scalar_pass'] is True
    assert record['grad_pass'] is False
    assert record['grad_controls']['method'] == 'independent_radial_angular_gradient'


def test_off_plane_projection_on_edge_is_valid():
    tri = _equilateral(0.3)
    x = 0.5 * (tri[0] + tri[1]) + np.array([0.0, 0.0, 0.08])
    value, normal, status = ppd.radial_triangle_scalar(x, tri, 64, with_normal=True)
    assert status == 'ok'
    analytic, _ = ppd.analytic_triangle_integrals(x, tri)
    assert abs(value - analytic[0]) < 1e-11
    assert abs(-normal - analytic[1][2]) < 1e-10
    record, _ = _node_record(x, tri)
    assert record['scalar_pass']
    assert record['grad_pass']


def test_radial_gradient_against_direct_volume_quadrature():
    tri = _equilateral(0.3)
    for x in (tri.mean(axis=0) + np.array([0.02, 0.01, 0.15]),
              np.array([0.4, 0.1, 0.0])):
        phi, normal, tangential, status = ppd.radial_triangle_scalar(
            x, tri, 64, with_normal=True, with_tangential=True)
        assert status == 'ok'
        n = ppd._plane_frame(tri)[0]
        gradient = (tangential - normal * n) / (4.0 * math.pi)
        direct, status = ppd.duffy_node_integrals(x, tri, (64,))
        assert status == 'ok'
        assert np.linalg.norm(gradient - direct[64]['grad']) < 1e-11
