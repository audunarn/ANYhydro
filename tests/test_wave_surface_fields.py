"""Field-capable retained snapshots: records, cache refusals and same-solution eta/pressure conventions.

Stub-evaluator tests need no Capytaine.  ``test_native_*`` solves a tiny box with Capytaine and is skipped unless
ANYHYDRO_NATIVE_FIELD=1 (so the parent can schedule it explicitly).
"""
import math
import os

import numpy as np
import pytest

from anyhydro.wave_surface_fields import (FieldSnapshot, WaveFieldError, evaluate_surface_fields, field_request,
    missing_field_frequencies, prepare_field_snapshots, recover_field_snapshot, save_field_snapshot, snapshot_from_dict,
    snapshot_to_dict)

RHO, G = 1025., 9.81
V = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], float)
F = np.array([[0, 3, 2, 1], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
LABELS = ('bottom', 'y-', 'x+', 'y+', 'x-')


def snapshot(omega=1., heading=0.):
    request = field_request(V, F, LABELS, heading=heading, water_depth=50., rho=RHO, gravity=G)
    return FieldSnapshot(request, omega, V, F, LABELS, np.arange(5) * (1 + 0.5j), dict(backend='stub'))


def airy(snap, points):
    """Unit deep-water incident potential; exp(-i w t), heading 0."""
    w = snap.omega; k = w * w / G
    return -(1j * G / w) * np.exp(k * points[:, 2]) * np.exp(1j * k * points[:, 0])


def test_snapshot_identity_roundtrip_and_tamper_refusal(tmp_path):
    s = snapshot()
    save_field_snapshot(tmp_path, s)
    again = recover_field_snapshot(tmp_path, s.request, s.omega)
    assert again.snapshot_id == s.snapshot_id and np.array_equal(again.sources, s.sources)
    assert snapshot_from_dict(snapshot_to_dict(s)).snapshot_id == s.snapshot_id
    bad = snapshot_to_dict(s); bad['sources_real'][0] += 1.
    with pytest.raises(WaveFieldError):
        snapshot_from_dict(bad)


def test_missing_corrupt_and_unsupported_refuse_with_actionable_error_and_never_solve(tmp_path, monkeypatch):
    s = snapshot()
    assert missing_field_frequencies(tmp_path, s.request, [s.omega, 2.]) == (s.omega, 2.)
    with pytest.raises(WaveFieldError, match='prepare'):
        recover_field_snapshot(tmp_path, s.request, s.omega)
    path = save_field_snapshot(tmp_path, s)
    path.write_bytes(b'garbage')
    with pytest.raises(WaveFieldError, match='corrupt|unsupported'):         # corrupt is NOT "missing"
        missing_field_frequencies(tmp_path, s.request, [s.omega])
    with pytest.raises(WaveFieldError, match='corrupt|unsupported'):
        recover_field_snapshot(tmp_path, s.request, s.omega)

    def solve(*args, **kwargs): raise AssertionError('solved')
    monkeypatch.setattr('anyhydro.wave_surface_fields.solve_field_snapshot', solve)
    with pytest.raises(WaveFieldError, match='corrupt|unsupported'):          # never silently overwritten
        prepare_field_snapshots(V, F, LABELS, [s.omega], cache_directory=tmp_path, heading=0., water_depth=50.)
    path.unlink()
    with pytest.raises(AssertionError, match='solved'):                       # truly absent: explicit preparation solves
        prepare_field_snapshots(V, F, LABELS, [s.omega], cache_directory=tmp_path, heading=0., water_depth=50.)
    save_field_snapshot(tmp_path, s)
    assert prepare_field_snapshots(V, F, LABELS, [s.omega], cache_directory=tmp_path, heading=0., water_depth=50.)[0].snapshot_id == s.snapshot_id


def test_retained_decomposition_is_hash_bound_persisted_and_metadata_mutation_is_refused(tmp_path):
    base = snapshot()
    prony = np.array([[0.5, 1.5, 2.5], [1., -.5, .25]])
    s = FieldSnapshot(base.request, base.omega, V, F, LABELS, base.sources, base.provider, '', prony, 2.5, 'python',
                      {'tabulation_nr': 328})
    assert s.snapshot_id != base.snapshot_id
    save_field_snapshot(tmp_path, s)
    again = recover_field_snapshot(tmp_path, s.request, s.omega)
    assert np.array_equal(again.prony, prony) and again.prony_kh == 2.5 and again.snapshot_id == s.snapshot_id
    assert snapshot_from_dict(snapshot_to_dict(s)).snapshot_id == s.snapshot_id
    again.provider['version'] = 'tampered'                                    # public dict mutation is detected at use
    with pytest.raises(WaveFieldError, match='changed'):
        evaluate_surface_fields(again, np.zeros((1, 3)), np.array([[1., 0, 0]]), evaluator=airy)
    with pytest.raises(WaveFieldError):
        FieldSnapshot(base.request, base.omega, V, F, LABELS, base.sources, base.provider, '', prony * np.nan, 2.5, 'python', {})


def test_eta_and_pressure_come_from_one_potential_with_pd0_equal_rho_g_eta():
    s = snapshot(omega=1.3)
    points = np.array([[1., 0., -0.5], [0., 1., 0.3], [0., -1., -1.0]])
    normals = np.array([[1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
    seen = []

    def evaluator(snap, p):
        seen.append(np.array(p)); return airy(snap, p)
    fields = evaluate_surface_fields(s, points, normals, evaluator=evaluator, offset=0.01)
    k = s.omega ** 2 / G
    assert np.allclose(fields.pd0, RHO * G * fields.eta, rtol=1e-12)
    assert np.allclose(fields.eta, np.exp(1j * k * (points[:, 0] + .01 * normals[:, 0])), rtol=1e-12)   # unit amplitude
    below = np.exp(k * np.minimum(points[:, 2], 0.)) * np.exp(1j * k * (points[:, 0] + .01 * normals[:, 0]))
    assert np.allclose(fields.pd, RHO * G * below, rtol=1e-3)
    # waterline evaluation is horizontal-only fluid-side offset at z = 0; submerged evaluation is clamped at z <= 0
    assert np.allclose(seen[0][:, 2], 0.) and np.all(seen[1][:, 2] <= 0.)
    assert np.allclose(seen[0][:, :2], points[:, :2] + .01 * normals[:, :2])


def test_waterline_mask_reports_exact_zero_for_excluded_points_and_never_evaluates_them():
    s = snapshot()
    points = np.array([[0., 0., -1.0], [1., 0., 0.2]])
    normals = np.array([[0., 0., -1.], [1., 0., 0.]])
    called = []

    def evaluator(snap, p):
        called.append(len(p)); return airy(snap, p)
    fields = evaluate_surface_fields(s, points, normals, evaluator=evaluator, waterline=[False, True])
    assert fields.eta[0] == 0 and fields.pd0[0] == 0 and fields.pd[0] != 0 and called == [1, 2]


def test_invalid_inputs_are_refused():
    s = snapshot()
    with pytest.raises(WaveFieldError):
        evaluate_surface_fields(s, np.zeros((1, 3)), np.zeros((1, 3)), evaluator=airy)          # zero fluid normal
    with pytest.raises(WaveFieldError):
        evaluate_surface_fields(s, np.array([[np.nan, 0, 0]]), np.array([[1., 0, 0]]), evaluator=airy)


@pytest.mark.skipif(os.environ.get('ANYHYDRO_NATIVE_FIELD') != '1', reason='native Capytaine field capture is scheduled by the parent')
def test_native_small_box_field_reconstruction_matches_panel_potential_and_never_solves_on_recovery(tmp_path):
    import capytaine as cpt
    from capytaine.bem.airy_waves import airy_waves_potential
    from anyhydro._pressure_body import prepare_body
    from anyhydro.wave_surface_fields import solve_field_snapshot
    w = 1.2
    snap = solve_field_snapshot(V, F, LABELS, w, heading=0., water_depth=50., rho=RHO, gravity=G)
    save_field_snapshot(tmp_path, snap)
    again = recover_field_snapshot(tmp_path, snap.request, w)
    assert again.snapshot_id == snap.snapshot_id and abs(again.sources).max() > 0
    assert again.prony is not None and again.prony.shape[0] == 2           # realised Green-function decomposition retained
    first = evaluate_surface_fields(again, np.array([[1.0, 0.0, -0.5]]), np.array([[1., 0., 0.]]), offset=1e-3)
    second = evaluate_surface_fields(recover_field_snapshot(tmp_path, snap.request, w), np.array([[1.0, 0.0, -0.5]]),
                                     np.array([[1., 0., 0.]]), offset=1e-3)
    assert np.array_equal(first.pd, second.pd)                              # replay is deterministic across instances
    v, f, labels, mesh, body = prepare_body(cpt, V, F, LABELS, period=math.tau / w, heading=0., water_depth=50., rho=RHO, max_panels=600)
    problem = cpt.DiffractionProblem(body=body, omega=w, wave_direction=0., water_depth=50., rho=RHO, g=G)
    result = cpt.BEMSolver().solve(problem, method='indirect', keep_details=True, n_threads=1)
    centers = np.asarray(mesh.faces_centers); normals = np.asarray(mesh.faces_normals)
    reference = np.asarray(result.potential) + np.asarray(airy_waves_potential(centers, problem))
    wet = centers[:, 2] < -.2
    fields = evaluate_surface_fields(again, centers[wet], normals[wet], offset=1e-3)
    recon = fields.pd / (1j * w * RHO)
    assert np.allclose(recon, reference[wet], rtol=5e-2, atol=5e-2 * np.abs(reference).max())
    top = evaluate_surface_fields(again, np.array([[1.0, 0.0, 0.0]]), np.array([[1., 0., 0.]]), offset=1e-3)
    assert np.allclose(top.pd0, RHO * G * top.eta, rtol=1e-10)
