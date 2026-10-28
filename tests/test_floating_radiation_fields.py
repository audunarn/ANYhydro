"""Retained floating family: records, mask/mutation refusals (no native work) and a gated native smoke."""
import os

import numpy as np
import pytest

from anyhydro import floating_radiation_fields as ff
from anyhydro.wave_surface_fields import WaveFieldError, WaveFieldMissing


def box_quads(L=4.0, B=2.0, T=1.0):
    v = np.array([[x, y, z] for z in (-T, 0.0) for y in (-B / 2, B / 2) for x in (-L / 2, L / 2)], float)
    # indices: z-major; bottom 0..3, top 4..7; panels with outward normals (verified by prepare_body volume check natively)
    f = np.array([[0, 2, 3, 1], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]])
    return v, f


def make_snapshot(scale=1.0):
    v, f = box_quads()
    ids = tuple(f"p{i}" for i in range(len(f)))
    Q0 = np.eye(3); cg = (0.0, 0.0, -0.4)
    req = ff.family_request(v, f, ids, heading=0.3, water_depth=50.0, rho=1025.0, gravity=9.81, Q0=Q0, cg_world=cg)
    n = len(f)
    rng = np.random.default_rng(2)
    rad = rng.normal(size=(6, n)) + 1j * rng.normal(size=(6, n))
    dif = rng.normal(size=n) + 0j
    A = np.diag(np.arange(1.0, 7.0)) * scale
    return ff.FloatingFamilySnapshot(req, 0.8, v, f, ids, rad, dif, A, 0.1 * A, np.arange(6) + 1j, {"backend": "capytaine"},
                                     panel_pressure_radiation=rad[:, :n] * 2, panel_pressure_incident_diffraction=dif + 1j,
                                     centers=np.ones((n, 3)), normals=np.tile([0.0, 0.0, 1.0], (n, 1)), areas=np.ones(n))


def test_snapshot_identity_is_content_bound_and_immutable():
    s = make_snapshot()
    s.validate()
    assert not s.radiation_sources.flags.writeable
    assert make_snapshot(2.0).snapshot_id != s.snapshot_id
    with pytest.raises(WaveFieldError):
        ff.FloatingFamilySnapshot(s.request, s.omega, s.vertices, s.faces, s.surface_ids, s.radiation_sources,
                                  s.diffraction_sources, s.added_mass, s.damping, s.excitation, s.provider,
                                  snapshot_id="0" * 64)


def test_cache_roundtrip_never_solves_and_missing_is_explicit(tmp_path):
    s = make_snapshot()
    assert ff.missing_family_frequencies(tmp_path, s.request, [s.omega]) == (s.omega,)
    with pytest.raises(WaveFieldMissing):
        ff.recover_family_snapshot(tmp_path, s.request, s.omega)
    ff.save_family_snapshot(tmp_path, s)
    back = ff.recover_family_snapshot(tmp_path, s.request, s.omega)
    assert back.snapshot_id == s.snapshot_id
    np.testing.assert_array_equal(back.radiation_sources, s.radiation_sources)
    assert ff.missing_family_frequencies(tmp_path, s.request, [s.omega]) == ()


def test_payload_roundtrip_is_self_contained_and_identity_checked():
    s = make_snapshot()
    meta, arrays = ff.snapshot_to_payload(s)
    back = ff.snapshot_from_payload(meta, {k: np.array(v) for k, v in arrays.items()})
    assert back.snapshot_id == s.snapshot_id
    tampered = {k: np.array(v) for k, v in arrays.items()}
    tampered['A'] = tampered['A'] * 2
    with pytest.raises(WaveFieldError):
        ff.snapshot_from_payload(meta, tampered)


def test_radiation_basis_is_heading_independent_and_missing_rhs_count_is_actual(tmp_path):
    s = make_snapshot()
    ff.save_family_snapshot(tmp_path, s)
    other = dict(s.request, heading=(s.request['heading'] + 0.7) % (2 * np.pi))
    assert ff.radiation_key(other) == ff.radiation_key(s.request)
    assert ff.missing_radiation_rhs(tmp_path, s.request, s.omega) == 0           # cached exactly
    assert ff.missing_radiation_rhs(tmp_path, other, s.omega) == 1               # only the new heading's diffraction RHS
    assert ff.missing_radiation_rhs(tmp_path, other, s.omega * 1.1) == 7         # other frequency: full basis
    donor = ff.recover_radiation_donor(tmp_path, other, s.omega)
    assert donor is not None and donor.snapshot_id == s.snapshot_id
    moved = dict(s.request, Q0=np.diag([1.0, -1.0, -1.0]).tolist())              # different axes: not the same basis
    assert ff.radiation_key(moved) != ff.radiation_key(s.request)


def test_elevation_recovery_preserves_arbitrary_evaluator_calls_and_results():
    s = make_snapshot()
    pts = np.array([[0.0, 1.0, -0.5], [0.0, -1.0, -0.2], [1.0, 0.0, 0.0]])
    nrm = np.array([[0.0, 1, 0], [0, -1, 0], [1, 0, 0]])
    full_calls, only_calls = [], []

    def make(log):
        def ev(snap, q, k):
            log.append((None if k is None else int(k), np.array(q)))
            base = 0.3 if k is None else 0.1 * (k + 1)
            return (base + 0.01 * q[:, 0] - 0.02 * q[:, 2]) * (1.0 + 0.5j)
        return ev
    full = ff.evaluate_floating_fields(s, pts, nrm, evaluator=make(full_calls), output_mask=[True, False, True])
    only = ff.evaluate_floating_elevations(s, pts, nrm, evaluator=make(only_calls), output_mask=[True, False, True])
    np.testing.assert_array_equal(only.eta_incident_diffraction, full.eta_incident_diffraction)     # bit identical
    np.testing.assert_array_equal(only.eta_radiation, full.eta_radiation)
    assert len(full_calls) == len(only_calls) == 14
    assert [k for k, _ in only_calls] == [k for k, _ in full_calls]
    for (_, a), (_, b) in zip(only_calls, full_calls):
        np.testing.assert_array_equal(a, b)
    assert only.snapshot_id == s.snapshot_id


def test_elevation_recovery_preserves_stateful_evaluator_order_and_values():
    s = make_snapshot()
    points = np.array([[0.0, 1.0, -0.5], [0.0, -1.0, -0.2]])
    normals = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    traces = [[], []]

    def callback(trace):
        def evaluate(snap, query, radiation_index):
            trace.append((radiation_index, np.array(query)))
            return np.full(len(query), len(trace) * (1.0 + 0.5j))
        return evaluate

    full = ff.evaluate_floating_fields(s, points, normals, evaluator=callback(traces[0]))
    only = ff.evaluate_floating_elevations(s, points, normals, evaluator=callback(traces[1]))
    np.testing.assert_array_equal(only.eta_incident_diffraction, full.eta_incident_diffraction)
    np.testing.assert_array_equal(only.eta_radiation, full.eta_radiation)
    assert len(traces[0]) == len(traces[1]) == 14
    for (old_index, old_query), (new_index, new_query) in zip(*traces):
        assert old_index == new_index
        np.testing.assert_array_equal(old_query, new_query)


def test_changed_basis_changes_request():
    v, f = box_quads()
    ids = tuple(str(i) for i in range(len(f)))
    a = ff.family_request(v, f, ids, heading=0, water_depth=50, rho=1025, gravity=9.81, Q0=np.eye(3), cg_world=(0, 0, 0))
    b = ff.family_request(v, f, ids, heading=0, water_depth=50, rho=1025, gravity=9.81, Q0=np.eye(3), cg_world=(0, 0, 0.1))
    assert a != b
    with pytest.raises(WaveFieldError):
        ff.family_request(v, f, ids, heading=0, water_depth=50, rho=1025, gravity=9.81, Q0=np.diag([1, 1, -1.0]), cg_world=(0, 0, 0))


def test_output_mask_zeroes_outputs_but_keeps_sources_and_uses_family_evaluator():
    s = make_snapshot()
    pts = np.array([[0.0, 1.0, -0.5], [0.0, -1.0, -0.5], [1.0, 0.0, -1.0]])
    nrm = np.array([[0.0, 1, 0], [0, -1, 0], [0, 0, -1]])
    calls = []
    def ev(snap, q, k):
        calls.append((len(q), k)); return np.full(len(q), 1.0 + 0j if k is None else (k + 1) * 1j)
    out = ff.evaluate_floating_fields(s, pts, nrm, output_mask=[True, False, True], evaluator=ev)
    assert out.incident_diffraction_pressure[1] == 0 and np.all(out.radiation_pressure[:, 1] == 0)
    w, rho = s.omega, 1025.0
    np.testing.assert_allclose(out.incident_diffraction_pressure[[0, 2]], 1j * w * rho)
    # per-unit-displacement radiation pressure is i w rho phi with NO additional -i w
    np.testing.assert_allclose(out.radiation_pressure[2, 0], 1j * w * rho * (3j))
    assert s.radiation_sources.shape[1] == len(s.faces)           # sources untouched by output masking


@pytest.mark.skipif(os.environ.get(ff.NATIVE_GATE) != "1", reason=f"native BEM gated by {ff.NATIVE_GATE}=1")
def test_native_family_smoke_coefficients_and_pressure_integrals():
    """Parent-run only: one 5-panel, 7-solve frequency; A/B/Fexc and recovered pressure from the same retained state."""
    v, f = box_quads()
    ids = [str(i) for i in range(len(f))]
    c, s_ = np.cos(0.05), np.sin(0.05)
    Q0 = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])           # body axes differ from world axes
    snap = ff.solve_family_snapshot(v, f, ids, 1.0, heading=0.4, water_depth=60.0, Q0=Q0, cg_world=(0, 0, -0.4))
    snap.validate()
    assert snap.radiation_sources.shape == (6, len(f)) and np.all(np.isfinite(snap.added_mass))
    assert np.all(np.diag(snap.damping) >= -1e-6 * np.abs(snap.damping).max())
    # reciprocity of the native operator (raw, unsymmetrised): modest asymmetry only on this coarse mesh
    scale = np.abs(snap.added_mass).max()
    assert np.abs(snap.added_mass - snap.added_mass.T).max() < 0.05 * scale
    pts = np.array([[0.0, 1.0, -0.5]])
    fields = ff.evaluate_floating_fields(snap, pts, np.array([[0.0, 1.0, 0.0]]))
    assert fields.radiation_pressure.shape == (6, 1) and np.all(np.isfinite(fields.radiation_pressure))
