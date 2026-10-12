# SPDX-License-Identifier: MPL-2.0
"""Far-field damping and the far-field weak-mode derived-operator policy."""
from __future__ import annotations

import math

import numpy as np
import pytest

from anyhydro import operators as ops
from anyhydro.green_function import far_field_damping, far_field_theta_count, solve_wavenumber
from anyhydro.schema import SchemaError, validate_hydro_dataset

from test_derived_operators import RETAINED4, make_settings, make_norm
from test_radiation import make_oracle_dataset

LAM_NEAR = np.array([-1.0e-4, 0.3, 30.0, 200.0])
LAM_FAR = np.array([2.0e-4, 0.31, 31.0, 205.0])


def _rotation(seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(4, 4)))
    return q


def _dataset_with_far_field(near=LAM_NEAR, far=LAM_FAR, seed=3):
    ds = make_oracle_dataset().copy(deep=True)
    labels = [str(v) for v in ds["influenced_dof"].values.tolist()]
    idx = [labels.index(k) for k in RETAINED4]
    B = ds["radiation_damping"].values.copy()
    F = B.copy()
    for o in range(B.shape[0]):
        V = _rotation(seed + o)
        B[o][np.ix_(idx, idx)] = V @ np.diag(near) @ V.T
        F[o][np.ix_(idx, idx)] = V @ np.diag(far) @ V.T
    ds["radiation_damping"].values[...] = B
    ds["radiation_damping_far_field"] = (("omega", "influenced_dof", "radiating_dof"), F)
    ds["radiation_damping_far_field"].attrs["source"] = "test far-field damping"
    return validate_hydro_dataset(ds), idx


def far_settings(**kw):
    return make_settings(policy=ops.CORRECTION_FAR_FIELD_WEAK_MODES, **kw)


def test_far_field_damping_is_psd_and_uses_the_analytic_constant():
    rng = np.random.default_rng(1)
    H = rng.normal(size=(2, 3, 64)) + 1j * rng.normal(size=(2, 3, 64))
    omegas = np.array([0.8, 1.5])
    ks = np.array([solve_wavenumber(w, 9.81, 30.0) for w in omegas])
    B = far_field_damping(H, omegas, ks, 30.0, 1025.0)
    for o in range(2):
        assert np.allclose(B[o], B[o].T, rtol=0, atol=0)
        assert np.linalg.eigvalsh(B[o])[0] >= -1e-12 * np.abs(B[o]).max()
        k, w = ks[o], omegas[o]
        D = math.tanh(k * 30.0) + k * 30.0 / math.cosh(k * 30.0) ** 2
        expected = 4 * math.pi * 1025.0 * k / (w * D) * np.mean(np.abs(H[o, 1]) ** 2) * 2 * math.pi
        assert B[o][1, 1] == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError):
        far_field_damping(H, omegas[:1], ks[:1], 30.0, 1025.0)


def test_wavenumber_and_theta_count():
    for h in (5.0, 40.0, math.inf):
        k = solve_wavenumber(1.3, 9.81, h)
        lhs = k * (1.0 if math.isinf(h) else math.tanh(k * h))
        assert lhs == pytest.approx(1.3 ** 2 / 9.81, rel=1e-12)
    assert far_field_theta_count([0.5], 150.0, 9.81, 2.0) == 720
    k8 = solve_wavenumber(8.0, 9.81, 150.0)
    assert far_field_theta_count([8.0], 150.0, 9.81, 100.0) >= 4 * k8 * 100.0


def test_schema_requires_far_field_provenance():
    ds, _ = _dataset_with_far_field()
    ds["radiation_damping_far_field"].attrs.pop("source")
    with pytest.raises(SchemaError):
        validate_hydro_dataset(ds)


def test_policy_replaces_only_weak_modes():
    ds, _ = _dataset_with_far_field()
    art = ops.derived_operators_from_dataset(ds, far_settings())
    for o in range(art.omegas.size):
        lam = np.linalg.eigvalsh(art.derived_damping[o])
        np.testing.assert_allclose(lam, [LAM_FAR[0], *LAM_NEAR[1:]], rtol=1e-9, atol=1e-13)
    assert art.report.damping_psd_pass
    np.testing.assert_array_equal(art.derived_added_mass,
                                  0.5 * (art.raw_added_mass + np.transpose(art.raw_added_mass, (0, 2, 1))))
    # without the policy the same source is not PSD
    plain = ops.derived_operators_from_dataset(ds, make_settings())
    assert not plain.report.damping_psd_pass


def _reference_derivation(Bsym, F, scales):
    """Independent implementation for one frequency (normalized coordinates)."""
    S = np.diag(scales)
    Bh, Fh = S @ Bsym @ S, S @ F @ S
    lam, V = np.linalg.eigh(Bh)
    weak = lam < ops.WEAK_MODE_FRACTION * lam[-1]
    P = V[:, weak] @ V[:, weak].T
    Q = np.eye(len(lam)) - P
    new = Q @ Bh @ Q + P @ Fh @ P
    Si = np.diag(1.0 / np.asarray(scales))
    return Si @ new @ Si, int(weak.sum())


def test_weak_subspace_uses_normalized_coordinates_and_far_compression():
    scales = [1.0, 10.0, 0.5, 2.0]
    rng = np.random.default_rng(11)
    # normalized near damping with two EQUAL weak eigenvalues; far field does
    # not commute with it
    V = _rotation(5)
    lam_hat = np.array([-3.0e-3, -3.0e-3, 40.0, 300.0])
    Sinv = np.diag(1.0 / np.array(scales))
    Bsi = Sinv @ (V @ np.diag(lam_hat) @ V.T) @ Sinv
    G = rng.normal(size=(4, 4))
    Fsi = Sinv @ (G @ G.T + V @ np.diag([0.0, 0.0, 40.0, 300.0]) @ V.T) @ Sinv
    ds, idx = _dataset_with_far_field()
    B = ds["radiation_damping"].values
    F = ds["radiation_damping_far_field"].values
    for o in range(B.shape[0]):
        B[o][np.ix_(idx, idx)] = Bsi
        F[o][np.ix_(idx, idx)] = Fsi
    art = ops.derived_operators_from_dataset(
        ds, far_settings(normalization=make_norm(4, scales=scales)))
    expected, n_weak = _reference_derivation(Bsi, Fsi, scales)
    assert n_weak == 2
    np.testing.assert_allclose(art.derived_damping[0], expected, rtol=1e-9, atol=1e-12)
    assert np.linalg.eigvalsh(art.derived_damping[0])[0] > -1e-12
    # the SI-coordinate weak set is different here, so the coordinates matter
    lam_si = np.linalg.eigvalsh(Bsi)
    assert int((lam_si < ops.WEAK_MODE_FRACTION * lam_si[-1]).sum()) != n_weak or \
        not np.allclose(_reference_derivation(Bsi, Fsi, [1.0] * 4)[0], expected)


def test_missing_far_field_is_refused():
    ds = make_oracle_dataset()
    with pytest.raises(ops.DerivedOperatorError, match='radiation_damping_far_field'):
        ops.derived_operators_from_dataset(ds, far_settings())


def test_far_field_array_is_refused_under_other_policies():
    ds, _ = _dataset_with_far_field()
    art = ops.derived_operators_from_dataset(ds, far_settings())
    import dataclasses
    with pytest.raises(ops.DerivedOperatorError):
        dataclasses.replace(art, settings=make_settings(),
                            derived_damping=art.raw_damping,
                            derived_added_mass=art.derived_added_mass)
    with pytest.raises(ops.DerivedOperatorError):
        dataclasses.replace(art, raw_damping_far_field=None)
    with pytest.raises(ops.DerivedOperatorError):
        dataclasses.replace(art, raw_damping_far_field=art.raw_damping_far_field * 1.01)


def test_save_load_and_raw_source_proof(tmp_path):
    ds, _ = _dataset_with_far_field()
    art = ops.derived_operators_from_dataset(ds, far_settings())
    path = tmp_path / 'far.npz'
    ops.save_derived_operators(art, path)
    back = ops.load_derived_operators(path, source_dataset=ds)
    assert back.content_hash() == art.content_hash()
    np.testing.assert_array_equal(back.raw_damping_far_field, art.raw_damping_far_field)
    # a different far-field source is detected by the raw-source proof
    ds2 = ds.copy(deep=True)
    ds2["radiation_damping_far_field"].values[...] *= 1.001
    assert not back.matches_raw_source(ds2)
    # artifacts of the old policies carry no far-field array (unchanged format)
    plain = ops.derived_operators_from_dataset(ds, make_settings())
    ops.save_derived_operators(plain, tmp_path / 'plain.npz')
    with np.load(tmp_path / 'plain.npz') as z:
        assert 'raw_damping_far_field' not in z.files
    # removing the array from a far-field artifact is refused
    with np.load(path) as z:
        data = {k: z[k] for k in z.files if k != 'raw_damping_far_field'}
    np.savez(tmp_path / 'stripped.npz', **data)
    with pytest.raises(ops.DerivedOperatorSerializationError):
        ops.load_derived_operators(tmp_path / 'stripped.npz')


def test_capytaine_backend_far_field_output():
    pytest.importorskip('capytaine')
    from anyhydro.capytaine_backend import CapytaineBackend, CapytaineExecutionLimits
    from anyhydro.first_slice import make_first_slice_bodies
    from anyhydro.records import HydroRequest
    bodies = make_first_slice_bodies(subdivisions=2)
    req = HydroRequest(omegas=np.array([1.0, 3.0]), wave_directions=np.array([0.0]),
                       water_depth=150.0, rho=1025.0, gravity=9.81,
                       include_zero_frequency=False, include_infinite_frequency=False,
                       backend_settings={})
    lim = CapytaineExecutionLimits(max_total_panels=2000, max_finite_omegas=2,
                                   max_wave_directions=1, n_jobs=1, n_threads=1)
    ds = CapytaineBackend(execution_limits=lim).solve(bodies, req)
    Bf = ds['radiation_damping_far_field'].values
    Bn = ds['radiation_damping'].values
    assert 'Kochin' in ds['radiation_damping_far_field'].attrs['source']
    for o in range(2):
        assert np.linalg.eigvalsh(Bf[o])[0] >= -1e-9 * np.abs(Bf[o]).max()
        lam, V = np.linalg.eigh(0.5 * (Bn[o] + Bn[o].T))
        top = V[:, -1]
        assert top @ Bf[o] @ top == pytest.approx(lam[-1], rel=0.1)   # coarse mesh
