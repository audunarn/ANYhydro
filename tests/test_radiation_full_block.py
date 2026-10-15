# SPDX-License-Identifier: MPL-2.0
"""W5 K1-K3 reconstruction: band-consistent A_inf, body-diagonal tail, full-block criteria."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from anyhydro import operators as ops
from anyhydro import radiation as rad

from test_derived_operators import make_settings
from test_radiation import _oracle_matrices, make_oracle_dataset

OMEGAS = np.linspace(0.05, 3.0, 120)
N_TIME = 4001


def kernel_settings(**extra):
    C, lam, _ = _oracle_matrices(12)
    base = dict(dt=0.01, n_time=N_TIME, symmetry_tol=1e-9, passivity_tol=1e-9,
                low_tail_policy="explicit_zero_sample", zero_frequency_damping=C / lam,
                quadrature_method="piecewise_linear_exact",
                execution_limits=rad.RadiationExecutionLimits(max_time_samples=N_TIME))
    base.update(extra)
    return rad.RadiationSettings(**base)


BAND = dict(a_inf_policy="band_consistent", a_inf_calibration_min=0.1, a_inf_calibration_max=0.4)
FULL_BLOCK = dict(form="full_block", omega_min=0.5, omega_max=2.0, block_rel_tol=0.02,
                  modal_rel_tol=0.02, energetic_fraction=1e-3, cluster_fraction=0.01)


def test_band_consistent_a_inf_removes_truncation_offset_and_is_recorded():
    ds = make_oracle_dataset(omegas=OMEGAS)
    plain = rad.radiation_kernel_from_dataset(ds, kernel_settings())
    band = rad.radiation_kernel_from_dataset(ds, kernel_settings(**BAND))
    om = ds["omega"].values
    s = (om >= 0.5) & (om <= 2.0)
    A = ds["added_mass"].values[s]
    e_plain = np.abs(plain.reconstruct(om[s])[0] - A).max()
    e_band = np.abs(band.reconstruct(om[s])[0] - A).max()
    assert e_band < 0.5 * e_plain
    prov = band.provenance
    np.testing.assert_array_equal(np.asarray(prov["a_inf_computed"]), plain.added_mass_infinite)
    np.testing.assert_allclose(band.added_mass_infinite - plain.added_mass_infinite,
                               np.asarray(prov["a_inf_band_consistent_delta"]), rtol=0, atol=1e-12)
    assert any("band_consistent" in a for a in band.assumptions)
    # the kernel samples themselves are unchanged
    np.testing.assert_array_equal(band.kernel, plain.kernel)


def test_band_consistent_verifier_proves_and_refuses():
    ds = make_oracle_dataset(omegas=OMEGAS)
    k = rad.radiation_kernel_from_dataset(ds, kernel_settings(**BAND))
    a_inf = ds["added_mass_infinite"].values
    out = rad.verify_band_consistent_added_mass_infinite(k, a_inf, ds["omega"].values,
                                                        ds["added_mass"].values)
    np.testing.assert_array_equal(out, k.added_mass_infinite)
    with pytest.raises(rad.RadiationError, match='computed A_inf'):
        rad.verify_band_consistent_added_mass_infinite(k, a_inf * 1.001, ds["omega"].values,
                                                      ds["added_mass"].values)
    shifted = k.added_mass_infinite + 1.0
    forged = dataclasses.replace(k, added_mass_infinite=shifted, provenance={
        **dict(k.provenance), "a_inf_band_consistent_delta":
        (shifted - np.asarray(k.provenance["a_inf_computed"])).tolist()})
    with pytest.raises(rad.RadiationError, match='not band-consistent'):
        rad.verify_band_consistent_added_mass_infinite(forged, a_inf, ds["omega"].values,
                                                      ds["added_mass"].values)
    plain = rad.radiation_kernel_from_dataset(ds, kernel_settings())
    with pytest.raises(rad.RadiationError, match='does not declare'):
        rad.verify_band_consistent_added_mass_infinite(plain, a_inf, ds["omega"].values,
                                                      ds["added_mass"].values)


def test_band_consistent_settings_validation_and_serialization():
    st = kernel_settings(**BAND)
    back = rad.RadiationSettings.from_dict(st.to_dict())
    assert (back.a_inf_policy, back.a_inf_calibration_min, back.a_inf_calibration_max) == \
        ("band_consistent", 0.1, 0.4)
    assert "a_inf_policy" not in kernel_settings().to_dict()
    for bad in (dict(a_inf_calibration_min=0.1), dict(a_inf_policy="band_consistent"),
                dict(a_inf_policy="band_consistent", a_inf_calibration_min=0.5,
                     a_inf_calibration_max=0.4),
                dict(high_tail_scope="body_diagonal")):
        with pytest.raises(rad.RadiationSettingsError):
            kernel_settings(**bad)
    ds = make_oracle_dataset(omegas=OMEGAS)
    with pytest.raises(rad.RadiationSettingsError, match='at least two'):
        rad.radiation_kernel_from_dataset(ds, kernel_settings(
            a_inf_policy="band_consistent", a_inf_calibration_min=0.06, a_inf_calibration_max=0.07))


def test_body_diagonal_tail_leaves_inter_body_coupling_truncated():
    ds = make_oracle_dataset(omegas=OMEGAS)
    tail = dict(high_tail_policy="power_law_extension", high_tail_exponent=2.0,
                high_tail_upper=300.0, high_tail_name="test")
    full = rad.radiation_kernel_from_dataset(ds, kernel_settings(**tail))
    diag = rad.radiation_kernel_from_dataset(ds, kernel_settings(**tail, high_tail_scope="body_diagonal"))
    trunc = rad.radiation_kernel_from_dataset(ds, kernel_settings())
    owner = [k.split("::")[0] for k in full.dof_keys]
    same = np.array([[a == b for b in owner] for a in owner])
    # inside each body the tail is applied; across bodies only the linear ramp
    # to zero over the first tail interval (ratio 1.01) remains
    b_m = np.abs(ds["radiation_damping"].values[-1][~same]).max()
    ramp = (2 / np.pi) * b_m * 0.5 * (rad.HIGH_TAIL_RATIO - 1.0) * OMEGAS[-1]
    assert np.abs(diag.kernel[:, ~same] - trunc.kernel[:, ~same]).max() <= 1.01 * ramp
    assert np.abs(full.kernel[:, ~same] - trunc.kernel[:, ~same]).max() > 10 * ramp
    np.testing.assert_array_equal(diag.kernel[:, same], full.kernel[:, same])
    assert diag.settings.to_dict()["high_tail_scope"] == "body_diagonal"


def test_criteria_factory_forms():
    fb = rad.reconstruction_criteria_from_dict(FULL_BLOCK)
    assert isinstance(fb, rad.FullBlockReconstructionCriteria)
    assert fb.to_dict() == dict(FULL_BLOCK, modal_reference="spectral_radius")
    legacy = rad.reconstruction_criteria_from_dict(dict(
        omega_min=1.0, omega_max=2.0, added_mass_abs_tol=0.1, added_mass_rel_tol=0.01,
        damping_abs_tol=0.1, damping_rel_tol=0.01))
    assert isinstance(legacy, rad.ReconstructionCriteria)
    for bad in (dict(FULL_BLOCK, form="other"), dict(FULL_BLOCK, block_rel_tol=0.0),
                dict(FULL_BLOCK, energetic_fraction=1.5), {k: v for k, v in FULL_BLOCK.items() if k != "modal_rel_tol"}):
        with pytest.raises(rad.ReconstructionCriteriaError):
            rad.reconstruction_criteria_from_dict(bad)


def test_full_block_report_on_derived_kernel():
    ds = make_oracle_dataset(omegas=OMEGAS)
    der = ops.derived_operators_from_dataset(ds, make_settings())
    C, lam, _ = _oracle_matrices(12)
    P = der.projection
    st = kernel_settings(zero_frequency_damping=P.T @ (C / lam) @ P, **BAND)
    k = rad.radiation_kernel_from_derived_operators(der, st)
    crit = rad.reconstruction_criteria_from_dict(FULL_BLOCK)
    rep = rad.compare_derived_reconstruction(k, der, crit, source_dataset=ds)
    assert isinstance(rep, rad.FullBlockReconstructionReport)
    s = rep.summary()
    assert s["overall_pass"] and s["max_impedance_rel_error"] < 0.01
    assert s["reported_a_inf_difference"] is not None
    # an impossible tolerance fails, and a forged band-consistent A_inf is refused
    strict = rad.reconstruction_criteria_from_dict(dict(FULL_BLOCK, block_rel_tol=1e-9, modal_rel_tol=1e-9))
    assert not rad.compare_derived_reconstruction(k, der, strict).overall_pass
    forged = dataclasses.replace(k, added_mass_infinite=k.added_mass_infinite + 1.0)
    with pytest.raises(rad.RadiationError):
        rad.compare_derived_reconstruction(forged, der, crit)


def test_modal_error_is_relative_to_spectral_radius():
    ref = np.diag([100.0, 100.0, 10.0, 1e-5])
    rec = np.diag([102.0, 100.0, 10.0, 5e-5])          # cluster error 2 of rho 100
    assert rad._modal_error(rec, ref, 1e-3, 0.01) == pytest.approx(0.02)
    rec2 = np.diag([100.0, 100.0, 10.3, 0.0])          # 0.3 on a small mode: 0.3 % of rho
    assert rad._modal_error(rec2, ref, 1e-3, 0.01) == pytest.approx(0.003)
    # subdominant cluster: spectral norm 0.3 over rho 100, not over the cluster's 10
    ref3 = np.diag([100.0, 10.0, 10.0, 1.0])
    rec3 = np.diag([100.0, 10.3, 10.0, 1.0])
    assert rad._modal_error(rec3, ref3, 1e-3, 0.01) == pytest.approx(0.003)
    crit = rad.reconstruction_criteria_from_dict(FULL_BLOCK)
    assert crit.to_dict()["modal_reference"] == "spectral_radius"
    assert rad.reconstruction_criteria_from_dict(crit.to_dict()) == crit
    with pytest.raises(rad.ReconstructionCriteriaError):
        rad.reconstruction_criteria_from_dict(dict(FULL_BLOCK, modal_reference="own_eigenvalue"))


def _band_kernel(ds):
    return rad.radiation_kernel_from_dataset(ds, kernel_settings(**BAND))


def test_verifier_binds_calibration_set_and_delta():
    ds = make_oracle_dataset(omegas=OMEGAS)
    k = _band_kernel(ds)
    om, A, ainf = ds["omega"].values, ds["added_mass"].values, ds["added_mass_infinite"].values
    prov = dict(k.provenance)
    # substitute evaluation frequencies and shift A_inf to match them: refused
    sub = om[(om >= 1.0) & (om <= 1.3)]
    a_rec, _ = k.reconstruct(sub)
    idx = [int(np.flatnonzero(om == w)[0]) for w in sub]
    shift = np.mean(A[idx] - a_rec, axis=0)
    forged = dataclasses.replace(k, added_mass_infinite=k.added_mass_infinite + shift,
                                 provenance={**prov, "a_inf_calibration_omegas": sub.tolist()})
    with pytest.raises(rad.RadiationError, match='calibration'):
        rad.verify_band_consistent_added_mass_infinite(forged, ainf, om, A)
    bad_delta = dataclasses.replace(k, provenance={**prov, "a_inf_band_consistent_delta":
                                                   (np.asarray(prov["a_inf_band_consistent_delta"]) * 2).tolist()})
    with pytest.raises(rad.RadiationError, match='difference'):
        rad.verify_band_consistent_added_mass_infinite(bad_delta, ainf, om, A)


def test_overlapping_calibration_and_legacy_criteria_are_refused():
    ds = make_oracle_dataset(omegas=OMEGAS)
    der = ops.derived_operators_from_dataset(ds, make_settings())
    C, lam, _ = _oracle_matrices(12)
    P = der.projection
    st = kernel_settings(zero_frequency_damping=P.T @ (C / lam) @ P,
                         a_inf_policy="band_consistent", a_inf_calibration_min=0.1,
                         a_inf_calibration_max=1.0)
    k = rad.radiation_kernel_from_derived_operators(der, st)
    with pytest.raises(rad.ReconstructionCriteriaError, match='below the evaluation window'):
        rad.compare_derived_reconstruction(k, der, rad.reconstruction_criteria_from_dict(FULL_BLOCK))
    legacy = rad.ReconstructionCriteria(omega_min=1.5, omega_max=2.0, added_mass_abs_tol=1.0,
                                        added_mass_rel_tol=0.1, damping_abs_tol=1.0, damping_rel_tol=0.1)
    with pytest.raises(rad.ReconstructionCriteriaError, match='full-block'):
        rad.compare_derived_reconstruction(k, der, legacy)


def test_zero_reference_blocks_and_weak_mode_report():
    z = np.zeros((1, 4, 4))
    crit = rad.reconstruction_criteria_from_dict(FULL_BLOCK)
    a = np.eye(4)[None] * 10.0
    rep = rad._full_block_report(crit, np.array([1.0]), a, z, a, z, None)
    assert rep.overall_pass                      # exact zero damping reconstruction
    ref = np.diag([100.0, 50.0, 1.0, 1e-6])[None]
    rec = np.diag([100.0, 50.0, 1.0, 3e-6])[None]
    rep = rad._full_block_report(crit, np.array([1.0]), ref, rec, ref, ref, None)
    weak = rep.summary()["reported_damping_weak_modes"]
    assert weak == [[[1e-6, 3e-6]]]


def test_band_consistent_kernel_round_trip(tmp_path):
    ds = make_oracle_dataset(omegas=OMEGAS)
    k = _band_kernel(ds)
    rad.save_radiation_kernel(k, tmp_path / 'k.nc')
    back = rad.load_radiation_kernel(tmp_path / 'k.nc')
    assert back.content_hash() == k.content_hash()
    np.testing.assert_array_equal(
        rad.verify_band_consistent_added_mass_infinite(back, ds["added_mass_infinite"].values,
                                                      ds["omega"].values, ds["added_mass"].values),
        k.added_mass_infinite)


def _paired_derived_kernels(**extra):
    """Derived kernels at dt and 2 dt (same horizon), A_inf calibrated at dt."""
    ds = make_oracle_dataset(omegas=OMEGAS)
    der = ops.derived_operators_from_dataset(ds, make_settings())
    C, lam, _ = _oracle_matrices(12)
    zero = der.projection.T @ (C / lam) @ der.projection
    fine = kernel_settings(zero_frequency_damping=zero, a_inf_calibration_dt=0.01, **BAND, **extra)
    coarse = kernel_settings(zero_frequency_damping=zero, dt=0.02, n_time=(N_TIME - 1) // 2 + 1,
                             a_inf_calibration_dt=0.01, **BAND, **extra)
    return (ds, der, rad.radiation_kernel_from_derived_operators(der, coarse),
            rad.radiation_kernel_from_derived_operators(der, fine))


def test_separate_calibration_step_shares_one_a_inf_across_dt():
    ds, der, coarse, fine = _paired_derived_kernels()
    # Calibrated once on the fine grid: bit-identical A_inf for both time steps.
    np.testing.assert_array_equal(coarse.added_mass_infinite, fine.added_mass_infinite)
    own = rad.radiation_kernel_from_derived_operators(
        der, dataclasses.replace(coarse.settings, a_inf_calibration_dt=None))
    assert not np.array_equal(own.added_mass_infinite, fine.added_mass_infinite)
    # Declaring the kernel's own step is the plain band-consistent kernel.
    plain_fine = rad.radiation_kernel_from_derived_operators(
        der, dataclasses.replace(fine.settings, a_inf_calibration_dt=None))
    np.testing.assert_array_equal(plain_fine.added_mass_infinite, fine.added_mass_infinite)
    np.testing.assert_array_equal(plain_fine.kernel, fine.kernel)
    assert coarse.provenance["a_inf_calibration_dt"] == 0.01
    assert any("rebuilt at dt=0.01" in a for a in coarse.assumptions)
    # Both prove against their own derived artifact; the comparator passes it.
    for k in (coarse, fine):
        out = rad.verify_band_consistent_added_mass_infinite(
            k, der.derived_added_mass_infinite, der.omegas, der.derived_added_mass,
            calibration_source=der)
        np.testing.assert_array_equal(out, fine.added_mass_infinite)
    crit = rad.reconstruction_criteria_from_dict(FULL_BLOCK)
    assert rad.compare_derived_reconstruction(coarse, der, crit, source_dataset=ds).overall_pass


def test_separate_calibration_step_refusals(tmp_path):
    ds, der, coarse, fine = _paired_derived_kernels()
    args = (der.derived_added_mass_infinite, der.omegas, der.derived_added_mass)
    with pytest.raises(rad.RadiationError, match="derived operator artifact"):
        rad.verify_band_consistent_added_mass_infinite(coarse, *args)
    # the kernel's own (dt) calibration is not the declared one
    own = rad.radiation_kernel_from_derived_operators(
        der, dataclasses.replace(coarse.settings, a_inf_calibration_dt=None))
    swapped = dataclasses.replace(own, settings=coarse.settings,
                                  provenance={**own.provenance, "a_inf_calibration_dt": 0.01})
    with pytest.raises(rad.RadiationError):
        rad.verify_band_consistent_added_mass_infinite(swapped, *args, calibration_source=der)
    # a kernel without its recorded step is refused at construction (see
    # test_separate_calibration_claims_need_a_derived_kernel); the verifier
    # also checks the step for same-grid kernels
    same = rad.radiation_kernel_from_derived_operators(
        der, dataclasses.replace(fine.settings, a_inf_calibration_dt=None))
    claimed = dataclasses.replace(same, provenance={**same.provenance, "a_inf_calibration_dt": 0.01})
    with pytest.raises(rad.RadiationError, match="calibration step"):
        rad.verify_band_consistent_added_mass_infinite(claimed, *args, calibration_source=der)
    # settings: finer-or-equal step dividing the horizon, band-consistent only
    base = coarse.settings
    for bad in (dict(a_inf_calibration_dt=0.03), dict(a_inf_calibration_dt=0.0),
                dict(a_inf_calibration_dt=0.0123), dict(a_inf_calibration_dt=True),
                dict(a_inf_policy="computed", a_inf_calibration_min=None,
                     a_inf_calibration_max=None, a_inf_calibration_dt=0.01)):
        with pytest.raises(rad.RadiationSettingsError):
            dataclasses.replace(base, **bad)
    data = base.to_dict()
    assert data["a_inf_calibration_dt"] == 0.01
    assert rad.RadiationSettings.from_dict(data).to_dict() == data
    assert "a_inf_calibration_dt" not in dataclasses.replace(base, a_inf_calibration_dt=None).to_dict()
    # save/load keeps the declaration and the proof
    rad.save_radiation_kernel(coarse, tmp_path / "coarse.nc")
    back = rad.load_radiation_kernel(tmp_path / "coarse.nc")
    assert back.content_hash() == coarse.content_hash()
    rad.verify_band_consistent_added_mass_infinite(back, *args, calibration_source=der)


def test_dataset_route_refuses_separate_calibration_step():
    ds = make_oracle_dataset(omegas=OMEGAS)
    settings = kernel_settings(dt=0.02, n_time=(N_TIME - 1) // 2 + 1, a_inf_calibration_dt=0.01, **BAND)
    with pytest.raises(rad.RadiationSettingsError, match="derived-operator route"):
        rad.radiation_kernel_from_dataset(ds, settings)
    rad.radiation_kernel_from_dataset(ds, dataclasses.replace(settings, a_inf_calibration_dt=0.02))


def test_separate_calibration_claims_need_a_derived_kernel(tmp_path):
    ds = make_oracle_dataset(omegas=OMEGAS)
    same = rad.radiation_kernel_from_dataset(ds, kernel_settings(**BAND))
    claim = dataclasses.replace(same.settings, a_inf_calibration_dt=0.005)
    with pytest.raises(rad.RadiationError, match="derived-operator kernel only"):
        dataclasses.replace(same, settings=claim,
                            provenance={**same.provenance, "a_inf_calibration_dt": 0.005})
    _, der, coarse, _ = _paired_derived_kernels()
    with pytest.raises(rad.RadiationError, match="recorded step"):
        dataclasses.replace(coarse, provenance={k: v for k, v in coarse.provenance.items()
                                                if k != "a_inf_calibration_dt"})
    legacy = rad.ReconstructionCriteria(omega_min=0.5, omega_max=2.0, added_mass_abs_tol=1.0,
                                        added_mass_rel_tol=0.1, damping_abs_tol=1.0, damping_rel_tol=0.1)
    with pytest.raises(rad.RadiationError, match="derived-operator route only"):
        rad.compare_reconstruction(coarse, ds, legacy)


def test_overflowing_calibration_ratio_is_a_typed_refusal():
    with pytest.raises(rad.RadiationSettingsError):
        kernel_settings(dt=0.01, n_time=101, a_inf_calibration_dt=1e-320, **BAND)
    data = kernel_settings(**BAND).to_dict()
    data.update(n_time=101, a_inf_calibration_dt=1e-320)
    with pytest.raises(rad.RadiationSettingsError):
        rad.RadiationSettings.from_dict(data)


# ---------------------------------------------------------------- ADR-015 L6-4 H-c: normalised full-block report

def _synthetic_blocks(seed=3):
    rng = np.random.default_rng(seed)
    window = np.array([0.5, 0.8, 1.1])
    def spd(scale):
        Q = rng.normal(size=(4, 4))
        X = Q @ Q.T + 4 * np.eye(4)
        return X * scale[:, None] * scale[None, :]
    si = np.array([1.0, 1.0, 50.0, 50.0])                 # rotations ~ L^2 larger in SI
    a_ref = np.array([spd(si) for _ in window])
    b_ref = np.array([spd(si) for _ in window])
    a_rec = a_ref * (1 + 1e-3 * rng.normal(size=a_ref.shape))
    b_rec = b_ref * (1 + 1e-3 * rng.normal(size=b_ref.shape))
    return window, a_rec, b_rec, a_ref, b_ref


def test_normalised_report_equals_the_report_of_scaled_operators():
    window, a_rec, b_rec, a_ref, b_ref = _synthetic_blocks()
    S = np.array([1.0, 1.0, 1 / 50.0, 1 / 50.0])
    base = rad.reconstruction_criteria_from_dict(FULL_BLOCK)
    normalised = dataclasses.replace(base, dof_scales=tuple(S))
    got = rad._full_block_report(normalised, window, a_rec, b_rec, a_ref, b_ref, None)
    M = S[:, None] * S[None, :]
    want = rad._full_block_report(base, window, a_rec * M, b_rec * M, a_ref * M, b_ref * M, None)
    assert np.array_equal(got.impedance_rel_error, want.impedance_rel_error)
    assert np.array_equal(got.added_mass_modal_error, want.added_mass_modal_error)
    assert np.array_equal(got.damping_coefficient_error, want.damping_coefficient_error)
    unit = dataclasses.replace(base, dof_scales=(1.0, 1.0, 1.0, 1.0))      # translation-only: a no-op
    plain = rad._full_block_report(base, window, a_rec, b_rec, a_ref, b_ref, None)
    same = rad._full_block_report(unit, window, a_rec, b_rec, a_ref, b_ref, None)
    assert np.array_equal(plain.impedance_rel_error, same.impedance_rel_error)


def test_dof_scales_serialisation_legacy_form_and_refusals():
    base = rad.reconstruction_criteria_from_dict(FULL_BLOCK)
    assert "dof_scales" not in base.to_dict()
    scaled = dataclasses.replace(base, dof_scales=(1.0, 0.02))
    assert rad.reconstruction_criteria_from_dict(scaled.to_dict()) == scaled
    with pytest.raises(rad.ReconstructionCriteriaError, match="dof_scales"):
        dataclasses.replace(base, dof_scales=(1.0, -1.0))
    window, a_rec, b_rec, a_ref, b_ref = _synthetic_blocks()
    with pytest.raises(rad.ReconstructionCriteriaError, match="4 DOFs"):
        rad._full_block_report(scaled, window, a_rec, b_rec, a_ref, b_ref, None)
