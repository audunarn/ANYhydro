"""Focused derived-operator and retained-coordinate radiation tests.

Development tests only: they confer no scientific qualification. They use
the independent exponential-kernel oracle from ``test_radiation`` and the
public assemble/fixture APIs; no backend is involved.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import anyhydro
from anyhydro import operators as ops
from anyhydro import radiation as rad
from anyhydro.errors import TimeDomainAdmissionError

from test_radiation import (
    ORACLE_DT,
    ORACLE_N_TIME,
    _oracle_matrices,
    make_oracle_dataset,
)

RETAINED4 = (
    "fixture-body-alpha::surge",
    "fixture-body-alpha::heave",
    "fixture-body-beta::surge",
    "fixture-body-beta::heave",
)
ALPHA6 = tuple(f"fixture-body-alpha::{d}" for d in
               ("surge", "sway", "heave", "roll", "pitch", "yaw"))

# Captured from the pre-refactor implementation (evidence
# derived-operator-worker-01): the dataset bridge must reproduce this
# full-12 oracle kernel hash bit-for-bit after the shared-core refactor.
PRE_REFACTOR_FULL12_HASH = "10bd41545ea153aa94cae508bcf1da3b7d0f3e2145553812de73060ccfc9091c"


def make_norm(n, scales=None, name="test-normalization"):
    if scales is None:
        scales = [1.0] * n
    return ops.NormalizationSpec(
        name=name,
        unit_convention="test: q = S q_hat, Q_hat = S^T Q (declared test units)",
        scales=scales,
    )


def make_criteria(**overrides):
    defaults = dict(
        reciprocity_abs_tol=1.0e3,
        reciprocity_rel_tol=0.0,
        reciprocity_floor=1.0e-12,
        correction_abs_tol=1.0e3,
        correction_rel_tol=0.0,
        correction_floor=1.0e-12,
        derived_symmetry_tol=1.0e-9,
        damping_psd_tol=1.0e-9,
    )
    defaults.update(overrides)
    return ops.DerivedOperatorCriteria(**defaults)


def make_settings(retained=RETAINED4, policy=ops.CORRECTION_RECIPROCAL_AVERAGE,
                  criteria=None, normalization=None):
    return ops.DerivedOperatorSettings(
        correction_policy=policy,
        retained_dofs=retained,
        normalization=normalization or make_norm(len(retained)),
        criteria=criteria or make_criteria(),
    )


@pytest.fixture(scope="module")
def oracle_ds():
    return make_oracle_dataset()


@pytest.fixture(scope="module")
def derived4(oracle_ds):
    return ops.derived_operators_from_dataset(oracle_ds, make_settings())


# ---------------------------------------------------------------------------
# projection, virtual work, full coupling


def test_projection_is_exact_label_selector(derived4, oracle_ds):
    source = [str(v) for v in oracle_ds["influenced_dof"].values.tolist()]
    P = derived4.projection
    assert P.shape == (12, 4)
    # each column has exactly one 1, at the label's own position (no
    # hardcoded numeric positions anywhere in this check)
    for j, label in enumerate(RETAINED4):
        col = P[:, j]
        assert np.count_nonzero(col) == 1
        assert col[source.index(label)] == 1.0
    assert np.array_equal(P, ops.build_projection(source, RETAINED4))
    assert np.array_equal(P.T @ P, np.eye(4))
    assert derived4.source_dof_order == tuple(source)


def test_projection_preserves_virtual_work_and_full_coupling(derived4):
    P = derived4.projection
    rng = np.random.default_rng(7)
    q_ret = rng.standard_normal(4)
    Q_full = rng.standard_normal(12)
    q_full = P @ q_ret
    Q_ret = P.T @ Q_full
    # q_full = P q_ret and Q_ret = P^T Q_full preserve virtual work exactly
    assert Q_ret @ q_ret == pytest.approx(Q_full @ q_full, rel=1e-15, abs=1e-15)
    # full coupled cross-body blocks survive in the retained operators
    i_alpha_surge = RETAINED4.index("fixture-body-alpha::surge")
    i_beta_surge = RETAINED4.index("fixture-body-beta::surge")
    assert np.all(np.abs(derived4.raw_damping[:, i_alpha_surge, i_beta_surge]) > 0.0)
    assert np.all(np.abs(derived4.raw_added_mass[:, i_beta_surge, i_alpha_surge]) > 0.0)


def test_work_conjugate_normalization_translations_only(derived4):
    s = np.asarray(derived4.settings.normalization.scales)
    rng = np.random.default_rng(11)
    q_hat = rng.standard_normal(4)
    Q_ret = rng.standard_normal(4)
    q_ret = s * q_hat          # q = S q_hat
    Q_hat = s * Q_ret          # Q_hat = S^T Q (S diagonal)
    assert Q_hat @ q_hat == pytest.approx(Q_ret @ q_ret, rel=1e-15, abs=1e-15)


def test_work_conjugate_normalization_mixed_translational_rotational(oracle_ds):
    # mixed retained subspace: translations scaled by L, rotations by one
    # (q_hat = (u/L, theta), Q_hat = (L F, M))
    L = 2.5
    settings = make_settings(
        retained=ALPHA6,
        normalization=make_norm(6, scales=[L, L, L, 1.0, 1.0, 1.0],
                                name="translations-length-rotations-one"),
    )
    art = ops.derived_operators_from_dataset(oracle_ds, settings)
    s = np.asarray(art.settings.normalization.scales)
    rng = np.random.default_rng(13)
    q_hat = rng.standard_normal(6)
    Q_ret = rng.standard_normal(6)
    q_ret = s * q_hat
    Q_hat = s * Q_ret
    assert Q_hat @ q_hat == pytest.approx(Q_ret @ q_ret, rel=1e-15, abs=1e-15)
    # full chain through P: Q_full^T (P S q_hat) = (S P^T Q_full)^T q_hat
    Q_full = rng.standard_normal(12)
    lhs = Q_full @ (art.projection @ (s * q_hat))
    rhs = (s * (art.projection.T @ Q_full)) @ q_hat
    assert lhs == pytest.approx(rhs, rel=1e-14, abs=1e-14)


def test_normalization_is_strict_congruence(oracle_ds):
    s = np.array([2.0, 0.5, 3.0, 1.5])
    settings = make_settings(normalization=make_norm(4, scales=s))
    art = ops.derived_operators_from_dataset(oracle_ds, settings)
    raw = art.raw_added_mass_infinite
    expected_defect = np.abs(s[:, None] * raw * s[None, :]
                             - (s[:, None] * raw * s[None, :]).T)
    np.testing.assert_allclose(
        art.report.reciprocity["defect"]["added_mass_infinite"], expected_defect)
    # threshold formula: abs + rel * max(global matrix scale, floor)
    crit = settings.criteria
    scale = float(np.max(np.abs(s[:, None] * raw * s[None, :])))
    expected_thr = crit.reciprocity_abs_tol + crit.reciprocity_rel_tol * max(
        scale, crit.reciprocity_floor)
    np.testing.assert_allclose(
        art.report.reciprocity["threshold"]["added_mass_infinite"], expected_thr)


# ---------------------------------------------------------------------------
# explicit policies


def test_policy_none_keeps_raw_and_average_symmetrizes(oracle_ds):
    none_art = ops.derived_operators_from_dataset(
        oracle_ds, make_settings(policy=ops.CORRECTION_NONE))
    avg_art = ops.derived_operators_from_dataset(oracle_ds, make_settings())
    for arr in ("added_mass", "damping", "added_mass_infinite"):
        raw = getattr(none_art, f"raw_{arr}")
        np.testing.assert_array_equal(getattr(none_art, f"derived_{arr}"), raw)
        derived = getattr(avg_art, f"derived_{arr}")
        expected = 0.5 * (raw + np.transpose(raw, (0, 2, 1))) \
            if raw.ndim == 3 else 0.5 * (raw + raw.T)
        np.testing.assert_array_equal(derived, expected)
    # hydrostatics and excitation are projected only, never policy-modified
    for art in (none_art, avg_art):
        C_full = np.asarray(oracle_ds["hydrostatic_stiffness"].values)
        np.testing.assert_array_equal(
            art.hydrostatic_stiffness, art.projection.T @ C_full @ art.projection)
        F_full = np.asarray(oracle_ds["excitation_force"].values)
        np.testing.assert_allclose(
            art.excitation_force,
            np.einsum("ia,owi->owa", art.projection, F_full))
    assert none_art.hydrostatic_stiffness.shape == (4, 4)
    assert avg_art.excitation_force.dtype == np.complex128


def test_no_default_policy_or_subspace():
    with pytest.raises(TypeError):
        ops.DerivedOperatorSettings(
            retained_dofs=RETAINED4,
            normalization=make_norm(4),
            criteria=make_criteria(),
        )
    with pytest.raises(ops.DerivedOperatorSettingsError, match="no default policy"):
        ops.DerivedOperatorSettings(
            correction_policy="auto",
            retained_dofs=RETAINED4,
            normalization=make_norm(4),
            criteria=make_criteria(),
        )


# ---------------------------------------------------------------------------
# admission: failures retained, radiation refused, nothing clamped


def test_correction_failure_is_retained_and_refused(oracle_ds):
    settings = make_settings(criteria=make_criteria(correction_abs_tol=0.0,
                                                    correction_rel_tol=0.0))
    art = ops.derived_operators_from_dataset(oracle_ds, settings)
    assert not art.report.correction_pass
    assert not art.admitted
    # complete residual/threshold arrays are retained for inspection
    assert np.any(~art.report.correction["pass"]["damping"])
    assert np.all(art.report.correction["threshold"]["damping"] == 0.0)
    assert np.any(art.report.correction["defect"]["damping"] > 0.0)
    with pytest.raises(ops.DerivedOperatorAdmissionError, match="not admitted"):
        rad.radiation_kernel_from_derived_operators(
            art, make_derived_settings(art))


def test_policy_none_refuses_asymmetric_raw(oracle_ds):
    settings = make_settings(
        policy=ops.CORRECTION_NONE,
        criteria=make_criteria(reciprocity_abs_tol=1.0e-9,
                               correction_abs_tol=1.0e-9,
                               derived_symmetry_tol=1.0e-9))
    art = ops.derived_operators_from_dataset(oracle_ds, settings)
    assert not art.report.raw_reciprocity_pass
    assert not art.report.derived_symmetry_pass
    assert not art.admitted
    with pytest.raises(ops.DerivedOperatorAdmissionError):
        rad.radiation_kernel_from_derived_operators(
            art, make_derived_settings(art))


def test_negative_b_excluded_mode_is_rawfull_diagnostic_only(oracle_ds):
    C, lam, a_inf = _oracle_matrices(12)
    C[3, 3] = -4000.0  # roll of body alpha: excluded from the retained four
    ds = make_oracle_dataset(C=C, lam=lam, a_inf=a_inf)
    art = ops.derived_operators_from_dataset(ds, make_settings())
    report = art.report
    # the excluded-DOF PSD defect stays visible as an informational
    # unscaled mixed-SI raw-full diagnostic, explicitly labelled
    assert report.raw_full_coordinate_label == ops.RAW_FULL_DIAGNOSTIC_COORDINATES
    assert report.raw_full_coordinate_label == "unscaled_mixed_si_informational"
    assert np.min(report.raw_full_unscaled_damping_min_eigenvalues) < 0.0
    assert report.raw_full_unscaled_negative_mode_indices.size >= 1
    assert report.raw_full_unscaled_negative_mode_retained_alignment.shape == \
        report.raw_full_unscaled_negative_mode_indices.shape
    assert np.all(report.raw_full_unscaled_negative_mode_retained_alignment < 0.1)
    # ... but it is not a retained failure and the retained model is admitted
    assert report.damping_psd_pass
    assert art.admitted
    assert np.min(report.damping_eigenvalues) > 0.0


def test_negative_b_included_mode_not_clamped(oracle_ds):
    C, lam, a_inf = _oracle_matrices(12)
    C[0, 0] = -4000.0  # surge of body alpha: inside the retained four
    ds = make_oracle_dataset(C=C, lam=lam, a_inf=a_inf)
    art = ops.derived_operators_from_dataset(ds, make_settings())
    report = art.report
    assert not report.damping_psd_pass
    assert not art.admitted
    # negative modes are recorded with eigenvalues and vectors, never clamped
    assert report.damping_negative_mode_frequency_indices.size >= 1
    assert np.all(report.damping_negative_mode_eigenvalues
                  < -art.settings.criteria.damping_psd_tol)
    assert report.damping_negative_mode_vectors.shape[1] == 4
    assert np.all(art.derived_damping[:, 0, 0] < 0.0)
    with pytest.raises(ops.DerivedOperatorAdmissionError):
        rad.radiation_kernel_from_derived_operators(
            art, make_derived_settings(art))


def test_normalized_psd_gate_and_physical_direction():
    # parent discriminator: a finite PSD tolerance is not congruence-
    # invariant. Same SI data, different declared normalization, different
    # admission. Scales S=100, B diagonal entry -1e-11, tolerance 1e-10.
    ds = make_oracle_dataset(omegas=np.array([1.0, 2.0, 3.0]),
                             C=np.eye(12), lam=np.ones((12, 12)),
                             a_inf=np.eye(12) * 2.0)
    labels = [str(v) for v in ds["influenced_dof"].values.tolist()]
    i = labels.index(RETAINED4[0])
    ds["radiation_damping"].values[:, i, i] = -1e-11
    # unscaled declaration: the -1e-11 SI defect stays inside 1e-10
    unscaled = ops.derived_operators_from_dataset(
        ds, make_settings(criteria=make_criteria(damping_psd_tol=1e-10)))
    assert unscaled.admitted
    # declared S=100 normalization: Bhat = S^T B S puts the same defect at
    # -1e-7 in normalized coordinates and the gate must fail there
    scaled = ops.derived_operators_from_dataset(
        ds, make_settings(normalization=make_norm(4, scales=[100.0] * 4,
                                                  name="scale-by-100"),
                           criteria=make_criteria(damping_psd_tol=1e-10)))
    assert not scaled.admitted
    assert not scaled.report.damping_psd_pass
    assert np.all(scaled.report.damping_eigenvalues[:, 0] < -1e-10)
    # the mechanic arrays remain SI and identical in both declarations
    np.testing.assert_array_equal(scaled.derived_damping, unscaled.derived_damping)
    assert scaled.derived_damping[0, 0, 0] == pytest.approx(-1e-11)
    # a normalized eigenvector v_hat maps to the physical direction S v_hat
    v_hat = scaled.report.damping_negative_mode_vectors[0]
    physical = np.asarray(scaled.settings.normalization.scales) * v_hat
    assert np.argmax(np.abs(physical)) == 0
    np.testing.assert_allclose(np.abs(physical[0]), 100.0)


def test_all_negative_modes_and_complete_eigenbasis():
    # every negative retained mode is recorded (not only the worst per
    # frequency) alongside the complete normalized eigenbasis
    ds = make_oracle_dataset(omegas=np.array([1.0, 2.0, 3.0]),
                             C=-np.eye(12), lam=np.ones((12, 12)),
                             a_inf=np.eye(12) * 2.0)
    art = ops.derived_operators_from_dataset(
        ds, make_settings(criteria=make_criteria(damping_psd_tol=1e-9)))
    report = art.report
    assert not report.damping_psd_pass
    assert not art.admitted
    n_w = 3
    assert report.damping_eigenvectors.shape == (n_w, 4, 4)
    # 3 frequencies x 4 negative retained modes: all of them recorded
    assert report.damping_negative_mode_frequency_indices.size == 12
    assert report.damping_negative_mode_basis_indices.size == 12
    assert np.all(report.damping_negative_mode_eigenvalues < -1e-9)
    s = np.asarray(art.settings.normalization.scales)
    for o in range(n_w):
        V = report.damping_eigenvectors[o]
        np.testing.assert_allclose(V.T @ V, np.eye(4), atol=1e-12)
        B_hat = s[:, None] * art.derived_damping[o] * s[None, :]
        np.testing.assert_allclose(
            V @ np.diag(report.damping_eigenvalues[o]) @ V.T, B_hat, atol=1e-9)


def test_coupled_negative_mode_vectors_satisfy_eigen_equation():
    # regression for the parent-02 defect (max mismatch 74.11): eigh
    # returns eigenvectors as COLUMNS, so the negative mode at (frequency
    # f, basis b) is eigvecs[f, :, b] -- never the row eigvecs[f, b, :].
    # Coupled indefinite retained block with declared scales: every
    # reported negative mode must satisfy B_hat v = lambda v with unit
    # norm and must be the matching column of the complete eigenbasis.
    ds = make_oracle_dataset(omegas=np.array([1.0, 2.0, 3.0]),
                             C=np.eye(12), lam=np.ones((12, 12)),
                             a_inf=np.eye(12) * 2.0)
    scales = np.array([1.0, 2.0, 3.0, 4.0])
    spec = make_settings(
        normalization=make_norm(4, scales=list(scales), name="coupled-scales"))
    indices = [list(ds["influenced_dof"].values).index(k)
               for k in spec.retained_dofs]
    B = np.array([[2.0, 1.0, 0.0, 0.5],
                  [1.0, -2.0, 0.3, 0.0],
                  [0.0, 0.3, 3.0, 0.2],
                  [0.5, 0.0, 0.2, -3.0]])
    for a, i in enumerate(indices):
        for b, j in enumerate(indices):
            ds["radiation_damping"].values[:, i, j] = B[a, b]
    art = ops.derived_operators_from_dataset(ds, spec)
    report = art.report
    assert not report.damping_psd_pass
    assert not art.admitted
    B_hat = scales[:, None] * B * scales[None, :]
    tol = art.settings.criteria.damping_psd_tol
    # multiple coupled negative modes per frequency, not only the worst
    expected = 3 * int(np.count_nonzero(np.linalg.eigvalsh(B_hat) < -tol))
    assert expected >= 6
    freqs = report.damping_negative_mode_frequency_indices
    basis = report.damping_negative_mode_basis_indices
    values = report.damping_negative_mode_eigenvalues
    vectors = report.damping_negative_mode_vectors
    assert values.size == expected
    assert vectors.shape == (expected, 4)
    for value, v, f, b in zip(values, vectors, freqs, basis):
        np.testing.assert_allclose(B_hat @ v, value * v, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-12)
        # the reported mode is the matching COLUMN of the eigenbasis
        np.testing.assert_allclose(report.damping_eigenvectors[f][:, b], v,
                                   atol=1e-12)


def test_residuals_complete_and_frequency_indexed(derived4, oracle_ds):
    report = derived4.report
    n_w = int(oracle_ds["omega"].size)
    for key in ("added_mass", "damping"):
        for group in ("reciprocity", "correction"):
            for kind in ("defect", "threshold", "pass"):
                assert getattr(report, group)[kind][key].shape == (n_w, 4, 4), \
                    f"{group}.{kind}.{key} must be frequency-indexed"
        assert report.symmetry_defect[key].shape == (n_w, 4, 4)
    # A_inf has no frequency axis: its complete residual is the single matrix
    for group in ("reciprocity", "correction"):
        for kind in ("defect", "threshold", "pass"):
            assert getattr(report, group)[kind]["added_mass_infinite"].shape == (4, 4)
    # independent recomputation of one frequency slice of the damping defect
    o = 7
    s = np.asarray(derived4.settings.normalization.scales)
    Xh = s[:, None] * derived4.raw_damping[o] * s[None, :]
    np.testing.assert_allclose(
        report.reciprocity["defect"]["damping"][o], np.abs(Xh - Xh.T))
    # maxima appear in the summary only, never instead of the full arrays
    summary = report.summary()
    assert summary["max_reciprocity_defect"]["damping"] >= 0.0
    assert summary["max_correction_defect"]["damping"] >= 0.0


def make_derived_settings(art, **overrides):
    """Ordinary RadiationSettings with B(0) projected into retained coords."""
    C, lam, _ = _oracle_matrices(12)
    b0_full = C / lam
    b0 = art.projection.T @ b0_full @ art.projection
    defaults = dict(
        dt=ORACLE_DT,
        n_time=ORACLE_N_TIME,
        symmetry_tol=1.0e-9,
        passivity_tol=1.0e-9,
        low_tail_policy=rad.LOW_TAIL_EXPLICIT_ZERO,
        zero_frequency_damping=b0,
    )
    defaults.update(overrides)
    return rad.RadiationSettings(**defaults)


# ---------------------------------------------------------------------------
# settings and label refusals


def test_dof_label_refusals():
    with pytest.raises(ops.DerivedOperatorSettingsError, match="not be empty"):
        make_settings(retained=())
    with pytest.raises(ops.DerivedOperatorSettingsError, match="duplicate"):
        make_settings(retained=(RETAINED4[0], RETAINED4[0], RETAINED4[2], RETAINED4[3]))
    with pytest.raises(ops.DerivedOperatorSettingsError, match="body-qualified"):
        make_settings(retained=("surge", "heave", "sway", "roll"))
    with pytest.raises(ops.DerivedOperatorSettingsError, match="got str"):
        make_settings(retained="fixture-body-alpha::surge")


def test_unknown_retained_label_refused_at_build(oracle_ds):
    settings = make_settings(retained=(
        "fixture-body-alpha::surge", "fixture-body-alpha::heave",
        "fixture-body-gamma::surge", "fixture-body-gamma::heave"))
    with pytest.raises(ops.DerivedOperatorSettingsError, match="not present"):
        ops.derived_operators_from_dataset(oracle_ds, settings)


def test_scale_refusals():
    for bad in ([0.0, 1.0, 1.0, 1.0], [1.0, -1.0, 1.0, 1.0],
                [1.0, np.nan, 1.0, 1.0], [np.inf, 1.0, 1.0, 1.0]):
        with pytest.raises(ops.DerivedOperatorSettingsError):
            make_norm(4, scales=bad)
    with pytest.raises(ops.DerivedOperatorSettingsError, match="not bools"):
        make_norm(4, scales=[True, 1.0, 1.0, 1.0])
    with pytest.raises(ops.DerivedOperatorSettingsError, match="not be empty"):
        make_norm(0, scales=[])


def test_criteria_and_settings_refusals():
    with pytest.raises(ops.DerivedOperatorCriteriaError, match="not a bool"):
        make_criteria(reciprocity_abs_tol=True)
    with pytest.raises(ops.DerivedOperatorCriteriaError, match="finite"):
        make_criteria(correction_abs_tol=np.nan)
    with pytest.raises(ops.DerivedOperatorCriteriaError, match=">= 0"):
        make_criteria(derived_symmetry_tol=-1.0)
    with pytest.raises(ops.DerivedOperatorCriteriaError, match="> 0"):
        make_criteria(reciprocity_floor=0.0)
    with pytest.raises(ops.DerivedOperatorSettingsError, match="unknown"):
        ops.DerivedOperatorSettings(
            correction_policy=ops.CORRECTION_NONE,
            retained_dofs=RETAINED4,
            normalization=make_norm(4),
            criteria=make_criteria(),
            surprise=1,
        )
    with pytest.raises(ops.DerivedOperatorSettingsError, match="one entry per"):
        ops.derived_operators_from_dataset(
            make_oracle_dataset(), make_settings(normalization=make_norm(3)))


def test_missing_a_inf_refused(oracle_ds):
    import anyhydro.fixtures as fx
    ds = fx.make_two_body_fixture(include_added_mass_infinite=False)
    with pytest.raises(TimeDomainAdmissionError):
        ops.derived_operators_from_dataset(ds, make_settings())


# ---------------------------------------------------------------------------
# immutability


def test_artifact_is_deeply_immutable(derived4):
    art = derived4
    with pytest.raises(ValueError):
        art.excitation_force[0, 0, 0] = 1.0 + 2.0j
    with pytest.raises(ValueError):
        art.raw_damping[0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        art.projection[0, 0] = 1.0
    with pytest.raises(ValueError):
        art.report.damping_eigenvalues[0, 0] = 1.0
    with pytest.raises(ValueError):
        art.report.damping_eigenvectors[0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        art.report.reciprocity["pass"]["damping"][0, 0] = False
    with pytest.raises(TypeError):
        art.provenance["new_key"] = 1
    with pytest.raises(TypeError):
        art.report.symmetry_pass["damping"] = True
    with pytest.raises(dataclasses.FrozenInstanceError):
        art.settings.correction_policy = ops.CORRECTION_NONE
    with pytest.raises(ValueError):
        art.settings.normalization.scales[0] = 2.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        art.settings.criteria.damping_psd_tol = 1.0
    assert art.settings.criteria.damping_psd_tol >= 0.0


def test_defensive_copies_at_construction(oracle_ds):
    scales = [2.0, 2.0, 2.0, 2.0]
    norm = make_norm(4, scales=scales)
    settings = make_settings(normalization=norm)
    scales[0] = 99.0  # caller mutation after construction changes nothing
    art = ops.derived_operators_from_dataset(oracle_ds, settings)
    assert art.settings.normalization.scales[0] == 2.0


# ---------------------------------------------------------------------------
# hash coverage and reader recomputation


def test_content_hash_covers_settings_criteria_and_data(oracle_ds):
    base = ops.derived_operators_from_dataset(oracle_ds, make_settings())
    same = ops.derived_operators_from_dataset(oracle_ds, make_settings())
    assert base.content_hash() == same.content_hash()
    # different criteria
    other_crit = ops.derived_operators_from_dataset(
        oracle_ds, make_settings(criteria=make_criteria(damping_psd_tol=1.0e-6)))
    assert base.content_hash() != other_crit.content_hash()
    # different policy
    other_pol = ops.derived_operators_from_dataset(
        oracle_ds, make_settings(policy=ops.CORRECTION_NONE))
    assert base.content_hash() != other_pol.content_hash()
    # different normalization
    other_norm = ops.derived_operators_from_dataset(
        oracle_ds, make_settings(normalization=make_norm(4, scales=[2.0] * 4)))
    assert base.content_hash() != other_norm.content_hash()
    # different source data
    C, lam, a_inf = _oracle_matrices(12)
    C[0, 0] += 1.0
    other_src = ops.derived_operators_from_dataset(
        make_oracle_dataset(C=C, lam=lam, a_inf=a_inf), make_settings())
    assert base.content_hash() != other_src.content_hash()


def test_reader_recomputes_invariants_and_refuses_bad_report(derived4):
    art = derived4
    # stored pass bit tampered: recomputation disagrees
    bad_report = dataclasses.replace(art.report, damping_psd_pass=False)
    with pytest.raises(ops.DerivedOperatorError, match="recomputed"):
        dataclasses.replace(art, report=bad_report)
    # derived array tampered: not the declared policy applied to raw
    bad_derived = art.derived_damping.copy()
    bad_derived[0, 0, 0] += 1.0
    with pytest.raises(ops.DerivedOperatorError, match="policy"):
        dataclasses.replace(art, derived_damping=bad_derived)
    # selector tampered
    bad_P = art.projection.copy()
    bad_P[0, 0] = 0.0
    with pytest.raises(ops.DerivedOperatorError, match="exact selector"):
        dataclasses.replace(art, projection=bad_P)


def test_raw_source_verification_by_full_rederivation(oracle_ds, derived4):
    assert derived4.matches_raw_source(oracle_ds)
    assert derived4.provenance["source_dataset_hash"] == \
        anyhydro.dataset_content_hash(oracle_ds)
    mutated = oracle_ds.copy(deep=True)
    mutated["excitation_force"].values[0, 0, 0] += 1.0
    assert not derived4.matches_raw_source(mutated)
    with pytest.raises(ops.DerivedOperatorError):
        derived4.verify_raw_source(mutated)
    with pytest.raises(ops.DerivedOperatorError, match="xarray Dataset"):
        derived4.verify_raw_source([1, 2, 3])


def test_forged_source_hash_cannot_attest_projection(oracle_ds, derived4):
    # parent counterexample: a valid artifact derived from a DIFFERENT
    # dataset, with its recorded source hash forged to match this one.
    # The hash alone proves nothing; only the full rederivation does.
    C, lam, a_inf = _oracle_matrices(12)
    C[0, 0] += 1.0
    changed = make_oracle_dataset(C=C, lam=lam, a_inf=a_inf)
    forged = ops.derived_operators_from_dataset(changed, make_settings())
    provenance = dict(forged.provenance)
    provenance["source_dataset_hash"] = anyhydro.dataset_content_hash(oracle_ds)
    forged = dataclasses.replace(forged, provenance=provenance)
    assert forged.provenance["source_dataset_hash"] == \
        derived4.provenance["source_dataset_hash"]
    assert not forged.matches_raw_source(oracle_ds)
    with pytest.raises(ops.DerivedOperatorError, match="rederivation"):
        forged.verify_raw_source(oracle_ds)
    # the honest artifact of the same source still proves out
    assert derived4.matches_raw_source(oracle_ds)


def test_source_controls_preserved_verbatim(oracle_ds, derived4):
    ds = oracle_ds.copy(deep=True)
    ds.attrs.update(backend_settings='{"method": "parent_analytic"}',
                    rho=1025.0, g=9.81, water_depth=150.0)
    art = ops.derived_operators_from_dataset(ds, make_settings())
    evidence = repr(art.provenance)
    for key in ("backend_settings", "method", "rho", "g", "water_depth",
                "length_unit"):
        assert key in evidence, f"missing producing source control {key}"
    assert art.provenance["source_attrs"]["rho"] == 1025.0
    assert art.provenance["source_attrs"]["backend_settings"] == \
        {"method": "parent_analytic"}
    # nothing invented: controls absent from the source stay absent
    assert "rho" not in derived4.provenance["source_attrs"]
    assert "backend_settings" not in derived4.provenance["source_attrs"]
    # relevant variable attrs are preserved (Ainf source evidence)
    assert derived4.provenance["source_variable_attrs"][
        "added_mass_infinite"]["source"].startswith("test oracle")
    assert art.matches_raw_source(ds)


def test_constructor_caps_and_body_consistency(derived4):
    # 13 source DOFs: refused by the constructor before array freezes
    thirteen = tuple(derived4.source_dof_order) + ("fixture-body-gamma::surge",)
    with pytest.raises(ops.DerivedOperatorLimitError, match="12"):
        dataclasses.replace(derived4, source_dof_order=thirteen)
    # 514 source frequencies: refused the same way
    with pytest.raises(ops.DerivedOperatorLimitError, match="513"):
        dataclasses.replace(
            derived4, omegas=np.linspace(0.05, 50.0, 514))
    # a source body missing one of its six DOFs is inconsistent
    eleven = tuple(l for k, l in enumerate(derived4.source_dof_order) if k != 5)
    with pytest.raises(ops.DerivedOperatorError, match="six DOFs"):
        dataclasses.replace(derived4, source_dof_order=eleven)
    # a body order that disagrees with the DOF axis is inconsistent
    with pytest.raises(ops.DerivedOperatorError, match="consistent"):
        dataclasses.replace(
            derived4, source_body_order=("fixture-body-alpha", "other-body"))


def test_loader_refuses_oversized_metadata_typed(derived4, tmp_path):
    path = tmp_path / "derived.npz"
    ops.save_derived_operators(derived4, path)

    def mutate_meta(mutator):
        data = np.load(path)
        arrays = {k: data[k] for k in data.files}
        import json
        meta = json.loads(str(arrays.pop("meta")[()]))
        mutator(meta)
        np.savez(path, meta=np.array(json.dumps(
            meta, sort_keys=True, separators=(",", ":"))), **arrays)

    # 13 declared source DOFs: typed loader refusal via the constructor cap
    mutate_meta(lambda m: m.update(
        {"source_dof_order": list(m["source_dof_order"]) + ["x::surge"]}))
    with pytest.raises(ops.DerivedOperatorSerializationError):
        ops.load_derived_operators(path)


def test_loader_raw_source_proof_keyword(derived4, oracle_ds, tmp_path):
    path = tmp_path / "derived.npz"
    ops.save_derived_operators(derived4, path)
    loaded = ops.load_derived_operators(path, source_dataset=oracle_ds)
    assert loaded.content_hash() == derived4.content_hash()
    mutated = oracle_ds.copy(deep=True)
    mutated["added_mass"].values[0, 0, 0] += 1.0
    with pytest.raises(ops.DerivedOperatorSerializationError,
                       match="raw-source proof"):
        ops.load_derived_operators(path, source_dataset=mutated)


def test_consumers_invoke_raw_source_proof(oracle_ds, derived4):
    settings = make_derived_settings(derived4)
    kernel = rad.radiation_kernel_from_derived_operators(
        derived4, settings, source_dataset=oracle_ds)
    criteria = rad.ReconstructionCriteria(
        omega_min=0.5, omega_max=8.0,
        added_mass_abs_tol=2.0, added_mass_rel_tol=1.0e-2,
        damping_abs_tol=2.0, damping_rel_tol=1.0e-2,
    )
    report = rad.compare_derived_reconstruction(
        kernel, derived4, criteria, source_dataset=oracle_ds)
    assert report.overall_pass
    # a mutated source is refused by the proof before any consumption
    mutated = oracle_ds.copy(deep=True)
    mutated["radiation_damping"].values[0, 0, 0] += 1.0
    with pytest.raises(ops.DerivedOperatorError, match="hash"):
        rad.radiation_kernel_from_derived_operators(
            derived4, settings, source_dataset=mutated)
    with pytest.raises(ops.DerivedOperatorError, match="hash"):
        rad.compare_derived_reconstruction(
            kernel, derived4, criteria, source_dataset=mutated)


# ---------------------------------------------------------------------------
# persistence


def test_save_reload_exact_roundtrip(derived4, tmp_path):
    path = tmp_path / "derived.npz"
    ops.save_derived_operators(derived4, path)
    loaded = ops.load_derived_operators(path)
    assert loaded.content_hash() == derived4.content_hash()
    assert loaded.admitted == derived4.admitted
    assert loaded.retained_dofs == derived4.retained_dofs
    assert loaded.source_dof_order == derived4.source_dof_order
    np.testing.assert_array_equal(loaded.raw_damping, derived4.raw_damping)
    np.testing.assert_array_equal(loaded.derived_added_mass, derived4.derived_added_mass)
    np.testing.assert_array_equal(loaded.excitation_force, derived4.excitation_force)
    np.testing.assert_array_equal(loaded.projection, derived4.projection)
    assert loaded.report.summary() == derived4.report.summary()
    assert dict(loaded.provenance) == dict(derived4.provenance)
    with pytest.raises(ops.DerivedOperatorSerializationError, match="does not exist"):
        ops.save_derived_operators(derived4, tmp_path / "missing_dir" / "x.npz")


def _rewrite_npz(path, out, array_overrides=None, meta_mutator=None):
    data = np.load(path)
    arrays = {k: data[k] for k in data.files}
    meta = arrays.pop("meta")
    if array_overrides:
        arrays.update(array_overrides)
    meta_obj = __import__("json").loads(str(meta[()]))
    if meta_mutator:
        meta_mutator(meta_obj)
    np.savez(out, meta=np.array(__import__("json").dumps(
        meta_obj, sort_keys=True, separators=(",", ":"))), **arrays)


def test_tampered_arrays_and_metadata_refused(derived4, tmp_path):
    import json

    path = tmp_path / "derived.npz"
    ops.save_derived_operators(derived4, path)

    # tampered data array: invariant verification refuses
    bad = tmp_path / "bad_data.npz"
    data = np.load(path)
    arrays = {k: data[k] for k in data.files}
    arrays["derived_damping"] = arrays["derived_damping"] + 1.0
    np.savez(bad, **arrays)
    with pytest.raises(ops.DerivedOperatorSerializationError):
        ops.load_derived_operators(bad)

    # tampered stored pass bit: recomputation refuses
    bad2 = tmp_path / "bad_pass.npz"
    _rewrite_npz(path, bad2, meta_mutator=lambda m: m["report_scalars"].update(
        {"damping_psd_pass": not m["report_scalars"]["damping_psd_pass"]}))
    with pytest.raises(ops.DerivedOperatorSerializationError):
        ops.load_derived_operators(bad2)

    # unknown npz field
    bad3 = tmp_path / "bad_field.npz"
    data = np.load(path)
    arrays = {k: data[k] for k in data.files}
    arrays["surprise"] = np.zeros(3)
    np.savez(bad3, **arrays)
    with pytest.raises(ops.DerivedOperatorSerializationError, match="unknown"):
        ops.load_derived_operators(bad3)

    # missing npz field
    bad4 = tmp_path / "missing_field.npz"
    arrays.pop("raw_damping")
    np.savez(bad4, **arrays)
    with pytest.raises(ops.DerivedOperatorSerializationError, match="missing"):
        ops.load_derived_operators(bad4)

    # format version mismatch
    bad5 = tmp_path / "bad_version.npz"
    _rewrite_npz(path, bad5, meta_mutator=lambda m: m.update(
        {"artifact_format": "anyhydro-derived-operators-v0"}))
    with pytest.raises(ops.DerivedOperatorSerializationError, match="not a"):
        ops.load_derived_operators(bad5)

    # unknown meta field
    bad6 = tmp_path / "bad_meta.npz"
    _rewrite_npz(path, bad6, meta_mutator=lambda m: m.update({"extra": 1}))
    with pytest.raises(ops.DerivedOperatorSerializationError, match="meta field"):
        ops.load_derived_operators(bad6)

    # unknown serialized settings field
    bad7 = tmp_path / "bad_settings.npz"
    _rewrite_npz(path, bad7, meta_mutator=lambda m: m["settings"].update({"dt": 0.1}))
    with pytest.raises(ops.DerivedOperatorSerializationError, match="settings"):
        ops.load_derived_operators(bad7)

    # not an npz artifact at all
    bad8 = tmp_path / "not_npz.npz"
    bad8.write_bytes(b"not a numpy file")
    with pytest.raises(ops.DerivedOperatorSerializationError):
        ops.load_derived_operators(bad8)
    with pytest.raises(ops.DerivedOperatorSerializationError, match="exist"):
        ops.load_derived_operators(tmp_path / "no_such_file.npz")
    assert json  # silence linters; json used inside _rewrite_npz


# ---------------------------------------------------------------------------
# retained-coordinate radiation


@pytest.fixture(scope="module")
def derived4_kernel(derived4):
    settings = make_derived_settings(derived4)
    return rad.radiation_kernel_from_derived_operators(derived4, settings)


def test_valid4_radiation_kernel_and_public_compare(derived4, derived4_kernel):
    kernel = derived4_kernel
    assert kernel.dof_keys == RETAINED4
    assert kernel.n_dof == 4
    assert kernel.qualification_status == "unqualified"
    prov = kernel.provenance
    assert prov["source_kind"] == "derived_operators"
    assert prov["derived_content_hash"] == derived4.content_hash()
    assert prov["source_dataset_hash"] == derived4.provenance["source_dataset_hash"]
    assert prov["derived_correction_policy"] == ops.CORRECTION_RECIPROCAL_AVERAGE
    assert any("derived_operators_source" in a for a in kernel.assumptions)

    criteria = rad.ReconstructionCriteria(
        omega_min=0.5, omega_max=8.0,
        added_mass_abs_tol=2.0, added_mass_rel_tol=1.0e-2,
        damping_abs_tol=2.0, damping_rel_tol=1.0e-2,
    )
    report = rad.compare_derived_reconstruction(kernel, derived4, criteria)
    assert report.overall_pass
    assert report.max_added_mass_error > 0.0
    assert report.max_damping_error > 0.0
    # the reference is the derived artifact itself, not the raw source
    sel = (derived4.omegas >= 0.5) & (derived4.omegas <= 8.0)
    np.testing.assert_allclose(
        report.added_mass_threshold,
        criteria.added_mass_abs_tol
        + criteria.added_mass_rel_tol * np.max(np.abs(derived4.derived_added_mass[sel]), axis=0))


def test_derived_kernel_persists_through_public_radiation_api(derived4_kernel, tmp_path):
    path = tmp_path / "derived_kernel.nc"
    rad.save_radiation_kernel(derived4_kernel, path)
    loaded = rad.load_radiation_kernel(path)
    assert loaded.content_hash() == derived4_kernel.content_hash()
    assert loaded.dof_keys == derived4_kernel.dof_keys
    assert dict(loaded.provenance) == dict(derived4_kernel.provenance)


def test_wrong_source_refusals(oracle_ds, derived4, derived4_kernel):
    # a different derived artifact (different policy, same raw source)
    other = ops.derived_operators_from_dataset(
        oracle_ds, make_settings(policy=ops.CORRECTION_NONE))
    criteria = rad.ReconstructionCriteria(
        omega_min=0.5, omega_max=8.0,
        added_mass_abs_tol=2.0, added_mass_rel_tol=1.0e-2,
        damping_abs_tol=2.0, damping_rel_tol=1.0e-2,
    )
    with pytest.raises(rad.RadiationError, match="does not match this artifact"):
        rad.compare_derived_reconstruction(derived4_kernel, other, criteria)

    # a derived artifact from a different raw source
    C, lam, a_inf = _oracle_matrices(12)
    C[0, 0] += 5.0
    other_source = ops.derived_operators_from_dataset(
        make_oracle_dataset(C=C, lam=lam, a_inf=a_inf), make_settings())
    with pytest.raises(rad.RadiationError, match="does not match this artifact"):
        rad.compare_derived_reconstruction(derived4_kernel, other_source, criteria)

    # a kernel built from the dataset path is not a derived-source kernel
    C0, lam0, _ = _oracle_matrices(12)
    full_kernel = rad.radiation_kernel_from_dataset(
        oracle_ds, make_derived_settings(derived4,
                                         zero_frequency_damping=C0 / lam0))
    with pytest.raises(rad.RadiationError, match="not built from a derived"):
        rad.compare_derived_reconstruction(full_kernel, derived4, criteria)

    with pytest.raises(rad.RadiationError, match="must be a DerivedHydroOperators"):
        rad.compare_derived_reconstruction(derived4_kernel, oracle_ds, criteria)
    with pytest.raises(rad.ReconstructionCriteriaError, match="required"):
        rad.compare_derived_reconstruction(derived4_kernel, derived4, None)


def test_derived_kernel_requires_admitted_artifact(oracle_ds):
    settings = make_settings(criteria=make_criteria(correction_abs_tol=0.0,
                                                     correction_rel_tol=0.0))
    art = ops.derived_operators_from_dataset(oracle_ds, settings)
    assert not art.admitted
    with pytest.raises(ops.DerivedOperatorAdmissionError, match="not admitted"):
        rad.radiation_kernel_from_derived_operators(art, make_derived_settings(art))
    with pytest.raises(rad.RadiationError, match="must be a DerivedHydroOperators"):
        rad.radiation_kernel_from_derived_operators(oracle_ds, make_derived_settings(art))


# ---------------------------------------------------------------------------
# old full-12 compatibility


def test_old_full12_dataset_bridge_hash_and_results(oracle_ds):
    from test_radiation import make_oracle_settings
    settings = make_oracle_settings()
    kernel = rad.radiation_kernel_from_dataset(oracle_ds, settings)
    # bit-for-bit the pre-refactor artifact hash (captured baseline)
    assert kernel.content_hash() == PRE_REFACTOR_FULL12_HASH
    # reconstruction still matches the analytic oracle
    C, lam, a_inf = _oracle_matrices(12)
    w = np.array([1.0, 2.0, 4.0, 8.0])
    a_rec, b_rec = kernel.reconstruct(w)
    b_ref = C[None] * lam[None] / (lam[None] ** 2 + w[:, None, None] ** 2)
    a_ref = a_inf[None] - C[None] / (lam[None] ** 2 + w[:, None, None] ** 2)
    assert np.max(np.abs(b_rec - b_ref)) < 2.0
    assert np.max(np.abs(a_rec - a_ref)) < 2.0


def test_public_exports_available():
    for name in (
        "DerivedHydroOperators", "DerivedOperatorSettings",
        "DerivedOperatorCriteria", "NormalizationSpec", "DerivedOperatorReport",
        "derived_operators_from_dataset", "save_derived_operators",
        "load_derived_operators", "build_projection",
        "radiation_kernel_from_derived_operators",
        "compare_derived_reconstruction", "DerivedOperatorAdmissionError",
        "CORRECTION_NONE", "CORRECTION_RECIPROCAL_AVERAGE", "CORRECTION_POLICIES",
    ):
        assert hasattr(anyhydro, name)

@pytest.mark.parametrize("refusal", ["success", "field", "metadata", "raw_source"])
def test_npz_closed_before_refusal_traceback_or_success(
    derived4, oracle_ds, tmp_path, refusal
):
    """Retained errors must not keep a private source archive open."""
    import json
    path = tmp_path / "owned.npz"
    ops.save_derived_operators(derived4, path)
    source = oracle_ds
    if refusal in ("field", "metadata"):
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        if refusal == "field":
            arrays["unexpected"] = np.zeros(1)
        else:
            meta = json.loads(str(arrays["meta"][()]))
            meta["artifact_format"] = "not-a-derived-artifact"
            arrays["meta"] = np.array(json.dumps(meta))
        np.savez(path, **arrays)
    elif refusal == "raw_source":
        source = oracle_ds.copy(deep=True)
        source["added_mass"].values[0, 0, 0] += 1.0
    if refusal == "success":
        loaded = ops.load_derived_operators(path, source_dataset=source)
        assert loaded.content_hash() == derived4.content_hash()
    else:
        with pytest.raises(ops.DerivedOperatorSerializationError) as retained:
            ops.load_derived_operators(path, source_dataset=source)
        assert retained.value.__traceback__ is not None
    # On Windows this fails while an archive file handle is retained.
    moved = tmp_path / "moved.npz"
    path.rename(moved)
    moved.unlink()
    assert not path.exists() and not moved.exists()
