"""Focused radiation preprocessing tests (development slice).

The independent exponential-kernel oracle has
    B(w) = C*lambda/(lambda^2 + w^2)
    A(w) = A_inf - C/(lambda^2 + w^2)
    K(t) = C*exp(-lambda*t)            (closed form)
    B(0) = C/lambda,  A(0) = A_inf - C/lambda^2

which exercises the exact sign convention of the reconstruction
(A_rec = A_inf - integral K sin(wt) dt / w; never the flipped sign) and the
explicit zero-frequency limit. These are development tests: they confer no
scientific qualification.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

import anyhydro
from anyhydro import radiation as rad
from anyhydro.errors import SchemaError, TimeDomainAdmissionError
from anyhydro.fixtures import make_fixture_bodies
from anyhydro.hashing import dataset_content_hash
from anyhydro.records import HydroRequest
from anyhydro.schema import assemble_hydro_dataset

# Valid discretization regime for the 513-frequency development cap:
# kernel duration * frequency spacing ~= 1 (the spacing-vs-duration
# diagnostic), omega_max*dt < pi, and T >> 1/lambda.
ORACLE_OMEGAS = np.linspace(0.05, 50.0, 513)
ORACLE_DT = 0.01
ORACLE_N_TIME = 1001  # duration 10 s >> 1/lambda; 10 * domega ~= 0.98


def _oracle_matrices(n_dof: int):
    """Full coupled C, lambda, A_inf with nonzero cross-body blocks."""
    rng_rows = np.arange(n_dof)
    C = np.empty((n_dof, n_dof))
    lam = np.empty((n_dof, n_dof))
    for i in range(n_dof):
        for j in range(n_dof):
            same_body = (i // 6) == (j // 6)
            if i == j:
                C[i, j] = 800.0 + 40.0 * (i % 6)
                lam[i, j] = 1.0 + 0.1 * (i % 6)
            elif same_body:
                C[i, j] = 60.0 + 5.0 * ((i * 7 + j * 3) % 11)
                lam[i, j] = 0.9 + 0.05 * ((i + j) % 9)
            else:
                C[i, j] = 25.0 + 4.0 * ((i * 5 + j * 11) % 7)
                lam[i, j] = 1.1 + 0.04 * ((i * 3 + j * 7) % 5)
    a_inf = 2.0 * C / lam**2 + 50.0  # keeps A(w) positive; distinct from any A(w)
    return C, lam, a_inf


def make_oracle_dataset(
    omegas=ORACLE_OMEGAS,
    n_dof: int = 12,
    qualification_status: str = "development_fixture",
    C=None,
    lam=None,
    a_inf=None,
    bodies=None,
):
    if bodies is None:
        bodies = make_fixture_bodies()[: n_dof // 6]
    else:
        assert 6 * len(bodies) == n_dof
    if C is None or lam is None or a_inf is None:
        C, lam, a_inf = _oracle_matrices(n_dof)
    request = HydroRequest(
        omegas=np.asarray(omegas, dtype=np.float64),
        wave_directions=np.array([0.0]),
        water_depth=150.0,
        rho=1025.0,
        gravity=9.81,
        include_zero_frequency=False,
        include_infinite_frequency=False,
        backend_settings={"fixture": "radiation-exponential-oracle"},
    )
    n_w = request.omegas.shape[0]
    w2 = request.omegas[:, None, None] ** 2
    B = C[None] * lam[None] / (lam[None] ** 2 + w2)
    A = a_inf[None] - C[None] / (lam[None] ** 2 + w2)
    excitation = np.zeros((n_w, 1, n_dof), dtype=np.complex128)
    stiffness = np.zeros((n_dof, n_dof))
    centers = np.array([b.rotation_center for b in bodies], dtype=np.float64)
    coms = np.array([b.center_of_mass for b in bodies], dtype=np.float64)
    return assemble_hydro_dataset(
        bodies,
        request,
        added_mass=A,
        radiation_damping=B,
        excitation_force=excitation,
        hydrostatic_stiffness=stiffness,
        rotation_center=centers,
        center_of_mass=coms,
        center_of_buoyancy=centers,
        added_mass_infinite=a_inf,
        added_mass_infinite_source="test oracle: analytic exponential kernel, not a backend solve",
        backend="exponential-oracle",
        qualification_status=qualification_status,
    )


def make_oracle_settings(n_dof: int = 12, **overrides):
    C, lam, _ = _oracle_matrices(n_dof)
    defaults = dict(
        dt=ORACLE_DT,
        n_time=ORACLE_N_TIME,
        symmetry_tol=1.0e-9,
        passivity_tol=1.0e-9,
        low_tail_policy=rad.LOW_TAIL_EXPLICIT_ZERO,
        zero_frequency_damping=C / lam,  # oracle B(0)
    )
    defaults.update(overrides)
    return rad.RadiationSettings(**defaults)


@pytest.fixture(scope="module")
def oracle():
    ds = make_oracle_dataset()
    settings = make_oracle_settings()
    kernel = rad.radiation_kernel_from_dataset(ds, settings)
    C, lam, a_inf = _oracle_matrices(12)
    return ds, settings, kernel, C, lam, a_inf


# ---------------------------------------------------------------------------
# forward kernel and reconstruction against the analytic oracle


def test_kernel_matches_analytic_exponential(oracle):
    _, _, kernel, C, lam, _ = oracle
    t = kernel.time
    analytic = C[None] * np.exp(-lam[None] * t[:, None, None])
    err = np.abs(kernel.kernel - analytic)
    # the only visible error is the explicit omega_max tail truncation, which
    # is absolute-small but relative-huge at late t (analytic ~ 1e-13 there):
    # compare on the kernel's own scale C = K(0)
    assert np.max(err / C[None]) < 5.0e-2
    rel_t0 = err[0] / analytic[0]
    assert np.max(rel_t0) < 5.0e-2


def test_reconstruction_recovers_both_matrices(oracle):
    ds, _, kernel, C, lam, a_inf = oracle
    w = np.array([1.0, 2.0, 4.0, 8.0])
    a_rec, b_rec = kernel.reconstruct(w)
    b_ref = C[None] * lam[None] / (lam[None] ** 2 + w[:, None, None] ** 2)
    a_ref = a_inf[None] - C[None] / (lam[None] ** 2 + w[:, None, None] ** 2)
    assert np.max(np.abs(b_rec - b_ref)) < 2.0
    assert np.max(np.abs(a_rec - a_ref)) < 2.0


def test_added_mass_sign_convention_is_not_flipped(oracle):
    _, _, kernel, C, lam, a_inf = oracle
    w = np.array([1.0])
    a_rec, _ = kernel.reconstruct(w)
    correct = a_inf - C / (lam**2 + w[0] ** 2)
    flipped = a_inf + C / (lam**2 + w[0] ** 2)
    assert np.allclose(a_rec[0], correct, rtol=1.0e-2)
    assert not np.allclose(a_rec[0], flipped, rtol=5.0e-2)


def _sampled_analytic_kernel(oracle):
    """RadiationKernel whose kernel is the analytically sampled exponential.

    Unlike the bridge kernel (whose omega_max band truncation and finite
    duration are real, documented error sources), this artifact carries the
    exact closed form ``K(t) = C*exp(-lambda*t)`` on the same time grid, so
    the zero-limit formula is verified against the independent analytic
    reference alone.
    """
    _, settings, kernel, C, lam, a_inf = oracle
    t = kernel.time
    analytic = C[None] * np.exp(-lam[None] * t[:, None, None])
    return rad.RadiationKernel(
        settings=settings,
        time=t,
        dof_keys=kernel.dof_keys,
        body_ids=kernel.body_ids,
        kernel=analytic,
        added_mass_infinite=a_inf,
        diagnostics=kernel.diagnostics,
        provenance=dict(kernel.provenance),
        assumptions=kernel.assumptions,
    )


def test_zero_frequency_limit_strict_on_sampled_exponential_kernel(oracle):
    # Original strict criterion, restored: the explicit w=0 limit
    # A(0) = A_inf - integral t*K(t) dt must hold to 1% on the analytically
    # known sampled exponential kernel. Only trapezoid quadrature and the
    # 10 s grid tail contribute (observed worst ~1e-3), so this verifies
    # the actual formula against an independent reference.
    _, _, kernel, C, lam, a_inf = oracle
    sampled = _sampled_analytic_kernel(oracle)
    a_rec, b_rec = sampled.reconstruct(np.array([0.0, 0.0]))  # repeated zero queries
    assert a_rec.shape == (2, 12, 12) and b_rec.shape == a_rec.shape
    np.testing.assert_allclose(a_rec[0], a_inf - C / lam**2, rtol=1.0e-2)
    np.testing.assert_allclose(a_rec[1], a_inf - C / lam**2, rtol=1.0e-2)
    np.testing.assert_allclose(b_rec[0], C / lam, rtol=1.0e-2)


def test_zero_frequency_limit_finite_band_discrepancy_is_preserved(oracle):
    # The bridge kernel is truncated at omega_max and its duration is capped
    # by the spacing-vs-duration criterion, so its w=0 limit carries a real
    # few-percent truncation error (observed worst ~6% at check-a2). The
    # discrepancy is reported here; it must NOT be admitted by loosening the
    # strict criterion above, and the strict criterion intentionally fails.
    _, _, kernel, C, lam, a_inf = oracle
    a_rec, b_rec = kernel.reconstruct(np.array([0.0]))
    a0_ref = a_inf - C / lam**2
    rel_err = np.abs(a_rec[0] - a0_ref) / np.abs(a0_ref)
    assert not np.allclose(a_rec[0], a0_ref, rtol=1.0e-2)
    assert np.max(rel_err) < 0.15  # gross sanity bound; documentation, not admission


def test_compare_reconstruction_passes_declared_criteria(oracle):
    ds, _, kernel, *_ = oracle
    criteria = rad.ReconstructionCriteria(
        omega_min=0.5,
        omega_max=8.0,
        added_mass_abs_tol=2.0,
        added_mass_rel_tol=1.0e-2,
        damping_abs_tol=2.0,
        damping_rel_tol=1.0e-2,
    )
    report = rad.compare_reconstruction(kernel, ds, criteria)
    assert report.overall_pass
    assert report.max_added_mass_error > 0.0  # observed errors are reported, not hidden
    assert report.max_damping_error > 0.0


def test_threshold_formula_and_near_zero_absolute_allowance():
    # cross entries with C=0 and A_inf = C/lambda^2 give near-zero references:
    # they must pass through the absolute allowance, not a relative one.
    n_dof = 12
    C, lam, a_inf = _oracle_matrices(n_dof)
    C[:6, 6:] = 0.0
    C[6:, :6] = 0.0
    a_inf[:6, 6:] = 0.0
    a_inf[6:, :6] = 0.0
    ds = make_oracle_dataset(C=C, lam=lam, a_inf=a_inf)
    settings = make_oracle_settings(zero_frequency_damping=C / lam)
    kernel = rad.radiation_kernel_from_dataset(ds, settings)
    criteria = rad.ReconstructionCriteria(
        omega_min=0.5,
        omega_max=8.0,
        added_mass_abs_tol=2.0,
        added_mass_rel_tol=1.0e-2,
        damping_abs_tol=2.0,
        damping_rel_tol=1.0e-2,
    )
    report = rad.compare_reconstruction(kernel, ds, criteria)
    assert report.overall_pass
    # the zeroed cross-body block passes purely through the absolute allowance
    assert np.all(report.added_mass_threshold[:6, 6:] == criteria.added_mass_abs_tol)
    assert np.all(report.damping_threshold[:6, 6:] == criteria.damping_abs_tol)

    w = ds["omega"].values
    sel = (w >= criteria.omega_min) & (w <= criteria.omega_max)
    a_ref = ds["added_mass"].values[sel]
    b_ref = ds["radiation_damping"].values[sel]
    expected_a_thr = criteria.added_mass_abs_tol + criteria.added_mass_rel_tol * np.max(np.abs(a_ref), axis=0)
    expected_b_thr = criteria.damping_abs_tol + criteria.damping_rel_tol * np.max(np.abs(b_ref), axis=0)
    assert np.allclose(report.added_mass_threshold, expected_a_thr)
    assert np.allclose(report.damping_threshold, expected_b_thr)


def test_no_admission_on_damping_alone(oracle):
    ds, _, kernel, *_ = oracle
    criteria = rad.ReconstructionCriteria(
        omega_min=0.1,
        omega_max=8.0,
        added_mass_abs_tol=0.0,
        added_mass_rel_tol=0.0,  # exact: guaranteed to fail
        damping_abs_tol=1.0e6,
        damping_rel_tol=1.0e6,
    )
    report = rad.compare_reconstruction(kernel, ds, criteria)
    assert report.summary()["damping_pass"]
    assert not report.summary()["added_mass_pass"]
    assert not report.overall_pass


# ---------------------------------------------------------------------------
# admission, refusals and limits


def test_missing_a_inf_is_refused(two_body_dataset_no_a_inf):
    settings = make_oracle_settings()
    with pytest.raises(TimeDomainAdmissionError):
        rad.radiation_kernel_from_dataset(two_body_dataset_no_a_inf, settings)


def test_default_policy_requires_explicit_zero_sample():
    with pytest.raises(rad.RadiationSettingsError, match="zero_frequency_damping"):
        rad.RadiationSettings(dt=0.05, n_time=101, symmetry_tol=1e-9, passivity_tol=1e-9)


def test_zero_sample_shape_checked_at_bridge(two_body_dataset):
    settings = rad.RadiationSettings(
        dt=0.05, n_time=101, symmetry_tol=1e-9, passivity_tol=1e-9,
        zero_frequency_damping=np.zeros((6, 6)),
    )
    with pytest.raises(rad.RadiationSettingsError, match="shape"):
        rad.radiation_kernel_from_dataset(two_body_dataset, settings)


def test_linear_to_zero_extension_recorded_as_assumption():
    ds = make_oracle_dataset(omegas=np.linspace(0.1, 60.0, 121))
    settings = rad.RadiationSettings(
        dt=ORACLE_DT, n_time=201, symmetry_tol=1e-9, passivity_tol=1e-9,
        low_tail_policy=rad.LOW_TAIL_LINEAR_TO_ZERO,
        extension_name="dev-linear-to-zero-low-tail",
    )
    kernel = rad.radiation_kernel_from_dataset(ds, settings)
    assert np.all(kernel.diagnostics.zero_frequency_damping == 0.0)
    assert any("dev-linear-to-zero-low-tail" in a for a in kernel.assumptions)
    assert any("B(0)=0" in a for a in kernel.assumptions)


def test_linear_extension_requires_separate_name():
    with pytest.raises(rad.RadiationSettingsError, match="separately named"):
        rad.RadiationSettings(
            dt=0.05, n_time=101, symmetry_tol=1e-9, passivity_tol=1e-9,
            low_tail_policy=rad.LOW_TAIL_LINEAR_TO_ZERO,
        )
    with pytest.raises(rad.RadiationSettingsError, match="must not be supplied"):
        rad.RadiationSettings(
            dt=0.05, n_time=101, symmetry_tol=1e-9, passivity_tol=1e-9,
            low_tail_policy=rad.LOW_TAIL_LINEAR_TO_ZERO,
            extension_name="x",
            zero_frequency_damping=np.zeros((12, 12)),
        )


def test_refuses_non_finite_inputs(two_body_dataset):
    bad_b0 = np.zeros((12, 12))
    bad_b0[0, 0] = np.nan
    with pytest.raises(rad.RadiationSettingsError, match="non-finite"):
        rad.RadiationSettings(
            dt=0.05, n_time=101, symmetry_tol=1e-9, passivity_tol=1e-9,
            zero_frequency_damping=bad_b0,
        )
    broken = two_body_dataset.copy(deep=True)
    broken["radiation_damping"].values[0, 0, 0] = np.inf
    settings = make_oracle_settings()
    with pytest.raises(SchemaError):
        rad.radiation_kernel_from_dataset(broken, settings)


def test_frequency_grid_needs_three_samples():
    ds = make_oracle_dataset(omegas=np.array([0.5, 1.5]))
    settings = make_oracle_settings()
    with pytest.raises(rad.RadiationError, match="at least 3"):
        rad.radiation_kernel_from_dataset(ds, settings)


def test_frequency_limit_refused():
    ds = make_oracle_dataset(omegas=np.linspace(0.1, 50.0, 514))
    settings = make_oracle_settings()
    with pytest.raises(rad.RadiationLimitError, match="513"):
        rad.radiation_kernel_from_dataset(ds, settings)


def test_time_sample_limit_refused():
    with pytest.raises(rad.RadiationLimitError, match="1001"):
        rad.RadiationSettings(
            dt=0.01, n_time=1002, symmetry_tol=1e-9, passivity_tol=1e-9,
            zero_frequency_damping=np.zeros((12, 12)),
        )


def test_dof_limit_refused():
    from anyhydro.dof import body_dof_keys
    from anyhydro.fixtures import _box_mesh
    from anyhydro.records import HydroBody

    gamma_mesh = _box_mesh("fixture-body-gamma", (24.0, 0.0, -0.5), (1.0, 0.6, 0.8))
    gamma = HydroBody(
        body_id="fixture-body-gamma",
        name="synthetic fixture body gamma",
        mesh=gamma_mesh,
        lid_mesh=None,
        rotation_center=(24.0, 0.0, -0.5),
        center_of_mass=(24.0, 0.0, -0.1),
        enabled_dofs=body_dof_keys("fixture-body-gamma"),
    )
    bodies = (*make_fixture_bodies(), gamma)
    n_dof = 18
    C, lam, a_inf = _oracle_matrices(n_dof)
    ds = make_oracle_dataset(
        omegas=np.linspace(0.1, 40.0, 41), n_dof=n_dof, bodies=bodies,
        C=C, lam=lam, a_inf=a_inf,
    )
    settings = rad.RadiationSettings(
        dt=0.05, n_time=101, symmetry_tol=1e-9, passivity_tol=1e-9,
        zero_frequency_damping=C / lam,
    )
    with pytest.raises(rad.RadiationLimitError, match="development limit of 12"):
        rad.radiation_kernel_from_dataset(ds, settings)


def test_reconstruct_rejects_bad_frequencies(oracle):
    _, _, kernel, *_ = oracle
    with pytest.raises(rad.RadiationError, match=">= 0"):
        kernel.reconstruct(np.array([-1.0]))
    with pytest.raises(rad.RadiationError, match="finite"):
        kernel.reconstruct(np.array([np.nan]))


# ---------------------------------------------------------------------------
# diagnostics: symmetry, passivity, tail (errors preserved, never clamped)


def test_symmetry_diagnostics_preserve_violations():
    n_dof = 12
    C, lam, a_inf = _oracle_matrices(n_dof)
    C[0, 1] += 30.0  # asymmetric damping by construction
    ds = make_oracle_dataset(C=C, lam=lam, a_inf=a_inf)
    settings = make_oracle_settings(zero_frequency_damping=C / lam)
    kernel = rad.radiation_kernel_from_dataset(ds, settings)
    d = kernel.diagnostics
    assert d.max_damping_asymmetry > 1.0
    assert d.damping_symmetry_violations >= 1
    # preserved, not clamped: the kernel itself remains asymmetric
    asym = np.abs(kernel.kernel[1] - kernel.kernel[1].T)
    assert asym.max() > 0.0


def test_passivity_diagnostics_preserve_violations():
    n_dof = 12
    C, lam, a_inf = _oracle_matrices(n_dof)
    C[0, 0] = -400.0  # negative diagonal damping: non-passive by construction
    ds = make_oracle_dataset(C=C, lam=lam, a_inf=a_inf)
    settings = make_oracle_settings(zero_frequency_damping=C / lam)
    kernel = rad.radiation_kernel_from_dataset(ds, settings)
    d = kernel.diagnostics
    assert d.min_damping_eigenvalue < 0.0
    assert d.passivity_violations >= 1


def test_tail_diagnostics(oracle):
    ds, settings, kernel, *_ = oracle
    d = kernel.diagnostics
    b_last = ds["radiation_damping"].values[-1]
    assert np.allclose(d.damping_at_omega_max, b_last)
    assert d.omega_max == pytest.approx(ds["omega"].values[-1])
    assert d.omega_max_times_dt == pytest.approx(ds["omega"].values[-1] * settings.dt)
    assert d.max_frequency_spacing > 0.0
    assert d.duration_over_spacing_period == pytest.approx(
        d.kernel_duration * d.max_frequency_spacing / (2.0 * np.pi)
    )


def test_explicit_time_window_zeroes_kernel_tail():
    ds = make_oracle_dataset(omegas=np.linspace(0.1, 60.0, 121))
    settings = rad.RadiationSettings(
        dt=0.05, n_time=401, symmetry_tol=1e-9, passivity_tol=1e-9,
        zero_frequency_damping=np.zeros((12, 12)),
        time_window=10.0,
    )
    kernel = rad.radiation_kernel_from_dataset(ds, settings)
    assert np.all(kernel.kernel[kernel.time > 10.0] == 0.0)
    assert np.any(kernel.kernel[kernel.time <= 10.0] != 0.0)
    assert any("time_window=10.0" in a for a in kernel.assumptions)


# ---------------------------------------------------------------------------
# immutability, provenance and serialization


def test_artifact_is_immutable_with_defensive_copies(oracle):
    ds, settings, kernel, *_ = oracle
    with pytest.raises(ValueError):
        kernel.kernel[0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        kernel.added_mass_infinite[0, 0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        kernel.assumptions = ()
    with pytest.raises(ValueError):
        settings.zero_frequency_damping[0, 0] = 1.0

    # defensive copy: mutating caller inputs after construction changes nothing
    b0 = np.full((12, 12), 7.0)
    s2 = rad.RadiationSettings(
        dt=0.05, n_time=101, symmetry_tol=1e-9, passivity_tol=1e-9,
        zero_frequency_damping=b0,
    )
    b0[0, 0] = 99.0
    assert s2.zero_frequency_damping[0, 0] == 7.0

    ds_mut = ds.copy(deep=True)
    before = np.array(kernel.kernel[0])
    ds_mut["radiation_damping"].values[:] = 0.0
    assert np.array_equal(kernel.kernel[0], before)


def test_provenance_is_preserved_and_status_not_inherited():
    ds = make_oracle_dataset(qualification_status="qualified")
    settings = make_oracle_settings()
    kernel = rad.radiation_kernel_from_dataset(ds, settings)
    assert kernel.provenance["source_dataset_hash"] == dataset_content_hash(ds)
    assert kernel.provenance["source_qualification_status"] == "qualified"
    assert kernel.provenance["a_inf_source"].startswith("test oracle")
    assert kernel.provenance["phasor_convention"] == anyhydro.schema.PHASOR_CONVENTION
    assert kernel.qualification_status == "unqualified"  # never inherited
    assert any("cannot inherit qualified" in a for a in kernel.assumptions)


def test_serialization_roundtrip_reproducible_and_verified(oracle, tmp_path):
    _, _, kernel, *_ = oracle
    p1 = tmp_path / "kernel_a.nc"
    p2 = tmp_path / "kernel_b.nc"
    rad.save_radiation_kernel(kernel, p1)
    rad.save_radiation_kernel(kernel, p2)
    assert p1.read_bytes() == p2.read_bytes()  # deterministic bytes, no timestamps

    loaded = rad.load_radiation_kernel(p1)
    assert loaded.content_hash() == kernel.content_hash()
    assert np.array_equal(loaded.kernel, kernel.kernel)
    assert loaded.dof_keys == kernel.dof_keys
    assert dict(loaded.provenance) == dict(kernel.provenance)
    assert loaded.assumptions == kernel.assumptions
    assert loaded.qualification_status == "unqualified"

    a1, b1 = kernel.reconstruct(np.array([0.5, 2.0]))
    a2, b2 = loaded.reconstruct(np.array([0.5, 2.0]))
    assert np.array_equal(a1, a2) and np.array_equal(b1, b2)


def test_serialization_integrity_check_rejects_tampering(oracle, tmp_path):
    import xarray as xr

    _, _, kernel, *_ = oracle
    path = tmp_path / "kernel.nc"
    rad.save_radiation_kernel(kernel, path)
    tampered = xr.load_dataset(path, engine="scipy")
    tampered["kernel"].values[0, 0, 0] += 1.0
    tampered.to_netcdf(tmp_path / "tampered.nc", engine="scipy", format="NETCDF3_64BIT")
    with pytest.raises(rad.RadiationSerializationError, match="integrity"):
        rad.load_radiation_kernel(tmp_path / "tampered.nc")


def test_compare_reconstruction_is_source_bound(oracle, two_body_dataset):
    _, _, kernel, *_ = oracle
    criteria = rad.ReconstructionCriteria(
        omega_min=0.1, omega_max=8.0,
        added_mass_abs_tol=1.0, added_mass_rel_tol=1e-2,
        damping_abs_tol=1.0, damping_rel_tol=1e-2,
    )
    with pytest.raises(rad.RadiationError, match="does not match"):
        rad.compare_reconstruction(kernel, two_body_dataset, criteria)
    with pytest.raises(rad.ReconstructionCriteriaError, match="required"):
        rad.compare_reconstruction(kernel, oracle[0], None)


def test_window_must_contain_reference_frequencies(oracle):
    ds, _, kernel, *_ = oracle
    criteria = rad.ReconstructionCriteria(
        omega_min=1.0e5, omega_max=2.0e5,
        added_mass_abs_tol=1.0, added_mass_rel_tol=1e-2,
        damping_abs_tol=1.0, damping_rel_tol=1e-2,
    )
    with pytest.raises(rad.ReconstructionCriteriaError, match="no reference grid"):
        rad.compare_reconstruction(kernel, ds, criteria)


def test_public_exports_available():
    for name in (
        "RadiationSettings", "ReconstructionCriteria", "RadiationKernel",
        "RadiationDiagnostics", "ReconstructionReport",
        "radiation_kernel_from_dataset", "compare_reconstruction",
        "save_radiation_kernel", "load_radiation_kernel",
        "radiation_kernel_hash", "RadiationError",
    ):
        assert hasattr(anyhydro, name)
