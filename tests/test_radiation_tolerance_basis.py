"""Declared per-sample relative tolerance basis for the radiation kernel diagnostics (slice F2a).

The default basis is absolute (SI) and must stay bit-identical to the parent: settings
dicts, diagnostics scalars, persisted files and kernel content hashes. The optional
``relative_to_sample_norm`` basis judges every checked damping / added-mass sample against
the spectral norm of that sample itself (a large sample cannot mask a violation in a smaller
one); violations are still counted and reported, never clamped or repaired. The recorded
violation counts are bound to the declared basis and tolerances (fail closed).
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import xarray as xr

from anyhydro import radiation as rad
from anyhydro.fixtures import make_fixture_bodies
from anyhydro.records import HydroRequest
from anyhydro.schema import assemble_hydro_dataset

from test_radiation import ORACLE_OMEGAS, make_oracle_dataset, make_oracle_settings

N_DOF = 12
SCALE = 1.0e8
LAM = 2.0
RATIOS = ("min_damping_eigenvalue_ratio", "max_damping_asymmetry_ratio",
          "max_added_mass_asymmetry_ratio")
RELATIVE = "relative_to_sample_norm"


def _c_matrix(min_eig_rel=-1.0e-12, asym_rel=1.0e-13):
    """Symmetric C (to round-off) with eigenvalues SCALE*d, smallest min_eig_rel."""
    rng = np.random.default_rng(7)
    q, _ = np.linalg.qr(rng.standard_normal((N_DOF, N_DOF)))
    d = np.linspace(1.0, 0.1, N_DOF)
    d[-1] = min_eig_rel
    c = SCALE * (q * d) @ q.T
    c = 0.5 * (c + c.T)
    c[0, 1] += asym_rel * SCALE
    return c


def _build(c, **overrides):
    lam = np.full((N_DOF, N_DOF), LAM)
    a_inf = 2.0 * c / lam**2 + 50.0
    ds = make_oracle_dataset(C=c, lam=lam, a_inf=a_inf)
    settings = make_oracle_settings(zero_frequency_damping=c / lam, **overrides)
    return ds, settings, rad.radiation_kernel_from_dataset(ds, settings)


REL = dict(tolerance_basis=RELATIVE, symmetry_tol=1.0e-2, passivity_tol=1.0e-9)


def _kernel_from_samples(added, damping, b0, **settings_overrides):
    """Kernel from explicit (non-proportional) added mass / damping samples on the oracle grid."""
    n_w = ORACLE_OMEGAS.size
    assert added.shape == damping.shape == (n_w, N_DOF, N_DOF)
    bodies = make_fixture_bodies()[: N_DOF // 6]
    request = HydroRequest(
        omegas=np.asarray(ORACLE_OMEGAS, dtype=np.float64), wave_directions=np.array([0.0]),
        water_depth=150.0, rho=1025.0, gravity=9.81, include_zero_frequency=False,
        include_infinite_frequency=False, backend_settings={"fixture": "tolerance-basis"})
    centers = np.array([b.rotation_center for b in bodies], dtype=np.float64)
    coms = np.array([b.center_of_mass for b in bodies], dtype=np.float64)
    a_inf = np.eye(N_DOF) * 50.0
    ds = assemble_hydro_dataset(
        bodies, request, added_mass=added, radiation_damping=damping,
        excitation_force=np.zeros((n_w, 1, N_DOF), dtype=np.complex128),
        hydrostatic_stiffness=np.zeros((N_DOF, N_DOF)), rotation_center=centers,
        center_of_mass=coms, center_of_buoyancy=centers, added_mass_infinite=a_inf,
        added_mass_infinite_source="test fixture, not a backend solve", backend="tolerance-basis-fixture",
        qualification_status="development_fixture")
    settings = make_oracle_settings(zero_frequency_damping=b0, **settings_overrides)
    return rad.radiation_kernel_from_dataset(ds, settings)


def _non_proportional(big_added=None, big_damping=None, small_damping=None, small_added=None,
                      small=1.0e2, big=1.0e8):
    """Sample 0 is huge (SCALE-sized identity), every other sample ``small``-sized."""
    n_w = ORACLE_OMEGAS.size
    eye = np.eye(N_DOF)
    damping = np.broadcast_to(small * eye, (n_w, N_DOF, N_DOF)).copy()
    added = np.broadcast_to(50.0 * eye, (n_w, N_DOF, N_DOF)).copy()
    damping[0] = big * eye
    added[0] = big * eye
    if small_damping is not None:
        damping[1] = small_damping
    if small_added is not None:
        added[1] = small_added
    return added, damping, small * eye


# --- a. absolute default is unchanged -------------------------------------

def test_absolute_default_serialization_and_hash_unchanged():
    s = make_oracle_settings()
    assert s.tolerance_basis == rad.TOLERANCE_BASIS_ABSOLUTE == "absolute"
    assert "tolerance_basis" not in s.to_dict()
    back = rad.RadiationSettings.from_dict(s.to_dict())
    assert back.to_dict() == s.to_dict() and back.tolerance_basis == "absolute"
    ds, _, k_default = _build(_c_matrix())
    _, _, k_explicit = _build(_c_matrix(), tolerance_basis="absolute")
    assert k_default.content_hash() == k_explicit.content_hash()
    assert k_default.settings.to_dict() == k_explicit.settings.to_dict()
    scalars = k_default.diagnostics.scalars()
    assert not set(RATIOS) & set(scalars)
    for name in RATIOS:
        assert getattr(k_default.diagnostics, name) is None


# --- a2. parent golden: byte-for-byte legacy compatibility ------------------

# Produced by the parent commit d7a9d46 (git worktree, PYTHONPATH at its src) from the kernel built in
# ``test_absolute_matches_parent_golden``; the branch must reproduce them exactly.
PARENT_GOLDEN_HASH = "1fe67584333f27a797e8303013924d62557e3f4769911dff3c893f8ecff8e3e8"
PARENT_GOLDEN_SCALARS = {
    "added_mass_symmetry_violations": 1, "damping_symmetry_violations": 1,
    "duration_over_spacing_period": 0.15526932436406812, "kernel_duration": 10.0,
    "max_abs_damping_at_omega_max": 70796.96417941595,
    "max_added_mass_asymmetry": 4.996021743863821e-06, "max_damping_asymmetry": 5.0000089686363935e-06,
    "max_frequency_spacing": 0.09755859375000142, "min_damping_eigenvalue": -4.982959944754839e-05,
    "omega_max": 50.0, "omega_max_times_dt": 0.5, "passivity_violations": 144,
}
PARENT_GOLDEN_SETTINGS_SCALARS = {
    "dt": 0.01, "extension_name": None, "low_tail_policy": "explicit_zero_sample", "n_time": 1001,
    "passivity_tol": 1e-06, "symmetry_tol": 1e-06, "time_window": None,
}


def test_absolute_matches_parent_golden(tmp_path):
    _, _, k = _build(_c_matrix(), symmetry_tol=1.0e-6, passivity_tol=1.0e-6)
    assert k.diagnostics.scalars() == PARENT_GOLDEN_SCALARS
    settings_dict = k.settings.to_dict()
    assert {key: v for key, v in settings_dict.items() if key != "zero_frequency_damping"} \
        == PARENT_GOLDEN_SETTINGS_SCALARS
    assert k.content_hash() == PARENT_GOLDEN_HASH
    # persistence path: the branch-written absolute file carries the parent hash and loads back to it
    path = tmp_path / "abs.nc"
    rad.save_radiation_kernel(k, path)
    loaded = rad.load_radiation_kernel(path)
    assert loaded.content_hash() == PARENT_GOLDEN_HASH
    assert loaded.diagnostics.scalars() == PARENT_GOLDEN_SCALARS
    attrs = xr.load_dataset(path, engine="scipy").attrs
    assert json.loads(attrs["diagnostics_scalars"]) == PARENT_GOLDEN_SCALARS
    assert "tolerance_basis" not in json.loads(attrs["settings"])


# --- b. relative basis ----------------------------------------------------

def test_absolute_flags_roundoff_but_relative_does_not():
    c = _c_matrix()
    _, _, k_abs = _build(c, symmetry_tol=1.0e-6, passivity_tol=1.0e-6)
    da = k_abs.diagnostics
    assert da.passivity_violations > 0
    assert da.damping_symmetry_violations > 0 or da.added_mass_symmetry_violations > 0

    _, settings, k_rel = _build(c, **REL)
    d = k_rel.diagnostics
    assert settings.to_dict()["tolerance_basis"] == RELATIVE
    assert d.passivity_violations == 0
    assert d.damping_symmetry_violations == 0
    assert d.added_mass_symmetry_violations == 0
    assert d.min_damping_eigenvalue < 0.0  # reported, not repaired
    assert d.min_damping_eigenvalue_ratio == pytest.approx(-1.0e-12, abs=2.0e-13)
    assert 0.0 <= d.max_damping_asymmetry_ratio < 1.0e-2
    assert 0.0 <= d.max_added_mass_asymmetry_ratio < 1.0e-2
    assert set(RATIOS) <= set(d.scalars())
    # the kernel content differs from the absolute one only by declared settings/diagnostics
    assert k_rel.content_hash() != k_abs.content_hash()


def test_relative_still_flags_real_passivity_violation():
    _, _, k = _build(_c_matrix(min_eig_rel=-1.0e-6), **REL)
    assert k.diagnostics.passivity_violations > 0
    assert k.diagnostics.min_damping_eigenvalue_ratio < -1.0e-9


def test_relative_still_flags_real_asymmetry():
    _, _, k = _build(_c_matrix(asym_rel=1.0e-1), **REL)
    d = k.diagnostics
    assert d.damping_symmetry_violations > 0
    assert d.max_damping_asymmetry_ratio > 1.0e-2


# --- b2. a large sample cannot mask a violation in a smaller one --------------

def test_large_sample_does_not_mask_passivity_violation():
    small_damping = 1.0e2 * np.eye(N_DOF)
    small_damping[-1, -1] = -1.0e-3 * 1.0e2  # smallest eigenvalue -1e-3 x own norm
    added, damping, b0 = _non_proportional(small_damping=small_damping)
    k = _kernel_from_samples(added, damping, b0, tolerance_basis=RELATIVE, symmetry_tol=1.0e-2,
                             passivity_tol=1.0e-8)
    d = k.diagnostics
    assert d.passivity_violations >= 1
    assert d.min_damping_eigenvalue_ratio == pytest.approx(-1.0e-3, rel=1.0e-6)
    # the same data judged against the largest sample (the rejected basis) would be admitted
    assert -d.min_damping_eigenvalue < 1.0e-8 * 1.0e8


def test_large_sample_does_not_mask_damping_asymmetry():
    small_damping = 1.0e2 * np.eye(N_DOF)
    small_damping[0, 1] = 1.0e-1 * 1.0e2  # |B01 - B10| = 1e-1 x own norm
    added, damping, b0 = _non_proportional(small_damping=small_damping)
    k = _kernel_from_samples(added, damping, b0, tolerance_basis=RELATIVE, symmetry_tol=1.0e-2,
                             passivity_tol=1.0e-8)
    d = k.diagnostics
    assert d.damping_symmetry_violations >= 1
    assert d.max_damping_asymmetry_ratio == pytest.approx(10.0 / np.linalg.norm(small_damping, 2), rel=1.0e-9)
    assert d.added_mass_symmetry_violations == 0
    assert 1.0e-1 * 1.0e2 < 1.0e-2 * 1.0e8  # the largest-sample tolerance would admit it


def test_large_sample_does_not_mask_added_mass_asymmetry():
    small_added = 50.0 * np.eye(N_DOF)
    small_added[2, 3] = 1.0e-1 * 50.0
    added, damping, b0 = _non_proportional(small_added=small_added)
    k = _kernel_from_samples(added, damping, b0, tolerance_basis=RELATIVE, symmetry_tol=1.0e-2,
                             passivity_tol=1.0e-8)
    d = k.diagnostics
    assert d.added_mass_symmetry_violations >= 1
    assert d.max_added_mass_asymmetry_ratio == pytest.approx(5.0 / np.linalg.norm(small_added, 2), rel=1.0e-9)
    assert d.damping_symmetry_violations == 0 and d.passivity_violations == 0


def test_large_scale_roundoff_is_admitted_with_recorded_ratios():
    big = 1.0e8
    sample = big * np.eye(N_DOF)
    sample[-1, -1] = -1.0e-12 * big
    sample[0, 1] += 1.0e-6 * big
    added = np.broadcast_to(sample, (ORACLE_OMEGAS.size, N_DOF, N_DOF)).copy()
    damping = added.copy()
    added[:, -1, -1] = big  # added mass: asymmetry only
    k = _kernel_from_samples(added, damping, sample, tolerance_basis=RELATIVE, symmetry_tol=1.0e-2,
                             passivity_tol=1.0e-8)
    d = k.diagnostics
    assert (d.passivity_violations, d.damping_symmetry_violations, d.added_mass_symmetry_violations) == (0, 0, 0)
    assert d.min_damping_eigenvalue_ratio == pytest.approx(-1.0e-12, abs=1.0e-14)
    assert d.max_damping_asymmetry_ratio == pytest.approx(1.0e-6, rel=1.0e-3)
    assert d.max_added_mass_asymmetry_ratio == pytest.approx(1.0e-6, rel=1.0e-3)
    assert d.min_damping_eigenvalue < 0.0  # reported, not repaired


# --- c. persistence -------------------------------------------------------

def test_relative_kernel_round_trip(tmp_path):
    _, _, k = _build(_c_matrix(), **REL)
    p = tmp_path / "rel.nc"
    rad.save_radiation_kernel(k, p)
    loaded = rad.load_radiation_kernel(p)
    assert loaded.settings.tolerance_basis == rad.TOLERANCE_BASIS_RELATIVE
    assert loaded.content_hash() == k.content_hash()
    for name in RATIOS:
        assert getattr(loaded.diagnostics, name) == getattr(k.diagnostics, name)
        assert getattr(loaded.diagnostics, name) is not None


def test_absolute_kernel_round_trip(tmp_path):
    _, _, k = _build(_c_matrix())
    p = tmp_path / "abs.nc"
    rad.save_radiation_kernel(k, p)
    loaded = rad.load_radiation_kernel(p)
    assert loaded.settings.tolerance_basis == rad.TOLERANCE_BASIS_ABSOLUTE
    assert loaded.content_hash() == k.content_hash()
    for name in RATIOS:
        assert getattr(loaded.diagnostics, name) is None


def test_tampered_persisted_diagnostics_are_refused(tmp_path):
    _, _, k = _build(_c_matrix(min_eig_rel=-1.0e-6), **REL)
    assert k.diagnostics.passivity_violations > 0
    p = tmp_path / "rel.nc"
    rad.save_radiation_kernel(k, p)
    raw = xr.load_dataset(p, engine="scipy")
    scalars = json.loads(raw.attrs["diagnostics_scalars"])
    scalars["min_damping_eigenvalue_ratio"] = 0.0  # contradicts passivity_violations > 0
    raw.attrs["diagnostics_scalars"] = json.dumps(scalars, sort_keys=True)
    bad = tmp_path / "tampered.nc"
    raw.to_netcdf(bad, engine="scipy", format="NETCDF3_64BIT")
    with pytest.raises(rad.RadiationError, match="inconsistent"):
        rad.load_radiation_kernel(bad)
    # dropping the ratios from a relative file is refused as well
    scalars = json.loads(xr.load_dataset(p, engine="scipy").attrs["diagnostics_scalars"])
    for name in RATIOS:
        scalars.pop(name)
    raw.attrs["diagnostics_scalars"] = json.dumps(scalars, sort_keys=True)
    raw.to_netcdf(bad, engine="scipy", format="NETCDF3_64BIT")
    with pytest.raises(rad.RadiationError):
        rad.load_radiation_kernel(bad)


# --- d. validation --------------------------------------------------------

@pytest.mark.parametrize("bad", ["Relative", "", "relative", None, 1, ("absolute",),
                                 "relative_to_damping_scale"])
def test_invalid_basis_is_refused(bad):
    with pytest.raises(rad.RadiationSettingsError):
        make_oracle_settings(tolerance_basis=bad)


def test_from_dict_refuses_unknown_basis():
    for bad in ("relative_to_something_else", "relative_to_damping_scale"):
        data = make_oracle_settings().to_dict()
        data["tolerance_basis"] = bad
        with pytest.raises(rad.RadiationSettingsError):
            rad.RadiationSettings.from_dict(data)


def test_from_dict_relative_round_trip():
    s = make_oracle_settings(**REL)
    assert rad.RadiationSettings.from_dict(s.to_dict()).tolerance_basis == rad.TOLERANCE_BASIS_RELATIVE


def test_zero_samples_give_zero_ratios_and_no_violations():
    zero = np.zeros((N_DOF, N_DOF))
    ds = make_oracle_dataset(C=zero, lam=np.full((N_DOF, N_DOF), LAM), a_inf=zero)
    settings = make_oracle_settings(zero_frequency_damping=zero, **REL)
    d = rad.radiation_kernel_from_dataset(ds, settings).diagnostics
    assert d.min_damping_eigenvalue_ratio == 0.0
    assert d.max_damping_asymmetry_ratio == 0.0 and d.max_added_mass_asymmetry_ratio == 0.0
    assert (d.passivity_violations, d.damping_symmetry_violations, d.added_mass_symmetry_violations) == (0, 0, 0)


# --- e. diagnostics are bound to the basis and tolerances -----------------------

@pytest.fixture(scope="module")
def kernels():
    _, _, rel = _build(_c_matrix(), **REL)
    _, _, ab = _build(_c_matrix(), symmetry_tol=1.0e-6, passivity_tol=1.0e-6)
    return rel, ab


def _with(kernel, **fields):
    diagnostics = dataclasses.replace(kernel.diagnostics, **fields)
    return dataclasses.replace(kernel, diagnostics=diagnostics)


def test_binding_accepts_consistent_kernels(kernels):
    rel, ab = kernels
    assert dataclasses.replace(rel).content_hash() == rel.content_hash()
    assert dataclasses.replace(ab).content_hash() == ab.content_hash()


def test_relative_basis_without_ratios_is_refused(kernels):
    rel, _ = kernels
    with pytest.raises(rad.RadiationError):
        _with(rel, min_damping_eigenvalue_ratio=None, max_damping_asymmetry_ratio=None,
              max_added_mass_asymmetry_ratio=None)


def test_absolute_basis_with_ratios_is_refused(kernels):
    rel, ab = kernels
    with pytest.raises(rad.RadiationError):
        _with(ab, min_damping_eigenvalue_ratio=0.0, max_damping_asymmetry_ratio=0.0,
              max_added_mass_asymmetry_ratio=0.0)
    # relative settings glued onto absolute diagnostics (and vice versa)
    with pytest.raises(rad.RadiationError):
        dataclasses.replace(ab, settings=dataclasses.replace(ab.settings, tolerance_basis=RELATIVE))
    with pytest.raises(rad.RadiationError):
        dataclasses.replace(rel, settings=dataclasses.replace(rel.settings, tolerance_basis="absolute"))


@pytest.mark.parametrize("keep", [("min_damping_eigenvalue_ratio",),
                                  ("min_damping_eigenvalue_ratio", "max_damping_asymmetry_ratio"),
                                  ("max_added_mass_asymmetry_ratio",)])
def test_partial_ratio_set_is_refused(kernels, keep):
    rel, _ = kernels
    fields = {name: None for name in RATIOS if name not in keep}
    with pytest.raises(rad.RadiationError):
        _with(rel, **fields)


@pytest.mark.parametrize("name,value", [
    ("min_damping_eigenvalue_ratio", float("nan")), ("min_damping_eigenvalue_ratio", float("-inf")),
    ("max_damping_asymmetry_ratio", float("nan")), ("max_damping_asymmetry_ratio", -0.5),
    ("max_added_mass_asymmetry_ratio", float("inf")), ("max_added_mass_asymmetry_ratio", -1.0e-3)])
def test_non_finite_or_negative_ratios_are_refused(kernels, name, value):
    rel, _ = kernels
    with pytest.raises(rad.RadiationError):
        _with(rel, **{name: value})


def test_relative_counts_must_match_ratios(kernels):
    rel, _ = kernels
    d = rel.diagnostics
    assert (d.passivity_violations, d.damping_symmetry_violations, d.added_mass_symmetry_violations) == (0, 0, 0)
    with pytest.raises(rad.RadiationError, match="inconsistent"):  # ratio violates, count says none
        _with(rel, min_damping_eigenvalue_ratio=-0.5)
    with pytest.raises(rad.RadiationError, match="inconsistent"):
        _with(rel, max_damping_asymmetry_ratio=0.5)
    with pytest.raises(rad.RadiationError, match="inconsistent"):
        _with(rel, max_added_mass_asymmetry_ratio=0.5)
    for name in ("passivity_violations", "damping_symmetry_violations", "added_mass_symmetry_violations"):
        with pytest.raises(rad.RadiationError, match="inconsistent"):  # count says violation, ratio fine
            _with(rel, **{name: 1})


def test_absolute_counts_must_match_extrema(kernels):
    _, ab = kernels
    d = ab.diagnostics
    assert d.passivity_violations > 0 and d.damping_symmetry_violations > 0
    with pytest.raises(rad.RadiationError, match="inconsistent"):
        _with(ab, passivity_violations=0)
    with pytest.raises(rad.RadiationError, match="inconsistent"):
        _with(ab, damping_symmetry_violations=0)
    with pytest.raises(rad.RadiationError, match="inconsistent"):
        _with(ab, added_mass_symmetry_violations=0)
    with pytest.raises(rad.RadiationError, match="inconsistent"):  # extremum inside tolerance, count > 0
        _with(ab, min_damping_eigenvalue=0.0)
    with pytest.raises(rad.RadiationError, match="inconsistent"):
        _with(ab, max_damping_asymmetry=0.0)
    with pytest.raises(rad.RadiationError, match="inconsistent"):
        _with(ab, max_added_mass_asymmetry=0.0)
