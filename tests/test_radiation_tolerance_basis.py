"""Declared relative tolerance basis for the radiation kernel diagnostics (slice F2a).

The default basis is absolute (SI) and must stay bit-identical: settings
dicts, diagnostics scalars, persisted files and kernel content hashes. The
optional ``relative_to_damping_scale`` basis scales the declared tolerances by
the spectral-norm scale of the checked damping / added-mass samples; violations
are still counted and reported, never clamped or repaired.
"""

from __future__ import annotations

import numpy as np
import pytest

from anyhydro import radiation as rad

from test_radiation import make_oracle_dataset, make_oracle_settings

N_DOF = 12
SCALE = 1.0e8
LAM = 2.0
NEW_SCALARS = (
    "damping_scale", "added_mass_scale", "effective_damping_symmetry_tol",
    "effective_added_mass_symmetry_tol", "effective_passivity_tol",
)


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


REL = dict(tolerance_basis="relative_to_damping_scale", symmetry_tol=1.0e-2, passivity_tol=1.0e-9)


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
    assert not set(NEW_SCALARS) & set(scalars)
    for name in NEW_SCALARS:
        assert getattr(k_default.diagnostics, name) is None


# --- b. relative basis ----------------------------------------------------

def test_absolute_flags_roundoff_but_relative_does_not():
    c = _c_matrix()
    _, _, k_abs = _build(c, symmetry_tol=1.0e-6, passivity_tol=1.0e-6)
    da = k_abs.diagnostics
    assert da.passivity_violations > 0
    assert da.damping_symmetry_violations > 0 or da.added_mass_symmetry_violations > 0

    _, settings, k_rel = _build(c, **REL)
    d = k_rel.diagnostics
    assert settings.to_dict()["tolerance_basis"] == "relative_to_damping_scale"
    assert d.passivity_violations == 0
    assert d.damping_symmetry_violations == 0
    assert d.added_mass_symmetry_violations == 0
    assert d.min_damping_eigenvalue < 0.0  # reported, not repaired
    # recorded scales and effective tolerances
    assert d.damping_scale == pytest.approx(SCALE / LAM, rel=1.0e-6)
    assert d.damping_scale > 0.0 and d.added_mass_scale > 0.0
    assert d.effective_damping_symmetry_tol == pytest.approx(1.0e-2 * d.damping_scale)
    assert d.effective_added_mass_symmetry_tol == pytest.approx(1.0e-2 * d.added_mass_scale)
    assert d.effective_passivity_tol == pytest.approx(1.0e-9 * d.damping_scale)
    assert set(NEW_SCALARS) <= set(d.scalars())
    # the kernel content differs from the absolute one only by declared settings/diagnostics
    assert k_rel.content_hash() != k_abs.content_hash()


def test_relative_still_flags_real_passivity_violation():
    _, _, k = _build(_c_matrix(min_eig_rel=-1.0e-6), **REL)
    assert k.diagnostics.passivity_violations > 0


def test_relative_still_flags_real_asymmetry():
    _, _, k = _build(_c_matrix(asym_rel=1.0e-1), **REL)
    d = k.diagnostics
    assert d.damping_symmetry_violations > 0
    assert d.max_damping_asymmetry > d.effective_damping_symmetry_tol


# --- c. persistence -------------------------------------------------------

def test_relative_kernel_round_trip(tmp_path):
    _, _, k = _build(_c_matrix(), **REL)
    p = tmp_path / "rel.nc"
    rad.save_radiation_kernel(k, p)
    loaded = rad.load_radiation_kernel(p)
    assert loaded.settings.tolerance_basis == rad.TOLERANCE_BASIS_RELATIVE
    assert loaded.content_hash() == k.content_hash()
    for name in NEW_SCALARS:
        assert getattr(loaded.diagnostics, name) == getattr(k.diagnostics, name)
        assert getattr(loaded.diagnostics, name) is not None


def test_absolute_kernel_round_trip(tmp_path):
    _, _, k = _build(_c_matrix())
    p = tmp_path / "abs.nc"
    rad.save_radiation_kernel(k, p)
    loaded = rad.load_radiation_kernel(p)
    assert loaded.settings.tolerance_basis == rad.TOLERANCE_BASIS_ABSOLUTE
    assert loaded.content_hash() == k.content_hash()
    for name in NEW_SCALARS:
        assert getattr(loaded.diagnostics, name) is None


# --- d. validation --------------------------------------------------------

@pytest.mark.parametrize("bad", ["Relative", "", "relative", None, 1, ("absolute",)])
def test_invalid_basis_is_refused(bad):
    with pytest.raises(rad.RadiationSettingsError):
        make_oracle_settings(tolerance_basis=bad)


def test_from_dict_refuses_unknown_basis():
    data = make_oracle_settings().to_dict()
    data["tolerance_basis"] = "relative_to_something_else"
    with pytest.raises(rad.RadiationSettingsError):
        rad.RadiationSettings.from_dict(data)


def test_from_dict_relative_round_trip():
    s = make_oracle_settings(**REL)
    assert rad.RadiationSettings.from_dict(s.to_dict()).tolerance_basis == rad.TOLERANCE_BASIS_RELATIVE


def test_zero_scale_gives_zero_effective_tolerance():
    zero = np.zeros((N_DOF, N_DOF))
    ds = make_oracle_dataset(C=zero, lam=np.full((N_DOF, N_DOF), LAM), a_inf=zero)
    settings = make_oracle_settings(zero_frequency_damping=zero, **REL)
    d = rad.radiation_kernel_from_dataset(ds, settings).diagnostics
    assert d.damping_scale == 0.0 and d.effective_passivity_tol == 0.0
    assert d.effective_damping_symmetry_tol == 0.0
