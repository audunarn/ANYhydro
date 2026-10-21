"""B1: declared producer-side kernel symmetrisation (symmetric part, recorded and bounded skew, hash-bound).

Default ("none") is omitted everywhere so settings dicts, diagnostics, saved files and hashes are unchanged.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from anyhydro import radiation as rad

from test_radiation_tolerance_basis import (N_DOF, REL, _kernel_from_samples, _non_proportional)

BASE = dict(dt=0.1, n_time=11, symmetry_tol=1e-6, passivity_tol=1e-6,
            low_tail_policy=rad.LOW_TAIL_LINEAR_TO_ZERO, extension_name="ext")
SYM = dict(kernel_symmetrisation="symmetric_part", kernel_skew_bound=1.0e-2)


def _skewed(skew=1.0e-4):
    """Damping/added-mass samples with a small off-diagonal skew in samples 1 and 2 (within symmetry_tol 1e-2)."""
    s = np.zeros((N_DOF, N_DOF))
    s[0, 1], s[1, 0] = skew * 1e2, -skew * 1e2          # skew pair relative to the 1e2 diagonal
    added, damping, b0 = _non_proportional(small_damping=1.0e2 * np.eye(N_DOF) + s,
                                           small_added=50.0 * np.eye(N_DOF) + s)
    return added, damping, b0


def _pair(**extra):
    added, damping, b0 = _skewed(**extra)
    off = _kernel_from_samples(added, damping, b0, **REL)
    on = _kernel_from_samples(added, damping, b0, **REL, **SYM)
    return off, on


def test_default_is_none_and_omitted():
    s = rad.RadiationSettings(**BASE)
    assert s.kernel_symmetrisation == "none" and s.kernel_skew_bound is None
    assert "kernel_symmetrisation" not in s.to_dict() and "kernel_skew_bound" not in s.to_dict()


def test_settings_validation():
    base = BASE
    s = rad.RadiationSettings(**base, **SYM)
    assert rad.RadiationSettings.from_dict(s.to_dict()).to_dict() == s.to_dict()
    for bad in (dict(kernel_symmetrisation="sym"), dict(kernel_symmetrisation="symmetric_part"),
                dict(kernel_symmetrisation="symmetric_part", kernel_skew_bound=0.0),
                dict(kernel_symmetrisation="symmetric_part", kernel_skew_bound=float("nan")),
                dict(kernel_symmetrisation="none", kernel_skew_bound=1e-2)):
        with pytest.raises(rad.RadiationSettingsError):
            rad.RadiationSettings(**base, **bad)


def test_kernel_is_the_symmetric_part_of_the_unsymmetrised_kernel():
    off, on = _pair()
    K = np.asarray(off.kernel)
    assert np.abs(K - K.transpose(0, 2, 1)).max() > 0.0
    Kon = np.asarray(on.kernel)
    assert np.array_equal(Kon, 0.5 * (K + K.transpose(0, 2, 1)))
    assert np.array_equal(Kon, Kon.transpose(0, 2, 1))
    assert np.array_equal(np.asarray(on.added_mass_infinite), np.asarray(off.added_mass_infinite))


def test_recorded_skew_and_hash_binding():
    off, on = _pair()
    rec = dict(on.provenance["kernel_symmetrisation"])
    K = np.asarray(off.kernel)
    assert rec["option"] == "symmetric_part" and rec["skew_bound"] == 1.0e-2
    assert rec["max_abs_change_over_max_abs_kernel"] == float(np.abs(K - 0.5 * (K + K.transpose(0, 2, 1))).max()
                                                              / np.abs(K).max())
    assert 0.0 < rec["pairwise_skew_over_peak"] <= 1.0e-2
    assert rec["worst_sample_skew_ratio"] >= rec["pairwise_skew_over_peak"] * 0.0
    assert off.content_hash() != on.content_hash()
    assert "kernel_symmetrisation" not in off.provenance
    tampered = dict(on.provenance)
    tampered["kernel_symmetrisation"] = {**rec, "pairwise_skew_over_peak": 0.0}
    assert dataclasses.replace(on, provenance=tampered).content_hash() != on.content_hash()


def test_kernel_object_fails_closed_on_inconsistent_symmetrisation():
    off, on = _pair()
    with pytest.raises(rad.RadiationError):          # option declared but data not symmetric
        dataclasses.replace(on, kernel=np.asarray(off.kernel))
    prov = {k: v for k, v in on.provenance.items() if k != "kernel_symmetrisation"}
    with pytest.raises(rad.RadiationError):          # option declared but no record
        dataclasses.replace(on, provenance=prov)
    with pytest.raises(rad.RadiationError):          # record without the declared option
        dataclasses.replace(off, provenance={**off.provenance, "kernel_symmetrisation": on.provenance["kernel_symmetrisation"]})
    bad = {**on.provenance, "kernel_symmetrisation": {**on.provenance["kernel_symmetrisation"], "skew_bound": 0.5}}
    with pytest.raises(rad.RadiationError):          # recorded bound differs from the settings
        dataclasses.replace(on, provenance=bad)
    big = {**on.provenance, "kernel_symmetrisation": {**on.provenance["kernel_symmetrisation"],
                                                      "pairwise_skew_over_peak": 0.5}}
    with pytest.raises(rad.RadiationError):          # recorded skew above the bound
        dataclasses.replace(on, provenance=big)


def test_skew_above_bound_is_refused():
    added, damping, b0 = _skewed(skew=5.0e-3)
    with pytest.raises(rad.RadiationError, match="kernel skew"):
        _kernel_from_samples(added, damping, b0, **REL, kernel_symmetrisation="symmetric_part",
                             kernel_skew_bound=1.0e-12)


def test_save_load_round_trip_and_tamper_detection(tmp_path):
    _, on = _pair()
    p = tmp_path / "k.nc"
    rad.save_radiation_kernel(on, p)
    back = rad.load_radiation_kernel(p)
    assert back.content_hash() == on.content_hash() and back.settings.kernel_symmetrisation == "symmetric_part"
    import xarray as xr
    raw = xr.load_dataset(p, engine="scipy")
    raw.attrs["provenance"] = raw.attrs["provenance"].replace("pairwise_skew_over_peak", "pairwise_skew_over_peakx")
    q = tmp_path / "t.nc"
    raw.to_netcdf(q, engine="scipy")
    with pytest.raises(rad.RadiationError):
        rad.load_radiation_kernel(q)


def test_per_sample_ratio_can_exceed_bound_while_whole_kernel_skew_is_small():
    """The ill-conditioned per-sample measure is information only; the bound is on the whole kernel."""
    _, on = _pair()
    rec = on.provenance["kernel_symmetrisation"]
    assert set(rec) >= {"worst_sample", "worst_sample_skew_ratio", "n_samples_over_1e-2_per_sample"}


def test_symmetrised_kernel_is_still_passive_on_the_relative_basis():
    _, on = _pair()
    assert on.diagnostics.passivity_violations == 0


def test_band_consistent_a_inf_is_calibrated_against_the_symmetrised_kernel():
    """Codex review round 1: A_inf must be proved against the kernel the artifact actually carries."""
    from test_radiation_tolerance_basis import ORACLE_OMEGAS
    band = dict(a_inf_policy="band_consistent", a_inf_calibration_min=float(ORACLE_OMEGAS[2]),
                a_inf_calibration_max=float(ORACLE_OMEGAS[-3]))
    added, damping, b0 = _skewed(skew=1.0e-3)
    k = _kernel_from_samples(added, damping, b0, **REL, **SYM, **band)
    assert np.array_equal(np.asarray(k.kernel), np.asarray(k.kernel).transpose(0, 2, 1))
    out = rad.verify_band_consistent_added_mass_infinite(k, np.eye(N_DOF) * 50.0, ORACLE_OMEGAS, added)
    assert np.array_equal(out, k.added_mass_infinite)
