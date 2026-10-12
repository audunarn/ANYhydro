# SPDX-License-Identifier: MPL-2.0
"""Declared power-law high-frequency tail of the radiation kernel."""
from __future__ import annotations

import numpy as np
import pytest

from anyhydro import radiation as rad

from test_radiation import _oracle_matrices, make_oracle_dataset

# Lorentzian oracle B = C lam/(lam^2 + w^2) has the exact asymptote C lam / w^2.
OMEGAS = np.linspace(0.05, 3.0, 120)   # truncated where the tail still matters
DT = 0.01
N_TIME = 4001


def settings(**extra):
    C, lam, _ = _oracle_matrices(12)
    base = dict(dt=DT, n_time=N_TIME, symmetry_tol=1e-9, passivity_tol=1e-9,
                low_tail_policy="explicit_zero_sample", zero_frequency_damping=C / lam,
                quadrature_method="piecewise_linear_exact",
                execution_limits=rad.RadiationExecutionLimits(max_time_samples=N_TIME))
    base.update(extra)
    return rad.RadiationSettings(**base)


def power(exponent=2.0, upper=300.0):
    return dict(high_tail_policy="power_law_extension", high_tail_exponent=exponent,
                high_tail_upper=upper, high_tail_name="test asymptote")


def _errors(kernel, ds, lo=0.3, hi=1.5):
    om = ds["omega"].values
    s = (om >= lo) & (om <= hi)
    a_rec, b_rec = kernel.reconstruct(om[s])
    a = ds["added_mass"].values[s]
    b = ds["radiation_damping"].values[s]
    return np.abs(a_rec - a).max() / np.abs(a).max(), np.abs(b_rec - b).max() / np.abs(b).max()


def test_power_law_tail_removes_the_truncation_offset():
    ds = make_oracle_dataset(omegas=OMEGAS)
    trunc = rad.radiation_kernel_from_dataset(ds, settings())
    tail = rad.radiation_kernel_from_dataset(ds, settings(**power(2.0)))
    ea_t, eb_t = _errors(trunc, ds)
    ea_p, eb_p = _errors(tail, ds)
    assert ea_t > 5e-3                    # truncation: visible added-mass offset
    assert ea_p < 0.1 * ea_t              # exact asymptote: offset removed
    assert eb_p <= eb_t + 1e-6
    assert any('power_law_extension' in a for a in tail.assumptions)
    assert not any('high_tail_truncated' in a for a in tail.assumptions)


def test_exponent_sensitivity_is_monotone_and_bounded():
    ds = make_oracle_dataset(omegas=OMEGAS)
    errs = [_errors(rad.radiation_kernel_from_dataset(ds, settings(**power(n))), ds)[0]
            for n in (1.5, 2.0, 3.0)]
    assert errs[1] < errs[0] and errs[1] < errs[2]


def test_default_settings_serialization_is_unchanged():
    d = settings().to_dict()
    assert not any(k.startswith('high_tail') for k in d)
    back = rad.RadiationSettings.from_dict(d)
    assert back.high_tail_policy == rad.HIGH_TAIL_TRUNCATE
    p = settings(**power(3.0, 60.0))
    back = rad.RadiationSettings.from_dict(p.to_dict())
    assert (back.high_tail_policy, back.high_tail_exponent, back.high_tail_upper,
            back.high_tail_name) == ("power_law_extension", 3.0, 60.0, "test asymptote")


@pytest.mark.parametrize('bad', [
    dict(high_tail_exponent=3.0),                                     # truncate + exponent
    dict(**{**power(), 'quadrature_method': 'frequency_trapezoid'}),  # needs exact quadrature
    dict(**{**power(), 'high_tail_exponent': 1.0}),
    dict(**{**power(), 'high_tail_upper': float('nan')}),
    dict(**{**power(), 'high_tail_name': ' '}),
    dict(high_tail_policy='fit_everything'),
])
def test_invalid_high_tail_settings_are_refused(bad):
    with pytest.raises(rad.RadiationSettingsError):
        settings(**bad)


def test_upper_must_exceed_the_grid():
    ds = make_oracle_dataset(omegas=OMEGAS)
    with pytest.raises(rad.RadiationSettingsError, match='must exceed'):
        rad.radiation_kernel_from_dataset(ds, settings(**power(2.0, 2.5)))


def test_kernel_round_trip_keeps_the_tail_settings(tmp_path):
    ds = make_oracle_dataset(omegas=OMEGAS)
    k = rad.radiation_kernel_from_dataset(ds, settings(**power(2.0)))
    rad.save_radiation_kernel(k, tmp_path / 'k.nc')
    back = rad.load_radiation_kernel(tmp_path / 'k.nc')
    assert back.settings.to_dict() == k.settings.to_dict()
    np.testing.assert_array_equal(back.kernel, k.kernel)


def test_tail_above_nyquist_and_huge_spans_are_refused():
    ds = make_oracle_dataset(omegas=OMEGAS)
    with pytest.raises(rad.RadiationSettingsError, match='Nyquist'):
        rad.radiation_kernel_from_dataset(ds, settings(**power(2.0, 1000.0)))   # 1000*0.01 > pi
    # Nyquist-admissible but a huge logarithmic span: typed allocation refusal
    tiny = make_oracle_dataset(omegas=np.geomspace(1e-95, 1e-94, 8))
    with pytest.raises(rad.RadiationLimitError):
        rad.radiation_kernel_from_dataset(tiny, settings(**{**power(2.0, 1000.0), 'dt': 1e-3}))
