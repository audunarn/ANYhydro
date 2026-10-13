"""Pairwise reciprocity diagnostic (owner decision 2026-10-13; anyhydro.reciprocity)."""
import json
import warnings

import numpy as np
import pytest
import xarray as xr

from anyhydro.reciprocity import (RECIPROCITY_WARNING, ReciprocityWarning, dataset_reciprocity, pairwise_defect,
                                  reciprocity_diagnostic, warn_if_exceeded)

KEYS = ['b::surge', 'b::sway', 'b::roll']


def _matrices():
    M = np.zeros((3, 3, 3))
    for o, s in enumerate((1.0, 2.0, 3.0)):
        M[o] = np.diag([4.0, 1.0, 9.0]) * s
        M[o, 2, 1], M[o, 1, 2] = -3.0 * s, -3.3 * s                         # defect 0.3 s / (3 s) = 0.1
    return M


def test_pairwise_defect_value_and_location():
    value, o, pair = pairwise_defect(_matrices())
    assert value == pytest.approx(0.1) and pair == (1, 2) and o in (0, 1, 2)


def test_symmetric_matrices_have_zero_defect():
    M = _matrices()
    M = 0.5 * (M + np.transpose(M, (0, 2, 1)))
    assert pairwise_defect(M)[0] == 0.0


def test_pairs_below_the_scale_floor_or_with_zero_scale_are_skipped():
    M = _matrices()
    M[:, 0, 0] = 0.0                                                        # zero scale: surge pairs skipped
    M[:, 0, 1], M[:, 1, 0] = 5.0, -5.0
    assert pairwise_defect(M)[0] == pytest.approx(0.1)
    B = np.zeros((2, 2, 2))
    B[0] = [[1.0, 0.5], [0.0, 1.0]]                                         # large defect at a negligible scale
    B[0] *= 1e-6
    B[1] = [[1.0, 0.01], [0.01, 1.0]]
    assert pairwise_defect(B)[0] == 0.0


def test_diagnostic_is_json_ready_and_flags_exceedance():
    d = reciprocity_diagnostic(_matrices(), 0.5 * (_matrices() + np.transpose(_matrices(), (0, 2, 1))),
                               [0.5, 1.0, 1.5], KEYS)
    assert json.loads(json.dumps(d)) == d
    assert d['added_mass']['dofs'] == ['b::sway', 'b::roll'] and d['radiation_damping']['max'] == 0.0
    assert d['exceeds'] and d['threshold'] == RECIPROCITY_WARNING
    assert not reciprocity_diagnostic(_matrices(), _matrices(), [0.5, 1.0, 1.5], KEYS, threshold=0.2)['exceeds']


def test_warning_names_the_pair_and_is_silent_below_the_threshold():
    d = reciprocity_diagnostic(_matrices(), _matrices(), [0.5, 1.0, 1.5], KEYS)
    with pytest.warns(ReciprocityWarning, match='b::sway / b::roll'):
        warn_if_exceeded(d)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        warn_if_exceeded(reciprocity_diagnostic(_matrices(), _matrices(), [0.5, 1.0, 1.5], KEYS, threshold=0.5))


def test_dataset_reciprocity_reads_any_dataset_order():
    M = _matrices()
    da = xr.DataArray(M, dims=('omega', 'influenced_dof', 'radiating_dof'),
                      coords={'omega': [0.5, 1.0, 1.5], 'influenced_dof': KEYS, 'radiating_dof': KEYS})
    ds = xr.Dataset({'added_mass': da.transpose('radiating_dof', 'omega', 'influenced_dof'), 'radiation_damping': da})
    assert dataset_reciprocity(ds)['added_mass']['max'] == pytest.approx(0.1)


@pytest.mark.parametrize('bad', [np.zeros((3, 3)), np.full((1, 2, 2), np.nan)])
def test_malformed_input_is_refused(bad):
    with pytest.raises(ValueError):
        pairwise_defect(bad)


@pytest.mark.parametrize('size', [1e200, 1e-200])
def test_extreme_magnitudes_neither_overflow_nor_underflow(size):
    """Review: sqrt(M_ii M_jj) formed as a product overflowed (1e200) or underflowed (1e-200)."""
    M = np.array([[[size, 0.1 * size], [0.0, size]]])
    assert pairwise_defect(M)[0] == pytest.approx(0.1)


def test_near_overflow_difference_stays_finite_and_json_safe():
    """Review round 2: M - M^T overflowed to inf for finite entries near 1e308."""
    M = np.array([[[1e308, 1e308], [-1e308, 1e308]]])
    value = pairwise_defect(M)[0]
    assert value == pytest.approx(2.0)
    d = reciprocity_diagnostic(M, M, [1.0], ['a', 'b'])
    assert 'Infinity' not in json.dumps(d) and json.loads(json.dumps(d))['added_mass']['max'] == pytest.approx(2.0)


def test_subnormal_and_wide_range_entries_round_3():
    """Review round 3: halving erased subnormal asymmetry; the quotient overflowed for wide-range entries."""
    u = np.nextafter(0.0, 1.0)
    assert pairwise_defect(np.array([[[16 * u, u], [0.0, 16 * u]]]))[0] == pytest.approx(1 / 16)
    with np.errstate(all='raise'):
        value = pairwise_defect(np.array([[[1e-200, 1e200], [0.0, 1e-200]]]))[0]
    assert np.isfinite(value) and value == pytest.approx(np.finfo(np.float64).max / 4, rel=1e-12)
    assert json.loads(json.dumps(reciprocity_diagnostic(np.array([[[1e-200, 1e200], [0.0, 1e-200]]]),
                                                        np.eye(2)[None], [1.0], ['a', 'b'])))


def test_pair_exactly_at_the_floor_is_kept_round_4():
    """Review round 4: log rounding excluded a pair whose scale equals the floor of its peak."""
    B = np.array([[[2.0, 0.2], [0.0, 2.0]], [[2000.0, 0.0], [0.0, 2000.0]]])
    value, o, pair = pairwise_defect(B)
    assert value == pytest.approx(0.1) and o == 0 and pair == (0, 1)


def test_opposite_extreme_off_diagonals_stay_finite_round_4():
    """Review round 4: unit = 2 times the capped ratio could round past float64 max."""
    with np.errstate(all='raise'):
        value = pairwise_defect(np.array([[[1e-200, 1e200], [-1e200, 1e-200]]]))[0]
    assert np.isfinite(value)
