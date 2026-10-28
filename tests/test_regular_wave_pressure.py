"""Refusal before entering a BEM solve (provider remains a lazy optional dep)."""
import pytest
from anyhydro.regular_wave_pressure import solve_regular_wave_pressure


def test_noninteger_topology_is_not_silently_truncated():
    pytest.importorskip('capytaine')
    with pytest.raises(ValueError,match='integer face'):
        solve_regular_wave_pressure([[0,0,-1],[1,0,-1],[1,1,-1],[0,1,-1]],[[0.,1.,2.,3.]],['bottom'])


def test_open_underwater_seams_are_refused_before_solving():
    pytest.importorskip('capytaine')
    with pytest.raises(ValueError,match='open hull seams'):
        solve_regular_wave_pressure([[0,0,-1],[1,0,-1],[1,1,-1],[0,1,-1]],[[0,1,2,3]],['bottom'])
