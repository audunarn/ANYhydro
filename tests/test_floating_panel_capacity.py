"""Capacity is an explicit cache/recipe contract, never a wider default."""
import numpy as np
import pytest
from anyhydro import floating_radiation_fields as ff
from anyhydro.wave_surface_fields import WaveFieldError


def request(count=1, **kwargs):
    return ff.family_request(np.array([[0,0,-1],[1,0,-1],[1,1,-1],[0,1,-1.]]),
        np.tile([0,1,2,3],(count,1)), ['hull']*count, heading=0.,water_depth=50.,
        rho=1025.,gravity=9.81,Q0=np.eye(3),cg_world=[0,0,0],**kwargs)


def test_default_identity_has_no_explicit_capacity_and_remains_600():
    r=request()
    assert 'source_capacity' not in r
    assert ff._panel_ceiling(r)==600


@pytest.mark.parametrize('capacity,limit', [('faceted_hull_v1',2500),('faceted_quad_hull_v1',4000)])
def test_explicit_capacity_is_bound_to_direct_recipe_and_refuses_overflow(capacity,limit):
    with pytest.raises(WaveFieldError,match='direct family recipe'):
        request(source_capacity=capacity)
    accepted=request(limit,recipe=ff.DIRECT_RECIPE,source_capacity=capacity)
    assert ff._panel_ceiling(accepted)==limit
    with pytest.raises(WaveFieldError,match='exceed'):
        request(limit+1,recipe=ff.DIRECT_RECIPE,source_capacity=capacity)


def test_distinct_profiles_cannot_share_request_identity():
    tri=request(recipe=ff.DIRECT_RECIPE,source_capacity='faceted_hull_v1')
    quad=request(recipe=ff.DIRECT_RECIPE,source_capacity='faceted_quad_hull_v1')
    assert tri!=quad
    assert tri['geometry']==quad['geometry']
    with pytest.raises(WaveFieldError,match='known profile'):
        request(recipe=ff.DIRECT_RECIPE,source_capacity='unknown')
