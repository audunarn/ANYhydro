import json
import math
import sys
import numpy as np
import pytest
from anyloads import RegularWave, WaveRealization, JonswapSpectrum, DirectionalSpectrum
from anyhydro.fixtures import make_two_body_fixture
from anyhydro import (build_excitation_series, ExcitationSeries,
    ExcitationSeriesError, verify_excitation_binding, UNIT_INCIDENT_AMPLITUDE)


def build(ds,r,**kw):
    return build_excitation_series(ds,r,amplitude_normalization=kw.get('amplitude_normalization',UNIT_INCIDENT_AMPLITUDE),
        phase_origin_xy=kw.get('phase_origin_xy',(0.,0.)))


def test_single_phasor_interpolation_full_dof_identity():
    ds=make_two_body_fixture(); r=RegularWave(.4,.9,math.pi/4,math.pi/2).realize()
    series=build(ds,r); t=np.array([0.,.3,1.,2.])
    x=(ds.excitation_force.values[1,1]+ds.excitation_force.values[2,1])/2
    expected=.4*(x.real[None,:]*np.cos(math.pi/2-.9*t[:,None])-x.imag[None,:]*np.sin(math.pi/2-.9*t[:,None]))
    np.testing.assert_allclose(series.evaluate(t),expected,rtol=1e-12,atol=1e-12)
    assert series.dof_keys==tuple(ds.influenced_dof.values) and len(series.dof_keys)==12
    assert series.source_qualification_status=='development_fixture'
    for omega,index in ((.3,0),(2.2,4)):
        s=build(ds,RegularWave(1.,omega).realize())
        np.testing.assert_allclose(s.evaluate([0.])[0],ds.excitation_force.values[index,0].real,rtol=1e-12,atol=1e-12)


def test_directional_common_realization_saved_inputs_and_binding(monkeypatch):
    ds=make_two_body_fixture()
    r=WaveRealization.from_jonswap(JonswapSpectrum(.5,8.),(.7,1.1,1.6),(.2,.2,.2),
        seed=71,directions=DirectionalSpectrum((0.,math.pi/2),(.3,.7)))
    s=build(ds,r); t=np.array([0.,.2,1.])
    expected=np.zeros((len(t),12))
    for i,c in enumerate(r.components):
        j=0 if i%2==0 else 2; k=(1,2,3)[i//2]; x=ds.excitation_force.values[k,j]
        expected+=c.amplitude*(np.cos(c.phase-c.omega*t[:,None])*x.real-np.sin(c.phase-c.omega*t[:,None])*x.imag)
    np.testing.assert_allclose(s.evaluate(t),expected,rtol=1e-12,atol=1e-12)
    data=json.loads(json.dumps(dict(waves=r.to_dict(),forces=s.to_dict())))
    monkeypatch.setattr(np.random,'PCG64',lambda *a:pytest.fail('regenerated phases'))
    r2=WaveRealization.from_dict(data['waves']); s2=ExcitationSeries.from_dict(data['forces'])
    assert r2.content_hash==r.content_hash and s2.content_hash==s.content_hash
    assert verify_excitation_binding(s2,ds,r2)
    np.testing.assert_array_equal(s2.evaluate(t),s.evaluate(t))
    np.testing.assert_array_equal(r2.environment(20.).evaluate([[1.,2.,-3.]],.2).velocity,
                                 r.environment(20.).evaluate([[1.,2.,-3.]],.2).velocity)
    changed=ds.copy(deep=True); changed.excitation_force.values[1,0,0]+=1.
    with pytest.raises(ExcitationSeriesError): verify_excitation_binding(s,changed,r)
    data['forces']['weighted_real'][0][0]+=1.
    with pytest.raises(ExcitationSeriesError): verify_excitation_binding(ExcitationSeries.from_dict(data['forces']),ds,r)
    wave_data=r.to_dict(); wave_data['components'][0]['phase']+=.1
    with pytest.raises(ExcitationSeriesError): verify_excitation_binding(s,ds,WaveRealization.from_dict(wave_data))
    changed=ds.copy(deep=True);changed.attrs['qualification_status']='unqualified'
    with pytest.raises(ExcitationSeriesError): verify_excitation_binding(s,changed,r)


def test_heading_modulo_and_ambiguity():
    ds=make_two_body_fixture(); ds=ds.assign_coords(wave_direction=ds.wave_direction.values+2*math.pi)
    build(ds,RegularWave(.2,.7,math.pi/4).realize())
    ds=ds.assign_coords(wave_direction=[0.,2*math.pi,math.pi/2])
    with pytest.raises(ExcitationSeriesError): build(ds,RegularWave(.2,.7).realize())


def test_immutable_array_storage_no_alias():
    s=build(make_two_body_fixture(),RegularWave(.2,.7).realize()); data=s.to_dict()
    for a in (s.omegas,s.weighted_real,s.weighted_imag):
        with pytest.raises(ValueError): a.setflags(write=True)
        with pytest.raises(ValueError): a.flat[0]=99.
    data['weighted_real'][0][0]=99.
    assert s.weighted_real[0,0]!=99.


@pytest.mark.parametrize('omega,heading,kw',[(.1,0.,{}),(3.,0.,{}),(.7,.3,{}),
    (.7,0.,{'phase_origin_xy':(1.,0.)}),(.7,0.,{'amplitude_normalization':'unknown'}),
    (.7,0.,{'phase_origin_xy':(False,False)})])
def test_synthesis_refusals(omega,heading,kw):
    with pytest.raises(ExcitationSeriesError): build(make_two_body_fixture(),RegularWave(.2,omega,heading).realize(),**kw)


@pytest.mark.parametrize('key,value',[('api_version','unknown'),('provider','python'),('omegas',[False]),
    ('source_qualification_status','accepted'),('hydro_content_hash','bad'),('dof_keys',['surge']),
    ('weighted_imag',[[float('nan')]*12])])
def test_serialized_refusals(key,value):
    data=build(make_two_body_fixture(),RegularWave(.2,.7).realize()).to_dict();data[key]=value
    with pytest.raises(ExcitationSeriesError): ExcitationSeries.from_dict(data)


def test_mixed_booleans_and_chunk_boundary():
    s=build(make_two_body_fixture(),RegularWave(.2,.7).realize())
    for times in ([0.,True], [np.bool_(False),1.]):
        with pytest.raises(ExcitationSeriesError):s.evaluate(times)
    data=s.to_dict();data['weighted_real'][0][0]=True
    with pytest.raises(ExcitationSeriesError):ExcitationSeries.from_dict(data)
    with pytest.raises(ExcitationSeriesError):build(make_two_body_fixture(),RegularWave(.2,.7).realize(),phase_origin_xy=(False,0.))
    t=np.linspace(0.,2.,2051)
    np.testing.assert_array_equal(s.evaluate(t),np.vstack((s.evaluate(t[:2048]),s.evaluate(t[2048:]))))


def test_missing_optional_sea_state_dependency(monkeypatch):
    ds=make_two_body_fixture(); r=RegularWave(.2,.7).realize()
    monkeypatch.setitem(sys.modules,'anyloads',None)
    with pytest.raises(ExcitationSeriesError,match='extra'): build(ds,r)
