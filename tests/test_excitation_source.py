"""Transport and phasor algebra only; no BEM or source qualification."""
import copy
from dataclasses import replace
import json
import math

import numpy as np
import pytest

from anyhydro import (EXCITATION_SOURCE_ATTR, CURRENT_POLICY, ExcitationSourceError,
    attach_excitation_source, verify_excitation_source, request_to_dict,
    verify_excitation_source_declaration,
    request_from_dict, build_source_bound_excitation_series,
    dataset_content_hash, save_hydro_dataset, load_hydro_dataset)
from anyhydro.fixtures import make_two_body_fixture, make_fixture_request, make_fixture_bodies
from anyloads import RegularWave


def source():
    raw=make_two_body_fixture();r=make_fixture_request();b=make_fixture_bodies()
    return raw,attach_excitation_source(raw,r,b,convention_evidence='declared fixture convention test double'),r,b


def environment(r):
    return dict(depth_kind='finite',water_depth=r.water_depth,rho=r.rho,
                gravity=r.gravity,still_water_level=0.,current_policy=CURRENT_POLICY)


def test_detached_declaration_preserves_legacy_source_and_status(tmp_path):
    raw,ds,r,b=source();old=dataset_content_hash(raw)
    assert EXCITATION_SOURCE_ATTR not in raw.attrs
    c=verify_excitation_source(ds,r,b)
    assert c['dataset_binding_hash']==old and dataset_content_hash(raw)==old
    assert ds.attrs['qualification_status']=='development_fixture'
    assert dataset_content_hash(ds)!=old
    c['conventions']['reference_incident_amplitude_m']=3.
    assert verify_excitation_source(ds,r,b)['conventions']['reference_incident_amplitude_m']==1.
    path=tmp_path/'declared.nc';save_hydro_dataset(ds,path);restored=load_hydro_dataset(path)
    assert dataset_content_hash(restored)==dataset_content_hash(ds)
    assert verify_excitation_source(restored,request_from_dict(request_to_dict(r)),b)
    with pytest.raises(ExcitationSourceError):attach_excitation_source(ds,r,b,convention_evidence='duplicate')


@pytest.mark.parametrize('infinite',[False,True])
def test_complete_request_roundtrip_without_defaults(infinite):
    r=replace(make_fixture_request(),water_depth=math.inf if infinite else 20.,
              backend_settings={'nested':{'table':[1,2]},'label':'saved'})
    d=request_to_dict(r);assert request_from_dict(json.loads(json.dumps(d))).content_hash()==r.content_hash()
    d['backend_settings']['nested']['table'][0]=8
    assert r.backend_settings['nested']['table'][0]==1
    d=request_to_dict(r);d.pop('gravity')
    with pytest.raises(ExcitationSourceError):request_from_dict(d)
    d=request_to_dict(r);d['water_depth']=150. if infinite else None
    with pytest.raises(ExcitationSourceError):request_from_dict(d)


@pytest.mark.parametrize('key,value',[('rho',True),('gravity',float('nan')),('omegas',[True]),('include_infinite_frequency',1)])
def test_malformed_request_refused(key,value):
    d=request_to_dict(make_fixture_request());d[key]=value
    with pytest.raises(ExcitationSourceError):request_from_dict(d)


def test_source_mutation_missing_sidecars_and_inventory_refused():
    raw,ds,r,b=source()
    with pytest.raises(ExcitationSourceError):verify_excitation_source(raw,r,b)
    with pytest.raises(ExcitationSourceError):verify_excitation_source(ds,None,b)
    with pytest.raises(ExcitationSourceError):verify_excitation_source(ds,r,None)
    for changed_r in (replace(r,rho=r.rho+1),replace(r,water_depth=r.water_depth+1)):
        with pytest.raises(ExcitationSourceError):verify_excitation_source(ds,changed_r,b)
    for changed_b in ((replace(b[0],rotation_center=(1.,0.,0.)),b[1]),(b[0],b[0]),b[:1]):
        with pytest.raises(ExcitationSourceError):verify_excitation_source(ds,r,changed_b)
    changed=ds.copy(deep=True);changed.excitation_force.values[0,0,0]+=1j
    with pytest.raises(ExcitationSourceError):verify_excitation_source(changed,r,b)
    for key,value in (('qualification_status','qualified'),('backend_version','unknown')):
        changed=ds.copy(deep=True);changed.attrs[key]=value
        with pytest.raises(ExcitationSourceError):verify_excitation_source(changed,r,b)


@pytest.mark.parametrize('key,value',[('version','unknown'),('conventions',{}),('convention_evidence','')])
def test_unknown_or_contradictory_contract_refused(key,value):
    raw,ds,r,b=source();c=json.loads(ds.attrs[EXCITATION_SOURCE_ATTR]);c[key]=value
    ds.attrs[EXCITATION_SOURCE_ATTR]=json.dumps(c)
    with pytest.raises(ExcitationSourceError):verify_excitation_source(ds,r,b)


@pytest.mark.parametrize('phase',[0.,math.pi/2])
@pytest.mark.parametrize('heading,index',[(0.,0),(math.pi/2,2)])
def test_source_bound_unit_amplitude_phase_sum(phase,heading,index):
    raw,ds,r,b=source();wave=RegularWave(.4,.7,heading,phase).realize()
    series=build_source_bound_excitation_series(ds,wave,request=r,bodies=b,environment=environment(r))
    t=np.array([0.,.1,.4]);x=raw.excitation_force.values[1,index]
    expected=.4*np.real(x[None,:]*np.exp(1j*(phase-.7*t[:,None])))
    np.testing.assert_allclose(series.evaluate(t),expected,rtol=1e-12,atol=1e-12)
    assert series.hydro_content_hash==dataset_content_hash(ds)
    assert series.source_qualification_status=='development_fixture'
    for key,value in (('rho',1024.),('still_water_level',1.),('current_policy','forward_speed')):
        env=environment(r);env[key]=value
        with pytest.raises(ExcitationSourceError):build_source_bound_excitation_series(ds,wave,request=r,bodies=b,environment=env)


def test_public_backend_emits_version_scoped_declaration_without_solve(monkeypatch):
    from anyhydro.capytaine_backend import CapytaineBackend
    raw=make_two_body_fixture();raw.attrs.update(backend='capytaine',backend_version='3.0.0')
    # This injected backend result is explicitly fixture data, never BEM evidence.
    backend=CapytaineBackend();r=replace(make_fixture_request(),backend_settings={},omegas=[.7],wave_directions=[0.]);b=make_fixture_bodies()
    raw=raw.sel(omega=[.7],wave_direction=[0.]);raw.attrs['request_hash']=r.content_hash()
    monkeypatch.setattr(backend,'_admit_geometry',lambda *a, **kwargs:None)
    monkeypatch.setattr(backend,'_solve_fresh',lambda *a:raw.copy(deep=True))
    ds=backend.solve(b,r);assert verify_excitation_source(ds,r,b)
    assert ds.attrs['qualification_status']=='development_fixture'
    raw.attrs['backend_version']='3.0.1'
    assert EXCITATION_SOURCE_ATTR not in backend.solve(b,r).attrs


def test_hydrostatic_attachment_order_preserves_both_contract_bindings():
    from anyhydro import HydrostaticContract, attach_hydrostatic_contract, hydrostatic_contract_binding_matches
    raw,ds,r,b=source()
    hydro=HydrostaticContract(role='BUOYANCY_ONLY',force_basis='GENERALIZED_DOF',
        stiffness_derivative='INCREMENTAL_ABOUT_EQUILIBRIUM',bodies=tuple(x.body_id for x in b),
        baseline_wrench=None,provenance={'source':'explicit transport test'},full_source=None)
    combined=attach_hydrostatic_contract(ds,hydro)
    assert verify_excitation_source(combined,r,b) and hydrostatic_contract_binding_matches(combined)
    with pytest.raises(ExcitationSourceError):
        attach_excitation_source(attach_hydrostatic_contract(raw,hydro),r,b,convention_evidence='rebuild required')


@pytest.mark.parametrize('value',[True,float('nan'),object(),None])
def test_environment_malformed_values_are_typed_refusals(value):
    raw,ds,r,b=source();env=environment(r);env['gravity']=value
    with pytest.raises(ExcitationSourceError):
        build_source_bound_excitation_series(ds,RegularWave(.4,.7).realize(),request=r,bodies=b,environment=env)


def test_integer_environment_values_and_invalid_attach_inventory():
    raw,ds,r,b=source();env=environment(r);env['rho']=1025;env['water_depth']=150;env['still_water_level']=0
    assert build_source_bound_excitation_series(ds,RegularWave(.4,.7).realize(),request=r,bodies=b,environment=env)
    with pytest.raises(ExcitationSourceError):attach_excitation_source(raw,r,None,convention_evidence='invalid')


@pytest.mark.parametrize('subset',[True,False])
def test_public_solve_preserves_subset_and_reordered_enabled_dofs(monkeypatch,subset):
    from anyhydro.capytaine_backend import CapytaineBackend
    r=replace(make_fixture_request(),backend_settings={},omegas=[.7],wave_directions=[0.])
    b=make_fixture_bodies();keys=b[0].enabled_dofs[:1] if subset else b[0].enabled_dofs[::-1]
    b=(replace(b[0],enabled_dofs=keys),b[1])
    raw=make_two_body_fixture().sel(omega=[.7],wave_direction=[0.])
    raw.attrs.update(backend='capytaine',backend_version='3.0.0',request_hash=r.content_hash())
    inventory=json.loads(raw.attrs['body_identity']);inventory[b[0].body_id]['enabled_dofs']=list(keys)
    raw.attrs['body_identity']=json.dumps(inventory)
    backend=CapytaineBackend()
    monkeypatch.setattr(backend,'_admit_geometry',lambda *a, **kwargs:None)
    monkeypatch.setattr(backend,'_solve_fresh',lambda *a:raw.copy(deep=True))
    result=backend.solve(b,r)
    assert EXCITATION_SOURCE_ATTR not in result.attrs
    assert dataset_content_hash(result)==dataset_content_hash(raw)


def test_declared_source_fresh_warm_public_cache_identity(tmp_path,monkeypatch):
    import anyhydro.capytaine_backend as cb
    from test_cache_identity_repair import _dev_bodies, _dev_request, _mocked_solve_fresh
    b=_dev_bodies();r=_dev_request();backend=cb.CapytaineBackend(cache_dir=tmp_path)
    monkeypatch.setattr(cb,'_capytaine_version',lambda:'3.0.0')
    monkeypatch.setattr(cb,'_require_capytaine',lambda:pytest.fail('real BEM forbidden'))
    monkeypatch.setattr(cb.CapytaineBackend,'_solve_fresh',_mocked_solve_fresh)
    fresh=backend.solve(b,r);assert verify_excitation_source(fresh,r,b)
    files=list(tmp_path.glob('*.nc'));assert len(files)==1
    stored=load_hydro_dataset(files[0]);assert 'cache_identity' in stored.attrs
    monkeypatch.setattr(cb.CapytaineBackend,'_solve_fresh',lambda *a:pytest.fail('warm cache must avoid solve'))
    warm=backend.solve(b,r);assert 'cache_identity' not in warm.attrs
    assert verify_excitation_source(warm,r,b)
    assert dataset_content_hash(warm)==dataset_content_hash(fresh)


def test_ordered_source_inventory_has_distinct_cache_identity(tmp_path,monkeypatch):
    import anyhydro.capytaine_backend as cb
    from test_cache_identity_repair import _dev_bodies, _dev_request, _mocked_solve_fresh
    canonical=_dev_bodies();reversed_b=(replace(canonical[0],enabled_dofs=canonical[0].enabled_dofs[::-1]),)
    r=_dev_request();backend=cb.CapytaineBackend(cache_dir=tmp_path);calls=[]
    monkeypatch.setattr(cb,'_capytaine_version',lambda:'3.0.0')
    monkeypatch.setattr(cb,'_require_capytaine',lambda:pytest.fail('real BEM forbidden'))
    def fresh(self,bodies,request,settings):
        calls.append(tuple(bodies[0].enabled_dofs))
        return _mocked_solve_fresh(self,bodies,request,settings)
    monkeypatch.setattr(cb.CapytaineBackend,'_solve_fresh',fresh)
    hashes={}
    for b in (canonical,reversed_b,canonical,reversed_b):
        ds=backend.solve(b,r);order=tuple(b[0].enabled_dofs)
        assert 'cache_identity' not in ds.attrs
        stored=ds.attrs['body_identity'];stored=json.loads(stored) if isinstance(stored,str) else stored
        assert tuple(stored[b[0].body_id]['enabled_dofs'])==order
        if b is canonical:assert verify_excitation_source(ds,r,b)
        else:assert EXCITATION_SOURCE_ATTR not in ds.attrs
        if order in hashes:assert dataset_content_hash(ds)==hashes[order]
        hashes[order]=dataset_content_hash(ds)
    assert len(calls)==2 and len(list(tmp_path.glob('*.nc')))==2


def test_declaration_verifier_without_caller_request():
    raw,ds,r,b=source()
    contract,restored=verify_excitation_source_declaration(ds)
    assert restored.content_hash()==r.content_hash() and contract['request']==request_to_dict(r)
    with pytest.raises(ExcitationSourceError):verify_excitation_source_declaration(raw)
    changed=ds.copy(deep=True);changed.excitation_force.values[0,0,0]+=1j
    with pytest.raises(ExcitationSourceError,match='stale'):verify_excitation_source_declaration(changed)
    for key,value in (('version','unknown'),('convention_evidence',' ')):
        c=json.loads(ds.attrs[EXCITATION_SOURCE_ATTR]);c[key]=value
        changed=ds.copy(deep=True);changed.attrs[EXCITATION_SOURCE_ATTR]=json.dumps(c)
        with pytest.raises(ExcitationSourceError):verify_excitation_source_declaration(changed)
    c=json.loads(ds.attrs[EXCITATION_SOURCE_ATTR]);c['conventions']['phase_origin_xy_m']=[1.,0.]
    changed=ds.copy(deep=True);changed.attrs[EXCITATION_SOURCE_ATTR]=json.dumps(c)
    with pytest.raises(ExcitationSourceError,match='convention'):verify_excitation_source_declaration(changed)
    c=json.loads(ds.attrs[EXCITATION_SOURCE_ATTR]);c['request']['water_depth']+=1.
    changed=ds.copy(deep=True);changed.attrs[EXCITATION_SOURCE_ATTR]=json.dumps(c)
    with pytest.raises(ExcitationSourceError,match='request hash'):verify_excitation_source_declaration(changed)
    with pytest.raises(ExcitationSourceError,match='request'):verify_excitation_source(changed,r,b)
