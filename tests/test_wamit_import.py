# SPDX-License-Identifier: MPL-2.0
from dataclasses import replace
import hashlib
import json
import math
import sys

import numpy as np
import pytest

from anyhydro.backends import ImportedHydroDatasetBackend
from anyhydro.dof import body_dof_keys
from anyhydro.errors import CapabilityError
from anyhydro.excitation_source import verify_excitation_source
from anyhydro.fixtures import make_fixture_bodies
from anyhydro.frames import CoefficientFrame
from anyhydro.hashing import dataset_content_hash
from anyhydro.hydrostatic_contract import read_hydrostatic_contract, require_hydrostatic_composition, HydrostaticContractError
from anyhydro.persistence import save_hydro_dataset, load_hydro_dataset
from anyhydro.records import HydroRequest
from anyhydro.schema import validate_for_time_domain
from anyhydro.wamit_import import Wamit7ImportContract, WamitImportError, import_wamit7


def hashes(paths):
    return {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}


@pytest.fixture
def case(tmp_path):
    original=make_fixture_bodies();bodies=tuple(reversed(original))
    request=HydroRequest([.5,1.25],[0,.7],50.,1025.,9.81,False,True,{})
    n=12;omega=request.omegas;L=2.5
    # Deliberately dense asymmetric cross blocks; never construct a diagonal oracle.
    a=np.arange(1,n*n+1).reshape(n,n)/100
    b=np.arange(n*n,0,-1).reshape(n,n)/70
    c=np.arange(1,n*n+1).reshape(n,n)/90
    f=(np.arange(1,n+1)+1j*np.arange(n,0,-1))/11
    paths={k:tmp_path/('coefficient.'+k) for k in ('1','3','hst')}
    rows=[]
    for k,w in enumerate(omega):
        for i in range(n):
            for j in range(n):rows.append(f'{2*math.pi/w:.12E} {i+1} {j+1} {a[i,j]*(k+1):.12E} {b[i,j]*(k+1):.12E}')
    for i in range(n):
        for j in range(n):rows.append(f'0.00000000E+00 {i+1} {j+1} {3*a[i,j]:.12E}')
    paths['1'].write_text('\n'.join(rows)+'\n')
    rows=[]
    for k,w in enumerate(omega):
        for h,theta in enumerate(request.wave_directions):
            for i in range(n):
                z=f[i]*(k+1)*(h+1)
                rows.append(f'{2*math.pi/w:.12E} {math.degrees(theta):.12E} {i+1} {abs(z):.12E} {math.degrees(math.atan2(z.imag,z.real)):.12E} {z.real:.12E} {z.imag:.12E}')
    paths['3'].write_text('\n'.join(rows)+'\n')
    paths['hst'].write_text('\n'.join(f'{i+1} {j+1} {c[i,j]:.12E}' for i in range(n) for j in range(n))+'\n')
    frames={}
    for index,body in enumerate(original):
        angle=.3*(index+1);s,co=math.sin(angle),math.cos(angle)
        axes=np.array([[co,-s,0],[s,co,0],[0,0,1.]])
        center=np.asarray(body.rotation_center)+[.4,-.7,.2]
        frames[body.body_id]=CoefficientFrame(axes,tuple(center))
    contract=Wamit7ImportContract(tuple(b.body_id for b in original),frames,
        {b.body_id:b.mesh.mesh_hash for b in original},{b.body_id:None for b in original},
        {b.body_id:(float(index),0.,-1.) for index,b in enumerate(original)},L,request.rho,request.gravity,
        request.water_depth,request.content_hash(),'1'*64,'2'*64,hashes(paths),
        '7.5-development-fixture','synthetic-coefficient-oracle-v1','synthetic signed/phasor/dimensional/frame fixture; no physical qualification')
    return dict(bodies=bodies,original=original,request=request,contract=contract,paths=paths,a=a,b=b,c=c,f=f)


def run(case, **changes):
    return import_wamit7(case['bodies'],case['request'],changes.pop('contract',case['contract']),
        radiation_path=case['paths']['1'],excitation_path=case['paths']['3'],hydrostatic_path=case['paths']['hst'],**changes)


def rehash(case):
    case['contract']=replace(case['contract'],file_sha256=hashes(case['paths']))


def independent_map(case):
    # q_source = R.T*[v_target - skew(r_source-r_target)*w_target, w_target].
    # This oracle does not call the production owner's frame-map helper.
    T=np.zeros((12,12))
    for s,body in enumerate(case['original']):
        t=case['bodies'].index(body);frame=case['contract'].source_frames[body.body_id]
        x,y,z=np.asarray(frame.reference_center)-np.asarray(body.rotation_center)
        skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
        block=np.block([[frame.axes.T,-frame.axes.T@skew],[np.zeros((3,3)),frame.axes.T]])
        T[s*6:s*6+6,t*6:t*6+6]=block
    return T


def test_complete_normalization_replay_and_roundtrip(case,tmp_path):
    ds=run(case);T=independent_map(case);contract=case['contract'];request=case['request']
    exponents=np.tile([0,0,0,1,1,1],2)
    mass=contract.source_rho*contract.reference_length_m**(3+exponents[:,None]+exponents[None,:])
    stiffness=contract.source_rho*contract.source_gravity*contract.reference_length_m**(2+exponents[:,None]+exponents[None,:])
    force=contract.source_rho*contract.source_gravity*contract.reference_length_m**(2+exponents)
    for k,w in enumerate(request.omegas):
        np.testing.assert_allclose(ds.added_mass[k],T.T@(case['a']*(k+1)*mass)@T,rtol=2e-8,atol=1e-8)
        np.testing.assert_allclose(ds.radiation_damping[k],T.T@(case['b']*(k+1)*mass*w)@T,rtol=2e-8,atol=1e-8)
        for h in range(2):
            np.testing.assert_allclose(ds.excitation_force[k,h],T.T@(case['f'].conjugate()*(k+1)*(h+1)*force),rtol=2e-8,atol=1e-7)
    np.testing.assert_allclose(ds.hydrostatic_stiffness,T.T@(case['c']*stiffness)@T,rtol=2e-8,atol=1e-7)
    np.testing.assert_allclose(ds.added_mass_infinite,T.T@(3*case['a']*mass)@T,rtol=2e-8,atol=1e-8)
    assert not np.allclose(ds.added_mass[0],ds.added_mass[0].T)
    assert ds.attrs['qualification_status']=='unqualified'
    assert list(ds.body.values)==[b.body_id for b in case['bodies']]
    np.testing.assert_array_equal(ds.rotation_center,[b.rotation_center for b in case['bodies']])
    verify_excitation_source(ds,request,case['bodies'])
    assert read_hydrostatic_contract(ds).role=='UNKNOWN'
    with pytest.raises(HydrostaticContractError):require_hydrostatic_composition(ds)
    validate_for_time_domain(ds)  # explicit Ainf transport only, no physical source admission
    path=tmp_path/'imported.nc';save_hydro_dataset(ds,path);restored=load_hydro_dataset(path)
    assert dataset_content_hash(restored)==dataset_content_hash(ds)
    verify_excitation_source(restored,request,case['bodies'])
    backend=ImportedHydroDatasetBackend(restored)
    replay=backend.solve(case['bodies'],request)
    assert dataset_content_hash(replay)==dataset_content_hash(ds)
    replay.added_mass.values[:]=0
    assert dataset_content_hash(backend.dataset)==dataset_content_hash(ds)
    assert 'period0' in ds.added_mass_infinite.attrs['source']
    assert contract.file_sha256['1'] in ds.added_mass_infinite.attrs['source']


def test_no_limit_damping_fabricated_and_absent_ainf_refuses_time_domain(case):
    request=replace(case['request'],include_infinite_frequency=False)
    case['request']=request
    p=case['paths']['1'];p.write_text('\n'.join(x for x in p.read_text().splitlines() if not x.startswith('0.'))+'\n')
    case['contract']=replace(case['contract'],source_request_hash=request.content_hash(),file_sha256=hashes(case['paths']))
    ds=run(case)
    assert 'added_mass_infinite' not in ds
    with pytest.raises(CapabilityError):validate_for_time_domain(ds)


@pytest.mark.parametrize('kind',['1','3','hst'])
def test_missing_and_duplicate_rows_refuse_with_line_diagnostics(case,kind):
    p=case['paths'][kind];original=p.read_text();p.write_text('\n'.join(original.splitlines()[1:])+'\n');rehash(case)
    with pytest.raises(WamitImportError) as caught:run(case)
    assert any(d.code=='WAMIT_MISSING' for d in caught.value.diagnostics)
    p.write_text(original+original.splitlines()[0]+'\n');rehash(case)
    with pytest.raises(WamitImportError) as caught:run(case)
    assert any(d.code=='WAMIT_DUPLICATE' and d.line_start for d in caught.value.diagnostics)


def test_corrupt_phase_never_uses_rectangular_fallback(case):
    p=case['paths']['3'];rows=p.read_text().splitlines();fields=rows[0].split();fields[4]='9.99900000E+03';rows[0]=' '.join(fields)
    p.write_text('\n'.join(rows)+'\n');rehash(case)
    before=hashes(case['paths'])
    with pytest.raises(WamitImportError) as caught:run(case)
    assert any(d.code=='WAMIT_PHASE' for d in caught.value.diagnostics)
    assert hashes(case['paths'])==before


@pytest.mark.parametrize('field',['source_request_hash','input_sha256','executable_sha256'])
def test_missing_identity_contract_refuses(case,field):
    with pytest.raises(WamitImportError):replace(case['contract'],**{field:''})


def test_wrong_source_files_mesh_request_and_fluid_refuse(case):
    with pytest.raises(WamitImportError):run(case,contract=replace(case['contract'],file_sha256={'1':'a'*64,**{k:v for k,v in case['contract'].file_sha256.items() if k!='1'}}))
    with pytest.raises(WamitImportError):run(case,contract=replace(case['contract'],source_mesh_hashes={b.body_id:'a'*64 for b in case['bodies']}))
    with pytest.raises(WamitImportError):run(case,contract=replace(case['contract'],source_request_hash='a'*64))
    with pytest.raises(WamitImportError):run(case,contract=replace(case['contract'],source_gravity=10.))


def test_missing_cb_invalid_axes_and_hams_profile_refuse(case):
    with pytest.raises(WamitImportError):replace(case['contract'],center_of_buoyancy={})
    first=case['original'][0].body_id
    pitch=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
    frames={**case['contract'].source_frames,first:CoefficientFrame(pitch,(0,0,0))}
    with pytest.raises(CapabilityError):replace(case['contract'],source_frames=frames)
    with pytest.raises(CapabilityError):replace(case['contract'],profile='hams_v2.2_ouft3')


def test_negative_period_and_requested_zero_limit_refuse(case):
    p=case['paths']['1'];p.write_text(p.read_text()+'-1.00000000E+00 1 1 2.00000000E+00\n');rehash(case)
    with pytest.raises(CapabilityError):run(case)
    request=replace(case['request'],include_zero_frequency=True);case['request']=request
    case['contract']=replace(case['contract'],source_request_hash=request.content_hash())
    with pytest.raises(CapabilityError):run(case)


def test_missing_ainf_and_nonzero_limit_damping_refuse(case):
    p=case['paths']['1'];original=p.read_text()
    p.write_text('\n'.join(x for x in original.splitlines() if not x.startswith('0.'))+'\n');rehash(case)
    with pytest.raises(WamitImportError):run(case)
    rows=original.splitlines();index=next(i for i,x in enumerate(rows) if x.startswith('0.'));rows[index]+=' 1.00000000E+00'
    p.write_text('\n'.join(rows)+'\n');rehash(case)
    with pytest.raises(WamitImportError):run(case)


def test_duplicate_circular_heading_bins_refuse(case):
    request=replace(case['request'],wave_directions=[0,2*math.pi]);case['request']=request
    case['contract']=replace(case['contract'],source_request_hash=request.content_hash())
    with pytest.raises(WamitImportError,match='ambiguous'):run(case)


def test_overlapping_printed_period_bins_refuse(case):
    request=replace(case['request'],omegas=[.5,.5+1e-15]);case['request']=request
    case['contract']=replace(case['contract'],source_request_hash=request.content_hash())
    with pytest.raises(WamitImportError,match='ambiguous'):run(case)


def test_optional_syntax_dependency_failure_is_typed(case,monkeypatch):
    monkeypatch.setitem(sys.modules,'anyfileio.wamit',None)
    with pytest.raises(CapabilityError,match='optional'):run(case)


def test_nonstandard_modes_and_foreign_headings_refuse(case):
    case['bodies']=tuple(replace(b,enabled_dofs=(body_dof_keys(b.body_id)[0],)) for b in case['bodies'])
    with pytest.raises(CapabilityError):run(case)
    case['bodies']=tuple(reversed(case['original']))
    request=replace(case['request'],wave_directions=[1.,2.]);case['request']=request
    case['contract']=replace(case['contract'],source_request_hash=request.content_hash())
    with pytest.raises(WamitImportError):run(case)


@pytest.mark.parametrize('length',[1e-200,1e200])
def test_unrepresentable_scale_never_substitutes_zero_or_inf(case,length):
    with pytest.raises(WamitImportError):run(case,contract=replace(case['contract'],reference_length_m=length))


def test_nonzero_coefficient_underflow_refuses(case):
    p=case['paths']['1'];rows=p.read_text().splitlines()
    index=next(i for i,x in enumerate(rows) if x.split()[1:3]==['4','4'])
    fields=rows[index].split();fields[3]='1.000000000000E-100';rows[index]=' '.join(fields)
    p.write_text('\n'.join(rows)+'\n');rehash(case)
    with pytest.raises(WamitImportError):run(case,contract=replace(case['contract'],reference_length_m=1e-50))


def test_subnormal_power_cannot_be_amplified_into_a_wrong_finite_scale(case):
    request=replace(case['request'],rho=1e300);case['request']=request
    contract=replace(case['contract'],source_rho=request.rho,source_request_hash=request.content_hash(),reference_length_m=2e-65)
    with pytest.raises(WamitImportError):run(case,contract=contract)


def test_indeterminate_phase_diagnostic_survives_import_and_reopen(case,tmp_path):
    p=case['paths']['3'];rows=p.read_text().splitlines();fields=rows[0].split()
    fields[3:]=['0.000000000000E+00','9.999000000000E+03','0.000000000000E+00','0.000000000000E+00']
    rows[0]=' '.join(fields);p.write_text('\n'.join(rows)+'\n');rehash(case)
    ds=run(case);info=json.loads(ds.attrs['wamit_import'])
    assert info['source_diagnostics'][0]['code']=='WAMIT_PHASE_INDETERMINATE'
    path=tmp_path/'nearzero.nc';save_hydro_dataset(ds,path);restored=load_hydro_dataset(path)
    assert restored.attrs['wamit_import']['source_diagnostics']==info['source_diagnostics']
    assert dataset_content_hash(restored)==dataset_content_hash(ds)


@pytest.mark.parametrize('change',[{'source_frames':None},{'source_mesh_hashes':[]},{'file_sha256':None},
                                  {'source_body_order':'body'}, {'center_of_buoyancy':{} }])
def test_malformed_contract_is_typed(case,change):
    with pytest.raises(WamitImportError):replace(case['contract'],**change)
