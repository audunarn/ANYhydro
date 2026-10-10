# SPDX-License-Identifier: MPL-2.0
from dataclasses import replace
from pathlib import Path
import hashlib,json,math
import numpy as np
import pytest
from test_hams_import import case,run,rehash_inputs,independent_map,sha
from anyfileio.hams import parse_control,render_control
from anyhydro.hams_import import GRAVITY_PROFILE,HamsImportError,_wave_number
from anyhydro.hams_producer import GRAVITY_SOURCE_IDENTITY
from anyhydro.hams_field_points import admit_field_points,normalize_field_points,HamsFieldPointError
from anyhydro.hams_fields import HamsFieldError,HamsFieldData,import_hams_fields,save_hams_fields,load_hams_fields,reframe_hams_fields
from anyhydro.frames import reframe_dataset
from anyhydro.hashing import dataset_content_hash
from anyhydro.errors import CapabilityError


@pytest.fixture
def field_case(case):
    points=((10.,8.,0.),(10.,8.,-2.))
    p=case['inputs']/'ControlFile.in';control=parse_control(p.read_text());p.write_text(render_control(replace(control,field_points=points)));rehash_inputs(case)
    def f(value):return f'{value:14.6E}'
    source=np.arange(1,49).reshape(2,2,12)/10+1j*np.arange(48,0,-1).reshape(2,2,12)/10
    total=np.zeros((2,2,2),complex);incident=np.zeros_like(total);rows=[];incrows=[]
    for k,w in enumerate(case['request'].omegas):
        for pindex in range(2):
            for body in range(2):
                header=f(w)+f'{pindex+1:10d}' if body==0 else ' '*24
                values=source[k,pindex,body*6:body*6+6]
                rows.append(header+''.join(f(v) for z in values for v in (z.real,z.imag)))
        for h,theta in enumerate(case['request'].wave_directions):
            for pindex,(x,y,z) in enumerate(points):
                wave=_wave_number(w,case['request'].water_depth)
                raw=math.cosh(wave*(z+50))/math.cosh(wave*50)*np.exp(-1j*wave*(x*math.cos(theta)+y*math.sin(theta)))
                raw=complex(float(np.float32(raw.real)),float(np.float32(raw.imag)))
                incrows.append(f(w)+f(math.degrees(theta))+f'{pindex+1:10d}'+''.join(f(v) for v in (abs(raw),math.degrees(math.atan2(raw.imag,raw.real)),raw.real,raw.imag)))
                incident[k,h,pindex]=complex(float(f(raw.real)),float(f(raw.imag)))
                total[k,h,pindex]=complex((k+1)*(h+1)*(pindex+1)/4,-(k+h+pindex+1)/8)
                t=total[k,h,pindex]
                rows.append(f(w)+f(math.degrees(theta))+f'{pindex+1:10d}'+f(t.real)+f(t.imag))
    paths={'combined':case['tmp']/'fields.6p','incident':case['tmp']/'incident.6p'} if 'tmp' in case else {'combined':case['paths']['1'].parent/'fields.6p','incident':case['paths']['1'].parent/'incident.6p'}
    paths['combined'].write_text('\n'.join(rows)+'\n');paths['incident'].write_text('\n'.join(incrows)+'\n')
    case['contract']=replace(case['contract'],source_tree_sha256=GRAVITY_SOURCE_IDENTITY,profile=GRAVITY_PROFILE,source_field_points=points,field_file_sha256={k:sha(p) for k,p in paths.items()})
    case.update(fieldpaths=paths,points=points,source_fields=source,total_native=total,incident_native=incident)
    return case


def fields(c,dataset=None):
    ds=run(c) if dataset is None else dataset
    return import_hams_fields(c['bodies'],c['request'],c['contract'],dataset=ds,input_directory=c['inputs'],executable_path=c['exe'],native_directory=c['native'],
        radiation_path=c['paths']['1'],excitation_path=c['paths']['3'],hydrostatic_path=c['paths']['hst'],combined_path=c['fieldpaths']['combined'],incident_path=c['fieldpaths']['incident'])


def test_units_phase_rotational_scaling_permutation_reference_shift_and_total(field_case):
    c=field_case;data=fields(c);scale=np.array([1.,9806.65]);local=np.tile([1,1,1,2.5,2.5,2.5],2)
    expected=(c['source_fields'].conjugate()*scale[None,:,None]*local[None,None,:])@independent_map(c)
    np.testing.assert_allclose(data.radiation,expected,rtol=1e-14,atol=1e-10)
    np.testing.assert_allclose(data.incident,c['incident_native'].conjugate()*scale,rtol=0,atol=0)
    np.testing.assert_allclose(data.total,c['total_native'].conjugate()*scale,rtol=0,atol=0)
    assert np.array_equal(data.scattered,data.total-data.incident)
    assert data.quantities==('elevation','dynamic_pressure') and data.dof_keys[:6]==c['bodies'][0].enabled_dofs
    assert data.metadata['pressure_kind']=='dynamic_wave_pressure_excluding_hydrostatic'
    with pytest.raises(ValueError):data.radiation[0,0,0]=0
    for array in (data.radiation,data.incident,data.total,data.scattered):
        with pytest.raises(ValueError):array.setflags(write=True)
    with pytest.raises(TypeError):data.metadata['target_frames']['body-a']={}
    old=data.content_hash();c['points']=((0.,0.,0.),);assert data.content_hash()==old


def test_rotation_right_map_and_parent_binding(field_case):
    c=field_case;ds=run(c);data=fields(c,ds);keys=[b.body_id for b in c['bodies']]
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    axes={key:rotation for key in keys};centers={key:tuple(np.asarray(c['bodies'][i].rotation_center)+[.2,-.3,.4]) for i,key in enumerate(keys)}
    new=reframe_dataset(ds,new_axes=axes,new_reference_centers=centers)
    mapped=reframe_hams_fields(data,ds,new,new_axes=axes,new_reference_centers=centers)
    T=np.zeros((12,12));x,y,z=.2,-.3,.4;skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
    for i in range(2):T[6*i:6*i+3,6*i:6*i+3]=rotation;T[6*i:6*i+3,6*i+3:6*i+6]=skew@rotation;T[6*i+3:6*i+6,6*i+3:6*i+6]=rotation
    np.testing.assert_allclose(mapped.radiation,data.radiation@T,rtol=1e-14,atol=1e-9)
    assert np.array_equal(mapped.incident,data.incident) and mapped.parent_dataset_hash==dataset_content_hash(new)
    with pytest.raises(ValueError):mapped.radiation.setflags(write=True)
    with pytest.raises(HamsFieldError,match='New field parent'):reframe_hams_fields(data,ds,ds,new_axes=axes,new_reference_centers=centers)


def test_save_reopen_parent_quantity_missing_and_resealed_semantics(field_case,tmp_path):
    data=fields(field_case);p=tmp_path/'fields.json';save_hams_fields(data,p)
    stored=load_hams_fields(p,expected_parent_hash=data.parent_dataset_hash);assert data.content_hash()==stored.content_hash()
    with pytest.raises(ValueError):stored.radiation.setflags(write=True)
    with pytest.raises(HamsFieldError,match='New field artifact'):save_hams_fields(data,p)
    with pytest.raises(HamsFieldError,match='parent'):load_hams_fields(p,expected_parent_hash='0'*64)
    raw=json.loads(p.read_text());raw['payload']['quantities'][0]='dynamic_pressure';p.write_text(json.dumps(raw))
    with pytest.raises(HamsFieldError,match='identity/quantity'):load_hams_fields(p)
    with pytest.raises(HamsFieldError,match='Incomplete'):load_hams_fields(tmp_path/'missing.json')


@pytest.mark.parametrize('kind',['combined','incident'])
def test_raw_hash_or_complete_inventory_refuses(field_case,kind):
    c=field_case;p=c['fieldpaths'][kind];raw=p.read_text();p.write_text(raw.rsplit('\n',2)[0]+'\n')
    with pytest.raises(HamsFieldError,match='Raw field'):fields(c)
    c['contract']=replace(c['contract'],field_file_sha256={k:sha(p) for k,p in c['fieldpaths'].items()})
    with pytest.raises(HamsFieldError,match='layout'):fields(c)


def test_wrong_incident_redundancy_and_stale_parent_refuse(field_case):
    c=field_case;ds=run(c);bad=ds.copy(deep=True);bad['added_mass'].values[0,0,0]+=1
    with pytest.raises(HamsFieldError,match='parent'):fields(c,bad)
    p=c['fieldpaths']['incident'];rows=p.read_text().splitlines();rows[0]=rows[0][:38]+f'{2.:14.6E}'+rows[0][52:];p.write_text('\n'.join(rows)+'\n')
    c['contract']=replace(c['contract'],field_file_sha256={k:sha(p) for k,p in c['fieldpaths'].items()})
    with pytest.raises(HamsFieldError,match='Incident normalization'):fields(c)


def test_old_source_and_new_source_limits_remain_refused(field_case):
    c=field_case
    from anyhydro.hams_import import PROFILE
    from anyhydro.hams_producer import SOURCE_IDENTITIES
    with pytest.raises(HamsImportError,match='Fields require'):replace(c['contract'],source_tree_sha256=next(iter(SOURCE_IDENTITIES)),profile=PROFILE)
    request=replace(c['request'],include_infinite_frequency=True)
    c['contract']=replace(c['contract'],source_request_hash=request.content_hash());c['request']=request
    with pytest.raises(CapabilityError,match='Finite-only'):fields(c)


@pytest.mark.parametrize('point',[(10,8,1e-9),(10,8,-1e-7),(10,8,-1e-6),(10,8,-50),(10,8,-51)])
def test_near_surface_above_water_or_bed_refuses(field_case,point):
    with pytest.raises(HamsFieldPointError):admit_field_points(field_case['bodies'],field_case['request'],(point,))


def test_aabb_and_native_tolerant_onbody_refuse(field_case):
    c=field_case;body=c['bodies'][0];vertices=body.mesh.vertices;center=(vertices.min(0)+vertices.max(0))/2
    with pytest.raises(HamsFieldPointError,match='AABB'):admit_field_points(c['bodies'],c['request'],(tuple(center),))
    point=(float(vertices[:,0].min())-1e-7,float(center[1]),float(center[2]))
    assert point[0]<vertices[:,0].min()
    with pytest.raises(HamsFieldPointError,match='ONBODY'):admit_field_points(c['bodies'],c['request'],(point,))
    assert admit_field_points(c['bodies'],c['request'],c['points'])==c['points']


@pytest.mark.parametrize('points',[[[True,8,0]],[[10,8,float('nan')]],[[10,8,0],[10,8,0]],np.asarray(1),[[10,8]]])
def test_point_identity_and_shape_refuse(points):
    with pytest.raises(HamsFieldPointError):normalize_field_points(points)

@pytest.mark.parametrize('which',['frames','frequencies','headings','order'])
def test_reframe_checks_parent_coordinates_before_mapping(field_case,which):
    c=field_case;ds=run(c);data=fields(c,ds);meta=json.loads(json.dumps(data.payload(),default=lambda value:dict(value)))['metadata']
    if which=='frames':meta['target_frames'][meta['target_body_order'][0]]['reference_center'][0]+=.1;data=replace(data,metadata=meta)
    if which=='frequencies':data=replace(data,omegas=tuple(w+.01 for w in data.omegas))
    if which=='headings':data=replace(data,headings=tuple(h+.01 for h in data.headings))
    if which=='order':
        meta['target_body_order']=list(reversed(meta['target_body_order']));keys=tuple(k for b in meta['target_body_order'] for k in c['bodies'][[x.body_id for x in c['bodies']].index(b)].enabled_dofs);data=replace(data,metadata=meta,dof_keys=keys)
    with pytest.raises(HamsFieldError,match='coordinates/frames'):reframe_hams_fields(data,ds,ds,new_axes=None,new_reference_centers=None)

def test_mutated_contract_axes_cannot_enter_field_import(field_case):
    c=field_case;frame=next(iter(c['contract'].source_frames.values()));frame.axes.setflags(write=True);frame.axes[:]=[[0,-1,0],[1,0,0],[0,0,1]]
    with pytest.raises(HamsImportError,match='WORLD frame changed'):fields(c)
