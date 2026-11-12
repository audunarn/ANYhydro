import json
import math
import numpy as np
import pytest
from anyfileio.orcawave import OrcaWavePressureDocument
from anyhydro.orcawave_pressure import import_pressure_bank, ImportedPressureBank


def document():
    return OrcaWavePressureDocument(np.array([[[0,0,-1],[1,0,-1],[1,1,-1],[0,1,-1]]]*3),
        [[.5,.5,-1]]*3,[[0,0,1]]*3,[1]*3,(0,1,0),(3,4,5),(1,1,2),(0,0,1),[3.,6.],[0.],
        ('fixed_diffraction','total_with_source_motion'),np.ones((2,1,2,3)),np.ones((2,1,2,3))*2,
        json.dumps(dict(source_sha256='a'*64,units=dict(length='m',mass='te',force='kN'),water_depth=None,
            rho=1.025,gravity=9.81,phasor='exp(+i omega t)',normals='out of fluid',phase_origin=[0,0,0])))


def test_conversion_and_body_lid_selection():
    bank=import_pressure_bank(document(),body_id=0,heading_degrees=0)
    assert bank.panel_ids==(3,); assert bank.surface_ids==('body:0',)
    assert bank.rho==1025 and bank.water_depth is None
    assert np.allclose(bank.responses,1000-2000j)
    assert np.array_equal(bank.normals,[[0,0,-1]])
    p=bank.vertices[bank.faces[0]]
    assert np.dot(np.cross(p[1]-p[0],p[2]-p[0]),bank.normals[0])>0
    # At quarter period provider Re[(1+2i)*i] is -2 kPa, canonical yields same.
    assert np.real(bank.responses[0,0]*np.exp(-1j*math.pi/2))==pytest.approx(-2000.)
    assert ImportedPressureBank.from_dict(bank.to_dict()).bank_id==bank.bank_id
    assert not bank.responses.flags.writeable


def test_total_distinct_and_refusals():
    d=document(); bank=import_pressure_bank(d,body_id=1,heading_degrees=0,composition='total_with_source_motion')
    assert bank.composition=='total_with_source_motion'
    with pytest.raises(ValueError,match='heading'):import_pressure_bank(d,body_id=0,heading_degrees=1)
    with pytest.raises(ValueError,match='no physical'):import_pressure_bank(d,body_id=9,heading_degrees=0)
    data=d._payload(); data['panel_types']=(2,0,1)
    jump=import_pressure_bank(OrcaWavePressureDocument(**data),body_id=0,heading_degrees=0)
    assert jump.kinds==('dipole_jump',) and jump.to_dict()['type']=='imported_pressure_bank_v3'


def test_non_si_conversion_and_identity():
    data=document()._payload(); metadata=json.loads(data['metadata_json'])
    metadata['units']=dict(length='ft',mass='lb',force='lbf');metadata['water_depth']=10
    data['metadata_json']=json.dumps(metadata)
    bank=import_pressure_bank(OrcaWavePressureDocument(**data),body_id=0,heading_degrees=0)
    assert bank.responses[0,0]==pytest.approx((1-2j)*4.4482216152605/.3048**3)
    assert bank.areas[0]==pytest.approx(.3048**2)
    assert bank.rho==pytest.approx(1.025*.45359237/.3048**3)
    assert bank.water_depth==pytest.approx(3.048)
    stored=bank.to_dict();stored['heading']=.5
    with pytest.raises(ValueError,match='identity'):ImportedPressureBank.from_dict(stored)


def test_triangular_panel_repeats_index_and_preserves_orientation():
    data=document()._payload()
    data['vertices'][0]=[[0,0,-1],[1,0,-1],[0,1,-1],[0,1,-1]]
    data['areas'][0]=.5;data['centers'][0]=[1/3,1/3,-1]
    bank=import_pressure_bank(OrcaWavePressureDocument(**data),body_id=0,heading_degrees=0)
    assert bank.faces[0,2]==bank.faces[0,3]
    assert len(set(bank.faces[0]))==3
    poly=bank.vertices[bank.faces[0,:3]]
    assert np.dot(np.cross(poly[1]-poly[0],poly[2]-poly[0]),bank.normals[0])>0
    assert bank.panel_ids==(3,)


def test_raw_winding_is_not_assumed():
    data=document()._payload()
    data['vertices'][0]=data['vertices'][0][::-1]
    bank=import_pressure_bank(OrcaWavePressureDocument(**data),body_id=0,heading_degrees=0)
    poly=bank.vertices[bank.faces[0]]
    assert np.dot(np.cross(poly[1]-poly[0],poly[2]-poly[0]),bank.normals[0])>0
    assert bank.responses[0,0]==1000-2000j


@pytest.mark.parametrize('field,value,message',[
    ('normals',[1.,0.,0.],'normal'),('areas',1.01,'area'),('centers',[.6,.5,-1.],'centroid')])
def test_individual_panel_metadata_must_match_geometry(field,value,message):
    data=document()._payload();data[field][0]=value
    with pytest.raises(ValueError,match=message):
        import_pressure_bank(OrcaWavePressureDocument(**data),body_id=0,heading_degrees=0)


@pytest.mark.parametrize('vertices',[
    [[0,0,-1],[1,0,-1],[.2,.2,-1],[0,1,-1]],
    [[0,0,-1],[1,0,-1],[1,1,-1],[0,1,-.9]]])
def test_nonconvex_or_warped_physical_panel_refused(vertices):
    data=document()._payload();data['vertices'][0]=vertices
    with pytest.raises(ValueError):
        import_pressure_bank(OrcaWavePressureDocument(**data),body_id=0,heading_degrees=0)


@pytest.mark.parametrize('origin',[None,[1,0,0],[False,0,0]])
def test_phase_origin_is_explicit_and_fixed(origin):
    data=document()._payload();metadata=json.loads(data['metadata_json'])
    if origin is None:metadata.pop('phase_origin')
    else:metadata['phase_origin']=origin
    data['metadata_json']=json.dumps(metadata)
    with pytest.raises(ValueError,match='phase origin'):OrcaWavePressureDocument(**data)


def test_manual_and_retained_bank_geometry_cannot_bypass_metadata_checks():
    from dataclasses import replace
    bank=import_pressure_bank(document(),body_id=0,heading_degrees=0)
    bank.validate_geometry()
    malformed=replace(bank,areas=bank.areas*2,bank_id='')
    malformed.validate()  # data shape/hash validation intentionally stays data-only
    retained=ImportedPressureBank.from_dict(malformed.to_dict())
    with pytest.raises(ValueError,match='area'):retained.validate_geometry()
    reversed_normal=replace(bank,normals=-bank.normals,bank_id='')
    with pytest.raises(ValueError,match='normal'):reversed_normal.validate_geometry()


def test_dipole_jump_orientation_mixed_schema_and_legacy_identity():
    import hashlib
    from anyloads.hydro_pressure import pressure_wrench
    body=import_pressure_bank(document(),body_id=0,heading_degrees=0)
    old=body.to_dict();assert old['type']=='imported_pressure_bank_v2' and 'panel_kinds' not in old
    expected=hashlib.sha256(json.dumps(body._payload(),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert ImportedPressureBank.from_dict(old).bank_id==expected==body.bank_id
    data=document()._payload();data['panel_types']=(0,0,2)
    mixed=import_pressure_bank(OrcaWavePressureDocument(**data),body_id=0,heading_degrees=0)
    assert mixed.kinds==('body_pressure','dipole_jump')
    assert ImportedPressureBank.from_dict(mixed.to_dict()).bank_id==mixed.bank_id
    assert 'dipole_jump' in mixed.conventions['reference_conventions']
    w=pressure_wrench(mixed.centers,mixed.normals,mixed.areas,mixed.responses[0])
    data['normals'][2]=(-np.asarray(data['normals'][2])).tolist()
    for key in ('pressure_real','pressure_imag'):
        a=np.asarray(data[key]);a[:,:,:,2]*=-1;data[key]=a.tolist()
    reversed_source=import_pressure_bank(OrcaWavePressureDocument(**data),body_id=0,heading_degrees=0)
    assert np.allclose(pressure_wrench(reversed_source.centers,reversed_source.normals,reversed_source.areas,reversed_source.responses[0]),w)
    # A +1 real provider jump on a +Z reference pushes +Z once, not twice.
    assert mixed.normals[1,2]==-1 and mixed.responses[0,1]==1000-2000j
