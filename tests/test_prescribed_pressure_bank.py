"""Stored native jump transport; synthetic coefficients are not qualification."""
from dataclasses import replace
import json
import numpy as np
import pytest
import anyhydro as h
from test_surfaces import body_set,pressure_dataset
from anyhydro.excitation_source import attach_excitation_source


def native_source(fixture_bodies,fixture_request,two_body_dataset,*,shuffle=False):
    bodies=body_set(fixture_bodies);ds=pressure_dataset(bodies,two_body_dataset)
    ds.pressure_jump.values[:]=np.array([10+2j,20+4j,30+6j,40+8j])
    if shuffle:ds=ds.isel(sheet_panel=[2,0,3,1])
    ds=attach_excitation_source(ds,fixture_request,bodies,convention_evidence='synthetic sign/phase oracle')
    return bodies,ds


@pytest.mark.parametrize('shuffle',[False,True])
def test_native_jump_identity_selection_phase_and_portable(fixture_bodies,fixture_request,two_body_dataset,tmp_path,shuffle):
    bodies,ds=native_source(fixture_bodies,fixture_request,two_body_dataset,shuffle=shuffle)
    bank=h.bank_from_anyhydro_pressure_jump(ds,bodies,fixture_request,body_id=bodies[1].body_id,heading=float(ds.wave_direction.values[0]))
    np.testing.assert_allclose(bank.responses,np.tile([30+6j,40+8j],(len(bank.omegas),1)))
    np.testing.assert_allclose(bank.normals,np.tile([0,0,-1],(2,1)))
    traction=-bank.responses[0,:,None]*bank.normals*bank.areas[:,None]
    assert traction.sum(axis=0)==pytest.approx([0,0,70+14j])
    assert np.real(traction*np.exp(-1j*np.pi/2)).sum(axis=0)==pytest.approx([0,0,14])
    meta=json.loads(bank.provenance_json)
    assert meta['selected_body_id']==bodies[1].body_id and len(meta['body_ids'])==2
    assert meta['qualification_status']==ds.attrs['qualification_status']
    assert [p['face_index'] for p in meta['panel_identities']]==[0,1]
    assert bank.to_dict()['type']=='imported_pressure_bank_v3' and bank.composition=='fixed_diffraction'
    path=tmp_path/'native.json';bank.save(path);reopened=h.PrescribedPressureBank.load(path)
    assert reopened.bank_id==bank.bank_id


def test_native_source_requires_phase_inventory_and_exact_requests(fixture_bodies,fixture_request,two_body_dataset):
    bodies,ds=native_source(fixture_bodies,fixture_request,two_body_dataset)
    kwargs=dict(body_id=bodies[0].body_id,heading=float(ds.wave_direction.values[0]))
    broken=ds.copy(deep=True);del broken.attrs['excitation_source_contract']
    with pytest.raises(h.HydroError):h.bank_from_anyhydro_pressure_jump(broken,bodies,fixture_request,**kwargs)
    changed=ds.copy(deep=True);changed.pressure_jump.values[0,0,0]+=1
    with pytest.raises(h.HydroError,match='stale'):h.bank_from_anyhydro_pressure_jump(changed,bodies,fixture_request,**kwargs)
    with pytest.raises(h.HydroError):h.bank_from_anyhydro_pressure_jump(ds,bodies[:1],fixture_request,**kwargs)
    with pytest.raises(ValueError,match='heading'):h.bank_from_anyhydro_pressure_jump(ds,bodies,fixture_request,**{**kwargs,'heading':.123456})
    changed_bodies=(replace(bodies[0],attached_surfaces=()),bodies[1])
    with pytest.raises(h.HydroError):h.bank_from_anyhydro_pressure_jump(ds,changed_bodies,fixture_request,**kwargs)


def test_native_sheet_requires_potential_flow_owner(fixture_bodies,fixture_request,two_body_dataset):
    bodies=body_set(fixture_bodies)
    sheet=bodies[0].attached_surfaces[0]
    sheet=replace(sheet,surface=replace(sheet.surface,primary_hydrodynamics='none'),mesh_hash='')
    bodies=(replace(bodies[0],attached_surfaces=(sheet,)),bodies[1])
    ds=pressure_dataset(bodies,two_body_dataset)
    # Schema versions may refuse this already; either owner refuses unsupported loading.
    try:ds=attach_excitation_source(ds,fixture_request,bodies,convention_evidence='synthetic wrong-owner refusal')
    except h.HydroError:return
    with pytest.raises(ValueError,match='potential_flow'):
        h.bank_from_anyhydro_pressure_jump(ds,bodies,fixture_request,body_id=bodies[0].body_id,heading=float(ds.wave_direction.values[0]))
