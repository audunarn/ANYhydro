"""Surface transport/refusal contracts. Synthetic coefficients are not sheet evidence."""
from dataclasses import replace
import copy
import json
import numpy as np
import pytest
import anyhydro as h
from anyhydro.surfaces import sheet_geometry

def plate(owner='float-a', **kwargs):
    semantics=h.SurfaceSemantics('plate-'+owner,owner,'thin_sheet','potential_flow','plate-geometry',
                                minus_side='lower-fluid',plus_side='upper-fluid',**kwargs)
    return h.HydroPanelMesh(np.array([[0.,0.,-2.],[2.,0.,-2.],[2.,1.,-2.],[0.,1.,-2.]]),
        np.array([[0,1,2],[0,2,3]]),np.array([3,3]),owner,False,('plate-geometry',),'',surface=semantics)

def body_set(fixture_bodies):
    return tuple(replace(b,attached_surfaces=(plate(b.body_id),)) for b in fixture_bodies)

def source():
    # Deliberately declared synthetic test provenance; this never changes fixture qualification.
    return dict(backend='synthetic-schema-test-only',version='test-v1',source_commit='test-source',
        capability_evidence_sha256='1'*64,coupling='full_hull_sheet_and_cross_body',
        pressure_jump_convention='p_minus-p_plus;normal_minus_to_plus',source_capability_status='verified')

def dataset(bodies, original):
    ds=original.copy(deep=True)
    ds.attrs['schema_version']=h.SURFACE_DATASET_SCHEMA
    ds.attrs['surface_identity']=json.dumps(h.surface_identity(bodies),sort_keys=True)
    ds.attrs['sheet_source_provenance']=json.dumps(source(),sort_keys=True)
    return h.validate_hydro_dataset(ds)

def test_sheet_orientation_pressure_work_and_offset_moment():
    m=plate(); f,t=h.sheet_pressure_load(m,np.array([3.,3.]),np.array([1.,1.]),reference=(0.,0.,0.))
    np.testing.assert_allclose(f,[0,0,4]); np.testing.assert_allclose(t,[2,-4,0])
    # Under consistent reversal, labels and values swap as face winding reverses.
    reverse=replace(m,faces=m.faces[:,::-1],mesh_hash='',surface=replace(m.surface,minus_side='upper-fluid',plus_side='lower-fluid'))
    fr,tr=h.sheet_pressure_load(reverse,np.array([1.,1.]),np.array([3.,3.]))
    np.testing.assert_allclose(fr,f);np.testing.assert_allclose(tr,t)
    velocity=np.array([1.,-2.,.5]);omega=np.array([.2,.3,-.4])
    _,area,centers=sheet_geometry(m)
    panel_forces=2*area
    point_velocity=velocity+np.cross(omega,centers)
    assert np.sum(panel_forces*point_velocity)==pytest.approx(f@velocity+t@omega)

def test_complex_pressure_and_centroid():
    m=plate(); f,t=h.sheet_pressure_load(m,np.array([1j,1j]),np.zeros(2))
    np.testing.assert_allclose(f,[0,0,2j]);np.testing.assert_allclose(t,[1j,-2j,0])

def test_frozen_explicit_mesh_and_identity():
    m=plate(); before=m.mesh_hash
    with pytest.raises(ValueError):m.vertices[0,0]=9
    changed=replace(m,surface=replace(m.surface,source_geometry_id='new-geometry'),source_geometry_refs=('new-geometry',),mesh_hash='')
    assert changed.mesh_hash!=before
    changed=replace(m,surface=replace(m.surface,minus_side='changed-fluid'),mesh_hash='')
    assert changed.mesh_hash!=before
    assert len(sheet_geometry(m)[0])==4
    assert h.SurfaceSemantics.from_dict(m.surface.to_dict())==m.surface

@pytest.mark.parametrize('field,value',[
    ('surface_kind','shell'),('primary_hydrodynamics','potential_flow_and_morison'),
    ('primary_hydrodynamics','morison'),('viscous_correction','member_drag'),
    ('minus_side','upper-fluid'),('minus_side',None),('source_geometry_id',''),
    ('owning_body_id',''),('surface_id',''),('plate_drag',h.PlateDragCorrection(1.,1.))])
def test_invalid_sheet_semantics(field,value):
    with pytest.raises(h.RecordValidationError):replace(plate().surface,**{field:value})

@pytest.mark.parametrize('field,value',[
    ('outward_normals_verified',True),('body_id','other'),('source_geometry_refs',()),
    ('faces',[[0.,1.,2.],[0.,2.,3.]]),('face_sizes',[3.5,3.5]),
    ('faces',np.array([[0,1,1],[0,2,3]])),('faces',np.array([[0,1,2],[0,3,2]])),
    ('vertices',np.zeros((4,3)))])
def test_invalid_sheet_geometry(field,value):
    with pytest.raises(h.RecordValidationError):replace(plate(),mesh_hash='',**{field:value})

def test_plate_drag_single_owner_and_area_convention():
    m=plate(viscous_correction='plate_drag',plate_drag=h.PlateDragCorrection(2.,.8))
    assert m.surface.plate_drag.reference_area_m2==2
    assert h.SurfaceSemantics.from_dict(m.surface.to_dict())==m.surface
    with pytest.raises(h.RecordValidationError):h.PlateDragCorrection(True,1.)
    with pytest.raises(h.RecordValidationError):h.PlateDragCorrection(1.,1.,force_direction='equivalent_diameter')
    s=m.surface.to_dict(); s['displaced_volume_m3']=.01
    with pytest.raises(h.RecordValidationError):h.SurfaceSemantics.from_dict(s)
    s=m.surface.to_dict(); s['morison_inertia']=True
    with pytest.raises(h.RecordValidationError):h.SurfaceSemantics.from_dict(s)

def test_explicit_migration_and_numerical_lid():
    m=h.migrate_surface_role('bem',surface_kind='body_boundary',surface_id='hull',owning_body_id='a',source_geometry_id='hull-src')
    assert m.primary_hydrodynamics=='potential_flow'
    with pytest.raises(h.RecordValidationError):h.migrate_surface_role('drag_only',surface_kind='thin_sheet',surface_id='s',owning_body_id='a',source_geometry_id='src')
    lid=h.migrate_surface_role('numerical_lid',surface_kind='numerical_lid',surface_id='lid',owning_body_id='a',source_geometry_id='lid-src')
    assert lid.primary_hydrodynamics=='none'
    with pytest.raises(h.RecordValidationError):replace(lid,viscous_correction='plate_drag')
    with pytest.raises(h.RecordValidationError):replace(lid,primary_hydrodynamics='potential_flow')

def test_exact_native_refusal_before_import_cache_or_geometry(fixture_bodies,fixture_request,monkeypatch,tmp_path):
    b=h.CapytaineBackend(cache_dir=str(tmp_path))
    def forbidden(*args,**kwargs):raise AssertionError('backend/cache/geometry touched before unsupported-sheet refusal')
    monkeypatch.setattr(b,'_admit_geometry',forbidden);monkeypatch.setattr(b,'_cache_load',forbidden)
    with pytest.raises(h.UnsupportedOperationError,match='genuine two-sided'):
        b.solve(body_set(fixture_bodies),fixture_request)
    caps=b.capabilities()
    assert (caps.native_thin_sheet,caps.mixed_hull_sheet,caps.multibody_with_sheets,caps.sheet_pressure_jump)==('unsupported',)*4

def test_body_ownership_lid_identity_and_no_extra_dofs(fixture_bodies):
    b=fixture_bodies[0]; p=plate(b.body_id)
    attached=replace(b,attached_surfaces=[p])
    assert attached.all_dof_keys()==b.all_dof_keys() and attached.enabled_dofs==b.enabled_dofs
    assert attached.mesh.mesh_hash==b.mesh.mesh_hash
    with pytest.raises(h.RecordValidationError):replace(b,mesh=p)
    with pytest.raises(h.RecordValidationError):replace(b,lid_mesh=p)
    with pytest.raises(h.RecordValidationError):replace(b,attached_surfaces=(p,p))
    with pytest.raises(h.RecordValidationError):replace(b,attached_surfaces=(plate('other'),))

def test_import_save_reopen_coefficients_without_pressure(fixture_bodies,fixture_request,two_body_dataset,tmp_path):
    bodies=body_set(fixture_bodies); ds=dataset(bodies,two_body_dataset)
    assert 'pressure_jump' not in ds and ds.attrs['qualification_status']=='development_fixture'
    path=tmp_path/'sheet.nc'; h.save_hydro_dataset(ds,path); restored=h.load_hydro_dataset(path)
    assert h.dataset_content_hash(restored)==h.dataset_content_hash(ds)
    imported=h.ImportedHydroDatasetBackend(restored)
    got=imported.solve(bodies,fixture_request)
    np.testing.assert_array_equal(got.added_mass,two_body_dataset.added_mass)
    assert imported.capabilities().native_thin_sheet=='unsupported'
    with pytest.raises(h.BodyIdentityMismatchError,match='surface identity'):
        imported.solve(fixture_bodies,fixture_request)
    with pytest.raises(h.BodyIdentityMismatchError,match='surface identity'):
        h.ImportedHydroDatasetBackend(two_body_dataset).solve(bodies,fixture_request)
    changed=(replace(bodies[0],attached_surfaces=(replace(bodies[0].attached_surfaces[0],surface=replace(bodies[0].attached_surfaces[0].surface,plus_side='edited'),mesh_hash=''),)),bodies[1])
    with pytest.raises(h.BodyIdentityMismatchError):imported.solve(changed,fixture_request)

@pytest.mark.parametrize('change', ['absent','unverified','summed','sign','hash','unknown','null'])
def test_source_provenance_fails_closed(change,fixture_bodies,two_body_dataset):
    ds=dataset(body_set(fixture_bodies),two_body_dataset); p=source()
    if change=='absent':del ds.attrs['sheet_source_provenance']
    else:
        if change=='unverified':p['source_capability_status']='unverified'
        elif change=='summed':p['coupling']='hull_plus_plate_separately'
        elif change=='sign':p['pressure_jump_convention']='p_plus-p_minus'
        elif change=='hash':p['capability_evidence_sha256']='not-a-hash'
        elif change=='unknown':p['assumed_native']=True
        else:p=None
        ds.attrs['sheet_source_provenance']=json.dumps(p)
    with pytest.raises(h.SchemaError):h.validate_hydro_dataset(ds)

def pressure_dataset(bodies,original):
    ds=dataset(bodies,original); ids=[]; faces=[]
    for b in bodies:
        sid=b.attached_surfaces[0].surface.surface_id
        ids.extend([sid,sid]);faces.extend([0,1])
    ds=ds.assign_coords(sheet_panel=range(len(ids)),sheet_panel_surface_id=('sheet_panel',ids),sheet_panel_face=('sheet_panel',faces))
    ds['pressure_jump']=(('omega','wave_direction','sheet_panel'),np.ones((ds.sizes['omega'],ds.sizes['wave_direction'],len(ids)),dtype=np.complex128)*(1+2j))
    ds.attrs[h.PRESSURE_JUMP_CONTRACT_ATTR]=dict(quantity='total_excitation_pressure_jump',unit='Pa/m',wave_amplitude='unit_complex_incident_amplitude_1_m',coordinate_frame='world',phasor_convention=ds.attrs['phasor_convention'],sign='p_minus-p_plus;normal_minus_to_plus')
    return ds

def test_optional_pressure_save_reopen_and_hash_binding(fixture_bodies,two_body_dataset,tmp_path):
    ds=h.validate_hydro_dataset(pressure_dataset(body_set(fixture_bodies),two_body_dataset))
    path=tmp_path/'pressure.nc'; h.save_hydro_dataset(ds,path); restored=h.load_hydro_dataset(path)
    np.testing.assert_array_equal(restored.pressure_jump,ds.pressure_jump)
    assert h.dataset_content_hash(ds)==h.dataset_content_hash(restored)
    changed=ds.copy(deep=True); changed.pressure_jump.values[0,0,0]+=1
    assert h.dataset_content_hash(changed)!=h.dataset_content_hash(ds)
    attrs=ds.copy(deep=True);attrs.attrs['surface_identity']=json.loads(attrs.attrs['surface_identity'])
    assert h.dataset_content_hash(attrs)==h.dataset_content_hash(ds)

@pytest.mark.parametrize('change',['no-surface','missing-coord','duplicate-face','wrong-surface','real','nan'])
def test_optional_pressure_identity_refusal(change,fixture_bodies,two_body_dataset):
    ds=pressure_dataset(body_set(fixture_bodies),two_body_dataset)
    if change=='no-surface':del ds.attrs['surface_identity']
    elif change=='missing-coord':ds=ds.drop_vars('sheet_panel_face')
    elif change=='duplicate-face':ds.sheet_panel_face.values[1]=0
    elif change=='wrong-surface':ds.sheet_panel_surface_id.values[0]='wrong'
    elif change=='real':ds['pressure_jump']=ds.pressure_jump.real
    else:ds.pressure_jump.values[0,0,0]=np.nan
    with pytest.raises(h.SchemaError):h.validate_hydro_dataset(ds)

def test_legacy_absence_preserves_import_and_hash(fixture_bodies,fixture_request,two_body_dataset):
    assert h.surface_identity(fixture_bodies) is None
    for b in fixture_bodies:
        assert b.attached_surfaces==() and b.mesh.surface is None
        assert h.mesh_hash(b.body_id,b.mesh.vertices,b.mesh.faces,b.mesh.face_sizes)==b.mesh.mesh_hash
    assert h.dataset_content_hash(h.ImportedHydroDatasetBackend(two_body_dataset).solve(fixture_bodies,fixture_request))==h.dataset_content_hash(two_body_dataset)


@pytest.mark.parametrize('index',[2**32,2**63-1,-2**32])
def test_raw_index_cannot_wrap_before_narrowing(index):
    faces=np.array([[index,1,2],[0,2,3]],dtype=np.int64)
    with pytest.raises(h.RecordValidationError,match='narrowing'):
        replace(plate(),faces=faces,mesh_hash='')

def test_morison_full_explicit_migration():
    s=h.migrate_surface_role('morison_full',surface_kind='body_boundary',surface_id='hull',owning_body_id='a',source_geometry_id='src')
    assert s.primary_hydrodynamics=='morison' and s.viscous_correction=='none'

@pytest.mark.parametrize('faces,vertices',[
    ([[0,1,2],[0,2,1],[3,4,5]],[[0,0,-2],[1,0,-2],[0,1,-2],[2,0,-2],[3,0,-2],[2,1,-2]]),
    ([[0,1,2],[3,5,4]],[[0,0,-2],[1,0,-2],[0,1,-2],[0,0,-2],[1,0,-2],[0,1,-2]]),
    ([[0,2,1],[0,1,3],[0,3,2],[1,2,3],[4,5,6]],[[0,0,-3],[1,0,-3],[0,1,-3],[0,0,-2],[2,0,-3],[3,0,-3],[2,1,-3]])])
def test_opposing_coincident_faces_and_closed_components_refused(faces,vertices):
    with pytest.raises(h.RecordValidationError):
        replace(plate(),faces=np.array(faces),vertices=np.array(vertices,dtype=float),face_sizes=np.array([3]*len(faces)),mesh_hash='')

def test_schema_version_and_surface_downgrade_refusal(fixture_bodies,two_body_dataset):
    ds=dataset(body_set(fixture_bodies),two_body_dataset)
    assert ds.attrs['schema_version']=='1.2.0'
    ds.attrs['schema_version']='1.1.0'
    with pytest.raises(h.SchemaError,match='legacy relabeling'):h.validate_hydro_dataset(ds)
    ds=two_body_dataset.copy(deep=True);ds.attrs['schema_version']='1.2.0'
    with pytest.raises(h.SchemaError,match='requires explicit'):h.validate_hydro_dataset(ds)

@pytest.mark.parametrize('field,value',[('unit','Pa'),('quantity','radiation_pressure_jump'),('coordinate_frame','body'),('wave_amplitude','arbitrary'),('sign','p_plus-p_minus')])
def test_pressure_quantity_units_and_normalization_refusal(field,value,fixture_bodies,two_body_dataset):
    ds=pressure_dataset(body_set(fixture_bodies),two_body_dataset)
    ds.attrs[h.PRESSURE_JUMP_CONTRACT_ATTR][field]=value
    with pytest.raises(h.SchemaError):h.validate_hydro_dataset(ds)

def test_pressure_world_scalar_survives_shift_rotation_roundtrip(fixture_bodies,two_body_dataset,tmp_path):
    ds=h.validate_hydro_dataset(pressure_dataset(body_set(fixture_bodies),two_body_dataset))
    before=h.dataset_content_hash(ds); ids=copy.deepcopy(ds.attrs['surface_identity'])
    original_centers={str(b):tuple(ds.rotation_center.values[i]) for i,b in enumerate(ds.body.values)}
    moved_centers={b:tuple(np.array(c)+[.3,-.2,.4]) for b,c in original_centers.items()}
    shifted=h.shift_dataset_reference_point(ds,moved_centers)
    axes=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    rotated=h.rotate_axes(shifted,{b:axes for b in original_centers})
    np.testing.assert_array_equal(rotated.pressure_jump,ds.pressure_jump)
    assert rotated.attrs['surface_identity']==ids
    assert rotated.attrs[h.PRESSURE_JUMP_CONTRACT_ATTR]==ds.attrs[h.PRESSURE_JUMP_CONTRACT_ATTR]
    assert h.dataset_content_hash(rotated)!=before and h.dataset_content_hash(ds)==before
    restored_axes=h.rotate_axes(rotated,{b:np.eye(3) for b in original_centers})
    roundtrip=h.shift_dataset_reference_point(restored_axes,original_centers)
    for name in ('added_mass','radiation_damping','excitation_force','hydrostatic_stiffness','added_mass_infinite'):
        np.testing.assert_allclose(roundtrip[name],ds[name],rtol=1e-12,atol=1e-12)
    np.testing.assert_array_equal(roundtrip.pressure_jump,ds.pressure_jump)
    path=tmp_path/'transformed.nc';h.save_hydro_dataset(rotated,path);saved=h.load_hydro_dataset(path)
    np.testing.assert_array_equal(saved.pressure_jump,ds.pressure_jump)
    assert h.dataset_content_hash(saved)==h.dataset_content_hash(rotated)

def test_production_assembler_surfaces_and_provenance(fixture_bodies,fixture_request,two_body_dataset):
    bodies=body_set(fixture_bodies);ds=two_body_dataset
    values={name:ds[name].values for name in ('added_mass','radiation_damping','excitation_force','hydrostatic_stiffness','rotation_center','center_of_mass','center_of_buoyancy','added_mass_infinite')}
    values.update(added_mass_infinite_source=ds.added_mass_infinite.attrs['source'],backend='schema-test',backend_version='test-v1',backend_commit='test-source',qualification_status='development_fixture',created_utc=ds.attrs['created_utc'])
    with pytest.raises(h.SchemaError,match='source provenance'):h.assemble_hydro_dataset(bodies,fixture_request,**values)
    produced=h.assemble_hydro_dataset(bodies,fixture_request,**values,extra_attrs={'sheet_source_provenance':source()})
    assert produced.attrs['schema_version']=='1.2.0'
    assert json.loads(produced.attrs['surface_identity'])==h.surface_identity(bodies)
    assert produced.added_mass.shape[-2:]==(12,12) and 'pressure_jump' not in produced
    h.ImportedHydroDatasetBackend(produced).solve(bodies,fixture_request)
    with pytest.raises(h.SchemaError,match='cannot be overridden'):
        h.assemble_hydro_dataset(bodies,fixture_request,**values,extra_attrs={'surface_identity':{},'sheet_source_provenance':source()})
