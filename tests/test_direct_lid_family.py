"""Additive direct internal-lid record contract; public synthetic geometry only."""
from copy import deepcopy
from dataclasses import replace
import json
import numpy as np
import pytest
from anyhydro import floating_radiation_fields as ff
from anyhydro.direct_lid_profile import DirectLidProfile, profile_from_dict, profile_from_request, verify_profile_body
from anyhydro.wave_surface_fields import WaveFieldError

KW=dict(heading=0.,water_depth=50.,rho=1025.,gravity=9.81,Q0=np.eye(3),cg_world=[0.,0.,-.4])


@pytest.fixture(autouse=True)
def neutral_contract_fixture(request,monkeypatch):
    if not request.node.name.startswith('test_native_'):
        monkeypatch.setattr('anyhydro.direct_lid_profile.verify_canonical_profile',lambda *a,**k:None)


def fixture():
    v=np.array([[0,0,-1],[0,1,-1],[1,1,-1],[1,0,-1.]],float)
    f=np.array([[0,1,2,3]])
    r=ff.family_request(v,f,['hull'],**KW)
    lid=v.copy();lid[:,2]=-.1
    p=DirectLidProfile(r['geometry'],lid,f,np.vstack([v,lid]),np.vstack([f,f+4]),np.array([True,False]),-.1,2.)
    return v,f,p


def snapshot():
    v,f,p=fixture();r=ff.family_request(v,f,['hull'],recipe=ff.DIRECT_LID_RECIPE,direct_lid_profile=p,**KW)
    phi=np.arange(12).reshape(6,2)+1j
    return ff.FloatingFamilySnapshot(r,.8,v,f,('hull',),np.tile([1.,0.],(6,1)).astype(complex),np.array([1.,0.],complex),
        np.eye(6),np.eye(6),np.ones(6,complex),dict(method='direct'),
        panel_pressure_radiation=1j*.8*1025*phi[:,:1],panel_pressure_incident_diffraction=np.ones(1,complex),
        centers=np.array([[.5,.5,-1.]]),normals=np.array([[0,0,-1.]]),areas=np.ones(1),
        full_potential_radiation=phi,full_potential_diffraction=np.array([2+1j,7+2j]))


def test_profile_settings_and_bound_request_roundtrip():
    v,f,p=fixture();assert profile_from_dict(p.to_dict()).content_hash()==p.content_hash()
    r=ff.family_request(v,f,['hull'],recipe=ff.DIRECT_LID_RECIPE,direct_lid_profile=p.to_dict(),**KW)
    assert profile_from_request(r).content_hash()==p.content_hash()
    for field in ('method','hull_mask','source_faces','max_work_bytes'):
        bad=deepcopy(p.to_dict());bad.pop(field)
        with pytest.raises(WaveFieldError):profile_from_dict(bad)
    bad=deepcopy(r);bad['direct_lid_profile']['hull_mask'].reverse()
    with pytest.raises(WaveFieldError):profile_from_request(bad)


def test_recipe_is_explicit_and_legacy_requests_stay_unchanged():
    v,f,p=fixture()
    with pytest.raises(WaveFieldError):ff.family_request(v,f,['hull'],recipe=ff.DIRECT_LID_RECIPE,**KW)
    with pytest.raises(WaveFieldError):ff.family_request(v,f,['hull'],direct_lid_profile=p,**KW)
    r=ff.family_request(v,f,['hull'],**KW)
    assert 'direct_lid_profile' not in r and ff._panel_ceiling(r)==600
    assert ff.ACTIVE_RECIPE==ff.RECIPE


def test_full_potentials_are_required_hashed_and_preserved_offline(tmp_path):
    s=snapshot()
    with pytest.raises(WaveFieldError):replace(s,full_potential_radiation=None,snapshot_id='')
    with pytest.raises(WaveFieldError):replace(s,full_potential_diffraction=s.full_potential_diffraction[:1],snapshot_id='')
    with pytest.raises(WaveFieldError,match='identity'):replace(s,full_potential_diffraction=s.full_potential_diffraction+1)
    ff.save_family_snapshot(tmp_path,s)
    t=ff.recover_family_snapshot(tmp_path,s.request,s.omega)
    assert t.snapshot_id==s.snapshot_id
    assert np.array_equal(t.full_potential_radiation,s.full_potential_radiation)
    assert t.panel_pressure_radiation.shape==(6,1)
    assert t.radiation_sources.shape==(6,2)
    assert ff.recover_radiation_donor(tmp_path,s.request,s.omega).snapshot_id==s.snapshot_id
    m,a=ff.snapshot_to_payload(s)
    del a['full_dif_phi']
    with pytest.raises(WaveFieldError):ff.snapshot_from_payload(m,a)


def test_full_source_order_mask_and_geometry_are_verified():
    _,_,p=fixture();s=snapshot()
    class Mesh:
        vertices=p.source_vertices;faces=p.source_faces;nb_faces=2
    class Body:
        mesh_including_lid=Mesh();hull_mask=p.hull_mask
    verify_profile_body(Body(),s.request)
    body=Body();body.hull_mask=np.array([False,True])
    with pytest.raises(WaveFieldError):verify_profile_body(body,s.request)
    bad=deepcopy(s.request);bad['direct_lid_profile']['source_faces'].reverse()
    with pytest.raises(WaveFieldError):replace(s,request=bad,snapshot_id='')


def test_direct_recovery_uses_complete_potentials_and_neumann_not_zero_lid():
    s=snapshot();r=object.__new__(ff._NativeRecovery);r.snap=s
    class Prob: pass
    class Cpt:
        @staticmethod
        def RadiationProblem(**kwargs):return Prob()
    r.cpt=Cpt();r.kw={};r.diffraction=Prob();calls=[]
    def representation(pts,prob,phi,neumann):
        calls.append((np.array(phi),np.array(neumann)))
        return np.full(len(pts),phi.sum()+neumann.sum())
    r._green_representation=representation
    pts=np.array([[2.,2.,-.5]])
    actual=r.potential(pts,2)
    assert actual[0]==s.full_potential_radiation[2].sum()+s.radiation_sources[2].sum()
    assert calls[0][0][-1]!=0


def test_new_full_arrays_cannot_be_attached_to_legacy_records():
    s=snapshot();r=dict(s.request);r.pop('direct_lid_profile');r['recipe']=ff.DIRECT_RECIPE
    with pytest.raises(WaveFieldError):replace(s,request=r,snapshot_id='')


def test_pure_depth_duplicate_and_mask_shape_refusals():
    _,_,p=fixture()
    with pytest.raises(WaveFieldError):replace(p,hull_mask=np.array([[True,False]]))
    bad=p.to_dict();bad['lid_z']=-2.
    bad['lid_vertices']=(np.array(bad['lid_vertices'])+[0,0,-1.9]).tolist()
    bad['source_vertices'][4:]=bad['lid_vertices']
    with pytest.raises(WaveFieldError,match='keel'):profile_from_dict(bad)
    with pytest.raises(WaveFieldError,match='duplicate'):
        replace(p,source_faces=np.vstack([p.source_faces,p.source_faces[1]]),
                lid_faces=np.vstack([p.lid_faces,p.lid_faces]),hull_mask=np.array([True,False,False]))
    bad=p.to_dict();bad['source_faces'][0]=[0]*10000
    with pytest.raises(WaveFieldError):profile_from_dict(bad)


def test_capacity_and_cross_array_and_provider_refusals():
    s=snapshot()
    with pytest.raises(WaveFieldError,match='pressure'):replace(s,panel_pressure_radiation=s.panel_pressure_radiation+1,snapshot_id='')
    q=s.radiation_sources.copy();q[:,-1]=1
    with pytest.raises(WaveFieldError,match='Neumann'):replace(s,radiation_sources=q,snapshot_id='')
    changed=replace(s,provider=dict(method='direct',revision='other'),snapshot_id='')
    with pytest.raises(WaveFieldError,match='provider'):ff.validate_family_coherence([s,changed])
    changed=replace(s,gf_settings=dict(imaginary_part='other'),snapshot_id='')
    with pytest.raises(WaveFieldError,match='settings'):ff.validate_family_coherence([s,changed])
    v,f,p=fixture();many=np.tile(f,(601,1));ids=['hull']*601
    raw=ff.family_request(v,many,ids,**KW)
    p=replace(p,hull_hash=raw['geometry'])
    with pytest.raises(WaveFieldError,match='capacity'):
        ff.family_request(v,many,ids,recipe=ff.DIRECT_LID_RECIPE,direct_lid_profile=p,**KW)


@pytest.mark.skipif(__import__('os').environ.get(ff.NATIVE_GATE)!='1',reason='explicit native small fixture')
def test_native_lid_hull_only_normalization_complete_recovery_and_offline_reload(tmp_path,monkeypatch):
    from test_floating_passivity import box
    from anyhydro.direct_lid_profile import create_source_profile
    from anyhydro.wave_surface_fields import _replay_green_function
    from capytaine.bem.airy_waves import airy_waves_potential
    import capytaine as cpt
    v,f,labels=box(8,4,2)
    p=create_source_profile(v,f,labels,lid_z=-.1,faces_max_radius=.5,water_depth=50.)
    bad=p.to_dict();lv=np.array(bad['lid_vertices']);lv[:,:2]*=2;bad['lid_vertices']=lv.tolist()
    sv=np.array(bad['source_vertices']);indices=np.unique(np.array(bad['source_faces'])[~p.hull_mask]);sv[indices,:2]*=2
    bad['source_vertices']=sv.tolist()
    with pytest.raises(WaveFieldError,match='canonical'):
        ff.family_request(v,f,labels,recipe=ff.DIRECT_LID_RECIPE,direct_lid_profile=bad,**KW)
    s=ff.solve_family_snapshot(v,f,labels,1.2,recipe=ff.DIRECT_LID_RECIPE,direct_lid_profile=p,**KW)
    mask=p.hull_mask;n=len(p.source_faces)
    assert s.full_potential_radiation.shape==(6,n) and s.full_potential_diffraction.shape==(n,)
    assert np.linalg.norm(s.full_potential_radiation[:,~mask])>0
    np.testing.assert_array_equal(s.panel_pressure_radiation,1j*s.omega*1025*s.full_potential_radiation[:,mask])
    modes=np.vstack([s.normals.T,np.cross(s.centers-np.array(KW['cg_world']),s.normals).T])
    force=-(s.panel_pressure_radiation*s.areas)@modes.T
    Z=s.omega**2*s.added_mass+1j*s.omega*s.damping
    np.testing.assert_allclose(force.T,Z,rtol=0,atol=1e-8*np.abs(Z).max())
    # This coarse development box is not passivity-qualified. Keep its raw damping;
    # the production admission gate must still adjudicate each actual source family.
    assert not np.array_equal(s.damping,s.damping.T)
    _,_,_,_,body=ff._body_for_family(cpt,v,f,labels,s.request,s.omega)
    problem=cpt.DiffractionProblem(body=body,omega=s.omega,water_depth=50.,rho=1025.,g=9.81)
    gf=_replay_green_function(cpt,s,problem)
    solver=cpt.BEMSolver(green_function=gf)
    native=solver.solve(cpt.RadiationProblem(body=body,omega=s.omega,water_depth=50.,rho=1025.,g=9.81,
                         radiating_dof='Heave'),method='direct',keep_details=True,n_threads=1)
    np.testing.assert_allclose(s.damping[:,2],[native.radiation_damping[name] for name in ff.DOF_NAMES],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(s.full_potential_radiation[2],native.potential,rtol=1e-12,atol=1e-12)
    points=np.array([[4.,0.,-.3],[0.,2.,-.3]])
    S,D=gf.evaluate(points,body.mesh_including_lid,free_surface=0.,water_depth=50.,wavenumber=problem.wavenumber,
                    adjoint_double_layer=False,early_dot_product=True,diagonal_term_in_double_layer=False)
    reference=S@s.diffraction_sources-D@s.full_potential_diffraction+airy_waves_potential(points,problem)
    actual=ff._potentials(s,points)
    np.testing.assert_allclose(actual,reference,rtol=1e-12,atol=1e-12)
    # A deliberate missing-lid representation must be detectable on this fixture.
    wrong=S[:,mask]@s.diffraction_sources[mask]-D[:,mask]@s.full_potential_diffraction[mask]+airy_waves_potential(points,problem)
    assert np.max(np.abs(reference-wrong))>1e-5
    ff.save_family_snapshot(tmp_path,s)
    monkeypatch.setattr(cpt.BEMSolver,'solve',lambda *a,**k:pytest.fail('offline recovery called BEM solve'))
    t=ff.recover_family_snapshot(tmp_path,s.request,s.omega)
    normal=np.array([[1.,0,0],[0,1.,0]])
    full=ff.evaluate_floating_fields(t,points,normal)
    eta=ff.evaluate_floating_elevations(t,points,normal)
    np.testing.assert_array_equal(full.eta_radiation,eta.eta_radiation)
    np.testing.assert_array_equal(full.eta_incident_diffraction,eta.eta_incident_diffraction)
    masked=ff.evaluate_floating_fields(t,points,normal,output_mask=[True,False])
    assert masked.incident_diffraction_pressure[1]==0 and np.all(masked.radiation_pressure[:,1]==0)
