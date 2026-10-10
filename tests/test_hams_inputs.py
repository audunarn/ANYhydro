# SPDX-License-Identifier: MPL-2.0
from dataclasses import replace
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pytest
from anyfileio.hams import HamsMechanicalInput,parse_control,parse_mesh,parse_mechanical
from anyhydro.fixtures import make_fixture_bodies
from anyhydro.first_slice import make_first_slice_bodies
from anyhydro.records import HydroRequest,HydroPanelMesh
from anyhydro.surfaces import SurfaceSemantics
from anyhydro.errors import CapabilityError
from anyhydro.hams_inputs import HamsPreparedDeck,HamsDeckError,prepare_hams_deck


@pytest.fixture
def case(tmp_path,monkeypatch):
    # Synthetic executable-identity fixture; never executed or qualified.
    exe=tmp_path/'fixture.exe';exe.write_bytes(b'identity-fixture-never-executed')
    import anyhydro.hams_inputs as hi
    monkeypatch.setattr(hi,'HAMS_EXECUTABLE_SHA256',hashlib.sha256(exe.read_bytes()).hexdigest())
    bodies=make_fixture_bodies();request=HydroRequest([.5,1.25],[0.,.7],50.,1000.,9.80665,False,True,{})
    eye=tuple(tuple(float(i==j) for j in range(6)) for i in range(6));zero=tuple((0.,)*6 for _ in range(6))
    mechanics={b.body_id:HamsMechanicalInput(b.center_of_mass,eye,zero,zero,eye,zero) for b in bodies}
    return dict(directory=tmp_path/'deck',bodies=bodies,request=request,mechanics=mechanics,executable_path=exe)


def test_full_multibody_deck_geometry_reference_identity_and_no_launch(case):
    deck=prepare_hams_deck(**case);m=deck.verify(bodies=case['bodies'],request=case['request'],executable_path=case['executable_path'])
    assert m['qualification']=='development_unqualified';assert m['excitation_admission'].startswith('unsupported')
    assert set(m['input_sha256'])=={'ControlFile.in','HullMesh_1.pnl','HullMesh_2.pnl','Hydrostatic_1.in','Hydrostatic_2.in'}
    c=parse_control((deck.directory/'Input/ControlFile.in').read_text());assert c.include_limits
    assert c.body_origins_xyzyaw==((0.,0.,0.,0.),)*2
    np.testing.assert_array_equal(c.reference_centers,[b.rotation_center for b in case['bodies']])
    for i,b in enumerate(case['bodies'],1):
        mesh=parse_mesh((deck.directory/'Input'/f'HullMesh_{i}.pnl').read_text())
        np.testing.assert_array_equal(mesh.vertices,b.mesh.vertices)
        assert mesh.faces==tuple(tuple(int(x) for x in row[:n]) for row,n in zip(b.mesh.faces,b.mesh.face_sizes))
        mech=parse_mechanical((deck.directory/'Input'/f'Hydrostatic_{i}.in').read_text());assert mech==case['mechanics'][b.body_id]
    assert not list((deck.directory/'Output').rglob('*.*'))
    assert (deck.directory/'Output/Wamit_format').is_dir()
    with pytest.raises(FileExistsError):prepare_hams_deck(**case)


def test_single_body_deep_water_has_native_names_and_world_z(case):
    b=case['bodies'][1];case.update(bodies=(b,),mechanics={b.body_id:case['mechanics'][b.body_id]},request=replace(case['request'],water_depth=float('inf')))
    deck=prepare_hams_deck(**case);c=parse_control((deck.directory/'Input/ControlFile.in').read_text())
    assert c.water_depth==-1.;assert (deck.directory/'Input/HullMesh.pnl').exists()
    np.testing.assert_array_equal(parse_mesh((deck.directory/'Input/HullMesh.pnl').read_text()).vertices,b.mesh.vertices)


def test_irregular_frequency_lid_policy_exact_plane_and_no_omission(case):
    bodies=make_first_slice_bodies(subdivisions=4);mechanics=case['mechanics'];values=next(iter(mechanics.values()))
    case.update(bodies=bodies,mechanics={b.body_id:replace(values,center_of_gravity=b.center_of_mass) for b in bodies})
    with pytest.raises(HamsDeckError):prepare_hams_deck(**case)
    assert not case['directory'].exists()
    deck=prepare_hams_deck(**case,irregular_frequency_removal=True)
    assert len(list((deck.directory/'Input').glob('WaterPlaneMesh*')))==2
    assert parse_control((deck.directory/'Input/ControlFile.in').read_text()).irregular_frequency_removal


@pytest.mark.parametrize('mutation',['file','sidecar','extra','missing'])
def test_sealed_deck_changes_refuse(case,mutation):
    deck=prepare_hams_deck(**case)
    if mutation=='file':(deck.directory/'Input/HullMesh_1.pnl').write_text('corrupt')
    elif mutation=='sidecar':(deck.directory/'case.json').write_text('{}')
    elif mutation=='extra':(deck.directory/'Input/GenMode_1_7.dat').write_text('implicit mode')
    else:(deck.directory/'Input/Hydrostatic_2.in').unlink()
    with pytest.raises(HamsDeckError):deck.verify()


def test_changed_caller_request_reference_and_executable_refuse(case,tmp_path):
    deck=prepare_hams_deck(**case)
    with pytest.raises(HamsDeckError):deck.verify(request=replace(case['request'],rho=1025.))
    changed=(replace(case['bodies'][0],rotation_center=(1.,2.,3.)),case['bodies'][1])
    with pytest.raises(HamsDeckError):deck.verify(bodies=changed)
    fake=tmp_path/'fake.exe';fake.write_bytes(b'not pinned')
    with pytest.raises(HamsDeckError):deck.verify(executable_path=fake)
    new=case.copy();new.update(directory=tmp_path/'new',executable_path=fake)
    with pytest.raises(HamsDeckError):prepare_hams_deck(**new)
    assert not new['directory'].exists()


@pytest.mark.parametrize('change',[{'rho':1025.},{'gravity':9.81},{'backend_settings':{'unsupported':'option'}},{'include_zero_frequency':True}])
def test_native_properties_unmapped_controls_and_zero_request_refuse(case,change):
    case['request']=replace(case['request'],**change)
    with pytest.raises(HamsDeckError):prepare_hams_deck(**case)
    assert not case['directory'].exists()


def test_mechanical_rounding_explicit_report_and_reference_rounding_refuse(case):
    b=case['bodies'][0];r=case['mechanics'][b.body_id];values=[list(row) for row in r.mass];values[0][1]=1.23456789
    case['mechanics'][b.body_id]=replace(r,mass=values)
    with pytest.raises(HamsDeckError):prepare_hams_deck(**case)
    assert not case['directory'].exists()
    deck=prepare_hams_deck(**case,allow_mechanical_rounding=True);m=deck.verify()
    assert m['mechanical_quantization'][b.body_id]['mass']['changed_values']==1
    assert m['mechanical_quantization'][b.body_id]['mass']['max_absolute_change']>0


def test_missing_mechanics_cog_subset_lid_and_nonmillimeter_reference_refuse(case):
    original=case.copy()
    for alteration in ('mechanics','cog','subset','reference','lid'):
        candidate={**original,'mechanics':dict(original['mechanics'])};b=candidate['bodies'][0]
        if alteration=='mechanics':candidate['mechanics']={}
        elif alteration=='cog':candidate['mechanics'][b.body_id]=replace(candidate['mechanics'][b.body_id],center_of_gravity=(0.,0.,0.))
        elif alteration=='subset':candidate['bodies']=(replace(b,enabled_dofs=b.enabled_dofs[:1]),candidate['bodies'][1])
        elif alteration=='reference':candidate['bodies']=(replace(b,rotation_center=(.0001,0.,0.)),candidate['bodies'][1])
        else:
            with pytest.raises(HamsDeckError):prepare_hams_deck(**candidate,irregular_frequency_removal=True)
            continue
        with pytest.raises(HamsDeckError):prepare_hams_deck(**candidate)
        assert not candidate['directory'].exists()


def test_optional_syntax_missing_is_typed_before_writes(case,monkeypatch):
    monkeypatch.setitem(sys.modules,'anyfileio.hams',None)
    import anyfileio;monkeypatch.delattr(anyfileio,'hams',raising=False)
    with pytest.raises(CapabilityError):prepare_hams_deck(**case)
    assert not case['directory'].exists()


def test_changed_attached_surface_and_normal_verification_refuse(case):
    deck=prepare_hams_deck(**case);body=case['bodies'][0]
    semantics=SurfaceSemantics('added-sheet',body.body_id,'thin_sheet','potential_flow','source-sheet',minus_side='minus',plus_side='plus')
    sheet=HydroPanelMesh(body_id=body.body_id,vertices=np.array([[0.,0.,-1.],[1.,0.,-1.],[0.,1.,-1.]]),faces=np.array([[0,1,2]]),face_sizes=np.array([3]),source_geometry_refs=('source-sheet',),mesh_hash=None,surface=semantics,outward_normals_verified=False)
    changed=(replace(body,attached_surfaces=(sheet,)),case['bodies'][1])
    with pytest.raises(HamsDeckError):deck.verify(bodies=changed)
    with pytest.raises(HamsDeckError):prepare_hams_deck(**{**case,'directory':case['directory'].parent/'sheets','bodies':changed})
    unverified=replace(body,mesh=replace(body.mesh,outward_normals_verified=False))
    with pytest.raises(HamsDeckError):deck.verify(bodies=(unverified,case['bodies'][1]))


def test_interrupted_publication_keeps_incomplete_evidence_and_no_seal(case,monkeypatch):
    original=Path.write_bytes
    def interrupt(path,data):
        if path.name=='case.json':raise OSError('simulated disk interruption')
        return original(path,data)
    monkeypatch.setattr(Path,'write_bytes',interrupt)
    with pytest.raises(HamsDeckError):prepare_hams_deck(**case)
    assert (case['directory']/'Input/ControlFile.in').is_file()
    assert not (case['directory']/'case.json').exists()
    with pytest.raises(HamsDeckError):HamsPreparedDeck(case['directory'],'0'*64).verify()


def test_caller_mechanical_identity_is_optional_explicit_and_complete(case):
    deck=prepare_hams_deck(**case);deck.verify(mechanics=case['mechanics'])
    changed=dict(case['mechanics']);b=case['bodies'][0];m=changed[b.body_id]
    changed[b.body_id]=replace(m,mass=m.external_restoring,external_restoring=m.linear_damping)
    with pytest.raises(HamsDeckError):deck.verify(mechanics=changed)
    with pytest.raises(HamsDeckError):deck.verify(mechanics={})
    with pytest.raises(HamsDeckError):deck.verify(mechanics={'invalid':None})


@pytest.mark.parametrize('depth,omega',[(501.,.5),(float('inf'),1.e-6),(50.,1.e8)])
def test_inconsistent_incident_and_limit_branches_refuse_before_publication(case,depth,omega):
    case['request']=replace(case['request'],water_depth=depth,omegas=[omega])
    with pytest.raises(HamsDeckError):prepare_hams_deck(**case)
    assert not case['directory'].exists()
