import json,math,sys,types
from pathlib import Path
from dataclasses import replace
import numpy as np
import pytest
from anyhydro import excitation_cache as module
from anyhydro.pressure_bank import PressureResponseBank,PressureBankError,PressurePreparationError


def pressure_bank():
    return PressureResponseBank([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],[[0,1,2,3]],('bottom',),
        [[.5,.5,0]],[[0,0,1]],[1.],[.7,1.1],[[1+2j],[3-4j]])


@pytest.fixture
def fake_backend(monkeypatch):
    import anyhydro._pressure_body as body_module
    b=pressure_bank();calls=[];constructors=[]
    path=str(Path('fake/capytaine/__init__.py').resolve())
    fp=dict(backend='capytaine',version='3.0.0',path=path,code={'adapter':'a'*64},numpy='test',scipy='test',recipe=module.RECIPE)
    monkeypatch.setattr(module,'backend_fingerprint',lambda:dict(fp))
    def wrench(p):return dict(Surge=0j,Sway=0j,Heave=-p,Roll=-.5*p,Pitch=.5*p,Yaw=0j)
    class Solver:
        exportable_settings=dict(method='indirect',engine='fake')
        engine=types.SimpleNamespace(green_function='fake-default')
        def __init__(self):constructors.append(1)
        def solve(self,pb,**kwargs):
            calls.append(pb.omega);phi=np.array([complex(pb.omega,-.1)])
            return types.SimpleNamespace(potential=phi,forces=wrench(1j*pb.omega*pb.rho*phi[0]))
    class Problem:
        def __init__(self,**kwargs):self.__dict__.update(kwargs)
    cpt=types.ModuleType('capytaine');cpt.__file__=path;cpt.__version__='3.0.0';cpt.BEMSolver=Solver;cpt.DiffractionProblem=Problem
    airy=types.ModuleType('capytaine.bem.airy_waves')
    airy.airy_waves_potential=lambda *args:np.array([.2+0j])
    airy.froude_krylov_force=lambda pb:wrench(1j*pb.omega*pb.rho*.2)
    monkeypatch.setitem(sys.modules,'capytaine',cpt);monkeypatch.setitem(sys.modules,'capytaine.bem.airy_waves',airy)
    mesh=types.SimpleNamespace(faces_centers=b.centers,faces_normals=b.normals,faces_areas=b.areas)
    monkeypatch.setattr(body_module,'prepare_body',lambda *args,**kw:(b.vertices,b.faces,b.surface_ids,mesh,object()))
    return b,calls,constructors,cpt,fp


def test_warm_force_only_and_deferred_pressure_recovery_no_solve(tmp_path,fake_backend,monkeypatch):
    b,calls,constructors,cpt,fp=fake_backend
    bank=module.prepare_excitation_bank(b.vertices,b.faces,b.surface_ids,b.omegas,cache_directory=tmp_path)
    assert calls==list(b.omegas) and len(constructors)==1
    with monkeypatch.context() as scope:
        def forbidden(*args,**kwargs):raise AssertionError('warm preview must not load panel data or initialize solver')
        scope.setattr(module.np,'load',forbidden);scope.setattr(cpt,'BEMSolver',forbidden)
        warm=module.prepare_excitation_bank(b.vertices,b.faces,b.surface_ids,b.omegas,cache_directory=tmp_path)
        assert warm.bank_id==bank.bank_id
    def no_solve(*args,**kwargs):raise AssertionError('recovery must never solve')
    monkeypatch.setattr(cpt,'BEMSolver',no_solve)
    recovered=module.recover_pressure_bank(bank,cache_directory=tmp_path)
    expected=np.array([[1j*w*1025*complex(w,-.1)+1j*w*1025*.2] for w in b.omegas])
    np.testing.assert_array_equal(recovered.responses,expected)
    assert calls==list(b.omegas)


def test_compact_schema_digest_and_solution_corruption_refuse(tmp_path,fake_backend):
    b,*_=fake_backend
    bank=module.prepare_excitation_bank(b.vertices,b.faces,b.surface_ids,b.omegas,cache_directory=tmp_path)
    for change in (dict(bank_id=''),dict(bank_id='bad'),dict(real=[[99]*6]*2),dict(providers=[]),
                   dict(conventions={'units':'kN'})):
        with pytest.raises(PressureBankError):module.ExcitationResponseBank.from_dict({**bank.to_dict(),**change})
    path=tmp_path/'solutions'/(bank.solution_ids[0]+'.npz');path.write_bytes(b'PK\x03\x04truncated')
    with pytest.raises(PressureBankError,match='missing/corrupt'):module.recover_pressure_bank(bank,cache_directory=tmp_path)


def test_partial_resume_migration_isolation_and_nonworld_refusal(tmp_path,fake_backend):
    b,calls,*_=fake_backend
    with pytest.raises(PressurePreparationError) as caught:
        module.prepare_excitation_bank(b.vertices,b.faces,b.surface_ids,b.omegas,cache_directory=tmp_path,cancel=lambda:len(calls)==1)
    assert caught.value.completed==(.7,) and caught.value.missing==(1.1,)
    bank=module.prepare_excitation_bank(b.vertices,b.faces,b.surface_ids,b.omegas,cache_directory=tmp_path)
    assert calls==[.7,1.1]
    migrated=module.migrate_pressure_bank(b,cache_directory=tmp_path)
    assert 'migration' in json.loads(migrated.request_json)
    unchanged=module.prepare_excitation_bank(b.vertices,b.faces,b.surface_ids,b.omegas,cache_directory=tmp_path)
    assert unchanged.bank_id==bank.bank_id and unchanged.bank_id!=migrated.bank_id
    recovered=module.recover_pressure_bank(migrated,cache_directory=tmp_path)
    np.testing.assert_allclose(recovered.responses,b.responses,rtol=0,atol=1e-12)
    other=tmp_path/'forbidden'
    with pytest.raises(PressureBankError,match='WORLD0'):
        module.migrate_pressure_bank(replace(b,phase_origin=(1,0,0),bank_id=''),cache_directory=other)
    assert not other.exists()


def test_explicit_repair_only_bad_rows_and_assembled_manifest(tmp_path,fake_backend):
    b,calls,*_=fake_backend
    bank=module.prepare_excitation_bank(b.vertices,b.faces,b.surface_ids,b.omegas,cache_directory=tmp_path)
    (tmp_path/'solutions'/(bank.solution_ids[0]+'.npz')).unlink()
    warm=module.prepare_excitation_bank(b.vertices,b.faces,b.surface_ids,b.omegas,cache_directory=tmp_path)
    assert warm.bank_id==bank.bank_id and len(calls)==2
    with pytest.raises(PressureBankError):module.recover_pressure_bank(warm,cache_directory=tmp_path)
    repaired=module.prepare_excitation_bank(b.vertices,b.faces,b.surface_ids,b.omegas,cache_directory=tmp_path,repair_cache=True)
    assert calls==[.7,1.1,.7] and repaired.solution_ids[1]==bank.solution_ids[1]
    module.recover_pressure_bank(repaired,cache_directory=tmp_path)
    manifest=next((tmp_path/'force-banks').glob('*.json'));manifest.write_text('corrupt')
    with pytest.raises(PressureBankError):module.prepare_excitation_bank(b.vertices,b.faces,b.surface_ids,b.omegas,cache_directory=tmp_path)
    restored=module.prepare_excitation_bank(b.vertices,b.faces,b.surface_ids,b.omegas,cache_directory=tmp_path,repair_cache=True)
    assert len(calls)==3 and restored.bank_id==repaired.bank_id
    row=module._paths(tmp_path,json.loads(restored.request_json),float(b.omegas[0]))
    for bad in ('null','[]'):
        row.write_text(bad)
        module.prepare_excitation_bank(b.vertices,b.faces,b.surface_ids,b.omegas,cache_directory=tmp_path,repair_cache=True)
    assert calls==[.7,1.1,.7,.7,.7]


def test_pressure_digests_custom_panel_ids_and_heading_preflight(tmp_path,fake_backend):
    b,calls,*_=fake_backend
    migrated=module.migrate_pressure_bank(replace(b,panel_ids=(97,),bank_id=''),cache_directory=tmp_path)
    recovered=module.recover_pressure_bank(migrated,cache_directory=tmp_path)
    assert recovered.panel_ids==(97,)
    module.validate_pressure_binding(migrated,recovered)
    wrong=replace(recovered,rho=1000.,bank_id='')
    with pytest.raises(PressureBankError,match='physical'):module.validate_pressure_binding(migrated,wrong)
    assert module.canonical_heading(math.radians(-1e-300))==0.
    module.prepare_excitation_bank(b.vertices,b.faces,b.surface_ids,[.7],cache_directory=tmp_path,heading=-1e-300)
    assert len(calls)==1


def test_current_provider_recovery_pressure_owner_codec_binding(tmp_path,fake_backend):
    b,calls,*_=fake_backend
    forces=module.prepare_excitation_bank(b.vertices,b.faces,b.surface_ids,b.omegas,cache_directory=tmp_path)
    assert 'migration' not in json.loads(forces.request_json)
    recovered=module.recover_pressure_bank(forces,cache_directory=tmp_path)
    from anyhydro.pressure_bank import PressureResponseBank
    restored=PressureResponseBank.from_dict(recovered.to_dict())
    module.validate_pressure_binding(forces,restored)
    assert restored.bank_id==recovered.bank_id and calls==list(b.omegas)
