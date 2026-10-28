import sys
import types
import numpy as np
import pytest
from anyhydro.pressure_bank import (PressureResponseBank, PressureBankError,
    PressurePreparationError, prepare_pressure_bank)


def bank(**changes):
    data = dict(vertices=[[0,0,0],[1,0,0],[1,1,0],[0,1,0]], faces=[[0,1,2,3]],
        surface_ids=('bottom',), centers=[[.5,.5,0]], normals=[[0,0,1]], areas=[1.],
        omegas=[.7,1.1], responses=[[1+2j],[3-4j]])
    data.update(changes)
    return PressureResponseBank(**data)


def test_immutable_identity_roundtrip(tmp_path):
    b = bank(); path = tmp_path/'bank.npz'; b.save(path)
    restored = PressureResponseBank.load(path)
    assert restored.bank_id == b.bank_id
    for a in (b.responses, b.vertices, b.omegas):
        with pytest.raises(ValueError): a.setflags(write=True)
    assert bank(heading=.2).bank_id != b.bank_id
    assert bank(surface_ids=('other',)).bank_id != b.bank_id
    with pytest.raises(PressureBankError): bank(bank_id='stale')
    path.write_bytes(b'corrupt')
    with pytest.raises(PressureBankError): PressureResponseBank.load(path)


def test_owner_record_roundtrip_preserves_complex_signed_zero_identity():
    responses=np.empty((2,1),complex)
    responses.real=[[-0.],[1.]];responses.imag=[[0.],[-0.]]
    b=bank(responses=responses)
    restored=PressureResponseBank.from_dict(b.to_dict())
    assert restored.bank_id==b.bank_id
    assert restored.responses.tobytes()==b.responses.tobytes()


def test_bank_refuses_bad_data():
    for change in (dict(omegas=[.7,.7]), dict(responses=[[np.nan],[1]]),
                   dict(normals=[[0,0,2]]), dict(panel_ids=(True,)),
                   dict(omegas=[.7+1j,1.1]),dict(omegas=[True,1.1]),
                   dict(vertices=[[1j,0,0]]*4)):
        with pytest.raises(PressureBankError): bank(**change)


@pytest.fixture
def backend(monkeypatch):
    import anyhydro._pressure_body as prep
    import anyhydro.pressure_bank as module
    calls=[]
    class Solver:
        engine = types.SimpleNamespace(green_function='default')
        exportable_settings = dict(method='indirect', linear_solver='lu_decomposition')
        def solve(self, problem, **settings):
            calls.append((problem.omega, settings))
            if getattr(self, 'fail', False): raise RuntimeError('failure')
            pressure=problem.omega+1j
            return types.SimpleNamespace(pressure=np.array([pressure]), forces=dict(
                Surge=0j,Sway=0j,Heave=-pressure,Roll=-.5*pressure,Pitch=.5*pressure,Yaw=0j))
    class Problem:
        def __init__(self, **kwargs): self.__dict__.update(kwargs)
    cpt = types.ModuleType('capytaine'); cpt.__file__=__file__
    cpt.BEMSolver=Solver; cpt.DiffractionProblem=Problem
    airy=types.ModuleType('capytaine.bem.airy_waves')
    airy.airy_waves_pressure=lambda centers, problem: np.array([0j])
    airy.froude_krylov_force=lambda problem:dict.fromkeys(('Surge','Sway','Heave','Roll','Pitch','Yaw'),0j)
    monkeypatch.setitem(sys.modules, 'capytaine', cpt)
    monkeypatch.setitem(sys.modules, 'capytaine.bem.airy_waves', airy)
    monkeypatch.setattr(module.importlib.metadata, 'version', lambda name: '3.0.0')
    def prepare(cpt, vertices, faces, labels, **kw):
        b=bank(omegas=[1], responses=[[0]])
        mesh=types.SimpleNamespace(faces_centers=b.centers, faces_normals=b.normals, faces_areas=b.areas)
        return np.asarray(vertices), np.asarray(faces), tuple(labels), mesh, object()
    monkeypatch.setattr(prep, 'prepare_body', prepare)
    return calls, Solver


def test_exact_omega_cancel_resume_extension_and_cache(tmp_path, backend):
    calls, _ = backend; b=bank()
    frequencies=[float.fromhex('0x1.0000000000001p+0'), 1.3]
    settings=dict(cache_directory=tmp_path, max_wall_seconds=30, max_missing_solves=3)
    progress=[]
    with pytest.raises(PressurePreparationError) as caught:
        prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,frequencies,
            progress=progress.append, cancel=lambda: len(calls)==1, **settings)
    assert caught.value.completed == (frequencies[0],)
    assert caught.value.missing == (frequencies[1],)
    assert len(list(tmp_path.glob('*.npz'))) == 1
    resumed_progress=[]
    full=prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,frequencies,
        progress=resumed_progress.append,**settings)
    estimate=resumed_progress[0].estimate
    assert estimate.panels==1 and estimate.requested_frequency_count==2
    assert estimate.cache_hits==1 and estimate.missing_solve_count==1
    assert estimate.response_storage_bytes==32 and estimate.dense_matrix_bytes==48
    assert estimate.max_missing_solves==3 and estimate.max_wall_seconds==30.
    assert resumed_progress[0].solved_count==0
    assert [x[0] for x in calls] == frequencies
    assert tuple(full.omegas) == tuple(frequencies)
    again=prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,frequencies,**settings)
    assert again.bank_id == full.bank_id and len(calls)==2
    prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,frequencies+[1.7],**settings)
    assert len(calls)==3
    prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,frequencies,heading=.2,**settings)
    assert len(calls)==5
    entry=next(tmp_path.glob('*.npz')); entry.write_bytes(b'broken')
    # Find the owning heading key by running both headings; corruption must be diagnosed.
    failures=0
    for heading in (0.,.2):
        try: prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,frequencies,heading=heading,**settings)
        except PressurePreparationError: failures+=1
    assert failures==1


def test_failed_preparation_preserves_entries(tmp_path, backend):
    calls, solver=backend; b=bank(); solver.fail=True
    with pytest.raises(PressurePreparationError) as caught:
        prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,[1.,2.],cache_directory=tmp_path)
    assert caught.value.completed == () and caught.value.missing == (1.,2.)
    assert not list(tmp_path.glob('*.npz'))


def test_limits_and_nonzero_origin_refused(tmp_path, backend):
    b=bank()
    for settings in (dict(max_missing_solves=True),dict(max_wall_seconds=0),
                     dict(max_panels=601),dict(phase_origin=(1,0,0))):
        with pytest.raises(PressureBankError):
            prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,[1],cache_directory=tmp_path,**settings)
    with pytest.raises(PressurePreparationError) as caught:
        prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,[1],cache_directory=tmp_path,max_missing_solves=0)
    assert caught.value.missing==(1.,)
    assert caught.value.progress.estimate.missing_solve_count==1
    assert caught.value.progress.estimate.response_storage_bytes==16
    assert caught.value.progress.estimate.max_missing_solves==0


def test_overall_wall_limit_and_progress(tmp_path, backend, monkeypatch):
    import anyhydro.pressure_bank as module
    b=bank(); ticks=iter([0.,2.,2.])
    monkeypatch.setattr(module.time,'perf_counter',lambda:next(ticks))
    with pytest.raises(PressurePreparationError) as caught:
        prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,[1.],cache_directory=tmp_path,max_wall_seconds=1.)
    assert caught.value.missing==(1.,) and caught.value.progress.elapsed_seconds==2.
    assert not backend[0]


def test_cached_only_deadline_after_final_cache_load(tmp_path,backend,monkeypatch):
    import anyhydro.pressure_bank as module
    b=bank()
    settings=dict(cache_directory=tmp_path,max_missing_solves=1,max_wall_seconds=1.)
    prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,[1.],**settings)
    real_load=PressureResponseBank.load
    clock=[0.]
    def slow_load(path):
        result=real_load(path); clock[0]=2.; return result
    monkeypatch.setattr(PressureResponseBank,'load',slow_load)
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    with pytest.raises(PressurePreparationError) as caught:
        prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,[1.],**settings)
    assert caught.value.completed==(1.,) and caught.value.missing==()
    assert len(backend[0])==1


def test_cached_only_deadline_after_final_bank_assembly(tmp_path,backend,monkeypatch):
    import anyhydro.pressure_bank as module
    b=bank(); settings=dict(cache_directory=tmp_path,max_wall_seconds=1.)
    prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,[1.],**settings)
    clock=[0.]; stack=np.stack
    def slow_stack(values):
        result=stack(values); clock[0]=2.; return result
    monkeypatch.setattr(module.np,'stack',slow_stack)
    monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    with pytest.raises(PressurePreparationError) as caught:
        prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,[1.],**settings)
    assert caught.value.completed==(1.,) and caught.value.missing==()
    assert len(backend[0])==1


def test_source_at_or_below_seabed_refused_before_backend():
    from anyhydro._pressure_body import prepare_body
    b=bank()
    for depth in (1.,1.1):
        vertices=b.vertices.copy();vertices[:,2]=-depth
        with pytest.raises(ValueError,match='seabed'):
            prepare_body(object(),vertices,b.faces,b.surface_ids,period=6.,heading=0.,water_depth=1.,rho=1025.,max_panels=600)


@pytest.mark.parametrize('frequency',[True,np.bool_(False),1+0j,np.complex128(1+2j),'1.0'])
def test_preparation_rejects_nonreal_frequency_before_backend(tmp_path,frequency):
    b=bank()
    with pytest.raises(PressureBankError):
        prepare_pressure_bank(b.vertices,b.faces,b.surface_ids,[frequency],cache_directory=tmp_path)
