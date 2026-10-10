# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from dataclasses import replace
import hashlib,json,os,sys
import pytest
from anyfileio.hams import HamsMechanicalInput
from anyhydro.first_slice import make_first_slice_bodies
from anyhydro.records import HydroRequest
from anyhydro.hams_producer import HamsProducerBuild,SOURCE_IDENTITIES
from anyhydro.hams_inputs import prepare_hams_deck,HamsDeckError
from anyhydro.hams_backend import HamsBackend,HamsExecutionLimits,HamsExecutionError,HamsCancelledError,HamsDeadlineError,load_completed_hams_run,execute_prepared_hams
from anyhydro.hashing import dataset_content_hash
from anyhydro.errors import CapabilityError

@pytest.fixture
def case(tmp_path,monkeypatch):
    source=os.environ.get('HAMS_SUPERVISOR_STORED_FIXTURE')
    if not source:pytest.fail('Explicit stored-output lifecycle fixture required; no numerical producer may execute')
    source=Path(source)
    bodies=make_first_slice_bodies(subdivisions=4);request=HydroRequest([.7],[0.],150.,1000.,9.80665,False,False,{})
    eye=tuple(tuple(float(i==j) for j in range(6)) for i in range(6));zero=tuple((0.,)*6 for _ in range(6))
    mechanics={b.body_id:HamsMechanicalInput(b.center_of_mass,eye,zero,zero,zero,zero) for b in bodies}
    centers={b.body_id:(b.rotation_center[0],0.,-.25) for b in bodies}
    exe=Path(sys.executable);build=HamsProducerBuild(exe,hashlib.sha256(exe.read_bytes()).hexdigest(),next(iter(SOURCE_IDENTITIES)),
        'windows' if os.name=='nt' else 'linux','synthetic producer lifecycle fixture; stored HAMS output only')
    script=tmp_path/'fixture.py'
    script.write_text('''import pathlib,sys,os,json,shutil,time,subprocess
mode,source=sys.argv[1:3];path=pathlib.Path.cwd()
print(json.dumps({key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')}),flush=True)
(path/'started').write_text('started')
if mode=='tree':
 child='import pathlib,time; p=pathlib.Path("heartbeat"); n=0\\nwhile True:\\n p.write_text(str(n));n+=1;time.sleep(.01)'
 subprocess.Popen([sys.executable,'-c',child]);time.sleep(60)
if mode=='delay-success':time.sleep(.2)
if mode=='crash':
 (path/'Output'/'partial.txt').write_text('incomplete');print('crash diagnostics',file=sys.stderr,flush=True);sys.exit(7)
for item in pathlib.Path(source).rglob('*'):
 if item.is_file():
  target=path/'Output'/item.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(item,target)
if mode=='partial':(path/'Output/Hams_format/OAMASS1.txt').unlink()
print('fixture finished',flush=True)
''')
    mode=['success']
    import anyhydro.hams_backend as hb
    # Explicit test-private command override: tests run Python copying stored
    # evidence, never the real numerical producer. Production has no override API.
    monkeypatch.setattr(hb,'_command',lambda b:(str(b.executable_path),str(script),mode[0],str(source),'Input','Output'))
    backend=HamsBackend(build,mechanics,centers,workroot=tmp_path/'runs',limits=HamsExecutionLimits(wall_seconds=4.),irregular_frequency_removal=True)
    return dict(bodies=bodies,request=request,mechanics=mechanics,centers=centers,build=build,backend=backend,mode=mode,tmp=tmp_path)

def run(c):return c['backend'].solve(c['bodies'],c['request'])
def receipt(c):return json.loads((c['backend'].last_run_directory/'run.json').read_text())

def test_connected_success_seal_replay_full_coupling_and_effective_threads(case):
    ds=run(case);path=case['backend'].last_run_directory;r=receipt(case)
    assert r['state']=='completed' and r['exit_code']==0 and r['wall_seconds']>0
    assert all(v=='1' for v in r['environment'].values())
    assert ds.added_mass.shape==(1,12,12) and ds.attrs['qualification_status']=='unqualified'
    restored=load_completed_hams_run(path,bodies=case['bodies'],request=case['request'],producer_build=case['build'])
    assert dataset_content_hash(restored)==dataset_content_hash(ds)
    assert json.loads((path/'stdout.log').read_text().splitlines()[0])==r['environment']
    assert not case['backend'].capabilities().provides_added_mass_infinite

@pytest.mark.parametrize('mode',['partial','crash'])
def test_partial_or_crashing_producer_never_completes(case,mode):
    case['mode'][0]=mode
    with pytest.raises(HamsExecutionError):run(case)
    path=case['backend'].last_run_directory;assert receipt(case)['state']=='failed' and not (path/'completed.json').exists()
    assert (path/'stdout.log').exists() and (path/'stderr.log').exists()
    with pytest.raises(HamsExecutionError):load_completed_hams_run(path,bodies=case['bodies'],request=case['request'],producer_build=case['build'])

@pytest.mark.parametrize('when',['queued','tree','after-result'])
def test_cancellation_preserves_evidence_and_no_seal(case,when):
    if when=='tree':case['mode'][0]='tree'
    def cancelled():
        p=case['backend'].last_run_directory
        return when=='queued' or (p/('heartbeat' if when=='tree' else 'result.nc')).exists()
    case['backend'].should_cancel=cancelled
    with pytest.raises(HamsCancelledError):run(case)
    p=case['backend'].last_run_directory;assert receipt(case)['state']=='cancelled' and not (p/'completed.json').exists()
    if when=='queued':assert not (p/'stdout.log').exists()
    if when=='tree':
        import time
        value=(p/'heartbeat').read_bytes();time.sleep(.15);assert (p/'heartbeat').read_bytes()==value
    if when=='after-result':assert (p/'result.nc').exists()

def test_deadline_terminates_owned_tree(case):
    case['mode'][0]='tree';case['backend'].limits=replace(case['backend'].limits,wall_seconds=.6)
    with pytest.raises(HamsDeadlineError):run(case)
    assert receipt(case)['state']=='failed' and receipt(case)['error_type']=='HamsDeadlineError'
    assert not (case['backend'].last_run_directory/'completed.json').exists()

def test_input_change_during_job_cannot_complete(case):
    case['mode'][0]='delay-success';changed=[False]
    def mutate_input():
        p=case['backend'].last_run_directory
        if (p/'started').exists() and not changed[0]:
            with (p/'Input/HullMesh_1.pnl').open('ab') as stream:stream.write(b'\n! changed during execution\n')
            changed[0]=True
        return False
    case['backend'].should_cancel=mutate_input
    with pytest.raises(HamsExecutionError):run(case)
    assert changed[0] and receipt(case)['state']=='failed'

@pytest.mark.parametrize('which',['result','receipt','raw','input','missing-seal','request','body'])
def test_saved_completed_artifact_tamper_staleness_and_missing_sidecar_refuse(case,which):
    run(case);p=case['backend'].last_run_directory;request=case['request'];bodies=case['bodies']
    if which=='missing-seal':(p/'completed.json').unlink()
    elif which=='request':request=replace(request,omegas=[.8])
    elif which=='body':bodies=(replace(bodies[0],rotation_center=(1.,0.,0.)),bodies[1])
    else:
        path=p/{'result':'result.nc','receipt':'run.json','raw':'Output/Wamit_format/ExcForce.3','input':'Input/HullMesh_1.pnl'}[which]
        with path.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(HamsExecutionError):load_completed_hams_run(p,bodies=bodies,request=request,producer_build=case['build'])

def test_corrected_deck_opt_in_and_legacy_pin_preserved(case):
    args=dict(directory=case['tmp']/'deck',bodies=case['bodies'],request=case['request'],mechanics=case['mechanics'],executable_path=case['build'].executable_path,irregular_frequency_removal=True)
    with pytest.raises(HamsDeckError):prepare_hams_deck(**args)
    deck=prepare_hams_deck(**args,producer_build=case['build']);m=deck.verify(producer_build=case['build'],executable_path=case['build'].executable_path)
    assert m['format']=='anyhydro-corrected-hams-deck-v2' and m['producer_build']==case['build'].identity()
    with pytest.raises(HamsDeckError):deck.verify(producer_build=replace(case['build'],build_description='different build'))

@pytest.mark.parametrize('kind',['body','frequency','panels','limits'])
def test_unadmitted_inventory_never_launches(case,kind):
    if kind=='body':case['backend'].limits=replace(case['backend'].limits,max_bodies=1)
    elif kind=='frequency':case['backend'].limits=replace(case['backend'].limits,max_frequencies=1);case['request']=replace(case['request'],omegas=[.7,.8])
    elif kind=='panels':case['backend'].limits=replace(case['backend'].limits,max_panels=1)
    else:case['request']=replace(case['request'],include_infinite_frequency=True)
    with pytest.raises((HamsExecutionError,CapabilityError)):run(case)
    assert case['backend'].last_run_directory is None

def test_reentrant_run_refuses_and_does_not_change_owned_case(case):
    seen=[]
    def attempt():
        if not seen:
            path=case['backend'].last_run_directory
            with pytest.raises(HamsExecutionError,match='active run'):run(case)
            assert case['backend'].last_run_directory==path
            seen.append(True)
        return False
    case['backend'].should_cancel=attempt
    assert run(case) is not None and seen

def test_completed_case_cannot_be_rerun(case):
    run(case);path=case['backend'].last_run_directory
    from anyhydro.hams_inputs import HamsPreparedDeck
    original=(path/'run.json').read_bytes()
    deck=HamsPreparedDeck(path,json.loads(original)['case_hash'])
    with pytest.raises(HamsExecutionError,match='already claimed'):
        execute_prepared_hams(deck,case['bodies'],case['request'],case['build'],case['centers'],limits=case['backend'].limits)
    assert (path/'run.json').read_bytes()==original

def test_cleanup_failure_retains_cancellation_and_diagnostics(case,monkeypatch):
    import anyhydro.hams_backend as hb
    original=hb.OwnedProcess
    class BrokenCleanup:
        def __init__(self,*args,**kwargs):
            self.owned=original(*args,**kwargs);self.pid=self.owned.pid
        def poll(self):return self.owned.poll()
        def close(self):
            self.owned.close();raise OSError('simulated cleanup failure after tree stopped')
    monkeypatch.setattr(hb,'OwnedProcess',BrokenCleanup)
    case['mode'][0]='tree'
    case['backend'].should_cancel=lambda:(case['backend'].last_run_directory/'heartbeat').exists()
    with pytest.raises(HamsCancelledError):run(case)
    assert receipt(case)['state']=='cancelled' and 'cleanup failure' in receipt(case)['cleanup_error']
    assert not (case['backend'].last_run_directory/'completed.json').exists()

def test_linked_output_refuses_before_launch(case,monkeypatch):
    import anyhydro.hams_backend as hb
    prepare=hb.prepare_hams_deck
    def linked(*args,**kwargs):
        deck=prepare(*args,**kwargs)
        # Mock detection rather than requiring elevated Windows symlink privilege.
        original=Path.is_symlink
        monkeypatch.setattr(Path,'is_symlink',lambda p:p.name=='Hams_format' or original(p))
        return deck
    monkeypatch.setattr(hb,'prepare_hams_deck',linked)
    with pytest.raises(HamsExecutionError,match='Linked'):run(case)
    assert not (case['backend'].last_run_directory/'run.json').exists() and not (case['backend'].last_run_directory/'stdout.log').exists()

def test_nested_backend_settings_refuse_as_capability(case):
    case['request']=replace(case['request'],backend_settings={'unsupported':{'x':1}})
    with pytest.raises(CapabilityError,match='backend_settings'):run(case)
    assert case['backend'].last_run_directory is None

@pytest.mark.parametrize('event',['cancel','deadline'])
def test_cancellation_or_deadline_during_completed_receipt_cannot_seal(case,monkeypatch,event):
    import anyhydro.hams_backend as hb
    original=hb._write;triggered=[False]
    case['backend'].should_cancel=lambda:triggered[0] if event=='cancel' else False
    def write(path,value):
        original(path,value)
        if path.name=='run.json' and value['state']=='completed':
            triggered[0]=True
            if event=='deadline':
                # Cross the exact run deadline without a costly wall-clock delay.
                started=value['started_monotonic'];wall=value['execution_limits']['wall_seconds']
                monkeypatch.setattr(hb.time,'perf_counter',lambda:started+wall+1.)
    monkeypatch.setattr(hb,'_write',write)
    with pytest.raises(HamsCancelledError if event=='cancel' else HamsDeadlineError):run(case)
    assert triggered[0] and not (case['backend'].last_run_directory/'completed.json').exists()
    assert receipt(case)['state']==('cancelled' if event=='cancel' else 'failed')

def test_editing_backend_configuration_does_not_change_run_snapshot(case):
    original=case['build'];changed=[]
    def edit():
        if not changed:
            case['backend'].build=replace(original,build_description='edited configuration')
            case['backend'].limits=replace(case['backend'].limits,max_bodies=1)
            changed.append(True)
        return False
    case['backend'].should_cancel=edit
    assert run(case) is not None
    assert receipt(case)['producer']==original.identity() and receipt(case)['execution_limits']['max_bodies']==2

def test_caller_mesh_edit_cannot_mutate_running_snapshot(case):
    import numpy as np
    vertices=case['bodies'][0].mesh.vertices;original=vertices.copy();edited=[]
    def edit():
        if not edited:
            vertices.setflags(write=True);vertices[0,0]+=1.;vertices.setflags(write=False);edited.append(True)
        return False
    case['backend'].should_cancel=edit
    try:
        ds=run(case)
        assert edited and not np.array_equal(vertices,original)
        hashes=ds.attrs['mesh_hashes']
        if isinstance(hashes,str):hashes=json.loads(hashes)
        assert hashes[case['bodies'][0].body_id]==case['bodies'][0].mesh.mesh_hash
    finally:
        vertices.setflags(write=True);vertices[:]=original;vertices.setflags(write=False)

def test_replay_refuses_linked_result_even_with_matching_stored_hash(case,monkeypatch):
    run(case);path=case['backend'].last_run_directory
    original=Path.is_symlink
    monkeypatch.setattr(Path,'is_symlink',lambda p:p.name=='result.nc' or original(p))
    with pytest.raises(HamsExecutionError,match='Linked'):
        load_completed_hams_run(path,bodies=case['bodies'],request=case['request'],producer_build=case['build'])

def test_existing_result_refuses_before_launch(case,monkeypatch):
    import anyhydro.hams_backend as hb
    prepare=hb.prepare_hams_deck
    def occupied(*args,**kwargs):
        deck=prepare(*args,**kwargs);(deck.directory/'result.nc').write_bytes(b'preserve existing result');return deck
    monkeypatch.setattr(hb,'prepare_hams_deck',occupied)
    with pytest.raises(HamsExecutionError,match='Exclusive unexecuted'):run(case)
    path=case['backend'].last_run_directory
    assert (path/'result.nc').read_bytes()==b'preserve existing result' and not (path/'stdout.log').exists()

def test_non_os_receipt_failure_still_cleans_live_tree_and_keeps_primary_cause(case,monkeypatch):
    import anyhydro.hams_backend as hb
    import time
    original=hb._write;failed=[False];case['mode'][0]='tree'
    def write(path,value):
        if value.get('state')=='running':
            end=time.perf_counter()+2.
            while not (path.parent/'heartbeat').exists() and time.perf_counter()<end:time.sleep(.02)
            assert (path.parent/'heartbeat').exists()
            raise HamsExecutionError('primary injected receipt refusal')
        if value.get('state')=='failed' and not failed[0]:
            failed[0]=True;raise RuntimeError('secondary injected persistence failure')
        original(path,value)
    monkeypatch.setattr(hb,'_write',write)
    with pytest.raises(HamsExecutionError,match='primary injected'):run(case)
    p=case['backend'].last_run_directory;r=receipt(case)
    assert r['state']=='failed' and 'secondary injected' in r['receipt_error']
    value=(p/'heartbeat').read_bytes();time.sleep(.15);assert (p/'heartbeat').read_bytes()==value
    assert not (p/'completed.json').exists()

def test_windows_reparse_attribute_detection_does_not_depend_on_new_path_api(tmp_path,monkeypatch):
    import anyhydro.hams_backend as hb
    from types import SimpleNamespace
    p=tmp_path/'reparse'
    monkeypatch.setattr(Path,'is_symlink',lambda self:False)
    monkeypatch.setattr(Path,'lstat',lambda self:SimpleNamespace(st_file_attributes=0x400))
    assert hb._linked(p)
