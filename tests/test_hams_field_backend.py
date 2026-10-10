# SPDX-License-Identifier: MPL-2.0
"""Stored-output lifecycle tests. The test-private producer copies files only."""
from pathlib import Path
from dataclasses import replace
import hashlib,json,os
import numpy as np
import pytest
from test_hams_backend import case,run,receipt
from anyhydro.hams_producer import GRAVITY_SOURCE_IDENTITY
from anyhydro.hams_backend import HamsBackend,HamsExecutionError,HamsCancelledError,HamsDeadlineError,load_completed_hams_run,load_completed_hams_fields
from anyhydro.hams_inputs import HamsDeckError,HamsPreparedDeck
from anyhydro.hams_fields import load_hams_fields,save_hams_fields
from anyhydro.hams_field_points import HamsFieldPointError
from anyhydro.errors import CapabilityError
from anyhydro.hashing import dataset_content_hash

POINTS=((-6.,-4.,0.),(6.,-4.,0.),(6.,4.,0.),(-6.,4.,0.),(-6.,-4.,-2.),(6.,-4.,-2.),(6.,4.,-2.),(-6.,4.,-2.))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

@pytest.fixture
def field_case(case,monkeypatch):
    c=case;stored=os.environ.get('HAMS_FIELD_STORED_FIXTURE')
    if not stored:pytest.fail('Explicit saved corrected field evidence required')
    c['build']=replace(c['build'],source_tree_sha256=GRAVITY_SOURCE_IDENTITY)
    c['backend']=HamsBackend(c['build'],c['mechanics'],c['centers'],workroot=c['tmp']/'field-runs',limits=c['backend'].limits,irregular_frequency_removal=True,field_points=POINTS)
    import anyhydro.hams_backend as hb
    script=c['tmp']/'fixture.py'
    monkeypatch.setattr(hb,'_command',lambda b:(str(b.executable_path),str(script),c['mode'][0],stored,'Input','Output'))
    return c

def replay(c):
    return load_completed_hams_fields(c['backend'].last_run_directory,bodies=c['bodies'],request=c['request'],producer_build=c['build'])

def reseal(path):
    r=json.loads((path/'run.json').read_text());s=json.loads((path/'completed.json').read_text())
    r['fields']['sha256']=sha(path/'fields.json');r['fields']['content_hash']=load_hams_fields(path/'fields.json').content_hash()
    (path/'run.json').write_text(json.dumps(r));s['receipt_sha256']=sha(path/'run.json');s['fields_sha256']=r['fields']['sha256'];(path/'completed.json').write_text(json.dumps(s))

def test_connected_fields_sealed_reopen_and_source_specific_capabilities(field_case):
    c=field_case;ds=run(c);data=replay(c);path=c['backend'].last_run_directory
    assert data.points==POINTS and data.radiation.shape==(1,8,12) and data.total.shape==(1,1,8)
    assert data.parent_dataset_hash==dataset_content_hash(ds) and receipt(c)['state']=='completed'
    assert np.array_equal(data.scattered,data.total-data.incident)
    assert data.quantities==('elevation',)*4+('dynamic_pressure',)*4
    for array in (data.incident,data.total,data.radiation,data.scattered):
        with pytest.raises(ValueError):array.setflags(write=True)
    caps=c['backend'].capabilities();assert caps.provides_free_surface and not caps.provides_added_mass_infinite and not caps.provides_panel_pressure
    assert json.loads((path/'case.json').read_text())['field_points']==[list(p) for p in POINTS]
    c['backend'].field_points=POINTS[4:];assert not c['backend'].capabilities().provides_free_surface

@pytest.mark.parametrize('which',['cap','infinite','zero','geometry'])
def test_field_admission_and_limits_before_any_launch(field_case,which):
    c=field_case
    if which=='cap':c['backend'].limits=replace(c['backend'].limits,max_field_points=7)
    if which=='infinite':c['request']=replace(c['request'],include_infinite_frequency=True)
    if which=='zero':c['request']=replace(c['request'],include_zero_frequency=True)
    if which=='geometry':c['backend'].field_points=(tuple(c['bodies'][0].mesh.vertices.mean(axis=0)),)
    with pytest.raises((HamsExecutionError,CapabilityError,HamsFieldPointError)):run(c)
    assert c['backend'].last_run_directory is None

def test_old_source_rejects_field_configuration_and_legacy_replay_remains(case):
    c=case
    with pytest.raises(CapabilityError):HamsBackend(c['build'],c['mechanics'],c['centers'],workroot=c['tmp']/'old',limits=c['backend'].limits,field_points=POINTS)
    run(c)
    with pytest.raises(CapabilityError,match='no requested'):replay(c)

@pytest.mark.parametrize('target',['fields.json','result.nc','run.json','Output/Wamit_format/PressureElevation.6p','Output/Wamit_format/AmssDamp.1','Input/ControlFile.in'])
def test_final_callback_artifact_mutation_cannot_commit(field_case,target):
    c=field_case;mutated=[False]
    def mutate():
        path=c['backend'].last_run_directory;r=path/'run.json'
        if r.exists() and json.loads(r.read_text()).get('state')=='completed' and not mutated[0]:
            p=path/target;p.write_bytes(p.read_bytes()+b'\n');mutated[0]=True
        return False
    c['backend'].should_cancel=mutate
    with pytest.raises(HamsExecutionError):run(c)
    assert mutated[0] and receipt(c)['state']=='failed' and not (c['backend'].last_run_directory/'completed.json').exists()

@pytest.mark.parametrize('when',['fields','final'])
def test_field_cancellation_keeps_unsealed_evidence(field_case,when):
    c=field_case
    def cancel():
        path=c['backend'].last_run_directory
        return (path/'fields.json').exists() if when=='fields' else (path/'run.json').exists() and json.loads((path/'run.json').read_text()).get('state')=='completed'
    c['backend'].should_cancel=cancel
    with pytest.raises(HamsCancelledError):run(c)
    path=c['backend'].last_run_directory
    assert receipt(c)['state']=='cancelled' and (path/'fields.json').exists() and not (path/'completed.json').exists()

@pytest.mark.parametrize('mode',['crash','partial'])
def test_failed_field_job_no_completed_result(field_case,mode):
    c=field_case;c['mode'][0]=mode
    with pytest.raises(HamsExecutionError):run(c)
    assert receipt(c)['state']=='failed' and not (c['backend'].last_run_directory/'completed.json').exists()

@pytest.mark.parametrize('which',['missing','bytes','reframed','total'])
def test_replay_rejects_missing_changed_or_resealed_semantics(field_case,which):
    c=field_case;run(c);path=c['backend'].last_run_directory;p=path/'fields.json'
    if which=='missing':p.unlink()
    if which=='bytes':p.write_bytes(p.read_bytes()+b' ')
    if which in ('reframed','total'):
        data=load_hams_fields(p)
        if which=='total':data=replace(data,total=data.total+1)
        else:
            m=json.loads(json.dumps(json.loads(p.read_text())['payload']['metadata']));key=m['target_body_order'][0];m['target_frames'][key]['reference_center'][0]+=.1;data=replace(data,metadata=m)
        p.unlink();save_hams_fields(data,p);reseal(path)
    with pytest.raises(HamsExecutionError):replay(c)
    with pytest.raises(HamsExecutionError):load_completed_hams_run(path,bodies=c['bodies'],request=c['request'],producer_build=c['build'])

def test_running_configuration_edits_do_not_change_requested_snapshot(field_case):
    c=field_case;c['mode'][0]='delay-success'
    def edit():
        if (c['backend'].last_run_directory/'started').exists():c['backend'].field_points=POINTS[4:]
        return False
    c['backend'].should_cancel=edit;run(c);assert replay(c).points==POINTS

def test_field_deck_caller_identity_is_checked(field_case):
    c=field_case;run(c);path=c['backend'].last_run_directory;r=receipt(c)
    deck=HamsPreparedDeck(path,r['case_hash'])
    with pytest.raises(HamsDeckError):deck.verify(bodies=c['bodies'],request=c['request'],producer_build=c['build'],executable_path=c['build'].executable_path,field_points=POINTS[4:])

def test_final_integrity_guard_cannot_exceed_deadline_and_complete(field_case,monkeypatch):
    import time
    from types import SimpleNamespace
    import anyhydro.hams_backend as hb
    c=field_case;offset=[0.];real_sha=hb._sha;real_clock=time.perf_counter
    def slow_guard(path):
        r=path.parent/'run.json'
        if path.name=='fields.json' and r.exists() and json.loads(r.read_text()).get('state')=='completed':offset[0]=10.
        return real_sha(path)
    monkeypatch.setattr(hb,'time',SimpleNamespace(perf_counter=lambda:real_clock()+offset[0],sleep=time.sleep))
    monkeypatch.setattr(hb,'_sha',slow_guard)
    with pytest.raises(HamsDeadlineError,match='before completion seal'):run(c)
    assert receipt(c)['state']=='failed' and receipt(c)['wall_seconds']>=10 and not (c['backend'].last_run_directory/'completed.json').exists()
