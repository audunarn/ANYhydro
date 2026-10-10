# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from dataclasses import replace
import json,os,sys
import pytest
from test_hams_backend import case
from anyhydro.hams_producer import LIMIT_SOURCE_IDENTITY
from anyhydro.hams_backend import HamsCancelledError,HamsExecutionError,load_completed_hams_run
from anyhydro.hashing import dataset_content_hash
from anyhydro.errors import CapabilityError

def configure(case,monkeypatch):
    import anyhydro.hams_backend as hb
    stored=os.environ.get('HAMS_LIMIT_STORED_FIXTURE')
    if not stored:pytest.fail('Explicit stored limit output required; no numerical producer may execute')
    output=Path(stored)
    script=case['tmp']/'fixture.py'
    monkeypatch.setattr(hb,'_command',lambda b:(str(b.executable_path),str(script),case['mode'][0],str(output),'Input','Output'))
    case['build']=replace(case['build'],source_tree_sha256=LIMIT_SOURCE_IDENTITY)
    case['backend'].build=case['build']
    case['request']=replace(case['request'],include_infinite_frequency=True)
    return case

def test_repaired_source_connected_ainf_completion_reopen(case,monkeypatch):
    c=configure(case,monkeypatch);ds=c['backend'].solve(c['bodies'],c['request'])
    path=c['backend'].last_run_directory
    caps=c['backend'].capabilities()
    assert caps.provides_added_mass_infinite and caps.supports_infinite_frequency and not caps.provides_free_surface
    assert ds.added_mass_infinite.shape==(12,12) and ds.attrs['qualification_status']=='unqualified'
    reopened=load_completed_hams_run(path,bodies=c['bodies'],request=c['request'],producer_build=c['build'])
    assert dataset_content_hash(ds)==dataset_content_hash(reopened)
    provenance=reopened.attrs['wamit_import']
    if isinstance(provenance,str):provenance=json.loads(provenance)
    assert provenance['native_source']['zero_frequency_evidence']['admitted'] is False
    assert json.loads((path/'run.json').read_text())['state']=='completed'

def test_limit_run_cancel_after_result_preserves_unsealed_evidence(case,monkeypatch):
    c=configure(case,monkeypatch)
    c['backend'].should_cancel=lambda:(c['backend'].last_run_directory/'result.nc').exists()
    with pytest.raises(HamsCancelledError):c['backend'].solve(c['bodies'],c['request'])
    path=c['backend'].last_run_directory
    assert (path/'result.nc').exists() and not (path/'completed.json').exists()
    assert json.loads((path/'run.json').read_text())['state']=='cancelled'

@pytest.mark.parametrize('which',['old-source-infinite','zero'])
def test_unadmitted_requests_refuse_before_launch(case,which):
    c=case
    c['request']=replace(c['request'],include_infinite_frequency=which=='old-source-infinite',include_zero_frequency=which=='zero')
    with pytest.raises(CapabilityError):c['backend'].solve(c['bodies'],c['request'])
    assert c['backend'].last_run_directory is None

def test_frequency_budget_counts_both_native_limit_branches(case,monkeypatch):
    c=configure(case,monkeypatch);c['backend'].limits=replace(c['backend'].limits,max_frequencies=2)
    with pytest.raises(HamsExecutionError,match='inventory exceeds execution limits'):c['backend'].solve(c['bodies'],c['request'])
    assert c['backend'].last_run_directory is None
