# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
import importlib.util,json,hashlib,shutil,os
import pytest

REPO=Path(__file__).resolve().parents[1]
def module(path):
    spec=importlib.util.spec_from_file_location('recipe',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
@pytest.fixture
def tool():return module(REPO/'tools/prepare_hams_phase_candidate.py')
@pytest.fixture
def source():
    path=Path(os.environ['HAMS_LIMIT_PRISTINE_SOURCE']);assert path.is_dir();return path

@pytest.mark.parametrize('gnu,limits',[(False,False),(True,False),(True,True)])
def test_exact_recipe_profiles_preserve_unmodified_sources_and_not_execute(tool,source,tmp_path,gnu,limits):
    out=tmp_path/'candidate';m=tool.prepare_candidate(source,out,gnu_compatible=gnu,preserve_frequency_limits=limits)
    assert m['qualified'] is False and m['solver_executed'] is False and len(m['candidate_sha256'])==33
    assert len(m['changed_files'])==(3 if gnu else 2)
    assert m['format']==('anyhydro-hams-limit-source-candidate-v2' if limits else 'anyhydro-hams-phase-source-candidate-v1')
    for name,value in m['source_sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==value
    for name,value in m['candidate_sha256'].items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==value
    if limits:
        assert m['limit_damping'].startswith('absent')
        assert b'ADVANCE=\'NO\'' in (out/'src/PotentWavForceMulti.f90').read_bytes()
        # This pinned tree carries its upstream notices in source headers.
        original=(source/'src/PotentWavForceMulti.f90').read_bytes().split(b'MODULE PotentWavForceMulti')[0]
        assert b'Apache License' in original
        assert (out/'src/PotentWavForceMulti.f90').read_bytes().startswith(original)

@pytest.mark.parametrize('gnu,limits',[(True,1),(True,None),('true',False),(False,True)])
def test_unproven_platform_or_nonboolean_flag_refuses_before_writing(tool,source,tmp_path,gnu,limits):
    out=tmp_path/'candidate'
    with pytest.raises(tool.ProducerRecipeError):tool.prepare_candidate(source,out,gnu_compatible=gnu,preserve_frequency_limits=limits)
    assert not out.exists()

def test_mutated_replacement_refuses_before_destination(tool,source,tmp_path):
    tools=tmp_path/'tools';tools.mkdir()
    for name in ('prepare_hams_phase_candidate.py','hams_phase_source_recipe.json','hams_reorder_limits.f90.inc'):shutil.copyfile(REPO/'tools'/name,tools/name)
    with (tools/'hams_reorder_limits.f90.inc').open('ab') as stream:stream.write(b'! changed\n')
    t=module(tools/'prepare_hams_phase_candidate.py');out=tmp_path/'candidate'
    with pytest.raises(t.ProducerRecipeError,match='identity'):t.prepare_candidate(source,out,gnu_compatible=True,preserve_frequency_limits=True)
    assert not out.exists()

def test_overwrite_and_inplace_refuse(tool,source,tmp_path):
    out=tmp_path/'candidate';out.mkdir();(out/'protected').write_text('keep')
    with pytest.raises(FileExistsError):tool.prepare_candidate(source,out,gnu_compatible=True,preserve_frequency_limits=True)
    assert (out/'protected').read_text()=='keep'
    with pytest.raises(tool.ProducerRecipeError):tool.prepare_candidate(source,source/'nested-candidate')

def test_cli_limit_option_reports_no_execution(tool,source,tmp_path,monkeypatch,capsys):
    import sys
    out=tmp_path/'cli';monkeypatch.setattr(sys,'argv',['prepare','--source',str(source),'--output',str(out),'--gnu-compatible','--preserve-frequency-limits'])
    tool.main();message=json.loads(capsys.readouterr().out)
    assert message['qualified'] is False and message['solver_executed'] is False
    assert json.loads((out/'phase-candidate.json').read_text())['preserve_frequency_limits'] is True
