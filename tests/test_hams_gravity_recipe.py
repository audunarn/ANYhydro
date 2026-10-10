# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
import importlib.util, json, hashlib, os
import pytest

ROOT=Path(__file__).resolve().parents[1]


def tool():
    spec=importlib.util.spec_from_file_location('gravity_recipe',ROOT/'tools/prepare_hams_phase_candidate.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def test_exact_gravity_patch_preserves_every_other_source_and_old_profiles(tmp_path):
    source=Path(os.environ['HAMS_LIMIT_PRISTINE_SOURCE']);m=tool()
    old=m.prepare_candidate(source,tmp_path/'old',gnu_compatible=True,preserve_frequency_limits=True)
    fixed=m.prepare_candidate(source,tmp_path/'fixed',gnu_compatible=True,preserve_frequency_limits=True,consistent_gravity=True)
    assert fixed['format']=='anyhydro-hams-gravity-source-candidate-v3'
    assert fixed['gravity']==9.80665 and fixed['qualified'] is False and fixed['solver_executed'] is False
    changes=[n for n in old['candidate_sha256'] if old['candidate_sha256'][n]!=fixed['candidate_sha256'][n]]
    assert changes==['src/WavDynSubs.f90']
    raw=(tmp_path/'fixed/src/WavDynSubs.f90').read_bytes()
    assert raw.count(b'USE Const_mod, ONLY: G')==1
    assert b'DATA G,PI/9.807' not in raw and b'DATA PI/3.141592653589793d0/' in raw
    for name,pin in fixed['source_sha256'].items():assert hashlib.sha256((source/name).read_bytes()).hexdigest()==pin
    assert old['format']=='anyhydro-hams-limit-source-candidate-v2' and 'consistent_gravity' not in old


@pytest.mark.parametrize('gnu,limits,gravity',[(False,False,True),(True,False,True),(False,True,True),(True,True,1),(True,True,None)])
def test_unproven_profile_or_nonboolean_refuses_before_writing(tmp_path,gnu,limits,gravity):
    m=tool();out=tmp_path/'refused'
    with pytest.raises(m.ProducerRecipeError):m.prepare_candidate(Path(os.environ['HAMS_LIMIT_PRISTINE_SOURCE']),out,gnu_compatible=gnu,preserve_frequency_limits=limits,consistent_gravity=gravity)
    assert not out.exists()
