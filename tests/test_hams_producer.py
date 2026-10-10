# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from dataclasses import replace
import hashlib,sys,os
import pytest
from anyhydro.hams_producer import HamsProducerBuild,HamsProducerError,SOURCE_IDENTITIES
from anyhydro.errors import CapabilityError

def build():
    exe=Path(sys.executable);return HamsProducerBuild(exe,hashlib.sha256(exe.read_bytes()).hexdigest(),next(iter(SOURCE_IDENTITIES)),
        'windows' if os.name=='nt' else 'linux','synthetic Python lifecycle fixture; never HAMS')

def test_build_identity_relocation_and_no_execution():
    b=build();b.verify(require_native=True)
    assert HamsProducerBuild.from_identity(b.identity(),executable_path=b.executable_path).content_hash()==b.content_hash()
    assert b.identity()['qualified'] is False and 'executable_path' not in b.identity()

@pytest.mark.parametrize('field,value',[('source_tree_sha256','a'*64),('source_commit','unknown'),('executable_sha256',''),('platform','wsl'),('build_description','')])
def test_unknown_or_incomplete_build_refuses(field,value):
    with pytest.raises(HamsProducerError):replace(build(),**{field:value})

def test_missing_changed_and_wrong_native_platform(tmp_path):
    b=build();fake=tmp_path/'wrong';fake.write_bytes(b'changed')
    with pytest.raises(HamsProducerError):replace(b,executable_path=fake).verify()
    with pytest.raises(HamsProducerError):replace(b,executable_path=tmp_path/'missing').verify()
    wrong='linux' if os.name=='nt' else 'windows'
    with pytest.raises((CapabilityError,HamsProducerError)):replace(b,platform=wrong).verify(require_native=True)
