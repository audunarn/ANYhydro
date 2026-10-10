# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
import os,sys,time,json,subprocess
import pytest
from anyhydro._owned_process import OwnedProcess

def wait_for(predicate,seconds=3.):
    end=time.perf_counter()+seconds
    while time.perf_counter()<end:
        if predicate():return
        time.sleep(.02)
    raise AssertionError('bounded fixture condition did not occur')

def test_native_stdout_stderr_environment_and_normal_exit(tmp_path):
    env=os.environ.copy();env['OWNED_FIXTURE']='effective'
    with (tmp_path/'out').open('wb') as out,(tmp_path/'err').open('wb') as err:
        p=OwnedProcess([sys.executable,'-c','import os,sys;print(os.environ["OWNED_FIXTURE"]);print("diagnostic",file=sys.stderr)'],cwd=tmp_path,env=env,stdout=out,stderr=err)
        wait_for(lambda:p.poll() is not None);assert p.poll()==0;p.close();p.close()
    assert b'effective' in (tmp_path/'out').read_bytes() and b'diagnostic' in (tmp_path/'err').read_bytes()

def test_owned_grandchild_stops_with_tree_cleanup(tmp_path):
    child='import pathlib,time; p=pathlib.Path("heartbeat"); n=0\nwhile True:\n p.write_text(str(n));n+=1;time.sleep(.01)'
    parent=f'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",{child!r}]);print("started",flush=True);time.sleep(60)'
    with (tmp_path/'out').open('wb') as out,(tmp_path/'err').open('wb') as err:
        p=OwnedProcess([sys.executable,'-c',parent],cwd=tmp_path,env=os.environ.copy(),stdout=out,stderr=err)
        try:
            wait_for(lambda:(tmp_path/'heartbeat').exists());p.close()
            value=(tmp_path/'heartbeat').read_bytes();time.sleep(.15);assert (tmp_path/'heartbeat').read_bytes()==value
            assert p.poll() is not None
        finally:p.close()

@pytest.mark.skipif(os.name!='nt',reason='Windows atomic job close protects supervisor death; Linux Qt parent-death route unadmitted')
def test_windows_supervisor_death_closes_job_and_grandchild(tmp_path):
    # This owned helper creates the producer job. Killing only that helper must
    # close its non-inherited job handle and stop the native tree automatically.
    module=Path(__import__('anyhydro._owned_process',fromlist=['x']).__file__)
    child='import pathlib,time\np=pathlib.Path("heartbeat");n=0\nwhile True:\n p.write_text(str(n));n+=1;time.sleep(.01)'
    parent=f'import subprocess,sys,time;subprocess.Popen([sys.executable,"-c",{child!r}]);time.sleep(60)'
    helper=tmp_path/'helper.py'
    helper.write_text(f'import importlib.util,pathlib,os,sys,time\ns=importlib.util.spec_from_file_location("owned",{str(module)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\no=open("out","wb");e=open("err","wb")\np=m.OwnedProcess([sys.executable,"-c",{parent!r}],cwd=pathlib.Path.cwd(),env=os.environ.copy(),stdout=o,stderr=e)\ntime.sleep(60)\n')
    proc=subprocess.Popen([sys.executable,str(helper)],cwd=tmp_path,creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        wait_for(lambda:(tmp_path/'heartbeat').exists());proc.kill();proc.wait(timeout=2)
        time.sleep(.15);value=(tmp_path/'heartbeat').read_bytes();time.sleep(.15);assert (tmp_path/'heartbeat').read_bytes()==value
    finally:
        if proc.poll() is None:proc.kill();proc.wait(timeout=2)
