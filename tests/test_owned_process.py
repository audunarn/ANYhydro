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

def test_close_records_parent_return_status_before_returning(tmp_path):
    # close() must reap the owned parent and record its return status: a
    # stable integer afterwards, never None, on every supported platform.
    parent='import time;time.sleep(60)'
    with (tmp_path/'out').open('wb') as out,(tmp_path/'err').open('wb') as err:
        p=OwnedProcess([sys.executable,'-c',parent],cwd=tmp_path,env=os.environ.copy(),stdout=out,stderr=err)
        try:
            p.close()
            code=p.poll()
            assert isinstance(code,int)
            assert code!=0  # early cancellation is terminal non-success
            assert p.poll()==code  # recorded status is stable across further polls
        finally:p.close()

def test_close_after_normal_exit_keeps_producer_code(tmp_path):
    # A producer that already exited keeps its exact exit status through close.
    with (tmp_path/'out').open('wb') as out,(tmp_path/'err').open('wb') as err:
        p=OwnedProcess([sys.executable,'-c','print("done",flush=True)'],cwd=tmp_path,env=os.environ.copy(),stdout=out,stderr=err)
        try:
            wait_for(lambda:p.poll() is not None)
            assert p.poll()==0
            p.close()
            assert p.poll()==0
        finally:p.close()

def test_close_preserves_reported_nonzero_producer_exit(tmp_path):
    # A verified nonzero producer completion packet stays exact through close.
    with (tmp_path/'out').open('wb') as out,(tmp_path/'err').open('wb') as err:
        p=OwnedProcess([sys.executable,'-c','import sys;print("boom",flush=True);sys.exit(7)'],cwd=tmp_path,env=os.environ.copy(),stdout=out,stderr=err)
        try:
            wait_for(lambda:p.poll() is not None)
            assert p.poll()==7
            p.close()
            assert p.poll()==7
        finally:p.close()

linux_only=pytest.mark.skipif(not sys.platform.startswith('linux'),
    reason='focused Linux owned close/status refusal contract')

# Pinned lifecycle stub, not the production guard: it completes the startup
# handshake and starts a real producer, then exits 0 without ever delivering
# a valid producer completion packet (OWNED_STUB_MODE=invalid_packet sends a
# schema-invalid packet instead). It models a guard whose own success is not
# producer success; the false-success/refusal contract is what is under test.
STUB_GUARD=r'''# Pinned lifecycle stub guard for the false-success refusal contract.
import json,os,socket,subprocess,sys

def _send(channel,value):
    channel.sendall(json.dumps(value,separators=(',',':')).encode()+b'\n')

def main(fd,parent_pid):
    channel=socket.socket(fileno=fd)
    os.set_inheritable(fd,False)
    if os.getpgrp()!=os.getpid():os._exit(126)
    _send(channel,dict(event='ready',guard_pid=os.getpid()))
    raw=bytearray()
    while b'\n' not in raw:
        part=channel.recv(4096)
        if not part:raise EOFError('lifeline closed before launch')
        raw.extend(part)
    command=json.loads(bytes(raw).split(b'\n',1)[0])['command']
    producer=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=sys.stdout,stderr=sys.stderr,close_fds=True)
    _send(channel,dict(event='started',producer_pid=producer.pid))
    if os.environ.get('OWNED_STUB_MODE')=='invalid_packet':
        _send(channel,dict(event='exit',code='not-an-int'))
    os._exit(0)

if __name__=='__main__':main(int(sys.argv[1]),int(sys.argv[2]))
'''

def _isolated_owned_process(tmp_path,guard_source):
    # Private copy of the production module pinned to the stub guard: the
    # guard path resolves relative to the module file, so this temp copy
    # supervises with the stub without touching production files, the
    # installed package or any other test.
    import importlib.util,shutil
    from anyhydro import _owned_process
    isolated=tmp_path/'isolated-owned';isolated.mkdir()
    shutil.copy(Path(_owned_process.__file__),isolated/'_owned_process.py')
    (isolated/'_linux_process_guard.py').write_text(guard_source)
    spec=importlib.util.spec_from_file_location('isolated_owned_process',isolated/'_owned_process.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def _proc_state(pid):
    # Non-reaping liveness probe; Popen.poll()/wait() here would steal the
    # owned child and turn a close into an ownership refusal.
    try:
        stat=Path(f'/proc/{int(pid)}/stat').read_text()
        return stat[stat.rindex(')')+2:].split(' ',1)[0]
    except (FileNotFoundError,ProcessLookupError,ValueError):return 'gone'

@linux_only
def test_close_guard_exit0_without_producer_packet_is_terminal_nonsuccess(tmp_path):
    # False-success refusal contract: a guard that exits 0 while hiding the
    # producer completion packet must not make close/poll report 0 or None.
    module=_isolated_owned_process(tmp_path,STUB_GUARD)
    env=os.environ.copy();env['OWNED_STUB_MODE']='missing_packet'
    with (tmp_path/'out').open('wb') as out,(tmp_path/'err').open('wb') as err:
        p=module.OwnedProcess([sys.executable,'-c','import time;time.sleep(60)'],cwd=tmp_path,env=env,stdout=out,stderr=err)
        try:
            wait_for(lambda:_proc_state(p.supervision['guard_pid'])=='Z')
            p.close()
            code=p.poll()
            assert isinstance(code,int) and code!=0
            assert p.poll()==code  # terminal: stable across further polls
            p.close()  # idempotent
            assert p.poll()==code
            wait_for(lambda:_proc_state(p.pid)=='gone',seconds=5)  # owned tree cleaned
        finally:p.close()

@linux_only
def test_close_invalid_completion_packet_refuses_and_cleans_up(tmp_path):
    # A schema-invalid completion packet is a protocol refusal: cleanup still
    # bounds the owned tree, the error is preserved and no success is reported.
    module=_isolated_owned_process(tmp_path,STUB_GUARD)
    env=os.environ.copy();env['OWNED_STUB_MODE']='invalid_packet'
    with (tmp_path/'out').open('wb') as out,(tmp_path/'err').open('wb') as err:
        p=module.OwnedProcess([sys.executable,'-c','import time;time.sleep(60)'],cwd=tmp_path,env=env,stdout=out,stderr=err)
        try:
            wait_for(lambda:_proc_state(p.supervision['guard_pid'])=='Z')
            with pytest.raises(OSError):p.close()
            code=p.poll()
            assert isinstance(code,int) and code!=0
            assert p.poll()==code
            wait_for(lambda:_proc_state(p.pid)=='gone',seconds=5)
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
