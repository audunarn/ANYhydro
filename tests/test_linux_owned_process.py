# SPDX-License-Identifier: MPL-2.0
"""Stdlib process fixtures; no HAMS, BEM or mechanical calculations."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

MODULE = Path(os.environ.get('ANYHYDRO_PROCESS_MODULE',Path(__file__).parents[1]/'src/anyhydro/_owned_process.py')).resolve()
EVIDENCE = Path(os.environ.get('OWNED_PROCESS_EVIDENCE',Path(tempfile.gettempdir())/'anyhydro-owned-process-tests')).resolve()
LEDGER = EVIDENCE / 'launches.jsonl'

def append_record(kind, pid):
    fd = os.open(LEDGER, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try: os.write(fd, (json.dumps(dict(kind=kind, pid=pid))+'\n').encode())
    finally: os.close(fd)

def load_module():
    spec = importlib.util.spec_from_file_location('owned_process_fixture', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

NATIVE = r'''
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]);mode=sys.argv[2]
def record(kind):
 fd=os.open(os.environ['OWNED_PROCESS_LEDGER'],os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
 try:os.write(fd,(json.dumps(dict(kind=kind,pid=os.getpid()))+'\n').encode())
 finally:os.close(fd)
if mode!='heartbeat':record('producer')
(root/'producer.json').write_text(json.dumps(dict(pid=os.getpid(),group=os.getpgrp(),token=os.environ['FIXTURE_TOKEN'])))
print('native stdout',flush=True);print('native stderr',file=sys.stderr,flush=True)
if mode.startswith('tree'):
 child=subprocess.Popen([sys.executable,'-I',__file__,str(root),'heartbeat'],close_fds=True)
 deadline=time.monotonic()+3
 while not (root/'heartbeat.pid').exists():
  if time.monotonic()>deadline:raise RuntimeError('heartbeat startup timeout')
  time.sleep(.01)
if mode=='heartbeat':
 record('heartbeat')
 (root/'heartbeat.pid').write_text(str(os.getpid()))
 while True:
  with (root/'heartbeat.log').open('ab') as out:out.write(b'x')
  time.sleep(.02)
if mode=='tree-exit':sys.exit(0)
if mode in ('sleep','tree'):
 while True:time.sleep(.1)
sys.exit(int(mode))
'''

SUPERVISOR = r'''
import importlib.util,json,os,subprocess,sys,time
from pathlib import Path
module_path,root_path,mode=sys.argv[1:];root=Path(root_path)
spec=importlib.util.spec_from_file_location('owned_fixture',module_path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
real=subprocess.Popen
def launch(*a,**kw):
 p=real(*a,**kw)
 if any(str(v).endswith('_linux_process_guard.py') for v in a[0]):
  fd=os.open(os.environ['OWNED_PROCESS_LEDGER'],os.O_CREAT|os.O_APPEND|os.O_WRONLY,0o600)
  try:os.write(fd,(json.dumps(dict(kind='guard',pid=p.pid))+'\n').encode())
  finally:os.close(fd)
 return p
subprocess.Popen=launch
if mode=='before-command':
 original=m._LinuxProcess._packet
 def gate(self):
  value=original(self)
  (root/'guard.pid').write_text(str(self._helper.pid))
  os._exit(18)
 m._LinuxProcess._packet=gate
with (root/'stdout').open('wb') as out,(root/'stderr').open('wb') as err:
 p=m.OwnedProcess([sys.executable,'-I',str(root/'native.py'),str(root),'tree'],cwd=root,env=os.environ.copy(),stdout=out,stderr=err)
 (root/'guard.pid').write_text(str(p.supervision['guard_pid']))
 deadline=time.monotonic()+3
 while not (root/'heartbeat.pid').exists():
  if time.monotonic()>deadline:raise RuntimeError('supervisor heartbeat timeout')
  time.sleep(.01)
 (root/'ready').write_text(str(p.pid))
 if mode=='exit':os._exit(19)
 while True:time.sleep(.1)
'''

def wait_for(predicate, seconds=3):
    end = time.monotonic()+seconds
    while time.monotonic()<end:
        if predicate(): return
        time.sleep(.01)
    raise AssertionError('bounded process fixture wait expired')

def stopped(pid):
    try: state = Path('/proc', str(pid), 'stat').read_text().rsplit(')',1)[1].split()[0]
    except (FileNotFoundError,ProcessLookupError): return True
    return state in ('Z','X')

@unittest.skipUnless(sys.platform.startswith('linux'), 'Linux process ownership')
class LinuxOwnedProcessTests(unittest.TestCase):
    def setUp(self):
        EVIDENCE.mkdir(parents=True,exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=EVIDENCE)
        self.root = Path(self.tmp.name)
        (self.root/'native.py').write_text(NATIVE)
        (self.root/'supervisor.py').write_text(SUPERVISOR)
        self.module = load_module()
        self.processes = []
        self.streams = []
        self.env = os.environ.copy()
        self.env.update(FIXTURE_TOKEN='kept',OWNED_PROCESS_LEDGER=str(LEDGER))
        self.real_popen = subprocess.Popen
        def launch(*a,**kw):
            p = self.real_popen(*a,**kw)
            if any(str(v).endswith('_linux_process_guard.py') for v in a[0]): append_record('guard',p.pid)
            return p
        self.mock = patch.object(subprocess,'Popen',side_effect=launch)
        self.mock.start()

    def tearDown(self):
        try:
            for p in self.processes: p.close()
        finally:
            for stream in self.streams: stream.close()
            self.mock.stop()
            self.tmp.cleanup()

    def create(self,mode):
        out=(self.root/'stdout').open('wb');err=(self.root/'stderr').open('wb')
        self.streams.extend((out,err))
        p=self.module.OwnedProcess([sys.executable,'-I',str(self.root/'native.py'),str(self.root),mode],cwd=self.root,env=self.env,stdout=out,stderr=err)
        self.processes.append(p)
        return p

    def heartbeat(self):
        wait_for(lambda:(self.root/'heartbeat.pid').exists())
        return int((self.root/'heartbeat.pid').read_text())

    def assert_clean(self,pid):
        wait_for(lambda:stopped(pid))
        before=(self.root/'heartbeat.log').read_bytes() if (self.root/'heartbeat.log').exists() else b''
        time.sleep(.06)
        self.assertEqual(before,(self.root/'heartbeat.log').read_bytes() if (self.root/'heartbeat.log').exists() else b'')

    def test_success_identity_environment_and_logs(self):
        p=self.create('0')
        wait_for(lambda:p.poll() is not None)
        self.assertEqual(p.poll(),0)
        data=json.loads((self.root/'producer.json').read_text())
        self.assertEqual(data,dict(pid=p.pid,group=p.supervision['guard_pid'],token='kept'))
        self.assertIn(b'native stdout',(self.root/'stdout').read_bytes())
        self.assertIn(b'native stderr',(self.root/'stderr').read_bytes())
        self.assertTrue(stopped(p.supervision['guard_pid']))

    def test_nonzero_native_exit(self):
        p=self.create('7');wait_for(lambda:p.poll() is not None);self.assertEqual(p.poll(),7)

    def test_close_removes_descendant(self):
        p=self.create('tree');pid=self.heartbeat();p.close();p.close();self.assert_clean(pid)

    def test_normal_exit_removes_descendant_before_completion(self):
        p=self.create('tree-exit');pid=self.heartbeat();wait_for(lambda:p.poll() is not None)
        self.assertEqual(p.poll(),0);self.assert_clean(pid)

    def supervisor(self,mode):
        controller=self.real_popen([sys.executable,'-I',str(self.root/'supervisor.py'),str(MODULE),str(self.root),mode],env=self.env,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            if mode=='before-command':
                controller.wait(4);wait_for(lambda:(self.root/'guard.pid').exists())
                wait_for(lambda:stopped(int((self.root/'guard.pid').read_text())))
                self.assertFalse((self.root/'producer.json').exists());return
            wait_for(lambda:(self.root/'ready').exists());pid=self.heartbeat()
            if mode=='kill':controller.kill()
            controller.wait(4);self.assert_clean(pid)
            wait_for(lambda:stopped(int((self.root/'guard.pid').read_text())))
        finally:
            if controller.poll() is None:controller.kill();controller.wait(2)

    def test_supervisor_os_exit(self):self.supervisor('exit')
    def test_supervisor_sigkill(self):self.supervisor('kill')
    def test_supervisor_dies_before_command_authorization(self):self.supervisor('before-command')

    def test_stopped_guard_cancellation(self):
        p=self.create('tree');pid=self.heartbeat()
        signal.pidfd_send_signal(p._pidfd,signal.SIGSTOP)
        p.close();self.assert_clean(pid)

    def test_exit_packet_then_external_guard_sigkill_cleans_descendant(self):
        # Pause the private guard exactly after the exit packet. An external
        # SIGKILL there must not be confused with the guard's own group cleanup.
        variant=self.root/'variant';variant.mkdir()
        owned=variant/'_owned_process.py';owned.write_bytes(MODULE.read_bytes())
        guard=MODULE.with_name('_linux_process_guard.py').read_text()
        guard=guard.replace("_send(channel, dict(event='exit', code=code))", "_send(channel, dict(event='exit', code=code))\n        open(os.environ['OWNED_EXIT_GATE'],'w').write('ready')\n        os.kill(os.getpid(),signal.SIGSTOP)")
        (variant/'_linux_process_guard.py').write_text(guard)
        spec=importlib.util.spec_from_file_location('fault_owned',owned)
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.env['OWNED_EXIT_GATE']=str(self.root/'exit-gate')
        p=self.create('tree-exit');pid=self.heartbeat()
        wait_for(lambda:(self.root/'exit-gate').exists())
        signal.pidfd_send_signal(p._pidfd,signal.SIGKILL)
        wait_for(lambda:p.poll() is not None)
        self.assertEqual(p.poll(),0);self.assert_clean(pid)

    def test_guard_sigkill_without_exit_packet_fails_and_cleans(self):
        p=self.create('sleep');wait_for(lambda:(self.root/'producer.json').exists())
        signal.pidfd_send_signal(p._pidfd,signal.SIGKILL)
        wait_for(lambda:stopped(p.supervision['guard_pid']))
        with self.assertRaises(OSError):p.poll()
        wait_for(lambda:stopped(p.pid))

    def test_cleanup_timeout_retains_retry_handle(self):
        p=self.create('sleep')
        with patch.object(p._helper,'wait',side_effect=subprocess.TimeoutExpired('private guard',2)):
            with self.assertRaises(subprocess.TimeoutExpired):p.close()
        self.assertIsNotNone(p._pidfd);self.assertFalse(p._reaped)
        p.close();self.assertTrue(p._reaped);self.assertIsNone(p._pidfd)

    def test_external_reaping_refuses_numeric_group_signal(self):
        p=self.create('0');p._helper.wait(timeout=3)
        try:
            with patch.object(os,'killpg') as kill_group:
                with self.assertRaisesRegex(OSError,'ownership lost'):p.close()
                kill_group.assert_not_called()
        finally:
            self.processes.remove(p)
            if p._pidfd is not None:os.close(p._pidfd);p._pidfd=None

    def test_close_refusal_after_valid_zero_packet_fails_closed(self):
        # A drain that consumes a valid zero completion and then hits a
        # malformed/duplicate packet must refuse: terminal status is 125,
        # never the consumed zero, while bounded cleanup still completes.
        p=self.create('0')
        packets=[{'event':'exit','code':0},{'event':'exit','code':0}]
        with patch.object(p,'_packet',side_effect=packets):
            with self.assertRaisesRegex(OSError,'Invalid Linux native completion packet'):p.close()
        self.assertIsNone(p._lifeline);self.assertTrue(p._reaped);self.assertIsNone(p._pidfd)
        self.assertEqual(p.poll(),125)
        p.close()
        self.assertEqual(p.poll(),125)

    def test_completion_packet_arriving_after_eagain_is_drained(self):
        p=self.create('0')
        wait_for(lambda:os.waitid(os.P_PIDFD,p._pidfd,os.WEXITED|os.WNOHANG|os.WNOWAIT) is not None)
        packet=p._packet;first=[True]
        def late_packet():
            if first[0]:first[0]=False;raise BlockingIOError()
            return packet()
        with patch.object(p,'_packet',side_effect=late_packet):self.assertEqual(p.poll(),0)

    def test_echild_failure_stays_lost_until_close(self):
        p=self.create('sleep')
        try:
            with patch.object(os,'waitid',side_effect=ChildProcessError()):
                with self.assertRaisesRegex(OSError,'ownership lost'):p.poll()
            with patch.object(os,'killpg') as kill_group:
                with self.assertRaisesRegex(OSError,'ownership lost'):p.close()
                kill_group.assert_not_called()
        finally:
            # Only the test injected ECHILD; the actual child remains owned.
            p._ownership_lost=False;p.close()

    def test_threaded_start_does_not_use_preexec(self):
        result=[]
        def start():
            try:result.append(self.create('sleep'))
            except BaseException as error:result.append(error)
        thread=threading.Thread(target=start);thread.start();thread.join(4)
        self.assertFalse(thread.is_alive());self.assertFalse(isinstance(result[0],BaseException),str(result[0]))
        # prctl parent-thread death cancels conservatively when this thread exits.
        result[0].close()

    def test_missing_executable_fails_and_cleans_guard(self):
        out=(self.root/'stdout').open('wb');err=(self.root/'stderr').open('wb');self.streams.extend((out,err))
        before=len(LEDGER.read_text().splitlines()) if LEDGER.exists() else 0
        with self.assertRaises(OSError):
            self.module.OwnedProcess([str(self.root/'absent')],cwd=self.root,env=self.env,stdout=out,stderr=err)
        rows=[json.loads(v) for v in LEDGER.read_text().splitlines()[before:]]
        for row in rows:
            if row['kind']=='guard':self.assertTrue(stopped(row['pid']))

    def test_pidfd_failure_refuses_before_native_launch(self):
        with patch.object(os,'pidfd_open',side_effect=OSError('kernel refuses pidfd')):
            with self.assertRaises(OSError):self.create('sleep')
        self.assertFalse((self.root/'producer.json').exists())

    def test_nondefault_sigchld_refuses_before_guard(self):
        with patch.object(signal,'getsignal',return_value=signal.SIG_IGN):
            with self.assertRaises(OSError):self.create('sleep')
        self.assertFalse((self.root/'producer.json').exists())

if __name__=='__main__':unittest.main()
