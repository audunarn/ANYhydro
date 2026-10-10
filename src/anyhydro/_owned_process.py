# SPDX-License-Identifier: MPL-2.0
"""Owned native process trees; stdlib only, no shell or global process searches."""
import os,signal,subprocess,sys


def native_supervision_status():
    """No-launch runtime probe; version numbers alone do not prove ownership."""
    import hashlib
    from pathlib import Path
    platform='windows' if os.name=='nt' else 'linux' if sys.platform.startswith('linux') else sys.platform
    contract='windows-atomic-job-list-v1' if platform=='windows' else 'linux-owned-session-guard-v1' if platform=='linux' else None
    result=dict(platform=platform,contract=contract,available=False,reason=None,guard_sha256=None)
    if platform=='windows':
        result['available']=sys.getwindowsversion().major>=10
        if not result['available']:result['reason']='Atomic JobList native supervision requires Windows10+'
        return result
    if platform!='linux':
        result['reason']='Native supervision supports Windows10+ and Linux only'
        return result
    if not all(hasattr(os,name) for name in ('pidfd_open','waitid','P_PIDFD')):
        result['reason']='Linux native supervision requires pidfd wait APIs and kernel5.4+'
        return result
    if signal.getsignal(signal.SIGCHLD)!=signal.SIG_DFL:
        result['reason']='Linux native supervision requires default SIGCHLD and exclusive child reaping'
        return result
    fd=None
    try:
        fd=os.pidfd_open(os.getpid())
        try:os.waitid(os.P_PIDFD,fd,os.WEXITED|os.WNOHANG|os.WNOWAIT)
        except ChildProcessError:pass  # Self is not a child; syscall is supported.
        else:raise OSError('Unexpected Linux pidfd/waitid self-probe result')
        guard=Path(__file__).with_name('_linux_process_guard.py')
        result['guard_sha256']=hashlib.sha256(guard.read_bytes()).hexdigest()
        result['available']=True
    except OSError as error:result['reason']='Linux native supervision unavailable: '+str(error)
    finally:
        if fd is not None:os.close(fd)
    return result


class OwnedProcess:
    def __new__(cls,command,*,cwd,env,stdout,stderr):
        if os.name=='nt':return _WindowsProcess(command,cwd,env,stdout,stderr)
        if sys.platform.startswith('linux'):return _LinuxProcess(command,cwd,env,stdout,stderr)
        raise OSError('Native process supervision supports Windows10+ and Linux only')


class _LinuxProcess:
    def __init__(self,command,cwd,env,stdout,stderr):
        import hashlib,json,socket
        from pathlib import Path
        if not hasattr(os,'pidfd_open') or not hasattr(os,'P_PIDFD') or not hasattr(os,'waitid'):
            raise OSError('Linux native supervision requires pidfd wait APIs and kernel5.4+')
        if signal.getsignal(signal.SIGCHLD)!=signal.SIG_DFL:
            raise OSError('Linux native supervision requires default SIGCHLD and exclusive child reaping')
        self._helper=None;self._pidfd=None;self._lifeline=None
        self._buffer=b'';self._reported=None;self._code=None;self._closed=False;self._reaped=False;self._ownership_lost=False
        parent,child=socket.socketpair();self._lifeline=parent
        guard=Path(__file__).with_name('_linux_process_guard.py')
        try:
            guard_sha=hashlib.sha256(guard.read_bytes()).hexdigest()
            helper_command=[sys.executable,'-I',str(guard),str(child.fileno()),str(os.getpid())]
            self._helper=subprocess.Popen(helper_command,cwd=cwd,env=env,stdout=stdout,stderr=stderr,
                stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True,pass_fds=(child.fileno(),))
            child.close()
            self._pidfd=os.pidfd_open(self._helper.pid)
            parent.settimeout(2.)
            ready=self._packet()
            if ready!={'event':'ready','guard_pid':self._helper.pid}:
                raise OSError('Linux guard startup handshake differs')
            raw=json.dumps(dict(command=[str(v) for v in command]),allow_nan=False,separators=(',',':')).encode()+b'\n'
            if len(raw)>32768:raise OSError('Oversize Linux native command')
            parent.sendall(raw)
            started=self._packet()
            if (type(started) is not dict or set(started)!={'event','producer_pid'} or
                started['event']!='started' or type(started['producer_pid']) is not int or started['producer_pid']<=0):
                raise OSError('Linux producer did not acknowledge owned startup: '+str(started))
            self.pid=started['producer_pid']
            parent.setblocking(False)
            self.supervision=dict(format='linux-owned-session-guard-v1',guard_command=helper_command,
                guard_sha256=guard_sha,guard_pid=self._helper.pid,producer_pid=self.pid,
                parent_death_signal='SIGTERM',lifeline='private_noninherited_socket',
                close_control='unreaped_child_pinned_group_SIGKILL',native_group=self._helper.pid)
        except BaseException as error:
            try:self.close()
            except BaseException as cleanup:
                if hasattr(error,'add_note'):error.add_note('Linux guard cleanup: '+str(cleanup))
            raise
        finally:child.close()
    def _packet(self):
        import json
        while b'\n' not in self._buffer:
            chunk=self._lifeline.recv(4096)
            if not chunk:raise EOFError('Linux guard lifeline ended without a complete packet')
            self._buffer+=chunk
            if len(self._buffer)>32768:raise OSError('Oversize Linux guard response')
        line,self._buffer=self._buffer.split(b'\n',1)
        return json.loads(line)
    def _drain(self):
        while True:
            try:packet=self._packet()
            except BlockingIOError:break
            except EOFError:break
            if (type(packet) is not dict or set(packet)!={'event','code'} or packet['event']!='exit' or
                type(packet['code']) is not int or self._reported is not None):
                raise OSError('Invalid Linux native completion packet: '+str(packet))
            self._reported=packet['code']
    def poll(self):
        if self._closed:return self._code
        self._drain()
        # Never Popen.poll()/wait() before group cleanup: the unreaped direct
        # child pins this exact PID even when the guard has already died.
        if self._reported is not None:
            self._finish_group()
            self._code=self._reported
            return self._code
        try:ended=os.waitid(os.P_PIDFD,self._pidfd,os.WEXITED|os.WNOHANG|os.WNOWAIT)
        except ChildProcessError as error:
            self._ownership_lost=True
            raise OSError('Linux guard child ownership lost; group signal refused') from error
        if ended is None:return None
        # A packet can arrive between EAGAIN above and the termination check.
        self._drain()
        self._finish_group()
        if self._reported is None:
            raise OSError('Linux guard ended without owned-session completion')
        self._code=self._reported
        return self._code
    def _finish_group(self):
        if self._reaped or self._helper is None:return
        if self._ownership_lost or self._helper.returncode is not None:
            self._ownership_lost=True
            raise OSError('Linux guard child ownership lost; group signal refused')
        # Fail closed if SIGCHLD handling or an external reaper stole our child.
        # The host contract excludes concurrent external reaping; WNOWAIT never
        # releases the anchor, including when it has already become a zombie.
        try:
            os.waitid(os.P_PIDFD if self._pidfd is not None else os.P_PID,
                self._pidfd if self._pidfd is not None else self._helper.pid,
                os.WEXITED|os.WNOHANG|os.WNOWAIT)
        except ChildProcessError as error:
            self._ownership_lost=True
            raise OSError('Linux guard child ownership lost; group signal refused') from error
        # This PID cannot be recycled until we reap our own child. The producer
        # never leads a separate group; no global process lookup is involved.
        try:os.killpg(self._helper.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        self._helper.wait(timeout=2.)
        self._reaped=True
    def close(self):
        if self._closed:return
        refusal=None
        if self._lifeline is not None:
            # Capture a pending owned completion packet before revoking the
            # lifeline; a producer that already ended keeps its exact status.
            # A drain/protocol failure must not become successful completion:
            # bounded cleanup still runs below, then the error is refused.
            try:self._drain()
            except (OSError,ValueError) as error:refusal=error
            import socket
            try:self._lifeline.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            self._lifeline.close();self._lifeline=None
        self._finish_group()
        if refusal is not None:
            # A refused drain is terminal non-success even when a valid
            # completion (including zero) was consumed first: a malformed
            # or duplicate packet must never leave a success status behind.
            self._code=125
        elif self._code is None:
            # close() reaps the owned parent above; record its return status
            # before returning so a later poll() reports the effective owned
            # result instead of None. Bounded: no additional waiting happens
            # here. Guard exit is not producer exit: without a verified owned
            # completion packet the status is terminal non-success, never
            # the helper's own zero/unknown returncode.
            helper_code=self._helper.returncode if self._helper is not None else None
            self._code=(self._reported if self._reported is not None
                else helper_code if isinstance(helper_code,int) and helper_code!=0
                else 125)
        self._closed=True
        if self._pidfd is not None:os.close(self._pidfd);self._pidfd=None
        if refusal is not None:raise refusal


class _WindowsProcess:
    def __init__(self,command,cwd,env,stdout,stderr):
        import ctypes as c
        from ctypes import wintypes as w
        import msvcrt
        k=c.WinDLL('kernel32',use_last_error=True);size=c.c_size_t;HANDLE=w.HANDLE;DWORD=w.DWORD
        class BASIC(c.Structure):
            _fields_=[('process_time',c.c_int64),('job_time',c.c_int64),('flags',DWORD),('min_ws',size),('max_ws',size),('active',DWORD),('affinity',size),('priority',DWORD),('scheduling',DWORD)]
        class IO(c.Structure):_fields_=[(x,c.c_uint64) for x in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class LIMITS(c.Structure):_fields_=[('basic',BASIC),('io',IO),('process_memory',size),('job_memory',size),('peak_process',size),('peak_job',size)]
        class START(c.Structure):
            _fields_=[('cb',DWORD),('reserved',w.LPWSTR),('desktop',w.LPWSTR),('title',w.LPWSTR),('x',DWORD),('y',DWORD),('width',DWORD),('height',DWORD),('xchars',DWORD),('ychars',DWORD),('fill',DWORD),('flags',DWORD),('show',w.WORD),('reserved_size',w.WORD),('reserved_bytes',c.POINTER(c.c_byte)),('stdin',HANDLE),('stdout',HANDLE),('stderr',HANDLE)]
        class STARTEX(c.Structure):_fields_=[('start',START),('attributes',c.c_void_p)]
        class INFO(c.Structure):_fields_=[('process',HANDLE),('thread',HANDLE),('pid',DWORD),('tid',DWORD)]
        def api(name,argtypes,restype):
            f=getattr(k,name);f.argtypes=argtypes;f.restype=restype;return f
        create_job=api('CreateJobObjectW',[c.c_void_p,w.LPCWSTR],HANDLE)
        set_info=api('SetInformationJobObject',[HANDLE,c.c_int,c.c_void_p,DWORD],w.BOOL)
        init=api('InitializeProcThreadAttributeList',[c.c_void_p,DWORD,DWORD,c.POINTER(size)],w.BOOL)
        update=api('UpdateProcThreadAttribute',[c.c_void_p,DWORD,size,c.c_void_p,size,c.c_void_p,c.c_void_p],w.BOOL)
        delete=api('DeleteProcThreadAttributeList',[c.c_void_p],None)
        create=api('CreateProcessW',[w.LPCWSTR,w.LPWSTR,c.c_void_p,c.c_void_p,w.BOOL,DWORD,c.c_void_p,w.LPCWSTR,c.POINTER(START),c.POINTER(INFO)],w.BOOL)
        self._close=api('CloseHandle',[HANDLE],w.BOOL)
        self._terminate=api('TerminateJobObject',[HANDLE,w.UINT],w.BOOL)
        self._wait=api('WaitForSingleObject',[HANDLE,DWORD],DWORD)
        self._exit=api('GetExitCodeProcess',[HANDLE,c.POINTER(DWORD)],w.BOOL)
        self._ctypes=c;self._w=w;self._handle=None;self._job=None;self._code=None
        def checked(value):
            if not value:raise c.WinError(c.get_last_error())
            return value
        info=INFO();attributes=None;attributes_live=False
        # The job handle is not inherited. Last-handle closure contains the whole
        # tree even when the supervising worker itself crashes or is terminated.
        self._job=checked(create_job(None,None));limits=LIMITS();limits.basic.flags=0x2000
        try:
            checked(set_info(self._job,9,c.byref(limits),c.sizeof(limits)))
            count=size();init(None,2,0,c.byref(count));attributes=c.create_string_buffer(count.value)
            checked(init(attributes,2,0,c.byref(count)));attributes_live=True
            jobs=(HANDLE*1)(self._job)
            checked(update(attributes,0,0x2000D,jobs,c.sizeof(jobs),None,None))
            with open(os.devnull,'rb') as stdin:
                handles=[msvcrt.get_osfhandle(stream.fileno()) for stream in (stdin,stdout,stderr)]
                whitelist=(HANDLE*3)(*handles)
                checked(update(attributes,0,0x20002,whitelist,c.sizeof(whitelist),None,None))
                startup=STARTEX();startup.start.cb=c.sizeof(startup);startup.start.flags=0x100
                startup.start.stdin,startup.start.stdout,startup.start.stderr=handles
                startup.attributes=c.cast(attributes,c.c_void_p)
                environment=c.create_unicode_buffer('\0'.join(f'{key}={value}' for key,value in sorted(env.items(),key=lambda x:x[0].upper()))+'\0\0')
                line=c.create_unicode_buffer(subprocess.list2cmdline([str(x) for x in command]))
                inheritance=[os.get_handle_inheritable(h) for h in handles]
                try:
                    for h in handles:os.set_handle_inheritable(h,True)
                    # Atomic job-list assignment happens before the first thread
                    # runs. No suspended-but-unowned interval or fallback exists.
                    checked(create(str(command[0]),line,None,None,True,0x80000|0x400|0x8000000,
                        environment,str(cwd),c.cast(c.byref(startup),c.POINTER(START)),c.byref(info)))
                    self._handle=info.process;self.pid=int(info.pid)
                finally:
                    for h,value in zip(handles,inheritance):os.set_handle_inheritable(h,value)
            self._close(info.thread)
        except BaseException:
            if info.thread:self._close(info.thread)
            self.close();raise
        finally:
            if attributes_live:delete(attributes)

    def poll(self):
        if self._handle is None:return self._code
        if self._wait(self._handle,0)==258:return None
        code=self._w.DWORD()
        if not self._exit(self._handle,self._ctypes.byref(code)):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        self._code=int(code.value);return self._code

    def close(self):
        try:
            if self._job:
                self._terminate(self._job,1)
                if self._handle:
                    if self._wait(self._handle,2000)==258:raise TimeoutError('Owned process did not terminate within cleanup bound')
                    self.poll()
        finally:
            if self._handle:self._close(self._handle);self._handle=None
            if self._job:self._close(self._job);self._job=None
