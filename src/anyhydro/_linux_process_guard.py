# SPDX-License-Identifier: MPL-2.0
"""Isolated stdlib Linux session guard. Not an arbitrary public launcher.

The parent starts this script with -I, an inherited private socket and an exact
parent PID. No producer starts until the parent authorizes the bounded command.
All producer descendants must retain this trusted executable's session/group.
"""
import ctypes
import json
import os
import signal
import socket
import subprocess
import sys
import threading

MAX_PACKET = 32768


def _kill_session(*_):
    # The guard itself keeps this process-group identity alive until cleanup.
    # Do not search for processes or signal the caller's/unrelated groups.
    pid = os.getpid()
    if os.getpgrp() != pid:
        os._exit(126)
    os.killpg(pid, signal.SIGKILL)
    os._exit(126)


def _send(channel, value):
    raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode() + b'\n'
    if len(raw) > MAX_PACKET:
        raise ValueError('Oversize guard packet')
    channel.sendall(raw)


def _command(channel):
    raw = bytearray()
    while b'\n' not in raw:
        part = channel.recv(4096)
        if not part:
            raise EOFError('Parent lifeline closed before launch')
        raw.extend(part)
        if len(raw) > MAX_PACKET:
            raise ValueError('Oversize guard command')
    packet, remainder = bytes(raw).split(b'\n', 1)
    if remainder:
        raise ValueError('Only one parent launch command is admitted')
    value = json.loads(packet)
    if type(value) is not dict or set(value) != {'command'}:
        raise ValueError('Guard command schema differs')
    command = value['command']
    if (type(command) is not list or not command or
        any(type(v) is not str or not v or '\x00' in v for v in command) or
        not os.path.isabs(command[0])):
        raise ValueError('Explicit absolute executable and bounded string arguments required')
    return command


def main(fd, parent_pid):
    channel = socket.socket(fileno=fd)
    os.set_inheritable(fd, False)
    try:
        if os.getpgrp() != os.getpid():
            raise RuntimeError('Guard must lead a new owned session')
        # prctl is executed here, in the new interpreter before it has threads,
        # never in a preexec_fn of the possibly threaded application.
        signal.signal(signal.SIGTERM, _kill_session)
        libc = ctypes.CDLL(None, use_errno=True)
        prctl = libc.prctl
        prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
                          ctypes.c_ulong, ctypes.c_ulong]
        prctl.restype = ctypes.c_int
        if prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
            raise OSError(ctypes.get_errno(), 'Cannot bind Linux parent-death signal')
        if os.getppid() != parent_pid:
            raise EOFError('Parent died before parent-death binding')
        _send(channel, dict(event='ready', guard_pid=os.getpid()))
        command = _command(channel)

        def lifeline():
            try:
                # EOF or an unexpected second command both revoke ownership.
                channel.recv(1)
            finally:
                _kill_session()

        watcher = threading.Thread(target=lifeline, daemon=True)
        watcher.start()
        # Inherited session/group ownership exists before the native child's
        # first instruction. The private socket is not inherited by the child.
        producer = subprocess.Popen(command, stdin=subprocess.DEVNULL,
                                    stdout=sys.stdout, stderr=sys.stderr,
                                    close_fds=True)
        _send(channel, dict(event='started', producer_pid=producer.pid))
        code = producer.wait()
        _send(channel, dict(event='exit', code=code))
    except BaseException as error:
        try:
            _send(channel, dict(event='error', message=f'{type(error).__name__}: {error}'))
        except (OSError, ValueError):
            pass
    finally:
        # Includes normal producer exit: no leftover same-group descendants
        # may remain, even if the supervising application forgot explicit close.
        _kill_session()


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
