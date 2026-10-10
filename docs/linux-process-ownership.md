# Linux native process ownership (development)

`OwnedProcess` launches an isolated stdlib guard in a new session. Native HAMS
inherits that group before its first instruction. A private noninherited socket
authorizes one bounded absolute command and supplies a lifeline. The guard binds
Linux parent-death SIGTERM before creating threads; neither the GUI nor worker
uses a thread-unsafe `preexec_fn`. Producer stdout/stderr and the explicit thread
environment remain unchanged. The receipt records guard command/source hash,
guard/native PIDs and the cleanup policy as optional Linux supervision metadata.

The supervising process retains the guard as its unreaped direct child. Valid
native exit packets do not prove cleanup: before reporting that exit code, the
parent kills the pinned group and waits for its guard. Abnormal guard death is
observed using pidfd `waitid(...WNOWAIT)` and likewise cleaned before error.
Explicit close kills the pinned group, including a stopped guard. Cleanup
timeouts retain the recovery handle and do not establish completion. Never use
`Popen.poll()` or an external reaper on this guard before group cleanup: releasing
its child PID would destroy the identity pin. Completed objects never signal
that numeric group again. Windows Job Object source remains unchanged.

Requirements: Linux kernel5.4+, Python pidfd/waitid APIs, default SIGCHLD, and
exclusive child reaping by this owner. Trusted native executables/descendants
must remain in the inherited process group. This is process ownership, not an
OS sandbox. The parent-death signal refers to the creating thread, so handing
off from a thread that exits conservatively cancels its job. A guard separately
stopped or killed while the supervisor also dies cannot enforce cleanup; this
double failure and escaped sessions are unsupported. Use a kernel-backed cgroup
scope if that stronger fault model is required. No complete-cleanup guarantee
for arbitrary hostile processes or system shutdown is claimed.

The Linux regression fixture uses real Ubuntu24.04/WSL2 Python3.12.3 and kernel
6.6.87.2 process operations. It directly loads the module extracted from the
built wheel and uses private Python producers, not native HAMS calculations.
This verifies the module's Linux path; it does not establish a fully installed
Linux ANYhydro/Qt stack, a native desktop, actual HAMS output accuracy, or Linux
ANYmarine job admission. The application's Windows-only native admission remains.

Kernel references: [parent-death signal](https://man7.org/linux/man-pages/man2/PR_SET_PDEATHSIG.2const.html),
[waitid/WNOWAIT](https://man7.org/linux/man-pages/man2/waitid.2.html),
[process-group signals](https://man7.org/linux/man-pages/man2/kill.2.html).
