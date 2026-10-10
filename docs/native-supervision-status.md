# Native supervision status (development)

`HamsBackend.native_supervision_status()` is a public no-launch prerequisite
probe. It returns platform, contract, available (bool), reason and guard_sha256.
The result is fresh; imported packages and queries do not start a native job.
Linux consumers require `linux-owned-session-guard-v1` and literal available=True.
An older ANYhydro0.1.0 without this method must fail admission: the package version
alone does not identify the repaired ownership implementation.

Linux checks Python pidfd/waitid APIs, default SIGCHLD, a kernel P_PIDFD self-probe
(ECHILD is expected), and readable installed guard bytes. It closes the pidfd.
Windows reports the existing atomic JobList contract on Windows10+. Actual guard
prctl/session startup and native producer launch remain runtime obligations; the
probe does not prove exclusive child reaping, hostile containment, producer/input
readiness, scientific qualification or complete application/desktop support.
The existing guarded classes and Windows implementation are unchanged.

Qt, CLI snapshot, worker and saved-output consumer admission use the same service.
Refusals preserve typed non-success. See [Linux ownership](linux-process-ownership.md)
for creator-thread lifetime, group membership and guard/supervisor fault exclusions.
