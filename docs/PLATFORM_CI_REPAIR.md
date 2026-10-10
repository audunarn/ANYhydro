# Platform CI repair record — 2026-10-10 (gui-ci-platform-repair-20261010)

Clean base CI run 38079385447 (ubuntu-latest, Python 3.13) failed with three
defect classes. This record documents the diagnosis, the repairs and the
verification evidence. Failed evidence is preserved verbatim in
`C:/ANYworkspaceWork/anymarine-evidence/gui-ci-repair-20261010/hydro-clean-ci-failed.log`.

## 1. Missing `threadpoolctl` on clean installs (118 errors + cascade)

`anyhydro/panel_integration_candidate.py` imports `threadpoolctl` at solve time
(`threadpool_limits`, `threadpool_info`) to pin native thread pools. The `dev`
extra only declared `pytest` and `build`, so every panel/micro test that
reached the candidate failed with `ModuleNotFoundError: No module named
'threadpoolctl'` (wrapped as `BackendSolveError`).

Repair: `pyproject.toml` dev extra now declares `threadpoolctl>=3.6`
(compatible with the current 3.6.0). No production source changed.

## 2. Three golden legacy hash regressions on Linux

Failing tests and observed deltas (Linux actual vs literal golden):

| Test | Linux hash | Golden hash |
| --- | --- | --- |
| `test_coefficient_frames.py::TestLegacyUnchanged::test_legacy_hashes_match_preedit_baseline` | `cd9a6930124a247650b94a42efca3b38d44d2ec9d22b7576131b553f5862bf8d` | `2db117e6b2b19b5bf9278cb21410d3aff2e266cc33f14bbc2646e05cd6fd52b9` |
| `test_execution_limits_persistence_hash.py::TestExecutionLimitsHashIdentity::test_dataset_without_execution_limits_hash_unchanged` | `cd9a6930...` (same fixture) | `2db117e6...` |
| `test_derived_operators.py::test_old_full12_dataset_bridge_hash_and_results` | `489d8e9e8abebedea18d40db556c072f138cc61147684467ce4ccce235f8cb2b` | `10bd41545ea153aa94cae508bcf1da3b7d0f3e2145553812de73060ccfc9091c` |

### Diagnosis (per-field differential, identical package versions)

A per-variable/coord/attr digest diff between the Windows runtime
(Python 3.13.9) and the local Linux venv (Ubuntu-24.04, Python 3.12.3), both
with numpy 2.5.3 / scipy 1.18.1 / xarray 2026.9.0 / threadpoolctl 3.6.0:

- Two-body fixture: exactly one field differs — `excitation_force`
  (5, 3, 12) complex128, Windows digest `72937e25a128d587...`, Linux digest
  `6fed8d1f86f128b5...`. Every other variable, coordinate and attribute
  digests identically. The analytic generator's complex excitation values
  differ across the observed numeric environments; the specific library cause
  was not isolated.
- Full-12 oracle dataset: byte-identical on both platforms
  (`dataset_content_hash = 8375e93f817e417c6b88760cb6b1b5a6beea9c8a7486df5bf4100bc39b39ae10`).
  The kernel hash nevertheless differs per environment: Windows
  `10bd4154...` (golden), CI Linux `489d8e9e...`, local Linux venv
  `325f11b4...` — three different hashes from identical dataset bytes, i.e.
  the kernel quadrature itself is environment-dependent.

Conclusion: these are generated-input float differences, not hashing or
serialization compatibility defects. The hasher binds exact bytes; identical
bytes hash identically everywhere (demonstrated below). Per the reserved
contract, the literal golden hashes cannot be promised for newly recomputed
analytic fixtures/kernel quadrature across numeric environments.

### Repair: portable legacy oracle (frozen artifact bytes)

Newly generated reference artifacts whose canonical content hashes equal the
existing legacy goldens are frozen under
`tests/fixtures/legacy-compatibility/` with `provenance.json` (golden hashes,
generation environment, evidence references, observed cross-environment
hashes). The original historical serialized file bytes were not recovered;
the frozen files are regenerated artifacts that reproduce the legacy golden
content hashes:

- `two_body_fixture_legacy.nc` — regenerated
  `save_hydro_dataset(make_two_body_fixture())` artifact, content hash
  `2db117e6...` equal to the legacy golden (verified at freeze time).
- `two_body_fixture_legacy_shifted.nc` — the regenerated shifted variant,
  content hash `70813430e6343da3d67dda4ca8dc7c8d3a900e409ffb3c643502e015236c6022`.
- `full12_kernel_legacy.nc` — regenerated deterministic kernel artifact
  (`save_radiation_kernel`, byte-stable re-save) with
  `content_hash = 10bd4154...` equal to the legacy golden (verified at
  freeze time).

The three golden tests now verify the literal golden hashes against the
frozen immutable bytes (portable), and keep meaningful same-input
compatibility and independent numerical checks:

- `test_coefficient_frames.py`: frozen plain/shifted datasets hash to the
  literal goldens; the live recomputed fixture and its shifted variant must
  match the frozen content field-by-field within `rtol=atol=1e-12`
  (independent numerical check of the generator, not a hash tolerance).
- `test_execution_limits_persistence_hash.py`: frozen fixture hashes to the
  literal golden and carries no `execution_limits` attribute.
- `test_derived_operators.py`: the frozen kernel hashes to the literal golden;
  its provenance `source_dataset_hash` must equal the live oracle dataset
  hash (same-input compatibility); both the frozen kernel and the live
  dataset-bridge kernel must reconstruct the analytic oracle within the
  existing tolerance-2.0 checks.

No expected hash was replaced with a Linux hash, no case was skipped, the
production hasher is untouched, and no data was quantized or rounded.

## 3. Linux owned-process close did not record the parent return status

`tests/test_owned_process.py::test_owned_grandchild_stops_with_tree_cleanup`
failed on Linux: after `p.close()`, `p.poll()` returned `None`.
`_LinuxProcess.close()` shut down the lifeline and reaped the guard via
`_finish_group()` but never recorded a return status into `self._code`, so
`poll()` after close returned `None`. (Windows `close()` already records via
`TerminateJobObject` + `poll()`.)

Repair (Linux close lifecycle only, `src/anyhydro/_owned_process.py`):
`close()` now drains any pending guard exit packet before revoking the
lifeline (a producer that already ended keeps its exact status), then reaps
via the existing bounded `_finish_group()` (2 s helper wait, group SIGKILL,
idempotence, ownership checks unchanged), then records the effective parent
return status — the producer's reported exit code, else the reaped guard's
nonzero returncode (or sentinel 125 if no producer result was verified) — before returning. A guard's zero exit is never substituted for producer success. No unbounded wait, no Windows job behavior
change, no broad refactor. Process ownership, descendant stop, bounded
cancellation and idempotent close are preserved; focused coverage added in
`tests/test_owned_process.py` (close-while-running records a stable integer
status; close after normal exit keeps the producer's exact code).

Preimage of the edited file before repair:
`sha256 src/anyhydro/_owned_process.py = 3a76c441d09bfc0b136c4a5d24e6cecbf46714c728c144b8e5ee4ce001743e35`
(the guard `src/anyhydro/_linux_process_guard.py` is untouched,
`sha256 = 79d6892e9c28424648fa254638cedc25ca9c2461211bb2d924001ec5fabb4669`).

## Verification receipts

All executions accounted through
`C:/ANYworkspaceWork/anymarine-evidence/gui-ci-platform-repair-20261010/run.py checks ...`:

- `focused-win-01`: pytest `tests/test_owned_process.py`,
  `test_coefficient_frames.py::TestLegacyUnchanged`,
  `test_execution_limits_persistence_hash.py`,
  `test_derived_operators.py::test_old_full12_dataset_bridge_hash_and_results`
  — 10 passed (Windows runtime, repo `src` first on path).
- `verify-linux-01`: plain-script Linux verification — frozen oracle hashes
  equal the literal goldens, `source_dataset_hash` matches the live oracle,
  live-vs-frozen field compatibility, kernel reconstruction within tolerance
  for both frozen and live kernels, owned-process close/reap semantics
  (grandchild stops, `poll()` after close is a stable integer, normal-exit
  code preserved, idempotent close) — all assertions passed.
- `contract-win-01`: `tests/test_portable_ci_contract.py` — 8 passed; dev
  extra parsed as `['pytest>=7', 'build>=1', 'threadpoolctl>=3.6']`.
- `gen-legacy-fixtures-01`: fixture freeze run (golden hashes asserted at
  freeze time).

Remaining uncertainty: GitHub's effective clean matrix (Python 3.11/3.13,
fresh dependency resolution) is the final authority and is observed by root
after publication. The frozen oracle removes the platform dependence of the
golden hash assertions; the threadpoolctl addition removes the import
cascade; the Linux reap repair addresses the observed poll-after-close
failure. No native numerical execution, no full 796-suite run, no production
numerical/hash/fixture source edits were made.

## Final root reconciliation before main publication

Linux close keeps a verified producer exit, uses terminal non-success when a
producer completion is absent, preserves protocol refusal after bounded tree
cleanup, and sets sentinel 125 even if a valid zero packet preceded a duplicate
or malformed packet. Repeated close and subsequent poll cannot promote the
refusal to success. Windows job implementation and Linux guard are unchanged.
The new valid-zero/duplicate regression actually ran on Linux via unittest:
1 test, OK. Earlier Linux plain-script lifecycle checks cover 7 cases; the
Windows owned-process selection also passed. No full local suite was rerun.

Final repair continuation completed with exit 0 but did not update its expected
handoff file. Root reconstructed its delivered edits and actual check receipts;
the earlier worker handoff remains historical, not final acceptance evidence.
The first final Linux launch failed because Git Bash converted its WSL path;
the corrected launch disabled that conversion. Failed evidence is retained.

- `owned-close-win-01`: finished, exit 0, process wall 3.409581600 s.
- `owned-close-linux-01`: finished, exit 0, process wall 5.323563000 s.
- `close-refusal-linux-01`: finished, exit 127, process wall 4.208386000 s.
- `close-refusal-linux-02`: finished, exit 0, process wall 0.230178100 s.
- `close-refusal-win-01`: finished, exit 0, process wall 0.082142300 s.
- `final-hydro-install-01`: finished, exit 0, process wall 3.539667600 s.

Installed hydro wheel SHA-256 `d75c9c4cf8e9fefae7b55a98e401fd73c3c0b93ca56bc9b4178c7e528b84410c` matches all
37 source package files (zero mismatches). Final process
module SHA-256 `3223d5acb86a74908a2f57afdacaeec4c0e46ef439a0fab7807ac5995e36055c`.

The historical protected set contains 101 files:
95 unchanged; six prospective changes are the Linux process
module, its two lifecycle test modules, and three legacy-oracle test modules.
All numerical/hash/fixture production algorithms and native campaign evidence
remain unchanged. The exact preimages remain in Git and local evidence;
`protected-source-reconciliation.json` records the before/after identities.
Claims of all 101 files unchanged in older records apply to their earlier time.

Frozen reference artifacts were newly generated in the recorded Windows
installed-package environment and reproduce existing legacy canonical goldens.
Original historical serialized files were not recovered. Production numerical
checks still run with the existing reconstruction criteria; no hash, physics,
sign or scientific tolerance was changed. No new native/BEM campaign occurred.

Earlier CI-check accounting gap/overrun remains recorded and frozen. This repair
uses a separate prospective pool: worker 674.413916500 s / 720 s; focused checks
31.104830100 s / 180 s; build/install 3.539667600 s / 120 s. Remaining time is not
carried into resumed scientific work. Clean GitHub CI is still pending at this
source-publication point; the final restart records the actual observed result.

## Final clean-CI tail repair

Run 38081165749 on source 8530b92 failed three assertions on Linux3.11:
816 passed, 3 failed, 1 skipped; other matrix legs were cancelled by fail-fast.
Full failed logs remain in gui-ci-platform-repair-20261010. The CI matrix now
uses fail-fast:false so each platform's result remains observable.

Mistral's first tail assignment was deliberately stopped at 266.846531000 s
after missing the early checkpoint and producing no candidate. The same-owner
focused continuation timed out at 93.473584700 s without edits or handoff.
Root completed the small test-infrastructure outcome directly because two
bounded attempts failed to deliver and continued worker inspection was blocking
publication. This is execution unreliability for this delivery, not a claim of
provider outage. Logs and partial evidence are preserved. Worker accounting is
360.320115700 s against360 s; the0.320115700 s timeout/termination overrun is
recorded without retrospective authority or renewal. No worker remains active.

The missing-completion test no longer assumes /proc zombie visibility and pidfd
waitability occur simultaneously. It retries only None within the existing3 s
fixture wait; a terminal integer fails, and OSError is required before checking
the producer stopped. Production supervision is unchanged by this tail repair.
The full19 Linux lifecycle tests passed, including ownership, refusal, cleanup,
cancellation and supervisor-death routes.

The two remaining long-kernel golden checks now use newly frozen canonical data
matching the unchanged historical golden8a40ee08c0568484bb78f666ab74d307bcdf87ecbc8d4e7f9c10daff448e208e.
Live public computation is retained and checked against the frozen kernel using
settings/source identity and new1e-12 compatibility assertions. The immutable
reference is saved/reopened for the existing exact golden roundtrip check.
All other live long-kernel reconstruction/limit/serialization tests remain.
Original historical serialized bytes were not recovered; provenance explicitly
records the newly generated matching reference, source path and file identity.
No production numerical/hash algorithm or existing tolerance changed.

Windows focused long-kernel/portable-contract selection:50 passed.
Linux:19 lifecycle tests passed, then the verification shim lacked pytest.mark
(the local venv has no pytest), causing a harness-only failure after those
passes. The shim was repaired; only the unrun hash/roundtrip part was repeated
and passed. Failed attempt preserved. Tests were not skipped or weakened.

- `root-tail-reference-01`: finished, exit 0, process wall 1.462589000 s.
- `root-tail-win-01`: finished, exit 0, process wall 3.085888100 s.
- `root-tail-linux-01`: finished, exit 1, process wall 8.517257300 s.
- `root-tail-linux-hash-02`: finished, exit 0, process wall 3.692241000 s.

Focused check pool used 16.757975400 s /120 s. No native/BEM/
compiler campaign occurred. Seven of101 historical protected files differ
prospectively (process module, two lifecycle tests, four legacy-oracle tests);
94 unchanged. Numeric/hash/fixture production algorithms and the Linux guard
are unchanged; before/after hashes are retained in source reconciliation.
The installed hydro wheel remains current because this tail changes only tests,
reference data, CI and documentation. Clean GitHub CI is pending at this commit;
the final checkpoint will record the actually observed source matrix result.

## Observed source CI success and final checkpoint

Final source commit `4f10a55d8b45b57249b0aa8ca37c3b1cf1b44022` passed all four
clean GitHub matrix jobs: https://github.com/audunarn/ANYhydro/actions/runs/38082088864.
Linux Python3.11/3.13:819 passed,1 platform skip each. Windows Python3.11/3.13:
799 passed,21 platform skips each. Wheels built successfully on all four legs.
Marine source `77c308bdc875c43ad4065f2332a5d64fde23d5a1` also passed its complete
Windows/Linux3.11/3.13 matrix:
https://github.com/audunarn/ANYmarine/actions/runs/38079388189.
These are the observed source baselines; a documentation-only commit saves the
final checkpoint. Both CHECKPOINT.md files identify the current restart state,
which keeps scientific development paused and W5 NO_GO. Remaining numerical,
optional integration and qualification gates are not promoted by CI success.
All owned workers are terminal and all delivery execution pools frozen.
