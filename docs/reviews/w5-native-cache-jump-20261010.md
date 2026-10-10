# W5 native cache/jump review — 2026-10-10 (final integration/handoff)

Repository: C:/Github/ANYhydro. Owned boundary:
`.development-evidence/w5-native-cache-jump-20261010` (native_check.py,
test_mock_checks.py, mock_backend.py, check.py pools) plus this docs tree.
Root controls, `src/`, old inputs, Git and siblings untouched. Development
evidence only; no scientific qualification is conferred by any fixture,
mock or native receipt below.

## Outcome

Complete for this slice and verified by execution (mock-connected control
algorithm; native execution NOT executed, admission pending). Installed
source identity unchanged: 36 anyhydro modules verified against
`installed-identity.json`; 15 protected old inputs verified unchanged
against `input-identities.json`.

## Root source-inspection finding and correction (this session)

`run()` captured `threadpool_info()` before `_stage_profile` imports SciPy
and `_stage_backend` imports Capytaine. That pre-import list ("Environment1")
is insufficient evidence about pools loaded later, so no measured claim may
be derived from it.

Correction in `native_check.py` `_stage_backend`, after the actual/injected
backend module is resolved:

- Re-query the current `threadpool_info()` (default: threadpoolctl; a
  `HarnessConfig.backend_threadpool_info` callable is the mock-only control
  injection point, same pattern as `capt_module`/`delhommeau_cls`/`clock`).
- Validate every actual pool has `num_threads == 1`; refuse with
  `IdentityRefused` BEFORE any native constructor can run (the backend stage
  precedes observer installation and the caller).
- Bind the backend-stage list to all records: `self.pools`,
  `identity.json` (`threadpools_stage: 'backend'`), the failure `status.json`
  record (`threadpools` field), `trace.json` (`threadpools` field) and the
  finalize summary (`threadpools_stage: 'backend'`). The pre-import capture
  in `run()` remains only as an early refusal and is labelled
  `threadpools_stage: 'pre_import'` in `identity.json`.

Verified public `_create_or_load_tabulation` returns the cache filename
(pinned actual installed source), so the existing observer capture of the
cache path is correct and unchanged.

Preserved unchanged: production Rankine API correction
(`evaluate_rankine_only(mesh1, mesh2, *, adjoint_double_layer,
early_dot_product, diagonal_term_in_double_layer)`; the Rankine kernel
takes no water_depth/wavenumber), supervisor wall deadline matching, raw
mutation-return preservation, unchanged real caller path and the captured
same green function. No physics or threshold changes.

## New focused mock test

`test_backend_stage_late_multithread_pool_refuses_before_constructor`
injects a late pool with `num_threads=2` (invisible to the pre-import
capture) and proves: backend-stage refusal (`IdentityRefused`, stage
`backend`), observer never installed (counters empty), no constructor body
entered, and partial evidence preserved (`status.json` + `identity.json`
under `results/mock/late-pool-refusal/`, both carrying the late pool and
`threadpools_stage: 'backend'`). The unchanged happy path additionally
asserts the backend-stage binding with all actual pools at one thread
(unchanged env/source case).

## Windows launcher interoperability repair (this session)

Diagnosed blocker, measured by the root ordinary attempt
`launcher-identity-1` (receipt preserved): on Windows the pinned venv
`Scripts/python.exe` is a redirector, so check.py's Popen pid (launcher
5816) differs from the interpreter pid (63764) whose `os.getppid()` is 5816.
The strict `receipt pid == os.getpid()` check in `check_authority` therefore
refused the consumed `native-once` authority before any native work
(numerical wall 0.18020719999913126 s, zero constructor/Green/system calls;
failure evidence preserved under `attempts/native-once/`). The consumed
authority is NOT reusable and was NOT edited.

Narrow process identity contract adopted in `native_check.py`
`check_authority` (no scientific investigation reopened):

- Direct receipt pid == interpreter pid remains valid everywhere.
- Additionally, ONLY on `os.name == 'nt'`, ONLY when `sys.executable` is
  exactly the pinned runtime executable AND the receipt `command[0]` is
  exactly that same runtime, the receipt launcher pid may equal this
  process's single direct parent pid
  (`windows-venv-redirector-single-parent`).
- Any other pid (wrong parent, non-Windows, wrong runtime on either side,
  non-integer pid) refuses; the running-receipt, criteria/script hash,
  attempt, deadline and command binding checks are untouched.
- The accepted lineage method plus launcher pid, interpreter pid and
  interpreter parent pid are recorded separately in the authority evidence
  (`process_lineage` in the `check_authority` record, bound into
  `identity.json`/status by the existing authority stage). The receipt
  launcher pid remains the process-tree timeout/cancellation target in
  check.py (unchanged); the parent check.py owns the active Popen launcher
  and the Windows launcher stays waiting on its child.

Focused mock regression `test_process_lineage_contract`: direct match pass
(platform/runtime independent); Windows exact-runtime single-parent pass
with full lineage recording; wrong parent, non-Windows, wrong child
runtime, wrong receipt command runtime and non-integer pid all refuse.
Existing wrong-command/criteria/script/deadline refusals unchanged
(`test_attempt_receipt_binding_refusals`).

Metadata-only control probe `windows-lineage-probe.py` (owned dir), run
ONCE through check.py ordinary as `windows-lineage-probe-1`: receives the
existing `native_check.py` path as argument (satisfying the receipt
harness-path guard; the parent receipt already binds scripts/command and
deadline), imports that module by source, instantiates no backend, writes a
clearly MOCK fixture authority in its own unique attempt temp dir (the root
`native-authority.json` untouched), copies the parent receipt
deadline/criteria and attempt, and calls the product `check_authority` on
the real ordinary receipt. Accepted lineage
`windows-venv-redirector-single-parent`: launcher pid 40800 (receipt
Popen target), interpreter pid 65096, interpreter parent pid 40800, real
`getpid`/`getppid` reported, zero capytaine imports, zero constructor
calls, authority/receipt deadlines bound (98300.470293). This proves the
function process-lineage path only — NOT the full native CLI (exact
command-line equality remains separately exercised by the mock suite when
`sys.argv[0]` is `native_check.py`), NOT the native runtime, NOT any
science, and it does NOT admit the next real native run.

## Actual receipts (prior results preserved; changed scope run once each)

| Attempt (check.py ordinary) | Result | Process wall |
|---|---|---|
| mock-connected-parent-1 (prior) | 6 fail / 10 pass | preserved |
| mock-connected-parent-2 (prior) | 16 pass | 13.240441600006307 s |
| mock-latepool-parent-1 (prior) | 17 pass, 0 errors/failures/skipped (JUnit `tests="17" time="11.512"`) | 13.121849900009693 s |
| mock-lineage-parent-1 (this session, changed scope) | 18 pass, 0 errors/failures/skipped (JUnit `tests="18" time="11.915"`) | 13.787977700005285 s |
| windows-lineage-probe-1 (this session, launcher-only probe) | accepted single-parent lineage, exit 0 | 0.16494740000052843 s |

All prior failure evidence (including the earlier 20/17-case failures and
the failed `native-once`) is preserved under `attempts/` and
`results/mock/`; nothing was re-run except the single changed-scope suite
and the single launcher-only probe above, both required because
`native_check.py` changed. The prior 17-pass receipt is preserved as-is and
is a mock control result, not a native pass.

Resource pools after this session: ordinary pool 180 s budget,
58.19665040001564 s used, 121.80334959998436 s remaining; numerical pool
180 s budget, 0.18020719999913126 s used (consumed by the failed
`native-once`), not frozen. Prior continuations cumulative
385.4541201000102 s of the old 420 s allocation (frozen; unspent not
carried).

## Native admission (pending, NOT executed; prior attempt consumed and failed)

The prior `native-once` attempt consumed its authority and failed at the
now-repaired process-lineage refusal; that authority is spent and is NOT
reusable. A next real native run requires a NEW root-issued
`native-authority.json` admitted and consumed by the check.py launcher for
that exact attempt; none was created or requested this session. Exact
command shape (NOT executed):

```
C:/ANYworkspaceWork/anymarine-evidence/runtime/Scripts/python.exe \
  C:/Github/ANYhydro/.development-evidence/w5-native-cache-jump-20261010/check.py \
  numerical native-once \
  C:/ANYworkspaceWork/anymarine-evidence/runtime/Scripts/python.exe \
  C:/Github/ANYhydro/.development-evidence/w5-native-cache-jump-20261010/native_check.py \
  --attempt native-once
```

## Scope and open qualification

Local face-0 Rankine-only convention/scope only. Explicitly open: near-plane
behaviour below the 1e-4*face_radius AT-suppression cutoff, full-wave
qualification, and full operator qualification. The conditional full batch
is one measured 864-source batch cost only; no cost extrapolation. Native
self finite-difference consistency is retained separately and is not an
admission gate.
