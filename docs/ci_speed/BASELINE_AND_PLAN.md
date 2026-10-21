# Test-regime speed-up for ANYhydro and ANYmarine: baseline, options, plan

Date 2026-10-21. Status: **baseline measured, plan proposed, nothing implemented, nothing pushed.**
Worktrees: ANYhydro `claude/beautiful-sutherland-85c3dd` (this one, HEAD 49dafc0);
ANYmarine `C:/Github/wt/anymarine-ci-speed`, branch `claude/ci-speed` (HEAD e5ed5aa). The
main checkouts were not touched. Raw data: `docs/ci_speed/raw/`.

## 1. Baseline (measured)

Machine: 32 logical cores, 127 GB RAM, Windows 11. All runs single thread BLAS
(`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`, as CI).

### Key finding: the two repos have different problems

| | ANYhydro | ANYmarine |
|---|---|---|
| Test files | 75 (54 in the portable CI allowlist) | 101 |
| What GitHub CI runs | the 54-module portable selection, 1291 pass + 61 skip | **6 test files only** (headless subset) + wheel build |
| GitHub CI wall (last 8 runs) | 5.8-7.7 min (limit 10) | 0.7-0.9 min |
| Latest run, per cell: tests / pip install | ubuntu 170-179 s / 16-18 s; **windows 3.13: 296 s / 51 s (6.1 min of 10)** | not significant |
| Local "everything" | portable selection 180 s (3m03 wall) | 1126 s (18.8 min) when run one file per process |
| Real problem | CI cell nearing its 10 min limit; 4 cells per push | the 100-file local suite is slow, and one test crashes the whole pytest process |

So for ANYmarine GitHub CI there is nothing to speed up (0.9 min); the cost is the local full suite.
For ANYhydro the CI cell is the cost, and 65 % of its test time is two files.

### ANYhydro (portable selection, serial, py3.13, `run_portable_checks.py`, 1352 tests)
Result: 1291 passed, 61 skipped, 180.3 s pytest time, 183.7 s wall. Collection 2.1 s (0.9 s
Python start). Disk: **19 GB** basetemp for one run (one basetemp reused).

| Where time goes | seconds | share |
|---|---|---|
| `test_check_energy_artifact.py` (about 60 tests at 2-4.3 s) | 82.2 | 46 % |
| `test_check_operator_artifact.py` (about 15 tests at 2-2.2 s) | 33.4 | 19 % |
| next 18 files | about 45 | 25 % |
| 1166 tests under 0.1 s | 18.7 | 10 % |

Slowest 30 tests are all in those two files (2.1-4.3 s each, 77 s together). Cause: a
function-scoped `route` fixture rebuilds a mocked 864-panel artifact case (writes the npz
fixtures, the 19 GB) for every test. Slowest 10 tests = 34 s, top 50 = 116 s.
Full table: `raw/hydro-portable-analysis.txt`.

### ANYmarine (100 files; `test_native_member_authoring_qt.py` excluded, see below)
Interpreter: py3.14 (the one with PySide6 6.12 and the editable any3dview), PYTHONPATH =
marine `src` + hydro `src`. One pytest process per file, serial: **1125.8 s**, 1482 cases;
**164 fail/error in this environment** (missing `HAMS_QT_STORED_OUTPUT`, `anyloads`,
`ANYMARINE_ADOPTION_EVIDENCE`, etc.; they fail the same way on every run, set stored in
`raw/marine-baseline-fail-ids.txt`).

| Where time goes | seconds | share |
|---|---|---|
| `test_rao_mooring.py` (4 tests) | 338 | 30 % |
| `test_rao_review_fixes.py` | 122 | 11 % |
| `test_rao_extraction.py` (2 tests) | 98 | 9 % |
| remaining 97 files | about 570 | 50 % (median file about 3 s, mostly interpreter + Qt start) |

GUI/Qt files are not slow (4-12 s each); they are numerous. Solver-style time is the three
`test_rao_*` files.

**Local defect found (not fixed, not mine to fix):** `tests/test_native_member_authoring_qt.py::
test_production_handler_adds_member_updates_project_and_preview` ends the interpreter with a
Windows access violation (reproduced twice, line 191), and the pytest process then hangs. A
plain `pytest` over the whole suite therefore died at 57 % and sat until my 55 min timeout
(first attempt, no JUnit written). Any runner must use one process per job with a hard timeout.
The file is excluded from the timing baseline and must be run (and fail visibly) in the full tier.

### Parallel check (already measured, no new software)
Same 100 files, 8 concurrent pytest processes, each its own `--basetemp`: **455.9 s** (was
1125.8 s, 2.5x), and the pass/fail set is **identical** (1482 cases, the same 164 ids).
Limit is the single 338 s file; splitting the large files by test id would bring it under about 200 s.

## 2. Options, expected gain, risk

| # | Option | Gain | Risk |
|---|---|---|---|
| A | Own parallel runner (`tools/dev_checks.py`), one process per file/test-id job, longest first, per-job basetemp deleted after the job, hard timeout, completeness check. **No new dependency.** | marine local 19 -> about 3-8 min (measured 7.6 min with files only); hydro 3 -> about 1 min | low; isolation per process is stricter than today (module-level state cannot leak between jobs) |
| B | `slow` marker from a recorded duration table (as ANYgeometry C2) with `conftest.py` | fast tier: hydro about 60 s serial (tests under 1 s), marine drops the 3 `rao_*` files (-558 s serial) | a test becomes unmarked-slow when it grows: needs the guard (as `ci_shards.py guard`) |
| C | Duration-balanced test-level shards in CI (as C3), hydro matrix cell split in 3 shards | hydro Windows cell 296 s -> about 110 s per shard (parallel jobs); ends the 10 min limit pressure | more jobs per push (billable minutes about equal; wall shorter) |
| D | pytest-xdist | similar to A | **needs install, owner said ask first**; shares one process tree, Qt access violation would kill a worker. A already gives the gain without it. Not recommended now |
| E | Hydro matrix: development path = ubuntu 3.13 + windows 3.11 only (or 1 OS), final path = 4 cells | halves CI cost per push | the other 2 cells only run in the final path; a 3.11/3.13 break is found later |
| F | Run on `workflow_dispatch` + weekly + PR to main; push to main only the fast path | removes the most runs (today every push AND every PR push runs all 4 cells; branches with an open PR run twice) | needs owner decision on cadence |
| G | `concurrency` group with cancel-in-progress (per ref) | stops stacked runs (owner's complaint 2026-10-21) | none for tests; do not cancel on `main` final path |
| H | Path filters (docs-only changes skip tests) | rare, small | a path list that is wrong skips tests; use a `plan` job as ANYgeometry does, or leave out |
| I | pip cache (`cache: pip`) | hydro install 16-51 s -> about 10-25 s per cell | none |
| J | Make the `route` fixture cheaper (build the 864-panel case once per module and copy, or in-memory) | hydro 116 s -> about 25 s serial, 19 GB -> a few GB | **medium**: these tests exercise refusal paths and mutate the files; sharing must not weaken what a test mutates. Needs its own review slice. Not in this first pass |
| K | Marine CI: nothing (0.9 min) | - | - |

## 3. Proposed implementation (needs owner go-ahead for the workflow parts)

Slices, each with its acceptance criterion:

1. **S1 duration tables** (`reports/ci/durations.json` per repo, tool `tools/ci_durations.py`): built from the JUnit of the baseline runs. *Accept:* table lists every collected test id with seconds.
2. **S2 tiers**: `slow` marker applied in `tests/conftest.py` from the table (hydro threshold 1.0 s; marine 10 s, so only `test_rao_*`); `ci_shards.py`-style `verify`/`guard`. *Accept:* `-m slow` plus `-m "not slow"` collected ids = full collected ids, equal as sets (script `tools/ci_shards.py verify`), checked for both repos and the hydro portable selection.
3. **S3 local runner** `tools/dev_checks.py`: `--tier fast|full`, `--jobs N` (default min(8, cores/2)), test-id jobs for slow files, per-job basetemp removed afterwards (disk), 900 s hard timeout per job, report JSON with the ids run. *Accept:* full tier result identical to the baseline pass/fail set (hydro 1291/61; marine the 164 ids), fast tier + slow tier ids = full ids.
4. **S4 workflows** (written, **not pushed**): hydro `ci.yml` -> `development.yml` (PR + push to non-main: fast tier, ubuntu 3.13 and windows 3.11, sharded 3 ways) and `ci.yml` final path (`workflow_dispatch` + weekly cron + PR to main: 4 cells x 3 shards, all tests, with a completeness job); `concurrency` cancel-in-progress; `cache: pip`. Marine: keep its 6-file job (0.9 min), add `concurrency` and `cache: pip`, and a weekly/dispatch full headless job only for tests runnable without Qt/solver. *Accept:* a local dry run of each workflow's commands (the shard command lines) reproduces the same ids; YAML parsed (`yaml.safe_load`) and `actionlint` if available.
5. **S5 report** with before/after table.

Not planned in this pass: option J (fixture cost), option D (xdist), changing any test.

## 4. Decisions needed from the owner
1. OK to implement S1-S3 (local tooling, no workflow change, nothing pushed) now? (Expected: yes.)
2. CI cadence (S4): my recommendation is fast tier on PR/push, full 4-cell matrix on `workflow_dispatch` + weekly + PR to main. Confirm or change.
3. Reduced hydro development matrix (ubuntu 3.13 + windows 3.11) acceptable?
4. ANYmarine: the `test_native_member_authoring_qt` access violation: separate task for whoever owns the GUI work; I will keep it in the full tier and flag it, not skip it.
5. pytest-xdist: not requested; install only if you want it.

## 5. Not verified
GitHub timings for ANYmarine's local suite (CI runs 6 files only); macOS/Linux behaviour of the new runner (Windows only locally); the hydro baseline ran Python 3.13 only (3.11 not installed with dependencies locally); the 164 marine failures are environment-specific to this machine and say nothing about CI.

## 6. S1-S3 result (2026-10-21, local, uncommitted, nothing pushed)

| | before (serial) | fast tier | full tier | same results? |
|---|---|---|---|---|
| ANYhydro portable selection (1366 tests incl. 14 new) | 184 s (original runner: 151 s on a quiet rerun) | **19 s**, 1308 tests | **46 s** | full: 1305 passed, 61 skipped, 0 failed = original runner |
| ANYmarine (1503 tests incl. 14 new) | 1126 s per-file serial (plain `pytest` crashes and hangs) | **94 s**, 1490 tests | **358 s** (floor: one 328 s test) | 0 new failures; the 163 environment failures of the baseline unchanged; 22 tests of the crashing file reported MISSING (fail-closed) |

- Partition check (`dev_checks.py --verify`): fast + slow = full, disjoint, for both repos.
- Baseline correction: one baseline failure (`test_verify_installed_application::test_native_preparation_control_refuses_missing_marine_compiler`)
  was an artefact of my relative `PYTHONPATH=src`; with an absolute path it passes. The expected-failure list drops it.
- New files per repo: `tools/ci_tiers.py`, `tools/dev_checks.py`, `reports/ci/durations.json`, `tests/test_ci_tiers.py`
  (hydro: also in the portable allowlist and `docs/PORTABLE_CI.md`), `tests/conftest.py` hooks, `docs/TESTING.md`.
- Not done: S4 workflows, option J (fixture cost), commits.

## 7. S4 workflows (written 2026-10-21, not pushed, not run on GitHub)

ANYhydro: `development.yml` (new; pull requests and pushes to main; fast tier on ubuntu/3.13 and windows/3.11;
partition check, tier guard at 4 s, wheel build; cancel-in-progress on pull requests) and `ci.yml` (final path;
`workflow_dispatch`, Monday 03:00 UTC, pull requests to main; a `completeness` job plus 4 cells x 3 shards, each
`dev_checks.py --tier full --shard k/3 --jobs 2`; wheel build on shard 1). Both use `cache: pip`.
ANYmarine: `ci.yml` keeps its six headless files (+ `test_ci_tiers.py`), push only on main, pull_request, dispatch,
concurrency, `cache: pip`.
Local reproduction: the three hydro shards with `--jobs 2` gave 500 + 401 + 404 = 1305 passed, 34 + 15 + 12 = 61
skipped, 0 failed, 0 missing (= the original runner); the development commands gave 1247 passed, 61 skipped in 37 s.
YAML parses; actionlint is not installed here. Not verified: the runner timings (2-core runners are slower; the
shards took 35-39 s here), the pip cache keys, and the changed job names versus any required status checks
(job names change: "base (...)" becomes "Full suite (...)" and a new "Fast tier (...)").

## 8. Review (Codex, static: its sandbox cannot run processes here, so the code was embedded in the prompt)
Round 1: 1 P1 + 3 P2, all fixed (job exit status must match its JUnit; table update gated on a complete run and merged;
empty selection refused; timeout kills the process tree). Round 2: 3 P2, fixed (empty shard; baseline file overwrite;
output to a file instead of a pipe). Round 3: 2 P2 + 1 P3: fixed the output bound, POSIX cleanup after a finished job and
the hard-link alias; **left open: on Windows a background process left by a finished job is not killed** (needs a Job
Object; documented in docs/TESTING.md). Not a clean review; three rounds are used, owner decides.
Final local checks: hydro full tier 1319 passed / 61 skipped / 0 failed in 44 s (1380 tests), `--verify` and
`--check-shards 3` clean; marine `--verify` clean, `test_ci_tiers.py` 28 passed, last marine full tier
1330 passed + the 163 known environment failures, 0 new failures, 22 reported missing (the crashing file), 478 s.
