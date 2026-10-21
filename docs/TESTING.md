# Running the tests (ANYhydro)

One command per tier, from the repository root; both run the portable base-CI selection
(`tools/run_portable_checks.py`, 54 modules) in parallel jobs and check afterwards that every selected
test reported.

```
python tools/dev_checks.py                 # fast tier: tests not recorded as slow; about 20 s here
python tools/dev_checks.py --tier full     # every test of the selection; about 50 s here (serial: 184 s)
```

- Tiers come from `reports/ci/durations.json` (seconds per test id, from a full local run). A test at or
  above `SLOW_SECONDS` (1 s, `tools/ci_tiers.py`) carries the `slow` mark, set by `tests/conftest.py`;
  a test that is not in the table is fast. `-m slow` plus `-m "not slow"` is exactly the selection:
  `python tools/dev_checks.py --verify`.
- Refresh the table after adding or changing heavy tests: `python tools/dev_checks.py --tier full --update-durations`.
  `python tools/ci_tiers.py guard .pytest-dev` (after a run) lists unmarked tests slower than 2 s.
- Options: `--jobs N` (default half the cores, at most 8), `--shard k/N` (a duration-balanced share, for a CI
  matrix), `--timeout S`, `--keep` (keep each job's basetemp), `--workdir DIR` (default `.pytest-dev/`, JUnit
  files, logs and `report.json` stay there).
- Each job has its own `--basetemp`, removed when the job ends: peak disk use is the sum of the running jobs,
  not 19 GB per run. Exit status 1 on a failing test, a selected test that did not report (crash, timeout),
  an unreadable JUnit file or an abnormal job exit.
- Run a single file as before: `python tools/run_portable_checks.py` has no file argument; use
  `python -m pytest tests/test_x.py --basetemp=<dir>` with `PYTHONPATH=src` for the full (non-portable) modules.
- Known limit: a timeout kills the job's whole process tree. On Windows a process that a finished job left running
  in the background is not killed (no process group to signal; a Job Object would be needed); on Linux it is.
