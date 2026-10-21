"""Local test runner: tiers, parallel jobs, a completeness check (CI speed-up slice S3).

    python tools/dev_checks.py                  # fast tier: tests not recorded as slow (reports/ci/durations.json)
    python tools/dev_checks.py --tier full      # every test of the selection
    python tools/dev_checks.py --verify         # the fast and slow tiers together equal the full selection
    python tools/dev_checks.py --tier full --update-durations   # refresh the duration table from this run

The selected tests are split into jobs by recorded duration (longest first); each job is its own pytest
process with its own ``--basetemp`` (removed afterwards, ``--keep`` to keep it) and a hard timeout, so a crash or
hang in one job cannot take the others down.  Afterwards the JUnit files are compared with the collected node ids:
a selected test that did not report fails the run (exit 1) -- nothing is skipped silently.  No extra package is needed.

Jobs: ``--jobs`` (default: half the cores, at most 8).  ``--shard k/N`` runs only the k-th of N duration-balanced
shares of the selection (for a CI matrix); every selected id is in exactly one shard.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ci_tiers  # noqa: E402

# ---- repository settings ------------------------------------------------------------------------------------
IDS_FILE_ENV = "ANYHYDRO_TEST_IDS_FILE"          # read by tests/conftest.py
RUN_PREFIX = [sys.executable, str(ROOT / "tools" / "run_portable_checks.py")]   # the portable base-CI selection
EXTRA_ARGS: list[str] = []      # extra arguments for a test job (the portable runner already adds -q and no:cacheprovider)
COLLECT_ARGS: list[str] = []    # extra arguments for collection (one -q lists node ids; the portable runner adds it)
PACK_JOBS = True      # True: pack the selection into --jobs processes; False: one process per file/split (isolation)
SPLIT_SECONDS = 60.0  # a file recorded above this is split by test id into parts of about 30 s
WORKDIR = ROOT / ".pytest-dev"
# --------------------------------------------------------------------------------------------------------------


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH", "")]))
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env.setdefault(name, "1")
    return env


def collect(marker: str | None) -> list[str]:
    """Node ids pytest selects (``marker`` is a ``-m`` expression); the collection must succeed."""
    cmd = [*RUN_PREFIX, "--collect-only", *COLLECT_ARGS]
    if marker:
        cmd += ["-m", marker]
    env = _env()
    env.pop(IDS_FILE_ENV, None)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    ids = [line.strip() for line in proc.stdout.splitlines() if "::" in line and not line.startswith(("ERROR", " "))]
    if proc.returncode != 0 or not ids:   # an empty selection never counts as success
        sys.exit(f"collection failed (exit {proc.returncode}):\n{proc.stdout[-1500:]}\n{proc.stderr[-800:]}")
    return ids


def estimate(table: dict, ids: list[str]) -> dict[str, float]:
    known = table.get("tests", {})
    mean = (sum(known.values()) / len(known)) if known else 0.1
    return {i: known.get(i, mean) for i in ids}


def lpt(weights: dict[str, float], count: int) -> list[list[str]]:
    """Longest first onto the lightest bin; deterministic (ties by name)."""
    bins: list[list[str]] = [[] for _ in range(count)]
    loads = [0.0] * count
    for name in sorted(weights, key=lambda n: (-weights[n], n)):
        k = min(range(count), key=lambda b: (loads[b], b))
        bins[k].append(name)
        loads[k] += weights[name]
    return bins


def make_units(ids: list[str], times: dict[str, float]) -> list[dict]:
    """One unit per file; a file recorded above SPLIT_SECONDS is split by test id."""
    by_file: dict[str, list[str]] = {}
    for node in ids:
        by_file.setdefault(node.split("::")[0], []).append(node)
    units = []
    for file, members in sorted(by_file.items()):
        total = sum(times[m] for m in members)
        if total > SPLIT_SECONDS and len(members) > 1:
            parts = min(len(members), math.ceil(total / (SPLIT_SECONDS / 2)))
            for part in lpt({m: times[m] for m in members}, parts):
                if part:
                    units.append({"files": [file], "ids": sorted(part), "est": sum(times[m] for m in part)})
        else:
            units.append({"files": [file], "ids": sorted(members), "est": total})
    return units


def make_jobs(ids: list[str], table: dict, jobs: int) -> list[dict]:
    units = make_units(ids, estimate(table, ids))
    if not PACK_JOBS:
        return sorted(units, key=lambda u: -u["est"])
    packed = lpt({str(i): u["est"] for i, u in enumerate(units)}, max(1, min(jobs, len(units))))
    result = []
    for members in packed:
        if members:
            chosen = [units[int(i)] for i in members]
            result.append({"files": sorted({f for u in chosen for f in u["files"]}),
                           "ids": sorted(i for u in chosen for i in u["ids"]), "est": sum(u["est"] for u in chosen)})
    return sorted(result, key=lambda u: -u["est"])


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return f" {pid} " in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            proc.kill()


def run_with_timeout(cmd: list[str], cwd: Path, env: dict[str, str], timeout: float) -> tuple[int | str, str]:
    """``(exit status or 'timeout', output)``; on a timeout the whole process tree is killed, not just the child.
    The output goes to a file, not a pipe, so a descendant that outlives a finished job cannot hold the job up."""
    import tempfile
    kwargs: dict = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    with tempfile.TemporaryFile() as sink:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=sink, stderr=subprocess.STDOUT, **kwargs)
        try:
            code: int | str = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            code = "timeout"
        if code != "timeout" and os.name != "nt":
            _kill_tree(proc)   # POSIX: nothing a finished job left behind outlives it (Windows has no process group to signal)
        sink.seek(0)
        return code, sink.read(8_000_000).decode("utf-8", "replace")   # bounded: a leftover child may still write


def job_consistent(code, failing: int) -> bool:
    """A job's exit status must be explained by its JUnit file: 0 with no failing test, 1 with at least one."""
    return (code == 0 and failing == 0) or (code == 1 and failing > 0)


def job_failures(path: Path) -> int:
    """Failing testcases in one JUnit file (-1 when it is missing or unreadable)."""
    try:
        cases = list(ET.parse(path).getroot().iter("testcase"))
    except (ET.ParseError, OSError):
        return -1
    return sum(1 for case in cases if any(child.tag in ("failure", "error") for child in case))


def run_job(index: int, job: dict, args, workdir: Path, marker: str | None) -> dict:
    name = f"job{index:03d}"
    ids_file = workdir / f"{name}.ids"
    ids_file.write_text("\n".join(job["ids"]) + "\n", encoding="utf-8")
    junit = workdir / f"{name}.xml"
    basetemp = workdir / "tmp" / name
    cmd = [*RUN_PREFIX, *([] if PACK_JOBS else job["files"]), *EXTRA_ARGS,
           f"--junitxml={junit}", f"--basetemp={basetemp}"]
    if marker:
        cmd += ["-m", marker]
    env = _env()
    env[IDS_FILE_ENV] = str(ids_file)
    limit = args.timeout or max(300.0, 2 * job["est"] + 120)
    started = time.time()
    code, out = run_with_timeout(cmd, ROOT, env, limit)
    (workdir / f"{name}.log").write_text(out, encoding="utf-8")
    if not args.keep:
        shutil.rmtree(basetemp, ignore_errors=True)
    return {"job": name, "exit": code, "seconds": round(time.time() - started, 1), "expected": len(job["ids"]),
            "estimate": round(job["est"], 1), "ids": job["ids"], "tail": out.strip().splitlines()[-1][:160] if out.strip() else ""}


def read_junit(workdir: Path) -> tuple[dict[str, str], list[str]]:
    """``({node id: outcome}, unmapped)`` over every job's JUnit file."""
    outcomes: dict[str, str] = {}
    unmapped: list[str] = []
    for path in sorted(workdir.glob("job*.xml")):
        try:
            cases = list(ET.parse(path).getroot().iter("testcase"))
        except (ET.ParseError, OSError):
            unmapped.append(f"unreadable {path.name}")
            continue
        for case in cases:
            node = ci_tiers.junit_node_id(case.get("classname", ""), case.get("name", ""))
            if node is None:
                unmapped.append(f"{case.get('classname')}::{case.get('name')}")
                continue
            kinds = {child.tag for child in case}
            outcome = ("error" if "error" in kinds else "failed" if "failure" in kinds
                       else "skipped" if "skipped" in kinds else "passed")
            if outcomes.get(node) not in ("error", "failed"):
                outcomes[node] = outcome
    return outcomes, unmapped


def shard_of(ids: list[str], table: dict, spec: str) -> list[str]:
    index, _, count = spec.partition("/")
    index, count = int(index), int(count)
    if not 1 <= index <= count:
        sys.exit(f"--shard expects k/N with 1 <= k <= N, got {spec!r}")
    return sorted(lpt(estimate(table, ids), count)[index - 1])


def verify() -> int:
    full, slow, fast = (set(collect(m)) for m in (None, "slow", "not slow"))
    table = ci_tiers.load_table()
    expected_slow = {node for node in full if ci_tiers.is_slow(table, node)}
    problems = []
    if slow | fast != full:
        problems.append(f"slow + fast differ from full by {len((slow | fast) ^ full)} ids")
    if slow & fast:
        problems.append(f"{len(slow & fast)} ids in both tiers")
    if slow != expected_slow:
        problems.append("the slow tier does not equal the table's slow ids")
    print(f"verify: full {len(full)}, fast {len(fast)}, slow {len(slow)}; table has {len(ci_tiers.slow_ids(table) - full)} stale slow ids")
    for problem in problems:
        print("FAIL:", problem)
    return 1 if problems else 0


def check_shards(count: int) -> int:
    """Every id of the full selection is in exactly one of the ``count`` shards (what the CI matrix relies on)."""
    table = ci_tiers.load_table()
    ids = collect(None)
    shards = [shard_of(ids, table, f"{k}/{count}") for k in range(1, count + 1)]
    flat = [i for shard in shards for i in shard]
    ok = sorted(flat) == sorted(ids) and len(flat) == len(set(flat))
    print(f"shards: {len(ids)} tests in {count} shards of {[len(s) for s in shards]}: {'complete, disjoint' if ok else 'FAIL'}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", choices=("fast", "full"), default="fast")
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--timeout", type=int, default=None,
                        help="hard limit per job, seconds (default: 300, or twice the recorded estimate plus 120 if larger)")
    parser.add_argument("--shard", help="k/N: run only this share of the selection")
    parser.add_argument("--workdir", type=Path, default=WORKDIR)
    parser.add_argument("--keep", action="store_true", help="keep each job's basetemp")
    parser.add_argument("--verify", action="store_true", help="check that the tiers partition the selection, then exit")
    parser.add_argument("--check-shards", type=int, metavar="N",
                        help="check that N shards partition the full selection, then exit")
    parser.add_argument("--update-durations", action="store_true", help="rebuild the duration table from this (full) run")
    parser.add_argument("--expect-failures", type=Path,
                        help="file of node ids known to fail here; the run passes iff no other test fails")
    parser.add_argument("--write-failures", type=Path, help="write the failing node ids to this file")
    args = parser.parse_args(argv)
    if args.verify:
        return verify()
    if args.check_shards:
        return check_shards(args.check_shards)
    table = ci_tiers.load_table()
    marker = "not slow" if args.tier == "fast" else None
    ids = collect(marker)
    expected = shard_of(ids, table, args.shard) if args.shard else ids
    if not expected:
        sys.exit(f"nothing to run: the selection{' shard ' + args.shard if args.shard else ''} is empty")
    known = None
    if args.expect_failures:
        if args.write_failures and (args.write_failures.resolve() == args.expect_failures.resolve() or (
                args.write_failures.exists() and os.path.samefile(args.write_failures, args.expect_failures))):
            sys.exit("--write-failures and --expect-failures name the same file: the baseline would be overwritten")
        known = {line for line in args.expect_failures.read_text(encoding="utf-8").splitlines() if line}
    workdir = args.workdir.resolve()
    shutil.rmtree(workdir, ignore_errors=True)
    (workdir / "tmp").mkdir(parents=True)
    jobs = make_jobs(expected, table, args.jobs)
    print(f"{args.tier} tier{' shard ' + args.shard if args.shard else ''}: {len(expected)} tests in {len(jobs)} job(s), "
          f"{min(args.jobs, len(jobs))} at a time; estimate {sum(j['est'] for j in jobs):.0f} s of work", flush=True)
    started = time.time()
    with ThreadPoolExecutor(max(1, args.jobs)) as pool:
        results = list(pool.map(lambda pair: run_job(pair[0], pair[1], args, workdir, marker), enumerate(jobs)))
    wall = time.time() - started
    outcomes, unmapped = read_junit(workdir)
    want = set(expected)
    missing = sorted(want - set(outcomes))
    extra = sorted(set(outcomes) - want)
    counts = {kind: sum(1 for o in outcomes.values() if o == kind) for kind in ("passed", "failed", "error", "skipped")}
    failing = sorted(n for n, o in outcomes.items() if o in ("failed", "error"))
    report = {"tier": args.tier, "shard": args.shard, "wall_seconds": round(wall, 1), "expected": len(expected),
              "counts": counts, "missing": missing, "extra": extra, "unmapped": unmapped, "failing": failing,
              "jobs": [{k: v for k, v in r.items() if k != "ids"} for r in results]}
    (workdir / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.write_failures:
        args.write_failures.write_text("\n".join(failing) + "\n", encoding="utf-8")
    for r in results:
        r["failing"] = job_failures(workdir / f"{r['job']}.xml")
    bad_jobs = [r for r in results if not job_consistent(r["exit"], r["failing"])]
    print(f"wall {wall:.0f} s; expected {len(expected)}; {counts}; missing {len(missing)}; extra {len(extra)}; "
          f"unmapped {len(unmapped)}; jobs with abnormal exit {len(bad_jobs)}")
    for r in sorted(results, key=lambda r: -r["seconds"])[:5]:
        print(f"  {r['job']} {r['seconds']:7.1f} s (estimate {r['estimate']:.0f}) exit {r['exit']} "
              f"{r['expected']} tests | {r['tail']}")
    for r in bad_jobs:
        print(f"ABNORMAL {r['job']} exit {r['exit']}, {r['failing']} failing in its JUnit file: see {workdir / (r['job'] + '.log')}")
    for node in missing[:10]:
        print("MISSING", node)
    ok = not (missing or extra or unmapped or bad_jobs)
    new_failures: list[str] = []
    if args.expect_failures:
        new_failures = sorted(set(failing) - known)
        now_passing = sorted(n for n in known if outcomes.get(n) in ("passed", "skipped"))
        print(f"against {args.expect_failures.name}: {len(new_failures)} new failures, {len(now_passing)} known failures now pass")
        for node in new_failures[:10]:
            print("NEW FAILURE", node)
        ok = ok and not new_failures
    elif failing:
        ok = False
        print(f"{len(failing)} failing, first: {failing[0]}")
    if args.update_durations:
        # only a full, unsharded run may rewrite the table; a test that did not report is tolerated only when its
        # whole file is listed under slow_files (a known crash), and a failure must be a known one
        known_bad = {n for n in want if n.split("::")[0] in table.get("slow_files", [])}
        clean = (args.tier == "full" and not args.shard and not extra and not unmapped
                 and set(missing) <= known_bad
                 and all(set(r["ids"]) <= known_bad for r in bad_jobs)
                 and (args.expect_failures is not None or not failing))
        if args.expect_failures is not None:
            clean = clean and not new_failures
        if not clean:
            print("--update-durations needs a complete full run (unreported tests only in slow_files, no new failures); table not changed")
            return 1
        durations, _ = ci_tiers.junit_durations(sorted(workdir.glob("job*.xml")))
        ci_tiers.write_table(durations, ci_tiers.TABLE, keep_ids=want)
        print(f"wrote {ci_tiers.TABLE} ({len(durations)} tests measured)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
