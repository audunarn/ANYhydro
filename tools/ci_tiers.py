"""Test tiers from a recorded duration table (CI speed-up slices S1 and S2).

``reports/ci/durations.json`` records seconds per test node id from a full local run.  ``tests/conftest.py``
marks the tests at or above ``SLOW_SECONDS`` as ``slow`` before ``-m`` deselection, so ``-m slow`` and
``-m "not slow"`` partition the suite: the fast tier is ``-m "not slow"``, the full tier is everything.
Tests that are not in the table are fast (a new test runs in the fast tier until the table is refreshed).

Commands (``python tools/ci_tiers.py <command>``):
  update JUNIT...   rebuild the table from JUnit files (``--out`` to choose the file)
  guard JUNIT...    fail when a test that is not marked slow took more than ``--limit`` seconds
  show              print the tier sizes
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "reports" / "ci" / "durations.json"
SLOW_SECONDS = 1.0       # seconds at or above which a test is in the slow tier (set per repository)
GUARD_SECONDS = 2.0      # guard: an unmarked test slower than this means the table needs a refresh
TEST_DIR = "tests"


def load_table(path: Path = TABLE) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"tests": {}}


def slow_ids(table: dict, threshold: float = SLOW_SECONDS) -> set[str]:
    return {name for name, seconds in table.get("tests", {}).items() if seconds >= threshold}


def is_slow(table: dict, node: str, threshold: float = SLOW_SECONDS) -> bool:
    """Slow by recorded time, or in a file listed under ``slow_files`` (a whole file kept out of the fast tier, e.g.
    one that crashes or hangs the interpreter here: it still runs, and fails visibly, in the full tier)."""
    return table.get("tests", {}).get(node, 0.0) >= threshold or node.split("::")[0] in table.get("slow_files", [])


def junit_node_id(classname: str, name: str) -> str | None:
    """pytest node id of a JUnit testcase (``tests.test_a.TestB`` + ``test_c`` -> ``tests/test_a.py::TestB::test_c``)."""
    parts = classname.split(".")
    for i, part in enumerate(parts):
        if part.startswith("test_") and (ROOT / TEST_DIR / (part + ".py")).is_file():
            return "::".join([f"{TEST_DIR}/{part}.py", *parts[i + 1:], name])
    return None


def junit_durations(paths: list[Path]) -> tuple[dict[str, float], list[str]]:
    """``({node id: seconds}, [unmapped 'classname::name'])`` over JUnit files (a repeated id keeps the larger time)."""
    result: dict[str, float] = {}
    unmapped: list[str] = []
    for path in paths:
        for case in ET.parse(path).getroot().iter("testcase"):
            node = junit_node_id(case.get("classname", ""), case.get("name", ""))
            if node is None:
                unmapped.append(f"{case.get('classname')}::{case.get('name')}")
                continue
            result[node] = max(result.get(node, 0.0), float(case.get("time", 0.0)))
    return result, unmapped


def write_table(durations: dict[str, float], path: Path = TABLE, keep_ids: set[str] | None = None) -> None:
    """Write the table.  With ``keep_ids`` the old entries of those ids that ``durations`` does not cover stay (a
    partial run must not erase the classification of tests that did not report); other old ids are dropped."""
    old = load_table(path)
    tests: dict[str, float] = {}
    if keep_ids is not None:
        tests = {k: v for k, v in old.get("tests", {}).items() if k in keep_ids}
    tests.update({k: round(v, 3) for k, v in durations.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"slow_seconds": SLOW_SECONDS, "source": "local JUnit, seconds per node id",
                                "slow_files": old.get("slow_files", []), "tests": dict(sorted(tests.items()))},
                               indent=0) + "\n", encoding="utf-8")


def _junit_files(args: list[str]) -> list[Path]:
    files: list[Path] = []
    for arg in args:
        path = Path(arg)
        files.extend(sorted(path.glob("*.xml")) if path.is_dir() else [path])
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("update", "guard", "show"))
    parser.add_argument("junit", nargs="*", help="JUnit files or directories of them")
    parser.add_argument("--out", type=Path, default=TABLE)
    parser.add_argument("--limit", type=float, default=GUARD_SECONDS)
    args = parser.parse_args(argv)
    if args.command == "show":
        table = load_table(args.out)
        slow = slow_ids(table)
        tests = table.get("tests", {})
        print(f"{len(tests)} tests in the table, {len(slow)} slow (>= {SLOW_SECONDS} s, "
              f"{sum(tests[i] for i in slow):.0f} s of {sum(tests.values()):.0f} s)")
        return 0
    files = _junit_files(args.junit)
    durations, unmapped = junit_durations(files)
    if not files or not durations:
        print(f"{args.command}: no JUnit results found in the input; nothing was checked", file=sys.stderr)
        return 1
    if args.command == "guard" and unmapped:
        print(f"guard: {len(unmapped)} JUnit cases could not be mapped to a test file, e.g. {unmapped[:3]}", file=sys.stderr)
        return 1
    if args.command == "update":
        if unmapped:
            print(f"{len(unmapped)} JUnit cases could not be mapped to a test file, e.g. {unmapped[:3]}", file=sys.stderr)
            return 1
        write_table(durations, args.out)
        print(f"wrote {args.out} ({len(durations)} tests, {sum(durations.values()):.0f} s)")
        return 0
    table = load_table()
    late = sorted(((s, n) for n, s in durations.items() if not is_slow(table, n) and s > args.limit), reverse=True)
    for seconds, node in late:
        print(f"UNMARKED SLOW {seconds:7.2f} s  {node}")
    print(f"guard: {len(late)} unmarked tests above {args.limit} s")
    return 1 if late else 0


if __name__ == "__main__":
    raise SystemExit(main())
