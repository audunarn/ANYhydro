"""Tests for the tier and parallel-job tooling (tools/ci_tiers.py, tools/dev_checks.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import ci_tiers  # noqa: E402
import dev_checks  # noqa: E402

IDS = [f"tests/test_a.py::test_{i}" for i in range(40)] + [f"tests/test_b.py::TestC::test_{i}[p{i}]" for i in range(25)]
TABLE = {"slow_files": ["tests/test_a.py"], "tests": {IDS[41]: 99.0, IDS[42]: 0.4, IDS[50]: 70.0}}


def _every_id_once(groups):
    flat = [i for group in groups for i in group]
    assert sorted(flat) == sorted(IDS) and len(flat) == len(set(flat))


@pytest.mark.parametrize("count", [1, 2, 3, 7, 100])
def test_lpt_places_every_id_exactly_once(count):
    _every_id_once(dev_checks.lpt(dev_checks.estimate(TABLE, IDS), count))


@pytest.mark.parametrize("spec", ["1/3", "2/3", "3/3"])
def test_shards_partition_the_selection(spec):
    _every_id_once([dev_checks.shard_of(IDS, TABLE, f"{k}/3") for k in (1, 2, 3)])
    assert set(dev_checks.shard_of(IDS, TABLE, spec)) <= set(IDS)


@pytest.mark.parametrize("pack", [True, False])
def test_jobs_cover_the_selection_exactly(monkeypatch, pack):
    monkeypatch.setattr(dev_checks, "PACK_JOBS", pack)
    jobs = dev_checks.make_jobs(IDS, TABLE, 4)
    _every_id_once([job["ids"] for job in jobs])
    if pack:
        assert len(jobs) <= 4


def test_big_file_is_split_by_test_id(monkeypatch):
    monkeypatch.setattr(dev_checks, "PACK_JOBS", False)
    jobs = dev_checks.make_jobs(IDS, TABLE, 4)
    assert len([job for job in jobs if job["files"] == ["tests/test_b.py"]]) > 1


def test_slow_rule_uses_time_and_slow_files():
    assert ci_tiers.is_slow(TABLE, IDS[41], threshold=1.0)
    assert not ci_tiers.is_slow(TABLE, IDS[42], threshold=1.0)
    assert ci_tiers.is_slow(TABLE, "tests/test_a.py::test_not_in_table", threshold=1.0)
    assert not ci_tiers.is_slow(TABLE, "tests/test_b.py::TestC::test_unknown", threshold=1.0)


def test_junit_names_map_to_node_ids():
    this = Path(__file__).stem
    assert ci_tiers.junit_node_id(f"tests.{this}", "test_x") == f"tests/{this}.py::test_x"
    assert ci_tiers.junit_node_id(f"tests.{this}.TestK", "test_x[a-1]") == f"tests/{this}.py::TestK::test_x[a-1]"
    assert ci_tiers.junit_node_id("tests.no_such_module", "test_x") is None


def test_missing_and_failed_tests_are_visible_in_the_junit_read(tmp_path):
    this = Path(__file__).stem
    (tmp_path / "job000.xml").write_text(
        f'<testsuites><testsuite>'
        f'<testcase classname="tests.{this}" name="test_one"/>'
        f'<testcase classname="tests.{this}" name="test_two"><failure message="x"/></testcase>'
        f'<testcase classname="tests.{this}" name="test_three"><skipped/></testcase>'
        f'<testcase classname="tests.{this}" name="test_four"><error message="x"/></testcase>'
        f'<testcase classname="other" name="t"/></testsuite></testsuites>', encoding="utf-8")
    outcomes, unmapped = dev_checks.read_junit(tmp_path)
    base = f"tests/{this}.py::"
    assert outcomes == {base + "test_one": "passed", base + "test_two": "failed",
                        base + "test_three": "skipped", base + "test_four": "error"}
    assert unmapped == ["other::t"]
    assert base + "test_five" not in outcomes   # an absent id stays absent, so the run reports it as missing


# --- review round 1 regressions -------------------------------------------------------------------------------
@pytest.mark.parametrize("code,failing,consistent", [
    (0, 0, True), (1, 2, True),          # clean pass; failing tests reported
    (0, 1, False),                        # exit 0 but a failure in the JUnit file
    (1, 0, False),                        # session-level failure: no failing test explains the exit status
    (5, 0, False), (2, 0, False), ("timeout", 0, False), (-11, 3, False),
])
def test_job_exit_status_must_match_its_junit(code, failing, consistent):
    assert dev_checks.job_consistent(code, failing) is consistent


def test_empty_collection_is_refused(monkeypatch):
    class Done:
        returncode, stdout, stderr = 5, "no tests collected\n", ""
    monkeypatch.setattr(dev_checks.subprocess, "run", lambda *a, **k: Done())
    with pytest.raises(SystemExit):
        dev_checks.collect(None)


def test_update_keeps_unreported_tests_and_drops_vanished_ones(tmp_path):
    table = tmp_path / "durations.json"
    table.write_text('{"slow_files": ["tests/x.py"], "tests": {"tests/a.py::old": 5.0, "tests/a.py::gone": 7.0}}',
                     encoding="utf-8")
    ci_tiers.write_table({"tests/a.py::new": 2.0}, table, keep_ids={"tests/a.py::old", "tests/a.py::new"})
    data = ci_tiers.load_table(table)
    assert data["tests"] == {"tests/a.py::new": 2.0, "tests/a.py::old": 5.0}
    assert data["slow_files"] == ["tests/x.py"]


def test_timeout_kills_the_whole_process_tree(tmp_path):
    pid_file = tmp_path / "child.pid"
    code = (f"import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']);"
            f"open(r'{pid_file}','w').write(str(p.pid)); time.sleep(120)")
    status, _ = dev_checks.run_with_timeout([sys.executable, "-c", code], cwd=tmp_path, env=dict(__import__("os").environ), timeout=4)
    assert status == "timeout"
    pid = int(pid_file.read_text())
    import time
    for _ in range(20):
        if not dev_checks.pid_alive(pid):
            break
        time.sleep(0.25)
    assert not dev_checks.pid_alive(pid)


# --- review round 2 regressions -------------------------------------------------------------------------------
def test_empty_shard_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(dev_checks, "collect", lambda marker: ["tests/test_a.py::test_only"])
    with pytest.raises(SystemExit):
        dev_checks.main(["--tier", "full", "--shard", "2/2", "--workdir", str(tmp_path / "w")])


def test_failures_file_cannot_overwrite_the_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr(dev_checks, "collect", lambda marker: ["tests/test_a.py::test_only"])
    baseline = tmp_path / "known.txt"
    baseline.write_text("tests/test_a.py::test_only\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        dev_checks.main(["--tier", "full", "--workdir", str(tmp_path / "w"),
                         "--expect-failures", str(baseline), "--write-failures", str(baseline)])
    assert baseline.read_text(encoding="utf-8") == "tests/test_a.py::test_only\n"


def test_a_child_holding_the_output_pipe_does_not_hold_up_a_finished_job(tmp_path):
    code = ("import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']);"
            "print('done')")
    import time
    started = time.time()
    status, out = dev_checks.run_with_timeout([sys.executable, "-c", code], cwd=tmp_path,
                                              env=dict(__import__("os").environ), timeout=20)
    assert status == 0 and "done" in out and time.time() - started < 15   # Windows: the leftover child is not killed (known limit)
