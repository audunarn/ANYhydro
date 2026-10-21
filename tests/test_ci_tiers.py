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


# --- fix round: review findings F1-F4 ---------------------------------------------------------------------------
def _run_main_with_workdir(monkeypatch, workdir):
    monkeypatch.setattr(dev_checks, "collect", lambda marker: ["tests/test_a.py::test_only"])
    removed = []
    monkeypatch.setattr(dev_checks.shutil, "rmtree", lambda p, *a, **k: removed.append(Path(p).resolve()))
    try:
        dev_checks.main(["--tier", "full", "--workdir", str(workdir)])
    except BaseException:
        pass
    return removed


@pytest.mark.parametrize("which", ["root", "parent", "fsroot", "home"])
def test_workdir_never_deletes_repo_ancestors_or_home(monkeypatch, which):
    target = {"root": dev_checks.ROOT, "parent": dev_checks.ROOT.parent,
              "fsroot": Path(dev_checks.ROOT.anchor), "home": Path.home()}[which]
    assert _run_main_with_workdir(monkeypatch, target) == []


def test_workdir_refuses_a_foreign_non_empty_directory(monkeypatch, tmp_path):
    (tmp_path / "precious.txt").write_text("x")
    assert _run_main_with_workdir(monkeypatch, tmp_path) == []
    assert (tmp_path / "precious.txt").exists()


def test_workdir_own_directory_is_reused_and_fresh_is_created(tmp_path):
    work = tmp_path / "w"
    dev_checks.prepare_workdir(work)
    assert (work / "tmp").is_dir()
    (work / "leftover.log").write_text("old")
    dev_checks.prepare_workdir(work)          # own (marker present): cleared and recreated
    assert not (work / "leftover.log").exists() and (work / "tmp").is_dir()


def test_workdir_symlink_is_refused(tmp_path):
    target = tmp_path / "real"
    target.mkdir()
    (target / "keep.txt").write_text("x")
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    with pytest.raises(SystemExit):
        dev_checks.prepare_workdir(link)
    assert (target / "keep.txt").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_finished_job_leaves_no_background_child(tmp_path):
    pid_file = tmp_path / "child.pid"
    code = (f"import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']);"
            f"open(r'{pid_file}','w').write(str(p.pid))")
    status, _ = dev_checks.run_with_timeout([sys.executable, "-c", code], cwd=tmp_path, env=dict(__import__("os").environ), timeout=30)
    assert status == 0
    import time
    pid = int(pid_file.read_text())
    for _ in range(20):
        if not dev_checks.pid_alive(pid):
            break
        time.sleep(0.25)
    assert not dev_checks.pid_alive(pid)


def test_kill_tree_uses_the_pgid_captured_at_start(monkeypatch):
    if sys.platform == "win32":
        pytest.skip("POSIX only")
    import os, signal
    calls = []
    monkeypatch.setattr(os, "getpgid", lambda pid: (_ for _ in ()).throw(ProcessLookupError()))
    monkeypatch.setattr(os, "killpg", lambda pg, sig: calls.append((pg, sig)))
    class P:
        pid = 4242
        def kill(self): raise AssertionError("fallback kill used")
    dev_checks._kill_tree(P())
    assert calls == [(4242, signal.SIGKILL)]


def test_guard_rejects_empty_input_and_unmapped_results(tmp_path, capsys):
    assert ci_tiers.main(["guard"]) != 0
    assert ci_tiers.main(["guard", str(tmp_path)]) != 0          # a directory without JUnit files
    bad = tmp_path / "j.xml"
    bad.write_text('<testsuite><testcase classname="nowhere.nothing" name="t" time="1"/></testsuite>')
    assert ci_tiers.main(["guard", str(bad)]) != 0


def test_verify_accepts_an_empty_slow_tier_but_not_an_empty_full_tier(monkeypatch):
    ids = ["tests/test_a.py::test_x"]
    monkeypatch.setattr(ci_tiers, "load_table", lambda *a, **k: {"tests": {}})
    def fake(marker, allow_empty=False):
        if marker == "slow":
            if not allow_empty:
                raise SystemExit("empty")
            return []
        return ids
    monkeypatch.setattr(dev_checks, "collect", fake)
    assert dev_checks.verify() == 0
    monkeypatch.setattr(dev_checks, "collect", lambda marker, allow_empty=False: (_ for _ in ()).throw(SystemExit("empty")) if marker is None else ids)
    with pytest.raises(SystemExit):
        dev_checks.verify()
