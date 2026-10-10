# SPDX-License-Identifier: MPL-2.0
"""Refusal, patch-validation, offline-input-identity and evidence tests for
build_e1_backend.

No WSL, no network, no compiler: run_wsl/subprocess are mocked and all WSL
output is synthetic. The real isolated build runs separately, once per phase,
through the workspace accounting runner. Real archived inputs are only read,
never written.
"""
import base64
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                       "tools" / "native_precision"))

import build_e1_backend as beb

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = (REPO / ".development-evidence" / "w5-isolated-backend-20261010" /
            "upstream-release" / "capytaine-3.0.0")
BASELINE_E1 = (UPSTREAM / beb.E1_REL).read_bytes()
ADOPTED = (REPO / ".development-evidence" / "w5-precision-candidate-20261010"
           / "candidate" / "Constants.f90")


def fake_proc(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


@pytest.fixture()
def owned_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(beb, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(beb, "SCRIPTS", tmp_path / "scripts")
    monkeypatch.setattr(beb, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(beb, "CANDIDATE_DIR", tmp_path / "candidate")
    monkeypatch.setattr(beb, "ROOT_RECEIPTS", tmp_path / "receipts")
    return tmp_path


class TestE1Patch:
    def test_real_baseline_patches_exactly_four_lines(self):
        patched, changed = beb.apply_e1_patch(BASELINE_E1)
        assert changed == list(beb.E1_CHANGED_LINES)
        check = beb.verify_e1_patch(BASELINE_E1, patched)
        assert check["changed_lines"] == [107, 111, 113, 149]
        assert check["patched_e1_sha256"] == beb.sha256_bytes(patched)
        lines = patched.decode().split("\n")
        assert lines[110] == beb.E1_NEW_DECL
        assert lines[112] == beb.E1_NEW_ASSIGN
        assert lines[148] == beb.E1_NEW_RETURN
        assert "kind(1e0)" not in "".join(
            lines[103:151])

    def test_wrong_parent_line_rejected(self):
        wrong = BASELINE_E1.replace(
            beb.E1_OLD_DECL.encode(),
            b"      complex(kind=kind(1e0))       :: g, y, z")
        with pytest.raises(beb.BuildError, match="line 111"):
            beb.apply_e1_patch(wrong)

    def test_extra_diff_rejected(self):
        patched, _ = beb.apply_e1_patch(BASELINE_E1)
        tampered = patched.replace(b"integrals(1) = integrals(1)/PI",
                                   b"integrals(1) = integrals(1)/PI ")
        with pytest.raises(beb.BuildError, match="exactly lines"):
            beb.verify_e1_patch(BASELINE_E1, tampered)

    def test_line_count_change_rejected(self):
        patched, _ = beb.apply_e1_patch(BASELINE_E1)
        with pytest.raises(beb.BuildError, match="line count"):
            beb.verify_e1_patch(BASELINE_E1, patched + b"extra\n")

    def test_coefficient_literals_preserved(self):
        patched, _ = beb.apply_e1_patch(BASELINE_E1)
        for literal in (b"0.711093", b"0.415775", b"0.278518", b"2.29428",
                        b"0.010389", b"6.2900", b"0.5772156649e+00",
                        b"0.23721365e+00", b"-3.777369, -4.510900",
                        b"1.3935496e-06"):
            assert literal in patched
        for branch in (b"real(z) < -16.0", b"abs(aimag(z)) > 10.0",
                       b"real(z) > -0.5", b"aimag(z) < 0"):
            assert branch in patched


class TestAdoptedConstants:
    def test_real_adopted_constants_verify(self):
        check = beb.verify_adopted_constants()
        assert check["changed_lines"] == [26, 28]
        assert check["candidate_constants_sha256"] == \
            beb.EXPECTED_CANDIDATE_CONSTANTS_SHA256

    def test_wrong_hash_rejected(self, monkeypatch):
        monkeypatch.setattr(beb, "EXPECTED_CANDIDATE_CONSTANTS_SHA256",
                            "0" * 64)
        with pytest.raises(beb.BuildError, match="SHA mismatch"):
            beb.verify_adopted_constants()

    def test_extra_constants_diff_rejected(self, tmp_path, monkeypatch):
        cand = tmp_path / "Constants.f90"
        lines = ADOPTED.read_text(encoding="utf-8").split("\n")
        lines[29] = lines[29] + " ! extra in-place change"
        cand.write_text("\n".join(lines), encoding="utf-8")
        monkeypatch.setattr(beb, "ADOPTED_CONSTANTS", cand)
        monkeypatch.setattr(beb, "EXPECTED_CANDIDATE_CONSTANTS_SHA256",
                            beb.sha256_file(cand))
        with pytest.raises(beb.BuildError, match="26/28"):
            beb.verify_adopted_constants()


class TestBuildRootSafety:
    def test_windows_mount_root_rejected(self, monkeypatch):
        monkeypatch.setattr(beb, "BUILD_ROOT",
                            "/mnt/c/var/tmp/anyhydro-e1-kind")
        with pytest.raises(beb.BuildError, match="Windows mount"):
            beb.check_build_root_safety()

    def test_root_overlapping_immutable_rejected(self, monkeypatch):
        monkeypatch.setattr(beb, "BUILD_ROOT",
                            "/mnt/c/Github/ANYhydro/sub")
        with pytest.raises(beb.BuildError, match="must not be on a "
                            "Windows mount|overlaps immutable"):
            beb.check_build_root_safety()

    def test_assigned_root_is_safe(self):
        beb.check_build_root_safety()


class TestReceiptSemantics:
    def test_fresh_receipt_refusal(self, owned_dirs):
        owned_dirs.joinpath("results").mkdir()
        (owned_dirs / "results" / "stage-a1-receipt.json").write_text("{}")
        with pytest.raises(beb.BuildError, match="refusing to overwrite"):
            beb.require_fresh_receipt("stage", "a1")

    def test_prerequisite_requires_ok_receipt(self, owned_dirs):
        owned_dirs.joinpath("results").mkdir()
        (owned_dirs / "results" / "build-b1-receipt.json").write_text(
            json.dumps({"ok": False}))
        with pytest.raises(beb.BuildError, match="no successful build"):
            beb.require_prerequisite("install", "build")

    def test_failed_child_evidence_preserved(self, owned_dirs):
        owned_dirs.joinpath("results").mkdir()
        receipt = {"gnu_timeout_seconds": 180, "build_root": beb.BUILD_ROOT}
        out = beb.finish_receipt(receipt, "build", "fail1",
                                 ["wsl.exe", "--"], fake_proc(
                                     returncode=124, stdout="partial out",
                                     stderr="killed"), 180.4,
                                 "WSL child returned 124")
        assert out["ok"] is False
        assert out["returncode"] == 124
        assert out["stdout"] == "partial out"
        assert out["stderr"] == "killed"
        assert out["validation_error"] == "WSL child returned 124"
        saved = json.loads((owned_dirs / "results" /
                            "build-fail1-receipt.json").read_text())
        assert saved["ok"] is False
        assert saved["process_wall_seconds"] == 180.4

    def test_save_text_refuses_differing_overwrite(self, owned_dirs):
        p = owned_dirs / "evidence.txt"
        beb.save_text(p, "first")
        beb.save_text(p, "first")
        with pytest.raises(beb.BuildError, match="refusing to overwrite"):
            beb.save_text(p, "second")


class TestOfflineInputs:
    def test_archived_build_wheels_verify(self):
        verified = beb.verify_wheel_sums(beb.DEP_WHEELS_SUMS,
                                         beb.DEP_WHEELS_DIR)
        assert len(verified) == 8
        assert beb.verify_lock_names(beb.BUILD_LOCK, verified) == {
            "pip": "26.2.1", "charset-normalizer": "3.5.2", "meson": "1.12.1",
            "meson-python": "0.22.1", "ninja": "1.13.2", "numpy": "2.5.3",
            "packaging": "26.3", "pyproject-metadata": "0.12.1"}

    def test_runtime_wheels_verify_excluding_baseline(self):
        verified = beb.verify_wheel_sums(beb.RUNTIME_WHEELS_SUMS,
                                         beb.RUNTIME_WHEELS_DIR,
                                         exclude="capytaine-3.0.0")
        assert len(verified) == 15
        assert not any(name.startswith("capytaine")
                       for name in verified)

    def test_tampered_wheel_rejected(self, tmp_path):
        wheel = tmp_path / "numpy-1.0-py3-none-any.whl"
        wheel.write_bytes(b"tampered bytes")
        sums = tmp_path / "SHA256SUMS"
        sums.write_text(f"{'0' * 64}  {tmp_path}/numpy-1.0-py3-none-any.whl\n")
        with pytest.raises(beb.BuildError, match="SHA mismatch"):
            beb.verify_wheel_sums(sums, tmp_path)

    def test_missing_wheel_rejected(self, tmp_path):
        sums = tmp_path / "SHA256SUMS"
        sums.write_text(f"{'0' * 64}  {tmp_path}/absent.whl\n")
        with pytest.raises(beb.BuildError, match="archived wheel missing"):
            beb.verify_wheel_sums(sums, tmp_path)

    def test_lock_requirement_not_in_wheels_rejected(self, tmp_path):
        verified = {"numpy-2.5.3-x.whl": "0" * 64}
        lock = tmp_path / "lock.txt"
        lock.write_text("numpy==2.5.3\nscipy==1.18.1\n")
        with pytest.raises(beb.BuildError, match="scipy"):
            beb.verify_lock_names(lock, verified)

    def test_bootstrap_pip_identity(self):
        assert beb.sha256_file(
            beb.BOOTSTRAP_PIP_WHEEL) == beb.EXPECTED_BOOTSTRAP_PIP_SHA256


class TestVersionShape:
    def test_actual_about_shape_assign(self):
        about = ('__title__ = "capytaine"\n\n__version__ = "3.0.0"\n\n'
                 '__build_info__ = {"compiler": "gcc 13.3.0"}\n')
        assert beb.validate_built_version(about) == "3.0.0"

    def test_annotated_assignment_shape(self):
        assert beb.parse_version_source(
            '__version__: str = "3.0.0"\n') == "3.0.0"

    def test_wrong_version_rejected(self):
        with pytest.raises(beb.BuildError, match="not '3.0.0'"):
            beb.validate_built_version('__version__ = "3.0.1"\n')

    def test_multiple_assignments_rejected(self):
        with pytest.raises(beb.BuildError, match="exactly one"):
            beb.parse_version_source(
                '__version__ = "3.0.0"\n__version__ = "3.0.0"\n')

    def test_missing_version_rejected(self):
        with pytest.raises(beb.BuildError, match="exactly one"):
            beb.parse_version_source('__title__ = "capytaine"\n')


class TestWslTransport:
    def test_base64_transport_and_bounds(self):
        script = 'echo "$ROOT"\nROOT=/var/tmp/anyhydro-e1-kind-backend-20261010\n'
        with mock.patch.object(beb.subprocess, "run") as run:
            run.return_value = fake_proc(0, "ok", "")
            cmd, proc = beb.run_wsl("Ubuntu-24.04", script, 180)
        assert cmd[:5] == ["wsl.exe", "-d", "Ubuntu-24.04", "--", "timeout"]
        assert "--kill-after=3s" in cmd and "180s" in cmd
        assert cmd[-3] == "bash" and cmd[-2] == "-c"
        inner = cmd[-1]
        assert inner.startswith("$(echo ") and inner.endswith(" | base64 -d)")
        b64 = inner[len("$(echo "):-len(" | base64 -d)")]
        assert base64.b64decode(b64).decode() == script
        assert "$ROOT" not in inner and "anyhydro-e1-kind" not in inner

    def test_gnu_timeout_bounds_enforced(self):
        with pytest.raises(beb.BuildError, match="180"):
            beb.main(["build", "--attempt", "x", "--gnu-timeout", "181"])
        with pytest.raises(beb.BuildError, match="180"):
            beb.main(["build", "--attempt", "x", "--gnu-timeout", "0"])

    def test_windows_local_phase_rejects_gnu_timeout(self):
        with pytest.raises(beb.BuildError, match="Windows-local"):
            beb.main(["patch", "--attempt", "x", "--gnu-timeout", "5"])

    def test_invalid_attempt_label_rejected(self):
        with pytest.raises(beb.BuildError, match="invalid attempt"):
            beb.main(["build", "--attempt", "bad attempt/1"])


class TestPhasePatchRefusals:
    def test_refuses_existing_candidate_output(self, owned_dirs, monkeypatch):
        cand = owned_dirs / "candidate"
        cand.mkdir()
        (cand / "Delhommeau_integrals.f90").write_bytes(b"stale")
        args = SimpleNamespace(attempt="r1", gnu_timeout=0)
        with pytest.raises(beb.BuildError,
                           match="refusing to overwrite candidate"):
            beb.phase_patch(args)

    def test_refuses_existing_receipt(self, owned_dirs):
        owned_dirs.joinpath("results").mkdir()
        (owned_dirs / "results" / "patch-p1-receipt.json").write_text("{}")
        args = SimpleNamespace(attempt="p1", gnu_timeout=0)
        with pytest.raises(beb.BuildError, match="refusing to overwrite"):
            beb.phase_patch(args)


WHEEL_NAME = "capytaine-3.0.0-cp312-cp312-linux_x86_64.whl"
NATIVE_F64 = ("Delhommeau_float64.cpython-312-x86_64-linux-gnu.so")
NATIVE_F32 = ("Delhommeau_float32.cpython-312-x86_64-linux-gnu.so")


def write_ok_receipt(phase, attempt, payload):
    path = beb.RESULTS / f"{phase}-{attempt}-receipt.json"
    path.write_text(json.dumps({"ok": True, **payload}), encoding="utf-8")
    return path


def write_full_ok_receipts(wheel_sha):
    """Synthetic ok receipts covering every field manifest/sidecar bind."""
    write_ok_receipt("patch", "p1", {
        "e1_verification": {"changed_lines": [107, 111, 113, 149],
                            "patched_e1_sha256": "f" * 64}})
    write_ok_receipt("stage", "s1", {
        "upstream_verification": {"verified_files": 262},
        "e1_verification": {"patched_e1_sha256": "f" * 64},
        "candidate_tree_diff_lines": [
            "libs/libDelhommeau/src/Delhommeau_integrals.f90",
            "libs/libDelhommeau/src/Delhommeau/constants.f90"]})
    write_ok_receipt("deps", "d1", {"dependency_acquisition": {
        "numpy": "2.5.3", "meson": "1.12.1"}})
    write_ok_receipt("build", "b1", {
        "wheel_name": WHEEL_NAME, "wheel_sha256": wheel_sha,
        "build_route": "pip wheel meson-python offline",
        "compiler": "gfortran 13.3.0",
        "ninja_args_proven": ["ninja", "-j1"],
        "wheel_record_verified": True,
        "built_about_py": '__version__ = "3.0.0"'})
    write_ok_receipt("install", "i1", {
        "isolated_install_env": "/var/tmp/anyhydro-e1-kind-backend-20261010"
                                "/install-env",
        "runtime_inputs": {"numpy": "2.5.3"},
        "runtime_identity": {
            "executable": "/install-env/bin/python",
            "executable_sha256": "e" * 64,
            "python": "3.12.3", "platform": "linux",
            "package_version": "3.0.0", "packages": {"numpy": "2.5.3"},
            "pre_kind": 8, "pi": 3.141592653589793,
            "euler_gamma": 0.5772156649015329,
            "pools": [{"user_api": "blas", "num_threads": 1}],
            "thread_env": {"OMP_NUM_THREADS": "1"},
            "installed_bytes_verified_against_wheel_record": 68,
            "native_modules": {
                NATIVE_F32: {"sha256": "3" * 64, "linked_libraries": []},
                NATIVE_F64: {"sha256": "6" * 64, "linked_libraries": []}},
            "classification": "metadata only"}})
    write_ok_receipt("collect", "c1", {"artifacts_dir": str(beb.ARTIFACTS)})


class TestCollectBoundaries:
    def test_collect_requires_all_prerequisite_ok_receipts(self, owned_dirs):
        owned_dirs.joinpath("results").mkdir()
        write_ok_receipt("patch", "p1", {})
        args = SimpleNamespace(attempt="c1", gnu_timeout=60, wsl_distro="Ubuntu-24.04")
        with pytest.raises(beb.BuildError,
                           match="no successful stage receipt"):
            beb.phase_collect(args)

    def test_collect_refusal_evidence_preserved(self, owned_dirs,
                                                monkeypatch):
        owned_dirs.joinpath("results").mkdir()
        for phase in ("patch", "stage", "deps", "build", "install"):
            write_ok_receipt(phase, "x1", {})
        captured = {}

        def refuse(distro, script, timeout):
            captured["script"] = script
            return (["wsl.exe", "--"], fake_proc(
                3, "REFUSED: build-root receipts already collected", ""))
        monkeypatch.setattr(beb, "run_wsl", refuse)
        args = SimpleNamespace(attempt="dup1", gnu_timeout=60, wsl_distro="Ubuntu-24.04")
        assert beb.phase_collect(args) == 1
        saved = json.loads((owned_dirs / "results" /
                            "collect-dup1-receipt.json").read_text())
        assert saved["ok"] is False
        assert saved["returncode"] == 3
        assert saved["validation_error"] == "WSL child returned 3"
        assert "REFUSED: build-root receipts already collected" \
            in saved["stdout"]
        assert "build-root-receipts" in captured["script"]
        assert "REFUSED: build-root receipts already collected" \
            in captured["script"]

    def test_collect_success_receipt(self, owned_dirs, monkeypatch):
        owned_dirs.joinpath("results").mkdir()
        for phase in ("patch", "stage", "deps", "build", "install"):
            write_ok_receipt(phase, "x1", {})
        monkeypatch.setattr(beb, "run_wsl", lambda d, s, t: (
            ["wsl.exe", "--"], fake_proc(0, "COLLECT_DONE", "")))
        args = SimpleNamespace(attempt="ok1", gnu_timeout=60, wsl_distro="Ubuntu-24.04")
        assert beb.phase_collect(args) == 0
        saved = json.loads((owned_dirs / "results" /
                            "collect-ok1-receipt.json").read_text())
        assert saved["ok"] is True
        assert saved["validation_error"] is None
        assert saved["artifacts_dir"] == str(owned_dirs / "artifacts")
        assert beb.SCRIPTS.joinpath("collect-ok1.sh").is_file()


class TestManifestBoundaries:
    def test_manifest_requires_ok_receipts(self, owned_dirs):
        owned_dirs.joinpath("results").mkdir()
        for phase in ("patch", "stage", "deps", "build", "install"):
            write_ok_receipt(phase, "x1", {})
        args = SimpleNamespace(attempt="m1", gnu_timeout=0)
        with pytest.raises(beb.BuildError,
                           match="no successful collect receipt"):
            beb.phase_manifest(args)

    def test_manifest_refuses_missing_and_mismatched_wheel(self, owned_dirs):
        owned_dirs.joinpath("results").mkdir()
        write_full_ok_receipts("a" * 64)
        args = SimpleNamespace(attempt="m1", gnu_timeout=0)
        with pytest.raises(beb.BuildError, match="collected wheel missing"):
            beb.phase_manifest(args)
        beb.ARTIFACTS.mkdir(parents=True)
        (beb.ARTIFACTS / WHEEL_NAME).write_bytes(b"not the built wheel")
        with pytest.raises(beb.BuildError, match="SHA mismatch"):
            beb.phase_manifest(args)

    def test_manifest_sidecar_identity_then_overwrite_refusal(self, owned_dirs):
        owned_dirs.joinpath("results").mkdir()
        wheel_bytes = b"synthetic collected wheel bytes"
        wheel_sha = beb.sha256_bytes(wheel_bytes)
        write_full_ok_receipts(wheel_sha)
        beb.ARTIFACTS.mkdir(parents=True)
        (beb.ARTIFACTS / WHEEL_NAME).write_bytes(wheel_bytes)
        runtime_copy = (beb.ROOT_RECEIPTS / "build-root-receipts" /
                        "runtime-identity.json")
        runtime_copy.parent.mkdir(parents=True)
        runtime_copy.write_text('{"pre_kind": 8}', encoding="utf-8")
        args = SimpleNamespace(attempt="m1", gnu_timeout=0)
        assert beb.phase_manifest(args) == 0
        manifest = json.loads((owned_dirs / "results" /
                              "build-manifest.json").read_text())
        assert manifest["format"] == "anyhydro-e1-kind-backend-manifest-v1"
        assert manifest["external_identity"] == beb.EXTERNAL_IDENTITY
        assert manifest["build"]["wheel_sha256"] == wheel_sha
        assert manifest["runtime"]["pre_kind"] == 8
        assert manifest["runtime"]["native_modules"][NATIVE_F64][
            "sha256"] == "6" * 64
        sidecar = json.loads((beb.ARTIFACTS /
                              f"{WHEEL_NAME}.anyhydro-backend.json")
                             .read_text())
        assert sidecar["format"] == "anyhydro-backend-artifact-identity-v1"
        assert sidecar["wheel_sha256"] == wheel_sha
        assert sidecar["admitted_for_numerical_execution"] is False
        assert sidecar["native_extension_sha256"][
            f"capytaine/green_functions/{NATIVE_F64}"] == "6" * 64
        assert sidecar["build_manifest_sha256"] == beb.sha256_file(
            owned_dirs / "results" / "build-manifest.json")
        assert sidecar["runtime_identity_sha256"] == beb.sha256_file(
            runtime_copy)
        assert sidecar["candidate_source_tree_sha256"] == \
            beb.candidate_tree_hash()
        with pytest.raises(beb.BuildError,
                           match="refusing to overwrite existing manifest"):
            beb.phase_manifest(SimpleNamespace(attempt="m2", gnu_timeout=0))
        (owned_dirs / "results" / "build-manifest.json").unlink()
        with pytest.raises(beb.BuildError,
                           match="refusing to overwrite existing sidecar"):
            beb.phase_manifest(SimpleNamespace(attempt="m3", gnu_timeout=0))
