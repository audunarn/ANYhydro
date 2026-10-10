# SPDX-License-Identifier: MPL-2.0
"""Refusal, verification and script-generation tests for build_isolated_backend.

No WSL, no network, no compiler: run_wsl/subprocess are mocked and all WSL
output is synthetic. The real isolated build runs separately, once per phase,
through check.py.
"""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_isolated_backend as bib

REPO = Path(__file__).resolve().parents[3]
ACQ = (REPO / ".development-evidence" / "w5-isolated-backend-20261010" /
       "upstream-release" / "source-acquisition.json")


def fake_proc(returncode=0, stdout="", stderr=""):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


class OwnedDirs:
    """Redirect RESULTS/SCRIPTS/ARTIFACTS into a temp dir for one test."""

    def __init__(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.results = self.tmp / "results"
        self.scripts = self.tmp / "scripts"
        self.artifacts = self.tmp / "artifacts"
        for d in (self.results, self.scripts, self.artifacts):
            d.mkdir()
        self.patches = [
            mock.patch.object(bib, "RESULTS", self.results),
            mock.patch.object(bib, "SCRIPTS", self.scripts),
            mock.patch.object(bib, "ARTIFACTS", self.artifacts),
        ]

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


def ok_receipt(dir_, phase, attempt="1", **extra):
    p = dir_ / f"{phase}-{attempt}-receipt.json"
    p.write_text(json.dumps({"ok": True, "attempt": attempt, **extra}),
                 encoding="utf-8")
    return p


class VerificationTests(unittest.TestCase):
    def test_verify_upstream_tree_matches_official_archive(self):
        info = bib.verify_upstream_tree()
        expected = len(json.loads(ACQ.read_text(encoding="utf-8"))
                       ["file_sha256"])
        self.assertEqual(info["verified_files"], expected)
        self.assertEqual(info["archive_sha256"],
                         bib.EXPECTED_ARCHIVE_SHA256)

    def test_verify_upstream_tree_refuses_wrong_archive_sha(self):
        with mock.patch.object(bib, "EXPECTED_ARCHIVE_SHA256", "0" * 64):
            with self.assertRaises(bib.BuildError):
                bib.verify_upstream_tree()

    def test_verify_candidate_two_line_matches(self):
        info = bib.verify_candidate_two_line()
        self.assertEqual(info["changed_lines"], [26, 28])
        self.assertIn("3.141592653589793238462643383279502884_PRE",
                      info["pi_line"])
        self.assertIn("0.577215664901532860606512090082402431_PRE",
                      info["euler_gamma_line"])

    def test_verify_candidate_refuses_third_line_change(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            base = (bib.UPSTREAM_TREE / bib.CONSTANTS_REL)
            cand = tmp / "Constants.f90"
            lines = bib.CANDIDATE_CONSTANTS.read_text(
                encoding="utf-8").splitlines()
            lines[10] = lines[10] + " ! extra edit"
            cand.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with mock.patch.object(bib, "CANDIDATE_CONSTANTS", cand), \
                 mock.patch.object(
                     bib, "EXPECTED_CANDIDATE_CONSTANTS_SHA256",
                     bib.sha256_file(cand)):
                with self.assertRaises(bib.BuildError):
                    bib.verify_candidate_two_line()


class RefusalTests(unittest.TestCase):
    def test_duplicate_attempt_receipt_refused(self):
        with OwnedDirs() as own:
            ok_receipt(own.results, "stage", "dup")
            with self.assertRaises(bib.BuildError):
                bib.require_fresh_receipt("stage", "dup")

    def test_missing_prerequisite_refused(self):
        with OwnedDirs() as own:
            with self.assertRaises(bib.BuildError):
                bib.require_prerequisite("build", "deps")

    def test_failed_prerequisite_not_accepted(self):
        with OwnedDirs() as own:
            p = own.results / "deps-1-receipt.json"
            p.write_text(json.dumps({"ok": False}), encoding="utf-8")
            self.assertIsNone(bib.find_ok_receipt("deps"))

    def test_gnu_timeout_bound_refused(self):
        with self.assertRaises(bib.BuildError):
            bib.main(["build", "--attempt", "x", "--gnu-timeout", "286"])
        with self.assertRaises(bib.BuildError):
            bib.main(["build", "--attempt", "x", "--gnu-timeout", "0"])

    def test_invalid_attempt_label_refused(self):
        with self.assertRaises(bib.BuildError):
            bib.main(["build", "--attempt", "../escape"])

    def test_save_script_identical_reuse_and_differing_refusal(self):
        with OwnedDirs() as own:
            p = bib.save_script("x-1.sh", "echo hi\n")
            self.assertTrue(p.exists())
            self.assertEqual(bib.save_script("x-1.sh", "echo hi\n"), p)
            with self.assertRaises(bib.BuildError):
                bib.save_script("x-1.sh", "echo changed\n")

    def test_build_root_safety_accepts_owned_tmp_root(self):
        bib.check_build_root_safety()

    def test_build_root_on_windows_mount_refused(self):
        with mock.patch.object(bib, "BUILD_ROOT", "/mnt/c/Github/ANYhydro/x"):
            with self.assertRaises(bib.BuildError):
                bib.check_build_root_safety()

    def test_build_root_inside_repo_refused(self):
        with mock.patch.object(bib, "BUILD_ROOT",
                               bib.to_wsl_path(bib.REPO) + "/sub"):
            with self.assertRaises(bib.BuildError):
                bib.check_build_root_safety()


class CommandTests(unittest.TestCase):
    def test_run_wsl_argv_shape(self):
        with mock.patch.object(bib.subprocess, "run") as run:
            run.return_value = fake_proc()
            cmd, _ = bib.run_wsl("Ubuntu-24.04", "echo hi", 285)
        self.assertEqual(cmd[:9], ["wsl.exe", "-d", "Ubuntu-24.04", "--",
                                   "timeout", "--kill-after=3s", "285s",
                                   "bash", "-c"])
        inner = cmd[9]
        self.assertTrue(inner.startswith("$(echo "), inner)
        self.assertIn(" | base64 -d)", inner)
        import base64 as b64mod
        decoded = b64mod.b64decode(inner[len("$(echo "):
                                         -len(" | base64 -d)")])
        self.assertEqual(decoded.decode("utf-8"), "echo hi")
        run.assert_called_once()

    def test_stage_script_content(self):
        with OwnedDirs() as own:
            ok = None
            captured = {}

            def fake_run(distro, script, timeout):
                captured["script"] = script
                n = len(json.loads(ACQ.read_text(encoding="utf-8"))
                        ["file_sha256"])
                return (["wsl.exe"], fake_proc(stdout=(
                    "STAGE_UPSTREAM_VERIFY\n"
                    "Files $ROOT/upstream/capytaine-3.0.0/"
                    "libs/libDelhommeau/src/constants.f90 and "
                    "$ROOT/candidate-src/capytaine-3.0.0/"
                    "libs/libDelhommeau/src/constants.f90 differ\n"
                    f"CANDIDATE_FILES={n}\n"
                    "STAGE_DONE\n")))
            with mock.patch.object(bib, "run_wsl", fake_run):
                rc = bib.main(["stage", "--attempt", "s1"])
            self.assertEqual(rc, 0)
            s = captured["script"]
            self.assertIn("set -euo pipefail", s)
            self.assertIn("export OMP_NUM_THREADS=1", s)
            self.assertIn("export NUMEXPR_NUM_THREADS=1", s)
            self.assertIn("REFUSED: build root exists", s)
            self.assertIn("sha256sum -c", s)
            self.assertIn("diff -qr", s)
            receipt = json.loads((own.results / "stage-s1-receipt.json")
                                 .read_text(encoding="utf-8"))
            self.assertTrue(receipt["ok"])
            self.assertEqual(receipt["external_identity"],
                             bib.EXTERNAL_IDENTITY)

    def test_stage_refuses_wrong_diff_count(self):
        with OwnedDirs():
            def fake_run(distro, script, timeout):
                return (["wsl.exe"], fake_proc(stdout=(
                    "STAGE_UPSTREAM_VERIFY\n"
                    "STAGE_DONE\n")))
            with mock.patch.object(bib, "run_wsl", fake_run):
                rc = bib.main(["stage", "--attempt", "s2"])
            self.assertEqual(rc, 1)

    def test_build_script_content_and_receipt(self):
        with OwnedDirs() as own:
            ok_receipt(own.results, "stage", "1")
            ok_receipt(own.results, "deps", "1")
            captured = {}

            def fake_run(distro, script, timeout):
                captured["script"] = script
                return (["wsl.exe"], fake_proc(stdout=(
                    "EFFECTIVE_ENV\nOMP_NUM_THREADS=1\n"
                    "EFFECTIVE_TOOLS\n"
                    "MESONPY_COMPILE_ARGS_MECHANISM\n"
                    "PIP_WHEEL_START\nPIP_WHEEL_DONE\n"
                    "capytaine-3.0.0-cp312-cp312-linux_x86_64.whl\n"
                    f"{'a' * 64}  /dist/capytaine-3.0.0-cp312-cp312-"
                    "linux_x86_64.whl\n"
                    "BUILD_DONE\n")))
            (own.results / "build-wheel-contents-b1.json").write_text(
                json.dumps({"about_py": '__version__ = "3.0.0"\n'}),
                encoding="utf-8")
            (own.results / "build-ninja-args-b1.log").write_text(
                "ninja_argv: -j1\n", encoding="utf-8")
            with mock.patch.object(bib, "run_wsl", fake_run):
                rc = bib.main(["build", "--attempt", "b1"])
            self.assertEqual(rc, 0)
            s = captured["script"]
            self.assertIn("--config-settings=compile-args=-j1", s)
            self.assertIn("--no-build-isolation", s)
            self.assertIn("grep -rn", s)
            self.assertIn("ninja-wrap", s)
            self.assertIn("REFUSED: dist dir not empty", s)
            receipt = json.loads((own.results / "build-b1-receipt.json")
                                 .read_text(encoding="utf-8"))
            self.assertTrue(receipt["ok"])
            self.assertEqual(receipt["wheel_sha256"], "a" * 64)
            self.assertEqual(receipt["ninja_args_proven"],
                             ["ninja_argv: -j1"])

    def test_build_refuses_wrong_wheel_version(self):
        with OwnedDirs() as own:
            ok_receipt(own.results, "stage", "1")
            ok_receipt(own.results, "deps", "1")

            def fake_run(distro, script, timeout):
                return (["wsl.exe"], fake_proc(stdout=(
                    f"{'a' * 64}  /dist/capytaine-3.1.0-cp312-"
                    "linux_x86_64.whl\nBUILD_DONE\n")))
            with mock.patch.object(bib, "run_wsl", fake_run):
                rc = bib.main(["build", "--attempt", "b2"])
            self.assertEqual(rc, 1)

    def test_install_receipt_validates_native_constants(self):
        with OwnedDirs() as own:
            for phase in ("stage", "deps", "build"):
                ok_receipt(own.results, phase, "1")
            native = {
                "loader": "importlib",
                "analysis": {
                    "pre_kind": 8,
                    "pi": math.pi,
                    "euler_gamma":
                        0.577215664901532860606512090082402431,
                    "minus_one_over_fourpi": -1.0 / (4 * math.pi),
                    "log_2": math.log(2.0),
                },
                "modules": {
                    "Delhommeau_float64.cpython-312-x86_64-linux-gnu.so": {
                        "sha256": "b" * 64,
                        "constants": {"pi": math.pi},
                        "floating_point_precision": {"pre": 8}},
                    "Delhommeau_float32.cpython-312-x86_64-linux-gnu.so": {
                        "sha256": "c" * 64},
                },
            }
            (own.results / "install-native-constants-i1.json").write_text(
                json.dumps(native), encoding="utf-8")

            def fake_run(distro, script, timeout):
                return (["wsl.exe"], fake_proc(stdout="INSTALL_DONE\n"))
            with mock.patch.object(bib, "run_wsl", fake_run):
                rc = bib.main(["install", "--attempt", "i1"])
            self.assertEqual(rc, 0)
            receipt = json.loads((own.results / "install-i1-receipt.json")
                                 .read_text(encoding="utf-8"))
            self.assertTrue(receipt["ok"])
            self.assertEqual(receipt["native_constants"]["analysis"]["pi"],
                             math.pi)
            self.assertEqual(receipt["extension_sha256"][
                "Delhommeau_float64.cpython-312-x86_64-linux-gnu.so"],
                "b" * 64)

    def test_install_refuses_missing_float32(self):
        with OwnedDirs() as own:
            for phase in ("stage", "deps", "build"):
                ok_receipt(own.results, phase, "1")
            native = {
                "analysis": {
                    "pre_kind": 8, "pi": math.pi,
                    "euler_gamma":
                        0.577215664901532860606512090082402431,
                    "minus_one_over_fourpi": -1.0 / (4 * math.pi),
                    "log_2": math.log(2.0)},
                "modules": {"Delhommeau_float64.so": {"sha256": "b" * 64}},
            }
            (own.results / "install-native-constants-i3.json").write_text(
                json.dumps(native), encoding="utf-8")

            def fake_run(distro, script, timeout):
                return (["wsl.exe"], fake_proc(stdout="INSTALL_DONE\n"))
            with mock.patch.object(bib, "run_wsl", fake_run):
                rc = bib.main(["install", "--attempt", "i3"])
            self.assertEqual(rc, 1)

    def test_install_refuses_wrong_native_pi(self):
        with OwnedDirs() as own:
            for phase in ("stage", "deps", "build"):
                ok_receipt(own.results, phase, "1")
            native = {"analysis": {"pre_kind": 8, "pi": 3.1415927410125732,
                                   "euler_gamma": 0.5772156649}}
            (own.results / "install-native-constants-i2.json").write_text(
                json.dumps(native), encoding="utf-8")

            def fake_run(distro, script, timeout):
                return (["wsl.exe"], fake_proc(stdout="INSTALL_DONE\n"))
            with mock.patch.object(bib, "run_wsl", fake_run):
                rc = bib.main(["install", "--attempt", "i2"])
            self.assertEqual(rc, 1)

    def test_deps_receipt_reads_toolchain_file(self):
        with OwnedDirs() as own:
            ok_receipt(own.results, "stage", "1")
            (own.results / "deps-toolchain-d1.json").write_text(
                json.dumps({"numpy_version": "2.1.3",
                            "meson_python_version": "0.17.0"}),
                encoding="utf-8")
            (own.results / "deps-wheels-d1.sha256").write_text(
                f"{'a' * 64}  /wheels/numpy-2.1.3-cp312-cp312-"
                "manylinux_2_17_x86_64.whl\n"
                f"{'b' * 64}  /wheels/meson_python-0.17.0-py3-none-"
                "any.whl\n", encoding="utf-8")

            def fake_run(distro, script, timeout):
                return (["wsl.exe"], fake_proc(stdout="DEPS_DONE\n"))
            with mock.patch.object(bib, "run_wsl", fake_run):
                rc = bib.main(["deps", "--attempt", "d1"])
            self.assertEqual(rc, 0)
            receipt = json.loads((own.results / "deps-d1-receipt.json")
                                 .read_text(encoding="utf-8"))
            self.assertEqual(receipt["toolchain"]["numpy_version"], "2.1.3")
            self.assertIn("numpy-2.1.3-cp312-cp312-manylinux_2_17_x86_64.whl",
                          receipt["dependency_acquisition"]
                          ["downloaded_wheels_sha256"])
            self.assertTrue((own.scripts / "deps-d1.sh").is_file())

    def test_deps_refuses_numpy1(self):
        with OwnedDirs() as own:
            ok_receipt(own.results, "stage", "1")
            (own.results / "deps-toolchain-d2.json").write_text(
                json.dumps({"numpy_version": "1.26.4"}),
                encoding="utf-8")
            (own.results / "deps-wheels-d2.sha256").write_text(
                f"{'a' * 64}  /wheels/numpy-1.26.4-cp312-cp312-"
                "manylinux_2_17_x86_64.whl\n", encoding="utf-8")

            def fake_run(distro, script, timeout):
                return (["wsl.exe"], fake_proc(stdout="DEPS_DONE\n"))
            with mock.patch.object(bib, "run_wsl", fake_run):
                rc = bib.main(["deps", "--attempt", "d2"])
            self.assertEqual(rc, 1)

    def test_deps_refuses_unexpected_dependency(self):
        with OwnedDirs() as own:
            ok_receipt(own.results, "stage", "1")
            (own.results / "deps-toolchain-d3.json").write_text(
                json.dumps({"numpy_version": "2.1.3"}),
                encoding="utf-8")
            (own.results / "deps-wheels-d3.sha256").write_text(
                f"{'a' * 64}  /wheels/requests-2.32.0-py3-none-any.whl\n",
                encoding="utf-8")

            def fake_run(distro, script, timeout):
                return (["wsl.exe"], fake_proc(stdout="DEPS_DONE\n"))
            with mock.patch.object(bib, "run_wsl", fake_run):
                rc = bib.main(["deps", "--attempt", "d3"])
            self.assertEqual(rc, 1)


class ManifestTests(unittest.TestCase):
    def test_manifest_consolidates_receipts(self):
        with OwnedDirs() as own:
            for phase in ("stage", "deps", "build", "install", "collect"):
                ok_receipt(own.results, phase, "1")
            (own.artifacts / "build-root-receipts").mkdir()
            (own.artifacts / "build-root-receipts" /
             "dependency-wheels.sha256").write_text(
                f"{'c' * 64}  /wheels/numpy-2.1.3-cp312.whl\n",
                encoding="utf-8")
            (own.artifacts / "build-root-receipts" /
             "build-env-freeze.txt").write_text("numpy==2.1.3\n",
                                                encoding="utf-8")
            ok_receipt(own.results, "install", "1",
                       native_constants={"analysis": {
                           "pre_kind": 8, "pi": math.pi},
                           "modules": {}})
            ok_receipt(own.results, "build", "1",
                       wheel_name="capytaine-3.0.0-cp312-cp312-"
                                  "linux_x86_64.whl",
                       wheel_sha256="a" * 64)
            rc = bib.main(["manifest", "--attempt", "m1"])
            self.assertEqual(rc, 0)
            m = json.loads((own.results / "build-manifest.json")
                           .read_text(encoding="utf-8"))
            self.assertEqual(m["external_identity"], bib.EXTERNAL_IDENTITY)
            self.assertEqual(m["upstream_semantic_version"], "3.0.0")
            self.assertEqual(m["dependencies"]["downloaded_wheels_sha256"]
                             ["numpy-2.1.3-cp312.whl"], "c" * 64)
            self.assertEqual(m["native_artifact"]["pi"], math.pi)

    def test_manifest_refuses_missing_prerequisite(self):
        with OwnedDirs() as own:
            for phase in ("stage", "deps", "build", "install"):
                ok_receipt(own.results, phase, "1")
            with self.assertRaises(bib.BuildError):
                bib.main(["manifest", "--attempt", "m2"])


if __name__ == "__main__":
    unittest.main()
