# SPDX-License-Identifier: MPL-2.0
"""Refusal, command, parsing and validation tests for probe_constants_source.

No WSL, no compiler: run_wsl/subprocess are mocked and all driver output is
synthetic. The single actual compiler probe is launched once, separately,
through check.py.
"""
import json
import math
import shlex
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import probe_constants_source as pcs

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / ".development-evidence" / "w5-precision-candidate-20261010"


def make_stdout(pi_base, gamma_base, pi_cand, gamma_cand, exe_sha="0" * 64):
    def block(variant, pi, gamma):
        mofpi = -1.0 / (4.0 * pi)
        return "\n".join([
            f"VARIANT={variant}",
            "PRE= 8",
            "PI_KIND= 8",
            "GAMMA_KIND= 8",
            f"PI= {pi:.17E}",
            f"EULER_GAMMA= {gamma:.17E}",
            f"MINUS_ONE_OVER_FOURPI= {mofpi:.17E}",
            f"LOG_2= {math.log(2.0):.17E}",
            f"{exe_sha}  ./probe_{variant}.exe",
        ])
    return ("GNU Fortran (Ubuntu 13.3.0-1ubuntu1~24.04) 13.3.0\n"
            + block("baseline", pi_base, gamma_base) + "\n"
            + block("candidate", pi_cand, gamma_cand) + "\n")


GOOD_STDOUT = make_stdout(
    pcs.float32_round(pcs.BASELINE_PI_LITERAL),
    pcs.float32_round(pcs.BASELINE_GAMMA_LITERAL),
    math.pi,
    float("0.577215664901532860606512090082402431"),
    exe_sha="a" * 64,
)


class RefusalTests(unittest.TestCase):
    def args(self, tmp, baseline="b.f90", candidate="c.f90", temp=None,
             out=None):
        return ["--baseline", str(tmp / baseline),
                "--candidate", str(tmp / candidate),
                "--temp", str(temp if temp else tmp / "temp"),
                "--out", str(out if out else tmp / "out")]

    def tmp(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        (d / "b.f90").write_text("baseline", encoding="utf-8")
        (d / "c.f90").write_text("candidate", encoding="utf-8")
        return d

    def test_timeout_above_bound_refused(self):
        with self.assertRaises(pcs.ProbeError):
            pcs.main(["--baseline", "x", "--candidate", "y", "--temp", "t",
                      "--out", "o", "--timeout", "101"])

    def test_timeout_nonpositive_refused(self):
        with self.assertRaises(pcs.ProbeError):
            pcs.main(["--baseline", "x", "--candidate", "y", "--temp", "t",
                      "--out", "o", "--timeout", "0"])

    def test_kill_after_above_bound_refused(self):
        with self.assertRaises(pcs.ProbeError):
            pcs.main(["--baseline", "x", "--candidate", "y", "--temp", "t",
                      "--out", "o", "--timeout", "100", "--kill-after", "4"])

    def test_missing_source_refused(self):
        with self.assertRaises(pcs.ProbeError):
            pcs.main(["--baseline", "does_not_exist.f90", "--candidate",
                      "also_missing.f90", "--temp", "t", "--out", "o"])

    def test_temp_inside_upstream_bundle_refused(self):
        tmp = self.tmp()
        temp = EVIDENCE / "upstream" / "probe_tmp_must_not_exist"
        with self.assertRaises(pcs.ProbeError):
            pcs.main(self.args(tmp, temp=temp))
        self.assertFalse(temp.exists())

    def test_temp_source_overlap_refused(self):
        tmp = self.tmp()
        (tmp / "temp").mkdir()
        (tmp / "temp" / "b.f90").write_text("baseline", encoding="utf-8")
        with self.assertRaises(pcs.ProbeError):
            pcs.main(self.args(tmp, baseline="temp/b.f90"))

    def test_nonempty_temp_refused(self):
        tmp = self.tmp()
        temp = tmp / "temp"
        temp.mkdir()
        (temp / "sentinel").write_text("x", encoding="utf-8")
        with self.assertRaises(pcs.ProbeError):
            pcs.main(self.args(tmp, temp=temp))

    def test_existing_receipt_refused(self):
        tmp = self.tmp()
        out = tmp / "out"
        out.mkdir()
        (out / "constants-source-probe-receipt.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(pcs.ProbeError):
            pcs.main(self.args(tmp, out=out))

    def test_temp_output_overlap_refused(self):
        tmp = self.tmp()
        with self.assertRaises(pcs.ProbeError):
            pcs.main(self.args(tmp, out=tmp / "temp" / "out"))

    def test_output_inside_upstream_bundle_refused(self):
        tmp = self.tmp()
        out = EVIDENCE / "upstream" / "out_must_not_exist"
        with self.assertRaises(pcs.ProbeError):
            pcs.main(self.args(tmp, out=out))
        self.assertFalse(out.exists())

    def test_output_containing_upstream_bundle_refused(self):
        tmp = self.tmp()
        with self.assertRaises(pcs.ProbeError):
            pcs.main(self.args(tmp, out=EVIDENCE))

    def test_output_containing_source_refused(self):
        tmp = self.tmp()
        with self.assertRaises(pcs.ProbeError):
            pcs.main(self.args(tmp, out=tmp))

    def test_output_is_existing_file_refused(self):
        tmp = self.tmp()
        with self.assertRaises(pcs.ProbeError):
            pcs.main(self.args(tmp, out=tmp / "b.f90"))

    def test_nonempty_output_refused(self):
        tmp = self.tmp()
        out = tmp / "out"
        out.mkdir()
        (out / "unrelated.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(pcs.ProbeError):
            pcs.main(self.args(tmp, out=out))

    def test_output_refusal_happens_before_temp_mutation(self):
        tmp = self.tmp()
        out = tmp / "out"
        out.mkdir()
        (out / "unrelated.txt").write_text("x", encoding="utf-8")
        with self.assertRaises(pcs.ProbeError):
            pcs.main(self.args(tmp, out=out))
        self.assertFalse((tmp / "temp").exists())


class CommandTests(unittest.TestCase):
    def test_run_wsl_argv_is_direct_no_json_escaping(self):
        with mock.patch.object(pcs.subprocess, "run") as run:
            run.return_value = SimpleNamespace(returncode=0, stdout="",
                                               stderr="")
            cmd, _proc = pcs.run_wsl("Ubuntu-24.04", "echo hi\n", 100, 3)
        run.assert_called_once()
        argv = run.call_args[0][0]
        self.assertEqual(argv, ["wsl.exe", "-d", "Ubuntu-24.04", "--",
                               "timeout", "--kill-after=3s", "100s",
                               "bash", "-c", "echo hi\n"])
        self.assertEqual(cmd, argv)
        self.assertNotIn(json.dumps("echo hi\n"), argv)

    def test_script_quotes_paths_and_sets_thread_env_inside_wsl(self):
        temp = Path("C:/Github/ANYhydro/probe temp dir")
        script = pcs.build_wsl_script(temp)
        self.assertIn("export OMP_NUM_THREADS=1", script)
        self.assertIn("export OPENBLAS_NUM_THREADS=1", script)
        self.assertIn("export MKL_NUM_THREADS=1", script)
        self.assertIn("export NUMEXPR_NUM_THREADS=1", script)
        self.assertEqual(script.count("echo VARIANT=baseline"), 1)
        self.assertEqual(script.count("echo VARIANT=candidate"), 1)
        self.assertIn(
            "cd " + shlex.quote("/mnt/c/Github/ANYhydro/probe temp dir/baseline"),
            script)
        self.assertIn(
            "cd " + shlex.quote("/mnt/c/Github/ANYhydro/probe temp dir/candidate"),
            script)
        self.assertNotIn("\\n", script)
        self.assertNotIn('\\"', script)


class ParsingTests(unittest.TestCase):
    def test_split_two_ordered_blocks(self):
        blocks = pcs.split_variant_blocks(GOOD_STDOUT)
        self.assertEqual(sorted(blocks), ["baseline", "candidate"])
        self.assertIn("PI=", blocks["baseline"])
        self.assertIn("PI=", blocks["candidate"])

    def test_split_missing_candidate_block_refused(self):
        text = GOOD_STDOUT.split("VARIANT=candidate")[0]
        with self.assertRaises(pcs.ProbeError):
            pcs.split_variant_blocks(text)

    def test_split_duplicate_marker_refused(self):
        text = GOOD_STDOUT + "VARIANT=baseline\nPI= 1.0\n"
        with self.assertRaises(pcs.ProbeError):
            pcs.split_variant_blocks(text)

    def test_split_wrong_order_refused(self):
        base = GOOD_STDOUT.split("VARIANT=baseline")[1]
        cand = GOOD_STDOUT.split("VARIANT=candidate")[1]
        with self.assertRaises(pcs.ProbeError):
            pcs.split_variant_blocks("VARIANT=candidate" + cand +
                                      "VARIANT=baseline" + base)

    def test_split_unknown_marker_refused(self):
        with self.assertRaises(pcs.ProbeError):
            pcs.split_variant_blocks("VARIANT=other\nPI= 1.0\n")

    def test_validate_good_blocks(self):
        blocks = pcs.split_variant_blocks(GOOD_STDOUT)
        base = pcs.validate_block(blocks["baseline"], "baseline")
        cand = pcs.validate_block(blocks["candidate"], "candidate")
        self.assertEqual(base["values"]["PI"],
                         pcs.float32_round(pcs.BASELINE_PI_LITERAL))
        self.assertEqual(cand["values"]["PI"], math.pi)
        self.assertEqual(cand["executable_sha256"], "a" * 64)
        self.assertEqual(base["pre"], "8")

    def test_validate_wrong_candidate_pi_refused(self):
        bad = make_stdout(pcs.float32_round(pcs.BASELINE_PI_LITERAL),
                          pcs.float32_round(pcs.BASELINE_GAMMA_LITERAL),
                          3.1415927410125732,
                          float("0.577215664901532860606512090082402431"))
        blocks = pcs.split_variant_blocks(bad)
        with self.assertRaises(pcs.ProbeError):
            pcs.validate_block(blocks["candidate"], "candidate")

    def test_validate_nonfinite_refused(self):
        block = ("PRE= 8\nPI_KIND= 8\nGAMMA_KIND= 8\n"
                 "PI= NaN\nEULER_GAMMA= 0.5E+00\n"
                 "MINUS_ONE_OVER_FOURPI= -1.0E-01\nLOG_2= 6.9E-01\n")
        with self.assertRaises(pcs.ProbeError):
            pcs.validate_block(block, "baseline")

    def test_validate_missing_key_refused(self):
        block = "PRE= 8\nPI_KIND= 8\n"
        with self.assertRaises(pcs.ProbeError):
            pcs.validate_block(block, "baseline")

    def test_validate_wrong_kind_refused(self):
        block = ("PRE= 8\nPI_KIND= 4\nGAMMA_KIND= 8\nPI= 3.14159274101257324E+00\n"
                 "EULER_GAMMA= 5.77215671539306641E-01\n"
                 "MINUS_ONE_OVER_FOURPI= -7.95774702263925281E-02\n"
                 "LOG_2= 6.93147180559945309E-01\n")
        with self.assertRaises(pcs.ProbeError):
            pcs.validate_block(block, "baseline")

    def test_validate_inconsistent_derived_refused(self):
        block = ("PRE= 8\nPI_KIND= 8\nGAMMA_KIND= 8\n"
                 "PI= 3.14159274101257324E+00\n"
                 "EULER_GAMMA= 5.77215671539306641E-01\n"
                 "MINUS_ONE_OVER_FOURPI= -1.0E-01\n"
                 "LOG_2= 6.93147180559945309E-01\n")
        with self.assertRaises(pcs.ProbeError):
            pcs.validate_block(block, "baseline")


class MainFlowTests(unittest.TestCase):
    def run_main_with_stdout(self, stdout, returncode=0):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        (tmp / "b.f90").write_text("baseline", encoding="utf-8")
        (tmp / "c.f90").write_text("candidate", encoding="utf-8")
        proc = SimpleNamespace(returncode=returncode, stdout=stdout,
                               stderr="")
        with mock.patch.object(pcs, "run_wsl",
                               return_value=(["wsl.exe"], proc)) as rw:
            rc = pcs.main(["--baseline", str(tmp / "b.f90"),
                           "--candidate", str(tmp / "c.f90"),
                           "--temp", str(tmp / "temp"),
                           "--out", str(tmp / "out")])
        receipt = json.loads((tmp / "out" / "constants-source-probe-receipt.json")
                             .read_text(encoding="utf-8"))
        return rc, receipt, rw

    def test_main_success_records_variants_and_exits_zero(self):
        rc, receipt, rw = self.run_main_with_stdout(GOOD_STDOUT)
        self.assertEqual(rc, 0)
        rw.assert_called_once()
        self.assertIsNone(receipt["validation_error"])
        self.assertEqual(sorted(receipt["compiled_variants"]),
                         ["baseline", "candidate"])
        self.assertEqual(receipt["compiled_variants"]["baseline"]["pi"],
                         pcs.float32_round(pcs.BASELINE_PI_LITERAL))
        self.assertEqual(receipt["compiled_variants"]["candidate"]["pi"],
                         math.pi)
        self.assertEqual(
            receipt["compiled_variants"]["candidate"]["executable_sha256"],
            "a" * 64)
        self.assertEqual(receipt["probe_compiler_version"],
                         "GNU Fortran (Ubuntu 13.3.0-1ubuntu1~24.04) 13.3.0")
        self.assertTrue(receipt["source_identity"]["sources_unchanged_after_probe"])
        self.assertEqual(receipt["exact_argv"], ["wsl.exe"])

    def test_main_empty_parsed_output_exits_nonzero(self):
        rc, receipt, _rw = self.run_main_with_stdout("")
        self.assertNotEqual(rc, 0)
        self.assertIn("baseline", receipt["validation_error"])

    def test_main_malformed_candidate_exits_nonzero(self):
        bad = make_stdout(pcs.float32_round(pcs.BASELINE_PI_LITERAL),
                          pcs.float32_round(pcs.BASELINE_GAMMA_LITERAL),
                          3.5, 0.577215664901532860606512090082402431)
        rc, receipt, _rw = self.run_main_with_stdout(bad)
        self.assertNotEqual(rc, 0)
        self.assertIn("wrong PI", receipt["validation_error"])
        self.assertNotIn("candidate", receipt["compiled_variants"])

    def test_main_wsl_error_exits_nonzero(self):
        rc, receipt, _rw = self.run_main_with_stdout("", returncode=124)
        self.assertNotEqual(rc, 0)
        self.assertIn("124", receipt["validation_error"])


class HelperTests(unittest.TestCase):
    def test_to_wsl_path(self):
        self.assertEqual(
            pcs.to_wsl_path(Path("C:/Github/ANYhydro/x")),
            "/mnt/c/Github/ANYhydro/x")
        self.assertEqual(
            pcs.to_wsl_path(Path("C:\\Github\\ANYhydro")),
            "/mnt/c/Github/ANYhydro")

    def test_parse_driver_output(self):
        parsed = pcs.parse_driver_output(
            "PRE= 8\nPI_KIND= 8\nPI= 3.14159265358979312E+00\n")
        self.assertEqual(parsed["PRE"], "8")
        self.assertEqual(parsed["PI"], "3.14159265358979312E+00")

    def test_probe_module_is_labelled_synthesized(self):
        self.assertIn("PROBE-SYNTHESIZED", pcs.FPP_MODULE)
        self.assertIn("kind(1d0)", pcs.FPP_MODULE)

    def test_driver_prints_kinds_and_values(self):
        self.assertIn("kind(pi)", pcs.DRIVER)
        self.assertIn("kind(euler_gamma)", pcs.DRIVER)
        self.assertIn("MINUS_ONE_OVER_FOURPI", pcs.DRIVER)

    def test_float32_round_matches_exposed_compiled_pi(self):
        self.assertEqual(pcs.float32_round(pcs.BASELINE_PI_LITERAL),
                         3.1415927410125732)


if __name__ == "__main__":
    unittest.main()
