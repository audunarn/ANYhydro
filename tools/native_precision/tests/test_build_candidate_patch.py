# SPDX-License-Identifier: MPL-2.0
"""Refusal/idempotency/hash/immutability tests for build_candidate_patch."""
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import build_candidate_patch as bcp

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / ".development-evidence" / "w5-precision-candidate-20261010"
PARENT = (EVIDENCE / "upstream" / "libs" / "libDelhommeau" / "src" / "constants.f90")
PARENT_SHA = "df6c9ae10f8f5c60fd397c5fe691c8ac527e971081d6d474138862eaea71cd96"


class Fixture(unittest.TestCase):
    def tmp(self):
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(d, ignore_errors=True))
        return d


class RefusalTests(Fixture):
    def test_refuses_nested_output(self):
        out = self.tmp() / "out"
        with self.assertRaises(bcp.CandidatePatchError):
            bcp.build(PARENT, PARENT.parent / "nested")

    def test_refuses_output_inside_upstream(self):
        with self.assertRaises(bcp.CandidatePatchError):
            bcp.build(PARENT, PARENT.parent / "candidate_dir")

    def test_refuses_existing_output(self):
        out = self.tmp() / "out"
        out.mkdir()
        (out / "sentinel").write_text("x")
        with self.assertRaises(FileExistsError):
            bcp.build(PARENT, out)

    def test_refuses_parent_sha_mismatch(self):
        with self.assertRaises(bcp.CandidatePatchError):
            bcp.build(PARENT, self.tmp() / "out", expected_parent_sha="0" * 64)

    def test_refuses_doctored_parent(self):
        doctored = self.tmp() / "constants.f90"
        text = PARENT.read_text(encoding="utf-8").replace(
            "3.141592653588979", "3.141592653588978")
        doctored.write_text(text, encoding="utf-8")
        with self.assertRaises(bcp.CandidatePatchError):
            bcp.build(doctored, self.tmp() / "out")


class CandidateTests(Fixture):
    def build_fresh(self, tmp):
        out = tmp / ("out_" + hashlib.sha256(str(tmp).encode()).hexdigest()[:8])
        manifest = bcp.build(PARENT, out, expected_parent_sha=PARENT_SHA)
        return out, manifest

    def test_parent_hash_pinned(self):
        self.assertEqual(hashlib.sha256(PARENT.read_bytes()).hexdigest(), PARENT_SHA)

    def test_idempotent_build(self):
        t1, m1 = self.build_fresh(self.tmp())
        t2, m2 = self.build_fresh(self.tmp())
        self.assertEqual((t1 / "Constants.f90").read_bytes(),
                         (t2 / "Constants.f90").read_bytes())
        self.assertEqual((t1 / "constants-candidate.patch").read_bytes(),
                         (t2 / "constants-candidate.patch").read_bytes())
        for key in ("format", "tool", "tool_version", "provenance", "changes",
                    "public_interface", "public_interface_unchanged"):
            self.assertEqual(m1[key], m2[key])
        self.assertEqual(m1["candidate"]["sha256"], m2["candidate"]["sha256"])
        self.assertEqual(m1["patch"]["sha256"], m2["patch"]["sha256"])

    def test_exactly_two_lines_changed(self):
        out, manifest = self.build_fresh(self.tmp())
        parent_lines = PARENT.read_text(encoding="utf-8").splitlines()
        cand_lines = (out / "Constants.f90").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(parent_lines), len(cand_lines))
        changed = [i for i, (a, b) in enumerate(zip(parent_lines, cand_lines),
                   start=1) if a != b]
        self.assertEqual(len(changed), 2)
        self.assertEqual([c["line"] for c in manifest["changes"]], changed)

    def test_only_named_constants_changed(self):
        out, _manifest = self.build_fresh(self.tmp())
        parent_lines = PARENT.read_text(encoding="utf-8").splitlines()
        cand_lines = (out / "Constants.f90").read_text(encoding="utf-8").splitlines()
        for i, (a, b) in enumerate(zip(parent_lines, cand_lines), start=1):
            if a == b:
                continue
            self.assertTrue(
                a.startswith("  REAL(KIND=PRE), PARAMETER    :: PI =")
                or a.startswith("  REAL(KIND=PRE), PARAMETER    :: EULER_GAMMA ="),
                f"unexpected changed line {i}: {a}")
            self.assertIn("_PRE", b)

    def test_candidate_values(self):
        out, _manifest = self.build_fresh(self.tmp())
        text = (out / "Constants.f90").read_text(encoding="utf-8")
        self.assertIn("PI = 3.141592653589793238462643383279502884_PRE", text)
        self.assertIn("EULER_GAMMA = 0.577215664901532860606512090082402431_PRE", text)
        self.assertNotIn("3.141592653588979", text)
        self.assertNotIn("0.5772156649\n", text)

    def test_apache_header_preserved(self):
        out, _manifest = self.build_fresh(self.tmp())
        parent_lines = PARENT.read_text(encoding="utf-8").splitlines()
        cand_lines = (out / "Constants.f90").read_text(encoding="utf-8").splitlines()
        self.assertEqual(parent_lines[:15], cand_lines[:15])
        self.assertIn("Apache License", "\n".join(cand_lines[:15]))

    def test_public_interface_unchanged(self):
        out, manifest = self.build_fresh(self.tmp())
        self.assertTrue(manifest["public_interface_unchanged"])
        self.assertEqual(manifest["public_interface"]["module"], "CONSTANTS")
        for name in ("PI", "EULER_GAMMA", "MINUS_ONE_OVER_FOURPI", "LOG_2",
                     "ZERO", "CZERO", "ONE", "II"):
            self.assertIn(name, manifest["public_interface"]["parameters"])
        self.assertIn("IS_INFINITY", manifest["public_interface"]["procedures"])

    def test_manifest_hashes_match_files(self):
        out, manifest = self.build_fresh(self.tmp())
        self.assertEqual(
            hashlib.sha256(PARENT.read_bytes()).hexdigest(),
            manifest["parent"]["sha256"])
        self.assertEqual(
            hashlib.sha256((out / "Constants.f90").read_bytes()).hexdigest(),
            manifest["candidate"]["sha256"])
        self.assertEqual(
            hashlib.sha256((out / "constants-candidate.patch").read_bytes()).hexdigest(),
            manifest["patch"]["sha256"])

    def test_patch_is_unified_diff_with_two_hunks(self):
        out, _manifest = self.build_fresh(self.tmp())
        patch = (out / "constants-candidate.patch").read_text(encoding="utf-8")
        self.assertTrue(patch.startswith("--- "))
        minus = [l for l in patch.splitlines() if l.startswith("-") and not
                 l.startswith("---")]
        plus = [l for l in patch.splitlines() if l.startswith("+") and not
                l.startswith("+++")]
        self.assertEqual(len(minus), 2)
        self.assertEqual(len(plus), 2)

    def test_parent_unchanged_after_build(self):
        before = hashlib.sha256(PARENT.read_bytes()).hexdigest()
        self.build_fresh(self.tmp())
        after = hashlib.sha256(PARENT.read_bytes()).hexdigest()
        self.assertEqual(before, after)


class UpstreamImmutabilityTests(Fixture):
    def test_pinned_bundle_matches_identities(self):
        identities = json.loads(
            (EVIDENCE / "upstream-identities.json").read_text(encoding="utf-8"))
        for rel, expected in identities["files"].items():
            path = EVIDENCE / "upstream" / "libs" / rel
            self.assertTrue(path.is_file(), f"missing pinned file {rel}")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual, expected, f"identity mismatch for {rel}")


if __name__ == "__main__":
    unittest.main()
