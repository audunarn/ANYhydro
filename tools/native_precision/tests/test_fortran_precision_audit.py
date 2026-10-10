# SPDX-License-Identifier: MPL-2.0
"""Lexical/parser/classification tests for fortran_precision_audit."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import fortran_precision_audit as fpa


def audit_text(tmp_path, source, filename="src/thing.f90"):
    root = tmp_path / "root"
    f = root / filename
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(source, encoding="utf-8")
    files, ident, mismatches = fpa.audit(root, None)
    findings = [x for entry in files for x in entry["findings"]]
    return files, findings


class CodeViewTests(unittest.TestCase):
    def test_comment_ignored(self):
        code, mask = fpa.code_view("x = 1.0 ! 2.5 is a comment")
        self.assertNotIn("2.5", code)
        self.assertIn("1.0", code)

    def test_string_ignored(self):
        code, mask = fpa.code_view("s = '3.14' // \"0.5\"")
        self.assertNotIn("3.14", code)
        self.assertNotIn("0.5", code)

    def test_doubled_quote_stays_string(self):
        code, _ = fpa.code_view("s = 'it''s 3.14'")
        self.assertNotIn("3.14", code)

    def test_positions_preserved(self):
        code, _ = fpa.code_view("x = 1.0")
        self.assertEqual(len(code), len("x = 1.0"))
        self.assertEqual(code[:4], "x = ")


class LiteralTests(unittest.TestCase):
    def test_integer_ignored(self):
        _files, findings = audit_text(self._tmp(), "n = 4\ni = 123\n")
        self.assertEqual(findings, [])

    def test_suffixed_and_double_ignored(self):
        _files, findings = audit_text(
            self._tmp(), "a = 1.0d0\nb = 2.5_pre\nc = 3d-5\nd = 1.0e5_pre\n")
        self.assertEqual(findings, [])

    def test_unsuffixed_flagged_with_float32_rounding(self):
        _files, findings = audit_text(
            self._tmp(),
            "module m\nreal(kind=pre), parameter :: pi = 3.141592653588979\nend module m\n")
        lits = [f for f in findings if f["category"] == "unsuffixed_real_literal"]
        self.assertEqual(len(lits), 1)
        self.assertTrue(lits[0]["material_float32_rounding"])
        self.assertEqual(lits[0]["float32_value"], 3.1415927410125732)
        self.assertEqual(lits[0]["context"], "parameter_initialization")
        self.assertEqual(lits[0]["line"], 2)

    def test_exact_literal_not_material(self):
        _files, findings = audit_text(
            self._tmp(), "module m\nreal(kind=pre), parameter :: one = 1.0\nend module m\n")
        lits = [f for f in findings if f["category"] == "unsuffixed_real_literal"]
        self.assertEqual(len(lits), 1)
        self.assertFalse(lits[0]["material_float32_rounding"])
        self.assertEqual(lits[0]["float32_rounding_delta"], 0.0)

    def test_e_exponent_is_default_real(self):
        _files, findings = audit_text(self._tmp(), "x = 1e-5\ny = 1.5e2\n")
        lits = [f for f in findings if f["category"] == "unsuffixed_real_literal"]
        self.assertEqual(len(lits), 2)

    def test_literal_in_comment_only_not_flagged(self):
        _files, findings = audit_text(self._tmp(), "! pi = 3.141592653588979\n")
        self.assertEqual(findings, [])

    def test_literal_in_string_only_not_flagged(self):
        _files, findings = audit_text(self._tmp(), "s = '0.5772156649'\n")
        self.assertEqual(findings, [])

    def test_continuation_line_number(self):
        _files, findings = audit_text(
            self._tmp(), "x = 1.0 &\n     + 2.5\n")
        lits = [f for f in findings if f["category"] == "unsuffixed_real_literal"]
        self.assertEqual([f["line"] for f in lits], [1, 2])

    def _tmp(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d


class DeclarationTests(unittest.TestCase):
    def test_default_real_declaration(self):
        _files, findings = audit_text(self._tmp(), "subroutine s\nreal :: x\nend subroutine s\n")
        decls = [f for f in findings if f["category"] == "default_real_declaration"]
        self.assertEqual(len(decls), 1)
        self.assertEqual(decls[0]["routine"], "S")

    def test_kinded_declaration_not_flagged(self):
        _files, findings = audit_text(
            self._tmp(), "real(kind=pre) :: x\nreal*8 :: y\ndouble precision :: z\ncomplex(kind=pre) :: w\n")
        decls = [f for f in findings if f["category"] == "default_real_declaration"]
        self.assertEqual(decls, [])

    def test_default_complex_declaration(self):
        _files, findings = audit_text(self._tmp(), "complex :: w\n")
        decls = [f for f in findings if f["category"] == "default_real_declaration"]
        self.assertEqual(len(decls), 1)

    def test_implicit_default_range(self):
        _files, findings = audit_text(self._tmp(), "implicit real (a-z)\n")
        decls = [f for f in findings if f["category"] == "default_real_declaration"]
        self.assertEqual(len(decls), 1)
        self.assertEqual(decls[0]["subtype"], "implicit_default_range")

    def test_implicit_kinded_not_flagged(self):
        _files, findings = audit_text(
            self._tmp(), "implicit real(kind=pre) (a-h)\nimplicit none\n")
        decls = [f for f in findings if f["category"] == "default_real_declaration"]
        self.assertEqual(decls, [])

    def _tmp(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d


class IntrinsicTests(unittest.TestCase):
    def test_real_without_kind(self):
        _files, findings = audit_text(self._tmp(), "y = real(x)\n")
        hits = [f for f in findings if f["category"] == "real_intrinsic_without_kind"]
        self.assertEqual(len(hits), 1)

    def test_real_with_kind_not_flagged(self):
        _files, findings = audit_text(self._tmp(), "y = real(x, kind=pre)\ny = real(x, pre)\n")
        hits = [f for f in findings if f["category"] == "real_intrinsic_without_kind"]
        self.assertEqual(hits, [])

    def test_declaration_kind_selector_not_intrinsic(self):
        _files, findings = audit_text(
            self._tmp(),
            "module m\nreal(kind=pre), parameter :: q = log(real(2d0, kind=pre))\nend module m\n")
        hits = [f for f in findings if f["category"] == "real_intrinsic_without_kind"]
        self.assertEqual(hits, [])

    def test_cmplx_without_kind(self):
        _files, findings = audit_text(self._tmp(), "z = cmplx(a, b)\n")
        hits = [f for f in findings if f["category"] == "cmplx_intrinsic_without_kind"]
        self.assertEqual(len(hits), 1)

    def test_cmplx_with_kind_not_flagged(self):
        _files, findings = audit_text(
            self._tmp(), "z = cmplx(a, b, kind=pre)\nz = cmplx(a, b, pre)\n")
        hits = [f for f in findings if f["category"] == "cmplx_intrinsic_without_kind"]
        self.assertEqual(hits, [])

    def test_default_kind_expression_statement(self):
        _files, findings = audit_text(self._tmp(), "y = 0.5*x + real(n)\n")
        exprs = [f for f in findings if f["category"] == "default_kind_expression"]
        self.assertEqual(len(exprs), 1)
        self.assertEqual(exprs[0]["default_kind_literals"], 1)
        self.assertEqual(exprs[0]["default_kind_intrinsics"], 1)

    def _tmp(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d


class ClassificationTests(unittest.TestCase):
    def test_active_and_alternative(self):
        root = self._tmp() / "root"
        for rel in ("libDelhommeau/src/constants.f90", "FinGreen3D/FinGreen3D.f90",
                    "libDelhommeau/benchmarks/rankine/benchmark_rankine.f90"):
            f = root / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("x = 1.0\n", encoding="utf-8")
        files, _ident, _m = fpa.audit(root, None)
        by_rel = {e["file"]: e for e in files}
        self.assertEqual(by_rel["libDelhommeau/src/constants.f90"]["classification"], "active")
        self.assertEqual(by_rel["FinGreen3D/FinGreen3D.f90"]["classification"],
                         "alternative_fingreen3d")
        self.assertEqual(
            by_rel["libDelhommeau/benchmarks/rankine/benchmark_rankine.f90"]["classification"],
            "unknown_auxiliary")

    def test_routine_attribution_nested(self):
        source = ("module mod_a\ncontains\nsubroutine sub_b\nx = 1.0\n"
                  "end subroutine sub_b\nend module mod_a\n")
        _files, findings = audit_text(self._tmp(), source)
        lits = [f for f in findings if f["category"] == "unsuffixed_real_literal"]
        self.assertEqual(lits[0]["routine"], "SUB_B")

    def test_report_counts_consistent(self):
        root = self._tmp() / "root"
        f = root / "libDelhommeau/src/constants.f90"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("module m\nreal :: x\nx = 1.0\nend module m\n", encoding="utf-8")
        files, ident, mismatches = fpa.audit(root, None)
        report = fpa.build_report(root, files, ident, mismatches,
                                  argparse_stub(root))
        total = sum(len(e["findings"]) for e in files)
        self.assertEqual(report["counts"]["total_findings"], total)
        self.assertIn("sha256", report["files"][0])
        self.assertEqual(mismatches, 0)

    def _tmp(self):
        import tempfile
        d = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(d, ignore_errors=True))
        return d


def argparse_stub(root):
    class A:
        pass
    a = A()
    a.root = root
    a.identities = None
    a.out = None
    return a


if __name__ == "__main__":
    unittest.main()
