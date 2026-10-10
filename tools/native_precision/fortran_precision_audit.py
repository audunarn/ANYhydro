# SPDX-License-Identifier: MPL-2.0
"""Central Fortran source precision audit for the pinned upstream native bundle.

Source-only lexical audit. It never compiles, executes or modifies anything.
It scans free-form Fortran sources for default-kind (float32) precision hazards:

- unsuffixed real literals (rounded through default REAL before PRE promotion),
- default REAL/COMPLEX declarations (including IMPLICIT default typing),
- REAL()/CMPLX() intrinsic conversions without a KIND argument,
- expression-only default-kind arithmetic statements.

Comments, string literals and integers are ignored. Findings preserve
source file, line, column, enclosing routine, file sha256 and, for literals,
the float32 rounding where material. Files are classified as the active
newer/low_freq/scaled_nemoh3 native path, the alternative FinGreen3D /
LiangWuNoblesse paths, or unknown/auxiliary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

TOOL = "fortran_precision_audit"
TOOL_VERSION = "1.0.0"

CLASSIFICATION = {
    "libDelhommeau/src/constants.f90": "active",
    "libDelhommeau/src/Delhommeau_integrals.f90": "active",
    "libDelhommeau/src/Green_Rankine.f90": "active",
    "libDelhommeau/src/Green_wave.f90": "active",
    "libDelhommeau/src/interface.f90": "active",
    "libDelhommeau/src/matrices.f90": "active",
    "libDelhommeau/src/old_Prony_decomposition.f90": "active",
    "libDelhommeau/src/float32.f90": "active",
    "libDelhommeau/src/float64.f90": "active",
    "FinGreen3D/FinGreen3D.f90": "alternative_fingreen3d",
    "libDelhommeau/src/FinGreen3D.f90": "alternative_fingreen3d",
    "LiangWuNoblesse/LiangWuNoblesseWaveTerm.f90": "alternative_liangwunoblesse",
    "libDelhommeau/src/LiangWuNoblesseWaveTerm.f90": "alternative_liangwunoblesse",
}
CLASSIFICATION_NOTES = {
    "libDelhommeau/src/old_Prony_decomposition.f90":
        "frozen legacy Prony decomposition; preserved unchanged per disposition",
    "libDelhommeau/src/float32.f90":
        "deliberate float32 precision module; kind(1e0) literal is by design",
    "libDelhommeau/src/float64.f90":
        "deliberate float64 precision module",
}
UNKNOWN_CLASSIFICATION = "unknown_auxiliary"

FORTRAN_SUFFIXES = {".f90", ".f95", ".f", ".f03", ".inc"}

REAL_LITERAL_RE = re.compile(
    r"""(?<![\w.])(?:
        \d+\.\d*(?:[eEdD][+-]?\d+)?
      | \.\d+(?:[eEdD][+-]?\d+)?
      | \d+[eEdD][+-]?\d+
    )(?![\w.])""",
    re.VERBOSE,
)
WORD_RE = re.compile(r"[A-Za-z_]\w*")
DECL_STARTERS = {"REAL", "COMPLEX", "INTEGER", "LOGICAL", "CHARACTER",
                 "DOUBLE", "TYPE", "CLASS", "PROCEDURE"}
TYPE_WORDS = {"REAL", "COMPLEX"}
UNIT_PUSH_RES = [
    re.compile(r"^\s*MODULE\s+(\w+)", re.I),
    re.compile(r"^\s*PROGRAM\s+(\w+)", re.I),
    re.compile(r"^\s*(?:(?:PURE|ELEMENTAL|RECURSIVE|IMPURE)\s+)*SUBROUTINE\s+(\w+)", re.I),
    re.compile(
        r"^\s*(?:(?:PURE|ELEMENTAL|RECURSIVE|IMPURE)\s+)*"
        r"(?:DOUBLE\s+PRECISION"
        r"|(?:REAL|COMPLEX|INTEGER|LOGICAL)(?:\s*\(\s*(?:KIND\s*=\s*)?[\w*]+\s*\))?"
        r"|CHARACTER(?:\s*\([^)]*\))?"
        r"|TYPE\s*\([^)]*\)|CLASS\s*\([^)]*\))?"
        r"\s*FUNCTION\s+(\w+)", re.I),
]
UNIT_POP_RE = re.compile(r"^\s*END\s+(MODULE|SUBROUTINE|FUNCTION|PROGRAM)\b", re.I)
MODULE_PROCEDURE_RE = re.compile(r"^\s*MODULE\s+PROCEDURE\b", re.I)


def code_view(line: str):
    """Blank out comments and string contents, preserving character positions.

    Returns (code_text, mask) where mask[i] is True when line[i] is code.
    """
    out = list(line)
    n = len(line)
    mask = [True] * n
    i = 0
    quote = None
    while i < n:
        c = line[i]
        if quote is not None:
            if c == quote:
                if i + 1 < n and line[i + 1] == quote:
                    out[i] = out[i + 1] = " "
                    mask[i] = mask[i + 1] = False
                    i += 2
                    continue
                quote = None
            out[i] = " "
            mask[i] = False
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            out[i] = " "
            mask[i] = False
            i += 1
            continue
        if c == "!":
            for j in range(i, n):
                out[j] = " "
                mask[j] = False
            break
        i += 1
    return "".join(out), mask


def iter_statements(code_lines):
    """Join continuation lines into logical statements.

    Yields (statement_text, positions) with positions[i] = (line_no, col)
    for statement_text[i].
    """
    buf, pos = [], []
    continuation = False
    for line_no, text in code_lines:
        t = text
        lead = re.match(r"\s*&", t)
        if lead:
            idx = t.index("&")
            t = t[:idx] + " " + t[idx + 1:]
        rstrip = t.rstrip()
        trail = rstrip.endswith("&")
        if trail:
            idx = len(rstrip) - 1
            t = t[:idx] + " " + t[idx + 1:]
        if not continuation and buf:
            yield "".join(buf), pos
            buf, pos = [], []
        if not t.strip() and not continuation:
            continue
        if buf:
            buf.append(" ")
            pos.append((line_no, 0))
        for i, ch in enumerate(t):
            buf.append(ch)
            pos.append((line_no, i))
        continuation = trail
    if buf:
        yield "".join(buf), pos


def split_semicolons(text, positions):
    """Split a statement on ';' separators, keeping positions aligned."""
    parts, cur, curpos = [], [], []
    for ch, p in zip(text, positions):
        if ch == ";":
            parts.append("".join(cur))
            cur, curpos = [], []
        else:
            cur.append(ch)
            curpos.append(p)
    parts.append("".join(cur))
    # positions of each part start at the same index list; trim leading blanks
    result = []
    start = 0
    for part in parts:
        stripped = part.lstrip()
        offset = len(part) - len(stripped)
        result.append((stripped, positions[start + offset:start + len(part)]))
        start += len(part) + 1
    return result


def float32_rounding(literal: str):
    """Return (value_float32, value_float64, abs_delta, rel_delta) for a literal."""
    s = literal.lower().replace("d", "e")
    val64 = float(s)
    val32 = struct.unpack(">f", struct.pack(">f", val64))[0]
    delta = abs(val64 - val32)
    rel = delta / abs(val64) if val64 else 0.0
    return val32, val64, delta, rel


def literal_kind(match, text):
    """Classify a real literal match: suffixed / double / default_real."""
    if text[match.end():].startswith("_"):
        return "suffixed"
    if "d" in match.group(0).lower():
        return "double"
    return "default_real"


def matching_paren(text, open_idx):
    """Index of the ')' matching text[open_idx] == '(', or -1."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def count_args(text, open_idx):
    """Number of top-level comma-separated arguments of the call at open_idx."""
    close = matching_paren(text, open_idx)
    if close < 0:
        return -1
    depth = 0
    args = 1 if close > open_idx + 1 else 0
    for i in range(open_idx + 1, close):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
        elif text[i] == "," and depth == 0:
            args += 1
    return args


def first_word(text):
    m = WORD_RE.match(text.lstrip())
    return m.group(0).upper() if m else ""


def current_routine(scopes):
    for kind, name in reversed(scopes):
        if kind == "proc":
            return name
    for kind, name in reversed(scopes):
        if kind == "module":
            return name
    return ""


def update_scope(text, scopes):
    if MODULE_PROCEDURE_RE.match(text):
        return
    if UNIT_POP_RE.match(text):
        if scopes:
            scopes.pop()
        return
    for regex in UNIT_PUSH_RES:
        m = regex.match(text)
        if m:
            name = m.group(1)
            kind = "module" if regex is UNIT_PUSH_RES[0] else (
                "program" if regex is UNIT_PUSH_RES[1] else "proc")
            scopes.append((kind, name.upper()))
            return


def _snippet(text, start, width=100):
    s = text[max(0, start - 30):start + width].strip()
    return s[:100]


def scan_statement(text, positions, rel, classification, scopes, findings):
    """Append findings for one logical statement."""
    upper = text.upper()
    head = first_word(text)
    is_decl = head in DECL_STARTERS
    routine = current_routine(scopes)

    def loc(index):
        return positions[index] if index < len(positions) else (0, 0)

    # --- default REAL/COMPLEX declarations (including IMPLICIT typing) ---
    decl_hazard = None
    tspec_paren = False
    if head in TYPE_WORDS:
        rest = re.match(r"\s*" + head, text, re.I).end()
        after = text[rest:].lstrip()
        tspec_paren = after.startswith("(")
        if after.startswith("*"):
            pass  # REAL*8 style: explicit size
        elif tspec_paren:
            close = matching_paren(text, text.index("(", rest))
            content = text[text.index("(", rest) + 1:close] if close > 0 else ""
            if "-" in content:  # IMPLICIT-style letter range without kind
                decl_hazard = "implicit_default_range"
        else:
            decl_hazard = "declaration_without_kind"
    elif head == "IMPLICIT":
        m = re.match(r"\s*IMPLICIT\s+(REAL|COMPLEX)\s*", text, re.I)
        if m:
            after = text[m.end(1):].lstrip()
            if after.startswith("("):
                close = matching_paren(text, text.index("(", m.end(1)))
                content = text[text.index("(", m.end(1)) + 1:close] if close > 0 else ""
                stripped = content.strip()
                if "-" in stripped or stripped == "" or (
                        len(stripped) == 1 and stripped.isalpha()):
                    decl_hazard = "implicit_default_range"
                # else: kind selector present
            else:
                decl_hazard = "implicit_default_range"
    if decl_hazard:
        line, col = loc(0)
        findings.append({
            "category": "default_real_declaration",
            "subtype": decl_hazard,
            "file": rel, "line": line, "column": col + 1,
            "routine": routine, "classification": classification,
            "snippet": text.strip()[:100],
        })

    # --- unsuffixed real literals ---
    literal_hazards = []
    for m in REAL_LITERAL_RE.finditer(text):
        kind = literal_kind(m, text)
        if kind != "default_real":
            continue
        line, col = loc(m.start())
        val32, val64, delta, rel_d = float32_rounding(m.group(0))
        if is_decl and "PARAMETER" in upper:
            context = "parameter_initialization"
        elif is_decl:
            context = "declaration_initialization"
        elif any(k == "proc" for k, _ in scopes):
            context = "procedure_body"
        else:
            context = "module_expression"
        literal_hazards.append(m.start())
        findings.append({
            "category": "unsuffixed_real_literal",
            "literal": m.group(0),
            "context": context,
            "file": rel, "line": line, "column": col + 1,
            "routine": routine, "classification": classification,
            "float32_value": val32,
            "float64_value": val64,
            "float32_rounding_delta": delta,
            "float32_rounding_relative": rel_d,
            "material_float32_rounding": delta > 0.0,
            "snippet": _snippet(text, m.start()),
        })

    # --- REAL()/CMPLX() intrinsics without KIND ---
    intrinsic_hazards = []
    skip_first_real = tspec_paren or head == "IMPLICIT"
    for m in re.finditer(r"\bREAL\s*\(", text, re.I):
        if skip_first_real:
            skip_first_real = False
            continue
        args = count_args(text, m.end() - 1)
        if args < 2:
            line, col = loc(m.start())
            intrinsic_hazards.append(m.start())
            findings.append({
                "category": "real_intrinsic_without_kind",
                "file": rel, "line": line, "column": col + 1,
                "routine": routine, "classification": classification,
                "snippet": _snippet(text, m.start()),
            })
    for m in re.finditer(r"\bCMPLX\s*\(", text, re.I):
        args = count_args(text, m.end() - 1)
        if args < 3:
            line, col = loc(m.start())
            intrinsic_hazards.append(m.start())
            findings.append({
                "category": "cmplx_intrinsic_without_kind",
                "file": rel, "line": line, "column": col + 1,
                "routine": routine, "classification": classification,
                "snippet": _snippet(text, m.start()),
            })

    # --- expression-only default-kind arithmetic (statement level) ---
    if not is_decl and (literal_hazards or intrinsic_hazards):
        line, col = loc(0)
        findings.append({
            "category": "default_kind_expression",
            "file": rel, "line": line, "column": col + 1,
            "routine": routine, "classification": classification,
            "default_kind_literals": len(literal_hazards),
            "default_kind_intrinsics": len(intrinsic_hazards),
            "snippet": text.strip()[:100],
        })


def scan_file(path: Path, root: Path):
    rel = path.relative_to(root).as_posix()
    classification = CLASSIFICATION.get(rel, UNKNOWN_CLASSIFICATION)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    text = data.decode("utf-8", errors="replace")
    code_lines = []
    for i, line in enumerate(text.splitlines(), start=1):
        code, _mask = code_view(line)
        code_lines.append((i, code))
    findings = []
    scopes = []
    for stmt, positions in iter_statements(code_lines):
        for sub, subpos in split_semicolons(stmt, positions):
            if not sub:
                continue
            update_scope(sub, scopes)
            scan_statement(sub, subpos, rel, classification, scopes, findings)
    return {
        "file": rel,
        "sha256": sha,
        "classification": classification,
        "classification_note": CLASSIFICATION_NOTES.get(rel),
        "lines": len(code_lines),
        "findings": findings,
    }


def audit(root: Path, identities: dict | None):
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in FORTRAN_SUFFIXES:
            files.append(scan_file(path, root))
    identity_verification = None
    mismatches = 0
    if identities:
        expected = identities.get("files", {})
        identity_verification = {}
        for entry in files:
            exp = expected.get(entry["file"])
            match = (exp == entry["sha256"]) if exp else None
            if exp is not None and exp != entry["sha256"]:
                mismatches += 1
            identity_verification[entry["file"]] = {
                "expected": exp, "actual": entry["sha256"], "match": match,
            }
        for rel, exp in expected.items():
            if not any(e["file"] == rel for e in files):
                identity_verification[rel] = {
                    "expected": exp, "actual": None, "match": False,
                }
                mismatches += 1
    return files, identity_verification, mismatches


def summarize(files):
    by_category, by_classification, by_file, by_context = {}, {}, {}, {}
    material_literals = 0
    for entry in files:
        for f in entry["findings"]:
            by_category[f["category"]] = by_category.get(f["category"], 0) + 1
            by_classification[f["classification"]] = (
                by_classification.get(f["classification"], 0) + 1)
            by_file[f["file"]] = by_file.get(f["file"], 0) + 1
            if f["category"] == "unsuffixed_real_literal":
                by_context[f["context"]] = by_context.get(f["context"], 0) + 1
                if f["material_float32_rounding"]:
                    material_literals += 1
    total = sum(by_category.values())
    return {
        "files_scanned": len(files),
        "total_findings": total,
        "material_unsuffixed_literals": material_literals,
        "by_category": by_category,
        "by_classification": by_classification,
        "by_file": by_file,
        "literal_contexts": by_context,
    }


def build_report(root, files, identity_verification, mismatches, args):
    counts = summarize(files)
    return {
        "format": "anyhydro-fortran-precision-audit-v1",
        "tool": TOOL,
        "tool_version": TOOL_VERSION,
        "root": str(root),
        "classification_legend": {
            "active": "active newer/low_freq/scaled_nemoh3 native path",
            "alternative_fingreen3d": "alternative FinGreen3D path",
            "alternative_liangwunoblesse": "alternative LiangWuNoblesse path",
            "unknown_auxiliary": "benchmarks, examples, test programs, other",
        },
        "counts": counts,
        "identity_verification": identity_verification,
        "identity_mismatches": mismatches,
        "files": files,
        "limitations": [
            "source-only lexical audit; no compilation, execution or semantics",
            "string scanner state resets at line end (no multi-line strings)",
            "declaration detection is line/statement-head heuristic",
            "advisory: findings are hazards, not proof of numerical error",
        ],
        "args": {k: str(v) for k, v in vars(args).items()},
    }


def write_summary(report, out_path: Path):
    c = report["counts"]
    lines = [
        f"{TOOL} v{TOOL_VERSION} summary",
        f"root: {report['root']}",
        f"files scanned: {c['files_scanned']}",
        f"total findings: {c['total_findings']}",
        f"material unsuffixed literals (float32 rounding): "
        f"{c['material_unsuffixed_literals']}",
        "findings by category:",
    ]
    for k in sorted(c["by_category"]):
        lines.append(f"  {k}: {c['by_category'][k]}")
    lines.append("findings by classification:")
    for k in sorted(c["by_classification"]):
        lines.append(f"  {k}: {c['by_classification'][k]}")
    lines.append("findings by file:")
    for k in sorted(c["by_file"], key=lambda x: -c["by_file"][x]):
        lines.append(f"  {k}: {c['by_file'][k]}")
    lines.append("literal contexts:")
    for k in sorted(c["literal_contexts"]):
        lines.append(f"  {k}: {c['literal_contexts'][k]}")
    lines.append(f"identity mismatches: {report['identity_mismatches']}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path,
                    help="pinned upstream libs root to scan")
    ap.add_argument("--identities", type=Path, default=None,
                    help="upstream-identities.json for sha256 immutability check")
    ap.add_argument("--out", required=True, type=Path,
                    help="output directory for report and summary")
    args = ap.parse_args(argv)

    root = args.root.resolve()
    if not root.is_dir():
        ap.error(f"root not found: {root}")
    identities = None
    if args.identities:
        identities = json.loads(args.identities.read_text(encoding="utf-8"))

    files, identity_verification, mismatches = audit(root, identities)
    report = build_report(root, files, identity_verification, mismatches, args)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "precision-audit-report.json"
    summary_path = out / "precision-audit-summary.txt"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = write_summary(report, summary_path)
    print(summary)
    print(f"report: {report_path}")
    print(f"summary: {summary_path}")
    if mismatches:
        print(f"ERROR: {mismatches} identity mismatch(es); sources not immutable",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
