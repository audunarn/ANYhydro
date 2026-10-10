# SPDX-License-Identifier: MPL-2.0
"""Build the minimal W5 precision-source candidate for upstream Constants.f90.

Reproducible, refusing-by-default tool. It copies the pinned upstream
constants.f90 into an owned evidence directory and changes exactly two named
mathematical constants to full-precision PRE-kind literals:

  PI          3.141592653588979                 -> 3.141592653589793238462643383279502884_PRE
  EULER_GAMMA 0.5772156649                       -> 0.577215664901532860606512090082402431_PRE

The Apache-2.0 license header, module name, public parameter names, kinds and
procedures are preserved byte-for-byte. MINUS_ONE_OVER_FOURPI is derived from
PI and therefore auto-corrects; it is not edited. No other file, coefficient,
ABI or numerical model is touched. The upstream original is never modified.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
from pathlib import Path

TOOL = "build_candidate_patch"
TOOL_VERSION = "1.0.0"
MANIFEST_FORMAT = "anyhydro-precision-candidate-manifest-v1"

CHANGES = [
    {
        "name": "PI",
        "old_token": "3.141592653588979",
        "new_token": "3.141592653589793238462643383279502884_PRE",
        "old_line": "  REAL(KIND=PRE), PARAMETER    :: PI = 3.141592653588979 ! π",
        "new_line": "  REAL(KIND=PRE), PARAMETER    :: PI = 3.141592653589793238462643383279502884_PRE ! π",
    },
    {
        "name": "EULER_GAMMA",
        "old_token": "0.5772156649",
        "new_token": "0.577215664901532860606512090082402431_PRE",
        "old_line": "  REAL(KIND=PRE), PARAMETER    :: EULER_GAMMA = 0.5772156649",
        "new_line": "  REAL(KIND=PRE), PARAMETER    :: EULER_GAMMA = 0.577215664901532860606512090082402431_PRE",
    },
]

PARAMETER_NAME_RE = re.compile(r"::\s*([A-Za-z_]\w*)\s*=")
PROCEDURE_NAME_RE = re.compile(r"\b(?:FUNCTION|SUBROUTINE)[ \t]+([A-Za-z_]\w*)", re.I)
MODULE_NAME_RE = re.compile(r"^\s*MODULE\s+(\w+)", re.I | re.MULTILINE)


class CandidatePatchError(ValueError):
    """Unsafe or invalid candidate patch request."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def public_interface(text: str):
    """Best-effort public interface fingerprint: module, parameters, procedures."""
    module = MODULE_NAME_RE.search(text)
    return {
        "module": module.group(1).upper() if module else None,
        "parameters": sorted({m.group(1).upper() for m in PARAMETER_NAME_RE.finditer(text)}),
        "procedures": sorted({m.group(1).upper() for m in PROCEDURE_NAME_RE.finditer(text)}),
    }


def build_candidate(parent_bytes: bytes) -> bytes:
    """Apply the two named constant edits; refuse anything unexpected."""
    for change in CHANGES:
        if parent_bytes.count(change["old_line"].encode("utf-8")) != 1:
            raise CandidatePatchError(
                f"expected exactly one {change['name']} declaration line in parent, "
                f"found {parent_bytes.count(change['old_line'].encode('utf-8'))}")
    candidate = parent_bytes
    for change in CHANGES:
        candidate = candidate.replace(
            change["old_line"].encode("utf-8"), change["new_line"].encode("utf-8"))
    parent_lines = parent_bytes.splitlines(keepends=True)
    candidate_lines = candidate.splitlines(keepends=True)
    if len(parent_lines) != len(candidate_lines):
        raise CandidatePatchError("line count changed")
    stripped = [l.strip() for l in parent_bytes.decode("utf-8").splitlines()]
    expected_lines = sorted(stripped.index(c["old_line"].strip()) + 1 for c in CHANGES)
    changed = [i for i, (a, b) in enumerate(zip(parent_lines, candidate_lines), start=1)
               if a != b]
    if sorted(changed) != expected_lines:
        raise CandidatePatchError(f"unexpected changed lines: {changed}")
    return candidate


def make_patch(parent_bytes: bytes, candidate_bytes: bytes,
               parent_label: str, candidate_label: str) -> str:
    parent_text = parent_bytes.decode("utf-8")
    candidate_text = candidate_bytes.decode("utf-8")
    diff = difflib.unified_diff(
        parent_text.splitlines(keepends=True),
        candidate_text.splitlines(keepends=True),
        fromfile=parent_label, tofile=candidate_label, n=3)
    return "".join(diff)


def build(source: Path, out_dir: Path, *, expected_parent_sha: str | None = None,
          parent_label: str = "a/libDelhommeau/src/constants.f90",
          candidate_label: str = "b/libDelhommeau/src/constants.f90"):
    source = source.resolve()
    out_dir = out_dir.resolve()
    if not source.is_file():
        raise CandidatePatchError(f"parent source not found: {source}")
    if out_dir == source or out_dir in source.parents or source in out_dir.parents:
        raise CandidatePatchError("source and output must be separate non-nested paths")
    if "upstream" in out_dir.parts:
        raise CandidatePatchError("refusing to write inside the immutable upstream bundle")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"output directory not empty: {out_dir}")
    parent_bytes = source.read_bytes()
    parent_sha = sha256_bytes(parent_bytes)
    if expected_parent_sha is not None and parent_sha != expected_parent_sha:
        raise CandidatePatchError(
            f"parent sha256 mismatch: expected {expected_parent_sha}, got {parent_sha}")

    candidate_bytes = build_candidate(parent_bytes)
    candidate_sha = sha256_bytes(candidate_bytes)

    parent_iface = public_interface(parent_bytes.decode("utf-8"))
    candidate_iface = public_interface(candidate_bytes.decode("utf-8"))
    if parent_iface != candidate_iface:
        raise CandidatePatchError(
            f"public interface changed: {parent_iface} -> {candidate_iface}")

    patch_text = make_patch(parent_bytes, candidate_bytes, parent_label, candidate_label)
    patch_bytes = patch_text.encode("utf-8")
    patch_sha = sha256_bytes(patch_bytes)

    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = out_dir / "Constants.f90"
    patch_path = out_dir / "constants-candidate.patch"
    candidate_path.write_bytes(candidate_bytes)
    patch_path.write_bytes(patch_bytes)

    parent_lines = [l.strip() for l in parent_bytes.decode("utf-8").splitlines()]
    manifest = {
        "format": MANIFEST_FORMAT,
        "tool": f"tools/native_precision/{TOOL}.py",
        "tool_version": TOOL_VERSION,
        "provenance": {
            "upstream_origin": (
                "C:\\ANYworkspaceWork\\anymarine-evidence\\w5-direction-power-20261010"
                "\\pinned-native-source\\capytaine-3.0.0\\libs"),
            "upstream_version": "3.0.0",
            "pinned_bundle": (
                "C:/Github/ANYhydro/.development-evidence/"
                "w5-precision-candidate-20261010/upstream"),
            "license": "Apache-2.0 header preserved verbatim; repo tooling MPL-2.0",
            "scope": "minimal precision-source candidate; mathematical constants only",
        },
        "parent": {"path": str(source), "sha256": parent_sha},
        "candidate": {"path": str(candidate_path), "sha256": candidate_sha},
        "patch": {"path": str(patch_path), "sha256": patch_sha},
        "changes": [
            {
                "name": c["name"],
                "line": parent_lines.index(c["old_line"].strip()) + 1,
                "old": c["old_token"],
                "new": c["new_token"],
            }
            for c in CHANGES
        ],
        "derived_auto_corrected": [
            "MINUS_ONE_OVER_FOURPI = -ONE/(4*PI) recomputed from corrected PI at PRE",
        ],
        "public_interface": candidate_iface,
        "public_interface_unchanged": True,
        "upstream_original_immutable": True,
        "not_adopted": [
            "no other literal, declaration, intrinsic, coefficient, ABI or model change",
            "no output rescaling, no compiler default-real promotion",
            "no claim that PI/EULER_GAMMA alone fixes kernel precision or old energy/PSD",
        ],
    }
    manifest_path = out_dir / "patch-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path,
                    help="pinned upstream constants.f90 (read-only)")
    ap.add_argument("--out", required=True, type=Path,
                    help="empty owned output directory for the candidate bundle")
    ap.add_argument("--expected-parent-sha", default=None,
                    help="require this sha256 for the parent source")
    args = ap.parse_args(argv)
    try:
        manifest = build(args.source, args.out,
                         expected_parent_sha=args.expected_parent_sha)
    except (CandidatePatchError, FileExistsError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "candidate_sha256": manifest["candidate"]["sha256"],
        "patch_sha256": manifest["patch"]["sha256"],
        "parent_sha256": manifest["parent"]["sha256"],
        "public_interface_unchanged": manifest["public_interface_unchanged"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
