# SPDX-License-Identifier: MPL-2.0
"""Connected E1 PRE-kind candidate: patch, offline isolated build, runtime
identity and provenance tool.

Builds a complete, distinctly identified native Capytaine 3.0.0 wheel from the
exact official sdist archived in the w5-isolated-backend evidence, applying
ONLY the adopted two-line PI/EULER_GAMMA constants candidate plus the minimal
exp_e1 PRE-kind patch to libs/libDelhommeau/src/Delhommeau_integrals.f90, in a
fresh owned WSL build root, using exclusively archived offline build and
runtime inputs (no network), and proves the installed native PRE-kind
constants and E1 module identity by direct extension loading and a
metadata-only package import.

exp_e1 candidate (exactly four lines of the original file change):
- precision comment updated to the selected precision,
- `complex(kind=kind(1e0)) :: g, y, z` -> `complex(kind=pre) :: g, y, z`,
- `z = cmplx(zz, kind=kind(1e0))` -> `z = zz`,
- `exp_e1 = cmplx(g, kind=pre)` -> `exp_e1 = g`.
Every coefficient literal byte, approximation formula and branch test is
preserved. Arithmetic precision can change branch selection at rounding
boundaries; no identical-path or universal accuracy claim is made.

This is artifact development verification only. It is NOT Green-function,
ABI, kernel, scientific or platform qualification. Zero GF constructors,
kernel evaluations, table generations, Prony fitting, systems, HAMS, dynamics
or campaigns are invoked; the package import is metadata-only and the native
extension is additionally loaded directly from its file path.

Safety contract:
- Every WSL phase is one bounded child launched as argv
  ['wsl.exe', '-d', <distro>, '--', 'timeout', '--kill-after=3s', '<t>s',
  'bash', '-c', <base64-decoded script>] with GNU timeout <= 180 s, one
  job/thread (OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 exported inside the WSL
  shell, ninja driven with -j1 via the meson-python compile-args config
  setting, proven by an owned PATH wrapper logging the actual ninja argv).
- The immutable upstream release tree, archive, acquisition record, archived
  dependency/runtime wheels and locks are verified by SHA-256 before anything
  is created and never written to; the candidate tree is a separate copy whose
  only differences are the two candidate files, verified by full-tree diff.
- Fresh owned environments only: refuses an existing/nonempty build root,
  build/install venvs, wheel or receipts; nothing is installed into any
  existing application, Windows or old-Linux runtime; no network access, no
  downloads, no resolver choices beyond the archived locks.
- Every phase writes a unique receipt with exact argv, full stdout/stderr,
  wall time, validation error and identities; failures are preserved.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

TOOL = "build_e1_backend"
TOOL_VERSION = "1.0.0"

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / ".development-evidence" / "w5-e1-kind-backend-20261010"
CANDIDATE_DIR = EVIDENCE / "candidate"
SCRIPTS = EVIDENCE / "scripts"
ARTIFACTS = EVIDENCE / "artifacts"
RESULTS = EVIDENCE / "results"
ROOT_RECEIPTS = EVIDENCE / "receipts"

ISOLATED = REPO / ".development-evidence" / "w5-isolated-backend-20261010"
UPSTREAM_RELEASE = ISOLATED / "upstream-release"
UPSTREAM_TREE = UPSTREAM_RELEASE / "capytaine-3.0.0"
ARCHIVE = UPSTREAM_RELEASE / "capytaine-3.0.0.tar.gz"
SOURCE_ACQUISITION = UPSTREAM_RELEASE / "source-acquisition.json"
DEP_WHEELS_DIR = ISOLATED / "artifacts" / "dependency-wheels"
DEP_WHEELS_SUMS = DEP_WHEELS_DIR / "SHA256SUMS"
BUILD_LOCK = ISOLATED / "artifacts" / "build-requirements.lock"
BOOTSTRAP_PIP_WHEEL = ISOLATED / "bootstrap" / "pip-26.2.1-py3-none-any.whl"
BOOTSTRAP_PIP_ACQ = ISOLATED / "bootstrap" / "pip-acquisition.json"

ADOPTED_CONSTANTS = (REPO / ".development-evidence" /
                     "w5-precision-candidate-20261010" / "candidate" /
                     "Constants.f90")

RUNTIME_BASE = Path("C:/ANYworkspaceWork/anymarine-evidence/"
                    "w5-rankine-artifact-20261010")
RUNTIME_WHEELS_DIR = RUNTIME_BASE / "runtime-wheels"
RUNTIME_WHEELS_SUMS = RUNTIME_WHEELS_DIR / "SHA256SUMS"
RUNTIME_LOCK = RUNTIME_BASE / "runtime-requirements.lock"

BUILD_ROOT = "/var/tmp/anyhydro-e1-kind-backend-20261010"
WSL_DISTRO_DEFAULT = "Ubuntu-24.04"
EXTERNAL_IDENTITY = "capytaine-3.0.0-anyhydro-pi-gamma-e1-pre-v2/linux-x86_64"
UPSTREAM_SEMANTIC_VERSION = "3.0.0"
MAX_GNU_TIMEOUT = 180
KILL_AFTER = 3

EXPECTED_ARCHIVE_SHA256 = ("2651fd1cd66624d6b5f56e84f5d0001eda34b5d108f38c"
                           "39d908f425f20bc968")
EXPECTED_UPSTREAM_CONSTANTS_SHA256 = ("df6c9ae10f8f5c60fd397c5fe691c8ac527e97"
                                      "1081d6d474138862eaea71cd96")
EXPECTED_CANDIDATE_CONSTANTS_SHA256 = ("5b8233205963e84710629ee9395ccb7e78ddb"
                                       "138bca411cfe8defb481b6b66bf")
EXPECTED_BASELINE_E1_SHA256 = ("01d740fdafd8bacaf156226fa092950459f8846e3"
                               "c9d5ee6069c2215a619fcb0")
EXPECTED_BOOTSTRAP_PIP_SHA256 = ("71138adf1f4ca900cdb7d289c21b7494329f2332b6d"
                                 "85f0e1c42108c0384ed3e")
CANDIDATE_GAMMA_REFERENCE = float("0.577215664901532860606512090082402431")

CONSTANTS_REL = "libs/libDelhommeau/src/constants.f90"
E1_REL = "libs/libDelhommeau/src/Delhommeau_integrals.f90"

E1_OLD_COMMENT = "      ! the computation is done is single precision."
E1_NEW_COMMENT = ("      ! the computation is done in the selected precision "
                   "(kind=pre).")
E1_OLD_DECL = "      complex(kind=kind(1e0))        :: g, y, z"
E1_NEW_DECL = "      complex(kind=pre)              :: g, y, z"
E1_OLD_ASSIGN = "      z = cmplx(zz, kind=kind(1e0))"
E1_NEW_ASSIGN = "      z = zz"
E1_OLD_RETURN = "      exp_e1 = cmplx(g, kind=pre)"
E1_NEW_RETURN = "      exp_e1 = g"
E1_CHANGED_LINES = (107, 111, 113, 149)

EXPECTED_PI_LINE = ("  REAL(KIND=PRE), PARAMETER    :: PI = "
                    "3.141592653589793238462643383279502884_PRE ! π")
EXPECTED_GAMMA_LINE = ("  REAL(KIND=PRE), PARAMETER    :: EULER_GAMMA = "
                       "0.577215664901532860606512090082402431_PRE")

EXPECTED_BUILD_DEPS = {
    "charset-normalizer": "3.5.2", "meson": "1.12.1",
    "meson-python": "0.22.1", "ninja": "1.13.2", "numpy": "2.5.3",
    "packaging": "26.3", "pyproject-metadata": "0.12.1",
}
EXPECTED_RUNTIME_DEPS = {
    "markdown-it-py": "4.2.0", "mdurl": "0.1.2", "numpy": "2.5.3",
    "packaging": "26.3", "pandas": "2.3.3", "Pygments": "2.21.0",
    "python-dateutil": "2.9.0.post0", "pytz": "2026.5", "rich": "15.0.0",
    "scipy": "1.18.1", "six": "1.17.0", "threadpoolctl": "3.6.0",
    "tzdata": "2026.5", "xarray": "2026.9.0",
}
THREAD_ENV_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

PHASES = ("patch", "stage", "deps", "build", "install", "collect", "manifest",
          "cleanenv")
DEFAULT_GNU_TIMEOUTS = {"patch": 0, "stage": 120, "deps": 150, "build": 180,
                        "install": 180, "collect": 60, "manifest": 0,
                        "cleanenv": 60}


class BuildError(ValueError):
    """Invalid or unsafe build request."""


def to_wsl_path(path: Path) -> str:
    s = str(path.resolve())
    if re.match(r"^[A-Za-z]:", s):
        return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")
    return s


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def receipt_path(phase: str, attempt: str) -> Path:
    return RESULTS / f"{phase}-{attempt}-receipt.json"


def require_fresh_receipt(phase: str, attempt: str) -> None:
    if receipt_path(phase, attempt).exists():
        raise BuildError(f"refusing to overwrite an existing receipt for "
                         f"{phase} attempt {attempt!r}")


def find_ok_receipt(phase: str) -> Path | None:
    for p in sorted(RESULTS.glob(f"{phase}-*-receipt.json")):
        try:
            if json.loads(p.read_text(encoding="utf-8")).get("ok"):
                return p
        except (OSError, ValueError):
            continue
    return None


def require_prerequisite(phase: str, prerequisite: str) -> Path:
    p = find_ok_receipt(prerequisite)
    if p is None:
        raise BuildError(f"{phase}: no successful {prerequisite} receipt; "
                         f"run {prerequisite} first")
    return p


def check_build_root_safety() -> None:
    if BUILD_ROOT.startswith("/mnt/"):
        raise BuildError(f"build root {BUILD_ROOT} must not be on a Windows "
                         f"mount")
    for immutable in (UPSTREAM_RELEASE, REPO, ADOPTED_CONSTANTS.parent,
                      Path("C:/ANYworkspaceWork")):
        wsl = to_wsl_path(immutable)
        if BUILD_ROOT == wsl or BUILD_ROOT.startswith(wsl + "/"):
            raise BuildError(f"build root {BUILD_ROOT} overlaps immutable "
                             f"path {immutable}")


def verify_upstream_tree() -> dict:
    """Verify archive and every extracted file against source-acquisition."""
    acq = json.loads(SOURCE_ACQUISITION.read_text(encoding="utf-8"))
    if acq.get("archive_sha256") != EXPECTED_ARCHIVE_SHA256:
        raise BuildError("source-acquisition archive SHA mismatch")
    if sha256_file(ARCHIVE) != EXPECTED_ARCHIVE_SHA256:
        raise BuildError(f"archive {ARCHIVE} SHA-256 mismatch")
    if sha256_file(UPSTREAM_TREE / CONSTANTS_REL) != \
            EXPECTED_UPSTREAM_CONSTANTS_SHA256:
        raise BuildError("upstream constants.f90 SHA mismatch")
    if sha256_file(UPSTREAM_TREE / E1_REL) != EXPECTED_BASELINE_E1_SHA256:
        raise BuildError("upstream Delhommeau_integrals.f90 SHA mismatch")
    expected = {k.replace("\\", "/"): v for k, v in
                acq.get("file_sha256", {}).items()}
    if not expected:
        raise BuildError("source-acquisition has no file SHA map")
    actual = {p.relative_to(UPSTREAM_TREE).as_posix(): sha256_file(p)
              for p in UPSTREAM_TREE.rglob("*") if p.is_file()}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    mismatched = sorted(k for k in expected.keys() & actual.keys()
                        if expected[k] != actual[k])
    if missing or extra or mismatched:
        raise BuildError(f"upstream tree mismatch: missing={missing[:5]} "
                         f"extra={extra[:5]} mismatched={mismatched[:5]}")
    return {"archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "archive_bytes": acq.get("archive_bytes"),
            "verified_files": len(expected),
            "upstream_constants_sha256": EXPECTED_UPSTREAM_CONSTANTS_SHA256,
            "baseline_e1_sha256": EXPECTED_BASELINE_E1_SHA256}


def verify_adopted_constants() -> dict:
    """Verify the adopted constants.f90 differs in exactly lines 26/28."""
    if sha256_file(ADOPTED_CONSTANTS) != EXPECTED_CANDIDATE_CONSTANTS_SHA256:
        raise BuildError(f"candidate constants SHA mismatch: "
                         f"{ADOPTED_CONSTANTS}")
    base = (UPSTREAM_TREE / CONSTANTS_REL).read_text(encoding="utf-8",
                                                     errors="strict")
    cand = ADOPTED_CONSTANTS.read_text(encoding="utf-8", errors="strict")
    base_lines = base.splitlines()
    cand_lines = cand.splitlines()
    if len(base_lines) != len(cand_lines):
        raise BuildError("candidate changes the line count")
    changed = [i + 1 for i, (b, c) in enumerate(zip(base_lines, cand_lines))
               if b != c]
    if changed != [26, 28]:
        raise BuildError(f"candidate must change exactly lines 26/28, "
                         f"changed: {changed}")
    if cand_lines[25] != EXPECTED_PI_LINE:
        raise BuildError(f"candidate PI line mismatch: {cand_lines[25]!r}")
    if cand_lines[27] != EXPECTED_GAMMA_LINE:
        raise BuildError(f"candidate EULER_GAMMA line mismatch: "
                         f"{cand_lines[27]!r}")
    return {"candidate_constants_sha256": EXPECTED_CANDIDATE_CONSTANTS_SHA256,
            "changed_lines": changed,
            "pi_line": cand_lines[25],
            "euler_gamma_line": cand_lines[27],
            "all_other_bytes_identical": True}


def apply_e1_patch(baseline: bytes) -> tuple[bytes, list[int]]:
    """Apply the exact four-line exp_e1 PRE-kind patch; return new bytes."""
    lines = baseline.decode("utf-8").split("\n")
    changed = []
    for no, old, new in ((E1_CHANGED_LINES[0], E1_OLD_COMMENT, E1_NEW_COMMENT),
                         (E1_CHANGED_LINES[1], E1_OLD_DECL, E1_NEW_DECL),
                         (E1_CHANGED_LINES[2], E1_OLD_ASSIGN, E1_NEW_ASSIGN),
                         (E1_CHANGED_LINES[3], E1_OLD_RETURN, E1_NEW_RETURN)):
        if lines[no - 1] != old:
            raise BuildError(f"E1 line {no} does not match the expected "
                             f"original: {lines[no - 1]!r}")
        lines[no - 1] = new
        changed.append(no)
    return "\n".join(lines).encode("utf-8"), changed


def verify_e1_patch(baseline: bytes, patched: bytes) -> dict:
    """Verify patched E1 differs from baseline in exactly the four lines."""
    base_lines = baseline.decode("utf-8").split("\n")
    new_lines = patched.decode("utf-8").split("\n")
    if len(base_lines) != len(new_lines):
        raise BuildError("E1 patch changes the line count")
    changed = [i + 1 for i, (b, c) in enumerate(zip(base_lines, new_lines))
               if b != c]
    if changed != list(E1_CHANGED_LINES):
        raise BuildError(f"E1 patch must change exactly lines "
                         f"{list(E1_CHANGED_LINES)}, changed: {changed}")
    expected_new = {E1_CHANGED_LINES[0]: E1_NEW_COMMENT,
                    E1_CHANGED_LINES[1]: E1_NEW_DECL,
                    E1_CHANGED_LINES[2]: E1_NEW_ASSIGN,
                    E1_CHANGED_LINES[3]: E1_NEW_RETURN}
    for no, text in expected_new.items():
        if new_lines[no - 1] != text:
            raise BuildError(f"E1 patched line {no} mismatch: "
                             f"{new_lines[no - 1]!r}")
    return {"changed_lines": changed,
            "patched_e1_sha256": sha256_bytes(patched),
            "line_count_unchanged": True,
            "coefficients_formulas_branch_tests_unchanged": True}


def unified_diff_text(before: bytes, after: bytes, fromfile: str,
                      tofile: str) -> str:
    import difflib
    return "".join(difflib.unified_diff(
        before.decode("utf-8").splitlines(keepends=True),
        after.decode("utf-8").splitlines(keepends=True),
        fromfile=fromfile, tofile=tofile))


def checklist_text() -> tuple[str, int]:
    acq = json.loads(SOURCE_ACQUISITION.read_text(encoding="utf-8"))
    lines = [f"{v}  {k.replace(chr(92), '/')}"
             for k, v in sorted(acq["file_sha256"].items())]
    return "\n".join(lines) + "\n", len(lines)


def verify_wheel_sums(sums_path: Path, wheels_dir: Path,
                      exclude: str = "") -> dict:
    """Verify archived wheels against their SHA256SUMS record."""
    if not sums_path.is_file():
        raise BuildError(f"wheel SHA256SUMS missing: {sums_path}")
    verified = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            raise BuildError(f"malformed SHA256SUMS line: {line!r}")
        digest, raw = parts
        name = Path(raw.replace("\\", "/")).name
        if exclude and name.startswith(exclude):
            continue
        wheel = wheels_dir / name
        if not wheel.is_file():
            raise BuildError(f"archived wheel missing: {wheel}")
        actual = sha256_file(wheel)
        if actual != digest:
            raise BuildError(f"archived wheel SHA mismatch for {name}: "
                             f"{actual} != {digest}")
        verified[name] = digest
    if not verified:
        raise BuildError("no archived wheels verified")
    return verified


def verify_lock_names(lock_path: Path, verified: dict) -> dict:
    """Verify every locked requirement is present among verified wheels."""
    locked = {}
    for line in lock_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or " @ " in line \
                or "file:" in line:
            continue
        name, _, version = line.partition("==")
        if not version:
            continue
        locked[name] = version
    wheel_names = {}
    wheel_versions = {}
    for filename in verified:
        parts = filename.split("-")
        dist = re.sub(r"[-_]", "", parts[0]).lower()
        wheel_names[dist] = filename
        wheel_versions[dist] = parts[1] if len(parts) > 1 else ""
    missing = []
    for name, version in locked.items():
        key = re.sub(r"[-_]", "", name).lower()
        if key not in wheel_names:
            missing.append(f"{name}=={version}")
        elif wheel_versions[key] != version:
            missing.append(f"{name}=={version} (wheel "
                           f"{wheel_names[key]})")
    if missing:
        raise BuildError(f"locked requirements not satisfied by archived "
                         f"wheels: {missing}")
    return locked


def run_wsl(distro: str, script: str, timeout: int):
    """Launch one bounded WSL child.

    wsl.exe routes the command line through the WSL default shell, which
    would expand $variables and $(...) in a raw script argument before
    bash -c receives it. The script is therefore passed base64-encoded and
    decoded by the inner bash via command substitution, which contains no
    expandable characters; the decoded script then executes inside the
    timeout-monitored bash process itself. The exact decoded script is
    saved verbatim under scripts/ for reproduction.
    """
    b64 = base64.b64encode(script.encode("utf-8")).decode("ascii")
    inner = f"$(echo {b64} | base64 -d)"
    cmd = ["wsl.exe", "-d", distro, "--", "timeout",
           f"--kill-after={KILL_AFTER}s", f"{timeout}s", "bash", "-c",
           inner]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return cmd, proc


def thread_env_lines() -> list[str]:
    return [f"export {k}=1" for k in THREAD_ENV_KEYS]


def save_text(path: Path, text: str) -> Path:
    """Write evidence text; refuse to overwrite differing existing bytes.

    Byte-identical regeneration reuses the existing file, so a new unique
    phase attempt never requires deleting earlier evidence; any differing
    content on an existing path is refused.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    if path.exists():
        if path.read_bytes() == data:
            return path
        raise BuildError(f"refusing to overwrite existing evidence {path} "
                         f"with differing content")
    path.write_bytes(data)
    return path


def save_script(name: str, script: str) -> Path:
    return save_text(SCRIPTS / name, script)


def finish_receipt(receipt: dict, phase: str, attempt: str, cmd, proc,
                   wall: float, validation_error, extra=None) -> dict:
    receipt.update({
        "tool": f"tools/native_precision/{TOOL}.py",
        "tool_sha256": sha256_file(Path(__file__)),
        "tool_version": TOOL_VERSION,
        "phase": phase,
        "attempt": attempt,
        "external_identity": EXTERNAL_IDENTITY,
        "upstream_semantic_version": UPSTREAM_SEMANTIC_VERSION,
        "exact_argv": cmd if cmd is not None else
            [sys.executable, str(Path(__file__).resolve()), phase,
             "--attempt", attempt],
        "gnu_timeout_seconds": receipt.get("gnu_timeout_seconds"),
        "gnu_kill_after_seconds": KILL_AFTER if cmd is not None else None,
        "process_wall_seconds": wall,
        "returncode": proc.returncode if proc is not None else None,
        "validation_error": validation_error,
        "stdout": proc.stdout if proc is not None else "",
        "stderr": proc.stderr if proc is not None else "",
        "ok": (proc.returncode == 0 if proc is not None else True)
              and validation_error is None,
    })
    if extra:
        receipt.update(extra)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = receipt_path(phase, attempt)
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in
                      ("phase", "attempt", "ok", "returncode",
                       "process_wall_seconds", "validation_error")},
                     indent=2))
    print(f"receipt: {out}")
    return receipt


def parse_version_source(source: str) -> str:
    """Read __version__ from generated source using AST (no execution)."""
    values = []
    for node in ast.parse(source).body:
        targets = []
        value = None
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        if value is not None and any(
                isinstance(t, ast.Name) and t.id == "__version__"
                for t in targets):
            values.append(ast.literal_eval(value))
    if len(values) != 1:
        raise BuildError(f"expected exactly one __version__ assignment, "
                         f"found {values!r}")
    return values[0]


def validate_built_version(about: str | None) -> str:
    if not about:
        raise BuildError("built __about__.py missing")
    version = parse_version_source(about)
    if version != UPSTREAM_SEMANTIC_VERSION:
        raise BuildError(f"built __version__ is {version!r}, not "
                         f"{UPSTREAM_SEMANTIC_VERSION!r}")
    return version


def phase_patch(args) -> int:
    """Windows-local candidate generation and validation (no WSL)."""
    require_fresh_receipt("patch", args.attempt)
    check_build_root_safety()
    upstream = verify_upstream_tree()
    constants = verify_adopted_constants()

    baseline_e1 = (UPSTREAM_TREE / E1_REL).read_bytes()
    if sha256_bytes(baseline_e1) != EXPECTED_BASELINE_E1_SHA256:
        raise BuildError("baseline E1 SHA mismatch")
    patched_e1, changed = apply_e1_patch(baseline_e1)
    e1_check = verify_e1_patch(baseline_e1, patched_e1)

    out_constants = CANDIDATE_DIR / "Constants.f90"
    out_e1 = CANDIDATE_DIR / "Delhommeau_integrals.f90"
    if out_constants.exists() or out_e1.exists():
        raise BuildError("candidate output already exists; refusing to "
                         "overwrite candidate evidence")
    save_text(out_constants, ADOPTED_CONSTANTS.read_text(encoding="utf-8"))
    save_text(out_e1, patched_e1.decode("utf-8"))

    diff_constants = unified_diff_text(
        (UPSTREAM_TREE / CONSTANTS_REL).read_bytes(),
        ADOPTED_CONSTANTS.read_bytes(),
        f"a/{CONSTANTS_REL}", f"b/{CONSTANTS_REL}")
    diff_e1 = unified_diff_text(baseline_e1, patched_e1,
                                f"a/{E1_REL}", f"b/{E1_REL}")
    save_text(CANDIDATE_DIR / "constants-candidate.diff", diff_constants)
    save_text(CANDIDATE_DIR / "e1-candidate.diff", diff_e1)

    manifest = {
        "format": "anyhydro-e1-kind-candidate-patch-v1",
        "external_identity": EXTERNAL_IDENTITY,
        "baseline_source": str(UPSTREAM_TREE),
        "baseline_e1_sha256": EXPECTED_BASELINE_E1_SHA256,
        "adopted_constants_source": str(ADOPTED_CONSTANTS),
        "candidate_constants_sha256": EXPECTED_CANDIDATE_CONSTANTS_SHA256,
        "constants_changed_lines": constants["changed_lines"],
        "e1_changed_lines": e1_check["changed_lines"],
        "patched_e1_sha256": e1_check["patched_e1_sha256"],
        "e1_patch": [
            {"line": E1_CHANGED_LINES[0], "before": E1_OLD_COMMENT,
             "after": E1_NEW_COMMENT},
            {"line": E1_CHANGED_LINES[1], "before": E1_OLD_DECL,
             "after": E1_NEW_DECL},
            {"line": E1_CHANGED_LINES[2], "before": E1_OLD_ASSIGN,
             "after": E1_NEW_ASSIGN},
            {"line": E1_CHANGED_LINES[3], "before": E1_OLD_RETURN,
             "after": E1_NEW_RETURN},
        ],
        "preserved": ("exact coefficient literal bytes, approximation "
                      "formulas, branch tests and every other source byte; "
                      "no blanket literal promotion, no approximation "
                      "redesign, no sign/physical/threshold change"),
        "non_claim": ("arithmetic precision can change branch selection at "
                      "rounding boundaries; no identical-path or universal "
                      "accuracy claim"),
        "upstream_verification": upstream,
    }
    save_text(CANDIDATE_DIR / "patch-manifest.json",
              json.dumps(manifest, indent=2) + "\n")

    receipt = {"build_root": BUILD_ROOT,
               "upstream_verification": upstream,
               "candidate_verification": constants,
               "e1_verification": e1_check,
               "candidate_files": {
                   "constants": str(out_constants),
                   "constants_sha256": EXPECTED_CANDIDATE_CONSTANTS_SHA256,
                   "e1": str(out_e1),
                   "e1_sha256": e1_check["patched_e1_sha256"]},
               "patch_manifest": str(CANDIDATE_DIR / "patch-manifest.json")}
    start = time.perf_counter()
    finish_receipt(receipt, "patch", args.attempt, None, None,
                   time.perf_counter() - start, None)
    return 0


def phase_stage(args) -> int:
    require_fresh_receipt("stage", args.attempt)
    require_prerequisite("stage", "patch")
    check_build_root_safety()
    upstream = verify_upstream_tree()
    constants = verify_adopted_constants()
    patched_e1 = (CANDIDATE_DIR / "Delhommeau_integrals.f90").read_bytes()
    e1_check = verify_e1_patch((UPSTREAM_TREE / E1_REL).read_bytes(),
                               patched_e1)
    checklist = SCRIPTS / f"stage-upstream-sha256-checklist-{args.attempt}.txt"
    text, n = checklist_text()
    save_text(checklist, text)

    mnt_upstream = to_wsl_path(UPSTREAM_TREE)
    mnt_constants = to_wsl_path(CANDIDATE_DIR / "Constants.f90")
    mnt_e1 = to_wsl_path(CANDIDATE_DIR / "Delhommeau_integrals.f90")
    mnt_checklist = to_wsl_path(checklist)
    root = shlex.quote(BUILD_ROOT)
    script = "\n".join([
        "set -euo pipefail",
        *thread_env_lines(),
        f"ROOT={root}",
        f'if [ -e "$ROOT" ]; then echo "REFUSED: build root exists: $ROOT"; '
        "exit 3; fi",
        'mkdir -p "$ROOT/upstream" "$ROOT/candidate-src" "$ROOT/wheels" '
        '"$ROOT/runtime-wheels" "$ROOT/dist" "$ROOT/receipts" "$ROOT/logs" '
        '"$ROOT/pip-cache" "$ROOT/ninja-wrap"',
        f"cp -a {shlex.quote(mnt_upstream + '/')} "
        '"$ROOT/upstream/capytaine-3.0.0/"',
        'cd "$ROOT/upstream/capytaine-3.0.0"',
        "echo STAGE_UPSTREAM_VERIFY",
        f"sha256sum -c {shlex.quote(mnt_checklist)} > "
        '"$ROOT/receipts/upstream-tree-verify.txt"',
        'echo "UPSTREAM_FILES=$(find "$ROOT/upstream" -type f | wc -l)"',
        'cp -a "$ROOT/upstream/capytaine-3.0.0" '
        '"$ROOT/candidate-src/capytaine-3.0.0"',
        f"cp {shlex.quote(mnt_constants)} "
        f'"$ROOT/candidate-src/capytaine-3.0.0/{CONSTANTS_REL}"',
        f"cp {shlex.quote(mnt_e1)} "
        f'"$ROOT/candidate-src/capytaine-3.0.0/{E1_REL}"',
        "echo STAGE_CANDIDATE_DIFF",
        'diff -qr "$ROOT/upstream/capytaine-3.0.0" '
        '"$ROOT/candidate-src/capytaine-3.0.0" > '
        '"$ROOT/receipts/candidate-tree-diff.txt" || true',
        f'diff -u "$ROOT/upstream/capytaine-3.0.0/{CONSTANTS_REL}" '
        f'"$ROOT/candidate-src/capytaine-3.0.0/{CONSTANTS_REL}" > '
        '"$ROOT/receipts/constants-candidate-diff.txt" || true',
        f'diff -u "$ROOT/upstream/capytaine-3.0.0/{E1_REL}" '
        f'"$ROOT/candidate-src/capytaine-3.0.0/{E1_REL}" > '
        '"$ROOT/receipts/e1-candidate-diff.txt" || true',
        f'sha256sum "$ROOT/candidate-src/capytaine-3.0.0/{CONSTANTS_REL}" '
        f'"$ROOT/candidate-src/capytaine-3.0.0/{E1_REL}" > '
        '"$ROOT/receipts/candidate-files.sha256"',
        'echo "CANDIDATE_FILES=$(find "$ROOT/candidate-src" -type f | wc -l)"',
        'cat "$ROOT/receipts/candidate-tree-diff.txt"',
        'cat "$ROOT/receipts/candidate-files.sha256"',
        "echo STAGE_DONE",
    ]) + "\n"
    script_path = save_script(f"stage-{args.attempt}.sh", script)

    start = time.perf_counter()
    cmd, proc = run_wsl(args.wsl_distro, script, args.gnu_timeout)
    wall = time.perf_counter() - start

    validation_error = None
    diff_lines = []
    if proc.returncode != 0:
        validation_error = f"WSL child returned {proc.returncode}"
    else:
        diff_lines = [ln for ln in proc.stdout.splitlines()
                      if ln.startswith("Files ") and ln.endswith(" differ")]
        expected_suffixes = (CONSTANTS_REL, E1_REL)
        if len(diff_lines) != 2 or sorted(
                ln[len("Files "):-len(" differ")].split("/")[-1]
                for ln in diff_lines) != sorted(
                s.split("/")[-1] for s in expected_suffixes):
            validation_error = (f"expected exactly two differing files "
                                f"(constants.f90, Delhommeau_integrals.f90), "
                                f"got: {diff_lines}")
        elif f"CANDIDATE_FILES={n}" not in proc.stdout:
            validation_error = "candidate file count mismatch"
        elif "STAGE_UPSTREAM_VERIFY" not in proc.stdout or \
                "STAGE_DONE" not in proc.stdout:
            validation_error = "stage markers missing"
        else:
            want = {EXPECTED_CANDIDATE_CONSTANTS_SHA256,
                    e1_check["patched_e1_sha256"]}
            got = {ln.split()[0] for ln in proc.stdout.splitlines()
                   if re.match(r"^[0-9a-f]{64}  ", ln)}
            if got != want:
                validation_error = (f"candidate file hashes in staged tree "
                                    f"mismatch: {sorted(got)}")

    receipt = {"gnu_timeout_seconds": args.gnu_timeout,
               "build_root": BUILD_ROOT,
               "upstream_verification": upstream,
               "candidate_verification": constants,
               "e1_verification": e1_check,
               "checklist": {"path": str(checklist),
                             "sha256": sha256_bytes(text.encode("utf-8")),
                             "expected_files": n},
               "script": str(script_path),
               "candidate_tree_diff_lines": diff_lines}
    finish_receipt(receipt, "stage", args.attempt, cmd, proc, wall,
                   validation_error)
    return 0 if receipt["ok"] else 1


def bootstrap_venv_lines(label: str) -> list[str]:
    """Bootstrap only an owned fresh venv without system ensurepip or apt.

    The archived pip wheel is verified against its acquisition receipt before
    launch. PYTHONPATH exists for this one bootstrap subprocess only; the
    resulting interpreter then uses its installed pip normally.
    """
    if label not in {"build-env", "install-env"}:
        raise BuildError("unregistered virtual environment label")
    acq = json.loads(BOOTSTRAP_PIP_ACQ.read_text(encoding="utf-8"))
    if acq.get("wheel_sha256") != EXPECTED_BOOTSTRAP_PIP_SHA256:
        raise BuildError("bootstrap pip acquisition receipt SHA mismatch")
    if sha256_file(BOOTSTRAP_PIP_WHEEL) != EXPECTED_BOOTSTRAP_PIP_SHA256:
        raise BuildError("bootstrap pip wheel identity mismatch")
    pip_wheel = shlex.quote(to_wsl_path(BOOTSTRAP_PIP_WHEEL))
    return [
        f'python3 -m venv --without-pip "$ROOT/{label}"',
        f'env PYTHONPATH={pip_wheel} "$ROOT/{label}/bin/python" -m pip '
        f'install --no-index --no-deps {pip_wheel} > '
        f'"$ROOT/logs/bootstrap-{label}-pip.log"',
    ]


def phase_deps(args) -> int:
    require_fresh_receipt("deps", args.attempt)
    require_prerequisite("deps", "stage")
    check_build_root_safety()
    verified = verify_wheel_sums(DEP_WHEELS_SUMS, DEP_WHEELS_DIR)
    locked = verify_lock_names(BUILD_LOCK, verified)
    if sha256_file(BOOTSTRAP_PIP_WHEEL) != EXPECTED_BOOTSTRAP_PIP_SHA256:
        raise BuildError("bootstrap pip wheel identity mismatch")

    mnt_wheels = to_wsl_path(DEP_WHEELS_DIR)
    root = shlex.quote(BUILD_ROOT)
    install_specs = " ".join(shlex.quote(f"{n}=={v}") for n, v in
                              sorted(EXPECTED_BUILD_DEPS.items()))
    script = "\n".join([
        "set -euo pipefail",
        *thread_env_lines(),
        f"ROOT={root}",
        'if [ -e "$ROOT/build-env" ]; then echo '
        '"REFUSED: build-env exists"; exit 3; fi',
        'if [ -n "$(ls -A "$ROOT/wheels" 2>/dev/null)" ]; then echo '
        '"REFUSED: wheels dir not empty"; exit 3; fi',
        "python3 --version",
        f'cp {shlex.quote(mnt_wheels)}/*.whl "$ROOT/wheels/"',
        'sha256sum "$ROOT"/wheels/* | tee '
        '"$ROOT/receipts/dependency-wheels.sha256"',
        *bootstrap_venv_lines("build-env"),
        '"$ROOT/build-env/bin/pip" install --no-index '
        '--find-links="$ROOT/wheels" --cache-dir "$ROOT/pip-cache" '
        f"{install_specs} > "
        '"$ROOT/logs/pip-install-build-deps.log"',
        '"$ROOT/build-env/bin/pip" freeze | tee '
        '"$ROOT/receipts/build-env-freeze.txt"',
        '"$ROOT/build-env/bin/python" - > "$ROOT/receipts/build-toolchain'
        '.json" <<\'PYEOF\'',
        "import json, platform, sys",
        "import numpy, mesonpy",
        "print(json.dumps({'python_version': sys.version, "
        "'platform': platform.platform(), 'numpy_version': numpy.__version__, "
        "'meson_python_version': mesonpy.__version__}))",
        "PYEOF",
        'cat "$ROOT/receipts/build-toolchain.json"',
        '"$ROOT/build-env/bin/meson" --version',
        '"$ROOT/build-env/bin/ninja" --version',
        "gfortran --version | head -1",
        "gcc --version | head -1",
        "echo DEPS_DONE",
    ]) + "\n"
    script_path = save_script(f"deps-{args.attempt}.sh", script)

    start = time.perf_counter()
    cmd, proc = run_wsl(args.wsl_distro, script, args.gnu_timeout)
    wall = time.perf_counter() - start

    validation_error = None
    toolchain = None
    if proc.returncode != 0:
        validation_error = f"WSL child returned {proc.returncode}"
    elif "DEPS_DONE" not in proc.stdout:
        validation_error = "deps markers missing"
    else:
        m = re.search(r"^\{.*\}$", proc.stdout, re.M | re.S)
        if m:
            try:
                toolchain = json.loads(m.group(0))
            except ValueError:
                toolchain = None
        if not toolchain or \
                not toolchain.get("numpy_version", "").startswith("2"):
            validation_error = f"unexpected toolchain receipt: {toolchain}"
        else:
            got = {}
            for line in proc.stdout.splitlines():
                mm = re.match(r"^([A-Za-z0-9._-]+)==([^\s]+)$", line)
                if mm:
                    got[mm.group(1).lower()] = mm.group(2)
            for name, version in EXPECTED_BUILD_DEPS.items():
                if got.get(name.lower()) != version:
                    validation_error = (f"build env freeze missing locked "
                                        f"{name}=={version}: freeze has "
                                        f"{sorted(got.items())}")
                    break
    receipt = {"gnu_timeout_seconds": args.gnu_timeout,
               "build_root": BUILD_ROOT,
               "script": str(script_path),
               "dependency_acquisition": {
                   "mode": ("archived offline wheels only; no network, no "
                            "downloads, no resolver choices beyond the "
                            "archived build-requirements.lock"),
                   "lock": str(BUILD_LOCK),
                   "lock_sha256": sha256_file(BUILD_LOCK),
                   "archived_wheels_sha256": verified,
                   "locked_requirements": locked,
                   "bootstrap_pip_wheel": str(BOOTSTRAP_PIP_WHEEL),
                   "bootstrap_pip_sha256": EXPECTED_BOOTSTRAP_PIP_SHA256,
               },
               "toolchain": toolchain}
    finish_receipt(receipt, "deps", args.attempt, cmd, proc, wall,
                   validation_error)
    return 0 if receipt["ok"] else 1


WHEEL_INSPECTOR = """\
import base64, csv, hashlib, io, json, sys, zipfile
w = sys.argv[1]
z = zipfile.ZipFile(w)
names = z.namelist()
record_name = next(n for n in names if n.endswith(".dist-info/RECORD"))
rows = list(csv.reader(io.TextIOWrapper(z.open(record_name), "utf-8")))
record = []
mismatches = []
for path, digest, size in rows:
    if path in (record_name, record_name + ".jws") or not path:
        record.append({"path": path, "verified": "self-excluded"})
        continue
    if digest.startswith("sha256="):
        digest = digest[len("sha256="):]
    data = z.read(path)
    actual = base64.urlsafe_b64encode(
        hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    ok = (actual == digest) and (str(len(data)) == size)
    if not ok:
        mismatches.append({"path": path, "expected": digest,
                           "actual": actual, "size": size,
                           "actual_size": len(data)})
    record.append({"path": path, "sha256": actual, "bytes": len(data),
                   "verified": ok})
about = z.read("capytaine/__about__.py").decode("utf-8")
print(json.dumps({"wheel": w, "names": names, "record": record,
                  "record_mismatches": mismatches,
                  "about_py": about}))
"""


def phase_build(args) -> int:
    require_fresh_receipt("build", args.attempt)
    require_prerequisite("build", "deps")
    check_build_root_safety()
    inspector = SCRIPTS / f"wheel-inspect-{args.attempt}.py"
    save_text(inspector, WHEEL_INSPECTOR)
    root = shlex.quote(BUILD_ROOT)
    mnt_inspector = to_wsl_path(inspector)
    script = "\n".join([
        "set -euo pipefail",
        *thread_env_lines(),
        f"ROOT={root}",
        'if [ -n "$(ls -A "$ROOT/dist" 2>/dev/null)" ]; then echo '
        '"REFUSED: dist dir not empty (no silent wheel reuse)"; exit 3; fi',
        'mkdir -p "$ROOT/ninja-wrap"',
        "printf '%s\\n' '#!/bin/bash' "
        '\'D="$(cd "$(dirname "$0")/.." && pwd)"\' '
        '\'echo "ninja_argv: $*" >> "$D/receipts/ninja-args.log"\' '
        '\'exec "$D/build-env/bin/ninja" "$@"\' > "$ROOT/ninja-wrap/ninja"',
        'chmod +x "$ROOT/ninja-wrap/ninja"',
        'export PATH="$ROOT/ninja-wrap:$ROOT/build-env/bin:$PATH"',
        "echo EFFECTIVE_ENV",
        "env | grep -E '^(OMP|OPENBLAS|MKL|NUMEXPR)_NUM_THREADS=' | sort",
        "echo EFFECTIVE_TOOLS",
        "which ninja meson gfortran pip",
        "ninja --version",
        "meson --version",
        "gfortran --version | head -1",
        'cd "$ROOT/candidate-src/capytaine-3.0.0"',
        "echo PIP_WHEEL_START",
        'pip wheel --no-deps --no-build-isolation --wheel-dir "$ROOT/dist" '
        '--config-settings=compile-args=-j1 . 2>&1 | tee '
        '"$ROOT/logs/pip-wheel.log"',
        "echo PIP_WHEEL_DONE",
        'ls -l "$ROOT/dist"',
        'sha256sum "$ROOT"/dist/*.whl | tee "$ROOT/receipts/wheel.sha256"',
        f'"$ROOT/build-env/bin/python" {shlex.quote(mnt_inspector)} '
        '"$ROOT"/dist/*.whl > "$ROOT/receipts/wheel-contents.json"',
        f'cp "$ROOT/receipts/wheel-contents.json" '
        f'{shlex.quote(to_wsl_path(RESULTS))}/build-wheel-contents-'
        f'{args.attempt}.json',
        f'cp "$ROOT/receipts/ninja-args.log" '
        f'{shlex.quote(to_wsl_path(RESULTS))}/build-ninja-args-'
        f'{args.attempt}.log',
        "echo BUILD_DONE",
    ]) + "\n"
    script_path = save_script(f"build-{args.attempt}.sh", script)

    start = time.perf_counter()
    cmd, proc = run_wsl(args.wsl_distro, script, args.gnu_timeout)
    wall = time.perf_counter() - start

    validation_error = None
    wheel_sha = None
    wheel_name = None
    about = None
    record_mismatches = None
    ninja_args = []
    if proc.returncode != 0:
        validation_error = (f"WSL child returned {proc.returncode} "
                            f"(partial output preserved)")
    elif "BUILD_DONE" not in proc.stdout:
        validation_error = "build did not reach BUILD_DONE"
    else:
        m = re.search(r"^([0-9a-f]{64})  \S*/(\S+\.whl)\s*$", proc.stdout,
                      re.M)
        if not m:
            validation_error = "wheel SHA line missing"
        else:
            wheel_sha, wheel_name = m.group(1), m.group(2)
            contents_path = (RESULTS /
                             f"build-wheel-contents-{args.attempt}.json")
            if contents_path.is_file():
                contents = json.loads(
                    contents_path.read_text(encoding="utf-8"))
                about = contents.get("about_py")
                record_mismatches = contents.get("record_mismatches")
            if not wheel_name.startswith("capytaine-3.0.0-"):
                validation_error = f"unexpected wheel name {wheel_name}"
            elif record_mismatches:
                validation_error = (f"wheel RECORD verification failed: "
                                    f"{record_mismatches[:3]}")
            else:
                try:
                    validate_built_version(about)
                except (BuildError, SyntaxError, ValueError) as exc:
                    validation_error = str(exc)
    if validation_error is None:
        ninja_log_path = RESULTS / f"build-ninja-args-{args.attempt}.log"
        if not ninja_log_path.is_file():
            validation_error = ("ninja argv evidence log missing "
                                "(meson-python may not invoke ninja via PATH)")
        else:
            ninja_args = [ln for ln in ninja_log_path.read_text(
                encoding="utf-8").splitlines() if ln.strip()]
            if not any("-j1" in ln for ln in ninja_args):
                validation_error = (f"ninja -j1 not proven in actual ninja "
                                    f"argv: {ninja_args}")
            elif any(re.search(r"-j\s*[2-9]", ln) for ln in ninja_args):
                validation_error = (f"parallel ninja job argument found: "
                                    f"{ninja_args}")
    receipt = {"gnu_timeout_seconds": args.gnu_timeout,
               "build_root": BUILD_ROOT,
               "script": str(script_path),
               "inspector": str(inspector),
               "build_route": ("pip wheel (meson-python backend, "
                               "--no-build-isolation, offline locked deps) "
                               "with --config-settings=compile-args=-j1; "
                               "f2py from the installed numpy; full log in "
                               "receipts/build-root-logs/pip-wheel.log"),
               "ninja_args_proven": ninja_args,
               "wheel_name": wheel_name,
               "wheel_sha256": wheel_sha,
               "wheel_record_verified": record_mismatches == [],
               "built_about_py": about,
               "compiler": ("gfortran (WSL Ubuntu-24.04 system compiler; "
                            "exact version line in stdout)")}
    finish_receipt(receipt, "build", args.attempt, cmd, proc, wall,
                   validation_error)
    return 0 if receipt["ok"] else 1


RUNTIME_METADATA = '''\
"""Imported metadata only. Never constructs a Green function or calls a kernel."""
import base64, csv, hashlib, importlib, importlib.metadata, io, json, os, \
    platform, subprocess, sys, zipfile
from pathlib import Path
import capytaine as cpt
import numpy, scipy.linalg, pandas, xarray
from threadpoolctl import threadpool_info

site = Path(sys.argv[1])
wheel = Path(sys.argv[2])
out_path = Path(sys.argv[3])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


native = importlib.import_module(
    "capytaine.green_functions.Delhommeau_float64")
pools = threadpool_info()
for pool in pools:
    assert pool["num_threads"] == 1, pool
    pool["sha256"] = sha(pool["filepath"])
assert int(native.floating_point_precision.pre) == 8

z = zipfile.ZipFile(wheel)
record_name = next(n for n in z.namelist()
                   if n.endswith(".dist-info/RECORD"))
rows = list(csv.reader(io.TextIOWrapper(z.open(record_name), "utf-8")))
installed_mismatches = []
verified_installed = 0
for path, digest, size in rows:
    if path in (record_name, record_name + ".jws") or not path:
        continue
    if digest.startswith("sha256="):
        digest = digest[len("sha256="):]
    installed = site / path
    if not installed.is_file():
        installed_mismatches.append({"path": path, "missing": True})
        continue
    data = installed.read_bytes()
    actual = base64.urlsafe_b64encode(
        hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    if actual != digest or str(len(data)) != size:
        installed_mismatches.append({"path": path, "expected": digest,
                                      "actual": actual})
    else:
        verified_installed += 1
assert not installed_mismatches, installed_mismatches[:3]

linked = {}
for so in sorted((site / "capytaine" / "green_functions")
                 .glob("Delhommeau_float*.so")):
    ldd = subprocess.run(["ldd", str(so)], capture_output=True, text=True)
    libs = []
    for line in ldd.stdout.splitlines():
        part = line.split("=>")
        if len(part) == 2 and "not found" not in line:
            lib = part[1].split("(")[0].strip()
            if lib and Path(lib).is_file():
                libs.append({"library": lib, "sha256": sha(lib)})
    linked[so.name] = {"sha256": sha(so), "linked_libraries": libs}

record = {
    "classification": ("complete isolated public package import; "
                      "metadata only; zero GF/kernel/table/fit/system "
                      "calls"),
    "executable": sys.executable,
    "executable_sha256": sha(sys.executable),
    "prefix": sys.prefix,
    "python": sys.version,
    "platform": platform.platform(),
    "package_file": cpt.__file__,
    "package_version": cpt.__version__,
    "green_function_source": cpt.Delhommeau.__module__,
    "native_file": native.__file__,
    "native_sha256": sha(native.__file__),
    "pre_kind": int(native.floating_point_precision.pre),
    "pi": float(native.constants.pi),
    "euler_gamma": float(native.constants.euler_gamma),
    "packages": {d.metadata["Name"]: d.version
                 for d in importlib.metadata.distributions()},
    "pools": pools,
    "thread_env": {k: os.environ.get(k) for k in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS")},
    "installed_bytes_verified_against_wheel_record": verified_installed,
    "installed_mismatches": installed_mismatches,
    "native_modules": linked,
}
out_path.write_text(json.dumps(record, indent=2) + "\\n", encoding="utf-8")
print(json.dumps({"public_import": True,
                  "packages": len(record["packages"]),
                  "actual_pools": len(pools),
                  "pre_kind": record["pre_kind"],
                  "native_sha256": record["native_sha256"],
                  "installed_verified": verified_installed}))
'''


def phase_install(args) -> int:
    require_fresh_receipt("install", args.attempt)
    require_prerequisite("install", "build")
    check_build_root_safety()
    verified = verify_wheel_sums(RUNTIME_WHEELS_SUMS, RUNTIME_WHEELS_DIR,
                                 exclude="capytaine-3.0.0")
    locked = verify_lock_names(RUNTIME_LOCK, verified)
    metadata_script = SCRIPTS / f"runtime-metadata-{args.attempt}.py"
    save_text(metadata_script, RUNTIME_METADATA)
    mnt_metadata = to_wsl_path(metadata_script)
    mnt_runtime_wheels = to_wsl_path(RUNTIME_WHEELS_DIR)
    root = shlex.quote(BUILD_ROOT)
    runtime_specs = " ".join(shlex.quote(f"{n}=={v}") for n, v in
                             sorted(EXPECTED_RUNTIME_DEPS.items()))
    script = "\n".join([
        "set -euo pipefail",
        *thread_env_lines(),
        f"ROOT={root}",
        'if [ -e "$ROOT/install-env" ]; then echo '
        '"REFUSED: install-env exists"; exit 3; fi',
        'WHEEL="$(ls "$ROOT"/dist/*.whl)"',
        'test "$(ls "$ROOT"/dist/*.whl | wc -l)" -eq 1',
        f'cp {shlex.quote(mnt_runtime_wheels)}/*.whl "$ROOT/runtime-wheels/"',
        'rm -f "$ROOT/runtime-wheels"/capytaine-3.0.0-*.whl',
        'sha256sum "$ROOT"/runtime-wheels/* | tee '
        '"$ROOT/receipts/runtime-wheels.sha256"',
        *bootstrap_venv_lines("install-env"),
        '"$ROOT/install-env/bin/pip" install --no-index '
        '--find-links="$ROOT/runtime-wheels" --cache-dir "$ROOT/pip-cache" '
        f"{runtime_specs} > "
        '"$ROOT/logs/pip-install-runtime-deps.log"',
        '"$ROOT/install-env/bin/pip" install --no-deps --no-index "$WHEEL" '
        '> "$ROOT/logs/pip-install-wheel.log"',
        '"$ROOT/install-env/bin/pip" check | tee '
        '"$ROOT/receipts/pip-check.txt"',
        '"$ROOT/install-env/bin/pip" freeze | tee '
        '"$ROOT/receipts/runtime-freeze.txt"',
        'SITE="$("$ROOT/install-env/bin/python" -c '
        '"import sysconfig; print(sysconfig.get_paths()[\'purelib\'])")"',
        'echo SITE_PACKAGES="$SITE"',
        f'"$ROOT/install-env/bin/python" {shlex.quote(mnt_metadata)} '
        '"$SITE" "$WHEEL" "$ROOT/receipts/runtime-identity.json"',
        f'cp "$ROOT/receipts/runtime-identity.json" '
        f'{shlex.quote(to_wsl_path(RESULTS))}/install-runtime-identity-'
        f'{args.attempt}.json',
        'cat "$ROOT/receipts/runtime-identity.json"',
        "echo INSTALL_DONE",
    ]) + "\n"
    script_path = save_script(f"install-{args.attempt}.sh", script)

    start = time.perf_counter()
    cmd, proc = run_wsl(args.wsl_distro, script, args.gnu_timeout)
    wall = time.perf_counter() - start

    validation_error = None
    runtime = None
    if proc.returncode != 0:
        validation_error = f"WSL child returned {proc.returncode}"
    elif "INSTALL_DONE" not in proc.stdout:
        validation_error = "install did not reach INSTALL_DONE"
    else:
        identity_path = (RESULTS /
                         f"install-runtime-identity-{args.attempt}.json")
        if not identity_path.is_file():
            validation_error = "runtime identity JSON missing"
        else:
            runtime = json.loads(identity_path.read_text(encoding="utf-8"))
            mods = runtime.get("native_modules") or {}
            f64 = next((m for k, m in mods.items()
                        if "Delhommeau_float64" in k), None)
            f32 = next((m for k, m in mods.items()
                        if "Delhommeau_float32" in k), None)
            pi = runtime.get("pi")
            gamma = runtime.get("euler_gamma")
            if f64 is None or f32 is None:
                validation_error = (f"both Delhommeau extensions required, "
                                    f"found: {sorted(mods)}")
            elif not _is_sha256(f64.get("sha256")) or \
                    not _is_sha256(f32.get("sha256")):
                validation_error = "extension SHA-256 hashes missing"
            elif runtime.get("pre_kind") != 8:
                validation_error = (f"PRE kind is not 8: "
                                    f"{runtime.get('pre_kind')}")
            elif not isinstance(pi, float) or pi != math.pi:
                validation_error = f"native PI mismatch: {pi!r}"
            elif not isinstance(gamma, float) or \
                    abs(gamma - CANDIDATE_GAMMA_REFERENCE) > 1e-15:
                validation_error = f"native EULER_GAMMA mismatch: {gamma!r}"
            elif runtime.get("installed_mismatches"):
                validation_error = (f"installed package bytes differ from "
                                    f"wheel RECORD: "
                                    f"{runtime['installed_mismatches'][:3]}")
            elif not runtime.get("pools") or \
                    not all(p.get("num_threads") == 1
                            for p in runtime["pools"]):
                validation_error = (f"threadpoolctl pools not all 1: "
                                    f"{runtime.get('pools')}")
            elif runtime.get("package_version") != UPSTREAM_SEMANTIC_VERSION:
                validation_error = (f"imported capytaine version "
                                    f"{runtime.get('package_version')!r} "
                                    f"is not 3.0.0")
            else:
                got = {k.lower(): v for k, v in
                       (runtime.get("packages") or {}).items()}
                for name, version in EXPECTED_RUNTIME_DEPS.items():
                    if got.get(name.lower()) != version:
                        validation_error = (f"runtime missing locked "
                                            f"{name}=={version}: has "
                                            f"{sorted(got.items())}")
                        break
                else:
                    if "capytaine" not in got:
                        validation_error = ("capytaine not installed in "
                                            "runtime")
    receipt = {"gnu_timeout_seconds": args.gnu_timeout,
               "build_root": BUILD_ROOT,
               "script": str(script_path),
               "metadata_script": str(metadata_script),
               "isolated_install_env": (f"{BUILD_ROOT}/install-env (fresh "
                                        "complete runtime venv built from "
                                        "the archived runtime wheels; the "
                                        "baseline capytaine wheel in the "
                                        "archive is explicitly excluded and "
                                        "never installed; the new candidate "
                                        "wheel is installed --no-deps "
                                        "--no-index; never an existing "
                                        "application, Windows or old-Linux "
                                        "runtime)"),
               "runtime_inputs": {
                   "wheels_dir": str(RUNTIME_WHEELS_DIR),
                   "lock": str(RUNTIME_LOCK),
                   "lock_sha256": sha256_file(RUNTIME_LOCK),
                   "archived_wheels_sha256": verified,
                   "locked_requirements": locked,
                   "excluded_baseline_wheel": ("capytaine-3.0.0-cp312-cp312-"
                                                "linux_x86_64.whl (baseline "
                                                "artifact; read-only, not "
                                                "installed)"),
               },
               "runtime_identity": runtime}
    finish_receipt(receipt, "install", args.attempt, cmd, proc, wall,
                   validation_error)
    return 0 if receipt["ok"] else 1


def _is_sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) \
        is not None


def phase_collect(args) -> int:
    require_fresh_receipt("collect", args.attempt)
    for prereq in ("patch", "stage", "deps", "build", "install"):
        require_prerequisite("collect", prereq)
    check_build_root_safety()
    root = shlex.quote(BUILD_ROOT)
    mnt_artifacts = to_wsl_path(ARTIFACTS)
    mnt_receipts = to_wsl_path(ROOT_RECEIPTS)
    script = "\n".join([
        "set -euo pipefail",
        *thread_env_lines(),
        f"ROOT={root}",
        f"ART={shlex.quote(mnt_artifacts)}",
        f"RCPT={shlex.quote(mnt_receipts)}",
        'if [ -e "$RCPT/build-root-receipts" ]; then echo '
        '"REFUSED: build-root receipts already collected"; exit 3; fi',
        'if [ -e "$ART/build-root-logs" ]; then echo '
        '"REFUSED: build-root logs already collected"; exit 3; fi',
        'mkdir -p "$ART" "$RCPT"',
        'cp -a "$ROOT/receipts" "$RCPT/build-root-receipts"',
        'cp -a "$ROOT/logs" "$ART/build-root-logs"',
        'cp "$ROOT"/dist/*.whl "$ART/"',
        'sha256sum "$ART"/*.whl',
        'du -sh "$ART" "$RCPT"',
        "echo COLLECT_DONE",
    ]) + "\n"
    script_path = save_script(f"collect-{args.attempt}.sh", script)

    start = time.perf_counter()
    cmd, proc = run_wsl(args.wsl_distro, script, args.gnu_timeout)
    wall = time.perf_counter() - start
    validation_error = None
    if proc.returncode != 0:
        validation_error = f"WSL child returned {proc.returncode}"
    elif "COLLECT_DONE" not in proc.stdout:
        validation_error = "collect did not reach COLLECT_DONE"
    receipt = {"gnu_timeout_seconds": args.gnu_timeout,
               "script": str(script_path),
               "artifacts_dir": str(ARTIFACTS),
               "root_receipts_dir": str(ROOT_RECEIPTS)}
    finish_receipt(receipt, "collect", args.attempt, cmd, proc, wall,
                   validation_error)
    return 0 if receipt["ok"] else 1


def candidate_tree_hash() -> str:
    """Deterministic hash over the staged candidate source tree identity."""
    stage = json.loads(find_ok_receipt("stage").read_text(encoding="utf-8"))
    payload = json.dumps({
        "external_identity": EXTERNAL_IDENTITY,
        "upstream_archive_sha256": EXPECTED_ARCHIVE_SHA256,
        "candidate_constants_sha256": EXPECTED_CANDIDATE_CONSTANTS_SHA256,
        "patched_e1_sha256": (stage.get("e1_verification") or {})
        .get("patched_e1_sha256"),
        "differing_files": sorted(
            stage.get("candidate_tree_diff_lines") or []),
    }, sort_keys=True)
    return sha256_bytes(payload.encode("utf-8"))


def phase_manifest(args) -> int:
    require_fresh_receipt("manifest", args.attempt)
    for prereq in ("patch", "stage", "deps", "build", "install", "collect"):
        require_prerequisite("manifest", prereq)
    receipts = {}
    for phase in ("patch", "stage", "deps", "build", "install", "collect"):
        p = find_ok_receipt(phase)
        if p is None:
            raise BuildError(f"missing ok receipt for {phase}")
        receipts[phase] = p
    data = {phase: json.loads(p.read_text(encoding="utf-8"))
            for phase, p in receipts.items()}
    build = data["build"]
    runtime = data["install"].get("runtime_identity") or {}
    mods = runtime.get("native_modules") or {}
    wheel_path = ARTIFACTS / build["wheel_name"]
    if not wheel_path.is_file():
        raise BuildError(f"collected wheel missing: {wheel_path}")
    if sha256_file(wheel_path) != build["wheel_sha256"]:
        raise BuildError("collected wheel SHA mismatch vs build receipt")

    manifest = {
        "format": "anyhydro-e1-kind-backend-manifest-v1",
        "tool": f"tools/native_precision/{TOOL}.py",
        "tool_version": TOOL_VERSION,
        "external_identity": EXTERNAL_IDENTITY,
        "upstream_semantic_version": UPSTREAM_SEMANTIC_VERSION,
        "scope": ("artifact development verification only; not "
                  "Green/kernel/ABI/scientific/platform qualification"),
        "source": {
            "archive": str(ARCHIVE),
            "archive_sha256": EXPECTED_ARCHIVE_SHA256,
            "upstream_tree": str(UPSTREAM_TREE),
            "verified_files": (data["stage"].get("upstream_verification")
                               or {}).get("verified_files"),
            "baseline_e1_sha256": EXPECTED_BASELINE_E1_SHA256,
            "candidate_constants": str(ADOPTED_CONSTANTS),
            "candidate_constants_sha256": EXPECTED_CANDIDATE_CONSTANTS_SHA256,
            "only_changes": (f"{CONSTANTS_REL} lines 26 (PI) and 28 "
                             f"(EULER_GAMMA); {E1_REL} exp_e1 lines "
                             f"{list(E1_CHANGED_LINES)} (precision comment, "
                             "complex(kind=pre) temporaries, direct z=zz "
                             "input, direct exp_e1=g return); all other "
                             "source bytes identical to the official sdist; "
                             "Apache-2.0 headers preserved; upstream default "
                             "meson.options hams_green_function=enabled "
                             "compiles the optional LGPL FinGreen3D and "
                             "LiangWuNoblesse sources as upstream does"),
        },
        "patch": data["patch"].get("e1_verification"),
        "build": {
            "route": build.get("build_route"),
            "platform": "WSL Ubuntu-24.04 linux-x86_64",
            "build_root": BUILD_ROOT,
            "compiler": build.get("compiler"),
            "one_job": ("ninja -j1 via meson-python compile-args config "
                        "setting, proven by the actual ninja argv recorded "
                        "in build-root receipts; OMP/OPENBLAS/MKL/"
                        "NUMEXPR_NUM_THREADS=1 exported inside the WSL "
                        "shell and echoed at runtime"),
            "ninja_args_proven": build.get("ninja_args_proven"),
            "wheel_name": build.get("wheel_name"),
            "wheel_sha256": build.get("wheel_sha256"),
            "wheel_record_verified": build.get("wheel_record_verified"),
            "built_about_py": build.get("built_about_py"),
        },
        "dependencies": {
            "build": data["deps"].get("dependency_acquisition"),
            "runtime": data["install"].get("runtime_inputs"),
        },
        "runtime": {
            "isolated_env": data["install"].get("isolated_install_env"),
            "executable": runtime.get("executable"),
            "executable_sha256": runtime.get("executable_sha256"),
            "python": runtime.get("python"),
            "platform": runtime.get("platform"),
            "package_version": runtime.get("package_version"),
            "packages": runtime.get("packages"),
            "pre_kind": runtime.get("pre_kind"),
            "pi": runtime.get("pi"),
            "euler_gamma": runtime.get("euler_gamma"),
            "threadpoolctl_pools": runtime.get("pools"),
            "thread_env": runtime.get("thread_env"),
            "installed_bytes_verified_against_wheel_record":
                runtime.get("installed_bytes_verified_against_wheel_record"),
            "native_modules": {
                k: {"sha256": m.get("sha256"),
                    "linked_libraries": m.get("linked_libraries")}
                for k, m in mods.items()},
            "classification": runtime.get("classification"),
        },
        "receipts": {k: str(v) for k, v in receipts.items()},
        "reproduction_recipe": [
            f"1. patch:    python tools/native_precision/{TOOL}.py patch "
            "--attempt <name>",
            f"2. stage:    python tools/native_precision/{TOOL}.py stage "
            "--attempt <name>",
            f"3. deps:     python tools/native_precision/{TOOL}.py deps "
            "--attempt <name>",
            f"4. build:    python tools/native_precision/{TOOL}.py build "
            "--attempt <name>",
            f"5. install:  python tools/native_precision/{TOOL}.py install "
            "--attempt <name>",
            f"6. collect:  python tools/native_precision/{TOOL}.py collect "
            "--attempt <name>",
            f"7. manifest: python tools/native_precision/{TOOL}.py manifest "
            "--attempt <name>",
            "Each phase runs through the workspace ordinary accounting "
            "runner; WSL phases run one GNU-timeout-bounded bash script "
            "(saved under scripts/) with base64 transport; exact argv, full "
            "stdout/stderr and wall time are in the per-phase receipts and "
            "the runner attempt logs. No bit-identical wheel claim is made "
            "(build-generated timestamps in capytaine/__about__.py).",
        ],
        "non_claims": [
            "not Green-function, kernel, ABI, scientific or platform "
            "qualification",
            "arithmetic precision can change branch selection at rounding "
            "boundaries; no identical-path or universal accuracy claim",
            "no current application runtime changed; nothing installed into "
            "any existing Windows or old-Linux runtime",
            "no blanket literal promotion, no approximation redesign, no "
            "sign/physical/threshold change; coefficient literal bytes, "
            "formulas and branch tests preserved",
            "upstream Apache/GPL-3 generated metadata discrepancy preserved "
            "unchanged; no third-source license edit or relicense",
            "import/constants metadata does not confer numerical authority; "
            "admitted_for_numerical_execution is false",
        ],
    }
    manifest_path = RESULTS / "build-manifest.json"
    if manifest_path.exists():
        raise BuildError(f"refusing to overwrite existing manifest "
                         f"{manifest_path}")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                             encoding="utf-8")

    runtime_identity_copy = (ROOT_RECEIPTS / "build-root-receipts" /
                             "runtime-identity.json")
    sidecar = {
        "format": "anyhydro-backend-artifact-identity-v1",
        "external_identity": EXTERNAL_IDENTITY,
        "wheel": build["wheel_name"],
        "wheel_sha256": build["wheel_sha256"],
        "native_extension_sha256": {
            f"capytaine/green_functions/{name}": m.get("sha256")
            for name, m in sorted(mods.items())},
        "candidate_source_tree_sha256": candidate_tree_hash(),
        "build_manifest_sha256": sha256_file(manifest_path),
        "runtime_identity_sha256": (sha256_file(runtime_identity_copy)
                                    if runtime_identity_copy.is_file()
                                    else None),
        "admitted_for_numerical_execution": False,
        "transfer_contract": ("Always transfer with this sidecar and the "
                              "hashed manifest/runtime identities. Filename/"
                              "version alone cannot admit this artifact. A "
                              "new scoped native admission is required "
                              "before any numerical execution."),
    }
    sidecar_path = ARTIFACTS / f"{build['wheel_name']}.anyhydro-backend.json"
    if sidecar_path.exists():
        raise BuildError(f"refusing to overwrite existing sidecar "
                         f"{sidecar_path}")
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n",
                            encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path),
                      "sidecar": str(sidecar_path), "ok": True}, indent=2))
    return 0


def phase_cleanenv(args) -> int:
    """Remove only partial environments left by a failed attempt.

    Repair within ownership: touches nothing outside the owned BUILD_ROOT,
    never the immutable upstream/candidate copies or collected evidence.
    """
    require_fresh_receipt("cleanenv", args.attempt)
    check_build_root_safety()
    root = shlex.quote(BUILD_ROOT)
    script = "\n".join([
        "set -euo pipefail",
        *thread_env_lines(),
        f"ROOT={root}",
        'if [ ! -d "$ROOT" ]; then echo "REFUSED: owned build root '
        'missing"; exit 3; fi',
        'rm -rf "$ROOT/build-env" "$ROOT/install-env" "$ROOT/ninja-wrap"',
        'find "$ROOT/wheels" "$ROOT/runtime-wheels" "$ROOT/dist" '
        '"$ROOT/pip-cache" -mindepth 1 -maxdepth 1 -delete 2>/dev/null '
        "|| true",
        'echo "REMAINING=$(find "$ROOT" -mindepth 1 -maxdepth 1 | '
        'sort | tr \'\\n\' \' \')"',
        "echo CLEANENV_DONE",
    ]) + "\n"
    script_path = save_script(f"cleanenv-{args.attempt}.sh", script)
    start = time.perf_counter()
    cmd, proc = run_wsl(args.wsl_distro, script, args.gnu_timeout)
    wall = time.perf_counter() - start
    validation_error = None
    if proc.returncode != 0:
        validation_error = f"WSL child returned {proc.returncode}"
    elif "CLEANENV_DONE" not in proc.stdout:
        validation_error = "cleanenv did not reach CLEANENV_DONE"
    receipt = {"gnu_timeout_seconds": args.gnu_timeout,
               "build_root": BUILD_ROOT,
               "script": str(script_path),
               "scope": ("owned build root only; removes partial venvs and "
                         "wheel/dist/cache contents left by a failed "
                         "attempt; upstream and candidate-src trees are "
                         "preserved")}
    finish_receipt(receipt, "cleanenv", args.attempt, cmd, proc, wall,
                   validation_error)
    return 0 if receipt["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=PHASES)
    ap.add_argument("--attempt", required=True,
                    help="unique attempt label for this phase")
    ap.add_argument("--wsl-distro", default=WSL_DISTRO_DEFAULT)
    ap.add_argument("--gnu-timeout", type=int, default=None)
    args = ap.parse_args(argv)
    default = DEFAULT_GNU_TIMEOUTS[args.phase]
    if args.gnu_timeout is None:
        args.gnu_timeout = default
    if args.phase in ("patch", "manifest"):
        if args.gnu_timeout not in (0, default):
            raise BuildError(f"{args.phase} is Windows-local; no GNU timeout")
    elif args.gnu_timeout > MAX_GNU_TIMEOUT or args.gnu_timeout <= 0:
        raise BuildError(f"GNU timeout must be in (0, {MAX_GNU_TIMEOUT}] s")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.attempt):
        raise BuildError(f"invalid attempt label {args.attempt!r}")
    return {
        "patch": phase_patch,
        "stage": phase_stage,
        "deps": phase_deps,
        "build": phase_build,
        "install": phase_install,
        "collect": phase_collect,
        "manifest": phase_manifest,
        "cleanenv": phase_cleanenv,
    }[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
