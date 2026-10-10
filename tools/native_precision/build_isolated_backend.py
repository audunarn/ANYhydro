# SPDX-License-Identifier: MPL-2.0
"""Central reproducible isolated native backend build/provenance tool.

Builds a complete, distinctly identified native Capytaine 3.0.0 wheel from the
exact official sdist archived in this evidence directory, applying ONLY the
adopted two-line PI/EULER_GAMMA precision candidate to
libs/libDelhommeau/src/constants.f90, in a fresh owned WSL build root, and
proves the installed native PRE-kind constants by direct extension loading.

This is artifact development verification only. It is NOT Green-function,
ABI, kernel, scientific or platform qualification. Zero GF constructors,
kernel evaluations, table generations, Prony fitting, systems, HAMS, dynamics
or campaigns are invoked; the extension is loaded directly from its file path
so capytaine/__init__.py (heavy imports) never executes, and Fortran module
specification parts contain no executable statements, so the f2py module init
registers procedures without calling any of them.

Safety contract:
- Every phase is one bounded WSL child launched as argv
  ['wsl.exe', '-d', <distro>, '--', 'timeout', '--kill-after=3s', '<t>s',
   'bash', '-c', <raw script>] with GNU timeout <= 285 s, one job/thread
  (OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 exported inside the WSL shell,
  ninja driven with -j1 via the meson-python compile-args config setting).
- The immutable upstream release tree, archive and acquisition record are
  verified by SHA-256 before anything is created and never written to; the
  candidate tree is a separate copy whose only difference is the two-line
  constants.f90 candidate, verified by full-tree diff.
- Fresh owned environments only: refuses an existing/nonempty build root,
  build/install venvs, wheel or receipts; nothing is installed into any
  existing application, Windows or old-Linux runtime; dependency acquisition
  is pip from PyPI/files.pythonhosted.org only, with every downloaded file
  SHA-256 recorded.
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

TOOL = "build_isolated_backend"
TOOL_VERSION = "1.1.0"

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / ".development-evidence" / "w5-isolated-backend-20261010"
UPSTREAM_RELEASE = EVIDENCE / "upstream-release"
UPSTREAM_TREE = UPSTREAM_RELEASE / "capytaine-3.0.0"
ARCHIVE = UPSTREAM_RELEASE / "capytaine-3.0.0.tar.gz"
SOURCE_ACQUISITION = UPSTREAM_RELEASE / "source-acquisition.json"
CANDIDATE_CONSTANTS = (REPO / ".development-evidence" /
                       "w5-precision-candidate-20261010" / "candidate" /
                       "Constants.f90")
RESULTS = EVIDENCE / "results"
SCRIPTS = EVIDENCE / "scripts"
ARTIFACTS = EVIDENCE / "artifacts"

BUILD_ROOT = "/var/tmp/anyhydro-precision-build-20261010"
WSL_DISTRO_DEFAULT = "Ubuntu-24.04"
EXTERNAL_IDENTITY = "capytaine-3.0.0-anyhydro-precision-pi-gamma-v1/linux-x86_64"
UPSTREAM_SEMANTIC_VERSION = "3.0.0"
MAX_GNU_TIMEOUT = 285
KILL_AFTER = 3

EXPECTED_ARCHIVE_SHA256 = ("2651fd1cd66624d6b5f56e84f5d0001eda34b5d108f38c"
                           "39d908f425f20bc968")
EXPECTED_UPSTREAM_CONSTANTS_SHA256 = ("df6c9ae10f8f5c60fd397c5fe691c8ac527e97"
                                     "1081d6d474138862eaea71cd96")
EXPECTED_CANDIDATE_CONSTANTS_SHA256 = ("5b8233205963e84710629ee9395ccb7e78ddb"
                                      "138bca411cfe8defb481b6b66bf")
EXPECTED_PI_LINE = ("  REAL(KIND=PRE), PARAMETER    :: PI = "
                   "3.141592653589793238462643383279502884_PRE ! π")
EXPECTED_GAMMA_LINE = ("  REAL(KIND=PRE), PARAMETER    :: EULER_GAMMA = "
                       "0.577215664901532860606512090082402431_PRE")
CANDIDATE_GAMMA_REFERENCE = float("0.577215664901532860606512090082402431")

CONSTANTS_REL = "libs/libDelhommeau/src/constants.f90"
EXPECTED_DEPENDENCY_NAMES = frozenset({
    "meson_python", "meson", "ninja", "pyproject_metadata", "packaging",
    "numpy", "charset_normalizer"})
THREAD_ENV_KEYS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                   "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

PHASES = ("stage", "deps", "build", "install", "collect", "manifest")


class BuildError(ValueError):
    """Invalid or unsafe build request."""


def to_wsl_path(path: Path) -> str:
    s = str(path.resolve())
    if re.match(r"^[A-Za-z]:", s):
        return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")
    return s


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


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
    for immutable in (UPSTREAM_RELEASE, REPO, CANDIDATE_CONSTANTS.parent):
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
            "upstream_constants_sha256": EXPECTED_UPSTREAM_CONSTANTS_SHA256}


def verify_candidate_two_line() -> dict:
    """Verify the candidate constants.f90 differs in exactly lines 26/28."""
    if sha256_file(CANDIDATE_CONSTANTS) != EXPECTED_CANDIDATE_CONSTANTS_SHA256:
        raise BuildError(f"candidate constants SHA mismatch: "
                         f"{CANDIDATE_CONSTANTS}")
    base = (UPSTREAM_TREE / CONSTANTS_REL).read_text(encoding="utf-8",
                                                     errors="strict")
    cand = CANDIDATE_CONSTANTS.read_text(encoding="utf-8", errors="strict")
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


def checklist_text() -> tuple[str, int]:
    acq = json.loads(SOURCE_ACQUISITION.read_text(encoding="utf-8"))
    lines = [f"{v}  {k.replace(chr(92), '/')}"
             for k, v in sorted(acq["file_sha256"].items())]
    return "\n".join(lines) + "\n", len(lines)


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
        "exact_argv": cmd,
        "gnu_timeout_seconds": receipt.get("gnu_timeout_seconds"),
        "gnu_kill_after_seconds": KILL_AFTER,
        "process_wall_seconds": wall,
        "returncode": proc.returncode,
        "validation_error": validation_error,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "ok": proc.returncode == 0 and validation_error is None,
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


def phase_stage(args) -> int:
    require_fresh_receipt("stage", args.attempt)
    check_build_root_safety()
    upstream = verify_upstream_tree()
    candidate = verify_candidate_two_line()
    checklist = SCRIPTS / f"stage-upstream-sha256-checklist-{args.attempt}.txt"
    text, n = checklist_text()
    save_text(checklist, text)

    mnt_upstream = to_wsl_path(UPSTREAM_TREE)
    mnt_candidate = to_wsl_path(CANDIDATE_CONSTANTS)
    mnt_checklist = to_wsl_path(checklist)
    root = shlex.quote(BUILD_ROOT)
    script = "\n".join([
        "set -euo pipefail",
        *thread_env_lines(),
        f"ROOT={root}",
        f'if [ -e "$ROOT" ]; then echo "REFUSED: build root exists: $ROOT"; '
        "exit 3; fi",
        'mkdir -p "$ROOT/upstream" "$ROOT/candidate-src" "$ROOT/wheels" '
        '"$ROOT/dist" "$ROOT/receipts" "$ROOT/logs"',
        f"cp -a {shlex.quote(mnt_upstream + '/')} "
        '"$ROOT/upstream/capytaine-3.0.0/"',
        'cd "$ROOT/upstream/capytaine-3.0.0"',
        f"echo STAGE_UPSTREAM_VERIFY",
        f"sha256sum -c {shlex.quote(mnt_checklist)} > "
        '"$ROOT/receipts/upstream-tree-verify.txt"',
        'echo "UPSTREAM_FILES=$(find "$ROOT/upstream" -type f | wc -l)"',
        'cp -a "$ROOT/upstream/capytaine-3.0.0" '
        '"$ROOT/candidate-src/capytaine-3.0.0"',
        f"cp {shlex.quote(mnt_candidate)} "
        f'"$ROOT/candidate-src/capytaine-3.0.0/{CONSTANTS_REL}"',
        'echo STAGE_CANDIDATE_DIFF',
        'diff -qr "$ROOT/upstream/capytaine-3.0.0" '
        '"$ROOT/candidate-src/capytaine-3.0.0" > '
        '"$ROOT/receipts/candidate-tree-diff.txt" || true',
        f'diff -u "$ROOT/upstream/capytaine-3.0.0/{CONSTANTS_REL}" '
        f'"$ROOT/candidate-src/capytaine-3.0.0/{CONSTANTS_REL}" > '
        '"$ROOT/receipts/constants-candidate-diff.txt" || true',
        f'sha256sum "$ROOT/candidate-src/capytaine-3.0.0/{CONSTANTS_REL}" > '
        '"$ROOT/receipts/candidate-constants.sha256"',
        'echo "CANDIDATE_FILES=$(find "$ROOT/candidate-src" -type f | wc -l)"',
        'cat "$ROOT/receipts/candidate-tree-diff.txt"',
        'echo STAGE_DONE',
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
        if len(diff_lines) != 1 or \
                not diff_lines[0].rstrip().endswith(
                    "libs/libDelhommeau/src/constants.f90 differ"):
            validation_error = (f"expected exactly one differing file "
                                f"(constants.f90), got: {diff_lines}")
        elif f"CANDIDATE_FILES={n}" not in proc.stdout:
            validation_error = "candidate file count mismatch"
        elif "STAGE_UPSTREAM_VERIFY" not in proc.stdout or \
                "STAGE_DONE" not in proc.stdout:
            validation_error = "stage markers missing"

    receipt = {"gnu_timeout_seconds": args.gnu_timeout,
               "build_root": BUILD_ROOT,
               "upstream_verification": upstream,
               "candidate_verification": candidate,
               "checklist": {"path": str(checklist),
                             "sha256": hashlib.sha256(
                                 text.encode("utf-8")).hexdigest(),
                             "expected_files": n},
               "script": str(script_path),
               "candidate_tree_diff_lines": diff_lines}
    finish_receipt(receipt, "stage", args.attempt, cmd, proc, wall,
                   validation_error)
    return 0 if receipt["ok"] else 1


def ln_ok(line: str) -> bool:
    return line.rstrip().endswith("constants.f90 differ")


def bootstrap_venv_lines(label: str) -> list[str]:
    """Bootstrap only an owned fresh venv without system ensurepip or apt.

    The archived pip wheel is verified against official release metadata during
    acquisition. PYTHONPATH exists for this one bootstrap subprocess only; the
    resulting interpreter then uses its installed pip normally.
    """
    if label not in {"build-env", "install-env"}:
        raise BuildError("unregistered virtual environment label")
    wheel = EVIDENCE / "bootstrap" / "pip-26.2.1-py3-none-any.whl"
    receipt = json.loads((EVIDENCE / "bootstrap" / "pip-acquisition.json").read_text())
    if sha256_file(wheel) != receipt["wheel_sha256"]:
        raise BuildError("bootstrap pip wheel identity mismatch")
    pip_wheel = shlex.quote(to_wsl_path(wheel))
    return [
        f'python3 -m venv --without-pip "$ROOT/{label}"',
        f'env PYTHONPATH={pip_wheel} "$ROOT/{label}/bin/python" -m pip '
        f'install --isolated --no-index --no-deps {pip_wheel} > '
        f'"$ROOT/logs/bootstrap-{label}-pip.log"',
    ]


def phase_deps(args) -> int:
    require_fresh_receipt("deps", args.attempt)
    require_prerequisite("deps", "stage")
    check_build_root_safety()
    root = shlex.quote(BUILD_ROOT)
    script = "\n".join([
        "set -euo pipefail",
        *thread_env_lines(),
        f"ROOT={root}",
        'if [ -e "$ROOT/build-env" ]; then echo '
        '"REFUSED: build-env exists"; exit 3; fi',
        'if [ -n "$(ls -A "$ROOT/wheels" 2>/dev/null)" ]; then echo '
        '"REFUSED: wheels dir not empty"; exit 3; fi',
        "python3 --version",
        *bootstrap_venv_lines("build-env"),
        '"$ROOT/build-env/bin/pip" download --isolated '
        '--index-url=https://pypi.org/simple --only-binary=:all: '
        '--cache-dir "$ROOT/pip-cache" -d "$ROOT/wheels" '
        "meson-python 'numpy>=2.0.0' charset-normalizer ninja > "
        '"$ROOT/logs/pip-download.log"',
        'sha256sum "$ROOT"/wheels/* | tee "$ROOT/receipts/dependency-wheels.sha256"',
        f'cp "$ROOT/receipts/dependency-wheels.sha256" '
        f'{shlex.quote(to_wsl_path(RESULTS))}/deps-wheels-{args.attempt}.sha256',
        '"$ROOT/build-env/bin/pip" install --no-index '
        '--find-links="$ROOT/wheels" meson-python "numpy>=2.0.0" '
        "charset-normalizer ninja > \"$ROOT/logs/pip-install.log\"",
        '"$ROOT/build-env/bin/pip" freeze | tee '
        '"$ROOT/receipts/build-env-freeze.txt"',
        '"$ROOT/build-env/bin/python" - > "$ROOT/receipts/build-toolchain.json" '
        " <<'PYEOF'",
        "import json, platform, sys",
        "import numpy, mesonpy",
        "print(json.dumps({'python_version': sys.version, "
        "'platform': platform.platform(), 'numpy_version': numpy.__version__, "
        "'meson_python_version': mesonpy.__version__}))",
        "PYEOF",
        f"cp \"$ROOT/receipts/build-toolchain.json\" "
        f"{shlex.quote(to_wsl_path(RESULTS))}/deps-toolchain-{args.attempt}"
        ".json",
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
    wheels = {}
    if proc.returncode != 0:
        validation_error = f"WSL child returned {proc.returncode}"
    elif "DEPS_DONE" not in proc.stdout:
        validation_error = "deps markers missing"
    else:
        tc_path = RESULTS / f"deps-toolchain-{args.attempt}.json"
        toolchain = (json.loads(tc_path.read_text(encoding="utf-8"))
                     if tc_path.is_file() else None)
        if not toolchain or not toolchain.get("numpy_version", "").startswith("2"):
            validation_error = f"unexpected toolchain receipt: {toolchain}"
        else:
            wheels_path = RESULTS / f"deps-wheels-{args.attempt}.sha256"
            if not wheels_path.is_file():
                validation_error = "downloaded wheels SHA record missing"
            else:
                for line in (wheels_path.read_text(encoding="utf-8")
                             .splitlines()):
                    parts = line.split()
                    if len(parts) == 2:
                        wheels[Path(parts[1]).name] = parts[0]
                unexpected = sorted(
                    name for name in wheels
                    if name.split("-")[0] not in EXPECTED_DEPENDENCY_NAMES)
                if not wheels:
                    validation_error = "no dependency wheels recorded"
                elif unexpected:
                    validation_error = (f"unexpected downloaded dependencies "
                                        f"(outside the expected pyproject "
                                        f"build requirements): {unexpected}")
    receipt = {"gnu_timeout_seconds": args.gnu_timeout,
               "build_root": BUILD_ROOT,
               "script": str(script_path),
               "dependency_acquisition": {
                   "index_url": "https://pypi.org/simple",
                   "isolated": True,
                   "only_binary": ":all:",
                   "cache_dir": f"{BUILD_ROOT}/pip-cache (owned; no user "
                                "pip configuration, no global cache "
                                "mutation)",
                   "expected_dependency_names":
                       sorted(EXPECTED_DEPENDENCY_NAMES),
                   "downloaded_wheels_sha256": wheels,
               },
               "toolchain": toolchain}
    finish_receipt(receipt, "deps", args.attempt, cmd, proc, wall,
                   validation_error)
    return 0 if receipt["ok"] else 1


def validate_built_version(about: str | None) -> str:
    """Read the generated version assignment without executing package code."""
    if not about:
        raise BuildError("built __about__.py missing")
    values = []
    for node in ast.parse(about).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "__version__"
                for t in node.targets):
            values.append(ast.literal_eval(node.value))
    if values != [UPSTREAM_SEMANTIC_VERSION]:
        raise BuildError("built __about__.py version is not 3.0.0")
    return values[0]


def phase_build(args) -> int:
    require_fresh_receipt("build", args.attempt)
    require_prerequisite("build", "deps")
    check_build_root_safety()
    root = shlex.quote(BUILD_ROOT)
    script = "\n".join([
        "set -euo pipefail",
        *thread_env_lines(),
        f"ROOT={root}",
        'if [ -n "$(ls -A "$ROOT/dist" 2>/dev/null)" ]; then echo '
        '"REFUSED: dist dir not empty (no silent wheel reuse)"; exit 3; fi',
        'mkdir -p "$ROOT/ninja-wrap"',
        'printf \'%s\\n\' \'#!/bin/bash\' '
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
        "echo MESONPY_COMPILE_ARGS_MECHANISM",
        'grep -rn "compile.args\\|compile-args\\|compile_args" '
        '"$ROOT"/build-env/lib/python3*/site-packages/mesonpy/*.py | head -10 '
        '|| true',
        "cd \"$ROOT/candidate-src/capytaine-3.0.0\"",
        "echo PIP_WHEEL_START",
        "pip wheel --no-deps --no-build-isolation --wheel-dir \"$ROOT/dist\" "
        "--config-settings=compile-args=-j1 . 2>&1 | tee "
        '"$ROOT/logs/pip-wheel.log"',
        "echo PIP_WHEEL_DONE",
        'ls -l "$ROOT/dist"',
        'sha256sum "$ROOT"/dist/*.whl | tee "$ROOT/receipts/wheel.sha256"',
        '"$ROOT/build-env/bin/python" - "$ROOT"/dist/*.whl > '
        '"$ROOT/receipts/wheel-contents.json" <<\'PYEOF\'',
        "import json, sys, zipfile",
        "w = sys.argv[1]",
        "z = zipfile.ZipFile(w)",
        "print(json.dumps({'wheel': w, 'names': z.namelist(), "
        "'about_py': z.read('capytaine/__about__.py').decode()}))",
        "PYEOF",
        f"cp \"$ROOT/receipts/wheel-contents.json\" "
        f"{shlex.quote(to_wsl_path(RESULTS))}/build-wheel-contents-"
        f"{args.attempt}.json",
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
            contents_path = RESULTS / f"build-wheel-contents-{args.attempt}.json"
            if contents_path.is_file():
                contents = json.loads(contents_path.read_text(encoding="utf-8"))
                about = contents.get("about_py")
            if not wheel_name.startswith("capytaine-3.0.0-"):
                validation_error = f"unexpected wheel name {wheel_name}"
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
               "build_route": "pip wheel (meson-python backend, "
                              "--no-build-isolation) with "
                              "--config-settings=compile-args=-j1; f2py from "
                              "the installed numpy; full log in "
                              "build-root-logs/pip-wheel.log",
               "ninja_args_proven": ninja_args,
               "wheel_name": wheel_name,
               "wheel_sha256": wheel_sha,
               "built_about_py": about,
               "compiler": "gfortran 13.3.0 (WSL Ubuntu-24.04 system "
                           "compiler; exact version line in stdout)"}
    finish_receipt(receipt, "build", args.attempt, cmd, proc, wall,
                   validation_error)
    return 0 if receipt["ok"] else 1


INSPECTOR = '''\
# SPDX-License-Identifier: MPL-2.0
"""Direct native extension constants inspector (no package imports).

Loads each Delhommeau extension module straight from its .so file path via
importlib (spec_from_file_location + exec_module) using the exact compiled
module name (the .so file name before the first dot, e.g.
Delhommeau_float64), so the capytaine package __init__ never executes.
Fortran module specification parts contain no executable statements and the
f2py module init only registers procedures, so this import invokes no
Green-function constructor, kernel, table or Prony evaluation. The f2py
module namespaces (constants, floating_point_precision) are Fortran module
objects, not dicts: only the explicit public scalar attributes
(pi, euler_gamma, minus_one_over_fourpi, log_2, pre) are read via getattr
and converted from 0-d NumPy values to ordinary Python scalars for JSON;
no module procedure is called.
"""
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

site = Path(sys.argv[1])
gfdir = site / "capytaine" / "green_functions"
out = {
    "loader": "importlib.util.spec_from_file_location + exec_module",
    "package_init_executed": False,
    "import_safety": (
        "Fortran module specification parts contain no executable "
        "statements; f2py module init registers procedures only; no "
        "Green-function constructor, kernel, table generation or Prony "
        "evaluation is invoked; no module procedure is called"),
}


def scalar(value):
    """Convert a 0-d NumPy value to an ordinary Python scalar."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            return None
    return None


mods = {}
for so in sorted(gfdir.glob("Delhommeau_float*.so")):
    module_name = so.name.split(".")[0]
    spec = importlib.util.spec_from_file_location(module_name, str(so))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    entry = {
        "module_name": module_name,
        "path": str(so),
        "sha256": hashlib.sha256(so.read_bytes()).hexdigest(),
        "attributes": sorted(a for a in dir(mod) if not a.startswith("_")),
    }
    constants = getattr(mod, "constants", None)
    if constants is not None:
        entry["constants"] = {
            name: scalar(getattr(constants, name, None))
            for name in ("pi", "euler_gamma", "minus_one_over_fourpi",
                         "log_2")}
    fpp = getattr(mod, "floating_point_precision", None)
    if fpp is not None:
        entry["floating_point_precision"] = {
            "pre": scalar(getattr(fpp, "pre", None))}
    mods[so.name] = entry
out["modules"] = mods
f64 = next((m for k, m in mods.items() if "float64" in k), None)
if f64 is not None:
    c = f64.get("constants") or {}
    pre = (f64.get("floating_point_precision") or {}).get("pre")
    pi = c.get("pi")
    gamma = c.get("euler_gamma")
    out["analysis"] = {
        "pre_kind": pre,
        "pi": pi,
        "pi_minus_math_pi": (pi - math.pi) if isinstance(pi, float)
                             else None,
        "euler_gamma": gamma,
        "gamma_minus_reference": (
            gamma - 0.577215664901532860606512090082402431)
        if isinstance(gamma, float) else None,
        "minus_one_over_fourpi": c.get("minus_one_over_fourpi"),
        "log_2": c.get("log_2"),
    }
print(json.dumps(out, indent=2))
'''


def _is_sha256(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) \
        is not None


def phase_install(args) -> int:
    require_fresh_receipt("install", args.attempt)
    require_prerequisite("install", "build")
    check_build_root_safety()
    inspector = SCRIPTS / f"inspect_native_constants-{args.attempt}.py"
    save_text(inspector, INSPECTOR)
    root = shlex.quote(BUILD_ROOT)
    mnt_inspector = to_wsl_path(inspector)
    script = "\n".join([
        "set -euo pipefail",
        *thread_env_lines(),
        f"ROOT={root}",
        'if [ -e "$ROOT/install-env" ]; then echo '
        '"REFUSED: install-env exists"; exit 3; fi',
        'WHEEL="$(ls "$ROOT"/dist/*.whl)"',
        'test "$(ls "$ROOT"/dist/*.whl | wc -l)" -eq 1',
        *bootstrap_venv_lines("install-env"),
        'NUMPY_WHEEL="$(ls "$ROOT"/wheels/numpy-*.whl)"',
        'test -n "$NUMPY_WHEEL"',
        '"$ROOT/install-env/bin/pip" install --no-index --no-deps '
        '"$NUMPY_WHEEL" > "$ROOT/logs/pip-install-numpy.log"',
        '"$ROOT/install-env/bin/python" -c "import numpy; '
        'print(numpy.__version__)" > '
        '"$ROOT/receipts/install-numpy-version.txt"',
        '"$ROOT/install-env/bin/pip" install --no-deps --no-index "$WHEEL" > '
        '"$ROOT/logs/pip-install-wheel.log"',
        'SITE="$("$ROOT/install-env/bin/python" -c '
        '"import sysconfig; print(sysconfig.get_paths()[\'purelib\'])")"',
        'echo SITE_PACKAGES="$SITE"',
        'find "$SITE/capytaine" -name "Delhommeau_float*.so" -o -name '
        '"__about__.py" | sort',
        'find "$SITE/capytaine" -name "Delhommeau_float*.so" -exec sha256sum '
        '{} \\; | tee "$ROOT/receipts/native-binaries.sha256"',
        'cat "$SITE/capytaine/__about__.py"',
        f'"$ROOT/install-env/bin/python" {shlex.quote(mnt_inspector)} '
        '"$SITE" > "$ROOT/receipts/native-constants.json"',
        f"cp \"$ROOT/receipts/native-constants.json\" "
        f"{shlex.quote(to_wsl_path(RESULTS))}/install-native-constants-"
        f"{args.attempt}.json",
        'cat "$ROOT/receipts/native-constants.json"',
        "echo INSTALL_DONE",
    ]) + "\n"
    script_path = save_script(f"install-{args.attempt}.sh", script)

    start = time.perf_counter()
    cmd, proc = run_wsl(args.wsl_distro, script, args.gnu_timeout)
    wall = time.perf_counter() - start

    validation_error = None
    native = None
    if proc.returncode != 0:
        validation_error = f"WSL child returned {proc.returncode}"
    elif "INSTALL_DONE" not in proc.stdout:
        validation_error = "install did not reach INSTALL_DONE"
    else:
        native_path = RESULTS / f"install-native-constants-{args.attempt}.json"
        if not native_path.is_file():
            validation_error = "native constants JSON missing"
        else:
            native = json.loads(native_path.read_text(encoding="utf-8"))
            modules = native.get("modules") or {}
            names = sorted(modules)
            f64 = next((m for k, m in modules.items()
                        if "Delhommeau_float64" in k), None)
            f32 = next((m for k, m in modules.items()
                        if "Delhommeau_float32" in k), None)
            analysis = native.get("analysis") or {}
            pi = analysis.get("pi")
            gamma = analysis.get("euler_gamma")
            if f64 is None or f32 is None:
                validation_error = (f"both Delhommeau extensions required, "
                                    f"found: {names}")
            elif not _is_sha256(f64.get("sha256")) or \
                    not _is_sha256(f32.get("sha256")):
                validation_error = "extension SHA-256 hashes missing"
            elif analysis.get("pre_kind") != 8:
                validation_error = f"PRE kind is not 8: {analysis}"
            elif not isinstance(pi, float) or pi != math.pi:
                validation_error = f"native PI mismatch: {pi!r}"
            elif not isinstance(gamma, float) or \
                    abs(gamma - CANDIDATE_GAMMA_REFERENCE) > 1e-15:
                validation_error = f"native EULER_GAMMA mismatch: {gamma!r}"
            elif any(not isinstance(analysis.get(k), float)
                     for k in ("minus_one_over_fourpi", "log_2")):
                validation_error = (f"native f64 constants incomplete: "
                                    f"{analysis}")
    receipt = {"gnu_timeout_seconds": args.gnu_timeout,
               "build_root": BUILD_ROOT,
               "script": str(script_path),
               "inspector": str(inspector),
               "isolated_install_env": f"{BUILD_ROOT}/install-env "
                                       "(fresh venv; the exact numpy wheel "
                                       "downloaded for the build env is "
                                       "installed offline first, then the "
                                       "capytaine wheel with --no-deps "
                                       "--no-index; never an existing "
                                       "application, Windows or old-Linux "
                                       "runtime)",
               "extension_sha256": {k: m.get("sha256")
                                    for k, m in
                                    (native.get("modules") or {}).items()}
               if native else {},
               "native_constants": native}
    finish_receipt(receipt, "install", args.attempt, cmd, proc, wall,
                   validation_error)
    return 0 if receipt["ok"] else 1


def phase_collect(args) -> int:
    require_fresh_receipt("collect", args.attempt)
    for prereq in ("stage", "deps", "build", "install"):
        require_prerequisite("collect", prereq)
    check_build_root_safety()
    root = shlex.quote(BUILD_ROOT)
    mnt_artifacts = to_wsl_path(ARTIFACTS)
    script = "\n".join([
        "set -euo pipefail",
        *thread_env_lines(),
        f"ROOT={root}",
        f"ART={shlex.quote(mnt_artifacts)}",
        'if [ -e "$ART/build-root-receipts" ]; then echo '
        '"REFUSED: artifacts already collected"; exit 3; fi',
        'mkdir -p "$ART"',
        'cp -a "$ROOT/receipts" "$ART/build-root-receipts"',
        'cp -a "$ROOT/logs" "$ART/build-root-logs"',
        'cp "$ROOT"/dist/*.whl "$ART/"',
        'sha256sum "$ART"/*.whl',
        'du -sh "$ART"',
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
               "artifacts_dir": str(ARTIFACTS)}
    finish_receipt(receipt, "collect", args.attempt, cmd, proc, wall,
                   validation_error)
    return 0 if receipt["ok"] else 1


def phase_manifest(args) -> int:
    require_fresh_receipt("manifest", args.attempt)
    for prereq in ("stage", "deps", "build", "install", "collect"):
        require_prerequisite("manifest", prereq)
    dep_wheels = {}
    wheels_sha = ARTIFACTS / "build-root-receipts" / "dependency-wheels.sha256"
    for line in wheels_sha.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2:
            dep_wheels[Path(parts[1]).name] = parts[0]
    freeze = (ARTIFACTS / "build-root-receipts" /
              "build-env-freeze.txt").read_text(encoding="utf-8").splitlines()
    receipts = {}
    for phase in ("stage", "deps", "build", "install", "collect"):
        p = find_ok_receipt(phase)
        if p is None:
            raise BuildError(f"missing ok receipt for {phase}")
        receipts[phase] = p
    install_receipt = json.loads(
        (RESULTS / receipts["install"].name).read_text(encoding="utf-8"))
    build_receipt = json.loads(
        (RESULTS / receipts["build"].name).read_text(encoding="utf-8"))
    deps_receipt = json.loads(
        (RESULTS / receipts["deps"].name).read_text(encoding="utf-8"))
    stage_receipt_data = json.loads(
        (RESULTS / receipts["stage"].name).read_text(encoding="utf-8"))
    native = install_receipt.get("native_constants") or {}
    analysis = native.get("analysis") or {}
    f64 = next((m for k, m in (native.get("modules") or {}).items()
                if "float64" in k), {})
    manifest = {
        "format": "anyhydro-isolated-native-backend-manifest-v1",
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
            "verified_files": (stage_receipt_data
                                .get("upstream_verification") or {})
            .get("verified_files"),
            "upstream_constants_sha256": EXPECTED_UPSTREAM_CONSTANTS_SHA256,
            "candidate_constants": str(CANDIDATE_CONSTANTS),
            "candidate_constants_sha256": EXPECTED_CANDIDATE_CONSTANTS_SHA256,
            "only_change": ("libs/libDelhommeau/src/constants.f90 lines "
                            "26 (PI) and 28 (EULER_GAMMA); all other source "
                            "bytes identical to the official sdist; Apache-2.0 "
                            "headers preserved; upstream default "
                            "meson.options hams_green_function=enabled "
                            "compiles the optional LGPL FinGreen3D and "
                            "LiangWuNoblesse sources as upstream does"),
        },
        "patch": {
            "pi_line": EXPECTED_PI_LINE,
            "euler_gamma_line": EXPECTED_GAMMA_LINE,
        },
        "build": {
            "route": build_receipt.get("build_route"),
            "platform": "WSL Ubuntu-24.04 linux-x86_64",
            "build_root": BUILD_ROOT,
            "compiler": build_receipt.get("compiler"),
            "one_job": "ninja -j1 via meson-python compile-args config "
                        "setting, proven by the actual ninja argv recorded "
                        "in build-root receipts (ninja-args.log, every "
                        "invocation logged by an owned PATH wrapper that "
                        "execs the real ninja unchanged); "
                        "OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 exported "
                        "inside the WSL shell and echoed at runtime",
            "ninja_args_proven": build_receipt.get("ninja_args_proven"),
            "wheel_name": build_receipt.get("wheel_name"),
            "wheel_sha256": build_receipt.get("wheel_sha256"),
            "built_about_py": build_receipt.get("built_about_py"),
        },
        "dependencies": {
            "source": "PyPI/files.pythonhosted.org only",
            "acquisition": deps_receipt.get("dependency_acquisition"),
            "resolved_freeze": freeze,
            "downloaded_wheels_sha256": dep_wheels,
            "install_env_numpy": (
                (ARTIFACTS / "build-root-receipts" /
                 "install-numpy-version.txt")
                .read_text(encoding="utf-8").strip()
                if (ARTIFACTS / "build-root-receipts" /
                    "install-numpy-version.txt").is_file() else None),
        },
        "native_artifact": {
            "extension_modules": {
                k: {"path": m.get("path"), "sha256": m.get("sha256"),
                    "attributes": m.get("attributes")}
                for k, m in (native.get("modules") or {}).items()},
            "extension_sha256": install_receipt.get("extension_sha256"),
            "pre_kind": analysis.get("pre_kind"),
            "pi": analysis.get("pi"),
            "pi_minus_math_pi": analysis.get("pi_minus_math_pi"),
            "euler_gamma": analysis.get("euler_gamma"),
            "gamma_minus_reference": analysis.get("gamma_minus_reference"),
            "minus_one_over_fourpi": analysis.get("minus_one_over_fourpi"),
            "log_2": analysis.get("log_2"),
            "loader": native.get("loader"),
            "import_safety": native.get("import_safety"),
        },
        "receipts": {k: str(v) for k, v in receipts.items()},
        "reproduction_recipe": [
            "1. stage:   python tools/native_precision/build_isolated_backend.py"
            " stage --attempt <name>",
            "2. deps:    python tools/native_precision/build_isolated_backend.py"
            " deps --attempt <name>",
            "3. build:   python tools/native_precision/build_isolated_backend.py"
            " build --attempt <name>",
            "4. install: python tools/native_precision/build_isolated_backend.py"
            " install --attempt <name>",
            "5. collect:  python tools/native_precision/build_isolated_backend.py"
            " collect --attempt <name>",
            "6. manifest: python tools/native_precision/build_isolated_backend.py"
            " manifest --attempt <name>",
            "Each phase runs one WSL GNU-timeout-bounded bash script (saved "
            "under scripts/) through the root check.py runner; exact argv, "
            "full stdout/stderr and wall time are in the per-phase receipts "
            "and the runner attempt logs. This recipe reproduces the artifact"
            " deterministically up to build-generated timestamps recorded in "
            "capytaine/__about__.py; no bit-identical wheel claim is made.",
        ],
        "non_claims": [
            "not Green-function, kernel, ABI, scientific or platform "
            "qualification",
            "no current application runtime changed; nothing installed into "
            "any existing Windows or old-Linux runtime",
            "no empirical scaling, no private GF overrides, no default-real "
            "promotion, no other literal rewritten",
            "import/constants metadata does not confer numerical authority",
        ],
    }
    out = RESULTS / "build-manifest.json"
    if out.exists():
        raise BuildError(f"refusing to overwrite existing manifest {out}")
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(out), "ok": True}, indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=PHASES)
    ap.add_argument("--attempt", required=True,
                    help="unique attempt label for this phase")
    ap.add_argument("--wsl-distro", default=WSL_DISTRO_DEFAULT)
    ap.add_argument("--gnu-timeout", type=int, default=MAX_GNU_TIMEOUT)
    args = ap.parse_args(argv)
    if args.gnu_timeout > MAX_GNU_TIMEOUT or args.gnu_timeout <= 0:
        raise BuildError(f"GNU timeout must be in (0, {MAX_GNU_TIMEOUT}] s")
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.attempt):
        raise BuildError(f"invalid attempt label {args.attempt!r}")
    return {
        "stage": phase_stage,
        "deps": phase_deps,
        "build": phase_build,
        "install": phase_install,
        "collect": phase_collect,
        "manifest": phase_manifest,
    }[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
