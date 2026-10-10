# SPDX-License-Identifier: MPL-2.0
"""ONE bounded source/compiler/constant-only probe for the W5 precision candidate.

Synthesizes a clearly labelled FLOATING_POINT_PRECISION module (PRE = kind(1d0)),
compiles the pinned upstream Constants.f90 (baseline) and the candidate
Constants.f90 with the existing gfortran in WSL Ubuntu-24.04, and prints
PI/EULER_GAMMA kinds and values for both variants behind explicit VARIANT
markers.

This proves source-kind correction of the two named constants only. It is NOT a
native backend build, NOT an ABI/Green-physics demonstration and NOT numerical
qualification. Zero Green-function constructors, kernels, tables, Prony
fitting, systems, HAMS, dynamics, oracles or campaigns are invoked.

Safety contract:
- The WSL child is launched as subprocess argv
  ['wsl.exe', '-d', <distro>, '--', 'timeout', '--kill-after=<k>s', '<t>s',
   'bash', '-c', <raw script>] with the script passed directly (no JSON shell
  escaping); fixed paths inside the script are shlex.quoted.
- One-thread environment is exported INSIDE the WSL shell (Windows-side env
  may not propagate through wsl.exe).
- Refuses any temp/source/upstream/output overlap (including arbitrary output
  inside or containing the immutable upstream bundle), nonempty temp, an
  existing/nonempty/non-directory output, and an existing
  receipt; source hashes are verified unchanged before and after the run.
- Output parsing requires exactly one baseline block and one candidate block
  (explicit VARIANT markers), numeric kind/value checks, and finite values;
  any missing/malformed/error/nonfinite result or wrong PI/Gamma value exits
  nonzero. The receipt is inspected, not the exit code alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import struct
import subprocess
import sys
import time
from pathlib import Path

TOOL = "probe_constants_source"
TOOL_VERSION = "2.0.2"
MAX_WSL_TIMEOUT = 100
MAX_KILL_AFTER = 3

REPO = Path(__file__).resolve().parents[2]
UPSTREAM_BUNDLE = (REPO / ".development-evidence" /
                   "w5-precision-candidate-20261010" / "upstream")

FPP_MODULE = """! PROBE-SYNTHESIZED FLOATING_POINT_PRECISION (not upstream source)
! Source-only constant probe: PRE = kind(1d0) (float64), per build recipe.
module floating_point_precision
  implicit none
  integer, parameter :: pre = kind(1d0)
end module floating_point_precision
"""

DRIVER = """program probe_driver
  use floating_point_precision, only: pre
  use constants, only: pi, euler_gamma, minus_one_over_fourpi, log_2
  implicit none
  print '(a,i0)', 'PRE=', pre
  print '(a,i0)', 'PI_KIND=', kind(pi)
  print '(a,i0)', 'GAMMA_KIND=', kind(euler_gamma)
  print '(a,es25.17e3)', 'PI=', pi
  print '(a,es25.17e3)', 'EULER_GAMMA=', euler_gamma
  print '(a,es25.17e3)', 'MINUS_ONE_OVER_FOURPI=', minus_one_over_fourpi
  print '(a,es25.17e3)', 'LOG_2=', log_2
end program probe_driver
"""

BASELINE_PI_LITERAL = 3.141592653588979
BASELINE_GAMMA_LITERAL = 0.5772156649
CANDIDATE_PI_REFERENCE = math.pi
CANDIDATE_GAMMA_REFERENCE = float("0.577215664901532860606512090082402431")
VALUE_TOLERANCE = 1e-15

REQUIRED_KEYS = ("PRE", "PI_KIND", "GAMMA_KIND", "PI", "EULER_GAMMA",
                 "MINUS_ONE_OVER_FOURPI", "LOG_2")


class ProbeError(ValueError):
    """Invalid or unsafe probe request."""


def to_wsl_path(path: Path) -> str:
    s = str(path.resolve())
    if re.match(r"^[A-Za-z]:", s):
        return "/mnt/" + s[0].lower() + s[2:].replace("\\", "/")
    return s


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def float32_round(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def parse_driver_output(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    return out


def build_wsl_script(temp: Path) -> str:
    """Raw bash script for the WSL child; fixed paths are shlex.quoted."""
    lines = [
        "set -eu",
        "export OMP_NUM_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "gfortran --version | head -1",
    ]
    for variant in ("baseline", "candidate"):
        wdir = shlex.quote(to_wsl_path(temp / variant))
        exe = shlex.quote(f"./probe_{variant}.exe")
        lines += [
            f"echo VARIANT={variant}",
            f"cd {wdir}",
            "gfortran -std=f2008 -Wall floating_point_precision.f90 "
            "constants.f90 probe_driver.f90 -o "
            + shlex.quote(f"probe_{variant}.exe"),
            f"sha256sum {exe}",
            f"{exe}",
        ]
    return "\n".join(lines) + "\n"


def run_wsl(distro: str, script: str, timeout: int, kill_after: int):
    """Launch the WSL child via argv; the raw script is passed as one argv
    element to bash -c (no JSON/shell re-escaping)."""
    cmd = ["wsl.exe", "-d", distro, "--", "timeout",
           f"--kill-after={kill_after}s", f"{timeout}s", "bash", "-c", script]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return cmd, proc


def split_variant_blocks(stdout: str) -> dict:
    """Split stdout into exactly one baseline and one candidate block, in that
    order, behind explicit VARIANT markers. Raises ProbeError otherwise."""
    markers = {}
    order = []
    for lineno, line in enumerate(stdout.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("VARIANT="):
            variant = stripped[len("VARIANT="):].strip()
            if variant not in ("baseline", "candidate"):
                raise ProbeError(f"unknown VARIANT marker at line {lineno}: "
                                 f"{stripped!r}")
            if variant in markers:
                raise ProbeError(f"duplicate VARIANT={variant} marker")
            markers[variant] = []
            order.append(variant)
        elif order:
            markers[order[-1]].append(line)
    if order != ["baseline", "candidate"]:
        raise ProbeError(f"expected exactly one baseline block then one "
                         f"candidate block, got markers in order: {order}")
    return {v: "\n".join(lines) for v, lines in markers.items()}


def _finite_float(parsed: dict, key: str, variant: str) -> float:
    if key not in parsed:
        raise ProbeError(f"{variant}: missing required key {key!r}")
    try:
        value = float(parsed[key])
    except ValueError as exc:
        raise ProbeError(f"{variant}: non-numeric {key}={parsed[key]!r}") from exc
    if not math.isfinite(value):
        raise ProbeError(f"{variant}: nonfinite {key}={parsed[key]!r}")
    return value


def _kind_matches(parsed: dict, key: str, pre: str, variant: str) -> None:
    if parsed.get(key) != pre:
        raise ProbeError(f"{variant}: {key}={parsed.get(key)!r} does not match "
                         f"PRE={pre!r}")


def validate_block(block: str, variant: str) -> dict:
    """Strict validation of one driver block. Returns parsed values; raises
    ProbeError on missing/malformed/nonfinite data or wrong PI/Gamma."""
    parsed = parse_driver_output(block)
    for key in REQUIRED_KEYS:
        if key not in parsed:
            raise ProbeError(f"{variant}: missing required key {key!r}")
    pre = parsed["PRE"]
    if not re.fullmatch(r"\d+", pre):
        raise ProbeError(f"{variant}: malformed PRE={pre!r}")
    _kind_matches(parsed, "PI_KIND", pre, variant)
    _kind_matches(parsed, "GAMMA_KIND", pre, variant)
    values = {key: _finite_float(parsed, key, variant) for key in
              REQUIRED_KEYS[3:]}
    if variant == "baseline":
        expected_pi = float32_round(BASELINE_PI_LITERAL)
        expected_gamma = float32_round(BASELINE_GAMMA_LITERAL)
    else:
        expected_pi = CANDIDATE_PI_REFERENCE
        expected_gamma = CANDIDATE_GAMMA_REFERENCE
    if abs(values["PI"] - expected_pi) > VALUE_TOLERANCE:
        raise ProbeError(f"{variant}: wrong PI={values['PI']!r} "
                         f"(expected ~{expected_pi!r})")
    if abs(values["EULER_GAMMA"] - expected_gamma) > VALUE_TOLERANCE:
        raise ProbeError(f"{variant}: wrong EULER_GAMMA={values['EULER_GAMMA']!r} "
                         f"(expected ~{expected_gamma!r})")
    if abs(values["MINUS_ONE_OVER_FOURPI"] + 1.0 / (4.0 * values["PI"])) > \
            VALUE_TOLERANCE:
        raise ProbeError(f"{variant}: MINUS_ONE_OVER_FOURPI="
                         f"{values['MINUS_ONE_OVER_FOURPI']!r} inconsistent "
                         f"with PI={values['PI']!r}")
    if abs(values["LOG_2"] - math.log(2.0)) > VALUE_TOLERANCE:
        raise ProbeError(f"{variant}: wrong LOG_2={values['LOG_2']!r}")
    exe_sha = None
    for line in block.splitlines():
        m = re.match(r"^([0-9a-f]{64})\s+\*?\./?probe_" + variant
                     + r"\.exe\s*$", line.strip())
        if m:
            exe_sha = m.group(1)
            break
    return {"parsed": parsed, "values": values, "pre": pre,
            "executable_sha256": exe_sha}


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def check_paths(baseline: Path, candidate: Path, temp: Path, out: Path) -> None:
    """Refuse temp/source/upstream/output overlap before anything is created."""
    for name, source in (("baseline", baseline), ("candidate", candidate)):
        if _is_relative_to(source, temp) or _is_relative_to(temp, source):
            raise ProbeError(f"temp/source overlap: {name} source {source} "
                             f"vs temp {temp}")
    if _is_relative_to(temp, UPSTREAM_BUNDLE):
        raise ProbeError(f"temp {temp} is inside the immutable upstream "
                         f"bundle {UPSTREAM_BUNDLE}")
    if _is_relative_to(out, temp) or _is_relative_to(temp, out):
        raise ProbeError(f"temp/output overlap: temp {temp} vs out {out}")
    if _is_relative_to(out, UPSTREAM_BUNDLE):
        raise ProbeError(f"output {out} is inside the immutable upstream "
                         f"bundle {UPSTREAM_BUNDLE}")
    if _is_relative_to(UPSTREAM_BUNDLE, out):
        raise ProbeError(f"output {out} contains the immutable upstream "
                         f"bundle {UPSTREAM_BUNDLE}")
    for name, source in (("baseline", baseline), ("candidate", candidate)):
        if _is_relative_to(source, out):
            raise ProbeError(f"output/source overlap: {name} source {source} "
                             f"is inside output {out}")
    if temp.exists() and any(temp.iterdir()):
        raise ProbeError(f"temp {temp} exists and is not empty")
    if out.exists() and not out.is_dir():
        raise ProbeError(f"output path exists and is not a directory: {out}")
    if (out / "constants-source-probe-receipt.json").exists():
        raise ProbeError("refusing to overwrite an existing probe receipt")
    if out.exists() and any(out.iterdir()):
        raise ProbeError(f"output {out} exists and is not empty")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, type=Path,
                    help="pinned upstream constants.f90 (read-only)")
    ap.add_argument("--candidate", required=True, type=Path,
                    help="candidate Constants.f90 (read-only)")
    ap.add_argument("--temp", required=True, type=Path,
                    help="isolated owned TEMP directory for probe artifacts")
    ap.add_argument("--out", required=True, type=Path,
                    help="owned results directory for the probe receipt")
    ap.add_argument("--wsl-distro", default="Ubuntu-24.04")
    ap.add_argument("--timeout", type=int, default=100)
    ap.add_argument("--kill-after", type=int, default=3)
    args = ap.parse_args(argv)

    if args.timeout > MAX_WSL_TIMEOUT or args.timeout <= 0:
        raise ProbeError(f"WSL GNU timeout must be in (0, {MAX_WSL_TIMEOUT}] s")
    if args.kill_after > MAX_KILL_AFTER or args.kill_after <= 0:
        raise ProbeError(f"kill-after must be in (0, {MAX_KILL_AFTER}] s")
    for name, path in (("baseline", args.baseline), ("candidate", args.candidate)):
        if not path.is_file():
            raise ProbeError(f"{name} source not found: {path}")

    temp = args.temp.resolve()
    out = args.out.resolve()
    check_paths(args.baseline.resolve(), args.candidate.resolve(), temp, out)

    source_hashes_before = {
        "baseline_constants_sha256": sha256_file(args.baseline),
        "candidate_constants_sha256": sha256_file(args.candidate),
    }

    temp.mkdir(parents=True, exist_ok=True)
    for variant, source in (("baseline", args.baseline), ("candidate", args.candidate)):
        vdir = temp / variant
        vdir.mkdir(exist_ok=True)
        (vdir / "floating_point_precision.f90").write_text(FPP_MODULE,
                                                          encoding="utf-8")
        (vdir / "constants.f90").write_bytes(source.read_bytes())
        (vdir / "probe_driver.f90").write_text(DRIVER, encoding="utf-8")

    script = build_wsl_script(temp)

    start = time.perf_counter()
    cmd, proc = run_wsl(args.wsl_distro, script, args.timeout, args.kill_after)
    wall = time.perf_counter() - start

    source_hashes_after = {
        "baseline_constants_sha256": sha256_file(args.baseline),
        "candidate_constants_sha256": sha256_file(args.candidate),
    }
    sources_unchanged = source_hashes_before == source_hashes_after

    compiler_version = next((ln.strip() for ln in proc.stdout.splitlines()
                             if ln.strip()), "")

    validation_error = None
    validated = {}
    if proc.returncode != 0:
        validation_error = f"WSL child returned {proc.returncode}"
    elif not sources_unchanged:
        validation_error = "source files changed during the probe"
    else:
        try:
            blocks = split_variant_blocks(proc.stdout)
            for variant in ("baseline", "candidate"):
                validated[variant] = validate_block(blocks[variant], variant)
        except ProbeError as exc:
            validation_error = str(exc)

    receipt = {
        "format": "anyhydro-constants-source-probe-v2",
        "tool": f"tools/native_precision/{TOOL}.py",
        "tool_version": TOOL_VERSION,
        "scope": ("source/compiler/constant-only probe; proves source-kind "
                  "correction of PI/EULER_GAMMA only; not a backend build, not "
                  "ABI/Green physics, not qualification"),
        "wsl_distro": args.wsl_distro,
        "gnu_timeout_seconds": args.timeout,
        "gnu_kill_after_seconds": args.kill_after,
        "parent_bound": "check.py ordinary attempt, max 120 s/process, one job/thread",
        "exact_argv": cmd,
        "wsl_script": script,
        "thread_env_set_inside_wsl_shell": [
            "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
            "NUMEXPR_NUM_THREADS=1",
        ],
        "windows_env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "TEMP": os.environ.get("TEMP"),
            "ANYHYDRO_PROCESS_DEADLINE_PERF_COUNTER":
                os.environ.get("ANYHYDRO_PROCESS_DEADLINE_PERF_COUNTER"),
        },
        "probe_compiler_version": compiler_version,
        "source_identity": {
            **source_hashes_before,
            "sources_unchanged_after_probe": sources_unchanged,
            "synthesized_fpp_module": "PROBE-SYNTHESIZED, PRE=kind(1d0), not upstream",
            "fpp_module_sha256": hashlib.sha256(
                FPP_MODULE.encode("utf-8")).hexdigest(),
            "driver_sha256": hashlib.sha256(
                DRIVER.encode("utf-8")).hexdigest(),
        },
        "compiled_variants": {
            variant: {
                "pre_kind": data["pre"],
                "pi_kind": data["parsed"]["PI_KIND"],
                "gamma_kind": data["parsed"]["GAMMA_KIND"],
                "pi": data["values"]["PI"],
                "euler_gamma": data["values"]["EULER_GAMMA"],
                "minus_one_over_fourpi": data["values"]["MINUS_ONE_OVER_FOURPI"],
                "log_2": data["values"]["LOG_2"],
                "executable_sha256": data["executable_sha256"],
            }
            for variant, data in validated.items()
        },
        "process_wall_seconds": wall,
        "returncode": proc.returncode,
        "validation_error": validation_error,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }

    if len(validated) == 2:
        cand = validated["candidate"]["values"]
        base = validated["baseline"]["values"]
        receipt["analysis"] = {
            "candidate_pi_minus_math_pi": cand["PI"] - math.pi,
            "candidate_gamma_minus_reference":
                cand["EULER_GAMMA"] - CANDIDATE_GAMMA_REFERENCE,
            "baseline_pi": base["PI"],
            "baseline_pi_is_float32_rounding_of_literal":
                base["PI"] == float32_round(BASELINE_PI_LITERAL),
            "candidate_pi_kind": validated["candidate"]["parsed"]["PI_KIND"],
            "candidate_gamma_kind":
                validated["candidate"]["parsed"]["GAMMA_KIND"],
        }

    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out / "constants-source-probe-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n",
                            encoding="utf-8")
    ok = proc.returncode == 0 and sources_unchanged and validation_error is None
    print(json.dumps({
        "ok": ok,
        "returncode": proc.returncode,
        "validation_error": validation_error,
        "process_wall_seconds": wall,
        "compiled_variants": {v: {
            "pi": d["values"]["PI"],
            "euler_gamma": d["values"]["EULER_GAMMA"],
            "executable_sha256": d["executable_sha256"],
        } for v, d in validated.items()},
        "receipt": str(receipt_path),
    }, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
