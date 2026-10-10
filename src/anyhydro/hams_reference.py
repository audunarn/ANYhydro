"""Pinned HAMS-MREL radiation reference tooling, not an admitted backend.

No solver is launched here. The release's gravity is fixed at 9.80665, so
radiation-only similarity preserves omega**2/g and explicitly re-dimensionalizes
A/rho and B/(rho*omega). Excitation, hydrostatics and motion outputs are excluded.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from .first_slice import (
    FIRST_SLICE_BODY_IDS, FIRST_SLICE_GRAVITY, FIRST_SLICE_RHO,
    FIRST_SLICE_WATER_DEPTH, make_first_slice_bodies,
)
from .dof import body_dof_keys

HAMS_RELEASE = "v2.2"
HAMS_SOURCE_COMMIT = "cf9583c1609a5d1b904d24e09dff81e383b2ffad"
HAMS_EXECUTABLE_SHA256 = "4abcddfa4d1ef1f75f58377bf55bd6cc14f97be4ac4ae3f0fdd35c9daa7f0fcb"
HAMS_NATIVE_GRAVITY = 9.80665
HAMS_NATIVE_DENSITY = 1000.0
_FORMAT = "anyhydro-hams-first-slice-radiation-v1"
_AXES = tuple(key for body in FIRST_SLICE_BODY_IDS for key in body_dof_keys(body))


class HamsReferenceError(ValueError):
    """Unbound, unsupported, incomplete or malformed reference evidence."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def verify_hams_executable(path: str | Path) -> str:
    """Verify the official asset bytes; this does not execute or qualify them."""
    digest = _sha(Path(path).read_bytes())
    if digest != HAMS_EXECUTABLE_SHA256:
        raise HamsReferenceError("executable differs from pinned official release asset")
    return digest


@dataclass(frozen=True)
class HamsReferenceCase:
    directory: Path
    case_hash: str

    def verify(self) -> dict:
        """Reject edited inputs, replaced sidecars and implicit generalized modes."""
        path = Path(self.directory)
        try:
            manifest = json.loads((path / "case.json").read_text(encoding="utf-8"))
            if _sha(_canonical(manifest)) != self.case_hash:
                raise HamsReferenceError("reference manifest identity changed")
            if manifest["format"] != _FORMAT or manifest["executable_sha256"] != HAMS_EXECUTABLE_SHA256:
                raise HamsReferenceError("unsupported reference format or executable pin")
            names = {p.name for p in (path / "Input").iterdir()}
            if names != set(manifest["input_sha256"]):
                raise HamsReferenceError("input inventory changed; implicit mode overrides are refused")
            for name, digest in manifest["input_sha256"].items():
                if Path(name).name != name or _sha((path / "Input" / name).read_bytes()) != digest:
                    raise HamsReferenceError("reference input identity changed")
            return manifest
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise HamsReferenceError("missing or malformed reference sidecar/input") from exc


def _field(label, value, skip, *, integer=False):
    # These are column widths from the pinned ReadControlFile, not loose tokens.
    return label.ljust(skip) + (f"{int(value):16d}" if integer else f"{value:30.15f}")


def _control(omega, irsp):
    lines = ["ANYhydro radiation reference", "", _field("Waterdepth", 150., 14), "",
             "# Frequencies", _field("0_inf_frequency_limits", 0, 27, integer=True),
             _field("Input_frequency_type", 3, 25, integer=True),
             "Output_frequency_type".ljust(25) + f"{3:17d}",
             _field("Number_of_frequencies", 1, 26, integer=True), f"{omega:.17g}",
             "# End frequencies", "", "# multi body reference",
             _field("Number_of_bodies", 2, 24, integer=True)]
    for i, x in enumerate((-2., 2.), 1):
        lines.append(f"LCS_{i}".ljust(26) + "".join(f"{v:12.3f}" for v in (x, 0., 0., 0.)))
    lines += ["# End bodies", "", "# Headings", _field("Number_of_headings", 1, 23, integer=True),
              "0.0", "# End headings", ""]
    for i, x in enumerate((-2., 2.), 1):
        lines.append(f"Reference_body_center_{i}".ljust(28) + "".join(f"{v:12.3f}" for v in (x, 0., 0.)))
    lines += [_field("Reference_body_length", 1., 26),
              _field("Wave_diffrac_solution", 1, 26, integer=True),
              _field("If_remove_irr_freq", irsp, 23, integer=True),
              _field("Number of threads", 1, 23, integer=True), "", "# Field points",
              _field("Number_of_field_points", 0, 27, integer=True), "# End field points", "",
              "# End control"]
    return "\n".join(lines) + "\n"


def _mesh(mesh, center):
    # Export solid-outward standard winding unchanged. HAMS computes its own
    # inward normal as (v1-v2) x (v3-v2); do not force Capytaine normal equality.
    vertices = mesh.vertices - np.asarray(center)
    lines = ["ANYhydro exact physical hull / numerical waterplane", "", "# Panels Nodes Symmetries",
             f"{len(mesh.faces)} {len(vertices)} 0 0", "", "# Node coordinates"]
    lines += [f"{i} " + " ".join(f"{v:.17g}" for v in xyz) for i, xyz in enumerate(vertices, 1)]
    lines += ["# End coordinates", "", "# Panel connectivity"]
    lines += [f"{i} 3 " + " ".join(str(int(v) + 1) for v in face[:3])
              for i, face in enumerate(mesh.faces, 1)]
    return "\n".join(lines) + "\n"


def _mechanical_stub():
    # The executable always computes motion too. This deliberately synthetic
    # nonsingular auxiliary input is not a physical mass/restoring model; none
    # of its motion/hydrostatic/excitation results may be imported by this route.
    lines = ["Center of Gravity (auxiliary only):", "0 0 -0.25"]
    for label, matrix in (("Body Mass Matrix (auxiliary identity):", np.eye(6)),
                          ("External Linear Damping Matrix:", np.zeros((6, 6))),
                          ("External Quadratic Damping Matrix:", np.zeros((6, 6))),
                          ("Hydrostatic Restoring Matrix (unused):", np.zeros((6, 6))),
                          ("External Restoring Matrix:", np.zeros((6, 6)))):
        lines.append(label)
        # ReadHydroStaticMulti FORMAT(6(2x,E12.5)).
        lines += ["".join(f"  {value:12.5E}" for value in row) for row in matrix]
    return "\n".join(lines) + "\n"


def prepare_hams_first_slice_reference(directory: str | Path, *, subdivisions: int = 6,
                                       omega: float = 1.5,
                                       irregular_frequency_removal: bool = True) -> HamsReferenceCase:
    """Write a new immutable, radiation-only two-box deck; never run a solver.

    The physical hull matches the existing first-slice fixture. HAMS' optional
    numerical waterplane is at z=0, required by its formulation; it is distinct
    from Capytaine's submerged lid. No sheets or other geometries are admitted.
    """
    if isinstance(omega, bool) or not isinstance(omega, (int, float)) or not math.isfinite(omega) or not 0.25 <= omega <= 8.:
        raise HamsReferenceError("omega must be finite and within the fixed-box 0.25..8 rad/s band")
    if type(irregular_frequency_removal) is not bool:
        raise HamsReferenceError("irregular-frequency policy must be an explicit boolean")
    bodies = make_first_slice_bodies(subdivisions=subdivisions)
    native_omega = float(omega) * math.sqrt(HAMS_NATIVE_GRAVITY / FIRST_SLICE_GRAVITY)
    files = {"ControlFile.in": _control(native_omega, int(irregular_frequency_removal))}
    for i, body in enumerate(bodies, 1):
        files[f"HullMesh_{i}.pnl"] = _mesh(body.mesh, body.rotation_center)
        files[f"Hydrostatic_{i}.in"] = _mechanical_stub()
        if irregular_frequency_removal:
            files[f"WaterPlaneMesh_{i}.pnl"] = _mesh(body.lid_mesh, body.rotation_center)
    manifest = dict(format=_FORMAT, release=HAMS_RELEASE, source_commit=HAMS_SOURCE_COMMIT,
                    executable_sha256=HAMS_EXECUTABLE_SHA256, qualification="development_unqualified",
                    source_association="release tag; executable compiler provenance not independently established",
                    omega_target=float(omega), omega_native=native_omega,
                    gravity_target=FIRST_SLICE_GRAVITY, gravity_native=HAMS_NATIVE_GRAVITY,
                    rho_target=FIRST_SLICE_RHO, rho_native=HAMS_NATIVE_DENSITY,
                    water_depth=FIRST_SLICE_WATER_DEPTH, reference_length=1.,
                    input_frequency_type=3, output_frequency_type=3, subdivisions=int(subdivisions),
                    axes=list(_AXES), threads=1, include_frequency_limits=False,
                    numerical_waterplane="z=0" if irregular_frequency_removal else "none",
                    normal_convention="solid-outward standard winding; HAMS native negative cross",
                    auxiliary_mechanics="synthetic; exclude motion/hydrostatics/excitation outputs",
                    hull_mesh_hashes=[b.mesh.mesh_hash for b in bodies],
                    input_sha256={name: _sha(value.encode("utf-8")) for name, value in files.items()})
    path = Path(directory).resolve()
    path.mkdir(parents=True, exist_ok=False)
    (path / "Input").mkdir()
    # The official Windows asset uses `mkdir -p` internally. Pre-create its
    # directories so that platform-specific upstream command is never needed.
    for name in ("Hams_format", "Wamit_format"):
        (path / "Output" / name).mkdir(parents=True)
    for name, value in files.items():
        (path / "Input" / name).write_bytes(value.encode("utf-8"))
    (path / "case.json").write_bytes(_canonical(manifest))
    return HamsReferenceCase(path, _sha(_canonical(manifest)))


@dataclass(frozen=True)
class HamsRadiationPoint:
    case_hash: str
    output_sha256: str
    native_output_sha256: tuple[tuple[str, str], ...]
    omega: float
    native_omega: float
    axes: tuple[str, ...]
    added_mass: np.ndarray
    radiation_damping: np.ndarray
    added_mass_print_error: np.ndarray
    damping_print_error: np.ndarray

    def mean_radiated_power(self, real_rate) -> tuple[float, float]:
        """Mean power and conservative output-rounding bound for real rates.

        This is not a near/far equality assessment or a passivity correction.
        """
        raw = np.asarray(real_rate)
        if np.iscomplexobj(raw):
            raise HamsReferenceError("complex rates require the full complex impedance work contract")
        rate = np.asarray(raw, dtype=float)
        if rate.shape != (12,) or not np.all(np.isfinite(rate)):
            raise HamsReferenceError("rate must contain all twelve finite real components")
        power = .5 * float(rate @ self.radiation_damping @ rate)
        bound = .5 * float(np.abs(rate) @ self.damping_print_error @ np.abs(rate))
        return power, bound


def _number(token, fractional_digits=6):
    match = re.fullmatch(r"[+-]?\d+\.(\d+)[EeDd]([+-]?\d+)", token)
    if not match or len(match[1]) != fractional_digits:
        raise HamsReferenceError("expected pinned ES14.6 numeric output, including exponent")
    number = float(token.replace("D", "E").replace("d", "e"))
    try:
        half_last_digit = .5 * 10. ** (int(match[2]) - len(match[1]))
    except OverflowError as exc:
        raise HamsReferenceError("output exponent out of range") from exc
    if not math.isfinite(number) or not math.isfinite(half_last_digit) or half_last_digit == 0.:
        raise HamsReferenceError("non-finite or unrepresentable output precision")
    return number, half_last_digit


def read_hams_radiation_point(case: HamsReferenceCase, output: str | Path,
                              native_directory: str | Path) -> HamsRadiationPoint:
    """Parse all 144 signed .1 entries, influenced rows / radiating columns.

    Reject truncated/duplicate/extra data and changed input evidence. The first
    column is native omega (OUFT=3), not a WAMIT period. No symmetrization,
    clipping, interpolation, missing-DOF padding or synthetic A-infinity occurs.
    """
    manifest = case.verify()
    try:
        data = Path(output).read_bytes()
        lines = data.decode("ascii").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise HamsReferenceError("missing or non-ASCII radiation output") from exc
    if len(data) > 100_000 or len(lines) != 144:
        raise HamsReferenceError("exactly 144 complete coupled radiation rows are required")
    a, b, ae, be = (np.empty((12, 12)) for _ in range(4))
    seen = set()
    for line in lines:
        if len(line) != 54:
            raise HamsReferenceError("row differs from pinned ES14.6,2I6,2ES14.6 layout")
        w, we = _number(line[:14].strip())
        if abs(w - manifest["omega_native"]) > we + 8 * np.finfo(float).eps * abs(w):
            raise HamsReferenceError("frequency does not match the producing reference case")
        try:
            row, col = int(line[14:20]), int(line[20:26])
        except ValueError as exc:
            raise HamsReferenceError("invalid DOF indices") from exc
        if not 1 <= row <= 12 or not 1 <= col <= 12 or (row, col) in seen:
            raise HamsReferenceError("out-of-range or duplicate DOF pair")
        seen.add((row, col))
        av, av_error = _number(line[26:40].strip())
        bv, bv_error = _number(line[40:54].strip())
        i, j = row - 1, col - 1
        a[i, j] = av * manifest["rho_target"]
        b[i, j] = bv * manifest["rho_target"] * manifest["omega_target"]
        ae[i, j] = av_error * manifest["rho_target"]
        be[i, j] = bv_error * manifest["rho_target"] * manifest["omega_target"]
    # ReorderAmssDamp can zero-fill missing .1 entries. Independently require
    # complete native per-influenced-DOF files and agreement at their precision.
    native_hashes = []
    kn = manifest["omega_native"] ** 2 / manifest["gravity_native"]
    lo, hi = kn, max(1., 2 * kn)
    for _ in range(80):
        mid = .5 * (lo + hi)
        if mid * math.tanh(mid * manifest["water_depth"]) < kn:
            lo = mid
        else:
            hi = mid
    wave_number = .5 * (lo + hi)
    for stem, actual, errors, factor in (
        ("OAMASS", a, ae, manifest["rho_target"] / manifest["rho_native"]),
        ("ODAMPING", b, be, manifest["rho_target"] * manifest["omega_target"] /
         (manifest["rho_native"] * manifest["omega_native"])),
    ):
        for local_row in range(6):
            name = f"{stem}{local_row + 1}.txt"
            try:
                raw = (Path(native_directory) / name).read_bytes()
                rows = raw.decode("ascii").splitlines()
            except (OSError, UnicodeDecodeError) as exc:
                raise HamsReferenceError("complete native radiation files are required") from exc
            if len(raw) > 10_000 or len(rows) != 4:
                raise HamsReferenceError("native radiation file must contain all four body-pair rows")
            native_hashes.append((name, _sha(raw)))
            for pair, line in enumerate(rows):
                if len(line) != 100:
                    raise HamsReferenceError("native row differs from F7.3,1x,F7.3,1x,6E14.5")
                try:
                    wk, w = float(line[:7]), float(line[8:15])
                except ValueError as exc:
                    raise HamsReferenceError("invalid native wave number/frequency") from exc
                if not math.isfinite(wk) or not math.isfinite(w) or abs(wk - wave_number) > .000500000001 or abs(w - manifest["omega_native"]) > .000500000001:
                    raise HamsReferenceError("native wave number/frequency does not match case")
                i = (pair // 2) * 6 + local_row
                for local_col in range(6):
                    value, error = _number(line[16 + 14 * local_col:30 + 14 * local_col].strip(), 5)
                    j = (pair % 2) * 6 + local_col
                    limit = error * factor + errors[i, j] + 16 * np.finfo(float).eps * abs(actual[i, j])
                    if abs(value * factor - actual[i, j]) > limit:
                        raise HamsReferenceError("native and normalized radiation outputs disagree")
    for array in (a, b, ae, be):
        if not np.all(np.isfinite(array)):
            raise HamsReferenceError("dimensionalized output is not finite")
        array.setflags(write=False)
    return HamsRadiationPoint(case.case_hash, _sha(data), tuple(native_hashes), manifest["omega_target"],
                              manifest["omega_native"], tuple(manifest["axes"]), a, b, ae, be)
