# SPDX-License-Identifier: MPL-2.0
"""ONE admitted local public-API corrected-artifact wave/reference check
(w5-wave-artifact-20261010).

Question (criteria.json): does the corrected artifact, evaluated with direct
untabulated 1001-point integration and the declared fixed Prony coefficients,
satisfy the original frozen finite-depth gradient/reference contract, and what
is its native self-FD consistency? Development local check, NOT W5
qualification; no force/PSD/energy/full-operator claim.

Reference pairing (explicit, no conflation): the original reference
manifest/backend bytes remain the OLD producer identity and are never reused as
a backend admission. The actual evaluator is the NEW corrected artifact: a
public subclass of capytaine.Delhommeau that inherits the public constructor,
evaluate and evaluate_rankine_only unchanged and overrides ONLY the public
find_best_exponential_decomposition to return the declared read-only
coefficient array from input.npz for the exact declared kh and method
None/'python'. No new fitting, no cache, no positive table.

Safety contract (mirrors check_rankine_artifact, which is hash-bound by the
criteria and the admission before any native use):
- Fail closed before the constructor on missing, consumed or wrong admission,
  criteria, inputs, sidecar/backend identity, runtime (sys.executable and its
  sha256), the imported Python GF module sha256, the f64 extension hash,
  f64 PRE/PI/Gamma values, thread pools, input model/receiver/coefficient
  shape/dtype/finiteness, declared coefficient hash, k/kh consistency or an
  existing output directory.
- The admission JSON (format anyhydro-wave-artifact-admission-v1) binds this
  script's sha256, the helper sha256, the criteria sha256, the SHA256 of every
  input file, the actual sys.executable, the sha256 of that executable, the
  sha256 of the imported Python delhommeau GF module and the installed f64
  extension sha256, and allows one execution only. It is consumed by the
  existing atomic exclusive hard-link helper; original bytes are preserved and
  admission content is never written or fabricated here.
- A sys.setprofile guard pinned to the exact Delhommeau method code objects
  counts the constructor (1), the empty tabulation (1, both dimensions 0), the
  public Rankine call (1) and the public finite-depth evaluate (1), and refuses
  _create_or_load_tabulation (cache), the default Prony fitting body and any
  budget excess BEFORE their bodies execute. The fixed-coefficient override is
  a pinned Python method with budget 1: it is not a new fit. Counts are honest
  body executions; a cached wrapper return that skips the body is neither
  counted nor refused.
- Raw S/K are copied and saved after EACH public call BEFORE any shape,
  finiteness or predicate validation, and each saved pair is validated
  immediately, so an invalid result is preserved and the run stops before any
  further native call.
- The output directory must not already exist: prior raw/evidence is never
  overwritten, and no failure record is written into a preexisting output.
- The original real-gradient/reference-FD gate is unchanged: real gradient
  versus reference high at ALL 43 rows with limit 1e-8+1e-6*max|native total
  real gradient at the center|, plus the reference FD versus native center on
  the original last two steps. The seven native self-FD errors/pass flags with
  the same limit remain separate diagnostics; a self-FD failure is an
  unresolved numerical inconsistency, never a qualification pass. Complex
  potential error, imaginary gradient error and the free-surface/finite-depth
  remainder (total minus direct Rankine, including reflected/image terms; NOT
  a pure angular wave) are recorded as diagnostics only, with no invented
  threshold. No wave/Rankine extra calls, no new oracle, no full matrix, no
  solve, no HAMS, no dynamics, no below-cutoff case, no retry or relaxation.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_rankine_artifact as helper  # noqa: E402  (hash-bound reused helper)

ADMISSION_FORMAT = "anyhydro-wave-artifact-admission-v1"
CRITERIA_FORMAT = "anyhydro-wave-artifact-local-v1"
SIDECAR_FORMAT = helper.SIDECAR_FORMAT
REFUSED = helper.Refused
INPUT_FILES = ("backend.json", "input.npz", "manifest.json", "reference-128.npz",
               "reference-64.npz", "reference.npz", "source-geometry.npz")
MANIFEST_FORMAT = "anyhydro-panel-accuracy-reference-v1"
MODEL_ARRAYS = ("vertices", "faces", "owner", "hull_mask", "points", "weights",
                "normals", "rigid_trace")
REFERENCE_KEYS = {
    "potential_low": ((43,), np.complex128), "potential_high": ((43,), np.complex128),
    "gradient_low": ((43, 3), np.complex128), "gradient_high": ((43, 3), np.complex128),
}
RICH_KEYS = {
    "s": ((43,), np.complex128), "gradient": ((43, 3), np.complex128),
    "rankine": ((22, 43), np.float64), "rankine_gradient": ((22, 43, 3), np.float64),
    "wave": ((4, 43), np.complex128), "wave_gradient": ((4, 43, 3), np.complex128),
    "receivers": ((43, 3), np.float64),
}
EXPECTED_CONSTRUCTOR = dict(
    tabulation_nr=0, tabulation_nz=0, tabulation_cache_dir=None,
    tabulation_nb_integration_points=1001, tabulation_grid_shape="scaled_nemoh3",
    finite_depth_method="newer",
    finite_depth_prony_decomposition_method="python",
    gf_singularities="low_freq", floating_point_precision="float64",
)
EXPECTED_LIMITS = dict(constructor=1, empty_tabulation=1, positive_tabulation=0,
                       rankine=1, wave=1, prony=0, fixed_coefficients=1,
                       systems=0, retries=0)
EXPECTED_STEPS = (0.001, 0.0005, 0.00025, 0.0001, 5e-05, 1e-05, 5e-06)
OVERRIDE_METHOD_NAME = "fixed_coefficient_override"
RANKINE_SHAPES = ((43, 1), (3, 43, 1))
WAVE_SHAPES = ((43, 1), (3, 43, 1))
FOUR_PI = 4.0 * math.pi


def sha256_file(path):
    return helper.sha256_file(path)


def load_json(path, label):
    return helper.load_json(path, label)


def dump_json(path, payload):
    helper.dump_json(path, payload)


def _array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _is_sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def validate_criteria(criteria):
    if not isinstance(criteria, dict) or criteria.get("format") != CRITERIA_FORMAT:
        raise REFUSED("criteria format mismatch")
    constructor = criteria.get("constructor")
    if not isinstance(constructor, dict):
        raise REFUSED("criteria constructor block missing")
    for key, expected in EXPECTED_CONSTRUCTOR.items():
        if constructor.get(key) != expected:
            raise REFUSED(f"criteria constructor {key}={constructor.get(key)!r} "
                          f"is not the declared zero-table 1001-point route")
    limits = criteria.get("native_limits")
    if not isinstance(limits, dict) or any(limits.get(k) != v for k, v in
                                           EXPECTED_LIMITS.items()):
        raise REFUSED("criteria native_limits do not match the local contract")
    contract = criteria.get("original_gradient_contract")
    if not isinstance(contract, dict):
        raise REFUSED("criteria original_gradient_contract block missing")
    if (contract.get("absolute") != 1e-08 or contract.get("relative") != 1e-06
            or contract.get("scale") != "max abs real native total gradient at center"
            or contract.get("reference_original_step_indices") != [-2, -1]
            or tuple(contract.get("steps") or ()) != EXPECTED_STEPS):
        raise REFUSED("criteria original gradient contract was changed")
    fixed = criteria.get("fixed_coefficients")
    if (not isinstance(fixed, dict) or fixed.get("dtype") != "<f8"
            or fixed.get("shape") != [2, 4] or not _is_sha(fixed.get("sha256"))):
        raise REFUSED("criteria fixed_coefficients identity is not the declared "
                      "2x4 float64 coefficient array")
    if not _is_sha(criteria.get("helper_sha256")):
        raise REFUSED("criteria helper_sha256 missing or malformed")
    if not isinstance(criteria.get("backend_identity"), str) or not isinstance(
            criteria.get("wheel_sha256"), str):
        raise REFUSED("criteria backend identity missing")
    inputs = criteria.get("inputs")
    if (not isinstance(inputs, dict) or set(inputs) != set(INPUT_FILES)
            or not all(_is_sha(v) for v in inputs.values())):
        raise REFUSED("criteria inputs map does not bind every input file")
    k_hex, kh_hex = criteria.get("k_hex"), criteria.get("kh_hex")
    depth, free_surface = criteria.get("water_depth"), criteria.get("free_surface")
    if not isinstance(k_hex, str) or not isinstance(kh_hex, str):
        raise REFUSED("criteria k_hex/kh_hex missing")
    if not isinstance(depth, float) or not isinstance(free_surface, float):
        raise REFUSED("criteria water_depth/free_surface must be fixed floats")
    try:
        k = float.fromhex(k_hex)
        kh = float.fromhex(kh_hex)
    except ValueError as exc:
        raise REFUSED(f"criteria k_hex/kh_hex are not float hex values: {exc}") from exc
    if not math.isfinite(k) or k <= 0.0 or (k * depth).hex() != kh.hex():
        raise REFUSED("criteria k/kh consistency failed (kh != k*water_depth)")
    return dict(constructor=constructor, limits=limits, contract=contract,
                fixed=fixed, k=k, kh=kh, water_depth=depth,
                free_surface=free_surface)


def _load_npz(path, label, keys):
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in keys if key not in data.files]
            if missing:
                raise REFUSED(f"{label} is missing arrays {missing}")
            return {key: np.array(data[key], copy=True) for key in keys}
    except REFUSED:
        raise
    except Exception as exc:
        raise REFUSED(f"{label} is not a readable npz: {exc}") from exc


def _check_arrays(arrays, label, keys):
    for key, (shape, dtype) in keys.items():
        array = arrays[key]
        if array.shape != shape or array.dtype != np.dtype(dtype):
            raise REFUSED(f"{label} {key} has shape/dtype {array.shape}/{array.dtype}, "
                          f"expected {shape}/{np.dtype(dtype).name}")
        if not np.all(np.isfinite(array)):
            raise REFUSED(f"nonfinite values in {label} {key}")


def _finite(arrays, label):
    for key, array in arrays.items():
        if not np.all(np.isfinite(array)):
            raise REFUSED(f"nonfinite values in {label} {key}")


def load_inputs(inputs_dir, criteria, settings):
    """Validate every immutable input before anything native is touched."""
    root = Path(inputs_dir)
    if not root.is_dir():
        raise REFUSED(f"inputs directory missing: {root}")
    for name, digest in criteria["inputs"].items():
        path = root / name
        if not path.is_file():
            raise REFUSED(f"input file missing: {path}")
        if sha256_file(path) != digest:
            raise REFUSED(f"input file hash mismatch for {name}")
    manifest = load_json(root / "manifest.json", "reference manifest")
    if not isinstance(manifest, dict) or manifest.get("format") != MANIFEST_FORMAT:
        raise REFUSED("reference manifest format mismatch")
    if manifest.get("k_hex") != criteria.get("k_hex") or \
            manifest.get("kh_hex") != criteria.get("kh_hex"):
        raise REFUSED("manifest k/kh differ from the criteria declaration")
    if manifest.get("coefficient_identity") != criteria["fixed_coefficients"]:
        raise REFUSED("manifest coefficient identity differs from the criteria")
    binding = manifest.get("case_binding")
    if not isinstance(binding, dict) or \
            binding.get("water_depth") != settings["water_depth"]:
        raise REFUSED("manifest case water_depth differs from the criteria")
    model = _load_npz(root / "input.npz", "input.npz",
                      (*MODEL_ARRAYS, "coefficients", "receivers"))
    _finite({name: model[name] for name in MODEL_ARRAYS}, "input model")
    snapshots = binding.get("snapshot_arrays")
    if not isinstance(snapshots, dict) or set(snapshots) != set(MODEL_ARRAYS):
        raise REFUSED("manifest model snapshot identity block mismatch")
    for name in MODEL_ARRAYS:
        actual = dict(shape=list(model[name].shape), dtype=str(model[name].dtype),
                      sha256=_array_sha256(model[name]))
        if snapshots.get(name) != actual:
            raise REFUSED(f"input model array {name} does not match the frozen "
                          f"original model identity")
    coefficients = model["coefficients"]
    if coefficients.dtype != np.dtype("float64") or coefficients.shape != (2, 4):
        raise REFUSED("declared coefficients are not the 2x4 float64 array")
    if not np.all(np.isfinite(coefficients)):
        raise REFUSED("nonfinite values in declared coefficients")
    if _array_sha256(coefficients) != settings["fixed"]["sha256"]:
        raise REFUSED("declared coefficient hash does not match the criteria "
                      "fixed_coefficients identity")
    receivers = model["receivers"]
    if receivers.dtype != np.dtype("float64") or receivers.shape != (43, 3):
        raise REFUSED("input receivers are not the fixed 43x3 float64 grid")
    if not np.all(np.isfinite(receivers)):
        raise REFUSED("nonfinite values in input receivers")
    vertices, faces = model["vertices"], model["faces"]
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2
            or faces.shape[1] != 3 or faces.dtype.kind not in "iu"):
        raise REFUSED("input model does not contain a complete triangular face 0")
    triangle = vertices[faces[0]]
    center = triangle.mean(axis=0).reshape(1, 3)
    center[0, :2] += [0.37, 0.19]
    depth = settings["water_depth"]
    center[0, 2] -= min(0.17, (depth + center[0, 2]) / 4)
    expected = np.concatenate(
        [center] + [center + sign * step * np.eye(3)[axis]
                    for step in settings["contract"]["steps"]
                    for sign in (1, -1) for axis in range(3)])
    if not np.array_equal(receivers, expected):
        raise REFUSED("input receivers differ from the fixed original step grid")
    reference = _load_npz(root / "reference.npz", "reference.npz", REFERENCE_KEYS)
    _check_arrays(reference, "reference.npz", REFERENCE_KEYS)
    for kind in ("potential", "gradient"):
        low, high = reference[kind + "_low"], reference[kind + "_high"]
        if np.any(np.abs(high - low) > 1e-13 + 1e-11 * np.maximum(np.abs(low),
                                                                  np.abs(high))):
            raise REFUSED("reference64/128 numerical convergence failed")
    rich = {}
    for name, order in (("reference-64.npz", "low"), ("reference-128.npz", "high")):
        rich[name] = _load_npz(root / name, name, RICH_KEYS)
        _check_arrays(rich[name], name, RICH_KEYS)
        if (not np.array_equal(rich[name]["s"], reference[f"potential_{order}"])
                or not np.array_equal(rich[name]["gradient"],
                                      reference[f"gradient_{order}"])):
            raise REFUSED(f"{name} s/gradient are not byte-identical to the "
                          f"original bundle {order} values")
        if not np.array_equal(rich[name]["receivers"], receivers):
            raise REFUSED(f"{name} receivers differ from the input receivers")
    geometry = helper.load_geometry(root / "source-geometry.npz")
    if not np.array_equal(geometry["triangle"], triangle):
        raise REFUSED("source geometry triangle differs from the original model "
                      "face 0")
    return dict(model=model, coefficients=coefficients, receivers=receivers,
                reference=reference, rich=rich, geometry=geometry,
                manifest=manifest)


def validate_admission(admission, criteria, paths, sidecar_f64_sha):
    if not isinstance(admission, dict) or admission.get("format") != ADMISSION_FORMAT:
        raise REFUSED("admission format mismatch")
    if admission.get("state") != "admitted":
        raise REFUSED(f"admission state {admission.get('state')!r} is not admitted")
    if admission.get("consumed") is True:
        raise REFUSED("admission is already consumed")
    if admission.get("one_execution_only") is not True:
        raise REFUSED("admission does not allow exactly one execution")
    bindings = (
        ("script_sha256", paths["script_sha"]),
        ("helper_sha256", paths["helper_sha"]),
        ("criteria_sha256", paths["criteria_sha"]),
    )
    for key, actual in bindings:
        if admission.get(key) != actual:
            raise REFUSED(f"admission {key} does not match the actual input")
    if admission.get("helper_sha256") != criteria.get("helper_sha256"):
        raise REFUSED("admission helper_sha256 differs from the criteria helper")
    if admission.get("inputs") != criteria.get("inputs"):
        raise REFUSED("admission input SHA map does not match the criteria inputs")
    if admission.get("python_executable") != sys.executable:
        raise REFUSED("admission python_executable does not match sys.executable")
    if admission.get("python_executable_sha256") != sha256_file(sys.executable):
        raise REFUSED("admission python_executable_sha256 does not match "
                      "the actual sys.executable")
    if admission.get("f64_extension_sha256") != sidecar_f64_sha:
        raise REFUSED("admission f64 extension hash does not match the sidecar")
    consumed = Path(paths["admission"]).with_name(
        Path(paths["admission"]).name + ".consumed")
    if consumed.exists():
        raise REFUSED(f"admission already consumed: {consumed}")
    return consumed


def build_evaluator_class(cpt, coefficients, kh_hex):
    """Public fixed-coefficient evaluator: only the public Prony entry is
    overridden; the inherited public constructor/evaluate/Rankine run the
    actual pinned native backend. The override is not a new fit."""

    class FixedCoefficientDelhommeau(cpt.Delhommeau):
        def find_best_exponential_decomposition(self, dimensionless_wavenumber,
                                                 *, method=None):
            if method not in (None, "python"):
                raise REFUSED(f"fixed-coefficient evaluator refuses decomposition "
                               f"method {method!r}")
            if float(dimensionless_wavenumber).hex() != kh_hex:
                raise REFUSED("fixed-coefficient evaluator refuses changed kh")
            return coefficients

    return FixedCoefficientDelhommeau


def pin_evaluator(cpt, delhommeau_cls, limits):
    """Pin the base methods plus the fixed-coefficient override code."""
    _, pinned, budgets, wrappers = helper.pin_methods(cpt, limits)
    override_code = delhommeau_cls.find_best_exponential_decomposition.__code__
    if override_code in pinned:
        raise REFUSED("fixed-coefficient override code collides with a pinned "
                      "base method")
    pinned[override_code] = OVERRIDE_METHOD_NAME
    budgets[OVERRIDE_METHOD_NAME] = int(limits["fixed_coefficients"])
    return pinned, budgets, wrappers


def run_public_calls(gf, mesh, receivers, settings, output_dir):
    """Exact two public calls; each raw pair is stored BEFORE validation and
    validated immediately, so an invalid result is preserved and no further
    native call is made."""
    base = dict(adjoint_double_layer=True, early_dot_product=False,
                diagonal_term_in_double_layer=False)
    raw = {}
    s_rankine, k_rankine = gf.evaluate_rankine_only(receivers, mesh, **base)
    raw["s_rankine"] = np.array(s_rankine, copy=True)
    raw["k_rankine"] = np.array(k_rankine, copy=True)
    np.savez(str(output_dir / "wave-raw.npz"), **raw)
    if (raw["s_rankine"].shape, raw["k_rankine"].shape) != RANKINE_SHAPES:
        raise REFUSED(f"rankine evaluation shapes "
                      f"{(raw['s_rankine'].shape, raw['k_rankine'].shape)} "
                      f"do not match the contract {RANKINE_SHAPES}")
    for key in ("s_rankine", "k_rankine"):
        if raw[key].dtype.kind != "f":
            raise REFUSED(f"rankine raw {key} is not real float64")
        if not np.all(np.isfinite(raw[key])):
            raise REFUSED(f"nonfinite values in raw {key}")
    s_wave, k_wave = gf.evaluate(receivers, mesh,
                                 free_surface=settings["free_surface"],
                                 water_depth=settings["water_depth"],
                                 wavenumber=settings["k"], **base)
    raw["s_wave"] = np.array(s_wave, copy=True)
    raw["k_wave"] = np.array(k_wave, copy=True)
    np.savez(str(output_dir / "wave-raw.npz"), **raw)
    if (raw["s_wave"].shape, raw["k_wave"].shape) != WAVE_SHAPES:
        raise REFUSED(f"wave evaluation shapes "
                      f"{(raw['s_wave'].shape, raw['k_wave'].shape)} "
                      f"do not match the contract {WAVE_SHAPES}")
    for key in ("s_wave", "k_wave"):
        if raw[key].dtype.kind != "c":
            raise REFUSED(f"wave raw {key} is not complex")
        if not (np.all(np.isfinite(raw[key].real))
                and np.all(np.isfinite(raw[key].imag))):
            raise REFUSED(f"nonfinite values in raw {key}")
    return raw


def compare_with_reference(raw, inputs, settings):
    """EXACT original comparison; self-FD, complex, imaginary and remainder
    errors remain separate diagnostics."""
    reference = inputs["reference"]
    potential_high = reference["potential_high"]
    gradient_high = reference["gradient_high"]
    rich_high = inputs["rich"]["reference-128.npz"]
    steps = settings["contract"]["steps"]
    absolute = float(settings["contract"]["absolute"])
    relative = float(settings["contract"]["relative"])
    s_wave = raw["s_wave"]
    k_wave = raw["k_wave"]
    s_rankine = raw["s_rankine"]
    k_rankine = raw["k_rankine"]
    center = k_wave[:, 0, 0]
    scale = float(np.max(np.abs(center.real)))
    limit = absolute + relative * scale
    errors = np.abs(k_wave[:, :, 0].T.real - gradient_high.real)
    reference_gate_passed = bool(np.all(errors <= limit))
    fd = np.stack([(s_wave[1 + 6 * i:4 + 6 * i, 0].real
                    - s_wave[4 + 6 * i:7 + 6 * i, 0].real) / (2 * h)
                   for i, h in enumerate(steps)])
    reference_fd = np.stack([(potential_high[1 + 6 * i:4 + 6 * i].real
                              - potential_high[4 + 6 * i:7 + 6 * i].real) / (2 * h)
                             for i, h in enumerate(steps)])
    native_fd_error = np.max(np.abs(fd - center.real), axis=1)
    reference_native_fd_error = np.max(np.abs(reference_fd - center.real), axis=1)
    original_steps_passed = bool(np.all(reference_native_fd_error[-2:] <= limit))
    reference_fd_errors = np.max(np.abs(reference_fd - gradient_high[0].real), axis=1)
    potential_error = float(np.max(np.abs(s_wave[:, 0] - potential_high)))
    imaginary_gradient_error = float(np.max(np.abs(k_wave[:, :, 0].T.imag
                                                   - gradient_high.imag)))
    # Free-surface/finite-depth remainder: total minus DIRECT Rankine. The rich
    # reference rankine[0]/rankine_gradient[0] is the unnormalized direct term,
    # so the direct reference is -term/(4*pi). This remainder includes
    # reflected/image terms; it is NOT a pure angular wave.
    direct_reference_s = -rich_high["rankine"][0] / FOUR_PI
    direct_reference_g = -rich_high["rankine_gradient"][0] / FOUR_PI
    reference_remainder_s = potential_high - direct_reference_s
    reference_remainder_g = gradient_high - direct_reference_g
    native_remainder_s = s_wave[:, 0] - s_rankine[:, 0]
    native_remainder_g = k_wave[:, :, 0] - k_rankine[:, :, 0]
    remainder_potential_error = float(np.max(np.abs(native_remainder_s
                                                   - reference_remainder_s)))
    remainder_gradient_error = float(np.max(np.abs(native_remainder_g.T
                                                  - reference_remainder_g)))
    audit = dict(
        method="corrected-artifact-wave-vs-original-frozen-reference-v1",
        error=float(errors.max()), limit=limit, scale=scale,
        passed=bool(reference_gate_passed and original_steps_passed),
        reference_gradient_gate_passed=reference_gate_passed,
        reference_original_steps_passed=original_steps_passed,
        steps=list(steps),
        reference_fd_vs_native_center_errors=reference_native_fd_error.tolist(),
        native_self_fd_errors=native_fd_error.tolist(),
        native_self_fd_passed=(native_fd_error <= limit).tolist(),
        reference_fd_errors=reference_fd_errors.tolist(),
        potential_error=potential_error,
        imaginary_gradient_error=imaginary_gradient_error,
        free_surface_finite_depth_remainder=dict(
            label=("total minus direct Rankine; includes reflected/image "
                   "terms; NOT a pure angular wave"),
            potential_error=remainder_potential_error,
            gradient_error=remainder_gradient_error),
        empirical_reference_convergence=True, producer_attested=False,
        qualified=False,
        self_fd_note=("native self-FD flags are separate diagnostics with the "
                      "same limit; any failure is an unresolved numerical "
                      "inconsistency, never a qualification pass"),
        diagnostic_only_note=("complex potential error, imaginary gradient "
                              "error and remainder errors are diagnostics "
                              "only; no threshold is invented and the "
                              "original reference gate is unchanged"),
    )
    arrays = dict(
        gradient_errors=errors, native_fd=fd, reference_fd=reference_fd,
        native_remainder_s=native_remainder_s,
        native_remainder_g=native_remainder_g,
        reference_remainder_s=reference_remainder_s,
        reference_remainder_g=reference_remainder_g)
    return audit, arrays


def write_failure(output_dir, state, exc):
    if not state.get("output_dir_created"):
        return
    payload = dict(
        format=CRITERIA_FORMAT,
        status="failed",
        stage=state.get("stage"),
        error=f"{type(exc).__name__}: {exc}",
        guard_counts=state.get("counts"),
        guard_trace=state.get("trace"),
        pinned_wrappers=state.get("pinned_wrappers"),
        runtime_identity=state.get("identity"),
        producer_identity=state.get("producer_identity"),
        evaluator_identity=state.get("evaluator_identity"),
        elapsed_seconds=state.get("elapsed_seconds"),
        partial_results=("wave-raw.npz holds every raw S/K stored before the "
                         "failing validation, when any call completed"),
        retries=0,
    )
    try:
        dump_json(Path(output_dir) / "failure.json", payload)
    except OSError:
        pass


def run(args):
    output_dir = Path(args.output)
    state = dict(stage="arguments", output_dir_created=False)
    started = time.monotonic()
    guard = None
    try:
        if output_dir.exists():
            raise REFUSED(f"output directory already exists; prior raw and "
                          f"evidence must never be overwritten: {output_dir}")
        output_dir.mkdir(parents=True)
        state["output_dir_created"] = True
        state["stage"] = "input_validation"
        criteria = load_json(args.criteria, "criteria")
        settings = validate_criteria(criteria)
        sidecar = load_json(Path(args.inputs) / "backend.json", "backend sidecar")
        sidecar_f64_sha = helper.validate_sidecar(sidecar, criteria)
        admission = load_json(args.admission, "admission")
        paths = dict(
            script_sha=sha256_file(Path(__file__).resolve()),
            helper_sha=sha256_file(Path(helper.__file__).resolve()),
            criteria_sha=sha256_file(args.criteria),
            admission=args.admission,
        )
        if paths["helper_sha"] != criteria.get("helper_sha256"):
            raise REFUSED("helper sha256 does not match the criteria helper_sha256; "
                          "the reused helper must be hash-bound before native use")
        inputs = load_inputs(args.inputs, criteria, settings)
        state["producer_identity"] = dict(
            role="OLD producer identity of the original reference bundle; "
                 "bytes unchanged and never reused as backend admission",
            manifest_backend=inputs["manifest"].get("backend"),
            manifest_files=inputs["manifest"].get("files"),
            criteria_backend_identity=criteria.get("backend_identity"),
            criteria_wheel_sha256=criteria.get("wheel_sha256"),
            reference_pairing=criteria.get("reference_pairing"))
        consumed_path = validate_admission(admission, criteria, paths,
                                           sidecar_f64_sha)
        state["stage"] = "dependency_import"
        cpt, identity = helper.import_dependencies()
        if identity["f64_extension_sha256"] != sidecar_f64_sha:
            raise REFUSED("imported f64 extension hash does not match the sidecar; "
                          "old or wrong library refused")
        if admission.get("delhommeau_module_sha256") != identity["delhommeau_module_sha256"]:
            raise REFUSED("admission delhommeau_module_sha256 does not match "
                          "the imported Python GF module")
        state["identity"] = identity
        state["stage"] = "admission_consumption"
        helper.consume_admission(args.admission, consumed_path)
        state["stage"] = "guarded_construction"
        coefficients = np.array(inputs["coefficients"], copy=True)
        coefficients.setflags(write=False)
        delhommeau_cls = build_evaluator_class(cpt, coefficients,
                                               criteria["kh_hex"])
        pinned, budgets, wrappers = pin_evaluator(cpt, delhommeau_cls,
                                                  criteria["native_limits"])
        state["pinned_wrappers"] = wrappers
        guard = helper.BudgetGuard(pinned, budgets)
        state["counts"] = guard.counts
        state["trace"] = guard.trace
        guard.install()
        constructor = settings["constructor"]
        gf = delhommeau_cls(
            tabulation_nr=constructor["tabulation_nr"],
            tabulation_nz=constructor["tabulation_nz"],
            tabulation_cache_dir=constructor["tabulation_cache_dir"],
            tabulation_nb_integration_points=constructor["tabulation_nb_integration_points"],
            tabulation_grid_shape=constructor["tabulation_grid_shape"],
            finite_depth_method=constructor["finite_depth_method"],
            finite_depth_prony_decomposition_method=constructor["finite_depth_prony_decomposition_method"],
            gf_singularities=constructor["gf_singularities"],
            floating_point_precision=constructor["floating_point_precision"],
        )
        effective = getattr(gf, "exportable_settings", None)
        if not isinstance(effective, dict):
            raise REFUSED("evaluator exposes no exportable settings")
        for key, expected in constructor.items():
            if key == "tabulation_cache_dir":
                continue
            if effective.get(key) != expected:
                raise REFUSED(f"evaluator effective setting {key}={effective.get(key)!r} "
                              f"differs from the exact criteria setting {expected!r}")
        state["evaluator_identity"] = dict(
            role="NEW corrected artifact evaluator; public subclass overriding "
                 "only the public fixed-coefficient Prony entry",
            green_function=effective.get("green_function"),
            subclass_module=str(Path(__file__).resolve()),
            script_sha256=paths["script_sha"],
            helper_sha256=paths["helper_sha"],
            criteria_sha256=paths["criteria_sha"],
            input_sha256=dict(criteria["inputs"]),
            fixed_coefficients_sha256=settings["fixed"]["sha256"],
            k_hex=criteria["k_hex"],
            kh_hex=criteria["kh_hex"],
            constructor_settings=dict(constructor))
        state["stage"] = "mesh_construction"
        mesh = helper.build_mesh(cpt, inputs["geometry"])
        state["stage"] = "public_calls"
        raw = run_public_calls(gf, mesh, inputs["receivers"], settings,
                               output_dir)
        state["stage"] = "comparison"
        audit, arrays = compare_with_reference(raw, inputs, settings)
        np.savez(str(output_dir / "wave-comparison.npz"), **arrays)
        dump_json(output_dir / "wave-comparison.json", audit)
        state["stage"] = "summary"
        state["elapsed_seconds"] = time.monotonic() - started
        self_fd_all_passed = bool(all(audit["native_self_fd_passed"]))
        summary = dict(
            format=CRITERIA_FORMAT,
            status="passed" if audit["passed"] else "failed",
            all_passed=audit["passed"],
            reference_gradient_gate_passed=audit["reference_gradient_gate_passed"],
            reference_original_steps_passed=audit["reference_original_steps_passed"],
            native_self_fd=dict(
                all_passed=self_fd_all_passed,
                errors=audit["native_self_fd_errors"],
                passed=audit["native_self_fd_passed"],
                note=(audit["self_fd_note"] if not self_fd_all_passed else
                      "all seven native self-FD flags passed at the same limit")),
            guard_counts=guard.counts,
            guard_trace=guard.trace,
            pinned_wrappers=wrappers,
            runtime_identity=identity,
            producer_identity=state["producer_identity"],
            evaluator_identity=state["evaluator_identity"],
            admission_consumed_path=str(consumed_path),
            elapsed_seconds=state["elapsed_seconds"],
            retries=0,
            qualified=False,
            scope=("development local admission; not W5 qualification; no "
                   "force/PSD/energy/full-operator claim; diagnostics are not "
                   "qualification"),
        )
        dump_json(output_dir / "summary.json", summary)
        return 0 if audit["passed"] else 1
    except Exception as exc:
        state["elapsed_seconds"] = time.monotonic() - started
        write_failure(output_dir, state, exc)
        return 2
    finally:
        if guard is not None:
            guard.uninstall()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="One admitted local public-API corrected-artifact "
                    "wave/reference check.")
    parser.add_argument("--criteria", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
