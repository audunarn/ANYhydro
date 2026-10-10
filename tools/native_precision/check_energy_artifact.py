# SPDX-License-Identifier: MPL-2.0
"""ONE admitted public coupled-energy point check (w5-e1-energy-check-20261010).

Question (criteria.json): can the public panel route, admitted separately by a
fresh root energy admission, complete the original 864-panel two-body coupled
radiation point and compare its retained physical work and coefficients to
the frozen HAMS reference under the unchanged registered policy? This tool
validates the root-issued admission and atomically consumes it, then runs the
actual public caller under a sys.setprofile guard pinned to exact Delhommeau
method code objects, the public NumPy solve, the caller's imaginary-pole
recomputation and the directional-evidence entry. It is a development
diagnostic, NOT W5 qualification, and confers no PSD/Ainf/RAO/project claim.

Safety contract:
- Fail closed BEFORE any backend import, constructor or numerical work on
  missing, wrong, mutated or consumed admission, criteria, registration
  controls, frozen intent, HAMS point controls/report, binding record,
  sidecar, reference bundle (manifest, input/reference byte hashes), runtime
  (sys.executable and its sha256), installed ANYhydro module hashes, run
  script, evaluator source map, f64/f32 extension hashes or thread pools.
  The admission is consumed by the tested atomic exclusive hard-link claim
  that never overwrites and preserves the original bytes. The reused
  hash-bound helper module is inside the same chain: the admission binds its
  exact sha256, and the checker enforces that pin against the actual helper
  bytes with the standard library (never the helper's own hashing) plus its
  own pinned helper identity, all before consumption, so a helper mutated
  between issuance and execution is refused.
- This CLI executes ONLY mode=energy admissions of its own format; probe
  admissions and any other mode are refused. General numerical backend
  admission remains FALSE: this is ONE separately allowed UNQUALIFIED
  diagnostic, never general backend admission.
- The guard counts the constructor (1), the empty tabulation (1 with
  tabulation_nr==0 and tabulation_nz==0), public evaluate (5: one 43x1 micro
  plus four 864x864 operator batches), the fixed public Prony override (5),
  the public NumPy linear solve (1, pinned through __wrapped__ to its terminal
  Python code object with recorded NumPy source identity), the caller's
  imaginary_pole recomputation (4) and directional_evidence (1), and refuses
  any default fitting body, cache load, positive table, Rankine call, BEM
  solver call or budget excess BEFORE their bodies execute. The systems
  count is enforced by the pinned solve budget, not a label.
- The caller saves every raw native batch BEFORE validation; the output
  directory must not already exist, so prior evidence is never overwritten.
- The retained physical HAMS rate is placed at 12-axis positions [0,2,6,8]
  (first-slice-a::surge/heave, first-slice-b::surge/heave) with zeros
  elsewhere; no reordering, sign or normalization change. Near-field mean
  power, far source-pole powers at angular 64/128/256 and raw node/mean
  residuals are DIAGNOSTICS ONLY: the source-pole far equality is
  constructed, not an independent oracle, and no new acceptance criterion is
  invented here.
- Comparison to the frozen HAMS reference uses the exact registered policy:
  relative 0.03, absolute 1e-8, per-coefficient scale max(|candidate|,
  |reference|) floored at 1e-8, pass only if diff+printed_bound<=limit, fail
  if diff-printed_bound>limit, otherwise unresolved. The candidate is an
  unrounded binary computation with an EXPLICIT zero print bound; the HAMS
  printed bound comes from the frozen metadata. A negative candidate mean
  power stops source admission and is preserved with its formulation
  question; no sign flip. Computational completion is reported separately
  from registered comparison failure; a failed or unresolved comparison can
  never promote completed radiation/project results, and even a fully
  passing point stays UNQUALIFIED with W5 NO_GO unchanged.
"""
import argparse
import hashlib
import importlib
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_rankine_artifact as helper  # noqa: E402  (hash-bound reused helper)

CRITERIA_FORMAT = "anyhydro-e1-energy-check-controls-v1"
ADMISSION_FORMAT = "anyhydro-energy-artifact-admission-v1"
REGISTRATION_FORMAT = "anyhydro-e1-energy-development-registration-v1"
FIXED_COEFFICIENTS_SHA256 = ("d5f10c3684cbb71377be9d51adc82a1ad7b4ea0b1dbf5c"
                             "4669af77e9fa1788e7")
HAMS_CONTROLS_SHA256 = ("75c763721123ef7c0d717cf0e0b3a284b2f0cd5dd7545acdc4f95"
                        "c72063a78cf")
HAMS_REPORT_SHA256 = ("35610ae4db3ba6c6162eaf3ce34946f02d88ec505a979ddc27d218c"
                      "fab0bb8d6")
HAMS_MEAN_POWER_W = 0.0015879742100973749
HAMS_PRINT_BOUND_W = 2.356963116054815e-06
HAMS_SIGN = "positive_resolved"
RELATIVE = 0.03
ABSOLUTE = 1e-8
NEAR_ZERO_FLOOR = 1e-8
SIGN_THRESHOLD = 1e-8
CANDIDATE_PRINT_BOUND = 0.0
SOURCE_PANELS = 864
RECEIVER_NODES = 3456
EXPECTED_BODIES = 2
QUADRATURE_METHOD = "Gauss-Legendre 2"
OMEGA = 1.5
WATER_DEPTH = 150.0
RHO = 1025.0
GRAVITY = 9.81
ROTATION_CENTERS = [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
RETAINED_DOFS = ("first-slice-a::surge", "first-slice-a::heave",
                 "first-slice-b::surge", "first-slice-b::heave")
RETAINED_INDICES = (0, 2, 6, 8)
RETAINED_RATE = (-0.6391982513286226, 0.3025309504266779,
                 -0.6395839022199424, -0.3013852211928673)
ANGULAR_COUNTS = (64, 128, 256)
EXPECTED_NATIVE_FUTURE = dict(one_shot=True, outer_seconds=2700, child_seconds=2400,
                              constructors=1, empty_tables=1, positive_tables=0,
                              green_evaluations=5, fixed_overrides=5,
                              default_fits=0, systems=1, retries=0, threads=1)
GUARD_LIMITS = dict(constructor=1, empty_tabulation=1, positive_tabulation=0,
                    wave=5, prony=0, rankine=0)
EXPECTED_REGISTRATION = dict(retries=0, imaginary_pole_calls=4,
                             negative_power_rejects_source_admission=True,
                             future_child_seconds=2400, threads=1,
                             fixed_overrides=5, native_worker_execution=False,
                             gf_evaluations=5, source_panels=SOURCE_PANELS,
                             receiver_nodes=RECEIVER_NODES,
                             absolute_comparison=ABSOLUTE,
                             dense_solves=1, qualification=False,
                             future_outer_seconds=2700, relative_comparison=RELATIVE,
                             body_count=EXPECTED_BODIES)
COMPARISON_PRECISION = ("pass only diff+printed_bound<=limit; fail if "
                        "diff-printed_bound>limit; otherwise unresolved")
MEAN_POWER_PRECISION = ("positive only if mean minus print bound >1e-8; negative "
                        "if mean plus bound < -1e-8; otherwise unresolved")
IMPLEMENTATION_MODULES = ("artifact_bound_panel_profile.py",
                          "panel_integration_candidate.py",
                          "panel_accuracy_reference.py")
BACKEND_FILES = ("green_functions/delhommeau.py", "tools/prony_decomposition.py",
                 "meshes/geometry.py")
BUNDLE_FILES = ("manifest.json", "input.npz", "reference.npz")
HELPER_MODULE_SHA256 = ("7cad6edb5ce3f7ef4e863d45ada88e9c3cc7a7107eb5b37b208901d"
                        "2c8136e4d")


class Refused(Exception):
    """Fail-closed refusal; the attempt stops and evidence is preserved."""


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _finite_number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and value == value and value not in (float("inf"), float("-inf")))


def _matrix(value, rows, cols):
    if (not isinstance(value, list) or len(value) != rows
            or any(not isinstance(row, list) or len(row) != cols for row in value)
            or not all(_finite_number(cell) for row in value for cell in row)):
        raise Refused(f"expected a finite {rows}x{cols} numeric matrix")
    return [[float(cell) for cell in row] for row in value]


def validate_criteria(criteria):
    if not isinstance(criteria, dict) or criteria.get("format") != CRITERIA_FORMAT:
        raise Refused("criteria format mismatch")
    if not _digest(criteria.get("wheel_sha256")) or not _digest(criteria.get("f64_sha256")):
        raise Refused("criteria wheel/f64 hashes missing or malformed")
    sources = criteria.get("evaluator_backend_source_sha256")
    if (not isinstance(sources, dict) or set(sources) != set(BACKEND_FILES)
            or not all(_digest(v) for v in sources.values())):
        raise Refused("criteria evaluator source map is incomplete or malformed")
    if (criteria.get("source_panels") != SOURCE_PANELS
            or criteria.get("receiver_nodes") != RECEIVER_NODES
            or criteria.get("body_count") != EXPECTED_BODIES
            or criteria.get("quadrature") != QUADRATURE_METHOD
            or criteria.get("omega") != OMEGA or criteria.get("water_depth") != WATER_DEPTH
            or criteria.get("rho") != RHO or criteria.get("gravity") != GRAVITY
            or criteria.get("rotation_centers") != ROTATION_CENTERS):
        raise Refused("criteria case is not the exact original 864-panel two-body "
                      "coupled energy point")
    if (tuple(criteria.get("retained_dofs") or ()) != RETAINED_DOFS
            or tuple(criteria.get("retained_indices") or ()) != RETAINED_INDICES
            or tuple(criteria.get("retained_rate") or ()) != RETAINED_RATE):
        raise Refused("criteria retained DOF/rate mapping differs from the frozen "
                      "HAMS physical contract")
    comparison = criteria.get("comparison")
    if (not isinstance(comparison, dict)
            or comparison.get("relative") != RELATIVE
            or comparison.get("absolute") != ABSOLUTE
            or comparison.get("near_zero_floor") != NEAR_ZERO_FLOOR
            or comparison.get("precision") != COMPARISON_PRECISION
            or comparison.get("candidate_print_bound") != CANDIDATE_PRINT_BOUND):
        raise Refused("criteria comparison policy differs from the registered contract")
    mean_power = criteria.get("mean_power")
    if (not isinstance(mean_power, dict)
            or mean_power.get("nonnegative_absolute_w") != SIGN_THRESHOLD
            or mean_power.get("precision") != MEAN_POWER_PRECISION
            or mean_power.get("hams_mean_power_w") != HAMS_MEAN_POWER_W
            or mean_power.get("hams_print_bound_w") != HAMS_PRINT_BOUND_W):
        raise Refused("criteria mean-power policy differs from the frozen HAMS contract")
    for key in ("registration_sha256", "intent_sha256", "run_script_sha256",
                "hams_controls_sha256", "hams_report_sha256"):
        if not _digest(criteria.get(key)):
            raise Refused(f"criteria {key} missing or malformed")
    if (criteria.get("hams_controls_sha256") != HAMS_CONTROLS_SHA256
            or criteria.get("hams_report_sha256") != HAMS_REPORT_SHA256):
        raise Refused("criteria does not bind the exact frozen HAMS point "
                      "controls/report hashes")
    if criteria.get("fixed_coefficients_sha256") != FIXED_COEFFICIENTS_SHA256:
        raise Refused("criteria fixed coefficient identity mismatch")
    if criteria.get("native_worker_execution") is not False:
        raise Refused("criteria must not request worker-side native execution")
    if criteria.get("negative_power_rejects_source_admission") is not True:
        raise Refused("criteria must keep negative power as a source-admission stop")
    if criteria.get("native_future") != EXPECTED_NATIVE_FUTURE:
        raise Refused("criteria native_future budgets do not match the energy contract")
    reference = Path(criteria.get("original_reference") or "")
    if not reference.is_dir() or any(not (reference/name).is_file() for name in BUNDLE_FILES):
        raise Refused("criteria original reference bundle is incomplete")
    return sources, reference


def validate_registration(registration):
    if (not isinstance(registration, dict)
            or registration.get("format") != REGISTRATION_FORMAT):
        raise Refused("registration controls format mismatch")
    for key, expected in EXPECTED_REGISTRATION.items():
        if registration.get(key) != expected:
            raise Refused(f"registration {key} does not match the adopted energy "
                          f"contract ({registration.get(key)!r} vs {expected!r})")


def validate_intent(intent):
    if not isinstance(intent, dict):
        raise Refused("frozen energy intent is not a JSON object")
    if (tuple(intent.get("retained_dofs") or ()) != RETAINED_DOFS
            or list(intent.get("real_retained_rate") or []) != list(RETAINED_RATE)):
        raise Refused("intent retained DOF/rate mapping differs from the frozen contract")
    comparison = intent.get("comparison")
    if (not isinstance(comparison, dict) or comparison.get("relative") != RELATIVE
            or comparison.get("absolute") != ABSOLUTE
            or comparison.get("near_zero_floor") != NEAR_ZERO_FLOOR
            or comparison.get("precision") != COMPARISON_PRECISION):
        raise Refused("intent comparison policy differs from the registered contract")
    mean_power = intent.get("mean_power")
    if (not isinstance(mean_power, dict)
            or mean_power.get("nonnegative_absolute_w") != SIGN_THRESHOLD
            or mean_power.get("precision") != MEAN_POWER_PRECISION):
        raise Refused("intent mean-power policy differs from the frozen contract")
    if (intent.get("original_controls_sha256") != HAMS_CONTROLS_SHA256
            or intent.get("original_report_sha256") != HAMS_REPORT_SHA256):
        raise Refused("intent does not bind the frozen HAMS point controls/report")
    if (intent.get("saved_hams_mean_power_w") != HAMS_MEAN_POWER_W
            or intent.get("saved_hams_print_bound_w") != HAMS_PRINT_BOUND_W):
        raise Refused("intent saved HAMS mean power differs from the frozen report")
    if intent.get("qualification") is not False:
        raise Refused("intent must keep the energy point unqualified")


def validate_hams_controls(controls):
    if not isinstance(controls, dict):
        raise Refused("HAMS point controls are not a JSON object")
    manifest = controls.get("input_manifest")
    if (not isinstance(manifest, dict) or manifest.get("rho_target") != RHO
            or manifest.get("gravity_target") != GRAVITY
            or manifest.get("omega_target") != OMEGA
            or manifest.get("water_depth") != WATER_DEPTH):
        raise Refused("HAMS controls physical units/rho/gravity/omega conversion "
                      "differ from the candidate case")
    axes = manifest.get("axes")
    if not isinstance(axes, list) or len(axes) != 12:
        raise Refused("HAMS controls must draw the twelve-axis first-slice case")
    for index, name in zip(RETAINED_INDICES, RETAINED_DOFS):
        if axes[index] != name:
            raise Refused(f"HAMS controls axis {index} is {axes[index]!r}, "
                          f"expected {name!r}")
    if (tuple(controls.get("retained_dofs") or ()) != RETAINED_DOFS
            or list(controls.get("real_retained_rate") or []) != list(RETAINED_RATE)):
        raise Refused("HAMS controls retained DOF/rate mapping differs from the "
                      "frozen contract")
    comparison = controls.get("comparison")
    if (not isinstance(comparison, dict) or comparison.get("relative") != RELATIVE
            or comparison.get("absolute") != ABSOLUTE
            or comparison.get("near_zero_floor") != NEAR_ZERO_FLOOR
            or comparison.get("precision") != COMPARISON_PRECISION):
        raise Refused("HAMS controls comparison policy differs from the registered "
                      "contract")
    mean_power = controls.get("mean_power")
    if (not isinstance(mean_power, dict)
            or mean_power.get("nonnegative_absolute_w") != SIGN_THRESHOLD
            or mean_power.get("precision") != MEAN_POWER_PRECISION):
        raise Refused("HAMS controls mean-power policy differs from the frozen contract")
    if controls.get("qualification") not in (False, "development_unqualified"):
        raise Refused("HAMS controls must stay development-unqualified")


def validate_hams_report(report):
    if not isinstance(report, dict):
        raise Refused("HAMS point report is not a JSON object")
    if (report.get("qualification") is not False
            or report.get("backend_admission") is not False):
        raise Refused("HAMS point report must stay unqualified and non-admitting")
    if (report.get("hams_mean_power_w") != HAMS_MEAN_POWER_W
            or report.get("hams_print_bound_w") != HAMS_PRINT_BOUND_W
            or report.get("hams_sign") != HAMS_SIGN):
        raise Refused("HAMS report mean power/sign differs from the frozen metadata")
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, dict):
        raise Refused("HAMS report comparisons packet missing")
    blocks = {}
    for name in ("added_mass", "radiation_damping"):
        block = comparisons.get(name)
        if not isinstance(block, dict):
            raise Refused(f"HAMS report {name} block missing")
        blocks[name] = dict(
            hams_retained=_matrix(block.get("hams_retained"), 4, 4),
            print_bound=_matrix(block.get("print_bound"), 4, 4))
    return blocks


def validate_admission(admission, paths, criteria):
    if not isinstance(admission, dict) or admission.get("format") != ADMISSION_FORMAT:
        raise Refused("admission format mismatch")
    if admission.get("state") != "admitted":
        raise Refused(f"admission state {admission.get('state')!r} is not admitted")
    if admission.get("consumed") is True:
        raise Refused("admission is already consumed")
    if admission.get("one_execution_only") is not True:
        raise Refused("admission does not allow exactly one execution")
    if admission.get("mode") != "energy":
        raise Refused(f"admission mode {admission.get('mode')!r} is not the single "
                      "fresh energy mode this CLI executes")
    for key, actual in (("script_sha256", paths["script_sha"]),
                        ("criteria_sha256", paths["criteria_sha"]),
                        ("binding_sha256", paths["binding_sha"]),
                        ("sidecar_sha256", paths["sidecar_sha"]),
                        ("reference_manifest_sha256", paths["reference_manifest_sha"]),
                        ("registration_sha256", paths["registration_sha"]),
                        ("intent_sha256", paths["intent_sha"]),
                        ("hams_controls_sha256", paths["controls_sha"]),
                        ("hams_report_sha256", paths["report_sha"]),
                        ("run_script_sha256", paths["run_script_sha"])):
        if admission.get(key) != actual:
            raise Refused(f"admission {key} does not match the actual input")
    if not _digest(admission.get("helper_module_sha256")):
        raise Refused("admission helper_module_sha256 missing or malformed")
    if admission["helper_module_sha256"] != paths["helper_sha"]:
        raise Refused("admission helper_module_sha256 does not match the actual "
                      "helper module bytes")
    if paths["helper_sha"] != HELPER_MODULE_SHA256:
        raise Refused("actual helper module does not match the checker's pinned "
                      "hash-bound helper identity")
    if admission.get("python_executable") != sys.executable:
        raise Refused("admission python_executable does not match sys.executable")
    if admission.get("python_executable_sha256") != helper.sha256_file(sys.executable):
        raise Refused("admission python_executable_sha256 does not match the actual "
                      "sys.executable")
    if admission.get("evaluator_backend_source_sha256") != criteria["evaluator_backend_source_sha256"]:
        raise Refused("admission evaluator source map does not match the criteria")
    if (admission.get("f64_extension_sha256") != criteria["f64_sha256"]
            or admission.get("wheel_sha256") != criteria["wheel_sha256"]):
        raise Refused("admission wheel/f64 identity does not match the criteria")
    modules = admission.get("installed_anyhydro_module_sha256")
    if (not isinstance(modules, dict) or set(modules) != set(IMPLEMENTATION_MODULES)
            or not all(_digest(v) for v in modules.values())):
        raise Refused("admission installed ANYhydro module map is incomplete or malformed")
    if not _digest(admission.get("delhommeau_module_sha256")):
        raise Refused("admission does not bind the imported Python GF module hash")
    if admission.get("limits") != criteria["native_future"]:
        raise Refused("admission limits do not match the criteria native_future")
    consumed = Path(paths["admission"]).with_name(Path(paths["admission"]).name + ".consumed")
    if consumed.exists():
        raise Refused(f"admission already consumed: {consumed}")
    return consumed


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def build_rate(dofs):
    """Place the frozen physical HAMS rate at its exact 12-axis positions."""
    dofs = tuple(dofs)
    if len(dofs) != 12:
        raise Refused(f"candidate exposes {len(dofs)} DOFs, not the twelve-axis "
                      "first-slice case")
    rate = [0.0]*12
    for index, name, value in zip(RETAINED_INDICES, RETAINED_DOFS, RETAINED_RATE):
        if dofs[index] != name:
            raise Refused(f"candidate DOF at axis {index} is {dofs[index]!r}, "
                          f"expected the retained {name!r}")
        rate[index] = value
    return rate


def coefficient_verdict(diff, printed_bound, limit):
    if diff + printed_bound <= limit:
        return "pass"
    if diff - printed_bound > limit:
        return "fail"
    return "unresolved"


def compare_block(candidate, reference, print_bound, *, relative=RELATIVE,
                  absolute=ABSOLUTE, floor=NEAR_ZERO_FLOOR):
    """Exact registered per-coefficient comparison; no new acceptance rule."""
    candidate = _matrix(candidate, 4, 4)
    reference = _matrix(reference, 4, 4)
    print_bound = _matrix(print_bound, 4, 4)
    entries = []
    counts = dict(passed=0, failed=0, unresolved=0)
    for i in range(4):
        for j in range(4):
            c, r, b = candidate[i][j], reference[i][j], print_bound[i][j]
            scale = max(abs(c), abs(r))
            if scale < floor:
                scale = floor
            limit = absolute + relative*scale
            diff = abs(c-r)
            verdict = coefficient_verdict(diff, b, limit)
            counts[{"pass": "passed", "fail": "failed",
                     "unresolved": "unresolved"}[verdict]] += 1
            entries.append(dict(row=int(i), column=int(j), candidate=c, reference=r,
                                printed_bound=b, diff=diff, scale=scale, limit=limit,
                                verdict=verdict))
    if counts["failed"]:
        verdict = "failed"
    elif counts["unresolved"]:
        verdict = "unresolved"
    else:
        verdict = "passed"
    return dict(verdict=verdict, counts=counts, coefficients=entries,
                normalization=("per retained translation coefficient max absolute "
                               "candidate/reference; floor 1e-8"),
                candidate_print_bound=CANDIDATE_PRINT_BOUND)


def compare_mean_power(candidate_w, reference_w, reference_bound, *,
                       relative=RELATIVE, absolute=ABSOLUTE,
                       floor=NEAR_ZERO_FLOOR):
    candidate_w = float(candidate_w)
    reference_w = float(reference_w)
    reference_bound = float(reference_bound)
    for label, value in (("candidate mean power", candidate_w),
                         ("reference mean power", reference_w),
                         ("reference print bound", reference_bound)):
        if not _finite_number(value):
            raise Refused(f"{label} is not finite; nonfinite mean power is "
                          "refused, never classified")
    scale = max(abs(candidate_w), abs(reference_w))
    if scale < floor:
        scale = floor
    limit = absolute + relative*scale
    diff = abs(candidate_w-reference_w)
    verdict = coefficient_verdict(diff, float(reference_bound), limit)
    return dict(candidate_w=candidate_w, candidate_print_bound=CANDIDATE_PRINT_BOUND,
                reference_w=reference_w, reference_print_bound=float(reference_bound),
                diff=diff, scale=scale, limit=limit, verdict=verdict)


def classify_sign(mean_w, print_bound, *, threshold=SIGN_THRESHOLD):
    if not _finite_number(mean_w) or not _finite_number(print_bound):
        raise Refused("mean-power sign classification requires finite inputs")
    if mean_w - print_bound > threshold:
        return "positive"
    if mean_w + print_bound < -threshold:
        return "negative"
    return "unresolved"


def _code_identity(code):
    return dict(filename=code.co_filename, first_line=int(code.co_firstlineno),
                name=code.co_name)


def _pin(pinned, budgets, wrappers, name, func, budget):
    code, depth = helper._pinned_code(name, func)
    pinned[code] = name
    budgets[name] = int(budget)
    if depth:
        wrappers[name] = depth
    return code


def run(args):
    output_dir = Path(args.output)
    state = dict(stage="arguments", output_dir_created=False)
    started = time.monotonic()
    guard = None
    try:
        if output_dir.exists():
            raise Refused(f"output directory already exists; prior evidence must "
                          f"never be overwritten: {output_dir}")
        output_dir.mkdir(parents=True)
        state["output_dir_created"] = True
        state["stage"] = "input_validation"
        criteria = helper.load_json(args.criteria, "criteria")
        registration = helper.load_json(args.registration, "registration controls")
        intent = helper.load_json(args.intent, "frozen energy intent")
        controls = helper.load_json(args.controls, "HAMS point controls")
        report = helper.load_json(args.report, "HAMS point report")
        admission = helper.load_json(args.admission, "admission")
        evaluator_sources, declared_reference = validate_criteria(criteria)
        if declared_reference.resolve() != Path(args.reference).resolve():
            raise Refused("--reference differs from the criteria original_reference bundle")
        validate_registration(registration)
        validate_intent(intent)
        validate_hams_controls(controls)
        hams_blocks = validate_hams_report(report)
        reference_manifest = helper.load_json(Path(args.reference)/"manifest.json",
                                              "reference bundle manifest")
        bundle_files = reference_manifest.get("files")
        if (not isinstance(bundle_files, dict)
                or bundle_files.get("input.npz") != helper.sha256_file(Path(args.reference)/"input.npz")
                or bundle_files.get("reference.npz") != helper.sha256_file(Path(args.reference)/"reference.npz")):
            raise Refused("reference bundle input/reference byte hashes do not match "
                          "the canonical manifest")
        paths = dict(
            script_sha=helper.sha256_file(Path(__file__).resolve()),
            criteria_sha=helper.sha256_file(args.criteria),
            binding_sha=helper.sha256_file(args.binding),
            sidecar_sha=helper.sha256_file(args.sidecar),
            reference_manifest_sha=hashlib.sha256(
                canonical(reference_manifest).encode("utf-8")).hexdigest(),
            registration_sha=helper.sha256_file(args.registration),
            intent_sha=helper.sha256_file(args.intent),
            controls_sha=helper.sha256_file(args.controls),
            report_sha=helper.sha256_file(args.report),
            run_script_sha=helper.sha256_file(args.run_script),
            # Standard-library hash of the actual helper bytes; never the
            # helper's own sha256_file, which a mutated helper could forge.
            helper_sha=hashlib.sha256(
                Path(helper.__file__).resolve().read_bytes()).hexdigest(),
            admission=args.admission,
        )
        for key, field in (("registration_sha", "registration_sha256"),
                           ("intent_sha", "intent_sha256"),
                           ("controls_sha", "hams_controls_sha256"),
                           ("report_sha", "hams_report_sha256")):
            if criteria[field] != paths[key]:
                raise Refused(f"criteria {field} does not match the actual input file")
        consumed_path = validate_admission(admission, paths, criteria)
        state["stage"] = "installed_candidate_validation"
        from anyhydro import make_first_slice_bodies
        from anyhydro.artifact_bound_panel_profile import (ArtifactBoundPanelProfile,
                                                            FIXED_COEFFICIENT_OVERRIDE_CODE)
        from anyhydro.capytaine_backend import CapytaineExecutionLimits
        from anyhydro.panel_integration_candidate import (PanelRadiationCandidate,
                                                          imaginary_pole,
                                                          run_panel_integration_candidate)
        import anyhydro
        module_files = {"artifact_bound_panel_profile.py": anyhydro.artifact_bound_panel_profile,
                        "panel_integration_candidate.py": anyhydro.panel_integration_candidate,
                        "panel_accuracy_reference.py": anyhydro.panel_accuracy_reference}
        installed = {name: helper.sha256_file(module.__file__)
                     for name, module in module_files.items()}
        if installed != admission["installed_anyhydro_module_sha256"]:
            raise Refused("installed ANYhydro module hashes do not match the admission")
        profile = ArtifactBoundPanelProfile.load(args.reference, args.binding, args.sidecar)
        binding = json.loads(profile.binding_json)
        if binding["implementation_modules"] != installed:
            raise Refused("binding record implementation modules differ from the "
                         "installed candidate")
        if binding["evaluator"]["backend_source_sha256"] != evaluator_sources:
            raise Refused("binding record evaluator source map differs from the criteria")
        if (binding["evaluator"]["wheel_sha256"] != criteria["wheel_sha256"]
                or binding["evaluator"]["native_extension_sha256"][
                    "capytaine/green_functions/Delhommeau_float64.cpython-312-x86_64-linux-gnu.so"]
                != criteria["f64_sha256"]):
            raise Refused("binding record evaluator wheel/f64 identity differs from "
                          "the criteria")
        state["stage"] = "admission_consumption"
        helper.consume_admission(args.admission, consumed_path)
        state["stage"] = "dependency_import"
        cpt, identity = helper.import_dependencies()
        state["identity"] = identity
        if identity["f64_extension_sha256"] != criteria["f64_sha256"]:
            raise Refused("imported f64 extension hash does not match the criteria; "
                         "old or wrong library refused")
        f32_key = next(key for key in binding["evaluator"]["native_extension_sha256"]
                       if "Delhommeau_float32" in key)
        try:
            f32_mod = importlib.import_module("capytaine.green_functions.Delhommeau_float32")
        except Exception as exc:
            raise Refused(f"float32 extension import failed: {exc}") from exc
        f32_path = getattr(f32_mod, "__file__", None)
        f32_sha = (helper.sha256_file(f32_path)
                   if f32_path and Path(f32_path).is_file() else None)
        if f32_sha != binding["evaluator"]["native_extension_sha256"][f32_key]:
            raise Refused("imported f32 extension hash does not match the binding "
                         "record; old or wrong library refused")
        state["f32_extension_sha256"] = f32_sha
        if admission["delhommeau_module_sha256"] != identity["delhommeau_module_sha256"]:
            raise Refused("admission delhommeau_module_sha256 does not match the "
                          "imported Python GF module")
        package = Path(cpt.__file__).parent
        actual_sources = {name: helper.sha256_file(package/name) for name in BACKEND_FILES}
        if actual_sources != evaluator_sources:
            raise Refused("installed evaluator backend sources do not match the "
                         "exact E1 source map")
        import numpy
        import scipy
        np = numpy
        numpy_identity = dict(
            numpy_version=numpy.__version__,
            scipy_version=scipy.__version__,
            numpy_module_file=str(Path(numpy.__file__).resolve()),
            numpy_module_sha256=helper.sha256_file(numpy.__file__),
            numpy_linalg_module_file=str(Path(numpy.linalg.__file__).resolve()),
            numpy_linalg_module_sha256=helper.sha256_file(numpy.linalg.__file__),
        )
        state["numpy_identity"] = numpy_identity
        state["stage"] = "guarded_energy_point"
        delhommeau_cls, pinned, budgets, wrappers = helper.pin_methods(cpt, GUARD_LIMITS)
        pinned[FIXED_COEFFICIENT_OVERRIDE_CODE] = "fixed_coefficient_override"
        budgets["fixed_coefficient_override"] = int(EXPECTED_NATIVE_FUTURE["fixed_overrides"])
        _pin(pinned, budgets, wrappers, "numpy_linalg_solve", numpy.linalg.solve,
             EXPECTED_NATIVE_FUTURE["systems"])
        _pin(pinned, budgets, wrappers, "imaginary_pole", imaginary_pole,
             EXPECTED_REGISTRATION["imaginary_pole_calls"])
        _pin(pinned, budgets, wrappers, "directional_evidence",
             PanelRadiationCandidate.directional_evidence, 1)
        bem_solver = getattr(cpt, "BEMSolver", None)
        bem_solve = getattr(bem_solver, "solve", None) if bem_solver is not None else None
        if callable(bem_solve):
            _pin(pinned, budgets, wrappers, "capytaine_bem_solve", bem_solve, 0)
        state["pinned_wrappers"] = wrappers
        state["pinned_code_identities"] = {
            name: _code_identity(code) for code, name in pinned.items()}
        guard = helper.BudgetGuard(pinned, budgets)
        state["counts"] = guard.counts
        state["trace"] = guard.trace
        guard.install()
        try:
            bodies = make_first_slice_bodies(subdivisions=6, lid_policy="submerged_0p1m_v1")
            if len(bodies) != EXPECTED_BODIES:
                raise Refused(f"case inventory is {len(bodies)} bodies, not the exact "
                              "two-body coupled case")
            centers = [list(map(float, body.rotation_center)) for body in bodies]
            if centers != ROTATION_CENTERS:
                raise Refused(f"case rotation centers {centers} are not the original "
                              f"coupled poses {ROTATION_CENTERS}")
            limits = CapytaineExecutionLimits(max_total_panels=SOURCE_PANELS,
                                              max_finite_omegas=1, max_wave_directions=1,
                                              n_jobs=1, n_threads=1)
            result = run_panel_integration_candidate(
                bodies, omega=OMEGA, water_depth=WATER_DEPTH, rho=RHO, gravity=GRAVITY,
                limits=limits, artifact_directory=output_dir/"case",
                accuracy_profile=profile, operator_probe_receivers=None)
            if not isinstance(result, PanelRadiationCandidate):
                raise Refused("caller did not return the coupled radiation candidate")
            rate = build_rate(result.dofs)
            evidence = result.directional_evidence(rate)
        finally:
            guard.uninstall()
        state["stage"] = "point_verification"
        expected_counts = {"__init__": GUARD_LIMITS["constructor"],
                           "_create_tabulation": GUARD_LIMITS["empty_tabulation"],
                           "_create_or_load_tabulation": GUARD_LIMITS["positive_tabulation"],
                           "evaluate": GUARD_LIMITS["wave"],
                           "find_best_exponential_decomposition": GUARD_LIMITS["prony"],
                           "evaluate_rankine_only": GUARD_LIMITS["rankine"],
                           "fixed_coefficient_override": EXPECTED_NATIVE_FUTURE["fixed_overrides"],
                           "numpy_linalg_solve": EXPECTED_NATIVE_FUTURE["systems"],
                           "imaginary_pole": EXPECTED_REGISTRATION["imaginary_pole_calls"],
                           "directional_evidence": 1}
        if "capytaine_bem_solve" in guard.counts:
            expected_counts["capytaine_bem_solve"] = 0
        if guard.counts != expected_counts:
            raise Refused(f"guard counts {guard.counts} do not match the energy "
                          f"contract {expected_counts}")
        status = helper.load_json(output_dir/"case"/"status.json", "case status")
        if (status.get("state") != "completed_unqualified"
                or status.get("numerical_systems") != EXPECTED_NATIVE_FUTURE["systems"]
                or status.get("new_green_evaluations") != EXPECTED_NATIVE_FUTURE["green_evaluations"]):
            raise Refused("case status does not record the completed unqualified "
                          "coupled radiation state")
        case_file = helper.load_json(output_dir/"case"/"case.json", "case record")
        metadata = case_file.get("artifact_evaluator") or {}
        if (metadata.get("evaluator", {}).get("backend_source_sha256") != evaluator_sources
                or metadata.get("producer", {}).get("backend") != reference_manifest["backend"]):
            raise Refused("case metadata does not record the distinct "
                          "producer/evaluator identities")
        if len(result.sources) != SOURCE_PANELS or len(result.weights) != SOURCE_PANELS:
            raise Refused("candidate shape does not match the original 864-panel case")
        state["stage"] = "retained_evidence"
        indices = list(RETAINED_INDICES)
        retained = dict(
            added_mass=np.asarray(result.added_mass)[np.ix_(indices, indices)].tolist(),
            radiation_damping=np.asarray(result.radiation_damping)[np.ix_(indices, indices)].tolist(),
            force_per_velocity=np.asarray(result.force_per_velocity)[np.ix_(indices, indices)].tolist(),
        )
        np.savez(output_dir/"retained-blocks.npz",
                 rate=np.asarray(rate, dtype=float),
                 dofs=np.asarray(result.dofs),
                 added_mass=np.asarray(retained["added_mass"]),
                 radiation_damping=np.asarray(retained["radiation_damping"]),
                 force_per_velocity=np.asarray(retained["force_per_velocity"]))
        if not _finite_number(evidence["near_mean_power_w"]):
            raise Refused("candidate near-field mean power is not finite; "
                          "refused before any machine JSON can carry NaN")
        helper.dump_json(output_dir/"directional-evidence.json", dict(
            rate=rate, retained_dofs=list(RETAINED_DOFS),
            retained_indices=list(RETAINED_INDICES), **evidence))
        state["stage"] = "hams_comparison"
        added_comparison = compare_block(retained["added_mass"],
                                         hams_blocks["added_mass"]["hams_retained"],
                                         hams_blocks["added_mass"]["print_bound"])
        damping_comparison = compare_block(retained["radiation_damping"],
                                            hams_blocks["radiation_damping"]["hams_retained"],
                                            hams_blocks["radiation_damping"]["print_bound"])
        power_comparison = compare_mean_power(evidence["near_mean_power_w"],
                                              HAMS_MEAN_POWER_W, HAMS_PRINT_BOUND_W)
        candidate_sign = classify_sign(evidence["near_mean_power_w"],
                                       CANDIDATE_PRINT_BOUND)
        reference_sign = classify_sign(HAMS_MEAN_POWER_W, HAMS_PRINT_BOUND_W)
        force_bound = [[hams_blocks["radiation_damping"]["print_bound"][i][j]
                        + OMEGA*hams_blocks["added_mass"]["print_bound"][i][j]
                        for j in range(4)] for i in range(4)]
        force_reference = [[-hams_blocks["radiation_damping"]["hams_retained"][i][j]
                            + 1j*OMEGA*hams_blocks["added_mass"]["hams_retained"][i][j]
                            for j in range(4)] for i in range(4)]
        force_candidate = [[complex(value) for value in row]
                           for row in retained["force_per_velocity"]]
        force_diff = [[abs(force_candidate[i][j]-force_reference[i][j])
                       for j in range(4)] for i in range(4)]
        force_entries = []
        force_counts = dict(passed=0, failed=0, unresolved=0)
        for i in range(4):
            for j in range(4):
                scale = max(abs(force_candidate[i][j]), abs(force_reference[i][j]))
                if scale < NEAR_ZERO_FLOOR:
                    scale = NEAR_ZERO_FLOOR
                limit = ABSOLUTE + RELATIVE*scale
                verdict = coefficient_verdict(force_diff[i][j], force_bound[i][j], limit)
                force_counts[{"pass": "passed", "fail": "failed",
                              "unresolved": "unresolved"}[verdict]] += 1
                force_entries.append(dict(row=i, column=j, diff=force_diff[i][j],
                                          printed_bound=force_bound[i][j], scale=scale,
                                          limit=limit, verdict=verdict))
        force_comparison = dict(
            verdict=("failed" if force_counts["failed"] else
                     "unresolved" if force_counts["unresolved"] else "passed"),
            counts=force_counts, coefficients=force_entries,
            derived_from=("complex force-per-velocity F=-B+i*omega*A with conservative "
                          "bound |dF|<=|dB|+omega*|dA|; DIAGNOSTIC ONLY, derived from "
                          "the same frozen HAMS numbers, not a new acceptance criterion"),
            registered_predicate=False)
        comparison = dict(
            format="anyhydro-e1-energy-hams-comparison-v1",
            policy=dict(relative=RELATIVE, absolute=ABSOLUTE,
                        near_zero_floor=NEAR_ZERO_FLOOR, precision=COMPARISON_PRECISION,
                        candidate_print_bound=CANDIDATE_PRINT_BOUND,
                        hams_print_bound_source="frozen point-report.json metadata"),
            added_mass=added_comparison,
            radiation_damping=damping_comparison,
            mean_power=power_comparison,
            force_per_velocity=force_comparison,
            candidate_sign=candidate_sign,
            hams_sign=reference_sign,
            sign_rule=MEAN_POWER_PRECISION,
            registered_predicates=dict(
                added_mass_block=added_comparison["verdict"],
                radiation_damping_block=damping_comparison["verdict"],
                mean_power={"pass": "passed", "fail": "failed",
                            "unresolved": "unresolved"}[power_comparison["verdict"]]),
            diagnostics_only=dict(
                far_source_pole_w=dict(zip(map(str, ANGULAR_COUNTS),
                                            evidence["far_mean_powers_w"])),
                far_equality="constructed, not an independent oracle",
                node_residual_max_w=evidence["node_boundary_residual_max"],
                mean_residual_max_w=evidence["mean_boundary_residual_max"],
                node_residual_pressure_work_w=evidence["node_residual_pressure_work_w"],
                hull_node_residual_pressure_work_w=evidence["hull_node_residual_pressure_work_w"],
                node_residual_pressure_work_abs_bound_w=evidence[
                    "node_residual_pressure_work_abs_bound_w"],
                hull_node_residual_pressure_work_abs_bound_w=evidence[
                    "hull_node_residual_pressure_work_abs_bound_w"],
                mechanical_node_work_w=evidence["mechanical_node_work_w"],
                force_work_error_w=evidence["force_work_error_w"]),
            qualification=False)
        helper.dump_json(output_dir/"energy-comparison.json", comparison)
        registered = comparison["registered_predicates"]
        all_passed = all(verdict == "passed" for verdict in registered.values())
        if all_passed:
            point_state = "energy_point_completed_unqualified"
            exit_code = 0
        elif any(verdict == "failed" for verdict in registered.values()):
            point_state = "energy_point_comparison_failed_unqualified"
            exit_code = 1
        else:
            point_state = "energy_point_comparison_unresolved_unqualified"
            exit_code = 1
        negative_power = candidate_sign == "negative"
        state["stage"] = "summary"
        state["elapsed_seconds"] = time.monotonic()-started
        summary = dict(
            format=CRITERIA_FORMAT, status=point_state,
            computation_state="completed_unqualified",
            registered_comparison=("passed" if all_passed else
                                   "failed" if any(v == "failed" for v in registered.values())
                                   else "unresolved"),
            registered_predicates=registered, qualified=False,
            candidate=dict(case_hash=result.case_hash, dofs=list(result.dofs),
                           omega=result.omega, rho=result.rho,
                           water_depth=result.water_depth, k=result.k,
                           source_panels=SOURCE_PANELS, receiver_nodes=RECEIVER_NODES,
                           rate=rate,
                           near_mean_power_w=evidence["near_mean_power_w"],
                           candidate_sign=candidate_sign),
            source_admission=dict(
                general_numerical_admission=False,
                state=("stopped_negative_power" if negative_power else "not_admitted"),
                negative_power_rejects_source_admission=True,
                formulation_question=(None if not negative_power else
                                      "negative near-field mean power at the retained "
                                      "physical rate contradicts the positive HAMS "
                                      "reference; resolve with concrete equations or "
                                      "counterexamples, never a sign flip")),
            guard_counts=guard.counts, guard_trace=guard.trace,
            pinned_wrappers=wrappers,
            pinned_code_identities=state.get("pinned_code_identities"),
            f32_extension_sha256=state.get("f32_extension_sha256"),
            producer=dict(role="original producer only, never the evaluator",
                          reference_manifest_sha256=binding["reference_manifest_sha256"],
                          backend=binding["producer_backend"]),
            evaluator=dict(role="corrected artifact evaluator",
                           external_identity=binding["evaluator"]["external_identity"],
                           wheel_sha256=binding["evaluator"]["wheel_sha256"],
                           sidecar_sha256=binding["evaluator"]["sidecar_sha256"],
                           native_extension_sha256=binding["evaluator"]["native_extension_sha256"],
                           backend_source_sha256=binding["evaluator"]["backend_source_sha256"]),
            runtime_identity=identity,
            numpy_identity=numpy_identity,
            input_hashes=dict(criteria=paths["criteria_sha"],
                              binding=paths["binding_sha"], sidecar=paths["sidecar_sha"],
                              reference_manifest=paths["reference_manifest_sha"],
                              registration=paths["registration_sha"],
                              intent=paths["intent_sha"],
                              hams_controls=paths["controls_sha"],
                              hams_report=paths["report_sha"],
                              run_script=paths["run_script_sha"],
                              python_executable=helper.sha256_file(sys.executable),
                              checker_script=paths["script_sha"],
                              helper_module=helper.sha256_file(helper.__file__)),
            installed_anyhydro_module_sha256=installed,
            admission_consumed_path=str(consumed_path),
            elapsed_seconds=state["elapsed_seconds"], retries=0,
            scope=("development energy diagnostic; unqualified; far source-pole "
                   "equality constructed; no PSD/Ainf/RAO/project claim; W5 NO_GO "
                   "unchanged"),
        )
        helper.dump_json(output_dir/"summary.json", summary)
        return exit_code
    except Exception as exc:
        state["elapsed_seconds"] = time.monotonic()-started
        payload = dict(
            format=CRITERIA_FORMAT, status="failed", stage=state.get("stage"),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
            guard_counts=state.get("counts"), guard_trace=state.get("trace"),
            pinned_wrappers=state.get("pinned_wrappers"),
            runtime_identity=state.get("identity"),
            numpy_identity=state.get("numpy_identity"),
            elapsed_seconds=state.get("elapsed_seconds"), retries=0,
        )
        if state.get("output_dir_created"):
            try:
                helper.dump_json(output_dir/"failure.json", payload)
            except OSError:
                pass
        return 2
    finally:
        if guard is not None:
            guard.uninstall()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="One admitted public coupled-energy point check.")
    parser.add_argument("--criteria", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--registration", required=True)
    parser.add_argument("--intent", required=True)
    parser.add_argument("--controls", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--run-script", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
