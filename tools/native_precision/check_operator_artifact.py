# SPDX-License-Identifier: MPL-2.0
"""ONE admitted public operator-probe route check (w5-e1-operator-route-20261010).

Question (criteria.json): can the public panel route admit the exact corrected
E1 artifact separately from the original producer and measure a bounded
preparation/probe on the original coupled geometry? This tool validates the
root-issued admission and atomically consumes it, then runs the actual public
caller under a sys.setprofile guard pinned to exact Delhommeau method code
objects. It is a development route check, NOT W5 qualification, and confers no
force/PSD/energy/full-operator claim.

Safety contract:
- The original producer identity stays the immutable reference-manifest backend;
  the evaluator identity (wheel, sidecar, both native extensions, all three
  Python source hashes) is pinned separately from the exact verified E1 wheel
  and compared against the actually installed backend BEFORE the constructor:
  both the f64 and the f32 extension files are runtime-hashed at execution
  entry. The evaluator is never compared against the producer map and the
  producer is never relabeled as evaluator.
- Fail closed before any optional import or construction on missing, consumed
  or wrong admission, criteria, binding record, sidecar, reference bundle,
  runtime (sys.executable and its sha256), installed ANYhydro module hashes,
  evaluator source map, f64 extension hash, f64 PRE/PI/Gamma, thread pools or
  limits. The admission is consumed by the tested atomic exclusive hard-link
  claim that never overwrites and preserves the original bytes.
- The guard counts the constructor (1), the empty tabulation (1 with
  tabulation_nr==0 and tabulation_nz==0), public evaluate (2) and the fixed
  public Prony override (2), and refuses any default fitting body, cache load,
  positive table, Rankine call or budget excess BEFORE their bodies execute.
- The run is the terminal operator probe on the original two-body coupled case
  (exact inventory asserted) with 864 panels and four receiver rows: never a
  radiation result, never a linear solve. Energy mode is reserved and
  refused; it requires a separate root admission.
- The output directory must not already exist: prior evidence is never
  overwritten. Raw native pairs are saved by the caller before validation.
"""
import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_rankine_artifact as helper  # noqa: E402  (hash-bound reused helper)

CRITERIA_FORMAT = "anyhydro-e1-operator-route-controls-v1"
ADMISSION_FORMAT = "anyhydro-operator-artifact-admission-v1"
PRECISION_LIMIT = "1e-8+1e-6*maxabs(real center gradient)"
FIXED_COEFFICIENTS_SHA256 = ("d5f10c3684cbb71377be9d51adc82a1ad7b4ea0b1dbf5c"
                             "4669af77e9fa1788e7")
EXPECTED_STEPS = [0.001, 0.0005, 0.00025, 0.0001, 5e-05, 1e-05, 5e-06]
SOURCE_PANELS = 864
PROBE_RECEIVERS = 4
EXPECTED_BODIES = 2
EXPECTED_NATIVE_FUTURE = dict(one_shot=True, outer_seconds=180, child_seconds=120,
                              constructors=1, empty_tables=1, positive_tables=0,
                              green_evaluations=2, fixed_overrides=2,
                              default_fits=0, systems=0, retries=0, threads=1)
GUARD_LIMITS = dict(constructor=1, empty_tabulation=1, positive_tabulation=0,
                    wave=2, prony=0, rankine=0)
IMPLEMENTATION_MODULES = ("artifact_bound_panel_profile.py",
                          "panel_integration_candidate.py",
                          "panel_accuracy_reference.py")
BACKEND_FILES = ("green_functions/delhommeau.py", "tools/prony_decomposition.py",
                 "meshes/geometry.py")
BUNDLE_FILES = ("manifest.json", "input.npz", "reference.npz")


class Refused(Exception):
    """Fail-closed refusal; the attempt stops and evidence is preserved."""


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


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
            or criteria.get("probe_receiver_rows") != PROBE_RECEIVERS):
        raise Refused("criteria case shape is not the original 864-panel four-receiver probe")
    if criteria.get("steps") != EXPECTED_STEPS or criteria.get("precision_limit") != PRECISION_LIMIT:
        raise Refused("criteria steps/precision limit differ from the original contract")
    if criteria.get("fixed_coefficients_sha256") != FIXED_COEFFICIENTS_SHA256:
        raise Refused("criteria fixed coefficient identity mismatch")
    if criteria.get("native_worker_execution") is not False:
        raise Refused("criteria must not request worker-side native execution")
    if criteria.get("native_future") != EXPECTED_NATIVE_FUTURE:
        raise Refused("criteria native_future budgets do not match the probe contract")
    reference = Path(criteria.get("original_reference") or "")
    if not reference.is_dir() or any(not (reference/name).is_file() for name in BUNDLE_FILES):
        raise Refused("criteria original reference bundle is incomplete")
    return sources, reference


def validate_admission(admission, paths, criteria):
    if not isinstance(admission, dict) or admission.get("format") != ADMISSION_FORMAT:
        raise Refused("admission format mismatch")
    if admission.get("state") != "admitted":
        raise Refused(f"admission state {admission.get('state')!r} is not admitted")
    if admission.get("consumed") is True:
        raise Refused("admission is already consumed")
    if admission.get("one_execution_only") is not True:
        raise Refused("admission does not allow exactly one execution")
    if "energy_admission" in admission or admission.get("mode") in ("energy", "full_matrix"):
        raise Refused("energy/full-matrix mode is reserved and refused; it requires "
                      "a separate root admission after measured-route adjudication")
    for key, actual in (("script_sha256", paths["script_sha"]),
                        ("criteria_sha256", paths["criteria_sha"]),
                        ("binding_sha256", paths["binding_sha"]),
                        ("sidecar_sha256", paths["sidecar_sha"]),
                        ("reference_manifest_sha256", paths["reference_manifest_sha"])):
        if admission.get(key) != actual:
            raise Refused(f"admission {key} does not match the actual input")
    if admission.get("python_executable") != sys.executable:
        raise Refused("admission python_executable does not match sys.executable")
    if admission.get("python_executable_sha256") != helper.sha256_file(sys.executable):
        raise Refused("admission python_executable_sha256 does not match the actual sys.executable")
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
        admission = helper.load_json(args.admission, "admission")
        evaluator_sources, declared_reference = validate_criteria(criteria)
        if declared_reference.resolve() != Path(args.reference).resolve():
            raise Refused("--reference differs from the criteria original_reference bundle")
        reference_manifest = helper.load_json(Path(args.reference)/"manifest.json",
                                               "reference bundle manifest")
        paths = dict(
            script_sha=helper.sha256_file(Path(__file__).resolve()),
            criteria_sha=helper.sha256_file(args.criteria),
            binding_sha=helper.sha256_file(args.binding),
            sidecar_sha=helper.sha256_file(args.sidecar),
            reference_manifest_sha=hashlib.sha256(
                canonical(reference_manifest).encode("utf-8")).hexdigest(),
            admission=args.admission,
        )
        consumed_path = validate_admission(admission, paths, criteria)
        state["stage"] = "installed_candidate_validation"
        from anyhydro import make_first_slice_bodies
        from anyhydro.artifact_bound_panel_profile import (ArtifactBoundPanelProfile,
                                                           FIXED_COEFFICIENT_OVERRIDE_CODE)
        from anyhydro.capytaine_backend import CapytaineExecutionLimits
        from anyhydro.panel_accuracy_reference import PanelOperatorProbe
        from anyhydro.panel_integration_candidate import run_panel_integration_candidate
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
            raise Refused("binding record implementation modules differ from the installed candidate")
        if binding["evaluator"]["backend_source_sha256"] != evaluator_sources:
            raise Refused("binding record evaluator source map differs from the criteria")
        if (binding["evaluator"]["wheel_sha256"] != criteria["wheel_sha256"]
                or binding["evaluator"]["native_extension_sha256"][
                    "capytaine/green_functions/Delhommeau_float64.cpython-312-x86_64-linux-gnu.so"]
                != criteria["f64_sha256"]):
            raise Refused("binding record evaluator wheel/f64 identity differs from the criteria")
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
            raise Refused("imported f32 extension hash does not match the binding record; "
                         "old or wrong library refused")
        state["f32_extension_sha256"] = f32_sha
        if admission["delhommeau_module_sha256"] != identity["delhommeau_module_sha256"]:
            raise Refused("admission delhommeau_module_sha256 does not match the "
                         "imported Python GF module")
        package = Path(cpt.__file__).parent
        actual_sources = {name: helper.sha256_file(package/name) for name in BACKEND_FILES}
        if actual_sources != evaluator_sources:
            raise Refused("installed evaluator backend sources do not match the "
                         "exact E1 source map")
        state["stage"] = "guarded_probe"
        delhommeau_cls, pinned, budgets, wrappers = helper.pin_methods(cpt, GUARD_LIMITS)
        pinned[FIXED_COEFFICIENT_OVERRIDE_CODE] = "fixed_coefficient_override"
        budgets["fixed_coefficient_override"] = int(EXPECTED_NATIVE_FUTURE["fixed_overrides"])
        state["pinned_wrappers"] = wrappers
        guard = helper.BudgetGuard(pinned, budgets)
        state["counts"] = guard.counts
        state["trace"] = guard.trace
        guard.install()
        try:
            bodies = make_first_slice_bodies(subdivisions=6, lid_policy="submerged_0p1m_v1")
            if len(bodies) != EXPECTED_BODIES:
                raise Refused(f"case inventory is {len(bodies)} bodies, not the exact "
                              f"two-body coupled case")
            limits = CapytaineExecutionLimits(max_total_panels=SOURCE_PANELS,
                                              max_finite_omegas=1, max_wave_directions=1,
                                              n_jobs=1, n_threads=1)
            probe = run_panel_integration_candidate(
                bodies, omega=1.5, water_depth=150., rho=1025., gravity=9.81,
                limits=limits, artifact_directory=output_dir/"case",
                accuracy_profile=profile, operator_probe_receivers=PROBE_RECEIVERS)
        finally:
            guard.uninstall()
        state["stage"] = "probe_verification"
        expected_counts = {"__init__": GUARD_LIMITS["constructor"],
                           "_create_tabulation": GUARD_LIMITS["empty_tabulation"],
                           "_create_or_load_tabulation": GUARD_LIMITS["positive_tabulation"],
                           "evaluate": GUARD_LIMITS["wave"],
                           "find_best_exponential_decomposition": GUARD_LIMITS["prony"],
                           "evaluate_rankine_only": GUARD_LIMITS["rankine"],
                           "fixed_coefficient_override": EXPECTED_NATIVE_FUTURE["fixed_overrides"]}
        if guard.counts != expected_counts:
            raise Refused(f"guard counts {guard.counts} do not match the probe contract "
                         f"{expected_counts}")
        if not isinstance(probe, PanelOperatorProbe):
            raise Refused("caller did not return the terminal operator probe")
        if probe.receiver_count != PROBE_RECEIVERS or probe.source_panel_count != SOURCE_PANELS:
            raise Refused("probe shape does not match the original coupled case")
        status = helper.load_json(output_dir/"case"/"status.json", "probe status")
        if (status.get("state") != "completed_probe_unqualified"
                or status.get("numerical_systems") != 0
                or status.get("new_green_evaluations") != EXPECTED_NATIVE_FUTURE["green_evaluations"]):
            raise Refused("probe status does not record the unqualified terminal state")
        case_file = helper.load_json(output_dir/"case"/"case.json", "probe case")
        metadata = case_file.get("artifact_evaluator") or {}
        if (metadata.get("evaluator", {}).get("backend_source_sha256") != evaluator_sources
                or metadata.get("producer", {}).get("backend") != reference_manifest["backend"]):
            raise Refused("case metadata does not record the distinct producer/evaluator identities")
        state["stage"] = "summary"
        state["elapsed_seconds"] = time.monotonic()-started
        summary = dict(
            format=CRITERIA_FORMAT, status="probe_unqualified", qualified=False,
            probe=dict(case_hash=probe.case_hash, receiver_count=probe.receiver_count,
                       source_panel_count=probe.source_panel_count,
                       body_count=len(bodies),
                       preparation_wall_seconds=probe.preparation_wall_seconds,
                       micro_wall_seconds=probe.micro_wall_seconds,
                       probe_wall_seconds=probe.probe_wall_seconds),
            guard_counts=guard.counts, guard_trace=guard.trace,
            pinned_wrappers=wrappers,
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
            admission_consumed_path=str(consumed_path),
            elapsed_seconds=state["elapsed_seconds"], retries=0,
            scope=("development route admission; terminal unqualified probe; not W5 "
                   "qualification; no force/PSD/energy/full-operator claim"),
        )
        helper.dump_json(output_dir/"summary.json", summary)
        return 0
    except Exception as exc:
        state["elapsed_seconds"] = time.monotonic()-started
        payload = dict(
            format=CRITERIA_FORMAT, status="failed", stage=state.get("stage"),
            error=f"{type(exc).__name__}: {exc}",
            guard_counts=state.get("counts"), guard_trace=state.get("trace"),
            pinned_wrappers=state.get("pinned_wrappers"),
            runtime_identity=state.get("identity"),
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
        description="One admitted public operator-probe route check.")
    parser.add_argument("--criteria", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
