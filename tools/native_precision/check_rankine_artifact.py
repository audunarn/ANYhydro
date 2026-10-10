# SPDX-License-Identifier: MPL-2.0
"""ONE admitted local public-API Rankine artifact check (w5-rankine-artifact-20261010).

Question (criteria.json): does the built precision-corrected backend satisfy the
unchanged Rankine local contract through its public API? This tool executes the
single real native attempt only after a root-issued admission is validated and
atomically consumed. It is a development local check, NOT W5 qualification, and
confers no force/PSD/energy/full-operator claim.

Safety contract:
- Fail closed before the constructor on missing, consumed or wrong admission,
  criteria, geometry, sidecar, backend identity, runtime (sys.executable and
  its sha256), the imported Python GF module sha256, the f64 extension hash,
  f64 PRE/PI/Gamma values, or thread pools.
- The admission JSON binds this script's sha256, the criteria/geometry/sidecar
  sha256s, the actual sys.executable, the sha256 of that executable, the
  sha256 of the imported Python delhommeau GF module and the installed f64
  extension sha256, and allows one execution only. It is consumed by an
  atomic exclusive same-directory hard-link claim to '<admission>.consumed'
  that fails if any consumed record already exists, never overwrites it and
  preserves the original admission bytes; admission content is never written
  or fabricated here.
- The f64 extension module actually imported by the constructor is imported
  first, its file is hashed, and its PRE/PI/EULER_GAMMA are read via public
  Fortran module attributes. float64 PI (math.pi) and Gamma (float64 reference)
  are required; the float32 module and any old library are refused by hash.
- A sys.setprofile guard pinned to the exact Delhommeau method code objects
  counts the constructor (1), the empty tabulation (1 with tabulation_nr==0
  and tabulation_nz==0 read from the frame at the call event) and the public
  Rankine calls (4), and refuses _create_or_load_tabulation (cache),
  evaluate (wave), find_best_exponential_decomposition (Prony) and any budget
  excess BEFORE their bodies execute. Pinned code objects are the innermost
  Python bodies behind supported wrappers: the public lru_cache Prony entry
  is unwrapped via __wrapped__, and any wrapper shape without a terminal
  Python code object is refused. Counts are honest body executions; a cached
  wrapper return that skips the body is neither counted nor refused, and no
  new fitting is allowed in this slice. No wave evaluation, no linear solve,
  no retry.
- Raw S/K are copied and saved after EACH public call BEFORE any shape,
  finiteness or predicate validation, and each saved pair is validated
  immediately, so an invalid result is preserved and the run stops before
  any further native call.
- The output directory must not already exist: prior raw/evidence is never
  overwritten, and no failure record is written into a preexisting output.
- The byte-frozen saved geometry is used as-is; no new geometry or tolerance.
  The six predicates, their relative scales and the near-plane exclusion are
  unchanged from the historical implementation. Expected solid angles use
  math.pi, never native PI.
- The parent applies a Linux process-tree 60 s timeout; this tool records
  elapsed wall seconds but never compares clocks or enforces a deadline.
"""
import argparse
import hashlib
import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ADMISSION_FORMAT = "anyhydro-rankine-artifact-admission-v1"
CRITERIA_FORMAT = "anyhydro-rankine-artifact-local-v1"
SIDECAR_FORMAT = "anyhydro-backend-artifact-identity-v1"
F64_MODULE_NAME = "capytaine.green_functions.Delhommeau_float64"
GAMMA_REFERENCE = float("0.577215664901532860606512090082402431")
QUADRATURE_METHOD = "Gauss-Legendre 2"
PREDICATE_NAMES = (
    "off_panel_adjoint_normal_vs_solid_angle",
    "source_gradient_is_minus_receiver_gradient",
    "on_panel_principal_value_normal_zero",
    "self_mesh_on_off_increment_half_normal",
    "normalized_gauss_weight_mean_preserves_half",
    "smallest_above_cutoff_disk_tail_bound",
)
CALL_NAMES = (
    "adjoint_cloud_no_jump",
    "source_cloud_no_jump",
    "self_mesh_adjoint_no_jump",
    "self_mesh_adjoint_with_jump",
)
EXPECTED_SHAPES = ((35, 1), (3, 35, 1), (35, 1), (3, 35, 1),
                   (1, 1), (3, 1, 1), (1, 1), (3, 1, 1))
GEOMETRY_KEYS = {
    "triangle": ((3, 3), np.float64), "cloud": ((35, 3), np.float64),
    "normal": ((3,), np.float64), "base_points": ((5, 3), np.float64),
    "gauss_points": ((4, 3), np.float64), "gauss_weights": ((4,), np.float64),
    "face_radius": ((), np.float64), "base_index": ((35,), None),
    "eps_factor": ((35,), np.float64), "side": ((35,), None),
    "expected_solid_angle": ((35,), np.float64),
    "edge_distance": ((35,), np.float64),
}
GEOMETRY_TOLERANCES = dict(
    points=dict(rtol=1e-13, atol=1e-13),
    weights=dict(rtol=1e-13, atol=1e-15),
    normals=dict(rtol=1e-13, atol=1e-14),
    radius=dict(rtol=1e-13, atol=1e-14),
)
NEAR_PLANE_CUTOFF_FACTOR = 1e-4
PINNED_METHODS = ("__init__", "_create_tabulation", "_create_or_load_tabulation",
                  "evaluate", "find_best_exponential_decomposition",
                  "evaluate_rankine_only")


class Refused(Exception):
    """Fail-closed refusal; the attempt stops and evidence is preserved."""


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path, label):
    p = Path(path)
    if not p.is_file():
        raise Refused(f"{label} missing: {p}")
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise Refused(f"{label} is not valid JSON: {p}") from exc


def dump_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n",
                          encoding="utf-8")


def scalar(value):
    """Convert a 0-d NumPy Fortran parameter to a plain Python scalar."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    item = getattr(value, "item", None)
    return item() if callable(item) else None


def validate_criteria(criteria):
    if not isinstance(criteria, dict) or criteria.get("format") != CRITERIA_FORMAT:
        raise Refused("criteria format mismatch")
    constructor = criteria.get("constructor")
    if not isinstance(constructor, dict):
        raise Refused("criteria constructor block missing")
    if (constructor.get("tabulation_nr") != 0
            or constructor.get("tabulation_nz") != 0
            or constructor.get("tabulation_cache_dir") is not None
            or constructor.get("floating_point_precision") != "float64"):
        raise Refused("criteria constructor is not the zero-table float64 route")
    limits = criteria.get("native_limits")
    expected_limits = dict(constructor=1, empty_tabulation=1, positive_tabulation=0,
                           rankine=4, wave=0, prony=0, systems=0, retries=0)
    if not isinstance(limits, dict) or any(limits.get(k) != v for k, v in
                                           expected_limits.items()):
        raise Refused("criteria native_limits do not match the local contract")
    absolute, relative = criteria.get("absolute"), criteria.get("relative")
    if not (isinstance(absolute, (int, float)) and absolute > 0
            and isinstance(relative, (int, float)) and relative > 0):
        raise Refused("criteria absolute/relative limits missing or nonpositive")
    if tuple(criteria.get("predicates") or ()) != PREDICATE_NAMES:
        raise Refused("criteria predicate list does not match the six predicates")
    return float(absolute), float(relative)


def validate_sidecar(sidecar, criteria):
    if not isinstance(sidecar, dict) or sidecar.get("format") != SIDECAR_FORMAT:
        raise Refused("backend sidecar format mismatch")
    if sidecar.get("external_identity") != criteria.get("backend_identity"):
        raise Refused("sidecar identity does not match criteria backend_identity")
    if sidecar.get("wheel_sha256") != criteria.get("wheel_sha256"):
        raise Refused("sidecar wheel hash does not match criteria wheel_sha256")
    extensions = sidecar.get("native_extension_sha256")
    if not isinstance(extensions, dict):
        raise Refused("sidecar native_extension_sha256 block missing")
    f64 = next((v for k, v in extensions.items() if "Delhommeau_float64" in k), None)
    if not isinstance(f64, str) or len(f64) != 64:
        raise Refused("sidecar does not bind a Delhommeau_float64 extension hash")
    return f64


def validate_admission(admission, paths, sidecar_f64_sha):
    if not isinstance(admission, dict) or admission.get("format") != ADMISSION_FORMAT:
        raise Refused("admission format mismatch")
    if admission.get("state") != "admitted":
        raise Refused(f"admission state {admission.get('state')!r} is not admitted")
    if admission.get("consumed") is True:
        raise Refused("admission is already consumed")
    if admission.get("one_execution_only") is not True:
        raise Refused("admission does not allow exactly one execution")
    bindings = (
        ("script_sha256", paths["script_sha"]),
        ("criteria_sha256", paths["criteria_sha"]),
        ("geometry_sha256", paths["geometry_sha"]),
        ("sidecar_sha256", paths["sidecar_sha"]),
    )
    for key, actual in bindings:
        if admission.get(key) != actual:
            raise Refused(f"admission {key} does not match the actual input")
    if admission.get("python_executable") != sys.executable:
        raise Refused("admission python_executable does not match sys.executable")
    if admission.get("python_executable_sha256") != sha256_file(sys.executable):
        raise Refused("admission python_executable_sha256 does not match "
                      "the actual sys.executable")
    if admission.get("f64_extension_sha256") != sidecar_f64_sha:
        raise Refused("admission f64 extension hash does not match the sidecar")
    consumed = Path(paths["admission"]).with_name(
        Path(paths["admission"]).name + ".consumed")
    if consumed.exists():
        raise Refused(f"admission already consumed: {consumed}")
    return consumed


def load_geometry(path):
    geometry_path = Path(path)
    if not geometry_path.is_file():
        raise Refused(f"geometry missing: {geometry_path}")
    try:
        with np.load(geometry_path) as data:
            geometry = {key: np.array(data[key], copy=True) for key in GEOMETRY_KEYS}
    except Exception as exc:
        raise Refused(f"geometry is not a readable npz: {geometry_path}") from exc
    for key, (shape, dtype) in GEOMETRY_KEYS.items():
        array = geometry[key]
        if array.shape != shape or (dtype is not None and array.dtype != dtype):
            raise Refused(f"geometry {key} has shape/dtype {array.shape}/{array.dtype}, "
                          f"expected {shape}/{dtype}")
    radius = float(geometry["face_radius"])
    cutoff = NEAR_PLANE_CUTOFF_FACTOR * radius
    positive = geometry["eps_factor"][geometry["eps_factor"] > 0]
    if not np.all(positive * radius > cutoff) or np.unique(positive).size != 3:
        raise Refused("frozen offsets do not stay above the near-plane cutoff")
    return geometry


def import_dependencies():
    """Import the public backend and read its identity before any constructor."""
    try:
        cpt = importlib.import_module("capytaine")
        dh_mod = importlib.import_module("capytaine.green_functions.delhommeau")
        f64 = importlib.import_module(F64_MODULE_NAME)
        threadpoolctl = importlib.import_module("threadpoolctl")
    except Exception as exc:
        raise Refused(f"dependency import failed: {exc}") from exc
    f64_path = getattr(f64, "__file__", None)
    if not f64_path or not Path(f64_path).is_file():
        raise Refused("imported f64 extension has no real module file")
    f64_sha = sha256_file(f64_path)
    constants = getattr(f64, "constants", None)
    fpp = getattr(f64, "floating_point_precision", None)
    if constants is None or fpp is None:
        raise Refused("f64 extension lacks constants/floating_point_precision")
    pre = scalar(getattr(fpp, "pre", None))
    pi = scalar(getattr(constants, "pi", None))
    gamma = scalar(getattr(constants, "euler_gamma", None))
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(float(v)) for v in (pre, pi, gamma)):
        raise Refused(f"f64 PRE/PI/Gamma missing or nonfinite: {pre!r} {pi!r} {gamma!r}")
    if int(pre) != 8:
        raise Refused(f"f64 PRE kind {pre!r} is not float64 (kind 8)")
    if float(pi) != math.pi:
        raise Refused(f"f64 PI {pi!r} is not the float64 math.pi value")
    if float(gamma) != GAMMA_REFERENCE:
        raise Refused(f"f64 EULER_GAMMA {gamma!r} is not the float64 reference value")
    pools = threadpoolctl.threadpool_info()
    if not isinstance(pools, list) or not pools:
        raise Refused("threadpoolctl reported no pools")
    bad = [p for p in pools if p.get("num_threads") != 1]
    if bad:
        raise Refused(f"thread pools not limited to one thread: {bad}")
    identity = dict(
        python_executable=sys.executable,
        python_version=sys.version,
        capytaine_file=str(getattr(cpt, "__file__", None)),
        capytaine_version=str(getattr(cpt, "__version__", None)),
        delhommeau_module_file=str(getattr(dh_mod, "__file__", None)),
        delhommeau_module_sha256=(sha256_file(dh_mod.__file__)
                                  if getattr(dh_mod, "__file__", None) else None),
        f64_module_file=str(f64_path),
        f64_extension_sha256=f64_sha,
        f64_pre=int(pre), f64_pi=float(pi), f64_euler_gamma=float(gamma),
        threadpools=pools,
    )
    return cpt, identity


def consume_admission(admission_path, consumed_path):
    """Claim the single execution exclusively; never overwrite a consumed record.

    An exclusive hard link to the admission is created at the consumed path:
    it fails when any consumed record already exists (preexisting or
    competing claim) instead of overwriting it, and it preserves the original
    admission bytes unchanged. The admission name is removed only after the
    claim exists, so any later invocation fails closed on the consumed record.
    """
    admission = Path(admission_path)
    consumed = Path(consumed_path)
    try:
        os.link(str(admission), str(consumed))
    except FileExistsError as exc:
        raise Refused(f"admission already consumed: {consumed}") from exc
    except OSError as exc:
        raise Refused(f"admission could not be consumed atomically: {exc}") from exc
    try:
        os.unlink(str(admission))
    except OSError as exc:
        raise Refused(f"admission claim could not be finalized: {exc}") from exc
    if admission.exists() or not consumed.exists():
        raise Refused("admission consumption was not atomic-effective")


class BudgetGuard:
    """sys.setprofile guard pinned to exact method code objects.

    Refusals raise at the 'call' event, before the pinned body executes.
    """

    def __init__(self, pinned, budgets):
        self.pinned = pinned      # {code object: method name}
        self.budgets = budgets    # {method name: allowed call count}
        self.counts = {name: 0 for name in budgets}
        self.trace = []

    def install(self):
        sys.setprofile(self._profile)

    def uninstall(self):
        sys.setprofile(None)

    def _profile(self, frame, event, arg):
        if event != "call":
            return
        name = self.pinned.get(frame.f_code)
        if name is None:
            return
        if name == "_create_tabulation":
            locals_ = frame.f_locals
            nr = locals_.get("tabulation_nr")
            nz = locals_.get("tabulation_nz")
            if nr is None or nz is None or nr != 0 or nz != 0:
                raise Refused(f"table construction refused before body: "
                              f"tabulation_nr={nr!r} tabulation_nz={nz!r}")
        if self.counts[name] >= self.budgets[name]:
            raise Refused(f"budget for {name} exhausted before body "
                          f"(limit {self.budgets[name]})")
        self.counts[name] += 1
        self.trace.append(name)


def _pinned_code(name, method):
    """Innermost Python code object behind supported wrappers, else refusal.

    functools.lru_cache wrappers (the public Prony entry) hide the body code
    behind __wrapped__; the guard pins the body so refusals fire before the
    body executes. Counts are honest body executions: a cached wrapper
    return that skips the body is neither counted nor refused. Any wrapper
    shape without a terminal Python code object, and any __wrapped__ cycle,
    is refused.
    """
    seen = []
    obj = method
    while True:
        code = getattr(obj, "__code__", None)
        if code is not None:
            return code, len(seen)
        wrapped = getattr(obj, "__wrapped__", None)
        if wrapped is None or any(wrapped is item for item in seen):
            raise Refused(f"Delhommeau.{name} is not a pinned Python method "
                          f"or a supported wrapper")
        seen.append(obj)
        obj = wrapped


def pin_methods(cpt, limits):
    delhommeau_cls = getattr(cpt, "Delhommeau", None)
    if delhommeau_cls is None:
        raise Refused("capytaine exposes no public Delhommeau class")
    pinned = {}
    wrappers = {}
    for name in PINNED_METHODS:
        method = getattr(delhommeau_cls, name, None)
        code, depth = _pinned_code(name, method)
        pinned[code] = name
        if depth:
            wrappers[name] = depth
    budgets = {
        "__init__": int(limits["constructor"]),
        "_create_tabulation": int(limits["empty_tabulation"]),
        "_create_or_load_tabulation": int(limits["positive_tabulation"]),
        "evaluate": int(limits["wave"]),
        "find_best_exponential_decomposition": int(limits["prony"]),
        "evaluate_rankine_only": int(limits["rankine"]),
    }
    return delhommeau_cls, pinned, budgets, wrappers


def build_mesh(cpt, geometry):
    mesh_cls = getattr(cpt, "Mesh", None)
    if mesh_cls is None:
        raise Refused("capytaine exposes no public Mesh class")
    mesh = mesh_cls(vertices=geometry["triangle"],
                   faces=np.array([[0, 1, 2, 2]])).with_quadrature(QUADRATURE_METHOD)
    qpoints, qweights = mesh.quadrature_points
    tol = GEOMETRY_TOLERANCES
    try:
        np.testing.assert_allclose(qpoints[0], geometry["gauss_points"],
                                    **tol["points"])
        np.testing.assert_allclose(qweights[0], geometry["gauss_weights"],
                                    **tol["weights"])
        np.testing.assert_allclose(mesh.faces_normals[0], geometry["normal"],
                                    **tol["normals"])
        np.testing.assert_allclose(mesh.faces_radiuses[0], geometry["face_radius"],
                                   **tol["radius"])
    except AssertionError as exc:
        raise Refused(f"constructed mesh does not match the frozen geometry: {exc}") from exc
    return mesh


def run_rankine_calls(gf, mesh, cloud, output_dir):
    """Exact four public calls; each raw pair is stored BEFORE validation and
    validated immediately, so an invalid result is preserved and no further
    native call is made."""
    base = dict(early_dot_product=False, diagonal_term_in_double_layer=False)
    calls = (
        ("adjoint_cloud_no_jump", cloud, mesh, dict(adjoint_double_layer=True)),
        ("source_cloud_no_jump", cloud, mesh, dict(adjoint_double_layer=False)),
        ("self_mesh_adjoint_no_jump", mesh, mesh, dict(adjoint_double_layer=True)),
        ("self_mesh_adjoint_with_jump", mesh, mesh,
         dict(adjoint_double_layer=True, diagonal_term_in_double_layer=True)),
    )
    raw = {}
    for index, (name, mesh1, mesh2, extra) in enumerate(calls):
        s, k = gf.evaluate_rankine_only(mesh1, mesh2, **{**base, **extra})
        raw[f"s_{name}"] = np.array(s, copy=True)
        raw[f"k_{name}"] = np.array(k, copy=True)
        np.savez(str(output_dir / "rankine-raw.npz"), **raw)
        shapes = (raw[f"s_{name}"].shape, raw[f"k_{name}"].shape)
        expected = (EXPECTED_SHAPES[2 * index], EXPECTED_SHAPES[2 * index + 1])
        if shapes != expected:
            raise Refused(f"rankine evaluation shapes {shapes} for {name} do "
                          f"not match the contract")
        for key in (f"s_{name}", f"k_{name}"):
            if not np.all(np.isfinite(raw[key])):
                raise Refused(f"nonfinite values in raw {key}")
    return raw


def evaluate_predicates(raw, geometry, absolute, relative):
    """Six unchanged predicates; scales and near-plane exclusion as historical."""
    normal = geometry["normal"]
    side = geometry["side"]
    base_index = geometry["base_index"]
    eps_factor = geometry["eps_factor"]
    omega = geometry["expected_solid_angle"]
    d_edge = geometry["edge_distance"]
    radius = float(geometry["face_radius"])
    k_a = raw["k_adjoint_cloud_no_jump"]
    k_b = raw["k_source_cloud_no_jump"]
    k_c = raw["k_self_mesh_adjoint_no_jump"]
    k_d = raw["k_self_mesh_adjoint_with_jump"]
    trace_a = np.einsum("d,di->i", normal, k_a[:, :, 0])
    off_plane = np.array([s != 0 for s in side])
    on_plane = ~off_plane
    expected_off = omega[off_plane] / (4 * math.pi)
    err1 = np.abs(trace_a[off_plane] - expected_off)
    lim1 = absolute + relative * float(np.max(np.abs(expected_off)))
    rows1 = [dict(row=int(i), error=float(err1[i]), limit=lim1,
                  passed=bool(err1[i] <= lim1)) for i in range(len(err1))]
    p1 = all(entry["passed"] for entry in rows1)
    err2 = np.abs(k_b + k_a)
    lim2 = absolute + relative * float(np.max(np.abs(k_a)))
    p2 = bool(np.all(err2 <= lim2))
    err3 = np.abs(trace_a[on_plane])
    lim3 = absolute
    rows3 = [dict(row=int(i), error=float(err3[i]), limit=lim3,
                  passed=bool(err3[i] <= lim3)) for i in range(len(err3))]
    p3 = all(entry["passed"] for entry in rows3)
    increment = k_d - k_c
    expected4 = 0.5 * normal
    err4 = np.abs(increment - expected4.reshape(3, 1, 1))
    lim4 = absolute + relative * 0.5
    p4 = bool(np.all(err4 <= lim4))
    gauss_rows = [i for i in range(len(side))
                  if base_index[i] in (1, 2, 3, 4) and side[i] == 0]
    weights_normalized = geometry["gauss_weights"] / geometry["gauss_weights"].sum()
    mean_pv = float(np.dot(weights_normalized, trace_a[gauss_rows]))
    lim5 = absolute
    p5 = bool(abs(mean_pv) <= lim5)
    tol6 = absolute + relative * 0.5
    smallest = float(np.min(eps_factor[eps_factor > 0]))
    disk = []
    for i in range(len(side)):
        if eps_factor[i] == smallest:
            eps = eps_factor[i] * radius
            bound = eps / (2 * math.sqrt(d_edge[i] ** 2 + eps ** 2)) + tol6
            target = 0.5 * side[i]
            error = abs(float(trace_a[i]) - target)
            clamped = (float(trace_a[i]) <= 0.5 + tol6 if side[i] > 0
                       else float(trace_a[i]) >= -0.5 - tol6)
            disk.append(dict(row=i, base=int(base_index[i]), side=int(side[i]),
                             edge_distance=float(d_edge[i]), bound=bound,
                             error=error, half_side_clamp=clamped,
                             passed=bool(error <= bound and clamped)))
    p6 = len(disk) == 10 and all(entry["passed"] for entry in disk)
    predicates = dict(
        off_panel_adjoint_normal_vs_solid_angle=dict(
            passed=p1, limit=lim1, max_error=float(err1.max()), rows=rows1),
        source_gradient_is_minus_receiver_gradient=dict(
            passed=p2, limit=lim2, max_error=float(err2.max())),
        on_panel_principal_value_normal_zero=dict(
            passed=p3, limit=lim3, max_error=float(err3.max()), rows=rows3),
        self_mesh_on_off_increment_half_normal=dict(
            passed=p4, limit=lim4, max_error=float(err4.max())),
        normalized_gauss_weight_mean_preserves_half=dict(
            passed=p5, limit=lim5, mean_principal_value=mean_pv,
            mean_plus_half_jump=mean_pv + 0.5),
        smallest_above_cutoff_disk_tail_bound=dict(
            passed=p6, tolerance=tol6, rows=disk),
        near_plane_limit=(
            "no numerical claim below 1e-4*face_radius; native AT suppression "
            "below that cutoff is explicitly outside off-panel qualification"),
        adjoint_diagonal_on_point_array_mesh1=(
            "deliberately not requested (dummy receiver normals are zero on "
            "that public path)"),
        limits=dict(
            absolute=absolute, relative=relative,
            normalization=("dimensionless integrated Rankine gradient; scale is "
                           "the max abs of the independently expected quantity "
                           "per comparison")),
    )
    all_passed = bool(p1 and p2 and p3 and p4 and p5 and p6)
    return predicates, all_passed


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
        elapsed_seconds=state.get("elapsed_seconds"),
        partial_results="rankine-raw.npz holds every raw S/K stored before "
                        "the failing validation, when any call completed",
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
            raise Refused(f"output directory already exists; prior raw and "
                          f"evidence must never be overwritten: {output_dir}")
        output_dir.mkdir(parents=True)
        state["output_dir_created"] = True
        state["stage"] = "input_validation"
        criteria = load_json(args.criteria, "criteria")
        sidecar = load_json(args.sidecar, "backend sidecar")
        admission = load_json(args.admission, "admission")
        absolute, relative = validate_criteria(criteria)
        sidecar_f64_sha = validate_sidecar(sidecar, criteria)
        paths = dict(
            script_sha=sha256_file(Path(__file__).resolve()),
            criteria_sha=sha256_file(args.criteria),
            geometry_sha=sha256_file(args.geometry),
            sidecar_sha=sha256_file(args.sidecar),
            admission=args.admission,
        )
        if paths["geometry_sha"] != criteria.get("geometry_sha256"):
            raise Refused("geometry hash does not match criteria geometry_sha256")
        consumed_path = validate_admission(admission, paths, sidecar_f64_sha)
        geometry = load_geometry(args.geometry)
        state["stage"] = "dependency_import"
        cpt, identity = import_dependencies()
        if identity["f64_extension_sha256"] != sidecar_f64_sha:
            raise Refused("imported f64 extension hash does not match the sidecar; "
                          "old or wrong library refused")
        if admission.get("delhommeau_module_sha256") != identity["delhommeau_module_sha256"]:
            raise Refused("admission delhommeau_module_sha256 does not match "
                          "the imported Python GF module")
        state["identity"] = identity
        state["stage"] = "admission_consumption"
        consume_admission(args.admission, consumed_path)
        state["stage"] = "guarded_construction"
        delhommeau_cls, pinned, budgets, wrappers = pin_methods(
            cpt, criteria["native_limits"])
        state["pinned_wrappers"] = wrappers
        guard = BudgetGuard(pinned, budgets)
        state["counts"] = guard.counts
        state["trace"] = guard.trace
        guard.install()
        constructor = criteria["constructor"]
        gf = delhommeau_cls(
            tabulation_nr=constructor["tabulation_nr"],
            tabulation_nz=constructor["tabulation_nz"],
            tabulation_cache_dir=constructor["tabulation_cache_dir"],
            floating_point_precision=constructor["floating_point_precision"],
        )
        mesh = build_mesh(cpt, geometry)
        state["stage"] = "rankine_calls"
        raw = run_rankine_calls(gf, mesh, geometry["cloud"], output_dir)
        state["stage"] = "predicates"
        predicates, all_passed = evaluate_predicates(raw, geometry, absolute, relative)
        dump_json(output_dir / "rankine-predicates.json", predicates)
        state["stage"] = "summary"
        state["elapsed_seconds"] = time.monotonic() - started
        summary = dict(
            format=CRITERIA_FORMAT,
            status="passed" if all_passed else "failed",
            all_passed=all_passed,
            predicates={name: predicates[name]["passed"] for name in PREDICATE_NAMES},
            guard_counts=guard.counts,
            guard_trace=guard.trace,
            pinned_wrappers=wrappers,
            runtime_identity=identity,
            admission_consumed_path=str(consumed_path),
            elapsed_seconds=state["elapsed_seconds"],
            retries=0,
            scope=("development local admission; not W5 qualification; no "
                   "force/PSD/energy/full-operator claim"),
        )
        dump_json(output_dir / "summary.json", summary)
        return 0 if all_passed else 1
    except Exception as exc:
        state["elapsed_seconds"] = time.monotonic() - started
        write_failure(output_dir, state, exc)
        return 2
    finally:
        if guard is not None:
            guard.uninstall()


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="One admitted local public-API Rankine artifact check.")
    parser.add_argument("--criteria", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--sidecar", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
