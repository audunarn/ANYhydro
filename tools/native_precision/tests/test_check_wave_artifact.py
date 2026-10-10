# SPDX-License-Identifier: MPL-2.0
"""Mock-only refusal, guard, retention and comparison tests for
check_wave_artifact.

No real Green function is imported, constructed or executed: the capytaine,
delhommeau, f64-extension and threadpoolctl modules are fakes injected into
sys.modules, and every native call returns scripted synthetic data echoing the
frozen reference (wave pair = original reference high arrays; direct Rankine
pair = -rich rankine[0]/(4*pi) and matching gradient, component-first). The
real criteria.json and the byte-frozen input/reference npz files are only read.

The fake backend mirrors the pinned public surface: the constructor accepts
every declared constructor setting and exposes exportable_settings, the base
public Prony entry is a genuine functools.lru_cache wrapper (never budgeted
here), and the fake finite-depth evaluate calls
self.find_best_exponential_decomposition(k*depth) so the checker's
fixed-coefficient override count is exercised; it then returns the scripted
reference pair. The helper's import_dependencies is wrapped by the test
harness: every helper import control (module files, f64 PRE/PI/Gamma,
thread pools) still runs against the fakes, and only the returned
f64_extension_sha256 field is aligned to the frozen Linux sidecar value that a
fake Windows extension file can never reproduce byte-wise; the caller's
sidecar/admission comparisons are still fully exercised.
"""
import functools
import hashlib
import json
import math
import shutil
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import check_rankine_artifact as cra
import check_wave_artifact as cwa

EVIDENCE = (Path(__file__).resolve().parents[3] / ".development-evidence"
            / "w5-wave-artifact-20261010")
CRITERIA_PATH = EVIDENCE / "criteria.json"
INPUTS = EVIDENCE / "inputs"
CRITERIA = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
with np.load(INPUTS / "source-geometry.npz") as _npz:
    GEOMETRY = {key: np.array(_npz[key], copy=True) for key in _npz.files}
with np.load(INPUTS / "reference.npz") as _npz:
    REFERENCE = {key: np.array(_npz[key], copy=True) for key in _npz.files}
with np.load(INPUTS / "reference-128.npz") as _npz:
    RICH_HIGH = {key: np.array(_npz[key], copy=True)
                 for key in ("rankine", "rankine_gradient")}
with np.load(INPUTS / "input.npz") as _npz:
    COEFFICIENTS = np.array(_npz["coefficients"], copy=True)
SIDECAR = json.loads((INPUTS / "backend.json").read_text(encoding="utf-8"))
REAL_F64_SHA = next(v for k, v in SIDECAR["native_extension_sha256"].items()
                    if "Delhommeau_float64" in k)
K_WAVENUMBER = float.fromhex(CRITERIA["k_hex"])
KH = float.fromhex(CRITERIA["kh_hex"])
FOUR_PI = 4.0 * math.pi

FAKE_MODULES = ("capytaine", "capytaine.green_functions",
                "capytaine.green_functions.delhommeau",
                "capytaine.green_functions.Delhommeau_float64",
                "threadpoolctl")

RECORD = {}


def reset_record(rankine=None, wave=None):
    RECORD.clear()
    RECORD.update(constructor=0, constructor_settings=[], create_tabulation=[],
                  load_tabulation=[], wave=[], prony=[], rankine=[],
                  decomposition=[], quadrature_method=None,
                  rankine_script=rankine, wave_script=wave)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@functools.lru_cache(maxsize=None)
def _prony_body(*args, **kwargs):
    RECORD["prony"].append(True)
    return (1.0,)


class FakeDelhommeau:
    """Public-surface fake; method names and signatures mirror the backend.
    The base Prony entry is a genuine lru_cache wrapper, as upstream."""

    find_best_exponential_decomposition = _prony_body

    def __init__(self, **kwargs):
        RECORD["constructor"] += 1
        RECORD["constructor_settings"].append(dict(kwargs))
        self._settings = dict(kwargs)
        if kwargs.get("tabulation_cache_dir") is None:
            self._create_tabulation(kwargs.get("tabulation_nr", 0), 50.0,
                                    kwargs.get("tabulation_nz", 0), -1.0,
                                    kwargs.get(
                                        "tabulation_nb_integration_points", 1001))
        else:
            self._create_or_load_tabulation(
                kwargs.get("tabulation_nr", 0), 50.0,
                kwargs.get("tabulation_nz", 0), -1.0,
                kwargs.get("tabulation_nb_integration_points", 1001),
                kwargs["tabulation_cache_dir"])

    @property
    def exportable_settings(self):
        return dict(self._settings)

    def _create_tabulation(self, tabulation_nr, tabulation_rmax, tabulation_nz,
                           tabulation_zmin, tabulation_nb_integration_points):
        RECORD["create_tabulation"].append((tabulation_nr, tabulation_nz))

    def _create_or_load_tabulation(self, tabulation_nr, tabulation_rmax,
                                   tabulation_nz, tabulation_zmin,
                                   tabulation_nb_integration_points, cache_dir):
        RECORD["load_tabulation"].append((tabulation_nr, tabulation_nz))

    def evaluate(self, mesh1, mesh2, *, free_surface, water_depth, wavenumber,
                 adjoint_double_layer=True, early_dot_product=True,
                 diagonal_term_in_double_layer=True):
        RECORD["wave"].append((free_surface, water_depth, wavenumber,
                               adjoint_double_layer, early_dot_product,
                               diagonal_term_in_double_layer))
        method = self._settings.get("finite_depth_prony_decomposition_method")
        coefficients = self.find_best_exponential_decomposition(
            wavenumber * water_depth, method=method)
        RECORD["decomposition"].append((wavenumber * water_depth, method,
                                        coefficients))
        s, k = RECORD["wave_script"]
        return (np.array(s, dtype=np.complex128),
                np.array(k, dtype=np.complex128))

    def evaluate_rankine_only(self, mesh1, mesh2, *, adjoint_double_layer=True,
                              early_dot_product=True,
                              diagonal_term_in_double_layer=True):
        RECORD["rankine"].append((adjoint_double_layer, early_dot_product,
                                  diagonal_term_in_double_layer))
        s, k = RECORD["rankine_script"]
        return np.array(s, dtype=np.float64), np.array(k, dtype=np.float64)


class PositiveTableDelhommeau(FakeDelhommeau):
    """Defective backend: builds a positive table inside the constructor."""

    def __init__(self, **kwargs):
        RECORD["constructor"] += 1
        RECORD["constructor_settings"].append(dict(kwargs))
        self._settings = dict(kwargs)
        self._create_tabulation(4, 50.0, 4, -1.0, 1001)


class CacheDirDelhommeau(FakeDelhommeau):
    """Defective backend: takes the forbidden cache route inside the
    constructor."""

    def __init__(self, **kwargs):
        RECORD["constructor"] += 1
        RECORD["constructor_settings"].append(dict(kwargs))
        self._settings = dict(kwargs)
        self._create_or_load_tabulation(0, 50.0, 0, -1.0, 1001, "/tmp/cache")


class DefaultFitDelhommeau(FakeDelhommeau):
    """Defective backend: the finite-depth evaluate calls the BASE public
    Prony entry (a default fit) instead of the fixed-coefficient override."""

    def evaluate(self, mesh1, mesh2, *, free_surface, water_depth, wavenumber,
                 adjoint_double_layer=True, early_dot_product=True,
                 diagonal_term_in_double_layer=True):
        RECORD["wave"].append((free_surface, water_depth, wavenumber,
                               adjoint_double_layer, early_dot_product,
                               diagonal_term_in_double_layer))
        FakeDelhommeau.find_best_exponential_decomposition(
            self, wavenumber * water_depth)


class ChangedKhDelhommeau(FakeDelhommeau):
    """Defective backend: evaluate asks the decomposition for a changed kh."""

    def evaluate(self, mesh1, mesh2, *, free_surface, water_depth, wavenumber,
                 adjoint_double_layer=True, early_dot_product=True,
                 diagonal_term_in_double_layer=True):
        RECORD["wave"].append((free_surface, water_depth, wavenumber,
                               adjoint_double_layer, early_dot_product,
                               diagonal_term_in_double_layer))
        self.find_best_exponential_decomposition(wavenumber * water_depth * 1.01,
                                                 method="python")


class FortranMethodDelhommeau(FakeDelhommeau):
    """Defective backend: evaluate asks for a non-declared decomposition
    method."""

    def evaluate(self, mesh1, mesh2, *, free_surface, water_depth, wavenumber,
                 adjoint_double_layer=True, early_dot_product=True,
                 diagonal_term_in_double_layer=True):
        RECORD["wave"].append((free_surface, water_depth, wavenumber,
                               adjoint_double_layer, early_dot_product,
                               diagonal_term_in_double_layer))
        self.find_best_exponential_decomposition(wavenumber * water_depth,
                                                 method="fortran")


class FakeMesh:
    def __init__(self, vertices=None, faces=None, bad_normals=False):
        self._bad_normals = bad_normals

    def with_quadrature(self, method):
        RECORD["quadrature_method"] = method
        return self

    @property
    def quadrature_points(self):
        return (GEOMETRY["gauss_points"].reshape(1, 4, 3),
                GEOMETRY["gauss_weights"].reshape(1, 4))

    @property
    def faces_normals(self):
        normal = GEOMETRY["normal"].copy()
        if self._bad_normals:
            normal = normal + 0.1
        return normal.reshape(1, 3)

    @property
    def faces_radiuses(self):
        return np.array([float(GEOMETRY["face_radius"])])


class BadNormalsMesh(FakeMesh):
    def __init__(self, vertices=None, faces=None):
        super().__init__(vertices, faces, bad_normals=True)


def passing_script():
    """Scripted raw pairs echoing the frozen reference: the wave pair is the
    original reference high array pair; the direct Rankine pair is
    -rich rankine[0]/(4*pi) and the matching gradient, component-first."""
    s_wave = np.array(REFERENCE["potential_high"],
                      dtype=np.complex128).reshape(43, 1)
    k_wave = np.ascontiguousarray(REFERENCE["gradient_high"].T).reshape(3, 43, 1)
    s_rankine = (-RICH_HIGH["rankine"][0] / FOUR_PI).reshape(43, 1)
    k_rankine = np.ascontiguousarray(
        (-RICH_HIGH["rankine_gradient"][0] / FOUR_PI).T).reshape(3, 43, 1)
    return dict(rankine=(s_rankine, k_rankine), wave=(s_wave, k_wave))


def self_fd_failure_script():
    """Same as passing_script, but the scripted potential carries a small
    offset on the sign=+1 axis-0 receiver of every step: only the native
    self-FD diagnostics fail; the real-gradient/reference gate is unchanged
    because it never reads the potential."""
    script = passing_script()
    s_wave = np.array(script["wave"][0], copy=True)
    for i in range(len(cwa.EXPECTED_STEPS)):
        s_wave[1 + 6 * i, 0] += 1e-4
    script["wave"] = (s_wave, script["wave"][1])
    return script


class FakeBackendTestCase(unittest.TestCase):
    """Base: temp dirs, fake module installation, admission writer, runner."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.criteria_path = CRITERIA_PATH
        self.f64_file = self.tmp / "Delhommeau_float64.fake.so"
        self.f64_file.write_bytes(b"fake f64 extension bytes")
        self.f64_sha = sha(self.f64_file)
        reset_record()
        self._saved = {name: sys.modules.get(name) for name in FAKE_MODULES}
        for name in FAKE_MODULES:
            sys.modules.pop(name, None)
        self.addCleanup(self._restore_modules)

    def _restore_modules(self):
        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def install_backend(self, delhommeau_cls=FakeDelhommeau, mesh_cls=FakeMesh,
                        pi=None, gamma=None, pre=8, pools=None, f64_sha=None):
        pi = math.pi if pi is None else pi
        gamma = cra.GAMMA_REFERENCE if gamma is None else gamma
        pools = [{"user_api": "openmp", "num_threads": 1}] if pools is None else pools
        cpt = types.ModuleType("capytaine")
        cpt.__file__ = str(self.tmp / "capytaine__init__.py")
        (self.tmp / "capytaine__init__.py").write_text("# fake capytaine\n")
        cpt.__version__ = "3.0.0-fake"
        cpt.Delhommeau = delhommeau_cls
        cpt.Mesh = mesh_cls
        gf = types.ModuleType("capytaine.green_functions")
        dh = types.ModuleType("capytaine.green_functions.delhommeau")
        dh.__file__ = str(self.tmp / "delhommeau.py")
        (self.tmp / "delhommeau.py").write_text("# fake delhommeau\n")
        f64 = types.ModuleType("capytaine.green_functions.Delhommeau_float64")
        f64.__file__ = str(self.f64_file)
        f64.constants = SimpleNamespace(
            pi=pi, euler_gamma=gamma,
            minus_one_over_fourpi=-1.0 / FOUR_PI, log_2=math.log(2.0))
        f64.floating_point_precision = SimpleNamespace(pre=pre)
        tp = types.ModuleType("threadpoolctl")
        tp.threadpool_info = lambda: pools
        cpt.green_functions = gf
        gf.delhommeau = dh
        gf.Delhommeau_float64 = f64
        sys.modules["capytaine"] = cpt
        sys.modules["capytaine.green_functions"] = gf
        sys.modules["capytaine.green_functions.delhommeau"] = dh
        sys.modules["capytaine.green_functions.Delhommeau_float64"] = f64
        sys.modules["threadpoolctl"] = tp
        real_import = cra.import_dependencies

        def _import():
            cpt_module, identity = real_import()
            identity["f64_extension_sha256"] = f64_sha or REAL_F64_SHA
            return cpt_module, identity

        cra.import_dependencies = _import
        self.addCleanup(setattr, cra, "import_dependencies", real_import)

    def modified_criteria(self, mutate):
        criteria = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
        mutate(criteria)
        path = self.tmp / "criteria.json"
        path.write_text(json.dumps(criteria), encoding="utf-8")
        self.criteria_path = path
        return path

    def write_admission(self, **overrides):
        admission = dict(
            format=cwa.ADMISSION_FORMAT, state="admitted",
            one_execution_only=True,
            script_sha256=sha(cwa.__file__),
            helper_sha256=sha(cra.__file__),
            criteria_sha256=sha(self.criteria_path),
            inputs=dict(CRITERIA["inputs"]),
            python_executable=sys.executable,
            python_executable_sha256=sha(sys.executable),
            delhommeau_module_sha256=sha(self.tmp / "delhommeau.py"),
            f64_extension_sha256=REAL_F64_SHA)
        admission.update(overrides)
        path = self.tmp / "admission.json"
        path.write_text(json.dumps(admission), encoding="utf-8")
        return path

    def run_check(self, admission_path, inputs_path=None, output=None):
        output = Path(output) if output else self.tmp / "out"
        argv = ["--criteria", str(self.criteria_path),
                "--inputs", str(inputs_path or INPUTS),
                "--admission", str(admission_path),
                "--output", str(output)]
        return cwa.main(argv), output

    def failure(self, output):
        return json.loads((Path(output) / "failure.json")
                          .read_text(encoding="utf-8"))

    def assert_refused_before_consumption(self, admission, fragment,
                                          inputs_path=None):
        rc, out = self.run_check(admission, inputs_path=inputs_path)
        self.assertEqual(rc, 2)
        self.assertIn(fragment, self.failure(out)["error"])
        self.assertTrue(admission.exists(),
                        "refusal must not consume the admission")


class CriteriaValidationTests(FakeBackendTestCase):
    """Declared criteria identity refusals, all before the constructor."""

    def run_modified(self, mutate, fragment):
        self.install_backend()
        reset_record(**passing_script())
        self.modified_criteria(mutate)
        admission = self.write_admission()
        self.assert_refused_before_consumption(admission, fragment)
        self.assertEqual(RECORD["constructor"], 0,
                         "no constructor may run on a refused criteria")

    def test_criteria_format_mismatch(self):
        self.run_modified(lambda c: c.update(format="other-v1"),
                          "criteria format mismatch")

    def test_criteria_kh_inconsistent_with_k(self):
        self.run_modified(lambda c: c.update(kh_hex="0x1.0p+5"),
                          "k/kh consistency")

    def test_criteria_coefficient_identity_mismatch(self):
        def mutate(criteria):
            criteria["fixed_coefficients"]["sha256"] = "0" * 64
        self.run_modified(mutate, "coefficient identity")

    def test_criteria_constructor_route_changed(self):
        def mutate(criteria):
            criteria["constructor"]["tabulation_nb_integration_points"] = 501
        self.run_modified(mutate, "tabulation_nb_integration_points")

    def test_criteria_native_limits_changed(self):
        def mutate(criteria):
            criteria["native_limits"]["wave"] = 2
        self.run_modified(mutate, "native_limits")

    def test_criteria_helper_sha_mismatch(self):
        def mutate(criteria):
            criteria["helper_sha256"] = "0" * 64
        self.run_modified(mutate, "helper sha256 does not match")


class AdmissionRefusalTests(FakeBackendTestCase):
    def test_missing_admission(self):
        self.install_backend()
        reset_record(**passing_script())
        rc, out = self.run_check(self.tmp / "absent.json")
        self.assertEqual(rc, 2)
        self.assertIn("admission missing", self.failure(out)["error"])

    def test_state_not_admitted(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission(state="revoked")
        rc, out = self.run_check(admission)
        self.assertEqual(rc, 2)
        self.assertIn("not admitted", self.failure(out)["error"])

    def test_consumed_flag(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission(consumed=True)
        rc, out = self.run_check(admission)
        self.assertEqual(rc, 2)
        self.assertIn("already consumed", self.failure(out)["error"])

    def test_consumed_sibling(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission()
        admission.with_name(admission.name + ".consumed").write_text("spent")
        self.assert_refused_before_consumption(admission, "already consumed")

    def test_one_execution_only_false(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission(one_execution_only=False)
        self.assert_refused_before_consumption(admission, "one execution")

    def test_wrong_script_sha(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission(script_sha256="0" * 64)
        self.assert_refused_before_consumption(admission, "script_sha256")

    def test_wrong_helper_sha(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission(helper_sha256="0" * 64)
        self.assert_refused_before_consumption(admission, "helper_sha256")

    def test_wrong_criteria_sha(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission(criteria_sha256="0" * 64)
        self.assert_refused_before_consumption(admission, "criteria_sha256")

    def test_admission_inputs_map_mismatch(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission(inputs={})
        self.assert_refused_before_consumption(admission, "input SHA map")

    def test_wrong_python_executable(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission(python_executable="/other/python")
        self.assert_refused_before_consumption(admission, "python_executable")

    def test_wrong_python_executable_sha(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission(python_executable_sha256="0" * 64)
        self.assert_refused_before_consumption(admission,
                                               "python_executable_sha256")

    def test_admission_f64_sha_mismatches_sidecar(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission(f64_extension_sha256="b" * 64)
        self.assert_refused_before_consumption(admission, "f64 extension hash")

    def test_wrong_delhommeau_module_sha(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission(delhommeau_module_sha256="0" * 64)
        self.assert_refused_before_consumption(admission,
                                               "delhommeau_module_sha256")

    def test_imported_f64_mismatch_refused(self):
        self.install_backend(f64_sha="c" * 64)
        reset_record(**passing_script())
        admission = self.write_admission()
        self.assert_refused_before_consumption(admission,
                                               "old or wrong library")

    def test_input_file_hash_mismatch(self):
        self.install_backend()
        reset_record(**passing_script())
        inputs_copy = self.tmp / "inputs"
        shutil.copytree(INPUTS, inputs_copy)
        with open(inputs_copy / "reference.npz", "ab") as handle:
            handle.write(b"tampered")
        admission = self.write_admission()
        self.assert_refused_before_consumption(admission,
                                               "input file hash mismatch",
                                               inputs_path=inputs_copy)

    def test_missing_input_file(self):
        self.install_backend()
        reset_record(**passing_script())
        inputs_copy = self.tmp / "inputs"
        shutil.copytree(INPUTS, inputs_copy)
        (inputs_copy / "reference-64.npz").unlink()
        admission = self.write_admission()
        self.assert_refused_before_consumption(admission, "input file missing",
                                               inputs_path=inputs_copy)

    def test_reissued_admission_rejected_after_consumption(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission()
        rc, out = self.run_check(admission)
        self.assertEqual(rc, 0)
        admission = self.write_admission()
        rc2, out2 = self.run_check(admission, output=self.tmp / "out2")
        self.assertEqual(rc2, 2)
        self.assertIn("already consumed", self.failure(out2)["error"])

    def test_happy_path_consumes_once_and_preserves_bytes(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission()
        original = admission.read_bytes()
        rc, out = self.run_check(admission)
        self.assertEqual(rc, 0)
        self.assertFalse(admission.exists())
        consumed = admission.with_name(admission.name + ".consumed")
        self.assertEqual(consumed.read_bytes(), original,
                         "consumption must preserve the original bytes")
        rc2, _ = self.run_check(admission, output=self.tmp / "out2")
        self.assertEqual(rc2, 2)


class OutputProtectionTests(FakeBackendTestCase):
    def test_preexisting_output_refused_and_untouched(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission()
        out = self.tmp / "out"
        out.mkdir()
        sentinel = out / "prior-evidence.json"
        sentinel.write_text("prior evidence", encoding="utf-8")
        rc, _ = self.run_check(admission, output=out)
        self.assertEqual(rc, 2)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "prior evidence")
        self.assertEqual([p.name for p in out.iterdir()], ["prior-evidence.json"],
                         "no failure record may modify a preexisting output")
        self.assertTrue(admission.exists(),
                        "output refusal must not consume the admission")
        self.assertEqual(RECORD["rankine"], [], "no native call may run")

    def test_output_path_existing_file_refused(self):
        self.install_backend()
        reset_record(**passing_script())
        admission = self.write_admission()
        out = self.tmp / "out"
        out.write_text("prior bytes", encoding="utf-8")
        rc, _ = self.run_check(admission, output=out)
        self.assertEqual(rc, 2)
        self.assertEqual(out.read_text(encoding="utf-8"), "prior bytes")
        self.assertTrue(admission.exists())


class EvaluatorOverrideTests(FakeBackendTestCase):
    def make_evaluator(self):
        cpt = sys.modules["capytaine"]
        coefficients = np.array(COEFFICIENTS, copy=True)
        coefficients.setflags(write=False)
        return cwa.build_evaluator_class(cpt, coefficients, CRITERIA["kh_hex"])

    def test_override_returns_declared_readonly_coefficients(self):
        self.install_backend()
        cls = self.make_evaluator()
        for method in (None, "python"):
            returned = cls.find_best_exponential_decomposition(None, KH,
                                                               method=method)
            self.assertTrue(np.array_equal(returned, COEFFICIENTS))
            self.assertFalse(returned.flags.writeable)

    def test_override_refuses_changed_kh(self):
        self.install_backend()
        cls = self.make_evaluator()
        with self.assertRaises(cwa.REFUSED) as ctx:
            cls.find_best_exponential_decomposition(None, KH * 1.01,
                                                    method="python")
        self.assertIn("refuses changed kh", str(ctx.exception))

    def test_override_refuses_non_declared_method(self):
        self.install_backend()
        cls = self.make_evaluator()
        with self.assertRaises(cwa.REFUSED) as ctx:
            cls.find_best_exponential_decomposition(None, KH, method="fortran")
        self.assertIn("refuses decomposition method", str(ctx.exception))

    def test_override_budget_is_one(self):
        self.install_backend()
        cls = self.make_evaluator()
        cpt = sys.modules["capytaine"]
        pinned, budgets, wrappers = cwa.pin_evaluator(
            cpt, cls, CRITERIA["native_limits"])
        self.assertEqual(budgets[cwa.OVERRIDE_METHOD_NAME], 1)
        guard = cra.BudgetGuard(pinned, budgets)
        guard.install()
        self.addCleanup(guard.uninstall)
        cls.find_best_exponential_decomposition(None, KH, method="python")
        with self.assertRaises(cra.Refused):
            cls.find_best_exponential_decomposition(None, KH, method="python")
        self.assertEqual(guard.counts[cwa.OVERRIDE_METHOD_NAME], 1)

    def test_base_prony_body_refused_before_body(self):
        self.install_backend()
        cls = self.make_evaluator()
        cpt = sys.modules["capytaine"]
        pinned, budgets, wrappers = cwa.pin_evaluator(
            cpt, cls, CRITERIA["native_limits"])
        self.assertEqual(budgets["find_best_exponential_decomposition"], 0)
        guard = cra.BudgetGuard(pinned, budgets)
        guard.install()
        self.addCleanup(guard.uninstall)
        gf = cls(**CRITERIA["constructor"])
        with self.assertRaises(cra.Refused):
            FakeDelhommeau.find_best_exponential_decomposition(gf, 1.0)
        self.assertEqual(RECORD["prony"], [], "Prony body must not execute")


class RunIntegrationTests(FakeBackendTestCase):
    def admitted_run(self, **kwargs):
        reset_record(**kwargs)
        admission = self.write_admission()
        return self.run_check(admission)

    def test_pass_with_exact_counters(self):
        self.install_backend()
        rc, out = self.admitted_run(**passing_script())
        self.assertEqual(rc, 0)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["all_passed"])
        self.assertTrue(summary["reference_gradient_gate_passed"])
        self.assertTrue(summary["reference_original_steps_passed"])
        self.assertTrue(summary["native_self_fd"]["all_passed"])
        self.assertFalse(summary["qualified"])
        self.assertEqual(summary["guard_counts"],
                         {"__init__": 1, "_create_tabulation": 1,
                          "_create_or_load_tabulation": 0, "evaluate": 1,
                          "find_best_exponential_decomposition": 0,
                          "evaluate_rankine_only": 1,
                          "fixed_coefficient_override": 1})
        self.assertEqual(summary["guard_trace"],
                         ["__init__", "_create_tabulation",
                          "evaluate_rankine_only", "evaluate",
                          "fixed_coefficient_override"])
        self.assertEqual(summary["pinned_wrappers"],
                         {"find_best_exponential_decomposition": 1})
        identity = summary["evaluator_identity"]
        self.assertEqual(identity["constructor_settings"],
                         CRITERIA["constructor"])
        self.assertEqual(identity["fixed_coefficients_sha256"],
                         CRITERIA["fixed_coefficients"]["sha256"])
        self.assertEqual(RECORD["constructor"], 1)
        self.assertEqual(RECORD["constructor_settings"],
                         [CRITERIA["constructor"]])
        self.assertEqual(RECORD["create_tabulation"], [(0, 0)])
        self.assertEqual(RECORD["load_tabulation"], [])
        self.assertEqual(RECORD["rankine"], [(True, False, False)])
        self.assertEqual(RECORD["wave"],
                         [(0.0, 150.0, K_WAVENUMBER, True, False, False)])
        self.assertEqual(RECORD["prony"], [])
        self.assertEqual(RECORD["quadrature_method"], cra.QUADRATURE_METHOD)
        self.assertEqual(len(RECORD["decomposition"]), 1)
        dimensionless, method, returned = RECORD["decomposition"][0]
        self.assertEqual(dimensionless, KH)
        self.assertEqual(method, "python")
        self.assertTrue(np.array_equal(returned, COEFFICIENTS))
        self.assertFalse(returned.flags.writeable)
        with np.load(out / "wave-raw.npz") as raw:
            self.assertEqual(sorted(raw.files),
                             ["k_rankine", "k_wave", "s_rankine", "s_wave"])
            self.assertEqual(raw["s_rankine"].shape, (43, 1))
            self.assertEqual(raw["k_rankine"].shape, (3, 43, 1))
            self.assertEqual(raw["s_wave"].shape, (43, 1))
            self.assertEqual(raw["k_wave"].shape, (3, 43, 1))
            self.assertTrue(np.array_equal(
                raw["s_wave"], REFERENCE["potential_high"].reshape(43, 1)))
            self.assertTrue(np.array_equal(
                raw["k_wave"],
                REFERENCE["gradient_high"].T.reshape(3, 43, 1)))
            self.assertTrue(np.array_equal(
                raw["s_rankine"],
                (-RICH_HIGH["rankine"][0] / FOUR_PI).reshape(43, 1)))
            self.assertTrue(np.array_equal(
                raw["k_rankine"],
                (-RICH_HIGH["rankine_gradient"][0] / FOUR_PI)
                .T.reshape(3, 43, 1)))
        audit = json.loads((out / "wave-comparison.json")
                           .read_text(encoding="utf-8"))
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["potential_error"], 0.0)
        self.assertEqual(audit["imaginary_gradient_error"], 0.0)
        self.assertEqual(audit["free_surface_finite_depth_remainder"]
                         ["potential_error"], 0.0)
        self.assertEqual(audit["free_surface_finite_depth_remainder"]
                         ["gradient_error"], 0.0)
        self.assertTrue(all(audit["native_self_fd_passed"]))

    def test_reference_gradient_failure(self):
        self.install_backend()
        script = passing_script()
        k_wave = np.array(script["wave"][1], copy=True)
        k_wave[0, 5, 0] += 1e-3
        script["wave"] = (script["wave"][0], k_wave)
        rc, out = self.admitted_run(**script)
        self.assertEqual(rc, 1)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["all_passed"])
        self.assertFalse(summary["reference_gradient_gate_passed"])
        self.assertTrue(summary["reference_original_steps_passed"])

    def test_reference_small_step_failure(self):
        self.install_backend()
        script = passing_script()
        k_wave = np.array(script["wave"][1], copy=True)
        k_wave[0, 0, 0] += 1e-3
        script["wave"] = (script["wave"][0], k_wave)
        rc, out = self.admitted_run(**script)
        self.assertEqual(rc, 1)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["all_passed"])
        self.assertFalse(summary["reference_original_steps_passed"])

    def test_self_fd_only_failure_is_explicit_diagnostic(self):
        self.install_backend()
        rc, out = self.admitted_run(**self_fd_failure_script())
        self.assertEqual(rc, 0)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["all_passed"])
        self.assertTrue(summary["reference_gradient_gate_passed"])
        self.assertTrue(summary["reference_original_steps_passed"])
        self.assertFalse(summary["native_self_fd"]["all_passed"])
        self.assertIn("unresolved numerical inconsistency",
                      summary["native_self_fd"]["note"])
        audit = json.loads((out / "wave-comparison.json")
                           .read_text(encoding="utf-8"))
        self.assertTrue(audit["passed"])
        self.assertFalse(all(audit["native_self_fd_passed"]))
        self.assertGreater(audit["potential_error"], 0.0)

    def test_changed_kh_refused_in_run(self):
        self.install_backend(delhommeau_cls=ChangedKhDelhommeau)
        rc, out = self.admitted_run(**passing_script())
        self.assertEqual(rc, 2)
        self.assertIn("refuses changed kh", self.failure(out)["error"])
        self.assertEqual(len(RECORD["wave"]), 1)
        with np.load(out / "wave-raw.npz") as raw:
            self.assertEqual(sorted(raw.files), ["k_rankine", "s_rankine"])

    def test_fortran_method_refused_in_run(self):
        self.install_backend(delhommeau_cls=FortranMethodDelhommeau)
        rc, out = self.admitted_run(**passing_script())
        self.assertEqual(rc, 2)
        self.assertIn("refuses decomposition method",
                      self.failure(out)["error"])

    def test_default_prony_fit_refused_in_run(self):
        self.install_backend(delhommeau_cls=DefaultFitDelhommeau)
        rc, out = self.admitted_run(**passing_script())
        self.assertEqual(rc, 2)
        failure = self.failure(out)
        self.assertIn("find_best_exponential_decomposition", failure["error"])
        self.assertIn("exhausted before body", failure["error"])
        self.assertEqual(RECORD["prony"], [], "default fit body must not run")
        self.assertEqual(failure["guard_counts"]
                         ["find_best_exponential_decomposition"], 0)

    def test_positive_table_refused_in_run(self):
        self.install_backend(delhommeau_cls=PositiveTableDelhommeau)
        rc, out = self.admitted_run(**passing_script())
        self.assertEqual(rc, 2)
        failure = self.failure(out)
        self.assertIn("refused before body", failure["error"])
        self.assertEqual(failure["guard_counts"]["_create_tabulation"], 0)
        self.assertEqual(RECORD["create_tabulation"], [])
        self.assertFalse(self.tmp.joinpath("admission.json").exists(),
                         "admission is consumed before the constructor and "
                         "stays spent after refusal")

    def test_cache_route_refused_in_run(self):
        self.install_backend(delhommeau_cls=CacheDirDelhommeau)
        rc, out = self.admitted_run(**passing_script())
        self.assertEqual(rc, 2)
        failure = self.failure(out)
        self.assertIn("_create_or_load_tabulation", failure["error"])
        self.assertIn("exhausted before body", failure["error"])
        self.assertEqual(RECORD["load_tabulation"], [])

    def test_raw_retained_on_rankine_shape_no_later_call(self):
        self.install_backend()
        script = passing_script()
        script["rankine"] = (script["rankine"][0], np.ones((3, 43, 2)))
        rc, out = self.admitted_run(**script)
        self.assertEqual(rc, 2)
        failure = self.failure(out)
        self.assertIn("shapes", failure["error"])
        self.assertEqual(RECORD["wave"], [],
                         "an invalid rankine result must stop before the "
                         "wave call")
        self.assertEqual(failure["guard_counts"]["evaluate"], 0)
        with np.load(out / "wave-raw.npz") as raw:
            self.assertEqual(sorted(raw.files), ["k_rankine", "s_rankine"])
            self.assertEqual(raw["k_rankine"].shape, (3, 43, 2),
                             "invalid raw output must be retained")

    def test_raw_retained_on_rankine_nonfinite_no_later_call(self):
        self.install_backend()
        script = passing_script()
        s_rankine = np.array(script["rankine"][0], copy=True)
        s_rankine[3, 0] = np.nan
        script["rankine"] = (s_rankine, script["rankine"][1])
        rc, out = self.admitted_run(**script)
        self.assertEqual(rc, 2)
        failure = self.failure(out)
        self.assertIn("nonfinite", failure["error"])
        self.assertEqual(RECORD["wave"], [])
        self.assertEqual(failure["guard_counts"]["evaluate"], 0)
        with np.load(out / "wave-raw.npz") as raw:
            self.assertTrue(np.isnan(raw["s_rankine"][3, 0]),
                            "nonfinite raw output must be retained")

    def test_raw_retained_on_wave_shape(self):
        self.install_backend()
        script = passing_script()
        script["wave"] = (np.ones((43, 2)), script["wave"][1])
        rc, out = self.admitted_run(**script)
        self.assertEqual(rc, 2)
        self.assertIn("shapes", self.failure(out)["error"])
        with np.load(out / "wave-raw.npz") as raw:
            self.assertEqual(sorted(raw.files),
                             ["k_rankine", "k_wave", "s_rankine", "s_wave"])
            self.assertEqual(raw["s_wave"].shape, (43, 2),
                             "invalid raw wave output must be retained")

    def test_raw_retained_on_wave_nonfinite(self):
        self.install_backend()
        script = passing_script()
        k_wave = np.array(script["wave"][1], copy=True)
        k_wave[1, 2, 0] = np.nan + 0j
        script["wave"] = (script["wave"][0], k_wave)
        rc, out = self.admitted_run(**script)
        self.assertEqual(rc, 2)
        self.assertIn("nonfinite", self.failure(out)["error"])
        with np.load(out / "wave-raw.npz") as raw:
            self.assertTrue(np.isnan(raw["k_wave"][1, 2, 0]),
                            "nonfinite raw wave output must be retained")

    def test_mesh_geometry_mismatch_refused(self):
        self.install_backend(mesh_cls=BadNormalsMesh)
        rc, out = self.admitted_run(**passing_script())
        self.assertEqual(rc, 2)
        self.assertIn("frozen geometry", self.failure(out)["error"])


if __name__ == "__main__":
    unittest.main()
