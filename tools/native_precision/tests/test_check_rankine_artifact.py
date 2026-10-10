# SPDX-License-Identifier: MPL-2.0
"""Mock-only refusal, guard, retention and predicate tests for check_rankine_artifact.

No real Green function is imported, constructed or executed: the capytaine,
delhommeau, f64-extension and threadpoolctl modules are fakes injected into
sys.modules, and every native call returns scripted synthetic data. The real
criteria.json and the byte-frozen rankine-geometry.npz are only read.
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

EVIDENCE = (Path(__file__).resolve().parents[3] / ".development-evidence"
            / "w5-rankine-artifact-20261010")
CRITERIA_PATH = EVIDENCE / "criteria.json"
GEOMETRY_PATH = EVIDENCE / "inputs" / "rankine-geometry.npz"
CRITERIA = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
with np.load(GEOMETRY_PATH) as _npz:
    GEOMETRY = {key: np.array(_npz[key], copy=True) for key in _npz.files}

FAKE_MODULES = ("capytaine", "capytaine.green_functions",
                "capytaine.green_functions.delhommeau",
                "capytaine.green_functions.Delhommeau_float64",
                "threadpoolctl")

RECORD = {}


def reset_record(scripted=None):
    RECORD.clear()
    RECORD.update(constructor=0, create_tabulation=[], load_tabulation=[],
                  wave=[], prony=[], rankine=[], quadrature_method=None,
                  scripted=list(scripted or []))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeDelhommeau:
    """Public-surface fake; method names and signatures mirror the backend."""

    def __init__(self, **kwargs):
        RECORD["constructor"] += 1
        if kwargs.get("tabulation_cache_dir") is None:
            self._create_tabulation(kwargs.get("tabulation_nr", 0), 50.0,
                                    kwargs.get("tabulation_nz", 0), -1.0, 1)
        else:
            self._create_or_load_tabulation(kwargs.get("tabulation_nr", 0), 50.0,
                                            kwargs.get("tabulation_nz", 0), -1.0, 1,
                                            kwargs["tabulation_cache_dir"])

    def _create_tabulation(self, tabulation_nr, tabulation_rmax, tabulation_nz,
                           tabulation_zmin, tabulation_nb_integration_points):
        RECORD["create_tabulation"].append((tabulation_nr, tabulation_nz))

    def _create_or_load_tabulation(self, tabulation_nr, tabulation_rmax,
                                   tabulation_nz, tabulation_zmin,
                                   tabulation_nb_integration_points, cache_dir):
        RECORD["load_tabulation"].append((tabulation_nr, tabulation_nz))

    def evaluate(self, *args, **kwargs):
        RECORD["wave"].append(True)

    def find_best_exponential_decomposition(self, *args, **kwargs):
        RECORD["prony"].append(True)

    def evaluate_rankine_only(self, mesh1, mesh2, *, adjoint_double_layer=True,
                              early_dot_product=True,
                              diagonal_term_in_double_layer=True):
        RECORD["rankine"].append((adjoint_double_layer, early_dot_product,
                                  diagonal_term_in_double_layer))
        s, k = RECORD["scripted"].pop(0)
        return np.array(s, dtype=np.float64), np.array(k, dtype=np.float64)


class PositiveTableDelhommeau(FakeDelhommeau):
    """Defective backend: builds a positive table inside the constructor."""

    def __init__(self, **kwargs):
        RECORD["constructor"] += 1
        self._create_tabulation(4, 50.0, 4, -1.0, 1)


@functools.lru_cache(maxsize=None)
def _prony_body(*args, **kwargs):
    RECORD["prony"].append(True)
    return (1.0,)


class LruCacheDelhommeau(FakeDelhommeau):
    """Backend shape whose public Prony entry is a genuine functools.lru_cache
    wrapper (module-level function exposed as a class attribute, as upstream)."""

    find_best_exponential_decomposition = _prony_body


class OpaqueWrapper:
    """Callable wrapper shape without __code__ or __wrapped__."""

    def __call__(self, *args, **kwargs):
        RECORD["prony"].append(True)


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
    """Synthetic raw S/K satisfying all six predicates on the frozen geometry."""
    normal = GEOMETRY["normal"]
    omega = GEOMETRY["expected_solid_angle"]
    side = GEOMETRY["side"]
    nn = float(np.dot(normal, normal))
    trace = np.where(side != 0, omega / (4 * math.pi), 0.0)
    k_a = np.ascontiguousarray(np.outer(normal, trace / nn)).reshape(3, 35, 1)
    k_b = -k_a
    k_c = np.zeros((3, 1, 1))
    k_d = k_c + 0.5 * normal.reshape(3, 1, 1)
    s_cloud = np.ones((35, 1))
    s_mesh = np.ones((1, 1))
    return [(s_cloud, k_a), (s_cloud, k_b), (s_mesh, k_c), (s_mesh, k_d)]


def failing_script():
    """Same as passing_script but off-panel traces are wrong by 1e-6."""
    normal = GEOMETRY["normal"]
    side = GEOMETRY["side"]
    omega = GEOMETRY["expected_solid_angle"]
    nn = float(np.dot(normal, normal))
    trace = np.where(side != 0, omega / (4 * math.pi), 0.0) \
        + np.where(side != 0, 1e-6, 0.0)
    k_a = np.ascontiguousarray(np.outer(normal, trace / nn)).reshape(3, 35, 1)
    script = passing_script()
    script[0] = (script[0][0], k_a)
    script[1] = (script[1][0], -k_a)
    return script


class FakeBackendTestCase(unittest.TestCase):
    """Base: temp dirs, fake module installation, sidecar/admission writers."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
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
                        pi=None, gamma=None, pre=8, pools=None):
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
            minus_one_over_fourpi=-1.0 / (4.0 * math.pi), log_2=math.log(2.0))
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

    def write_sidecar(self, f64_sha=None, **overrides):
        sidecar = dict(
            format=cra.SIDECAR_FORMAT,
            external_identity=CRITERIA["backend_identity"],
            wheel_sha256=CRITERIA["wheel_sha256"],
            native_extension_sha256={
                "capytaine/green_functions/Delhommeau_float32.cpython-312-x86_64-linux-gnu.so":
                    "0" * 64,
                "capytaine/green_functions/Delhommeau_float64.cpython-312-x86_64-linux-gnu.so":
                    f64_sha or self.f64_sha,
            },
            admitted_for_numerical_execution=False)
        sidecar.update(overrides)
        path = self.tmp / "backend.json"
        path.write_text(json.dumps(sidecar), encoding="utf-8")
        return path

    def write_admission(self, sidecar_path, **overrides):
        admission = dict(
            format=cra.ADMISSION_FORMAT, state="admitted",
            one_execution_only=True,
            script_sha256=sha(cra.__file__),
            criteria_sha256=sha(CRITERIA_PATH),
            geometry_sha256=sha(GEOMETRY_PATH),
            sidecar_sha256=sha(sidecar_path),
            python_executable=sys.executable,
            python_executable_sha256=sha(sys.executable),
            delhommeau_module_sha256=sha(self.tmp / "delhommeau.py"),
            f64_extension_sha256=self.f64_sha)
        admission.update(overrides)
        path = self.tmp / "admission.json"
        path.write_text(json.dumps(admission), encoding="utf-8")
        return path

    def run_check(self, admission_path, sidecar_path, output=None):
        output = Path(output) if output else self.tmp / "out"
        argv = ["--criteria", str(CRITERIA_PATH),
                "--geometry", str(GEOMETRY_PATH),
                "--sidecar", str(sidecar_path),
                "--admission", str(admission_path),
                "--output", str(output)]
        return cra.main(argv), output

    def failure(self, output):
        return json.loads((Path(output) / "failure.json").read_text(encoding="utf-8"))


class AdmissionRefusalTests(FakeBackendTestCase):
    def test_missing_admission(self):
        self.install_backend()
        sidecar = self.write_sidecar()
        rc, out = self.run_check(self.tmp / "absent.json", sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("admission missing", self.failure(out)["error"])

    def test_state_not_admitted(self):
        self.install_backend()
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar, state="revoked")
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("not admitted", self.failure(out)["error"])

    def test_consumed_flag(self):
        self.install_backend()
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar, consumed=True)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("already consumed", self.failure(out)["error"])

    def test_consumed_sibling(self):
        self.install_backend()
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar)
        admission.with_name(admission.name + ".consumed").write_text("spent")
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("already consumed", self.failure(out)["error"])

    def test_one_execution_only_false(self):
        self.install_backend()
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar, one_execution_only=False)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("one execution", self.failure(out)["error"])

    def assert_refused_before_consumption(self, admission, sidecar, fragment):
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn(fragment, self.failure(out)["error"])
        self.assertTrue(admission.exists(), "refusal must not consume the admission")

    def test_wrong_script_sha(self):
        self.install_backend()
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar, script_sha256="0" * 64)
        self.assert_refused_before_consumption(admission, sidecar, "script_sha256")

    def test_wrong_criteria_sha(self):
        self.install_backend()
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar, criteria_sha256="0" * 64)
        self.assert_refused_before_consumption(admission, sidecar, "criteria_sha256")

    def test_wrong_geometry_sha(self):
        self.install_backend()
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar, geometry_sha256="0" * 64)
        self.assert_refused_before_consumption(admission, sidecar, "geometry_sha256")

    def test_wrong_sidecar_sha(self):
        self.install_backend()
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar, sidecar_sha256="0" * 64)
        self.assert_refused_before_consumption(admission, sidecar, "sidecar_sha256")

    def test_wrong_python_executable(self):
        self.install_backend()
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar, python_executable="/other/python")
        self.assert_refused_before_consumption(admission, sidecar,
                                               "python_executable")

    def test_admission_f64_sha_mismatches_sidecar(self):
        self.install_backend()
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar, f64_extension_sha256="b" * 64)
        self.assert_refused_before_consumption(admission, sidecar,
                                               "f64 extension hash")

    def test_wrong_python_executable_sha(self):
        self.install_backend()
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar, python_executable_sha256="0" * 64)
        self.assert_refused_before_consumption(admission, sidecar,
                                               "python_executable_sha256")

    def test_wrong_delhommeau_module_sha(self):
        self.install_backend()
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar, delhommeau_module_sha256="0" * 64)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("delhommeau_module_sha256", self.failure(out)["error"])
        self.assertTrue(admission.exists(),
                        "GF module hash mismatch must refuse before consumption")

    def test_reissued_admission_rejected_after_consumption(self):
        self.install_backend()
        reset_record(passing_script())
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 0)
        admission = self.write_admission(sidecar)
        rc2, out2 = self.run_check(admission, sidecar, output=self.tmp / "out2")
        self.assertEqual(rc2, 2)
        self.assertIn("already consumed", self.failure(out2)["error"])

    def test_happy_path_consumes_once(self):
        self.install_backend()
        reset_record(passing_script())
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 0)
        self.assertFalse(admission.exists())
        self.assertTrue(admission.with_name(admission.name + ".consumed").exists())
        rc2, _ = self.run_check(admission, sidecar, output=self.tmp / "out2")
        self.assertEqual(rc2, 2)


class ConsumeAdmissionTests(FakeBackendTestCase):
    def test_claims_once_and_preserves_bytes(self):
        admission = self.tmp / "admission.json"
        admission.write_bytes(b"original admission bytes")
        consumed = admission.with_name(admission.name + ".consumed")
        cra.consume_admission(admission, consumed)
        self.assertFalse(admission.exists())
        self.assertEqual(consumed.read_bytes(), b"original admission bytes")

    def test_preexisting_consumed_claim_refused(self):
        admission = self.tmp / "admission.json"
        admission.write_bytes(b"original admission bytes")
        consumed = admission.with_name(admission.name + ".consumed")
        consumed.write_bytes(b"spent record")
        with self.assertRaises(cra.Refused) as ctx:
            cra.consume_admission(admission, consumed)
        self.assertIn("already consumed", str(ctx.exception))
        self.assertEqual(consumed.read_bytes(), b"spent record",
                         "a preexisting consumed record must never be overwritten")
        self.assertTrue(admission.exists())


class OutputProtectionTests(FakeBackendTestCase):
    def test_preexisting_output_refused_and_untouched(self):
        self.install_backend()
        reset_record(passing_script())
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar)
        out = self.tmp / "out"
        out.mkdir()
        sentinel = out / "prior-evidence.json"
        sentinel.write_text("prior evidence", encoding="utf-8")
        rc, _ = self.run_check(admission, sidecar, output=out)
        self.assertEqual(rc, 2)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "prior evidence")
        self.assertEqual([p.name for p in out.iterdir()], ["prior-evidence.json"],
                         "no failure record may modify a preexisting output")
        self.assertTrue(admission.exists(),
                        "output refusal must not consume the admission")
        self.assertEqual(RECORD["rankine"], [], "no native call may run")

    def test_output_path_existing_file_refused(self):
        self.install_backend()
        reset_record(passing_script())
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar)
        out = self.tmp / "out"
        out.write_text("prior bytes", encoding="utf-8")
        rc, _ = self.run_check(admission, sidecar, output=out)
        self.assertEqual(rc, 2)
        self.assertEqual(out.read_text(encoding="utf-8"), "prior bytes")
        self.assertTrue(admission.exists())


class BackendIdentityTests(FakeBackendTestCase):
    def make_admitted(self, **sidecar_overrides):
        sidecar = self.write_sidecar(**sidecar_overrides)
        admission = self.write_admission(sidecar)
        return sidecar, admission

    def test_sidecar_format(self):
        self.install_backend()
        sidecar, admission = self.make_admitted(format="other-v1")
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("sidecar format", self.failure(out)["error"])

    def test_identity_mismatch(self):
        self.install_backend()
        sidecar, admission = self.make_admitted(external_identity="other-backend")
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("backend_identity", self.failure(out)["error"])

    def test_wheel_mismatch(self):
        self.install_backend()
        sidecar, admission = self.make_admitted(wheel_sha256="0" * 64)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("wheel", self.failure(out)["error"])

    def test_old_library_hash_refused(self):
        self.install_backend()
        sidecar = self.write_sidecar(f64_sha="c" * 64)
        admission = self.write_admission(sidecar, f64_extension_sha256="c" * 64)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("old or wrong library", self.failure(out)["error"])

    def test_rounded_pi_refused(self):
        self.install_backend(pi=3.141592653588979)
        sidecar, admission = self.make_admitted()
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("PI", self.failure(out)["error"])

    def test_rounded_gamma_refused(self):
        self.install_backend(gamma=0.5772156649)
        sidecar, admission = self.make_admitted()
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("EULER_GAMMA", self.failure(out)["error"])

    def test_float32_pre_refused(self):
        self.install_backend(pre=4)
        sidecar, admission = self.make_admitted()
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("float64", self.failure(out)["error"])

    def test_threadpool_not_one(self):
        self.install_backend(pools=[{"user_api": "openmp", "num_threads": 2}])
        sidecar, admission = self.make_admitted()
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("thread", self.failure(out)["error"])


class BudgetGuardTests(FakeBackendTestCase):
    def make_guard(self):
        pinned = {getattr(FakeDelhommeau, name).__code__: name
                  for name in cra.PINNED_METHODS}
        budgets = {"__init__": 1, "_create_tabulation": 1,
                   "_create_or_load_tabulation": 0, "evaluate": 0,
                   "find_best_exponential_decomposition": 0,
                   "evaluate_rankine_only": 4}
        return cra.BudgetGuard(pinned, budgets)

    def test_constructor_budget(self):
        reset_record(passing_script())
        guard = self.make_guard()
        guard.install()
        self.addCleanup(guard.uninstall)
        FakeDelhommeau()
        with self.assertRaises(cra.Refused):
            FakeDelhommeau()
        self.assertEqual(guard.counts["__init__"], 1)

    def test_empty_table_counted_with_zero_dimensions(self):
        reset_record(passing_script())
        guard = self.make_guard()
        guard.install()
        self.addCleanup(guard.uninstall)
        FakeDelhommeau(tabulation_nr=0, tabulation_nz=0,
                       tabulation_cache_dir=None, floating_point_precision="float64")
        self.assertEqual(RECORD["create_tabulation"], [(0, 0)])
        self.assertEqual(guard.counts["_create_tabulation"], 1)

    def test_positive_table_refused_before_body(self):
        reset_record(passing_script())
        guard = self.make_guard()
        guard.install()
        self.addCleanup(guard.uninstall)
        with self.assertRaises(cra.Refused):
            PositiveTableDelhommeau()
        self.assertEqual(RECORD["create_tabulation"], [],
                         "positive table body must not run")

    def test_cache_route_refused_before_body(self):
        reset_record(passing_script())
        guard = self.make_guard()
        guard.install()
        self.addCleanup(guard.uninstall)
        with self.assertRaises(cra.Refused):
            FakeDelhommeau(tabulation_cache_dir="/tmp/cache")
        self.assertEqual(RECORD["load_tabulation"], [])

    def test_wave_refused_before_body(self):
        reset_record(passing_script())
        guard = self.make_guard()
        guard.install()
        self.addCleanup(guard.uninstall)
        gf = FakeDelhommeau()
        with self.assertRaises(cra.Refused):
            gf.evaluate(None, None, free_surface=0.0)
        self.assertEqual(RECORD["wave"], [])

    def test_prony_refused_before_body(self):
        reset_record(passing_script())
        guard = self.make_guard()
        guard.install()
        self.addCleanup(guard.uninstall)
        gf = FakeDelhommeau()
        with self.assertRaises(cra.Refused):
            gf.find_best_exponential_decomposition(1.0)
        self.assertEqual(RECORD["prony"], [])

    def test_rankine_budget(self):
        reset_record(passing_script())
        guard = self.make_guard()
        guard.install()
        self.addCleanup(guard.uninstall)
        gf = FakeDelhommeau()
        for _ in range(4):
            gf.evaluate_rankine_only(None, None)
        with self.assertRaises(cra.Refused):
            gf.evaluate_rankine_only(None, None)
        self.assertEqual(guard.counts["evaluate_rankine_only"], 4)


class PinMethodsTests(FakeBackendTestCase):
    def pin(self):
        cpt = sys.modules["capytaine"]
        return cra.pin_methods(cpt, CRITERIA["native_limits"])

    def test_genuine_lru_cache_unwrapped_to_body_code(self):
        self.install_backend(delhommeau_cls=LruCacheDelhommeau)
        cls, pinned, budgets, wrappers = self.pin()
        wrapper = LruCacheDelhommeau.find_best_exponential_decomposition
        body_code = wrapper.__wrapped__.__code__
        self.assertEqual(pinned[body_code], "find_best_exponential_decomposition")
        self.assertEqual(wrappers, {"find_best_exponential_decomposition": 1})

    def test_lru_cache_prony_body_refused_before_body(self):
        self.install_backend(delhommeau_cls=LruCacheDelhommeau)
        cls, pinned, budgets, wrappers = self.pin()
        guard = cra.BudgetGuard(pinned, budgets)
        guard.install()
        self.addCleanup(guard.uninstall)
        gf = cls()
        with self.assertRaises(cra.Refused):
            gf.find_best_exponential_decomposition(1.0)
        self.assertEqual(RECORD["prony"], [], "Prony body must not execute")

    def test_cached_wrapper_hit_skips_body_not_counted(self):
        self.install_backend(delhommeau_cls=LruCacheDelhommeau)
        cls, pinned, budgets, wrappers = self.pin()
        budgets = dict(budgets, find_best_exponential_decomposition=1)
        guard = cra.BudgetGuard(pinned, budgets)
        gf = cls()
        guard.install()
        self.addCleanup(guard.uninstall)
        gf.find_best_exponential_decomposition(1.0)
        gf.find_best_exponential_decomposition(1.0)
        self.assertEqual(guard.counts["find_best_exponential_decomposition"], 1)
        self.assertEqual(len(RECORD["prony"]), 1,
                         "a cached wrapper return must not execute the body")

    def test_unsupported_wrapper_without_terminal_code_refused(self):
        opaque = OpaqueWrapper()
        opaque.__wrapped__ = object()

        class BadWrapperDelhommeau(FakeDelhommeau):
            find_best_exponential_decomposition = opaque
        self.install_backend(delhommeau_cls=BadWrapperDelhommeau)
        cpt = sys.modules["capytaine"]
        with self.assertRaises(cra.Refused) as ctx:
            cra.pin_methods(cpt, CRITERIA["native_limits"])
        self.assertIn("supported wrapper", str(ctx.exception))

    def test_opaque_wrapper_without_wrapped_refused(self):
        class OpaqueDelhommeau(FakeDelhommeau):
            find_best_exponential_decomposition = OpaqueWrapper()
        self.install_backend(delhommeau_cls=OpaqueDelhommeau)
        cpt = sys.modules["capytaine"]
        with self.assertRaises(cra.Refused) as ctx:
            cra.pin_methods(cpt, CRITERIA["native_limits"])
        self.assertIn("supported wrapper", str(ctx.exception))


class RunIntegrationTests(FakeBackendTestCase):
    def test_pass(self):
        self.install_backend()
        reset_record(passing_script())
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 0)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertTrue(summary["all_passed"])
        self.assertEqual(summary["guard_counts"]["__init__"], 1)
        self.assertEqual(summary["guard_counts"]["_create_tabulation"], 1)
        self.assertEqual(summary["guard_counts"]["evaluate_rankine_only"], 4)
        self.assertEqual(summary["guard_counts"]["evaluate"], 0)
        self.assertEqual(RECORD["quadrature_method"], cra.QUADRATURE_METHOD)
        self.assertEqual(RECORD["wave"], [])
        self.assertEqual(RECORD["prony"], [])
        self.assertEqual(RECORD["load_tabulation"], [])
        self.assertEqual(RECORD["rankine"],
                         [(True, False, False), (False, False, False),
                          (True, False, False), (True, False, True)])
        predicates = json.loads((out / "rankine-predicates.json")
                               .read_text(encoding="utf-8"))
        for name in cra.PREDICATE_NAMES:
            self.assertTrue(predicates[name]["passed"], name)
        self.assertIn("near_plane_limit", predicates)
        with np.load(out / "rankine-raw.npz") as raw:
            self.assertEqual(sorted(raw.files), sorted(
                f"{prefix}_{name}" for name in cra.CALL_NAMES
                for prefix in ("s", "k")))

    def test_predicate_failure(self):
        self.install_backend()
        reset_record(failing_script())
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 1)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertFalse(summary["all_passed"])
        predicates = json.loads((out / "rankine-predicates.json")
                               .read_text(encoding="utf-8"))
        self.assertFalse(predicates["off_panel_adjoint_normal_vs_solid_angle"]
                         ["passed"])

    def test_raw_retained_on_invalid_shape(self):
        self.install_backend()
        script = passing_script()
        script[1] = (np.ones((35, 2)), script[1][1])
        reset_record(script)
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("shapes", self.failure(out)["error"])
        with np.load(out / "rankine-raw.npz") as raw:
            self.assertEqual(raw["s_adjoint_cloud_no_jump"].shape, (35, 1))
            self.assertEqual(raw["s_source_cloud_no_jump"].shape, (35, 2),
                             "invalid raw output must be retained")

    def test_raw_retained_on_nonfinite(self):
        self.install_backend()
        script = passing_script()
        k_bad = script[2][1].copy()
        k_bad[1, 0, 0] = np.nan
        script[2] = (script[2][0], k_bad)
        reset_record(script)
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("nonfinite", self.failure(out)["error"])
        with np.load(out / "rankine-raw.npz") as raw:
            self.assertTrue(np.isnan(raw["k_self_mesh_adjoint_no_jump"][1, 0, 0]),
                            "nonfinite raw output must be retained")

    def test_first_invalid_shape_stops_after_one_call(self):
        self.install_backend()
        script = passing_script()
        script[0] = (np.ones((35, 2)), script[0][1])
        reset_record(script)
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        failure = self.failure(out)
        self.assertIn("shapes", failure["error"])
        self.assertEqual(len(RECORD["rankine"]), 1,
                         "an invalid first result must stop before another call")
        self.assertEqual(failure["guard_counts"]["evaluate_rankine_only"], 1)
        with np.load(out / "rankine-raw.npz") as raw:
            self.assertEqual(sorted(raw.files),
                             ["k_adjoint_cloud_no_jump", "s_adjoint_cloud_no_jump"])
            self.assertEqual(raw["s_adjoint_cloud_no_jump"].shape, (35, 2),
                             "the invalid first result must be retained")

    def test_first_nonfinite_stops_after_one_call(self):
        self.install_backend()
        script = passing_script()
        k_bad = script[0][1].copy()
        k_bad[0, 0, 0] = np.inf
        script[0] = (script[0][0], k_bad)
        reset_record(script)
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        failure = self.failure(out)
        self.assertIn("nonfinite", failure["error"])
        self.assertEqual(len(RECORD["rankine"]), 1,
                         "a nonfinite first result must stop before another call")
        self.assertEqual(failure["guard_counts"]["evaluate_rankine_only"], 1)
        with np.load(out / "rankine-raw.npz") as raw:
            self.assertTrue(np.isinf(raw["k_adjoint_cloud_no_jump"][0, 0, 0]),
                            "the nonfinite first result must be retained")

    def test_geometry_mismatch_refused(self):
        self.install_backend(mesh_cls=BadNormalsMesh)
        reset_record(passing_script())
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        self.assertIn("frozen geometry", self.failure(out)["error"])

    def test_positive_table_refused_in_run(self):
        self.install_backend(delhommeau_cls=PositiveTableDelhommeau)
        reset_record(passing_script())
        sidecar = self.write_sidecar()
        admission = self.write_admission(sidecar)
        rc, out = self.run_check(admission, sidecar)
        self.assertEqual(rc, 2)
        failure = self.failure(out)
        self.assertIn("refused before body", failure["error"])
        self.assertEqual(failure["guard_counts"]["_create_tabulation"], 0)
        self.assertEqual(RECORD["create_tabulation"], [])
        self.assertFalse(admission.exists(), "admission is consumed before "
                         "the constructor and stays spent after refusal")


if __name__ == "__main__":
    unittest.main()
