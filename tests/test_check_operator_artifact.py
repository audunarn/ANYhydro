"""Mock-only admission, guard and probe tests for check_operator_artifact.

No real Green function is imported, constructed or executed: capytaine, the
delhommeau module, the f64 extension and threadpoolctl are fakes injected into
sys.modules, and every native call returns scripted synthetic data. The probe
case is the real public caller on the original 864-panel coupled geometry with
a mock backend; that is route evidence, not BEM evidence.
"""
import contextlib
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "native_precision"))
import check_operator_artifact as coa  # noqa: E402
import check_rankine_artifact as helper  # noqa: E402

import anyhydro
from test_artifact_bound_panel_profile import (MODULE_FILES, build_artifact_case,
                                               canonical, install_evaluator_backend)
from test_panel_integration_candidate import fake_backend

F64_KEY = "capytaine/green_functions/Delhommeau_float64.cpython-312-x86_64-linux-gnu.so"
RECORD = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture
def route(tmp_path, monkeypatch):
    """Full mocked route: 864-panel bundle, binding, sidecar, criteria, admission."""
    RECORD.clear()
    RECORD.update(constructor=0, create_tabulation=[], load_tabulation=0,
                  wave=[], prony=0, rankine=0, override=0)
    case = build_artifact_case(tmp_path, monkeypatch, subdivisions=6)
    install_evaluator_backend(case)
    green = case['green']
    delhommeau_file = tmp_path/'delhommeau.py'
    delhommeau_file.write_text('# fake delhommeau')
    sys.modules['capytaine.green_functions.delhommeau'].__file__ = str(delhommeau_file)
    f64 = types.SimpleNamespace(__file__=str(case['native']),
                                constants=types.SimpleNamespace(pi=math.pi,
                                                                euler_gamma=helper.GAMMA_REFERENCE),
                                floating_point_precision=types.SimpleNamespace(pre=8))
    monkeypatch.setitem(sys.modules, 'capytaine.green_functions.Delhommeau_float64', f64)
    tp = types.ModuleType('threadpoolctl')
    @contextlib.contextmanager
    def threadpool_limits(limits=None):
        yield
    tp.threadpool_limits = threadpool_limits
    tp.threadpool_info = lambda: [{'user_api': 'openmp', 'num_threads': 1}]
    monkeypatch.setitem(sys.modules, 'threadpoolctl', tp)

    def init(self, **kw):
        green._original_init(self, **kw)
        RECORD['constructor'] += 1
        if kw.get('tabulation_cache_dir') is None:
            self._create_tabulation(kw.get('tabulation_nr', 0), kw.get('tabulation_nz', 0))
        else:
            self._create_or_load_tabulation()
    def create_tabulation(self, tabulation_nr, tabulation_nz):
        RECORD['create_tabulation'].append((tabulation_nr, tabulation_nz))
    def load_tabulation(self, *a, **k):
        RECORD['load_tabulation'] += 1
    def prony(self, kh, method=None):
        RECORD['prony'] += 1
        return case['coeff']
    def rankine(self, *a, **k):
        RECORD['rankine'] += 1
    original_evaluate = green.evaluate
    def evaluate(self, receivers, source, **kwargs):
        RECORD['wave'].append(len(receivers))
        self.find_best_exponential_decomposition(kwargs['wavenumber']*kwargs['water_depth'])
        return original_evaluate(self, receivers, source, **kwargs)
    sys.modules['capytaine'].Delhommeau = green
    green._original_init = green.__init__
    green.__init__ = init
    green._create_tabulation = create_tabulation
    green._create_or_load_tabulation = load_tabulation
    green.find_best_exponential_decomposition = prony
    green.evaluate_rankine_only = rankine
    green.evaluate = evaluate

    manifest = json.loads((case['root']/'manifest.json').read_text())
    f64_sha = case['binding']['evaluator']['native_extension_sha256'][F64_KEY]
    criteria = dict(format=coa.CRITERIA_FORMAT,
                    external_identity=case['binding']['evaluator']['external_identity'],
                    wheel_sha256=case['binding']['evaluator']['wheel_sha256'],
                    f64_sha256=f64_sha,
                    original_reference=str(case['root']),
                    fixed_coefficients_sha256=coa.FIXED_COEFFICIENTS_SHA256,
                    source_panels=864, probe_receiver_rows=4,
                    precision_limit=coa.PRECISION_LIMIT, steps=list(coa.EXPECTED_STEPS),
                    native_worker_execution=False,
                    native_future=dict(coa.EXPECTED_NATIVE_FUTURE),
                    linux_new_root='/var/tmp/anyhydro-e1-operator-route-20261010',
                    evaluator_backend_source_sha256=case['evaluator_sources'],
                    producer_evaluator_separation='test separation statement')
    criteria_path = tmp_path/'criteria.json'
    criteria_path.write_text(json.dumps(criteria))
    admission = dict(format=coa.ADMISSION_FORMAT, state='admitted',
                     one_execution_only=True,
                     script_sha256=sha(coa.__file__),
                     criteria_sha256=sha(criteria_path),
                     binding_sha256=sha(case['binding_path']),
                     sidecar_sha256=sha(case['sidecar_path']),
                     reference_manifest_sha256=hashlib.sha256(canonical(manifest).encode()).hexdigest(),
                     python_executable=sys.executable,
                     python_executable_sha256=sha(sys.executable),
                     evaluator_backend_source_sha256=case['evaluator_sources'],
                     f64_extension_sha256=f64_sha,
                     wheel_sha256=criteria['wheel_sha256'],
                     installed_anyhydro_module_sha256={
                         name: sha(MODULE_FILES[name]) for name in coa.IMPLEMENTATION_MODULES},
                     delhommeau_module_sha256=sha(delhommeau_file),
                     limits=dict(coa.EXPECTED_NATIVE_FUTURE))
    admission_path = tmp_path/'admission.json'
    admission_path.write_text(json.dumps(admission))
    case.update(criteria=criteria, criteria_path=criteria_path,
                admission=admission, admission_path=admission_path, f64_sha=f64_sha)
    return case


def run_check(case, output, **arg_changes):
    argv = dict(criteria=str(case['criteria_path']), reference=str(case['root']),
                binding=str(case['binding_path']), sidecar=str(case['sidecar_path']),
                admission=str(case['admission_path']), output=str(output))
    argv.update(arg_changes)
    return coa.main([item for key, value in argv.items() for item in (f'--{key}', value)])


def failure(output):
    return json.loads((Path(output)/'failure.json').read_text(encoding='utf-8'))


def test_successful_mocked_probe(route, tmp_path):
    output = tmp_path/'out'
    admission_bytes = route['admission_path'].read_bytes()
    rc = run_check(route, output)
    assert rc == 0
    summary = json.loads((output/'summary.json').read_text())
    assert summary['status'] == 'probe_unqualified' and summary['qualified'] is False
    assert summary['probe']['receiver_count'] == 4
    assert summary['probe']['source_panel_count'] == 864
    assert summary['probe']['body_count'] == 2
    assert summary['f32_extension_sha256'] == route['binding']['evaluator']['native_extension_sha256'][
        'capytaine/green_functions/Delhommeau_float32.cpython-312-x86_64-linux-gnu.so']
    assert summary['guard_counts'] == dict(__init__=1, _create_tabulation=1,
                                           _create_or_load_tabulation=0, evaluate=2,
                                           find_best_exponential_decomposition=0,
                                           evaluate_rankine_only=0,
                                           fixed_coefficient_override=2)
    assert summary['producer']['backend'] == route['binding']['producer_backend']
    assert summary['evaluator']['backend_source_sha256'] == route['evaluator_sources']
    assert not route['admission_path'].exists()
    consumed = route['admission_path'].with_name(route['admission_path'].name+'.consumed')
    assert consumed.read_bytes() == admission_bytes
    status = json.loads((output/'case'/'status.json').read_text())
    assert status['state'] == 'completed_probe_unqualified'
    assert status['numerical_systems'] == 0 and status['new_green_evaluations'] == 2
    assert RECORD['constructor'] == 1 and RECORD['create_tabulation'] == [(0, 0)]
    assert RECORD['wave'] == [43, 4]
    assert RECORD['prony'] == 0 and RECORD['rankine'] == 0 and RECORD['load_tabulation'] == 0
    assert (output/'case'/'native-operator-probe.npz').is_file()
    assert not (output/'case'/'radiation.npz').exists()


def test_criteria_format_mismatch_refused_before_consume(route, tmp_path):
    route['criteria']['format'] = 'wrong'
    route['criteria_path'].write_text(json.dumps(route['criteria']))
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'criteria format' in failure(output)['error']
    assert route['admission_path'].exists()


def test_admission_script_hash_mismatch_refused(route, tmp_path):
    route['admission']['script_sha256'] = '0'*64
    route['admission_path'].write_text(json.dumps(route['admission']))
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'script_sha256' in failure(output)['error']
    assert route['admission_path'].exists()


def test_preconsumed_admission_refused(route, tmp_path):
    route['admission_path'].with_name(
        route['admission_path'].name+'.consumed').write_text('spent')
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'already consumed' in failure(output)['error']


def test_admission_evaluator_map_mismatch_refused(route, tmp_path):
    route['admission']['evaluator_backend_source_sha256'] = {
        name: '0'*64 for name in route['evaluator_sources']}
    route['admission_path'].write_text(json.dumps(route['admission']))
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'evaluator source map' in failure(output)['error']
    assert route['admission_path'].exists()


def test_installed_module_hash_mismatch_refused(route, tmp_path):
    route['admission']['installed_anyhydro_module_sha256'][
        'panel_integration_candidate.py'] = '0'*64
    route['admission_path'].write_text(json.dumps(route['admission']))
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'installed ANYhydro module' in failure(output)['error']
    assert route['admission_path'].exists()


def test_energy_mode_reserved_and_refused(route, tmp_path):
    route['admission']['mode'] = 'energy'
    route['admission_path'].write_text(json.dumps(route['admission']))
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'reserved and refused' in failure(output)['error']
    assert route['admission_path'].exists()


def test_reference_directory_mismatch_refused(route, tmp_path):
    output = tmp_path/'out'
    rc = run_check(route, output, reference=str(route['root'].with_name('other')))
    assert rc == 2 and 'original_reference' in failure(output)['error']


def test_default_prony_fit_refused_before_body(route, tmp_path, monkeypatch):
    green = route['green']
    original = green.evaluate
    def evaluate(self, receivers, source, **kwargs):
        RECORD['wave'].append(len(receivers))
        green.find_best_exponential_decomposition(
            self, kwargs['wavenumber']*kwargs['water_depth'])
        return original(self, receivers, source, **kwargs)
    monkeypatch.setattr(green, 'evaluate', evaluate)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2
    assert 'find_best_exponential_decomposition' in failure(output)['error']
    assert not route['admission_path'].exists()


def test_fixed_override_budget_exhausted(route, tmp_path, monkeypatch):
    green = route['green']
    original = green.evaluate
    def evaluate(self, receivers, source, **kwargs):
        RECORD['wave'].append(len(receivers))
        kh = kwargs['wavenumber']*kwargs['water_depth']
        self.find_best_exponential_decomposition(kh)
        self.find_best_exponential_decomposition(kh)
        return original(self, receivers, source, **kwargs)
    monkeypatch.setattr(green, 'evaluate', evaluate)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'fixed_coefficient_override' in failure(output)['error']


def test_wrong_effective_settings_refused(route, tmp_path, monkeypatch):
    green = route['green']
    original = green.__init__
    def init(self, **kw):
        original(self, **kw)
        self.exportable_settings['gf_singularities'] = 'high_freq'
    monkeypatch.setattr(green, '__init__', init)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'effective settings' in failure(output)['error']


def test_threadpool_not_one_refused(route, tmp_path, monkeypatch):
    tp = types.ModuleType('threadpoolctl')
    @contextlib.contextmanager
    def threadpool_limits(limits=None):
        yield
    tp.threadpool_limits = threadpool_limits
    tp.threadpool_info = lambda: [{'user_api': 'openmp', 'num_threads': 2}]
    monkeypatch.setitem(sys.modules, 'threadpoolctl', tp)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'one thread' in failure(output)['error']


def test_f64_extension_mismatch_refused(route, tmp_path):
    route['native'].write_bytes(b'old or wrong library bytes')
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'f64 extension hash' in failure(output)['error']


def test_f32_extension_mismatch_refused(route, tmp_path):
    route['native32'].write_bytes(b'old or wrong float32 library')
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'f32 extension' in failure(output)['error']


def test_body_count_not_two_refused(route, tmp_path, monkeypatch):
    monkeypatch.setattr(anyhydro, 'make_first_slice_bodies', lambda **kw: [object()]*3)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'two-body' in failure(output)['error']


def test_existing_output_refused(route, tmp_path):
    output = tmp_path/'out'
    output.mkdir()
    (output/'prior.txt').write_text('prior evidence')
    rc = run_check(route, output)
    assert rc == 2
    assert (output/'prior.txt').read_text() == 'prior evidence'
    assert not (output/'failure.json').exists()
