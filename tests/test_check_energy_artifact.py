"""Mock-only admission, guard, comparison and energy-point tests.

No real Green function is imported, constructed or executed: capytaine, the
delhommeau module, the f64 extension and threadpoolctl are fakes injected into
sys.modules, and every native call returns scripted synthetic data. The energy
point is the real public caller on the original 864-panel coupled geometry
with a mock backend; that is route evidence, not BEM evidence and never a
qualification. The frozen HAMS reference values are the real registered
numbers; the mock candidate cannot match them, which is exactly the
mismatching-basis path. Matching-basis, promotion-branch, helper-pin and
nonfinite behaviour is covered by the pure comparison/classifier functions
and by route tests whose injection oracle supplies stored retained
blocks/evidence while every production pin and rule stays active.
"""
import contextlib
import dataclasses
import hashlib
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "native_precision"))
import check_energy_artifact as cea  # noqa: E402
import check_rankine_artifact as helper  # noqa: E402

import anyhydro
import anyhydro.panel_integration_candidate as panel_candidate_module
from anyhydro.records import HydroBody
from anyhydro.panel_accuracy_reference import PanelCandidateCancelled
from anyhydro.panel_integration_candidate import PanelRadiationCandidate
from test_artifact_bound_panel_profile import (MODULE_FILES, build_artifact_case,
                                               canonical, install_evaluator_backend)

F64_KEY = "capytaine/green_functions/Delhommeau_float64.cpython-312-x86_64-linux-gnu.so"
F32_KEY = "capytaine/green_functions/Delhommeau_float32.cpython-312-x86_64-linux-gnu.so"
AXES = ["first-slice-a::surge", "first-slice-a::sway", "first-slice-a::heave",
        "first-slice-a::roll", "first-slice-a::pitch", "first-slice-a::yaw",
        "first-slice-b::surge", "first-slice-b::sway", "first-slice-b::heave",
        "first-slice-b::roll", "first-slice-b::pitch", "first-slice-b::yaw"]
HAMS_ADDED_MASS = [[361.7871775, -0.19149009, -6.081979975, -18.163635499999998],
                   [-0.18144867750000002, 425.170205, 18.02088375, 7.250498425],
                   [-6.133797824999999, 18.18373575, 361.844885, 0.27757245999999997],
                   [-18.00345875, 7.199904425, -0.1683815675, 425.01819750000004]]
HAMS_DAMPING = [[2.88396255, 0.63574810125, 2.0947668750000004, -10.1883114],
                [0.44811267749999995, 114.03154725000002, 10.2446653875, 90.79916887499999],
                [2.0645965125, 10.4567235375, 2.8826818125, -0.296816835],
                [-10.244714587499999, 90.77390774999999, -0.44918508375, 114.047937]]
HAMS_MASS_BOUND = [[5.125e-05, 5.125e-08, 5.125000000000001e-07, 5.125e-06],
                   [5.125e-08, 5.125e-05, 5.125e-06, 5.125000000000001e-07],
                   [5.125000000000001e-07, 5.125e-06, 5.125e-05, 5.125e-08],
                   [5.125e-06, 5.125000000000001e-07, 5.125e-08, 5.125e-05]]
HAMS_DAMP_BOUND = [[7.6875e-07, 7.6875e-08, 7.6875e-07, 7.6875e-07],
                   [7.6875e-08, 7.6875e-06, 7.6875e-07, 7.6875e-06],
                   [7.6875e-07, 7.6875e-07, 7.6875e-07, 7.6875e-08],
                   [7.6875e-07, 7.6875e-06, 7.6875e-08, 7.6875e-06]]
RECORD = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hams_controls():
    return dict(format="hams-single-radiation-reference-diagnostic-v1",
                input_manifest=dict(rho_target=1025.0, gravity_target=9.81,
                                   omega_target=1.5, water_depth=150.0, axes=list(AXES)),
                retained_dofs=list(cea.RETAINED_DOFS),
                real_retained_rate=list(cea.RETAINED_RATE),
                comparison=dict(relative=cea.RELATIVE, absolute=cea.ABSOLUTE,
                                near_zero_floor=cea.NEAR_ZERO_FLOOR,
                                precision=cea.COMPARISON_PRECISION),
                mean_power=dict(nonnegative_absolute_w=cea.SIGN_THRESHOLD,
                                precision=cea.MEAN_POWER_PRECISION),
                qualification="development_unqualified")


def hams_report():
    return dict(qualification=False, backend_admission=False,
                hams_mean_power_w=cea.HAMS_MEAN_POWER_W,
                hams_print_bound_w=cea.HAMS_PRINT_BOUND_W,
                hams_sign=cea.HAMS_SIGN,
                comparisons=dict(
                    added_mass=dict(hams_retained=[row[:] for row in HAMS_ADDED_MASS],
                                    print_bound=[row[:] for row in HAMS_MASS_BOUND]),
                    radiation_damping=dict(hams_retained=[row[:] for row in HAMS_DAMPING],
                                            print_bound=[row[:] for row in HAMS_DAMP_BOUND])))


def hams_intent():
    return dict(retained_dofs=list(cea.RETAINED_DOFS),
                real_retained_rate=list(cea.RETAINED_RATE),
                comparison=dict(relative=cea.RELATIVE, absolute=cea.ABSOLUTE,
                                near_zero_floor=cea.NEAR_ZERO_FLOOR,
                                precision=cea.COMPARISON_PRECISION),
                mean_power=dict(nonnegative_absolute_w=cea.SIGN_THRESHOLD,
                                precision=cea.MEAN_POWER_PRECISION),
                original_controls_sha256=cea.HAMS_CONTROLS_SHA256,
                original_report_sha256=cea.HAMS_REPORT_SHA256,
                saved_hams_mean_power_w=cea.HAMS_MEAN_POWER_W,
                saved_hams_print_bound_w=cea.HAMS_PRINT_BOUND_W,
                qualification=False)


@pytest.fixture
def route(tmp_path, monkeypatch):
    """Full mocked energy route: 864-panel bundle, binding, sidecar, HAMS
    packet, registration, intent, criteria and a fresh energy admission."""
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

    controls_path = tmp_path/'point-controls.json'
    controls_path.write_text(json.dumps(hams_controls()))
    report_path = tmp_path/'point-report.json'
    report_path.write_text(json.dumps(hams_report()))
    # Local fixture-only pin: the route generates its own controls/report
    # bytes, so the frozen-hash equality in validate_criteria is monkey-
    # patched to those generated hashes. Production keeps the strict pins.
    monkeypatch.setattr(cea, 'HAMS_CONTROLS_SHA256', sha(controls_path))
    monkeypatch.setattr(cea, 'HAMS_REPORT_SHA256', sha(report_path))
    intent_path = tmp_path/'future-energy-contract-intent.json'
    intent_path.write_text(json.dumps(hams_intent()))
    registration_path = tmp_path/'registration-criteria.json'
    registration_path.write_text(json.dumps(dict(format=cea.REGISTRATION_FORMAT,
                                                 **cea.EXPECTED_REGISTRATION)))
    run_script_path = tmp_path/'run.py'
    run_script_path.write_text('# fake ordinary run script')

    manifest = json.loads((case['root']/'manifest.json').read_text())
    f64_sha = case['binding']['evaluator']['native_extension_sha256'][F64_KEY]
    criteria = dict(format=cea.CRITERIA_FORMAT,
                    scope='test energy point', question='test',
                    external_identity=case['binding']['evaluator']['external_identity'],
                    wheel_sha256=case['binding']['evaluator']['wheel_sha256'],
                    f64_sha256=f64_sha,
                    original_reference=str(case['root']),
                    fixed_coefficients_sha256=cea.FIXED_COEFFICIENTS_SHA256,
                    source_panels=864, receiver_nodes=3456, body_count=2,
                    quadrature=cea.QUADRATURE_METHOD,
                    omega=cea.OMEGA, water_depth=cea.WATER_DEPTH,
                    rho=cea.RHO, gravity=cea.GRAVITY,
                    rotation_centers=[list(c) for c in cea.ROTATION_CENTERS],
                    retained_dofs=list(cea.RETAINED_DOFS),
                    retained_indices=list(cea.RETAINED_INDICES),
                    retained_rate=list(cea.RETAINED_RATE),
                    comparison=dict(relative=cea.RELATIVE, absolute=cea.ABSOLUTE,
                                    near_zero_floor=cea.NEAR_ZERO_FLOOR,
                                    precision=cea.COMPARISON_PRECISION,
                                    candidate_print_bound=cea.CANDIDATE_PRINT_BOUND),
                    mean_power=dict(nonnegative_absolute_w=cea.SIGN_THRESHOLD,
                                    precision=cea.MEAN_POWER_PRECISION,
                                    hams_mean_power_w=cea.HAMS_MEAN_POWER_W,
                                    hams_print_bound_w=cea.HAMS_PRINT_BOUND_W),
                    hams_controls_sha256=sha(controls_path),
                    hams_report_sha256=sha(report_path),
                    registration_sha256=sha(registration_path),
                    intent_sha256=sha(intent_path),
                    run_script_sha256=sha(run_script_path),
                    native_worker_execution=False,
                    negative_power_rejects_source_admission=True,
                    native_future=dict(cea.EXPECTED_NATIVE_FUTURE),
                    evaluator_backend_source_sha256=case['evaluator_sources'],
                    producer_evaluator_separation='test separation statement')
    criteria_path = tmp_path/'criteria.json'
    criteria_path.write_text(json.dumps(criteria))
    admission = dict(format=cea.ADMISSION_FORMAT, state='admitted',
                     one_execution_only=True, mode='energy',
                     script_sha256=sha(cea.__file__),
                     helper_module_sha256=sha(helper.__file__),
                     criteria_sha256=sha(criteria_path),
                     binding_sha256=sha(case['binding_path']),
                     sidecar_sha256=sha(case['sidecar_path']),
                     reference_manifest_sha256=hashlib.sha256(canonical(manifest).encode()).hexdigest(),
                     registration_sha256=sha(registration_path),
                     intent_sha256=sha(intent_path),
                     hams_controls_sha256=sha(controls_path),
                     hams_report_sha256=sha(report_path),
                     run_script_sha256=sha(run_script_path),
                     python_executable=sys.executable,
                     python_executable_sha256=sha(sys.executable),
                     evaluator_backend_source_sha256=case['evaluator_sources'],
                     f64_extension_sha256=f64_sha,
                     wheel_sha256=criteria['wheel_sha256'],
                     installed_anyhydro_module_sha256={
                         name: sha(MODULE_FILES[name]) for name in cea.IMPLEMENTATION_MODULES},
                     delhommeau_module_sha256=sha(delhommeau_file),
                     limits=dict(cea.EXPECTED_NATIVE_FUTURE))
    admission_path = tmp_path/'admission.json'
    admission_path.write_text(json.dumps(admission))
    case.update(criteria=criteria, criteria_path=criteria_path,
                admission=admission, admission_path=admission_path, f64_sha=f64_sha,
                controls_path=controls_path, report_path=report_path,
                intent_path=intent_path, registration_path=registration_path,
                run_script_path=run_script_path, delhommeau_file=delhommeau_file)
    return case


def run_check(case, output, **arg_changes):
    argv = dict(criteria=str(case['criteria_path']), reference=str(case['root']),
                binding=str(case['binding_path']), sidecar=str(case['sidecar_path']),
                registration=str(case['registration_path']),
                intent=str(case['intent_path']), controls=str(case['controls_path']),
                report=str(case['report_path']), **{'run-script': str(case['run_script_path'])},
                admission=str(case['admission_path']), output=str(output))
    argv.update(arg_changes)
    return cea.main([item for key, value in argv.items() for item in (f'--{key}', value)])


def failure(output):
    return json.loads((Path(output)/'failure.json').read_text(encoding='utf-8'))


def rewrite(path, payload):
    path.write_text(json.dumps(payload))


def test_full_mocked_energy_point_mismatching_hams(route, tmp_path):
    output = tmp_path/'out'
    admission_bytes = route['admission_path'].read_bytes()
    rc = run_check(route, output)
    assert rc == 1
    summary = json.loads((output/'summary.json').read_text())
    assert summary['computation_state'] == 'completed_unqualified'
    assert summary['status'] == 'energy_point_comparison_failed_unqualified'
    assert summary['registered_comparison'] == 'failed'
    assert summary['qualified'] is False
    assert set(summary['registered_predicates']) == {'added_mass_block',
                                                     'radiation_damping_block',
                                                     'mean_power'}
    assert all(v == 'failed' for v in summary['registered_predicates'].values())
    assert summary['guard_counts'] == dict(
        __init__=1, _create_tabulation=1, _create_or_load_tabulation=0, evaluate=5,
        find_best_exponential_decomposition=0, evaluate_rankine_only=0,
        fixed_coefficient_override=5, numpy_linalg_solve=1, imaginary_pole=4,
        directional_evidence=1)
    assert summary['pinned_code_identities']['numpy_linalg_solve']['filename'].endswith('.py')
    assert 'numpy' in summary['numpy_identity']['numpy_linalg_module_file']
    assert RECORD['constructor'] == 1 and RECORD['create_tabulation'] == [(0, 0)]
    assert RECORD['wave'] == [43, 864, 864, 864, 864]
    assert RECORD['prony'] == 0 and RECORD['rankine'] == 0 and RECORD['load_tabulation'] == 0
    assert summary['f32_extension_sha256'] == route['binding']['evaluator'][
        'native_extension_sha256'][F32_KEY]
    assert not route['admission_path'].exists()
    consumed = route['admission_path'].with_name(route['admission_path'].name+'.consumed')
    assert consumed.read_bytes() == admission_bytes
    status = json.loads((output/'case'/'status.json').read_text())
    assert status['state'] == 'completed_unqualified'
    assert status['numerical_systems'] == 1 and status['new_green_evaluations'] == 5
    for qi in range(4):
        assert (output/'case'/f'native-operator-{qi}.npz').is_file()
        assert (output/'case'/f'operator-{qi}.npz').is_file()
    assert (output/'case'/'radiation.npz').is_file()
    evidence = json.loads((output/'directional-evidence.json').read_text())
    assert evidence['rate'][0] == cea.RETAINED_RATE[0]
    assert evidence['rate'][1] == 0.0 and evidence['rate'][8] == cea.RETAINED_RATE[3]
    assert len(evidence['far_mean_powers_w']) == 3
    assert evidence['mean_residual_is_not_pointwise_acceptance'] is True
    comparison = json.loads((output/'energy-comparison.json').read_text())
    assert comparison['registered_predicates']['mean_power'] == 'failed'
    assert comparison['force_per_velocity']['registered_predicate'] is False
    assert comparison['diagnostics_only']['far_equality'] == (
        'constructed, not an independent oracle')
    assert comparison['policy']['candidate_print_bound'] == 0.0
    blocks = np.load(output/'retained-blocks.npz')
    assert blocks['added_mass'].shape == (4, 4)
    assert blocks['force_per_velocity'].shape == (4, 4)
    assert blocks['force_per_velocity'].dtype.kind == 'c'
    assert list(blocks['dofs'][[0, 2, 6, 8]]) == list(cea.RETAINED_DOFS)
    assert summary['source_admission']['general_numerical_admission'] is False


def test_negative_candidate_power_stops_source_admission(route, tmp_path, monkeypatch):
    monkeypatch.setattr(cea, 'classify_sign', lambda mean, bound, **kw: 'negative')
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 1
    summary = json.loads((output/'summary.json').read_text())
    assert summary['source_admission']['state'] == 'stopped_negative_power'
    assert summary['source_admission']['formulation_question']
    assert summary['candidate']['candidate_sign'] == 'negative'


def inject_stored_candidate(monkeypatch, near_mean_power_w):
    """Test oracle inside the mock boundaries: keep the real public caller
    (raw saves, status/case records, exact guard counts) and independently
    supply stored retained blocks and stored near-field mean power. The
    production HAMS literal pins, per-coefficient rules, physical rate and
    sign rules stay fully active; only the candidate payload is scripted."""
    original_runner = panel_candidate_module.run_panel_integration_candidate
    index = np.ix_(list(cea.RETAINED_INDICES), list(cea.RETAINED_INDICES))

    def stored_runner(bodies, **kwargs):
        result = original_runner(bodies, **kwargs)
        added = np.array(result.added_mass, dtype=float)
        damping = np.array(result.radiation_damping, dtype=float)
        added[index] = HAMS_ADDED_MASS
        damping[index] = HAMS_DAMPING
        return dataclasses.replace(result, added_mass=added,
                                   radiation_damping=damping)

    original_evidence = PanelRadiationCandidate.directional_evidence

    def stored_evidence(self, rate):
        evidence = dict(original_evidence(self, rate))
        evidence['near_mean_power_w'] = near_mean_power_w
        return evidence

    monkeypatch.setattr(panel_candidate_module, 'run_panel_integration_candidate',
                        stored_runner)
    monkeypatch.setattr(PanelRadiationCandidate, 'directional_evidence',
                        stored_evidence)


def test_matching_blocks_and_work_complete_unqualified(route, tmp_path, monkeypatch):
    inject_stored_candidate(monkeypatch, cea.HAMS_MEAN_POWER_W)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 0
    summary = json.loads((output/'summary.json').read_text())
    assert summary['status'] == 'energy_point_completed_unqualified'
    assert summary['registered_comparison'] == 'passed'
    assert summary['computation_state'] == 'completed_unqualified'
    assert summary['qualified'] is False
    assert all(v == 'passed' for v in summary['registered_predicates'].values())
    assert summary['source_admission']['general_numerical_admission'] is False
    assert summary['source_admission']['state'] == 'not_admitted'
    assert summary['candidate']['near_mean_power_w'] == cea.HAMS_MEAN_POWER_W
    assert summary['guard_counts'] == dict(
        __init__=1, _create_tabulation=1, _create_or_load_tabulation=0, evaluate=5,
        find_best_exponential_decomposition=0, evaluate_rankine_only=0,
        fixed_coefficient_override=5, numpy_linalg_solve=1, imaginary_pole=4,
        directional_evidence=1)
    status = json.loads((output/'case'/'status.json').read_text())
    assert status['state'] == 'completed_unqualified'
    comparison = json.loads((output/'energy-comparison.json').read_text())
    assert comparison['mean_power']['verdict'] == 'pass'
    assert comparison['candidate_sign'] == 'positive'
    assert not route['admission_path'].exists()


def test_mean_power_only_mismatch_fails_not_unresolved(route, tmp_path, monkeypatch):
    inject_stored_candidate(monkeypatch, cea.HAMS_MEAN_POWER_W + 1.0)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 1
    summary = json.loads((output/'summary.json').read_text())
    assert summary['status'] == 'energy_point_comparison_failed_unqualified'
    assert summary['registered_comparison'] == 'failed'
    assert summary['registered_predicates']['added_mass_block'] == 'passed'
    assert summary['registered_predicates']['radiation_damping_block'] == 'passed'
    assert summary['registered_predicates']['mean_power'] == 'failed'
    assert summary['qualified'] is False
    assert summary['source_admission']['general_numerical_admission'] is False


def test_aggregate_unresolved_refuses_promotion(route, tmp_path, monkeypatch):
    # Candidate mean power placed exactly at the registered limit: inside the
    # printed-bound band, so the aggregate is UNRESOLVED, never a promotion.
    band = (cea.ABSOLUTE + cea.RELATIVE*cea.HAMS_MEAN_POWER_W)/(1.0 - cea.RELATIVE)
    inject_stored_candidate(monkeypatch, cea.HAMS_MEAN_POWER_W + band)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 1
    summary = json.loads((output/'summary.json').read_text())
    assert summary['status'] == 'energy_point_comparison_unresolved_unqualified'
    assert summary['registered_comparison'] == 'unresolved'
    assert summary['registered_predicates']['mean_power'] == 'unresolved'
    assert all(v == 'passed' for key, v in summary['registered_predicates'].items()
               if key != 'mean_power')
    assert summary['qualified'] is False
    assert summary['source_admission']['state'] == 'not_admitted'


@pytest.mark.parametrize('bad_power', [float('nan'), float('inf'), float('-inf')])
def test_nonfinite_mean_power_refused_raw_preserved(route, tmp_path, monkeypatch,
                                                    bad_power):
    inject_stored_candidate(monkeypatch, bad_power)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'not finite' in failure(output)['error']
    assert failure(output)['stage'] == 'retained_evidence'
    assert (output/'retained-blocks.npz').is_file()
    assert (output/'case'/'status.json').is_file()
    assert not (output/'directional-evidence.json').exists()
    assert not (output/'energy-comparison.json').exists()
    assert not (output/'summary.json').exists()


def test_admission_helper_pin_missing_or_mutated_refused_before_consume(route,
                                                                        tmp_path):
    for case, bad in (('missing', None), ('mutated', '0'*64),
                      ('malformed', 'not-a-sha256')):
        admission = dict(route['admission'])
        if bad is None:
            admission.pop('helper_module_sha256')
        else:
            admission['helper_module_sha256'] = bad
        rewrite(route['admission_path'], admission)
        output = tmp_path/f'out-{case}'
        rc = run_check(route, output)
        assert rc == 2 and 'helper_module_sha256' in failure(output)['error'], case
        assert route['admission_path'].exists()
        assert RECORD['constructor'] == 0


def test_mutated_helper_module_refused_before_consume(route, tmp_path, monkeypatch):
    """The review's counterexample: the helper mutated between admission
    issuance and execution. The admission pin is correct; the loaded helper
    bytes no longer match the checker's pinned helper identity, so the
    production route fails closed before consumption or any constructor."""
    monkeypatch.setattr(cea, 'HELPER_MODULE_SHA256', '1'*64)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'pinned hash-bound helper' in failure(output)['error']
    assert route['admission_path'].exists()
    assert RECORD['constructor'] == 0


def test_criteria_format_mismatch_refused_before_consume(route, tmp_path):
    route['criteria']['format'] = 'wrong'
    rewrite(route['criteria_path'], route['criteria'])
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'criteria format' in failure(output)['error']
    assert route['admission_path'].exists()


def test_criteria_native_future_mismatch_refused(route, tmp_path):
    route['criteria']['native_future']['green_evaluations'] = 4
    rewrite(route['criteria_path'], route['criteria'])
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'native_future budgets' in failure(output)['error']
    assert route['admission_path'].exists()


def test_registration_controls_mismatch_refused(route, tmp_path):
    registration = json.loads(route['registration_path'].read_text())
    registration['gf_evaluations'] = 4
    rewrite(route['registration_path'], registration)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'registration' in failure(output)['error']
    assert route['admission_path'].exists()


def test_intent_hams_power_mismatch_refused(route, tmp_path):
    intent = hams_intent()
    intent['saved_hams_mean_power_w'] = 0.5
    rewrite(route['intent_path'], intent)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'intent' in failure(output)['error']
    assert route['admission_path'].exists()


def test_controls_unit_conversion_mismatch_refused(route, tmp_path):
    controls = hams_controls()
    controls['input_manifest']['rho_target'] = 1000.0
    rewrite(route['controls_path'], controls)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'physical units' in failure(output)['error']
    assert route['admission_path'].exists()


def test_mutated_hams_report_file_refused(route, tmp_path):
    report = hams_report()
    report['hams_mean_power_w'] = 0.0
    rewrite(route['report_path'], report)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'mean power' in failure(output)['error']
    assert route['admission_path'].exists()
    report = hams_report()
    report['byte_mutation_marker'] = 'content-valid but byte-different'
    rewrite(route['report_path'], report)
    output = tmp_path/'out2'
    rc = run_check(route, output)
    assert rc == 2 and 'hams_report_sha256' in failure(output)['error']
    assert route['admission_path'].exists()


def test_mutated_reference_input_bytes_refused(route, tmp_path):
    with open(route['root']/'input.npz', 'ab') as handle:
        handle.write(b'mutated')
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'byte hashes' in failure(output)['error']
    assert route['admission_path'].exists()


def test_admission_mode_must_be_energy(route, tmp_path):
    for mode in (None, 'probe', 'full_matrix'):
        admission = dict(route['admission'])
        if mode is None:
            admission.pop('mode')
        else:
            admission['mode'] = mode
        rewrite(route['admission_path'], admission)
        output = tmp_path/'out'
        rc = run_check(route, output)
        assert rc == 2 and 'energy mode' in failure(output)['error'], mode
        assert route['admission_path'].exists()


def test_probe_admission_format_refused(route, tmp_path):
    admission = dict(route['admission'])
    admission['format'] = 'anyhydro-operator-artifact-admission-v1'
    rewrite(route['admission_path'], admission)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'admission format' in failure(output)['error']


def test_preconsumed_admission_refused(route, tmp_path):
    route['admission_path'].with_name(
        route['admission_path'].name+'.consumed').write_text('spent')
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'already consumed' in failure(output)['error']


def test_admission_script_hash_mismatch_refused(route, tmp_path):
    route['admission']['script_sha256'] = '0'*64
    rewrite(route['admission_path'], route['admission'])
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'script_sha256' in failure(output)['error']
    assert route['admission_path'].exists()


def test_admission_run_script_hash_mismatch_refused(route, tmp_path):
    route['admission']['run_script_sha256'] = '0'*64
    rewrite(route['admission_path'], route['admission'])
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'run_script_sha256' in failure(output)['error']
    assert route['admission_path'].exists()


def test_installed_module_hash_mismatch_refused(route, tmp_path):
    route['admission']['installed_anyhydro_module_sha256'][
        'panel_integration_candidate.py'] = '0'*64
    rewrite(route['admission_path'], route['admission'])
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'installed ANYhydro module' in failure(output)['error']
    assert route['admission_path'].exists()


def test_evaluator_source_map_mismatch_refused(route, tmp_path):
    route['admission']['evaluator_backend_source_sha256'] = {
        name: '0'*64 for name in route['evaluator_sources']}
    rewrite(route['admission_path'], route['admission'])
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'evaluator source map' in failure(output)['error']
    assert route['admission_path'].exists()


def test_backend_source_mutation_refused_after_consume(route, tmp_path):
    (route['package']/'green_functions/delhommeau.py').write_bytes(b'mutated evaluator')
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'evaluator backend sources' in failure(output)['error']
    assert not route['admission_path'].exists()


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


def test_existing_output_refused(route, tmp_path):
    output = tmp_path/'out'
    output.mkdir()
    (output/'prior.txt').write_text('prior evidence')
    rc = run_check(route, output)
    assert rc == 2
    assert (output/'prior.txt').read_text() == 'prior evidence'
    assert not (output/'failure.json').exists()


def test_body_count_not_two_refused(route, tmp_path, monkeypatch):
    monkeypatch.setattr(anyhydro, 'make_first_slice_bodies', lambda **kw: [object()]*3)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'two-body' in failure(output)['error']


def test_rotation_center_mutation_refused(route, tmp_path, monkeypatch):
    original = anyhydro.make_first_slice_bodies
    def replacement(**kw):
        bodies = list(original(**kw))
        bodies[0] = dataclasses.replace(bodies[0], rotation_center=(0.0, 0.0, 0.0))
        return bodies
    monkeypatch.setattr(anyhydro, 'make_first_slice_bodies', replacement)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'rotation centers' in failure(output)['error']


def test_dof_name_mutation_refused_no_completed_promotion(route, tmp_path, monkeypatch):
    original = HydroBody.all_dof_keys
    def replacement(self):
        keys = list(original(self))
        keys[0], keys[1] = keys[1], keys[0]  # valid unique inventory, wrong axis order
        return tuple(keys)
    monkeypatch.setattr(HydroBody, 'all_dof_keys', replacement)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'candidate DOF at axis 0' in failure(output)['error']
    assert not (output/'summary.json').exists()
    status = json.loads((output/'case'/'status.json').read_text())
    assert status['state'] == 'completed_unqualified'


def test_extra_override_budget_refused(route, tmp_path, monkeypatch):
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


def test_nonfinite_batch_preserves_raw_evidence(route, tmp_path, monkeypatch):
    green = route['green']
    previous = green.evaluate
    def evaluate(self, receivers, source, **kwargs):
        s, grad = previous(self, receivers, source, **kwargs)
        if len(RECORD['wave']) == 2:  # first 864x864 operator batch
            s = s.copy()
            s[0, 0] = np.inf
        return s, grad
    monkeypatch.setattr(green, 'evaluate', evaluate)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'nonfinite' in failure(output)['error']
    assert (output/'case'/'native-operator-0.npz').is_file()
    status = json.loads((output/'case'/'status.json').read_text())
    assert status['state'] == 'failed' and status['numerical_systems'] == 0
    assert not (output/'summary.json').exists()


def test_cancellation_preserves_evidence_without_promotion(route, tmp_path, monkeypatch):
    calls = {'n': 0}
    def cancelled(callback):
        calls['n'] += 1
        if calls['n'] > 1:
            raise PanelCandidateCancelled('panel integration candidate cancelled')
    monkeypatch.setattr('anyhydro.panel_integration_candidate.check_cancel', cancelled)
    output = tmp_path/'out'
    rc = run_check(route, output)
    assert rc == 2 and 'cancelled' in failure(output)['error']
    status = json.loads((output/'case'/'status.json').read_text())
    assert status['state'] == 'cancelled'
    assert not (output/'summary.json').exists()
    assert not (output/'case'/'radiation.npz').exists()


def test_guard_refuses_second_public_numpy_solve():
    """Tiny NumPy solve guard unit test; never the original BEM solver case."""
    import numpy
    code, depth = helper._pinned_code('numpy_linalg_solve', numpy.linalg.solve)
    guard = helper.BudgetGuard({code: 'numpy_linalg_solve'},
                               {'numpy_linalg_solve': 1})
    guard.install()
    try:
        matrix = np.eye(2)
        rhs = np.ones(2)
        guard.counts['numpy_linalg_solve'] = 1
        with pytest.raises(helper.Refused, match='exhausted'):
            numpy.linalg.solve(matrix, rhs)
    finally:
        guard.uninstall()
    assert depth == 0 or isinstance(depth, int)


# ---- pure validation and classifier functions (independently testable) ----

def minimal_criteria():
    return dict(format=cea.CRITERIA_FORMAT, wheel_sha256='1'*64, f64_sha256='2'*64,
                evaluator_backend_source_sha256={name: '3'*64 for name in cea.BACKEND_FILES},
                source_panels=864, receiver_nodes=3456, body_count=2,
                quadrature=cea.QUADRATURE_METHOD, omega=1.5, water_depth=150.0,
                rho=1025.0, gravity=9.81,
                rotation_centers=[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                retained_dofs=list(cea.RETAINED_DOFS),
                retained_indices=[0, 2, 6, 8], retained_rate=list(cea.RETAINED_RATE),
                comparison=dict(relative=0.03, absolute=1e-8, near_zero_floor=1e-8,
                                precision=cea.COMPARISON_PRECISION,
                                candidate_print_bound=0.0),
                mean_power=dict(nonnegative_absolute_w=1e-8,
                                precision=cea.MEAN_POWER_PRECISION,
                                hams_mean_power_w=cea.HAMS_MEAN_POWER_W,
                                hams_print_bound_w=cea.HAMS_PRINT_BOUND_W),
                hams_controls_sha256=cea.HAMS_CONTROLS_SHA256,
                hams_report_sha256=cea.HAMS_REPORT_SHA256,
                registration_sha256='4'*64, intent_sha256='5'*64,
                run_script_sha256='6'*64,
                fixed_coefficients_sha256=cea.FIXED_COEFFICIENTS_SHA256,
                native_worker_execution=False,
                negative_power_rejects_source_admission=True,
                native_future=dict(cea.EXPECTED_NATIVE_FUTURE))


def test_validate_criteria_accepts_minimal_contract(tmp_path):
    reference = tmp_path/'bundle'
    reference.mkdir()
    for name in cea.BUNDLE_FILES:
        (reference/name).write_bytes(b'x')
    criteria = minimal_criteria()
    criteria['original_reference'] = str(reference)
    sources, resolved = cea.validate_criteria(criteria)
    assert set(sources) == set(cea.BACKEND_FILES)
    assert resolved == reference


@pytest.mark.parametrize('mutation,match', [
    ('format', 'criteria format'),
    ('rate', 'retained DOF/rate'),
    ('policy', 'comparison policy'),
    ('future', 'native_future'),
    ('hams', 'frozen HAMS'),
    ('negative', 'negative power'),
])
def test_validate_criteria_refuses_mutations(tmp_path, mutation, match):
    reference = tmp_path/'bundle'
    reference.mkdir()
    for name in cea.BUNDLE_FILES:
        (reference/name).write_bytes(b'x')
    criteria = minimal_criteria()
    criteria['original_reference'] = str(reference)
    if mutation == 'format':
        criteria['format'] = 'other'
    elif mutation == 'rate':
        criteria['retained_rate'] = [0.0]*4
    elif mutation == 'policy':
        criteria['comparison']['relative'] = 0.05
    elif mutation == 'future':
        criteria['native_future']['systems'] = 2
    elif mutation == 'hams':
        criteria['hams_report_sha256'] = '7'*64
    elif mutation == 'negative':
        criteria['negative_power_rejects_source_admission'] = False
    with pytest.raises(cea.Refused, match=match):
        cea.validate_criteria(criteria)


def test_validate_hams_report_requires_full_coefficient_packet():
    blocks = cea.validate_hams_report(hams_report())
    assert blocks['added_mass']['hams_retained'] == HAMS_ADDED_MASS
    assert blocks['radiation_damping']['print_bound'] == HAMS_DAMP_BOUND
    incomplete = hams_report()
    del incomplete['comparisons']['added_mass']['hams_retained']
    with pytest.raises(cea.Refused, match='finite 4x4'):
        cea.validate_hams_report(incomplete)
    wrong_power = hams_report()
    wrong_power['hams_mean_power_w'] = 0.0
    with pytest.raises(cea.Refused, match='mean power'):
        cea.validate_hams_report(wrong_power)


def test_compare_block_matching_basis_passes():
    reference = [[100.0, 1.0, 2.0, 3.0], [1.0, 200.0, 4.0, 5.0],
                 [2.0, 4.0, 300.0, 6.0], [3.0, 5.0, 6.0, 400.0]]
    zero = [[0.0]*4]*4
    result = cea.compare_block([row[:] for row in reference], reference, zero)
    assert result['verdict'] == 'passed' and result['counts']['passed'] == 16


def test_compare_block_mismatching_basis_fails():
    reference = [[100.0, 1.0, 2.0, 3.0], [9.0, 200.0, 4.0, 5.0],
                 [2.0, 4.0, 300.0, 6.0], [7.0, 5.0, 6.0, 400.0]]
    candidate = [[row[i] for row in reference] for i in range(4)]  # transposed basis
    zero = [[0.0]*4]*4
    result = cea.compare_block(candidate, reference, zero)
    assert result['verdict'] == 'failed'
    assert result['counts']['failed'] == 4  # only the asymmetric entries differ


def test_compare_block_printed_bound_rules():
    reference = [[100.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    near = [[101.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    far = [[110.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    passing = cea.compare_block(near, reference, [[2.0]+[0.0]*3]+[[0.0]*4]*3)
    assert passing['coefficients'][0]['verdict'] == 'pass'
    unresolved = cea.compare_block(near, reference, [[2.5]+[0.0]*3]+[[0.0]*4]*3)
    assert unresolved['coefficients'][0]['verdict'] == 'unresolved'
    failing = cea.compare_block(far, reference, [[2.0]+[0.0]*3]+[[0.0]*4]*3)
    assert failing['coefficients'][0]['verdict'] == 'fail'


def test_compare_block_near_zero_floor():
    zero = [[0.0]*4]*4
    tiny = [[5e-9, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    result = cea.compare_block(tiny, zero, zero)
    assert result['coefficients'][0]['scale'] == cea.NEAR_ZERO_FLOOR
    assert result['coefficients'][0]['verdict'] == 'pass'


def test_compare_mean_power_and_sign_classifier():
    matching = cea.compare_mean_power(cea.HAMS_MEAN_POWER_W, cea.HAMS_MEAN_POWER_W,
                                      cea.HAMS_PRINT_BOUND_W)
    assert matching['verdict'] == 'pass'
    assert matching['candidate_print_bound'] == 0.0
    historical = cea.compare_mean_power(-0.007632760362971955, cea.HAMS_MEAN_POWER_W,
                                        cea.HAMS_PRINT_BOUND_W)
    assert historical['verdict'] == 'fail'
    assert cea.classify_sign(2e-8, 0.0) == 'positive'
    assert cea.classify_sign(1e-8, 0.0) == 'unresolved'
    assert cea.classify_sign(-2e-8, 0.0) == 'negative'
    assert cea.classify_sign(5e-8, 4e-8) == 'unresolved'
    assert cea.classify_sign(cea.HAMS_MEAN_POWER_W, cea.HAMS_PRINT_BOUND_W) == 'positive'


def test_nonfinite_mean_power_refused_by_pure_comparators():
    for bad in (float('nan'), float('inf'), float('-inf')):
        with pytest.raises(cea.Refused, match='not finite'):
            cea.compare_mean_power(bad, cea.HAMS_MEAN_POWER_W, cea.HAMS_PRINT_BOUND_W)
        with pytest.raises(cea.Refused, match='finite'):
            cea.classify_sign(bad, 0.0)
    with pytest.raises(cea.Refused, match='not finite'):
        cea.compare_mean_power(1.0, float('nan'), 0.0)
    with pytest.raises(cea.Refused, match='not finite'):
        cea.compare_mean_power(1.0, 1.0, float('inf'))
    with pytest.raises(cea.Refused, match='finite'):
        cea.classify_sign(1.0, float('nan'))


def test_build_rate_maps_retained_axes_and_refuses_mutations():
    rate = cea.build_rate(list(AXES))
    assert rate == [cea.RETAINED_RATE[0], 0.0, cea.RETAINED_RATE[1], 0.0, 0.0, 0.0,
                     cea.RETAINED_RATE[2], 0.0, cea.RETAINED_RATE[3], 0.0, 0.0, 0.0]
    mutated = list(AXES)
    mutated[2] = 'first-slice-a::sway'
    with pytest.raises(cea.Refused, match='candidate DOF at axis 2'):
        cea.build_rate(mutated)
    with pytest.raises(cea.Refused, match='twelve-axis'):
        cea.build_rate(AXES[:6])
