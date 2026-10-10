# SPDX-License-Identifier: MPL-2.0
"""Production caller lifecycle/admission contracts with a mock, not BEM evidence."""
import dataclasses
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys
import types
import zipfile

import numpy as np
import pytest
from scipy.optimize import brentq

from anyhydro import make_first_slice_bodies
from anyhydro.errors import BackendSolveError, CapabilityError
from anyhydro.panel_accuracy_reference import (BACKEND_FILES, FORMAT, MODEL_ARRAYS, STEPS,
    FrozenUntabulatedProfile, FrozenTabulatedProfile, TABULATED_FORMAT,
    PanelCandidateCancelled, PanelOperatorProbe, case_binding, operator_probe)
from anyhydro.saved_tabulation_profile import (SAVED_TABULATED_FORMAT, FrozenSavedTabulatedProfile,
                                               public_cache_filename)
from test_panel_integration_candidate import call, fake_backend


def sha(data):
    return hashlib.sha256(data).hexdigest()


def seal(root, manifest):
    manifest['files'] = {n: sha((root/n).read_bytes()) for n in ('input.npz', 'reference.npz')}
    (root/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def build_bundle(tmp_path, monkeypatch):
    green = fake_backend(monkeypatch)
    package = tmp_path/'mock-capytaine';package.mkdir()
    init = package/'__init__.py';init.write_text('# mock')
    sys.modules['capytaine'].__file__ = str(init)
    for name in BACKEND_FILES:
        p = package/name;p.parent.mkdir(parents=True, exist_ok=True);p.write_text('# mock '+name)
    native = package/'native.mock';native.write_bytes(b'mock native implementation')
    green.fortran_core = types.SimpleNamespace(__file__=str(native))
    monkeypatch.setitem(sys.modules, 'capytaine.green_functions.Delhommeau_float64', green.fortran_core)
    call(tmp_path/'seed')
    green.calls = 0
    case = json.loads((tmp_path/'seed/candidate/case.json').read_text())
    with np.load(tmp_path/'seed/candidate/input.npz') as z:
        inputs = {n: z[n] for n in MODEL_ARRAYS}
    center = inputs['vertices'][inputs['faces'][0]].mean(axis=0).reshape(1, 3)
    center[0, :2] += [.37, .19];center[0, 2] -= .17
    receivers = np.concatenate([center]+[center+sign*h*np.eye(3)[d] for h in STEPS for sign in (1, -1) for d in range(3)])
    coeff = np.array([[-1., -2., -3., -4.], [.1, .2, .3, .4]])
    area = inputs['weights'][0].sum();direction = np.array([.12, .17, .07])
    potential = ((1+receivers@direction)*area).astype(complex)
    gradient = np.broadcast_to(direction*area, (43, 3)).astype(complex).copy()
    root = tmp_path/'bundle';root.mkdir()
    np.savez(root/'input.npz', **inputs, coefficients=coeff, receivers=receivers)
    np.savez(root/'reference.npz', potential_low=potential, potential_high=potential,
             gradient_low=gradient, gradient_high=gradient)
    k = brentq(lambda x: x*np.tanh(x*150)-1.5**2/9.81, 0., 1.5**2/9.81+1/150, xtol=1e-15)
    manifest = dict(format=FORMAT, case_binding=case_binding(case), orders=[64, 128], steps=STEPS,
                    provenance=dict(mock=True, qualified=False), k_hex=k.hex(), kh_hex=(k*150).hex(),
                    coefficient_identity=dict(shape=[2, 4], dtype='<f8', sha256=sha(coeff.tobytes())),
                    backend=dict(capytaine='3.0.0', native_library_sha256=sha(native.read_bytes()),
                                 source_sha256={n: sha((package/n).read_bytes()) for n in BACKEND_FILES}))
    seal(root, manifest)
    return root, manifest, green


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    return build_bundle(tmp_path, monkeypatch)


def test_terminal_probe_keeps_raw_arrays_and_never_solves(bundle, tmp_path, monkeypatch):
    root, _, green = bundle
    profile = FrozenUntabulatedProfile.load(root)
    monkeypatch.setattr(np.linalg, 'solve', lambda *a: pytest.fail('probe must never solve'))
    result = call(tmp_path/'run', accuracy_profile=profile, operator_probe_receivers=4)
    assert isinstance(result, PanelOperatorProbe) and not hasattr(result, 'added_mass')
    assert green.calls == 2 and result.receiver_count == 4 and result.source_panel_count == 24
    output = tmp_path/'run/candidate'
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'completed_probe_unqualified' and status['numerical_systems'] == 0
    assert not (output/'radiation.npz').exists() and not (output/'operator-0.npz').exists()
    with np.load(output/'native-operator-probe.npz') as z:
        np.testing.assert_array_equal(z['indices'], [[0, 0], [0, 1], [0, 2], [0, 3]])
        assert z['s'].shape == (4, 24) and z['gradient'].shape == (3, 4, 24)


def test_profile_full_caller_keeps_coupled_result_contract(bundle, tmp_path):
    root, _, green = bundle
    result = call(tmp_path/'run', accuracy_profile=FrozenUntabulatedProfile.load(root))
    assert result.force_per_velocity.shape == (12, 12) and green.calls == 5
    assert np.max(np.abs(result.mean_residual)) < 1e-12
    assert np.max(np.abs(result.node_residual)) > 1e-4


def test_full_case_mismatch_with_unchanged_micro_geometry(bundle, tmp_path):
    root, _, green = bundle
    bodies = list(make_first_slice_bodies(subdivisions=1, lid_policy='submerged_0p1m_v1'))
    bodies[1] = dataclasses.replace(bodies[1], rotation_center=(1., 2., -3.))
    with pytest.raises(CapabilityError, match='complete producing case'):
        call(tmp_path/'run', bodies, accuracy_profile=FrozenUntabulatedProfile.load(root), operator_probe_receivers=1)
    assert green.calls == 0 and not (tmp_path/'run/candidate').exists()


@pytest.mark.parametrize('count', [True, False, 0, 17, 1.0])
def test_probe_count_refusal(bundle, tmp_path, count):
    root, _, green = bundle
    with pytest.raises(CapabilityError, match='integer1..16'):
        call(tmp_path/'run', accuracy_profile=FrozenUntabulatedProfile.load(root), operator_probe_receivers=count)
    assert green.calls == 0


def test_probe_requires_profile(tmp_path):
    with pytest.raises(CapabilityError, match='profile'):
        call(tmp_path, operator_probe_receivers=1)
    with pytest.raises(CapabilityError, match='load'):
        FrozenUntabulatedProfile()


@pytest.mark.parametrize('after_calls', [1, 2])
def test_cancel_after_micro_or_probe_keeps_incomplete_evidence(bundle, tmp_path, after_calls):
    root, _, green = bundle
    with pytest.raises(PanelCandidateCancelled):
        call(tmp_path/'run', accuracy_profile=FrozenUntabulatedProfile.load(root), operator_probe_receivers=4,
             cancel_requested=lambda: green.calls >= after_calls)
    output = tmp_path/'run/candidate'
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'cancelled' and status['new_green_evaluations'] == after_calls
    assert status['numerical_systems'] == 0 and not (output/'radiation.npz').exists()
    assert (output/'accuracy-native-micro.npz').is_file()
    assert (output/'native-operator-probe.npz').exists() == (after_calls == 2)


def test_resealed_reference_still_must_converge(bundle):
    root, manifest, _ = bundle
    with np.load(root/'reference.npz') as z:
        values = {n: z[n] for n in z.files}
    values['gradient_high'][10, 2] += 1e-6
    np.savez(root/'reference.npz', **values);seal(root, manifest)
    with pytest.raises(CapabilityError, match='convergence'):
        FrozenUntabulatedProfile.load(root)


def test_resealed_inconsistent_potential_cannot_pass_original_step_guard(bundle, tmp_path):
    root, manifest, green = bundle
    with np.load(root/'reference.npz') as z:
        values = {n: z[n] for n in z.files}
    values['potential_low'][:] = values['potential_low'][0]
    values['potential_high'][:] = values['potential_high'][0]
    np.savez(root/'reference.npz', **values);seal(root, manifest)
    with pytest.raises(BackendSolveError, match='independent-reference'):
        call(tmp_path/'run', accuracy_profile=FrozenUntabulatedProfile.load(root), operator_probe_receivers=1)
    assert green.calls == 1
    audit = json.loads((tmp_path/'run/candidate/accuracy-comparison.json').read_text())
    assert not audit['reference_original_steps_passed'] and not audit['passed']


def test_oversized_npy_shape_refused_before_numpy_load(monkeypatch):
    from anyhydro.panel_accuracy_reference import _arrays
    member = BytesIO()
    np.lib.format.write_array_header_1_0(member, dict(descr='<f8', fortran_order=False, shape=(2**42,)))
    payload = BytesIO()
    with zipfile.ZipFile(payload, 'w') as z:z.writestr('huge.npy', member.getvalue())
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('allocation attempted before header admission'))
    with pytest.raises(CapabilityError, match='allocation/payload'):
        _arrays(payload.getvalue(), ('huge',))


def test_resealed_full_geometry_still_must_match_snapshot(bundle):
    root, manifest, _ = bundle
    with np.load(root/'input.npz') as z:
        values = {n: z[n] for n in z.files}
    values['vertices'][-1, 0] += .001
    np.savez(root/'input.npz', **values);seal(root, manifest)
    with pytest.raises(CapabilityError, match='full model snapshot'):
        FrozenUntabulatedProfile.load(root)


def test_wrong_native_identity_refused_before_any_call(bundle, tmp_path):
    root, manifest, green = bundle
    manifest['backend']['native_library_sha256'] = '0'*64;seal(root, manifest)
    with pytest.raises(CapabilityError, match='implementation mismatch'):
        call(tmp_path/'run', accuracy_profile=FrozenUntabulatedProfile.load(root), operator_probe_receivers=1)
    assert green.calls == 0
    status = json.loads((tmp_path/'run/candidate/status.json').read_text())
    assert status['state'] == 'failed' and status['numerical_systems'] == 0


def test_frozen_arrays_and_decomposition_refuse_mutation_or_fallback(bundle):
    root, _, green = bundle
    profile = FrozenUntabulatedProfile.load(root)
    for array in (profile.coefficients, profile.receivers, profile.gradient, profile.potential):
        with pytest.raises(ValueError):array.setflags(write=True)
    gf = profile.green_function(green, sys.modules['capytaine'], k=float.fromhex(json.loads(profile.manifest_json)['k_hex']), water_depth=150.)
    kh = float.fromhex(json.loads(profile.manifest_json)['kh_hex'])
    assert gf.find_best_exponential_decomposition(kh) is profile.coefficients
    for value, method in [(kh+1, None), (kh, 'fortran'), (kh, ''), (kh, False)]:
        with pytest.raises(CapabilityError):gf.find_best_exponential_decomposition(value, method)


def test_load_snapshots_bundle_before_external_files_change(bundle, tmp_path):
    root, _, _ = bundle
    profile = FrozenUntabulatedProfile.load(root)
    saved = profile.input_bytes
    (root/'input.npz').write_bytes(b'changed after loading')
    call(tmp_path/'run', accuracy_profile=profile, operator_probe_receivers=1)
    assert (tmp_path/'run/candidate/accuracy-input.npz').read_bytes() == saved


def test_failed_reference_guard_stops_probe_with_raw_micro(bundle, tmp_path, monkeypatch):
    root, _, green = bundle
    evaluate = green.evaluate
    def bad(self, *args, **kwargs):
        s, grad = evaluate(self, *args, **kwargs)
        return s, grad+1
    monkeypatch.setattr(green, 'evaluate', bad)
    with pytest.raises(BackendSolveError, match='independent-reference'):
        call(tmp_path/'run', accuracy_profile=FrozenUntabulatedProfile.load(root), operator_probe_receivers=1)
    assert green.calls == 1
    output = tmp_path/'run/candidate'
    assert (output/'accuracy-native-micro.npz').is_file()
    assert not (output/'native-operator-probe.npz').exists()


def build_table_backend(bundle, monkeypatch):
    root, manifest, green = bundle
    initialize = green.__init__
    green.table_constructors = 0
    def create(self, **kwargs):
        initialize(self, **kwargs)
        self.tabulation_cache_dir = kwargs.get('tabulation_cache_dir')
        if kwargs.get('tabulation_nr', 0):
            green.table_constructors += 1
            self.tabulated_r_range = np.linspace(1e-10, 100., kwargs['tabulation_nr'])
            self.tabulated_z_range = -np.geomspace(1e-9, 251., kwargs['tabulation_nz'])
            self.tabulated_integrals = np.zeros((kwargs['tabulation_nr'], kwargs['tabulation_nz'], 5))
    monkeypatch.setattr(green, '__init__', create)
    return bundle


@pytest.fixture
def table_backend(bundle, monkeypatch):
    return build_table_backend(bundle, monkeypatch)


def test_distinct_tabulated_identity(table_backend):
    root, _, green = table_backend
    untabulated = FrozenUntabulatedProfile.load(root)
    tabulated = FrozenTabulatedProfile.load(root)
    first = json.loads(untabulated.manifest_json);second = json.loads(tabulated.manifest_json)
    assert first['format'] == FORMAT and second['format'] == TABULATED_FORMAT
    assert second['reference_manifest_sha256'] == sha(untabulated.manifest_json.encode())
    assert second['constructor_settings']['tabulation_nr'] == 1352
    assert second['constructor_settings']['tabulation_nz'] == 744
    assert 'constructor_settings' not in first and green.table_constructors == 0
    np.testing.assert_array_equal(untabulated.coefficients, tabulated.coefficients)
    with pytest.raises(ValueError):tabulated.coefficients.setflags(write=True)
    with pytest.raises(dataclasses.FrozenInstanceError):tabulated.constructor_options = lambda: {}


def test_tabulated_probe_binds_and_preserves_actual_tables(table_backend, tmp_path):
    root, _, green = table_backend
    result = call(tmp_path/'run', accuracy_profile=FrozenTabulatedProfile.load(root), operator_probe_receivers=4)
    assert isinstance(result, PanelOperatorProbe) and green.calls == 2 and green.table_constructors == 1
    output = tmp_path/'run/candidate'
    case = json.loads((output/'case.json').read_text())
    assert case['actual_tabulation']['format'] == TABULATED_FORMAT
    assert case['planned_case_hash'] != case['case_hash'] == result.case_hash
    actual = case['actual_tabulation']['actual_tables']
    with np.load(output/'frozen-tabulation.npz') as z:
        assert z['tabulated_integrals'].shape == (1352, 744, 5)
        for name in z.files:assert sha(z[name].tobytes()) == actual[name]['sha256']
    for phase in ('before_micro', 'after_micro', 'before_probe', 'after_probe'):
        record = json.loads((output/('tabulation-hashes-'+phase+'.json')).read_text())
        assert record['passed'] and record['immutable'] and record['actual_tables'] == actual
    status = json.loads((output/'status.json').read_text())
    assert status['new_table_constructions'] == 1 and status['numerical_systems'] == 0
    assert status['table_constructor_wall_seconds'] >= 0 and status['table_serialization_wall_seconds'] > 0
    assert not (output/'radiation.npz').exists()


def test_table_rebinding_during_acquisition_stops_probe(table_backend, tmp_path, monkeypatch):
    root, _, green = table_backend
    evaluate = green.evaluate
    def corrupt(self, *args, **kwargs):
        result = evaluate(self, *args, **kwargs)
        self.tabulated_integrals = self.tabulated_integrals.copy()
        return result
    monkeypatch.setattr(green, 'evaluate', corrupt)
    with pytest.raises(BackendSolveError, match='table identity or immutability'):
        call(tmp_path/'run', accuracy_profile=FrozenTabulatedProfile.load(root), operator_probe_receivers=1)
    assert green.calls == 1 and green.table_constructors == 1
    output = tmp_path/'run/candidate'
    assert (output/'accuracy-native-micro.npz').is_file()
    assert not (output/'native-operator-probe.npz').exists()
    assert not json.loads((output/'tabulation-hashes-after_micro.json').read_text())['passed']


def test_construction_failure_retains_attempt_and_zero_green_calls(table_backend, tmp_path, monkeypatch):
    root, _, green = table_backend
    def fail(self, **kwargs):raise BackendSolveError('mock table construction failed')
    monkeypatch.setattr(green, '__init__', fail)
    with pytest.raises(BackendSolveError, match='construction failed'):
        call(tmp_path/'run', accuracy_profile=FrozenTabulatedProfile.load(root), operator_probe_receivers=1)
    output = tmp_path/'run/candidate'
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'failed' and status['table_construction_attempts'] == 1
    assert status['new_green_evaluations'] == 0 and status['numerical_systems'] == 0 and green.calls == 0
    assert status['evaluator_preparation_wall_seconds'] > 0
    assert (output/'accuracy-profile.json').is_file() and not (output/'frozen-tabulation.npz').exists()


def test_cancellation_after_table_preservation_stops_before_micro(table_backend, tmp_path):
    root, _, green = table_backend
    output = tmp_path/'run/candidate'
    with pytest.raises(PanelCandidateCancelled):
        call(tmp_path/'run', accuracy_profile=FrozenTabulatedProfile.load(root), operator_probe_receivers=4,
             cancel_requested=lambda: (output/'tabulation-receipt.json').exists())
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'cancelled' and status['new_green_evaluations'] == 0
    assert green.calls == 0 and green.table_constructors == 1 and (output/'frozen-tabulation.npz').is_file()


def test_unnamed_profile_subclass_refused_before_backend(bundle, tmp_path):
    root, _, green = bundle
    class Unnamed(FrozenUntabulatedProfile):pass
    with pytest.raises(CapabilityError, match='named frozen'):
        call(tmp_path/'run', accuracy_profile=Unnamed.load(root), operator_probe_receivers=1)
    assert green.calls == 0 and not (tmp_path/'run/candidate').exists()


def test_table_asymptotic_inventory_discloses_out_of_table_points(table_backend, tmp_path):
    root, _, green = table_backend
    profile = FrozenTabulatedProfile.load(root)
    k = float.fromhex(json.loads(profile.manifest_json)['k_hex'])
    gf = profile.green_function(green, sys.modules['capytaine'], k=k, water_depth=150., artifact_directory=tmp_path, runtime={})
    points = np.array([[[0., 0., -1.], [1., 0., -1.], [0., 1., -1.], [1., 1., -1.]]])
    mesh = types.SimpleNamespace(quadrature_points=(points, np.ones((1, 4))))
    profile.record_branches(gf, np.array([[1e6, 0., -1.]]), mesh,
                            dict(wavenumber=k, water_depth=150.), tmp_path, 'synthetic')
    inventory = json.loads((tmp_path/'tabulation-branches-synthetic.json').read_text())
    assert not inventory['instrumented_native_trace']
    assert all(v['asymptotic_predicate_rows'] == 4 and v['table_predicate_rows'] == 0 for v in inventory['counts'].values())
    assert green.calls == 0


def test_wrong_native_identity_consumes_no_table_construction(table_backend, tmp_path):
    root, manifest, green = table_backend
    manifest['backend']['native_library_sha256'] = '0'*64;seal(root, manifest)
    with pytest.raises(CapabilityError, match='implementation mismatch'):
        call(tmp_path/'run', accuracy_profile=FrozenTabulatedProfile.load(root), operator_probe_receivers=1)
    assert green.calls == 0 and green.table_constructors == 0
    assert not (tmp_path/'run/candidate/acquired-tabulation.npz').exists()


@pytest.mark.parametrize('failure', ['nonfinite', 'shape', 'range', 'settings'])
def test_failed_generated_table_preserved_before_refusal(table_backend, tmp_path, monkeypatch, failure):
    root, _, green = table_backend
    initialize = green.__init__
    def invalid(self, **kwargs):
        initialize(self, **kwargs)
        if failure == 'nonfinite':self.tabulated_integrals[0, 0, 0] = np.nan
        elif failure == 'shape':self.tabulated_integrals = self.tabulated_integrals[:-1]
        elif failure == 'range':self.tabulated_r_range[-1] = 200.
        else:self.exportable_settings['tabulation_nr'] = 10
    monkeypatch.setattr(green, '__init__', invalid)
    with pytest.raises(CapabilityError):
        call(tmp_path/'run', accuracy_profile=FrozenTabulatedProfile.load(root), operator_probe_receivers=1)
    output = tmp_path/'run/candidate'
    assert green.table_constructors == 1 and green.calls == 0
    with np.load(output/'acquired-tabulation.npz') as z:
        if failure == 'nonfinite':assert np.isnan(z['tabulated_integrals'][0, 0, 0])
        elif failure == 'shape':assert z['tabulated_integrals'].shape[0] == 1351
        elif failure == 'range':assert z['tabulated_r_range'][-1] == 200.
    status = json.loads((output/'status.json').read_text())
    assert status['new_table_constructions'] == 1 and status['new_green_evaluations'] == 0
    assert (output/'acquired-tabulation.json').is_file() and not (output/'frozen-tabulation.npz').exists()


def test_oversized_probe_count_refused_before_native(table_backend, tmp_path):
    root, _, green = table_backend
    profile = FrozenTabulatedProfile.load(root)
    k = float.fromhex(json.loads(profile.manifest_json)['k_hex'])
    gf = profile.green_function(green, sys.modules['capytaine'], k=k, water_depth=150.,
                                artifact_directory=tmp_path, runtime={})
    points = np.array([[[0., 0., -1.], [1., 0., -1.], [0., 1., -1.], [1., 1., -1.]]])
    mesh = types.SimpleNamespace(quadrature_points=(points, np.ones((1, 4))), nb_faces=1)
    for count in (5, 0, -1, True, 2.0):
        with pytest.raises(CapabilityError, match='flattened node inventory'):
            operator_probe(gf, mesh, points, dict(wavenumber=k, water_depth=150.), count,
                           tmp_path, dict(new_green_evaluations=0), 'case', None, profile)
    assert green.calls == 0 and not (tmp_path/'native-operator-probe.npz').exists()


def test_tabulated_micro_guard_failure_preserves_raw_micro(table_backend, tmp_path, monkeypatch):
    root, _, green = table_backend
    evaluate = green.evaluate
    def bad(self, *args, **kwargs):
        s, grad = evaluate(self, *args, **kwargs)
        return s, grad+1
    monkeypatch.setattr(green, 'evaluate', bad)
    with pytest.raises(BackendSolveError, match='independent-reference'):
        call(tmp_path/'run', accuracy_profile=FrozenTabulatedProfile.load(root), operator_probe_receivers=1)
    assert green.calls == 1 and green.table_constructors == 1
    output = tmp_path/'run'/'candidate'
    assert (output/'accuracy-native-micro.npz').is_file()
    assert not (output/'native-operator-probe.npz').exists()
    assert json.loads((output/'tabulation-hashes-after_micro.json').read_text())['passed']
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'failed' and status['numerical_systems'] == 0


def test_after_probe_table_mutation_refusal_preserves_probe(table_backend, tmp_path, monkeypatch):
    root, _, green = table_backend
    evaluate = green.evaluate
    def corrupt(self, *args, **kwargs):
        result = evaluate(self, *args, **kwargs)
        if green.calls == 2:
            self.tabulated_integrals = self.tabulated_integrals.copy()
        return result
    monkeypatch.setattr(green, 'evaluate', corrupt)
    with pytest.raises(BackendSolveError, match='table identity or immutability'):
        call(tmp_path/'run', accuracy_profile=FrozenTabulatedProfile.load(root), operator_probe_receivers=1)
    assert green.calls == 2 and green.table_constructors == 1
    output = tmp_path/'run'/'candidate'
    assert (output/'native-operator-probe.npz').is_file()
    assert (output/'accuracy-native-micro.npz').is_file()
    assert not json.loads((output/'tabulation-hashes-after_probe.json').read_text())['passed']
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'failed' and status['numerical_systems'] == 0


@pytest.fixture
def saved_cache_backend(bundle, monkeypatch):
    root, manifest, green = bundle
    initialize = green.__init__
    green.table_constructors = 0
    green.cache_loads = 0
    def create(self, **kwargs):
        initialize(self, **kwargs)
        self.tabulation_cache_dir = kwargs.get('tabulation_cache_dir')
        if kwargs.get('tabulation_nr', 0):
            cache_dir = kwargs.get('tabulation_cache_dir')
            path = Path(cache_dir)/public_cache_filename(kwargs) if cache_dir is not None else None
            if path is not None and path.exists():
                with np.load(path, allow_pickle=False) as z:
                    self.tabulated_r_range = z['r_range']
                    self.tabulated_z_range = z['z_range']
                    self.tabulated_integrals = z['values']
                green.cache_loads += 1
            else:
                green.table_constructors += 1
                self.tabulated_r_range = np.linspace(1e-10, 100., kwargs['tabulation_nr'])
                self.tabulated_z_range = -np.geomspace(1e-9, 251., kwargs['tabulation_nz'])
                self.tabulated_integrals = np.zeros((kwargs['tabulation_nr'], kwargs['tabulation_nz'], 5))
    monkeypatch.setattr(green, '__init__', create)
    return bundle


@pytest.fixture
def saved_tables(saved_cache_backend, tmp_path):
    root, manifest, green = saved_cache_backend
    call(tmp_path/'producer', accuracy_profile=FrozenTabulatedProfile.load(root), operator_probe_receivers=2)
    green.calls = 0
    return root, tmp_path/'producer'/'candidate', green


def test_saved_tabulated_profile_reuses_cache_without_new_construction(saved_tables, tmp_path):
    root, producer, green = saved_tables
    profile = FrozenSavedTabulatedProfile.load(root, producer)
    result = call(tmp_path/'run', accuracy_profile=profile, operator_probe_receivers=2)
    assert isinstance(result, PanelOperatorProbe)
    assert green.cache_loads == 1 and green.table_constructors == 1 and green.calls == 2
    output = tmp_path/'run'/'candidate'
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'completed_probe_unqualified'
    assert status['saved_table_cache_attempts'] == 1 and status['saved_table_cache_loads'] == 1
    assert status['new_table_constructions'] == 0 and status['saved_table_cache_reuse_confirmed']
    assert status['numerical_systems'] == 0 and not (output/'radiation.npz').exists()
    cache = output/'saved-tabulation-cache'
    assert [p.name for p in cache.iterdir()] == [status['saved_table_cache_filename']]
    assert public_cache_filename(profile.constructor_options()) == status['saved_table_cache_filename']
    loaded = json.loads((output/'loaded-saved-tabulation.json').read_text())
    assert loaded['state'] == 'loaded_not_validated'
    assert loaded['external_concurrent_mutation_outside_trust_boundary']
    case = json.loads((output/'case.json').read_text())
    tabulation = case['actual_tabulation']
    assert tabulation['format'] == SAVED_TABULATED_FORMAT
    assert tabulation['cache_load_observed_via_public_constructor'] and not tabulation['native_internal_trace']
    assert tabulation['hash_match_alone_does_not_prove_absence_of_native_construction']
    producer_receipt = json.loads((producer/'tabulation-receipt.json').read_text())
    assert tabulation['saved_table_provenance']['producer_implementation_modules'] == producer_receipt['implementation_modules']
    assert tabulation['saved_table_provenance']['archive_sha256'] == producer_receipt['archive_sha256']
    assert tabulation['actual_tables'] == producer_receipt['actual_tables']
    assert case['planned_case_hash'] != case['case_hash'] == result.case_hash
    for phase in ('before_micro', 'after_micro', 'before_probe', 'after_probe'):
        assert json.loads((output/('tabulation-hashes-'+phase+'.json')).read_text())['passed']
    with np.load(output/'frozen-tabulation.npz') as z:
        assert z['tabulated_integrals'].shape == (1352, 744, 5)


def test_saved_profile_missing_archive_refused_before_backend(saved_tables, tmp_path):
    root, producer, green = saved_tables
    empty = tmp_path/'empty';empty.mkdir()
    with pytest.raises(CapabilityError, match='bounded regular files'):
        FrozenSavedTabulatedProfile.load(root, empty)
    assert green.calls == 0 and green.cache_loads == 0


def test_saved_profile_corrupt_archive_refused(saved_tables, tmp_path):
    root, producer, green = saved_tables
    (producer/'frozen-tabulation.npz').write_bytes(b'not an npz archive')
    with pytest.raises(CapabilityError):
        FrozenSavedTabulatedProfile.load(root, producer)
    assert green.calls == 0




def test_saved_profile_receipt_table_mismatch_refused(saved_tables, tmp_path):
    root, producer, green = saved_tables
    receipt = json.loads((producer/'tabulation-receipt.json').read_text())
    receipt['actual_tables']['tabulated_z_range']['sha256'] = '0'*64
    (producer/'tabulation-receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    with pytest.raises(CapabilityError, match='inconsistent'):
        FrozenSavedTabulatedProfile.load(root, producer)
    assert green.calls == 0


def test_saved_profile_model_mismatch_refused_before_backend(saved_tables, tmp_path):
    root, producer, green = saved_tables
    profile = FrozenSavedTabulatedProfile.load(root, producer)
    bodies = list(make_first_slice_bodies(subdivisions=1, lid_policy='submerged_0p1m_v1'))
    bodies[1] = dataclasses.replace(bodies[1], rotation_center=(1., 2., -3.))
    with pytest.raises(CapabilityError, match='complete producing case'):
        call(tmp_path/'run', bodies, accuracy_profile=profile, operator_probe_receivers=1)
    assert green.calls == 0 and green.cache_loads == 0 and not (tmp_path/'run').exists()


def test_saved_profile_wrong_backend_identity_consumes_no_cache(saved_tables, tmp_path):
    root, producer, green = saved_tables
    manifest = json.loads((producer/'accuracy-profile.json').read_text())
    manifest['backend']['native_library_sha256'] = '0'*64
    (producer/'accuracy-profile.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(CapabilityError, match='implementation mismatch'):
        call(tmp_path/'run', accuracy_profile=FrozenSavedTabulatedProfile.load(root, producer),
             operator_probe_receivers=1)
    assert green.calls == 0 and green.cache_loads == 0 and green.table_constructors == 1
    assert not (tmp_path/'run'/'candidate'/'saved-tabulation-cache').exists()


def test_saved_profile_loaded_mutation_refusal_preserves_probe(saved_tables, tmp_path, monkeypatch):
    root, producer, green = saved_tables
    evaluate = green.evaluate
    def corrupt(self, *args, **kwargs):
        result = evaluate(self, *args, **kwargs)
        if green.calls == 2:
            self.tabulated_integrals = self.tabulated_integrals.copy()
        return result
    monkeypatch.setattr(green, 'evaluate', corrupt)
    with pytest.raises(BackendSolveError, match='table identity or immutability'):
        call(tmp_path/'run', accuracy_profile=FrozenSavedTabulatedProfile.load(root, producer),
             operator_probe_receivers=1)
    assert green.calls == 2 and green.cache_loads == 1 and green.table_constructors == 1
    output = tmp_path/'run'/'candidate'
    assert (output/'native-operator-probe.npz').is_file()
    assert (output/'loaded-saved-tabulation.npz').is_file()
    assert not json.loads((output/'tabulation-hashes-after_probe.json').read_text())['passed']


def test_saved_profile_constructor_failure_preserves_prepared_cache(saved_tables, tmp_path, monkeypatch):
    root, producer, green = saved_tables
    def fail(self, **kwargs):raise BackendSolveError('mock constructor failed')
    monkeypatch.setattr(green, '__init__', fail)
    with pytest.raises(BackendSolveError, match='failed'):
        call(tmp_path/'run', accuracy_profile=FrozenSavedTabulatedProfile.load(root, producer),
             operator_probe_receivers=1)
    output = tmp_path/'run'/'candidate'
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'failed' and status['saved_table_cache_attempts'] == 1
    assert status['new_green_evaluations'] == 0 and status['numerical_systems'] == 0
    assert status['saved_table_cache_writes'] == 1
    assert (output/'saved-tabulation-cache').is_dir() and not (output/'loaded-saved-tabulation.npz').exists()
    assert not (output/'frozen-tabulation.npz').exists()


def test_saved_profile_cancellation_after_cache_preparation(saved_tables, tmp_path):
    root, producer, green = saved_tables
    output = tmp_path/'run'/'candidate'
    with pytest.raises(PanelCandidateCancelled):
        call(tmp_path/'run', accuracy_profile=FrozenSavedTabulatedProfile.load(root, producer),
             operator_probe_receivers=2,
             cancel_requested=lambda: (output/'loaded-saved-tabulation.json').exists())
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'cancelled' and status['saved_table_cache_loads'] == 1
    assert status['new_table_constructions'] == 0 and status['numerical_systems'] == 0
    assert green.calls == 0 and green.cache_loads == 1
    assert (output/'frozen-tabulation.npz').is_file() and not (output/'accuracy-native-micro.npz').exists()
