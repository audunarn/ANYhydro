"""Artifact-bound profile contracts with a mock backend; not BEM evidence.

The fixture binding models the verified real fact: the corrected E1 evaluator
differs from the original producer in ALL THREE backend source hashes, so the
producer map can never serve as the evaluator map. Native binding v2
intentionally also admits unchanged Python maps (all exactly pinned) with a
distinct pinned float64 library and guarded E1 interface; the real artifact
exercises the changed-source path, so no real Python map equals the producer.
"""
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from scipy.optimize import brentq

import anyhydro
from anyhydro import make_first_slice_bodies
from anyhydro.capytaine_backend import CapytaineExecutionLimits
from anyhydro.errors import BackendSolveError, CapabilityError
from anyhydro.panel_accuracy_reference import (BACKEND_FILES, FORMAT, MODEL_ARRAYS,
    STEPS, case_binding)
from anyhydro.panel_integration_candidate import run_panel_integration_candidate
from anyhydro.artifact_bound_panel_profile import (ARTIFACT_BOUND_PANEL_PROFILE_FORMAT,
    CONSTRUCTOR_OPTIONS, FIXED_COEFFICIENT_OVERRIDE_CODE, IMPLEMENTATION_MODULES,
    ArtifactBoundPanelProfile)
from test_panel_integration_candidate import call, fake_backend
from test_panel_accuracy_reference import sha, seal

SIDECAR_FORMAT = 'anyhydro-backend-artifact-identity-v1'
MODULE_FILES = {'artifact_bound_panel_profile.py': Path(anyhydro.artifact_bound_panel_profile.__file__),
                'panel_integration_candidate.py': Path(anyhydro.panel_integration_candidate.__file__),
                'panel_accuracy_reference.py': Path(anyhydro.panel_accuracy_reference.__file__)}


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'))


def build_artifact_case(tmp_path, monkeypatch, subdivisions=1):
    green = fake_backend(monkeypatch)
    package = tmp_path/'mock-capytaine';package.mkdir()
    init = package/'__init__.py';init.write_text('# mock')
    sys.modules['capytaine'].__file__ = str(init)
    for name in BACKEND_FILES:
        p = package/name;p.parent.mkdir(parents=True, exist_ok=True);p.write_bytes(('producer '+name).encode())
    native = package/'native.mock';native.write_bytes(b'producer native implementation')
    native32 = package/'native32.mock';native32.write_bytes(b'producer native32 implementation')
    green.fortran_core = types.SimpleNamespace(__file__=str(native))
    monkeypatch.setitem(sys.modules, 'capytaine.green_functions.Delhommeau_float64', green.fortran_core)
    monkeypatch.setitem(sys.modules, 'capytaine.green_functions.Delhommeau_float32',
                         types.SimpleNamespace(__file__=str(native32)))
    bodies = make_first_slice_bodies(subdivisions=subdivisions, lid_policy='submerged_0p1m_v1')
    run_panel_integration_candidate(bodies, omega=1.5, water_depth=150., rho=1025., gravity=9.81,
                                    limits=CapytaineExecutionLimits(max_total_panels=864,
                                                                    max_finite_omegas=1),
                                    artifact_directory=tmp_path/'seed'/'candidate')
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
    evaluator_sources = {n: sha(('evaluator '+n).encode()) for n in BACKEND_FILES}
    f64_bytes = b'evaluator f64 extension'
    f32_bytes = b'evaluator f32 extension'
    sidecar = dict(format=SIDECAR_FORMAT,
                   external_identity='capytaine-3.0.0-anyhydro-pi-gamma-e1-pre-v2/linux-x86_64',
                   wheel_sha256='4c'*32,
                   native_extension_sha256={
                       'capytaine/green_functions/Delhommeau_float32.cpython-312-x86_64-linux-gnu.so':
                           sha(b'evaluator f32 extension'),
                       'capytaine/green_functions/Delhommeau_float64.cpython-312-x86_64-linux-gnu.so':
                           sha(f64_bytes)},
                   admitted_for_numerical_execution=False)
    sidecar_path = tmp_path/'sidecar.json';sidecar_path.write_text(json.dumps(sidecar))
    binding = dict(format=ARTIFACT_BOUND_PANEL_PROFILE_FORMAT,
                   reference_manifest_sha256=sha(canonical(manifest).encode()),
                   producer_backend=manifest['backend'],
                   evaluator=dict(external_identity=sidecar['external_identity'],
                                  wheel_sha256=sidecar['wheel_sha256'],
                                  sidecar_sha256=sha(sidecar_path.read_bytes()),
                                  native_extension_sha256=sidecar['native_extension_sha256'],
                                  backend_source_sha256=evaluator_sources),
                   constructor_options=dict(CONSTRUCTOR_OPTIONS),
                   implementation_modules={n: sha(MODULE_FILES[n].read_bytes()) for n in IMPLEMENTATION_MODULES},
                   coefficient_identity=manifest['coefficient_identity'])
    binding_path = tmp_path/'artifact-binding.json';binding_path.write_text(json.dumps(binding))
    return dict(root=root, manifest=manifest, package=package, native=native, native32=native32,
                f32_bytes=f32_bytes, green=green,
                binding=binding, binding_path=binding_path, sidecar_path=sidecar_path,
                evaluator_sources=evaluator_sources, f64_bytes=f64_bytes, coeff=coeff, k=k,
                bodies=bodies, case=case)


def rebinding(case, **changes):
    binding = json.loads(json.dumps(case['binding']))
    binding.update(changes)
    path = case['binding_path'].with_name('artifact-binding-alt.json')
    path.write_text(json.dumps(binding))
    return path


def install_evaluator_backend(case):
    """Install the evaluator's own bytes; producer bytes stay pinned in the manifest."""
    for name, digest in case['evaluator_sources'].items():
        (case['package']/name).write_bytes(('evaluator '+name).encode())
        assert sha((case['package']/name).read_bytes()) == digest
    case['native'].write_bytes(case['f64_bytes'])
    case['native32'].write_bytes(case['f32_bytes'])


@pytest.fixture
def case(tmp_path, monkeypatch):
    return build_artifact_case(tmp_path, monkeypatch)


def load(case, binding_path=None):
    return ArtifactBoundPanelProfile.load(case['root'], binding_path or case['binding_path'],
                                          case['sidecar_path'])


def test_load_separates_producer_and_evaluator_identities(case):
    profile = load(case)
    manifest = json.loads(profile.manifest_json)
    assert manifest['format'] == ARTIFACT_BOUND_PANEL_PROFILE_FORMAT
    assert manifest['backend'] == case['binding']['producer_backend']
    assert manifest['artifact_binding']['evaluator']['backend_source_sha256'] == case['evaluator_sources']
    assert manifest['artifact_binding']['evaluator']['backend_source_sha256'] != manifest['backend']['source_sha256']
    assert profile.coefficients.shape == (2, 4) and not profile.coefficients.flags.writeable
    with pytest.raises(ValueError):profile.receivers.setflags(write=True)


def test_relabeling_producer_as_evaluator_refused(case):
    binding = json.loads(json.dumps(case['binding']))
    binding['evaluator']['backend_source_sha256'] = binding['producer_backend']['source_sha256']
    path = rebinding(case)
    Path(path).write_text(json.dumps(binding))
    with pytest.raises(CapabilityError, match='relabel'):load(case, path)


def test_partial_producer_relabeling_refused(case):
    binding = json.loads(json.dumps(case['binding']))
    name = 'meshes/geometry.py'
    binding['evaluator']['backend_source_sha256'][name] = binding['producer_backend']['source_sha256'][name]
    path = rebinding(case)
    Path(path).write_text(json.dumps(binding))
    with pytest.raises(CapabilityError, match='relabel'):load(case, path)


def test_reference_manifest_rebind_refused(case):
    with pytest.raises(CapabilityError, match='canonical original reference'):
        load(case, rebinding(case, reference_manifest_sha256='0'*64))


def test_producer_backend_rewrite_refused(case):
    binding = json.loads(json.dumps(case['binding']))
    binding['producer_backend']['native_library_sha256'] = '0'*64
    path = rebinding(case)
    Path(path).write_text(json.dumps(binding))
    with pytest.raises(CapabilityError, match='producer identity'):load(case, path)


def test_implementation_module_hash_mismatch_refused(case):
    binding = json.loads(json.dumps(case['binding']))
    binding['implementation_modules']['panel_integration_candidate.py'] = '0'*64
    path = rebinding(case)
    Path(path).write_text(json.dumps(binding))
    with pytest.raises(CapabilityError, match='implementation modules'):load(case, path)


def test_constructor_options_change_refused(case):
    binding = json.loads(json.dumps(case['binding']))
    binding['constructor_options']['tabulation_nr'] = 1
    path = rebinding(case)
    Path(path).write_text(json.dumps(binding))
    with pytest.raises(CapabilityError, match='zero-table'):load(case, path)


def test_sidecar_identity_mismatch_refused(case):
    sidecar = json.loads(case['sidecar_path'].read_text())
    sidecar['wheel_sha256'] = 'ff'*32
    case['sidecar_path'].write_text(json.dumps(sidecar))
    with pytest.raises(CapabilityError, match='sidecar'):load(case)


def test_sidecar_numerical_admission_flag_refused(case):
    sidecar = json.loads(case['sidecar_path'].read_text())
    sidecar['admitted_for_numerical_execution'] = True
    case['sidecar_path'].write_text(json.dumps(sidecar))
    binding = json.loads(json.dumps(case['binding']))
    binding['evaluator']['sidecar_sha256'] = sha(case['sidecar_path'].read_bytes())
    path = rebinding(case)
    Path(path).write_text(json.dumps(binding))
    with pytest.raises(CapabilityError, match='sidecar'):load(case, path)


def test_coefficient_identity_mismatch_refused(case):
    binding = json.loads(json.dumps(case['binding']))
    binding['coefficient_identity']['sha256'] = '0'*64
    path = rebinding(case)
    Path(path).write_text(json.dumps(binding))
    with pytest.raises(CapabilityError, match='coefficient'):load(case, path)


def test_green_function_compares_evaluator_map_not_producer(case):
    """Producer-installed sources must be refused: the evaluator map is separate."""
    profile = load(case)
    with pytest.raises(CapabilityError, match='evaluator backend source'):
        profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert case['green'].calls == 0


def test_green_function_accepts_evaluator_sources(case):
    install_evaluator_backend(case)
    profile = load(case)
    gf = profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert gf.exportable_settings['gf_singularities'] == 'low_freq'
    assert gf.find_best_exponential_decomposition.__code__ is FIXED_COEFFICIENT_OVERRIDE_CODE
    kh = float.fromhex(json.loads(profile.manifest_json)['kh_hex'])
    np.testing.assert_array_equal(gf.find_best_exponential_decomposition(kh), case['coeff'])
    with pytest.raises(CapabilityError, match='fallback'):
        gf.find_best_exponential_decomposition(kh, method='fortran')
    with pytest.raises(CapabilityError, match='changed kh'):
        gf.find_best_exponential_decomposition(kh*(1+1e-12))


def test_green_function_f64_extension_mismatch_refused(case):
    for name, digest in case['evaluator_sources'].items():
        (case['package']/name).write_bytes(('evaluator '+name).encode())
    profile = load(case)
    with pytest.raises(CapabilityError, match='float64 extension'):
        profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert case['green'].calls == 0


def test_green_function_f32_extension_mismatch_refused_before_constructor(case):
    install_evaluator_backend(case)
    case['native32'].write_bytes(b'old or wrong float32 library')
    profile = load(case)
    with pytest.raises(CapabilityError, match='float32 extension'):
        profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert case['green'].calls == 0


def test_green_function_implementation_modules_reverified_at_execution_entry(case, monkeypatch):
    install_evaluator_backend(case)
    profile = load(case)
    module = anyhydro.artifact_bound_panel_profile
    real_sha = module._sha
    changed = Path(anyhydro.panel_integration_candidate.__file__).read_bytes()
    def fake_sha(data):
        return '0'*64 if data == changed else real_sha(data)
    monkeypatch.setattr(module, '_sha', fake_sha)
    with pytest.raises(CapabilityError, match='changed after'):
        profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert case['green'].calls == 0


def test_green_function_refuses_effective_tabulation_cache(case, monkeypatch):
    install_evaluator_backend(case)
    profile = load(case)
    green = case['green']
    original = green.__init__
    def cached(self, **kw):
        original(self, **kw)
        self.exportable_settings['tabulation_cache_dir'] = '/tmp/positive-cache'
    monkeypatch.setattr(green, '__init__', cached)
    with pytest.raises(CapabilityError, match='tabulation cache'):
        profile.green_function(green, sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert green.calls == 0


def test_green_function_refuses_positive_cache_instance_attribute(case, monkeypatch):
    """E1 exportable_settings excludes the cache key; the attribute is checked instead."""
    install_evaluator_backend(case)
    profile = load(case)
    green = case['green']
    original = green.__init__
    def cached(self, **kw):
        original(self, **kw)
        self.exportable_settings.pop('tabulation_cache_dir', None)
        self.tabulation_cache_dir = '/tmp/positive-cache'
    monkeypatch.setattr(green, '__init__', cached)
    with pytest.raises(CapabilityError, match='tabulation cache'):
        profile.green_function(green, sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert green.calls == 0


def test_green_function_accepts_backend_without_exposed_cache_property(case, monkeypatch):
    """No exposed cache property is not invented as a setting; the route still passes."""
    install_evaluator_backend(case)
    profile = load(case)
    green = case['green']
    original = green.__init__
    def unexposed(self, **kw):
        original(self, **kw)
        self.exportable_settings.pop('tabulation_cache_dir', None)
    monkeypatch.setattr(green, '__init__', unexposed)
    gf = profile.green_function(green, sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert gf.exportable_settings['gf_singularities'] == 'low_freq'


def test_green_function_wavenumber_mismatch_refused_before_constructor(case):
    install_evaluator_backend(case)
    profile = load(case)
    with pytest.raises(CapabilityError, match='wavenumber'):
        profile.green_function(case['green'], sys.modules['capytaine'], k=case['k']*1.000001, water_depth=150.)
    assert case['green'].calls == 0


def test_green_function_effective_settings_mismatch_refused(case, monkeypatch):
    install_evaluator_backend(case)
    profile = load(case)
    green = case['green']
    original = green.__init__
    def corrupted(self, **kw):
        original(self, **kw)
        self.exportable_settings['gf_singularities'] = 'high_freq'
    monkeypatch.setattr(green, '__init__', corrupted)
    with pytest.raises(CapabilityError, match='effective settings'):
        profile.green_function(green, sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert green.calls == 0


def test_optional_backend_absence_is_typed_capability_error(case, monkeypatch):
    install_evaluator_backend(case)
    profile = load(case)
    monkeypatch.delitem(sys.modules, 'capytaine.green_functions.Delhommeau_float64', raising=False)
    with pytest.raises(CapabilityError, match='optional float64 backend unavailable'):
        profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert case['green'].calls == 0


def test_bind_case_rechecks_implementation_modules(case, monkeypatch):
    profile = load(case)
    identity = case_binding(case['case'])
    profile.bind_case(identity)
    module = anyhydro.artifact_bound_panel_profile
    real_sha = module._sha
    changed = Path(anyhydro.panel_integration_candidate.__file__).read_bytes()
    def fake_sha(data):
        return '0'*64 if data == changed else real_sha(data)
    monkeypatch.setattr(module, '_sha', fake_sha)
    with pytest.raises(CapabilityError, match='changed after'):
        profile.bind_case(identity)


def test_check_micro_requires_all_seven_native_self_fd(case, tmp_path, monkeypatch):
    install_evaluator_backend(case)
    profile = load(case)
    green = case['green']
    original = green.evaluate
    def perturbed(self, receivers, source, **kwargs):
        s, grad = original(self, receivers, source, **kwargs)
        if len(receivers) == 43:
            signs = np.tile([1, 1, 1, -1, -1, -1], len(STEPS))
            steps = np.repeat(STEPS, 6)
            s = np.array(s, copy=True)
            s[1:, 0] = s[1:, 0]+1e-3*signs*steps
        return s, grad
    monkeypatch.setattr(green, 'evaluate', perturbed)
    gf = profile.green_function(green, sys.modules['capytaine'], k=case['k'], water_depth=150.)
    directory = tmp_path/'micro';directory.mkdir()
    with np.load(case['root']/'input.npz') as z:
        mesh = sys.modules['capytaine'].Mesh(z['vertices'], z['faces'][:1])
    params = dict(water_depth=150., wavenumber=case['k'], adjoint_double_layer=True,
                  early_dot_product=False, diagonal_term_in_double_layer=False)
    with pytest.raises(BackendSolveError, match='all seven native self-FD'):
        profile.check_micro(gf, mesh, params, directory,
                            dict(new_green_evaluations=0, numerical_systems=0))
    gate = json.loads((directory/'artifact-bound-micro-gate.json').read_text())
    assert gate['passed'] is False and gate['original_reference_gate_passed'] is True
    assert not any(gate['native_self_fd_passed'])
    comparison = json.loads((directory/'accuracy-comparison.json').read_text())
    assert comparison['passed'] is True


def test_check_micro_passes_with_consistent_gradient(case, tmp_path):
    install_evaluator_backend(case)
    profile = load(case)
    gf = profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    directory = tmp_path/'micro';directory.mkdir()
    with np.load(case['root']/'input.npz') as z:
        mesh = sys.modules['capytaine'].Mesh(z['vertices'], z['faces'][:1])
    params = dict(water_depth=150., wavenumber=case['k'], adjoint_double_layer=True,
                  early_dot_product=False, diagonal_term_in_double_layer=False)
    audit = profile.check_micro(gf, mesh, params, directory,
                                dict(new_green_evaluations=0, numerical_systems=0))
    assert audit['passed'] is True
    gate = json.loads((directory/'artifact-bound-micro-gate.json').read_text())
    assert gate['passed'] is True and all(gate['native_self_fd_passed'])


def test_evaluator_case_metadata_records_both_identities(case):
    install_evaluator_backend(case)
    profile = load(case)
    gf = profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    metadata = profile.evaluator_case_metadata(gf)
    assert metadata['format'] == ARTIFACT_BOUND_PANEL_PROFILE_FORMAT
    assert metadata['producer']['backend']['source_sha256'] == case['manifest']['backend']['source_sha256']
    assert metadata['evaluator']['backend_source_sha256'] == case['evaluator_sources']
    assert metadata['evaluator']['backend_source_sha256'] != metadata['producer']['backend']['source_sha256']
    assert metadata['qualified'] is False
    assert metadata['effective_settings']['floating_point_precision'] == 'float64'
    assert metadata['producer']['role'] == 'original producer only, never the evaluator'
    assert metadata['evaluator']['role'] == 'corrected artifact evaluator'


def test_preserve_writes_binding_record(case, tmp_path):
    profile = load(case)
    directory = tmp_path/'kept';directory.mkdir()
    profile.preserve(directory)
    assert (directory/'artifact-binding.json').read_text() == profile.binding_json
    assert json.loads((directory/'accuracy-profile.json').read_text())['format'] == ARTIFACT_BOUND_PANEL_PROFILE_FORMAT


def test_caller_admits_artifact_bound_profile_and_returns_terminal_probe(case, tmp_path):
    install_evaluator_backend(case)
    profile = load(case)
    result = call(tmp_path/'run', accuracy_profile=profile, operator_probe_receivers=4)
    assert type(result).__name__ == 'PanelOperatorProbe'
    assert result.receiver_count == 4
    output = tmp_path/'run/candidate'
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'completed_probe_unqualified'
    assert status['numerical_systems'] == 0 and status['new_green_evaluations'] == 2
    case_file = json.loads((output/'case.json').read_text())
    metadata = case_file['artifact_evaluator']
    assert 'actual_tabulation' not in case_file
    assert metadata['format'] == ARTIFACT_BOUND_PANEL_PROFILE_FORMAT
    assert metadata['evaluator']['backend_source_sha256'] == case['evaluator_sources']
    assert metadata['producer']['backend']['source_sha256'] == case['manifest']['backend']['source_sha256']
    assert metadata['qualified'] is False
    assert metadata['producer']['role'] == 'original producer only, never the evaluator'
    assert metadata['evaluator']['role'] == 'corrected artifact evaluator'
    assert (output/'artifact-binding.json').read_text() == profile.binding_json
    assert (output/'artifact-bound-micro-gate.json').is_file()
    assert not (output/'radiation.npz').exists()


def native_binding(case):
    from anyhydro.artifact_bound_panel_profile import NATIVE_FORMAT
    evaluator = json.loads(json.dumps(case['binding']['evaluator']))
    evaluator['backend_source_sha256'] = case['manifest']['backend']['source_sha256']
    path = rebinding(case, format=NATIVE_FORMAT, evaluator=evaluator)
    case['native'].write_bytes(case['f64_bytes'])
    case['native32'].write_bytes(case['f32_bytes'])
    core = case['green'].fortran_core
    core.floating_point_precision = types.SimpleNamespace(pre=8)
    core.delhommeau_integrals = types.SimpleNamespace(exp_e1_native=lambda z: (_ for _ in ()).throw(AssertionError('metadata must never execute E1')))
    return path


def test_native_binding_accepts_unchanged_python_sources_and_pins_evaluator(case):
    path = native_binding(case)
    profile = load(case, path)
    gf = profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    from anyhydro.artifact_bound_panel_profile import NATIVE_FORMAT
    metadata = profile.evaluator_case_metadata(gf)
    assert metadata['format'] == NATIVE_FORMAT
    assert metadata['producer']['backend'] == case['manifest']['backend']
    assert metadata['qualified'] is False
    assert metadata['evaluator']['backend_source_sha256'] == metadata['producer']['backend']['source_sha256']
    assert gf.find_best_exponential_decomposition.__code__ is FIXED_COEFFICIENT_OVERRIDE_CODE


def test_native_binding_refuses_original_native_library(case):
    path = native_binding(case)
    binding = json.loads(path.read_text())
    key = next(k for k in binding['evaluator']['native_extension_sha256'] if 'float64' in k)
    binding['evaluator']['native_extension_sha256'][key] = case['manifest']['backend']['native_library_sha256']
    path.write_text(json.dumps(binding))
    with pytest.raises(CapabilityError, match='distinct pinned float64'):load(case, path)


@pytest.mark.parametrize('kind,entry', [(4, True), (8, False), (None, True)])
def test_native_binding_refuses_wrong_precision_or_missing_interface_before_constructor(case, kind, entry):
    path = native_binding(case)
    core = case['green'].fortran_core
    core.floating_point_precision.pre = kind
    if not entry:core.delhommeau_integrals.exp_e1_native = None
    profile = load(case, path)
    before = case['green'].calls
    with pytest.raises(CapabilityError, match='float64 guarded E1'):
        profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert case['green'].calls == before


def test_native_binding_rechecks_native_bytes(case):
    profile = load(case, native_binding(case))
    case['native'].write_bytes(b'mutated evaluator')
    with pytest.raises(CapabilityError, match='float64 extension mismatch'):
        profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)


def test_native_binding_public_caller_preserves_version_and_unqualified_status(case, tmp_path):
    from anyhydro.artifact_bound_panel_profile import NATIVE_FORMAT
    profile = load(case, native_binding(case))
    result = call(tmp_path/'native-v2', accuracy_profile=profile, operator_probe_receivers=4)
    assert type(result).__name__ == 'PanelOperatorProbe'
    output = tmp_path/'native-v2/candidate'
    metadata = json.loads((output/'case.json').read_text())['artifact_evaluator']
    assert metadata['format'] == NATIVE_FORMAT
    assert metadata['qualified'] is False
    assert metadata['producer']['role'] == 'original producer only, never the evaluator'
    assert metadata['evaluator']['role'] == 'corrected artifact evaluator'
    assert json.loads((output/'artifact-bound-micro-gate.json').read_text())['format'] == NATIVE_FORMAT
    assert json.loads((output/'status.json').read_text())['state'] == 'completed_probe_unqualified'
    assert not (output/'radiation.npz').exists()


def test_native_binding_preserves_actual_version_in_profile_manifest(case):
    from anyhydro.artifact_bound_panel_profile import NATIVE_FORMAT
    profile = load(case, native_binding(case))
    manifest = json.loads(profile.manifest_json)
    assert manifest['format'] == NATIVE_FORMAT
    assert manifest['artifact_binding']['format'] == NATIVE_FORMAT
    assert json.loads(load(case).manifest_json)['format'] == ARTIFACT_BOUND_PANEL_PROFILE_FORMAT


def test_native_binding_preserve_writes_versioned_manifest(case, tmp_path):
    from anyhydro.artifact_bound_panel_profile import NATIVE_FORMAT
    profile = load(case, native_binding(case))
    directory = tmp_path/'kept-native';directory.mkdir()
    profile.preserve(directory)
    kept = json.loads((directory/'accuracy-profile.json').read_text())
    assert kept['format'] == NATIVE_FORMAT
    assert (directory/'artifact-binding.json').read_text() == profile.binding_json


def test_native_binding_refuses_swapped_core_object_after_constructor(case, monkeypatch):
    path = native_binding(case)
    profile = load(case, path)
    green = case['green']
    original = green.__init__
    def swapped(self, **kw):
        original(self, **kw)
        self.fortran_core = types.SimpleNamespace(__file__=str(case['native']))
    monkeypatch.setattr(green, '__init__', swapped)
    with pytest.raises(CapabilityError, match='exact imported float64 core'):
        profile.green_function(green, sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert green.calls == 0


@pytest.mark.parametrize('mutate', ['kind', 'interface'])
def test_native_binding_rechecks_guarded_e1_after_constructor(case, monkeypatch, mutate):
    path = native_binding(case)
    profile = load(case, path)
    green = case['green']
    core = green.fortran_core
    original = green.__init__
    def degraded(self, **kw):
        original(self, **kw)
        if mutate == 'kind':
            core.floating_point_precision = types.SimpleNamespace(pre=4)
        else:
            core.delhommeau_integrals = types.SimpleNamespace()
    monkeypatch.setattr(green, '__init__', degraded)
    with pytest.raises(CapabilityError, match='guarded E1 interface after construction'):
        profile.green_function(green, sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert green.calls == 0


def test_native_file_missing_is_typed_capability_error(case):
    path = native_binding(case)
    profile = load(case, path)
    case['native'].unlink()
    with pytest.raises(CapabilityError, match='native file missing or malformed'):
        profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert case['green'].calls == 0


def test_native_file_malformed_identity_is_typed_capability_error(case, monkeypatch):
    path = native_binding(case)
    profile = load(case, path)
    monkeypatch.setattr(case['green'].fortran_core, '__file__', None)
    with pytest.raises(CapabilityError, match='native file missing or malformed'):
        profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert case['green'].calls == 0


def test_float32_native_file_missing_is_typed_capability_error(case):
    install_evaluator_backend(case)
    profile = load(case)
    case['native32'].unlink()
    with pytest.raises(CapabilityError, match='native file missing or malformed'):
        profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert case['green'].calls == 0


def test_missing_fortran_core_after_constructor_is_typed_capability_error(case, monkeypatch):
    install_evaluator_backend(case)
    profile = load(case)
    green = case['green']
    original = green.__init__
    def stripped(self, **kw):
        original(self, **kw)
        self.fortran_core = None
    monkeypatch.setattr(green, '__init__', stripped)
    with pytest.raises(CapabilityError, match='native file missing or malformed'):
        profile.green_function(green, sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert green.calls == 0


def test_native_extension_pathlike_identity_accepted(case, monkeypatch):
    """A Path native file identity pins the same regular target bytes as str."""
    path = native_binding(case)
    profile = load(case, path)
    monkeypatch.setattr(case['green'].fortran_core, '__file__', Path(case['native']))
    gf = profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert gf.exportable_settings['gf_singularities'] == 'low_freq'
    monkeypatch.setattr(case['green'].fortran_core, '__file__', str(case['native']))
    gf = profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert gf.exportable_settings['gf_singularities'] == 'low_freq'


def test_native_extension_pathlike_missing_is_typed_capability_error(case, monkeypatch):
    """A Path identity without a regular file is still a typed refusal."""
    path = native_binding(case)
    profile = load(case, path)
    monkeypatch.setattr(case['green'].fortran_core, '__file__', Path(case['native'].with_name('absent.mock')))
    with pytest.raises(CapabilityError, match='native file missing or malformed'):
        profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert case['green'].calls == 0


def test_native_extension_symlink_target_followed_and_pinned(case):
    """v1 compatibility: a valid symlink extension path pins its regular target bytes."""
    path = native_binding(case)
    profile = load(case, path)
    real = case['native'].with_name('native-real.mock')
    case['native'].rename(real)
    try:
        case['native'].symlink_to(real)
    except OSError as exc:
        pytest.skip(f'symlink creation unavailable on this platform: {exc}')
    gf = profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert gf.exportable_settings['gf_singularities'] == 'low_freq'


def test_native_extension_broken_symlink_is_typed_capability_error(case):
    path = native_binding(case)
    profile = load(case, path)
    case['native'].unlink()
    try:
        case['native'].symlink_to(case['native'].with_name('missing-target.mock'))
    except OSError as exc:
        pytest.skip(f'symlink creation unavailable on this platform: {exc}')
    with pytest.raises(CapabilityError, match='native file missing or malformed'):
        profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert case['green'].calls == 0


def test_native_file_directory_path_is_typed_capability_error(case, tmp_path, monkeypatch):
    path = native_binding(case)
    profile = load(case, path)
    directory = tmp_path/'not-a-native-file';directory.mkdir()
    monkeypatch.setattr(case['green'].fortran_core, '__file__', str(directory))
    with pytest.raises(CapabilityError, match='native file missing or malformed'):
        profile.green_function(case['green'], sys.modules['capytaine'], k=case['k'], water_depth=150.)
    assert case['green'].calls == 0


def rebuild_producer_native_case(case, tmp_path, mutate_backend):
    """Rebuild the sealed bundle with a mutated producer native-library SHA."""
    path = native_binding(case)
    binding = json.loads(path.read_text())
    manifest = json.loads(json.dumps(case['manifest']))
    mutate_backend(manifest['backend'])
    root = tmp_path/'bundle-producer-native';root.mkdir()
    for name in ('input.npz', 'reference.npz'):
        (root/name).write_bytes((case['root']/name).read_bytes())
    seal(root, manifest)
    binding['producer_backend'] = manifest['backend']
    binding['reference_manifest_sha256'] = sha(canonical(manifest).encode())
    binding_path = tmp_path/'artifact-binding-producer-native.json'
    binding_path.write_text(json.dumps(binding))
    return root, binding_path


def test_missing_producer_native_sha_refused(case, tmp_path):
    root, path = rebuild_producer_native_case(
        case, tmp_path, lambda backend: backend.pop('native_library_sha256'))
    with pytest.raises(CapabilityError, match='complete pinned backend identity'):
        ArtifactBoundPanelProfile.load(root, path, case['sidecar_path'])


def test_malformed_producer_native_sha_refused(case, tmp_path):
    root, path = rebuild_producer_native_case(
        case, tmp_path, lambda backend: backend.update(native_library_sha256='0'*63))
    with pytest.raises(CapabilityError, match='invalid backend SHA256'):
        ArtifactBoundPanelProfile.load(root, path, case['sidecar_path'])
