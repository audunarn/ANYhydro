# SPDX-License-Identifier: MPL-2.0
"""Artifact-bound accuracy profile: original producer reference, corrected evaluator.

The immutable reference bundle keeps identifying its ORIGINAL producer backend.
Binding v1 requires three distinct Python source hashes. Native binding v2
intentionally permits unchanged OR changed Python source maps, all exactly
pinned, and requires a distinct pinned float64 native library with a guarded
E1 interface. Both pin producer and evaluator independently without
relabeling the reference producer. The real artifact's three evaluator Python
maps all differ from the producer's, so it exercises the v2 changed-source
path; no real Python map equals the producer map. Loading is headless: no
Capytaine, Qt or native import occurs before ``green_function``. Hashes prove
binding integrity, not scientific qualification.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import json
import os
from pathlib import Path
import time

import numpy as np

from .errors import BackendSolveError, CapabilityError
from .panel_accuracy_reference import (BACKEND_FILES, FrozenUntabulatedProfile,
                                       _json, _sha)

FORMAT = 'anyhydro-artifact-bound-panel-profile-v1'
NATIVE_FORMAT = 'anyhydro-artifact-bound-native-panel-profile-v2'
SIDECAR_FORMAT = 'anyhydro-backend-artifact-identity-v1'
BINDING_KEYS = ('format', 'reference_manifest_sha256', 'producer_backend',
                'evaluator', 'constructor_options', 'implementation_modules',
                'coefficient_identity')
EVALUATOR_KEYS = ('external_identity', 'wheel_sha256', 'sidecar_sha256',
                  'native_extension_sha256', 'backend_source_sha256')
IMPLEMENTATION_MODULES = ('artifact_bound_panel_profile.py',
                          'panel_integration_candidate.py',
                          'panel_accuracy_reference.py')
CONSTRUCTOR_OPTIONS = dict(tabulation_nr=0, tabulation_nz=0,
                           tabulation_cache_dir=None, gf_singularities='low_freq',
                           floating_point_precision='float64',
                           finite_depth_prony_decomposition_method='python')
ARTIFACT_BOUND_PANEL_PROFILE_FORMAT = FORMAT


def _digest(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _bounded_file(path, maximum):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise CapabilityError('artifact binding requires bounded regular files')
    return path.read_bytes()


def _native_extension_bytes(module, label):
    """Typed missing/malformed native-file refusal; never an untyped OSError.

    v1 compatibility: a symlink extension path is followed and the bytes of
    its regular target are pinned, exactly as before; only the resulting
    bytes are trusted. Native file identities may be str or os.PathLike
    (including Path). Any missing/malformed path, non-regular target or
    filesystem error is a typed refusal.
    """
    path = getattr(module, '__file__', None)
    if isinstance(path, (str, os.PathLike)) and path:
        try:
            candidate = Path(path)
            if candidate.is_file():
                return candidate.read_bytes()
        except (OSError, TypeError, ValueError):
            pass
    raise CapabilityError(f'artifact-bound evaluator {label} native file missing or malformed')


def _fixed_coefficient_decomposition(self, kh, method=None):
    """Public fixed-coefficient override; not a new fit, no cache, no fallback."""
    kh_hex = getattr(self, '_artifact_bound_kh_hex', None)
    coefficients = getattr(self, '_artifact_bound_coefficients', None)
    if kh_hex is None or coefficients is None:
        raise CapabilityError('artifact-bound evaluator lacks its frozen decomposition binding')
    if method not in (None, 'python') or float(kh).hex() != kh_hex:
        raise CapabilityError('artifact-bound evaluator refuses decomposition fallback or changed kh')
    return coefficients


# Exact code object guarded by the operator-route checker (fixed_overrides budget).
FIXED_COEFFICIENT_OVERRIDE_CODE = _fixed_coefficient_decomposition.__code__


@dataclass(frozen=True, init=False)
class ArtifactBoundPanelProfile(FrozenUntabulatedProfile):
    """Original producer reference checked against the corrected E1 evaluator.

    Construct only through :meth:`load`. The producer identity stays the
    immutable original manifest backend; the evaluator identity (wheel,
    sidecar, native extensions, three source hashes) is pinned independently
    in the versioned binding record and compared against the actually
    installed backend before the constructor runs.
    """
    binding_json: str
    # Explicit case.json identity key for this profile; old tabulated/saved
    # profiles keep the historical 'actual_tabulation' schema unchanged.
    CASE_METADATA_KEY = 'artifact_evaluator'

    @classmethod
    def load(cls, reference_directory, binding_record_path, sidecar_path):
        try:
            reference = FrozenUntabulatedProfile.load(reference_directory)
            manifest = json.loads(reference.manifest_json)
            binding = json.loads(_bounded_file(binding_record_path, 256*1024))
            if (set(binding) != set(BINDING_KEYS) or binding['format'] not in (FORMAT, NATIVE_FORMAT)):
                raise CapabilityError('unsupported artifact binding record')
            if binding['reference_manifest_sha256'] != _sha(reference.manifest_json.encode()):
                raise CapabilityError('binding does not pin the canonical original reference manifest')
            if binding['producer_backend'] != manifest['backend']:
                raise CapabilityError('binding producer identity differs from the immutable reference manifest')
            evaluator = binding['evaluator']
            if set(evaluator) != set(EVALUATOR_KEYS):
                raise CapabilityError('incomplete evaluator identity required')
            sources = evaluator['backend_source_sha256']
            if (set(sources) != set(BACKEND_FILES)
                    or not all(_digest(v) for v in sources.values())
                    or not _digest(evaluator['wheel_sha256']) or not _digest(evaluator['sidecar_sha256'])
                    or not all(_digest(v) for v in evaluator['native_extension_sha256'].values())):
                raise CapabilityError('invalid evaluator SHA256 identity')
            if binding['format'] == FORMAT and (sources == manifest['backend']['source_sha256'] or any(
                    sources[name] == manifest['backend']['source_sha256'][name] for name in BACKEND_FILES)):
                raise CapabilityError('evaluator source map must not relabel the original producer backend')
            extensions = evaluator['native_extension_sha256']
            if (len(extensions) != 2
                    or sum('Delhommeau_float64' in key for key in extensions) != 1
                    or sum('Delhommeau_float32' in key for key in extensions) != 1):
                raise CapabilityError('evaluator must pin both native extension hashes')
            if binding['format'] == NATIVE_FORMAT:
                producer_native = manifest['backend'].get('native_library_sha256')
                f64_key = next(key for key in extensions if 'Delhommeau_float64' in key)
                if not _digest(producer_native) or extensions[f64_key] == producer_native:
                    raise CapabilityError('native binding requires a distinct pinned float64 evaluator')
            if binding['constructor_options'] != CONSTRUCTOR_OPTIONS:
                raise CapabilityError('binding constructor options are not the fixed zero-table route')
            if (set(binding['implementation_modules']) != set(IMPLEMENTATION_MODULES)
                    or binding['implementation_modules'] != {
                        name: _sha(Path(__file__).with_name(name).read_bytes())
                        for name in IMPLEMENTATION_MODULES}):
                raise CapabilityError('artifact-bound implementation modules changed after binding')
            if binding['coefficient_identity'] != manifest['coefficient_identity']:
                raise CapabilityError('binding coefficient identity differs from the frozen reference')
            sidecar_bytes = _bounded_file(sidecar_path, 256*1024)
            sidecar = json.loads(sidecar_bytes)
            if (_sha(sidecar_bytes) != evaluator['sidecar_sha256']
                    or sidecar.get('format') != SIDECAR_FORMAT
                    or sidecar.get('external_identity') != evaluator['external_identity']
                    or sidecar.get('wheel_sha256') != evaluator['wheel_sha256']
                    or sidecar.get('native_extension_sha256') != extensions
                    or sidecar.get('admitted_for_numerical_execution') is not False):
                raise CapabilityError('backend sidecar does not complete the exact evaluator identity')
            manifest['format'] = binding['format']
            manifest['reference_manifest_sha256'] = binding['reference_manifest_sha256']
            manifest['artifact_binding'] = binding
            profile = object.__new__(cls)
            values = dict(manifest_json=_json(manifest), binding_json=_json(binding))
            for name in ('input_bytes', 'reference_bytes', 'coefficients',
                         'receivers', 'potential', 'gradient'):
                values[name] = getattr(reference, name)
            for name, value in values.items():
                object.__setattr__(profile, name, value)
            return profile
        except CapabilityError:
            raise
        except (ValueError, TypeError, KeyError, IndexError, OSError) as exc:
            raise CapabilityError(f'invalid artifact-bound profile: {exc}') from exc

    def bind_case(self, identity):
        super().bind_case(identity)
        binding = json.loads(self.binding_json)
        if binding['implementation_modules'] != {
                name: _sha(Path(__file__).with_name(name).read_bytes())
                for name in IMPLEMENTATION_MODULES}:
            raise CapabilityError('artifact-bound implementation modules changed after loading')

    def preserve(self, directory):
        super().preserve(directory)
        (Path(directory)/'artifact-binding.json').write_text(self.binding_json, encoding='utf-8')

    def green_function(self, base, capytaine, *, k, water_depth):
        manifest = json.loads(self.manifest_json)
        binding = json.loads(self.binding_json)
        evaluator = binding['evaluator']
        if binding['implementation_modules'] != {
                name: _sha(Path(__file__).with_name(name).read_bytes())
                for name in IMPLEMENTATION_MODULES}:
            raise CapabilityError('artifact-bound implementation modules changed after loading')
        if k.hex() != manifest['k_hex'] or (k*water_depth).hex() != manifest['kh_hex']:
            raise CapabilityError('frozen wavenumber/depth mismatch')
        package = Path(capytaine.__file__).parent
        sources = {name: _sha((package/name).read_bytes()) for name in BACKEND_FILES}
        if sources != evaluator['backend_source_sha256']:
            raise CapabilityError('artifact-bound evaluator backend source mismatch')
        try:
            core = import_module('capytaine.green_functions.Delhommeau_float64')
        except Exception as exc:
            raise CapabilityError(f'optional float64 backend unavailable: {exc}') from exc
        try:
            core32 = import_module('capytaine.green_functions.Delhommeau_float32')
        except Exception as exc:
            raise CapabilityError(f'optional float32 backend unavailable: {exc}') from exc
        extensions = evaluator['native_extension_sha256']
        f64_key = next(key for key in extensions if 'Delhommeau_float64' in key)
        f32_key = next(key for key in extensions if 'Delhommeau_float32' in key)
        if _sha(_native_extension_bytes(core, 'float64')) != extensions[f64_key]:
            raise CapabilityError('artifact-bound evaluator float64 extension mismatch')
        if _sha(_native_extension_bytes(core32, 'float32')) != extensions[f32_key]:
            raise CapabilityError('artifact-bound evaluator float32 extension mismatch')
        if binding['format'] == NATIVE_FORMAT:
            kind = getattr(getattr(core, 'floating_point_precision', None), 'pre', None)
            entry = getattr(getattr(core, 'delhommeau_integrals', None), 'exp_e1_native', None)
            if kind != 8 or not callable(entry):
                raise CapabilityError('native binding requires float64 guarded E1 interface')

        class ArtifactBoundGreen(base):
            find_best_exponential_decomposition = _fixed_coefficient_decomposition

        start = time.perf_counter()
        gf = ArtifactBoundGreen(**CONSTRUCTOR_OPTIONS)
        gf.frozen_constructor_wall_seconds = time.perf_counter()-start
        gf._artifact_bound_kh_hex = manifest['kh_hex']
        gf._artifact_bound_coefficients = self.coefficients
        actual_core = getattr(gf, 'fortran_core', None)
        if binding['format'] == NATIVE_FORMAT and actual_core is not core:
            raise CapabilityError('native binding requires the exact imported float64 core after construction')
        if _sha(_native_extension_bytes(actual_core, 'float64')) != extensions[f64_key]:
            raise CapabilityError('artifact-bound evaluator float64 extension mismatch')
        if _sha(_native_extension_bytes(core32, 'float32')) != extensions[f32_key]:
            raise CapabilityError('artifact-bound evaluator float32 extension mismatch')
        if binding['format'] == NATIVE_FORMAT:
            kind = getattr(getattr(actual_core, 'floating_point_precision', None), 'pre', None)
            entry = getattr(getattr(actual_core, 'delhommeau_integrals', None), 'exp_e1_native', None)
            if kind != 8 or not callable(entry):
                raise CapabilityError('native binding requires float64 guarded E1 interface after construction')
        if {name: _sha((package/name).read_bytes()) for name in BACKEND_FILES} != sources:
            raise CapabilityError('artifact-bound evaluator backend source mismatch')
        settings = gf.exportable_settings
        expected = dict(CONSTRUCTOR_OPTIONS)
        # The exact E1 Delhommeau.exportable_settings intentionally excludes
        # tabulation_cache_dir (E1 delhommeau.py keeps it only as the instance
        # attribute assigned from the pinned constructor argument), so the
        # effective cache property is read from that attribute when the
        # settings map does not expose it; any positive cache is refused and
        # no setting is invented.
        cache_dir = settings.get('tabulation_cache_dir')
        if 'tabulation_cache_dir' not in settings:
            cache_dir = getattr(gf, 'tabulation_cache_dir', None)
        if cache_dir is not None:
            raise CapabilityError('artifact-bound evaluator refuses an effective tabulation cache')
        expected.pop('tabulation_cache_dir')
        for key, expected_value in expected.items():
            if settings.get(key) != expected_value:
                raise CapabilityError('artifact-bound evaluator effective settings mismatch')
        return gf

    def check_micro(self, gf, mesh, params, directory, runtime):
        audit = super().check_micro(gf, mesh, params, directory, runtime)
        all_passed = bool(all(audit['native_self_fd_passed']))
        gate = dict(format=json.loads(self.binding_json)['format'], requires_all_seven_native_self_fd=True,
                    limit=audit['limit'], steps=audit['steps'],
                    native_self_fd_errors=audit['native_self_fd_errors'],
                    native_self_fd_passed=audit['native_self_fd_passed'],
                    original_reference_gate_passed=audit['passed'], passed=all_passed)
        (Path(directory)/'artifact-bound-micro-gate.json').write_text(json.dumps(gate, indent=2), encoding='utf-8')
        if not all_passed:
            raise BackendSolveError('artifact-bound profile requires all seven native self-FD steps within the original limit')
        return audit

    def evaluator_case_metadata(self, gf):
        manifest = json.loads(self.manifest_json)
        binding = json.loads(self.binding_json)
        return dict(format=binding['format'],
                    binding_sha256=_sha(self.binding_json.encode()),
                    producer=dict(role='original producer only, never the evaluator',
                                  reference_manifest_sha256=binding['reference_manifest_sha256'],
                                  backend=manifest['backend']),
                    evaluator=dict(role='corrected artifact evaluator',
                                   external_identity=binding['evaluator']['external_identity'],
                                   wheel_sha256=binding['evaluator']['wheel_sha256'],
                                   sidecar_sha256=binding['evaluator']['sidecar_sha256'],
                                   native_extension_sha256=binding['evaluator']['native_extension_sha256'],
                                   backend_source_sha256=binding['evaluator']['backend_source_sha256']),
                    constructor_options=binding['constructor_options'],
                    implementation_modules=binding['implementation_modules'],
                    effective_settings=gf.exportable_settings,
                    qualified=False)
