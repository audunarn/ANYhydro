# SPDX-License-Identifier: MPL-2.0
"""Explicit saved frozen-tabulation reuse candidate; distinct consumer identity.

Loads the protected saved producer tabulation (archive, receipt, producing
case) and reuses it through the public Delhommeau constructor/cache route and
inherited public evaluate. The old ``FrozenTabulatedProfile`` cacheNone
producer route is untouched. Producer module identities are historical
provenance; consumer module identities are separate and current. The prepared
per-job cache directory is exclusively owned trusted job storage; external
concurrent mutation is outside this candidate's trust boundary. Matching
loaded hashes do not by themselves prove the absence of native table
construction; the public constructor's documented load path is the route, and
no instrumented native trace is claimed. Actual native execution is not
performed by this module.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .errors import CapabilityError
from .panel_accuracy_reference import (BINDING_KEYS, TABULATED_FORMAT, FrozenTabulatedProfile,
                                       FrozenUntabulatedProfile, _arrays, _immutable, _json, _sha,
                                       _table_identity)

SAVED_TABULATED_FORMAT = 'anyhydro-panel-accuracy-saved-tabulation-v1'
COMMON_REFERENCE_KEYS = ('backend', 'case_binding', 'coefficient_identity', 'files', 'k_hex',
                         'kh_hex', 'orders', 'provenance', 'steps')
SAVED_TABLE_KEYS = ('tabulated_r_range', 'tabulated_z_range', 'tabulated_integrals')
CACHE_FILENAME_SETTINGS = ('floating_point_precision', 'tabulation_grid_shape', 'tabulation_nr',
                           'tabulation_rmax', 'tabulation_nz', 'tabulation_zmin',
                           'tabulation_nb_integration_points')
CONSUMER_MODULES = ('saved_tabulation_profile.py', 'panel_accuracy_reference.py',
                    'panel_integration_candidate.py')


def public_cache_filename(settings):
    """Exact public Delhommeau tabulation cache filename from pinned contract."""
    return 'tabulation_{}_{}_{}_{}_{}_{}_{}.npz'.format(*(settings[name] for name in CACHE_FILENAME_SETTINGS))


def _array_identity(values):
    return dict(shape=list(values.shape), dtype=str(values.dtype),
                sha256=_sha(np.ascontiguousarray(values).tobytes()))


def _hex64(value):
    return (isinstance(value, str) and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


@dataclasses.dataclass(frozen=True, init=False)
class FrozenSavedTabulatedProfile(FrozenTabulatedProfile):
    """Explicit saved-table reuse consumer; never selected implicitly.

    Construct only through :meth:`load`; the producing case must bind through
    the inherited reference contract before any backend construction. The
    subclass is itself a frozen dataclass so attribute replacement and
    instance-level constructor overrides are refused after admission.
    """

    tabulated_r_range: np.ndarray
    tabulated_z_range: np.ndarray
    tabulated_integrals: np.ndarray
    saved_table_identity: str

    @staticmethod
    def constructor_options():
        # tabulation_cache_dir stays None here; the exclusively owned per-job
        # directory is injected at green_function time, never a shared cache.
        return FrozenTabulatedProfile.constructor_options()

    @classmethod
    def load(cls, reference_bundle_directory, saved_directory):
        reference = FrozenUntabulatedProfile.load(reference_bundle_directory)
        saved = Path(saved_directory)
        for name in ('frozen-tabulation.npz', 'tabulation-receipt.json', 'case.json', 'accuracy-profile.json'):
            path = saved/name
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 96*1024*1024:
                raise CapabilityError('saved tabulation evidence requires bounded regular files')
        archive_bytes = (saved/'frozen-tabulation.npz').read_bytes()
        receipt_bytes = (saved/'tabulation-receipt.json').read_bytes()
        case_bytes = (saved/'case.json').read_bytes()
        profile_bytes = (saved/'accuracy-profile.json').read_bytes()
        try:
            producer_profile = json.loads(profile_bytes.decode('utf-8'),
                                           parse_constant=lambda _: (_ for _ in ()).throw(
                                               ValueError('nonfinite JSON')))
        except (ValueError, UnicodeDecodeError) as exc:
            raise CapabilityError('saved producing accuracy profile is not valid UTF-8 JSON') from exc
        reference_manifest = json.loads(reference.manifest_json)
        if (producer_profile.get('format') != TABULATED_FORMAT
                or producer_profile.get('reference_manifest_sha256') != _sha(reference.manifest_json.encode())
                or any(producer_profile.get(key) != reference_manifest[key] for key in COMMON_REFERENCE_KEYS)
                or producer_profile.get('constructor_settings') != cls.constructor_options()
                or not isinstance(producer_profile.get('implementation_modules'), dict)
                or not producer_profile['implementation_modules']
                or not all(_hex64(value) for value in producer_profile['implementation_modules'].values())):
            raise CapabilityError('saved producing accuracy profile differs from the independent '
                                  'reference bundle: backend implementation mismatch or producer '
                                  'descriptor is not the tabulated producer of this reference')
        try:
            tables = _arrays(archive_bytes, SAVED_TABLE_KEYS)
        except CapabilityError:
            raise
        except Exception as exc:
            raise CapabilityError('saved tabulation archive is not a readable bounded npz archive') from exc
        try:
            receipt = json.loads(receipt_bytes.decode('utf-8'),
                                 parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
            case = json.loads(case_bytes.decode('utf-8'),
                              parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
        except (ValueError, UnicodeDecodeError) as exc:
            raise CapabilityError('saved tabulation receipt/case are not valid UTF-8 JSON') from exc
        case_identity = {key: value for key, value in case.items() if key != 'case_hash'}
        if (receipt.get('format') != TABULATED_FORMAT
                or receipt.get('constructor_settings') != cls.constructor_options()
                or receipt.get('constructor_settings') != producer_profile['constructor_settings']
                or receipt.get('implementation_modules') != producer_profile['implementation_modules']
                or receipt.get('planned_profile_sha256') != _sha(profile_bytes)
                or receipt.get('archive_sha256') != _sha(archive_bytes)
                or {name: _array_identity(values) for name, values in tables.items()} != receipt.get('actual_tables')
                or case.get('actual_tabulation', {}).get('actual_tables') != receipt.get('actual_tables')
                or case.get('actual_tabulation', {}).get('format') != TABULATED_FORMAT
                or case.get('actual_tabulation', {}).get('constructor_settings') != cls.constructor_options()
                or case.get('actual_tabulation', {}).get('implementation_modules') != producer_profile['implementation_modules']
                or case.get('accuracy_profile_sha256') != receipt.get('planned_profile_sha256')
                or not case.get('planned_case_hash') or not case.get('case_hash')
                or not _hex64(case.get('module_sha256'))
                or not _hex64(case.get('accuracy_implementation_sha256'))
                or hashlib.sha256(_json(case_identity).encode()).hexdigest() != case.get('case_hash')
                or any(case.get(key) != reference_manifest['case_binding'][key] for key in BINDING_KEYS)):
            raise CapabilityError('saved tabulation receipt/case/table identities are inconsistent '
                                  'with the producing case or the independent reference binding')
        expected_shapes = {'tabulated_r_range': (1352,), 'tabulated_z_range': (744,),
                           'tabulated_integrals': (1352, 744, 5)}
        for name, shape in expected_shapes.items():
            values = tables[name]
            if values.shape != shape or values.dtype != np.dtype('float64'):
                raise CapabilityError('saved tabulation shape/dtype mismatch')
        if (not np.all(np.diff(tables['tabulated_r_range']) > 0)
                or not np.all(np.diff(tables['tabulated_z_range']) < 0)
                or tables['tabulated_r_range'][0] < 0 or tables['tabulated_z_range'][0] >= 0
                or not np.isclose(tables['tabulated_r_range'][-1], 100., rtol=0, atol=1e-12)
                or not np.isclose(tables['tabulated_z_range'][-1], -251., rtol=0, atol=1e-10)):
            raise CapabilityError('saved tabulation coordinate range/ordering mismatch')
        manifest = json.loads(reference.manifest_json)
        manifest['format'] = SAVED_TABULATED_FORMAT
        manifest['reference_manifest_sha256'] = _sha(reference.manifest_json.encode())
        manifest['constructor_settings'] = cls.constructor_options()
        manifest['saved_table_provenance'] = dict(
            producer_receipt_format=receipt['format'],
            archive_sha256=receipt['archive_sha256'],
            receipt_sha256=_sha(receipt_bytes),
            producing_case_hash=case['case_hash'],
            producing_case_hash_recomputed=True,
            producing_planned_case_hash=case['planned_case_hash'],
            planned_profile_sha256=receipt['planned_profile_sha256'],
            producer_profile_sha256=_sha(profile_bytes),
            producer_implementation_modules=receipt['implementation_modules'],
            producer_candidate_module_sha256=case['module_sha256'],
            producer_accuracy_module_sha256=case['accuracy_implementation_sha256'],
            declared_tables=receipt['actual_tables'],
            producer_identities_are_historical_not_current_source=True,
            qualified=False)
        manifest['consumer_modules'] = {name: _sha(Path(__file__).with_name(name).read_bytes())
                                        for name in CONSUMER_MODULES}
        profile = object.__new__(cls)
        for name, values in tables.items():
            object.__setattr__(profile, name, _immutable(values))
        object.__setattr__(profile, 'saved_table_identity', _json(
            {name: _array_identity(values) for name, values in tables.items()}))
        for field_name in ('manifest_json', 'input_bytes', 'reference_bytes', 'coefficients',
                           'receivers', 'potential', 'gradient'):
            object.__setattr__(profile, field_name, getattr(reference, field_name))
        object.__setattr__(profile, 'manifest_json', _json(manifest))
        return profile

    def bind_case(self, identity):
        FrozenUntabulatedProfile.bind_case(self, identity)
        manifest = json.loads(self.manifest_json)
        actual = {name: _sha(Path(__file__).with_name(name).read_bytes()) for name in CONSUMER_MODULES}
        if (manifest['constructor_settings'] != self.constructor_options()
                or actual != manifest['consumer_modules']
                or self.saved_table_identity != _json({name: _array_identity(getattr(self, name))
                                                        for name in SAVED_TABLE_KEYS})):
            raise CapabilityError('saved-tabulation consumer implementation/settings/tables changed after loading')

    def green_function(self, base, capytaine, *, k, water_depth, artifact_directory, runtime):
        directory = Path(artifact_directory)
        cache_directory = directory/'saved-tabulation-cache'
        cache_directory.mkdir(parents=False, exist_ok=False)
        settings = self.constructor_options()
        filename = public_cache_filename(settings)
        start = time.perf_counter()
        np.savez(cache_directory/filename, r_range=self.tabulated_r_range,
                 z_range=self.tabulated_z_range, values=self.tabulated_integrals)
        runtime['saved_table_cache_writes'] = 1
        runtime['saved_table_cache_write_wall_seconds'] = time.perf_counter()-start
        runtime['saved_table_cache_filename'] = filename
        options = dict(settings, tabulation_cache_dir=str(cache_directory))

        def acquired(gf):
            loaded_start = time.perf_counter()
            runtime['saved_table_cache_loads'] = 1
            # Expected source-path bookkeeping: the public constructor returned
            # tables without this module observing a native construction. It
            # is not an instrumented native trace; the mock backend proves the
            # cache-load route in tests, nothing is claimed for native runs.
            runtime['new_table_constructions'] = 0
            runtime['new_table_constructions_zero_basis'] = 'expected_source_path_bookkeeping'
            arrays = {name: getattr(gf, name) for name in SAVED_TABLE_KEYS}
            np.savez(directory/'loaded-saved-tabulation.npz', **arrays)
            identities = {name: _array_identity(values) for name, values in arrays.items()}
            (directory/'loaded-saved-tabulation.json').write_text(json.dumps(dict(
                state='loaded_not_validated', actual_arrays=identities,
                cache_filename=filename, exclusive_job_cache_directory=str(cache_directory),
                external_concurrent_mutation_outside_trust_boundary=True), indent=2), encoding='utf-8')
            if _json(identities) != self.saved_table_identity:
                raise CapabilityError('loaded saved tables differ from declared producer tables')
            runtime['saved_table_load_validation_wall_seconds'] = time.perf_counter()-loaded_start

        gf = self._green_function(base, capytaine, k=k, water_depth=water_depth,
                                  options=options, acquired=acquired)
        if gf.tabulation_cache_dir is None:
            raise CapabilityError('saved-table evaluator must load through the exclusively owned per-job cache')
        start = time.perf_counter()
        expected_shapes = {'tabulated_r_range': (1352,), 'tabulated_z_range': (744,),
                           'tabulated_integrals': (1352, 744, 5)}
        for name, shape in expected_shapes.items():
            values = getattr(gf, name)
            if values.shape != shape or values.dtype != np.dtype('float64') or not np.all(np.isfinite(values)):
                raise CapabilityError('loaded saved table shape/dtype/finite mismatch')
            setattr(gf, name, _immutable(values))
        if (not np.all(np.diff(gf.tabulated_r_range) > 0) or not np.all(np.diff(gf.tabulated_z_range) < 0)
                or gf.tabulated_r_range[0] < 0 or gf.tabulated_z_range[0] >= 0
                or not np.isclose(gf.tabulated_r_range[-1], 100., rtol=0, atol=1e-12)
                or not np.isclose(gf.tabulated_z_range[-1], -251., rtol=0, atol=1e-10)):
            raise CapabilityError('loaded saved table coordinate range/ordering mismatch')
        gf.frozen_table_identity_json = _json(_table_identity(gf))
        gf.frozen_table_freezing_wall_seconds = time.perf_counter()-start
        return gf

    def evaluator_case_metadata(self, gf):
        manifest = json.loads(self.manifest_json)
        return dict(format=SAVED_TABULATED_FORMAT,
                    constructor_settings=manifest['constructor_settings'],
                    saved_table_provenance=manifest['saved_table_provenance'],
                    consumer_modules=manifest['consumer_modules'],
                    planned_profile_sha256=_sha(self.manifest_json.encode()),
                    actual_tables=json.loads(gf.frozen_table_identity_json),
                    cache_load_observed_via_public_constructor=True,
                    cache_load_observation_basis='returned_arrays_expected_source_path_bookkeeping',
                    cache_load_mock_observed_only=True,
                    new_table_constructions_zero_is='expected_source_path_bookkeeping_not_native_instrumented_proof',
                    instrumented_native_trace=False,
                    native_internal_trace=False,
                    native_zero_construction_claimed=False,
                    hash_match_alone_does_not_prove_absence_of_native_construction=True,
                    qualified=False)
