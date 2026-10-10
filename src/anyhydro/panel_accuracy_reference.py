# SPDX-License-Identifier: MPL-2.0
"""Declared, case-bound accuracy evidence for an opt-in development candidate.

Hashes prove bundle integrity, not oracle independence or scientific acceptance.
No backend imports, execution, arbitrary paths or caller-selected tolerances occur
when loading. The numerical caller remains responsible for execution supervision.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from io import BytesIO
import hashlib
from importlib import import_module
import json
import math
import numbers
from pathlib import Path
import time
import zipfile

import numpy as np

from .errors import BackendSolveError, CapabilityError

FORMAT = 'anyhydro-panel-accuracy-reference-v1'
TABULATED_FORMAT = 'anyhydro-panel-accuracy-finer-tabulation-v1'
STEPS = (1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5, 1e-5, 5e-6)
BINDING_KEYS = ('inventory', 'body_ids', 'rotation_centers', 'omega',
                'water_depth', 'rho', 'gravity', 'snapshot_arrays')
MODEL_ARRAYS = ('vertices', 'faces', 'owner', 'hull_mask', 'points', 'weights',
                'normals', 'rigid_trace')
BACKEND_FILES = ('green_functions/delhommeau.py', 'tools/prony_decomposition.py',
                 'meshes/geometry.py')


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _immutable(value):
    value = np.ascontiguousarray(value)
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


def case_binding(identity):
    """Full hull/lid snapshots and body/fluid contract, independent of run budget."""
    return {key: identity[key] for key in BINDING_KEYS}


def _arrays(data, keys):
    # Bound decompressed data before NumPy allocates; disallow pickle/object arrays.
    with zipfile.ZipFile(BytesIO(data)) as archive:
        members = archive.infolist()
        if (len(members) != len(keys) or {m.filename for m in members} != {k+'.npy' for k in keys}
                or sum(m.file_size for m in members) > 64*1024*1024):
            raise CapabilityError('invalid or oversized reference array archive')
        for member in members:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1, 0), (2, 0)):
                    raise CapabilityError('unsupported reference NPY header version')
                read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                               else np.lib.format.read_array_header_2_0)
                shape, _, dtype = read_header(stream, max_header_size=10000)
                if (len(shape) > 4 or any(type(n) is not int or n < 0 for n in shape)
                        or dtype.kind not in 'bifuc' or dtype.itemsize > 16
                        or math.prod(shape)*dtype.itemsize > 64*1024*1024
                        or stream.tell()+math.prod(shape)*dtype.itemsize != member.file_size):
                    raise CapabilityError('reference NPY declared allocation/payload mismatch')
    with np.load(BytesIO(data), allow_pickle=False) as archive:
        result = {key: archive[key] for key in keys}
    for a in result.values():
        if a.dtype.kind not in 'bifu c'.replace(' ', '') or not np.all(np.isfinite(a)):
            raise CapabilityError('reference arrays require finite numeric values')
    return result


@dataclass(frozen=True, init=False)
class FrozenUntabulatedProfile:
    """Immutable portable profile; construct only through :meth:`load`."""
    manifest_json: str
    input_bytes: bytes
    reference_bytes: bytes
    coefficients: np.ndarray
    receivers: np.ndarray
    potential: np.ndarray
    gradient: np.ndarray

    def __new__(cls):
        raise CapabilityError('construct frozen profile through load() only')

    @classmethod
    def load(cls, directory):
        try:
            root = Path(directory)
            files = {}
            for name, maximum in [('manifest.json', 256*1024), ('input.npz', 32*1024*1024),
                                  ('reference.npz', 32*1024*1024)]:
                path = root/name
                if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
                    raise CapabilityError('reference bundle requires bounded regular files')
                files[name] = path.read_bytes()
            manifest = json.loads(files['manifest.json'], parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
            required = {'format', 'files', 'case_binding', 'backend', 'kh_hex', 'k_hex',
                        'orders', 'steps', 'provenance', 'coefficient_identity'}
            if set(manifest) != required or manifest['format'] != FORMAT:
                raise CapabilityError('unsupported accuracy reference manifest')
            if (manifest['orders'] != [64, 128] or manifest['steps'] != list(STEPS)
                    or set(manifest['case_binding']) != set(BINDING_KEYS)
                    or not isinstance(manifest['provenance'], dict) or not manifest['provenance']):
                raise CapabilityError('fixed reference orders, steps and provenance required')
            if manifest['files'] != {n: _sha(files[n]) for n in ('input.npz', 'reference.npz')}:
                raise CapabilityError('reference file hash mismatch')
            backend = manifest['backend']
            if (set(backend) != {'capytaine', 'native_library_sha256', 'source_sha256'}
                    or backend['capytaine'] != '3.0.0'
                    or set(backend['source_sha256']) != set(BACKEND_FILES)):
                raise CapabilityError('complete pinned backend identity required')
            for digest in [backend['native_library_sha256'], *backend['source_sha256'].values()]:
                if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                    raise CapabilityError('invalid backend SHA256')
            inputs = _arrays(files['input.npz'], (*MODEL_ARRAYS, 'coefficients', 'receivers'))
            snapshots = manifest['case_binding']['snapshot_arrays']
            actual = {n: dict(shape=list(inputs[n].shape), dtype=str(inputs[n].dtype),
                              sha256=_sha(np.ascontiguousarray(inputs[n]).tobytes())) for n in MODEL_ARRAYS}
            if snapshots != actual:
                raise CapabilityError('full model snapshot hash/shape/dtype mismatch')
            coeff = inputs['coefficients']; receivers = inputs['receivers']
            if coeff.dtype != np.dtype('float64') or coeff.shape != (2, 4):
                raise CapabilityError('actual frozen lambda/amplitude2x4 float64 required')
            if manifest['coefficient_identity'] != dict(shape=[2, 4], dtype='<f8', sha256=_sha(coeff.tobytes())):
                raise CapabilityError('frozen coefficient identity mismatch')
            vertices, faces = inputs['vertices'], inputs['faces']
            if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
                raise CapabilityError('reference requires complete triangular model')
            center = vertices[faces[0]].mean(axis=0).reshape(1, 3)
            center[0, :2] += [.37, .19]
            depth = manifest['case_binding']['water_depth']
            center[0, 2] -= min(.17, (depth+center[0, 2])/4)
            expected = np.concatenate([center] + [center+sign*s*np.eye(3)[d]
                                      for s in STEPS for sign in (1, -1) for d in range(3)])
            if receivers.dtype != np.dtype('float64') or receivers.shape != (43, 3) or not np.array_equal(receivers, expected):
                raise CapabilityError('reference receiver grid differs from fixed original steps')
            values = _arrays(files['reference.npz'], ('potential_low', 'potential_high', 'gradient_low', 'gradient_high'))
            for name, a in values.items():
                shape = (43, 3) if name.startswith('gradient') else (43,)
                if a.shape != shape or a.dtype != np.dtype('complex128'):
                    raise CapabilityError('reference requires complex128 potential and receiver gradient')
            for kind in ('potential', 'gradient'):
                low, high = values[kind+'_low'], values[kind+'_high']
                if np.any(np.abs(high-low) > 1e-13+1e-11*np.maximum(np.abs(low), np.abs(high))):
                    raise CapabilityError('reference64/128 numerical convergence failed')
            # The declared k/kh must be exact finite values for this complete case.
            k = float.fromhex(manifest['k_hex']); kh = float.fromhex(manifest['kh_hex'])
            if not np.isfinite(k) or k <= 0 or kh != k*depth:
                raise CapabilityError('invalid frozen k/kh')
            profile = object.__new__(cls)
            for name, value in dict(manifest_json=_json(manifest), input_bytes=files['input.npz'],
                                    reference_bytes=files['reference.npz'], coefficients=_immutable(coeff),
                                    receivers=_immutable(receivers), potential=_immutable(values['potential_high']),
                                    gradient=_immutable(values['gradient_high'])).items():
                object.__setattr__(profile, name, value)
            return profile
        except CapabilityError:
            raise
        except (ValueError, TypeError, KeyError, IndexError, OSError, zipfile.BadZipFile) as exc:
            raise CapabilityError(f'invalid panel accuracy reference: {exc}') from exc

    def bind_case(self, identity):
        if _json(case_binding(identity)) != _json(json.loads(self.manifest_json)['case_binding']):
            raise CapabilityError('accuracy profile differs from complete producing case')

    def preserve(self, directory):
        directory = Path(directory)
        (directory/'accuracy-profile.json').write_text(self.manifest_json, encoding='utf-8')
        (directory/'accuracy-input.npz').write_bytes(self.input_bytes)
        (directory/'accuracy-reference.npz').write_bytes(self.reference_bytes)

    def green_function(self, base, capytaine, *, k, water_depth):
        return self._green_function(base, capytaine, k=k, water_depth=water_depth,
                                    options=dict(tabulation_nr=0, tabulation_nz=0, tabulation_cache_dir=None))

    def _green_function(self, base, capytaine, *, k, water_depth, options, acquired=None):
        manifest = json.loads(self.manifest_json)
        if k.hex() != manifest['k_hex'] or (k*water_depth).hex() != manifest['kh_hex']:
            raise CapabilityError('frozen wavenumber/depth mismatch')
        coefficients = self.coefficients
        kh_hex = manifest['kh_hex']

        class FrozenGreen(base):
            def find_best_exponential_decomposition(self, kh, method=None):
                if method not in (None, 'python') or float(kh).hex() != kh_hex:
                    raise CapabilityError('frozen evaluator refuses decomposition fallback or changed kh')
                return coefficients

        package = Path(capytaine.__file__).parent
        sources = {n: _sha((package/n).read_bytes()) for n in BACKEND_FILES}
        if sources != manifest['backend']['source_sha256']:
            raise CapabilityError('frozen accuracy profile backend implementation mismatch')
        # Resolve exactly the core selected by the pinned float64 constructor.
        core = import_module('capytaine.green_functions.Delhommeau_float64')
        if _sha(Path(core.__file__).read_bytes()) != manifest['backend']['native_library_sha256']:
            raise CapabilityError('frozen accuracy profile backend implementation mismatch')
        start = time.perf_counter()
        gf = FrozenGreen(**dict(dict(gf_singularities='low_freq', floating_point_precision='float64',
                                     finite_depth_prony_decomposition_method='python'), **options))
        gf.frozen_constructor_wall_seconds = time.perf_counter()-start
        if acquired is not None:acquired(gf)
        actual = dict(capytaine='3.0.0', native_library_sha256=_sha(Path(gf.fortran_core.__file__).read_bytes()),
                      source_sha256=sources)
        if actual != manifest['backend']:
            raise CapabilityError('frozen accuracy profile backend implementation mismatch')
        settings = gf.exportable_settings
        expected_settings = dict(dict(gf_singularities='low_freq', floating_point_precision='float64',
                                      finite_depth_prony_decomposition_method='python'), **options)
        expected_settings.pop('tabulation_cache_dir')
        for key, expected in expected_settings.items():
            if settings.get(key) != expected:
                raise CapabilityError('frozen evaluator effective settings mismatch')
        return gf

    def preserve_evaluator(self, gf, directory, runtime):
        """Untabulated route has no positive-size table evidence."""

    def evaluator_case_metadata(self, gf):
        return None

    def record_branches(self, gf, receivers, mesh, params, directory, phase):
        """No claim of table use for the untabulated profile."""

    def verify_evaluator(self, gf, directory, phase):
        """Untabulated route retains its original immutable coefficient contract."""

    def check_micro(self, gf, mesh, params, directory, runtime):
        start = time.perf_counter()
        self.record_branches(gf, self.receivers, mesh, params, directory, 'micro')
        runtime['new_green_evaluations'] += 1
        (Path(directory)/'status.json').write_text(json.dumps(dict(state='frozen_micro_running', **runtime), indent=2), encoding='utf-8')
        s, gradient = gf.evaluate(self.receivers, mesh, **params)
        runtime['micro_wall_seconds'] = time.perf_counter()-start
        np.savez(Path(directory)/'accuracy-native-micro.npz', receivers=self.receivers, s=s, gradient=gradient)
        self.verify_evaluator(gf, directory, 'after_micro')
        if s.shape != (43, 1) or gradient.shape != (3, 43, 1) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(gradient)):
            raise BackendSolveError('invalid native frozen micro; raw evidence preserved')
        scale = float(np.max(np.abs(gradient[:, 0, 0].real)))
        limit = 1e-8+1e-6*scale
        errors = np.abs(gradient[:, :, 0].T.real-self.gradient.real)
        fd = np.stack([(s[1+6*i:4+6*i, 0].real-s[4+6*i:7+6*i, 0].real)/(2*h) for i, h in enumerate(STEPS)])
        reference_fd = np.stack([(self.potential[1+6*i:4+6*i].real-self.potential[4+6*i:7+6*i].real)/(2*h) for i, h in enumerate(STEPS)])
        native_fd_error = np.max(np.abs(fd-gradient[:, 0, 0].real), axis=1)
        reference_native_fd_error = np.max(np.abs(reference_fd-gradient[:, 0, 0].real), axis=1)
        original_steps_passed = bool(np.all(reference_native_fd_error[-2:] <= limit))
        audit = dict(method='native-gradient-vs-declared-converged-reference-v1',
                     error=float(errors.max()), limit=limit, scale=scale,
                     passed=bool(np.all(errors <= limit)) and original_steps_passed, steps=STEPS,
                     reference_original_steps_passed=original_steps_passed,
                     reference_fd_vs_native_center_errors=reference_native_fd_error.tolist(),
                     native_self_fd_errors=native_fd_error.tolist(),
                     native_self_fd_passed=(native_fd_error <= limit).tolist(),
                     reference_fd_errors=np.max(np.abs(reference_fd-self.gradient[0].real), axis=1).tolist(),
                     potential_error=float(np.max(np.abs(s[:, 0]-self.potential))),
                     empirical_reference_convergence=True, producer_attested=False, qualified=False)
        np.savez(Path(directory)/'accuracy-comparison.npz', gradient_errors=errors, native_fd=fd, reference_fd=reference_fd)
        (Path(directory)/'accuracy-comparison.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
        if not audit['passed']:
            raise BackendSolveError('native Green gradient fails frozen independent-reference comparison')
        return audit


def _table_identity(gf):
    return {name: dict(shape=list(values.shape), dtype=str(values.dtype),
                       sha256=_sha(np.ascontiguousarray(values).tobytes()))
            for name in ('tabulated_r_range', 'tabulated_z_range', 'tabulated_integrals')
            for values in (getattr(gf, name),)}


@dataclass(frozen=True, init=False)
class FrozenTabulatedProfile(FrozenUntabulatedProfile):
    """Distinct, fixed finer-tabulation candidate; never selected implicitly.

    Loading reuses the immutable independent reference, not a new oracle. Actual
    generated grids/tables complete the producing case's identity at runtime.
    """

    @staticmethod
    def constructor_options():
        return dict(tabulation_nr=1352, tabulation_nz=744, tabulation_rmax=100.,
                    tabulation_zmin=-251., tabulation_nb_integration_points=1001,
                    tabulation_grid_shape='scaled_nemoh3', finite_depth_method='newer',
                    gf_singularities='low_freq', floating_point_precision='float64',
                    finite_depth_prony_decomposition_method='python', tabulation_cache_dir=None)

    @classmethod
    def load(cls, directory):
        reference = FrozenUntabulatedProfile.load(directory)
        manifest = json.loads(reference.manifest_json)
        manifest['format'] = TABULATED_FORMAT
        manifest['reference_manifest_sha256'] = _sha(reference.manifest_json.encode())
        manifest['constructor_settings'] = cls.constructor_options()
        manifest['implementation_modules'] = {name: _sha(Path(__file__).with_name(name).read_bytes())
            for name in ('panel_accuracy_reference.py', 'panel_integration_candidate.py')}
        profile = object.__new__(cls)
        for field in fields(reference):
            object.__setattr__(profile, field.name, getattr(reference, field.name))
        object.__setattr__(profile, 'manifest_json', _json(manifest))
        return profile

    def bind_case(self, identity):
        super().bind_case(identity)
        manifest = json.loads(self.manifest_json)
        actual = {n: _sha(Path(__file__).with_name(n).read_bytes()) for n in manifest['implementation_modules']}
        if manifest['constructor_settings'] != self.constructor_options() or actual != manifest['implementation_modules']:
            raise CapabilityError('finer-tabulated profile implementation/settings changed after loading')

    def green_function(self, base, capytaine, *, k, water_depth, artifact_directory, runtime):
        def acquired(gf):
            # Preserve the returned numerical payload before any post-constructor
            # settings/shape/finiteness/range check can discard failed evidence.
            start = time.perf_counter()
            runtime['new_table_constructions'] = 1
            runtime['table_constructor_wall_seconds'] = gf.frozen_constructor_wall_seconds
            arrays = {n: getattr(gf, n) for n in ('tabulated_r_range', 'tabulated_z_range', 'tabulated_integrals')
                      if isinstance(getattr(gf, n, None), np.ndarray) and not getattr(gf, n).dtype.hasobject}
            path = Path(artifact_directory)/'acquired-tabulation.npz'
            np.savez(path, **arrays)
            identities = {n: dict(shape=list(a.shape), dtype=str(a.dtype), sha256=_sha(np.ascontiguousarray(a).tobytes()))
                          for n, a in arrays.items()}
            (Path(artifact_directory)/'acquired-tabulation.json').write_text(json.dumps(dict(
                state='acquired_not_validated', actual_arrays=identities,
                archive_sha256=_sha(path.read_bytes()), constructor_wall_seconds=gf.frozen_constructor_wall_seconds),
                indent=2), encoding='utf-8')
            runtime['raw_table_acquisition_wall_seconds'] = time.perf_counter()-start
        gf = self._green_function(base, capytaine, k=k, water_depth=water_depth,
                                  options=self.constructor_options(), acquired=acquired)
        if gf.tabulation_cache_dir is not None:
            raise CapabilityError('finer-tabulated evaluator must construct without cache')
        start = time.perf_counter()
        expected_shapes = {'tabulated_r_range': (1352,), 'tabulated_z_range': (744,),
                           'tabulated_integrals': (1352, 744, 5)}
        for name, shape in expected_shapes.items():
            values = getattr(gf, name)
            if values.shape != shape or values.dtype != np.dtype('float64') or not np.all(np.isfinite(values)):
                raise CapabilityError('generated finer table shape/dtype/finite mismatch')
            setattr(gf, name, _immutable(values))
        if (not np.all(np.diff(gf.tabulated_r_range) > 0) or not np.all(np.diff(gf.tabulated_z_range) < 0)
                or gf.tabulated_r_range[0] < 0 or gf.tabulated_z_range[0] >= 0
                or not np.isclose(gf.tabulated_r_range[-1], 100., rtol=0, atol=1e-12)
                or not np.isclose(gf.tabulated_z_range[-1], -251., rtol=0, atol=1e-10)):
            raise CapabilityError('generated finer table coordinate range/ordering mismatch')
        gf.frozen_table_identity_json = _json(_table_identity(gf))
        gf.frozen_table_freezing_wall_seconds = time.perf_counter()-start
        return gf

    def preserve_evaluator(self, gf, directory, runtime):
        start = time.perf_counter()
        path = Path(directory)/'frozen-tabulation.npz'
        np.savez(path, **{n: getattr(gf, n) for n in json.loads(gf.frozen_table_identity_json)})
        runtime['table_constructor_wall_seconds'] = gf.frozen_constructor_wall_seconds
        runtime['table_freezing_wall_seconds'] = gf.frozen_table_freezing_wall_seconds
        receipt = self.evaluator_case_metadata(gf)
        receipt['archive_sha256'] = _sha(path.read_bytes())
        receipt['constructor_wall_seconds'] = gf.frozen_constructor_wall_seconds
        receipt['freezing_wall_seconds'] = gf.frozen_table_freezing_wall_seconds
        (Path(directory)/'tabulation-receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
        runtime['table_serialization_wall_seconds'] = time.perf_counter()-start
        self.verify_evaluator(gf, directory, 'before_micro')

    def evaluator_case_metadata(self, gf):
        manifest = json.loads(self.manifest_json)
        return dict(format=TABULATED_FORMAT, constructor_settings=manifest['constructor_settings'],
                    implementation_modules=manifest['implementation_modules'],
                    planned_profile_sha256=_sha(self.manifest_json.encode()),
                    actual_tables=json.loads(gf.frozen_table_identity_json), qualified=False)

    def verify_evaluator(self, gf, directory, phase):
        actual = _table_identity(gf)
        immutable = all(not getattr(gf, n).flags.writeable for n in actual)
        passed = immutable and _json(actual) == gf.frozen_table_identity_json
        (Path(directory)/('tabulation-hashes-'+phase+'.json')).write_text(
            json.dumps(dict(actual_tables=actual, immutable=immutable, passed=passed), indent=2), encoding='utf-8')
        if not passed:
            raise BackendSolveError('frozen table identity or immutability changed during acquisition')

    def record_branches(self, gf, receivers, mesh, params, directory, phase):
        self.verify_evaluator(gf, directory, 'before_'+phase)
        sources = mesh.quadrature_points[0].reshape(-1, 3)
        k, depth = params['wavenumber'], params['water_depth']
        r = k*np.linalg.norm(receivers[:, None, :2]-sources[None, :, :2], axis=2)
        counts = {}
        for receiver_reflected in (False, True):
            rz = -2*depth-receivers[:, 2] if receiver_reflected else receivers[:, 2]
            for source_reflected in (False, True):
                sz = -2*depth-sources[:, 2] if source_reflected else sources[:, 2]
                z = k*(rz[:, None]+sz[None, :])
                inside = (np.abs(z) < abs(gf.tabulated_z_range[-1])) & (r < gf.tabulated_r_range[-1])
                name = f'receiver_reflected_{receiver_reflected}_source_reflected_{source_reflected}'
                counts[name] = dict(table_predicate_rows=int(np.sum(inside)), asymptotic_predicate_rows=int(np.sum(~inside)),
                                    radial_gradient_zero_limit_rows=int(np.sum(r <= 16*np.finfo(float).eps)),
                                    r_min=float(r.min()), r_max=float(r.max()), z_min=float(z.min()), z_max=float(z.max()))
        (Path(directory)/('tabulation-branches-'+phase+'.json')).write_text(json.dumps(
            dict(counts=counts, source_derived_predicates=True, instrumented_native_trace=False,
                 rankine_and_prony_terms_are_not_table_lookups=True), indent=2), encoding='utf-8')


class PanelCandidateCancelled(BackendSolveError):
    """Cooperative cancellation; acquired evidence remains incomplete."""


def check_cancel(callback):
    if callback is not None and callback():
        raise PanelCandidateCancelled('panel integration candidate cancelled')


@dataclass(frozen=True)
class PanelOperatorProbe:
    """Terminal unqualified sample, deliberately no radiation-result interface."""
    case_hash: str
    receiver_count: int
    source_panel_count: int
    artifact_directory: str
    preparation_wall_seconds: float
    micro_wall_seconds: float
    probe_wall_seconds: float


def operator_probe(gf, mesh, points, params, count, directory, runtime, case_hash, cancel_requested, profile=None):
    if (isinstance(count, bool) or not isinstance(count, numbers.Integral) or count < 1
            or count > points.shape[0]*points.shape[1]):
        raise CapabilityError('operator probe count must be an integer within the flattened node inventory')
    count = int(count)
    indices = np.column_stack(np.unravel_index(np.arange(count), points.shape[:2]))
    receivers = points.reshape(-1, 3)[:count].copy()
    if profile is not None:profile.record_branches(gf, receivers, mesh, params, directory, 'probe')
    runtime['new_green_evaluations'] += 1
    (directory/'status.json').write_text(json.dumps(dict(state='operator_probe_running', **runtime), indent=2), encoding='utf-8')
    start = time.perf_counter()
    s, grad = gf.evaluate(receivers, mesh, **params)
    runtime['probe_wall_seconds'] = time.perf_counter()-start
    np.savez(directory/'native-operator-probe.npz', receivers=receivers, indices=indices, s=s, gradient=grad)
    if profile is not None:profile.verify_evaluator(gf, directory, 'after_probe')
    if s.shape != (count, mesh.nb_faces) or grad.shape != (3, count, mesh.nb_faces) or not np.all(np.isfinite(s)) or not np.all(np.isfinite(grad)):
        raise BackendSolveError('invalid native operator probe; raw evidence preserved')
    check_cancel(cancel_requested)
    result = PanelOperatorProbe(case_hash, count, mesh.nb_faces, str(directory),
                                runtime['preparation_wall_seconds'], runtime['micro_wall_seconds'], runtime['probe_wall_seconds'])
    runtime['probe_receiver_count'] = count
    runtime['probe_source_panel_count'] = mesh.nb_faces
    (directory/'status.json').write_text(json.dumps(dict(state='completed_probe_unqualified', **runtime), indent=2), encoding='utf-8')
    return result
