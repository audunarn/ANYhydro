# SPDX-License-Identifier: MPL-2.0
"""Headless diagnostics over saved W5 panel evidence; zero native calls.

Only pinned, byte-verified saved artifacts and the portable reference bundle
are loaded. The saved native imaginary operator is compared against the
governing analytic propagating pole at the exact saved receivers/source
quadrature, the signed finite-difference defect identity between saved native
and reference potentials is recorded with the original steps and cross-checked
against the raw pinned native micro gradient/potentials, and coincident
horizontal node pairs are confirmed from saved coordinates. No new oracle,
fitting, acceptance threshold or qualification claim is introduced; the
original W5 NO_GO disposition is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .errors import CapabilityError
from .panel_accuracy_reference import (BINDING_KEYS, STEPS, TABULATED_FORMAT, FrozenTabulatedProfile,
                                       FrozenUntabulatedProfile, _arrays, _json, _sha)
from .panel_integration_candidate import imaginary_pole

SAVED_DIAGNOSTICS_FORMAT = 'anyhydro-saved-panel-diagnostics-v1'
WALL_SCOPE = 'service_invocation_only; process_wall_from_supervisor_receipt'
ORIGINAL_STEPS = STEPS[-2:]
MAX_JSON_BYTES = 4*1024*1024
MAX_NPZ_BYTES = 96*1024*1024
NEAR_ZERO_RELATIVE = 1e-12
THREAD_ENV_KEYS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
INPUT_ARRAYS = ('vertices', 'faces', 'owner', 'hull_mask', 'points', 'weights', 'normals', 'rigid_trace')
PROBE_ARRAYS = ('receivers', 'indices', 's', 'gradient')
MICRO_ARRAYS = ('receivers', 's', 'gradient')
COMPARISON_ARRAYS = ('gradient_errors', 'native_fd', 'reference_fd')
TABLE_ARRAYS = ('tabulated_r_range', 'tabulated_z_range', 'tabulated_integrals')
COMMON_REFERENCE_KEYS = ('backend', 'case_binding', 'coefficient_identity', 'files', 'k_hex',
                         'kh_hex', 'orders', 'provenance', 'steps')


def _bounded_bytes(path, maximum=MAX_NPZ_BYTES):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise CapabilityError(f'saved evidence requires a bounded regular file: {path.name}')
    if path.stat().st_size > maximum:
        raise CapabilityError(f'saved evidence file exceeds bounded admission: {path.name}')
    return path.read_bytes()


def _load_json(data, name):
    try:
        return json.loads(data.decode('utf-8'),
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    except (ValueError, UnicodeDecodeError) as exc:
        raise CapabilityError(f'invalid saved evidence JSON: {name}') from exc


def _case_hash(case):
    identity = {key: value for key, value in case.items() if key != 'case_hash'}
    return hashlib.sha256(_json(identity).encode()).hexdigest()


def _identity(values):
    return dict(shape=list(values.shape), dtype=str(values.dtype),
                sha256=_sha(np.ascontiguousarray(values).tobytes()))


def _signed_stats(values):
    values = np.asarray(values)
    stats = dict(count=int(values.size))
    if values.size:
        if np.iscomplexobj(values):
            stats.update(max_abs=float(np.max(np.abs(values))),
                         rms=float(np.sqrt(np.mean(np.abs(values)**2))),
                         l2=float(np.linalg.norm(values)),
                         signed_min_real=float(values.real.min()),
                         signed_max_real=float(values.real.max()),
                         signed_min_imag=float(values.imag.min()),
                         signed_max_imag=float(values.imag.max()))
        else:
            stats.update(max_abs=float(np.max(np.abs(values))), rms=float(np.sqrt(np.mean(values**2))),
                          l2=float(np.linalg.norm(values)), signed_min=float(values.min()),
                          signed_max=float(values.max()))
    return stats


def _atomic_bytes(path, data):
    temporary = path.with_name(path.name+'.partial')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _atomic_npz(path, **arrays):
    temporary = path.with_name(path.name+'.partial.npz')
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def _coincident_panels(receivers, points):
    flat = points.reshape(-1, 3)
    horizontal = np.linalg.norm(receivers[:, None, :2]-flat[None, :, :2], axis=2)
    coincident_nodes = horizontal == 0.0
    coincident_panels = np.zeros((len(receivers), points.shape[0]), dtype=bool)
    for receiver_index in range(len(receivers)):
        coincident_panels[receiver_index, np.unique(np.flatnonzero(coincident_nodes[receiver_index])
                                                    // points.shape[1])] = True
    return coincident_panels


def _validate_pinned_inputs(inputs, identities):
    pinned = {}
    for key, digest in identities.items():
        relative = key.replace('\\', '/')
        data = _bounded_bytes(inputs/relative)
        if _sha(data) != digest:
            raise CapabilityError(f'pinned saved input hash mismatch: {relative}')
        pinned[relative] = data
    return pinned


def _validate_case_and_arrays(pinned, inputs):
    case = _load_json(pinned['case.json'], 'case.json')
    arrays = _arrays(pinned['input.npz'], INPUT_ARRAYS)
    for name, declared in case.get('snapshot_arrays', {}).items():
        if name not in arrays or _identity(arrays[name]) != declared:
            raise CapabilityError(f'saved input array differs from producing case snapshot: {name}')
    if _case_hash(case) != case.get('case_hash'):
        raise CapabilityError('saved producing case hash mismatch')
    vertices, faces = arrays['vertices'], arrays['faces']
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3
            or faces.dtype.kind not in 'iu' or np.any(faces < 0) or np.any(faces >= len(vertices))
            or arrays['points'].shape != (len(faces), 4, 3) or arrays['weights'].shape != (len(faces), 4)
            or np.any(arrays['weights'] <= 0) or arrays['hull_mask'].dtype != np.dtype('bool')
            or arrays['owner'].dtype.kind not in 'iu' or arrays['normals'].shape != (len(faces), 3)):
        raise CapabilityError('saved input arrays are not a consistent quadrature model')
    return case, arrays


def _validate_probe(pinned, arrays):
    probe = _arrays(pinned['native-operator-probe.npz'], PROBE_ARRAYS)
    receivers, indices, s, gradient = (probe[name] for name in PROBE_ARRAYS)
    flat = arrays['points'].reshape(-1, 3)
    if (receivers.ndim != 2 or receivers.shape[1] != 3 or receivers.dtype != np.dtype('float64')
            or indices.shape != (len(receivers), 2) or indices.dtype.kind not in 'iu'
            or not np.array_equal(receivers, flat[:len(receivers)])
            or not np.array_equal(indices, np.column_stack(np.unravel_index(
                np.arange(len(receivers)), arrays['points'].shape[:2])))
            or s.shape != (len(receivers), len(arrays['faces']))
            or gradient.shape != (3, len(receivers), len(arrays['faces']))
            or s.dtype != np.dtype('complex128') or gradient.dtype != np.dtype('complex128')):
        raise CapabilityError('saved native operator probe is inconsistent with saved quadrature')
    return probe


def _coincident_inventory(receivers, points, hull_mask, owner):
    flat = points.reshape(-1, 3)
    nodes_per_panel = points.shape[1]
    horizontal = np.linalg.norm(receivers[:, None, :2]-flat[None, :, :2], axis=2)
    coincident_nodes = horizontal == 0.0
    coincident_panels = np.zeros((len(receivers), points.shape[0]), dtype=bool)
    inventory = []
    for receiver_index in range(len(receivers)):
        nodes = np.flatnonzero(coincident_nodes[receiver_index])
        coincident_panels[receiver_index, np.unique(nodes//nodes_per_panel)] = True
        inventory.append(dict(
            receiver_index=receiver_index,
            receiver=[float(v) for v in receivers[receiver_index]],
            coincident_node_count=int(len(nodes)),
            coincident_nodes=[dict(flat_node=int(node), panel=int(node//nodes_per_panel),
                                   quadrature=int(node % nodes_per_panel), z=float(flat[node, 2]),
                                   hull=bool(hull_mask[node//nodes_per_panel]), owner=int(owner[node//nodes_per_panel]))
                              for node in nodes],
            coincident_panels=[int(panel) for panel in np.flatnonzero(coincident_panels[receiver_index])]))
    totals = dict(receivers=int(len(receivers)), coincident_node_rows=int(coincident_nodes.sum()),
                  coincident_panel_entries=int(coincident_panels.sum()),
                  receivers_with_two_coincident_nodes=int(sum(1 for entry in inventory
                                                              if entry['coincident_node_count'] == 2)))
    return dict(inventory=inventory, totals=totals), coincident_nodes, coincident_panels, horizontal


def _imaginary_pole_comparison(receivers, points, weights, native_s, native_gradient, k, water_depth):
    pole, pole_gradient = imaginary_pole(receivers, points, weights, k=k, water_depth=water_depth)
    native_imag = native_s.imag
    difference = pole-native_imag
    gradient_difference = pole_gradient-native_gradient.imag
    scale = float(np.max(np.abs(native_imag)))
    near_zero = np.abs(native_imag) <= NEAR_ZERO_RELATIVE*scale
    significant = ~near_zero
    coincident_panels = _coincident_panels(receivers, points)
    coincident_significant = coincident_panels & significant
    sign_reversals = int(np.sum(np.sign(pole[coincident_significant])
                                 * np.sign(native_imag[coincident_significant]) < 0)) if np.any(coincident_significant) else 0
    relative = np.abs(difference)[significant]/np.abs(native_imag)[significant]
    report = dict(
        method='saved-native-imaginary-vs-analytic-pole-v1',
        receivers=int(len(receivers)), source_panels=int(points.shape[0]),
        source_quadrature_nodes=int(points.shape[0]*points.shape[1]),
        normalization_scale_max_abs_native_imag=scale,
        near_zero_relative_threshold=NEAR_ZERO_RELATIVE,
        near_zero_native_entries=int(near_zero.sum()), significant_native_entries=int(significant.sum()),
        value_difference=_signed_stats(difference),
        value_difference_noncoincident=_signed_stats(difference[~coincident_panels]),
        value_difference_coincident=_signed_stats(difference[coincident_panels]),
        value_native=_signed_stats(native_imag), value_pole=_signed_stats(pole),
        gradient_difference=_signed_stats(gradient_difference),
        max_abs_relative_difference_on_significant_entries=float(np.max(relative)) if relative.size else 0.0,
        coincident_sign_reversals_among_significant=sign_reversals,
        acceptance_threshold=None, retroactive_imaginary_acceptance=False,
        known_coincident_row_sign_reversals_are_themselves_diagnostic=True)
    raw = dict(receivers=receivers, native_imag=native_imag, pole=pole, difference=difference,
               native_gradient_imag=native_gradient.imag, pole_gradient=pole_gradient,
               gradient_difference=gradient_difference, near_zero_mask=near_zero,
               coincident_panels=coincident_panels)
    return report, raw


def _validate_micro(pinned, profile):
    if 'accuracy-native-micro.npz' not in pinned:
        raise CapabilityError('saved evidence requires the pinned raw native micro packet')
    micro = _arrays(pinned['accuracy-native-micro.npz'], MICRO_ARRAYS)
    receivers, s, gradient = (micro[name] for name in MICRO_ARRAYS)
    if (receivers.shape != (43, 3) or receivers.dtype != np.dtype('float64')
            or not np.array_equal(receivers, profile.receivers)
            or s.shape != (43, 1) or s.dtype != np.dtype('complex128')
            or gradient.shape != (3, 43, 1) or gradient.dtype != np.dtype('complex128')
            or not np.all(np.isfinite(s)) or not np.all(np.isfinite(gradient))):
        raise CapabilityError('saved raw native micro packet is inconsistent with the reference receivers')
    return micro


def _fd_defect_identity(comparison, profile, audit, micro):
    native_fd, reference_fd = comparison['native_fd'], comparison['reference_fd']
    reference_center = profile.gradient[0].real
    native_s, native_gradient = micro['s'], micro['gradient']
    native_center = native_gradient[:, 0, 0].real
    recomputed_native_fd = np.stack([(native_s[1+6*i:4+6*i, 0].real-native_s[4+6*i:7+6*i, 0].real)/(2*h)
                                     for i, h in enumerate(STEPS)])
    recomputed_reference_fd = np.stack([(profile.potential[1+6*i:4+6*i].real
                                        - profile.potential[4+6*i:7+6*i].real)/(2*h)
                                        for i, h in enumerate(STEPS)])
    recomputed_native_self_fd = np.max(np.abs(recomputed_native_fd-native_center), axis=1)
    recomputed_reference_fd_errors = np.max(np.abs(recomputed_reference_fd-reference_center), axis=1)
    recomputed_reference_vs_native_center = np.max(np.abs(recomputed_reference_fd-native_center), axis=1)
    recomputed_gradient_error = float(np.max(np.abs(native_gradient[:, :, 0].T.real-profile.gradient.real)))
    recomputed_potential_error = float(np.max(np.abs(native_s[:, 0]-profile.potential)))
    recomputed_scale = float(np.max(np.abs(native_center)))
    steps = list(audit['steps'])
    if (steps != list(STEPS) or native_fd.shape != (len(steps), 3) or reference_fd.shape != (len(steps), 3)
            or not np.array_equal(recomputed_native_fd, native_fd)
            or not np.array_equal(recomputed_reference_fd, reference_fd)
            or not np.array_equal(recomputed_native_self_fd, np.array(audit['native_self_fd_errors']))
            or not np.array_equal(recomputed_reference_fd_errors, np.array(audit['reference_fd_errors']))
            or not np.array_equal(recomputed_reference_vs_native_center,
                                  np.array(audit['reference_fd_vs_native_center_errors']))
            or recomputed_gradient_error != audit['error']
            or recomputed_potential_error != audit['potential_error']
            or recomputed_scale != audit['scale']):
        raise CapabilityError('recorded finite-difference audit is inconsistent with the raw saved native micro packet')
    native_minus_reference = native_fd-reference_fd
    reference_minus_gradient = reference_fd-reference_center
    native_minus_gradient = native_fd-reference_center
    native_minus_native_gradient = native_fd-native_center
    reference_minus_native_gradient = reference_fd-native_center
    native_gradient_minus_reference = native_center-reference_center
    native_potential_minus_reference = native_s[:, 0]-profile.potential
    original_indices = [steps.index(step) for step in ORIGINAL_STEPS]
    report = dict(
        method='saved-native-reference-fd-signed-defect-identity-v1',
        steps=steps, original_steps=list(ORIGINAL_STEPS), original_step_indices=original_indices,
        raw_native_micro_recomputed_fd=True,
        recorded_audit_cross_checked_against_raw_native_micro=True,
        native_potential_fd_minus_reference_potential_fd=_signed_stats(native_minus_reference),
        reference_potential_fd_minus_reference_gradient=_signed_stats(reference_minus_gradient),
        native_potential_fd_minus_reference_gradient=_signed_stats(native_minus_gradient),
        native_potential_fd_minus_native_gradient_center=_signed_stats(native_minus_native_gradient),
        reference_potential_fd_minus_native_gradient_center=_signed_stats(reference_minus_native_gradient),
        native_gradient_center_minus_reference_gradient=_signed_stats(native_gradient_minus_reference),
        native_potential_minus_reference_potential=_signed_stats(native_potential_minus_reference),
        per_step_max_abs=dict(
            native_minus_reference=[float(v) for v in np.max(np.abs(native_minus_reference), axis=1)],
            reference_minus_gradient=[float(v) for v in np.max(np.abs(reference_minus_gradient), axis=1)],
            native_minus_gradient=[float(v) for v in np.max(np.abs(native_minus_gradient), axis=1)],
            native_minus_native_gradient=[float(v) for v in np.max(np.abs(native_minus_native_gradient), axis=1)],
            reference_minus_native_gradient=[float(v) for v in np.max(np.abs(reference_minus_native_gradient), axis=1)]),
        original_step_max_abs_native_minus_reference=[float(np.max(np.abs(native_minus_reference[i])))
                                                      for i in original_indices],
        recorded_audit=dict(native_self_fd_errors=audit['native_self_fd_errors'],
                            reference_fd_vs_native_center_errors=audit['reference_fd_vs_native_center_errors'],
                            reference_fd_errors=audit['reference_fd_errors'],
                            potential_error=audit['potential_error'],
                            gradient_error=audit['error'], limit=audit['limit'], scale=audit['scale'],
                            passed=audit['passed']),
        distinctions=dict(
            native_potential_fd='finite differences of the raw saved native micro potential S at the original steps',
            reference_potential_fd='finite differences of declared-converged reference potential',
            reference_gradient='declared-converged reference gradient array (row 0 = center)',
            native_gradient='raw saved native center gradient from the pinned native micro packet',
            native_vs_reference_components_are_distinguished=True),
        unique_cause_claimed=False,
        attribution_note='Signed defects are recorded without causal attribution; the earlier untabulated '
                         'self-FD comparison also failed, so a unique table-interpolation cause is unsupported.',
        acceptance_threshold=None)
    raw = dict(steps=np.array(steps), native_fd=native_fd, reference_fd=reference_fd,
               recomputed_native_fd=recomputed_native_fd,
               recomputed_reference_fd=recomputed_reference_fd,
               reference_gradient_center=profile.gradient[0],
               native_gradient_center=native_gradient[:, 0, 0],
               native_minus_reference_fd=native_minus_reference,
               reference_fd_minus_reference_gradient=reference_minus_gradient,
               native_fd_minus_reference_gradient=native_minus_gradient,
               native_fd_minus_native_gradient=native_minus_native_gradient,
               reference_fd_minus_native_gradient=reference_minus_native_gradient,
               native_gradient_minus_reference_gradient=native_gradient_minus_reference,
               native_potential_minus_reference_potential=native_potential_minus_reference,
               gradient_errors=comparison['gradient_errors'])
    return report, raw


def run_saved_panel_diagnostics(inputs_directory, output_directory, identities_path=None):
    """Validate pinned saved evidence and produce headless comparison artifacts.

    Makes zero native Green calls, constructs no backend, fits nothing and
    introduces no acceptance threshold. All outputs are written atomically;
    a refusal preserves a failure report with completed stages.
    """
    started = time.perf_counter()
    inputs = Path(inputs_directory)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    identities_path = Path(identities_path) if identities_path is not None else inputs.parent/'input-identities.json'
    stages = []
    report = dict(format=SAVED_DIAGNOSTICS_FORMAT, state='running', native_calls=0, new_oracle=False,
                 qualified=False, inputs_directory=str(inputs), output_directory=str(output),
                 effective_thread_env={key: os.environ.get(key) for key in THREAD_ENV_KEYS})
    try:
        identities = _load_json(_bounded_bytes(identities_path, MAX_JSON_BYTES), 'input-identities.json')
        if not isinstance(identities, dict) or not identities:
            raise CapabilityError('saved input identities must be a nonempty mapping')
        stages.append('identities_loaded')
        pinned = _validate_pinned_inputs(inputs, identities)
        stages.append('pinned_bytes_verified')
        case, arrays = _validate_case_and_arrays(pinned, inputs)
        stages.append('case_and_input_arrays_validated')
        probe = _validate_probe(pinned, arrays)
        stages.append('probe_validated')
        comparison = _arrays(pinned['accuracy-comparison.npz'], COMPARISON_ARRAYS)
        audit = _load_json(pinned['accuracy-comparison.json'], 'accuracy-comparison.json')
        if (audit.get('method') != 'native-gradient-vs-declared-converged-reference-v1'
                or comparison['gradient_errors'].shape != (43, 3)):
            raise CapabilityError('saved accuracy comparison audit is inconsistent')
        stages.append('comparison_validated')
        profile = FrozenUntabulatedProfile.load(inputs/'reference-bundle')
        reference_manifest = json.loads(profile.manifest_json)
        producer_profile = _load_json(pinned['accuracy-profile.json'], 'accuracy-profile.json')
        if (producer_profile.get('format') != TABULATED_FORMAT
                or producer_profile.get('reference_manifest_sha256') != _sha(profile.manifest_json.encode())
                or any(producer_profile.get(key) != reference_manifest[key] for key in COMMON_REFERENCE_KEYS)
                or producer_profile.get('constructor_settings') != FrozenTabulatedProfile.constructor_options()
                or not isinstance(producer_profile.get('implementation_modules'), dict)
                or not producer_profile['implementation_modules']):
            raise CapabilityError('preserved producing accuracy profile is not the tabulated producer '
                                  'descriptor of the independent reference bundle')
        if (pinned['accuracy-input.npz'] != profile.input_bytes
                or pinned['accuracy-reference.npz'] != profile.reference_bytes):
            raise CapabilityError('preserved accuracy bundle copies differ from the reference bundle')
        stages.append('reference_bundle_validated')
        receipt = _load_json(pinned['tabulation-receipt.json'], 'tabulation-receipt.json')
        tables = _arrays(pinned['frozen-tabulation.npz'], TABLE_ARRAYS)
        if (receipt.get('format') != TABULATED_FORMAT
                or receipt.get('constructor_settings') != FrozenTabulatedProfile.constructor_options()
                or receipt.get('constructor_settings') != producer_profile['constructor_settings']
                or receipt.get('implementation_modules') != producer_profile['implementation_modules']
                or receipt.get('planned_profile_sha256') != _sha(pinned['accuracy-profile.json'])
                or receipt.get('archive_sha256') != _sha(pinned['frozen-tabulation.npz'])
                or {name: _identity(values) for name, values in tables.items()} != receipt.get('actual_tables')
                or case.get('actual_tabulation', {}).get('actual_tables') != receipt.get('actual_tables')
                or case.get('accuracy_profile_sha256') != receipt.get('planned_profile_sha256')
                or case.get('actual_tabulation', {}).get('format') != TABULATED_FORMAT
                or any(case.get(key) != reference_manifest['case_binding'][key] for key in BINDING_KEYS)):
            raise CapabilityError('saved tabulation receipt/case/table identities are inconsistent')
        stages.append('tabulation_receipt_validated')
        status = _load_json(pinned['status.json'], 'status.json')
        if (status.get('state') != 'completed_probe_unqualified'
                or status.get('numerical_systems') != 0
                or status.get('probe_receiver_count') != len(probe['receivers'])
                or status.get('probe_source_panel_count') != len(arrays['faces'])):
            raise CapabilityError('saved status is not the pinned completed terminal probe')
        stages.append('status_validated')
        manifest = json.loads(profile.manifest_json)
        k = float.fromhex(manifest['k_hex'])
        water_depth = case['water_depth']
        if k <= 0 or not np.isfinite(k) or (k*water_depth).hex() != manifest['kh_hex']:
            raise CapabilityError('saved wavenumber/depth identity mismatch')
        stages.append('parameters_validated')
        micro = _validate_micro(pinned, profile)
        stages.append('native_micro_validated')

        inventory, _, _, _ = _coincident_inventory(probe['receivers'], arrays['points'],
                                                   arrays['hull_mask'], arrays['owner'])
        stages.append('coincident_inventory_confirmed')
        imaginary_report, imaginary_raw = _imaginary_pole_comparison(
            probe['receivers'], arrays['points'], arrays['weights'], probe['s'], probe['gradient'],
            k, water_depth)
        stages.append('imaginary_pole_compared')
        fd_report, fd_raw = _fd_defect_identity(comparison, profile, audit, micro)
        stages.append('fd_defect_identity_recorded')

        _atomic_npz(output/'imaginary-pole-comparison.npz', **imaginary_raw)
        _atomic_npz(output/'fd-signed-defects.npz', **fd_raw)
        _atomic_bytes(output/'coincident-inventory.json',
                      json.dumps(inventory, indent=2, allow_nan=False).encode('utf-8'))
        elapsed = time.perf_counter()-started
        report.update(state='completed', completed_stages=stages,
                      validation=dict(pinned_files=sorted(pinned), case_hash_recomputed=True,
                                      reference_bundle_format=manifest['format'],
                                      status_state=status['state']),
                      imaginary_pole=imaginary_report, fd_defects=fd_report, coincident=inventory['totals'],
                      service_wall_seconds=elapsed, wall_scope=WALL_SCOPE,
                      process_wall_seconds=elapsed,
                      process_wall_seconds_is_legacy_service_alias=True)
        _atomic_bytes(output/'report.json', json.dumps(report, indent=2, allow_nan=False).encode('utf-8'))
        return report
    except Exception as exc:
        elapsed = time.perf_counter()-started
        failure = dict(format=SAVED_DIAGNOSTICS_FORMAT, state='failed',
                        error_type=type(exc).__name__, error=str(exc), completed_stages=stages,
                        native_calls=0, qualified=False,
                        service_wall_seconds=elapsed, wall_scope=WALL_SCOPE,
                        process_wall_seconds=elapsed,
                        process_wall_seconds_is_legacy_service_alias=True)
        _atomic_bytes(output/'failure-report.json', json.dumps(failure, indent=2, allow_nan=False).encode('utf-8'))
        if isinstance(exc, CapabilityError):
            raise
        raise CapabilityError(f'saved panel diagnostics failed: {exc}') from exc
