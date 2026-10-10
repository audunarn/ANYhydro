"""Versioned excitation declarations and source binding, not qualification.

The raw dataset remains authoritative. This extension does not admit a source to
native mechanics or certify backend accuracy, equilibrium or radiation data.
"""
from collections.abc import Mapping
import json
import math

import numpy as np

from .dof import DOF_NAMES
from .errors import HydroError
from .frames import require_global_dof_axes
from .hashing import dataset_content_hash
from .records import HydroBody, HydroRequest
from .schema import (validate_hydro_dataset, PHASOR_CONVENTION,
                     HEADING_CONVENTION, NORMAL_CONVENTION)

EXCITATION_SOURCE_ATTR = 'excitation_source_contract'
EXCITATION_SOURCE_VERSION = 'hydro_excitation_source_v1'
CURRENT_POLICY = 'still_water_airy_plus_prescribed_current_for_morison_drag'
_BINDING_EXCLUDES = (EXCITATION_SOURCE_ATTR, 'hydrostatic_contract')
_REQUEST_KEYS = {'omegas', 'wave_directions', 'depth_kind', 'water_depth', 'rho',
                 'gravity', 'include_zero_frequency', 'include_infinite_frequency',
                 'backend_settings'}
_CONVENTIONS = {'reference_incident_amplitude_m': 1., 'phase_origin_xy_m': [0., 0.],
                'still_water_level_m': 0., 'forward_speed_m_s': 0.,
                'phasor': PHASOR_CONVENTION, 'heading': HEADING_CONVENTION,
                'normal': NORMAL_CONVENTION, 'force_unit': 'N', 'moment_unit': 'N.m'}


class ExcitationSourceError(HydroError):
    """Missing, stale or incompatible source declarations."""


def _plain(value):
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def request_to_dict(request):
    """Persist complete request values, with explicit finite/infinite depth."""
    if type(request) is not HydroRequest:
        raise ExcitationSourceError('authoritative HydroRequest required')
    return dict(omegas=request.omegas.tolist(), wave_directions=request.wave_directions.tolist(),
                depth_kind='infinite' if math.isinf(request.water_depth) else 'finite',
                water_depth=None if math.isinf(request.water_depth) else request.water_depth,
                rho=request.rho, gravity=request.gravity,
                include_zero_frequency=request.include_zero_frequency,
                include_infinite_frequency=request.include_infinite_frequency,
                backend_settings=_plain(request.backend_settings))


def request_from_dict(data):
    """Restore exact saved request; absent sidecars never use defaults."""
    try:
        if not isinstance(data, Mapping) or set(data) != _REQUEST_KEYS:
            raise ValueError('exact saved request fields required')
        kind = data['depth_kind']
        depth = data['water_depth']
        if kind == 'infinite':
            if depth is not None:
                raise ValueError('infinite depth must have a null numeric value')
            depth = math.inf
        elif kind != 'finite' or isinstance(depth, bool) or not isinstance(depth, (int, float)) or not math.isfinite(depth):
            raise ValueError('explicit finite or infinite depth required')
        for key in ('rho', 'gravity'):
            if isinstance(data[key], bool) or not isinstance(data[key], (int, float)):
                raise ValueError(f'{key} must be numeric')
        for key in ('omegas', 'wave_directions'):
            if any(isinstance(x, (bool, np.bool_)) for x in np.asarray(data[key], dtype=object).flat):
                raise ValueError(f'{key} cannot contain booleans')
        return HydroRequest(data['omegas'], data['wave_directions'], depth,
                            data['rho'], data['gravity'], data['include_zero_frequency'],
                            data['include_infinite_frequency'], data['backend_settings'])
    except (TypeError, ValueError, HydroError, OverflowError) as error:
        raise ExcitationSourceError(f'invalid saved source request: {error}') from error


def _verify_inventory(dataset, request, bodies):
    validate_hydro_dataset(dataset)
    require_global_dof_axes(dataset)
    if type(request) is not HydroRequest or request.content_hash() != dataset.attrs['request_hash']:
        raise ExcitationSourceError('source request identity mismatch or missing sidecar')
    if not np.array_equal(request.omegas, dataset.omega.values) or not np.array_equal(request.wave_directions, dataset.wave_direction.values):
        raise ExcitationSourceError('source frequency/heading inventory mismatch')
    if request.include_zero_frequency:
        raise ExcitationSourceError('zero-frequency source is unsupported')
    try:
        bodies = tuple(bodies)
    except TypeError as error:
        raise ExcitationSourceError('authoritative HydroBody inventory required') from error
    if not bodies or any(type(b) is not HydroBody for b in bodies) or len({b.body_id for b in bodies}) != len(bodies):
        raise ExcitationSourceError('unique authoritative HydroBody inventory required')
    inventory = {b.body_id: dict(mesh_hash=b.mesh.mesh_hash,
                                lid_mesh_hash=None if b.lid_mesh is None else b.lid_mesh.mesh_hash,
                                rotation_center=list(b.rotation_center), center_of_mass=list(b.center_of_mass),
                                enabled_dofs=list(b.enabled_dofs)) for b in bodies}
    stored = dataset.attrs['body_identity']
    if isinstance(stored, str):
        stored = json.loads(stored)
    if inventory != stored or any(tuple(b.enabled_dofs) != tuple(f'{b.body_id}::{d}' for d in DOF_NAMES) for b in bodies):
        raise ExcitationSourceError('source hull/lid/reference/full six-DOF inventory mismatch')


def attach_excitation_source(dataset, request, bodies, *, convention_evidence):
    """Return a detached dataset with explicit audited caller declarations.

    Evidence identifies the convention audit; it cannot prove its scientific
    truth. Qualification status is preserved. No missing source is fabricated.
    """
    try:
        _verify_inventory(dataset, request, bodies)
    except (TypeError, ValueError, HydroError) as error:
        raise ExcitationSourceError(f'invalid source inventory: {error}') from error
    if 'hydrostatic_contract' in dataset.attrs:
        raise ExcitationSourceError('attach excitation before hydrostatic binding; explicit rebuild required')
    if EXCITATION_SOURCE_ATTR in dataset.attrs:
        raise ExcitationSourceError('source declaration already attached; explicit rebuild required')
    if not isinstance(convention_evidence, str) or not convention_evidence.strip():
        raise ExcitationSourceError('explicit convention audit evidence required')
    contract = dict(version=EXCITATION_SOURCE_VERSION,
                    dataset_binding_hash=dataset_content_hash(dataset, exclude_attrs=_BINDING_EXCLUDES),
                    binding_excludes=list(_BINDING_EXCLUDES),
                    request=request_to_dict(request), conventions=_CONVENTIONS,
                    backend={k: dataset.attrs[k] for k in ('backend', 'backend_version')},
                    convention_evidence=convention_evidence)
    result = dataset.copy(deep=True)
    result.attrs[EXCITATION_SOURCE_ATTR] = _json(contract)
    return result


def verify_excitation_source(dataset, request, bodies):
    """Verify transport/identity only; does not perform scientific admission."""
    try:
        _verify_inventory(dataset, request, bodies)
        raw = dataset.attrs.get(EXCITATION_SOURCE_ATTR)
        if not isinstance(raw, (str, Mapping)):
            raise ValueError('saved source declaration required')
        contract = json.loads(raw) if isinstance(raw, str) else json.loads(_json(raw))
        if not isinstance(contract, dict) or set(contract) != {'version', 'dataset_binding_hash', 'binding_excludes', 'request', 'conventions', 'backend', 'convention_evidence'}:
            raise ValueError('exact source declaration fields required')
        if contract['binding_excludes'] != list(_BINDING_EXCLUDES):
            raise ValueError('unsupported source binding exclusions')
        if contract['version'] != EXCITATION_SOURCE_VERSION or _json(contract['conventions']) != _json(_CONVENTIONS):
            raise ValueError('unsupported source convention/version')
        if not isinstance(contract['convention_evidence'], str) or not contract['convention_evidence'].strip():
            raise ValueError('convention evidence required')
        if contract['backend'] != {k: dataset.attrs[k] for k in ('backend', 'backend_version')}:
            raise ValueError('backend identity mismatch')
        restored = request_from_dict(contract['request'])
        if restored.content_hash() != request.content_hash():
            raise ValueError('saved source request mismatch')
        if contract['dataset_binding_hash'] != dataset_content_hash(dataset, exclude_attrs=_BINDING_EXCLUDES):
            raise ValueError('stale source dataset binding')
        return contract
    except (TypeError, ValueError, KeyError, HydroError, OverflowError) as error:
        raise ExcitationSourceError(f'excitation source verification failed: {error}') from error


def build_source_bound_excitation_series(dataset, realization, *, request, bodies, environment):
    """Verify raw source and intended environment before shared phasor synthesis.

    This returns the existing development series, not a native admission result.
    """
    verify_excitation_source(dataset, request, bodies)
    expected = dict(depth_kind='finite', water_depth=request.water_depth, rho=request.rho,
                    gravity=request.gravity, still_water_level=0., current_policy=CURRENT_POLICY)
    if not math.isfinite(request.water_depth):
        raise ExcitationSourceError('infinite-depth Airy environment is not yet admitted')
    try:
        if not isinstance(environment, Mapping) or set(environment) != set(expected):
            raise ValueError('exact environment fields required')
        normalized = dict(environment)
        for key in ('water_depth', 'rho', 'gravity', 'still_water_level'):
            value = normalized[key]
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{key} requires finite numeric data')
            normalized[key] = float(value)
        if _json(normalized) != _json(expected):
            raise ValueError('environment/source approximation mismatch')
    except (TypeError, ValueError, OverflowError) as error:
        raise ExcitationSourceError(f'invalid intended wave environment: {error}') from error
    from .excitation_series import build_excitation_series, UNIT_INCIDENT_AMPLITUDE
    return build_excitation_series(dataset, realization,
                                   amplitude_normalization=UNIT_INCIDENT_AMPLITUDE,
                                   phase_origin_xy=(0., 0.))
