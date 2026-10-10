# SPDX-License-Identifier: MPL-2.0
"""Strict standard-WAMIT7 coefficient import; transport is not qualification.

ANYfileIO owns literal syntax. This owner converts declared standard period,
heading, phasor, dimensions and reference frames into the canonical dataset.
No solver is launched, coefficient repaired, or missing entry synthesized.
"""
from dataclasses import dataclass, replace
from collections.abc import Mapping
from decimal import Decimal
import hashlib
import math
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from .dof import body_dof_keys
from .errors import CapabilityError, HydroError
from .excitation_source import attach_excitation_source
from .frames import COEFFICIENT_FRAMES_ATTR, CoefficientFrame, reframe_dataset
from .hydrostatic_contract import HydrostaticContract, attach_hydrostatic_contract
from .records import HydroBody, HydroRequest
from .schema import assemble_hydro_dataset

__all__ = ['WamitImportError', 'Wamit7ImportContract', 'import_wamit7']
PROFILE = 'wamit7_period_exp_plus_i_v1'


class WamitImportError(HydroError):
    """Unbound, incomplete, inconsistent or unsupported coefficient import."""
    def __init__(self, message, *, diagnostics=()):
        super().__init__(message)
        self.diagnostics = tuple(diagnostics)


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise WamitImportError(f'{name} requires an explicit SHA-256 hex digest')
    return value


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise WamitImportError(f'{name} must be positive finite')
    return float(value)


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise WamitImportError(f'{name} requires an explicit mapping')
    return dict(value)


def _scale(value):
    if not math.isfinite(value) or value <= 0:
        raise WamitImportError('Unrepresentable positive dimensional scale; no zero/inf coefficient replacement')
    return np.float64(value)


def _dimensional_scale(rho, length, exponent, gravity=None):
    # Entire construction belongs inside the caller's NumPy error context:
    # a subnormal power must not be rounded then amplified by a huge density.
    value = np.multiply(np.float64(rho), np.power(np.float64(length), exponent))
    if gravity is not None:
        value = np.multiply(value, np.float64(gravity))
    return _scale(value)


@dataclass(frozen=True)
class _CoefficientImportContract:
    """Explicit immutable source declarations for one standard-WAMIT7 case.

    Frames and CB are WORLD records. The table's generalized DOF axes are the
    declared source frames. Unknown hydrostatic baseline/role stays unadmitted.
    Source hashes establish binding, not independent truth of declarations.
    """
    source_body_order: tuple
    source_frames: dict
    source_mesh_hashes: dict
    source_lid_hashes: dict
    center_of_buoyancy: dict
    reference_length_m: float
    source_rho: float
    source_gravity: float
    source_water_depth: float
    source_request_hash: str
    input_sha256: str
    executable_sha256: str
    file_sha256: dict
    producer_version: str
    producer_build: str
    convention_evidence: str
    profile: str = PROFILE
    _SUPPORTED_PROFILES: ClassVar[tuple] = (PROFILE,)

    def __post_init__(self):
        if not isinstance(self.source_body_order, (tuple, list)):
            raise WamitImportError('Explicit source body order sequence required')
        order = tuple(self.source_body_order)
        if not order or any(not isinstance(x, str) or not x for x in order) or len(set(order)) != len(order):
            raise WamitImportError('Unique explicit source body order required')
        if self.profile not in self._SUPPORTED_PROFILES:
            raise CapabilityError('Only standard WAMIT7 period/exp(+i omega t) profile is supported; no HAMS profile or phase repair')
        object.__setattr__(self, 'source_body_order', order)
        for field in ('source_frames', 'source_mesh_hashes', 'source_lid_hashes', 'center_of_buoyancy'):
            values = _mapping(getattr(self, field), field)
            if set(values) != set(order):
                raise WamitImportError(f'{field} must cover exactly source bodies')
            if field == 'source_frames':
                result = {}
                for key, value in values.items():
                    if not isinstance(value, CoefficientFrame):
                        raise WamitImportError('Source frames require public CoefficientFrame records')
                    frame = CoefficientFrame(value.axes, value.reference_center)
                    if not np.allclose(frame.axes[:, 2], [0, 0, 1], rtol=0, atol=1e-10):
                        raise CapabilityError('Standard WAMIT body axes must have vertical positive z')
                    result[key] = frame
                values = result
            elif field == 'center_of_buoyancy':
                result = {}
                for key, value in values.items():
                    arr = np.asarray(value, dtype=object)
                    if arr.shape != (3,) or any(isinstance(x, (bool, np.bool_)) for x in arr):
                        raise WamitImportError('Explicit finite WORLD center of buoyancy required')
                    try:
                        arr = np.asarray(value, dtype=float)
                    except (TypeError, ValueError, OverflowError) as error:
                        raise WamitImportError('Explicit finite numeric WORLD center of buoyancy required') from error
                    if not np.all(np.isfinite(arr)):
                        raise WamitImportError('Explicit finite WORLD center of buoyancy required')
                    result[key] = tuple(float(x) for x in arr)
                values = result
            else:
                values = {key: (_digest(value, field) if value is not None or field == 'source_mesh_hashes' else None)
                          for key, value in values.items()}
            object.__setattr__(self, field, MappingProxyType(values))
        for field in ('reference_length_m', 'source_rho', 'source_gravity'):
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        depth = self.source_water_depth
        if isinstance(depth, bool) or not isinstance(depth, (int, float)) or math.isnan(depth) or depth <= 0:
            raise WamitImportError('Source depth must be positive finite or +inf')
        object.__setattr__(self, 'source_water_depth', float(depth))
        for field in ('source_request_hash', 'input_sha256', 'executable_sha256'):
            _digest(getattr(self, field), field)
        hashes = _mapping(self.file_sha256, 'file_sha256')
        if set(hashes) != {'1', '3', 'hst'}:
            raise WamitImportError('Explicit hashes required for .1/.3/.hst files')
        object.__setattr__(self, 'file_sha256', MappingProxyType({k: _digest(v, k) for k, v in hashes.items()}))
        for field in ('producer_version', 'producer_build', 'convention_evidence'):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise WamitImportError(f'{field} must be explicit nonempty text')


@dataclass(frozen=True)
class Wamit7ImportContract(_CoefficientImportContract):
    """Explicit standard-WAMIT7 period/exp(+i omega t) import contract."""


def _syntax_api():
    try:
        from anyfileio.wamit import parse_wamit, inventory_diagnostics, excitation_diagnostics, require_valid, WamitError
    except ImportError as error:
        raise CapabilityError('WAMIT import needs the optional anyhydro[wamit] dependency with anyfileio.wamit available') from error
    return parse_wamit, inventory_diagnostics, excitation_diagnostics, require_valid, WamitError


def _map_coordinates(document, targets, *, heading=False, allowed_limits=(Decimal(0),)):
    unique = {}
    for row in document.records:
        number = row.heading if heading else row.frequency
        if not heading and number.decimal in allowed_limits:
            continue
        if not heading and number.decimal < 0:
            raise CapabilityError('Negative-period zero-frequency limit is unsupported by canonical persistence')
        if not heading and number.value <= 0:
            raise WamitImportError('Finite period must be representable and positive')
        unique.setdefault(number.decimal, []).append(number)
    mapped = {}
    for coordinate, numbers in unique.items():
        candidates = []
        for index, target in enumerate(targets):
            errors = [abs(math.remainder(n.value - target, 360)) if heading else abs(n.value - target) for n in numbers]
            if all(error <= n.rounding_bound + 8 * math.ulp(target) for error, n in zip(errors, numbers)):
                candidates.append(index)
        if len(candidates) != 1 or candidates[0] in mapped.values():
            raise WamitImportError(f'{document.source}: missing, foreign or ambiguous printed coordinate {coordinate}')
        mapped[coordinate] = candidates[0]
    if set(mapped.values()) != set(range(len(targets))):
        raise WamitImportError(f'{document.source}: incomplete requested coordinate grid')
    return mapped


def _import_coefficients(bodies, request, contract, *, radiation_path, excitation_path, hydrostatic_path, hams=False, source_provenance=None):
    """Return a detached, unqualified canonical dataset; never write or solve.

    All files are hashed before parsing the exact captured bytes. Only the
    positive requested grid and genuine period-zero Ainf are admitted. No A0,
    frequency interpolation, phase repair, zero-fill or symmetrization exists.
    """
    bodies = tuple(bodies)
    if not bodies or any(type(b) is not HydroBody for b in bodies) or type(request) is not HydroRequest or not isinstance(contract, _CoefficientImportContract):
        raise WamitImportError('Authoritative HydroBody/HydroRequest/import contract records required')
    ids = tuple(b.body_id for b in bodies)
    if len(set(ids)) != len(ids) or set(ids) != set(contract.source_body_order):
        raise WamitImportError('Source/caller body identity mismatch')
    if request.include_zero_frequency:
        raise CapabilityError('Canonical dataset cannot retain the zero-frequency limit; it is not silently dropped')
    if request.content_hash() != contract.source_request_hash:
        raise WamitImportError('Source request hash mismatch')
    if (request.rho, request.gravity, request.water_depth) != (contract.source_rho, contract.source_gravity, contract.source_water_depth):
        raise WamitImportError('Source/target fluid properties differ; no implicit similarity transformation')
    for body in bodies:
        if tuple(body.enabled_dofs) != body_dof_keys(body.body_id) or body.attached_surfaces:
            raise CapabilityError('Standard import requires full rigid six-DOF hull bodies; sheets/generalized modes unsupported')
        if body.mesh.mesh_hash != contract.source_mesh_hashes[body.body_id] or (None if body.lid_mesh is None else body.lid_mesh.mesh_hash) != contract.source_lid_hashes[body.body_id]:
            raise WamitImportError('Source hull/lid identity mismatch')
    parse, inventory, excitation_check, validate, FormatError = _syntax_api()
    paths = {'1': Path(radiation_path), '3': Path(excitation_path), 'hst': Path(hydrostatic_path)}
    docs = {}
    try:
        for kind, path in paths.items():
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != contract.file_sha256[kind]:
                raise WamitImportError(f'{path}: source file hash mismatch')
            docs[kind] = parse(data.decode('ascii'), kind=kind, source=str(path))
        # Refuse inconsistent source representation before any physical conversion.
        excitation_issues = excitation_check(docs['3'])
        validate(docs['3'], excitation_issues)
    except FormatError as error:
        raise WamitImportError(str(error), diagnostics=error.diagnostics) from error
    except (OSError, UnicodeError) as error:
        raise WamitImportError(f'Cannot capture authoritative coefficient files: {error}') from error
    periods = [float(omega) if hams else 2 * math.pi / float(omega) for omega in request.omegas]
    headings = [math.degrees(theta) for theta in request.wave_directions]
    if not all(math.isfinite(x) and x > 0 for x in periods) or not all(math.isfinite(x) for x in headings):
        raise WamitImportError('Unrepresentable source period/heading coordinates')
    frequency = {kind: _map_coordinates(docs[kind], periods, allowed_limits=((Decimal(0),Decimal(-1)) if hams and request.include_infinite_frequency and kind=='1' else (Decimal(0),))) for kind in ('1', '3')}
    directions = _map_coordinates(docs['3'], headings, heading=True)
    size = 6 * len(bodies)
    radiation_keys = [(f, i, j) for f in frequency['1'] for i in range(1, size + 1) for j in range(1, size + 1)]
    limit_coordinates=(Decimal(0),Decimal(-1)) if hams else (Decimal(0),)
    if request.include_infinite_frequency:
        radiation_keys += [(f, i, j) for f in limit_coordinates for i in range(1, size + 1) for j in range(1, size + 1)]
    excitation_keys = [(f, h, i) for f in frequency['3'] for h in directions for i in range(1, size + 1)]
    try:
        validate(docs['1'], inventory(docs['1'], expected_keys=radiation_keys))
        validate(docs['3'], inventory(docs['3'], expected_keys=excitation_keys))
        validate(docs['hst'], inventory(docs['hst'], expected_keys=[(i,j) for i in range(1,size+1) for j in range(1,size+1)]))
    except FormatError as error:
        raise WamitImportError(str(error), diagnostics=error.diagnostics) from error
    source_to_target = [ids.index(body) * 6 + dof for body in contract.source_body_order for dof in range(6)]
    A = np.empty((len(periods), size, size)); B = np.empty_like(A)
    F = np.empty((len(periods), len(headings), size), dtype=complex); C = np.empty((size,size))
    Ainf = np.empty_like(C) if request.include_infinite_frequency else None
    L, rho, g = contract.reference_length_m, request.rho, request.gravity
    try:
        with np.errstate(over='raise', under='raise', invalid='raise', divide='raise'):
            for row in docs['1'].records:
                i,j = row.influenced-1,row.radiating-1
                ti,tj = source_to_target[i],source_to_target[j]
                factor = _scale(rho) if hams else _dimensional_scale(rho,L,3 + int(i % 6 >= 3) + int(j % 6 >= 3))
                if row.frequency.decimal in limit_coordinates:
                    if row.damping is not None and (hams or row.damping.decimal != 0):
                        raise WamitImportError('Computed HAMS limit damping or nonzero WAMIT limit damping unsupported')
                    if not hams or row.frequency.decimal == -1:
                        Ainf[ti,tj] = np.multiply(row.added_mass.value, factor)
                else:
                    if row.damping is None:
                        raise WamitImportError('Finite-frequency radiation damping is missing')
                    k = frequency['1'][row.frequency.decimal]
                    A[k,ti,tj] = np.multiply(row.added_mass.value, factor)
                    B[k,ti,tj] = np.multiply(np.multiply(row.damping.value, factor), request.omegas[k])
            for row in docs['3'].records:
                i=row.influenced-1;ti=source_to_target[i]
                k=frequency['3'][row.frequency.decimal];h=directions[row.heading.decimal]
                factor=_scale(np.multiply(rho,g)) if hams else _dimensional_scale(rho,L,2 + int(i % 6 >= 3),g)
                F[k,h,ti] = np.multiply(np.complex128(complex(row.real.value,-row.imaginary.value)), factor)
            for row in docs['hst'].records:
                i,j=row.influenced-1,row.radiating-1
                factor=_scale(np.multiply(rho,g)) if hams else _dimensional_scale(rho,L,2+int(i%6>=3)+int(j%6>=3),g)
                C[source_to_target[i],source_to_target[j]] = np.multiply(row.coefficient.value,factor)
    except (OverflowError, FloatingPointError) as error:
        raise WamitImportError(f'Unrepresentable dimensional conversion: {error}') from error
    source_bodies = [replace(b, rotation_center=contract.source_frames[b.body_id].reference_center) for b in bodies]
    provenance = dict(profile=contract.profile, source_body_order=list(contract.source_body_order),
        source_frames={k:v.to_json() for k,v in contract.source_frames.items()},
        file_sha256=dict(contract.file_sha256), input_sha256=contract.input_sha256,
        executable_sha256=contract.executable_sha256, source_request_hash=contract.source_request_hash,
        reference_length_m=L, source_rho=rho, source_gravity=g,
        source_depth_kind='infinite' if math.isinf(request.water_depth) else 'finite',
        source_water_depth=None if math.isinf(request.water_depth) else request.water_depth,
        phasor_conversion=('F/a=rho*g*conj(.3); physical potential=i*native stored potential' if hams else 'conjugate exp(+i omega t) amplitude to exp(-i omega t)'),
        source_frequency_coordinate=('angular_frequency_rad_s' if hams else 'period_seconds'),
        dimensional_normalization=('A=rho*Ahat; B=rho*omega*Bhat; C,F/a=rho*g*hat; REFL unused in coefficient writers' if hams else 'standard WAMIT7 ULEN powers'),
        incident_amplitude_normalization='per unit incident elevation amplitude at WORLD X=Y=0',
        printing_contract='round to nearest; exact decimal coordinate identity',
        source_diagnostics=[d.as_dict() for d in excitation_issues],
        hydrostatic_ownership='UNKNOWN; .hst may include gravity; no baseline declared')
    if source_provenance is not None: provenance['native_source'] = source_provenance
    ds = assemble_hydro_dataset(source_bodies,request,added_mass=A,radiation_damping=B,excitation_force=F,
        hydrostatic_stiffness=C,rotation_center=[b.rotation_center for b in source_bodies],
        center_of_mass=[b.center_of_mass for b in bodies],center_of_buoyancy=[contract.center_of_buoyancy[b.body_id] for b in bodies],
        added_mass_infinite=Ainf,added_mass_infinite_source=((f'HAMS angular-frequency sentinel -1 .1 SHA256={contract.file_sha256["1"]}; {contract.profile}; source={contract.source_tree_sha256}; rho={rho}; source frames in wamit_import' if hams else f'WAMIT period0 .1 SHA256={contract.file_sha256["1"]}; {PROFILE}; L={L}; rho={rho}') if Ainf is not None else None),
        backend=('HAMS-import' if hams else 'WAMIT-import'),backend_version=contract.producer_version,backend_commit=contract.producer_build,
        qualification_status='unqualified',extra_attrs={'wamit_import':provenance,
            COEFFICIENT_FRAMES_ATTR:{k:v.to_json() for k,v in contract.source_frames.items()}})
    ds = reframe_dataset(ds,new_axes={b.body_id:np.eye(3) for b in bodies},
                          new_reference_centers={b.body_id:b.rotation_center for b in bodies})
    ds = attach_excitation_source(ds,request,bodies,convention_evidence=contract.convention_evidence)
    return attach_hydrostatic_contract(ds,HydrostaticContract(role='UNKNOWN',force_basis='UNKNOWN',
        stiffness_derivative='UNKNOWN',bodies=ids,baseline_wrench=None,
        provenance={'source':'.hst source ownership unresolved','input_sha256':contract.input_sha256},full_source=None))


def import_wamit7(bodies, request, contract, *, radiation_path, excitation_path, hydrostatic_path):
    """Import only the public standard period/phasor profile; never solve or repair."""
    if type(contract) is not Wamit7ImportContract or contract.profile != PROFILE:
        raise WamitImportError('Authoritative standard WAMIT7 import contract required')
    return _import_coefficients(bodies,request,contract,radiation_path=radiation_path,
        excitation_path=excitation_path,hydrostatic_path=hydrostatic_path)
