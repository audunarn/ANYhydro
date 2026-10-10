# SPDX-License-Identifier: MPL-2.0
"""Immutable pinned-HAMS input preparation. Never executes or admits a solver."""
from dataclasses import dataclass
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .hams_producer import HamsProducerBuild,HamsProducerError,GRAVITY_SOURCE_IDENTITY
from .hams_field_points import normalize_field_points,admit_field_points,HamsFieldPointError
from .dof import body_dof_keys
from .errors import CapabilityError, HydroError
from .records import HydroBody, HydroRequest
from .hams_reference import HAMS_SOURCE_COMMIT, HAMS_EXECUTABLE_SHA256, HAMS_NATIVE_GRAVITY, HAMS_NATIVE_DENSITY

__all__ = ['HamsDeckError', 'HamsPreparedDeck', 'prepare_hams_deck']
_FORMAT = 'anyhydro-pinned-hams-deck-v1'
_CORRECTED_FORMAT = 'anyhydro-corrected-hams-deck-v2'
_MECHANICAL_FIELDS=('center_of_gravity','mass','linear_damping','quadratic_damping','hydrostatic_restoring','external_restoring')


class HamsDeckError(HydroError):
    """Missing, changed, unsupported or unrepresentable input evidence."""


def _syntax():
    try:
        from anyfileio import hams
    except ImportError as exc:
        raise CapabilityError('HAMS input preparation requires the optional ANYfileio package with public hams syntax') from exc
    return hams


def _bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')


def _sha(data):return hashlib.sha256(data).hexdigest()


def _mechanical_identity(record):
    return _sha(_bytes({name:getattr(record,name) for name in _MECHANICAL_FIELDS}))


def _body_identity(body):
    return dict(body_id=body.body_id,mesh_hash=body.mesh.mesh_hash,
                lid_mesh_hash=None if body.lid_mesh is None else body.lid_mesh.mesh_hash,
                rotation_center=list(body.rotation_center),center_of_mass=list(body.center_of_mass),
                enabled_dofs=list(body.enabled_dofs),outward_normals_verified=body.mesh.outward_normals_verified,
                attached_surface_hashes=[mesh.mesh_hash for mesh in body.attached_surfaces])


@dataclass(frozen=True)
class HamsPreparedDeck:
    directory: Path
    case_hash: str

    def verify(self, *, bodies=None, request=None, mechanics=None, executable_path=None, producer_build=None, field_points=None):
        """Verify a sealed input inventory; this is not permission to execute it."""
        try:
            path=Path(self.directory);raw=(path/'case.json').read_bytes()
            if _sha(raw)!=self.case_hash:raise HamsDeckError('Deck manifest identity changed')
            manifest=json.loads(raw)
            if manifest['source_commit']!=HAMS_SOURCE_COMMIT:raise HamsDeckError('Unknown source pin')
            expected=HAMS_EXECUTABLE_SHA256
            if manifest['format']==_CORRECTED_FORMAT:
                recorded=HamsProducerBuild.from_identity(manifest['producer_build'],executable_path=manifest['executable_path'])
                expected=recorded.executable_sha256
                if producer_build is not None and (type(producer_build) is not HamsProducerBuild or producer_build.identity()!=recorded.identity()):
                    raise HamsDeckError('Caller producer build differs from prepared identity')
                if expected!=manifest['executable_sha256']:raise HamsDeckError('Producer/deck executable identity differs')
            elif manifest['format']!=_FORMAT or manifest['executable_sha256']!=expected or producer_build is not None:
                raise HamsDeckError('Unsupported deck/executable/source profile')
            names={p.name for p in (path/'Input').iterdir()}
            if names!=set(manifest['input_sha256']):raise HamsDeckError('Input inventory changed; implicit mode/extra overrides refused')
            for name,digest in manifest['input_sha256'].items():
                if Path(name).name!=name or _sha((path/'Input'/name).read_bytes())!=digest:
                    raise HamsDeckError('Input file identity changed')
            stored_points=normalize_field_points(manifest.get('field_points',()))
            actual_points=_syntax().parse_control((path/'Input/ControlFile.in').read_text(encoding='ascii')).field_points
            if actual_points!=stored_points:raise HamsDeckError('Prepared field points differ from sealed control')
            if field_points is not None and normalize_field_points(field_points)!=stored_points:raise HamsDeckError('Caller field-point identity changed')
            if bodies is not None and request is not None:admit_field_points(bodies,request,stored_points)
            if bodies is not None and [_body_identity(b) for b in bodies]!=manifest['bodies']:
                raise HamsDeckError('Caller body geometry/reference/DOF identity changed')
            if request is not None and (not isinstance(request,HydroRequest) or request.content_hash()!=manifest['request_hash']):
                raise HamsDeckError('Caller request identity changed')
            if mechanics is not None:
                if not isinstance(mechanics,Mapping) or {key:_mechanical_identity(value) for key,value in mechanics.items()}!=manifest['mechanical_input_sha256']:
                    raise HamsDeckError('Caller mechanical input identity changed')
            if executable_path is not None and _sha(Path(executable_path).read_bytes())!=expected:
                raise HamsDeckError('Executable identity changed')
            return manifest
        except (OSError,KeyError,TypeError,AttributeError,json.JSONDecodeError,HamsProducerError,HamsFieldPointError) as exc:
            raise HamsDeckError('Missing or malformed sealed deck/input/caller') from exc


def _require_native_regime(request):
    """Reject finite inputs entering inconsistent native incident/limit branches."""
    if math.isfinite(request.water_depth) and request.water_depth > 500:
        raise HamsDeckError('Pinned HAMS finite depth >500 mixes deep VINP and finite DINP')
    for omega in request.omegas:
        if 2 * math.pi / omega <= 1.e-6 or omega * omega / HAMS_NATIVE_GRAVITY < 1.e-8:
            raise HamsDeckError('Finite frequency enters native zero/infinite limit substitutions')


def prepare_hams_deck(directory, bodies, request, mechanics, *, executable_path,
                      irregular_frequency_removal=False, allow_mechanical_rounding=False,
                      reference_length=1., threads=1, producer_build=None, field_points=()):
    """Prepare exact WORLD geometry and explicit caller mechanical records.

    Mechanical matrices use WORLD-aligned six rigid DOFs about each supplied
    reference center. They are inputs, never independent hydrostatic evidence.
    Matrix quantization requires explicit opt-in and is recorded. WORLD mesh Z
    is retained because this pinned multi-body reader ignores LCS Z translation.
    """
    api=_syntax()
    try:field_points=normalize_field_points(field_points)
    except HamsFieldPointError as exc:raise HamsDeckError(str(exc)) from exc
    if field_points and (type(producer_build) is not HamsProducerBuild or producer_build.source_tree_sha256!=GRAVITY_SOURCE_IDENTITY):raise HamsDeckError('Fields require exact gravity-consistent source')
    if not isinstance(request,HydroRequest):raise HamsDeckError('Expected HydroRequest')
    try:bodies=tuple(bodies)
    except TypeError as exc:raise HamsDeckError('Expected explicit body records') from exc
    if not bodies or any(not isinstance(b,HydroBody) for b in bodies) or len({b.body_id for b in bodies})!=len(bodies):
        raise HamsDeckError('Unique nonempty body records required')
    if not isinstance(mechanics,Mapping) or set(mechanics)!={b.body_id for b in bodies}:
        raise HamsDeckError('Explicit mechanical input for every body required')
    if request.rho!=HAMS_NATIVE_DENSITY or request.gravity!=HAMS_NATIVE_GRAVITY:
        raise HamsDeckError('Pinned HAMS native rho1000/g9.80665 required; no implicit similarity')
    _require_native_regime(request)
    try:admit_field_points(bodies,request,field_points)
    except HamsFieldPointError as exc:raise HamsDeckError(str(exc)) from exc
    if producer_build is not None and producer_build.source_tree_sha256==GRAVITY_SOURCE_IDENTITY and request.include_infinite_frequency:raise HamsDeckError('Corrected gravity source has finite-only acquisition evidence')
    if request.backend_settings:raise HamsDeckError('Unmapped backend settings refused')
    if request.include_zero_frequency:raise HamsDeckError('Zero-frequency canonical request unsupported')
    if not isinstance(irregular_frequency_removal,bool) or not isinstance(allow_mechanical_rounding,bool):
        raise HamsDeckError('Explicit boolean controls required')
    path=Path(directory).resolve()
    if path.exists():raise FileExistsError(path)
    try:
        exe=Path(executable_path).resolve();digest=_sha(exe.read_bytes())
    except (OSError,TypeError) as exc:raise HamsDeckError('Explicit readable executable required') from exc
    if producer_build is None:
        if digest!=HAMS_EXECUTABLE_SHA256:raise HamsDeckError('Executable differs from pinned release')
    else:
        if type(producer_build) is not HamsProducerBuild or producer_build.executable_path!=exe:
            raise HamsDeckError('Explicit matching corrected producer record required')
        try:producer_build.verify()
        except HamsProducerError as exc:raise HamsDeckError(str(exc)) from exc
        if digest!=producer_build.executable_sha256:raise HamsDeckError('Corrected producer executable identity changed')
    multi=len(bodies)>1
    # Keep mesh coordinates in WORLD directly. Nonzero LCS origins would incur
    # subtraction/re-addition rounding and the pinned reader ignores origin Z.
    origins=tuple((0.,0.,0.,0.) for b in bodies)
    depth=-1. if math.isinf(request.water_depth) else request.water_depth
    files={};quantization={};mechanical_identity={}
    try:
        control=api.HamsControl(depth,tuple(float(x) for x in request.omegas),tuple(float(math.degrees(x)) for x in request.wave_directions),
                    origins,tuple(b.rotation_center for b in bodies),reference_length,bool(request.include_infinite_frequency),
                    irregular_frequency_removal=irregular_frequency_removal,threads=threads,field_points=field_points)
        files['ControlFile.in']=api.render_control(control)
        for index,body in enumerate(bodies,1):
            if body.enabled_dofs!=body_dof_keys(body.body_id) or body.attached_surfaces:
                raise HamsDeckError('Full ordered rigid6DOF required; attached sheets/generalized modes unsupported')
            if body.mesh.surface is not None and body.mesh.surface.primary_hydrodynamics!='potential_flow':
                raise HamsDeckError('Hull primary ownership must be potential_flow')
            if not body.mesh.outward_normals_verified:raise HamsDeckError('Verified hull winding required')
            if not irregular_frequency_removal and body.lid_mesh is not None:
                raise HamsDeckError('Explicit lid cannot be silently omitted; enable the declared lid policy')
            if irregular_frequency_removal and body.lid_mesh is None:
                raise HamsDeckError('Irregular-frequency policy requires every body lid')
            suffix=f'_{index}' if multi else ''
            for mesh,prefix in ((body.mesh,'HullMesh'),(body.lid_mesh,'WaterPlaneMesh')):
                if mesh is None:continue
                if prefix=='WaterPlaneMesh' and np.any(mesh.vertices[:,2]!=0):
                    raise HamsDeckError('Pinned HAMS lid must be exactly on WORLD z=0; submerged lids unsupported')
                if np.any(mesh.face_sizes>4):raise HamsDeckError('Pinned HAMS triangles/quads only; no topology conversion')
                vertices=mesh.vertices
                faces=tuple(tuple(int(v) for v in row[:size]) for row,size in zip(mesh.faces,mesh.face_sizes))
                files[prefix+suffix+'.pnl']=api.render_mesh(api.HamsPanelMesh(vertices.tolist(),faces))
            mechanical=mechanics[body.body_id]
            if not isinstance(mechanical,api.HamsMechanicalInput) or tuple(mechanical.center_of_gravity)!=body.center_of_mass:
                raise HamsDeckError('Explicit mechanical record must bind the WORLD body center of mass')
            text=api.render_mechanical(mechanical,allow_rounding=allow_mechanical_rounding)
            represented=api.parse_mechanical(text);stats={}
            for name in ('mass','linear_damping','quadratic_damping','hydrostatic_restoring','external_restoring'):
                pairs=tuple(zip(np.asarray(getattr(mechanical,name)).flat,np.asarray(getattr(represented,name)).flat))
                changes=[abs(float(a)-float(b)) for a,b in pairs if a!=b]
                stats[name]=dict(changed_values=len(changes),max_absolute_change=max(changes,default=0.))
            quantization[body.body_id]=stats
            mechanical_identity[body.body_id]=_mechanical_identity(mechanical)
            files['Hydrostatic'+suffix+'.in']=text
        # Round-trip the exact serialized controls/geometry before publishing.
        if api.parse_control(files['ControlFile.in'])!=control:raise HamsDeckError('Control representation changed')
        for name,text in files.items():
            if name.endswith('.pnl'):api.parse_mesh(text)
    except api.HamsInputError as exc:raise HamsDeckError(str(exc)) from exc
    manifest=dict(format=(_FORMAT if producer_build is None else _CORRECTED_FORMAT),syntax_profile=api.PROFILE,source_commit=HAMS_SOURCE_COMMIT,
        executable_sha256=digest,executable_path=str(exe),qualification='development_unqualified',
        request_hash=request.content_hash(),bodies=[_body_identity(b) for b in bodies],
        source_axes='WORLD',source_origins=origins,normal_convention='unchanged solid-outward winding; pinned HAMS uses negative cross',
        native_rho=HAMS_NATIVE_DENSITY,native_gravity=HAMS_NATIVE_GRAVITY,
        input_frequency_type=3,output_frequency_type=3,reference_length=float(reference_length),threads=threads,
        mechanical_role='caller supplied, unqualified; not independent fluid hydrostatics',
        mechanical_input_sha256=mechanical_identity,allow_mechanical_rounding=allow_mechanical_rounding,
        mechanical_quantization=quantization,irregular_frequency_removal=irregular_frequency_removal,
        excitation_admission='unsupported: pinned producer phase inconsistency unresolved',
        input_sha256={name:_sha(text.encode('ascii')) for name,text in files.items()})
    if producer_build is not None:
        manifest['producer_build']=producer_build.identity()
        manifest['excitation_admission']='requires explicit corrected import; unqualified; limits/fields not admitted'
    if field_points:manifest['field_points']=field_points
    raw=_bytes(manifest)
    # Exclusive directory creation owns all following writes. Seal is written
    # last: interrupted writes leave inspectable incomplete evidence, not success.
    try:
        path.mkdir(parents=True,exist_ok=False);(path/'Input').mkdir()
        for name in ('Hams_format','Wamit_format'):(path/'Output'/name).mkdir(parents=True)
        for name,text in files.items():(path/'Input'/name).write_bytes(text.encode('ascii'))
        (path/'case.json').write_bytes(raw)
    except FileExistsError:raise
    except OSError as exc:raise HamsDeckError('Deck publication interrupted; incomplete evidence retained, no success seal') from exc
    deck=HamsPreparedDeck(path,_sha(raw));deck.verify(bodies=bodies,request=request,mechanics=mechanics,executable_path=exe,producer_build=producer_build)
    return deck
