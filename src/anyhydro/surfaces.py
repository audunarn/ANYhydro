"""Explicit surface semantics and sheet sign/identity contracts; no BEM solver.

Face winding points from the named minus side to the plus side. Net sheet
traction is (p_minus - p_plus)*n. Physical mass/buoyancy remain mechanical
authorities and cannot be assigned by this hydrodynamic surface record.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from numbers import Real
import numpy as np
from .errors import RecordValidationError, SchemaError

SURFACE_SCHEMA = 'anyhydro-surface-v1'
SURFACE_IDENTITY_ATTR = 'surface_identity'
SHEET_SOURCE_ATTR = 'sheet_source_provenance'
SURFACE_DATASET_SCHEMA = '1.2.0'
PRESSURE_JUMP_CONTRACT_ATTR = 'pressure_jump_contract'
CAPABILITY_STATES = ('unsupported', 'unverified', 'implemented', 'verified')

def _string(value, field):
    if not isinstance(value, str) or not value.strip():
        raise RecordValidationError(f'{field} must be a nonempty string')
    return value

@dataclass(frozen=True)
class PlateDragCorrection:
    """Declared plate-drag convention, not an implemented force law."""
    reference_area_m2: float
    coefficient: float
    coefficient_convention: str = 'F=0.5*rho*Cd*area*abs(vn)*vn*n'
    relative_velocity: str = 'wave_plus_current_minus_surface'
    force_direction: str = 'normal_projection'

    def __post_init__(self):
        for field in ('reference_area_m2', 'coefficient'):
            v = getattr(self, field)
            if isinstance(v, bool) or not isinstance(v, Real) or not math.isfinite(v) or v <= 0:
                raise RecordValidationError(f'{field} must be finite and positive')
            object.__setattr__(self, field, float(v))
        expected = ('F=0.5*rho*Cd*area*abs(vn)*vn*n', 'wave_plus_current_minus_surface', 'normal_projection')
        if (self.coefficient_convention, self.relative_velocity, self.force_direction) != expected:
            raise RecordValidationError('plate drag requires an explicit supported area/relative-velocity/normal convention')

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) != set(cls.__dataclass_fields__):
            raise RecordValidationError('plate_drag must contain exactly its declared convention fields')
        return cls(**data)

@dataclass(frozen=True)
class SurfaceSemantics:
    surface_id: str
    owning_body_id: str
    surface_kind: str
    primary_hydrodynamics: str
    source_geometry_id: str
    viscous_correction: str = 'none'
    minus_side: str | None = None
    plus_side: str | None = None
    plate_drag: PlateDragCorrection | None = None

    def __post_init__(self):
        for field in ('surface_id', 'owning_body_id', 'source_geometry_id'):
            _string(getattr(self, field), field)
        if self.surface_kind not in ('body_boundary', 'thin_sheet', 'numerical_lid'):
            raise RecordValidationError('unknown surface_kind; an open mesh does not infer a thin sheet')
        if self.primary_hydrodynamics not in ('potential_flow', 'morison', 'none'):
            raise RecordValidationError('one primary_hydrodynamics owner is required')
        if self.viscous_correction not in ('none', 'member_drag', 'plate_drag'):
            raise RecordValidationError('unknown viscous_correction')
        if self.surface_kind == 'numerical_lid':
            if self.primary_hydrodynamics != 'none' or self.viscous_correction != 'none':
                raise RecordValidationError('a numerical lid is not a physical plate or load owner')
        if self.surface_kind == 'thin_sheet':
            _string(self.minus_side, 'minus_side'); _string(self.plus_side, 'plus_side')
            if self.minus_side == self.plus_side:
                raise RecordValidationError('thin_sheet requires two distinct fluid sides')
            if self.primary_hydrodynamics == 'morison':
                raise RecordValidationError('full Morison sheet inertia is unspecified; use potential_flow with optional explicit plate_drag')
            if self.viscous_correction == 'member_drag':
                raise RecordValidationError('a sheet cannot silently use circular member drag or equivalent diameter')
        elif self.minus_side is not None or self.plus_side is not None:
            raise RecordValidationError('two-sided pressure labels belong only to thin_sheet')
        if self.viscous_correction == 'member_drag' and self.primary_hydrodynamics != 'potential_flow':
            raise RecordValidationError('member_drag correction requires potential_flow; duplicate Morison drag is refused')
        if self.viscous_correction == 'plate_drag':
            if self.surface_kind != 'thin_sheet' or not isinstance(self.plate_drag, PlateDragCorrection):
                raise RecordValidationError('plate_drag requires a thin_sheet and explicit PlateDragCorrection')
        elif self.plate_drag is not None:
            raise RecordValidationError('plate_drag data requires the plate_drag correction')

    @property
    def normal_convention(self):
        return {'body_boundary':'outward', 'thin_sheet':'minus_to_plus', 'numerical_lid':'downward'}[self.surface_kind]

    def to_dict(self):
        return dict(schema=SURFACE_SCHEMA, surface_id=self.surface_id, owning_body_id=self.owning_body_id,
            surface_kind=self.surface_kind, primary_hydrodynamics=self.primary_hydrodynamics,
            source_geometry_id=self.source_geometry_id, viscous_correction=self.viscous_correction,
            minus_side=self.minus_side, plus_side=self.plus_side,
            plate_drag=None if self.plate_drag is None else self.plate_drag.to_dict(),
            normal_convention=self.normal_convention)

    @classmethod
    def from_dict(cls, data):
        fields=set(cls.__dataclass_fields__)
        if not isinstance(data, dict) or set(data) != fields | {'schema','normal_convention'}:
            raise RecordValidationError('surface semantics fields are incomplete or unknown; migration must be explicit')
        if data['schema'] != SURFACE_SCHEMA:
            raise RecordValidationError('unsupported surface schema')
        values={name:data[name] for name in fields}
        if values['plate_drag'] is not None:
            values['plate_drag']=PlateDragCorrection.from_dict(values['plate_drag'])
        result=cls(**values)
        if data['normal_convention'] != result.normal_convention:
            raise RecordValidationError('surface normal convention disagrees with surface kind')
        return result

def migrate_surface_role(role, *, surface_kind, surface_id, owning_body_id, source_geometry_id,
                         minus_side=None, plus_side=None):
    """Explicit migration, never an inference from mesh openness or shell type.

    Ambiguous drag-only/exclusive combined roles require a caller decision.
    Existing records are never automatically changed during load.
    """
    roles={'potential_flow':'potential_flow', 'bem':'potential_flow', 'morison':'morison', 'morison_full':'morison', 'none':'none', 'numerical_lid':'none'}
    if role not in roles or (role == 'numerical_lid' and surface_kind != 'numerical_lid'):
        raise RecordValidationError('ambiguous legacy hydrodynamic role; explicitly select kind, primary owner and drag correction')
    return SurfaceSemantics(surface_id, owning_body_id, surface_kind, roles[role], source_geometry_id,
                            minus_side=minus_side, plus_side=plus_side)

def sheet_geometry(mesh):
    """Oriented manifold connectivity, boundary edges and panel area vectors."""
    incidences={}; vectors=[]; centers=[]; face_shapes=set(); adjacency={}; boundary_faces=set()
    for face_index, (face, size) in enumerate(zip(mesh.faces, mesh.face_sizes)):
        row=tuple(int(v) for v in face[:int(size)])
        if len(set(row)) != len(row):
            raise RecordValidationError('thin_sheet face contains repeated vertex indices')
        xyz=mesh.vertices[list(row)]
        shape=tuple(sorted(tuple(float(v) for v in point) for point in xyz))
        if shape in face_shapes:
            raise RecordValidationError('thin_sheet has duplicate/coincident faces, including opposing panels')
        face_shapes.add(shape)
        adjacency[face_index]=set()
        vector=sum((np.cross(xyz[i]-xyz[0],xyz[i+1]-xyz[0])*.5 for i in range(1,len(row)-1)), np.zeros(3))
        area=np.linalg.norm(vector)
        if not np.isfinite(area) or area <= 0:
            raise RecordValidationError('thin_sheet contains a degenerate face')
        # Straight planar panels only; do not quietly collapse a twisted midsurface.
        scale=max(1.0,float(np.max(np.abs(xyz-xyz[0]))))
        if np.max(np.abs((xyz-xyz[0]) @ (vector/area))) > 1e-10*scale:
            raise RecordValidationError('thin_sheet panel is not planar')
        # Area centroid of the triangulated polygon for exact constant-panel moments.
        weights=[np.linalg.norm(np.cross(xyz[i]-xyz[0],xyz[i+1]-xyz[0]))*.5 for i in range(1,len(row)-1)]
        if abs(sum(weights)-area) > 1e-10*area:
            raise RecordValidationError('thin_sheet polygon has inconsistent or overlapping triangulation')
        centers.append(sum((w*(xyz[0]+xyz[i]+xyz[i+1])/3 for i,w in enumerate(weights,1)),np.zeros(3))/area)
        vectors.append(vector)
        for a,b in zip(row,row[1:]+row[:1]):
            key=tuple(sorted((a,b))); incidences.setdefault(key,[]).append(((a,b),face_index))
    boundary=[]
    for key, directions in incidences.items():
        if len(directions) > 2 or (len(directions)==2 and directions[0][0] != directions[1][0][::-1]):
            raise RecordValidationError('thin_sheet connectivity is nonmanifold or has inconsistent winding')
        if len(directions)==1:
            boundary.append(directions[0][0]); boundary_faces.add(directions[0][1])
        else:
            a,b=directions[0][1],directions[1][1]; adjacency[a].add(b); adjacency[b].add(a)
    unvisited=set(adjacency)
    while unvisited:
        component=set(); pending=[next(iter(unvisited))]
        while pending:
            face=pending.pop()
            if face in component: continue
            component.add(face); pending.extend(adjacency[face]-component)
        unvisited-=component
        if not component & boundary_faces:
            raise RecordValidationError('every connected thin_sheet component must have open boundary edges')
    if not boundary:
        raise RecordValidationError('thin_sheet must be an open midsurface with boundary edges')
    return tuple(sorted(boundary)),np.asarray(vectors),np.asarray(centers)

def sheet_pressure_load(mesh, pressure_minus, pressure_plus, reference=(0.,0.,0.)):
    """Integrate supplied constant-panel pressure jump; no pressure is calculated."""
    if mesh.surface is None or mesh.surface.surface_kind != 'thin_sheet':
        raise RecordValidationError('pressure jump requires an explicit thin_sheet')
    _,vectors,centers=sheet_geometry(mesh)
    arrays=[np.asarray(v) for v in (pressure_minus,pressure_plus)]
    if any(v.shape!=(len(vectors),) or v.dtype.kind not in 'fc' or not np.all(np.isfinite(v)) for v in arrays):
        raise RecordValidationError('both pressure fields must be finite floating/complex values per sheet panel')
    origin=np.asarray(reference,dtype=float)
    if origin.shape!=(3,) or not np.all(np.isfinite(origin)):
        raise RecordValidationError('pressure moment reference must be a finite world 3-vector')
    forces=(arrays[0]-arrays[1])[:,None]*vectors
    return np.sum(forces,axis=0),np.sum(np.cross(centers-origin,forces),axis=0)

def surface_identity(bodies):
    """Complete explicit surface inventory; absent for untouched legacy bodies."""
    if not any(b.attached_surfaces or b.mesh.surface is not None or
               (b.lid_mesh is not None and b.lid_mesh.surface is not None) for b in bodies):
        return None
    result={}; seen=set()
    for body in bodies:
        items=[]
        for role,mesh in [('hull',body.mesh),('lid',body.lid_mesh)]+[('attached',s) for s in body.attached_surfaces]:
            if mesh is None or mesh.surface is None: continue
            sid=mesh.surface.surface_id
            if sid in seen: raise RecordValidationError('duplicate surface_id across bodies')
            seen.add(sid)
            edges=sheet_geometry(mesh)[0] if mesh.surface.surface_kind=='thin_sheet' else ()
            items.append(dict(role=role,mesh_hash=mesh.mesh_hash,panel_count=len(mesh.face_sizes),
                              boundary_edges=[list(pair) for pair in edges],semantics=mesh.surface.to_dict()))
        result[body.body_id]=items
    return result

def _decoded(attrs,key):
    data=attrs[key]
    if isinstance(data,str):
        try: data=json.loads(data)
        except (ValueError,TypeError) as exc: raise SchemaError(f'{key} is invalid JSON') from exc
    return data

def validate_surface_dataset(ds):
    """Validate transport/declared provenance; never confer scientific qualification."""
    present=SURFACE_IDENTITY_ATTR in ds.attrs
    if not present:
        if ds.attrs.get('schema_version')==SURFACE_DATASET_SCHEMA:
            raise SchemaError('surface dataset version requires explicit surface_identity; downgrade/omission refused')
        if SHEET_SOURCE_ATTR in ds.attrs or PRESSURE_JUMP_CONTRACT_ATTR in ds.attrs or 'pressure_jump' in ds:
            raise SchemaError('sheet source/pressure jump requires explicit surface_identity')
        return
    if ds.attrs.get('schema_version')!=SURFACE_DATASET_SCHEMA:
        raise SchemaError('explicit surface intent requires dataset schema 1.2.0; legacy relabeling refused')
    inventory=_decoded(ds.attrs,SURFACE_IDENTITY_ATTR)
    bodies=list(map(str,ds.body.values)); sheets={}; seen=set()
    if not isinstance(inventory,dict) or set(inventory)!=set(bodies):
        raise SchemaError('surface_identity must cover exactly all dataset bodies')
    for body,entries in inventory.items():
        if not isinstance(entries,list): raise SchemaError('surface_identity entries must be lists')
        roles=[]
        for entry in entries:
            if not isinstance(entry,dict) or set(entry)!={'role','mesh_hash','panel_count','boundary_edges','semantics'}:
                raise SchemaError('surface identity fields are incomplete or unknown')
            try: s=SurfaceSemantics.from_dict(entry['semantics'])
            except RecordValidationError as exc: raise SchemaError(str(exc)) from exc
            if s.owning_body_id!=body or s.surface_id in seen:
                raise SchemaError('surface ownership or unique surface_id mismatch')
            seen.add(s.surface_id)
            role=entry['role']; roles.append(role)
            if role in ('hull','lid') and roles.count(role)>1:
                raise SchemaError('only one hull and one numerical lid per body surface inventory')
            expected={'hull':'body_boundary','lid':'numerical_lid','attached':'thin_sheet'}
            if role not in expected or s.surface_kind!=expected[role]:
                raise SchemaError('surface kind does not match its hull/lid/attached role')
            digest=entry['mesh_hash']
            if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
                raise SchemaError('surface mesh_hash must be a SHA-256 digest')
            n=entry['panel_count']; edges=entry['boundary_edges']
            if isinstance(n,bool) or not isinstance(n,int) or n<=0 or not isinstance(edges,list):
                raise SchemaError('surface panel count/boundary edges are invalid')
            if role=='hull' and digest!=_decoded(ds.attrs,'mesh_hashes')[body]:
                raise SchemaError('explicit hull surface hash does not match dataset hull')
            if role=='lid' and digest!=_decoded(ds.attrs,'body_identity')[body]['lid_mesh_hash']:
                raise SchemaError('explicit numerical lid surface hash mismatch')
            if s.surface_kind=='thin_sheet':
                if not edges or any(not isinstance(pair,list) or len(pair)!=2 or any(isinstance(v,bool) or not isinstance(v,int) or v<0 for v in pair) or pair[0]==pair[1] for pair in edges):
                    raise SchemaError('thin_sheet requires explicit oriented boundary edges')
                sheets[s.surface_id]=n
            elif edges: raise SchemaError('boundary edge semantics belong only to thin_sheet')
    if sheets:
        if SHEET_SOURCE_ATTR not in ds.attrs:
            raise SchemaError('sheet coefficients require external/native source provenance; hull-only data cannot be relabeled')
        source=_decoded(ds.attrs,SHEET_SOURCE_ATTR)
        fields={'backend','version','source_commit','capability_evidence_sha256','coupling','pressure_jump_convention','source_capability_status'}
        if not isinstance(source,dict) or set(source)!=fields:
            raise SchemaError('sheet_source_provenance requires exact pinned source, evidence and conventions')
        if any(not isinstance(source[k],str) or not source[k].strip() for k in fields):
            raise SchemaError('sheet source provenance fields must be nonempty strings')
        digest=source['capability_evidence_sha256']
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise SchemaError('sheet capability evidence requires a SHA-256 reference')
        if source['source_capability_status']!='verified' or source['coupling']!='full_hull_sheet_and_cross_body' or source['pressure_jump_convention']!='p_minus-p_plus;normal_minus_to_plus':
            raise SchemaError('sheet source must declare verified genuine coupling and the signed pressure convention; no approximation substitution')
    elif SHEET_SOURCE_ATTR in ds.attrs:
        raise SchemaError('sheet source provenance has no sheet surface')
    if PRESSURE_JUMP_CONTRACT_ATTR in ds.attrs and 'pressure_jump' not in ds:
        raise SchemaError('pressure_jump_contract requires its stored pressure field')
    if 'pressure_jump' in ds:
        if PRESSURE_JUMP_CONTRACT_ATTR not in ds.attrs:
            raise SchemaError('pressure_jump requires explicit quantity/units/normalization contract')
        contract=_decoded(ds.attrs,PRESSURE_JUMP_CONTRACT_ATTR)
        expected_contract=dict(quantity='total_excitation_pressure_jump',unit='Pa/m',
            wave_amplitude='unit_complex_incident_amplitude_1_m',coordinate_frame='world',
            phasor_convention=ds.attrs['phasor_convention'],
            sign='p_minus-p_plus;normal_minus_to_plus')
        if contract!=expected_contract:
            raise SchemaError('pressure_jump supports only world total excitation pressure in Pa/m under the declared unit-amplitude phasor; radiation/other normalization is refused')
        pressure=ds['pressure_jump']
        if not sheets or tuple(pressure.dims)!=('omega','wave_direction','sheet_panel') or pressure.dtype!=np.complex128:
            raise SchemaError('pressure_jump must be complex128 on omega/wave_direction/sheet_panel')
        if not np.all(np.isfinite(pressure.values)):
            raise SchemaError('pressure_jump contains nonfinite values')
        for name in ('sheet_panel_surface_id','sheet_panel_face'):
            if name not in ds.coords or tuple(ds[name].dims)!=('sheet_panel',):
                raise SchemaError(f'pressure_jump requires {name} identity coordinate')
        ids=ds.sheet_panel_surface_id.values.tolist(); faces=ds.sheet_panel_face.values.tolist()
        if any(not isinstance(s,str) for s in ids) or any(isinstance(v,bool) or not isinstance(v,int) for v in faces):
            raise SchemaError('sheet pressure panel identities must be strings and integer face indices')
        actual=list(zip(ids,faces)); expected={(sid,i) for sid,n in sheets.items() for i in range(n)}
        if len(set(actual))!=len(actual) or set(actual)!=expected:
            raise SchemaError('pressure_jump must cover each declared sheet panel exactly once')

