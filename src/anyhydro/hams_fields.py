# SPDX-License-Identifier: MPL-2.0
"""Source-bound finite HAMS field transport, immutable artifacts and persistence.

Fields use exp(-i*omega*t). Radiation transfers multiply displacement (m/rad),
not velocity. Dynamic pressure excludes hydrostatic pressure. No producer runs.
"""
from dataclasses import dataclass,replace
from collections.abc import Mapping
from pathlib import Path
import hashlib,json,math,os,uuid
import numpy as np
from .errors import HydroError,CapabilityError
from .dof import body_dof_keys
from .frames import CoefficientFrame,coefficient_frames,frame_change_matrix,reframe_dataset
from .hashing import dataset_content_hash
from .hams_field_points import normalize_field_points,admit_field_points
from .hams_producer import GRAVITY_SOURCE_IDENTITY
from .records import _deep_freeze

__all__=['HamsFieldError','HamsFieldData','import_hams_fields','save_hams_fields',
         'load_hams_fields','reframe_hams_fields']
FORMAT='hams-canonical-field-data-v1'
_META={'source_tree_sha256','source_profile','executable_sha256','input_sha256',
       'request_hash','raw_field_sha256','source_body_order','target_body_order',
       'source_frames','target_frames','reference_length_m','rho','gravity',
       'water_depth','phase_origin_world_xy','phasor','native_clipping',
       'pressure_kind','response_units','qualification'}


class HamsFieldError(HydroError):pass


def _plain(value):
    if isinstance(value,Mapping):return {k:_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_plain(v) for v in value]
    return value


def _bytes(value):return json.dumps(_plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')
def _digest(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise HamsFieldError('Explicit SHA256 field binding required')
    return value


def _complex(value,shape,name):
    try:arr=np.array(value,dtype=np.complex128,copy=True,order='C')
    except (TypeError,ValueError,OverflowError) as exc:raise HamsFieldError('Invalid '+name+' response') from exc
    if arr.shape!=shape or not np.isfinite(arr).all():raise HamsFieldError('Complete finite '+name+' response inventory required')
    # Bytes-backed storage cannot be made writable again by a consumer.
    return np.frombuffer(arr.tobytes(order='C'),dtype=np.complex128).reshape(shape)


def _frames(raw,ids):
    if not isinstance(raw,Mapping) or set(raw)!=set(ids):raise HamsFieldError('Complete field frame identities required')
    return {key:CoefficientFrame(raw[key]['axes'],tuple(raw[key]['reference_center'])) for key in ids}


@dataclass(frozen=True)
class HamsFieldData:
    points: tuple
    omegas: tuple
    headings: tuple
    dof_keys: tuple
    incident: np.ndarray
    total: np.ndarray
    radiation: np.ndarray
    parent_dataset_hash: str
    metadata: Mapping

    def __post_init__(self):
        points=normalize_field_points(self.points)
        if not points:raise HamsFieldError('Nonempty field inventory required')
        omegas=tuple(float(x) for x in self.omegas);headings=tuple(float(x) for x in self.headings);keys=tuple(self.dof_keys)
        if not omegas or len(set(omegas))!=len(omegas) or any(not math.isfinite(w) or w<=0 for w in omegas):raise HamsFieldError('Finite distinct positive field frequencies required')
        if not headings or len(set(headings))!=len(headings) or any(not math.isfinite(w) for w in headings):raise HamsFieldError('Finite distinct field headings required')
        if not isinstance(self.metadata,Mapping) or set(self.metadata)!=_META:raise HamsFieldError('Complete explicit field metadata required')
        meta=_plain(self.metadata)
        if (meta['source_tree_sha256']!=GRAVITY_SOURCE_IDENTITY or meta['source_profile']!='hams_mrel_cf9583c_phase_limit_gravity_corrected_finite_v3'
            or meta['phasor']!='exp(-i*omega*t)' or meta['phase_origin_world_xy']!=[0.,0.]
            or meta['pressure_kind']!='dynamic_wave_pressure_excluding_hydrostatic'
            or meta['qualification']!='development_unqualified' or meta['native_clipping']!=1e-15):raise HamsFieldError('Unknown field source/phase/physical declarations')
        source=tuple(meta['source_body_order']);target=tuple(meta['target_body_order'])
        if not source or len(set(source))!=len(source) or set(source)!=set(target) or len(target)!=len(source):raise HamsFieldError('Complete source/target body identity required')
        if keys!=tuple(k for body in target for k in body_dof_keys(body)):raise HamsFieldError('Complete ordered target rigid6DOF identity required')
        _frames(meta['source_frames'],source);_frames(meta['target_frames'],target)
        for key in ('executable_sha256','input_sha256','request_hash'):_digest(meta[key])
        if set(meta['raw_field_sha256'])!={'combined','incident'}:raise HamsFieldError('Both raw field identities required')
        for value in meta['raw_field_sha256'].values():_digest(value)
        if (meta['rho'],meta['gravity'])!=(1000.,9.80665) or not math.isfinite(meta['reference_length_m']) or meta['reference_length_m']<=0:raise HamsFieldError('Pinned field density/gravity/reference length required')
        depth=meta['water_depth']
        if depth!='infinite' and (not isinstance(depth,(float,int)) or not math.isfinite(depth) or depth<=0 or depth>500):raise HamsFieldError('Admitted field depth required')
        for point in points:
            if point[2]!=0 and point[2]>=-1e-6:raise HamsFieldError('Ambiguous point quantity refused')
            if depth!='infinite' and point[2]<=-depth:raise HamsFieldError('Point below bed refused')
        if meta['response_units']!={'wave':'elevation_m or dynamic_pressure_Pa per incident_amplitude_m','radiation':'elevation_m or dynamic_pressure_Pa per target_displacement_m_or_rad'}:raise HamsFieldError('Explicit field response units required')
        wave_shape=(len(omegas),len(headings),len(points));rad_shape=(len(omegas),len(points),len(keys))
        object.__setattr__(self,'points',points);object.__setattr__(self,'omegas',omegas);object.__setattr__(self,'headings',headings);object.__setattr__(self,'dof_keys',keys)
        for name,shape in (('incident',wave_shape),('total',wave_shape),('radiation',rad_shape)):object.__setattr__(self,name,_complex(getattr(self,name),shape,name))
        object.__setattr__(self,'parent_dataset_hash',_digest(self.parent_dataset_hash));object.__setattr__(self,'metadata',_deep_freeze(meta))

    @property
    def quantities(self):return tuple('elevation' if p[2]==0 else 'dynamic_pressure' for p in self.points)

    @property
    def scattered(self):
        return _complex(self.total-self.incident,self.total.shape,'scattered')

    def payload(self):
        return dict(format=FORMAT,points=self.points,quantities=self.quantities,omegas=self.omegas,headings=self.headings,dof_keys=self.dof_keys,
                    incident=[self.incident.real.tolist(),self.incident.imag.tolist()],total=[self.total.real.tolist(),self.total.imag.tolist()],
                    radiation=[self.radiation.real.tolist(),self.radiation.imag.tolist()],parent_dataset_hash=self.parent_dataset_hash,metadata=self.metadata)

    def content_hash(self):return hashlib.sha256(_bytes(self.payload())).hexdigest()


def save_hams_fields(fields,path):
    if type(fields) is not HamsFieldData:raise HamsFieldError('Canonical field artifact required')
    path=Path(path)
    if path.exists() or path.is_symlink():raise HamsFieldError('New field artifact path required')
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('xb') as stream:stream.write(_bytes(dict(payload=fields.payload(),content_hash=fields.content_hash())))
    os.replace(temp,path)


def load_hams_fields(path,*,expected_parent_hash=None):
    try:
        value=json.loads(Path(path).read_text(encoding='ascii'))
        if set(value)!={'payload','content_hash'}:raise HamsFieldError('Unknown field envelope')
        data=value['payload']
        if set(data)!={'format','points','quantities','omegas','headings','dof_keys','incident','total','radiation','parent_dataset_hash','metadata'} or data['format']!=FORMAT:raise HamsFieldError('Unknown field artifact profile')
        def arr(name):
            raw=data[name]
            if len(raw)!=2:raise HamsFieldError('Complete real/imaginary field arrays required')
            real=np.asarray(raw[0],dtype=float);imag=np.asarray(raw[1],dtype=float)
            if real.shape!=imag.shape:raise HamsFieldError('Real/imaginary field shape mismatch')
            return real+1j*imag
        fields=HamsFieldData(tuple(data['points']),tuple(data['omegas']),tuple(data['headings']),tuple(data['dof_keys']),arr('incident'),arr('total'),arr('radiation'),data['parent_dataset_hash'],data['metadata'])
        if fields.content_hash()!=value['content_hash'] or list(fields.quantities)!=data['quantities']:raise HamsFieldError('Stored field identity/quantity changed')
        if expected_parent_hash is not None and fields.parent_dataset_hash!=expected_parent_hash:raise HamsFieldError('Stored field parent dataset changed')
        return fields
    except (OSError,ValueError,KeyError,TypeError,OverflowError) as exc:raise HamsFieldError('Incomplete or invalid field artifact: '+str(exc)) from exc


def import_hams_fields(bodies,request,contract,*,dataset,input_directory,executable_path,native_directory,
                       radiation_path,excitation_path,hydrostatic_path,combined_path,incident_path):
    """Validate source/input/coefficient/field inventory and map all unit responses.

    Reconstruct the parent coefficients from raw evidence; an arbitrary or stale
    supplied dataset is not accepted as the parent merely because hashes agree
    with user metadata. Total includes incident, scattering is their difference.
    """
    from .hams_import import import_hams_coefficients,GRAVITY_PROFILE,_wave_number,HamsImportContract
    if type(contract) is not HamsImportContract or contract.profile!=GRAVITY_PROFILE or not contract.source_field_points:raise HamsFieldError('Explicit corrected finite field import contract required')
    bodies=tuple(bodies);points=admit_field_points(bodies,request,contract.source_field_points)
    parent=import_hams_coefficients(bodies,request,contract,input_directory=input_directory,executable_path=executable_path,native_directory=native_directory,
                                    radiation_path=radiation_path,excitation_path=excitation_path,hydrostatic_path=hydrostatic_path)
    if dataset_content_hash(parent)!=dataset_content_hash(dataset):raise HamsFieldError('Field parent does not reproduce source coefficients')
    try:from anyfileio.hams_fields import parse_hams_fields,RadiationFieldRecord,WaveFieldRecord
    except ImportError as exc:raise CapabilityError('Native field transport requires public ANYfileio hams_fields syntax') from exc
    raw={'combined':Path(combined_path).read_bytes(),'incident':Path(incident_path).read_bytes()}
    if any(hashlib.sha256(raw[k]).hexdigest()!=contract.field_file_sha256[k] for k in raw):raise HamsFieldError('Raw field identity changed')
    kwargs=dict(body_count=len(bodies),frequency_count=len(request.omegas),heading_count=len(request.wave_directions),point_count=len(points))
    try:docs={k:parse_hams_fields(v.decode('ascii'),kind=k,source=str(combined_path if k=='combined' else incident_path),**kwargs) for k,v in raw.items()}
    except (ValueError,UnicodeError) as exc:raise HamsFieldError('Native field layout refused: '+str(exc)) from exc
    incident=np.zeros((len(request.omegas),len(request.wave_directions),len(points)),complex);total=np.zeros_like(incident)
    radiation=np.zeros((len(request.omegas),len(points),6*len(bodies)),complex)
    from .wamit_import import _map_coordinates
    frequencies=_map_coordinates(docs['incident'],request.omegas,allowed_limits=())
    headings=_map_coordinates(docs['incident'],[math.degrees(x) for x in request.wave_directions],heading=True)
    # Apply the same strict printed coordinate maps to the complete mixed document.
    if _map_coordinates(docs['combined'],request.omegas,allowed_limits=())!=frequencies:raise HamsFieldError('Combined/incident frequency identities differ')
    scale=np.array([1. if p[2]==0 else request.rho*request.gravity for p in points])
    local_scale=np.array([contract.reference_length_m**(0 if j%6<3 else 1) for j in range(6*len(bodies))])
    for name,doc in docs.items():
        for r in doc.records:
            k=frequencies[r.frequency.decimal];p=r.point-1
            if isinstance(r,RadiationFieldRecord):
                values=np.array([complex(a.value,b.value) for a,b in r.values]);start=6*(r.body-1)
                radiation[k,p,start:start+6]=scale[p]*local_scale[start:start+6]*values.conjugate()
            else:
                if r.heading.decimal not in headings:raise HamsFieldError('Combined heading identity differs')
                h=headings[r.heading.decimal];z=complex(r.real.value,r.imaginary.value)
                if name=='incident':
                    omega=request.omegas[k];theta=request.wave_directions[h];x,y,depth=points[p];wave=_wave_number(omega,request.water_depth)
                    factor=math.exp(wave*depth) if math.isinf(request.water_depth) else math.cosh(wave*(depth+request.water_depth))/math.cosh(wave*request.water_depth)
                    expected=factor*np.exp(-1j*wave*(x*math.cos(theta)+y*math.sin(theta)))
                    if abs(expected)<1e-15:expected=0j
                    cast=complex(float(np.float32(expected.real)),float(np.float32(expected.imag)))
                    targets=(cast.real,cast.imag,abs(cast),math.degrees(math.atan2(cast.imag,cast.real)))
                    for number,target in zip((r.real,r.imaginary,r.magnitude,r.phase_degrees),targets):
                        error=abs(number.value-target)
                        if number is r.phase_degrees:error=abs((number.value-target+180)%360-180)
                        if error>number.rounding_bound+64*np.finfo(float).eps*max(1.,abs(target)):raise HamsFieldError('Incident normalization/phase/depth disagrees with pinned source')
                    incident[k,h,p]=scale[p]*z.conjugate()
                else:total[k,h,p]=scale[p]*z.conjugate()
    target=tuple(b.body_id for b in bodies);frames=coefficient_frames(dataset);T=np.zeros((6*len(bodies),6*len(bodies)))
    for s,key in enumerate(contract.source_body_order):
        t=target.index(key);T[6*s:6*s+6,6*t:6*t+6]=frame_change_matrix(contract.source_frames[key],frames[key])
    radiation=radiation@T
    metadata=dict(source_tree_sha256=contract.source_tree_sha256,source_profile=contract.profile,executable_sha256=contract.executable_sha256,input_sha256=contract.input_sha256,
                  request_hash=request.content_hash(),raw_field_sha256=dict(contract.field_file_sha256),source_body_order=contract.source_body_order,target_body_order=target,
                  source_frames={k:v.to_json() for k,v in contract.source_frames.items()},target_frames={k:v.to_json() for k,v in frames.items()},reference_length_m=contract.reference_length_m,
                  rho=request.rho,gravity=request.gravity,water_depth='infinite' if math.isinf(request.water_depth) else request.water_depth,phase_origin_world_xy=(0.,0.),phasor='exp(-i*omega*t)',
                  native_clipping=1e-15,pressure_kind='dynamic_wave_pressure_excluding_hydrostatic',qualification='development_unqualified',
                  response_units={'wave':'elevation_m or dynamic_pressure_Pa per incident_amplitude_m','radiation':'elevation_m or dynamic_pressure_Pa per target_displacement_m_or_rad'})
    return HamsFieldData(points,tuple(request.omegas),tuple(request.wave_directions),tuple(str(k) for k in dataset.radiating_dof.values),incident,total,radiation,dataset_content_hash(dataset),metadata)


def reframe_hams_fields(fields,old_dataset,new_dataset,*,new_axes,new_reference_centers):
    """Right-map displacement transfer arrays; verify both parent dataset identities."""
    if fields.parent_dataset_hash!=dataset_content_hash(old_dataset):raise HamsFieldError('Old field parent differs')
    old=coefficient_frames(old_dataset)
    target=tuple(fields.metadata['target_body_order'])
    if (fields.dof_keys!=tuple(str(k) for k in old_dataset.radiating_dof.values)
        or tuple(k for b in target for k in body_dof_keys(b))!=fields.dof_keys
        or set(target)!=set(old)
        or fields.omegas!=tuple(float(w) for w in old_dataset.omega.values)
        or fields.headings!=tuple(float(h) for h in old_dataset.wave_direction.values)
        or _plain(fields.metadata['target_frames'])!={k:v.to_json() for k,v in old.items()}):
        raise HamsFieldError('Field parent coordinates/frames differ')
    expected=reframe_dataset(old_dataset,new_axes=new_axes,new_reference_centers=new_reference_centers)
    if dataset_content_hash(expected)!=dataset_content_hash(new_dataset):raise HamsFieldError('New field parent is not the declared frame change')
    new=coefficient_frames(new_dataset);T=np.zeros((len(fields.dof_keys),len(fields.dof_keys)))
    for i,key in enumerate(target):T[i*6:i*6+6,i*6:i*6+6]=frame_change_matrix(old[key],new[key])
    metadata=_plain(fields.metadata);metadata['target_frames']={k:v.to_json() for k,v in new.items()}
    return replace(fields,radiation=fields.radiation@T,parent_dataset_hash=dataset_content_hash(new_dataset),metadata=metadata)
