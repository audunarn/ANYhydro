"""Data-only imported pressure banks. Does not invoke a hydrodynamic solver."""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SCHEMA='imported_pressure_bank_v2'
MIXED_SCHEMA='imported_pressure_bank_v3'
COMPOSITIONS=('fixed_diffraction','total_with_source_motion')
PANEL_KINDS=('body_pressure','dipole_jump')
REFERENCE_CONVENTIONS=dict(body_pressure='positive pressure acts opposite body-outward normal',
    dipole_jump='P=p_minus-p_plus in canonical phasor, reference normal=-minus-to-plus normal; traction=-P*n_reference')


def _json(x): return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)


def _ro(x,dtype=float):
    a=np.asarray(x,dtype=dtype)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


def _validate_panel_metadata(polygons,normals,centers,areas,*,allow_reverse=False):
    """Compare declared metadata to geometry-owner truth with fixed roundoff limits."""
    from anygeometry.panel_pressure_projection import validate_panel_geometry
    polygons,actual_normals,actual_centers,actual_areas=validate_panel_geometry(polygons)
    alignments=[]
    for k,p in enumerate(polygons):
        extent=max(float(np.ptp(p,axis=0).max()),np.finfo(float).tiny)
        center_error=float(np.linalg.norm(centers[k]-actual_centers[k]))
        # Translation-aware roundoff floor only; never a mapping-gap allowance.
        center_tolerance=1e-9*extent+32*np.finfo(float).eps*float(np.abs(p).max())
        if center_error>center_tolerance:
            raise ValueError('declared panel centroid disagrees with physical polygon')
        if abs(areas[k]-actual_areas[k])>1e-9*actual_areas[k]:
            raise ValueError('declared panel area disagrees with physical polygon')
        alignment=float(actual_normals[k]@normals[k])
        comparison=abs(alignment) if allow_reverse else alignment
        if abs(comparison-1)>1e-10:
            raise ValueError('declared panel normal disagrees with physical polygon')
        alignments.append(alignment)
    return polygons,alignments


@dataclass(frozen=True)
class ImportedPressureBank:
    vertices: np.ndarray
    faces: np.ndarray
    centers: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    omegas: np.ndarray
    responses: np.ndarray
    panel_ids: tuple
    surface_ids: tuple
    heading: float
    composition: str
    body_id: int
    water_depth: float | None
    rho: float
    gravity: float
    provenance_json: str
    phase_origin: tuple=(0.,0.,0.)
    bank_id: str=''
    panel_kinds: tuple=()

    @property
    def kinds(self):
        return self.panel_kinds or tuple('body_pressure' for _ in self.panel_ids)

    @property
    def conventions(self):
        if self.panel_kinds:
            return dict(units='Pa/m',phasor='exp(-i omega t)',physics=self.composition,
                        panel_kinds=self.panel_kinds,reference_conventions=dict(REFERENCE_CONVENTIONS))
        return dict(units='Pa/m',phasor='exp(-i omega t)',normals='outward',
                    pressure='positive inward',physics=self.composition)

    def __post_init__(self):
        for name in ('vertices','centers','normals','areas','omegas'):
            raw=np.asarray(getattr(self,name))
            if raw.dtype.kind not in 'fiu': raise ValueError('real numeric bank arrays required')
            object.__setattr__(self,name,_ro(raw))
        if np.asarray(self.faces).dtype.kind not in 'iu': raise ValueError('integer faces required')
        object.__setattr__(self,'faces',_ro(self.faces,np.int64))
        object.__setattr__(self,'responses',_ro(self.responses,complex))
        for name in ('panel_ids','surface_ids','phase_origin','panel_kinds'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        object.__setattr__(self,'provenance_json',_json(json.loads(self.provenance_json)))
        self._check()
        digest=hashlib.sha256(_json(self._payload()).encode()).hexdigest()
        if self.bank_id and self.bank_id!=digest: raise ValueError('imported bank identity mismatch')
        object.__setattr__(self,'bank_id',digest)

    def _check(self):
        n=len(self.panel_ids); k=len(self.omegas)
        if self.panel_kinds and (len(self.panel_kinds)!=n or any(k not in PANEL_KINDS for k in self.panel_kinds)):
            raise ValueError('one explicit supported pressure kind per source panel required')
        if (not n or not k or self.vertices.ndim!=2 or self.vertices.shape[1:]!=(3,)
            or self.faces.shape!=(n,4) or self.centers.shape!=(n,3) or self.normals.shape!=(n,3)
            or self.areas.shape!=(n,) or self.omegas.shape!=(k,) or self.responses.shape!=(k,n)
            or self.faces.min()<0 or self.faces.max()>=len(self.vertices)):
            raise ValueError('imported bank array shapes differ')
        if (any(not np.isfinite(getattr(self,x)).all() for x in
                ('vertices','centers','normals','areas','omegas','responses'))
            or np.any(self.areas<=0) or np.any(self.omegas<=0) or np.any(np.diff(self.omegas)<=0)
            or not np.allclose(np.linalg.norm(self.normals,axis=1),1,rtol=0,atol=1e-8)):
            raise ValueError('invalid imported bank values')
        if (len(set(self.panel_ids))!=n or any(type(i) is not int or i<0 for i in self.panel_ids)
            or len(self.surface_ids)!=n or any(type(s) is not str for s in self.surface_ids)
            or self.composition not in COMPOSITIONS or type(self.body_id) is not int or self.body_id<0
            or not math.isfinite(self.heading) or not 0<=self.heading<math.tau
            or any(not math.isfinite(v) or v<=0 for v in (self.rho,self.gravity))
            or (self.water_depth is not None and (not math.isfinite(self.water_depth) or self.water_depth<=0))
            or self.phase_origin!=(0.,0.,0.)):
            raise ValueError('invalid imported bank conventions/identities/environment')

    def _payload(self):
        d={n:getattr(self,n).tolist() for n in ('vertices','faces','centers','normals','areas','omegas')}
        d.update(responses_real=self.responses.real.tolist(),responses_imag=self.responses.imag.tolist())
        d.update({n:getattr(self,n) for n in ('panel_ids','surface_ids','heading','composition','body_id',
                'water_depth','rho','gravity','provenance_json','phase_origin')})
        if self.panel_kinds:
            d.update(panel_kinds=self.panel_kinds,reference_conventions=dict(REFERENCE_CONVENTIONS))
        return d

    def validate(self):
        self._check()
        if hashlib.sha256(_json(self._payload()).encode()).hexdigest()!=self.bank_id:
            raise ValueError('imported bank changed')

    def validate_geometry(self):
        """Admit physical panel geometry, including retained/manual bank records.

        Unlike data-only ``validate``, this explicitly requires ANYgeometry.
        Canonical bank winding must agree with its stored reference normals.
        """
        self.validate()
        polygons=[self.vertices[f[:3] if f[2]==f[3] else f] for f in self.faces]
        _validate_panel_metadata(polygons,self.normals,self.centers,self.areas)

    def to_dict(self): return dict(type=MIXED_SCHEMA if self.panel_kinds else SCHEMA,**self._payload(),bank_id=self.bank_id)

    @classmethod
    def from_dict(cls,data):
        d=dict(data)
        version=d.pop('type',None)
        if version not in (SCHEMA,MIXED_SCHEMA) or not d.get('bank_id'): raise ValueError('unsupported imported bank record')
        if version==MIXED_SCHEMA:
            if not d.get('panel_kinds') or d.pop('reference_conventions',None)!=REFERENCE_CONVENTIONS:
                raise ValueError('explicit mixed pressure reference conventions required')
        elif 'panel_kinds' in d or 'reference_conventions' in d:
            raise ValueError('v2 banks cannot carry mixed pressure semantics')
        real=np.asarray(d.pop('responses_real')); imag=np.asarray(d.pop('responses_imag'))
        if real.shape!=imag.shape or real.dtype.kind not in 'fiu' or imag.dtype.kind not in 'fiu':
            raise ValueError('invalid complex pressure parts')
        return cls(**d,responses=real+1j*imag)

    def save(self,path):
        self.validate(); Path(path).write_text(_json(self.to_dict()),encoding='utf8')

    @classmethod
    def load(cls,path): return cls.from_dict(json.loads(Path(path).read_text(encoding='utf8')))


def import_pressure_bank(document, *, body_id, heading_degrees, composition='fixed_diffraction'):
    """Normalize one selected body and exact heading, retaining multibody provenance."""
    document.validate()
    if type(body_id) is not int or body_id<0: raise ValueError('explicit included body index required')
    if composition not in document.compositions: raise ValueError('requested pressure composition unavailable')
    if not math.isfinite(heading_degrees): raise ValueError('finite heading required')
    headings=np.flatnonzero(np.remainder(document.headings,360)==heading_degrees%360)
    if len(headings)!=1: raise ValueError('exact requested heading unavailable')
    selected=np.asarray(document.body_ids)==body_id
    kinds=np.asarray(document.panel_types)
    if np.any(selected & ~np.isin(kinds,[0,1,2])): raise ValueError('unknown panel type')
    selected &= np.isin(kinds,[0,2])
    ix=np.flatnonzero(selected)
    if not len(ix): raise ValueError('selected body has no physical pressure panels')
    m=document.metadata; units=m['units']
    lengths={'m':1.,'mm':.001,'cm':.01,'ft':.3048}
    masses={'kg':1.,'te':1000.,'t':1000.,'lb':.45359237}
    forces={'N':1.,'kN':1000.,'lbf':4.4482216152605,'kip':4448.2216152605}
    try: L=lengths[units['length']]; M=masses[units['mass']]; F=forces[units['force']]
    except KeyError as e: raise ValueError(f'unsupported OrcaWave units: {units}') from e
    if tuple(m.get('phase_origin',()))!=(0.,0.,0.):
        raise ValueError('only explicit source phase origin (0,0,0) is supported')
    panels=np.array(document.vertices[ix]*L)
    # Physical polygon truth belongs to ANYgeometry. Its strict planar/convex
    # admission is separate from user-configurable source-to-target tolerances.
    polygons=[p[:3] if np.array_equal(p[2],p[3]) else p for p in panels]
    canonical_normals=-document.normals[ix]
    polygons,alignments=_validate_panel_metadata(polygons,canonical_normals,
        document.centers[ix]*L,document.areas[ix]*L**2,allow_reverse=True)
    for k,(p,alignment) in enumerate(zip(polygons,alignments)):
        # OrcaWave winding is not assumed. Align only after physical checks.
        oriented=p if alignment>0 else p[np.r_[0,np.arange(len(p)-1,0,-1)]]
        panels[k,:len(p)]=oriented
        if len(p)==3: panels[k,3]=oriented[2]
    omega=math.tau/document.periods; order=np.argsort(omega)
    z=document.pressure_real[document.compositions.index(composition),int(headings[0])]+1j*document.pressure_imag[document.compositions.index(composition),int(headings[0])]
    provenance=dict(document_id=document.content_sha256,source=document.metadata,
        included_body_ids=sorted(set(document.body_ids)),result_panel_ids=[document.panel_ids[i] for i in ix],
        mesh_panel_ids=[document.mesh_panel_ids[i] for i in ix],conversion='orcawave_to_si_exp_minus_v1')
    faces=np.arange(len(ix)*4).reshape(-1,4)
    triangular=np.all(panels[:,2]==panels[:,3],axis=1)
    faces[triangular,3]=faces[triangular,2]
    return ImportedPressureBank(panels.reshape(-1,3),faces,
        document.centers[ix]*L,-document.normals[ix],document.areas[ix]*L**2,omega[order],
        np.conjugate(z[order][:,ix])*(F/L**3),tuple(document.panel_ids[i] for i in ix),
        tuple(f'body:{body_id}' for _ in ix),math.radians(heading_degrees%360),composition,body_id,
        None if m['water_depth'] is None else float(m['water_depth'])*L,
        float(m['rho'])*M/L**3,float(m['gravity'])*L,_json(provenance),
        panel_kinds=(tuple('dipole_jump' if kinds[i]==2 else 'body_pressure' for i in ix)
                     if np.any(kinds[ix]==2) else ()))
