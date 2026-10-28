"""Field-capable retained fixed-body solutions: same-solution surface elevation and pressure.

Sign/phase convention is exp(-i omega t), unit incident amplitude (1 m).  With total
(incident + diffraction) potential phi: dynamic pressure p = i omega rho phi and, from the linear
free-surface condition phi_t + g eta = 0, eta = i omega phi / g, hence p(z=0) = rho g eta from ONE solution.

Neutral records (snapshot, cache directory, validation, evaluation with a supplied evaluator) never import
Capytaine; only ``solve_field_snapshot`` and the default evaluator do, lazily.  Retained recovery
(``recover_field_snapshot``) post-processes only and never solves.  This is a separate additive cache
namespace (``field-snapshots/``); the older ``solutions/`` potential-only entries remain valid for their own
modes but carry no source strengths, so a field-using mode needs one-time recomputation per frequency.
"""
from dataclasses import dataclass
import hashlib,importlib.metadata,json,math,os,re,tempfile
from pathlib import Path
import numpy as np

VERSION='wave_surface_field_v1'
RECIPE='fixed_zero_speed_source_strengths_v1'
CONVENTIONS=dict(phasor='exp(-i omega t)',amplitude='unit incident amplitude 1 m',frame='hydro z=0 mean waterline',
    pressure='dynamic incident+diffraction p=i*omega*rho*phi',elevation='eta=i*omega*phi/g',normals='outward from hull into fluid')
DEFAULT_FLUID_OFFSET=1e-3
NAMESPACE='field-snapshots'


class WaveFieldError(ValueError):
    """Actionable refusal: missing/corrupt/unsupported retained field capability."""


class WaveFieldMissing(WaveFieldError):
    """The retained entry is truly absent (the only case preparation may fill)."""


def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def _sha(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def _numeric_hash(metadata,arrays):
    h=hashlib.sha256(_json(metadata).encode())
    for name,a in sorted(arrays.items()):
        a=np.ascontiguousarray(a);h.update(_json((name,a.shape,a.dtype.str)).encode());h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def _frozen(value,dtype):
    a=np.array(value,dtype=dtype,copy=True);a.setflags(write=False);return a


def field_request(vertices,faces,surface_ids,*,heading,water_depth,rho=1025.,gravity=9.81):
    """Physical identity of a field-capable solution family (frequency is added per entry)."""
    v=np.asarray(vertices,dtype=np.float64);f=np.asarray(faces)
    if f.dtype.kind not in 'iu':raise WaveFieldError('integer source faces required')
    labels=tuple(str(x) for x in surface_ids)
    values=(heading,water_depth,rho,gravity)
    if (v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=4 or len(labels)!=len(f)
            or not 0<len(f)<=600 or any(isinstance(x,bool) or not math.isfinite(x) for x in values)
            or min(water_depth,rho,gravity)<=0 or v[:,2].max()>1e-9):
        raise WaveFieldError('invalid admitted source geometry/physical settings')
    return dict(version=VERSION,recipe=RECIPE,heading=float(heading)%math.tau,water_depth=float(water_depth),
        rho=float(rho),gravity=float(gravity),conventions=CONVENTIONS,
        geometry=_numeric_hash(dict(surface_ids=labels),dict(vertices=v,faces=f.astype(np.int64))))


def _path(cache,request,omega):
    return Path(cache)/NAMESPACE/(hashlib.sha256(_json(dict(request=request,omega=float(omega).hex())).encode()).hexdigest()+'.npz')


@dataclass(frozen=True)
class FieldSnapshot:
    """One retained solution: geometry, source strengths, physics and backend identity, content hash."""
    request: dict
    omega: float
    vertices: np.ndarray
    faces: np.ndarray
    surface_ids: tuple
    sources: np.ndarray
    provider: dict
    snapshot_id: str=''
    # Realised numerical Green-function state: Capytaine refits the finite-depth Prony decomposition with fresh
    # randomness per solver instance, so sources alone do not define the field.  Retained and replayed verbatim.
    prony: object=None          # (2,n) array from GF.find_best_exponential_decomposition(prony_kh, method=prony_method)
    prony_kh: float=0.
    prony_method: str=''
    gf_settings: object=None    # effective Green-function constructor settings (JSON dict)

    def __post_init__(self):
        if self.prony is not None:
            object.__setattr__(self,'prony',_frozen(self.prony,np.float64))
            if self.prony.ndim!=2 or self.prony.shape[0]!=2 or not np.isfinite(self.prony).all() or not math.isfinite(self.prony_kh) \
                    or self.prony_kh<=0 or self.prony_method not in ('python','fortran') or type(self.gf_settings) is not dict:
                raise WaveFieldError('invalid retained Green-function decomposition')
        object.__setattr__(self,'vertices',_frozen(self.vertices,np.float64))
        object.__setattr__(self,'faces',_frozen(self.faces,np.int64))
        object.__setattr__(self,'sources',_frozen(self.sources,np.complex128))
        object.__setattr__(self,'surface_ids',tuple(self.surface_ids))
        if (not math.isfinite(self.omega) or self.omega<=0 or self.sources.shape!=(len(self.faces),)
                or not np.isfinite(self.sources).all() or len(self.surface_ids)!=len(self.faces)
                or self.request.get('version')!=VERSION or self.request.get('recipe')!=RECIPE):
            raise WaveFieldError('invalid or unsupported field snapshot')
        if _numeric_hash(dict(surface_ids=self.surface_ids),dict(vertices=self.vertices,faces=self.faces))!=self.request['geometry']:
            raise WaveFieldError('field snapshot geometry differs from its request')
        identity=self._identity()
        if self.snapshot_id and self.snapshot_id!=identity:raise WaveFieldError('field snapshot content identity mismatch')
        object.__setattr__(self,'snapshot_id',identity)

    def _identity(self):
        arrays=dict(vertices=self.vertices,faces=self.faces,sources=self.sources)
        meta=dict(request=self.request,omega=float(self.omega).hex(),provider=self.provider,surface_ids=self.surface_ids)
        if self.prony is not None:
            arrays['prony']=self.prony
            meta.update(prony_kh=float(self.prony_kh).hex(),prony_method=self.prony_method,gf_settings=self.gf_settings)
        return _numeric_hash(meta,arrays)

    def validate(self):
        """Public request/provider dicts are mutable: re-verify identity at every use."""
        if self._identity()!=self.snapshot_id:raise WaveFieldError('field snapshot metadata changed after validation')


def snapshot_to_dict(snapshot):
    """JSON-safe self-contained record (no backend objects) so a project can retain its source and remap later."""
    return dict(request=snapshot.request,omega=float(snapshot.omega),surface_ids=list(snapshot.surface_ids),
        vertices=snapshot.vertices.tolist(),faces=snapshot.faces.tolist(),sources_real=snapshot.sources.real.tolist(),
        sources_imag=snapshot.sources.imag.tolist(),provider=snapshot.provider,snapshot_id=snapshot.snapshot_id,
        prony=None if snapshot.prony is None else snapshot.prony.tolist(),prony_kh=snapshot.prony_kh,
        prony_method=snapshot.prony_method,gf_settings=snapshot.gf_settings)


def snapshot_from_dict(data):
    try:
        s=np.empty(len(data['sources_real']),dtype=np.complex128);s.real=data['sources_real'];s.imag=data['sources_imag']
        prony=data.get('prony')
        return FieldSnapshot(data['request'],float(data['omega']),np.asarray(data['vertices'],dtype=np.float64),
            np.asarray(data['faces'],dtype=np.int64),tuple(data['surface_ids']),s,data['provider'],data['snapshot_id'],
            None if prony is None else np.asarray(prony,dtype=np.float64),float(data.get('prony_kh',0.)),
            data.get('prony_method',''),data.get('gf_settings'))
    except WaveFieldError:raise
    except (KeyError,TypeError,ValueError) as error:raise WaveFieldError('corrupt embedded field snapshot') from error


def save_field_snapshot(cache,snapshot):
    path=_path(cache,snapshot.request,snapshot.omega);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez(stream,metadata=_json(dict(request=snapshot.request,omega=float(snapshot.omega).hex(),
                surface_ids=snapshot.surface_ids,provider=snapshot.provider,snapshot_id=snapshot.snapshot_id,
                prony_kh=snapshot.prony_kh,prony_method=snapshot.prony_method,gf_settings=snapshot.gf_settings,
                has_prony=snapshot.prony is not None)),
                vertices=snapshot.vertices,faces=snapshot.faces,sources=snapshot.sources,
                **({} if snapshot.prony is None else dict(prony=snapshot.prony)))
            stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return path


def recover_field_snapshot(cache,request,omega):
    """Load and fully validate one retained snapshot.  Never solves; refuses with an actionable error."""
    path=_path(cache,request,omega)
    if not path.exists():
        raise WaveFieldMissing(f'field-capable solution for omega={omega!r} is missing; prepare hydrodynamic fields explicitly')
    try:
        with np.load(path,allow_pickle=False) as data:
            meta=json.loads(str(data['metadata']));v=data['vertices'];f=data['faces'];s=data['sources']
            prony=data['prony'] if meta.get('has_prony') else None
        if _json(meta['request'])!=_json(request) or meta['omega']!=float(omega).hex() or not _sha(meta['snapshot_id']):
            raise WaveFieldError('field snapshot request/frequency mismatch')
        return FieldSnapshot(meta['request'],float(omega),v,f,tuple(meta['surface_ids']),s,meta['provider'],meta['snapshot_id'],
            prony,float(meta.get('prony_kh',0.)),meta.get('prony_method',''),meta.get('gf_settings'))
    except WaveFieldError:raise
    except Exception as error:
        raise WaveFieldError(f'corrupt or unsupported field snapshot {path.name}; delete it and prepare hydrodynamic '
            'fields explicitly (corrupt data is never silently overwritten)') from error


def missing_field_frequencies(cache,request,omegas):
    """Truly ABSENT frequencies (cheap, no Capytaine); corrupt/unsupported entries raise instead of counting as missing."""
    missing=[]
    for w in omegas:
        try:recover_field_snapshot(cache,request,w)
        except WaveFieldMissing:missing.append(float(w))
    return tuple(missing)


def solve_field_snapshot(vertices,faces,surface_ids,omega,*,heading,water_depth,rho=1025.,gravity=9.81,max_panels=600):
    """One bounded Capytaine 3.0 indirect fixed-body solve retaining source strengths (the only BEM entry)."""
    import capytaine as cpt
    from ._pressure_body import prepare_body
    if importlib.metadata.version('capytaine').split('.')[:2]!=['3','0']:raise WaveFieldError('field capture requires Capytaine 3.0')
    request=field_request(vertices,faces,surface_ids,heading=heading,water_depth=water_depth,rho=rho,gravity=gravity)
    v,f,labels,mesh,body=prepare_body(cpt,vertices,faces,surface_ids,period=math.tau/omega,heading=request['heading'],
        water_depth=water_depth,rho=rho,max_panels=max_panels)
    solver=cpt.BEMSolver()
    problem=cpt.DiffractionProblem(body=body,omega=omega,wave_direction=request['heading'],water_depth=water_depth,rho=rho,g=gravity)
    result=solver.solve(problem,method='indirect',keep_details=True,n_threads=1)
    if getattr(result,'sources',None) is None:raise WaveFieldError('backend did not retain source strengths')
    gf=solver.engine.green_function
    settings={k:v for k,v in gf.exportable_settings.items() if k!='green_function'}
    provider=dict(backend='capytaine',version=cpt.__version__,path=str(Path(cpt.__file__).resolve()),method='indirect',
        recipe=RECIPE,engine=type(solver.engine).__name__,green_function=gf.exportable_settings['green_function'])
    prony=None;kh=0.;method=''
    if math.isfinite(water_depth):
        # Capture the decomposition the solver ACTUALLY used: the public method is lru-cached per instance, so the
        # same call returns the realised (randomly fitted) coefficients.  A cache miss means we cannot vouch for them.
        method=str(settings.get('finite_depth_prony_decomposition_method','python')).lower()
        kh=float(problem.wavenumber*water_depth)
        cached=type(gf).find_best_exponential_decomposition
        before=cached.cache_info().hits if hasattr(cached,'cache_info') else None
        prony=np.array(gf.find_best_exponential_decomposition(kh),dtype=np.float64)
        if before is None or cached.cache_info().hits<=before:
            raise WaveFieldError('could not capture the realised finite-depth Green-function decomposition')
    return FieldSnapshot(request,float(omega),v,f,labels,np.asarray(result.sources),provider,'',prony,kh,method,settings)


def prepare_field_snapshots(vertices,faces,surface_ids,omegas,*,cache_directory,heading,water_depth,rho=1025.,
        gravity=9.81,max_missing=32,cancel=None,progress=None):
    """Return all snapshots, solving ONLY the frequencies missing from the retained cache."""
    request=field_request(vertices,faces,surface_ids,heading=heading,water_depth=water_depth,rho=rho,gravity=gravity)
    missing=missing_field_frequencies(cache_directory,request,omegas)
    if len(missing)>max_missing:raise WaveFieldError(f'{len(missing)} field-capable solves needed; limit {max_missing}')
    for w in missing:
        if cancel is not None and cancel():raise WaveFieldError('field preparation cancelled')
        save_field_snapshot(cache_directory,solve_field_snapshot(vertices,faces,surface_ids,w,heading=heading,
            water_depth=water_depth,rho=rho,gravity=gravity))
        if progress is not None:progress(w)
    return tuple(recover_field_snapshot(cache_directory,request,w) for w in omegas)


def _replay_green_function(cpt,snapshot,problem):
    """Default Green function with the finite-depth decomposition REPLACED by the retained realised coefficients.

    Only the PUBLIC ``find_best_exponential_decomposition`` is overridden, for the exact retained kh and method; any
    other request is refused.  The effective constructor settings must equal the default provider's, and the
    retained settings must equal what the capture recorded (no changed backend settings are accepted).
    """
    if snapshot.provider.get('green_function') == 'TransientDelhommeau':
        from .transient_native_profile import replay_profile_green_function
        return replay_profile_green_function(cpt, snapshot, problem)
    default=cpt.BEMSolver().engine.green_function
    effective={k:v for k,v in default.exportable_settings.items() if k!='green_function'}
    if snapshot.prony is None or snapshot.gf_settings is None:
        raise WaveFieldError('snapshot lacks the realised Green-function decomposition; prepare fields again')
    if _json(effective)!=_json(snapshot.gf_settings) or default.exportable_settings['green_function']!=snapshot.provider.get('green_function'):
        raise WaveFieldError('installed Green-function settings differ from the captured backend settings')
    kh=float(problem.wavenumber*problem.water_depth)
    if kh!=snapshot.prony_kh:raise WaveFieldError('retained decomposition belongs to a different dimensionless wavenumber')
    retained=np.array(snapshot.prony);method=snapshot.prony_method;retained_kh=snapshot.prony_kh

    class _Replay(type(default)):
        def find_best_exponential_decomposition(self,dimensionless_wavenumber,*,method=None):
            requested=(self.finite_depth_prony_decomposition_method if method is None else method).lower()
            if float(dimensionless_wavenumber)!=retained_kh or requested!=snapshot.prony_method:
                raise WaveFieldError('replayed Green function asked for a decomposition that was not retained')
            return np.array(retained)
        __hash__=type(default).__hash__
    _Replay.__name__=type(default).__name__
    return _Replay(**effective)


def _capytaine_total_potential(snapshot,points):
    """Public Capytaine postprocessing at hydro-frame points (unit amplitude); never solves."""
    import capytaine as cpt
    from capytaine.bem.airy_waves import airy_waves_potential
    from ._pressure_body import prepare_body
    snapshot.validate()
    r=snapshot.request
    # Retained recovery is bound to the backend that produced the sources: refuse a different installed provider.
    if (snapshot.provider.get('backend')!='capytaine' or snapshot.provider.get('version')!=cpt.__version__
            or snapshot.provider.get('path')!=str(Path(cpt.__file__).resolve())):
        raise WaveFieldError('retained field sources were produced by a different Capytaine provider; prepare fields again')
    v,f,labels,mesh,body=prepare_body(cpt,snapshot.vertices,snapshot.faces,snapshot.surface_ids,period=math.tau/snapshot.omega,
        heading=r['heading'],water_depth=r['water_depth'],rho=r['rho'],max_panels=600)
    problem=cpt.DiffractionProblem(body=body,omega=snapshot.omega,wave_direction=r['heading'],water_depth=r['water_depth'],
        rho=r['rho'],g=r['gravity'])
    result=problem.make_results_container(sources=np.asarray(snapshot.sources))
    solver=cpt.BEMSolver(green_function=_replay_green_function(cpt,snapshot,problem))
    pts=np.asarray(points,dtype=float)
    return np.asarray(solver.compute_potential(pts,result))+np.asarray(airy_waves_potential(pts,problem))


@dataclass(frozen=True)
class SurfaceFields:
    """Unit-amplitude complex fields at the requested target points, from one solution."""
    omega: float
    snapshot_id: str
    eta: np.ndarray       # at (x,y,0), fluid side
    pd0: np.ndarray       # incident+diffraction pressure at (x,y,0)
    pd: np.ndarray        # at (x,y,min(z,0))
    consistency: float    # max |pd0-rho g eta| / max(1,|pd0|)


def evaluate_surface_fields(snapshot,points,fluid_normals,*,offset=DEFAULT_FLUID_OFFSET,evaluator=None,waterline=None):
    """eta and pressure at target points (hydro frame, z relative to the mean waterline) for unit amplitude.

    Evaluation is moved ``offset`` m along the supplied fluid-side (outward hull) normal so that waterline and
    corner points are evaluated on the fluid side.  ``evaluator(snapshot, points)`` may replace the default
    Capytaine postprocessing (tests / alternative public providers); it must return the total potential.
    """
    snapshot.validate()
    p=np.asarray(points,dtype=float);n=np.asarray(fluid_normals,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or p.shape!=n.shape or not np.isfinite(p).all() or not np.isfinite(n).all():
        raise WaveFieldError('matching finite N by 3 points and fluid normals required')
    norm=np.linalg.norm(n,axis=1)
    if np.any(norm<1e-12) or not math.isfinite(offset) or offset<0:raise WaveFieldError('unit fluid normals and nonnegative offset required')
    n=n/norm[:,None]
    surface=np.array(p);surface[:,2]=0.
    below=np.array(p);below[:,2]=np.minimum(p[:,2],0.)
    evaluate=evaluator or _capytaine_total_potential
    r=snapshot.request
    # Waterline fields are only meaningful on the fluid side of the hull waterline contour; points whose
    # waterline projection would fall inside the hull (e.g. the bottom plate) are excluded by the caller's mask
    # and report exactly zero eta/pd0 (they are never used: those points are far below the surface).
    mask=np.ones(len(p),dtype=bool) if waterline is None else np.asarray(waterline,dtype=bool)
    if mask.shape!=(len(p),):raise WaveFieldError('waterline mask must have one entry per point')
    phi0=np.zeros(len(p),dtype=complex)
    if mask.any():
        phi0[mask]=np.asarray(evaluate(snapshot,(surface+offset*n*np.array([1.,1.,0.]))[mask]),dtype=complex)
    phi=np.asarray(evaluate(snapshot,below+offset*n),dtype=complex)
    w=snapshot.omega
    eta=1j*w*phi0/r['gravity'];pd0=1j*w*r['rho']*phi0;pd=1j*w*r['rho']*phi
    if not all(np.isfinite(a).all() for a in (eta,pd0,pd)):raise WaveFieldError('non-finite recovered field')
    residual=float(np.max(np.abs(pd0-r['rho']*r['gravity']*eta)/np.maximum(1.,np.abs(pd0)))) if len(eta) else 0.
    if residual>1e-9:raise WaveFieldError('waterline fields violate pd0=rho*g*eta')
    return SurfaceFields(snapshot.omega,snapshot.snapshot_id,eta,pd0,pd,residual)
