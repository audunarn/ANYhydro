"""Compact excitation responses and separately retained fixed-body solutions.

Neutral records/recovery do not import Capytaine. Only missing cold frequencies
instantiate its unchanged default Green function and indirect serial solver.
"""
from dataclasses import dataclass
import hashlib,importlib.metadata,importlib.util,json,math,os,re,tempfile,time
from pathlib import Path
import numpy as np
from .pressure_bank import (PressureResponseBank,PressureBankError,PreparationProgress,
    PressurePreparationError,CONVENTIONS,_array)

VERSION='excitation_bank_v2'
RECIPE='fixed_zero_speed_surface_potentials_v1'
FORCE_CONVENTIONS=dict(units=('N/m','N/m','N/m','Nm/m','Nm/m','Nm/m'),
    phasor='exp(-i omega t)',normals='outward',pressure='positive inward',
    physics='fixed incident+diffraction',moment_origin=(0.,0.,0.))


def _json(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def _hash(value):return hashlib.sha256(_json(value).encode()).hexdigest()
def _numeric_hash(metadata,arrays):
    h=hashlib.sha256(_json(metadata).encode())
    for name,a in sorted(arrays.items()):
        a=np.ascontiguousarray(a);h.update(_json((name,a.shape,a.dtype.str)).encode());h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def _sha(value):return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def canonical_heading(value):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not math.isfinite(value):
        raise PressureBankError('finite real propagation heading required')
    angle=float(value)%math.tau
    return 0. if angle==math.tau else angle


def pressure_field_digest(request,omega,pressure,*,vertices,faces,surface_ids,panel_ids,centers,normals,areas):
    """Exact geometry/order/physical-setting bound complex unit-pressure row."""
    metadata=dict(recipe='exact_unit_pressure_field_v1',omega=float(omega).hex(),surface_ids=tuple(surface_ids),
        panel_ids=tuple(panel_ids),heading=request['heading'],water_depth=request['water_depth'],rho=request['rho'],
        gravity=request['gravity'],phase_origin=request['phase_origin'],conventions=request['conventions'])
    return _numeric_hash(metadata,dict(vertices=np.asarray(vertices,dtype=np.float64),faces=np.asarray(faces,dtype=np.int64),
        centers=np.asarray(centers,dtype=np.float64),normals=np.asarray(normals,dtype=np.float64),areas=np.asarray(areas,dtype=np.float64),
        pressure=np.asarray(pressure,dtype=np.complex128)))


def validate_pressure_binding(forces,bank):
    forces.validate();bank.validate();request=json.loads(forces.request_json)
    if (bank.heading!=request['heading'] or bank.water_depth!=request['water_depth'] or bank.rho!=request['rho'] or
            bank.gravity!=request['gravity'] or tuple(bank.phase_origin)!=(0.,0.,0.) or bank.conventions!=CONVENTIONS or
            tuple(bank.omegas)!=tuple(forces.omegas) or
            _numeric_hash(dict(surface_ids=bank.surface_ids),dict(vertices=bank.vertices,faces=bank.faces))!=request['geometry']):
        raise PressureBankError('offline pressure physical settings/geometry differ from excitation request')
    if request.get('migration',{}).get('original_pressure_bank_id')==bank.bank_id:return
    for w,row,digest in zip(bank.omegas,bank.responses,forces.pressure_digests):
        current=pressure_field_digest(request,w,row,vertices=bank.vertices,faces=bank.faces,surface_ids=bank.surface_ids,
            panel_ids=bank.panel_ids,centers=bank.centers,normals=bank.normals,areas=bank.areas)
        if current!=digest:raise PressureBankError('offline pressure field identity/digest differs from excitation response')


def _validate_request(request,providers,count):
    keys={'version','recipe','geometry','heading','water_depth','rho','gravity','phase_origin','conventions','backend'}
    if type(request) is not dict or set(request) not in (keys,keys|{'migration'}):raise PressureBankError('invalid excitation request schema')
    if (request['version']!=VERSION or request['recipe']!=RECIPE or not _sha(request['geometry']) or
            _json(request['conventions'])!=_json(CONVENTIONS) or tuple(request['phase_origin'])!=(0.,0.,0.)):
        raise PressureBankError('invalid excitation physical conventions')
    for name in ('heading','water_depth','rho','gravity'):
        value=request[name]
        if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or (value<=0 if name!='heading' else not 0<=value<math.tau):
            raise PressureBankError('invalid excitation physical settings')
    backend=request['backend']
    if (type(backend) is not dict or set(backend)!={'backend','version','path','code','numpy','scipy','recipe'} or
            backend['backend']!='capytaine' or backend['recipe']!=RECIPE or
            any(type(backend[k]) is not str or not backend[k] for k in ('version','path','numpy','scipy')) or
            type(backend['code']) is not dict or not backend['code'] or any(type(k) is not str or not _sha(v) for k,v in backend['code'].items())):
        raise PressureBankError('invalid excitation backend identity')
    if type(providers) is not list or len(providers)!=count:raise PressureBankError('provider inventory differs from frequency inventory')
    migration=request.get('migration')
    if migration is not None and (type(migration) is not dict or set(migration)!={'kind','original_pressure_bank_id','original_provenance'} or
            migration['kind']!='legacy_pressure_to_equivalent_total_potential_v1' or not _sha(migration['original_pressure_bank_id']) or
            type(migration['original_provenance']) is not str):raise PressureBankError('invalid explicit legacy migration namespace')
    for provider in providers:
        if type(provider) is not dict:raise PressureBankError('invalid provider record')
        if migration is not None:
            if provider!={**migration,'recipe':RECIPE}:raise PressureBankError('legacy provider identity mismatch')
        elif (set(provider)!={'backend','version','path','effective_settings','method','n_threads','jobs','recipe','engine','green_function'} or
                provider['backend']!='capytaine' or provider['version']!=backend['version'] or provider['path']!=backend['path'] or
                provider['method']!='indirect' or provider['n_threads']!=1 or provider['jobs']!=1 or provider['recipe']!=RECIPE or
                type(provider['effective_settings']) is not dict or provider['effective_settings'].get('method')!='indirect' or
                any(type(provider[k]) is not str or not provider[k] for k in ('engine','green_function'))):
            raise PressureBankError('captured default provider settings mismatch')


def _atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,suffix='.tmp')
    try:
        with os.fdopen(fd,'w',encoding='utf8') as stream:
            stream.write(_json(value));stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def backend_fingerprint():
    spec=importlib.util.find_spec('capytaine')
    if spec is None or not spec.origin:raise PressureBankError('Capytaine provider unavailable')
    provider=Path(spec.origin).resolve();root=provider.parent
    paths=[p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.py','.pyd','.so','.dll')]
    paths += [Path(__file__).resolve(),Path(__file__).with_name('_pressure_body.py').resolve()]
    return dict(backend='capytaine',version=importlib.metadata.version('capytaine'),path=str(provider),
        code={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},
        numpy=importlib.metadata.version('numpy'),scipy=importlib.metadata.version('scipy'),recipe=RECIPE)


def _request(vertices,faces,surface_ids,*,heading,water_depth,rho,gravity,fingerprint):
    heading=canonical_heading(heading)
    v=_array(vertices,np.float64);raw=np.asarray(faces)
    if raw.dtype.kind not in 'iu':raise PressureBankError('integer source faces required')
    f=_array(faces,np.int64);labels=tuple(surface_ids)
    if (v.ndim!=2 or v.shape[1:]!=(3,) or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1:]!=(4,) or
            not 0<len(f)<=600 or len(labels)!=len(f) or any(type(x) is not str for x in labels) or
            f.min()<0 or f.max()>=len(v) or v[:,2].max()>1e-9 or v[:,2].min()<=-water_depth or
            not math.isfinite(heading) or any(not math.isfinite(x) or x<=0 for x in (water_depth,rho,gravity))):
        raise PressureBankError('invalid admitted source geometry/physical settings')
    request=dict(version=VERSION,recipe=RECIPE,geometry=_numeric_hash(dict(surface_ids=labels),dict(vertices=v,faces=f)),
        heading=heading,water_depth=float(water_depth),rho=float(rho),gravity=float(gravity),
        phase_origin=(0.,0.,0.),conventions=CONVENTIONS,backend=fingerprint)
    return v,f,labels,request


def _frequencies(omegas):
    raw=tuple(omegas)
    if not raw or any(isinstance(x,(bool,np.bool_,complex,np.complexfloating)) or not isinstance(x,(int,float,np.integer,np.floating)) for x in raw):
        raise PressureBankError('exact real frequencies required')
    result=tuple(float(x) for x in raw)
    if any(not math.isfinite(x) or x<=0 for x in result) or len(set(result))!=len(result):
        raise PressureBankError('unique positive exact frequencies required')
    return result


@dataclass(frozen=True)
class ExcitationResponseBank:
    omegas: np.ndarray
    responses: np.ndarray
    solution_ids: tuple
    request_json: str
    providers_json: str
    bank_id: str=''
    pressure_digests: tuple=()
    def __post_init__(self):
        object.__setattr__(self,'omegas',_array(_frequencies(self.omegas),np.float64))
        object.__setattr__(self,'responses',_array(self.responses,np.complex128))
        object.__setattr__(self,'solution_ids',tuple(self.solution_ids))
        object.__setattr__(self,'pressure_digests',tuple(self.pressure_digests))
        object.__setattr__(self,'request_json',_json(json.loads(self.request_json)))
        object.__setattr__(self,'providers_json',_json(json.loads(self.providers_json)))
        _validate_request(json.loads(self.request_json),json.loads(self.providers_json),len(self.omegas))
        if len(self.pressure_digests)!=len(self.omegas) or any(not _sha(x) for x in self.pressure_digests):
            raise PressureBankError('exact pressure-field digest inventory required')
        if self.responses.shape!=(len(self.omegas),6) or not np.isfinite(self.responses).all() or len(self.solution_ids)!=len(self.omegas) or any(type(x) is not str or re.fullmatch('[0-9a-f]{64}',x) is None for x in self.solution_ids):
            raise PressureBankError('invalid compact excitation inventory')
        identity=_numeric_hash(self._metadata(),dict(omegas=self.omegas,responses=self.responses))
        if self.bank_id and self.bank_id!=identity:raise PressureBankError('excitation bank content identity mismatch')
        object.__setattr__(self,'bank_id',identity)
    def _metadata(self):return dict(version=VERSION,conventions=FORCE_CONVENTIONS,solution_ids=self.solution_ids,
        pressure_digests=self.pressure_digests,request=json.loads(self.request_json),providers=json.loads(self.providers_json))
    @property
    def heading(self):return json.loads(self.request_json)['heading']
    @property
    def phase_origin(self):return (0.,0.,0.)
    @property
    def conventions(self):return dict(FORCE_CONVENTIONS)
    @property
    def panel_ids(self):return (0,1,2,3,4,5)
    def validate(self):
        if _numeric_hash(self._metadata(),dict(omegas=self.omegas,responses=self.responses))!=self.bank_id:
            raise PressureBankError('excitation bank changed')
    def to_dict(self):
        self.validate();return dict(**self._metadata(),omegas=self.omegas.tolist(),real=self.responses.real.tolist(),
            imag=self.responses.imag.tolist(),bank_id=self.bank_id)
    @classmethod
    def from_dict(cls,data):
        try:
            if set(data)!={'version','conventions','solution_ids','pressure_digests','request','providers','omegas','real','imag','bank_id'} or data['version']!=VERSION or _json(data['conventions'])!=_json(FORCE_CONVENTIONS) or not _sha(data['bank_id']):
                raise PressureBankError('unsupported excitation bank record')
            real=np.asarray(data['real']);imag=np.asarray(data['imag'])
            if real.shape!=imag.shape or real.dtype.kind not in 'fiu' or imag.dtype.kind not in 'fiu':raise PressureBankError('real force parts required')
            values=np.empty(real.shape,complex);values.real=real;values.imag=imag
            return cls(data['omegas'],values,tuple(data['solution_ids']),_json(data['request']),_json(data['providers']),data['bank_id'],tuple(data['pressure_digests']))
        except PressureBankError:raise
        except (ValueError,TypeError,KeyError) as error:raise PressureBankError('corrupt compact excitation bank') from error


def _paths(cache,request,omega):
    return Path(cache)/'force-entries'/(_hash(dict(request=request,omega=float(omega).hex()))+'.json')


def _save_solution(cache,metadata,arrays):
    identity=_numeric_hash(metadata,arrays);folder=Path(cache)/'solutions';folder.mkdir(parents=True,exist_ok=True)
    path=folder/(identity+'.npz');fd,tmp=tempfile.mkstemp(dir=folder,suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez(stream,metadata=_json(metadata),**arrays);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return identity


def _entry(cache,request,w,force,solution_id,provider,pressure_digest):
    value=dict(request=request,omega=float(w),real=np.asarray(force).real.tolist(),imag=np.asarray(force).imag.tolist(),
        solution_id=solution_id,provider=provider,pressure_digest=pressure_digest)
    value['entry_id']=_hash(value);_atomic_json(_paths(cache,request,w),value);return value


def _load_entry(cache,request,w):
    path=_paths(cache,request,w)
    if not path.exists():return None
    try:
        value=json.loads(path.read_text())
        if type(value) is not dict:raise PressureBankError('invalid per-frequency excitation entry schema')
        identity=value.pop('entry_id')
        if _hash(value)!=identity or _json(value['request'])!=_json(request) or value['omega']!=w or not _sha(value['solution_id']) or not _sha(value['pressure_digest']):
            raise PressureBankError('excitation entry integrity mismatch')
        if np.shape(value['real'])!=(6,) or np.shape(value['imag'])!=(6,):raise PressureBankError('six force entries required')
        value['entry_id']=identity;return value
    except PressureBankError:raise
    except (ValueError,TypeError,KeyError,OSError) as error:raise PressureBankError('corrupt excitation entry') from error


def _assemble(cache,request,omegas,entries):
    bank=ExcitationResponseBank(omegas,np.array([np.array(e['real'])+1j*np.array(e['imag']) for e in entries]),
        tuple(e['solution_id'] for e in entries),_json(request),_json([e['provider'] for e in entries]),
        pressure_digests=tuple(e['pressure_digest'] for e in entries))
    _atomic_json(Path(cache)/'force-banks'/(_hash(dict(request=request,omegas=[w.hex() for w in omegas]))+'.json'),bank.to_dict())
    return bank


def prepare_excitation_bank(vertices,faces,surface_ids,omegas,*,cache_directory,heading=0.,water_depth=200.,rho=1025.,
        gravity=9.81,max_panels=600,max_missing_solves=32,max_wall_seconds=120.,cancel=None,progress=None,repair_cache=False):
    started=time.perf_counter();omegas=_frequencies(omegas)
    if type(repair_cache) is not bool:raise PressureBankError('repair_cache must be an explicit boolean')
    if type(max_panels) is not int or not 0<max_panels<=600 or type(max_missing_solves) is not int or max_missing_solves<0 or isinstance(max_wall_seconds,bool) or not math.isfinite(max_wall_seconds) or max_wall_seconds<=0:
        raise PressureBankError('explicit valid preparation limits required')
    v,f,labels,request=_request(vertices,faces,surface_ids,heading=heading,water_depth=water_depth,rho=rho,gravity=gravity,fingerprint=backend_fingerprint())
    if len(f)>max_panels:raise PressureBankError('source exceeds preparation panel limit')
    entries={};cached=0;solved=0
    def state():return PreparationProgress(tuple(w for w in omegas if w in entries),tuple(w for w in omegas if w not in entries),cached,solved,time.perf_counter()-started)
    def stop():
        if (cancel is not None and cancel()) or time.perf_counter()-started>=max_wall_seconds:
            raise PressurePreparationError('excitation preparation cancelled or wall budget reached',state())
    stop()
    assembled=Path(cache_directory)/'force-banks'/(_hash(dict(request=request,omegas=[w.hex() for w in omegas]))+'.json')
    if assembled.exists() and not repair_cache:
        try:bank=ExcitationResponseBank.from_dict(json.loads(assembled.read_text()))
        except (ValueError,TypeError,OSError) as error:raise PressureBankError('corrupt complete excitation bank') from error
        if bank.request_json!=_json(request) or tuple(bank.omegas)!=omegas:raise PressureBankError('stale complete excitation bank')
        entries.update(dict.fromkeys(omegas));cached=len(omegas);stop()
        if progress is not None:progress(state())
        return bank
    for w in omegas:
        stop()
        try:
            entry=_load_entry(cache_directory,request,w)
            if entry is not None and repair_cache:
                check=ExcitationResponseBank([w],[np.array(entry['real'])+1j*np.array(entry['imag'])],(entry['solution_id'],),
                    _json(request),_json([entry['provider']]),pressure_digests=(entry['pressure_digest'],))
                recover_pressure_bank(check,cache_directory=cache_directory,cancel=cancel)
        except PressureBankError:
            if not repair_cache:raise
            entry=None
        if entry is not None:entries[w]=entry;cached+=1
    if progress is not None:progress(state())
    if len(omegas)-len(entries)>max_missing_solves:raise PressurePreparationError('missing excitation solve count exceeds limit',state())
    if len(entries)<len(omegas):
        import capytaine as cpt
        from capytaine.bem.airy_waves import airy_waves_potential,froude_krylov_force
        from ._pressure_body import prepare_body
        if request['backend']['version'].split('.')[:2]!=['3','0']:raise PressureBankError('requires Capytaine3.0')
        v,f,labels,mesh,body=prepare_body(cpt,v,f,labels,period=math.tau/omegas[0],heading=request['heading'],water_depth=water_depth,rho=rho,max_panels=max_panels)
        solver=cpt.BEMSolver()
        provider=dict(backend='capytaine',version=cpt.__version__,path=str(Path(cpt.__file__).resolve()),
            effective_settings=solver.exportable_settings,method='indirect',n_threads=1,jobs=1,recipe=RECIPE,
            engine=repr(solver.engine),green_function=repr(solver.engine.green_function))
        for w in omegas:
            if w in entries:continue
            stop()
            try:
                pb=cpt.DiffractionProblem(body=body,omega=w,wave_direction=request['heading'],water_depth=water_depth,rho=rho,g=gravity)
                result=solver.solve(pb,method='indirect',keep_details=True,n_threads=1)
                incident=airy_waves_potential(mesh.faces_centers,pb)
                fk=froude_krylov_force(pb)
                force=np.array([result.forces[key]+fk[key] for key in ('Surge','Sway','Heave','Roll','Pitch','Yaw')])
                pressure=1j*w*rho*np.asarray(result.potential)+1j*w*rho*incident
                _check_wrench(pressure,mesh.faces_centers,mesh.faces_normals,mesh.faces_areas,force)
                panel_ids=tuple(range(len(f)))
                digest=pressure_field_digest(request,w,pressure,vertices=v,faces=f,surface_ids=labels,panel_ids=panel_ids,
                    centers=mesh.faces_centers,normals=mesh.faces_normals,areas=mesh.faces_areas)
                metadata=dict(request=request,omega=w,surface_ids=labels,panel_ids=panel_ids,provider=provider,
                    pressure_digest=digest,representation='diffraction+incident_potentials')
                arrays=dict(vertices=v,faces=f,centers=mesh.faces_centers,normals=mesh.faces_normals,areas=mesh.faces_areas,
                    diffraction_potential=np.asarray(result.potential),incident_potential=np.asarray(incident),force=force)
                sid=_save_solution(cache_directory,metadata,arrays)
                entries[w]=_entry(cache_directory,request,w,force,sid,provider,digest);solved+=1
                del result,pb,pressure,incident
                if progress is not None:progress(state())
                stop()
            except PressurePreparationError:raise
            except Exception as error:raise PressurePreparationError(f'excitation solve failed: {error}',state()) from error
    bank=_assemble(cache_directory,request,omegas,[entries[w] for w in omegas]);stop();return bank


def _check_wrench(pressure,centers,normals,areas,force):
    values=-np.asarray(pressure)[:,None]*np.asarray(areas)[:,None]*np.asarray(normals)
    integrated=np.r_[values.sum(axis=0),np.cross(centers,values).sum(axis=0)]
    if not np.isfinite(integrated).all() or not np.allclose(integrated,force,rtol=1e-8,atol=1e-6):
        raise PressureBankError('recovered panel pressure differs from excitation wrench')


def recover_pressure_bank(bank,*,cache_directory,cancel=None):
    """Recover selected-source panel fields from retained data; never BEM solve."""
    bank.validate();rows=[];first=None;providers=json.loads(bank.providers_json)
    for index,(w,sid) in enumerate(zip(bank.omegas,bank.solution_ids)):
        if cancel is not None and cancel():raise PressureBankError('pressure recovery cancelled')
        path=Path(cache_directory)/'solutions'/(sid+'.npz')
        try:
            with np.load(path,allow_pickle=False) as data:
                metadata=json.loads(str(data['metadata']));arrays={name:data[name].copy() for name in data.files if name!='metadata'}
            if _numeric_hash(metadata,arrays)!=sid or _json(metadata['request'])!=bank.request_json or metadata['omega']!=w or metadata['provider']!=providers[index]:
                raise PressureBankError('retained solution identity/provider mismatch')
            request=metadata['request'];pressure=1j*float(w)*request['rho']*arrays['diffraction_potential']+1j*float(w)*request['rho']*arrays['incident_potential']
            if not 0<len(arrays['faces'])<=600:raise PressureBankError('retained solution panel bound exceeded')
            if _numeric_hash(dict(surface_ids=metadata['surface_ids']),dict(vertices=arrays['vertices'],faces=arrays['faces']))!=request['geometry']:
                raise PressureBankError('retained solution geometry mismatch')
            _check_wrench(pressure,arrays['centers'],arrays['normals'],arrays['areas'],bank.responses[index])
            digest=pressure_field_digest(request,w,pressure,vertices=arrays['vertices'],faces=arrays['faces'],
                surface_ids=metadata['surface_ids'],panel_ids=metadata['panel_ids'],centers=arrays['centers'],normals=arrays['normals'],areas=arrays['areas'])
            if digest!=bank.pressure_digests[index] or digest!=metadata['pressure_digest']:
                raise PressureBankError('retained exact pressure field digest differs from compact response')
            if not np.array_equal(arrays['force'],bank.responses[index]):raise PressureBankError('retained excitation response changed')
            if first is None:first=(metadata,arrays)
            else:
                for key in ('vertices','faces','centers','normals','areas'):
                    if not np.array_equal(first[1][key],arrays[key]):raise PressureBankError('retained solution geometry/order changed')
                if first[0]['panel_ids']!=metadata['panel_ids'] or first[0]['surface_ids']!=metadata['surface_ids']:
                    raise PressureBankError('retained source panel IDs/order changed')
            rows.append(pressure)
        except PressureBankError:raise
        except Exception as error:raise PressureBankError(f'missing/corrupt retained solution {sid}; prepare hydrodynamics explicitly') from error
    metadata,a=first;r=metadata['request']
    result=PressureResponseBank(a['vertices'],a['faces'],tuple(metadata['surface_ids']),a['centers'],a['normals'],a['areas'],bank.omegas,
        np.stack(rows),r['heading'],r['water_depth'],r['rho'],r['gravity'],panel_ids=tuple(metadata['panel_ids']),provenance_json=_json(dict(excitation_bank_id=bank.bank_id,
            providers=providers,recovery_recipe=RECIPE,solution_ids=bank.solution_ids)))
    if cancel is not None and cancel():raise PressureBankError('pressure recovery cancelled')
    return result


def migrate_pressure_bank(bank,*,cache_directory,progress=None):
    """Explicit legacy pressure-only migration, not an original BEM potential."""
    bank.validate()
    if tuple(bank.phase_origin)!=(0.,0.,0.) or bank.conventions!=CONVENTIONS:
        raise PressureBankError('legacy migration requires exact WORLD0 SI pressure conventions')
    v,f,labels,request=_request(bank.vertices,bank.faces,bank.surface_ids,heading=bank.heading,
        water_depth=bank.water_depth,rho=bank.rho,gravity=bank.gravity,fingerprint=backend_fingerprint())
    migration=dict(kind='legacy_pressure_to_equivalent_total_potential_v1',original_pressure_bank_id=bank.bank_id,
        original_provenance=bank.provenance_json)
    request['migration']=migration
    provider={**migration,'recipe':RECIPE}
    entries=[]
    for w,pressure in zip(bank.omegas,bank.responses):
        values=-pressure[:,None]*bank.areas[:,None]*bank.normals
        force=np.r_[values.sum(axis=0),np.cross(bank.centers,values).sum(axis=0)]
        potential=pressure/(1j*float(w)*bank.rho);incident=np.zeros_like(potential)
        recovered=1j*float(w)*bank.rho*potential+1j*float(w)*bank.rho*incident
        bound=1e-12*max(1.,float(np.max(np.abs(pressure))))
        if np.max(np.abs(recovered-pressure))>bound:raise PressureBankError('legacy potential migration roundoff certificate failed')
        _check_wrench(recovered,bank.centers,bank.normals,bank.areas,force)
        digest=pressure_field_digest(request,w,recovered,vertices=v,faces=f,surface_ids=labels,panel_ids=bank.panel_ids,
            centers=bank.centers,normals=bank.normals,areas=bank.areas)
        metadata=dict(request=request,omega=float(w),surface_ids=labels,panel_ids=bank.panel_ids,provider=provider,
            pressure_digest=digest,representation='legacy_equivalent_total_potential',original_pressure_bank_id=bank.bank_id,roundoff_bound=bound)
        sid=_save_solution(cache_directory,metadata,dict(vertices=v,faces=f,centers=bank.centers,normals=bank.normals,areas=bank.areas,
            diffraction_potential=potential,incident_potential=incident,force=force))
        entries.append(_entry(cache_directory,request,float(w),force,sid,provider,digest))
    if progress is not None:progress('Legacy pressure-only source explicitly migrated to equivalent retained solution/force cache (no BEM)')
    return _assemble(cache_directory,request,tuple(float(w) for w in bank.omegas),entries)
