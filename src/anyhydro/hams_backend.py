# SPDX-License-Identifier: MPL-2.0
"""Finite-grid corrected HAMS native backend, with preserved failure evidence."""
from dataclasses import dataclass,replace
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
import hashlib,json,math,os,time,uuid,threading,stat
import numpy as np
from ._owned_process import OwnedProcess
from .backends import HydrodynamicsBackend,HydroBackendCapabilities,Diagnostic
from .errors import CapabilityError,HydroError
from .frames import CoefficientFrame
from .hashing import dataset_content_hash
from .hams_inputs import HamsPreparedDeck,prepare_hams_deck,_syntax,_bytes
from .hams_import import HamsImportContract,import_hams_coefficients,PROFILE,LIMIT_PROFILE,GRAVITY_PROFILE
from .hams_producer import HamsProducerBuild,LIMIT_SOURCE_IDENTITY,GRAVITY_SOURCE_IDENTITY
from .hams_field_points import normalize_field_points,admit_field_points
from .hams_fields import import_hams_fields,save_hams_fields,load_hams_fields
from .persistence import save_hydro_dataset,load_hydro_dataset
from .records import HydroBody,HydroRequest

__all__=['HamsBackend','HamsExecutionLimits','HamsExecutionError','HamsCancelledError',
         'HamsDeadlineError','execute_prepared_hams','load_completed_hams_run','load_completed_hams_fields']

class HamsExecutionError(HydroError):pass
class HamsCancelledError(HamsExecutionError):pass
class HamsDeadlineError(HamsExecutionError):pass


@dataclass(frozen=True)
class HamsExecutionLimits:
    wall_seconds: float=60.
    threads: int=1
    max_bodies: int=2
    max_frequencies: int=128
    max_headings: int=36
    max_panels: int=20000
    max_field_points: int=4096
    def __post_init__(self):
        if isinstance(self.wall_seconds,bool) or not isinstance(self.wall_seconds,(int,float)) or not math.isfinite(self.wall_seconds) or self.wall_seconds<=0:
            raise HamsExecutionError('Positive finite wall budget required')
        if type(self.threads) is not int or self.threads!=1:raise HamsExecutionError('Admitted producer uses one thread')
        for name in ('max_bodies','max_frequencies','max_headings','max_panels','max_field_points'):
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:raise HamsExecutionError('Positive explicit inventory bounds required')
    def identity(self):return dict(vars(self))


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _linked(path):
    if path.is_symlink():return True
    try:attributes=getattr(path.lstat(),'st_file_attributes',0)
    except FileNotFoundError:return False
    return bool(attributes & getattr(stat,'FILE_ATTRIBUTE_REPARSE_POINT',0x400))
def _case_paths(path):
    if _linked(path):raise HamsExecutionError('Linked case directory is not admitted')
    for item in path.rglob('*'):
        if _linked(item) or not (item.is_dir() or item.is_file()):
            raise HamsExecutionError('Linked or nonregular case artifact is not admitted')
def _output_files(path):
    if _linked(path) or not path.is_dir():raise HamsExecutionError('Owned output directory required')
    files=[]
    for item in path.rglob('*'):
        if _linked(item):
            raise HamsExecutionError('Linked producer artifacts are not admitted')
        if item.is_file():files.append(item)
        elif not item.is_dir():raise HamsExecutionError('Nonregular producer artifact')
    return files
def _write(path,value):
    temp=path.with_name(path.name+'.tmp')
    if _linked(path):raise HamsExecutionError('Linked evidence artifact is not admitted')
    with temp.open('xb') as stream:stream.write(_bytes(value))
    os.replace(temp,path)
def _command(build):return (str(build.executable_path),'Input','Output')
def _snapshot(bodies,request):
    if type(request) is not HydroRequest or not bodies or any(type(b) is not HydroBody for b in bodies):
        raise HamsExecutionError('Authoritative body/request records required')
    if request.backend_settings:raise CapabilityError('This explicit HAMS profile does not admit backend_settings')
    return tuple(replace(b,mesh=replace(b.mesh),lid_mesh=None if b.lid_mesh is None else replace(b.lid_mesh),
        attached_surfaces=tuple(replace(m) for m in b.attached_surfaces)) for b in bodies),replace(request)
def _cb(values):
    if not isinstance(values,Mapping) or not values:raise HamsExecutionError('Explicit WORLD CB records required')
    result={}
    for key,value in values.items():
        raw=np.asarray(value,dtype=object)
        if raw.shape!=(3,) or any(isinstance(x,(bool,np.bool_)) for x in raw):raise HamsExecutionError('CB requires three finite numbers')
        try:arr=np.asarray(value,dtype=float)
        except (TypeError,ValueError) as exc:raise HamsExecutionError('Finite CB record required') from exc
        if not np.isfinite(arr).all():raise HamsExecutionError('Finite CB record required')
        result[key]=tuple(float(x) for x in arr)
    return MappingProxyType(result)
def _limits(bodies,request,limits,build,field_points=()):
    if type(limits) is not HamsExecutionLimits:raise HamsExecutionError('Explicit execution limits required')
    if request.include_zero_frequency:
        raise CapabilityError('Canonical zero-frequency operator unsupported; incidental A0 evidence only')
    if request.include_infinite_frequency and build.source_tree_sha256!=LIMIT_SOURCE_IDENTITY:
        raise CapabilityError('Ainf requires the exact limit-preserving producer source')
    if field_points and build.source_tree_sha256!=GRAVITY_SOURCE_IDENTITY:raise CapabilityError('Fields require exact gravity-consistent producer source')
    if len(field_points)>limits.max_field_points:raise HamsExecutionError('Field point inventory exceeds execution limits')
    frequency_count=len(request.omegas)+(2 if request.include_infinite_frequency else 0)
    if len(bodies)>limits.max_bodies or frequency_count>limits.max_frequencies or len(request.wave_directions)>limits.max_headings:
        raise HamsExecutionError('Requested body/frequency/heading inventory exceeds execution limits')
    if sum(len(m.faces) for b in bodies for m in (b.mesh,b.lid_mesh,*b.attached_surfaces) if m is not None)>limits.max_panels:
        raise HamsExecutionError('Requested panel inventory exceeds execution limits')


def _import_contract(bodies,request,build,centers,manifest,output,field_points):
    paths={'1':output/'Wamit_format/AmssDamp.1','3':output/'Wamit_format/ExcForce.3','hst':output/'Wamit_format/Hydrostat.hst'}
    native=output/'Hams_format';inputs=manifest['input_sha256']
    field_hashes={k:_sha(output/'Wamit_format'/name) for k,name in (('combined','PressureElevation.6p'),('incident','PressureElevationIncidence.6p'))} if field_points else {}
    contract=HamsImportContract(tuple(b.body_id for b in bodies),{b.body_id:CoefficientFrame(np.eye(3),b.rotation_center) for b in bodies},
        {b.body_id:b.mesh.mesh_hash for b in bodies},{b.body_id:None if b.lid_mesh is None else b.lid_mesh.mesh_hash for b in bodies},dict(centers),
        manifest['reference_length'],request.rho,request.gravity,request.water_depth,request.content_hash(),
        hashlib.sha256(_bytes(inputs)).hexdigest(),build.executable_sha256,{k:_sha(v) for k,v in paths.items()},
        build.version,build.build_description,'pinned source-derived phi=i*psi; F=i*EXFC; WORLD XW0 AMP1; unqualified transport',
        profile=(GRAVITY_PROFILE if build.source_tree_sha256==GRAVITY_SOURCE_IDENTITY else LIMIT_PROFILE if build.source_tree_sha256==LIMIT_SOURCE_IDENTITY else PROFILE),
        input_file_sha256=inputs,source_tree_sha256=build.source_tree_sha256,native_file_sha256={v.name:_sha(v) for v in native.iterdir()},
        source_field_points=field_points,field_file_sha256=field_hashes)
    return contract,paths,native


def _field_import(path,bodies,request,contract,paths,native,dataset,build):
    return import_hams_fields(bodies,request,contract,dataset=dataset,input_directory=path/'Input',executable_path=build.executable_path,
        native_directory=native,radiation_path=paths['1'],excitation_path=paths['3'],hydrostatic_path=paths['hst'],
        combined_path=path/'Output/Wamit_format/PressureElevation.6p',incident_path=path/'Output/Wamit_format/PressureElevationIncidence.6p')


def execute_prepared_hams(deck,bodies,request,build,center_of_buoyancy,*,limits,should_cancel=None,field_points=()):
    """Execute one exclusive native case; only validated sealed output succeeds.

    Cancellation/deadline checks precede launch, every poll, import, persistence
    and final seal. Partial files/logs remain; no failed/cancelled result returns.
    Caller records are copied and revalidated before the native process starts.
    """
    if type(deck) is not HamsPreparedDeck or type(build) is not HamsProducerBuild:
        raise HamsExecutionError('Prepared deck and corrected producer record required')
    if should_cancel is not None and not callable(should_cancel):raise HamsExecutionError('Cancellation callback must be callable')
    bodies,request=_snapshot(tuple(bodies),request);centers=_cb(center_of_buoyancy);field_points=admit_field_points(bodies,request,field_points);_limits(bodies,request,limits,build,field_points)
    if set(centers)!={b.body_id for b in bodies}:raise HamsExecutionError('CB must cover exactly the source bodies')
    path=Path(deck.directory);receipt=path/'run.json';start=time.perf_counter();deadline=start+limits.wall_seconds
    record=dict(format='hams-supervised-run-v1',state='queued',case_hash=deck.case_hash,request_hash=request.content_hash(),
        producer=build.identity(),execution_limits=limits.identity(),qualification='unqualified',started_monotonic=start)
    # Exclusive claim prevents rerunning a completed/partial case or competing
    # owners on the same prepared input. Reproduction uses a new case directory.
    _case_paths(path)
    if receipt.exists():raise HamsExecutionError('Prepared case already claimed; use a new case directory')
    if any((path/name).exists() for name in ('stdout.log','stderr.log','result.nc','fields.json','completed.json','run.json.tmp','completed.json.tmp')):
        raise HamsExecutionError('Exclusive unexecuted case artifacts required')
    try:
        with receipt.open('xb') as stream:stream.write(_bytes(record))
    except FileExistsError as exc:raise HamsExecutionError('Prepared case already claimed; use a new case directory') from exc
    process=None
    def checkpoint():
        if should_cancel is not None and should_cancel():raise HamsCancelledError('HAMS cancellation requested')
        if time.perf_counter()>=deadline:raise HamsDeadlineError('HAMS execution deadline exceeded')
    try:
        checkpoint();build.verify(require_native=True)
        manifest=deck.verify(bodies=bodies,request=request,executable_path=build.executable_path,producer_build=build,field_points=field_points)
        if manifest.get('producer_build')!=build.identity() or manifest['threads']!=limits.threads:
            raise HamsExecutionError('Prepared producer/thread settings differ from effective execution')
        output=path/'Output'
        if _output_files(output) or {p.name for p in output.iterdir()}!={'Hams_format','Wamit_format'} or any(not p.is_dir() for p in output.iterdir()):
            raise HamsExecutionError('Exclusive empty producer output directories required')
        env=os.environ.copy();env.pop('PYTHONPATH',None)
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]=str(limits.threads)
        env['OMP_DISPLAY_ENV']='VERBOSE'
        command=_command(build)
        if Path(command[0]).resolve()!=build.executable_path:raise HamsExecutionError('Launch executable differs from bound producer')
        record.update(command=list(command),environment={key:env[key] for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')})
        checkpoint();deck.verify(executable_path=build.executable_path,producer_build=build)
        with (path/'stdout.log').open('wb') as stdout,(path/'stderr.log').open('wb') as stderr:
            process=OwnedProcess(command,cwd=path,env=env,stdout=stdout,stderr=stderr)
            if hasattr(process,'supervision'):record['supervision']=dict(process.supervision)
            record.update(state='running',pid=process.pid);_write(receipt,record)
            while True:
                checkpoint();code=process.poll()
                if code is not None:break
                time.sleep(min(.02,max(0.,deadline-time.perf_counter())))
            process.close();process=None
        record['exit_code']=code
        checkpoint()
        if code!=0:raise HamsExecutionError(f'Native HAMS producer exited {code}; partial output retained')
        _case_paths(path);_output_files(output)
        deck.verify(bodies=bodies,request=request,executable_path=build.executable_path,producer_build=build)
        contract,paths,native=_import_contract(bodies,request,build,centers,manifest,output,field_points)
        checkpoint()
        dataset=import_hams_coefficients(bodies,request,contract,input_directory=path/'Input',executable_path=build.executable_path,
            native_directory=native,radiation_path=paths['1'],excitation_path=paths['3'],hydrostatic_path=paths['hst'])
        checkpoint();fields=_field_import(path,bodies,request,contract,paths,native,dataset,build) if field_points else None
        checkpoint();result=path/'result.nc';save_hydro_dataset(dataset,result)
        reopened=load_hydro_dataset(result)
        if dataset_content_hash(dataset)!=dataset_content_hash(reopened):raise HamsExecutionError('Persisted result identity changed')
        result_sha256=_sha(result)
        if fields is not None:
            checkpoint();save_hams_fields(fields,path/'fields.json')
            stored=load_hams_fields(path/'fields.json',expected_parent_hash=dataset_content_hash(reopened))
            if stored.content_hash()!=fields.content_hash():raise HamsExecutionError('Persisted field identity changed')
            record['fields']=dict(filename='fields.json',sha256=_sha(path/'fields.json'),content_hash=fields.content_hash())
            checkpoint()
            record['field_centers']=dict(centers)
        deck.verify(bodies=bodies,request=request,executable_path=build.executable_path,producer_build=build,field_points=field_points)
        raw_hashes={str(p.relative_to(output)).replace('\\','/'):_sha(p) for p in _output_files(output)}
        checkpoint()
        record.update(state='completed',wall_seconds=time.perf_counter()-start,result_sha256=result_sha256,dataset_hash=dataset_content_hash(reopened),output_sha256=raw_hashes)
        _write(receipt,record)
        seal=dict(format='hams-completed-seal-v1',receipt_sha256=_sha(receipt),case_hash=deck.case_hash,result_sha256=record['result_sha256'],dataset_hash=record['dataset_hash'])
        if fields is not None:seal['fields_sha256']=record['fields']['sha256']
        # The final seal is the completion commit. Later cancellation belongs to
        # a consumer workflow and cannot relabel an already completed run.
        checkpoint();_case_paths(path)
        # Callbacks may edit artifacts. Recheck every authoritative identity
        # after the last callback, before the completion commit.
        deck.verify(bodies=bodies,request=request,executable_path=build.executable_path,producer_build=build,field_points=field_points)
        if (_sha(receipt)!=seal['receipt_sha256'] or _sha(result)!=result_sha256
            or {str(p.relative_to(output)).replace('\\','/'):_sha(p) for p in _output_files(output)}!=raw_hashes
            or any(_sha(paths[k])!=v for k,v in contract.file_sha256.items())
            or any(_sha(native/k)!=v for k,v in contract.native_file_sha256.items())
            or (fields is not None and (_sha(path/'fields.json')!=record['fields']['sha256']
                or any(_sha(output/'Wamit_format'/name)!=contract.field_file_sha256[k]
                       for k,name in (('combined','PressureElevation.6p'),('incident','PressureElevationIncidence.6p')))))):
            raise HamsExecutionError('Authoritative artifacts changed before completion seal')
        record['wall_seconds']=time.perf_counter()-start
        _write(receipt,record);seal['receipt_sha256']=_sha(receipt)
        # This is a pure deadline check: invoking the callback again would
        # reopen the just-verified artifact mutation window.
        if time.perf_counter()>=deadline:raise HamsDeadlineError('HAMS execution deadline exceeded before completion seal')
        _write(path/'completed.json',seal)
        return reopened
    except BaseException as exc:
        record.update(state='cancelling' if isinstance(exc,HamsCancelledError) else 'failed',error_type=type(exc).__name__,error=str(exc))
        try:_write(receipt,record)
        except Exception as write_error:record['receipt_error']=str(write_error)
        finally:
            if process is not None:
                try:process.close()
                except Exception as cleanup_error:record['cleanup_error']=str(cleanup_error)
                process=None
        record.update(state='cancelled' if isinstance(exc,HamsCancelledError) else 'failed',wall_seconds=time.perf_counter()-start)
        try:_write(receipt,record)
        except Exception:pass  # Preserve the original error even if evidence persistence fails.
        if isinstance(exc,(HamsExecutionError,CapabilityError)):raise
        if not isinstance(exc,Exception):raise
        raise HamsExecutionError(str(exc)) from exc


def _load_completed_hams_result(directory,*,bodies,request,producer_build):
    """Verify the receipt, seal, raw bytes and original inputs before replay."""
    path=Path(directory)
    try:
        _case_paths(path)
        seal=json.loads((path/'completed.json').read_text());record=json.loads((path/'run.json').read_text())
        if seal['format']!='hams-completed-seal-v1' or record['format']!='hams-supervised-run-v1' or record['state']!='completed' or seal['receipt_sha256']!=_sha(path/'run.json'):
            raise HamsExecutionError('No authoritative completed run')
        if record['producer']!=producer_build.identity() or seal['case_hash']!=record['case_hash']:
            raise HamsExecutionError('Run producer/input binding changed')
        bodies=tuple(bodies);deck=HamsPreparedDeck(path,record['case_hash']);manifest=deck.verify(bodies=bodies,request=request,executable_path=producer_build.executable_path,producer_build=producer_build)
        actual={str(p.relative_to(path/'Output')).replace('\\','/'):_sha(p) for p in _output_files(path/'Output')}
        if actual!=record['output_sha256'] or _sha(path/'result.nc')!=seal['result_sha256'] or record['result_sha256']!=seal['result_sha256']:
            raise HamsExecutionError('Run/result/raw artifact identity changed')
        ds=load_hydro_dataset(path/'result.nc')
        if dataset_content_hash(ds)!=seal['dataset_hash'] or record['dataset_hash']!=seal['dataset_hash'] or ds.attrs['request_hash']!=request.content_hash():
            raise HamsExecutionError('Stored canonical result binding changed')
        points=normalize_field_points(manifest.get('field_points',()))
        fields=None
        if points:
            info=record.get('fields')
            if not isinstance(info,dict) or set(info)!={'filename','sha256','content_hash'} or info['filename']!='fields.json':raise HamsExecutionError('Missing authoritative field sidecar binding')
            if _sha(path/'fields.json')!=info['sha256'] or seal.get('fields_sha256')!=info['sha256']:raise HamsExecutionError('Stored field artifact bytes changed')
            fields=load_hams_fields(path/'fields.json',expected_parent_hash=dataset_content_hash(ds))
            if fields.content_hash()!=info['content_hash']:raise HamsExecutionError('Stored field content changed')
            contract,paths,native=_import_contract(bodies,request,producer_build,_cb(record['field_centers']),manifest,path/'Output',points)
            expected=_field_import(path,bodies,request,contract,paths,native,ds,producer_build)
            if fields.content_hash()!=expected.content_hash():raise HamsExecutionError('Stored field semantics do not reproduce authoritative raw source')
        elif 'fields' in record or 'fields_sha256' in seal or 'field_centers' in record:
            raise HamsExecutionError('Unrequested field sidecar binding refused')
        return ds,fields
    except (OSError,ValueError,KeyError,TypeError,HydroError) as exc:
        if isinstance(exc,HamsExecutionError):raise
        raise HamsExecutionError('Incomplete or invalid saved HAMS run: '+str(exc)) from exc


def load_completed_hams_run(directory,*,bodies,request,producer_build):
    return _load_completed_hams_result(directory,bodies=bodies,request=request,producer_build=producer_build)[0]


def load_completed_hams_fields(directory,*,bodies,request,producer_build):
    fields=_load_completed_hams_result(directory,bodies=bodies,request=request,producer_build=producer_build)[1]
    if fields is None:raise CapabilityError('Stored HAMS run has no requested canonical fields')
    return fields


class HamsBackend(HydrodynamicsBackend):
    """Explicit native corrected build; finite coupled coefficients and explicit repaired-source Ainf.

    Headless synchronous service. GUI callers require their isolated worker
    integration; constructing/importing this backend never runs a producer.
    """
    @staticmethod
    def native_supervision_status():
        """Report the effective no-launch ownership contract for this runtime.

        Consumer admission still requires an explicit native producer identity,
        complete deck validation and separate scientific qualification.
        """
        from ._owned_process import native_supervision_status
        return native_supervision_status()
    def __init__(self,build,mechanics,center_of_buoyancy,*,workroot,limits,should_cancel=None,irregular_frequency_removal=False,allow_mechanical_rounding=False,field_points=()):
        if type(build) is not HamsProducerBuild or type(limits) is not HamsExecutionLimits:raise HamsExecutionError('Explicit producer and limits required')
        api=_syntax()
        if not isinstance(mechanics,Mapping) or not mechanics or any(type(v) is not api.HamsMechanicalInput for v in mechanics.values()):
            raise HamsExecutionError('Explicit per-body mechanical records required')
        self.field_points=normalize_field_points(field_points)
        if self.field_points and build.source_tree_sha256!=GRAVITY_SOURCE_IDENTITY:raise CapabilityError('Fields require exact gravity-consistent producer source')
        self.build=build;self.mechanics=MappingProxyType(dict(mechanics));self.centers=_cb(center_of_buoyancy)
        self.workroot=Path(workroot).resolve();self.limits=limits;self.should_cancel=should_cancel
        self.irregular_frequency_removal=irregular_frequency_removal;self.allow_mechanical_rounding=allow_mechanical_rounding;self.last_run_directory=None
        self._run_lock=threading.Lock()
    def solve(self,bodies,request):
        if not self._run_lock.acquire(blocking=False):raise HamsExecutionError('This backend already owns an active run')
        try:return self._solve(bodies,request)
        finally:self._run_lock.release()
    def _solve(self,bodies,request):
        build,mechanics,centers,workroot,limits,cancel,irregular,rounding=(self.build,self.mechanics,self.centers,
            self.workroot,self.limits,self.should_cancel,self.irregular_frequency_removal,self.allow_mechanical_rounding)
        field_points=normalize_field_points(self.field_points)
        bodies,request=_snapshot(tuple(bodies),request);field_points=admit_field_points(bodies,request,field_points);_limits(bodies,request,limits,build,field_points)
        if set(mechanics)!=set(centers) or set(centers)!={b.body_id for b in bodies}:raise HamsExecutionError('Mechanics/CB/body identity mismatch')
        build.verify(require_native=True);workroot.mkdir(parents=True,exist_ok=True)
        directory=workroot/('hams-'+uuid.uuid4().hex);self.last_run_directory=directory
        deck=prepare_hams_deck(directory,bodies,request,mechanics,executable_path=build.executable_path,
            producer_build=build,threads=limits.threads,irregular_frequency_removal=irregular,allow_mechanical_rounding=rounding,field_points=field_points)
        return execute_prepared_hams(deck,bodies,request,build,centers,limits=limits,should_cancel=cancel,field_points=field_points)
    def capabilities(self):
        return HydroBackendCapabilities('HAMS-MREL-corrected',self.build.version,True,False,self.build.source_tree_sha256==LIMIT_SOURCE_IDENTITY,self.build.source_tree_sha256==LIMIT_SOURCE_IDENTITY,False,False,self.build.source_tree_sha256==GRAVITY_SOURCE_IDENTITY and any(p[2]==0 for p in self.field_points),False,
            notes=('native rigid6DOF finite coefficients and source-specific Ainf; explicit build required','unqualified; finite source-specific exterior elevation/dynamic-pressure fields when explicitly configured; not panel loads','A0 operator, new-gravity Ainf, limit damping/excitation, sheets and physical hydrostatic ownership unsupported','Qt worker integration requires separate admission'))
    def diagnose(self):
        try:self.build.verify(require_native=True)
        except (HydroError,OSError) as exc:return (Diagnostic('hams_producer_unavailable','error',str(exc)),)
        return (Diagnostic('hams_unqualified','warning','Configured source/build bound producer; no scientific qualification'),)
