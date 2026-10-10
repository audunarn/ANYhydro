# SPDX-License-Identifier: MPL-2.0
"""Explicit corrected-HAMS omega import; source transport is unqualified.

No producer is run and no coefficient is repaired. Input controls, complete
WORLD geometry, executable and raw output bytes are bound before transport.
"""
from dataclasses import dataclass
from decimal import Decimal
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import numpy as np

from .errors import CapabilityError
from .frames import CoefficientFrame
from .hams_inputs import HamsDeckError, _require_native_regime
from .hams_reference import HAMS_SOURCE_COMMIT, HAMS_NATIVE_DENSITY, HAMS_NATIVE_GRAVITY
from .hams_reference import _number, HamsReferenceError
from .records import HydroBody, HydroRequest
from .wamit_import import _CoefficientImportContract, _digest, _import_coefficients, _map_coordinates, _syntax_api, WamitImportError

__all__ = ['HamsImportContract', 'HamsImportError', 'import_hams_coefficients']
PROFILE = 'hams_mrel_cf9583c_phase_corrected_omega_v1'
from .hams_producer import SOURCE_IDENTITIES, ALL_SOURCE_IDENTITIES, LIMIT_SOURCE_IDENTITY, GRAVITY_SOURCE_IDENTITY
from .hams_field_points import normalize_field_points,admit_field_points,HamsFieldPointError
LIMIT_PROFILE = 'hams_mrel_cf9583c_phase_limit_corrected_omega_v2'
GRAVITY_PROFILE = 'hams_mrel_cf9583c_phase_limit_gravity_corrected_finite_v3'
HamsImportError = WamitImportError


def _sha(data):
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class HamsImportContract(_CoefficientImportContract):
    """Immutable exact source declarations, not independent producer truth.

    source_tree_sha256 identifies the corrected compiled source inventory;
    executable_sha256 binds the caller's separately recorded build. Neither
    digest asserts source accuracy or authorizes running that executable.
    """
    profile: str = PROFILE
    input_file_sha256: dict = None
    source_tree_sha256: str = ''
    source_commit: str = HAMS_SOURCE_COMMIT
    native_file_sha256: dict = None
    source_field_points: tuple = ()
    field_file_sha256: dict = None
    _SUPPORTED_PROFILES: ClassVar[tuple] = (PROFILE, LIMIT_PROFILE, GRAVITY_PROFILE)

    def __post_init__(self):
        super().__post_init__()
        if self.source_commit != HAMS_SOURCE_COMMIT or self.source_tree_sha256 not in ALL_SOURCE_IDENTITIES:
            raise HamsImportError('Unknown or uncorrected compiled producer source identity')
        expected_profile=(GRAVITY_PROFILE if self.source_tree_sha256==GRAVITY_SOURCE_IDENTITY else LIMIT_PROFILE if self.source_tree_sha256==LIMIT_SOURCE_IDENTITY else PROFILE)
        if self.profile!=expected_profile:
            raise HamsImportError('Import profile differs from exact compiled source identity')
        try:points=normalize_field_points(self.source_field_points)
        except HamsFieldPointError as exc:raise HamsImportError(str(exc)) from exc
        if points and self.profile!=GRAVITY_PROFILE:raise HamsImportError('Fields require exact gravity-consistent finite source profile')
        hashes=self.field_file_sha256 if self.field_file_sha256 is not None else {}
        if not isinstance(hashes,Mapping) or set(hashes)!=({'combined','incident'} if points else set()):raise HamsImportError('Both explicit field hashes required exactly when points requested')
        object.__setattr__(self,'source_field_points',points)
        object.__setattr__(self,'field_file_sha256',MappingProxyType({k:_digest(v,'native field') for k,v in hashes.items()}))
        if (self.source_rho,self.source_gravity)!=(HAMS_NATIVE_DENSITY,HAMS_NATIVE_GRAVITY):
            raise HamsImportError('Pinned native rho1000/g9.80665 required; no similarity conversion')
        if math.isfinite(self.source_water_depth) and self.source_water_depth>500:
            raise HamsImportError('Finite depth >500 uses inconsistent native incident branches')
        if any(not np.array_equal(frame.axes,np.eye(3)) for frame in self.source_frames.values()):
            raise HamsImportError('Pinned source coefficient axes must be WORLD')
        if not isinstance(self.input_file_sha256,Mapping) or not self.input_file_sha256:
            raise HamsImportError('Complete explicit input file hashes required')
        values={}
        for name,digest in self.input_file_sha256.items():
            if not isinstance(name,str) or Path(name).name!=name or '/' in name or '\\' in name:
                raise HamsImportError('Input file names must be plain basenames')
            values[name]=_digest(digest,'input file')
        aggregate=_sha(json.dumps(values,sort_keys=True,separators=(',',':')).encode('ascii'))
        if aggregate!=self.input_sha256:
            raise HamsImportError('Aggregate input identity differs from declared inventory')
        object.__setattr__(self,'input_file_sha256',MappingProxyType(values))
        names={f'{stem}{i}.txt' for stem in ('OAMASS','ODAMPING','OEXFOR') for i in range(1,7)}
        if not isinstance(self.native_file_sha256,Mapping) or set(self.native_file_sha256)!=names:
            raise HamsImportError('All18 native radiation/excitation file identities required')
        object.__setattr__(self,'native_file_sha256',MappingProxyType({k:_digest(v,'native output') for k,v in self.native_file_sha256.items()}))


def _verify_inputs(bodies,request,contract,directory,executable):
    try:
        from anyfileio.hams import parse_control,parse_mesh,parse_mechanical,HamsInputError
    except ImportError as exc:
        raise CapabilityError('HAMS import requires optional ANYfileIO syntax package') from exc
    path=Path(directory)
    try:
        if _sha(Path(executable).read_bytes())!=contract.executable_sha256:
            raise HamsImportError('Producer executable identity changed')
        if {p.name for p in path.iterdir()}!=set(contract.input_file_sha256):
            raise HamsImportError('Input inventory changed; unknown mode/override refused')
        captured={}
        for name,digest in contract.input_file_sha256.items():
            raw=(path/name).read_bytes()
            if _sha(raw)!=digest:raise HamsImportError('Input file identity changed: '+name)
            captured[name]=raw.decode('ascii')
        control=parse_control(captured['ControlFile.in'])
        order=contract.source_body_order;by_id={b.body_id:b for b in bodies}
        if len(by_id)!=len(bodies) or set(by_id)!=set(order):raise HamsImportError('Source body inventory mismatch')
        if any(not np.array_equal(frame.axes,np.eye(3)) for frame in contract.source_frames.values()):
            raise HamsImportError('Source WORLD frame changed after contract construction')
        if request.include_zero_frequency:
            raise CapabilityError('Requested canonical zero-frequency operator unsupported; incidental A0 retained only as source evidence')
        if contract.profile!=LIMIT_PROFILE and (control.include_limits or request.include_infinite_frequency):
            raise CapabilityError('Finite-only source profile does not admit limit transport')
        if control.include_limits!=request.include_infinite_frequency:
            raise HamsImportError('Native limit acquisition differs from authoritative request')
        depth=-1. if math.isinf(request.water_depth) else request.water_depth
        if (control.water_depth!=depth or control.frequencies!=tuple(request.omegas)
            or control.headings_degrees!=tuple(math.degrees(x) for x in request.wave_directions)
            or control.reference_centers!=tuple(contract.source_frames[key].reference_center for key in order)
            or control.body_origins_xyzyaw!=((0.,0.,0.,0.),)*len(order)
            or control.reference_length!=contract.reference_length_m or control.field_points!=contract.source_field_points):
            raise HamsImportError('Input controls differ from declared WORLD/omega/origin/request profile')
        try:admit_field_points(bodies,request,contract.source_field_points)
        except HamsFieldPointError as exc:raise HamsImportError(str(exc)) from exc
        if request.backend_settings:raise HamsImportError('Unmapped request backend settings refused')
        expected={'ControlFile.in'};restoring=[]
        for index,key in enumerate(order,1):
            body=by_id[key];suffix=f'_{index}' if len(order)>1 else ''
            if not body.mesh.outward_normals_verified or body.attached_surfaces:
                raise HamsImportError('Verified solid-outward hull winding required; sheets unsupported')
            if body.mesh.surface is not None and body.mesh.surface.primary_hydrodynamics!='potential_flow':
                raise HamsImportError('Hull primary ownership must be potential_flow')
            if bool(body.lid_mesh is not None)!=control.irregular_frequency_removal:
                raise HamsImportError('Exact explicit lid policy must match each body')
            for prefix,mesh in (('HullMesh',body.mesh),('WaterPlaneMesh',body.lid_mesh)):
                if mesh is None:continue
                name=prefix+suffix+'.pnl';expected.add(name);record=parse_mesh(captured[name])
                faces=tuple(tuple(int(x) for x in row[:size]) for row,size in zip(mesh.faces,mesh.face_sizes))
                if not np.array_equal(record.vertices,mesh.vertices) or record.faces!=faces:
                    raise HamsImportError('Actual source hull/lid geometry differs from caller')
                if prefix=='WaterPlaneMesh' and np.any(mesh.vertices[:,2]!=0):
                    raise HamsImportError('Native lid must be exactly WORLD z=0')
            name='Hydrostatic'+suffix+'.in';expected.add(name)
            mechanical=parse_mechanical(captured[name])
            restoring.append(mechanical.hydrostatic_restoring)
            if mechanical.center_of_gravity!=body.center_of_mass:
                raise HamsImportError('Source center of gravity differs from caller')
        if expected!=set(captured):raise HamsImportError('Unsupported or incomplete native input inventory')
        provenance=dict(input_file_sha256=dict(contract.input_file_sha256),source_tree_sha256=contract.source_tree_sha256,
            source_commit=contract.source_commit,native_controls='type3 omega; ISOL1; no symmetry/generalized; XW0; AMP1; WORLD/zero LCS',
            supplied_hydrostatics='caller mechanical input; UNKNOWN force ownership; no independent evidence')
        if contract.source_field_points:
            provenance.update(source_field_points=contract.source_field_points,field_file_sha256=dict(contract.field_file_sha256),field_transport='separate source-bound canonical artifact; dynamic pressure, not panel loads')
        return provenance,restoring
    except (OSError,UnicodeError,KeyError,HamsInputError) as exc:
        raise HamsImportError('Missing, malformed or unbound native inputs: '+str(exc)) from exc


def _wave_number(omega,depth):
    deep=omega*omega/HAMS_NATIVE_GRAVITY
    if math.isinf(depth):return deep
    lo=deep;hi=max(1.,2*deep)
    while hi*math.tanh(hi*depth)<deep:hi*=2
    for _ in range(80):
        mid=(lo+hi)/2
        if mid*math.tanh(mid*depth)<deep:lo=mid
        else:hi=mid
    return (lo+hi)/2


def _verify_native_outputs(request,contract,directory,restoring,paths):
    """Validate complete finite and source-specific limit acquisition before transport."""
    parse,inventory,excitation_check,validate,FormatError=_syntax_api()
    try:
        raw_docs={kind:Path(p).read_bytes() for kind,p in paths.items()}
        if any(_sha(raw)!=contract.file_sha256[kind] for kind,raw in raw_docs.items()):
            raise HamsImportError('Serialized coefficient identity changed')
        docs={kind:parse(raw.decode('ascii'),kind=kind) for kind,raw in raw_docs.items()}
        maps={kind:_map_coordinates(docs[kind],request.omegas,allowed_limits=((Decimal(0),Decimal(-1)) if request.include_infinite_frequency and kind=='1' else (Decimal(0),))) for kind in ('1','3')}
        headings=_map_coordinates(docs['3'],[math.degrees(x) for x in request.wave_directions],heading=True)
        n=len(contract.source_body_order);size=6*n
        coordinates=list(maps['1'])
        if request.include_infinite_frequency:coordinates += [Decimal(0),Decimal(-1)]
        validate(docs['1'],inventory(docs['1'],expected_keys=[(w,i,j) for w in coordinates for i in range(1,size+1) for j in range(1,size+1)]))
        validate(docs['3'],inventory(docs['3'],expected_keys=[(w,h,i) for w in maps['3'] for h in headings for i in range(1,size+1)]))
        validate(docs['3'],excitation_check(docs['3']))
        validate(docs['hst'],inventory(docs['hst'],expected_keys=[(i,j) for i in range(1,size+1) for j in range(1,size+1)]))
        a={};b={};f={}
        # Keys carry native coordinate, not finite index or WAMIT period semantics.
        for row in docs['1'].records:
            key=(row.frequency.decimal,row.influenced-1,row.radiating-1)
            if row.frequency.decimal<=0:
                if row.damping is not None:raise HamsImportError('Four-field native limits required; no computed damping')
            elif row.damping is None:raise HamsImportError('Missing finite-frequency damping')
            a[key]=row.added_mass
            if row.damping is not None:b[key]=row.damping
        for row in docs['3'].records:
            f[(maps['3'][row.frequency.decimal],headings[row.heading.decimal],row.influenced-1)]=(row.real,row.imaginary)
        rho,g=HAMS_NATIVE_DENSITY,HAMS_NATIVE_GRAVITY
        for row in docs['hst'].records:
            i,j=row.influenced-1,row.radiating-1
            expected=restoring[i//6][i%6][j%6]/(rho*g) if i//6==j//6 else 0.
            if abs(row.coefficient.value-expected)>row.coefficient.rounding_bound+16*math.ulp(expected):
                raise HamsImportError('Supplied restoring input and .hst output contradict each other')
        path=Path(directory);captured={}
        if {p.name for p in path.iterdir()}!=set(contract.native_file_sha256):
            raise HamsImportError('Complete exact native output inventory required')
        for name,digest in contract.native_file_sha256.items():
            raw=(path/name).read_bytes()
            if _sha(raw)!=digest:raise HamsImportError('Native acquisition file identity changed')
            captured[name]=raw.decode('ascii').splitlines()
        branches=([(Decimal(0),0.,0.),(Decimal(-1),-1.,-1.)] if request.include_infinite_frequency else [])
        branches += [(coord,float(request.omegas[index]),_wave_number(float(request.omegas[index]),request.water_depth)) for coord,index in sorted(maps['1'].items(),key=lambda item:item[1])]
        def row_coordinates(line,w,expected_k):
            if line[7:8]!=' ' or line[15:16]!=' ':raise HamsImportError('Native fixed row layout differs')
            k,omega=float(line[:7]),float(line[8:15])
            if (not math.isfinite(k) or not math.isfinite(omega)
                or abs(k-expected_k)>.0005+8*math.ulp(expected_k)
                or abs(omega-w)>.0005+8*math.ulp(w)):
                raise HamsImportError('Native acquisition coordinates disagree with source request')
        excluded={}
        def unadmitted_values(name,line,digits):
            # Only the native nonfinite spellings in unrequested auxiliary limit
            # rows are permitted. Never feed these into a canonical coefficient.
            nonfinite=0
            for offset in range(16,len(line),14):
                token=line[offset:offset+14].strip()
                if token in ('NaN','Infinity','-Infinity','Inf','-Inf'):nonfinite+=1
                else:_number(token,digits)
            value=excluded.setdefault(name,dict(rows=0,nonfinite_scalars=0,admitted=False))
            value['rows']+=1;value['nonfinite_scalars']+=nonfinite
        for stem,table in (('OAMASS',a),('ODAMPING',b)):
            for local_row in range(6):
                name=f'{stem}{local_row+1}.txt';rows=captured[name]
                if len(rows)!=len(branches)*n*n:raise HamsImportError('Missing native body-pair radiation acquisition')
                for branch,(coord,w,wave) in enumerate(branches):
                    factor=rho if stem=='OAMASS' else rho*w
                    for pair in range(n*n):
                        line=rows[branch*n*n+pair]
                        if len(line)!=100:raise HamsImportError('Native radiation format differs')
                        row_coordinates(line,w,wave);i=(pair//n)*6+local_row
                        if stem=='ODAMPING' and coord<=0:
                            unadmitted_values(name,line,5);continue
                        for col in range(6):
                            value,error=_number(line[16+14*col:30+14*col].strip(),5)
                            number=table[(coord,i,(pair%n)*6+col)];target=number.value*factor
                            if not math.isfinite(target) or abs(value-target)>error+number.rounding_bound*abs(factor)+16*math.ulp(target):
                                raise HamsImportError('Native/reordered radiation values disagree')
        finite_indices={coord:index for coord,index in maps['1'].items()}
        for local in range(6):
            name=f'OEXFOR{local+1}.txt';rows=captured[name]
            if len(rows)!=len(branches)*len(request.wave_directions)*n:raise HamsImportError('Missing native excitation acquisition')
            for branch,(coord,w,wave) in enumerate(branches):
                for h in range(len(request.wave_directions)):
                    for body in range(n):
                        line=rows[(branch*len(request.wave_directions)+h)*n+body]
                        if len(line)!=44:raise HamsImportError('Native excitation format differs')
                        row_coordinates(line,w,wave)
                        if coord<=0:unadmitted_values(name,line,6);continue
                        real,re=_number(line[16:30].strip());imag,im=_number(line[30:44].strip())
                        zr,zi=f[(finite_indices[coord],h,body*6+local)]
                        for actual,error,target,bound in ((-imag,im,rho*g*zr.value,rho*g*zr.rounding_bound),(real,re,-rho*g*zi.value,rho*g*zi.rounding_bound)):
                            if not math.isfinite(target) or abs(actual-target)>error+bound+16*math.ulp(target):
                                raise HamsImportError('Native excitation and signed .3 output disagree')
        evidence=dict(native_file_sha256=dict(contract.native_file_sha256),native_consistency='complete acquisition and printing-bound agreement; .hst matches supplied CRS/self blocks and zero cross blocks')
        if request.include_infinite_frequency:
            zero=[[rho*a[(Decimal(0),i,j)].value for j in range(size)] for i in range(size)]
            if not np.isfinite(zero).all():raise HamsImportError('Unrepresentable incidental A0 evidence')
            evidence.update(native_frequency_limits={'zero':0,'infinite':-1,'coordinate':'angular_frequency_rad_s'},
                zero_frequency_evidence=dict(admitted=False,representation='source-frame added mass, not a canonical operator',matrix_si=zero,
                    rho=rho,source_body_order=list(contract.source_body_order),source_frames={key:value.to_json() for key,value in contract.source_frames.items()},
                    raw_radiation_text=raw_docs['1'].decode('ascii'),radiation_sha256=contract.file_sha256['1'],source_tree_sha256=contract.source_tree_sha256),
                unadmitted_limit_auxiliary=excluded,limit_damping='absent from serialized coefficients; raw native values unadmitted',limit_excitation='unrequested native values unadmitted; finite .3 only')
        return evidence
    except FormatError as exc:
        raise HamsImportError(str(exc),diagnostics=exc.diagnostics) from exc
    except (OSError,UnicodeError,ValueError,HamsReferenceError) as exc:
        raise HamsImportError('Native acquisition validation failed: '+str(exc)) from exc


def import_hams_coefficients(bodies,request,contract,*,input_directory,executable_path,native_directory,
                             radiation_path,excitation_path,hydrostatic_path):
    """Import complete corrected native coefficients without running the producer."""
    if type(contract) is not HamsImportContract:raise HamsImportError('Explicit corrected HAMS import contract required')
    bodies=tuple(bodies)
    if type(request) is not HydroRequest or not bodies or any(type(b) is not HydroBody for b in bodies):
        raise HamsImportError('Authoritative HydroBody/HydroRequest records required')
    try:_require_native_regime(request)
    except HamsDeckError as exc:raise HamsImportError(str(exc)) from exc
    native,restoring=_verify_inputs(bodies,request,contract,input_directory,executable_path)
    native.update(_verify_native_outputs(request,contract,native_directory,restoring,
        {'1':radiation_path,'3':excitation_path,'hst':hydrostatic_path}))
    return _import_coefficients(bodies,request,contract,radiation_path=radiation_path,
        excitation_path=excitation_path,hydrostatic_path=hydrostatic_path,hams=True,
        source_provenance=native)
