# SPDX-License-Identifier: MPL-2.0
from dataclasses import replace
import hashlib,json,math,sys
import numpy as np
import pytest
from anyfileio.hams import HamsControl,HamsPanelMesh,HamsMechanicalInput,render_control,render_mesh,render_mechanical,parse_control,parse_mesh
from anyhydro.fixtures import make_fixture_bodies
from anyhydro.frames import CoefficientFrame
from anyhydro.records import HydroRequest
from anyhydro.errors import CapabilityError
from anyhydro.hams_import import HamsImportContract,HamsImportError,import_hams_coefficients,SOURCE_IDENTITIES
from anyhydro.wamit_import import import_wamit7
from anyhydro.hashing import dataset_content_hash
from anyhydro.persistence import save_hydro_dataset,load_hydro_dataset
from anyhydro.excitation_source import verify_excitation_source
from anyhydro.hydrostatic_contract import require_hydrostatic_composition,HydrostaticContractError
from anyhydro.schema import validate_for_time_domain

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
aggregate=lambda values:hashlib.sha256(json.dumps(values,sort_keys=True,separators=(',',':')).encode()).hexdigest()

@pytest.fixture
def case(tmp_path):
    original=make_fixture_bodies();bodies=tuple(reversed(original))
    request=HydroRequest([.5,1.25],[0.,math.pi/2],50.,1000.,9.80665,False,False,{})
    frames={b.body_id:CoefficientFrame(np.eye(3),tuple(np.asarray(b.rotation_center)+[.4,-.7,.2])) for b in original}
    inputs=tmp_path/'Input';inputs.mkdir();exe=tmp_path/'producer';exe.write_bytes(b'unqualified build identity fixture; never run')
    control=HamsControl(50.,tuple(request.omegas),(0.,90.),((0.,0.,0.,0.),)*2,tuple(frames[b.body_id].reference_center for b in original),reference_length=2.5)
    (inputs/'ControlFile.in').write_text(render_control(control))
    eye=tuple(tuple(float(i==j) for j in range(6)) for i in range(6));zero=tuple((0.,)*6 for _ in range(6))
    CRS=[]
    for index,body in enumerate(original,1):
        crs=tuple(tuple(float((index-1)*36+i*6+j+1) for j in range(6)) for i in range(6));CRS.append(crs)
        mesh=body.mesh;faces=tuple(tuple(int(x) for x in row[:n]) for row,n in zip(mesh.faces,mesh.face_sizes))
        (inputs/f'HullMesh_{index}.pnl').write_text(render_mesh(HamsPanelMesh(mesh.vertices.tolist(),faces)))
        (inputs/f'Hydrostatic_{index}.in').write_text(render_mechanical(HamsMechanicalInput(body.center_of_mass,eye,zero,zero,crs,zero)))
    A=np.arange(1,145).reshape(12,12)/100;B=np.arange(144,0,-1).reshape(12,12)/70;C=np.zeros((12,12));C[:6,:6]=np.array(CRS[0])/9806.65;C[6:,6:]=np.array(CRS[1])/9806.65
    F=(np.arange(1,13)+1j*np.arange(12,0,-1))/11
    paths={key:tmp_path/('coeff.'+key) for key in ('1','3','hst')}
    paths['1'].write_text('\n'.join(f'{w:.12E} {i+1} {j+1} {A[i,j]*(k+1):.12E} {B[i,j]*(k+1):.12E}' for k,w in enumerate(request.omegas) for i in range(12) for j in range(12))+'\n')
    rows=[]
    for k,w in enumerate(request.omegas):
        for h,theta in enumerate(request.wave_directions):
            for i in range(12):
                z=F[i]*(k+1)*(h+1);rows.append(f'{w:.12E} {math.degrees(theta):.12E} {i+1} {abs(z):.12E} {math.degrees(math.atan2(z.imag,z.real)):.12E} {z.real:.12E} {z.imag:.12E}')
    paths['3'].write_text('\n'.join(rows)+'\n')
    paths['hst'].write_text('\n'.join(f'{i+1} {j+1} {C[i,j]:.12E}' for i in range(12) for j in range(12))+'\n')
    native=tmp_path/'native';native.mkdir()
    from scipy.optimize import brentq
    for stem,matrix in (('OAMASS',A),('ODAMPING',B)):
        for local in range(6):
            rows=[]
            for k,w in enumerate(request.omegas):
                kn=w*w/9.80665;wave=brentq(lambda x:x*math.tanh(50*x)-kn,kn,2.)
                for body in range(2):
                    for other in range(2):
                        values=matrix[6*body+local,6*other:6*other+6]*(k+1)*1000*(w if stem=='ODAMPING' else 1)
                        rows.append(f'{wave:7.3f} {w:7.3f} '+''.join(f'{x:14.5E}' for x in values))
            (native/f'{stem}{local+1}.txt').write_text('\n'.join(rows)+'\n')
    for local in range(6):
        rows=[]
        for k,w in enumerate(request.omegas):
            kn=w*w/9.80665;wave=brentq(lambda x:x*math.tanh(50*x)-kn,kn,2.)
            for h in range(2):
                for body in range(2):
                    z=F[body*6+local]*(k+1)*(h+1);exfc=-1j*9806.65*z.conjugate()
                    rows.append(f'{wave:7.3f} {w:7.3f} {exfc.real:14.6E}{exfc.imag:14.6E}')
        (native/f'OEXFOR{local+1}.txt').write_text('\n'.join(rows)+'\n')
    input_hashes={p.name:sha(p) for p in inputs.iterdir()}
    contract=HamsImportContract(tuple(b.body_id for b in original),frames,{b.body_id:b.mesh.mesh_hash for b in original},
        {b.body_id:None for b in original},{b.body_id:(float(i),0.,-1.) for i,b in enumerate(original)},2.5,1000.,9.80665,50.,
        request.content_hash(),aggregate(input_hashes),sha(exe),{k:sha(p) for k,p in paths.items()},
        '2.2-development','source-derived synthetic oracle','pinned source psi=phi/i; inward-solid native normals; no physical qualification',
        input_file_sha256=input_hashes,source_tree_sha256=next(iter(SOURCE_IDENTITIES)),native_file_sha256={p.name:sha(p) for p in native.iterdir()})
    return dict(original=original,bodies=bodies,request=request,contract=contract,inputs=inputs,exe=exe,native=native,paths=paths,A=A,B=B,C=C,F=F)

def run(case):
    return import_hams_coefficients(case['bodies'],case['request'],case['contract'],input_directory=case['inputs'],executable_path=case['exe'],native_directory=case['native'],
        radiation_path=case['paths']['1'],excitation_path=case['paths']['3'],hydrostatic_path=case['paths']['hst'])

def rehash_inputs(case):
    hashes={p.name:sha(p) for p in case['inputs'].iterdir()}
    case['contract']=replace(case['contract'],input_file_sha256=hashes,input_sha256=aggregate(hashes))

def rehash_outputs(case):
    case['contract']=replace(case['contract'],file_sha256={k:sha(p) for k,p in case['paths'].items()})

def independent_map(case):
    T=np.zeros((12,12))
    for s,body in enumerate(case['original']):
        t=case['bodies'].index(body);x,y,z=np.asarray(case['contract'].source_frames[body.body_id].reference_center)-body.rotation_center
        skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
        T[s*6:s*6+3,t*6:t*6+3]=np.eye(3);T[s*6:s*6+3,t*6+3:t*6+6]=-skew;T[s*6+3:s*6+6,t*6+3:t*6+6]=np.eye(3)
    return T

def test_native_units_conjugation_cross_body_order_and_work_map(case,tmp_path):
    ds=run(case);T=independent_map(case)
    for k,w in enumerate(case['request'].omegas):
        np.testing.assert_allclose(ds.added_mass[k],T.T@(1000*case['A']*(k+1))@T,rtol=2e-11,atol=2e-9)
        np.testing.assert_allclose(ds.radiation_damping[k],T.T@(1000*w*case['B']*(k+1))@T,rtol=2e-11,atol=2e-9)
        for h in range(2):np.testing.assert_allclose(ds.excitation_force[k,h],T.T@(9806.65*case['F'].conj()*(k+1)*(h+1)),rtol=2e-11,atol=2e-8)
    np.testing.assert_allclose(ds.hydrostatic_stiffness,T.T@(9806.65*case['C'])@T,rtol=2e-11,atol=2e-8)
    assert ds.attrs['backend']=='HAMS-import' and ds.attrs['qualification_status']=='unqualified'
    p=tmp_path/'imported.nc';save_hydro_dataset(ds,p);reopened=load_hydro_dataset(p)
    assert dataset_content_hash(ds)==dataset_content_hash(reopened);verify_excitation_source(reopened,case['request'],case['bodies'])
    with pytest.raises(CapabilityError):validate_for_time_domain(ds)
    with pytest.raises(HydrostaticContractError):require_hydrostatic_composition(ds)
    with pytest.raises(HamsImportError):import_wamit7(case['bodies'],case['request'],case['contract'],radiation_path=case['paths']['1'],excitation_path=case['paths']['3'],hydrostatic_path=case['paths']['hst'])

def test_independent_incident_pressure_sign_and_native_writer_transform():
    # Crest pressure rho*g*a on underside: outward solid normal=-z. Integrate
    # physical pressure directly, independently of coefficient-import helpers.
    a=.3;g=9.80665;rho=1000.;omega=.7;area=2.;normal_inward=np.array([0.,0.,1.])
    psi=-a*g/omega;native=1j*rho*omega*psi*normal_inward*area
    z3=-1j*native.conj()/(rho*g*a)
    physical=rho*g*a*normal_inward*area
    np.testing.assert_allclose(rho*g*z3.conj(),physical/a)
    np.testing.assert_allclose(1j*native,physical)
    assert physical[2]>0 and native[2].real==0

@pytest.mark.parametrize('kind',['1','3','hst'])
def test_incomplete_or_extra_output_inventory_refuses(case,kind):
    p=case['paths'][kind];rows=p.read_text().splitlines();p.write_text('\n'.join(rows[1:])+'\n');rehash_outputs(case)
    with pytest.raises(HamsImportError):run(case)

def test_bad_phase_never_repaired(case):
    p=case['paths']['3'];rows=p.read_text().splitlines();f=rows[0].split();f[4]='999';rows[0]=' '.join(f);p.write_text('\n'.join(rows)+'\n');rehash_outputs(case)
    with pytest.raises(HamsImportError) as e:run(case)
    assert any(d.code=='WAMIT_PHASE' for d in e.value.diagnostics)

@pytest.mark.parametrize('which',['extra','mesh','control','missing','exe'])
def test_actual_input_and_producer_binding_refuse_even_redeclared_hashes(case,which):
    if which=='extra':(case['inputs']/'GenMode_1_7.dat').write_text('unknown')
    elif which=='missing':(case['inputs']/'Hydrostatic_1.in').unlink()
    elif which=='exe':case['exe'].write_bytes(b'changed')
    elif which=='control':
        p=case['inputs']/'ControlFile.in';c=parse_control(p.read_text());p.write_text(render_control(replace(c,reference_length=3.)))
    else:
        p=case['inputs']/'HullMesh_1.pnl';mesh=parse_mesh(p.read_text());vertices=list(mesh.vertices);vertices[0]=(vertices[0][0]+.1,*vertices[0][1:]);p.write_text(render_mesh(replace(mesh,vertices=vertices)))
    rehash_inputs(case)
    with pytest.raises(HamsImportError):run(case)

@pytest.mark.parametrize('field,value',[('source_tree_sha256','a'*64),('source_commit','unknown'),('source_rho',1025.),('source_water_depth',501.),('input_sha256','a'*64)])
def test_unadmitted_source_contract_refuses(case,field,value):
    with pytest.raises(HamsImportError):replace(case['contract'],**{field:value})

@pytest.mark.parametrize('omega',[1.e-6,1.e8])
def test_finite_native_limit_branches_refuse(case,omega):
    case['request']=replace(case['request'],omegas=[omega])
    with pytest.raises(HamsImportError):run(case)

def test_limits_fields_optional_syntax_and_missing_damping_refuse(case,monkeypatch):
    p=case['inputs']/'ControlFile.in';c=parse_control(p.read_text());p.write_text(render_control(replace(c,include_limits=True)));rehash_inputs(case)
    with pytest.raises(CapabilityError):run(case)
    p.write_text(render_control(replace(c,field_points=((0.,0.,0.),))));rehash_inputs(case)
    with pytest.raises(HamsImportError):run(case)
    p.write_text(render_control(c));rehash_inputs(case)
    monkeypatch.setitem(sys.modules,'anyfileio.hams',None)
    with pytest.raises(CapabilityError):run(case)
    monkeypatch.undo()
    p=case['paths']['1'];rows=p.read_text().splitlines();rows[0]=' '.join(rows[0].split()[:-1]);p.write_text('\n'.join(rows)+'\n');rehash_outputs(case)
    with pytest.raises(HamsImportError):run(case)


@pytest.mark.parametrize('which',['truncated','radiation-value','excitation-value','native-hash','native-extra','restoring-self','restoring-cross'])
def test_native_completeness_and_input_output_contradictions_refuse(case,which):
    if which.startswith('restoring'):
        p=case['paths']['hst'];rows=p.read_text().splitlines();index=0 if which=='restoring-self' else 6
        fields=rows[index].split();fields[-1]='1.00000000E+00';rows[index]=' '.join(fields);p.write_text('\n'.join(rows)+'\n');rehash_outputs(case)
    else:
        p=case['native']/('OEXFOR1.txt' if which=='excitation-value' else 'OAMASS1.txt')
        if which=='native-extra':(case['native']/'unexpected.txt').write_text('unknown')
        elif which=='truncated':p.write_text('\n'.join(p.read_text().splitlines()[1:])+'\n')
        else:
            rows=p.read_text().splitlines();rows[0]=rows[0][:16]+f'{1.e8:14.6E}' +rows[0][30:];p.write_text('\n'.join(rows)+'\n')
        if which not in ('native-hash','native-extra'):
            case['contract']=replace(case['contract'],native_file_sha256={p.name:sha(p) for p in case['native'].iterdir()})
    with pytest.raises(HamsImportError):run(case)
