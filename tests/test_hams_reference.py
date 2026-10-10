"""Signed public reference fixtures; no native solver execution."""
from pathlib import Path
import json
import math
import numpy as np
import pytest

from anyhydro.hams_reference import (
    HamsReferenceError, prepare_hams_first_slice_reference,
    read_hams_radiation_point, verify_hams_executable,
)
from anyhydro import make_first_slice_bodies


def outputs(case):
    m = case.verify()
    w = m['omega_native']
    # Deliberately nonsymmetric, signed cross-body data detects transpose,
    # clipping, isolated-body assembly and incorrect omega/rho scaling.
    a = np.arange(144, dtype=float).reshape(12, 12) / 10 + 1
    b = (np.arange(144, dtype=float).reshape(12, 12) - 80) / 100
    path = case.directory / 'Output' / 'Wamit_format' / 'AmssDamp.1'
    path.parent.mkdir(parents=True,exist_ok=True)
    lines = [f'{w:14.6E}{i+1:6d}{j+1:6d}{a[i,j]:14.6E}{b[i,j]:14.6E}' for i in range(12) for j in range(12)]
    path.write_text('\n'.join(lines)+'\n', encoding='ascii')
    native = case.directory / 'Output' / 'Hams_format'
    native.mkdir(exist_ok=True)
    kn = w*w/m['gravity_native']
    # This fixed deep finite-depth point has tanh(kh) within machine epsilon1.
    k = kn
    for prefix, values, factor in [('OAMASS',a,1000), ('ODAMPING',b,1000*w)]:
        for row in range(6):
            rows=[]
            for ib in range(2):
                for jb in range(2):
                    rows.append(f'{k:7.3f} {w:7.3f} '+''.join(f'{v*factor:14.5E}' for v in values[ib*6+row,jb*6:jb*6+6]))
            (native/f'{prefix}{row+1}.txt').write_text('\n'.join(rows)+'\n',encoding='ascii')
    return path, native, a, b


def test_signed_complete_coupled_and_precision(tmp_path):
    case=prepare_hams_first_slice_reference(tmp_path/'case', subdivisions=2)
    path,native,a,b=outputs(case)
    point=read_hams_radiation_point(case,path,native)
    np.testing.assert_allclose(point.added_mass,1025*a,rtol=1e-15)
    np.testing.assert_allclose(point.radiation_damping,1025*1.5*b,rtol=1e-15,atol=1e-12)
    assert point.added_mass[0,7] != point.added_mass[7,0]
    assert point.radiation_damping[0,7] < 0
    assert len(point.native_output_sha256)==12
    rate=np.arange(12,dtype=float)/12
    power,bound=point.mean_radiated_power(rate)
    assert power==pytest.approx(.5*rate@(1025*1.5*b)@rate)
    assert bound > 0
    with pytest.raises(ValueError): point.radiation_damping[0,0]=1
    with pytest.raises(HamsReferenceError): point.mean_radiated_power(rate.astype(complex))


@pytest.mark.parametrize('failure',['missing','duplicate','out_of_range','frequency','nan','overflow','short','filled_zero','coarse'])
def test_refuses_damaged_radiation(tmp_path,failure):
    case=prepare_hams_first_slice_reference(tmp_path/'case')
    path,native,_,_=outputs(case)
    lines=path.read_text(encoding='ascii').splitlines()
    if failure=='missing':lines.pop()
    elif failure=='duplicate':lines[-1]=lines[0]
    elif failure=='out_of_range':lines[0]=lines[0][:14]+f'{13:6d}'+lines[0][20:]
    elif failure=='frequency':lines[0]=f'{2:14.6E}'+lines[0][14:]
    elif failure=='nan':lines[0]=lines[0][:40]+f'{"NaN":>14}'
    elif failure=='overflow':lines[0]=lines[0][:40]+f'{"1.0E+999":>14}'
    elif failure=='short':lines[0]=lines[0][1:]
    elif failure=='filled_zero':lines[0]=lines[0][:40]+f'{0.:14.6E}'
    elif failure=='coarse':lines[0]=lines[0][:40]+f'{"-0.8E+00":>14}'
    path.write_text('\n'.join(lines)+'\n',encoding='ascii')
    with pytest.raises(HamsReferenceError):read_hams_radiation_point(case,path,native)


@pytest.mark.parametrize('failure',['missing_native','short_native','extra_input','edited_input','edited_manifest','coarse_native'])
def test_refuses_missing_or_changed_evidence(tmp_path,failure):
    case=prepare_hams_first_slice_reference(tmp_path/'case')
    path,native,_,_=outputs(case)
    if failure=='missing_native':(native/'ODAMPING6.txt').unlink()
    elif failure=='short_native':(native/'OAMASS1.txt').write_text('0\n',encoding='ascii')
    elif failure=='extra_input':(case.directory/'Input'/'gen_mod_1_3.txt').write_text('1 1\n')
    elif failure=='edited_input':(case.directory/'Input'/'ControlFile.in').write_text('changed')
    elif failure=='edited_manifest':(case.directory/'case.json').write_text('{}')
    elif failure=='coarse_native':
        p=native/'OAMASS1.txt';rows=p.read_text(encoding='ascii').splitlines()
        rows[0]=rows[0][:16]+f'{"1.0E+03":>14}'+rows[0][30:]
        p.write_text('\n'.join(rows)+'\n',encoding='ascii')
    with pytest.raises(HamsReferenceError):read_hams_radiation_point(case,path,native)


def test_control_columns_similarity_and_signed_mesh(tmp_path):
    case=prepare_hams_first_slice_reference(tmp_path/'case',subdivisions=6)
    assert (case.directory/'Output/Hams_format').is_dir()
    assert (case.directory/'Output/Wamit_format').is_dir()
    m=case.verify()
    assert m['omega_native']**2/m['gravity_native']==pytest.approx(1.5**2/9.81,rel=1e-15)
    control=(case.directory/'Input'/'ControlFile.in').read_text().splitlines()
    assert float(control[2][14:44])==150
    assert int(control[6][25:41])==3
    assert int(control[7][25:42])==3
    assert 'multi' in control[12]
    assert int(control[13][24:40])==2
    assert int(control[28][23:39])==1  # one thread
    bodies=make_first_slice_bodies(subdivisions=6)
    for i,body in enumerate(bodies,1):
        lines=(case.directory/'Input'/f'HullMesh_{i}.pnl').read_text().splitlines()
        panels,nodes,sx,sy=map(int,lines[3].split())
        assert panels==360 and (sx,sy)==(0,0)
        vertices=np.array([list(map(float,line.split()[1:])) for line in lines[6:6+nodes]])
        np.testing.assert_array_equal(vertices+body.rotation_center,body.mesh.vertices)
        faces=np.array([list(map(int,line.split()[2:])) for line in lines[9+nodes:]])-1
        np.testing.assert_array_equal(faces,body.mesh.faces[:,:3])
        p=vertices[faces]
        native_n=np.cross(p[:,0]-p[:,1],p[:,2]-p[:,1])
        # dot with centroid - solid center is negative (fluid-domain outward).
        assert np.all(np.einsum('ij,ij->i',native_n,p.mean(axis=1)-[0,0,-.25])<0)
        native_rot=np.cross(p.mean(axis=1),native_n)
        np.testing.assert_allclose(native_rot[:,0],p.mean(axis=1)[:,1]*native_n[:,2]-p.mean(axis=1)[:,2]*native_n[:,1])
        assert m['hull_mesh_hashes'][i-1]==body.mesh.mesh_hash


def test_no_ir_and_no_overwrite_or_unpinned_executable(tmp_path):
    case=prepare_hams_first_slice_reference(tmp_path/'case',irregular_frequency_removal=False)
    assert case.verify()['numerical_waterplane']=='none'
    assert not list((case.directory/'Input').glob('WaterPlaneMesh*'))
    with pytest.raises(FileExistsError):prepare_hams_first_slice_reference(case.directory)
    binary=tmp_path/'fake.exe';binary.write_bytes(b'not official executable')
    with pytest.raises(HamsReferenceError):verify_hams_executable(binary)


@pytest.mark.parametrize('omega',[True,float('nan'),0.,8.1])
def test_refuses_unsupported_frequency(tmp_path,omega):
    with pytest.raises(HamsReferenceError):prepare_hams_first_slice_reference(tmp_path/'case',omega=omega)
    assert not (tmp_path/'case').exists()
