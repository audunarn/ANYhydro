# SPDX-License-Identifier: MPL-2.0
from dataclasses import replace
from decimal import Decimal
import copy,hashlib,json,math,sys
from pathlib import Path
import numpy as np
import pytest
from test_hams_import import case,run,rehash_inputs,rehash_outputs,independent_map,sha
from anyfileio.hams import parse_control,render_control
from anyhydro.hams_import import HamsImportError,LIMIT_PROFILE,PROFILE
from anyhydro.hams_producer import LIMIT_SOURCE_IDENTITY,HamsProducerBuild
from anyhydro.hashing import dataset_content_hash
from anyhydro.persistence import save_hydro_dataset,load_hydro_dataset
from anyhydro.schema import validate_for_time_domain
from anyhydro.errors import CapabilityError

@pytest.fixture
def limits_case(case):
    request=replace(case['request'],include_infinite_frequency=True)
    case['request']=request
    p=case['inputs']/'ControlFile.in';c=parse_control(p.read_text());p.write_text(render_control(replace(c,include_limits=True)))
    rehash_inputs(case)
    p=case['paths']['1'];finite=p.read_text()
    rows=[]
    for w,multiplier in ((0,3),(-1,7)):
        for i in range(12):
            for j in range(12):rows.append(f'{w:.12E} {i+1} {j+1} {case["A"][i,j]*multiplier:.12E}')
    p.write_text('\n'.join(rows)+'\n'+finite);rehash_outputs(case)
    for stem in ('OAMASS','ODAMPING'):
        for local in range(6):
            p=case['native']/f'{stem}{local+1}.txt';finite=p.read_text();rows=[]
            for w,multiplier in ((0,3),(-1,7)):
                for body in range(2):
                    for other in range(2):
                        values=1000*multiplier*case['A'][6*body+local,6*other:6*other+6] if stem=='OAMASS' else np.zeros(6)
                        rows.append(f'{w:7.3f} {w:7.3f} '+''.join(f'{x:14.5E}' for x in values))
            p.write_text('\n'.join(rows)+'\n'+finite)
    for local in range(6):
        p=case['native']/f'OEXFOR{local+1}.txt';finite=p.read_text();rows=[]
        for w in (0,-1):
            for h in range(2):
                for body in range(2):rows.append(f'{w:7.3f} {w:7.3f} '+(f'{0.:14.6E}{0.:14.6E}' if w==0 else f'{"NaN":>14}{"NaN":>14}'))
        p.write_text('\n'.join(rows)+'\n'+finite)
    case['contract']=replace(case['contract'],profile=LIMIT_PROFILE,source_tree_sha256=LIMIT_SOURCE_IDENTITY,
        source_request_hash=request.content_hash(),native_file_sha256={p.name:sha(p) for p in case['native'].iterdir()})
    return case

def rehash_native(case):
    case['contract']=replace(case['contract'],native_file_sha256={p.name:sha(p) for p in case['native'].iterdir()})

def test_ainf_exact_sentinel_cross_body_work_map_and_lossless_a0_save(limits_case,tmp_path):
    c=limits_case;ds=run(c);T=independent_map(c)
    np.testing.assert_allclose(ds.added_mass_infinite,T.T@(7000*c['A'])@T,rtol=2e-11,atol=2e-9)
    np.testing.assert_allclose(ds.added_mass[0],T.T@(1000*c['A'])@T,rtol=2e-11,atol=2e-9)
    assert not np.allclose(ds.added_mass_infinite,T.T@(3000*c['A'])@T)
    assert 'HAMS angular-frequency sentinel -1' in ds.added_mass_infinite.attrs['source']
    evidence=json.loads(ds.attrs['wamit_import'])['native_source'];zero=evidence['zero_frequency_evidence']
    np.testing.assert_allclose(zero['matrix_si'],3000*c['A'])
    assert zero['raw_radiation_text']==c['paths']['1'].read_bytes().decode('ascii') and zero['admitted'] is False
    assert zero['source_body_order']==list(c['contract'].source_body_order)
    assert zero['source_frames']=={key:value.to_json() for key,value in c['contract'].source_frames.items()}
    assert sum(v['nonfinite_scalars'] for v in evidence['unadmitted_limit_auxiliary'].values())==48
    validate_for_time_domain(ds) # schema admission only; unqualified status stays.
    assert ds.attrs['qualification_status']=='unqualified' and 'added_mass_zero' not in ds
    p=tmp_path/'limits.nc';save_hydro_dataset(ds,p);reopened=load_hydro_dataset(p)
    assert dataset_content_hash(ds)==dataset_content_hash(reopened)
    reopened_provenance=reopened.attrs['wamit_import']
    if isinstance(reopened_provenance,str):reopened_provenance=json.loads(reopened_provenance)
    assert reopened_provenance['native_source']==evidence
    altered=ds.copy(deep=True);provenance=json.loads(ds.attrs['wamit_import'])
    provenance['native_source']['zero_frequency_evidence']['matrix_si'][0][0]+=1
    altered.attrs['wamit_import']=json.dumps(provenance,sort_keys=True,allow_nan=False)
    assert dataset_content_hash(altered)!=dataset_content_hash(ds)

@pytest.mark.parametrize('mutation',['missing-zero','missing-inf','duplicate','fifth-field','swap-sentinels','finite-nan'])
def test_limit_serialized_refusals(limits_case,mutation):
    c=limits_case;p=c['paths']['1'];rows=p.read_text().splitlines()
    if mutation=='missing-zero':rows=rows[1:]
    elif mutation=='missing-inf':del rows[144]
    elif mutation=='duplicate':rows.append(rows[0])
    elif mutation=='fifth-field':rows[0]+=' 0.0'
    elif mutation=='swap-sentinels':
        for i in range(288):
            fields=rows[i].split();fields[0]='-1' if i<144 else '0';rows[i]=' '.join(fields)
    else:
        fields=rows[288].split();fields[3]='NaN';rows[288]=' '.join(fields)
    p.write_text('\n'.join(rows)+'\n');rehash_outputs(c)
    with pytest.raises(HamsImportError):run(c)

@pytest.mark.parametrize('which',['missing','bad-zero','bad-inf','finite-nan','bad-limit-literal','coordinate','native-inf-nan','native-zero-nan'])
def test_limit_native_refusals(limits_case,which):
    c=limits_case;p=c['native']/('OEXFOR1.txt' if which in ('finite-nan','bad-limit-literal') else 'OAMASS1.txt')
    rows=p.read_text().splitlines()
    if which=='missing':rows.pop(0)
    elif which in ('bad-zero','bad-inf'):
        i=0 if which=='bad-zero' else 4;rows[i]=rows[i][:16]+f'{99999.:14.5E}'+rows[i][30:]
    elif which=='coordinate':rows[0]=f'{1.:7.3f}'+rows[0][7:]
    elif which in ('native-inf-nan','native-zero-nan'):
        i=4 if which=='native-inf-nan' else 0;rows[i]=rows[i][:16]+f'{"NaN":>14}'+rows[i][30:]
    else:
        i=8 if which=='finite-nan' else 4;token='NaN' if which=='finite-nan' else 'garbage'
        rows[i]=rows[i][:16]+f'{token:>14}'+rows[i][30:]
    p.write_text('\n'.join(rows)+'\n');rehash_native(c)
    with pytest.raises(HamsImportError):run(c)

def test_source_profile_pairing_and_explicit_zero_refuse(limits_case):
    c=limits_case
    with pytest.raises(HamsImportError):replace(c['contract'],profile=PROFILE)
    c['request']=replace(c['request'],include_zero_frequency=True)
    c['contract']=replace(c['contract'],source_request_hash=c['request'].content_hash())
    with pytest.raises(CapabilityError):run(c)

def test_limit_preserving_source_can_import_finite_only(case):
    case['contract']=replace(case['contract'],profile=LIMIT_PROFILE,source_tree_sha256=LIMIT_SOURCE_IDENTITY)
    ds=run(case);assert 'added_mass_infinite' not in ds

def test_new_build_identity_is_unqualified_and_platform_explicit():
    exe=Path(sys.executable);build=HamsProducerBuild(exe,hashlib.sha256(exe.read_bytes()).hexdigest(),LIMIT_SOURCE_IDENTITY,'windows' if sys.platform=='win32' else 'linux','synthetic identity only; never numerical')
    assert build.verify(require_native=True)['qualified'] is False

def test_unknown_negative_coordinate_never_admitted(limits_case):
    c=limits_case;p=c['paths']['1'];rows=p.read_text().splitlines();fields=rows[144].split();fields[0]='-2';rows[144]=' '.join(fields)
    p.write_text('\n'.join(rows)+'\n');rehash_outputs(c)
    with pytest.raises((CapabilityError,HamsImportError)):run(c)
