"""Neutral, unqualified fixed-body free-surface preview records (SI/WORLD).

No Capytaine, Qt, ANYsolver or application imports at module import. These
fields never admit hydrodynamic coefficients or supply mechanical forces.
"""
import hashlib
import json
import math
import numpy as np

API_VERSION = 'wave_field_preview_v1'
SOLVER_SETTINGS = dict(green_function='Delhommeau',tabulation_nr=0,tabulation_nz=0,
                       tabulation_nb_integration_points=1001,tabulation_cache_dir=None,
                       finite_depth_method='newer',finite_depth_prony_decomposition_method='python',
                       floating_point_precision='float64',gf_singularities='low_freq',n_threads=1)

def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_request(data):
    names={'bodies','omega','heading_degrees','amplitude','phase','water_depth','bounds','nx','ny','project_input_hash'}
    if type(data) is not dict or set(data)!=names:
        raise ValueError('wave preview request fields must match the named schema')
    value=json.loads(json.dumps(data,allow_nan=False))
    for name in ('omega','heading_degrees','amplitude','phase','water_depth'):
        number=value[name]
        if type(number) not in (int,float) or not math.isfinite(number):
            raise ValueError(f'{name} must be finite')
        value[name]=float(number)
    if value['omega']<=0 or value['water_depth']<=0 or value['amplitude']<=0:
        raise ValueError('frequency, amplitude and depth must be positive')
    if any(type(value[n]) is not int or value[n]<3 for n in ('nx','ny')) or value['nx']*value['ny']>1024:
        raise ValueError('wave preview grid must have 3+ nodes per axis and at most1024 points')
    if (not isinstance(value['project_input_hash'],str) or len(value['project_input_hash'])!=64
            or any(c not in '0123456789abcdef' for c in value['project_input_hash'])):
        raise ValueError('wave preview needs a project input hash')
    bounds=np.asarray(value['bounds'],dtype=float)
    if bounds.shape!=(4,) or not np.isfinite(bounds).all() or bounds[0]>=bounds[1] or bounds[2]>=bounds[3]:
        raise ValueError('wave bounds must be [xmin,xmax,ymin,ymax]')
    value['bounds']=bounds.tolist()
    if not isinstance(value['bodies'],list) or not 1<=len(value['bodies'])<=2:
        raise ValueError('this preview supports one or two captured bodies')
    ids=set()
    for body in value['bodies']:
        if type(body) is not dict or set(body)!={'body_id','vertices','triangles'}:
            raise ValueError('preview body fields must match the named schema')
        bid=body['body_id']
        if not isinstance(bid,str) or not bid or bid in ids:
            raise ValueError('preview body identities must be unique')
        ids.add(bid)
        vertices=np.asarray(body['vertices'],dtype=float)
        faces=np.asarray(body['triangles'])
        if vertices.ndim!=2 or vertices.shape[1]!=3 or len(vertices)<4 or not np.isfinite(vertices).all():
            raise ValueError('preview mesh needs finite WORLD vertices')
        if faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu' or not len(faces):
            raise ValueError('preview mesh needs integer triangles')
        if faces.min()<0 or faces.max()>=len(vertices) or np.any(np.sort(faces,axis=1)[:,1:]==np.sort(faces,axis=1)[:,:-1]):
            raise ValueError('preview mesh triangle indices are invalid')
        # Closed oriented source envelope; Capytaine subsequently clips it.
        edges={};counts={}
        for triangle in faces:
            for i,j in zip(triangle,np.roll(triangle,-1)):
                key=tuple(sorted((int(i),int(j))))
                edges[key]=edges.get(key,0)+(1 if i<j else -1)
                counts[key]=counts.get(key,0)+1
        if any(v!=0 or counts[k]!=2 for k,v in edges.items()):
            raise ValueError('preview requires a closed consistently oriented body envelope')
        volume=np.einsum('ij,ij->i',vertices[faces[:,0]],np.cross(vertices[faces[:,1]],vertices[faces[:,2]])).sum()/6.
        if not volume>0 or np.any(np.linalg.norm(np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]]),axis=1)<=1e-12):
            raise ValueError('preview requires outward nondegenerate panels')
        if vertices[:,2].min()<=-value['water_depth']:
            raise ValueError('preview body must stay clear of the flat seabed')
        if vertices[:,2].min()>=0:
            raise ValueError('preview body has no immersed part')
    return value

def request_hash(request):
    return _hash(validate_request(request))

def sample_grid(request):
    """Retained triangular grid, conservatively excluding body XY envelopes."""
    r=validate_request(request)
    xs=np.linspace(*r['bounds'][:2],r['nx']);ys=np.linspace(*r['bounds'][2:],r['ny'])
    xx,yy=np.meshgrid(xs,ys)
    vertices=np.column_stack((xx.ravel(),yy.ravel(),np.zeros(xx.size)))
    boxes=[(np.min(b['vertices'],axis=0)[:2],np.max(b['vertices'],axis=0)[:2]) for b in r['bodies']]
    faces=[]
    for j in range(r['ny']-1):
        for i in range(r['nx']-1):
            low=np.array([xs[i],ys[j]]);high=np.array([xs[i+1],ys[j+1]])
            if any(np.all(high>=a-1e-3) and np.all(low<=b+1e-3) for a,b in boxes):
                continue
            a=j*r['nx']+i;b=a+1;c=a+r['nx'];d=c+1
            faces.extend(((a,b,d),(a,d,c)))
    if not faces:
        raise ValueError('wave grid is covered by the captured body envelopes')
    faces=np.asarray(faces,dtype=np.int64);used=np.unique(faces)
    mapping=np.full(len(vertices),-1,dtype=np.int64);mapping[used]=np.arange(len(used))
    return vertices[used],mapping[faces]

def incident_field(request):
    from anyloads.marine import AiryEnvironment,AiryComponent
    r=validate_request(request)
    env=AiryEnvironment(r['water_depth'],(AiryComponent(r['amplitude'],r['omega'],r['heading_degrees'],r['phase']),))
    vertices,triangles=sample_grid(r)
    beta=math.radians(r['heading_degrees'])
    phase=env.wave_numbers[0]*(vertices[:,0]*math.cos(beta)+vertices[:,1]*math.sin(beta))+r['phase']
    eta=r['amplitude']*np.exp(1j*phase)
    return field_record(r,vertices,triangles,eta,np.zeros_like(eta),'incident_only',None,0)

def field_record(request,vertices,triangles,incident,diffraction,kind,version,panels):
    record=dict(api_version=API_VERSION,request=validate_request(request),request_hash=request_hash(request),
        vertices=np.asarray(vertices).tolist(),triangles=np.asarray(triangles).tolist(),
        incident_real=np.real(incident).tolist(),incident_imag=np.imag(incident).tolist(),
        diffraction_real=np.real(diffraction).tolist(),diffraction_imag=np.imag(diffraction).tolist(),
        phasor='exp(-i omega t)',kind=kind,capytaine_version=version,panels=panels,
        solver_settings=dict(SOLVER_SETTINGS) if kind=='fixed_body_diffraction_preview' else None)
    return validate_field(record)

def validate_field(data,expected_request=None):
    keys={'api_version','request','request_hash','vertices','triangles','incident_real','incident_imag',
          'diffraction_real','diffraction_imag','phasor','kind','capytaine_version','panels','solver_settings'}
    if type(data) is not dict or set(data)!=keys or data['api_version']!=API_VERSION or data['phasor']!='exp(-i omega t)':
        raise ValueError('wave field fields/version/phasor must match the named schema')
    r=validate_request(data['request'])
    if data['request_hash']!=request_hash(r) or expected_request is not None and request_hash(expected_request)!=data['request_hash']:
        raise ValueError('wave field belongs to different captured inputs')
    vertices,triangles=sample_grid(r)
    if not np.array_equal(vertices,np.asarray(data['vertices'])) or not np.array_equal(triangles,np.asarray(data['triangles'])):
        raise ValueError('wave field grid does not match its request')
    for name in ('incident_real','incident_imag','diffraction_real','diffraction_imag'):
        values=np.asarray(data[name],dtype=float)
        if values.shape!=(len(vertices),) or not np.isfinite(values).all():
            raise ValueError('wave field must contain finite complex elevations at every retained node')
    from anyloads.marine import AiryEnvironment,AiryComponent
    env=AiryEnvironment(r['water_depth'],(AiryComponent(r['amplitude'],r['omega'],r['heading_degrees'],r['phase']),))
    beta=math.radians(r['heading_degrees'])
    expected=r['amplitude']*np.exp(1j*(env.wave_numbers[0]*(vertices[:,0]*math.cos(beta)+vertices[:,1]*math.sin(beta))+r['phase']))
    actual=np.asarray(data['incident_real'])+1j*np.asarray(data['incident_imag'])
    if not np.allclose(actual,expected,rtol=1e-12,atol=1e-12):
        raise ValueError('incident wave phasor does not match captured wave inputs')
    if data['kind']=='incident_only':
        if data['solver_settings'] is not None or data['panels']!=0 or data['capytaine_version'] is not None or np.any(np.asarray(data['diffraction_real'])) or np.any(np.asarray(data['diffraction_imag'])):
            raise ValueError('incident-only field cannot claim body diffraction')
    elif data['kind']=='fixed_body_diffraction_preview':
        if data['solver_settings']!=SOLVER_SETTINGS or data['capytaine_version']!='3.0.0' or type(data['panels']) is not int or not 1<=data['panels']<=48:
            raise ValueError('unsupported diffraction preview source/panel count')
    else:
        raise ValueError('unknown wave field kind')
    return json.loads(json.dumps(data,allow_nan=False))

def positions_at_time(field,time,*,disturbed=False):
    f=validate_field(field)
    if isinstance(time,bool) or not math.isfinite(float(time)):
        raise ValueError('wave time must be finite')
    if type(disturbed) is not bool or disturbed and f['kind']!='fixed_body_diffraction_preview':
        raise ValueError('disturbed view requires an actual diffraction field')
    eta=np.asarray(f['incident_real'])+1j*np.asarray(f['incident_imag'])
    if disturbed:
        eta+=np.asarray(f['diffraction_real'])+1j*np.asarray(f['diffraction_imag'])
    points=np.array(f['vertices'],dtype=float)
    points[:,2]=np.real(eta*np.exp(-1j*f['request']['omega']*float(time)))
    return points

def compute_diffraction_preview(request,*,cancel=None):
    """One coarse fixed-body diffraction solve; never a dynamics qualification."""
    import importlib.metadata
    r=validate_request(request)
    if importlib.metadata.version('capytaine')!='3.0.0':
        raise ValueError('wave preview requires the tested Capytaine3.0.0')
    if cancel is not None and cancel():
        raise InterruptedError('wave preview cancelled before backend import')
    import capytaine as cpt
    bodies=[]
    for b in r['bodies']:
        mesh=cpt.Mesh(vertices=np.array(b['vertices']),faces=[list(t) for t in b['triangles']])
        mesh=mesh.immersed_part(water_depth=r['water_depth'])
        bodies.append(cpt.FloatingBody(mesh=mesh,name=b['body_id']))
    panels=sum(b.mesh.nb_faces for b in bodies)
    if not 1<=panels<=48:
        raise ValueError(f'wave preview panel limit48 exceeded ({panels})')
    with np.errstate(invalid='ignore',divide='ignore'):
        combined=bodies[0] if len(bodies)==1 else cpt.Multibody(bodies,name='wave-preview-fixed-bodies')
    if cancel is not None and cancel():
        raise InterruptedError('wave preview cancelled before solve')
    # Public direct-integration option avoids global tabulation startup/cache
    # work for this bounded coarse preview. Retain the default1001-point rule.
    settings={k:v for k,v in SOLVER_SETTINGS.items() if k not in ('green_function','n_threads')}
    solver=cpt.BEMSolver(green_function=cpt.Delhommeau(**settings))
    result=solver.solve(cpt.DiffractionProblem(body=combined,omega=r['omega'],
        wave_direction=math.radians(r['heading_degrees']),water_depth=r['water_depth']),
        keep_details=True,n_threads=1)
    if cancel is not None and cancel():
        raise InterruptedError('wave preview cancelled after solve')
    incident=incident_field(r)
    vertices=np.asarray(incident['vertices'])
    scattered=solver.compute_free_surface_elevation(vertices[:,:2],result)
    scattered=np.asarray(scattered)*r['amplitude']*np.exp(1j*r['phase'])
    eta=np.asarray(incident['incident_real'])+1j*np.asarray(incident['incident_imag'])
    return field_record(r,vertices,incident['triangles'],eta,scattered,
                        'fixed_body_diffraction_preview','3.0.0',panels)
