# SPDX-License-Identifier: MPL-2.0
"""Fabricated closed-form and refusal controls; no Green evaluation or solve."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("retained_audit", Path(__file__).parents[1] / "tools/audit_retained_panel_energy.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_single_source_closed_form_far_and_pole_normalization():
    # At xy=0,z=0 the angular amplitude is constant sigma*area/(4*pi).
    # Im(S) for this single panel is -k*area/(2*D). Derive both powers
    # directly from that amplitude, independently of the implementation.
    k, h, rho, omega, area = .2, 10., 1025., 1.4, 2.
    sigma = 2 + 3j
    D = np.tanh(k*h) + k*h / np.cosh(k*h)**2
    expected = rho*omega*k*abs(sigma)**2*area**2/(4*D)
    points = np.zeros((1,1,3))
    far = audit.far_mean_powers(points, np.array([[area]]), np.array([sigma]), k=k, water_depth=h, rho=rho, omega=omega, angular_counts=(64,128,256))
    for value in far.values():
        assert value == pytest.approx(expected, rel=2e-15)
    form = np.array([[-k*area**2*abs(sigma)**2/(2*D)]])
    assert audit.source_pole_power_matrix(form, rho, omega)[0,0] == pytest.approx(expected, rel=2e-15)


def test_complex_rate_mechanical_work():
    # F=-diag(2,4)+i*diag(7,9); reactive forces do no mean work.
    force=np.diag([-2+7j,-4+9j]); rate=np.array([1+2j,3-1j])
    assert audit.near_power(force,rate) == 25


def test_residual_hull_lid_partition():
    weights=np.ones((2,1)); pressure=np.array([[2j],[4.]])
    residual=np.array([[3j],[-2.]])
    r=audit.residual_diagnostics(weights,np.array([True,False]),pressure,residual)
    assert r['residual_work_hull_w']==3
    assert r['residual_work_lid_w']==-4
    assert r['residual_work_all_w']==-1


def fixture_case(path, bad=None):
    (path/'case.json').write_text('{}'); (path/'status.json').write_text('{}')
    values=dict(points=np.zeros((1,1,3)),weights=np.ones((1,1)),normals=np.array([[0.,0.,1.]]),hull_mask=np.array([True]),rigid_trace=np.ones((1,1,1)),vertices=np.zeros((3,3)),faces=np.array([[0,1,2]]),owner=np.array([0]))
    if bad=='object': values['points']=np.array([object()],dtype=object)
    if bad=='shape': values['points']=np.zeros((1,3))
    if bad=='nan': values['weights'][0,0]=np.nan
    np.savez(path/'input.npz',**values)
    np.savez(path/'radiation.npz',sources=np.ones((1,1),complex),pressure=np.ones((1,1,1),complex),node_residual=np.zeros((1,1,1)),mean_residual=np.zeros((1,1)),force_per_velocity=-np.ones((1,1)),added_mass=np.zeros((1,1)),radiation_damping=np.ones((1,1)),dofs=np.array(['body/surge']))


@pytest.mark.parametrize('bad',['object','shape','nan'])
def test_invalid_stored_fields_refused(tmp_path,bad):
    fixture_case(tmp_path,bad)
    with pytest.raises((audit.AuditRefusal,ValueError)):
        audit.load_case(tmp_path,(0,))


def test_snapshot_tampering_refused(tmp_path):
    fixture_case(tmp_path)
    (tmp_path/'case.json').write_text(json.dumps({'snapshot_arrays':{'weights':{'sha256':'wrong'}}}))
    with pytest.raises(audit.AuditRefusal,match='identity mismatch'):
        audit.load_case(tmp_path,(0,))


def test_missing_and_existing_output_refusal(tmp_path):
    assert audit.main(['--case-dir',str(tmp_path/'missing'),'--parent-dir',str(tmp_path/'parent'),'--output-dir',str(tmp_path/'out')])==2
    before=list(tmp_path.iterdir())
    assert audit.main(['--case-dir',str(tmp_path),'--parent-dir',str(tmp_path),'--output-dir',str(tmp_path)])==2
    assert list(tmp_path.iterdir())==before


def test_public_cli_fabricated_identity_and_read_only_workflow(tmp_path):
    parent=tmp_path/'parent'; case=parent/'case'; case.mkdir(parents=True)
    fixture_case(case)
    k,h,rho=.2,10.,1025.
    omega=float(np.sqrt(9.81*k*np.tanh(k*h)))
    D=np.tanh(k*h)+k*h/np.cosh(k*h)**2
    S=np.array([[-1j*k/(2*D)]])
    p=1j*rho*omega*S
    power=float(p[0,0].real/2)
    np.savez(case/'operator-0.npz',s=S,k=np.ones((1,1)))
    np.savez(case/'radiation.npz',sources=np.ones((1,1),complex),pressure=p.reshape(1,1,1),node_residual=np.zeros((1,1,1)),mean_residual=np.zeros((1,1)),force_per_velocity=-p,added_mass=np.zeros((1,1)),radiation_damping=2*np.array([[power]]),dofs=np.array(['fabricated/surge']))
    (case/'case.json').write_text(json.dumps({'case_hash':'fabricated-fixture','gravity':9.81}))
    (parent/'summary.json').write_text(json.dumps({'candidate':{'case_hash':'fabricated-fixture','rate':[1.],'omega':omega,'rho':rho,'water_depth':h,'k':k}}))
    (parent/'energy-comparison.json').write_text(json.dumps({'added_mass':{'coefficients':[{'row':0,'column':0,'candidate':0.,'reference':0.}]},'radiation_damping':{'coefficients':[{'row':0,'column':0,'candidate':2*power,'reference':2*power}]},'mean_power':{'candidate_w':power,'reference_w':power}}))
    before={p:audit.sha256_file(p) for p in parent.rglob('*') if p.is_file()}
    args=['--case-dir',str(case),'--parent-dir',str(parent),'--output-dir',str(tmp_path/'output'),'--retained-dofs','0','--skip-native','--candidate-source',str(tmp_path/'absent-source')]
    assert audit.main(args)==0
    r=json.loads((tmp_path/'output'/audit.AUDIT_JSON_NAME).read_text())
    assert r['limits']['qualification'] is False
    assert r['energy']['near_w']==pytest.approx(power)
    assert r['energy']['far_closed_form_w']==pytest.approx(power)
    assert abs(r['energy']['closure_far_minus_mech_w'])<1e-12
    assert {p:audit.sha256_file(p) for p in before}==before
    (parent/'summary.json').write_text(json.dumps({'candidate':{'case_hash':'wrong','rate':[1.],'omega':omega,'rho':rho,'water_depth':h,'k':k}}))
    args[args.index('--output-dir')+1]=str(tmp_path/'refused')
    assert audit.main(args)==2
    assert not (tmp_path/'refused').exists()


@pytest.mark.parametrize('coefficients',[
 [{'row':0,'column':1,'candidate':0.,'reference':0.}],
 [{'row':0,'column':0,'candidate':1.,'reference':1.}]*2,
 [{'row':0,'column':0,'candidate':float('nan'),'reference':1.}],
])
def test_reference_coefficients_refuse_holes_duplicates_and_nonfinite(coefficients):
    comparison={k:{'coefficients':coefficients} for k in ('added_mass','radiation_damping')}
    with pytest.raises(audit.AuditRefusal):
        audit.parse_reference_blocks(comparison)


def test_mean_covariance_split_exact_complex_conjugate_and_refusals():
    # Independent hand computation on a fabricated 2-panel, 3-quadrature case.
    w=np.array([[1.,2.,1.],[.5,.5,1.]])
    areas=w.sum(axis=1)
    p=np.array([[1+2j,3-1j,2+0j],[1j,2+1j,-1j]])
    u=np.array([[2-1j,1+1j,.5+0j],[3,1j,2-2j]])
    mean_w,cov_w=audit.mean_covariance_split(w,areas,p,u)
    wbar=w/areas[:,None]
    pbar=(wbar*p).sum(axis=1); ubar=(wbar*u).sum(axis=1)
    assert mean_w==pytest.approx(.5*np.real((areas*pbar*np.conj(ubar)).sum()))
    dp=p-pbar[:,None]; du=u-ubar[:,None]
    assert cov_w==pytest.approx(.5*np.real((w*dp*np.conj(du)).sum()))
    pointwise=.5*np.real((w*p*np.conj(u)).sum())
    assert mean_w+cov_w==pytest.approx(pointwise,rel=1e-14,abs=1e-300)
    # complex-conjugate control: pairing u with conj(p) keeps the real power
    mean2,cov2=audit.mean_covariance_split(w,areas,u,p)
    assert mean2+cov2==pytest.approx(pointwise,rel=1e-14,abs=1e-300)
    with pytest.raises(audit.AuditRefusal):
        audit.mean_covariance_split(w,areas,p,u[:,:2])
    bad=p.copy(); bad[0,0]=np.nan
    with pytest.raises(audit.AuditRefusal):
        audit.mean_covariance_split(w,areas,bad,u)


def test_green_defect_decision_classification():
    assert audit.green_defect_decision(1.,0.,0.)=='discrete_green_relation_closed'
    # pointwise relation closes, panel-mean picture does not: covariance carries it
    assert audit.green_defect_decision(1.,1e-15,.2)=='within_panel_covariance_accounted'
    assert audit.green_defect_decision(1.,.1,.02)=='largely_within_panel_covariance'
    assert audit.green_defect_decision(1.,.1,.09)=='persists_in_saved_trace_green_relation'


def fabricated_pointwise_case():
    # 3 panels (2 hull + 1 lid), 2 quadrature points, 2 DOFs, complex fields.
    w=np.array([[1.,1.],[2.,1.],[1.,3.]])
    areas=w.sum(axis=1)
    hull=np.array([True,True,False])
    p=np.array([[[2+1j,1.],[4.,-1j]],[[3.,2j],[3-2j,1+1j]],[[1+1j,.5],[2.,-1j]]])
    g=np.array([[[1.,0.],[1.,2.]],[[2.,1.],[2.,0.]],[[0.,0.],[0.,0.]]])
    r=np.array([[[.5,0.],[-.5+1j,1j]],[[0.,2.],[1j,0.]],[[2.,1.],[-1.,0.]]])
    return w,areas,hull,p,g,r


def independent_decomposition(w,areas,hull,p,g,r):
    # Explicit-loop reference implementation independent of the tool.
    u=g+r
    flux=.5*float(np.real(sum(w[i,q]*p[i,q]*np.conj(u[i,q])
                              for i in range(len(w)) for q in range(w.shape[1]))))
    work=.5*float(np.real(sum(w[i,q]*p[i,q]*np.conj(g[i,q])
                              for i in range(len(w)) if hull[i] for q in range(w.shape[1]))))
    wbar=w/areas[:,None]
    pbar=np.einsum('iq,iq->i',wbar,p); ubar=np.einsum('iq,iq->i',wbar,u)
    gbar=np.einsum('iq,iq->i',wbar,g)
    flux_mean=.5*float(np.real((areas*pbar*np.conj(ubar)).sum()))
    work_mean=.5*float(np.real((areas[hull]*pbar[hull]*np.conj(gbar[hull])).sum()))
    flux_cov=.5*float(np.real((w*(p-pbar[:,None])*np.conj(u-ubar[:,None])).sum()))
    work_cov=.5*float(np.real((w[hull]*(p[hull]-pbar[hull,None])
                                *np.conj(g[hull]-gbar[hull,None])).sum()))
    return flux,work,flux_mean,work_mean,flux_cov,work_cov


def test_rate_work_adjoint_decomposition_identities_and_unit_power():
    w,areas,hull,p,g,r=fabricated_pointwise_case()
    flux,work,flux_mean,work_mean,flux_cov,work_cov=independent_decomposition(
        w,areas,hull,p[:, :, 0],g[:, :, 0],r[:, :, 0])
    # unit-power control: DOF 0 unit rate equals the DOF-0 fields directly
    d0=audit.rate_work_adjoint_decomposition(w,areas,hull,p[:,:,0],g[:,:,0],r[:,:,0],far_w=flux)
    assert d0['flux_pointwise_w']==pytest.approx(flux)
    assert d0['work_pointwise_w']==pytest.approx(work)
    assert d0['residual_work_pointwise_w']==pytest.approx(flux-work)
    assert d0['flux_panel_mean_w']==pytest.approx(flux_mean)
    assert d0['work_panel_mean_w']==pytest.approx(work_mean)
    assert d0['flux_within_panel_covariance_w']==pytest.approx(flux_cov)
    assert d0['work_within_panel_covariance_w']==pytest.approx(work_cov)
    for key in ('split_flux_identity_max_abs_w','split_work_identity_max_abs_w',
                'green_decomposition_identity_max_abs_w',
                'closure_decomposition_identity_max_abs_w'):
        assert d0[key]<1e-14
    # pointwise relation closes while the panel-mean picture does not:
    # the within-panel covariance is what carries the closure
    assert d0['decision']=='within_panel_covariance_accounted'
    # covariance-carried closure: far equals the panel-mean flux
    d1=audit.rate_work_adjoint_decomposition(w,areas,hull,p[:,:,0],g[:,:,0],r[:,:,0],far_w=flux_mean)
    assert d1['mean_persistent_defect_far_minus_flux_mean_w']==pytest.approx(0.,abs=1e-15)
    assert d1['decision']=='largely_within_panel_covariance'
    # persistent defect: far far away from both flux pictures
    d2=audit.rate_work_adjoint_decomposition(w,areas,hull,p[:,:,0],g[:,:,0],r[:,:,0],far_w=flux+10.)
    assert d2['decision']=='persists_in_saved_trace_green_relation'
    with pytest.raises(audit.AuditRefusal):
        audit.rate_work_adjoint_decomposition(w,areas,hull,p[:,:,0],g[:,:,0],r[:, :1],far_w=flux)
    bad=r.copy(); bad[0,0]=np.inf
    with pytest.raises(audit.AuditRefusal):
        audit.rate_work_adjoint_decomposition(w,areas,hull,p[:,:,0],g[:,:,0],bad,far_w=flux)


def test_projected_adjoint_transpose_and_work_conjugacy():
    # F = -M_work^T exactly for a real trace; the conjugated pairing carries the power.
    w,areas,hull,p,g,r=fabricated_pointwise_case()
    ndof=2
    m_work=np.zeros((ndof,ndof),complex); m_flux=np.zeros((ndof,ndof),complex)
    force=np.zeros((ndof,ndof),complex)
    for i in range(len(w)):
        for q in range(w.shape[1]):
            for d in range(ndof):
                for e in range(ndof):
                    if hull[i]:
                        m_work[d,e]+=w[i,q]*p[i,q,d]*np.conj(g[i,q,e])
                        force[d,e]-=w[i,q]*g[i,q,d]*p[i,q,e]
                    m_flux[d,e]+=w[i,q]*p[i,q,d]*np.conj(g[i,q,e]+r[i,q,e])
    assert np.max(np.abs(force+m_work.T))<1e-15
    m_res=m_flux-m_work
    rate=np.array([.7,-1.1])
    flux=.5*float(np.real(np.vdot(rate,m_flux@rate)))
    work=.5*float(np.real(np.vdot(rate,m_work@rate)))
    res=.5*float(np.real(np.vdot(rate,m_res@rate)))
    direct=audit.rate_work_adjoint_decomposition(
        w,areas,hull,np.einsum('iqd,d->iq',p,rate),
        np.einsum('iqd,d->iq',g,rate),np.einsum('iqd,d->iq',r,rate),far_w=flux)
    assert direct['flux_pointwise_w']==pytest.approx(flux)
    assert direct['work_pointwise_w']==pytest.approx(work)
    assert direct['residual_work_pointwise_w']==pytest.approx(res)
    assert direct['residual_work_pointwise_w']==pytest.approx(flux-work)
    assert direct['split_flux_identity_max_abs_w']<1e-14
    # work conjugacy: the unconjugated force reproduces the conjugated hull work
    assert audit.near_power(force,rate)==pytest.approx(work)
    # root counterexample at helper level: G=I, P=[[1+2i,3+4i],[5+6i,7+8i]],
    # equal weights, v=[1,i]. The pressure-first stencil is M=P^T; the
    # motion-coordinate matrix M.T=P gives mean power 5, the untransposed
    # pressure-first matrix gives 3.
    Pc=np.array([[1+2j,3+4j],[5+6j,7+8j]])
    m_first=Pc.T
    vc=np.array([1.,1j])
    assert .5*float(np.real(np.vdot(vc,m_first.T@vc)))==pytest.approx(5.)
    assert .5*float(np.real(np.vdot(vc,m_first@vc)))==pytest.approx(3.)


def test_public_cli_discrete_work_adjoint_section(tmp_path):
    parent=tmp_path/'parent'; case=parent/'case'; case.mkdir(parents=True)
    fixture_case(case)
    k,h,rho=.2,10.,1025.
    omega=float(np.sqrt(9.81*k*np.tanh(k*h)))
    D=np.tanh(k*h)+k*h/np.cosh(k*h)**2
    S=np.array([[-1j*k/(2*D)]])
    p=1j*rho*omega*S
    power=float(p[0,0].real/2)
    np.savez(case/'operator-0.npz',s=S,k=np.ones((1,1)))
    np.savez(case/'radiation.npz',sources=np.ones((1,1),complex),pressure=p.reshape(1,1,1),node_residual=np.zeros((1,1,1)),mean_residual=np.zeros((1,1)),force_per_velocity=-p,added_mass=np.zeros((1,1)),radiation_damping=2*np.array([[power]]),dofs=np.array(['fabricated/surge']))
    (case/'case.json').write_text(json.dumps({'case_hash':'fabricated-dw','gravity':9.81}))
    (parent/'summary.json').write_text(json.dumps({'candidate':{'case_hash':'fabricated-dw','rate':[1.],'omega':omega,'rho':rho,'water_depth':h,'k':k}}))
    (parent/'energy-comparison.json').write_text(json.dumps({'added_mass':{'coefficients':[{'row':0,'column':0,'candidate':0.,'reference':0.}]},'radiation_damping':{'coefficients':[{'row':0,'column':0,'candidate':2*power,'reference':2*power}]},'mean_power':{'candidate_w':power,'reference_w':power}}))
    args=['--case-dir',str(case),'--parent-dir',str(parent),'--output-dir',str(tmp_path/'output'),'--retained-dofs','0','--skip-native','--candidate-source',str(tmp_path/'absent-source')]
    assert audit.main(args)==0
    r=json.loads((tmp_path/'output'/audit.AUDIT_JSON_NAME).read_text())
    dw=r['discrete_work_adjoint']
    assert dw['frozen_rate']['decision']=='discrete_green_relation_closed'
    assert dw['decision_summary']['overall']=='discrete_green_relation_closed'
    assert dw['frozen_rate']['flux_pointwise_w']==pytest.approx(power)
    assert dw['frozen_rate']['flux_within_panel_covariance_w']==pytest.approx(0.,abs=1e-15)
    assert dw['frozen_rate']['split_flux_identity_max_abs_w']<1e-15
    assert r['energy']['per_dof'][0]['work_adjoint_decision']=='discrete_green_relation_closed'
    assert r['energy']['per_dof'][0]['flux_pointwise_w']==pytest.approx(power)
    assert len(dw['projected_retained']['flux_pointwise']['real'])==1
    assert dw['weighted_s_stencil_asymmetry_per_quadrature'][0]['weighted_s_asym_rel']==0.
    assert dw['weighted_s_summed_reciprocity']['weighted_s_summed_asym_rel']==0.
    assert dw['mean_enforcement_max_abs_velocity_m_per_s']<1e-15
    assert dw['adjoint_force_transpose_identity_max_abs']<1e-15
    md=(tmp_path/'output'/audit.AUDIT_MD_NAME).read_text()
    assert '## Discrete work/adjoint diagnosis' in md
    assert 'discrete_green_relation_closed' in md


def test_public_cli_complex_velocity_motion_coordinate_orientation(tmp_path):
    # Root counterexample exercised through the public audit/caller matrix
    # fields: v=[1,i], G=I, P=[[1+2i,3+4i],[5+6i,7+8i]], equal weights.
    # Direct boundary / negative-force mean power is 5; the wrongly oriented
    # (untransposed pressure-first) matrix gives 3. Exported projected matrices
    # must be motion-coordinate: power = .5 Re(v^H A v).
    parent=tmp_path/'parent'; case=parent/'case'; case.mkdir(parents=True)
    k,h,rho=.2,10.,1025.
    omega=float(np.sqrt(9.81*k*np.tanh(k*h)))
    P=np.array([[1+2j,3+4j],[5+6j,7+8j]])
    G=np.eye(2)
    values=dict(points=np.zeros((2,1,3)),weights=np.ones((2,1)),
                normals=np.array([[0.,0.,1.]]*2),hull_mask=np.array([True,True]),
                rigid_trace=G.reshape(2,1,2),vertices=np.zeros((3,3)),
                faces=np.array([[0,1,2],[0,1,2]]),owner=np.array([0,0]))
    np.savez(case/'input.npz',**values)
    np.savez(case/'radiation.npz',sources=G.astype(complex),pressure=P.reshape(2,1,2),
             node_residual=np.zeros((2,1,2)),mean_residual=np.zeros((2,2)),
             force_per_velocity=-P,added_mass=-P.imag/omega,radiation_damping=P.real,
             dofs=np.array(['fabricated/surge','fabricated/heave']))
    np.savez(case/'operator-0.npz',s=P/(1j*rho*omega),k=G)
    (case/'case.json').write_text(json.dumps({'case_hash':'complex-dw','gravity':9.81}))
    (case/'status.json').write_text('{}')
    (parent/'summary.json').write_text(json.dumps({'candidate':{'case_hash':'complex-dw',
        'rate':[1.,0.],'omega':omega,'rho':rho,'water_depth':h,'k':k}}))
    blocks={key:{'coefficients':[{'row':i,'column':j,'candidate':float(M[i,j]),
                                 'reference':float(M[i,j])}
                                for i in range(2) for j in range(2)]}
            for key,M in (('added_mass',-P.imag/omega),('radiation_damping',P.real))}
    (parent/'energy-comparison.json').write_text(json.dumps(blocks))
    args=['--case-dir',str(case),'--parent-dir',str(parent),'--output-dir',str(tmp_path/'output'),
          '--retained-dofs','0,1','--skip-native','--candidate-source',str(tmp_path/'absent-source')]
    assert audit.main(args)==0
    r=json.loads((tmp_path/'output'/audit.AUDIT_JSON_NAME).read_text())
    pr=r['discrete_work_adjoint']['projected_retained']
    A=np.array(pr['flux_pointwise']['real'])+1j*np.array(pr['flux_pointwise']['imag'])
    # exported matrix is motion-coordinate G^H W P = P, not its transpose
    assert np.max(np.abs(A-P))<1e-12
    v=np.array([1.,1j])
    direct=.5*float(np.real(np.sum((P@v)*np.conj(G@v))))
    assert direct==pytest.approx(5.)
    assert audit.near_power(-P,v)==pytest.approx(5.)
    assert .5*float(np.real(np.vdot(v,A@v)))==pytest.approx(5.)
    # wrong orientation counterexample: the untransposed pressure-first matrix
    assert .5*float(np.real(np.vdot(v,A.T@v)))==pytest.approx(3.)
    # .5 normalization: mean-power eigenvalues are half the work-form ones
    work_form=np.linalg.eigvalsh(.5*(A+A.conj().T))
    mean_power=np.array(pr['flux_mean_power_matrix_hermitian_eigenvalues'])
    assert np.max(np.abs(mean_power-.5*work_form))<1e-12
    assert np.max(np.abs(mean_power-np.linalg.eigvalsh(.25*(A+A.conj().T))))<1e-12
    # pole form is already motion-coordinate and is not transposed
    pole=np.array(pr['pole_form']['real'])+1j*np.array(pr['pole_form']['imag'])
    assert np.max(np.abs(pole-(-P.real/(rho*omega))))<1e-12


def test_public_cli_summed_weighted_s_reciprocity_cancellation(tmp_path):
    # Per-quadrature weighted-S asymmetry cancels in the quadrature sum:
    # S_0=[[0,1],[0,0]] and S_1=[[0,0],[1,0]] are each maximally asymmetric
    # under diag(w), but Sum_q diag(w_q) S_q is symmetric. Per-q entries are
    # stencil info; only the summed metric is a reciprocity measure.
    parent=tmp_path/'parent'; case=parent/'case'; case.mkdir(parents=True)
    k,h,rho=.2,10.,1025.
    omega=float(np.sqrt(9.81*k*np.tanh(k*h)))
    S=[np.array([[0.,1.],[0.,0.]]),np.array([[0.,0.],[1.,0.]])]
    values=dict(points=np.zeros((2,2,3)),weights=np.ones((2,2)),
                normals=np.array([[0.,0.,1.]]*2),hull_mask=np.array([True,True]),
                rigid_trace=np.ones((2,2,1)),vertices=np.zeros((3,3)),
                faces=np.array([[0,1,2],[0,1,2]]),owner=np.array([0,0]))
    np.savez(case/'input.npz',**values)
    pressure=np.stack([1j*rho*omega*(S[q]@np.ones((2,1))) for q in range(2)],axis=1)
    np.savez(case/'radiation.npz',sources=np.ones((2,1),complex),pressure=pressure,
             node_residual=np.zeros((2,2,1)),mean_residual=np.zeros((2,1)),
             force_per_velocity=np.array([[-2j*rho*omega]]),added_mass=np.array([[-2.*rho]]),
             radiation_damping=np.array([[0.]]),dofs=np.array(['fabricated/surge']))
    for q in range(2):
        np.savez(case/f'operator-{q}.npz',s=S[q],k=.5*np.ones((2,2)))
    (case/'case.json').write_text(json.dumps({'case_hash':'summed-dw','gravity':9.81}))
    (case/'status.json').write_text('{}')
    (parent/'summary.json').write_text(json.dumps({'candidate':{'case_hash':'summed-dw',
        'rate':[1.],'omega':omega,'rho':rho,'water_depth':h,'k':k}}))
    blocks={key:{'coefficients':[{'row':0,'column':0,'candidate':float(M[0,0]),
                                 'reference':float(M[0,0])}]}
            for key,M in (('added_mass',np.array([[-2.*rho]])),
                          ('radiation_damping',np.array([[0.]])))}
    (parent/'energy-comparison.json').write_text(json.dumps(blocks))
    args=['--case-dir',str(case),'--parent-dir',str(parent),'--output-dir',str(tmp_path/'output'),
          '--retained-dofs','0','--skip-native','--candidate-source',str(tmp_path/'absent-source')]
    assert audit.main(args)==0
    r=json.loads((tmp_path/'output'/audit.AUDIT_JSON_NAME).read_text())
    dw=r['discrete_work_adjoint']
    per_q=dw['weighted_s_stencil_asymmetry_per_quadrature']
    assert [e['qi'] for e in per_q]==[0,1]
    assert all(e['weighted_s_asym_rel']==pytest.approx(1.) for e in per_q)
    summed=dw['weighted_s_summed_reciprocity']
    assert summed['weighted_s_summed_asym_rel']==pytest.approx(0.,abs=1e-15)
    assert summed['weighted_s_summed_imag_asym_rel']==0.


def test_physical_hull_and_artificial_lid_flux_partition():
    weights=np.ones((3,1));areas=np.ones(3);hull=np.array([True,True,False])
    p=np.array([[2j],[4.],[6.]])
    g=np.array([[1j],[1.],[0.]])
    residual=np.array([[2j],[-3.],[2.]])
    # Actual u=[3i,-2,2]; half Re(p*conj(u)) gives [3,-4,6].
    d=audit.rate_work_adjoint_decomposition(weights,areas,hull,p,g,residual,far_w=10.)
    assert d['flux_physical_hull_pointwise_w']==-1.
    assert d['flux_artificial_lid_pointwise_w']==6.
    assert d['flux_pointwise_w']==5.
    assert d['far_minus_physical_hull_flux_w']==11.
    assert 'artificial internal lid' in d['physical_boundary_note']
