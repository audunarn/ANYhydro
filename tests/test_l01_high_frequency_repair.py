"""Source integration accuracy and public prepare/retained-field contracts."""
from dataclasses import replace
import os

import numpy as np
import pytest

from anyhydro import floating_radiation_fields as ff
from anyhydro.direct_lid_profile import create_source_profile, verify_profile_body
from anyhydro.source_quadrature import RULES, bilinear_quadrature, configure, verify
from anyhydro.wave_surface_fields import WaveFieldError

KW = dict(heading=0., water_depth=50., rho=1025., gravity=9.81, Q0=np.eye(3), cg_world=[0.,0.,-.4])


def box():
    v = np.array([[-3,-2,-1],[3,-2,-1],[3,2,-1],[-3,2,-1],[-3,-2,0],[3,-2,0],[3,2,0],[-3,2,0]],float)
    f = np.array([[0,3,2,1],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]])
    return v,f,['hull']*len(f)


@pytest.mark.parametrize('order',[3,4])
def test_rule_integrates_known_rectangle_and_triangle_moments(order):
    # Independent analytic moments: rectangle[0,2]x[0,3], triangle x+y<=2.
    v=np.array([[0.,0.,-1.],[2.,0.,-1.],[2.,3.,-1.],[0.,3.,-1.],
                [0.,0.,-2.],[2.,0.,-2.],[0.,2.,-2.]])
    p,w=bilinear_quadrature(v,np.array([[0,1,2,3],[4,5,6,6]]),order)
    np.testing.assert_allclose(w.sum(axis=1),[6.,2.],rtol=1e-14)
    np.testing.assert_allclose((w*p[:,:,0]).sum(axis=1),[6.,4/3],rtol=1e-14)
    np.testing.assert_allclose((w*p[:,:,0]**4).sum(axis=1),[96/5,32/15],rtol=1e-14)
    assert (w>0).all() and p.shape==(2,order**2,3)


def test_identities_are_opt_in_and_refuse_unsupported_families():
    v,f,ids=box()
    legacy=ff.family_request(v,f,ids,recipe=ff.DIRECT_RECIPE,**KW)
    gauss2=ff.family_request(v,f,ids,recipe=ff.DIRECT_RECIPE,source_quadrature='Gauss-Legendre 2',**KW)
    keys={ff.radiation_key(legacy),ff.radiation_key(gauss2)}
    for rule in RULES:
        req=ff.family_request(v,f,ids,recipe=ff.DIRECT_RECIPE,source_quadrature=rule,**KW)
        keys.add(ff.radiation_key(req))
        with pytest.raises(WaveFieldError,match='quadrature'):
            ff.family_request(v,f,ids,source_quadrature=rule,**KW)
        with pytest.raises(WaveFieldError,match='quadrature'):
            ff._source_quadrature(dict(req,transient_source_profile={}))
    assert len(keys)==4 and 'source_quadrature' not in legacy


def test_folded_face_is_refused_without_mutating_inputs():
    v=np.array([[0.,0.,-1.],[2.,0.,-1.],[.5,.5,-1.],[0.,2.,-1.]])
    f=np.array([[0,1,2,3]]);before=v.copy(),f.copy()
    for order in [3,4]:
        with pytest.raises(ValueError,match='folded'):
            bilinear_quadrature(v,f,order)
    np.testing.assert_array_equal(v,before[0]);np.testing.assert_array_equal(f,before[1])


@pytest.mark.parametrize('rule',list(RULES))
def test_joined_body_rule_and_mutation_refusal(rule):
    cpt=pytest.importorskip('capytaine')
    v,f,ids=box();profile=create_source_profile(v,f,ids,lid_z=-.1,faces_max_radius=1.,water_depth=50.)
    req=ff.family_request(v,f,ids,recipe=ff.DIRECT_LID_RECIPE,direct_lid_profile=profile,source_quadrature=rule,**KW)
    _,_,_,_,body=ff._body_for_family(cpt,v,f,ids,req,1.2)
    for mesh in [body.mesh,body.lid_mesh,body.mesh_including_lid]:
        verify(mesh,rule)
        assert mesh.quadrature_points[0].shape[1]==RULES[rule]**2
    full=body.mesh_including_lid
    points,weights=full.quadrature_points
    full.__dict__['quadrature_points']=(points,weights*2)
    with pytest.raises(WaveFieldError,match='quadrature'):
        verify_profile_body(body,req)
    full.__dict__['quadrature_points']=()
    with pytest.raises(WaveFieldError,match='quadrature'):
        verify_profile_body(body,req)


@pytest.mark.parametrize('rule',['Gauss-Legendre 2',*RULES])
def test_native_source_derivative_and_exact_imaginary_agree_with_untabulated(rule):
    cpt=pytest.importorskip('capytaine')
    from anyhydro.green_function import anyhydro_green_function
    from anyhydro.source_quadrature import native_options
    # Quad plus triangle, separated from receivers; no singular quadrature in oracle.
    v=np.array([[0,0,-1],[1,0,-1],[1,1,-1],[0,1,-1],[2,0,-2],[3,0,-2],[2,1,-2]],float)
    src=cpt.Mesh(vertices=v,faces=np.array([[0,1,2,3],[4,5,6,6]]),auto_check=False,**native_options(rule))
    configure(src,rule)
    x=np.array([[4.,2.,-.3],[-2.,-1.,-.7]])
    gf=anyhydro_green_function(tabulation_nr=0,tabulation_nz=0)
    base=type(gf).__mro__[1]
    args=dict(water_depth=50.,wavenumber=.8,adjoint_double_layer=False,early_dot_product=True,diagonal_term_in_double_layer=False)
    S0,K0=base.evaluate(gf,x,src,**args);S,K=gf.evaluate(x,src,**args)
    np.testing.assert_allclose(S.imag,S0.imag,rtol=0,atol=1e-6*np.abs(S0.imag).max())
    np.testing.assert_allclose(K.imag,K0.imag,rtol=0,atol=1e-6*np.abs(K0.imag).max())


@pytest.mark.skipif(os.environ.get(ff.NATIVE_GATE)!='1',reason='explicit native family solve')
@pytest.mark.parametrize('rule',list(RULES))
def test_public_lid_solve_cache_donor_recovery_and_closure(rule,tmp_path,monkeypatch):
    cpt=pytest.importorskip('capytaine')
    v,f,ids=box();original=(v.copy(),f.copy())
    profile=create_source_profile(v,f,ids,lid_z=-.1,faces_max_radius=1.,water_depth=50.)
    kwargs=dict(recipe=ff.DIRECT_LID_RECIPE,direct_lid_profile=profile,source_quadrature=rule,**KW)
    s=ff.solve_family_snapshot(v,f,ids,1.2,**kwargs)
    assert s.provider['source_quadrature']==rule
    W=np.column_stack([s.normals,np.cross(s.centers-np.array(KW['cg_world']),s.normals)])*s.areas[:,None]
    np.testing.assert_allclose(-W.T@s.panel_pressure_radiation.T,s.omega**2*s.added_mass+1j*s.omega*s.damping,rtol=1e-12,atol=1e-8)
    ff.save_family_snapshot(tmp_path,s)
    loaded=ff.recover_family_snapshot(tmp_path,s.request,s.omega)
    assert loaded.snapshot_id==s.snapshot_id
    donor=ff.solve_family_snapshot(v,f,ids,1.2,donor=s,**dict(kwargs,heading=.2))
    np.testing.assert_array_equal(donor.damping,s.damping)
    with pytest.raises(WaveFieldError,match='basis'):
        ff.solve_family_snapshot(v,f,ids,1.2,donor=s,**dict(kwargs,source_quadrature='Gauss-Legendre 2'))
    with pytest.raises(WaveFieldError,match='provider'):
        replace(s,provider=dict(s.provider,source_quadrature='Gauss-Legendre 2'),snapshot_id='')
    monkeypatch.setattr(cpt.BEMSolver,'solve',lambda *a,**k:pytest.fail('recovery attempted solve'))
    rec=ff._NativeRecovery(loaded)
    verify(rec.kw['body'].mesh_including_lid,rule)
    for j in [*range(6),None]:assert np.isfinite(rec.potential(np.array([[4.,0.,-.5]]),j)).all()
    np.testing.assert_array_equal(v,original[0]);np.testing.assert_array_equal(f,original[1])


@pytest.mark.skipif(os.environ.get(ff.NATIVE_GATE)!='1',reason='explicit native family solve')
def test_public_without_lid_and_native_mutation_refusal(monkeypatch):
    cpt=pytest.importorskip('capytaine')
    v,f,ids=box();rule=next(iter(RULES))
    s=ff.solve_family_snapshot(v,f,ids,1.2,recipe=ff.DIRECT_RECIPE,source_quadrature=rule,**KW)
    rec=ff._NativeRecovery(s);verify(rec.kw['body'].mesh,rule)
    assert np.isfinite(rec.potential(np.array([[4.,0.,-.5]]),0)).all()
    native=cpt.RadiationProblem
    def mutate(*args,**kw):
        prob=native(*args,**kw)
        p,w=prob.body.mesh_including_lid.quadrature_points
        prob.body.mesh_including_lid.__dict__['quadrature_points']=(p,w*2)
        return prob
    monkeypatch.setattr(cpt,'RadiationProblem',mutate)
    with pytest.raises(WaveFieldError,match='quadrature'):
        ff.solve_family_snapshot(v,f,ids,1.2,recipe=ff.DIRECT_RECIPE,source_quadrature=rule,**KW)
