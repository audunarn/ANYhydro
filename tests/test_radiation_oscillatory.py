"""Exact interpolant quadrature tests; no BEM, dynamics or qualification."""
import dataclasses
import numpy as np
import pytest
from anyhydro import radiation as rad
from test_radiation import make_oracle_dataset

def settings(**options):
    return rad.RadiationSettings(dt=.05,n_time=641,symmetry_tol=1e-10,passivity_tol=1e-10,
        low_tail_policy='linear_to_zero_extension',extension_name='TEST low-tail assumption',**options)

@pytest.mark.parametrize('grid', [[0.,.2,1.3,3.7],[0.,1.,2.,3.],[0.,1e-5,.001,2.]])
def test_affine_interval_gauss_oracle(grid):
    omega=np.array(grid);rng=np.random.default_rng(932)
    # Deliberately asymmetric signed matrices: no repair/clamp or diagonal-only path.
    b=rng.normal(size=(len(grid),4,4))
    t=np.array([0.,1e-12,1e-6,.01,.0999999,.1000001,1.7,12.3,70.])
    actual=rad._piecewise_linear_cosine_integral(omega,b,t)
    x,weights=np.polynomial.legendre.leggauss(180);oracle=np.zeros_like(actual)
    for i in range(len(omega)-1):
        half=(omega[i+1]-omega[i])/2;mid=(omega[i]+omega[i+1])/2
        interp=(b[i]+b[i+1])/2+x[:,None,None]*(b[i+1]-b[i])/2
        oracle+=np.einsum('tn,n,nij->tij',np.cos(t[:,None]*(mid+half*x)),half*weights,interp)
    np.testing.assert_allclose(actual,oracle,rtol=1e-10,atol=2e-12)
    np.testing.assert_allclose(actual[0],np.trapezoid(b,omega,axis=0),rtol=1e-14,atol=1e-14)

def test_uniform_sample_recurrence_removed_by_public_builder():
    ds=make_oracle_dataset(omegas=np.array([1.,2.,3.]),n_dof=6)
    old=dataclasses.replace(settings(),dt=2*np.pi,n_time=21)
    legacy=rad.radiation_kernel_from_dataset(ds,old)
    exact=rad.radiation_kernel_from_dataset(ds,dataclasses.replace(old,quadrature_method='piecewise_linear_exact'))
    np.testing.assert_allclose(legacy.kernel,np.broadcast_to(legacy.kernel[0],legacy.kernel.shape),rtol=1e-14,atol=1e-10)
    np.testing.assert_allclose(exact.kernel[0],legacy.kernel[0],rtol=1e-14,atol=1e-10)
    assert np.max(np.abs(exact.kernel[1:])) < 1e-12*np.max(np.abs(exact.kernel[0]))
    assert exact.qualification_status=='unqualified'

def test_settings_legacy_identity_unknown_refusal_and_roundtrip(tmp_path):
    ds=make_oracle_dataset(omegas=np.array([.5,1.,2.]),n_dof=6)
    legacy=settings();old=rad.radiation_kernel_from_dataset(ds,legacy)
    assert 'quadrature_method' not in legacy.to_dict()
    np.testing.assert_array_equal(old.kernel,rad.radiation_kernel_from_dataset(ds,rad.RadiationSettings.from_dict(legacy.to_dict())).kernel)
    exact=rad.radiation_kernel_from_dataset(ds,dataclasses.replace(legacy,quadrature_method='piecewise_linear_exact'))
    assert exact.content_hash()!=old.content_hash()
    assert exact.settings.to_dict()['quadrature_method']=='piecewise_linear_exact'
    path=tmp_path/'exact.nc';rad.save_radiation_kernel(exact,path)
    loaded=rad.load_radiation_kernel(path)
    assert loaded.content_hash()==exact.content_hash()
    assert loaded.settings.quadrature_method=='piecewise_linear_exact'
    np.testing.assert_array_equal(loaded.kernel,exact.kernel)
    for method in ('unknown',None,True,1):
        with pytest.raises(rad.RadiationSettingsError):
            rad.RadiationSettings.from_dict({**legacy.to_dict(),'quadrature_method':method})

def test_time_window_is_applied_after_exact_integral():
    ds=make_oracle_dataset(omegas=np.array([.5,1.,2.]),n_dof=6)
    full=rad.radiation_kernel_from_dataset(ds,settings(quadrature_method='piecewise_linear_exact'))
    windowed=rad.radiation_kernel_from_dataset(ds,settings(quadrature_method='piecewise_linear_exact',time_window=1.))
    np.testing.assert_array_equal(windowed.kernel[windowed.time<=1],full.kernel[full.time<=1])
    assert np.all(windowed.kernel[windowed.time>1]==0)

def test_exponential_oracle_reconstruction_both_matrices():
    ds=make_oracle_dataset(n_dof=6)
    b0=np.asarray(ds.radiation_damping)[0]*0
    # Source helper default exponential matrix parameters are independently known.
    from test_radiation import _oracle_matrices
    c,lam,ainf=_oracle_matrices(6);b0=c/lam
    s=rad.RadiationSettings(dt=.01,n_time=1001,symmetry_tol=1e-10,passivity_tol=1e-10,
        zero_frequency_damping=b0,quadrature_method='piecewise_linear_exact')
    k=rad.radiation_kernel_from_dataset(ds,s)
    # Finite-band PL approximation is checked, not asserted exact exponential.
    np.testing.assert_allclose(k.kernel[100:600],c[None]*np.exp(-k.time[100:600,None,None]*lam[None]),rtol=.12,atol=.5)
    criteria=rad.ReconstructionCriteria(omega_min=.5,omega_max=5.,added_mass_abs_tol=1.,
        added_mass_rel_tol=.02,damping_abs_tol=1.,damping_rel_tol=.02)
    report=rad.compare_reconstruction(k,ds,criteria)
    assert report.overall_pass

def test_constant_and_triangle_closed_form_and_knot_insertion():
    t=np.array([0.,1e-12,.003,.09999,.10001,.3,2.,20.,100.])
    w=np.array([0.,.3,1.7,4.])
    matrix=np.array([[1.,-2.],[3.,.1]])
    const=np.broadcast_to(matrix,(len(w),2,2)).copy()
    tri=(1-w/4)[:,None,None]*matrix
    result=rad._piecewise_linear_cosine_integral(w,const,t)
    np.testing.assert_allclose(result,4*np.sinc(4*t/np.pi)[:,None,None]*matrix,rtol=1e-10,atol=1e-13)
    result=rad._piecewise_linear_cosine_integral(w,tri,t)
    np.testing.assert_allclose(result,2*np.sinc(2*t/np.pi)[:,None,None]**2*matrix,rtol=1e-10,atol=1e-13)
    rng=np.random.default_rng(28);b=rng.normal(size=(len(w),2,2))
    refined=np.sort(np.r_[w,(w[1:]+w[:-1])/2])
    values=np.stack([np.interp(refined,w,b[:,i,j]) for i in range(2) for j in range(2)],axis=-1).reshape(-1,2,2)
    np.testing.assert_allclose(rad._piecewise_linear_cosine_integral(w,b,t),
        rad._piecewise_linear_cosine_integral(refined,values,t),rtol=1e-10,atol=1e-13)

def test_small_z_switch_and_block_boundary():
    w=np.array([0.,1.,3.]);b=np.array([[[1.]],[[2.]],[[3.]]])
    t=np.linspace(0,2,1027); t[512:514]=[.099999999,.100000001]
    all_at_once=rad._piecewise_linear_cosine_integral(w,b,t)
    separately=np.concatenate([rad._piecewise_linear_cosine_integral(w,b,t[i:i+1]) for i in range(len(t))])
    # Matrix/vector BLAS paths may round differently; this is a machine-precision
    # equivalence check, not a physical acceptance tolerance.
    np.testing.assert_allclose(all_at_once,separately,rtol=0,atol=16*np.finfo(float).eps*np.max(np.abs(all_at_once)))
    assert abs(all_at_once[512,0,0]-all_at_once[513,0,0])<1e-7

from test_derived_operators import oracle_ds, derived4

def test_public_derived_builder_retains_crossbody_and_source_proof(oracle_ds,derived4):
    s=settings(quadrature_method='piecewise_linear_exact')
    full=rad.radiation_kernel_from_dataset(oracle_ds,s)
    retained=rad.radiation_kernel_from_derived_operators(derived4,s,source_dataset=oracle_ds)
    indices=[0,2,6,8]
    # Fixture explicitly registers reciprocal_average; integrating is linear,
    # so its declared correction must commute with the common cosine integral.
    projected=full.kernel[:,indices][:,:,indices]
    expected=(projected+projected.transpose(0,2,1))/2
    np.testing.assert_allclose(retained.kernel,expected,rtol=1e-12,atol=1e-12)
    assert np.any(retained.kernel[:,:2,2:]!=0)
    assert retained.settings.quadrature_method=='piecewise_linear_exact'
    changed=oracle_ds.copy(deep=True)
    changed['radiation_damping'].values[0,0,0]+=1
    from anyhydro.operators import DerivedOperatorError
    with pytest.raises(DerivedOperatorError,match='source dataset hash'):
        rad.radiation_kernel_from_derived_operators(derived4,s,source_dataset=changed)
