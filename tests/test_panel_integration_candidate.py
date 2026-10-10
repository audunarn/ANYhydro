"""Quadrature, trace and complete public caller contracts; mock Green is not BEM evidence."""
import dataclasses
import importlib.metadata
import json
import sys
import types

import numpy as np
import pytest

from anyhydro import make_first_slice_bodies
from anyhydro.capytaine_backend import CapytaineExecutionLimits
from anyhydro.errors import BackendSolveError,CapabilityError,GeometryValidationError
from anyhydro.panel_accuracy_reference import PanelCandidateCancelled
from anyhydro.panel_integration_candidate import triangle_quadrature,imaginary_pole,run_panel_integration_candidate


def test_quadratic_moments_on_oblique_unequal_triangles():
    vertices=np.array([[[1.,2.,-1.],[3.,2.,-1.],[1.,5.,-2.]],
                       [[-2.,0.,-2.],[-1.,2.,-3.],[.1,0.,-4.]]])
    x,w,n=triangle_quadrature(vertices)
    area=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)/2
    total=vertices.sum(axis=1)
    moment=area[:,None,None]*(np.einsum('ivd,ive->ide',vertices,vertices)+np.einsum('id,ie->ide',total,total))/12
    np.testing.assert_allclose(w.sum(axis=1),area,rtol=1e-14)
    np.testing.assert_allclose(np.einsum('iq,iqd->id',w,x),area[:,None]*vertices.mean(axis=1),rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(np.einsum('iq,iqd,iqe->ide',w,x,x),moment,rtol=1e-14,atol=1e-14)
    assert np.all(w>0)
    np.testing.assert_allclose(np.linalg.norm(n,axis=1),1.)
    with pytest.raises(ValueError):x.setflags(write=True)


def test_pole_receiver_derivative_including_coincident_horizontal_nodes():
    tri=np.array([[[0.,0.,-.2],[1.,0.,-.2],[0.,1.,-.2]]])
    source,w,_=triangle_quadrature(tri)
    receivers=np.array([source[0,0],[.1,.3,-.4]])
    _,gradient=imaginary_pole(receivers,source,w,k=.25,water_depth=150.)
    eps=1e-5
    for d in range(3):
        plus,_=imaginary_pole(receivers+eps*np.eye(3)[d],source,w,k=.25,water_depth=150.)
        minus,_=imaginary_pole(receivers-eps*np.eye(3)[d],source,w,k=.25,water_depth=150.)
        np.testing.assert_allclose((plus-minus)/(2*eps),gradient[d],atol=1e-11,rtol=1e-6)


@pytest.mark.parametrize('vertices',[np.zeros((1,3,3)),np.zeros((3,3)),np.full((1,3,3),np.nan)])
def test_bad_triangles_refused(vertices):
    with pytest.raises(GeometryValidationError):triangle_quadrature(vertices)


def call(tmp_path,bodies=None,**changes):
    args=dict(omega=1.5,water_depth=150.,rho=1025.,gravity=9.81,
              limits=CapytaineExecutionLimits(max_total_panels=96,max_finite_omegas=1),artifact_directory=tmp_path/'candidate')
    args.update(changes)
    return run_panel_integration_candidate(bodies or make_first_slice_bodies(subdivisions=1,lid_policy='submerged_0p1m_v1'),**args)


def test_free_surface_refusal_before_backend_import_or_artifact(tmp_path,monkeypatch):
    monkeypatch.setattr(importlib.metadata,'version',lambda _:pytest.fail('backend import before geometry refusal'))
    with pytest.raises(CapabilityError,match='free-surface'):
        call(tmp_path,make_first_slice_bodies(subdivisions=1))
    assert not (tmp_path/'candidate').exists()


def test_stale_mesh_identity_refused_before_backend(tmp_path,monkeypatch):
    monkeypatch.setattr(importlib.metadata,'version',lambda _:pytest.fail('backend import before identity refusal'))
    bodies=make_first_slice_bodies(subdivisions=1,lid_policy='submerged_0p1m_v1')
    bodies[0].mesh.vertices.setflags(write=True)
    bodies[0].mesh.vertices[0,0]+=.001
    with pytest.raises(GeometryValidationError,match='identity'):call(tmp_path,bodies)
    assert not (tmp_path/'candidate').exists()


@pytest.mark.parametrize('change',[dict(omega=True),dict(omega='1.5'),dict(rho=float('inf')),
                                dict(limits=None),dict(limits=CapytaineExecutionLimits(max_total_panels=1))])
def test_effective_controls_refused_before_backend(tmp_path,monkeypatch,change):
    monkeypatch.setattr(importlib.metadata,'version',lambda _:pytest.fail('backend admission too early'))
    with pytest.raises(CapabilityError):call(tmp_path,**change)


def fake_backend(monkeypatch,*,bad_gradient=False,bad_full_batch=False):
    class Mesh:
        def __init__(self,vertices,faces):
            self.vertices=vertices;self.faces=faces;self.nb_faces=len(faces)
            self.quadrature_points=triangle_quadrature(vertices[faces[:,:3]])[:2]
            self.faces_normals=triangle_quadrature(vertices[faces[:,:3]])[2]
            self.faces_centers=vertices[faces[:,:3]].mean(axis=1)
        def with_quadrature(self,method):
            assert method=='Gauss-Legendre 2';return self
        def extract_faces(self,indices):return Mesh(self.vertices,self.faces[indices])
    class Green:
        calls=0
        def __init__(self,**kw):self.exportable_settings=dict(**kw,test_mock=True)
        def evaluate(self,receivers,source,**kwargs):
            Green.calls+=1
            assert kwargs['early_dot_product'] is False
            assert kwargs['diagonal_term_in_double_layer'] is False
            area=source.quadrature_points[1].sum(axis=1)
            direction=np.array([.12,.17,.07])
            s=(1+receivers@direction)[:,None]*area[None,:]
            grad=np.broadcast_to(direction[:,None,None]*area[None,None,:],(3,len(receivers),len(area))).astype(complex).copy()
            if bad_gradient:grad+=1
            if bad_full_batch and Green.calls>2:s[0,0]=np.inf
            return s.astype(complex),grad
    cpt=types.ModuleType('capytaine');cpt.__path__=[];cpt.Mesh=Mesh
    greens=types.ModuleType('capytaine.green_functions');greens.__path__=[]
    module=types.ModuleType('capytaine.green_functions.delhommeau');module.Delhommeau=Green
    for name,value in [('capytaine',cpt),('capytaine.green_functions',greens),('capytaine.green_functions.delhommeau',module)]:
        monkeypatch.setitem(sys.modules,name,value)
    monkeypatch.setattr(importlib.metadata,'version',lambda _:'3.0.0')
    return Green


def test_public_coupled_pipeline_and_failure_evidence(tmp_path,monkeypatch):
    Green=fake_backend(monkeypatch)
    result=call(tmp_path)
    assert result.force_per_velocity.shape==(12,12)
    assert np.linalg.norm(result.force_per_velocity[:6,6:])>0
    assert Green.calls==6
    assert np.max(np.abs(result.mean_residual))<1e-12
    # Finite quadrature trace generally fails pointwise satisfaction: retain it.
    assert np.max(np.abs(result.node_residual))>1e-4
    evidence=result.directional_evidence(np.arange(12)/10)
    assert evidence['force_work_error_w']<1e-8
    assert evidence['mean_residual_is_not_pointwise_acceptance']
    assert evidence['node_residual_pressure_work_abs_bound_w']>0
    np.testing.assert_allclose(result.added_mass,result.force_per_velocity.imag/1.5)
    with pytest.raises(ValueError):result.sources.setflags(write=True)
    status=json.loads((tmp_path/'candidate'/'status.json').read_text())
    assert status['state']=='completed_unqualified'
    assert status['numerical_systems']==1
    with pytest.raises(FileExistsError):call(tmp_path)
    assert Green.calls==6


def test_real_derivative_failure_stops_all_full_batches(tmp_path,monkeypatch):
    Green=fake_backend(monkeypatch,bad_gradient=True)
    with pytest.raises(BackendSolveError,match='real Green'):call(tmp_path)
    assert Green.calls==2
    status=json.loads((tmp_path/'candidate'/'status.json').read_text())
    assert status['state']=='failed' and status['numerical_systems']==0
    assert (tmp_path/'candidate'/'real-derivative.npz').is_file()
    assert not (tmp_path/'candidate'/'operator-0.npz').exists()
    assert (tmp_path/'candidate'/'micro-gradient-native.npz').is_file()


def test_native_acquisition_saved_before_nonfinite_refusal(tmp_path,monkeypatch):
    Green=fake_backend(monkeypatch,bad_full_batch=True)
    with pytest.raises(BackendSolveError,match='nonfinite native'):call(tmp_path)
    assert Green.calls==3
    raw=np.load(tmp_path/'candidate'/'native-operator-0.npz',allow_pickle=False)
    assert not np.isfinite(raw['s'][0,0])
    assert raw['gradient'].shape[0]==3
    assert not (tmp_path/'candidate'/'operator-0.npz').exists()
    assert not (tmp_path/'candidate'/'radiation.npz').exists()


def test_force_moments_against_direct_traction_and_reference_shift(tmp_path,monkeypatch):
    fake_backend(monkeypatch)
    bodies=make_first_slice_bodies(subdivisions=1,lid_policy='submerged_0p1m_v1')
    result=call(tmp_path/'first',bodies)
    inputs=np.load(tmp_path/'first'/'candidate'/'input.npz',allow_pickle=False)
    expected=np.zeros((12,12),complex)
    triangles=inputs['vertices'][inputs['faces']]
    for panel in np.flatnonzero(result.hull_mask):
        triangle=triangles[panel]
        normal=np.cross(triangle[1]-triangle[0],triangle[2]-triangle[0]);normal/=np.linalg.norm(normal)
        body=0 if triangle[:,0].mean()<0 else 1
        for q in range(4):
            arm=result.source_points[panel,q]-bodies[body].rotation_center
            for col in range(12):
                traction=-result.pressure[panel,q,col]*normal*result.weights[panel,q]
                expected[body*6:body*6+3,col]+=traction
                expected[body*6+3:body*6+6,col]+=np.cross(arm,traction)
    np.testing.assert_allclose(result.force_per_velocity,expected,rtol=1e-12,atol=1e-9)
    delta=np.array([.23,-.31,.17])
    changed=(dataclasses.replace(bodies[0],rotation_center=tuple(np.array(bodies[0].rotation_center)+delta)),bodies[1])
    shifted=call(tmp_path/'second',changed)
    translations=[0,1,2,6,7,8]
    np.testing.assert_allclose(shifted.force_per_velocity[6:,translations],result.force_per_velocity[6:,translations],rtol=1e-12,atol=1e-9)
    for col in translations:
        np.testing.assert_allclose(shifted.force_per_velocity[3:6,col],result.force_per_velocity[3:6,col]-np.cross(delta,result.force_per_velocity[:3,col]),rtol=1e-12,atol=1e-9)


def test_wrong_version_preserves_input_and_refuses_green(tmp_path,monkeypatch):
    monkeypatch.setattr(importlib.metadata,'version',lambda _:'3.0.1')
    with pytest.raises(CapabilityError,match='exact tested'):call(tmp_path)
    assert (tmp_path/'candidate'/'input.npz').is_file()
    assert json.loads((tmp_path/'candidate'/'status.json').read_text())['new_green_evaluations']==0


def test_cancellation_between_full_batches_preserves_partial_artifacts(tmp_path, monkeypatch):
    fake_backend(monkeypatch)
    output = tmp_path/'candidate'
    with pytest.raises(PanelCandidateCancelled):
        call(tmp_path, cancel_requested=lambda: (output/'operator-0.npz').exists())
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'cancelled' and status['numerical_systems'] == 0
    assert (output/'operator-0.npz').is_file() and not (output/'operator-1.npz').exists()
    assert not (output/'radiation.npz').exists()


def test_cancellation_before_solve_keeps_all_batches(tmp_path, monkeypatch):
    fake_backend(monkeypatch)
    output = tmp_path/'candidate'
    with pytest.raises(PanelCandidateCancelled):
        call(tmp_path, cancel_requested=lambda: (output/'operator-3.npz').exists())
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'cancelled' and status['numerical_systems'] == 0
    for qi in range(4):
        assert (output/f'operator-{qi}.npz').is_file()
    assert not (output/'radiation.npz').exists()


def test_cancellation_before_completion_never_marks_completed(tmp_path, monkeypatch):
    fake_backend(monkeypatch)
    output = tmp_path/'candidate'
    with pytest.raises(PanelCandidateCancelled):
        call(tmp_path, cancel_requested=lambda: (output/'radiation.npz').exists())
    status = json.loads((output/'status.json').read_text())
    assert status['state'] == 'cancelled' and status['numerical_systems'] == 1
    assert (output/'radiation.npz').is_file()
    assert status['state'] != 'completed_unqualified'
