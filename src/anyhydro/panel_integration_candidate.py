# SPDX-License-Identifier: MPL-2.0
"""Experimental, versioned panel integration. Never a canonical hydro dataset.

Capytaine is optional and imported only by ``run_panel_integration_candidate``.
This route tests panel-mean Neumann equations, not pointwise boundary satisfaction.
Raw node residuals and pressure work are authoritative diagnostic outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import numbers
import time
from pathlib import Path

import numpy as np

from .capytaine_backend import CapytaineExecutionLimits
from .errors import BackendSolveError, CapabilityError, GeometryValidationError
from .hashing import mesh_hash
from .records import HydroBody
from .artifact_bound_panel_profile import ArtifactBoundPanelProfile
from .panel_accuracy_reference import (FrozenUntabulatedProfile, FrozenTabulatedProfile, PanelCandidateCancelled,
                                      check_cancel, operator_probe)
from .saved_tabulation_profile import FrozenSavedTabulatedProfile

PANEL_INTEGRATION_VERSION = 'triangle-mean-neumann-pole-gauss2-v1'
QUADRATURE_METHOD = 'Gauss-Legendre 2'


def _immutable(a):
    a = np.ascontiguousarray(a)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


def triangle_quadrature(vertices):
    """Positive tensor Gauss rule on the collapsed [a,b,c,c] map.

    Physical weights include area. Exact polynomial moments through degree two;
    Green/pressure integrals are approximations, not exact triangle integrals.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 3 or v.shape[1:] != (3, 3) or not np.all(np.isfinite(v)):
        raise GeometryValidationError('triangles must be finite (panels,3,3)')
    cross = np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0])
    area = np.linalg.norm(cross, axis=1)/2
    if np.any(area <= 0):
        raise GeometryValidationError('degenerate source triangle')
    q = []
    w = []
    # u,v in [0,1]; x=u*v*a+u*(1-v)*b+(1-u)*c, dA=2*A*u du dv.
    for u in ((1+1/math.sqrt(3))/2, (1-1/math.sqrt(3))/2):
        for t in ((1+1/math.sqrt(3))/2, (1-1/math.sqrt(3))/2):
            q.append(u*t*v[:, 0]+u*(1-t)*v[:, 1]+(1-u)*v[:, 2])
            w.append(area*u/2)
    points = np.stack(q, axis=1)
    weights = np.stack(w, axis=1)
    return _immutable(points), _immutable(weights), _immutable(cross/(2*area[:, None]))


def imaginary_pole(receivers, source_points, source_weights, *, k, water_depth):
    """Source-quadrature integrated ImG and receiver gradient, same normalization.

    Gradient shape is (3,receivers,source panels). There is no singular jump in
    this smooth imaginary term. Caller handles the real normal-trace jump.
    """
    from scipy.special import j0, j1
    x = np.asarray(receivers, dtype=float)
    y = np.asarray(source_points, dtype=float)
    w = np.asarray(source_weights, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 3 or y.ndim != 3 or y.shape[2] != 3
            or w.shape != y.shape[:2] or not np.all(np.isfinite(x))
            or not np.all(np.isfinite(y)) or not np.all(np.isfinite(w)) or np.any(w <= 0)
            or not math.isfinite(k) or k <= 0 or not math.isfinite(water_depth) or water_depth <= 0):
        raise GeometryValidationError('invalid propagating-pole geometry/parameters')
    kh = k*water_depth
    D = math.tanh(kh)+kh*(1-math.tanh(kh)**2)
    denominator = 1+math.exp(-2*kh)
    def mode(z):
        e = np.exp(k*z)
        f = np.exp(-k*(z+2*water_depth))
        return (e+f)/denominator, k*(e-f)/denominator
    Z, dZ = mode(x[:, 2])
    out = np.zeros((len(x), len(y)))
    grad = np.zeros((3, len(x), len(y)))
    factor = -k/(2*D)
    for qi in range(y.shape[1]):
        yz, _ = mode(y[:, qi, 2])
        delta = x[:, None, :2]-y[None, :, qi, :2]
        r = np.linalg.norm(delta, axis=2)
        J = j0(k*r)
        weighted = factor*yz[None, :]*w[None, :, qi]
        out += weighted*Z[:, None]*J
        horizontal = np.divide(j1(k*r), r, out=np.zeros_like(r), where=r > 0)
        for direction in (0, 1):
            grad[direction] += -k*weighted*Z[:, None]*horizontal*delta[:, :, direction]
        grad[2] += weighted*dZ[:, None]*J
    return out, grad


def _rigid_trace(points, normals, owner, centers):
    shape = np.zeros((*points.shape[:2], 6*len(centers)))
    for body, center in enumerate(centers):
        offsets = points-np.asarray(center)
        own = owner == body
        for dof in range(6):
            axis = np.eye(3)[dof % 3]
            velocity = np.broadcast_to(axis, offsets.shape) if dof < 3 else np.cross(axis, offsets)
            shape[own, :, 6*body+dof] = np.einsum('iqj,ij->iq', velocity[own], normals[own])
    return shape


@dataclass(frozen=True)
class PanelRadiationCandidate:
    """Raw immutable velocity-column result; no qualification/canonical admission."""
    case_hash: str
    dofs: tuple[str, ...]
    omega: float
    rho: float
    water_depth: float
    k: float
    source_points: np.ndarray
    weights: np.ndarray
    hull_mask: np.ndarray
    rigid_trace: np.ndarray
    sources: np.ndarray
    pressure: np.ndarray
    node_residual: np.ndarray
    mean_residual: np.ndarray
    force_per_velocity: np.ndarray
    added_mass: np.ndarray
    radiation_damping: np.ndarray

    def __post_init__(self):
        count, nq = np.asarray(self.weights).shape
        nd = len(self.dofs)
        expected = dict(source_points=(count,nq,3),hull_mask=(count,),rigid_trace=(count,nq,nd),
            sources=(count,nd),pressure=(count,nq,nd),node_residual=(count,nq,nd),
            mean_residual=(count,nd),force_per_velocity=(nd,nd),added_mass=(nd,nd),radiation_damping=(nd,nd))
        if not nd or len(set(self.dofs)) != nd or nq != 4 or count < 1 or np.any(self.weights <= 0):
            raise GeometryValidationError('invalid candidate inventory')
        for name, shape in expected.items():
            if np.asarray(getattr(self,name)).shape != shape:
                raise GeometryValidationError(f'invalid candidate {name} shape')
        for name in ('source_points', 'weights', 'hull_mask', 'rigid_trace', 'sources',
                     'pressure', 'node_residual', 'mean_residual', 'force_per_velocity',
                     'added_mass', 'radiation_damping'):
            values = np.asarray(getattr(self,name))
            if not np.all(np.isfinite(values)):
                raise BackendSolveError(f'nonfinite candidate {name}')
            object.__setattr__(self, name, _immutable(values))

    def directional_evidence(self, rate, *, angular_counts=(64, 128, 256)):
        """Mechanical work, outgoing waves and actual node residual work.

        Passing mean residual or force/work algebra does not admit this source.
        Source-pole far equality is constructed and is not independent evidence.
        """
        rate = np.asarray(rate, dtype=complex)
        if rate.shape != (len(self.dofs),) or not np.all(np.isfinite(rate)):
            raise CapabilityError('rate must contain one finite peak velocity per DOF')
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 16 or n > 4096 for n in angular_counts):
            raise CapabilityError('angular counts must be integers in [16,4096]')
        if not angular_counts:
            raise CapabilityError('angular inventory must not be empty')
        p = self.pressure@rate
        g = self.rigid_trace@rate
        residual = self.node_residual@rate
        hull = self.hull_mask
        near = -.5*float(np.real(np.vdot(rate, self.force_per_velocity@rate)))
        work = .5*float(np.real(np.sum(self.weights[hull]*p[hull]*g[hull].conj())))
        powers = []
        sigma = self.sources@rate
        kh = self.k*self.water_depth
        D = math.tanh(kh)+kh*(1-math.tanh(kh)**2)
        Z = (np.exp(self.k*self.source_points[:, :, 2])+np.exp(-self.k*(self.source_points[:, :, 2]+2*self.water_depth)))/(1+math.exp(-2*kh))
        coeff = sigma[:, None]*self.weights*Z
        for count in angular_counts:
            theta = 2*np.pi*np.arange(count)/count
            H = np.zeros(count, complex)
            for qi in range(self.weights.shape[1]):
                phase = np.exp(-1j*self.k*(np.cos(theta)[:, None]*self.source_points[None, :, qi, 0]+np.sin(theta)[:, None]*self.source_points[None, :, qi, 1]))
                H += phase@coeff[:, qi]/(4*np.pi)
            powers.append(float(2*np.pi*self.rho*self.omega*self.k/D*(2*np.pi/count)*np.sum(np.abs(H)**2)))
        signed = .5*np.real(self.weights*p*residual.conj())
        absolute = .5*self.weights*np.abs(p)*np.abs(residual)
        return dict(near_mean_power_w=near,mechanical_node_work_w=work,
            force_work_error_w=abs(near-work),far_mean_powers_w=powers,
            mean_boundary_residual_max=float(np.max(np.abs(self.mean_residual@rate))),
            node_boundary_residual_max=float(np.max(np.abs(residual))),
            node_residual_pressure_work_w=float(np.sum(signed)),
            hull_node_residual_pressure_work_w=float(np.sum(signed[hull])),
            node_residual_pressure_work_abs_bound_w=float(np.sum(absolute)),
            hull_node_residual_pressure_work_abs_bound_w=float(np.sum(absolute[hull])),
            mean_residual_is_not_pointwise_acceptance=True,qualification=False)


def run_panel_integration_candidate(bodies, *, omega, water_depth, rho, gravity,
                                    limits, artifact_directory, accuracy_profile=None,
                                    operator_probe_receivers=None, cancel_requested=None):
    """Run exactly one finite radiation candidate through public Green interfaces.

    Four full Green batches plus two one-triangle real-derivative micro calls.
    Explicit execution limits are required; external job supervision owns timeout.
    Refuses sheets/free-surface panels before importing or evaluating Capytaine.
    Saves inputs, derivative failure and every acquired batch before solving.
    """
    started = time.perf_counter()
    if accuracy_profile is not None and type(accuracy_profile) not in (FrozenUntabulatedProfile, FrozenTabulatedProfile,
                                                                       FrozenSavedTabulatedProfile,
                                                                       ArtifactBoundPanelProfile):
        raise CapabilityError('explicit named frozen untabulated, tabulated, saved-tabulated or artifact-bound profile required')
    if operator_probe_receivers is not None:
        if (accuracy_profile is None or isinstance(operator_probe_receivers, bool)
                or not isinstance(operator_probe_receivers, numbers.Integral)
                or not 1 <= operator_probe_receivers <= 16):
            raise CapabilityError('operator probe requires profile and integer1..16 receiver rows')
        operator_probe_receivers = int(operator_probe_receivers)
    if cancel_requested is not None and not callable(cancel_requested):
        raise CapabilityError('cancellation requires callable')
    check_cancel(cancel_requested)
    bodies = tuple(bodies)
    if not bodies or not all(isinstance(b, HydroBody) for b in bodies):
        raise CapabilityError('candidate requires nonempty HydroBody inventory')
    if len({b.body_id for b in bodies}) != len(bodies):
        raise CapabilityError('duplicate body identity')
    if not isinstance(limits, CapytaineExecutionLimits):
        raise CapabilityError('explicit CapytaineExecutionLimits required')
    for name, value in [('omega', omega), ('water_depth', water_depth), ('rho', rho), ('gravity', gravity)]:
        if isinstance(value, bool) or not isinstance(value,numbers.Real) or not math.isfinite(float(value)) or float(value) <= 0:
            raise CapabilityError(f'{name} must be positive finite SI scalar')
    omega, water_depth, rho, gravity = map(float, (omega, water_depth, rho, gravity))
    vertices = []; faces = []; owner = []; hull = []; inventory = []
    for role in ('mesh', 'lid_mesh'):
        for bi, body in enumerate(bodies):
            if body.attached_surfaces:
                raise CapabilityError('candidate does not support attached sheets')
            panel = getattr(body, role)
            if panel is None:
                continue
            if ((role == 'mesh' and not panel.outward_normals_verified)
                    or (role == 'lid_mesh' and panel.outward_normals_verified)
                    or np.any(panel.face_sizes != 3)
                    or panel.surface is not None and (panel.surface.surface_kind == 'thin_sheet'
                    or panel.surface.primary_hydrodynamics != 'potential_flow')):
                raise GeometryValidationError('candidate requires verified potential-flow triangles')
            if np.any(panel.vertices[:, 2] > 0) or np.any(panel.vertices[:, 2] <= -water_depth):
                raise GeometryValidationError('source vertices must be in water, above seabed')
            if mesh_hash(panel.body_id,panel.vertices,panel.faces,panel.face_sizes,surface=panel.surface) != panel.mesh_hash:
                raise GeometryValidationError('input mesh bytes no longer match their immutable record identity')
            offset = sum(len(v) for v in vertices)
            vertices.append(panel.vertices); faces.append(panel.faces+offset)
            owner.extend([bi]*len(panel.faces)); hull.extend([role == 'mesh']*len(panel.faces))
            inventory.append(dict(body_id=body.body_id,role=role,mesh_hash=panel.mesh_hash,source_geometry_refs=panel.source_geometry_refs))
    vertices = np.concatenate(vertices); triangles = np.concatenate(faces); owner = np.array(owner); hull = np.array(hull)
    if len(triangles) > limits.max_total_panels:
        raise CapabilityError('candidate exceeds explicit panel execution limit')
    points, weights, normals = triangle_quadrature(vertices[triangles])
    if np.any(points[:, :, 2] >= -1e-6):
        raise CapabilityError('quadrature candidate refuses free-surface source panels; no offset/clamp')
    if np.any(normals[~hull, 2] > -1+1e-12):
        raise GeometryValidationError('auxiliary lid triangles must point down')
    areas = weights.sum(axis=1)
    refs = [b.rotation_center for b in bodies]
    trace = _rigid_trace(points,normals,owner,refs);trace[~hull] = 0
    mean_rhs = np.einsum('iq,iqd->id', weights/areas[:, None], trace)
    identity = dict(version=PANEL_INTEGRATION_VERSION,capytaine='3.0.0',quadrature=QUADRATURE_METHOD,
        inventory=inventory,body_ids=[b.body_id for b in bodies],rotation_centers=refs,
        omega=omega,water_depth=water_depth,rho=rho,gravity=gravity,
        execution_limits=limits.as_dict(),mean_neumann_testing=True,source_admission=False,
        module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    identity['snapshot_arrays'] = {name:dict(shape=values.shape,dtype=str(values.dtype),sha256=hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest())
        for name,values in dict(vertices=vertices,faces=triangles,owner=owner,hull_mask=hull,points=points,weights=weights,normals=normals,rigid_trace=trace).items()}
    if accuracy_profile is not None:
        accuracy_profile.bind_case(identity)
        identity['accuracy_profile_sha256'] = hashlib.sha256(accuracy_profile.manifest_json.encode()).hexdigest()
        identity['accuracy_implementation_sha256'] = hashlib.sha256(Path(__file__).with_name('panel_accuracy_reference.py').read_bytes()).hexdigest()
        identity['operator_probe_receivers'] = operator_probe_receivers
    case_hash = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
    directory = Path(artifact_directory);directory.mkdir(parents=True,exist_ok=False)
    (directory/'case.json').write_text(json.dumps(dict(case_hash=case_hash,**identity),indent=2),encoding='utf-8')
    np.savez(directory/'input.npz',vertices=vertices,faces=triangles,points=points,weights=weights,normals=normals,owner=owner,hull_mask=hull,rigid_trace=trace)
    runtime = dict(new_green_evaluations=0,numerical_systems=0)
    if accuracy_profile is not None:
        accuracy_profile.preserve(directory)
    try:
        check_cancel(cancel_requested)
        if importlib.metadata.version('capytaine') != '3.0.0':
            raise CapabilityError('candidate requires exact tested Capytaine3.0.0')
        import capytaine as cpt
        from capytaine.green_functions.delhommeau import Delhommeau
        from scipy.optimize import brentq
        from threadpoolctl import threadpool_limits,threadpool_info
        k = brentq(lambda k:k*math.tanh(k*water_depth)-omega**2/gravity,
                   0.,omega**2/gravity+1/water_depth,xtol=1e-15)
        mesh = cpt.Mesh(vertices=vertices,faces=np.column_stack((triangles,triangles[:, -1])))
        mesh = mesh.with_quadrature(QUADRATURE_METHOD)
        native_points,native_weights = mesh.quadrature_points
        np.testing.assert_allclose(native_points,points,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(native_weights,weights,rtol=1e-13,atol=1e-15)
        tri = vertices[triangles]
        exact_area = np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        exact_first = exact_area[:,None]*tri.mean(axis=1)
        vertex_sum = tri.sum(axis=1)
        exact_second = exact_area[:,None,None]*(np.einsum('ivd,ive->ide',tri,tri)+np.einsum('id,ie->ide',vertex_sum,vertex_sum))/12
        np.testing.assert_allclose(native_weights.sum(axis=1),exact_area,rtol=1e-13,atol=1e-15)
        np.testing.assert_allclose(np.einsum('iq,iqd->id',native_weights,native_points),exact_first,rtol=1e-13,atol=1e-14)
        np.testing.assert_allclose(np.einsum('iq,iqd,iqe->ide',native_weights,native_points,native_points),exact_second,rtol=1e-13,atol=1e-14)
        np.testing.assert_allclose(mesh.faces_normals,normals,rtol=1e-13,atol=1e-14)
        if accuracy_profile is None:
            gf = Delhommeau(gf_singularities='low_freq',floating_point_precision='float64')
        else:
            if type(accuracy_profile) is FrozenTabulatedProfile:
                runtime['table_construction_attempts'] = 1
                runtime['intended_table_settings'] = accuracy_profile.constructor_options()
                (directory/'status.json').write_text(json.dumps(dict(state='finer_table_constructor_running',**runtime),indent=2),encoding='utf-8')
            elif type(accuracy_profile) is FrozenSavedTabulatedProfile:
                runtime['saved_table_cache_attempts'] = 1
                runtime['intended_saved_table_settings'] = accuracy_profile.constructor_options()
                (directory/'status.json').write_text(json.dumps(dict(state='saved_table_cache_preparation_running',**runtime),indent=2),encoding='utf-8')
            preparation_start = time.perf_counter()
            try:
                with threadpool_limits(limits=1):
                    runtime['effective_constructor_threadpools'] = threadpool_info()
                    if type(accuracy_profile) in (FrozenTabulatedProfile, FrozenSavedTabulatedProfile):
                        gf = accuracy_profile.green_function(Delhommeau,cpt,k=k,water_depth=water_depth,
                                                              artifact_directory=directory,runtime=runtime)
                    else:
                        gf = accuracy_profile.green_function(Delhommeau,cpt,k=k,water_depth=water_depth)
            finally:
                runtime['evaluator_preparation_wall_seconds'] = time.perf_counter()-preparation_start
            if type(accuracy_profile) is FrozenTabulatedProfile:
                runtime['new_table_constructions'] = 1
            elif type(accuracy_profile) is FrozenSavedTabulatedProfile:
                runtime['saved_table_cache_reuse_confirmed'] = runtime.get('saved_table_cache_loads') == 1
            accuracy_profile.preserve_evaluator(gf,directory,runtime)
            metadata = accuracy_profile.evaluator_case_metadata(gf)
            if metadata is not None:
                identity['planned_case_hash'] = case_hash
                # Old tabulated/saved profiles keep the historical actual_tabulation
                # schema; the artifact-bound profile records its evaluator identity
                # under its own explicit key.
                identity[getattr(accuracy_profile,'CASE_METADATA_KEY','actual_tabulation')] = metadata
                case_hash = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',', ':')).encode()).hexdigest()
                (directory/'case.json').write_text(json.dumps(dict(case_hash=case_hash,**identity),indent=2),encoding='utf-8')
        params = dict(water_depth=water_depth,wavenumber=k,adjoint_double_layer=True,
                      early_dot_product=False,diagonal_term_in_double_layer=False)
        runtime.update(effective_green_settings=gf.exportable_settings)
        runtime['preparation_wall_seconds'] = time.perf_counter()-started
        with threadpool_limits(limits=1):
            runtime['effective_threadpools'] = threadpool_info()
            micro_mesh = mesh.extract_faces([0]).with_quadrature(QUADRATURE_METHOD)
            receiver = mesh.faces_centers[:1].copy();receiver[0, :2] += [.37,.19]
            receiver[0, 2] -= min(.17,(water_depth+receiver[0, 2])/4)
            check_cancel(cancel_requested)
            if accuracy_profile is not None:
                accuracy_profile.check_micro(gf,micro_mesh,params,directory,runtime)
                check_cancel(cancel_requested)
                if operator_probe_receivers is not None:
                    return operator_probe(gf,mesh,points,params,operator_probe_receivers,
                                          directory,runtime,case_hash,cancel_requested,accuracy_profile)
            else:
                step = 1e-5
                shifted = np.concatenate([receiver+step*np.eye(3)[d] for d in range(3)]+[receiver-step*np.eye(3)[d] for d in range(3)])
                runtime['new_green_evaluations'] += 1
                (directory/'status.json').write_text(json.dumps(dict(state='micro_gradient_running',**runtime),indent=2),encoding='utf-8')
                micro_s, gradient = gf.evaluate(receiver,micro_mesh,**params)
                np.savez(directory/'micro-gradient-native.npz',receiver=receiver,s=micro_s,gradient=gradient)
                runtime['new_green_evaluations'] += 1
                (directory/'status.json').write_text(json.dumps(dict(state='micro_difference_running',**runtime),indent=2),encoding='utf-8')
                values,_ = gf.evaluate(shifted,micro_mesh,**params)
                np.savez(directory/'micro-difference-native.npz',shifted=shifted,s=values)
                fd = (values[:3, 0].real-values[3:, 0].real)/(2*step)
                derivative_error = float(np.max(np.abs(fd-gradient[:, 0, 0].real)))
                derivative_limit = 1e-8+1e-6*float(np.max(np.abs(gradient[:, 0, 0].real)))
                audit = dict(error=derivative_error,limit=derivative_limit,passed=derivative_error<=derivative_limit)
                np.savez(directory/'real-derivative.npz',receiver=receiver,shifted=shifted,gradient=gradient,values=values,fd=fd)
                (directory/'real-derivative.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
                if not audit['passed']:
                    raise BackendSolveError('real Green receiver gradient fails independent finite-difference consistency')
            singles = [];trace_nodes = []
            for qi in range(4):
                check_cancel(cancel_requested)
                if accuracy_profile is not None:
                    accuracy_profile.record_branches(gf,points[:, qi],mesh,params,directory,f'operator_{qi}')
                runtime['new_green_evaluations'] += 1
                (directory/'status.json').write_text(json.dumps(dict(state=f'operator_{qi}_running',**runtime),indent=2),encoding='utf-8')
                s,grad = gf.evaluate(points[:, qi],mesh,**params)
                np.savez(directory/f'native-operator-{qi}.npz',s=s,gradient=grad)
                if accuracy_profile is not None:
                    accuracy_profile.verify_evaluator(gf,directory,f'after_operator_{qi}')
                if not np.all(np.isfinite(s)) or not np.all(np.isfinite(grad)):
                    raise BackendSolveError('nonfinite native Green batch; acquired evidence preserved')
                pole,pole_grad = imaginary_pole(points[:, qi],points,weights,k=k,water_depth=water_depth)
                s = s.real+1j*pole
                normal_grad = np.einsum('dim,id->im',grad.real,normals)+1j*np.einsum('dim,id->im',pole_grad,normals)
                # All candidate source nodes are strictly submerged, so .5 jump.
                # Each row belongs to its source PANEL, not flattened-node index.
                normal_grad[np.arange(len(areas)),np.arange(len(areas))] += .5
                if not np.all(np.isfinite(s)) or not np.all(np.isfinite(normal_grad)):
                    raise BackendSolveError('nonfinite integrated Green batch')
                np.savez(directory/f'operator-{qi}.npz',s=s,k=normal_grad)
                singles.append(s);trace_nodes.append(normal_grad)
            check_cancel(cancel_requested)
            singles = np.stack(singles,axis=1);trace_nodes = np.stack(trace_nodes,axis=1)
            mean_k = np.einsum('iq,iqj->ij',weights/areas[:, None],trace_nodes)
            runtime['numerical_systems'] += 1
            (directory/'status.json').write_text(json.dumps(dict(state='radiation_system_running',**runtime),indent=2),encoding='utf-8')
            sigma = np.linalg.solve(mean_k,mean_rhs)
            phi = np.einsum('iqj,jd->iqd',singles,sigma);pressure = 1j*rho*omega*phi
            residual = np.einsum('iqj,jd->iqd',trace_nodes,sigma)-trace
            mean_residual = mean_k@sigma-mean_rhs
            force = -np.einsum('iq,iqd,iqe->de',weights[hull],trace[hull],pressure[hull])
            A = force.imag/omega;B = -force.real
        dofs = tuple(key for b in bodies for key in b.all_dof_keys())
        np.savez(directory/'radiation.npz',sources=sigma,pressure=pressure,node_residual=residual,
                 mean_residual=mean_residual,force_per_velocity=force,added_mass=A,radiation_damping=B,dofs=np.array(dofs))
        result = PanelRadiationCandidate(case_hash,dofs,omega,rho,water_depth,k,points,weights,hull,
            trace,sigma,pressure,residual,mean_residual,force,A,B)
        check_cancel(cancel_requested)
        (directory/'status.json').write_text(json.dumps(dict(state='completed_unqualified',**runtime),indent=2),encoding='utf-8')
        return result
    except Exception as exc:
        runtime = locals().get('runtime',dict(new_green_evaluations=0,numerical_systems=0))
        (directory/'status.json').write_text(json.dumps(dict(state='cancelled' if isinstance(exc,PanelCandidateCancelled) else 'failed',error_type=type(exc).__name__,error=str(exc),**runtime),indent=2),encoding='utf-8')
        if isinstance(exc,(BackendSolveError,CapabilityError,GeometryValidationError)):
            raise
        raise BackendSolveError(f'panel integration candidate failed: {exc}') from exc
