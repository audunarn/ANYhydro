# SPDX-License-Identifier: MPL-2.0
"""Conservative WORLD field-point admission for the pinned native solid solver."""
import math
import numpy as np
from .errors import HydroError

__all__=['HamsFieldPointError','normalize_field_points','admit_field_points']


class HamsFieldPointError(HydroError):pass


def normalize_field_points(points):
    if not isinstance(points,(tuple,list,np.ndarray)):
        raise HamsFieldPointError('Explicit ordered field-point array required')
    if isinstance(points,np.ndarray) and points.ndim==0:raise HamsFieldPointError('Ordered field points required')
    if len(points)==0:return ()
    raw=np.asarray(points,dtype=object)
    if raw.ndim!=2 or raw.shape[1]!=3 or any(isinstance(x,(bool,np.bool_)) for x in raw.flat):
        raise HamsFieldPointError('Field points require three finite numbers per row')
    try:values=np.asarray(points,dtype=float)
    except (TypeError,ValueError,OverflowError) as exc:raise HamsFieldPointError('Finite field points required') from exc
    if not np.isfinite(values).all():raise HamsFieldPointError('Finite field points required')
    result=tuple(tuple(float(x) for x in row) for row in values)
    if len(set(result))!=len(result):raise HamsFieldPointError('Duplicate field points refused')
    return result


def _cross(a,b):
    return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])


def _dot(a,b):return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]
def _sub(a,b):return tuple(a[i]-b[i] for i in range(3))


def _native_onbody(point,mesh):
    # Mirrors NormalProcess CalPanelArea_Normal/CalPanelSz and the NSYS=1
    # PressureElevationMulti ONBODY test. Roundoff-near comparisons also refuse.
    for face,size in zip(mesh.faces,mesh.face_sizes):
        if size not in (3,4):raise HamsFieldPointError('Pinned triangular/quadrilateral field geometry required')
        vertices=[tuple(float(v) for v in mesh.vertices[int(i)]) for i in face[:size]]
        if size==3:un=_cross(_sub(vertices[0],vertices[1]),_sub(vertices[2],vertices[1]))
        else:un=_cross(_sub(vertices[3],vertices[1]),_sub(vertices[2],vertices[0]))
        length=math.sqrt(_dot(un,un))
        if not math.isfinite(length) or length==0:raise HamsFieldPointError('Degenerate native field panel refused')
        normal=tuple(v/length for v in un)
        edges=[_sub(vertices[(i+1)%size],vertices[i]) for i in range(size)]
        pnsz=max(math.sqrt(_dot(e,e)) for e in edges)
        tolp=max(1e-6,1e-3*pnsz);tola=1e-3*pnsz*pnsz
        scale=max(1.,*(abs(x) for v in vertices for x in v),*(abs(x) for x in point))
        guard_plane=128*np.finfo(float).eps*scale
        guard_area=512*np.finfo(float).eps*scale*scale
        if not math.isfinite(guard_area):raise HamsFieldPointError('Field geometry representation range refused')
        if abs(_dot(_sub(point,vertices[0]),normal))>tolp+guard_plane:continue
        signs=[_dot(_cross(edge,_sub(point,vertex)),normal) for edge,vertex in zip(edges,vertices)]
        if min(signs)>=-tola-guard_area or max(signs)<=tola+guard_area:return True
    return False


def admit_field_points(bodies,request,points):
    """Refuse interior/AABB, near-plane, below-bed and native tolerant ONBODY points.

    AABB exclusion is deliberately conservative, including cavities. The exact
    native polygon/plane predicate adds its source tolerances and conservative
    floating-comparison guards; they are refusal margins, not solver tolerances.
    """
    points=normalize_field_points(points)
    bodies=tuple(bodies)
    if points and len(bodies)<2:raise HamsFieldPointError('Pinned combined fields require multibody input')
    for point in points:
        z=point[2]
        if z!=0. and z>=-1e-6:raise HamsFieldPointError('Exact z=0 elevation or z<-1e-6 pressure required')
        if math.isfinite(request.water_depth) and z<=-request.water_depth:
            raise HamsFieldPointError('Field point must be above the bed')
        for body in bodies:
            if body.attached_surfaces:raise HamsFieldPointError('Field profile does not admit attached sheets')
            for mesh in (body.mesh,body.lid_mesh):
                if mesh is None:continue
                vertices=mesh.vertices
                if all(float(vertices[:,i].min())<=point[i]<=float(vertices[:,i].max()) for i in range(3)):
                    raise HamsFieldPointError('Field point must lie outside every closed hull/lid AABB')
                if _native_onbody(point,mesh):
                    raise HamsFieldPointError('Native tolerant ONBODY or comparison-boundary field point refused')
    return points
