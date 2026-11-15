"""Explicit higher-order source integration for the pinned Capytaine 3.0 mesh.

The tensor rule integrates the bilinear face map and its surface Jacobian.
It changes neither panels nor receiver collocation points.  Rule names include
the mapping revision because they are persisted in family/cache identities.
"""
from __future__ import annotations

import numpy as np

RULES = {'anyhydro-bilinear-gauss-3-v1': 3, 'anyhydro-bilinear-gauss-4-v1': 4}


def bilinear_quadrature(vertices, faces, order):
    """Return physical points and area weights, including collapsed quad triangles."""
    if isinstance(order, bool) or order not in (3, 4):
        raise ValueError('supported bilinear quadrature orders are 3 and 4')
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces)
    if (v.ndim != 2 or v.shape[1] != 3 or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1] != 4 or f.dtype.kind not in 'iu'
            or not len(f) or f.min() < 0 or f.max() >= len(v)):
        raise ValueError('finite vertices and integer quad/degenerate-triangle faces required')
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x, y = np.meshgrid(nodes, nodes, indexing='ij')
    x, y = x.ravel(), y.ravel()
    w = np.multiply.outer(weights, weights).ravel()
    N = np.column_stack([(1+x)*(1+y), (1+x)*(1-y), (1-x)*(1-y), (1-x)*(1+y)]) / 4
    Nx = np.column_stack([1+y, 1-y, -(1-y), -(1+y)]) / 4
    Ny = np.column_stack([1+x, -(1+x), -(1-x), 1-x]) / 4
    corners = v[f]
    # The norm of a Jacobian alone hides folded maps. Its normal projection
    # is affine in the two parameters, so corner signs bound the entire face.
    cx = (corners[:,0]+corners[:,1]-corners[:,2]-corners[:,3])/4
    cy = (corners[:,0]-corners[:,1]-corners[:,2]+corners[:,3])/4
    reference = np.cross(cx, cy)
    reference_sq = np.sum(reference*reference, axis=1)
    corner_x = np.array([1.,1.,-1.,-1.])
    corner_y = np.array([1.,-1.,-1.,1.])
    Cx = np.column_stack([1+corner_y,1-corner_y,-(1-corner_y),-(1+corner_y)])/4
    Cy = np.column_stack([1+corner_x,-(1+corner_x),-(1-corner_x),1-corner_x])/4
    jac = np.cross(np.einsum('qa,fac->fqc',Cx,corners),np.einsum('qa,fac->fqc',Cy,corners))
    projected = np.einsum('fqc,fc->fq',jac,reference)
    if (reference_sq <= 0).any() or (projected < -1e-13*reference_sq[:,None]).any():
        raise ValueError('folded or degenerate bilinear face map')
    points = np.einsum('qa,fac->fqc', N, corners)
    dx = np.einsum('qa,fac->fqc', Nx, corners)
    dy = np.einsum('qa,fac->fqc', Ny, corners)
    area_weights = np.linalg.norm(np.cross(dx, dy), axis=-1) * w
    if not (np.isfinite(points).all() and np.isfinite(area_weights).all() and (area_weights > 0).all()):
        raise ValueError('degenerate bilinear quadrature map')
    return points, area_weights


def native_options(rule):
    # Native 3.0 recognizes only centroid and Gauss2. Install custom cached data
    # after construction, including after the body joins/reorients its lid.
    return {} if rule is None or rule in RULES else {'quadrature_method': rule}


def configure(mesh, rule):
    if rule in RULES:
        points, weights = bilinear_quadrature(mesh.vertices, mesh.faces, RULES[rule])
        points.setflags(write=False)
        weights.setflags(write=False)
        mesh.quadrature_method = rule
        mesh.__dict__['quadrature_points'] = (points, weights)
    return mesh


def verify(mesh, rule):
    if rule in RULES:
        if mesh.quadrature_method != rule or 'quadrature_points' not in mesh.__dict__:
            raise ValueError('native source quadrature rule/cache changed')
        expected = bilinear_quadrature(mesh.vertices, mesh.faces, RULES[rule])
        actual = mesh.quadrature_points
        if (not isinstance(actual, tuple) or len(actual) != 2
                or not all(np.array_equal(a, b) for a, b in zip(actual, expected))):
            raise ValueError('native source quadrature points/weights changed')
