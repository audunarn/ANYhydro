# SPDX-License-Identifier: MPL-2.0
"""ANYhydro Green function: Capytaine Delhommeau with two accuracy repairs.

1. The finite-depth remainder uses the validated Prony decomposition of
   :mod:`anyhydro.prony_fit`.
2. The imaginary part of the kernel is evaluated exactly.  For the
   outgoing finite-depth Green function the imaginary part is entirely the
   propagating mode,

       Im G(x, xi) = -(k / (2 D)) Z(x3) Z(xi3) J0(k r),
       Z(z) = cosh k(z + h) / cosh kh,   D = tanh kh + kh sech^2 kh,

   (deep water: Z(z) = exp(k z), D = 1), integrated with the same source
   quadrature that Capytaine uses for its wave terms.  Capytaine's default
   tabulated wave term interpolates this smooth function; the interpolation
   error is small but indefinite, and it produced negative radiation-damping
   eigenvalues that do not converge away under refinement (W5 evidence
   ``w5-route-comparison-20261012``).  Without tabulation Capytaine's own
   imaginary part equals this expression to ~1e-10.

The real part, the singular/Rankine terms and the double-layer jump are
Capytaine's.  Capytaine is imported lazily (never at base import time).
"""
from __future__ import annotations

import math

import numpy as np

from .prony_fit import PRONY_FIT_FORMAT, _validated_class

IMAGINARY_PART_FORMAT = 'anyhydro-exact-propagating-imaginary-v1'
GREEN_FUNCTION_FORMAT = 'anyhydro-green-function-v1'
_ROW_BLOCK = 512

_CLASS = None


def _depth_mode(z, k, water_depth):
    """Z(z) and dZ/dz, overflow-free; deep water when water_depth is inf."""
    z = np.asarray(z, dtype=np.float64)
    if math.isinf(water_depth):
        e = np.exp(k * z)
        return e, k * e
    e = np.exp(k * z)
    f = np.exp(-k * (z + 2 * water_depth))
    g = 1.0 + math.exp(-2 * k * water_depth)
    return (e + f) / g, k * (e - f) / g


def dispersion_factor(k, water_depth):
    if math.isinf(water_depth):
        return 1.0
    kh = k * water_depth
    t = math.tanh(kh)
    return t + kh * (1.0 - t * t)


def propagating_imaginary(points, quad_points, quad_weights, *, k, water_depth,
                          gradient='receiver', normals=None):
    """Exact Im S (n, m) and Im of the gradient.

    ``gradient`` selects the derivative with respect to the receiver point
    (adjoint double layer, indirect method) or the source point (double
    layer, direct method).  Without ``normals`` the full gradient (3, n, m)
    is returned; with ``normals`` ((n, 3) receiver normals for 'receiver',
    (m, 3) source normals for 'source') only its normal projection (n, m)
    is formed, block by block, so no (3, n, m) array is allocated.
    """
    x = np.asarray(points, dtype=np.float64)
    y = np.asarray(quad_points, dtype=np.float64)
    w = np.asarray(quad_weights, dtype=np.float64)
    if y.ndim == 2:              # one quadrature point per face
        y, w = y[:, None, :], w[:, None]
    if not (x.ndim == 2 and x.shape[1] == 3 and y.ndim == 3 and y.shape[2] == 3
            and w.shape == y.shape[:2]):
        raise ValueError('invalid receiver/quadrature shapes')
    if gradient not in ('receiver', 'source'):
        raise ValueError("gradient must be 'receiver' or 'source'")
    n, m = x.shape[0], y.shape[0]
    if normals is not None:
        nrm = np.asarray(normals, dtype=np.float64)
        if nrm.shape != ((n, 3) if gradient == 'receiver' else (m, 3)):
            raise ValueError('normals must match the differentiated points')
    from scipy.special import j0, j1
    factor = -k / (2.0 * dispersion_factor(k, water_depth))
    sgn = 1.0 if gradient == 'receiver' else -1.0
    S = np.zeros((n, m))
    G = np.zeros((n, m)) if normals is not None else np.zeros((3, n, m))
    Zx, dZx = _depth_mode(x[:, 2], k, water_depth)
    for q in range(y.shape[1]):
        Zy, dZy = _depth_mode(y[:, q, 2], k, water_depth)
        wq = factor * w[:, q]
        for a in range(0, n, _ROW_BLOCK):
            b = min(a + _ROW_BLOCK, n)
            delta = x[a:b, None, :2] - y[None, :, q, :2]
            r = np.sqrt(np.sum(delta * delta, axis=2))
            kr = k * r
            J0 = j0(kr)
            # d J0(k r)/d x_receiver = -k J1(k r) (x - xi)/r ; source gradient flips sign
            radial = np.divide(-k * j1(kr), r, out=np.zeros_like(r), where=r > 0)
            zz = wq[None, :] * Zy[None, :] * Zx[a:b, None]
            S[a:b] += zz * J0
            gx = sgn * zz * radial * delta[:, :, 0]
            gy = sgn * zz * radial * delta[:, :, 1]
            if gradient == 'receiver':
                gz = (wq * Zy)[None, :] * dZx[a:b, None] * J0
            else:
                gz = (wq * dZy)[None, :] * Zx[a:b, None] * J0
            if normals is None:
                G[0, a:b] += gx
                G[1, a:b] += gy
                G[2, a:b] += gz
            elif gradient == 'receiver':
                G[a:b] += gx * nrm[a:b, None, 0] + gy * nrm[a:b, None, 1] + gz * nrm[a:b, None, 2]
            else:
                G[a:b] += gx * nrm[None, :, 0] + gy * nrm[None, :, 1] + gz * nrm[None, :, 2]
    return S, G


def _anyhydro_class():
    global _CLASS
    if _CLASS is None:
        Base = _validated_class()

        class AnyhydroDelhommeau(Base):
            """Validated Prony remainder + exact propagating imaginary part."""

            def evaluate(self, mesh1, mesh2, *, free_surface=0.0, water_depth=np.inf,
                         wavenumber=1.0, adjoint_double_layer=True, early_dot_product=True,
                         diagonal_term_in_double_layer=True):
                S, K = super().evaluate(
                    mesh1, mesh2, free_surface=free_surface, water_depth=water_depth,
                    wavenumber=wavenumber, adjoint_double_layer=adjoint_double_layer,
                    early_dot_product=early_dot_product,
                    diagonal_term_in_double_layer=diagonal_term_in_double_layer)
                k = float(wavenumber)
                if free_surface != 0.0 or not (0.0 < k < np.inf):
                    return S, K      # no propagating mode: Capytaine's result stands
                points, normals = self._get_colocation_points_and_normals(
                    mesh1, mesh2, adjoint_double_layer)
                qp, qw = mesh2.quadrature_points
                Si, Gi = propagating_imaginary(
                    points, qp, qw, k=k, water_depth=float(water_depth),
                    gradient='receiver' if adjoint_double_layer else 'source',
                    normals=normals if early_dot_product else None)
                S.imag = Si          # in place: no complex temporaries
                del Si
                if early_dot_product:
                    K.imag = Gi.reshape(K.shape)
                elif K.shape[0] == 3:
                    K.imag = Gi
                else:
                    K.imag = np.moveaxis(Gi, 0, -1)
                return S, K

        AnyhydroDelhommeau.__module__ = __name__
        AnyhydroDelhommeau.__qualname__ = 'AnyhydroDelhommeau'
        _CLASS = AnyhydroDelhommeau
    return _CLASS


def __getattr__(name):
    if name == 'AnyhydroDelhommeau':
        return _anyhydro_class()
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def anyhydro_green_function(**options):
    """Production Green function; ``options`` go to Capytaine's constructor."""
    gf = _anyhydro_class()(**options)
    gf.exportable_settings = dict(gf.exportable_settings,
                                  finite_depth_prony_decomposition=PRONY_FIT_FORMAT,
                                  imaginary_part=IMAGINARY_PART_FORMAT)
    gf._hash = hash(tuple(gf.exportable_settings.values()))
    return gf
