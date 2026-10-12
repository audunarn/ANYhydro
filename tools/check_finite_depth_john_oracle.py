# SPDX-License-Identifier: MPL-2.0
"""Independent finite-depth Green-function oracle (John eigenfunction series).

Attributes the native free-surface Robin residual of a physical-field
record written by ``check_physical_green_field.py`` (format with
``primitive_*`` arrays).  Reads the saved record only; never imports
Capytaine and never calls native code.

Kernel convention (Capytaine theory manual, ``fin_depth_green_function``):
``Gt = -4*pi*G = 1/R + ...``; the native single-layer value of one panel
is ``S = -1/(4*pi) * integral(Gt)``.

Exact kernel, real part (John 1950 eigenfunction expansion)::

    Gt = 2*pi*(-k^2)/(k^2 sech^2(kh) h + nu) Z(z) Z(zeta) Y0(k r)
       + 4 sum_n (kn^2+nu^2)/((kn^2+nu^2) h - nu)
               cos kn(z+h) cos kn(zeta+h) K0(kn r)

with ``Z(z) = cosh k(z+h)/cosh kh``, ``nu = k tanh kh`` and
``kn tan(kn h) = -nu``.  Every mode satisfies ``dG/dz = nu G`` at
``z = 0`` and ``dG/dz = 0`` at ``z = -h`` exactly, so the oracle's own
Robin residual is a constructed zero (it validates the series only).

Native model (the identity this tool tests against the record)::

    S_native = -1/(4 pi) [ Gauss4(Gt_exact) - Gauss4(M_eps) + MIX ]

* ``M_eps = 2 int eps(kappa) e^{-2 kappa h} cosh(kappa(x3+h))
  cosh(kappa(xi3+h)) J0(kappa r) d kappa`` is the field of the Prony fit
  error ``eps(x) = ref(x) - sum a_i e^{lam_i x}`` with ``x = kappa h`` and
  ``ref(x) = F(x) - A/(x - kh) - 2`` (upstream ``delhommeau.py``);
* ``MIX = area * Rankine(centroid) - Gauss4(Rankine)`` over the 6 base and
  4N Prony Rankine images: the kernel integrates Rankine-type terms with a
  one-point rule far from the panel and the wave terms with the panel
  quadrature, so their sum is not a single consistent face rule.

Only ``M_eps`` and ``MIX`` violate the free-surface condition, so the
predicted native Robin residual is fully determined by them.  The tool
reports the measured native residual, the prediction and the split.
Qualification is always false.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import j0, k0, y0

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / 'src') not in sys.path:
    sys.path.insert(0, str(_REPO / 'src'))

from anyhydro.prony_fit import reference_function, tail_coefficient  # noqa: E402

DEFAULT_MODES = 1200
X_MAX = 30.0            # minimum integration cut-off in x = kappa h
X_CUT_CAP = 2000.0      # largest cut-off extended to for slow candidate terms
TAIL_LIMIT = 1e-9       # bound on the neglected tail of eps beyond the cut-off
GATE_REL_TOL = 1e-4     # check_physical_green_field ROBIN_REL_TOL
GATE_ABS_TOL = 1e-9     # check_physical_green_field ROBIN_ABS_TOL


def depth_products(x, a, b):
    """``e^{-2x} cosh(x(1+a)) cosh(x(1+b))`` and the sinh-cosh variant.

    Written as sums of exponentials with non-positive exponents for
    ``a, b`` in ``[-1, 0]`` (receiver/source depth over h), so that long
    integration ranges never overflow.
    """
    e1 = np.exp(x * (a + b))
    e2 = np.exp(x * (a - b - 2))
    e3 = np.exp(x * (b - a - 2))
    e4 = np.exp(-x * (4 + a + b))
    return 0.25 * (e1 + e2 + e3 + e4), 0.25 * (e1 + e2 - e3 - e4)


class FiniteDepthKernel:
    """Exact finite-depth kernel (John series) plus Prony-error fields."""

    def __init__(self, k, water_depth, n_modes=DEFAULT_MODES):
        if not (k > 0 and water_depth > 0 and n_modes > 0):
            raise ValueError('k, water depth and mode count must be positive')
        self.k = float(k)
        self.h = float(water_depth)
        self.kh = self.k * self.h
        self.nu = self.k * math.tanh(self.kh)
        self.kn = self._evanescent_roots(n_modes)
        sech2 = 1.0 / math.cosh(self.kh) ** 2 if self.kh < 350 else 0.0
        # (nu^2 - k^2) = -k^2 sech^2(kh); sech^2 is folded into Z(z) Z(zeta)
        self.c0 = 2 * math.pi * (-self.k ** 2) / (self.k ** 2 * sech2 * self.h + self.nu)
        kn2 = self.kn ** 2
        self.cn = 4 * (kn2 + self.nu ** 2) / ((kn2 + self.nu ** 2) * self.h - self.nu)

    def _evanescent_roots(self, n_modes):
        nuh = self.nu * self.h
        roots = np.empty(n_modes)
        for n in range(1, n_modes + 1):
            roots[n - 1] = brentq(lambda u: u * math.tan(u) + nuh,
                                  (n - 0.5) * math.pi + 1e-12, n * math.pi,
                                  xtol=1e-15, rtol=1e-15, maxiter=200)
        return roots / self.h

    def _depth_mode(self, z):
        # cosh k(z+h)/cosh kh evaluated without overflow
        k, h = self.k, self.h
        e = np.exp(k * z)
        f = np.exp(-k * (z + 2 * h))
        g = 1.0 + math.exp(-2 * self.kh)
        return (e + f) / g, k * (e - f) / g

    def green(self, receivers, sources):
        """Real part of Gt and dGt/dx3 for receivers (R,3), sources (Q,3)."""
        x = np.asarray(receivers, dtype=float)
        y = np.asarray(sources, dtype=float)
        r = np.hypot(x[:, None, 0] - y[None, :, 0], x[:, None, 1] - y[None, :, 1])
        if np.any(r <= 0):
            raise ValueError('the series needs a horizontal offset r > 0')
        zx, dzx = self._depth_mode(x[:, None, 2])
        zy, _ = self._depth_mode(y[None, :, 2])
        g = self.c0 * zx * zy * y0(self.k * r)
        gz = self.c0 * dzx * zy * y0(self.k * r)
        kn = self.kn[None, None, :]
        cn = self.cn[None, None, :]
        rr = r[..., None]
        h = self.h
        cs = np.cos(kn * (y[None, :, 2][..., None] + h))
        K0 = k0(kn * rr)
        g = g + (cn * np.cos(kn * (x[:, None, 2][..., None] + h)) * cs * K0).sum(-1)
        gz = gz + (cn * (-kn) * np.sin(kn * (x[:, None, 2][..., None] + h)) * cs * K0).sum(-1)
        tail = float(np.abs(cn[..., -1] * K0[..., -1]).max())
        return g, gz, tail

    def wave_imag(self, receivers, sources):
        """Magnitude of the propagating imaginary part, c0 Z Z J0(kr)."""
        x = np.asarray(receivers, dtype=float)
        y = np.asarray(sources, dtype=float)
        r = np.hypot(x[:, None, 0] - y[None, :, 0], x[:, None, 1] - y[None, :, 1])
        zx, dzx = self._depth_mode(x[:, None, 2])
        zy, _ = self._depth_mode(y[None, :, 2])
        return self.c0 * zx * zy * j0(self.k * r), self.c0 * dzx * zy * j0(self.k * r)

    # ---- Prony remainder ------------------------------------------------
    def ref(self, x):
        """Upstream fitted function ref(x) = F(x) - A/(x-kh) - 2 (x = kappa h)."""
        return reference_function(x, self.kh)

    def integration_cutoff(self, coefficients=None):
        """Cut-off in x such that the neglected tail of eps is bounded.

        Two tails are bounded: the algebraic tail of ``ref(x) ~ c/x`` with
        ``c = 4 kh tanh kh - A`` (negligible for the W5 case kh = 34.4, not
        for kh below ~8; refused), and the candidate's own exponentials via
        ``int_X^inf |a_i| e^{lam_i x} dx = |a_i| e^{lam_i X}/|lam_i|``; the
        cut-off is extended (up to X_CUT_CAP) until that bound is below
        TAIL_LIMIT, otherwise refused.
        """
        c = tail_coefficient(self.kh)
        if abs(c) / X_MAX > TAIL_LIMIT:
            raise ValueError(f'algebraic tail c={c:.3e} of ref(x) is not negligible at '
                             f'kh={self.kh}; a finite cut-off would truncate the error field')
        if coefficients is None:
            return X_MAX
        lam, amp = (np.asarray(v, dtype=float) for v in coefficients)
        if np.any(lam >= 0):
            raise ValueError('candidate exponentials must decay (lam < 0)')
        bound = lambda x: float(np.sum(np.abs(amp) * np.exp(lam * x) / np.abs(lam)))  # noqa: E731
        cut = X_MAX
        while bound(cut) > TAIL_LIMIT:
            if cut >= X_CUT_CAP:
                raise ValueError(f'candidate exponential tail beyond x={X_CUT_CAP} bounded by '
                                 f'{bound(X_CUT_CAP):.3e} > {TAIL_LIMIT}')
            cut = min(cut * 1.5, X_CUT_CAP)
        return cut

    @staticmethod
    def fit(x, coefficients):
        lam, amp = (np.asarray(v, dtype=float) for v in coefficients)
        return (amp[:, None] * np.exp(lam[:, None] * np.asarray(x)[None, :])).sum(0)

    def eps_nodes(self, coefficients, eps_fn=None, x_cut=X_MAX, order=64):
        xg, wg = np.polynomial.legendre.leggauss(order)
        edges = np.linspace(0.0, x_cut, int(math.ceil(2 * x_cut)) + 1)
        xs = np.concatenate([(b - a) / 2 * xg + (a + b) / 2 for a, b in zip(edges[:-1], edges[1:])])
        ws = np.concatenate([(b - a) / 2 * wg for a, b in zip(edges[:-1], edges[1:])])
        eps = (eps_fn(xs) if eps_fn is not None
               else self.ref(xs) - self.fit(xs, coefficients))
        return xs, ws, eps

    def prony_error_field(self, receivers, sources, coefficients, eps_fn=None):
        """M_eps and dM_eps/dx3 (Gt units) for the fit error eps."""
        x_cut = X_MAX if eps_fn is not None else self.integration_cutoff(coefficients)
        x = np.asarray(receivers, dtype=float)
        y = np.asarray(sources, dtype=float)
        xs, ws, eps = self.eps_nodes(coefficients, eps_fn, x_cut)
        h = self.h
        r = np.hypot(x[:, None, 0] - y[None, :, 0], x[:, None, 1] - y[None, :, 1])[..., None]
        kap = xs / h
        x3 = x[:, None, 2][..., None]
        x3s = y[None, :, 2][..., None]
        cc, sc = depth_products(xs, x3 / h, x3s / h)
        base = ws * eps * j0(kap * r)
        m = (2 / h) * (base * cc).sum(-1)
        mz = (2 / h) * (base * kap * sc).sum(-1)
        return m, mz

    def rankine_images(self, receivers, sources, coefficients):
        """6 base + 4N Prony Rankine images of Gt and their x3-derivative."""
        x = np.asarray(receivers, dtype=float)
        y = np.asarray(sources, dtype=float)
        h = self.h
        r2 = (x[:, None, 0] - y[None, :, 0]) ** 2 + (x[:, None, 1] - y[None, :, 1]) ** 2
        a3 = x[:, None, 2]
        b3 = y[None, :, 2]
        terms = [(1.0, a3 - b3), (1.0, a3 - b3 - 2 * h), (1.0, a3 - b3 + 2 * h),
                 (1.0, a3 + b3 + 2 * h), (1.0, a3 + b3), (1.0, a3 + b3 + 4 * h)]
        lam, amp = (np.asarray(v, dtype=float) for v in coefficients)
        for a, lt in zip(amp, -lam):
            terms += [(a / 2, a3 - b3 - 2 * h - lt * h), (a / 2, a3 - b3 + 2 * h + lt * h),
                      (a / 2, a3 + b3 - lt * h), (a / 2, a3 + b3 + 4 * h + lt * h)]
        g = np.zeros_like(r2)
        gz = np.zeros_like(r2)
        for c, zz in terms:
            rr = np.sqrt(r2 + zz * zz)
            g += c / rr
            gz += -c * zz / rr ** 3
        return g, gz


def gate_ratio(residual, scale):
    return residual / (GATE_ABS_TOL + GATE_REL_TOL * scale)


def attribute_case(kernel, diag, source, coefficients):
    """Native-vs-model comparison for one source case of the record."""
    c = -1.0 / (4 * math.pi)
    meta = diag['primitive_metadata']
    n_base = 1 + max(meta['far_index'] + meta['interior_index']
                     + meta['surface_index'] + meta['bottom_index'])
    X = np.asarray(diag['receivers_saved'], dtype=float)[:n_base]
    Y = np.asarray(diag['source_points_saved'], dtype=float)[0]
    W = np.asarray(diag['source_weights_saved'], dtype=float)[0]
    centroid = np.asarray(source['vertices_translated'], dtype=float).mean(0)[None, :]
    area = float(W.sum())
    nS = np.asarray(diag['primitive_potential_native']['real'], dtype=float)[:n_base, 0]
    nKz = np.asarray(diag['primitive_gradient_native']['real'], dtype=float)[2, :n_base, 0]
    pS = np.asarray(diag['primitive_potential_pole']['real'], dtype=float)[:n_base, 0]
    pKz = np.asarray(diag['primitive_gradient_pole']['real'], dtype=float)[2, :n_base, 0]
    quad = lambda a: (a * W[None, :]).sum(1)  # noqa: E731
    g, gz, tail = kernel.green(X, Y)
    m, mz = kernel.prony_error_field(X, Y, coefficients)
    rq, rzq = kernel.rankine_images(X, Y, coefficients)
    rc, rzc = kernel.rankine_images(X, centroid, coefficients)
    mix_s = area * rc[:, 0] - quad(rq)
    mix_kz = area * rzc[:, 0] - quad(rzq)
    exact_s = c * quad(g)
    model_s = c * (quad(g) - quad(m) + mix_s)
    model_kz = c * (quad(gz) - quad(mz) + mix_kz)
    nu = kernel.nu
    surface = []
    for j in meta['surface_index']:
        r_prony = c * (-quad(mz)[j] + nu * quad(m)[j])
        r_mix = c * (mix_kz[j] - nu * mix_s[j])
        r_exact = c * (quad(gz)[j] - nu * quad(g)[j])
        measured = nKz[j] - nu * nS[j]
        comp = abs((nKz[j] + 1j * pKz[j]) - nu * (nS[j] + 1j * pS[j]))
        scale = max(abs(nKz[j] + 1j * pKz[j]), abs(nu * (nS[j] + 1j * pS[j])), 1e-12)
        # predictions use the scale of the MODELLED field for these coefficients
        mscale = max(abs(model_kz[j] + 1j * pKz[j]), abs(nu * (model_s[j] + 1j * pS[j])), 1e-12)
        surface.append(dict(
            receiver=int(j), measured_real_residual=float(measured),
            predicted_prony=float(r_prony), predicted_mixing=float(r_mix),
            exact_constructed=float(r_exact),
            predicted_total=float(r_prony + r_mix + r_exact),
            measured_over_predicted=float(measured / (r_prony + r_mix + r_exact)),
            composite_gate_ratio=float(gate_ratio(comp, scale)),
            predicted_gate_ratio=float(gate_ratio(abs(r_prony + r_mix + r_exact), mscale)),
            predicted_gate_ratio_without_prony=float(gate_ratio(abs(r_mix + r_exact), mscale)),
            predicted_gate_ratio_without_mixing=float(gate_ratio(abs(r_prony + r_exact), mscale))))
    return dict(
        series_tail=tail, normalization_fit=float(np.dot(exact_s, nS) / np.dot(exact_s, exact_s)),
        native_vs_model_max_rel=float(np.max(np.abs(nS - model_s) / np.abs(model_s))),
        native_kz_vs_model_max_rel=float(np.max(np.abs(nKz - model_kz))
                                         / np.max(np.abs(model_kz))),
        native_vs_exact_rel=[float(v) for v in np.abs(nS - exact_s) / np.abs(exact_s)],
        surface=surface)


def attribute_record(record, n_modes=DEFAULT_MODES, coefficients=None):
    if record.get('qualification') is not False:
        raise ValueError('record must be an unqualified physical-field record')
    k = record['case']['k']
    h = record['case']['water_depth']
    coeff = coefficients if coefficients is not None else \
        record['frozen_prony_coefficients']['values']
    kernel = FiniteDepthKernel(k, h, n_modes)
    xf = np.linspace(-0.1, 20.0, 2001)
    eps = kernel.ref(xf) - kernel.fit(xf, coeff)
    out = dict(tool='check_finite_depth_john_oracle', qualification=False,
               k=k, water_depth=h, kh=kernel.kh, nu=kernel.nu, n_modes=n_modes,
               prony_fit=dict(n_exp=len(coeff[0]), mse_fit_range=float(np.mean(eps ** 2)),
                              max_abs_eps_fit_range=float(np.abs(eps).max())),
               cases={})
    for label, diag in record['diagnostics'].items():
        out['cases'][label] = attribute_case(kernel, diag, record['sources'][label], coeff)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--record', required=True, type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--modes', type=int, default=DEFAULT_MODES)
    a = p.parse_args(argv)
    result = attribute_record(json.loads(a.record.read_text(encoding='utf-8')), a.modes)
    text = json.dumps(result, indent=1)
    if a.output:
        if a.output.exists():
            raise SystemExit(f'refusing to overwrite {a.output}')
        a.output.write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
