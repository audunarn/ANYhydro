# SPDX-License-Identifier: MPL-2.0
"""Direct BEM method in finite depth (direct-default programme D2): source gradient and assembly."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("capytaine", reason="optional dependency not installed")

import capytaine as cpt                                                # noqa: E402
from capytaine.bem.problems_and_results import LinearPotentialFlowProblem  # noqa: E402

from anyhydro.green_function import anyhydro_green_function            # noqa: E402

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import check_finite_depth_john_oracle as oracle                       # noqa: E402

K, H = 0.3, 10.0            # kh = 3: finite-depth effects O(1), validated Prony decomposition
SOURCE = np.array([0.0, 0.0, -2.0])
RECEIVERS = np.array([[1.7, -0.9, -4.0],     # unequal elevations
                      [3.0, 2.0, -0.3],      # receiver near the free surface
                      [0.8, 0.5, -8.0],      # receiver near the bottom (2 m above)
                      [-2.5, 1.2, -2.0]])    # equal elevations


def _tiny_panel(y, a=1e-3):
    v = np.array([[y[0] - a, y[1] - a, y[2]], [y[0] + a, y[1] - a, y[2]],
                  [y[0] + a, y[1] + a, y[2]], [y[0] - a, y[1] + a, y[2]]])
    return cpt.Mesh(vertices=v, faces=np.array([[0, 1, 2, 3]]))


def _evaluate(receivers, source, *, adjoint):
    mesh = _tiny_panel(source)
    S, G = anyhydro_green_function().evaluate(
        np.atleast_2d(receivers), mesh, free_surface=0.0, water_depth=H, wavenumber=K,
        adjoint_double_layer=adjoint, early_dot_product=False, diagonal_term_in_double_layer=False)
    area = mesh.faces_areas[0]
    return S[:, 0] / area, G[:, :, 0] / area          # value (n,), gradient (3, n)


@pytest.fixture(scope="module")
def kernel():
    return oracle.FiniteDepthKernel(K, H, n_modes=400)


def _oracle(kernel, x, y):
    """ANYhydro/Capytaine normalisation of the John series: G = (1/4 pi)(-Re_series + i Im_series)."""
    g, _, _ = kernel.green(np.atleast_2d(x), np.atleast_2d(y))
    wi, _ = kernel.wave_imag(np.atleast_2d(x), np.atleast_2d(y))
    return (-g[:, 0] + 1j * wi[:, 0]) / (4.0 * np.pi)


def test_source_gradient_matches_the_john_series(kernel):
    """The direct method's d/d(source) of G (all components, near surface and bottom) vs the oracle."""
    value, grad = _evaluate(RECEIVERS, SOURCE, adjoint=False)
    step = 1e-4
    for i, x in enumerate(RECEIVERS):
        ref_value = _oracle(kernel, x, SOURCE)[0]
        assert abs(value[i] - ref_value) <= 5e-3 * abs(ref_value)
        fd = np.empty(3, dtype=complex)
        for d in range(3):
            yp, ym = SOURCE.copy(), SOURCE.copy()
            yp[d] += step
            ym[d] -= step
            fd[d] = (_oracle(kernel, x, yp)[0] - _oracle(kernel, x, ym)[0]) / (2 * step)
        assert np.max(np.abs(grad[:, i] - fd)) <= 5e-3 * np.max(np.abs(fd)), (x, grad[:, i], fd)


def test_source_and_receiver_gradients_satisfy_the_interchange_identity():
    """G(x, y) = G(y, x): d/dy G(x, y) equals the receiver gradient with the roles swapped."""
    for x in RECEIVERS:
        _, grad_source = _evaluate(x, SOURCE, adjoint=False)
        _, grad_receiver = _evaluate(SOURCE, x, adjoint=True)
        assert np.max(np.abs(grad_source - grad_receiver)) <= 1e-3 * np.max(np.abs(grad_receiver))


@pytest.mark.parametrize("depth", [np.inf, 20.0])
def test_direct_solve_recovers_a_known_exterior_potential(depth):
    """Sources inside a closed submerged box give an exact exterior field; given its normal velocity on
    the box, the direct method (assembly incl. the double-layer jump) must return its potential."""
    gf = anyhydro_green_function()
    k = 0.6
    src = cpt.mesh_sphere(radius=0.3, center=(0.4, -0.2, -2.1), resolution=(6, 6))
    sigma = np.random.default_rng(1).normal(size=src.nb_faces) + 1j * np.random.default_rng(2).normal(size=src.nb_faces)
    errors = []
    for res in (6, 12):
        box = cpt.mesh_parallelepiped(size=(4.0, 2.0, 1.5), center=(0.0, 0.0, -2.0), resolution=(res, res, res))
        S, Kg = gf.evaluate(box.faces_centers, src, free_surface=0.0, water_depth=depth, wavenumber=k,
                            adjoint_double_layer=True, early_dot_product=False, diagonal_term_in_double_layer=False)
        exact = S @ sigma
        dphi_dn = np.sum(np.einsum("dnm,m->nd", Kg, sigma) * box.faces_normals, axis=1)
        body = cpt.FloatingBody(box, dofs={})
        problem = LinearPotentialFlowProblem(body=body, wavenumber=k, water_depth=depth, boundary_condition=dphi_dn)
        result = cpt.BEMSolver(green_function=gf, method="direct").solve(problem, keep_details=True)
        errors.append(np.max(np.abs(np.asarray(result.potential) - exact)) / np.max(np.abs(exact)))
    assert errors[1] < 0.02 and errors[1] < 0.6 * errors[0], errors
