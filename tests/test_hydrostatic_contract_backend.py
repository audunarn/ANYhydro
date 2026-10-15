# SPDX-License-Identifier: MPL-2.0
"""ADR-015 L6-4 H-a: exact baseline wrench, verified stiffness semantics, ADR-004 chart pullback, backend contract."""
from __future__ import annotations

import json

import numpy as np
import pytest

from anyhydro.hydrostatics_exact import HydrostaticsExactError, buoyancy_baseline_wrench, immersed_volume_moments

capytaine = pytest.importorskip("capytaine", reason="optional dependency not installed")
from scipy.spatial.transform import Rotation  # noqa: E402

RHO, G = 1025.0, 9.81
CENTER = np.array([0.5, 0.2, -0.3])


def _box(res, odd_layers=False):
    """4 x 2 m box, 1 m draft, centred at (1.5, -0.7); closed mesh straddling z = 0 (immersed part used).
    ``odd_layers``: no vertex row on z = 0 (robust clipping under small rotations)."""
    return capytaine.mesh_parallelepiped(size=(4.0, 2.0, 2.0), center=(1.5, -0.7, 0.0),
                                         resolution=(4 * res, 2 * res, 2 * res + (1 if odd_layers else 0)))


def _exact_wrench(mesh, q):
    u, theta = np.asarray(q[:3]), np.asarray(q[3:])
    R = Rotation.from_rotvec(theta).as_matrix()
    verts = (np.asarray(mesh.vertices) - CENTER) @ R.T + CENTER + u
    wet = capytaine.Mesh(vertices=verts, faces=mesh.faces).immersed_part()
    V, m = immersed_volume_moments(wet.vertices, wet.faces)
    return V, m, np.array(buoyancy_baseline_wrench(wet.vertices, wet.faces, CENTER + u, RHO, G))


def _fd(fun, n, h):
    """Central differences of a vector function of q (6,) -> (n,) columns per q_j, Richardson-extrapolated."""
    def once(step):
        out = np.zeros((n, 6))
        for j in range(6):
            e = np.zeros(6)
            e[j] = step
            out[:, j] = (fun(e) - fun(-e)) / (2 * step)
        return out
    return (4 * once(h / 2) - once(h)) / 3


# ---------------------------------------------------------------------------------------------- exact moments

def test_exact_moments_of_an_offset_box():
    wet = _box(4).immersed_part()
    V, m = immersed_volume_moments(wet.vertices, wet.faces)
    assert V == pytest.approx(8.0, rel=1e-13)
    assert np.allclose(m / V, [1.5, -0.7, -0.5], rtol=0, atol=1e-13)
    F = buoyancy_baseline_wrench(wet.vertices, wet.faces, CENTER, RHO, G)
    W = RHO * G * 8.0
    assert np.allclose(F, [0, 0, W, W * (-0.9), -W * 1.0, 0.0], rtol=1e-13, atol=1e-9)


def test_exact_moments_refuse_meshes_above_the_surface():
    with pytest.raises(HydrostaticsExactError, match="above the free surface"):
        immersed_volume_moments(_box(2).vertices, _box(2).faces)


# ---------------------------------------------------------------------------------------------- semantics

def _capytaine_stiffness(res):
    body = capytaine.FloatingBody(mesh=_box(res).immersed_part(),
                                  dofs=capytaine.rigid_body_dofs(rotation_center=tuple(CENTER)),
                                  center_of_mass=tuple(CENTER), mass=0.0)
    return np.asarray(body.compute_hydrostatic_stiffness(rho=RHO, g=G).values)


def test_capytaine_stiffness_is_the_incremental_wrench_derivative_about_the_rotation_centre():
    """K_source = -dQ/dh with Q = (F, M about the co-moving rotation centre), h = (translation, spatial rotation
    increment, world axes). On this axis-aligned box Capytaine's centroid quadrature is exact for the entries linear in
    the waterplane/volume moments; K44/K55 converge at second order (its panel quadrature of int x^2, int y^2 dA).
    (On inclined panels the centroid quadrature also affects the CB-based entries - see the chart-stiffness test.)"""
    mesh = _box(4, odd_layers=True)
    exact = -_fd(lambda q: _exact_wrench(mesh, q)[2], 6, 1e-3)
    errors = {}
    for res in (4, 8, 16):
        body = capytaine.FloatingBody(mesh=_box(res, odd_layers=True).immersed_part(),
                                      dofs=capytaine.rigid_body_dofs(rotation_center=tuple(CENTER)),
                                      center_of_mass=tuple(CENTER), mass=0.0)
        K = np.asarray(body.compute_hydrostatic_stiffness(rho=RHO, g=G).values)
        diff = K - exact
        mask = np.ones((6, 6), bool)
        mask[3, 3] = mask[4, 4] = False
        assert np.max(np.abs(diff[mask])) <= 1e-6 * np.abs(exact).max(), res
        errors[res] = (abs(diff[3, 3]), abs(diff[4, 4]))
    for k in (0, 1):
        assert 3.0 <= errors[4][k] / errors[8][k] <= 5.0 and 3.0 <= errors[8][k] / errors[16][k] <= 5.0
    assert exact[3, 5] == pytest.approx(-RHO * G * 8.0 * (1.5 - CENTER[0]), rel=1e-9)     # baseline-moment term
    assert exact[5, 3] == pytest.approx(0.0, abs=1e-6)                                     # (asymmetric source)


def test_adr004_chart_pullback_is_the_hessian_of_the_buoyancy_potential():
    """Buoyancy is conservative: in the solver chart (u, Exp(theta) about the co-moving centre) its stiffness is the
    Hessian of U = -rho g int_V z dV; K_chart = K_source - 1/2 skew(M_baseline) on the rotational block."""
    mesh = _box(4, odd_layers=True)

    def potential_gradient(q):
        def U(x):
            V, m, _ = _exact_wrench(mesh, x)
            return -RHO * G * m[2]
        grad = np.zeros(6)
        for j in range(6):
            e = np.zeros(6)
            e[j] = 1e-4
            grad[j] = (U(q + e) - U(q - e)) / 2e-4
        return grad
    hessian = _fd(potential_gradient, 6, 1e-3)
    hessian = 0.5 * (hessian + hessian.T)
    K_source = -_fd(lambda q: _exact_wrench(mesh, q)[2], 6, 1e-3)
    M = np.asarray(_exact_wrench(mesh, np.zeros(6))[2][3:])
    skew = np.array([[0, -M[2], M[1]], [M[2], 0, -M[0]], [-M[1], M[0], 0]])
    K_chart = K_source.copy()
    K_chart[3:, 3:] -= 0.5 * skew
    assert np.allclose(K_chart, K_chart.T, rtol=0, atol=1e-4 * np.abs(K_chart).max())
    assert np.allclose(K_chart, hessian, rtol=0, atol=1e-4 * np.abs(K_chart).max())


# ---------------------------------------------------------------------------------------------- backend

from anyhydro import CapytaineBackend                                          # noqa: E402
from anyhydro.capytaine_backend import CapytaineExecutionLimits, HYDROSTATIC_COORDINATES   # noqa: E402
from anyhydro.errors import CapabilityError                                    # noqa: E402
from anyhydro.hydrostatic_contract import HYDROSTATIC_CONTRACT_ATTR, chart_stiffness, read_hydrostatic_contract, \
    require_hydrostatic_composition                                            # noqa: E402

from test_capytaine_backend import _box_body, _dev_request                     # noqa: E402


def _solve(settings):
    body = _box_body("b0", (0.3, -0.2, -0.3))
    limits = CapytaineExecutionLimits(max_total_panels=100, max_finite_omegas=1, max_wave_directions=1)
    request = _dev_request(omegas=np.array([1.0]), wave_directions=np.array([0.0]), backend_settings=settings)
    return body, CapytaineBackend(execution_limits=limits).solve([body], request)


def test_backend_attaches_a_bound_admissible_contract_on_request():
    body, ds = _solve({"hydrostatic_contract": "attach"})
    contract = read_hydrostatic_contract(ds)
    assert contract.role == "BUOYANCY_ONLY" and contract.force_basis == "PHYSICAL_WRENCH"
    assert contract.rate_map == "spatial_increments_exp_chart_v1"
    assert contract.stiffness_derivative == "INCREMENTAL_ABOUT_EQUILIBRIUM"
    assert contract.provenance["coordinates"] == HYDROSTATIC_COORDINATES
    require_hydrostatic_composition(ds)                           # complete and bound to this dataset
    mesh = capytaine.Mesh(vertices=np.asarray(body.mesh.vertices), faces=np.asarray(body.mesh.faces)).immersed_part()
    expected = buoyancy_baseline_wrench(mesh.vertices, mesh.faces, body.rotation_center, 1025.0, 9.81)
    assert np.allclose(contract.baseline_wrench["b0"], expected, rtol=1e-12, atol=1e-9)


def test_legacy_requests_carry_no_contract_and_bad_values_are_refused():
    _, ds = _solve({})
    assert HYDROSTATIC_CONTRACT_ATTR not in ds.attrs
    with pytest.raises(CapabilityError, match="hydrostatic_contract"):
        _solve({"hydrostatic_contract": "yes"})



# ---------------------------------------------------------------------------------------------- review round 1

def test_chart_stiffness_of_the_capytaine_matrix_converges_to_the_potential_hessian():
    """Capytaine's matrix through the implemented conversion (hydrostatic_contract.chart_stiffness) equals the exact
    potential Hessian up to Capytaine's panel quadrature, which converges at second order (box, odd layers)."""
    from anyhydro.hydrostatic_contract import HydrostaticContract
    mesh_exact = _box(4, odd_layers=True)
    W = RHO * G * 8.0
    exact_K = -_fd(lambda q: _exact_wrench(mesh_exact, q)[2], 6, 1e-3)
    M = np.asarray(_exact_wrench(mesh_exact, np.zeros(6))[2][3:])
    skew = np.array([[0, -M[2], M[1]], [M[2], 0, -M[0]], [-M[1], M[0], 0]])
    exact_K[3:, 3:] -= 0.5 * skew
    errors = []
    for res in (4, 8, 16):
        mesh = _box(res, odd_layers=True)
        body = capytaine.FloatingBody(mesh=mesh.immersed_part(),
                                      dofs=capytaine.rigid_body_dofs(rotation_center=tuple(CENTER)),
                                      center_of_mass=tuple(CENTER), mass=0.0)
        K_source = np.asarray(body.compute_hydrostatic_stiffness(rho=RHO, g=G).values)
        wet = mesh.immersed_part()
        contract = HydrostaticContract(
            role="BUOYANCY_ONLY", force_basis="PHYSICAL_WRENCH", rate_map="spatial_increments_exp_chart_v1",
            stiffness_derivative="INCREMENTAL_ABOUT_EQUILIBRIUM", bodies=("b",),
            baseline_wrench={"b": buoyancy_baseline_wrench(wet.vertices, wet.faces, CENTER, RHO, G)},
            provenance={"source": "test"}, full_source=None)
        K = chart_stiffness(contract, "b", K_source)
        assert np.max(np.abs(K - K.T)) <= 1e-9 * W
        errors.append(np.max(np.abs(K - exact_K)) / W)
    assert errors[0] / errors[1] > 3.0 and errors[1] / errors[2] > 3.0


@pytest.mark.parametrize("case", ["open", "reversed", "cap", "warped"])
def test_exact_moments_refuse_invalid_wet_boundaries(case):
    """Codex round 1 probes: an opening below the waterplane, an inverted face, a waterplane cap, a warped quad."""
    tet = np.array([[0, 0, -1], [2, 0, -1], [0, 2, -1], [0, 0, -3]], float)
    faces = [[0, 1, 2, 2], [0, 3, 1, 1], [1, 3, 2, 2], [2, 3, 0, 0]]
    V, _ = immersed_volume_moments(tet, np.array(faces))                 # the closed tetrahedron is admitted
    assert V == pytest.approx(4.0 / 3.0, rel=1e-14)
    if case in ("open", "reversed"):
        bad = faces[1:] if case == "open" else [[0, 2, 1, 1]] + faces[1:]
        with pytest.raises(HydrostaticsExactError, match="not closed"):
            immersed_volume_moments(tet, np.array(bad))
        return
    box = np.array([[0, 0, -1], [2, 0, -1], [2, 2, -1], [0, 2, -1], [0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], float)
    sides = [[0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    if case == "cap":
        with pytest.raises(HydrostaticsExactError, match="free surface z = 0"):
            immersed_volume_moments(box, np.array([[0, 3, 2, 1]] + sides + [[4, 5, 6, 7]]))
    else:
        warped = box.copy()
        warped[2, 2] = -2.0
        with pytest.raises(HydrostaticsExactError, match="not planar"):
            immersed_volume_moments(warped, np.array([[0, 3, 2, 1]] + sides))


def test_physical_wrench_needs_an_implemented_rate_map():
    from anyhydro.errors import RecordValidationError
    from anyhydro.hydrostatic_contract import HydrostaticContract, HydrostaticContractError
    declared = dict(role="BUOYANCY_ONLY", force_basis="PHYSICAL_WRENCH",
                    stiffness_derivative="INCREMENTAL_ABOUT_EQUILIBRIUM", bodies=("b",),
                    baseline_wrench={"b": (0, 0, 1, 0, 0, 0)}, provenance={"source": "t"}, full_source=None)
    with pytest.raises(HydrostaticContractError, match="no implemented conversion"):
        chart_stiffness(HydrostaticContract(**declared, rate_map="some_other_map_v1"), "b", np.eye(6))
    with pytest.raises(RecordValidationError, match="rate_map"):
        HydrostaticContract(**dict(declared, force_basis="GENERALIZED_DOF"), rate_map="spatial_increments_exp_chart_v1")
