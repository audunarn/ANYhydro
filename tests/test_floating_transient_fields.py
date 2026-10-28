"""Native-family adapter and pointwise clearance guard: structural identity, policy split, unchanged builder checks.

The analytic family is the exponential-memory model of test_radiation_field_kernel (P_v = Mv exp(-lam t), elevation
E_v exp(-lam t), Proj Mv = -D, Proj P_a = -A_inf) expressed as RETAINED FloatingFamilySnapshot records plus a
NativeFieldCoefficients object of the same posed family.  No native backend is used: the elevation evaluator is an
injected stand-in for the owner's retained-source recovery (it is the only callable the adapter may use).
"""
import functools

import numpy as np
import pytest

from anyhydro.floating_radiation_fields import DOF_NAMES, FloatingFamilySnapshot, family_request
from anyhydro.floating_transient_fields import (
    ClearanceConsumedError,
    FamilyAdapterError,
    build_family_radiation_field_kernel,
    guarded_clearance,
)
from anyhydro.radiation import RadiationSettings, _kernel_from_arrays
from anyhydro.radiation_field_kernel import (
    FieldReconstructionCriteria,
    NativeBoundaryError,
    NativeFieldCoefficients,
    RadiationFieldError,
    RadiationFieldLimitError,
    build_radiation_field_kernel,
)

LAM, DT, N_T, H = 1.0, 0.1, 601, 60.0
N_P, N_E = 10, 5
CRIT = FieldReconstructionCriteria(0.5, 4.0, 0.1, 0.0)
G = 9.81


def _spd(rng, scale):
    x = rng.standard_normal((6, 6))
    m = scale * (x @ x.T / 6.0 + np.eye(6))
    return 0.5 * (m + m.T)


@pytest.fixture(scope="module")
def fam():
    rng = np.random.default_rng(2027)
    D, A_inf = _spd(rng, 2.0e3), _spd(rng, 1.0e3)
    proj = rng.standard_normal((6, N_P))
    pinv = np.linalg.pinv(proj)
    null = np.eye(N_P) - pinv @ proj
    Mv = -pinv @ D + null @ (rng.standard_normal((N_P, 6)) * 300.0)
    Pa = -pinv @ A_inf + null @ (rng.standard_normal((N_P, 6)) * 100.0)
    Ev = rng.standard_normal((N_E, 6)) * 0.3
    w = np.linspace(0.05, 30.0, 600)
    ww = w[:, None, None]

    def transfer(mv, pa):
        return -(ww**2) * pa[None] + mv[None] * (ww**2 - 1j * ww * LAM) / (LAM**2 + ww**2)

    hp, he = transfer(Mv, Pa), transfer(Ev, np.zeros((N_E, 6)))
    settings = RadiationSettings(dt=DT, n_time=N_T, symmetry_tol=1e-9, passivity_tol=1e-9, zero_frequency_damping=D / LAM,
                                 time_window=H, time_window_shape="bartlett", quadrature_method="piecewise_linear_exact")
    damping = D[None] * LAM / (LAM**2 + ww**2)
    kernel = _kernel_from_arrays(
        omegas=w, added_mass=A_inf[None] - D[None] / (LAM**2 + ww**2), damping=damping, a_inf=A_inf,
        dof_keys=tuple(f"body.{d.lower()}" for d in DOF_NAMES), body_ids=("body",), settings=settings,
        provenance={"source": "analytic-family-test"})
    # ---- one posed family: geometry, request, provider
    v = rng.standard_normal((4 * N_P, 3)); v[:, 2] = -np.abs(v[:, 2]) - 0.1
    f = np.arange(4 * N_P, dtype=np.int64).reshape(N_P, 4)
    labels = ("hull",) * N_P
    request = family_request(v, f, labels, heading=0.0, water_depth=100.0, rho=1025.0, gravity=G, Q0=np.eye(3),
                             cg_world=[0.0, 0.0, -1.0])
    provider = dict(backend="capytaine", version="fake-3.0", path="/fake/capytaine/__init__.py", method="indirect",
                    recipe="native_indirect_6rad_1diff_mean_body_axes_v1", engine="Engine", green_function="Delhommeau")
    centers = v.reshape(N_P, 4, 3).mean(axis=1)
    normals = np.tile([0.0, 0.0, -1.0], (N_P, 1)); areas = np.ones(N_P)

    def snapshot(i, **over):
        args = dict(request=request, omega=float(w[i]), vertices=v, faces=f, surface_ids=labels,
                    radiation_sources=np.zeros((6, N_P), complex), diffraction_sources=np.zeros(N_P, complex),
                    added_mass=A_inf, damping=damping[i], excitation=np.zeros(6, complex), provider=provider,
                    panel_pressure_radiation=hp[i].T.copy(), panel_pressure_incident_diffraction=np.zeros(N_P, complex),
                    centers=centers, normals=normals, areas=areas)
        args.update(over)
        return FloatingFamilySnapshot(**args)

    snaps = tuple(snapshot(i) for i in range(w.size))
    pts = np.column_stack([np.linspace(-3, 3, N_E), np.linspace(2, -2, N_E), np.zeros(N_E)])

    def native(policy="native_strict", e_inf=None, abs_s=None, **over):
        args = dict(pressure_inf=Pa, elevation_inf=np.zeros((N_E, 6)) if e_inf is None else e_inf,
                    added_mass_infinite=A_inf, projection=proj, centers=centers, normals=normals, areas=areas,
                    query_points=pts, panel_ids=tuple(f"{i}:hull" for i in range(N_P)),
                    query_point_ids=tuple(f"q{i}" for i in range(N_E)), dof_names=DOF_NAMES, projection_error=0.0,
                    provenance={"source": "test"}, vertices=v, faces=f, hull_mask=np.ones(N_P, bool), surface_ids=labels,
                    sources_over_omega=np.ones((6, N_P), complex), request=request, provider=provider,
                    gf_settings={"finite_depth_prony_decomposition_method": "python"}, prony=np.ones((2, 3)),
                    prony_kh=float("inf"), prony_method="python", reference_omega=0.8,
                    boundary_policy=policy, abs_s_query=abs_s)
        args.update(over)
        return NativeFieldCoefficients(**args)

    index = {float(x): i for i, x in enumerate(w)}

    def evaluator(calls=None):
        def ev(snap, points, k):
            if calls is not None:
                calls.append((float(snap.omega), k, np.array(points)))
            i = index[float(snap.omega)]
            return he[i, :, k] * G / (1j * w[i])          # phi such that i w phi / g = H_e
        return ev

    return dict(kernel=kernel, w=w, hp=hp, he=he, snaps=snaps, snapshot=snapshot, native=native, pts=pts,
                evaluator=evaluator, proj=proj, Pa=Pa, A_inf=A_inf, D=D, Mv=Mv, Ev=Ev)


def _build(fam, **kw):
    args = dict(snapshots=fam["snaps"], native=fam["native"](), query_points=fam["pts"], kernel=fam["kernel"],
                pressure_criteria=CRIT, elevation_criteria=CRIT,
                pressure_zero_frequency=None, elevation_zero_frequency=None, evaluator=fam["evaluator"](), chunk_size=4)
    args.update(kw)
    pz = args.pop("pressure_zero_frequency"); ez = args.pop("elevation_zero_frequency")
    return build_family_radiation_field_kernel(
        args.pop("snapshots"), args.pop("native"), args.pop("query_points"),
        pressure_zero_frequency=_zero(fam, "p") if pz is None else pz,
        elevation_zero_frequency=_zero(fam, "e") if ez is None else ez, **args)


def _zero(fam, kind):
    # explicit zero-frequency sample C(0) = Mv / lam (pressure) and Ev / lam (elevation) of the analytic model
    return (fam["Mv"] if kind == "p" else fam["Ev"]) / LAM


# ----------------------------------------------------------------------------- adapter

def test_adapter_builds_the_same_kernel_as_the_generic_builder_and_binds_actual_sources(fam):
    calls = []
    fk = _build(fam, evaluator=fam["evaluator"](calls))
    generic = build_radiation_field_kernel(
        kernel=fam["kernel"], omegas=fam["w"], pressure_transfer=fam["hp"], elevation_transfer=fam["he"],
        pressure_inf=fam["Pa"], elevation_inf=np.zeros((N_E, 6)), projection=fam["proj"],
        panel_ids=tuple(f"{i}:hull" for i in range(N_P)), query_point_ids=tuple(f"q{i}" for i in range(N_E)),
        pressure_zero_frequency=_zero(fam, "p"), elevation_zero_frequency=_zero(fam, "e"),
        pressure_criteria=CRIT, elevation_criteria=CRIT, chunk_size=4)
    assert np.array_equal(fk.kernel_dense("pressure"), generic.kernel_dense("pressure"))
    assert np.allclose(fk.kernel_dense("elevation"), generic.kernel_dense("elevation"), rtol=1e-12, atol=1e-15)
    ident = fk.identity
    assert ident["binding"] == "native_family_adapter" and generic.identity["binding"] == "unbound_raw_arrays"
    assert list(ident["source_snapshot_ids"]) == [s.snapshot_id for s in fam["snaps"]]
    assert ident["native_hash"] == fam["native"]().content_hash() and ident["native_boundary_status"] == "admitted"
    assert ident["boundary_policy"] == "native_strict" and len(ident["family_radiation_key"]) == 64
    # the elevation transfer came ONLY from the retained-source evaluator at exactly the native points
    assert len(calls) == fam["w"].size * 6
    assert all(np.array_equal(p, fam["pts"]) for _, _, p in calls)
    assert fk.diagnostics["reconstruction"]["pressure"]["passed"] and fk.diagnostics["reconstruction"]["elevation"]["passed"]


def test_adapter_default_evaluator_is_the_owner_retained_source_routine(fam, monkeypatch):
    import anyhydro.floating_radiation_fields as ffr
    seen = []

    def recording(snap, points, radiation_index=None):
        seen.append(radiation_index)
        return fam["evaluator"]()(snap, points, radiation_index)

    monkeypatch.setattr(ffr, "_potentials", recording)
    fk = _build(fam, evaluator=None)
    assert len(seen) == fam["w"].size * 6 and sorted(set(seen)) == list(range(6))
    assert fk.identity["binding"] == "native_family_adapter"


def test_exact_dirichlet_policy_feeds_zero_but_keeps_raw_failure_visible(fam):
    raw = np.full((N_E, 6), 1.0e-6)                                     # raw native value, far above the 4096 eps gate
    abs_s = np.full((N_E, N_P), 1.0e-3)                                 # bound ~ 10 * 1e-3 / g >> 1e-6: consistent
    policy_native = fam["native"](policy="exact_dirichlet_z0", e_inf=raw, abs_s=abs_s)
    assert policy_native.boundary_status == "failed" and policy_native.policy_status == "admitted"
    fk = _build(fam, native=policy_native)
    assert np.all(fk.elevation_a == 0.0)                                 # effective coefficient fed to the builder
    assert np.array_equal(policy_native.elevation_inf, raw)               # raw retained untouched
    assert fk.identity["native_boundary_status"] == "failed" and fk.identity["policy_status"] == "admitted"
    assert fk.identity["boundary_policy"] == "exact_dirichlet_z0"
    strict_native = fam["native"](e_inf=raw, abs_s=abs_s)                # same raw data, default policy
    with pytest.raises(NativeBoundaryError):
        _build(fam, native=strict_native)
    inconsistent = fam["native"](policy="exact_dirichlet_z0", e_inf=np.full((N_E, 6), 1.0), abs_s=abs_s)
    with pytest.raises(NativeBoundaryError, match="inconsistent"):
        _build(fam, native=inconsistent)


@pytest.mark.parametrize("case", ["unsorted", "geometry", "provider", "points", "a_inf", "damping", "dofs", "few",
                                  "type", "nan_phi", "shape_phi", "no_pressure"])
def test_adapter_refuses_incoherent_family_data(fam, case):
    snaps = list(fam["snaps"])
    kw = {}
    if case == "unsorted":
        snaps[1], snaps[2] = snaps[2], snaps[1]
    elif case == "geometry":
        v2 = np.array(snaps[0].vertices); v2[0, 0] += 1e-6
        req2 = family_request(v2, snaps[0].faces, snaps[0].surface_ids, heading=0.0, water_depth=100.0, rho=1025.0,
                              gravity=G, Q0=np.eye(3), cg_world=[0.0, 0.0, -1.0])
        snaps[3] = fam["snapshot"](3, vertices=v2, request=req2)
    elif case == "provider":
        snaps[4] = fam["snapshot"](4, provider={**dict(snaps[4].provider), "version": "other"})
    elif case == "points":
        p = fam["pts"].copy(); p[0, 0] += 1e-9
        kw["query_points"] = p
    elif case == "a_inf":
        kw["native"] = fam["native"](added_mass_infinite=fam["A_inf"] * 1.01)
    elif case == "damping":
        snaps[-1] = fam["snapshot"](len(snaps) - 1, damping=np.asarray(snaps[-1].damping) * 1.01)
    elif case == "dofs":
        kw["native"] = fam["native"](dof_names=tuple(reversed(DOF_NAMES)))
    elif case == "few":
        snaps = snaps[:2]
    elif case == "type":
        snaps[0] = object()
    elif case == "nan_phi":
        kw["evaluator"] = lambda s, p, k: np.full(N_E, np.nan + 0j)
    elif case == "shape_phi":
        kw["evaluator"] = lambda s, p, k: np.zeros(N_E + 1, complex)
    elif case == "no_pressure":
        kw["native"] = fam["native"](query_points=fam["pts"])            # coherent native ...
        snaps[0] = fam["snapshot"](0, panel_pressure_radiation=None, panel_pressure_incident_diffraction=None,
                                   centers=None, normals=None, areas=None)   # ... but a snapshot without panel pressures
    with pytest.raises(FamilyAdapterError):
        _build(fam, snapshots=snaps, **kw)


def test_adapter_runs_the_unchanged_reconstruction_and_projection_checks(fam):
    with pytest.raises(RadiationFieldError, match="reconstruction failed"):
        _build(fam, elevation_criteria=FieldReconstructionCriteria(0.5, 4.0, 1e-12, 0.0))
    # a pressure_inf inconsistent with A_inf (projection check of the generic builder) is refused, not repaired
    bad = fam["native"](pressure_inf=fam["Pa"] * (1.0 + 1e-3))
    with pytest.raises(RadiationFieldError, match="projection of P_a"):
        _build(fam, native=bad)


def test_adapter_preflights_transfer_memory(fam):
    from anyhydro.radiation_field_kernel import FieldResourceLimits
    with pytest.raises(RadiationFieldLimitError, match="family transfers"):
        _build(fam, limits=FieldResourceLimits(max_work_bytes=100_000))


# ----------------------------------------------------------------------------- clearance guard

def test_guarded_clearance_envelope_is_pointwise_and_signals_consumed_margin():
    B = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 2.0], [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]])
    a = np.array([[1.0, -1.0, 0, 0, 0, 0.5], [-2.0, 4.0, 0, 0, 0, 0.0], [0.0, 0.0, 9.0, 9.0, 9.0, 0.0]])
    margins = np.array([3.0, 1.0])
    out = guarded_clearance(B, a, margins)
    expected = np.array([[1.0 + 2.0 * 0.5, 0.5], [2.0, 2.0], [0.0, 0.0]])    # sum_j B_ij |a_tj|
    assert np.array_equal(out.envelope, expected)
    assert np.array_equal(out.guarded_margin, np.array([[1.0, 0.5], [1.0, -1.0], [3.0, 1.0]]))
    assert out.status == "margin_consumed" and out.first_consumed == (1, 1) and out.n_samples == 3
    with pytest.raises(ClearanceConsumedError):
        out.require_clear()
    clear = guarded_clearance(B, a[[0, 2]], margins)
    assert clear.is_clear and clear.require_clear() is clear
    with pytest.raises(ValueError):
        out.envelope[0, 0] = 0.0


def test_guarded_clearance_is_independent_of_output_selection():
    rng = np.random.default_rng(3)
    B = np.abs(rng.standard_normal((4, 6))) * 1e-3
    a = rng.standard_normal((50, 6))
    m = np.abs(rng.standard_normal((50, 4))) + 0.01
    full = guarded_clearance(B, a, m)
    sel = np.array([3, 17, 49])
    part = guarded_clearance(B, a[sel], m[sel])
    assert np.array_equal(part.envelope, full.envelope[sel]) and np.array_equal(part.consumed, full.consumed[sel])


def test_guarded_clearance_refusals_and_memory_ceiling():
    B = np.ones((2, 6)); a = np.ones((3, 6))
    for bad in ((-B, a, np.ones(2)), (B, a, np.ones(3)), (B[:, :5], a, np.ones(2)), (B, a * np.nan, np.ones(2)),
                (B, a.astype(complex), np.ones(2))):
        with pytest.raises(RadiationFieldError):
            guarded_clearance(*bad)
    with pytest.raises(RadiationFieldLimitError):
        guarded_clearance(B, a, np.ones(2), max_work_bytes=10)
    with pytest.raises(RadiationFieldLimitError):
        guarded_clearance(B, a, np.ones(2), max_work_bytes=256 * 1024 * 1024 + 1)


def test_guard_uses_the_native_pointwise_bound(fam):
    abs_s = np.full((N_E, N_P), 1.0e-3)
    nat = fam["native"](policy="exact_dirichlet_z0", e_inf=np.full((N_E, 6), 1.0e-6), abs_s=abs_s)
    B = nat.pointwise_bound
    assert B.shape == (N_E, 6) and np.all(B >= np.abs(nat.elevation_inf))
    a = np.zeros((2, 6)); a[1, 2] = 1.0
    out = guarded_clearance(B, a, np.full(N_E, 1.0))
    assert np.allclose(out.envelope[1], B[:, 2]) and np.all(out.envelope[0] == 0.0) and out.is_clear
