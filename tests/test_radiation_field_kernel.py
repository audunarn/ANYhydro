"""Radiation field kernel tests against analytic exponential-memory references and independent oracles.

Model: P_v(t) = Mv exp(-lam t) per field/DOF, so C(w) = Mv lam/(lam^2+w^2), S(w) = Mv w/(lam^2+w^2) and
the native transfer is H(w) = -w^2 P_a + Mv (w^2 - i w lam)/(lam^2+w^2).  The generalized kernel is built
from B(w) = D lam/(lam^2+w^2), A(w) = A_inf - D/(lam^2+w^2) (the radiation.py exponential oracle), with
Proj @ Mv = -D and Proj @ P_a = -A_inf (signed physical-pressure projection).
"""
import functools

import numpy as np
import pytest

from anyhydro.radiation import RadiationSettings, _kernel_from_arrays
from anyhydro.radiation_field_kernel import (
    FieldReconstructionCriteria,
    FieldResourceLimits,
    RadiationFieldError,
    RadiationFieldKernel,
    RadiationFieldLimitError,
    RadiationFieldSerializationError,
    build_radiation_field_kernel,
    field_history_direct,
    field_history_fft,
    iter_field_history,
    load_radiation_field_kernel,
)

LAM, DT, N_T, H = 1.0, 0.1, 601, 60.0     # 60 s horizon; Nyquist 31 rad/s
N_P, N_E = 10, 5
CRIT = FieldReconstructionCriteria(0.5, 4.0, 0.1, 0.0)


def _spd(rng, scale):
    x = rng.standard_normal((6, 6))
    m = scale * (x @ x.T / 6.0 + np.eye(6))
    return 0.5 * (m + m.T)


@pytest.fixture(scope="module")
def model():
    rng = np.random.default_rng(2026)
    D, A_inf = _spd(rng, 2.0e3), _spd(rng, 1.0e3)
    proj = rng.standard_normal((6, N_P))
    pinv = np.linalg.pinv(proj)
    null = np.eye(N_P) - pinv @ proj                     # panel pressure patterns that carry no net wrench
    Mv = -pinv @ D + null @ (rng.standard_normal((N_P, 6)) * 300.0)
    Pa = -pinv @ A_inf + null @ (rng.standard_normal((N_P, 6)) * 100.0)
    Ev = rng.standard_normal((N_E, 6)) * 0.3
    Ea = np.zeros((N_E, 6))                              # no instantaneous radiation elevation
    w = np.linspace(0.05, 30.0, 600)

    def transfer(mv, pa):
        ww = w[:, None, None]
        return -(ww**2) * pa[None] + mv[None] * (ww**2 - 1j * ww * LAM) / (LAM**2 + ww**2)

    settings = RadiationSettings(
        dt=DT, n_time=N_T, symmetry_tol=1e-9, passivity_tol=1e-9, zero_frequency_damping=D / LAM,
        time_window=H, time_window_shape="bartlett", quadrature_method="piecewise_linear_exact")
    ww = w[:, None, None]
    kernel = _kernel_from_arrays(
        omegas=w, added_mass=A_inf[None] - D[None] / (LAM**2 + ww**2), damping=D[None] * LAM / (LAM**2 + ww**2),
        a_inf=A_inf, dof_keys=tuple(f"body.{d}" for d in ("surge", "sway", "heave", "roll", "pitch", "yaw")),
        body_ids=("body",), settings=settings, provenance={"source": "analytic-exponential-test"})
    args = dict(
        kernel=kernel, omegas=w, pressure_transfer=transfer(Mv, Pa), pressure_inf=Pa, projection=proj,
        panel_ids=[f"p{i}" for i in range(N_P)], elevation_transfer=transfer(Ev, Ea), elevation_inf=Ea,
        query_point_ids=[f"q{i}" for i in range(N_E)], pressure_zero_frequency=Mv / LAM,
        elevation_zero_frequency=Ev / LAM,
        pressure_criteria=CRIT, elevation_criteria=CRIT,
        chunk_size=4, identity={"body": "hull-1", "pose": "mean", "mesh": "m-abc"})
    fk = build_radiation_field_kernel(**args)
    return dict(args=args, fk=fk, kernel=kernel, w=w, D=D, A_inf=A_inf, Mv=Mv, Pa=Pa, Ev=Ev, proj=proj)


def _trajectory(n, dt=DT):
    t = np.arange(n) * dt
    v = np.stack([np.sin(0.7 * t + k) * (1 + 0.1 * k) for k in range(6)], axis=1)
    a = np.stack([0.7 * np.cos(0.7 * t + k) * (1 + 0.1 * k) for k in range(6)], axis=1)
    return a, v


# ----------------------------------------------------------------------------- construction and physics

def test_chunked_layout_identity_and_qualification(model):
    fk = model["fk"]
    assert len(fk.pressure_chunks) == 3 and fk.pressure_chunks[0].shape == (N_T, 4, 6)
    assert fk.pressure_chunks[2].shape == (N_T, 2, 6)
    assert fk.n_time == N_T and fk.dt == pytest.approx(DT)
    assert fk.kernel_hash == model["kernel"].content_hash()
    assert fk.identity["mesh"] == "m-abc" and fk.qualification_status == "unqualified"
    assert fk.kernel_dense("pressure").shape == (N_T, N_P, 6)
    assert np.all(fk.pressure_chunks[0][-1] == 0.0) and np.all(fk.elevation_chunks[0][-1] == 0.0)  # Bartlett endpoint


def test_kernel_matches_windowed_analytic_exponential_away_from_t0(model):
    fk, t = model["fk"], np.arange(N_T) * DT
    exact = model["Mv"][None] * (np.exp(-LAM * t) * np.clip(1.0 - t / H, 0, None))[:, None, None]
    got = fk.kernel_dense("pressure")
    sel = (t > 0.5) & (t < 15.0)           # band-limited at 60 rad/s: near-t0 truncation error excluded
    scale = np.max(np.abs(exact[sel]))
    assert np.max(np.abs(got[sel] - exact[sel])) < 2e-2 * scale


def test_signed_projection_gives_minus_ainf_and_minus_kr(model):
    fk, kernel, proj = model["fk"], model["kernel"], model["proj"]
    assert np.allclose(proj @ fk.pressure_a, -kernel.added_mass_infinite, rtol=0, atol=1e-8 * np.max(np.abs(kernel.added_mass_infinite)))
    proj_k = np.einsum("ip,tpj->tij", proj, fk.kernel_dense("pressure"))
    assert np.max(np.abs(proj_k + kernel.kernel)) < 1e-8 * np.max(np.abs(kernel.kernel))
    assert fk.diagnostics["projection_kernel_error"] < 1e-8 * np.max(np.abs(kernel.kernel))
    # elevation has no projection and no instantaneous part here
    assert np.all(fk.elevation_a == 0.0)


def test_projected_transfer_equals_global_reconstruction_exactly(model):
    """Proj H_rec = w^2 A_rec + i w B_rec: independent identity against RadiationKernel.reconstruct."""
    fk, kernel, proj = model["fk"], model["kernel"], model["proj"]
    w = np.array([0.3, 1.0, 2.5, 7.0, 20.0])
    a_rec, b_rec = kernel.reconstruct(w)
    lhs = np.einsum("ip,wpj->wij", proj, fk.reconstruct("pressure", w))
    rhs = (w**2)[:, None, None] * a_rec + 1j * w[:, None, None] * b_rec
    assert np.max(np.abs(lhs - rhs)) < 1e-9 * np.max(np.abs(rhs))


def test_pressure_and_elevation_reconstruct_the_native_transfers(model):
    fk, w = model["fk"], model["w"]
    for kind, key in (("pressure", "pressure_transfer"), ("elevation", "elevation_transfer")):
        sel = (w >= 0.5) & (w <= 4.0)
        rep = fk.reconstruction_report(kind, w, model["args"][key], CRIT)
        assert rep["passed"] and rep["max_rel_error"] < 0.1 and rep["n_frequencies"] == int(sel.sum())
        assert fk.diagnostics["reconstruction"][kind]["passed"]


def test_global_wrench_agreement_alone_does_not_admit_a_wrong_spatial_pressure(model):
    """A nullspace panel pattern leaves Proj unchanged but must fail the full panel-transfer reconstruction."""
    args = dict(model["args"])
    proj = model["proj"]
    null = np.eye(N_P) - np.linalg.pinv(proj) @ proj
    bad = np.array(args["pressure_transfer"]) + (null @ np.ones((N_P, 6)) * 1e3)[None]
    args["pressure_transfer"] = bad
    with pytest.raises(RadiationFieldError, match="panel-pressure transfer reconstruction failed"):
        build_radiation_field_kernel(**args)


# ----------------------------------------------------------------------------- convolution

def test_fft_matches_direct_oracle_on_history_longer_than_the_window(model):
    fk = model["fk"]
    a, v = _trajectory(900)                       # 45 s record versus the 30 s horizon
    for kind, n_f in (("pressure", N_P), ("elevation", N_E)):
        direct = field_history_direct(fk, kind, a, v)
        fast = field_history_fft(fk, kind, a, v)
        assert direct.shape == fast.shape == (900, n_f)
        assert np.max(np.abs(fast - direct)) <= 1e-10 * np.max(np.abs(direct))


def test_block_fft_and_chunking_do_not_change_the_result(model):
    fk = model["fk"]
    a, v = _trajectory(750)
    full = field_history_fft(fk, "pressure", a, v)
    blocked = field_history_fft(fk, "pressure", a, v, block_length=100)
    assert np.max(np.abs(blocked - full)) <= 1e-11 * np.max(np.abs(full))
    pieces = [part for _, part in iter_field_history(fk, "pressure", a, v, block_length=256)]
    assert len(pieces) == 3 and np.allclose(np.concatenate(pieces, axis=1), full, rtol=0, atol=1e-11 * np.max(np.abs(full)))


def test_discrete_rule_matches_numpy_convolve_for_one_field(model):
    fk = model["fk"]
    a, v = _trajectory(200)
    k = fk.kernel_dense("pressure")
    f = 3
    expected = a @ fk.pressure_a[f]
    for dof in range(6):
        full = np.convolve(k[:, f, dof], v[:, dof])[:200]
        mem = full - 0.5 * k[:200, f, dof] * v[0, dof] - 0.5 * k[0, f, dof] * v[:, dof]
        expected = expected + DT * mem
    expected[0] = a[0] @ fk.pressure_a[f]            # memory_0 = 0
    got = field_history_fft(fk, "pressure", a, v)[:, f]
    assert np.max(np.abs(got - expected)) <= 1e-10 * np.max(np.abs(expected))


def test_projected_pressure_history_equals_minus_global_added_mass_and_memory(model):
    fk, kernel, proj = model["fk"], model["kernel"], model["proj"]
    a, v = _trajectory(400)
    n = 400
    memory = np.zeros((n, 6))
    for i in range(6):
        for j in range(6):
            full = np.convolve(kernel.kernel[:, i, j], v[:, j])[:n]
            memory[:, i] += DT * (full - 0.5 * kernel.kernel[:n, i, j] * v[0, j] - 0.5 * kernel.kernel[0, i, j] * v[:, j])
    memory[0] = 0.0
    expected = -(a @ kernel.added_mass_infinite.T + memory)
    got = field_history_fft(fk, "pressure", a, v) @ proj.T
    assert np.max(np.abs(got - expected)) <= 1e-9 * np.max(np.abs(expected))


def test_history_longer_than_unwindowed_kernel_is_refused(model):
    fk = model["fk"]
    short = RadiationFieldKernel(
        kernel_hash=fk.kernel_hash, settings=fk.settings, dof_keys=fk.dof_keys, time=fk.time[:100],
        panel_ids=fk.panel_ids, query_point_ids=fk.query_point_ids, pressure_a=fk.pressure_a,
        pressure_chunks=tuple(c[:100] for c in fk.pressure_chunks), elevation_a=fk.elevation_a,
        elevation_chunks=tuple(c[:100] for c in fk.elevation_chunks), projection=fk.projection,
        added_mass_infinite=fk.added_mass_infinite, diagnostics={}, identity={}, chunk_size=fk.chunk_size)
    a, v = _trajectory(300)
    with pytest.raises(RadiationFieldError, match="zero windowed endpoint"):
        field_history_fft(short, "pressure", a, v)
    with pytest.raises(RadiationFieldError, match="zero windowed endpoint"):
        field_history_direct(short, "pressure", a, v)


# ----------------------------------------------------------------------------- admission

def test_build_refusals(model):
    args = model["args"]
    w = model["w"]
    with pytest.raises(RadiationFieldError, match="exactly"):
        build_radiation_field_kernel(**{**args, "omegas": w[:-1], "pressure_transfer": args["pressure_transfer"][:-1],
                                        "elevation_transfer": args["elevation_transfer"][:-1]})
    with pytest.raises(RadiationFieldError, match="projection of P_a"):
        build_radiation_field_kernel(**{**args, "pressure_inf": args["pressure_inf"] * (1.0 + 1e-3)})
    with pytest.raises(RadiationFieldError, match="required"):
        build_radiation_field_kernel(**{**args, "pressure_zero_frequency": None})
    with pytest.raises(RadiationFieldError, match="reconstruction failed"):
        build_radiation_field_kernel(**{**args, "elevation_criteria": FieldReconstructionCriteria(0.5, 4.0, 1e-12, 0.0)})
    with pytest.raises(RadiationFieldError, match="pressure_inf"):
        build_radiation_field_kernel(**{**args, "pressure_inf": np.full((N_P, 6), np.nan)})


def test_resource_limits_refuse_before_allocation(model):
    args = model["args"]
    with pytest.raises(RadiationFieldLimitError, match="panels"):
        build_radiation_field_kernel(**{**args, "limits": FieldResourceLimits(max_panels=N_P - 1)})
    with pytest.raises(RadiationFieldLimitError, match="storage"):
        build_radiation_field_kernel(**{**args, "limits": FieldResourceLimits(max_artifact_bytes=10_000)})
    with pytest.raises(RadiationFieldLimitError, match="work"):
        build_radiation_field_kernel(**{**args, "limits": FieldResourceLimits(max_work_bytes=10_000)})
    fk = model["fk"]
    a, v = _trajectory(600)
    tight = FieldResourceLimits(max_work_bytes=50_000)
    with pytest.raises(RadiationFieldLimitError):
        field_history_fft(fk, "pressure", a, v, limits=tight)
    with pytest.raises(RadiationFieldLimitError):
        fk.kernel_dense("pressure", tight)


def test_limits_and_projection_tolerance_can_only_tighten(model):
    for kwargs in (dict(max_panels=601), dict(max_query_points=2001), dict(max_work_bytes=256 * 1024**2 + 1),
                   dict(max_artifact_bytes=2 * 1024**3 + 1), dict(max_panels=0), dict(max_panels=True)):
        with pytest.raises(RadiationFieldLimitError):
            FieldResourceLimits(**kwargs)
    FieldResourceLimits(max_panels=100, max_work_bytes=1024)           # tightening is allowed
    for tol in (1.0e-7, 0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(RadiationFieldError, match="projection_rel_tol"):
            build_radiation_field_kernel(**{**model["args"], "projection_rel_tol": tol})


def test_default_ceilings_are_the_plan_values():
    lim = FieldResourceLimits()
    assert lim.max_panels == 600 and lim.max_work_bytes == 256 * 1024 * 1024 and lim.max_artifact_bytes == 2 * 1024**3


# ----------------------------------------------------------------------------- integrity

def test_payload_is_immutable_and_isolated(model):
    fk = model["fk"]
    with pytest.raises(ValueError):
        fk.pressure_chunks[0][0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        fk.pressure_a.setflags(write=True)
    with pytest.raises(TypeError):
        fk.identity["mesh"] = "other"
    src = np.array(model["args"]["pressure_inf"])
    assert fk.pressure_a is not src


def test_save_load_roundtrip_preserves_hash_and_arrays(model, tmp_path):
    fk = model["fk"]
    path = str(tmp_path / "field.npz")
    digest = fk.save(path)
    loaded = load_radiation_field_kernel(path, expected_kernel_hash=fk.kernel_hash)
    assert loaded.content_hash() == digest == fk.content_hash()
    assert np.array_equal(loaded.kernel_dense("pressure"), fk.kernel_dense("pressure"))
    assert np.array_equal(loaded.elevation_a, fk.elevation_a) and loaded.panel_ids == fk.panel_ids
    assert loaded.diagnostics["reconstruction"]["pressure"]["passed"]
    a, v = _trajectory(250)
    assert np.array_equal(field_history_fft(loaded, "pressure", a, v), field_history_fft(fk, "pressure", a, v))
    with pytest.raises(RadiationFieldSerializationError, match="different generalized"):
        load_radiation_field_kernel(path, expected_kernel_hash="0" * 64)


def test_tampering_truncation_and_ceiling_are_detected(model, tmp_path):
    fk = model["fk"]
    path = tmp_path / "field.npz"
    fk.save(str(path))
    with np.load(str(path), allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["pressure_chunk_1"] = data["pressure_chunk_1"].copy()
    data["pressure_chunk_1"][10, 0, 0] += 1e-9           # a single-sample payload mutation
    tampered = tmp_path / "tampered.npz"
    np.savez(str(tampered), **data)
    with pytest.raises(RadiationFieldSerializationError, match="hash mismatch"):
        load_radiation_field_kernel(str(tampered))
    cut = tmp_path / "cut.npz"
    cut.write_bytes(path.read_bytes()[:2000])
    with pytest.raises(RadiationFieldSerializationError):
        load_radiation_field_kernel(str(cut))
    with pytest.raises(RadiationFieldLimitError):
        load_radiation_field_kernel(str(path), limits=FieldResourceLimits(max_artifact_bytes=1000))
    with pytest.raises(RadiationFieldLimitError):
        fk.save(str(tmp_path / "x.npz"), limits=FieldResourceLimits(max_artifact_bytes=1000))
    assert not (tmp_path / "x.npz").exists()


def test_hash_depends_on_identity_and_values(model):
    args = model["args"]
    other = build_radiation_field_kernel(**{**args, "identity": {"body": "hull-2"}})
    assert other.content_hash() != model["fk"].content_hash()
    assert other.kernel_hash == model["fk"].kernel_hash


# ----------------------------------------------------------------------------- native omega = inf helper (fake API)

class _SM:
    """Minimal stand-in for capytaine.tools.symbolic_multiplication.SymbolicMultiplication."""

    def __init__(self, symbol, value=1.0):
        self.symbol, self.value = symbol, value

    @property
    def real(self):
        return _SM(self.symbol, np.real(self.value))

    def __neg__(self):
        return _SM(self.symbol, -self.value)

    def __mul__(self, x):
        return _SM(self.symbol, self.value * x)

    def __rmul__(self, x):
        return _SM(self.symbol, x * self.value)

    def __truediv__(self, x):
        if hasattr(x, "symbol") and x.symbol == self.symbol:
            return self.value / x.value
        return _SM(self.symbol, self.value / x)

    __array_ufunc__ = None            # numpy defers to the reflected operators below

    def __rtruediv__(self, x):        # finite / infinite -> symbolic zero, as in Capytaine
        return _SM("zero", x / self.value)

    def __float__(self):
        return float("inf") * float(self.value) if self.symbol == "inf" else 0.0


class _Mesh:
    def __init__(self, v, f, centers, normals, areas):
        self.vertices, self.faces = v, f
        self.faces_centers, self.faces_normals, self.faces_areas = centers, normals, areas
        self.nb_faces = len(f)


class _Body:
    def __init__(self, mesh, lid_extra=0):
        self.mesh = mesh
        self.mesh_including_lid = mesh if not lid_extra else _Mesh(mesh.vertices, np.vstack([mesh.faces] * 2), *[
            np.zeros((0, 3))] * 2, np.zeros(0))
        self.hull_mask = np.ones(mesh.nb_faces, dtype=bool)


class _GF:
    exportable_settings = {"green_function": "FakeGF", "finite_depth_prony_decomposition_method": "python"}

    @functools.lru_cache(maxsize=None)
    def find_best_exponential_decomposition(self, kh):
        return np.array([[1.0, 2.0, 3.0], [0.5, 0.25, 0.125]])


class _Problem:
    def __init__(self, *, body, omega, radiating_dof, water_depth, rho, g):
        self.body, self.omega, self.radiating_dof = body, omega, radiating_dof
        self.water_depth, self.rho, self.g = water_depth, rho, g
        inf = bool(np.isinf(omega))
        self.encounter_omega = _SM("inf", 1.0) if inf else omega
        self.wavenumber = _SM("inf", 1.0) if inf else 1.0

    def make_results_container(self, sources):
        res = type("Res", (), {})()
        res.sources, res.radiating_dof, res.encounter_omega = sources, self.radiating_dof, self.encounter_omega
        return res


NATIVE_DOFS = ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")


class _Native:
    """Fake `capytaine` module + solver exposing exactly the calls the helper makes."""

    def __init__(self, mesh, p_true, e_true, a_true, g, drop=None, plain_pressure=False, s_true=None, phi_true=None,
                 boundary_offset=0.0, s_scale=0.0):
        self.mesh, self.p, self.e, self.a, self.gr = mesh, p_true, e_true, a_true, g
        self.boundary_offset = boundary_offset
        self.s_scale = s_scale             # discrete |S(z0)| magnitude (continuum S(z0) = 0)
        self.events = []
        self.drop, self.plain_pressure = drop, plain_pressure
        rng = np.random.default_rng(3)
        n = mesh.nb_faces
        self.s = (rng.standard_normal((6, n)) + 1j * rng.standard_normal((6, n))) if s_true is None else s_true
        self.phi = (rng.standard_normal((6, n)) + 1j * rng.standard_normal((6, n))) if phi_true is None else phi_true
        self.calls, self.solve_kwargs, self.problems, self.potential_calls, self.gf_args = [], [], [], 0, []
        self.__version__ = "fake-3.0"
        self.__file__ = "fake/capytaine/__init__.py"
        self.engine = type("Engine", (), {})()
        self.engine.green_function = _GF()
        self.engine.build_S_matrix = self._build_S
        self.s_calls = []
        self.RadiationProblem = self._problem

    def _S(self, points):
        """Deterministic real influence matrix at the points (shape n_points x n_src)."""
        pts = np.asarray(points, dtype=float)
        n_src = self.mesh.nb_faces
        return self.s_scale * (1.0 + np.abs(pts[:, :1]) + 0.1 * np.arange(n_src)[None, :])

    def _build_S(self, points, mesh, **params):
        self.events.append("S")
        self.s_calls.append((np.array(points), params))
        return self._S(points)

    def BEMSolver(self, green_function=None):
        self.gf_args.append(green_function)
        return self

    def _problem(self, **kw):
        p = _Problem(**kw)
        self.problems.append(p)
        return p

    def solve(self, problem, **kw):
        self.solve_kwargs.append(kw)
        self.engine.green_function.find_best_exponential_decomposition(float("inf"))   # as the real engine does
        j = NATIVE_DOFS.index(problem.radiating_dof)
        col = -self.p[:, j].astype(complex)
        # plain_pressure models an unbalanced symbolic factor: only ONE infinite factor on the pressure
        pressure = _SM("inf", col) if self.plain_pressure else _SM("inf", _SM("inf", col))
        res = type("Res", (), {})()
        res.sources = None if self.drop == "sources" else _SM("inf", self.s[j])
        res.potential = _SM("inf", self.phi[j])
        res.pressure = pressure
        res.encounter_omega = problem.encounter_omega
        res.radiating_dof = problem.radiating_dof
        res.added_mass = {d: float(self.a[i, j]) for i, d in enumerate(NATIVE_DOFS)}
        return res

    def compute_potential(self, points, res):
        """Real potential on the free surface: the infinite-frequency acceleration coefficient is then exactly 0."""
        self.potential_calls += 1
        self.events.append("phi")
        src = np.asarray(res.sources.value)
        # real free-surface part (contributes nothing to E) + the discrete S(z0) @ sigma (raw E = Im(S sigma)/g)
        phi = np.real(src[0]) * np.asarray(points)[:, 0] + np.real(src[-1]) + self._S(points) @ src
        return _SM("inf", phi + 1j * self.boundary_offset if self.boundary_offset else phi)


def _native_fixture(n_q=3, perturb_a=0.0):
    rng = np.random.default_rng(7)
    v = rng.standard_normal((12, 3))
    f = np.array([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]])
    centers = np.array([[1.0, 0.0, -1.0], [0.0, 2.0, -0.5], [-1.0, 0.5, -2.0]])
    normals = np.array([[1.0, 0, 0], [0, 1.0, 0], [0.0, 0.6, -0.8]])
    areas = np.array([2.0, 3.0, 1.5])
    Q0 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    cg = np.array([0.1, -0.2, -1.0])
    # independent projection: generalized force of DOF j is -p (n . u_j) dS
    proj = np.zeros((6, 3))
    for p in range(3):
        for j in range(3):
            proj[j, p] = -areas[p] * float(normals[p] @ Q0[:, j])
            u = np.cross(Q0[:, j], centers[p] - cg)
            proj[3 + j, p] = -areas[p] * float(normals[p] @ u)
    p_true = rng.standard_normal((3, 6)) * 50.0
    e_true = np.zeros((n_q, 6))     # the infinite-frequency free-surface acceleration coefficient is exactly zero at z = 0
    a_true = -(proj @ p_true) + perturb_a
    pts = np.column_stack([np.linspace(-3, 3, n_q), np.linspace(1, -1, n_q), np.zeros(n_q)])
    mesh = _Mesh(v, f, centers, normals, areas)
    return dict(v=v, f=f, mesh=mesh, Q0=Q0, cg=cg, proj=proj, p=p_true, e=e_true, a=a_true, pts=pts)


def _helper(fx, fake=None, **kw):
    from anyhydro.radiation_field_kernel import native_infinite_frequency_field_coefficients
    fake = fake or _Native(fx["mesh"], fx["p"], fx["e"], fx["a"], 9.81)
    args = dict(vertices=fx["v"], faces=fx["f"], surface_ids=["hull"] * 3, query_points=fx["pts"], water_depth=100.0,
                rho=1025.0, gravity=9.81, Q0=fx["Q0"], cg_world=fx["cg"], body=_Body(fx["mesh"]), cpt=fake)
    args.update(kw)
    return native_infinite_frequency_field_coefficients(**args), fake


def test_native_helper_recovers_symbolically_normalised_finite_coefficients():
    fx = _native_fixture()
    out, fake = _helper(fx)
    assert np.allclose(out.pressure_inf, fx["p"], rtol=0, atol=1e-12)
    assert np.allclose(out.elevation_inf, fx["e"], rtol=0, atol=1e-13)
    assert np.allclose(out.added_mass_infinite, fx["a"])
    assert np.allclose(out.projection, fx["proj"], atol=1e-14)
    assert out.projection_error < 1e-8 * np.max(np.abs(fx["a"]))
    assert out.qualification_status == "unqualified" and out.provenance["method"] == "indirect"
    assert out.query_point_ids == ("q0", "q1", "q2") and out.panel_ids == ("0:hull", "1:hull", "2:hull")
    assert np.array_equal(out.query_points, fx["pts"])           # actual waterline points, order retained
    # complete retained source: both parts of sources/omega and potential/omega, geometry, GF state, provider
    assert out.sources_over_omega.dtype == np.complex128 and np.abs(out.sources_over_omega.imag).max() > 0
    assert np.array_equal(out.sources_over_omega, fake.s) and np.array_equal(out.potential_over_omega, fake.phi)
    assert out.is_recoverable and np.array_equal(out.vertices, fx["v"]) and out.hull_mask.all()
    assert out.provider["version"] == "fake-3.0" and out.gf_settings["finite_depth_prony_decomposition_method"] == "python"
    assert out.prony.shape == (2, 3) and np.isinf(out.prony_kh) and out.prony_method == "python"
    # boundary condition: observed residual reported (never clamped) against the declared roundoff scale
    assert np.max(out.boundary_residual) == 0.0 and np.all(out.boundary_residual <= out.boundary_limit)
    L = np.max(np.ptp(fx["v"], axis=0)); R = np.max(np.linalg.norm(fx["v"] - fx["cg"], axis=1)); eps = np.finfo(float).eps
    assert np.allclose(out.boundary_limit, 4096 * eps * np.array([L / 9.81] * 3 + [L * R / 9.81] * 3), rtol=1e-12)


def test_native_helper_issues_the_planned_omega_inf_indirect_solves():
    fx = _native_fixture()
    out, fake = _helper(fx)
    assert [p.radiating_dof for p in fake.problems] == list(NATIVE_DOFS)
    assert all(np.isinf(p.omega) for p in fake.problems)
    assert all((p.water_depth, p.rho, p.g) == (100.0, 1025.0, 9.81) for p in fake.problems)
    assert all(kw == dict(method="indirect", keep_details=True, n_threads=1) for kw in fake.solve_kwargs)
    assert len(fake.solve_kwargs) == 6


def test_native_helper_result_is_immutable_with_stable_hash():
    fx = _native_fixture()
    a, _ = _helper(fx)
    b, _ = _helper(fx)
    assert a.content_hash() == b.content_hash()
    for arr in (a.pressure_inf, a.elevation_inf, a.projection):
        with pytest.raises(ValueError):
            arr[0, 0] = 1.0
    with pytest.raises(TypeError):
        a.provenance["method"] = "direct"
    assert set(a.builder_arrays()) == {"pressure_inf", "elevation_inf", "projection", "panel_ids", "query_point_ids"}


def test_native_helper_refuses_inconsistent_or_unsafe_native_data():
    fx = _native_fixture()
    with pytest.raises(RadiationFieldError, match="signed integration"):
        _helper(_native_fixture(perturb_a=1.0))                    # A_inf of the same solves disagrees with the integral
    with pytest.raises(RadiationFieldError, match="symbolic"):
        _helper(fx, _Native(fx["mesh"], fx["p"], fx["e"], fx["a"], 9.81, plain_pressure=True))   # inf factor left over
    with pytest.raises(RadiationFieldError, match="retain"):
        _helper(fx, _Native(fx["mesh"], fx["p"], fx["e"], fx["a"], 9.81, drop="sources"))
    bad_pts = fx["pts"].copy(); bad_pts[0, 2] = -0.01
    with pytest.raises(RadiationFieldError, match="waterline"):
        _helper(fx, query_points=bad_pts)
    with pytest.raises(RadiationFieldError, match="exactly one"):
        _helper(fx, body=None)
    with pytest.raises(RadiationFieldError, match="exactly one"):
        _helper(fx, reference_omega=0.5)
    with pytest.raises(RadiationFieldError, match="projection_rel_tol"):
        _helper(fx, projection_rel_tol=1e-6)
    with pytest.raises(RadiationFieldLimitError):
        _helper(fx, limits=FieldResourceLimits(max_panels=2))
    wrong = _Mesh(fx["v"] + 1e-9, fx["f"], fx["mesh"].faces_centers, fx["mesh"].faces_normals, fx["mesh"].faces_areas)
    with pytest.raises(RadiationFieldError, match="geometry"):
        _helper(fx, body=_Body(wrong))
    with pytest.raises(RadiationFieldError, match="lid"):
        _helper(fx, body=_Body(fx["mesh"], lid_extra=1))


def test_native_helper_output_feeds_the_builder_arrays():
    """The helper's names/shapes are exactly what build_radiation_field_kernel consumes."""
    fx = _native_fixture()
    out, _ = _helper(fx)
    arrays = out.builder_arrays()
    assert arrays["pressure_inf"].shape == (3, 6) and arrays["elevation_inf"].shape == (3, 6)
    assert arrays["projection"].shape == (6, 3)
    assert np.allclose(arrays["projection"] @ arrays["pressure_inf"], -out.added_mass_infinite, atol=1e-8)


def _patched_family(monkeypatch, fx):
    """Route the family body preparation to the fake body (real preparation needs native Capytaine)."""
    import anyhydro.floating_radiation_fields as ffr
    calls = []

    def fake_body(cpt, vertices, faces, surface_ids, request, omega):
        calls.append(omega)
        return fx["v"], fx["f"], tuple(surface_ids), fx["mesh"], _Body(fx["mesh"])

    monkeypatch.setattr(ffr, "_body_for_family", fake_body)
    return calls


def test_reference_omega_prepares_the_body_and_never_passes_infinity(monkeypatch):
    fx = _native_fixture()
    calls = _patched_family(monkeypatch, fx)
    out, fake = _helper(fx, body=None, reference_omega=0.8)
    assert calls == [0.8] and out.reference_omega == 0.8
    assert all(np.isinf(p.omega) for p in fake.problems)


def test_recovery_uses_retained_sources_and_replayed_green_function_with_zero_solves(monkeypatch):
    fx = _native_fixture()
    _patched_family(monkeypatch, fx)
    out, fake = _helper(fx, body=None, reference_omega=0.8)
    solves_before = len(fake.solve_kwargs)
    new_pts = np.array([[1.5, 0.5, 0.0], [-2.0, 1.0, 0.0]])
    seen = []

    def replay(cpt, view, problem):
        seen.append(view)
        return ("replayed", view.prony_kh)

    from anyhydro.radiation_field_kernel import recover_waterline_coefficients
    rec = recover_waterline_coefficients(out, new_pts, cpt=fake, replay=replay, query_point_ids=("a", "b"))
    assert len(fake.solve_kwargs) == solves_before            # ZERO solver.solve calls
    assert fake.potential_calls == 6 + 6                         # six in the capture, six in the recovery
    assert fake.gf_args[-6:] == [("replayed", np.inf)] * 6        # the retained decomposition is replayed, not refitted
    assert np.array_equal(seen[0].prony, out.prony) and np.isinf(seen[0].prony_kh)
    assert rec.elevation_inf.shape == (2, 6) and np.max(np.abs(rec.elevation_inf)) == 0.0
    assert rec.query_point_ids == ("a", "b") and rec.native_hash == out.content_hash() and rec.solve_calls == 0
    with pytest.raises(ValueError):
        rec.elevation_inf[0, 0] = 1.0
    with pytest.raises(AttributeError):
        rec.solve_calls = 3
    # guards: wrong provider, z != 0, missing reference omega
    other = _Native(fx["mesh"], fx["p"], fx["e"], fx["a"], 9.81)
    other.__version__ = "other"
    with pytest.raises(RadiationFieldError, match="provider"):
        recover_waterline_coefficients(out, new_pts, cpt=other, replay=replay)
    bad = new_pts.copy(); bad[0, 2] = 0.1
    with pytest.raises(RadiationFieldError, match="z = 0"):
        recover_waterline_coefficients(out, bad, cpt=fake, replay=replay)
    no_ref, _ = _helper(fx)
    with pytest.raises(RadiationFieldError, match="reference_omega"):
        recover_waterline_coefficients(no_ref, new_pts, cpt=fake, replay=replay)


def test_native_hash_covers_every_identity_and_id(monkeypatch):
    fx = _native_fixture()
    base, _ = _helper(fx)
    renamed_q, _ = _helper(fx, query_point_ids=("x", "y", "z"))
    assert renamed_q.content_hash() != base.content_hash()
    renamed_p, _ = _helper(fx, surface_ids=["hull", "hull", "lid"])
    assert renamed_p.content_hash() != base.content_hash()
    from dataclasses import replace
    assert replace(base, panel_ids=("a", "b", "c")).content_hash() != base.content_hash()
    assert replace(base, prony_method="fortran").content_hash() != base.content_hash()
    assert replace(base, prony=base.prony * 1.0000001).content_hash() != base.content_hash()
    assert replace(base, normals=base.normals[::-1]).content_hash() != base.content_hash()
    sr = np.array(base.sources_over_omega); sr[0, 0] += 1j * 1e-12
    assert replace(base, sources_over_omega=sr).content_hash() != base.content_hash()
    with pytest.raises(RadiationFieldError):
        replace(base, query_point_ids=("q0", "q0", "q1"))


def test_native_roundtrip_preserves_complex_parts_inf_and_hash(tmp_path):
    from anyhydro.radiation_field_kernel import load_native_field_coefficients
    fx = _native_fixture()
    out, _ = _helper(fx)
    path = str(tmp_path / "native.npz")
    digest = out.save(path)
    back = load_native_field_coefficients(path, expected_hash=digest)
    assert back.content_hash() == digest == out.content_hash()
    assert np.array_equal(back.sources_over_omega, out.sources_over_omega) and back.sources_over_omega.dtype == np.complex128
    assert np.array_equal(back.potential_over_omega.imag, out.potential_over_omega.imag)
    assert np.isinf(back.prony_kh) and back.prony_kh > 0 and back.hull_mask.dtype == bool and back.is_recoverable
    assert back.panel_ids == out.panel_ids and back.provider == out.provider and back.gf_settings == out.gf_settings
    with pytest.raises(RadiationFieldError, match="stale"):
        load_native_field_coefficients(path, expected_hash="0" * 64)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    data["pressure_inf"] = data["pressure_inf"].copy(); data["pressure_inf"][0, 0] += 1e-9
    bad = str(tmp_path / "bad.npz"); np.savez(bad, **data)
    with pytest.raises(RadiationFieldError, match="hash mismatch"):
        load_native_field_coefficients(bad)
    data["unexpected"] = np.zeros(4096)
    extra = str(tmp_path / "extra.npz"); np.savez_compressed(extra, **data)
    with pytest.raises(RadiationFieldError, match="unexpected"):
        load_native_field_coefficients(extra)


def test_complex_sources_are_never_cast_to_real_and_symbolic_leftovers_are_refused():
    from anyhydro.radiation_field_kernel import _plain_complex
    z = _plain_complex(np.array([1 + 2j, -3j]), "z")
    assert z.dtype == np.complex128 and z[1] == -3j
    with pytest.raises(RadiationFieldError, match="symbolic"):
        _plain_complex(_SM("inf", np.ones(2)), "z")
    with pytest.raises(RadiationFieldError, match="object"):
        _plain_complex(np.array([_SM("zero", 1.0)], dtype=object), "z")


class _BoundResult:
    """Duck-typed bound transient result (what ANYsolver's FloatingTransientResult provides)."""

    def __init__(self, fk, a, v, **kw):
        self.status, self.complete, self.frame, self.digest_protocol = "complete", True, "body_attached_first_order", "numeric-array-v1"
        self.kernel_digest, self.ainf_digest, self.dt = fk.kernel_digest, fk.ainf_digest, fk.dt
        self.accelerations, self.velocities = a, v
        self.times, self.n_accepted = np.arange(len(a)) * fk.dt, len(a)
        self.__dict__.update(kw)


def test_bound_field_evaluation_checks_digests_dt_frame_and_status(model):
    from anyhydro.radiation_field_kernel import field_history_from_transient, numeric_array_digest
    fk, kernel = model["fk"], model["kernel"]
    assert fk.kernel_digest == numeric_array_digest(kernel.kernel) and fk.ainf_digest == numeric_array_digest(kernel.added_mass_infinite)
    a, v = _trajectory(150)
    ok = field_history_from_transient(fk, _BoundResult(fk, a, v), "pressure")
    assert np.array_equal(ok, field_history_fft(fk, "pressure", a, v))
    assert np.array_equal(field_history_from_transient(fk, _BoundResult(fk, a, v), "elevation", method="direct").shape, (150, N_E))
    for change, match in ((dict(kernel_digest="1" * 64), "DIFFERENT radiation kernel"),
                          (dict(ainf_digest="2" * 64), "DIFFERENT A_inf"), (dict(status="multiple_roots"), "not complete"),
                          (dict(complete=False), "not complete"), (dict(frame="fixed_mean_frame"), "frame"),
                          (dict(dt=fk.dt * 1.01), "dt"), (dict(digest_protocol="other"), "protocol")):
        with pytest.raises(RadiationFieldError, match=match):
            field_history_from_transient(fk, _BoundResult(fk, a, v, **change), "pressure")
    with pytest.raises(RadiationFieldError, match="not a bound"):
        field_history_from_transient(fk, object(), "pressure")
    # a K2 field kernel refuses a K1 result
    args = model["args"]
    k2 = build_radiation_field_kernel(**{
        **args, "kernel": _kernel_scaled(model, 1.5), "pressure_transfer": args["pressure_transfer"] * 1.5,
        "pressure_inf": args["pressure_inf"] * 1.5, "pressure_zero_frequency": args["pressure_zero_frequency"] * 1.5})
    with pytest.raises(RadiationFieldError, match="DIFFERENT radiation kernel"):
        field_history_from_transient(k2, _BoundResult(fk, a, v), "pressure")


def _kernel_scaled(model, factor):
    """A second generalized kernel (different A_inf / samples) of the same family settings."""
    from dataclasses import replace
    w = model["w"]; ww = w[:, None, None]
    base = model["kernel"].settings
    settings = replace(base, zero_frequency_damping=np.asarray(base.zero_frequency_damping) * factor)
    D, A_inf = model["D"] * factor, model["A_inf"] * factor
    return _kernel_from_arrays(
        omegas=w, added_mass=A_inf[None] - D[None] / (LAM**2 + ww**2), damping=D[None] * LAM / (LAM**2 + ww**2),
        a_inf=A_inf, dof_keys=model["kernel"].dof_keys, body_ids=("body",), settings=settings,
        provenance={"source": "analytic-exponential-test-2"})


def test_bound_wrapper_derives_dt_from_times_and_checks_every_array_length(model):
    from anyhydro.radiation_field_kernel import field_history_from_transient
    fk = model["fk"]
    a, v = _trajectory(40)
    t = np.arange(40) * fk.dt
    skip = t.copy(); skip[10:] += 0.5 * fk.dt                        # one long gap: not a uniform grid
    for change, match in ((dict(times=skip), "uniform"), (dict(times=t[::-1].copy()), "uniform"),
                          (dict(times=t[:-1]), "inconsistent"), (dict(n_accepted=39), "inconsistent"),
                          (dict(n_accepted=True), "inconsistent"), (dict(n_accepted=40.5), "inconsistent"),
                          (dict(times=t * 1.01, dt=fk.dt * 1.01), "field-kernel dt"),
                          (dict(times=t * (1 + 0j)), "complex"), (dict(dt=fk.dt * 1.5), "declared result dt"),
                          (dict(times=np.array([0.0])), "at least two")):
        with pytest.raises(RadiationFieldError, match=match):
            field_history_from_transient(fk, _BoundResult(fk, a, v, **change), "pressure")
    with pytest.raises(RadiationFieldError):                          # accelerations / velocities of different length
        field_history_from_transient(fk, _BoundResult(fk, a, v[:-1]), "pressure")
    with pytest.raises(RadiationFieldError):
        field_history_from_transient(fk, _BoundResult(fk, a[:, :5], v[:, :5]), "pressure")
    assert field_history_from_transient(fk, _BoundResult(fk, a, v), "pressure").shape == (40, N_P)


def test_digest_is_incremental_and_equals_the_reference_protocol(model):
    import hashlib
    from anyhydro.radiation_field_kernel import numeric_array_digest
    rng = np.random.default_rng(0)
    for shape in ((7,), (3, 4), (2, 70000), (1, 6, 6)):               # includes sizes above one 65536-element chunk
        x = rng.standard_normal(shape)
        h = hashlib.sha256(b"numeric-array-v1|" + repr(tuple(int(s) for s in shape)).encode() + b"|")
        h.update(np.ascontiguousarray(x, dtype="<f8").tobytes())
        assert numeric_array_digest(x) == h.hexdigest()
        assert numeric_array_digest(x.astype(np.float32)) == numeric_array_digest(x.astype(np.float32).astype(float))
    assert numeric_array_digest(np.zeros((0, 3))) == hashlib.sha256(b"numeric-array-v1|(0, 3)|").hexdigest()


def test_unbound_fixtures_are_refused_by_the_bound_wrapper_and_chunk_size_is_validated(model):
    from anyhydro.radiation_field_kernel import field_history_from_transient
    fk = model["fk"]
    kw = dict(kernel_hash=fk.kernel_hash, settings=fk.settings, dof_keys=fk.dof_keys, time=fk.time, panel_ids=fk.panel_ids,
              query_point_ids=fk.query_point_ids, pressure_a=fk.pressure_a, pressure_chunks=fk.pressure_chunks,
              elevation_a=fk.elevation_a, elevation_chunks=fk.elevation_chunks, projection=fk.projection,
              added_mass_infinite=fk.added_mass_infinite, diagnostics={}, identity={})
    unbound = RadiationFieldKernel(chunk_size=fk.chunk_size, **kw)
    a, v = _trajectory(20)
    with pytest.raises(RadiationFieldError, match="UNBOUND"):
        field_history_from_transient(unbound, _BoundResult(fk, a, v), "pressure")
    for bad in (0, -1, True, 2.5, "4", None):
        with pytest.raises(RadiationFieldError, match="chunk_size"):
            RadiationFieldKernel(chunk_size=bad, **kw)
    with pytest.raises(RadiationFieldError, match="digest"):
        RadiationFieldKernel(chunk_size=fk.chunk_size, kernel_digest="xyz", **kw)
    # the digests survive the artifact roundtrip and change the content hash
    assert fk.content_hash() != unbound.content_hash()


def test_real_coefficients_refuse_complex_input_but_complex_transfers_stay_intact(model):
    import warnings
    fk = model["fk"]
    base = dict(kernel_hash=fk.kernel_hash, settings=fk.settings, dof_keys=fk.dof_keys, time=fk.time, panel_ids=fk.panel_ids,
                query_point_ids=fk.query_point_ids, pressure_a=fk.pressure_a, pressure_chunks=fk.pressure_chunks,
                elevation_a=fk.elevation_a, elevation_chunks=fk.elevation_chunks, projection=fk.projection,
                added_mass_infinite=fk.added_mass_infinite, diagnostics={}, identity={}, chunk_size=fk.chunk_size)
    cx = lambda x: np.asarray(x) * (1.0 + 0j)           # even a zero imaginary part is refused
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for key in ("pressure_a", "elevation_a", "projection", "added_mass_infinite", "time"):
            with pytest.raises(RadiationFieldError, match="complex"):
                RadiationFieldKernel(**{**base, key: cx(base[key])})
        with pytest.raises(RadiationFieldError, match="complex"):
            RadiationFieldKernel(**{**base, "pressure_chunks": tuple(cx(c) for c in fk.pressure_chunks)})
        with pytest.raises(RadiationFieldError, match="complex"):
            RadiationFieldKernel(**{**base, "pressure_a": np.ones((N_P, 6), complex) * (1.0 + 2j)})
        args = model["args"]
        for key in ("pressure_inf", "elevation_inf", "projection", "pressure_zero_frequency", "omegas"):
            with pytest.raises(RadiationFieldError, match="complex"):
                build_radiation_field_kernel(**{**args, key: cx(args[key])})
        a, v = _trajectory(30)
        with pytest.raises(RadiationFieldError, match="complex"):
            field_history_fft(fk, "pressure", cx(a), v)
        with pytest.raises(RadiationFieldError, match="complex"):
            field_history_direct(fk, "pressure", a, cx(v))
        with pytest.raises(RadiationFieldError, match="complex"):
            fk.reconstruct("pressure", cx([1.0, 2.0]))
        from anyhydro.radiation_field_kernel import numeric_array_digest, physical_force_projection
        with pytest.raises(RadiationFieldError, match="complex"):
            numeric_array_digest(cx(np.zeros(3)))
        with pytest.raises(RadiationFieldError, match="complex"):
            physical_force_projection(cx(np.zeros((2, 3))), np.ones((2, 3)), np.ones(2), np.eye(3), np.zeros(3))
        fx = _native_fixture()
        with pytest.raises(RadiationFieldError, match="complex"):
            _helper(fx, query_points=cx(fx["pts"]))
        with pytest.raises(RadiationFieldError, match="complex"):
            _helper(fx, vertices=cx(fx["v"]))
    # the explicitly complex interfaces are untouched: the fixture's complex transfers and complex retained sources
    assert np.iscomplexobj(args["pressure_transfer"]) and np.iscomplexobj(args["elevation_transfer"])
    out, fake = _helper(fx)
    assert out.sources_over_omega.dtype == np.complex128 and np.abs(out.sources_over_omega.imag).max() > 0


def test_failed_boundary_check_fails_closed_keeps_raw_values_and_is_never_admitted(tmp_path, monkeypatch):
    from dataclasses import replace
    import anyhydro.radiation_field_kernel as rfk
    fx = _native_fixture()
    offset = 1.0e-3
    fake = _Native(fx["mesh"], fx["p"], fx["e"], fx["a"], 9.81, boundary_offset=offset)
    with pytest.raises(rfk.NativeBoundaryError, match="NOT admitted") as caught:
        _helper(fx, fake)
    diag = caught.value.diagnostic
    # raw evidence is preserved: nothing zeroed, clamped or loosened
    assert diag.boundary_status == "failed" and not diag.is_admitted
    assert np.allclose(diag.elevation_inf, offset / 9.81, rtol=1e-12)             # E_a = Im(phi)/g is KEPT
    assert np.allclose(diag.boundary_residual, offset / 9.81, rtol=1e-12) and np.all(diag.boundary_residual > diag.boundary_limit)
    unit = 4096 * np.finfo(float).eps
    L = np.max(np.ptp(fx["v"], axis=0)); R = np.max(np.linalg.norm(fx["v"] - fx["cg"], axis=1))
    assert np.allclose(diag.boundary_limit, unit * np.array([L / 9.81] * 3 + [L * R / 9.81] * 3), rtol=1e-12)
    assert diag.sources_over_omega.dtype == np.complex128 and diag.is_recoverable
    # production paths refuse the unadmitted object
    with pytest.raises(rfk.NativeBoundaryError):
        diag.builder_arrays()
    with pytest.raises(rfk.NativeBoundaryError):
        rfk.build_radiation_field_kernel_from_native(diag, kernel=None, omegas=None, pressure_transfer=None,
                                                    elevation_transfer=None, pressure_criteria=None, elevation_criteria=None)
    with pytest.raises(rfk.NativeBoundaryError):
        rfk.recover_waterline_coefficients(diag, np.array([[1.0, 0.0, 0.0]]), cpt=fake, replay=lambda *a: None)
    # the diagnostic snapshot can be saved and reloaded (status verified, still not admitted); the hash carries it
    path = str(tmp_path / "diag.npz")
    digest = diag.save(path)
    back = rfk.load_native_field_coefficients(path, expected_hash=digest)
    assert back.boundary_status == "failed" and np.array_equal(back.boundary_residual, diag.boundary_residual)
    ok, _ = _helper(fx)
    assert ok.is_admitted and ok.content_hash() != diag.content_hash()
    # the status is recomputed from the raw residual / limit: it cannot be asserted either way
    with pytest.raises(RadiationFieldError, match="contradicts"):
        replace(diag, boundary_status="admitted")
    with pytest.raises(RadiationFieldError, match="contradicts"):
        replace(ok, boundary_status="failed")
    big = ok.boundary_limit * 2.0
    assert replace(ok, elevation_inf=np.tile(big, (3, 1)), boundary_residual=big,
                   boundary_status="").boundary_status == "failed"
    with pytest.raises(RadiationFieldError, match="contradicts the raw elevation_inf"):
        replace(ok, boundary_residual=big, boundary_status="")          # reported residual not backed by the raw data
    unchecked = replace(ok, boundary_residual=None, boundary_limit=None, boundary_status="", vertices=None, faces=None, request=None)
    assert unchecked.boundary_status == "unchecked"
    with pytest.raises(rfk.NativeBoundaryError):
        unchecked.builder_arrays()
    # admitted object: the production binding forwards ONLY the native arrays and records the native identity
    seen = {}
    monkeypatch.setattr(rfk, "build_radiation_field_kernel", lambda **kw: seen.update(kw) or "built")
    assert rfk.build_radiation_field_kernel_from_native(
        ok, kernel="K", omegas="w", pressure_transfer="Hp", elevation_transfer="He",
        pressure_criteria="cp", elevation_criteria="ce", identity={"body": "b"}) == "built"
    assert np.array_equal(seen["pressure_inf"], ok.pressure_inf) and seen["panel_ids"] == ok.panel_ids
    binding = seen["_native_binding"]
    assert binding["native_hash"] == ok.content_hash() and binding["native_boundary_status"] == "admitted"
    assert binding["binding"] == "admitted_native_source" and seen["identity"] == {"body": "b"}


def test_status_follows_the_actual_coefficients_and_unchecked_refusal_is_typed():
    from dataclasses import replace
    import anyhydro.radiation_field_kernel as rfk
    fx = _native_fixture()
    ok, _ = _helper(fx)
    # raw E_a = 1 with a reported zero residual and a tiny limit must NOT be admitted (nor accepted as consistent)
    ones = np.ones_like(ok.elevation_inf)
    with pytest.raises(RadiationFieldError, match="contradicts the raw elevation_inf"):
        replace(ok, elevation_inf=ones, boundary_residual=np.zeros(6), boundary_limit=np.full(6, 1e-14), boundary_status="")
    honest = replace(ok, elevation_inf=ones, boundary_residual=np.ones(6), boundary_limit=np.full(6, 1e-14), boundary_status="")
    assert honest.boundary_status == "failed" and not honest.is_admitted
    assert np.all(honest.elevation_inf == 1.0)                           # raw data preserved, not clamped
    # without a reported residual the status is still computed from the raw coefficients
    assert replace(ok, elevation_inf=ones, boundary_residual=None, boundary_limit=np.full(6, 1e-14),
                   boundary_status="").boundary_status == "failed"
    for bad in (-np.ones(6), np.full(6, np.nan), np.full(6, -1e-30)):
        with pytest.raises(RadiationFieldError):
            replace(ok, boundary_limit=bad, boundary_status="")
    with pytest.raises(RadiationFieldError):
        replace(ok, boundary_residual=-np.zeros(6) - 1.0, boundary_status="")
    # an unchecked object raises the typed refusal (no TypeError from formatting unknown values)
    unchecked = replace(ok, boundary_residual=None, boundary_limit=None, boundary_status="", vertices=None, faces=None, request=None)
    for call in (unchecked.require_admitted, unchecked.builder_arrays):
        with pytest.raises(rfk.NativeBoundaryError, match="unchecked"):
            call()


def test_boundary_limit_is_an_immutable_derived_ceiling_that_can_only_tighten(tmp_path):
    from dataclasses import replace
    import json
    import anyhydro.radiation_field_kernel as rfk
    fx = _native_fixture()
    ok, _ = _helper(fx)
    ceiling = rfk._derived_boundary_limit(fx["v"], {"cg_world": fx["cg"].tolist(), "gravity": 9.81})
    assert np.array_equal(ok.boundary_limit, ceiling)                       # helper and constructor share one definition
    with pytest.raises(RadiationFieldError, match="widens"):
        replace(ok, boundary_limit=np.full(6, 1e300), boundary_status="")
    with pytest.raises(RadiationFieldError, match="widens"):
        replace(ok, boundary_limit=ceiling * (1 + 1e-9), boundary_status="")
    assert replace(ok, boundary_limit=ceiling * 0.5, boundary_status="").boundary_status == "admitted"   # tightening is fine
    assert np.array_equal(replace(ok, boundary_limit=None, boundary_status="").boundary_limit, ceiling)   # re-derived
    # a failed snapshot cannot be re-admitted by widening
    fake = _Native(fx["mesh"], fx["p"], fx["e"], fx["a"], 9.81, boundary_offset=1e-3)
    with pytest.raises(rfk.NativeBoundaryError) as caught:
        _helper(fx, fake)
    diag = caught.value.diagnostic
    with pytest.raises(RadiationFieldError, match="widens"):
        replace(diag, boundary_limit=np.full(6, 1e300), boundary_status="")
    # without complete valid geometry / environment nothing can be admitted, whatever limit is claimed
    assert replace(ok, vertices=None, faces=None, request=None, boundary_limit=np.full(6, 1e300),
                   boundary_status="").boundary_status == "unchecked"
    # saved-file tampering: widened limit array, and a forged header status, are both refused on load
    path = str(tmp_path / "diag.npz")
    diag.save(path)
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    wide = dict(data); wide["boundary_limit"] = np.full(6, 1e300)
    p1 = str(tmp_path / "wide.npz"); np.savez(p1, **wide)
    with pytest.raises(RadiationFieldError, match="widens"):
        rfk.load_native_field_coefficients(p1)
    header = json.loads(bytes(data["header"].tobytes()).decode())
    header["meta"]["boundary_status"] = "admitted"
    forged = dict(data); forged["header"] = np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)
    p2 = str(tmp_path / "forged.npz"); np.savez(p2, **forged)
    with pytest.raises(RadiationFieldError):
        rfk.load_native_field_coefficients(p2)


def test_direct_construction_requires_nonempty_shapes_with_typed_errors():
    from dataclasses import replace
    fx = _native_fixture()
    ok, _ = _helper(fx)
    with pytest.raises(RadiationFieldError, match="at least one"):
        replace(ok, elevation_inf=np.zeros((0, 6)), query_points=np.zeros((0, 3)), query_point_ids=(),
                boundary_residual=None, boundary_status="")
    with pytest.raises(RadiationFieldError):
        replace(ok, faces=np.zeros((0, 4), dtype=np.int64), boundary_status="")
    with pytest.raises(RadiationFieldError):
        replace(ok, vertices=np.zeros((0, 3)), boundary_status="")
    with pytest.raises(RadiationFieldError):
        replace(ok, pressure_inf=np.zeros((0, 6)), panel_ids=(), boundary_status="")


def test_recovered_points_are_checked_and_an_admitted_source_does_not_admit_a_new_field(monkeypatch):
    import anyhydro.radiation_field_kernel as rfk
    fx = _native_fixture()
    _patched_family(monkeypatch, fx)
    out, fake = _helper(fx, body=None, reference_omega=0.8)
    pts = np.array([[1.5, 0.5, 0.0], [-2.0, 1.0, 0.0]])
    rec = rfk.recover_waterline_coefficients(out, pts, cpt=fake, replay=lambda *a: "gf")
    assert rec.is_admitted and rec.boundary_status == "admitted" and np.array_equal(rec.boundary_limit, out.boundary_limit)
    assert rec.native_hash == out.content_hash() and "unverified" in rec.policy and np.all(rec.boundary_residual == 0.0)
    fake.boundary_offset = 1.0e-3                                  # the backend now returns a non-zero free-surface value
    with pytest.raises(rfk.NativeBoundaryError, match="recovered waterline") as caught:
        rfk.recover_waterline_coefficients(out, pts, cpt=fake, replay=lambda *a: "gf")
    diag = caught.value.diagnostic
    assert not diag.is_admitted and diag.boundary_status == "failed"
    assert np.allclose(diag.elevation_inf, 1.0e-3 / 9.81) and np.allclose(diag.boundary_residual, 1.0e-3 / 9.81)   # raw kept
    assert np.all(diag.boundary_residual > diag.boundary_limit) and diag.native_hash == out.content_hash()
    with pytest.raises(ValueError):
        diag.elevation_inf[0, 0] = 0.0


def test_raw_builder_cannot_claim_a_native_binding(model):
    args = model["args"]
    assert model["fk"].identity["binding"] == "unbound_raw_arrays"          # explicit, hashed marker of raw arrays
    for key in ("binding", "native_hash", "native_boundary_status", "native_helper"):
        with pytest.raises(RadiationFieldError, match="reserved"):
            build_radiation_field_kernel(**{**args, "identity": {key: "forged"}})
    other = build_radiation_field_kernel(**{**args, "identity": {"body": "hull-1", "pose": "mean", "mesh": "m-abc"}})
    assert other.content_hash() == model["fk"].content_hash()                # same raw build is deterministic


def test_digest_of_a_transposed_strided_array_has_no_full_copy_and_matches_the_protocol():
    import hashlib
    import tracemalloc
    from anyhydro.radiation_field_kernel import numeric_array_digest
    base = np.random.default_rng(5).standard_normal((1200, 1000))            # 9.6 MB
    view = base.T                                                               # F-ordered / strided view
    h = hashlib.sha256(b"numeric-array-v1|(1000, 1200)|")
    h.update(np.ascontiguousarray(view, dtype="<f8").tobytes())
    expected = h.hexdigest()
    tracemalloc.start()
    try:
        got = numeric_array_digest(view)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert got == expected and peak < 3 * 1024 * 1024 < base.nbytes
    assert numeric_array_digest(base.astype(">f8")) == numeric_array_digest(base)      # byte order does not matter


def test_wrapper_normalises_bad_counts_and_dt_to_typed_errors(model):
    from anyhydro.radiation_field_kernel import field_history_from_transient
    fk = model["fk"]
    a, v = _trajectory(20)
    for change in (dict(n_accepted=None), dict(n_accepted=float("nan")), dict(n_accepted=float("inf")),
                   dict(n_accepted="20"), dict(dt=None), dict(dt=float("nan")), dict(dt="x")):
        with pytest.raises(RadiationFieldError):
            field_history_from_transient(fk, _BoundResult(fk, a, v, **change), "pressure")


def test_faces_are_exact_integers_inside_the_vertex_array_and_waterline_points_are_nonempty():
    fx = _native_fixture()
    for faces in (fx["f"].astype(float), fx["f"] + 100, fx["f"] - 1, fx["f"][:, :3], fx["f"][:0]):
        with pytest.raises(RadiationFieldError, match="faces"):
            _helper(fx, faces=faces)
    with pytest.raises(RadiationFieldError):
        _helper(fx, query_points=np.zeros((0, 3)))
    out, _ = _helper(fx)
    from dataclasses import replace
    with pytest.raises(RadiationFieldError, match="faces"):
        replace(out, faces=out.faces.astype(float) + 0.5)
    with pytest.raises(RadiationFieldError, match="faces index"):
        replace(out, faces=out.faces + 1000)


def test_plain_limit_refuses_nonzero_imaginary_parts_but_accepts_exact_zero():
    from anyhydro.radiation_field_kernel import _plain_limit
    assert np.array_equal(_plain_limit(np.array([1.0 + 0j, -2.0 + 0j]), "c"), [1.0, -2.0])     # exact-zero imag: real value
    with pytest.raises(RadiationFieldError, match="imaginary"):
        _plain_limit(np.array([1.0 + 1e-300j]), "c")
    with pytest.raises(RadiationFieldError, match="imaginary"):
        _plain_limit(np.array([1.0, 2.0 + 0.5j]), "c")
    assert _plain_limit(np.array([3.0]), "r")[0] == 3.0


def test_recovery_refuses_a_mismatched_kh_and_wraps_replay_refusals(monkeypatch):
    from dataclasses import replace
    from anyhydro.radiation_field_kernel import recover_waterline_coefficients
    fx = _native_fixture()
    _patched_family(monkeypatch, fx)
    out, fake = _helper(fx, body=None, reference_omega=0.8)
    pts = np.array([[1.0, 0.0, 0.0]])
    with pytest.raises(RadiationFieldError, match="kh"):
        recover_waterline_coefficients(replace(out, prony_kh=3.0), pts, cpt=fake, replay=lambda *a: None)

    def refusing(cpt, view, problem):
        raise ValueError("installed Green-function settings differ from the captured backend settings")

    with pytest.raises(RadiationFieldError, match="replay refused"):
        recover_waterline_coefficients(out, pts, cpt=fake, replay=refusing)


def _expected_raw_and_bound(fake, pts, g=9.81):
    S = fake._S(pts)
    raw = np.stack([np.imag(S @ fake.s[j]) / g for j in range(6)], axis=1)
    m = (S.shape[1] + 4) * np.finfo(float).eps
    bound = (1.0 + m / (1.0 - m)) * (np.abs(S) @ np.abs(fake.s).T) / g
    return raw, bound


def test_strict_default_is_unchanged_and_still_refuses_a_raw_failure():
    import anyhydro.radiation_field_kernel as rfk
    fx = _native_fixture()
    fake = _Native(fx["mesh"], fx["p"], fx["e"], fx["a"], 9.81, s_scale=1e-3)
    with pytest.raises(rfk.NativeBoundaryError, match="native_strict") as caught:
        _helper(fx, fake)                                                   # default policy = native_strict
    diag = caught.value.diagnostic
    raw, bound = _expected_raw_and_bound(fake, fx["pts"])
    assert diag.boundary_policy == "native_strict" and diag.boundary_status == "failed" and diag.policy_status == "failed"
    assert np.allclose(diag.elevation_inf, raw, rtol=1e-12, atol=0) and not diag.effective_admitted
    assert np.allclose(diag.pointwise_bound, bound, rtol=1e-14)            # evidence captured in strict mode as well
    assert fake.events.index("S") < fake.events.index("phi")               # bound evidence BEFORE the signed potential


def test_exact_dirichlet_z0_splits_raw_and_effective_and_keeps_every_raw_value(tmp_path):
    import anyhydro.radiation_field_kernel as rfk
    from dataclasses import replace
    fx = _native_fixture()
    fake = _Native(fx["mesh"], fx["p"], fx["e"], fx["a"], 9.81, s_scale=1e-3)
    out, _ = _helper(fx, fake, boundary_policy="exact_dirichlet_z0")
    raw, bound = _expected_raw_and_bound(fake, fx["pts"])
    # raw native data and raw gate preserved; policy decides only the EFFECTIVE coefficient
    assert out.boundary_status == "failed" and not out.is_admitted
    assert out.boundary_policy == "exact_dirichlet_z0" and out.policy_status == "admitted" and out.effective_admitted
    assert np.allclose(out.elevation_inf, raw, rtol=1e-12, atol=0) and np.max(np.abs(out.elevation_inf)) > 0.0
    assert np.all(out.effective_elevation_inf == 0.0) and out.pointwise_bound.shape == (3, 6)
    assert np.allclose(out.pointwise_bound, bound, rtol=1e-14) and np.all(np.abs(out.elevation_inf) <= out.pointwise_bound)
    assert out.abs_s_query.shape == (3, 3) and np.all(out.abs_s_query >= 0.0)
    assert np.array_equal(out.pressure_inf, fx["p"]) and np.array_equal(out.added_mass_infinite, fx["a"])   # unchanged
    assert out.sources_over_omega.dtype == np.complex128 and out.prony is not None
    # strict production paths still refuse the raw failure; effective paths admit under the policy
    with pytest.raises(rfk.NativeBoundaryError):
        out.require_admitted()
    with pytest.raises(rfk.NativeBoundaryError):
        out.builder_arrays()
    eff = out.effective_builder_arrays()
    assert np.all(eff["elevation_inf"] == 0.0) and np.array_equal(eff["pressure_inf"], out.pressure_inf)
    with pytest.raises(ValueError):
        out.effective_elevation_inf[0, 0] = 1.0
    # statuses are recomputed, never asserted
    with pytest.raises(RadiationFieldError, match="contradicts"):
        replace(out, policy_status="refused")
    with pytest.raises(RadiationFieldError):
        replace(out, boundary_policy="anything")
    # exact hashed roundtrip; header tampering with the policy is detected
    path = str(tmp_path / "policy.npz")
    digest = out.save(path)
    back = rfk.load_native_field_coefficients(path, expected_hash=digest)
    assert back.boundary_policy == "exact_dirichlet_z0" and back.policy_status == "admitted" and back.boundary_status == "failed"
    assert np.array_equal(back.elevation_inf, out.elevation_inf) and np.array_equal(back.abs_s_query, out.abs_s_query)
    import json
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    header = json.loads(bytes(data["header"].tobytes()).decode())
    header["meta"]["boundary_policy"] = "native_strict"
    data["header"] = np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)
    forged = str(tmp_path / "forged.npz"); np.savez(forged, **data)
    with pytest.raises(RadiationFieldError):
        rfk.load_native_field_coefficients(forged)
    # widened |S| cannot be smuggled in: abs_s must match the sources' columns and is part of the hash
    assert replace(out, abs_s_query=out.abs_s_query * 2.0, policy_status="").content_hash() != out.content_hash()


def test_exact_dirichlet_policy_refuses_nonzero_z_inconsistency_and_missing_bound():
    import anyhydro.radiation_field_kernel as rfk
    from dataclasses import replace
    fx = _native_fixture()
    fake = _Native(fx["mesh"], fx["p"], fx["e"], fx["a"], 9.81, s_scale=1e-3)
    near = fx["pts"].copy(); near[0, 2] = 1e-13                             # within the helper's z tolerance, NOT exact
    with pytest.raises(rfk.NativeBoundaryError, match="z exactly 0"):
        _helper(fx, fake, boundary_policy="exact_dirichlet_z0", query_points=near)
    bad = _Native(fx["mesh"], fx["p"], fx["e"], fx["a"], 9.81, s_scale=1e-3, boundary_offset=1.0)   # beyond the bound
    with pytest.raises(rfk.NativeBoundaryError, match="inconsistent"):
        _helper(fx, bad, boundary_policy="exact_dirichlet_z0")
    ok, _ = _helper(fx, _Native(fx["mesh"], fx["p"], fx["e"], fx["a"], 9.81, s_scale=1e-3),
                    boundary_policy="exact_dirichlet_z0")
    # an artifact without retained |S| (e.g. an old strict artifact) cannot be converted into a policy admission
    old = replace(ok, abs_s_query=None, policy_status="")
    assert old.policy_status == "refused" and "unavailable" in old.policy_reason
    with pytest.raises(rfk.NativeBoundaryError):
        old.effective_builder_arrays()
    with pytest.raises(RadiationFieldError):
        replace(ok, abs_s_query=-ok.abs_s_query - 1.0, policy_status="")
    with pytest.raises(RadiationFieldError):
        replace(ok, abs_s_query=ok.abs_s_query[:, :2], policy_status="")


def test_policy_recovery_recomputes_raw_and_bound_with_the_replayed_gf_and_zero_solves(monkeypatch):
    import anyhydro.radiation_field_kernel as rfk
    fx = _native_fixture()
    _patched_family(monkeypatch, fx)
    fake = _Native(fx["mesh"], fx["p"], fx["e"], fx["a"], 9.81, s_scale=1e-3)
    out, _ = _helper(fx, fake, boundary_policy="exact_dirichlet_z0", body=None, reference_omega=0.8)
    solves = len(fake.solve_kwargs)
    pts = np.array([[1.5, 0.5, 0.0], [-2.0, 1.0, 0.0]])
    fake.events.clear()
    rec = rfk.recover_waterline_coefficients(out, pts, cpt=fake, replay=lambda *a: "replayed")
    raw, bound = _expected_raw_and_bound(fake, pts)
    assert len(fake.solve_kwargs) == solves and fake.gf_args[-1] == "replayed"     # zero solves (fake measurement)
    assert fake.events[0] == "S"                                                      # bound before the signed potential
    assert rec.boundary_policy == "exact_dirichlet_z0" and rec.effective_admitted and rec.boundary_status == "failed"
    assert np.allclose(rec.elevation_inf, raw, rtol=1e-12, atol=0) and np.allclose(rec.pointwise_bound, bound, rtol=1e-14)
    assert np.all(rec.effective_elevation_inf == 0.0) and rec.native_hash == out.content_hash()
    with pytest.raises(RadiationFieldError, match="EXACTLY 0"):
        rfk.recover_waterline_coefficients(out, np.array([[1.0, 0.0, 1e-13]]), cpt=fake, replay=lambda *a: "replayed")
    fake.boundary_offset = 1.0                                                        # inconsistent at the new points
    with pytest.raises(rfk.NativeBoundaryError, match="inconsistent") as caught:
        rfk.recover_waterline_coefficients(out, pts, cpt=fake, replay=lambda *a: "replayed")
    assert caught.value.diagnostic.policy_status == "refused"


def test_policy_binding_records_raw_status_policy_and_digests(monkeypatch, model):
    import anyhydro.radiation_field_kernel as rfk
    original_builder = rfk.build_radiation_field_kernel
    for key in ("boundary_policy", "policy_status", "source_snapshot_ids", "family_radiation_key"):
        with pytest.raises(RadiationFieldError, match="reserved"):          # raw callers cannot forge policy binding
            original_builder(**{**model["args"], "identity": {key: "x"}})
    fx = _native_fixture()
    out, _ = _helper(fx, _Native(fx["mesh"], fx["p"], fx["e"], fx["a"], 9.81, s_scale=1e-3),
                     boundary_policy="exact_dirichlet_z0")
    seen = {}
    monkeypatch.setattr(rfk, "build_radiation_field_kernel", lambda **kw: seen.update(kw) or "built")
    rfk.build_radiation_field_kernel_from_native(out, kernel="K", omegas="w", pressure_transfer="Hp",
                                                elevation_transfer="He", pressure_criteria="c", elevation_criteria="c")
    b = seen["_native_binding"]
    assert b["boundary_policy"] == "exact_dirichlet_z0" and b["policy_status"] == "admitted"
    assert b["native_boundary_status"] == "failed"                                   # raw failure stays visible
    assert b["raw_elevation_digest"] == rfk.numeric_array_digest(out.elevation_inf)
    assert b["pointwise_bound_digest"] == rfk.numeric_array_digest(out.pointwise_bound)
    assert np.all(seen["elevation_inf"] == 0.0)


def _reference_bound(abs_s, sources, g):
    """Independent exact reference: Fraction sum of the IEEE products (== correctly rounded sum), fixed scalar scaling."""
    import math
    from fractions import Fraction
    a = np.asarray(abs_s, dtype=float)
    s = np.asarray(sources, dtype=complex)
    m = float(a.shape[1] + 4) * np.finfo(float).eps
    factor = 1.0 + m / (1.0 - m)
    out = np.empty((a.shape[0], s.shape[0]))
    for i in range(a.shape[0]):
        for j in range(s.shape[0]):
            total = sum((Fraction(float(a[i, p]) * math.hypot(s[j, p].real, s[j, p].imag)) for p in range(a.shape[1])),
                        Fraction(0))
            out[i, j] = (factor * float(total)) / g
    return out


def test_pointwise_bound_is_bit_reproducible_across_layout_alignment_and_matches_exact_reference():
    from anyhydro.radiation_field_kernel import _pointwise_bound
    rng = np.random.default_rng(42)
    abs_s = np.abs(rng.standard_normal((7, 37))) * 10.0 ** rng.integers(-8, 3, (7, 37))
    sig = (rng.standard_normal((6, 37)) + 1j * rng.standard_normal((6, 37))) * 10.0 ** rng.integers(-5, 5, (6, 37))
    ref = _reference_bound(abs_s, sig, 9.81)
    base = _pointwise_bound(abs_s, sig, 9.81)
    assert np.array_equal(base, ref)                                          # bit-identical to the exact reference
    big = np.zeros((7, 74)); big[:, ::2] = abs_s                              # strided view
    raw = np.zeros(abs_s.size * 8 + 1, dtype=np.uint8)                        # deliberately misaligned buffer
    mis_w = np.ndarray(abs_s.shape, dtype=np.float64, buffer=raw, offset=1)
    mis_w[...] = abs_s
    assert not mis_w.flags.aligned
    for variant in (np.asfortranarray(abs_s), big[:, ::2], mis_w, abs_s.astype(">f8")):
        assert np.array_equal(_pointwise_bound(variant, sig, 9.81), base)
    for svariant in (np.asfortranarray(sig), sig.astype(">c16"), np.ascontiguousarray(sig.T).T):
        assert np.array_equal(_pointwise_bound(abs_s, svariant, 9.81), base)
    # complex magnitudes: math.hypot per element (finite even where |re|^2 would overflow)
    huge = np.full((6, 2), 1e200 + 1e200j)
    b = _pointwise_bound(np.ones((1, 2)), huge, 1.0)
    assert np.all(np.isfinite(b)) and b[0, 0] > 2e200


def test_abs_influence_is_exact_for_real_and_hypot_for_complex():
    import math
    from anyhydro.radiation_field_kernel import _abs_influence
    r = np.array([[-1.5, 0.0, 2.0]])
    assert np.array_equal(_abs_influence(r), np.abs(r))
    c = np.array([[3 + 4j, -1e-200 + 1e-200j]])
    got = _abs_influence(c)
    assert got[0, 0] == 5.0 and got[0, 1] == math.hypot(-1e-200, 1e-200)


def test_bound_algorithm_is_recorded_and_superseded_artifacts_are_refused(tmp_path):
    import json
    import anyhydro.radiation_field_kernel as rfk
    fx = _native_fixture()
    fake = _Native(fx["mesh"], fx["p"], fx["e"], fx["a"], 9.81, s_scale=1e-3)
    out, _ = _helper(fx, fake, boundary_policy="exact_dirichlet_z0")
    assert out._meta()["bound_algorithm"] == rfk.POINTWISE_BOUND_ALGORITHM
    assert rfk.native_binding_record(out, "x")["bound_algorithm"] == rfk.POINTWISE_BOUND_ALGORITHM
    path = str(tmp_path / "v2.npz")
    out.save(path)
    assert rfk.load_native_field_coefficients(path).content_hash() == out.content_hash()
    with np.load(path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    for forged_alg in (None, "pointwise-bound-matmul-v1"):
        header = json.loads(bytes(data["header"].tobytes()).decode())
        if forged_alg is None:
            header["meta"].pop("bound_algorithm")
        else:
            header["meta"]["bound_algorithm"] = forged_alg
        d2 = dict(data); d2["header"] = np.frombuffer(json.dumps(header).encode(), dtype=np.uint8)
        p2 = str(tmp_path / "old.npz"); np.savez(p2, **d2)
        with pytest.raises(rfk.RadiationFieldSerializationError, match="recapture"):
            rfk.load_native_field_coefficients(p2)
    # artifacts WITHOUT |S| evidence (old raw artifacts) carry no algorithm key: their identity is unchanged
    ok, _ = _helper(fx)
    from dataclasses import replace
    plain = replace(ok, abs_s_query=None)
    assert "bound_algorithm" not in plain._meta()


def test_pointwise_bound_at_default_limits_uses_bounded_memory():
    import tracemalloc
    from anyhydro.radiation_field_kernel import _pointwise_bound
    rng = np.random.default_rng(1)
    abs_s = np.abs(rng.standard_normal((2000, 600)))                          # q2000 / src600 default limits
    sig = rng.standard_normal((6, 600)) + 1j * rng.standard_normal((6, 600))
    tracemalloc.start()
    try:
        b = _pointwise_bound(abs_s, sig, 9.81)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert b.shape == (2000, 6) and peak < 2 * 1024 * 1024                    # no n_q * n_src * 6 temporary (57.6 MB)


def test_gated_native_node_requires_explicit_environment():
    """Real Capytaine run: scheduled by the parent only (ANYFEM_NATIVE_FLOATING_FIELD=1)."""
    import os
    if os.environ.get("ANYFEM_NATIVE_FLOATING_FIELD") != "1":
        pytest.skip("native omega=inf run is gated by ANYFEM_NATIVE_FLOATING_FIELD=1")
    cpt = pytest.importorskip("capytaine")
    from anyhydro.radiation_field_kernel import native_infinite_frequency_field_coefficients
    mesh = cpt.mesh_parallelepiped(size=(4.0, 4.0, 2.0), resolution=(4, 4, 2), center=(0.0, 0.0, -1.0)).immersed_part()
    v, f = np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces)
    pts = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [-2.0, 0.0, 0.0]])
    out = native_infinite_frequency_field_coefficients(
        vertices=v, faces=f, surface_ids=["hull"] * len(f), query_points=pts, water_depth=50.0, Q0=np.eye(3),
        cg_world=np.array([0.0, 0.0, -0.7]), reference_omega=1.0)
    assert np.all(np.isfinite(out.pressure_inf)) and np.all(np.isfinite(out.elevation_inf))
    assert out.projection_error < 1e-8 * np.max(np.abs(out.added_mass_infinite))
    # prospective declared roundoff scales (L/g translations, L R/g rotations); not widened if this fails
    L = float(np.max(np.ptp(v, axis=0))); R = float(np.max(np.linalg.norm(v - np.array([0.0, 0.0, -0.7]), axis=1)))
    eps = np.finfo(float).eps
    limit = 4096 * eps * np.array([L / 9.81] * 3 + [L * R / 9.81] * 3)
    assert np.all(out.boundary_residual <= limit), (out.boundary_residual, limit)
    assert out.is_recoverable and np.isinf(out.prony_kh)
