"""Native auxiliary geometry/retention checks; no BEM solve is used here."""
from dataclasses import replace
import copy
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from anyhydro.transient_native_profile import (
    TransientSourceProfileError, PROFILE_KEY, bind_profile, body_for_profile,
    create_source_profile, profile_from_request, verify_profile_body,
    _prony_target, solver_for_profile, replay_profile_green_function,
)
from anyhydro.floating_radiation_fields import family_request, FloatingFamilySnapshot, DOF_NAMES
from anyhydro.wave_surface_fields import WaveFieldError


@pytest.fixture(scope="module")
def geometry():
    cpt = pytest.importorskip("capytaine")
    mesh = cpt.mesh_parallelepiped(size=(4., 4., 4.), resolution=(4, 4, 4)).immersed_part()
    v, f = np.array(mesh.vertices), np.array(mesh.faces)
    assert not np.any(np.all(v[f, 2] == 0., axis=1))
    labels = ("hull",) * len(f)
    profile = create_source_profile(v, f, labels, lid_z=-.2, water_depth=50.)
    req = family_request(v, f, labels, heading=0., water_depth=50., rho=1025., gravity=9.81,
                         Q0=np.eye(3), cg_world=np.array([0., 0., -.7]))
    return cpt, v, f, labels, profile, req


def test_profile_is_complete_immutable_and_roundtrips(geometry):
    _, _, f, _, p, req = geometry
    bound = bind_profile(req, p)
    back = profile_from_request(bound)
    assert back.content_hash() == p.content_hash()
    assert np.count_nonzero(p.hull_mask) == len(f)
    assert len(p.source_faces) == len(f) + len(p.lid_faces) <= 600
    for a in (p.lid_vertices, p.lid_faces, p.source_vertices, p.source_faces, p.hull_mask):
        with pytest.raises(ValueError):
            a.setflags(write=True)


def test_restore_retains_exact_source_geometry_without_solving(geometry, monkeypatch):
    cpt, v, f, labels, p, req = geometry
    monkeypatch.setattr(cpt.BEMSolver, "solve", lambda *a, **k: pytest.fail("unexpected native solve"))
    _, _, _, _, body = body_for_profile(cpt, v, f, labels, bind_profile(req, p), 1.)
    verify_profile_body(body, bind_profile(req, p))
    assert np.array_equal(body.mesh.vertices, v) and np.array_equal(body.mesh.faces, f)
    assert np.array_equal(body.mesh_including_lid.vertices, p.source_vertices)
    assert np.array_equal(body.mesh_including_lid.faces, p.source_faces)
    assert tuple(body.dofs) == tuple(DOF_NAMES)


def test_profile_hash_hull_identity_and_geometry_refusals(geometry):
    cpt, v, f, labels, p, req = geometry
    corrupt = copy.deepcopy(bind_profile(req, p))
    corrupt[PROFILE_KEY]["profile_hash"] = "0" * 64
    with pytest.raises(TransientSourceProfileError):
        profile_from_request(corrupt)
    with pytest.raises(TransientSourceProfileError):
        bind_profile(dict(req, geometry="0" * 64), p)
    changed = replace(p, source_vertices=p.source_vertices + [1., 0., 0.])
    with pytest.raises(TransientSourceProfileError, match="source mesh"):
        body_for_profile(cpt, v, f, labels, bind_profile(req, changed), 1.)
    with pytest.raises(TransientSourceProfileError):
        replace(p, lid_faces=p.lid_faces[:, ::-1])


@pytest.mark.parametrize("z", [0., .1, -2., float("nan"), float("inf"), 1j, True])
def test_invalid_lid_depth_is_not_nudged(geometry, z):
    _, v, f, labels, _, _ = geometry
    with pytest.raises(TransientSourceProfileError):
        create_source_profile(v, f, labels, lid_z=z, water_depth=50.)


def test_source_ceiling_includes_the_lid(geometry):
    _, v, f, labels, _, _ = geometry
    with pytest.raises(TransientSourceProfileError, match="source panel limit"):
        create_source_profile(v, f, labels, lid_z=-.2, water_depth=50., source_panel_limit=len(f))
    with pytest.raises(TransientSourceProfileError, match="working-memory"):
        create_source_profile(v, f, labels, lid_z=-.2, water_depth=50., faces_max_radius=1.e-10)


def test_default_request_unchanged_and_snapshot_counts_cover_full_source(geometry):
    _, v, f, labels, p, req = geometry
    assert PROFILE_KEY not in req and profile_from_request(req) is None
    n = len(p.source_faces)
    snap = FloatingFamilySnapshot(bind_profile(req, p), 1., v, f, labels,
                                  np.zeros((6, n), complex), np.zeros(n, complex),
                                  np.zeros((6, 6)), np.zeros((6, 6)), np.zeros(6, complex), {})
    snap.validate()
    with pytest.raises(WaveFieldError):
        replace(snap, radiation_sources=np.zeros((6, len(f)), complex),
                diffraction_sources=np.zeros(len(f), complex), snapshot_id="")


@pytest.mark.parametrize("k", [.2, 1., 5., 10.])
def test_prony_removable_pole_against_independent_decimal_oracle(k):
    for x in (k, k - 1.e-7, k + 1.e-7):
        with localcontext() as ctx:
            ctx.prec = 80
            kd = Decimal.from_float(k)
            xd = Decimal.from_float(x) if x != k else kd + Decimal('1e-25')
            ek, emk = kd.exp(), (-kd).exp()
            t = (ek - emk) / (ek + emk)
            ex, emx = xd.exp(), (-xd).exp()
            c = (1 + t)**2 / (1 - t*t + t/kd)
            expected = (xd + kd*t)*ex / (xd*(ex-emx)/2 - kd*t*(ex+emx)/2) - c/(xd-kd) - 2
        assert float(_prony_target(k, np.array([x]))[0]) == pytest.approx(float(expected), abs=2.e-12)


def test_accuracy_profile_identity_and_native_fit_replay(geometry, monkeypatch):
    _, _, _, _, profile, req = geometry
    tight = replace(profile, prony_mse_tolerance=1.e-10, prony_max_terms=120)
    assert tight.content_hash() != profile.content_hash()
    assert profile_from_request(bind_profile(req, tight)).content_hash() == tight.content_hash()
    calls = []
    import capytaine.tools.prony_decomposition as pd
    def fit(f, low, high, terms, **kwargs):
        calls.append((low, high, tuple(terms), kwargs))
        assert np.all(np.isfinite(f(np.array([1., 1.1]))))
        return np.array([2.]), np.array([-1.])
    monkeypatch.setattr(pd, 'find_best_exponential_decomposition', fit)
    class GF:
        def __init__(self, finite_depth_prony_decomposition_method='python', **kwargs):
            self.finite_depth_prony_decomposition_method = finite_depth_prony_decomposition_method
            self.exportable_settings = dict(green_function=type(self).__name__,
                finite_depth_prony_decomposition_method=finite_depth_prony_decomposition_method)
    class Solver:
        def __init__(self, green_function=None):
            self.engine = SimpleNamespace(green_function=green_function or GF())
    fake = SimpleNamespace(BEMSolver=Solver)
    gf = solver_for_profile(fake, tight).engine.green_function
    fitted = gf.find_best_exponential_decomposition(1.)
    assert np.array_equal(gf.find_best_exponential_decomposition(1.), fitted)
    assert len(calls) == 1 and calls[0][-1]['tol'] == 1.e-10 and max(calls[0][2]) == 120
    snapshot = SimpleNamespace(gf_settings={k: v for k, v in gf.exportable_settings.items() if k != 'green_function'},
        provider={'green_function': type(gf).__name__}, prony=fitted, prony_kh=1., prony_method='python')
    monkeypatch.setattr(pd, 'find_best_exponential_decomposition', lambda *a, **k: pytest.fail('unexpected refit'))
    replay = replay_profile_green_function(fake, snapshot, SimpleNamespace(wavenumber=1., water_depth=1.))
    assert np.array_equal(replay.find_best_exponential_decomposition(1.), fitted)
    with pytest.raises(TransientSourceProfileError):
        replay.find_best_exponential_decomposition(2.)
