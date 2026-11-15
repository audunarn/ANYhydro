"""Reserved direct/owner-GF family: identity separation and legacy refusals (no native work), gated native coherence on a small box."""
import math
import os

import numpy as np
import pytest

from anyhydro import floating_radiation_fields as ff
from anyhydro.wave_surface_fields import WaveFieldError, _replay_green_function

native = pytest.mark.skipif(os.environ.get(ff.NATIVE_GATE) != '1', reason='native family solve')


def box(nx=6, ny=3, nz=1, L=6., B=3., T=1.):
    x, y, z = np.linspace(-L / 2, L / 2, nx + 1), np.linspace(-B / 2, B / 2, ny + 1), np.linspace(-T, 0., nz + 1)
    ids, V, F = {}, [], []

    def v(p):
        return ids.setdefault(tuple(round(float(c), 10) for c in p), len(ids))

    def quad(*corners):
        F.append([v(c) for c in corners])
    for i in range(nx):
        for j in range(ny):
            quad((x[i], y[j], -T), (x[i], y[j + 1], -T), (x[i + 1], y[j + 1], -T), (x[i + 1], y[j], -T))
    for i in range(nx):
        for k in range(nz):
            quad((x[i], y[0], z[k]), (x[i + 1], y[0], z[k]), (x[i + 1], y[0], z[k + 1]), (x[i], y[0], z[k + 1]))
            quad((x[i], y[-1], z[k]), (x[i], y[-1], z[k + 1]), (x[i + 1], y[-1], z[k + 1]), (x[i + 1], y[-1], z[k]))
    for j in range(ny):
        for k in range(nz):
            quad((x[0], y[j], z[k]), (x[0], y[j], z[k + 1]), (x[0], y[j + 1], z[k + 1]), (x[0], y[j + 1], z[k]))
            quad((x[-1], y[j], z[k]), (x[-1], y[j + 1], z[k]), (x[-1], y[j + 1], z[k + 1]), (x[-1], y[j], z[k + 1]))
    V = np.array(sorted(ids, key=ids.get), float)
    return V, np.array(F), tuple(f'p{i}' for i in range(len(F)))


KW = dict(water_depth=50., rho=1025., gravity=9.81, Q0=np.eye(3), cg_world=(0., 0., -.4))


def requests():
    V, F, S = box()
    return (ff.family_request(V, F, S, heading=0., recipe=ff.RECIPE, **KW),
            ff.family_request(V, F, S, heading=0., recipe=ff.DIRECT_RECIPE, **KW))


def record(recipe, provider=None, panels=True):
    V, F, S = box(); n = len(F)
    r = ff.family_request(V, F, S, heading=0., recipe=recipe, **KW)
    z = np.zeros((6, n), complex)
    extra = dict(panel_pressure_radiation=z, panel_pressure_incident_diffraction=z[0], centers=np.zeros((n, 3)),
                 normals=np.ones((n, 3)), areas=np.ones(n)) if panels else {}
    return ff.FloatingFamilySnapshot(r, .8, V, F, S, z, z[0], np.eye(6), np.eye(6), np.zeros(6, complex),
                                     provider or dict(backend='capytaine', method='direct' if recipe == ff.DIRECT_RECIPE else 'indirect'), **extra)


def test_production_default_stays_the_legacy_recipe_until_adoption():
    V, F, S = box()
    assert ff.ACTIVE_RECIPE == ff.RECIPE
    assert ff.family_request(V, F, S, heading=0., **KW)['recipe'] == ff.RECIPE
    with pytest.raises(WaveFieldError, match='unknown'):
        ff.family_request(V, F, S, heading=0., recipe='fake', **KW)


def test_recipes_never_share_a_cache_entry_or_a_radiation_donor(tmp_path):
    legacy, direct = requests()
    assert ff._path(tmp_path, legacy, .8) != ff._path(tmp_path, direct, .8)
    assert ff.radiation_key(legacy) != ff.radiation_key(direct)
    assert ff._donor_path(tmp_path, legacy, .8) != ff._donor_path(tmp_path, direct, .8)
    ff.save_family_snapshot(tmp_path, record(ff.RECIPE))
    assert ff.missing_family_frequencies(tmp_path, legacy, [.8]) == ()          # legacy record stays recoverable as itself
    assert ff.missing_family_frequencies(tmp_path, direct, [.8]) == (.8,)       # and is not the corrected state
    assert ff.recover_radiation_donor(tmp_path, direct, .8) is None
    assert ff.missing_radiation_rhs(tmp_path, direct, .8) == 7


def test_capacity_is_explicit_versioned_and_direct_only(tmp_path):
    V, F, S = box()
    default = ff.family_request(V, F, S, heading=0., **KW)
    assert 'source_capacity' not in default and ff._panel_ceiling(default) == 600
    refined = ff.family_request(V, F, S, heading=0., recipe=ff.DIRECT_RECIPE, source_capacity='refined_direct_quad_v1', **KW)
    assert ff._panel_ceiling(refined) == 2500
    assert ff.radiation_key(refined) != ff.radiation_key(ff.family_request(V, F, S, heading=0., recipe=ff.DIRECT_RECIPE, **KW))
    with pytest.raises(WaveFieldError, match='source capacity'):
        ff.family_request(V, F, S, heading=0., recipe=ff.RECIPE, source_capacity='refined_direct_quad_v1', **KW)
    with pytest.raises(WaveFieldError, match='source capacity'):
        ff.family_request(V, F, S, heading=0., recipe=ff.DIRECT_RECIPE, source_capacity='other', **KW)
    with pytest.raises(WaveFieldError, match='capacity 2500'):
        ff.family_request(V, np.repeat(F, 200, axis=0), S * 200, heading=0., recipe=ff.DIRECT_RECIPE,
                          source_capacity='refined_direct_quad_v1', **KW)


def test_panel_ceiling_defaults_to_600_and_is_parameterised():
    import capytaine as cpt
    from anyhydro._pressure_body import prepare_body
    V, F, S = box(48, 12, 1)
    assert len(F) > 600
    kw = dict(period=8., heading=0., water_depth=50., rho=1025.)
    with pytest.raises(ValueError, match='panel limit 600'):
        prepare_body(cpt, V, F, S, max_panels=2500, **kw)
    assert len(prepare_body(cpt, V, F, S, max_panels=2500, panel_ceiling=2500, **kw)[1]) == len(F)


def test_legacy_donor_is_refused_by_the_direct_route():
    V, F, S = box()
    with pytest.raises(WaveFieldError, match='radiation basis'):
        ff.solve_family_snapshot(V, F, S, .8, heading=0., donor=record(ff.RECIPE), recipe=ff.DIRECT_RECIPE, **KW)


def test_direct_record_needs_direct_provider_and_hull_potentials():
    with pytest.raises(WaveFieldError, match='direct family'):
        record(ff.DIRECT_RECIPE, provider=dict(backend='capytaine', method='indirect'))
    with pytest.raises(WaveFieldError, match='direct family'):
        record(ff.DIRECT_RECIPE, panels=False)


def test_transient_profile_cannot_select_the_direct_recipe():
    V, F, S = box()
    with pytest.raises(WaveFieldError, match='transient'):
        ff.solve_family_snapshot(V, F, S, .8, heading=0., source_profile=object(), recipe=ff.DIRECT_RECIPE, **KW)


def direct_family(nx=6, ny=3, nz=1, heading=0.):
    V, F, S = box(nx, ny, nz)
    return ff.solve_family_snapshot(V, F, S, 1.2, heading=heading, recipe=ff.DIRECT_RECIPE, **KW)


@native
def test_owner_replay_is_exact_and_refuses_other_state():
    import capytaine as cpt
    s = direct_family()
    assert s.provider['green_function'] == 'AnyhydroDelhommeau' and s.provider['method'] == 'direct'
    assert s.gf_settings['imaginary_part'] and s.gf_settings['finite_depth_prony_decomposition']
    _, _, _, _, body = ff._body_for_family(cpt, s.vertices, s.faces, s.surface_ids, s.request, s.omega)
    problem = cpt.DiffractionProblem(body=body, omega=s.omega, water_depth=50., rho=1025., g=9.81)
    replay = _replay_green_function(cpt, s, problem)
    np.testing.assert_array_equal(replay.find_best_exponential_decomposition(s.prony_kh), s.prony)
    # Re-solve the actual boundary problems: returning the retained Prony alone cannot prove it was the solved state.
    solver = cpt.BEMSolver(green_function=replay)
    radiation = cpt.RadiationProblem(body=body, omega=s.omega, water_depth=50., rho=1025., g=9.81,
                                     radiating_dof=ff.DOF_NAMES[2])
    r = solver.solve(radiation, method='direct', keep_details=True)
    d = solver.solve(problem, method='direct', keep_details=True)
    np.testing.assert_allclose(r.pressure, s.panel_pressure_radiation[2], rtol=1e-10, atol=1e-8)
    from capytaine.bem.airy_waves import airy_waves_pressure
    np.testing.assert_allclose(d.pressure + airy_waves_pressure(s.centers, problem),
                               s.panel_pressure_incident_diffraction, rtol=1e-10, atol=1e-8)
    with pytest.raises(WaveFieldError, match='not retained'):
        replay.find_best_exponential_decomposition(s.prony_kh * 1.01)

    class Wrong:
        provider = s.provider; prony = s.prony; prony_kh = s.prony_kh; prony_method = s.prony_method
        gf_settings = dict(s.gf_settings, imaginary_part='tabulated')
    with pytest.raises(WaveFieldError, match='settings differ'):
        _replay_green_function(cpt, Wrong, problem)

    class Kh(Wrong):
        gf_settings = s.gf_settings; prony_kh = s.prony_kh * 1.01
    with pytest.raises(WaveFieldError, match='dimensionless'):
        _replay_green_function(cpt, Kh, problem)


@native
def test_coefficients_close_against_the_retained_panel_pressures_and_reload_identically(tmp_path):
    s = direct_family()
    dofs = np.zeros((6, len(s.areas)))
    r = s.centers - np.array(KW['cg_world'])
    dofs[:3] = s.normals.T
    dofs[3:] = np.cross(r, s.normals).T
    # p per unit displacement; force on body = -int p n; Z = -w^2 A - i w B for the same exp(-i w t) displacement
    force = -(s.panel_pressure_radiation * s.areas) @ dofs.T
    Z = (s.omega ** 2 * s.added_mass + 1j * s.omega * s.damping)
    np.testing.assert_allclose(force.T, Z, rtol=0, atol=1e-8 * np.abs(Z).max())
    ff.save_family_snapshot(tmp_path, s)
    back = ff.recover_family_snapshot(tmp_path, s.request, s.omega)
    assert back.snapshot_id == s.snapshot_id
    meta, arrays = ff.snapshot_to_payload(s)
    assert ff.snapshot_from_payload(meta, {k: np.array(v) for k, v in arrays.items()}).snapshot_id == s.snapshot_id


@native
def test_green_representation_recovers_the_hull_potential_and_the_free_surface_field():
    s = direct_family(12, 6, 2)
    bottom = np.nonzero(s.normals[:, 2] < -.5)[0][len(s.normals) // 6]
    side = np.nonzero(np.abs(s.normals[:, 1]) > .5)[0][3]
    for panel in (bottom, side):
        point = (s.centers[panel] + .02 * s.normals[panel])[None]
        for k in (1, 2, 3, None):
            phi = ff._potentials(s, point, radiation_index=k)[0]
            if k is None:
                from capytaine.bem.airy_waves import airy_waves_potential
                import capytaine as cpt
                _, _, _, _, body = ff._body_for_family(cpt, s.vertices, s.faces, s.surface_ids, s.request, s.omega)
                problem = cpt.DiffractionProblem(body=body, omega=s.omega, water_depth=50., rho=1025., g=9.81)
                wave = np.asarray(airy_waves_potential(s.centers[panel][None], problem))[0]
                reference = (s.panel_pressure_incident_diffraction[panel] / (1j * s.omega * 1025.))
                assert abs(phi - reference) <= .05 * max(abs(reference), abs(wave))
            else:
                reference = s.panel_pressure_radiation[k, panel] / (1j * s.omega * 1025.)
                assert abs(phi - reference) <= .05 * max(np.abs(s.panel_pressure_radiation[k]).max() / (s.omega * 1025.), 1e-12)
    fields = ff.evaluate_floating_elevations(s, np.array([[3.5, 0., 0.], [0., 2.5, 0.]]), np.array([[1., 0., 0.], [0., 1., 0.]]))
    full = ff.evaluate_floating_fields(s, np.array([[3.5, 0., 0.], [0., 2.5, 0.]]), np.array([[1., 0., 0.], [0., 1., 0.]]))
    np.testing.assert_array_equal(fields.eta_radiation, full.eta_radiation)
    np.testing.assert_array_equal(fields.eta_incident_diffraction, full.eta_incident_diffraction)
    # the wave is present at a point well outboard of the hull, close to its unit incident amplitude
    far = ff.evaluate_floating_elevations(s, np.array([[60., 0., 0.]]), np.array([[1., 0., 0.]]))
    assert abs(abs(far.eta_incident_diffraction[0]) - 1.) < .2


@native
def test_donor_reuses_radiation_and_only_solves_the_new_heading():
    d = direct_family()
    V, F, S = box()
    h = ff.solve_family_snapshot(V, F, S, 1.2, heading=.7, donor=d, recipe=ff.DIRECT_RECIPE, **KW)
    np.testing.assert_array_equal(h.damping, d.damping)
    np.testing.assert_array_equal(h.prony, d.prony)
    fresh = direct_family(heading=.7)
    np.testing.assert_allclose(h.excitation, fresh.excitation, rtol=1e-10, atol=1e-6)
    np.testing.assert_allclose(fresh.damping, d.damping, rtol=1e-10, atol=1e-8)


@native
def test_record_keeps_the_raw_unsymmetrised_damping():
    s = direct_family(12, 4, 2)
    assert not np.array_equal(s.damping, s.damping.T)
