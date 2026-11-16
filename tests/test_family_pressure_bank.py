"""Family-cache -> ImportedPressureBank adapter: composition, closure, refusals, import hygiene."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from anyhydro import floating_radiation_fields as ff
from anyhydro.family_pressure_bank import (FamilyPressureBankError, floating_family_to_pressure_bank, linear_rao,
                                           snapshot_wrench_operator)
from anyhydro.orcawave_pressure import ImportedPressureBank
from anyhydro.wave_surface_fields import WaveFieldMissing

OMEGAS = (0.6, 0.9, 1.4)


def box():
    """Unit cube, six quads wound counter-clockwise seen from outside."""
    v = np.array([[x, y, z - 0.5] for z in (0., 1.) for y in (-.5, .5) for x in (-.5, .5)], float)
    f = np.array([[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]])
    return v, f


CG = (0.7, -0.4, 0.3)


def rotation(a, b, c):
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def body_wrench(Q, cg, centers, normals, areas, p):
    """-sum over panels of [dF; (c-cg) x dF] with dF = a n p in the world frame, then components along Q's columns."""
    force = np.zeros(3, complex)
    moment = np.zeros(3, complex)
    for c, n, a, pk in zip(centers, normals, areas, p):
        df = -a * n * pk
        force += df
        moment += np.cross(c - np.asarray(cg), df)
    return np.concatenate([Q.T @ force, Q.T @ moment])


def make_snapshot(omega, rng, warp=0.):
    v, f = box()
    v[3, 2] += warp
    n = len(f)
    Q0 = rotation(0.31, -0.22, 0.47)
    req = ff.family_request(v, f, [f'p{i}' for i in range(n)], heading=0.3, water_depth=50., rho=1025., gravity=9.81,
                            Q0=Q0, cg_world=CG)
    quad = v[f]
    centers = quad.mean(axis=1)
    normals = np.cross(quad[:, 2] - quad[:, 0], quad[:, 3] - quad[:, 1])
    areas = np.linalg.norm(normals, axis=1) / 2
    normals = normals / np.linalg.norm(normals, axis=1)[:, None]
    assert np.all(np.einsum('ij,ij->i', normals, centers - np.array([0, 0, 0.])) > 0)
    ppr = rng.normal(size=(6, n)) + 1j * rng.normal(size=(6, n))
    ppd = rng.normal(size=n) + 1j * rng.normal(size=n)
    # Independent oracle: physical definition, world frame, panel by panel, then expressed in the mean body axes.
    G = np.array([body_wrench(Q0, CG, centers, normals, areas, ppr[k]) for k in range(6)]).T
    A, B = G.real / omega ** 2, G.imag / omega      # = w^2 A + i w B per unit displacement
    F = body_wrench(Q0, CG, centers, normals, areas, ppd)
    return ff.FloatingFamilySnapshot(req, omega, v, f, tuple(f'p{i}' for i in range(n)), rng.normal(size=(6, n)) + 0j,
                                     rng.normal(size=n) + 0j, A, B, F, {'backend': 'synthetic'},
                                     panel_pressure_radiation=ppr, panel_pressure_incident_diffraction=ppd,
                                     centers=centers, normals=normals, areas=areas)


@pytest.fixture()
def family(tmp_path):
    rng = np.random.default_rng(7)
    snaps = [make_snapshot(w, rng) for w in OMEGAS]
    for s in snaps:
        ff.save_family_snapshot(tmp_path, s)
    return tmp_path, snaps


def test_fixed_diffraction_reproduces_saved_pressure_and_closes_on_force(family):
    pytest.importorskip("anygeometry", reason="validate_geometry needs ANYgeometry (the orcawave extra)")
    cache, snaps = family
    bank = floating_family_to_pressure_bank(cache, snaps[0].request, OMEGAS)
    assert bank.composition == 'fixed_diffraction'
    assert bank.heading == pytest.approx(0.3) and bank.rho == 1025. and bank.water_depth == 50.
    bank.validate_geometry()
    assert ImportedPressureBank.from_dict(bank.to_dict()).bank_id == bank.bank_id
    for k, s in enumerate(snaps):
        assert np.array_equal(bank.responses[k], s.panel_pressure_incident_diffraction)
        W = snapshot_wrench_operator(s)
        assert np.linalg.norm(-W.T @ bank.responses[k] - s.excitation) <= 1e-10 * np.linalg.norm(s.excitation)
        # world-frame force from the bank's own areas/normals equals Q F_body
        Q = np.asarray(s.request['Q0'])
        force = -(bank.areas[:, None] * bank.normals * bank.responses[k][:, None]).sum(axis=0)
        assert np.allclose(force, Q @ s.excitation[:3], rtol=1e-10, atol=1e-10)


def test_wrench_operator_matches_the_independent_moment_oracle_and_a_sign_flip_fails(family):
    _, snaps = family
    s = snaps[0]
    p = s.panel_pressure_incident_diffraction
    oracle = body_wrench(np.asarray(s.request['Q0']), CG, s.centers, s.normals, s.areas, p)
    assert np.abs(oracle[3:]).max() > 1e-3                                    # the moment is genuinely nonzero
    assert np.allclose(-snapshot_wrench_operator(s).T @ p, oracle, rtol=1e-12, atol=1e-12)
    flipped = snapshot_wrench_operator(s).copy()
    flipped[:, 3:] *= -1
    assert not np.allclose(-flipped.T @ p, oracle, rtol=1e-3, atol=1e-3)


def test_total_composition_with_rao_and_force_closure(family):
    cache, snaps = family
    rng = np.random.default_rng(3)
    rao = rng.normal(size=(3, 6)) + 1j * rng.normal(size=(3, 6))
    bank = floating_family_to_pressure_bank(cache, snaps[0].request, OMEGAS, rao)
    assert bank.composition == 'total_with_source_motion'
    prov = json.loads(bank.provenance_json)
    assert prov['composition_formula'].startswith('p_inc+dif +') and prov['rao_sha256']
    for k, s in enumerate(snaps):
        expect = s.panel_pressure_incident_diffraction + rao[k] @ s.panel_pressure_radiation
        assert np.array_equal(bank.responses[k], expect)
        W = snapshot_wrench_operator(s)
        radiation = (s.omega ** 2 * s.added_mass + 1j * s.omega * s.damping) @ rao[k]
        assert np.linalg.norm(-W.T @ bank.responses[k] - (s.excitation + radiation)) \
            <= 1e-10 * np.linalg.norm(s.excitation + radiation)
    # callable source gives the identical bank
    same = floating_family_to_pressure_bank(cache, snaps[0].request, OMEGAS, lambda w: rao[OMEGAS.index(w)])
    assert same.bank_id == bank.bank_id


def test_linear_rao_solves_equation_of_motion(family):
    _, snaps = family
    s = snaps[1]
    M = np.diag([5., 5, 5, 2, 2, 2]); C = np.diag([0, 0, 40., 10, 10, 0])
    xi = linear_rao(s.omega, s.added_mass, s.damping, s.excitation, M, C)
    lhs = (-s.omega ** 2 * (M + s.added_mass) - 1j * s.omega * s.damping + C) @ xi
    assert np.allclose(lhs, s.excitation, rtol=1e-12, atol=1e-12)
    with pytest.raises(FamilyPressureBankError):
        linear_rao(s.omega, s.added_mass, s.damping, s.excitation, M[:5], C)


def test_refusals(family):
    cache, snaps = family
    req = snaps[0].request
    with pytest.raises(WaveFieldMissing):
        floating_family_to_pressure_bank(cache, req, (0.6, 0.91))
    with pytest.raises(FamilyPressureBankError, match='increasing'):
        floating_family_to_pressure_bank(cache, req, (0.9, 0.6))
    with pytest.raises(FamilyPressureBankError, match='RAO'):
        floating_family_to_pressure_bank(cache, req, OMEGAS, np.ones((2, 6)))
    with pytest.raises(FamilyPressureBankError, match='RAO'):
        floating_family_to_pressure_bank(cache, req, OMEGAS, np.full((3, 6), np.nan))
    with pytest.raises(FamilyPressureBankError, match='contradicts'):
        floating_family_to_pressure_bank(cache, req, OMEGAS, composition='total_with_source_motion')
    with pytest.raises(FamilyPressureBankError, match='contradicts'):
        floating_family_to_pressure_bank(cache, req, OMEGAS, np.ones((3, 6)), composition='fixed_diffraction')


def test_warped_quad_is_carried_unchanged_and_refused_by_the_geometry_owner(tmp_path):
    pytest.importorskip("anygeometry", reason="validate_geometry needs ANYgeometry (the orcawave extra)")
    """The adapter never repairs geometry: a quad outside the owner's planarity gate is refused downstream."""
    warped = make_snapshot(0.6, np.random.default_rng(5), warp=3e-13)
    ff.save_family_snapshot(tmp_path, warped)
    bank = floating_family_to_pressure_bank(tmp_path, warped.request, (0.6,))
    assert np.array_equal(bank.vertices, warped.vertices)
    with pytest.raises(Exception, match='planar'):
        bank.validate_geometry()


def test_no_heavy_imports_at_base_import():
    root = str(Path(ff.__file__).resolve().parents[1])
    code = (f"import sys; sys.path.insert(0, {root!r})\n"
            "import anyhydro.family_pressure_bank as m\n"
            "bad=[n for n in ('capytaine','PySide6','PyQt5','PyQt6','anyfesolver','anysolver') if n in sys.modules]\n"
            "assert not bad, bad")
    subprocess.run([sys.executable, '-I', '-c', code], check=True)


REAL = os.environ.get('ANYHYDRO_L01_FAMILY_DIR')


@pytest.mark.skipif(not REAL, reason='set ANYHYDRO_L01_FAMILY_DIR to the refined-lid audit folder')
def test_local_refined_lid_cache_reproduces_saved_pressure_and_closes():
    root = Path(REAL)
    manifest = json.loads((root.parent / 'BAND-MANIFEST.json').read_text())
    request = manifest['request']
    raw = np.load(root / 'coherent-raw.npz')
    grid = raw['omega']
    idx = [int(np.argmin(abs(grid - w))) for w in (1.0554, 1.9553, 2.7734)]
    om = grid[idx]
    assert np.all(abs(om - (1.0554, 1.9553, 2.7734)) < 1e-3)
    bank = floating_family_to_pressure_bank(root / 'coherent-cache', request, om)
    for j, i in enumerate(idx):
        assert np.abs(bank.responses[j] - raw['pressure_incident_diffraction'][i]).max() <= 1e-12 * np.abs(
            raw['pressure_incident_diffraction'][i]).max()
        s = ff.recover_family_snapshot(root / 'coherent-cache', request, float(om[j]))
        W = snapshot_wrench_operator(s)
        assert np.linalg.norm(-W.T @ bank.responses[j] - raw['F'][i]) <= 1e-10 * np.linalg.norm(raw['F'][i])
    rao = np.array([np.linalg.solve(np.eye(6) * 1e3 * (j + 1), raw['F'][i]) for j, i in enumerate(idx)])
    total = floating_family_to_pressure_bank(root / 'coherent-cache', request, om, rao)
    for j, i in enumerate(idx):
        assert np.abs(total.responses[j] - (raw['pressure_incident_diffraction'][i]
                                            + rao[j] @ raw['pressure_radiation'][i])).max() <= 1e-12 * np.abs(
            total.responses[j]).max()
