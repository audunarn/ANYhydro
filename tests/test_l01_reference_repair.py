"""Public synthetic fixtures only; these tests do not qualify the private L01 case."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from anyhydro import floating_radiation_fields as ff
from anyhydro.direct_lid_profile import (create_source_profile, WATERLINE_PROFILE_METHOD,
                                       profile_from_dict, profile_from_request)
from anyhydro.wave_surface_fields import WaveFieldError
from anyhydro.waterline_lid import generate_waterline_lid

KW = dict(heading=0., water_depth=50., rho=1025., gravity=9.81, Q0=np.eye(3), cg_world=[0., 0., -.4])


def box():
    v = np.array([[-3,-2,-1], [3,-2,-1], [3,2,-1], [-3,2,-1],
                  [-3,-2,0], [3,-2,0], [3,2,0], [-3,2,0]], dtype=float)
    f = np.array([[0,3,2,1], [0,1,5,4], [1,2,6,5], [2,3,7,6], [3,0,4,7]])
    return v, f, ['hull'] * len(f)


def dependencies():
    pytest.importorskip('capytaine')
    pytest.importorskip('shapely')
    pytest.importorskip('anymesher.triangulation')


def test_legacy_quadrature_identity_and_invalid_new_rules():
    v, f, ids = box()
    old = ff.family_request(v, f, ids, recipe=ff.DIRECT_RECIPE, **KW)
    assert ff.family_request(v, f, ids, recipe=ff.DIRECT_RECIPE, source_quadrature=None, **KW) == old
    new = ff.family_request(v, f, ids, recipe=ff.DIRECT_RECIPE, source_quadrature='Gauss-Legendre 2', **KW)
    assert 'source_quadrature' not in old and ff.radiation_key(new) != ff.radiation_key(old)
    for rule in [True, 2, [], 'Gauss-Legendre 4', 'centroid']:
        with pytest.raises(WaveFieldError, match='quadrature'):
            ff.family_request(v, f, ids, recipe=ff.DIRECT_RECIPE, source_quadrature=rule, **KW)
    with pytest.raises(WaveFieldError, match='quadrature'):
        ff.family_request(v, f, ids, source_quadrature='Gauss-Legendre 2', **KW)
    with pytest.raises(WaveFieldError, match='quadrature'):
        ff._source_quadrature(dict(new, transient_source_profile={}))


def test_prechange_v1_record_has_pinned_serialized_identity():
    # Public synthetic record in the pre-change v1 fieldset; digest is fixed,
    # not produced by the serializer under test.
    record = dict(method='capytaine-3.0-direct-internal-lid-v1', hull_hash='0'*64,
        lid_z=-.1, faces_max_radius=2., source_panel_limit=4000,
        lid_vertices=[[0.,0.,-.1],[0.,1.,-.1],[1.,1.,-.1],[1.,0.,-.1]],
        lid_faces=[[0,1,2,3]],
        source_vertices=[[0.,0.,-1.],[0.,1.,-1.],[1.,1.,-1.],[1.,0.,-1.],
                         [0.,0.,-.1],[0.,1.,-.1],[1.,1.,-.1],[1.,0.,-.1]],
        source_faces=[[0,1,2,3],[4,5,6,7]], hull_mask=[True,False], max_work_bytes=536870912)
    p = profile_from_dict(record)
    assert p.to_dict() == record
    assert p.content_hash() == '17b39c81c26ccc5b448d1de57b6073ea87f4ab01d851bace6a2bda43f40cc4bf'


def test_base_import_does_not_load_optional_mesh_backends():
    code = ('import sys; import anyhydro.waterline_lid; import anyhydro.direct_lid_profile; '
            'assert not any(n in sys.modules for n in ("capytaine", "shapely", "anymesher", "PySide6", "anyfesolver", "anysolver"))')
    subprocess.run([sys.executable, '-c', code], check=True, timeout=15,
                   env=dict(os.environ, PYTHONPATH=str(__import__('pathlib').Path(ff.__file__).parents[1])))


def test_boundary_and_resource_refusals_precede_optional_generation():
    v, f, _ = box()
    for size in [0., -1., float('nan'), True, 1e-100, None]:
        with pytest.raises(WaveFieldError):
            generate_waterline_lid(v, f, cell_size=size, max_panels=100, max_work_bytes=2**20)
    with pytest.raises(WaveFieldError, match='generation work'):
        generate_waterline_lid(v, f, cell_size=1., max_panels=100, max_work_bytes=1)
    vv = v.copy(); vv[4:, 2] = -1e-8
    with pytest.raises(WaveFieldError, match='exactly z=0'):
        generate_waterline_lid(vv, f, cell_size=1., max_panels=100, max_work_bytes=2**20)
    with pytest.raises(WaveFieldError, match='one loop'):
        generate_waterline_lid(np.vstack([v, v+[10,0,0]]), np.vstack([f, f+len(v)]),
                               cell_size=1., max_panels=100, max_work_bytes=2**20)


def test_fitted_lid_covers_concave_polygon_and_preserves_boundary():
    dependencies()
    # L-shaped prism: no convex-hull substitution, boundary clipping or inserted hull vertices.
    xy = np.array([[0,0], [3,0], [3,1], [1,1], [1,3], [0,3]], float)
    v = np.column_stack([xy, np.zeros(6)])
    # Only boundary extraction is under test here; an open polygon face is sufficient.
    f = np.array([[0,1,2,2], [0,2,3,3], [0,3,5,5], [3,4,5,5]])
    before = v.copy(), f.copy()
    lv, lf = generate_waterline_lid(v, f, cell_size=.5, max_panels=150, max_work_bytes=2**24)
    p = lv[lf[:, :3]]
    normals = np.cross(p[:,1]-p[:,0], p[:,2]-p[:,0])
    assert np.all(normals[:,2] < 0) and np.all(lv[:,2] == 0)
    assert abs(np.sum(-normals[:,2]/2)-5.) < 1e-12
    assert set(map(tuple, xy)).issubset(set(map(tuple, lv[:, :2])))
    np.testing.assert_array_equal(v, before[0]); np.testing.assert_array_equal(f, before[1])


def test_waterline_profile_roundtrip_identity_tampering_and_bounds():
    dependencies(); v, f, ids = box()
    p = create_source_profile(v, f, ids, lid_z=0., method=WATERLINE_PROFILE_METHOD,
                              waterline_cell_size=1., water_depth=50.)
    assert p.lid_z == 0. and p.waterline_cell_size == 1.
    assert profile_from_dict(p.to_dict()).content_hash() == p.content_hash()
    r = ff.family_request(v, f, ids, recipe=ff.DIRECT_LID_RECIPE, direct_lid_profile=p, **KW)
    assert profile_from_request(r).content_hash() == p.content_hash()
    np.testing.assert_array_equal(p.source_vertices[p.source_faces[p.hull_mask]], v[f])
    bad = p.to_dict(); bad['waterline_cell_size'] = 2.
    with pytest.raises(WaveFieldError, match='canonical'):
        ff.family_request(v, f, ids, recipe=ff.DIRECT_LID_RECIPE, direct_lid_profile=bad, **KW)
    with pytest.raises(WaveFieldError):
        create_source_profile(v, f, ids, lid_z=-.1, method=WATERLINE_PROFILE_METHOD,
                              waterline_cell_size=1., water_depth=50.)
    with pytest.raises(WaveFieldError):
        create_source_profile(v, f, ids, lid_z=0., method=WATERLINE_PROFILE_METHOD,
                              waterline_cell_size=.5, water_depth=50., source_panel_limit=10)
    old = create_source_profile(v, f, ids, lid_z=-.1, faces_max_radius=1., water_depth=50.)
    assert 'waterline_cell_size' not in old.to_dict()
    assert profile_from_dict(old.to_dict()).content_hash() == old.content_hash()


@pytest.mark.skipif(os.environ.get(ff.NATIVE_GATE) != '1', reason='explicit native family solve')
def test_real_public_solve_cache_donor_recovery_and_quadrature_binding(tmp_path):
    dependencies(); import capytaine as cpt
    v, f, ids = box(); before = v.copy(), f.copy()
    p = create_source_profile(v, f, ids, lid_z=0., method=WATERLINE_PROFILE_METHOD,
                              waterline_cell_size=1., water_depth=50.)
    request = dict(recipe=ff.DIRECT_LID_RECIPE, direct_lid_profile=p,
                   source_quadrature='Gauss-Legendre 2', **KW)
    s = ff.solve_family_snapshot(v, f, ids, 1.2, **request)
    assert s.provider['source_quadrature'] == 'Gauss-Legendre 2'
    _,_,_,_,body = ff._body_for_family(cpt, v, f, ids, s.request, s.omega)
    assert body.mesh_including_lid.quadrature_points[0].shape[1] == 4
    wrench = np.column_stack([s.normals, np.cross(s.centers-np.array(KW['cg_world']), s.normals)])*s.areas[:,None]
    force = -wrench.T@s.panel_pressure_radiation.T
    np.testing.assert_allclose(force, s.omega**2*s.added_mass+1j*s.omega*s.damping, rtol=1e-12, atol=1e-8)
    ff.save_family_snapshot(tmp_path, s)
    restored = ff.recover_family_snapshot(tmp_path, s.request, s.omega)
    assert restored.snapshot_id == s.snapshot_id
    recovery = ff._NativeRecovery(restored)
    assert recovery.kw['body'].mesh_including_lid.quadrature_method == 'Gauss-Legendre 2'
    assert np.isfinite(recovery.potential(np.array([[4.,0.,-.5]]), 0)).all()
    changed_heading = dict(request, heading=.2)
    donor = ff.solve_family_snapshot(v, f, ids, 1.2, donor=s, **changed_heading)
    np.testing.assert_array_equal(donor.damping, s.damping)
    with pytest.raises(WaveFieldError, match='basis'):
        ff.solve_family_snapshot(v, f, ids, 1.2, donor=s, **dict(request, source_quadrature=None))
    with pytest.raises(WaveFieldError, match='provider'):
        replace(s, provider=dict(s.provider, source_quadrature='other'), snapshot_id='')
    np.testing.assert_array_equal(v, before[0]); np.testing.assert_array_equal(f, before[1])


@pytest.mark.skipif(os.environ.get(ff.NATIVE_GATE) != '1', reason='explicit native family solve')
def test_direct_without_lid_quadrature_solve_and_retained_recovery(tmp_path, monkeypatch):
    pytest.importorskip('capytaine'); import capytaine as cpt
    v, f, ids = box()
    s = ff.solve_family_snapshot(v, f, ids, 1.2, recipe=ff.DIRECT_RECIPE,
                                source_quadrature='Gauss-Legendre 2', **KW)
    assert s.full_potential_radiation is None and s.full_potential_diffraction is None
    _,_,_,_,body = ff._body_for_family(cpt, v, f, ids, s.request, s.omega)
    assert body.mesh_including_lid.quadrature_points[0].shape[1] == 4
    ff.save_family_snapshot(tmp_path, s)
    loaded = ff.recover_family_snapshot(tmp_path, s.request, s.omega)
    monkeypatch.setattr(cpt.BEMSolver, 'solve', lambda *a, **kw: pytest.fail('retained recovery attempted solve'))
    recovery = ff._NativeRecovery(loaded)
    assert recovery.kw['body'].mesh.quadrature_method == 'Gauss-Legendre 2'
    points = np.array([[4.,0.,-.5], [0.,3.,-.5]])
    for j in list(range(6))+[None]:
        assert np.isfinite(recovery.potential(points, j)).all()
    wrench = np.column_stack([s.normals, np.cross(s.centers-np.array(KW['cg_world']), s.normals)])*s.areas[:,None]
    np.testing.assert_allclose(-wrench.T@s.panel_pressure_radiation.T,
                              s.omega**2*s.added_mass+1j*s.omega*s.damping, rtol=1e-12, atol=1e-8)
