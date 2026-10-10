"""Opt-in numerical auxiliary-lid recipe contracts; NO BEM execution."""
from dataclasses import replace
import json
import numpy as np
import pytest
from anyhydro import CapytaineBackend, CapytaineExecutionLimits, GeometryValidationError, CapabilityError
from anyhydro.first_slice import make_first_slice_bodies, verify_first_slice_fixture
from anyhydro.source_study_case import (SourceStudyCase, SourceStudyCaseError,
    make_prospective_source_study_case, build_study_bodies, build_study_request,
    FIRST_SLICE_SUBMERGED_LID_GEOMETRY_RECIPE)
from anyhydro.source_study import prepare_source_study, load_source_study
from anyhydro.schema import assemble_hydro_dataset
POLICY = "submerged_0p1m_v1"
def case():
    data = make_prospective_source_study_case().to_dict()
    data.update(geometry_recipe=FIRST_SLICE_SUBMERGED_LID_GEOMETRY_RECIPE, levels=[2,3], omegas=[1.5,8.])
    return SourceStudyCase.from_dict(data)
def dataset(bodies, request):
    n=12;w=len(request.omegas);d=len(request.wave_directions)
    return assemble_hydro_dataset(bodies,request,added_mass=np.zeros((w,n,n)),
        radiation_damping=np.zeros((w,n,n)),excitation_force=np.zeros((w,d,n),complex),
        hydrostatic_stiffness=np.zeros((n,n)),rotation_center=[b.rotation_center for b in bodies],
        center_of_mass=[b.center_of_mass for b in bodies],center_of_buoyancy=[b.center_of_mass for b in bodies],
        backend="stub",backend_version="0",qualification_status="development_fixture")
@pytest.mark.parametrize("n", [2,3,4,6])
def test_physical_geometry_invariant_and_lid_identity_changed(n):
    old=make_first_slice_bodies(subdivisions=n)
    new=make_first_slice_bodies(subdivisions=n,lid_policy=POLICY)
    old_report=verify_first_slice_fixture(old,subdivisions=n)
    report=verify_first_slice_fixture(new,subdivisions=n,lid_policy=POLICY)
    assert report["auxiliary_lid"]["elevation_m"] == -.1
    for a,b in zip(old,new):
        assert a.mesh.mesh_hash==b.mesh.mesh_hash
        assert a.lid_mesh.mesh_hash!=b.lid_mesh.mesh_hash
        np.testing.assert_array_equal(a.mesh.vertices,b.mesh.vertices)
        np.testing.assert_array_equal(a.lid_mesh.vertices[:,:2],b.lid_mesh.vertices[:,:2])
        np.testing.assert_array_equal(b.lid_mesh.vertices[:,2],-.1)
        assert a.rotation_center==b.rotation_center and a.center_of_mass==b.center_of_mass
        assert a.enabled_dofs==b.enabled_dofs
        for key in ["submerged_volume_m3","center_of_buoyancy_m","waterplane_area_m2"]:
            assert old_report["bodies"][a.body_id][key]==report["bodies"][b.body_id][key]
    with pytest.raises(GeometryValidationError): verify_first_slice_fixture(new,subdivisions=n)
def test_public_prepare_roundtrip_and_request_identity(tmp_path):
    new=case();data=new.to_dict()
    assert SourceStudyCase.from_dict(data).content_hash()==new.content_hash()
    old=SourceStudyCase.from_dict(dict(data,geometry_recipe="first-slice-two-box-wetted-hull-lid-v1"))
    assert old.content_hash()!=new.content_hash()
    assert build_study_request(old).content_hash()!=build_study_request(new).content_hash()
    out=tmp_path/"case";prepare_source_study(new,out)
    assert load_source_study(out)["case_hash"]==new.content_hash()
    inv=json.loads((out/"geometry_inventory.json").read_text())
    assert inv["request"]["backend_settings"]=={"auxiliary_lid_policy":POLICY}
    assert inv["levels"]["3"]["mesh_hashes"] != {}
@pytest.mark.parametrize("value", ["unknown",None,True,3])
def test_unknown_policy_refuses(value):
    with pytest.raises(GeometryValidationError): make_first_slice_bodies(lid_policy=value)
def test_band_above_fixed_coverage_refuses_before_prepare():
    with pytest.raises(SourceStudyCaseError,match="8 rad/s"):
        SourceStudyCase.from_dict(dict(case().to_dict(),omegas=[1.5,8.01]))
def test_explicit_backend_admission_and_cache_identity(monkeypatch):
    c=case();bodies=build_study_bodies(c,2);request=build_study_request(c)
    backend=CapytaineBackend(execution_limits=CapytaineExecutionLimits(max_finite_omegas=2))
    monkeypatch.setattr(CapytaineBackend,"_solve_fresh",lambda self,b,r,s:dataset(b,r))
    result=backend.solve(bodies,request)
    assert result.attrs["auxiliary_lid_policy"]==POLICY
    old=SourceStudyCase.from_dict(dict(c.to_dict(),geometry_recipe="first-slice-two-box-wetted-hull-lid-v1"))
    old_request=build_study_request(old)
    assert backend._cache_identity(bodies,request,dict(request.backend_settings),"3.0.0") != backend._cache_identity(
        make_first_slice_bodies(),old_request,{},"3.0.0")
    with pytest.raises(GeometryValidationError): backend.solve(bodies,old_request)
    with pytest.raises(GeometryValidationError): backend.solve(make_first_slice_bodies(),request)
    with pytest.raises(CapabilityError): backend.solve(bodies,replace(request,backend_settings={"auxiliary_lid_policy":"bogus"}))
    with pytest.raises(CapabilityError): backend.solve(bodies,replace(request,omegas=np.array([8.1])))
def test_arbitrary_submerged_geometry_is_not_fixed_recipe():
    c=case();bodies=build_study_bodies(c,2)
    bad=replace(bodies[0],rotation_center=(-1.,0.,0.))
    with pytest.raises(GeometryValidationError):
        CapytaineBackend()._admit_geometry((bad,bodies[1]),150.,lid_policy=POLICY)

@pytest.mark.parametrize('policy', ['bogus', 4, None])
def test_neutral_dataset_rejects_bad_recognized_policy(policy):
    from anyhydro.schema import validate_hydro_dataset
    from anyhydro.errors import SchemaError
    c=case();ds=dataset(build_study_bodies(c,2),build_study_request(c));ds.attrs['auxiliary_lid_policy']=policy
    with pytest.raises(SchemaError):validate_hydro_dataset(ds)
