"""Deterministic hashing tests."""

from __future__ import annotations

import numpy as np
import pytest

from anyhydro import (
    RecordValidationError,
    dataset_content_hash,
    mesh_hash,
    request_hash,
)
from anyhydro.fixtures import make_fixture_request, make_two_body_fixture


class TestMeshHash:
    def test_detects_vertex_change(self):
        v1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        v2 = v1.copy()
        v2[1, 0] = 1.0000001
        faces = np.array([[0, 1, 1]], np.int32)
        sizes = np.array([3], np.uint8)
        assert mesh_hash("b", v1, faces, sizes) != mesh_hash("b", v2, faces, sizes)

    def test_detects_body_id_change(self):
        v = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        faces = np.array([[0, 1, 1]], np.int32)
        sizes = np.array([3], np.uint8)
        assert mesh_hash("b1", v, faces, sizes) != mesh_hash("b2", v, faces, sizes)

    def test_endian_independent(self):
        v_le = np.array([[0.5, 0.25, 0.0]], dtype="<f8")
        v_be = v_le.astype(">f8")
        faces = np.array([[0, 0, 0]], np.int32)
        sizes = np.array([3], np.uint8)
        assert mesh_hash("b", v_le, faces, sizes) == mesh_hash("b", v_be, faces, sizes)


class TestRequestHash:
    def test_stable(self):
        r1 = make_fixture_request()
        r2 = make_fixture_request()
        assert r1.content_hash() == r2.content_hash()

    def test_detects_omega_change(self):
        r = make_fixture_request()
        args = dict(
            omegas=r.omegas,
            wave_directions=r.wave_directions,
            water_depth=r.water_depth,
            rho=r.rho,
            gravity=r.gravity,
            include_zero_frequency=r.include_zero_frequency,
            include_infinite_frequency=r.include_infinite_frequency,
            backend_settings=r.backend_settings,
        )
        base = request_hash(**args)
        args["omegas"] = r.omegas * 1.0000001
        assert request_hash(**args) != base

    def test_detects_settings_change(self):
        r = make_fixture_request()
        args = dict(
            omegas=r.omegas,
            wave_directions=r.wave_directions,
            water_depth=r.water_depth,
            rho=r.rho,
            gravity=r.gravity,
            include_zero_frequency=r.include_zero_frequency,
            include_infinite_frequency=r.include_infinite_frequency,
            backend_settings={"different": "settings"},
        )
        assert request_hash(**args) != r.content_hash()


class TestDatasetContentHash:
    def test_deterministic(self):
        assert dataset_content_hash(
            make_two_body_fixture()
        ) == dataset_content_hash(make_two_body_fixture())

    def test_excludes_created_utc(self):
        ds1 = make_two_body_fixture()
        ds2 = make_two_body_fixture()
        ds2.attrs["created_utc"] = "1999-12-31T23:59:59+00:00"
        assert dataset_content_hash(ds1) == dataset_content_hash(ds2)

    def test_detects_data_change(self):
        ds1 = make_two_body_fixture()
        ds2 = make_two_body_fixture()
        values = ds2["added_mass"].values.copy()
        values[0, 0, 0] += 1e-6
        ds2["added_mass"].values = values
        assert dataset_content_hash(ds1) != dataset_content_hash(ds2)

    def test_detects_convention_change(self):
        ds1 = make_two_body_fixture()
        ds2 = make_two_body_fixture()
        ds2.attrs["phasor_convention"] = "wrong"
        assert dataset_content_hash(ds1) != dataset_content_hash(ds2)

    def test_detects_provenance_change(self):
        import json

        ds1 = make_two_body_fixture()
        ds2 = make_two_body_fixture()
        hashes = json.loads(ds2.attrs["mesh_hashes"])
        hashes[sorted(hashes)[0]] = "0" * 64
        ds2.attrs["mesh_hashes"] = json.dumps(hashes)
        assert dataset_content_hash(ds1) != dataset_content_hash(ds2)

    def test_invariant_under_dof_reordering(self):
        from anyhydro import reorder_dataset

        ds = make_two_body_fixture()
        dofs = [str(d) for d in ds["influenced_dof"].values.tolist()]
        shuffled = dofs[6:12] + dofs[0:6]
        shuffled_ds = ds.sel(influenced_dof=shuffled, radiating_dof=shuffled)
        assert dataset_content_hash(shuffled_ds) == dataset_content_hash(ds)
        assert dataset_content_hash(reorder_dataset(shuffled_ds)) == dataset_content_hash(ds)


class TestExtensionAttrHashing:
    """Focused additions: new JSON attrs hash canonically and tamper-proof."""

    def test_json_and_dict_forms_hash_identically(self):
        import json

        ds = make_two_body_fixture()
        payload = {"note": "identical"}
        a = ds.copy(deep=True)
        a.attrs["coefficient_frames"] = json.dumps(payload, sort_keys=True)
        b = ds.copy(deep=True)
        b.attrs["coefficient_frames"] = payload
        assert dataset_content_hash(a) == dataset_content_hash(b)

    def test_new_attrs_in_json_attr_registry(self):
        from anyhydro.hashing import JSON_DATASET_ATTRS

        assert "coefficient_frames" in JSON_DATASET_ATTRS
        assert "hydrostatic_contract" in JSON_DATASET_ATTRS

    def test_contract_attach_changes_hash(self):
        from anyhydro import HydrostaticContract, attach_hydrostatic_contract

        ds = make_two_body_fixture()
        contract = HydrostaticContract(
            role="BUOYANCY_ONLY",
            force_basis="GENERALIZED_DOF",
            stiffness_derivative="INCREMENTAL_ABOUT_EQUILIBRIUM",
            bodies=("fixture-body-alpha", "fixture-body-beta"),
            baseline_wrench=None,
            provenance={"source": "hashing test"},
            full_source=None,
        )
        attached = attach_hydrostatic_contract(ds, contract)
        assert dataset_content_hash(attached) != dataset_content_hash(ds)
