"""Synthetic fixture property tests."""

from __future__ import annotations

import numpy as np

from anyhydro import dataset_content_hash, validate_for_time_domain
from anyhydro.fixtures import (
    FIXTURE_BODY_IDS,
    make_fixture_bodies,
    make_fixture_request,
    make_two_body_fixture,
)


class TestFixture:
    def test_full_12x12_shapes(self, two_body_dataset):
        ds = two_body_dataset
        assert ds.sizes["influenced_dof"] == 12
        assert ds.sizes["radiating_dof"] == 12
        assert ds.sizes["body"] == 2
        assert ds["added_mass"].dims == ("omega", "influenced_dof", "radiating_dof")
        assert ds["excitation_force"].dtype == np.complex128
        assert ds["hydrostatic_stiffness"].shape == (12, 12)

    def test_cross_body_blocks_nonzero(self, two_body_dataset):
        ds = two_body_dataset
        for var in ("added_mass", "radiation_damping"):
            values = ds[var].values
            cross = values[:, 0:6, 6:12]
            assert np.all(cross != 0.0), f"{var} lost cross-body coupling"
            cross2 = values[:, 6:12, 0:6]
            assert np.all(cross2 != 0.0)

    def test_excitation_is_genuinely_complex(self, two_body_dataset):
        exc = two_body_dataset["excitation_force"].values
        assert np.any(exc.imag != 0.0)
        assert np.any(exc.real != 0.0)

    def test_deterministic_content_hash(self):
        ds1 = make_two_body_fixture()
        ds2 = make_two_body_fixture()
        assert dataset_content_hash(ds1) == dataset_content_hash(ds2)

    def test_created_utc_excluded_from_content_hash(self):
        ds1 = make_two_body_fixture()
        ds2 = make_two_body_fixture()
        ds2.attrs["created_utc"] = "2000-01-01T00:00:00+00:00"
        assert dataset_content_hash(ds1) == dataset_content_hash(ds2)

    def test_fixture_is_honestly_labeled(self, two_body_dataset):
        ds = two_body_dataset
        assert ds.attrs["qualification_status"] == "development_fixture"
        assert ds.attrs["backend"] == "synthetic-fixture"
        assert ds.attrs["backend_commit"] == "unknown"

    def test_fixture_admits_to_time_domain(self, two_body_dataset):
        validate_for_time_domain(two_body_dataset)

    def test_a_inf_is_not_last_sample_substitution(self, two_body_dataset):
        ds = two_body_dataset
        a_inf = ds["added_mass_infinite"].values
        last = ds["added_mass"].values[-1]
        assert not np.allclose(a_inf, last)
        assert ds["added_mass_infinite"].attrs["source"]

    def test_fixture_bodies_match_dataset_identity(self, two_body_dataset):
        import json

        ds = two_body_dataset
        bodies = make_fixture_bodies()
        hashes = json.loads(ds.attrs["mesh_hashes"])
        assert set(hashes) == set(FIXTURE_BODY_IDS)
        for body in bodies:
            assert hashes[body.body_id] == body.mesh.mesh_hash
        assert ds.attrs["request_hash"] == make_fixture_request().content_hash()
