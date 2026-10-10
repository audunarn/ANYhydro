"""Persistence tests: NetCDF round-trip, atomicity, validation."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import xarray as xr

from anyhydro import (
    PersistenceError,
    SchemaError,
    dataset_content_hash,
    load_hydro_dataset,
    save_hydro_dataset,
)


class TestRoundTrip:
    def test_real_values_bit_for_bit(self, two_body_dataset, tmp_path):
        path = tmp_path / "fixture.nc"
        save_hydro_dataset(two_body_dataset, path)
        loaded = load_hydro_dataset(path)
        for var in ("added_mass", "radiation_damping", "hydrostatic_stiffness",
                    "added_mass_infinite", "rotation_center", "center_of_mass",
                    "center_of_buoyancy"):
            assert np.array_equal(
                loaded[var].values, two_body_dataset[var].values
            ), var

    def test_complex_values_exact(self, two_body_dataset, tmp_path):
        path = tmp_path / "fixture.nc"
        save_hydro_dataset(two_body_dataset, path)
        loaded = load_hydro_dataset(path)
        assert loaded["excitation_force"].dtype == np.complex128
        assert np.array_equal(
            loaded["excitation_force"].values,
            two_body_dataset["excitation_force"].values,
        )

    def test_attrs_round_trip(self, two_body_dataset, tmp_path):
        path = tmp_path / "fixture.nc"
        save_hydro_dataset(two_body_dataset, path)
        loaded = load_hydro_dataset(path)
        assert loaded.attrs["created_utc"] == two_body_dataset.attrs["created_utc"]
        # JSON attrs are decoded on load; compare decoded forms
        assert loaded.attrs["mesh_hashes"] == json.loads(
            two_body_dataset.attrs["mesh_hashes"]
        )
        assert loaded.attrs["request_hash"] == two_body_dataset.attrs["request_hash"]
        assert loaded["added_mass_infinite"].attrs["source"] == (
            two_body_dataset["added_mass_infinite"].attrs["source"]
        )

    def test_content_hash_stable_across_round_trip(self, two_body_dataset, tmp_path):
        path = tmp_path / "fixture.nc"
        save_hydro_dataset(two_body_dataset, path)
        loaded = load_hydro_dataset(path)
        assert dataset_content_hash(loaded) == dataset_content_hash(two_body_dataset)

    def test_save_does_not_mutate_input(self, two_body_dataset, tmp_path):
        before = dataset_content_hash(two_body_dataset)
        save_hydro_dataset(two_body_dataset, tmp_path / "fixture.nc")
        assert dataset_content_hash(two_body_dataset) == before

    def test_no_temp_files_left_behind(self, two_body_dataset, tmp_path):
        target = tmp_path / "fixture.nc"
        save_hydro_dataset(two_body_dataset, target)
        save_hydro_dataset(two_body_dataset, target)  # overwrite must work
        leftovers = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == []
        assert target.exists()


class TestRejections:
    def test_missing_file(self, tmp_path):
        with pytest.raises(PersistenceError):
            load_hydro_dataset(tmp_path / "does-not-exist.nc")

    def test_missing_target_directory(self, two_body_dataset, tmp_path):
        with pytest.raises(PersistenceError):
            save_hydro_dataset(two_body_dataset, tmp_path / "no-such-dir" / "f.nc")

    def test_load_rejects_invalid_content(self, two_body_dataset, tmp_path):
        path = tmp_path / "fixture.nc"
        save_hydro_dataset(two_body_dataset, path)
        raw = xr.load_dataset(path, engine="scipy")
        raw = raw.drop_vars("excitation_force_real")
        bad = tmp_path / "bad.nc"
        raw.to_netcdf(bad, engine="scipy", format="NETCDF3_64BIT")
        with pytest.raises(SchemaError):
            load_hydro_dataset(bad)

    def test_save_rejects_invalid_dataset(self, tmp_path):
        import xarray as xr

        with pytest.raises(SchemaError):
            save_hydro_dataset(xr.Dataset(), tmp_path / "invalid.nc")


class TestFormatHygiene:
    def test_no_pickle_or_backend_objects_in_file(self, two_body_dataset, tmp_path):
        path = tmp_path / "fixture.nc"
        save_hydro_dataset(two_body_dataset, path)
        blob = path.read_bytes()
        assert b"pickle" not in blob.lower()
        assert b"capytaine" not in blob.lower()

    def test_attrs_are_plain_scalars_or_json(self, two_body_dataset, tmp_path):
        path = tmp_path / "fixture.nc"
        save_hydro_dataset(two_body_dataset, path)
        loaded = load_hydro_dataset(path)
        for key, value in loaded.attrs.items():
            # plain scalar or decoded JSON only: no backend objects, no
            # pickle; nested JSON structures (e.g. body_identity) are allowed
            assert isinstance(value, (str, int, float, dict, list, bool)), key
            json.dumps(value, allow_nan=False)


class TestExtensionAttrsPersistence:
    """Focused addition: frames/contract attrs survive the NetCDF roundtrip."""

    def test_frames_and_contract_round_trip(self, two_body_dataset, tmp_path):
        import numpy as np

        from anyhydro import (
            HydrostaticContract,
            attach_hydrostatic_contract,
            coefficient_frame_axes,
            read_hydrostatic_contract,
            reframe_dataset,
        )
        from anyhydro.fixtures import FIXTURE_BODY_IDS

        alpha, beta = FIXTURE_BODY_IDS
        reframed = reframe_dataset(
            two_body_dataset,
            new_axes={
                alpha: np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                beta: np.eye(3),
            },
        )
        contracted = attach_hydrostatic_contract(
            reframed,
            HydrostaticContract(
                role="BUOYANCY_ONLY",
                force_basis="GENERALIZED_DOF",
                stiffness_derivative="INCREMENTAL_ABOUT_EQUILIBRIUM",
                bodies=(alpha, beta),
                baseline_wrench=None,
                provenance={"source": "persistence test"},
                full_source=None,
            ),
        )
        path = tmp_path / "extended.nc"
        save_hydro_dataset(contracted, path)
        loaded = load_hydro_dataset(path)
        # decoded dict forms are valid metadata
        assert isinstance(loaded.attrs["coefficient_frames"], dict)
        assert isinstance(loaded.attrs["hydrostatic_contract"], dict)
        assert dataset_content_hash(loaded) == dataset_content_hash(contracted)
        assert np.allclose(
            coefficient_frame_axes(loaded)[alpha],
            coefficient_frame_axes(contracted)[alpha],
        )
        assert read_hydrostatic_contract(loaded).role == "BUOYANCY_ONLY"
