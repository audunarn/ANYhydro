"""Canonical schema validation tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from anyhydro import (
    SchemaError,
    TimeDomainAdmissionError,
    validate_for_time_domain,
    validate_hydro_dataset,
)


def _tamper(ds, **changes):
    out = ds.copy(deep=True)
    for key, value in changes.items():
        out.attrs[key] = value
    return out


class TestValidDataset:
    def test_fixture_validates(self, two_body_dataset):
        validate_hydro_dataset(two_body_dataset)

    def test_time_domain_admission_with_a_inf(self, two_body_dataset):
        validate_for_time_domain(two_body_dataset)

    def test_time_domain_rejects_missing_a_inf(self, two_body_dataset_no_a_inf):
        with pytest.raises(TimeDomainAdmissionError):
            validate_for_time_domain(two_body_dataset_no_a_inf)

    def test_a_inf_source_provenance_required(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        ds["added_mass_infinite"].attrs.clear()
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)


class TestRejections:
    def test_rejects_swapped_dims(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        data = ds["added_mass"].values
        ds = ds.drop_vars("added_mass")
        ds["added_mass"] = (("omega", "radiating_dof", "influenced_dof"), data)
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_rejects_duplicate_dof_coords(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        dofs = [str(d) for d in ds["influenced_dof"].values.tolist()]
        dofs[1] = dofs[0]
        ds = ds.assign_coords(influenced_dof=np.array(dofs, dtype=object))
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_rejects_incomplete_dofs(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        keep = [str(d) for d in ds["influenced_dof"].values.tolist()][:-1]
        drop = [str(d) for d in ds["influenced_dof"].values.tolist()][-1]
        ds = ds.sel(influenced_dof=keep)
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_rejects_nonfinite_data(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        values = ds["added_mass"].values.copy()
        values[0, 0, 0] = np.nan
        ds["added_mass"].values = values
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_rejects_nonfinite_complex_data(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        values = ds["excitation_force"].values.copy()
        values[0, 0, 0] = complex(np.inf, 0.0)
        ds["excitation_force"].values = values
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_rejects_missing_mandatory_variable(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        ds = ds.drop_vars("hydrostatic_stiffness")
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_rejects_missing_attrs(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        del ds.attrs["request_hash"]
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_rejects_wrong_convention_attr(self, two_body_dataset):
        ds = _tamper(two_body_dataset, phasor_convention="exp(+i*omega*t)")
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_rejects_unknown_variable_dry_mass(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        ds["dry_mass"] = (("body",), np.array([1.0, 2.0]))
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_rejects_zero_omega_coordinate(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        omegas = ds["omega"].values.copy()
        omegas[0] = 0.0
        ds = ds.assign_coords(omega=omegas)
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_rejects_unsorted_omega_coordinate(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        omegas = ds["omega"].values.copy()
        omegas = omegas[::-1]
        ds = ds.assign_coords(omega=omegas)
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_rejects_mesh_hashes_not_covering_bodies(self, two_body_dataset):
        import json

        ds = two_body_dataset.copy(deep=True)
        hashes = json.loads(ds.attrs["mesh_hashes"])
        hashes.pop(next(iter(hashes)))
        ds.attrs["mesh_hashes"] = json.dumps(hashes)
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_rejects_bad_qualification_status(self, two_body_dataset):
        ds = _tamper(two_body_dataset, qualification_status="definitely-qualified")
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_rejects_non_dataset_input(self):
        with pytest.raises(SchemaError):
            validate_hydro_dataset({"not": "a dataset"})


class TestOptionalMetadataExtensions:
    """Focused additions: coefficient_frames / hydrostatic_contract attrs."""

    def test_accepts_valid_frames_metadata(self, two_body_dataset):
        import json

        import numpy as np

        from anyhydro import reframe_dataset

        ds = reframe_dataset(
            two_body_dataset,
            new_axes={
                "fixture-body-alpha": np.eye(3),
                "fixture-body-beta": np.eye(3),
            },
        )
        validate_hydro_dataset(ds)
        assert isinstance(json.loads(ds.attrs["coefficient_frames"]), dict)

    def test_rejects_malformed_frames_metadata(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        ds.attrs["coefficient_frames"] = json.dumps({"only-one-body": {}})
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_rejects_malformed_contract_metadata(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        ds.attrs["hydrostatic_contract"] = json.dumps({"role": "NOT_A_ROLE"})
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_legacy_absence_is_global_convention(self, two_body_dataset):
        ds = two_body_dataset
        assert "coefficient_frames" not in ds.attrs
        assert "hydrostatic_contract" not in ds.attrs
        validate_hydro_dataset(ds)
