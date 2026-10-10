"""Hydrostatic contract transport tests.

These test transport validation and typed refusals only. A declared-complete
contract is NOT scientific qualification and NOT integrator admission; no
composition is performed anywhere in this module.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from anyhydro import (
    FORCE_BASES,
    HYDROSTATIC_ROLES,
    SOURCE_BINDING_VERSION,
    HydrostaticContract,
    HydrostaticContractError,
    RecordValidationError,
    SchemaError,
    TransformError,
    attach_hydrostatic_contract,
    contract_source_binding,
    dataset_content_hash,
    hydrostatic_composition_eligibility,
    hydrostatic_contract_binding_matches,
    load_hydro_dataset,
    read_hydrostatic_contract,
    reframe_dataset,
    require_hydrostatic_composition,
    save_hydro_dataset,
    validate_hydro_dataset,
)
from anyhydro.fixtures import FIXTURE_BODY_IDS

ALPHA, BETA = FIXTURE_BODY_IDS[0], FIXTURE_BODY_IDS[1]


def _contract(**overrides):
    base = dict(
        role="BUOYANCY_ONLY",
        force_basis="GENERALIZED_DOF",
        stiffness_derivative="INCREMENTAL_ABOUT_EQUILIBRIUM",
        bodies=(ALPHA, BETA),
        baseline_wrench=None,
        provenance={"source": "development test declaration"},
        full_source=None,
    )
    base.update(overrides)
    return HydrostaticContract(**base)


def _measured_wrench():
    return {
        ALPHA: (100.0, -20.0, 5.0, 3.0, -2.0, 1.0),
        BETA: (-50.0, 10.0, 8.0, 1.0, 2.0, -3.0),
    }


def _measured_contract(**overrides):
    overrides.setdefault("baseline_wrench", _measured_wrench())
    return _contract(**overrides)


class TestRecordValidation:
    def test_valid_buoyancy_only(self):
        c = _contract()
        assert c.role == "BUOYANCY_ONLY"

    def test_missing_role_vocabulary_rejected(self):
        with pytest.raises(RecordValidationError):
            _contract(role="ZERO_STIFFNESS")

    def test_missing_force_basis_rejected(self):
        with pytest.raises(RecordValidationError):
            _contract(force_basis="ASSUMED_GENERALIZED")

    def test_missing_derivative_rejected(self):
        with pytest.raises(RecordValidationError):
            _contract(stiffness_derivative="GUESSED")

    def test_full_requires_source(self):
        with pytest.raises(RecordValidationError, match="FULL requires"):
            _contract(role="FULL")

    def test_full_with_source_ok(self):
        c = _contract(
            role="FULL",
            full_source={
                "mass": 1.0e5,
                "center_of_gravity": [0.0, 0.0, -0.1],
                "composition": "declared source decomposition (development)",
            },
        )
        assert c.full_source["mass"] == 1.0e5

    def test_full_source_only_for_full(self):
        with pytest.raises(RecordValidationError, match="only permitted"):
            _contract(
                full_source={
                    "mass": 1.0,
                    "center_of_gravity": [0.0, 0.0, 0.0],
                    "composition": "x",
                }
            )

    def test_full_source_shape(self):
        for bad in (
            {"mass": -1.0, "center_of_gravity": [0, 0, 0], "composition": "x"},
            {"mass": 1.0, "center_of_gravity": [0, 0], "composition": "x"},
            {"mass": 1.0, "center_of_gravity": [0, 0, 0], "composition": " "},
            {"mass": 1.0, "center_of_gravity": [0, 0, 0]},
        ):
            with pytest.raises(RecordValidationError):
                _contract(role="FULL", full_source=bad)

    def test_baseline_wrench_validation(self):
        good = {ALPHA: (1.0, 2, 3, 4, 5, 6), BETA: (0, 0, 0, 0, 0, 0)}
        assert _contract(baseline_wrench=good).baseline_wrench[ALPHA] == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        with pytest.raises(RecordValidationError):
            _contract(baseline_wrench={ALPHA: (1.0, 2, 3, 4, 5, np.nan), BETA: (0,)*6})
        with pytest.raises(RecordValidationError):
            _contract(baseline_wrench={ALPHA: (1.0, 2, 3, 4, 5, 6)})

    def test_provenance_source_required(self):
        with pytest.raises(RecordValidationError):
            _contract(provenance={"note": "no source identity"})
        with pytest.raises(RecordValidationError):
            _contract(provenance={"source": " "})

    def test_duplicate_bodies_rejected(self):
        with pytest.raises(RecordValidationError):
            _contract(bodies=(ALPHA, ALPHA))

    def test_immutable(self):
        c = _contract()
        with pytest.raises(Exception):
            c.role = "FULL"


class TestAttachRead:
    def test_missing_contract_is_unknown(self, two_body_dataset):
        c = read_hydrostatic_contract(two_body_dataset)
        assert c.role == "UNKNOWN"
        assert c.force_basis == "UNKNOWN"
        assert c.stiffness_derivative == "UNKNOWN"
        assert c.baseline_wrench is None
        # UNKNOWN is not zero stiffness and not a gravity-free assumption
        assert "UNKNOWN" in hydrostatic_composition_eligibility(two_body_dataset).reasons[0]

    def test_attach_read_roundtrip(self, two_body_dataset):
        contract = _contract()
        attached = attach_hydrostatic_contract(two_body_dataset, contract)
        assert "hydrostatic_contract" in attached.attrs
        assert dataset_content_hash(attached) != dataset_content_hash(two_body_dataset)
        read = read_hydrostatic_contract(attached)
        assert read.role == "BUOYANCY_ONLY"
        assert read.force_basis == "GENERALIZED_DOF"
        assert read.stiffness_derivative == "INCREMENTAL_ABOUT_EQUILIBRIUM"
        assert read.bodies == (ALPHA, BETA)
        validate_hydro_dataset(attached)

    def test_attach_does_not_mutate_input(self, two_body_dataset):
        before = dataset_content_hash(two_body_dataset)
        attach_hydrostatic_contract(two_body_dataset, _contract())
        assert dataset_content_hash(two_body_dataset) == before

    def test_attach_body_mismatch(self, two_body_dataset):
        with pytest.raises(HydrostaticContractError, match="bodies"):
            attach_hydrostatic_contract(two_body_dataset, _contract(bodies=("other",)))

    def test_attach_rejects_non_record(self, two_body_dataset):
        with pytest.raises(HydrostaticContractError):
            attach_hydrostatic_contract(two_body_dataset, {"role": "FULL"})


class TestCompositionGate:
    def test_missing_contract_refuses(self, two_body_dataset):
        with pytest.raises(HydrostaticContractError, match="UNKNOWN"):
            require_hydrostatic_composition(two_body_dataset)

    def test_unknown_semantics_refuse(self, two_body_dataset):
        ds = attach_hydrostatic_contract(
            two_body_dataset,
            _contract(force_basis="UNKNOWN", stiffness_derivative="UNKNOWN"),
        )
        with pytest.raises(HydrostaticContractError):
            require_hydrostatic_composition(ds)

    def test_partial_semantics_refuse(self, two_body_dataset):
        ds = attach_hydrostatic_contract(
            two_body_dataset, _contract(stiffness_derivative="UNKNOWN")
        )
        eligibility = hydrostatic_composition_eligibility(ds)
        assert not eligibility.admissible
        assert any("derivative" in r for r in eligibility.reasons)
        with pytest.raises(HydrostaticContractError):
            require_hydrostatic_composition(ds)

    def test_complete_bound_generalized_dof_admissible(self, two_body_dataset):
        ds = attach_hydrostatic_contract(two_body_dataset, _measured_contract())
        eligibility = require_hydrostatic_composition(ds)
        assert eligibility.admissible
        assert eligibility.role == "BUOYANCY_ONLY"
        assert eligibility.reasons == ()

    def test_missing_baseline_wrench_refuses(self, two_body_dataset):
        # defect repair: absent baseline is UNMEASURED, not zero; a complete
        # role/basis/derivative declaration without a measured baseline is
        # not admissible.
        ds = attach_hydrostatic_contract(two_body_dataset, _contract())
        eligibility = hydrostatic_composition_eligibility(ds)
        assert not eligibility.admissible
        assert any("UNMEASURED" in r for r in eligibility.reasons)
        with pytest.raises(HydrostaticContractError, match="UNMEASURED"):
            require_hydrostatic_composition(ds)

    def test_declared_zero_baseline_is_valid_metadata(self, two_body_dataset):
        # a source-declared zero wrench is actual measured metadata and is
        # admissible; zero is never inferred from absence.
        ds = attach_hydrostatic_contract(
            two_body_dataset,
            _measured_contract(baseline_wrench={ALPHA: (0.0,) * 6, BETA: (0.0,) * 6}),
        )
        assert require_hydrostatic_composition(ds).admissible

    def test_physical_wrench_basis_refuses(self, two_body_dataset):
        # defect repair: PHYSICAL_WRENCH needs the source rate/coordinate
        # convention map and implemented conversion; none is admitted here.
        ds = attach_hydrostatic_contract(
            two_body_dataset, _measured_contract(force_basis="PHYSICAL_WRENCH")
        )
        eligibility = hydrostatic_composition_eligibility(ds)
        assert not eligibility.admissible
        assert any("PHYSICAL_WRENCH" in r for r in eligibility.reasons)
        with pytest.raises(HydrostaticContractError, match="PHYSICAL_WRENCH"):
            require_hydrostatic_composition(ds)

    def test_full_role_refuses_with_unimplemented_policy_reason(
        self, two_body_dataset
    ):
        # defect repair: FULL declarations are transported metadata only; the
        # decomposition/reference-only-gravity policy is not implemented in
        # this slice, so FULL never admits composition.
        ds = attach_hydrostatic_contract(
            two_body_dataset,
            _measured_contract(
                role="FULL",
                full_source={
                    "mass": {ALPHA: 2.0e5, BETA: 1.0e5},
                    "center_of_gravity": {ALPHA: [0.1, 0.0, -0.2], BETA: [0.0, 0.0, -0.1]},
                    "composition": "declared source decomposition (development)",
                },
            ),
        )
        eligibility = hydrostatic_composition_eligibility(ds)
        assert not eligibility.admissible
        assert any(
            "transported as declared metadata" in r and "not implemented" in r
            for r in eligibility.reasons
        )
        with pytest.raises(HydrostaticContractError, match="FULL"):
            require_hydrostatic_composition(ds)

    def test_full_legacy_single_value_shape_transports_but_refuses(
        self, two_body_dataset
    ):
        # legacy single-value FULL shape stays readable as declared
        # incomplete metadata on a two-body contract; it never admits.
        ds = attach_hydrostatic_contract(
            two_body_dataset,
            _measured_contract(
                role="FULL",
                full_source={
                    "mass": 1234.0,
                    "center_of_gravity": [0.0, 0.0, 0.0],
                    "composition": "gravity included",
                },
            ),
        )
        with pytest.raises(HydrostaticContractError):
            require_hydrostatic_composition(ds)

    def test_role_vocabulary_public(self):
        assert HYDROSTATIC_ROLES == ("UNKNOWN", "BUOYANCY_ONLY", "FULL")
        assert "PHYSICAL_WRENCH" in FORCE_BASES


class TestSchemaAndSerialization:
    def test_schema_rejects_malformed_contract(self, two_body_dataset):
        attached = attach_hydrostatic_contract(two_body_dataset, _contract())
        bad_payloads = []
        payload = json.loads(attached.attrs["hydrostatic_contract"])
        p = dict(payload); p["role"] = "ZERO"; bad_payloads.append(json.dumps(p))
        p = dict(payload); del p["provenance"]; bad_payloads.append(json.dumps(p))
        p = dict(payload); p["bodies"] = ["someone-else"]; bad_payloads.append(json.dumps(p))
        bad_payloads.append("{not json")
        for raw in bad_payloads:
            ds = attached.copy(deep=True)
            ds.attrs["hydrostatic_contract"] = raw
            with pytest.raises(SchemaError):
                validate_hydro_dataset(ds)

    def test_hash_detects_contract_tamper(self, two_body_dataset):
        attached = attach_hydrostatic_contract(two_body_dataset, _contract())
        tampered = attached.copy(deep=True)
        payload = json.loads(tampered.attrs["hydrostatic_contract"])
        payload["role"] = "FULL"
        tampered.attrs["hydrostatic_contract"] = json.dumps(payload, sort_keys=True)
        assert dataset_content_hash(tampered) != dataset_content_hash(attached)

    def test_persistence_roundtrip(self, two_body_dataset, tmp_path):
        attached = attach_hydrostatic_contract(two_body_dataset, _contract())
        path = tmp_path / "contracted.nc"
        save_hydro_dataset(attached, path)
        loaded = load_hydro_dataset(path)
        assert dataset_content_hash(loaded) == dataset_content_hash(attached)
        read = read_hydrostatic_contract(loaded)
        assert read.role == "BUOYANCY_ONLY"
        assert read.provenance["source"] == "development test declaration"


class TestFrameInteraction:
    def _wrench_contract(self, force_basis):
        return _contract(
            baseline_wrench={
                ALPHA: (100.0, -20.0, 5.0, 3.0, -2.0, 1.0),
                BETA: (-50.0, 10.0, 8.0, 1.0, 2.0, -3.0),
            },
            force_basis=force_basis,
        )

    def test_reframe_transports_generalized_wrench(self, two_body_dataset):
        from anyhydro import CoefficientFrame, frame_change_matrix

        ds = attach_hydrostatic_contract(two_body_dataset, self._wrench_contract("GENERALIZED_DOF"))
        old_frames = {
            ALPHA: CoefficientFrame(np.eye(3), (0.0, 0.0, 0.0)),
            BETA: CoefficientFrame(np.eye(3), (12.0, 0.0, -0.5)),
        }
        new_axes = {
            ALPHA: np.array(
                [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            BETA: np.eye(3),
        }
        new_centers = {ALPHA: (0.5, 0.0, 0.0), BETA: (12.0, 0.0, -0.5)}
        reframed = reframe_dataset(
            ds, new_axes=new_axes, new_reference_centers=new_centers
        )
        transported = read_hydrostatic_contract(reframed).baseline_wrench
        for body in (ALPHA, BETA):
            T = frame_change_matrix(
                old_frames[body],
                CoefficientFrame(new_axes[body], new_centers[body]),
            )
            expected = T.T @ np.asarray(
                self._wrench_contract("GENERALIZED_DOF").baseline_wrench[body]
            )
            assert np.allclose(transported[body], expected, rtol=1e-12, atol=0.0), body

    def test_reframe_refuses_physical_wrench_basis(self, two_body_dataset):
        ds = attach_hydrostatic_contract(
            two_body_dataset, self._wrench_contract("PHYSICAL_WRENCH")
        )
        before = dataset_content_hash(ds)
        with pytest.raises(TransformError, match="PHYSICAL_WRENCH"):
            reframe_dataset(
                ds,
                new_axes={ALPHA: np.eye(3), BETA: np.eye(3)},
                new_reference_centers={ALPHA: (1.0, 0.0, 0.0), BETA: (12.0, 0.0, -0.5)},
            )
        assert dataset_content_hash(ds) == before

    def test_reframe_refuses_unknown_basis_wrench(self, two_body_dataset):
        ds = attach_hydrostatic_contract(
            two_body_dataset, self._wrench_contract("UNKNOWN")
        )
        with pytest.raises(TransformError, match="UNKNOWN"):
            reframe_dataset(
                ds,
                new_axes={ALPHA: np.eye(3), BETA: np.eye(3)},
                new_reference_centers={ALPHA: (1.0, 0.0, 0.0), BETA: (12.0, 0.0, -0.5)},
            )

    def test_reframe_ok_without_wrench(self, two_body_dataset):
        ds = attach_hydrostatic_contract(two_body_dataset, _contract())
        reframed = reframe_dataset(
            ds,
            new_axes={ALPHA: np.eye(3), BETA: np.eye(3)},
            new_reference_centers={ALPHA: (1.0, 0.0, 0.0), BETA: (12.0, 0.0, -0.5)},
        )
        assert read_hydrostatic_contract(reframed).role == "BUOYANCY_ONLY"


class TestSourceBinding:
    """Defect 2 repair: versioned immutable binding to the exact dataset."""

    def test_attach_binds_to_exact_dataset_content(self, two_body_dataset):
        attached = attach_hydrostatic_contract(
            two_body_dataset, _measured_contract()
        )
        contract = read_hydrostatic_contract(attached)
        assert contract.source_binding is not None
        assert contract.source_binding["binding_version"] == SOURCE_BINDING_VERSION
        assert contract.source_binding == contract_source_binding(two_body_dataset)
        assert hydrostatic_contract_binding_matches(attached)

    def test_binding_hash_excludes_only_the_contract_payload(
        self, two_body_dataset
    ):
        # the noncircular identity excludes ONLY the contract/binding payload:
        # attaching a contract must not change the binding hash.
        assert contract_source_binding(
            attach_hydrostatic_contract(two_body_dataset, _measured_contract())
        ) == contract_source_binding(two_body_dataset)

    def test_stale_geometry_detected_and_refused(self, two_body_dataset):
        bound = attach_hydrostatic_contract(two_body_dataset, _measured_contract())
        foreign = bound.copy(deep=True)
        foreign["center_of_buoyancy"].values[:, 2] -= 0.25
        # transport stays possible; detection is explicit...
        assert not hydrostatic_contract_binding_matches(foreign)
        eligibility = hydrostatic_composition_eligibility(foreign)
        assert not eligibility.admissible
        assert any("stale/foreign" in r for r in eligibility.reasons)
        # ...and admission refuses the silently-following contract.
        with pytest.raises(HydrostaticContractError, match="stale/foreign"):
            require_hydrostatic_composition(foreign)

    def test_stale_coefficients_detected(self, two_body_dataset):
        bound = attach_hydrostatic_contract(two_body_dataset, _measured_contract())
        foreign = bound.copy(deep=True)
        foreign["hydrostatic_stiffness"].values[0, 0] += 1.0
        with pytest.raises(HydrostaticContractError):
            require_hydrostatic_composition(foreign)

    def test_unbound_contract_refuses_admission(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        ds.attrs["hydrostatic_contract"] = json.dumps(
            _measured_contract().to_json(), sort_keys=True
        )
        validate_hydro_dataset(ds)  # shape-valid transport stays possible
        eligibility = hydrostatic_composition_eligibility(ds)
        assert not eligibility.admissible
        assert any("not bound" in r for r in eligibility.reasons)
        with pytest.raises(HydrostaticContractError, match="not bound"):
            require_hydrostatic_composition(ds)

    def test_attach_refuses_foreign_bound_record(self, two_body_dataset):
        bound = attach_hydrostatic_contract(two_body_dataset, _measured_contract())
        foreign_ds = two_body_dataset.copy(deep=True)
        foreign_ds["center_of_buoyancy"].values[:, 2] -= 0.25
        bound_contract = read_hydrostatic_contract(bound)
        with pytest.raises(HydrostaticContractError, match="rebind=True"):
            attach_hydrostatic_contract(foreign_ds, bound_contract)
        # explicit rebinding is the caller's declared decision
        rebound = attach_hydrostatic_contract(
            foreign_ds, bound_contract, rebind=True
        )
        assert hydrostatic_contract_binding_matches(rebound)

    def test_attach_same_content_is_idempotent(self, two_body_dataset):
        attached = attach_hydrostatic_contract(two_body_dataset, _measured_contract())
        again = attach_hydrostatic_contract(
            attached, read_hydrostatic_contract(attached)
        )
        assert dataset_content_hash(again) == dataset_content_hash(attached)

    def test_malformed_binding_is_typed_schema_error(self, two_body_dataset):
        attached = attach_hydrostatic_contract(two_body_dataset, _measured_contract())
        for bad in (
            {"binding_version": SOURCE_BINDING_VERSION},
            {
                "binding_version": "other-v1",
                "dataset_content_hash": "0" * 64,
            },
            {
                "binding_version": SOURCE_BINDING_VERSION,
                "dataset_content_hash": "not-a-hash",
            },
            [SOURCE_BINDING_VERSION, "0" * 64],
        ):
            payload = json.loads(attached.attrs["hydrostatic_contract"])
            payload["source_binding"] = bad
            ds = attached.copy(deep=True)
            ds.attrs["hydrostatic_contract"] = json.dumps(payload, sort_keys=True)
            with pytest.raises(SchemaError):
                validate_hydro_dataset(ds)

    def test_binding_roundtrips_through_persistence(self, two_body_dataset, tmp_path):
        attached = attach_hydrostatic_contract(two_body_dataset, _measured_contract())
        path = tmp_path / "bound.nc"
        save_hydro_dataset(attached, path)
        loaded = load_hydro_dataset(path)
        assert hydrostatic_contract_binding_matches(loaded)
        assert require_hydrostatic_composition(loaded).admissible


class TestFullSourcePerBodyTransport:
    """Defect 4 repair: per-body FULL transport, no admission."""

    def test_per_body_full_source_transports(self):
        c = _contract(
            role="FULL",
            full_source={
                "mass": {ALPHA: 2.0e5, BETA: 1.0e5},
                "center_of_gravity": {ALPHA: [0.0, 0.0, -0.1], BETA: [0.5, 0.0, 0.0]},
                "composition": "declared per-body decomposition (development)",
            },
        )
        assert c.full_source["mass"][ALPHA] == 2.0e5
        assert c.full_source["center_of_gravity"][BETA] == (0.5, 0.0, 0.0)
        roundtrip = HydrostaticContract.from_json(
            json.loads(json.dumps(c.to_json()))
        )
        assert roundtrip.full_source == c.full_source

    def test_per_body_coverage_must_match_contract_bodies(self):
        with pytest.raises(RecordValidationError, match="cover exactly"):
            _contract(
                role="FULL",
                full_source={
                    "mass": {ALPHA: 1.0},
                    "center_of_gravity": {ALPHA: [0.0, 0.0, 0.0], BETA: [0.0, 0.0, 0.0]},
                    "composition": "x",
                },
            )

    def test_mixed_mass_and_cg_shapes_refused(self):
        with pytest.raises(RecordValidationError, match="same"):
            _contract(
                role="FULL",
                full_source={
                    "mass": {ALPHA: 1.0, BETA: 1.0},
                    "center_of_gravity": [0.0, 0.0, 0.0],
                    "composition": "x",
                },
            )

    def test_per_body_values_validated(self):
        for bad in (
            {"mass": {ALPHA: -1.0, BETA: 1.0}, "center_of_gravity": {ALPHA: [0, 0, 0], BETA: [0, 0, 0]}, "composition": "x"},
            {"mass": {ALPHA: 1.0, BETA: 1.0}, "center_of_gravity": {ALPHA: [0, 0], BETA: [0, 0, 0]}, "composition": "x"},
            {"mass": {ALPHA: 1.0, BETA: 1.0}, "center_of_gravity": {ALPHA: [0, 0, 0], BETA: [0, 0, np.nan]}, "composition": "x"},
        ):
            with pytest.raises(RecordValidationError):
                _contract(role="FULL", full_source=bad)


class TestFromJsonHardening:
    """Defect 5 repair: malformed containers are typed, never AttributeError."""

    def _payload(self):
        return _measured_contract().to_json()

    def test_baseline_wrench_list_is_typed(self):
        data = self._payload()
        data["baseline_wrench"] = [0.0, 0.0, 0.0]
        with pytest.raises(RecordValidationError, match="baseline_wrench"):
            HydrostaticContract.from_json(data)

    def test_baseline_wrench_scalar_entry_is_typed(self):
        data = self._payload()
        data["baseline_wrench"] = {ALPHA: 3.0, BETA: [0.0] * 6}
        with pytest.raises(RecordValidationError):
            HydrostaticContract.from_json(data)

    def test_baseline_wrench_non_numeric_entry_is_typed(self):
        data = self._payload()
        data["baseline_wrench"] = {ALPHA: ["a"] * 6, BETA: [0.0] * 6}
        with pytest.raises(RecordValidationError):
            HydrostaticContract.from_json(data)

    def test_bodies_not_a_list_is_typed(self):
        data = self._payload()
        data["bodies"] = "not-a-list"
        with pytest.raises(RecordValidationError, match="bodies"):
            HydrostaticContract.from_json(data)

    def test_provenance_not_an_object_is_typed(self):
        data = self._payload()
        data["provenance"] = [1, 2, 3]
        with pytest.raises(RecordValidationError, match="provenance"):
            HydrostaticContract.from_json(data)

    def test_full_source_not_an_object_is_typed(self):
        data = self._payload()
        data["full_source"] = [1, 2, 3]
        with pytest.raises(RecordValidationError, match="full_source"):
            HydrostaticContract.from_json(data)

    def test_unknown_and_missing_keys_are_typed(self):
        data = self._payload()
        data["extra"] = 1
        with pytest.raises(RecordValidationError, match="unknown"):
            HydrostaticContract.from_json(data)
        data = self._payload()
        del data["provenance"]
        with pytest.raises(RecordValidationError, match="missing"):
            HydrostaticContract.from_json(data)

    def test_schema_loader_refuses_malformed_baseline(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        data = self._payload()
        data["baseline_wrench"] = [0.0, 0.0, 0.0]
        ds.attrs["hydrostatic_contract"] = json.dumps(data, sort_keys=True)
        with pytest.raises(SchemaError):
            validate_hydro_dataset(ds)

    def test_persistence_loader_refuses_malformed_baseline(
        self, two_body_dataset, tmp_path
    ):
        attached = attach_hydrostatic_contract(two_body_dataset, _measured_contract())
        path = tmp_path / "malformed.nc"
        save_hydro_dataset(attached, path)
        import xarray as xr

        raw = xr.load_dataset(path, engine="scipy")
        payload = json.loads(raw.attrs["hydrostatic_contract"])
        payload["baseline_wrench"] = [0.0, 0.0, 0.0]
        raw.attrs["hydrostatic_contract"] = json.dumps(payload, sort_keys=True)
        raw.to_netcdf(path, engine="scipy")
        with pytest.raises(SchemaError):
            load_hydro_dataset(path)

    def test_legacy_seven_key_json_still_loads(self):
        data = self._payload()
        assert "source_binding" in data
        legacy = {k: v for k, v in data.items() if k != "source_binding"}
        contract = HydrostaticContract.from_json(legacy)
        assert contract.source_binding is None


class TestReframeRebindsBinding:
    """Connected reframe integration: transport then explicit rebind."""

    def _reframe(self, ds):
        return reframe_dataset(
            ds,
            new_axes={
                ALPHA: np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
                BETA: np.eye(3),
            },
            new_reference_centers={ALPHA: (0.5, 0.0, 0.0), BETA: (12.0, 0.0, -0.5)},
        )

    def test_reframe_rebinds_and_preserves_provenance(self, two_body_dataset):
        ds = attach_hydrostatic_contract(two_body_dataset, _measured_contract())
        reframed = self._reframe(ds)
        assert hydrostatic_contract_binding_matches(reframed)
        contract = read_hydrostatic_contract(reframed)
        assert contract.provenance["source"] == "development test declaration"
        assert contract.role == "BUOYANCY_ONLY"
        # the reframed dataset is bound to its own exact new content
        assert contract.source_binding == contract_source_binding(reframed)
        assert contract.source_binding != contract_source_binding(ds)

    def test_reframe_of_stale_bound_contract_refuses(self, two_body_dataset):
        bound = attach_hydrostatic_contract(two_body_dataset, _measured_contract())
        foreign = bound.copy(deep=True)
        foreign["center_of_buoyancy"].values[:, 2] -= 0.25
        with pytest.raises(HydrostaticContractError, match="stale"):
            self._reframe(foreign)

    def test_reframed_bound_contract_stays_admissible(self, two_body_dataset):
        ds = attach_hydrostatic_contract(two_body_dataset, _measured_contract())
        reframed = self._reframe(ds)
        assert require_hydrostatic_composition(reframed).admissible
