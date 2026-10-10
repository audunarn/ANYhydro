"""First-slice source-quality mesh study tests (mocked/analytic ONLY).

Every backend in this file is an EXPLICITLY LABELLED mock-rehearsal
double assembling analytic arrays through the PUBLIC
``assemble_hydro_dataset`` API: no real Capytaine import, no BEM solve,
and nothing here is scientific evidence. The tests cover the immutable
closed-world case record, the normalized refinement metric arithmetic
and the complete public prepare/execute/load/adjudicate workflow
(identity/caps checks before backend work, one solve per level,
fail-closed refusals, full raw-source proof at reload, report
reproduction without fresh BEM).
"""

from __future__ import annotations

import json
import shutil
import sys

import numpy as np
import pytest

import anyhydro.source_study
from anyhydro.errors import HydroError
from anyhydro.first_slice import (
    FIRST_SLICE_HEAVE_BUOYANCY_STIFFNESS,
    FIRST_SLICE_PROJECTED_DOFS,
    make_first_slice_bodies,
)
from anyhydro.operators import DerivedOperatorCriteria, NormalizationSpec
from anyhydro.records import HydroRequest
from anyhydro.persistence import load_hydro_dataset
from anyhydro.schema import assemble_hydro_dataset
from anyhydro.source_study import (
    CLI_EXIT_CANCELLED,
    CLI_EXIT_ERROR,
    CLI_EXIT_FAILED,
    CLI_EXIT_INCOMPLETE,
    CLI_EXIT_NO_GO,
    CLI_EXIT_OK,
    OUTCOME_CANCELLED,
    OUTCOME_FAILED,
    OUTCOME_NO_GO,
    OUTCOME_PREPARED_NOT_EXECUTED,
    OUTCOME_SOURCE_ADMISSION_PASSED_DEVELOPMENT_ONLY,
    RECEIVED_ARCHIVE_FORMAT,
    SourceStudyIdentityError,
    SourceStudyWorkflowError,
    adjudicate_source_study,
    execute_source_study,
    load_source_study,
    main,
    prepare_source_study,
)
from anyhydro.source_study_case import (
    DEVELOPMENT_QUALIFICATION_SCOPE,
    FIRST_SLICE_GEOMETRY_RECIPE,
    SOURCE_STUDY_FORMAT,
    SOURCE_STUDY_QUANTITIES,
    QuantityRefinementCriteria,
    RefinementCriteria,
    SourceStudyCase,
    SourceStudyCaseError,
    build_study_bodies,
    build_study_request,
    compare_level_refinement,
    make_prospective_source_study_case,
    study_operator_settings,
)

#: Explicit mock label: any artifact produced by the doubles in this file
#: is development rehearsal only and never scientific evidence.
MOCK_BACKEND_LABEL = (
    "mock-rehearsal-capytaine-labelled development double "
    "(NOT scientific evidence)"
)


# --------------------------------------------------------------------------- #
# case helpers
# --------------------------------------------------------------------------- #


def _operator_criteria() -> DerivedOperatorCriteria:
    return DerivedOperatorCriteria(
        reciprocity_abs_tol=1e-8,
        reciprocity_rel_tol=0.02,
        reciprocity_floor=1e-8,
        correction_abs_tol=1e-8,
        correction_rel_tol=0.01,
        correction_floor=1e-8,
        derived_symmetry_tol=1e-10,
        damping_psd_tol=1e-10,
    )


def _refinement_criteria(mesh_rel_tol: float = 0.02) -> RefinementCriteria:
    quantities = {
        name: QuantityRefinementCriteria(
            abs_tol=1e-9,
            rel_tol=mesh_rel_tol,
            floor=1e-9,
            unit=f"normalized {name} unit (S-scaled)",
        )
        for name in SOURCE_STUDY_QUANTITIES
    }
    return RefinementCriteria(
        mesh_rel_tol=mesh_rel_tol, quantities=quantities)


def _normalization() -> NormalizationSpec:
    return NormalizationSpec(
        name="translation-length-1m",
        unit_convention=(
            "q = S q_hat with S = diag(1, 1, 1, 1) m: q_hat is "
            "DIMENSIONLESS and Q_hat = S^T Q carries N*m = J; A_hat = "
            "S A S and A_inf_hat = S A_inf S carry kg*m^2; B_hat = S B S "
            "carries kg*m^2/s; H_hat = S H S carries N*m = J; the stored "
            "excitation is the force for a ONE-METRE reference wave "
            "(F_1m_hat = S^T F_1m carries N*m = J; the per-metre-wave "
            "transfer has N)"),
        scales=(1.0, 1.0, 1.0, 1.0),
    )


#: Honest mock-rehearsal backend identity: analytic/mock cases,
#: capabilities and raw source attrs must all match this identity —
#: never advertise capytaine 3.0.0 while the raw source says mock.
MOCK_BACKEND_NAME = "mock-rehearsal"
MOCK_BACKEND_VERSION = "0.0-mock-rehearsal"


def _admission_declaration() -> str:
    return (
        "admission requires the actual raw backend/version attrs to "
        "match the case and the actual ordered center data variables "
        "to match the fixed fixture (atol 1e-12 m, rtol 0)")


def make_case(**overrides) -> SourceStudyCase:
    """Small fast case (levels 2 and 3) with explicit test overrides.

    The expected backend identity is the honest mock-rehearsal identity
    so the analytic doubles' capabilities AND raw source attrs match
    the case exactly (no capytaine 3.0.0 advertisement while the raw
    source says mock).
    """
    kwargs = dict(
        case_id="small-source-study",
        revision="small-test-v1",
        geometry_recipe=FIRST_SLICE_GEOMETRY_RECIPE,
        levels=(2, 3),
        omegas=(2.0, 4.0, 6.0),
        wave_directions=(0.0,),
        include_zero_frequency=False,
        include_infinite_frequency=True,
        expected_backend_name=MOCK_BACKEND_NAME,
        expected_backend_version=MOCK_BACKEND_VERSION,
        method=None,
        method_declaration="unset method selects the declared pinned default",
        admission_declaration=_admission_declaration(),
        n_jobs=1,
        n_threads=1,
        retained_dofs=FIRST_SLICE_PROJECTED_DOFS,
        normalization=_normalization(),
        correction_policy="reciprocal_average",
        operator_criteria=_operator_criteria(),
        refinement=_refinement_criteria(),
    )
    kwargs.update(overrides)
    return SourceStudyCase(**kwargs)


def _prospective_mock_case() -> SourceStudyCase:
    """The prospective case with the HONEST mock-rehearsal identity.

    Same controls/levels/criteria as the parent-owned prospective
    candidate, but the expected backend identity is mock-rehearsal so
    the analytic double's capabilities AND raw source attrs match the
    case honestly (the real prospective case keeps capytaine 3.0.0
    for the actual run authority the parent owns).
    """
    data = make_prospective_source_study_case().to_dict()
    data["expected_backend_name"] = MOCK_BACKEND_NAME
    data["expected_backend_version"] = MOCK_BACKEND_VERSION
    return SourceStudyCase.from_dict(data)


# --------------------------------------------------------------------------- #
# explicitly labelled mock-rehearsal backend (NOT scientific evidence)
# --------------------------------------------------------------------------- #


class _MockCapabilities:
    """Duck-typed capability record of the mock-rehearsal double.

    The default identity is the honest mock-rehearsal identity — the
    analytic/mock tests never advertise capytaine 3.0.0.
    """

    def __init__(self, backend_name: str = MOCK_BACKEND_NAME,
                 backend_version: str = MOCK_BACKEND_VERSION,
                 supports_multibody: bool = True,
                 supports_infinite_frequency: bool = True,
                 provides_added_mass_infinite: bool = True,
                 supports_zero_frequency: bool = False) -> None:
        self.backend_name = backend_name
        self.backend_version = backend_version
        self.supports_multibody = supports_multibody
        self.supports_infinite_frequency = supports_infinite_frequency
        self.provides_added_mass_infinite = provides_added_mass_infinite
        self.supports_zero_frequency = supports_zero_frequency


class MockExecutionLimits:
    """Duck-typed explicit execution limits (strict caps)."""

    def __init__(self, max_total_panels: int = 1_000_000,
                 max_finite_omegas: int = 513,
                 max_wave_directions: int = 513,
                 n_jobs: int = 1, n_threads: int = 1) -> None:
        self.max_total_panels = int(max_total_panels)
        self.max_finite_omegas = int(max_finite_omegas)
        self.max_wave_directions = int(max_wave_directions)
        self.n_jobs = int(n_jobs)
        self.n_threads = int(n_threads)


def _mock_arrays(n_dof: int, omegas: np.ndarray, n_dir: int, scale: float,
                 variant: str):
    """Analytic full-12-DOF symmetric PSD rehearsal arrays.

    Level-independent unless the ``level_bias`` variant is selected, so
    the cross-level refinement residuals are exactly zero on the good
    route. Off-body coupling blocks are deliberately nonzero.
    """
    n_omega = int(omegas.size)
    idx = np.arange(n_dof)
    a_base = np.diag(500.0 + 10.0 * idx)
    a_base[0, 6] = a_base[6, 0] = 60.0
    a_base[2, 8] = a_base[8, 2] = 40.0
    A = np.tile(a_base * scale, (n_omega, 1, 1))
    v = np.zeros(n_dof)
    v[0] = 1.0
    v[6] = 1.0
    b_base = np.diag(20.0 + idx) + 5.0 * np.outer(v, v)
    B = np.tile(b_base * scale, (n_omega, 1, 1))
    a_inf = 0.5 * a_base * scale
    H = np.zeros((n_dof, n_dof))
    H[2, 2] = H[8, 8] = FIRST_SLICE_HEAVE_BUOYANCY_STIFFNESS * scale
    F = np.zeros((n_omega, n_dir, n_dof), dtype=np.complex128)
    F[:, :, 0] = (100.0 + 20.0j) * omegas[:, None]
    F[:, :, 6] = (100.0 + 20.0j) * omegas[:, None]
    F[:, :, 2] = (200.0 + 50.0j) * omegas[:, None]
    F[:, :, 8] = (200.0 + 50.0j) * omegas[:, None]
    F = F * scale
    if variant == "psd_fail":
        B[:, 0, 0] = -5.0
    elif variant == "asym":
        # large enough to exceed the relative reciprocity threshold
        # (0.02 * ~610 normalized unit) of the prospective criteria
        A[:, 0, 2] += 100.0
    elif variant == "block_diagonal":
        # otherwise identical positive-PSD source with EVERY off-body
        # block exactly zero: the fixed two-body physical case requires
        # nonzero off-body radiation, so this must produce NoGo
        A[:, 0, 6] = A[:, 6, 0] = 0.0
        A[:, 2, 8] = A[:, 8, 2] = 0.0
        B[:, 0, 6] = B[:, 6, 0] = 0.0
        B[:, 2, 8] = B[:, 8, 2] = 0.0
        a_inf[0, 6] = a_inf[6, 0] = 0.0
        a_inf[2, 8] = a_inf[8, 2] = 0.0
    return A, B, a_inf, H, F


class MockSourceBackend:
    """EXPLICITLY LABELLED MOCK-REHEARSAL backend (NOT scientific evidence).

    Assembles analytic arrays through the public
    ``anyhydro.assemble_hydro_dataset`` API. Variants select the failure
    mode under test; the ``grid_mismatch``/``foreign_mesh``/``qualified``
    variants act only on the second solve so the first level's artifacts
    are retained as prior evidence.
    """

    def __init__(self, variant: str = "good",
                 backend_name: str = MOCK_BACKEND_NAME,
                 backend_version: str = MOCK_BACKEND_VERSION,
                 dataset_backend: str = None,
                 dataset_backend_version: str = None,
                 cap_overrides: dict = None) -> None:
        self._variant = variant
        self._backend_name = backend_name
        self._backend_version = backend_version
        # raw dataset attrs default to the SAME identity as the
        # capabilities (honest mock-rehearsal); independent control
        # exists so mismatched-raw-attr refusals can be rehearsed
        self._dataset_backend = (backend_name if dataset_backend is None
                                 else dataset_backend)
        self._dataset_backend_version = (
            backend_version if dataset_backend_version is None
            else dataset_backend_version)
        self._cap_overrides = dict(cap_overrides or {})
        self.solve_calls: list = []

    def capabilities(self) -> _MockCapabilities:
        caps = _MockCapabilities(self._backend_name, self._backend_version)
        for key, value in self._cap_overrides.items():
            setattr(caps, key, value)
        return caps

    def solve(self, bodies, request):
        self.solve_calls.append(tuple(b.body_id for b in bodies))
        second = len(self.solve_calls) >= 2
        if self._variant == "fail_on_solve" and second:
            raise RuntimeError(
                "mock-rehearsal solve failure (explicitly labelled)")
        bodies_used = bodies
        request_used = request
        scale = 1.0
        qualification = "unqualified"
        n = int(round(np.sqrt(bodies[0].mesh.faces.shape[0] / 10.0)))
        if self._variant == "level_bias":
            scale = 1.0 + 0.05 * n
        elif self._variant == "grid_mismatch" and second:
            request_used = HydroRequest(
                omegas=np.array([2.0, 4.0], dtype=np.float64),
                wave_directions=np.array([0.0], dtype=np.float64),
                water_depth=150.0,
                rho=1025.0,
                gravity=9.81,
                include_zero_frequency=False,
                include_infinite_frequency=True,
                backend_settings={},
            )
        elif self._variant == "foreign_mesh" and second:
            bodies_used = make_first_slice_bodies(subdivisions=2)
        elif self._variant == "qualified" and second:
            qualification = "qualified"
        omegas = np.asarray(request_used.omegas, dtype=np.float64)
        n_dir = int(np.asarray(request_used.wave_directions).size)
        A, B, a_inf, H, F = _mock_arrays(
            6 * len(bodies_used), omegas, n_dir, scale, self._variant)
        dataset = assemble_hydro_dataset(
            bodies=bodies_used,
            request=request_used,
            added_mass=A,
            radiation_damping=B,
            excitation_force=F,
            hydrostatic_stiffness=H,
            rotation_center=np.array(
                [b.rotation_center for b in bodies_used], dtype=np.float64),
            center_of_mass=np.array(
                [b.center_of_mass for b in bodies_used], dtype=np.float64),
            center_of_buoyancy=np.array(
                [[b.rotation_center[0], 0.0, -0.25]
                 for b in bodies_used], dtype=np.float64),
            added_mass_infinite=a_inf,
            added_mass_infinite_source=(
                "mock-rehearsal analytic infinite-frequency source"),
            backend=self._dataset_backend,
            backend_version=self._dataset_backend_version,
            backend_commit=MOCK_BACKEND_LABEL,
            qualification_status=qualification,
        )
        if self._variant == "phasor_mismatch" and second:
            dataset.attrs["phasor_convention"] = (
                "reversed-exp-plus-iwt-mock")
        # center-array mutations with UNCHANGED metadata/request hash:
        # the body_identity attrs still carry the true fixture values,
        # so only the actual data-variable proof can catch these
        if second and self._variant in (
                "rotation_center_mutation", "center_of_mass_mutation",
                "center_of_buoyancy_mutation"):
            variable = self._variant[:-len("_mutation")]
            values = np.array(dataset[variable].values, dtype=np.float64)
            values[0, 0] += 0.1
            dataset[variable] = (("body", "xyz"), values)
        return dataset


def _prepare_and_execute(case, backend, limits, out_dir, **kwargs):
    prepare_source_study(case, out_dir)
    return execute_source_study(case, backend, limits, out_dir, **kwargs)


def _read_status(out_dir) -> dict:
    return json.loads((out_dir / "status.json").read_text(encoding="utf-8"))


def _read_report(out_dir) -> dict:
    return json.loads(
        (out_dir / "study_report.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# case record
# --------------------------------------------------------------------------- #


class TestCaseRecord:

    def test_prospective_case_exact_inventory(self):
        case = make_prospective_source_study_case()
        assert case.levels == (4, 6)
        assert case.level_panel_inventory() == {4: 384, 6: 864}
        assert case.max_total_panels() == 864
        assert case.omegas == (2.0, 4.0, 6.0)
        assert case.wave_directions == (0.0,)
        assert case.include_zero_frequency is False
        assert case.include_infinite_frequency is True
        assert case.expected_backend_name == "capytaine"
        assert case.expected_backend_version == "3.0.0"
        assert case.method is None
        # template revision v2: correct units/work conjugacy declared
        # BEFORE any actual run (none has run); thresholds unchanged
        assert case.revision == "prospective-candidate-2026-10-08-v2"
        convention = case.normalization.unit_convention
        assert "q_hat is DIMENSIONLESS" in convention
        assert "Q_hat = S^T Q carries N*m = J" in convention
        assert "kg*m^2" in convention
        assert "kg*m^2/s" in convention
        assert "H_hat = S H S carries N*m = J" in convention
        assert "ONE-METRE reference wave" in convention
        assert "GLOBAL-band linf relative control" in convention
        units = {name: qc.unit
                 for name, qc in case.refinement.quantities.items()}
        assert "kg*m^2" in units["added_mass"]
        assert "kg*m^2/s" in units["damping"]
        assert "kg*m^2" in units["added_mass_infinite"]
        assert "N*m = J" in units["hydrostatic_stiffness"]
        assert "N*m = J" in units["excitation_force"]
        assert "per-metre-wave transfer has N" in units["excitation_force"]
        # hash-covered admission declaration (v2): the fixed-fixture
        # center-array proof and the raw backend/version attr proof
        declaration = case.admission_declaration
        assert "backend/backend_version" in declaration
        assert "rotation_center/center_of_mass/center_of_buoyancy" \
            in declaration
        assert "(-2,0,0) m and (2,0,0) m" in declaration
        assert "(cx, 0, -0.25) m" in declaration
        assert "atol 1e-12 m rtol 0" in declaration
        assert "ADMISSION numerical tolerance, not a meshing tolerance" \
            in declaration
        assert case.n_jobs == 1
        assert case.n_threads == 1
        assert case.retained_dofs == FIRST_SLICE_PROJECTED_DOFS
        assert np.allclose(case.normalization.scales, 1.0)
        assert case.correction_policy == "reciprocal_average"
        assert case.qualification_scope == DEVELOPMENT_QUALIFICATION_SCOPE
        crit = case.operator_criteria
        assert crit.reciprocity_abs_tol == 1e-8
        assert crit.reciprocity_rel_tol == 0.02
        assert crit.reciprocity_floor == 1e-8
        assert crit.correction_abs_tol == 1e-8
        assert crit.correction_rel_tol == 0.01
        assert crit.correction_floor == 1e-8
        assert crit.derived_symmetry_tol == 1e-10
        assert crit.damping_psd_tol == 1e-10
        assert case.refinement.mesh_rel_tol == 0.02
        assert set(case.refinement.quantities) == set(SOURCE_STUDY_QUANTITIES)
        for qc in case.refinement.quantities.values():
            assert qc.abs_tol == 1e-9
            assert qc.rel_tol == 0.02
            assert qc.floor == 1e-9
            assert qc.unit

    def test_case_public_study_inputs(self):
        case = make_prospective_source_study_case()
        settings = study_operator_settings(case)
        assert settings.correction_policy == "reciprocal_average"
        assert tuple(settings.retained_dofs) == FIRST_SLICE_PROJECTED_DOFS
        request = build_study_request(case)
        assert np.array_equal(request.omegas, np.array([2.0, 4.0, 6.0]))
        assert np.array_equal(request.wave_directions, np.array([0.0]))
        assert request.include_zero_frequency is False
        assert request.include_infinite_frequency is True
        assert request.backend_settings == {}
        bodies = build_study_bodies(case, 4)
        assert sum(
            b.mesh.faces.shape[0] + b.lid_mesh.faces.shape[0]
            for b in bodies) == 384

    def test_build_study_bodies_refuses_unregistered_level(self):
        case = make_case()
        with pytest.raises(SourceStudyWorkflowError):
            build_study_bodies(case, 5)

    def test_case_roundtrip_and_hash_binding(self):
        case = make_case()
        payload = json.loads(json.dumps(case.to_dict()))
        restored = SourceStudyCase.from_dict(payload)
        assert restored.content_hash() == case.content_hash()
        # to_dict returns a fresh dict: mutating it cannot change the case
        snapshot = case.to_dict()
        snapshot["revision"] = "mutated-after-the-fact"
        assert case.to_dict()["revision"] == "small-test-v1"
        assert case.content_hash() == restored.content_hash()
        # frozen record
        with pytest.raises(Exception):
            case.levels = (2, 3, 4)  # type: ignore[misc]
        # changed controls change the bound hash
        other = make_case(revision="small-test-v2")
        assert other.content_hash() != case.content_hash()

    def test_case_closed_world_refusals(self):
        with pytest.raises(SourceStudyCaseError):
            make_case(unknown_field=1)
        for overrides in (
            dict(case_id="../evil"),
            dict(case_id="a/b"),
            dict(geometry_recipe="some-other-recipe"),
            dict(levels=(3, 2)),
            dict(levels=(2, 2)),
            dict(levels=(0, 2)),
            dict(levels=(17, 18)),
            dict(levels=(True, 2)),
            dict(levels=(2.0, 3)),
            dict(omegas=(0.0, 2.0)),
            dict(omegas=(4.0, 2.0)),
            dict(omegas=(2.0, float("nan"))),
            dict(omegas=()),
            dict(wave_directions=()),
            dict(include_zero_frequency=True),
            dict(include_infinite_frequency=False),
            dict(n_jobs=2),
            dict(n_threads=0),
            dict(retained_dofs=(
                "first-slice-a::heave", "first-slice-a::surge",
                "first-slice-b::surge", "first-slice-b::heave")),
            dict(correction_policy="clamp"),
            dict(normalization=NormalizationSpec(
                name="bad", unit_convention="bad",
                scales=(1.0, 1.0, 1.0))),
            dict(operator_criteria="not-criteria"),
        ):
            with pytest.raises(SourceStudyCaseError):
                make_case(**overrides)
        payload = make_case().to_dict()
        payload["extra"] = 1
        with pytest.raises(SourceStudyCaseError):
            SourceStudyCase.from_dict(payload)
        payload = make_case().to_dict()
        del payload["method_declaration"]
        with pytest.raises(SourceStudyCaseError):
            SourceStudyCase.from_dict(payload)
        payload = make_case().to_dict()
        del payload["admission_declaration"]
        with pytest.raises(SourceStudyCaseError):
            SourceStudyCase.from_dict(payload)
        payload = make_case().to_dict()
        payload["format"] = "some-other-format"
        with pytest.raises(SourceStudyCaseError):
            SourceStudyCase.from_dict(payload)

    def test_case_v1_format_explicitly_unsupported(self):
        # the admission_declaration extension changed the closed-world
        # case schema: v1 records are honestly refused by the v2
        # decoder — no automatic migration, no restamping
        assert SOURCE_STUDY_FORMAT == "anyhydro-source-study-v2"
        payload = make_case().to_dict()
        payload["format"] = "anyhydro-source-study-v1"
        with pytest.raises(SourceStudyCaseError,
                           match="explicitly unsupported|EXPLICITLY "
                                 "UNSUPPORTED"):
            SourceStudyCase.from_dict(payload)
        with pytest.raises(SourceStudyCaseError, match="v1"):
            SourceStudyCase.from_dict(payload)

    def test_case_content_hash_honest_v2_prefix(self):
        # the content-hash prefix is pinned at v2: the hash is sha256
        # over b"anyhydro-source-study-case-v2" + 8-byte little-endian
        # payload length + the canonical closed-world JSON
        import hashlib
        case = make_case()
        payload = json.dumps(case.to_dict(), sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
        h = hashlib.sha256()
        h.update(b"anyhydro-source-study-case-v2")
        h.update(len(payload.encode("utf-8")).to_bytes(8, "little"))
        h.update(payload.encode("utf-8"))
        assert case.content_hash() == h.hexdigest()
        # a v1-prefix hash is a DIFFERENT identity: no restamping
        h1 = hashlib.sha256()
        h1.update(b"anyhydro-source-study-case-v1")
        h1.update(len(payload.encode("utf-8")).to_bytes(8, "little"))
        h1.update(payload.encode("utf-8"))
        assert case.content_hash() != h1.hexdigest()
        payload = make_case().to_dict()
        payload["qualification_scope"] = "qualified"
        with pytest.raises(SourceStudyCaseError):
            SourceStudyCase.from_dict(payload)

    def test_refinement_criteria_validation(self):
        quantities = {
            name: QuantityRefinementCriteria(
                abs_tol=1e-9, rel_tol=0.02, floor=1e-9, unit="u")
            for name in SOURCE_STUDY_QUANTITIES
        }
        partial = dict(quantities)
        del partial["damping"]
        with pytest.raises(SourceStudyCaseError):
            RefinementCriteria(mesh_rel_tol=0.02, quantities=partial)
        loose = dict(quantities)
        loose["added_mass"] = QuantityRefinementCriteria(
            abs_tol=1e-9, rel_tol=0.05, floor=1e-9, unit="u")
        with pytest.raises(SourceStudyCaseError):
            RefinementCriteria(mesh_rel_tol=0.02, quantities=loose)
        for bad in (
            dict(abs_tol=-1e-9, rel_tol=0.02, floor=1e-9, unit="u"),
            dict(abs_tol=True, rel_tol=0.02, floor=1e-9, unit="u"),
            dict(abs_tol=1e-9, rel_tol=0.02, floor=0.0, unit="u"),
            dict(abs_tol=1e-9, rel_tol=0.02, floor=1e-9, unit=" "),
        ):
            with pytest.raises(SourceStudyCaseError):
                QuantityRefinementCriteria(**bad)
        with pytest.raises(SourceStudyCaseError):
            QuantityRefinementCriteria.from_dict(
                {"abs_tol": 1e-9, "rel_tol": 0.02, "floor": 1e-9,
                 "unit": "u", "extra": 1})
        criteria = _refinement_criteria()
        restored = RefinementCriteria.from_dict(json.loads(
            json.dumps(criteria.to_dict())))
        assert restored.to_dict() == criteria.to_dict()


# --------------------------------------------------------------------------- #
# refinement metric arithmetic
# --------------------------------------------------------------------------- #


def _metric_arrays() -> dict:
    return {
        "added_mass": np.zeros((1, 4, 4)),
        "damping": np.zeros((1, 4, 4)),
        "added_mass_infinite": np.zeros((4, 4)),
        "hydrostatic_stiffness": np.zeros((4, 4)),
        "excitation_force": np.zeros((1, 1, 4), dtype=np.complex128),
    }


def _scaled_arrays(value: float) -> dict:
    out = _metric_arrays()
    for name in ("added_mass", "damping"):
        out[name][0] = value * np.eye(4)
    out["added_mass_infinite"] = value * np.eye(4)
    out["hydrostatic_stiffness"] = value * np.eye(4)
    out["excitation_force"][:, :, 0] = complex(value, 0.0)
    return out


class TestRefinementMetric:

    def test_zero_residuals_pass(self):
        ref = _metric_arrays()
        cand = _metric_arrays()
        report = compare_level_refinement(
            ref, cand, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))
        assert report["mesh_pass"] is True
        for name in SOURCE_STUDY_QUANTITIES:
            assert report["quantities"][name]["pass"] is True
            assert report["quantities"][name]["residuals"] == [0.0]

    def test_nearzero_floor_pass_and_fail(self):
        ref = _metric_arrays()
        small = _metric_arrays()
        for arr in small.values():
            arr += 5e-10
        report = compare_level_refinement(
            ref, small, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))
        assert report["mesh_pass"] is True
        large = _metric_arrays()
        for arr in large.values():
            arr += 1e-7
        report = compare_level_refinement(
            ref, large, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))
        assert report["mesh_pass"] is False
        for name in SOURCE_STUDY_QUANTITIES:
            entry = report["quantities"][name]
            assert entry["pass"] is False
            # explicit near-zero floor in the quantity's own unit
            assert entry["threshold"] == pytest.approx(
                1e-9 + 0.02 * 1e-9, rel=1e-12)

    def test_relative_threshold_against_finest_scale(self):
        ref = _scaled_arrays(100.0)
        within = _scaled_arrays(101.0)
        report = compare_level_refinement(
            ref, within, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))
        assert report["mesh_pass"] is True
        outside = _scaled_arrays(103.0)
        report = compare_level_refinement(
            ref, outside, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))
        assert report["mesh_pass"] is False
        for name in SOURCE_STUDY_QUANTITIES:
            entry = report["quantities"][name]
            assert entry["reference_scale"] == pytest.approx(100.0)
            assert entry["threshold"] == pytest.approx(
                1e-9 + 0.02 * 100.0, rel=1e-9)

    def test_global_band_threshold_multi_frequency(self):
        # ADOPTED mesh norm rule: per-sample max-entry residual vs
        # abs + rel * max(finest quantity's GLOBAL registered-band
        # max-entry magnitude, floor) — a GLOBAL-band linf relative
        # control, NOT relative 2% of each local sample
        ref = _metric_arrays()
        ref["added_mass"] = np.zeros((3, 4, 4))
        ref["added_mass"][0] = 1000.0 * np.eye(4)
        ref["added_mass"][1] = 1e-12 * np.eye(4)
        ref["added_mass"][2] = 5.0 * np.eye(4)
        cand = {k: np.array(v, copy=True) for k, v in ref.items()}
        # 0.5 residual on the near-zero frequency: a local-sample
        # relative rule (2% of 1e-12) would refuse it; the GLOBAL-band
        # rule (2% of the 1000 global max) admits it
        cand["added_mass"][1] += 0.5 * np.eye(4)
        report = compare_level_refinement(
            ref, cand, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))
        entry = report["quantities"]["added_mass"]
        assert entry["pass"] is True
        assert entry["reference_scale"] == pytest.approx(1000.0)
        assert entry["threshold"] == pytest.approx(1e-9 + 0.02 * 1000.0)
        assert len(entry["residuals"]) == 3
        assert entry["residuals"][1] == pytest.approx(0.5)
        # a residual ABOVE the global-band threshold fails the quantity
        far = {k: np.array(v, copy=True) for k, v in ref.items()}
        far["added_mass"][2] += 30.0 * np.eye(4)
        report = compare_level_refinement(
            ref, far, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))
        assert report["quantities"]["added_mass"]["pass"] is False
        assert report["mesh_pass"] is False

    def test_nearzero_band_floor_governs_multi_frequency(self):
        # when the ENTIRE registered band of a quantity is near zero,
        # the explicit floor (never any local sample magnitude) governs
        # the threshold for every frequency
        ref = _metric_arrays()
        ref["damping"] = np.zeros((3, 4, 4))
        ref["damping"] += 4e-10 * np.eye(4)
        cand = {k: np.array(v, copy=True) for k, v in ref.items()}
        cand["damping"] += 3e-10 * np.eye(4)
        report = compare_level_refinement(
            ref, cand, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))
        entry = report["quantities"]["damping"]
        assert entry["pass"] is True
        assert entry["reference_scale"] == pytest.approx(4e-10)
        assert entry["threshold"] == pytest.approx(1e-9 + 0.02 * 1e-9)
        assert len(entry["residuals"]) == 3
        large = {k: np.array(v, copy=True) for k, v in ref.items()}
        large["damping"] += 2e-9 * np.eye(4)
        report = compare_level_refinement(
            ref, large, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))
        assert report["quantities"]["damping"]["pass"] is False

    def test_complex_excitation_residuals(self):
        ref = _metric_arrays()
        ref["excitation_force"][:, :, 0] = 100.0 + 50.0j
        cand = _metric_arrays()
        cand["excitation_force"][:, :, 0] = 100.0 + 50.005j
        report = compare_level_refinement(
            ref, cand, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))
        assert report["quantities"]["excitation_force"]["pass"] is True
        far = _metric_arrays()
        far["excitation_force"][:, :, 0] = 100.0 + 55.0j
        report = compare_level_refinement(
            ref, far, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))
        assert report["quantities"]["excitation_force"]["pass"] is False
        assert report["quantities"]["excitation_force"]["residuals"] == \
            [pytest.approx(5.0)]

    def test_partial_or_nonfinite_records_refused(self):
        ref = _metric_arrays()
        cand = _metric_arrays()
        cand["added_mass"] = np.zeros((2, 4, 4))
        with pytest.raises(SourceStudyWorkflowError):
            compare_level_refinement(
                ref, cand, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))
        cand = _metric_arrays()
        del cand["damping"]
        with pytest.raises(SourceStudyWorkflowError):
            compare_level_refinement(
                ref, cand, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))
        cand = _metric_arrays()
        cand["added_mass"][0, 0, 0] = np.nan
        with pytest.raises(SourceStudyWorkflowError):
            compare_level_refinement(
                ref, cand, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))
        cand = _metric_arrays()
        cand["excitation_force"][:, :, 0] = complex(np.inf, 0.0)
        with pytest.raises(SourceStudyWorkflowError):
            compare_level_refinement(
                ref, cand, _refinement_criteria(), (1.0, 1.0, 1.0, 1.0))


# --------------------------------------------------------------------------- #
# prepare-only path
# --------------------------------------------------------------------------- #


class TestPrepare:

    def test_prepare_writes_immutable_case_and_inventory(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        result = prepare_source_study(case, out)
        assert result["status"] == "not_executed"
        assert result["outcome"] == OUTCOME_PREPARED_NOT_EXECUTED
        assert result["qualification_scope"] == DEVELOPMENT_QUALIFICATION_SCOPE
        record = json.loads((out / "case.json").read_text(encoding="utf-8"))
        assert record["case_hash"] == case.content_hash()
        assert SourceStudyCase.from_dict(
            record["case"]).content_hash() == case.content_hash()
        inventory = json.loads(
            (out / "geometry_inventory.json").read_text(encoding="utf-8"))
        assert inventory["geometry_recipe"] == FIRST_SLICE_GEOMETRY_RECIPE
        assert inventory["levels"]["2"]["total_panels"] == 96
        assert inventory["levels"]["3"]["total_panels"] == 216
        assert inventory["levels"]["3"]["hull_panels_per_body"] == 90
        assert inventory["levels"]["3"]["lid_panels_per_body"] == 18
        assert inventory["levels"]["3"]["hull_vertices_per_body"] == 52
        assert inventory["levels"]["3"]["lid_vertices_per_body"] == 16
        assert inventory["request"]["omegas"] == [2.0, 4.0, 6.0]
        assert inventory["request"]["include_zero_frequency"] is False
        assert inventory["request"]["include_infinite_frequency"] is True
        assert inventory["request"]["backend_settings"] == {}
        status = _read_status(out)
        assert status["status"] == "not_executed"
        assert status["outcome"] == OUTCOME_PREPARED_NOT_EXECUTED

    def test_prepare_requires_exclusive_new_directory(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        out.mkdir()
        with pytest.raises(SourceStudyWorkflowError):
            prepare_source_study(case, out)

    def test_prepare_cli_public_path_no_backend_import(self, tmp_path):
        sys.modules.pop("capytaine", None)
        out = tmp_path / "study"
        assert main(["prepare", "--out", str(out)]) == 0
        assert (out / "case.json").is_file()
        assert (out / "geometry_inventory.json").is_file()
        assert _read_status(out)["status"] == "not_executed"
        # prepare-only path never imports the actual backend package
        assert "capytaine" not in sys.modules
        # prepare of a serialized case record round-trips through the CLI
        record = json.loads((out / "case.json").read_text(encoding="utf-8"))
        case_json = tmp_path / "case-record.json"
        case_json.write_text(json.dumps(record), encoding="utf-8")
        out2 = tmp_path / "study-2"
        assert main(["prepare", "--out", str(out2),
                     "--case-json", str(case_json)]) == 0
        assert _read_status(out2)["case_hash"] == record["case_hash"]


# --------------------------------------------------------------------------- #
# execute path (mock-rehearsal only)
# --------------------------------------------------------------------------- #


class TestExecute:

    def test_execute_success_two_levels_one_solve_each(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        backend = MockSourceBackend()
        result = _prepare_and_execute(case, backend, MockExecutionLimits(),
                                      out)
        assert result["outcome"] == \
            OUTCOME_SOURCE_ADMISSION_PASSED_DEVELOPMENT_ONLY
        assert result["no_go_reasons"] == []
        assert len(backend.solve_calls) == 2
        levels = result["levels"]
        assert levels["2"]["total_panels"] == 96
        assert levels["3"]["total_panels"] == 216
        for name in ("2", "3"):
            assert levels[name]["admitted"] is True
            assert levels[name]["solve_calls"] == 1
            assert levels[name]["qualification_status"] == "unqualified"
            assert "mock" in levels[name]["source_backend"]
        status = _read_status(out)
        assert status["status"] == "executed"
        assert status["outcome"] == \
            OUTCOME_SOURCE_ADMISSION_PASSED_DEVELOPMENT_ONLY
        report = _read_report(out)
        assert report["qualification_scope"] == \
            DEVELOPMENT_QUALIFICATION_SCOPE
        assert report["assumptions"]
        assert len(report["call_inventory"]) == 2
        assert all(entry["solve_calls"] == 1
                   for entry in report["call_inventory"])
        # full 12-source-DOF record preserved; nonzero off-body coupling
        loaded = load_source_study(out)
        artifact = loaded["levels"]["2"]["artifact"]
        assert len(artifact.source_dof_order) == 12
        assert artifact.admitted is True
        assert np.any(artifact.raw_added_mass[:, 0, 2] != 0.0)
        assert np.any(artifact.raw_damping[:, 0, 2] != 0.0)

    def test_execute_prospective_case_levels(self, tmp_path):
        # honest mock-rehearsal identity: the analytic double's
        # capabilities and raw source attrs match the case exactly
        case = _prospective_mock_case()
        out = tmp_path / "study"
        backend = MockSourceBackend()
        result = _prepare_and_execute(case, backend, MockExecutionLimits(),
                                      out)
        assert result["outcome"] == \
            OUTCOME_SOURCE_ADMISSION_PASSED_DEVELOPMENT_ONLY
        assert result["levels"]["4"]["total_panels"] == 384
        assert result["levels"]["6"]["total_panels"] == 864
        assert len(backend.solve_calls) == 2

    def test_execute_raw_backend_attr_mismatch_refused(self, tmp_path):
        # capabilities match the case, but the ACTUAL raw dataset attrs
        # claim a different backend: admission refuses the FIRST
        # received dataset (fail-closed, no artifact admitted, no
        # second solve) — but the RECEIVED schema-valid source is
        # preserved BEFORE admission so the rejection is diagnosable
        case = make_case()
        out = tmp_path / "study"
        backend = MockSourceBackend(dataset_backend="capytaine")
        with pytest.raises(SourceStudyIdentityError, match="backend"):
            _prepare_and_execute(case, backend, MockExecutionLimits(), out)
        assert len(backend.solve_calls) == 1
        assert _read_status(out)["status"] == "failed"
        assert not (out / "level-03").exists()
        # the received source survived the refusal, unmarked as admitted
        level = out / "level-02"
        assert level.is_dir()
        assert not (level / "report.json").exists()
        raw_files = list(level.glob("raw-*.nc"))
        assert len(raw_files) == 1
        received = load_hydro_dataset(raw_files[0])
        assert received.attrs["backend"] == "capytaine"
        report = _read_report(out)
        assert any(event["event"] == "failed" for event in report["log"])
        preserved = [e for e in report["log"]
                     if e["event"] == "received_source_preserved"]
        assert len(preserved) == 1
        assert preserved[0]["detail"]["file"] == raw_files[0].name

    def test_execute_raw_version_attr_mismatch_refused(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        backend = MockSourceBackend(dataset_backend_version="3.0.0")
        with pytest.raises(SourceStudyIdentityError,
                           match="backend_version"):
            _prepare_and_execute(case, backend, MockExecutionLimits(), out)
        assert len(backend.solve_calls) == 1
        assert _read_status(out)["status"] == "failed"
        assert not (out / "level-03").exists()
        # the received source survived the version-identity refusal
        level = out / "level-02"
        assert level.is_dir()
        raw_files = list(level.glob("raw-*.nc"))
        assert len(raw_files) == 1
        received = load_hydro_dataset(raw_files[0])
        assert received.attrs["backend_version"] == "3.0.0"

    @pytest.mark.parametrize("variant,expected", [
        # rotation_center/center_of_mass array mutations with unchanged
        # metadata are refused by the schema's own array-vs-attr proof
        # (typed SchemaError) inside admission; the fixed-fixture
        # admission check is the defense-in-depth behind it
        ("rotation_center_mutation", HydroError),
        ("center_of_mass_mutation", HydroError),
        # center_of_buoyancy has NO attr cross-check in the schema —
        # only the new fixed-fixture admission proof catches it
        ("center_of_buoyancy_mutation", SourceStudyIdentityError),
    ])
    def test_execute_center_array_mutation_refused(self, variant, expected,
                                                  tmp_path):
        # the metadata attrs and request hash are UNCHANGED; the actual
        # data-variable proof refuses the mutated source
        case = make_case()
        out = tmp_path / "study"
        backend = MockSourceBackend(variant=variant)
        if expected is SourceStudyIdentityError:
            with pytest.raises(expected, match="fixed-fixture mismatch"):
                _prepare_and_execute(case, backend, MockExecutionLimits(),
                                     out)
        else:
            with pytest.raises(expected):
                _prepare_and_execute(case, backend, MockExecutionLimits(),
                                     out)
        assert len(backend.solve_calls) == 2
        assert _read_status(out)["status"] == "failed"
        # prior level evidence is retained under the existing failure
        # behavior; the mutated level saved nothing
        assert (out / "level-02" / "report.json").is_file()
        level = out / "level-03"
        assert level.is_dir()
        assert not (level / "report.json").exists()
        if expected is SourceStudyIdentityError:
            # schema-valid but fixture-mismatched: the RECEIVED raw
            # source is preserved through the public save route
            raw_files = list(level.glob("raw-*.nc"))
            assert len(raw_files) == 1
            assert not list(level.glob("received-forensic-*.json"))
        else:
            # schema-invalid (the schema's own array-vs-attr proof):
            # a bounded labelled forensic archive, never a raw dataset
            assert not list(level.glob("raw-*.nc"))
            archives = list(level.glob("received-forensic-*.json"))
            assert len(archives) == 1
            archive = json.loads(
                archives[0].read_text(encoding="utf-8"))
            assert archive["format"] == RECEIVED_ARCHIVE_FORMAT
            assert archive["status"] == "rejected_received_schema_invalid"
            assert archive["admitted"] is False
            assert archive["level"] == 3
            assert archive["failure_type"].endswith("Error")
            # bounded forensic arrays/coords/attrs of the received source
            assert "rotation_center" in archive["data_variables"]
            assert archive["data_variables"]["rotation_center"]["truncated"] \
                is False
            assert len(archive["data_variables"]["rotation_center"]
                       ["elements"]) <= 32

    def test_execute_received_source_survives_center_mutation(self,
                                                              tmp_path):
        # center_of_buoyancy has no schema attr cross-check: the raw
        # source is schema-valid, saved through the PUBLIC save route
        # BEFORE admission, and survives the fixed-fixture refusal with
        # the mutated array intact for diagnosis
        case = make_case()
        out = tmp_path / "study"
        backend = MockSourceBackend(variant="center_of_buoyancy_mutation")
        with pytest.raises(SourceStudyIdentityError,
                           match="fixed-fixture mismatch"):
            _prepare_and_execute(case, backend, MockExecutionLimits(), out)
        assert _read_status(out)["status"] == "failed"
        raw_files = list((out / "level-03").glob("raw-*.nc"))
        assert len(raw_files) == 1
        received = load_hydro_dataset(raw_files[0])
        assert float(received["center_of_buoyancy"].values[0, 0]) \
            == pytest.approx(-2.0 + 0.1)
        report = _read_report(out)
        preserved = [e for e in report["log"]
                     if e["event"] == "received_source_preserved"]
        assert len(preserved) == 2
        assert any(e["detail"]["file"] == raw_files[0].name
                   for e in preserved)

    def test_execute_wrong_version_refused_before_any_solve(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        prepare_source_study(case, out)
        backend = MockSourceBackend(backend_version="2.0.0")
        with pytest.raises(SourceStudyIdentityError):
            execute_source_study(case, backend, MockExecutionLimits(), out)
        assert backend.solve_calls == []
        assert _read_status(out)["status"] == "not_executed"

    def test_execute_wrong_backend_name_refused(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        prepare_source_study(case, out)
        backend = MockSourceBackend(backend_name="foam")
        with pytest.raises(SourceStudyIdentityError):
            execute_source_study(case, backend, MockExecutionLimits(), out)
        assert backend.solve_calls == []

    def test_execute_extra_zero_capability_not_rejected(self, tmp_path):
        # include_zero_frequency=False means the zero-frequency limit is
        # NOT REQUESTED/COMPUTED: a backend advertising ADDITIONAL
        # zero-frequency capability is not a mismatch and must not be
        # rejected before the solves
        case = make_case()
        out = tmp_path / "study"
        prepare_source_study(case, out)
        backend = MockSourceBackend(
            cap_overrides={"supports_zero_frequency": True})
        result = execute_source_study(case, backend, MockExecutionLimits(),
                                      out)
        assert result["outcome"] == \
            OUTCOME_SOURCE_ADMISSION_PASSED_DEVELOPMENT_ONLY
        assert len(backend.solve_calls) == 2
        # the report states explicitly that zero frequency was NOT computed
        report = _read_report(out)
        assert report["effective_controls"][
            "zero_frequency_computed"] is False

    def test_execute_missing_multibody_refused_before_any_solve(
            self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        prepare_source_study(case, out)
        backend = MockSourceBackend(
            cap_overrides={"supports_multibody": False})
        with pytest.raises(SourceStudyIdentityError, match="multibody"):
            execute_source_study(case, backend, MockExecutionLimits(), out)
        assert backend.solve_calls == []

    def test_execute_missing_infinite_frequency_refused_before_any_solve(
            self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        prepare_source_study(case, out)
        backend = MockSourceBackend(
            cap_overrides={"supports_infinite_frequency": False})
        with pytest.raises(SourceStudyIdentityError,
                           match="infinite-frequency"):
            execute_source_study(case, backend, MockExecutionLimits(), out)
        assert backend.solve_calls == []

    def test_execute_missing_explicit_ainf_refused_before_any_solve(
            self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        prepare_source_study(case, out)
        backend = MockSourceBackend(
            cap_overrides={"provides_added_mass_infinite": False})
        with pytest.raises(SourceStudyIdentityError,
                           match="added_mass_infinite"):
            execute_source_study(case, backend, MockExecutionLimits(), out)
        assert backend.solve_calls == []

    def test_execute_strict_caps_refused_before_any_solve(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        prepare_source_study(case, out)
        tight = MockExecutionLimits(max_total_panels=100)
        backend = MockSourceBackend()
        with pytest.raises(SourceStudyIdentityError):
            execute_source_study(case, backend, tight, out)
        assert backend.solve_calls == []
        threads = MockExecutionLimits(n_threads=2)
        with pytest.raises(SourceStudyIdentityError):
            execute_source_study(case, MockSourceBackend(), threads, out)

    def test_execute_psd_fail_no_go_nonadmitted_artifact_saved(
            self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        backend = MockSourceBackend(variant="psd_fail")
        result = _prepare_and_execute(case, backend, MockExecutionLimits(),
                                      out)
        assert result["outcome"] == OUTCOME_NO_GO
        assert result["no_go_reasons"]
        for name in ("2", "3"):
            entry = result["levels"][name]
            assert entry["admitted"] is False
            # non-admitted artifacts are saved too
            level_dir = out / f"level-0{name}"
            assert (level_dir / entry["raw_file"]).is_file()
            assert (level_dir / entry["derived_artifact"]).is_file()
        assert _read_status(out)["status"] == "executed"

    def test_execute_asym_raw_preserved_no_go(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        backend = MockSourceBackend(variant="asym")
        result = _prepare_and_execute(case, backend, MockExecutionLimits(),
                                      out)
        assert result["outcome"] == OUTCOME_NO_GO
        loaded = load_source_study(out)
        artifact = loaded["levels"]["2"]["artifact"]
        assert artifact.admitted is False
        # no raw symmetrisation or clamping on the raw route
        asym = artifact.raw_added_mass[:, 0, 1] - artifact.raw_added_mass[:,
                                                                          1,
                                                                          0]
        assert np.all(np.abs(asym) > 0.5)
        # the derived route applies the declared reciprocal average
        assert np.allclose(
            artifact.derived_added_mass[:, 0, 1],
            artifact.derived_added_mass[:, 1, 0])

    def test_execute_level_bias_mesh_criterion_no_go(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        backend = MockSourceBackend(variant="level_bias")
        result = _prepare_and_execute(case, backend, MockExecutionLimits(),
                                      out)
        assert result["outcome"] == OUTCOME_NO_GO
        assert any("mesh refinement criterion" in reason
                   for reason in result["no_go_reasons"])
        # per-level admission still passes; only the cross-level mesh
        # criterion fails
        assert all(entry["admitted"]
                   for entry in result["levels"].values())
        report = result["refinement"]["levels"]["2"]
        assert report["mesh_pass"] is False
        assert result["refinement"]["reference_level"] == 3
        for name in SOURCE_STUDY_QUANTITIES:
            entry = report["quantities"][name]
            assert entry["pass"] is False
            assert entry["residuals"]
        # complete per-frequency/heading residuals, never only counts
        assert len(report["quantities"]["added_mass"]["residuals"]) == 3
        assert len(report["quantities"]["excitation_force"]["residuals"]) == 3

    def test_execute_grid_mismatch_fails_closed(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        backend = MockSourceBackend(variant="grid_mismatch")
        with pytest.raises(SourceStudyIdentityError):
            _prepare_and_execute(case, backend, MockExecutionLimits(), out)
        assert len(backend.solve_calls) == 2
        status = _read_status(out)
        assert status["status"] == "failed"
        assert status["outcome"] == OUTCOME_FAILED
        # prior level artifacts are retained
        assert (out / "level-02").is_dir()
        assert (out / "level-02" / "report.json").is_file()
        # the schema-valid but grid-mismatched received source is
        # preserved (unmarked as admitted) for diagnosis
        assert len(list((out / "level-03").glob("raw-*.nc"))) == 1
        assert not (out / "level-03" / "report.json").exists()
        report = _read_report(out)
        assert report["outcome"] == OUTCOME_FAILED
        assert any(event["event"] == "failed" for event in report["log"])

    def test_execute_foreign_mesh_fails_closed(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        backend = MockSourceBackend(variant="foreign_mesh")
        with pytest.raises(SourceStudyIdentityError):
            _prepare_and_execute(case, backend, MockExecutionLimits(), out)
        assert _read_status(out)["status"] == "failed"
        assert (out / "level-02").is_dir()

    def test_execute_qualified_label_refused(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        backend = MockSourceBackend(variant="qualified")
        with pytest.raises(SourceStudyIdentityError):
            _prepare_and_execute(case, backend, MockExecutionLimits(), out)
        assert _read_status(out)["status"] == "failed"
        assert (out / "level-02").is_dir()

    def test_execute_convention_mismatch_fails_closed(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        backend = MockSourceBackend(variant="phasor_mismatch")
        with pytest.raises(HydroError):
            _prepare_and_execute(case, backend, MockExecutionLimits(), out)
        assert _read_status(out)["status"] == "failed"
        assert (out / "level-02").is_dir()

    def test_execute_solve_failure_retains_prior_level_and_refuses_retry(
            self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        backend = MockSourceBackend(variant="fail_on_solve")
        with pytest.raises(RuntimeError):
            _prepare_and_execute(case, backend, MockExecutionLimits(), out)
        assert len(backend.solve_calls) == 2
        status = _read_status(out)
        assert status["status"] == "failed"
        assert (out / "level-02" / "report.json").is_file()
        assert not (out / "level-03").exists()
        # no automatic retry over the failed study
        with pytest.raises(SourceStudyWorkflowError):
            execute_source_study(case, MockSourceBackend(),
                                 MockExecutionLimits(), out)

    def test_execute_cancellation_preserves_evidence(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        prepare_source_study(case, out)
        backend = MockSourceBackend()
        result = execute_source_study(
            case, backend, MockExecutionLimits(), out,
            should_cancel=lambda level: True)
        assert result["outcome"] == OUTCOME_CANCELLED
        assert backend.solve_calls == []
        status = _read_status(out)
        assert status["status"] == "cancelled"
        assert status["outcome"] == OUTCOME_CANCELLED
        report = _read_report(out)
        assert report["outcome"] == OUTCOME_CANCELLED
        assert any(event["event"] == "cancelled" for event in report["log"])
        assert not (out / "level-02").exists()

    def test_execute_cancellation_preserves_prior_level_evidence(self,
                                                                 tmp_path):
        # cancelling BEFORE the second level keeps the completed first
        # level's full evidence (received source, report, artifacts)
        case = make_case()
        out = tmp_path / "study"
        prepare_source_study(case, out)
        backend = MockSourceBackend()
        result = execute_source_study(
            case, backend, MockExecutionLimits(), out,
            should_cancel=lambda level: int(level) >= 3)
        assert result["outcome"] == OUTCOME_CANCELLED
        assert len(backend.solve_calls) == 1
        assert _read_status(out)["status"] == "cancelled"
        level = out / "level-02"
        assert (level / "report.json").is_file()
        assert len(list(level.glob("raw-*.nc"))) == 1
        assert len(list(level.glob("derived-*.npz"))) == 1
        assert not (out / "level-03").exists()
        report = _read_report(out)
        assert report["outcome"] == OUTCOME_CANCELLED
        assert set(report["levels"]) == {"2"}

    def test_execute_changed_case_invalidates_identity(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        prepare_source_study(case, out)
        changed = make_case(revision="changed-revision")
        backend = MockSourceBackend()
        with pytest.raises(SourceStudyIdentityError):
            execute_source_study(changed, backend, MockExecutionLimits(),
                                out)
        assert backend.solve_calls == []
        assert _read_status(out)["status"] == "not_executed"

    def test_execute_reexecution_refused(self, completed_study):
        case = make_case()
        with pytest.raises(SourceStudyWorkflowError):
            execute_source_study(case, MockSourceBackend(),
                                 MockExecutionLimits(), completed_study)

    def test_execute_requires_prepared_directory(self, tmp_path):
        case = make_case()
        with pytest.raises(SourceStudyWorkflowError):
            execute_source_study(case, MockSourceBackend(),
                                 MockExecutionLimits(),
                                 tmp_path / "does-not-exist")


# --------------------------------------------------------------------------- #
# reload and adjudication
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def completed_study(tmp_path_factory):
    out = tmp_path_factory.mktemp("completed-study") / "study"
    case = make_case()
    prepare_source_study(case, out)
    result = execute_source_study(case, MockSourceBackend(),
                                  MockExecutionLimits(), out)
    assert result["outcome"] == \
        OUTCOME_SOURCE_ADMISSION_PASSED_DEVELOPMENT_ONLY
    return out


def _copy_study(completed_study, tmp_path):
    dst = tmp_path / "study"
    shutil.copytree(completed_study, dst)
    return dst


def _mutate_raw_dataset(out, level: int, mutate) -> None:
    """Mutate a saved raw dataset and make the recorded hashes consistent.

    The recorded ``raw_file_sha256``/``dataset_content_hash`` are
    recomputed after the mutation so the byte/content hash bookkeeping
    cannot masquerade the change as intact: any refusal must come from
    the actual raw-source admission proof (attrs or data variables).
    """
    import hashlib

    from anyhydro.hashing import dataset_content_hash
    from anyhydro.persistence import load_hydro_dataset, save_hydro_dataset

    level_dir = out / f"level-0{level}"
    report = json.loads(
        (level_dir / "report.json").read_text(encoding="utf-8"))
    raw_path = level_dir / report["raw_file"]
    dataset = load_hydro_dataset(raw_path)
    mutate(dataset)
    save_hydro_dataset(dataset, raw_path)
    report["raw_file_sha256"] = hashlib.sha256(
        raw_path.read_bytes()).hexdigest()
    report["dataset_content_hash"] = dataset_content_hash(dataset)
    (level_dir / "report.json").write_text(
        json.dumps(report), encoding="utf-8")


class TestLoadAdjudicate:

    def test_load_verifies_and_adjudicate_reproduces(self, completed_study):
        loaded = load_source_study(completed_study)
        assert loaded["status"]["status"] == "executed"
        assert set(loaded["levels"]) == {"2", "3"}
        artifact = loaded["levels"]["3"]["artifact"]
        assert artifact.admitted is True
        assert artifact.qualification_status == "unqualified"
        adjudicated = adjudicate_source_study(completed_study)
        assert adjudicated["reproduced"] is True
        assert adjudicated["outcome"] == \
            OUTCOME_SOURCE_ADMISSION_PASSED_DEVELOPMENT_ONLY
        assert adjudicated["no_go_reasons"] == []
        assert adjudicated["qualification_scope"] == \
            DEVELOPMENT_QUALIFICATION_SCOPE
        report = adjudicated["refinement"]["levels"]["2"]
        assert report["mesh_pass"] is True
        assert adjudicated["refinement"]["reference_level"] == 3
        for name in SOURCE_STUDY_QUANTITIES:
            entry = report["quantities"][name]
            assert entry["pass"] is True
            assert entry["residuals"] == [0.0] * len(entry["residuals"])

    def test_load_refuses_tampered_raw_bytes(self, completed_study,
                                             tmp_path):
        out = _copy_study(completed_study, tmp_path)
        level_report = json.loads(
            (out / "level-02" / "report.json").read_text(encoding="utf-8"))
        raw = out / "level-02" / level_report["raw_file"]
        with open(raw, "ab") as handle:
            handle.write(b"tamper")
        with pytest.raises(SourceStudyIdentityError):
            load_source_study(out)

    def test_load_refuses_swapped_derived_artifacts(self, completed_study,
                                                    tmp_path):
        # swapping derived artifacts between levels is caught by the full
        # raw-source proof, not only by the recorded hashes
        out = _copy_study(completed_study, tmp_path)
        report_a = json.loads(
            (out / "level-02" / "report.json").read_text(encoding="utf-8"))
        report_b = json.loads(
            (out / "level-03" / "report.json").read_text(encoding="utf-8"))
        path_a = out / "level-02" / report_a["derived_artifact"]
        path_b = out / "level-03" / report_b["derived_artifact"]
        backup = out / "backup.npz"
        shutil.move(str(path_a), str(backup))
        shutil.move(str(path_b), str(path_a))
        shutil.move(str(backup), str(path_b))
        with pytest.raises(HydroError):
            load_source_study(out)

    def test_load_refuses_changed_case_record(self, completed_study,
                                              tmp_path):
        out = _copy_study(completed_study, tmp_path)
        record = json.loads((out / "case.json").read_text(encoding="utf-8"))
        record["case"]["revision"] = "tampered"
        (out / "case.json").write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(SourceStudyIdentityError):
            load_source_study(out)

    def test_load_refuses_corrupt_case_json(self, completed_study, tmp_path):
        out = _copy_study(completed_study, tmp_path)
        (out / "case.json").write_text("not json at all", encoding="utf-8")
        with pytest.raises(SourceStudyWorkflowError):
            load_source_study(out)

    def test_load_refuses_missing_level(self, completed_study, tmp_path):
        out = _copy_study(completed_study, tmp_path)
        shutil.rmtree(out / "level-03")
        with pytest.raises(SourceStudyWorkflowError):
            load_source_study(out)

    def test_load_refuses_unknown_level_directory(self, completed_study,
                                                  tmp_path):
        out = _copy_study(completed_study, tmp_path)
        (out / "level-99").mkdir()
        with pytest.raises(SourceStudyWorkflowError):
            load_source_study(out)

    def test_load_refuses_traversal_file_names(self, completed_study,
                                               tmp_path):
        out = _copy_study(completed_study, tmp_path)
        report_path = out / "level-02" / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["raw_file"] = "../evil.nc"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(SourceStudyWorkflowError):
            load_source_study(out)

    def test_load_refuses_mismatched_raw_backend_attrs(self,
                                                       completed_study,
                                                       tmp_path):
        # reload proof: raw attrs claiming a backend the case does not
        # expect are refused even when the recorded hashes were made
        # consistent with the tampered file
        out = _copy_study(completed_study, tmp_path)

        def mutate(dataset):
            dataset.attrs["backend"] = "capytaine"

        _mutate_raw_dataset(out, 2, mutate)
        with pytest.raises(SourceStudyIdentityError, match="backend"):
            load_source_study(out)

    def test_load_refuses_mutated_center_arrays(self, completed_study,
                                                tmp_path):
        # center_of_buoyancy has NO metadata attr cross-check, the
        # metadata attrs and request hash are UNCHANGED and the
        # recorded hashes were made consistent: only the actual
        # fixed-fixture data-variable proof at reload catches the
        # mutated array
        out = _copy_study(completed_study, tmp_path)

        def mutate(dataset):
            values = np.array(dataset["center_of_buoyancy"].values,
                              dtype=np.float64)
            values[0, 0] = -1.9
            dataset["center_of_buoyancy"] = (("body", "xyz"), values)

        _mutate_raw_dataset(out, 2, mutate)
        with pytest.raises(SourceStudyIdentityError,
                           match="fixed-fixture mismatch"):
            load_source_study(out)
        # adjudicate goes through the same reload proof
        out2 = _copy_study(completed_study, tmp_path / "second")
        _mutate_raw_dataset(out2, 3, mutate)
        with pytest.raises(SourceStudyIdentityError,
                           match="fixed-fixture mismatch"):
            adjudicate_source_study(out2)

    def test_adjudicate_refuses_tampered_study_report(self, completed_study,
                                                      tmp_path):
        out = _copy_study(completed_study, tmp_path)
        report = _read_report(out)
        report["outcome"] = OUTCOME_NO_GO
        (out / "study_report.json").write_text(
            json.dumps(report), encoding="utf-8")
        with pytest.raises(SourceStudyIdentityError):
            adjudicate_source_study(out)

    def test_adjudicate_refuses_tampered_refinement(self, completed_study,
                                                    tmp_path):
        out = _copy_study(completed_study, tmp_path)
        report = _read_report(out)
        report["refinement"]["levels"]["2"]["quantities"]["added_mass"][
            "residuals"] = [1.0, 1.0, 1.0]
        (out / "study_report.json").write_text(
            json.dumps(report), encoding="utf-8")
        with pytest.raises(SourceStudyIdentityError):
            adjudicate_source_study(out)

    def test_adjudicate_not_executed_outcome(self, tmp_path):
        case = make_case()
        out = tmp_path / "study"
        prepare_source_study(case, out)
        result = adjudicate_source_study(out)
        assert result["outcome"] == OUTCOME_PREPARED_NOT_EXECUTED
        assert result["reproduced"] is True
        assert result["levels"] == {}


# --------------------------------------------------------------------------- #
# registered geometry/request inventory binding (parent-05 defect repair)
# --------------------------------------------------------------------------- #


class TestGeometryInventoryBinding:
    """The saved geometry/request inventory is bound to the immutable
    case on EVERY reload/execute/adjudicate: a missing, unreadable,
    unknown, removed or modified inventory (level panels, per-body mesh
    hashes, request hash) — and current-recipe drift under the same
    case hash — is a typed refusal BEFORE any backend capability check
    or solve. No autocorrection, no migration, no re-registration; a
    prepared case can never silently execute new geometry."""

    def _tamper_inventory(self, out, mutate) -> None:
        inventory = json.loads(
            (out / "geometry_inventory.json").read_text(encoding="utf-8"))
        mutate(inventory)
        (out / "geometry_inventory.json").write_text(
            json.dumps(inventory), encoding="utf-8")

    def test_prepared_reload_and_adjudicate_bind_inventory(self, tmp_path):
        out = tmp_path / "study"
        prepare_source_study(make_case(), out)
        loaded = load_source_study(out)
        assert loaded["status"]["status"] == "not_executed"
        assert adjudicate_source_study(out)["outcome"] == \
            OUTCOME_PREPARED_NOT_EXECUTED

    def test_completed_reload_and_adjudicate_bind_inventory(
            self, completed_study):
        loaded = load_source_study(completed_study)
        assert loaded["status"]["status"] == "executed"
        assert adjudicate_source_study(completed_study)["reproduced"] is True

    def test_load_refuses_missing_inventory_prepared(self, tmp_path):
        out = tmp_path / "study"
        prepare_source_study(make_case(), out)
        (out / "geometry_inventory.json").unlink()
        with pytest.raises(SourceStudyWorkflowError):
            load_source_study(out)
        with pytest.raises(SourceStudyWorkflowError):
            adjudicate_source_study(out)

    def test_load_refuses_missing_inventory_completed(self, completed_study,
                                                       tmp_path):
        # a missing COMPLETE-study inventory also refuses
        out = _copy_study(completed_study, tmp_path)
        (out / "geometry_inventory.json").unlink()
        with pytest.raises(SourceStudyWorkflowError):
            load_source_study(out)
        with pytest.raises(SourceStudyWorkflowError):
            adjudicate_source_study(out)

    def test_load_refuses_corrupt_inventory(self, completed_study,
                                             tmp_path):
        out = _copy_study(completed_study, tmp_path)
        (out / "geometry_inventory.json").write_text(
            "not json at all", encoding="utf-8")
        with pytest.raises(SourceStudyWorkflowError):
            load_source_study(out)

    def test_load_refuses_modified_level_panels(self, completed_study,
                                                tmp_path):
        out = _copy_study(completed_study, tmp_path)

        def mutate(inventory):
            inventory["levels"]["3"]["total_panels"] += 24

        self._tamper_inventory(out, mutate)
        with pytest.raises(SourceStudyIdentityError):
            load_source_study(out)

    def test_load_refuses_modified_mesh_hash(self, completed_study,
                                             tmp_path):
        out = _copy_study(completed_study, tmp_path)

        def mutate(inventory):
            inventory["levels"]["2"]["mesh_hashes"]["first-slice-a"][
                "hull"] = "0" * 64

        self._tamper_inventory(out, mutate)
        with pytest.raises(SourceStudyIdentityError):
            load_source_study(out)

    def test_load_refuses_modified_request_hash(self, completed_study,
                                                tmp_path):
        out = _copy_study(completed_study, tmp_path)

        def mutate(inventory):
            inventory["levels"]["2"]["request_hash"] = "tampered"

        self._tamper_inventory(out, mutate)
        with pytest.raises(SourceStudyIdentityError):
            load_source_study(out)

    def test_load_refuses_unknown_inventory_field(self, completed_study,
                                                   tmp_path):
        # closed world: an unknown extra field is a drift, not extra
        # harmless metadata
        out = _copy_study(completed_study, tmp_path)

        def mutate(inventory):
            inventory["extra_field"] = 1

        self._tamper_inventory(out, mutate)
        with pytest.raises(SourceStudyIdentityError):
            load_source_study(out)

    def test_load_refuses_removed_inventory_field(self, completed_study,
                                                   tmp_path):
        out = _copy_study(completed_study, tmp_path)

        def mutate(inventory):
            del inventory["levels"]["2"]["mesh_hashes"]

        self._tamper_inventory(out, mutate)
        with pytest.raises(SourceStudyIdentityError):
            load_source_study(out)

    def test_execute_refuses_missing_inventory_before_caps_and_solve(
            self, tmp_path):
        out = tmp_path / "study"
        case = make_case()
        prepare_source_study(case, out)
        (out / "geometry_inventory.json").unlink()

        class _NoCapsNoSolveBackend(MockSourceBackend):
            def capabilities(self):
                raise AssertionError(
                    "capabilities consulted before inventory validation")

            def solve(self, bodies, request):
                raise AssertionError(
                    "solve attempted before inventory validation")

        with pytest.raises(SourceStudyWorkflowError):
            execute_source_study(case, _NoCapsNoSolveBackend(),
                                 MockExecutionLimits(), out)
        assert _read_status(out)["status"] == "not_executed"

    def test_execute_refuses_tampered_inventory_before_any_solve(
            self, tmp_path):
        out = tmp_path / "study"
        case = make_case()
        prepare_source_study(case, out)

        def mutate(inventory):
            inventory["levels"]["2"]["mesh_hashes"]["first-slice-a"][
                "hull"] = "0" * 64

        self._tamper_inventory(out, mutate)
        backend = MockSourceBackend()
        with pytest.raises(SourceStudyIdentityError):
            execute_source_study(case, backend, MockExecutionLimits(), out)
        assert backend.solve_calls == []
        assert _read_status(out)["status"] == "not_executed"

    def test_load_refuses_current_recipe_drift(self, completed_study,
                                               tmp_path, monkeypatch):
        # TESTING HOOK ONLY (monkeypatch, no production hook): simulate
        # the CURRENT built recipe drifting under the SAME case hash
        # (e.g. a fixture change after preparation); the rebuilt
        # expected inventory no longer matches the saved registered one
        out = _copy_study(completed_study, tmp_path)
        original = anyhydro.source_study._geometry_inventory

        def drifted(case_):
            record = original(case_)
            record["levels"]["2"]["mesh_hashes"]["first-slice-a"][
                "hull"] = "0" * 64
            return record

        monkeypatch.setattr(anyhydro.source_study, "_geometry_inventory",
                            drifted)
        with pytest.raises(SourceStudyIdentityError):
            load_source_study(out)

    def test_execute_refuses_current_recipe_drift_before_any_solve(
            self, tmp_path, monkeypatch):
        out = tmp_path / "study"
        case = make_case()
        prepare_source_study(case, out)
        original = anyhydro.source_study._geometry_inventory

        def drifted(case_):
            record = original(case_)
            record["levels"]["2"]["total_panels"] += 24
            return record

        monkeypatch.setattr(anyhydro.source_study, "_geometry_inventory",
                            drifted)
        backend = MockSourceBackend()
        with pytest.raises(SourceStudyIdentityError):
            execute_source_study(case, backend, MockExecutionLimits(), out)
        assert backend.solve_calls == []
        assert _read_status(out)["status"] == "not_executed"


# --------------------------------------------------------------------------- #
# two-body coupling observability gate (scientific gate, mock-rehearsal only)
# --------------------------------------------------------------------------- #


class TestCouplingGate:

    def test_good_path_coupling_observable_per_level(self, completed_study):
        loaded = load_source_study(completed_study)
        for name in ("2", "3"):
            entry = loaded["levels"][name]["level_report"]
            coupling = entry["coupling"]
            assert coupling["observable"] is True
            assert coupling["bodies"] == ["first-slice-a", "first-slice-b"]
            # truthful per-frequency off-body block norms in the case's
            # declared normalized units, both ordered body pairs
            assert set(coupling["blocks"]) == {
                "added_mass", "damping", "added_mass_infinite"}
            for pair in ("first-slice-a::first-slice-b",
                         "first-slice-b::first-slice-a"):
                assert len(coupling["blocks"]["added_mass"][pair]) == 3
                assert len(coupling["blocks"]["damping"][pair]) == 3
                assert coupling["blocks"]["added_mass"][pair][0] > \
                    coupling["floors"]["added_mass"]
            # the predicate is stored separately from the artifact's own
            # admission verdict and never relabels it
            assert loaded["levels"][name]["artifact"].admitted is True

    def test_block_diagonal_source_no_go_and_reload_same(self, tmp_path):
        # an otherwise identical positive-PSD block-diagonal FakeSource
        # passes the shape-12/public operator admission but MUST fail the
        # fixed two-body coupling observability gate
        case = make_case()
        out = tmp_path / "study"
        backend = MockSourceBackend(variant="block_diagonal")
        result = _prepare_and_execute(case, backend, MockExecutionLimits(),
                                      out)
        assert result["outcome"] == OUTCOME_NO_GO
        assert any("coupling not observable" in reason
                   for reason in result["no_go_reasons"])
        for name in ("2", "3"):
            entry = result["levels"][name]
            # operator admission itself still passes; only the coupling
            # predicate fails — the artifact's own truth is untouched
            assert entry["admitted"] is True
            assert entry["coupling"]["observable"] is False
            assert all(
                norm == 0.0
                for norms in entry["coupling"]["blocks"]
                ["added_mass"].values() for norm in norms)
            # raw + operator artifacts are preserved for the NoGo study
            level_dir = out / f"level-0{name}"
            assert (level_dir / entry["raw_file"]).is_file()
            assert (level_dir / entry["derived_artifact"]).is_file()
        # reload + adjudicate reproduce the same NoGo with the same
        # coupling metrics, without any fresh BEM
        adjudicated = adjudicate_source_study(out)
        assert adjudicated["outcome"] == OUTCOME_NO_GO
        assert any("coupling not observable" in reason
                   for reason in adjudicated["no_go_reasons"])
        for name in ("2", "3"):
            assert adjudicated["levels"][name]["coupling"][
                "observable"] is False

    def test_adjudicate_refuses_tampered_coupling_record(self,
                                                         completed_study,
                                                         tmp_path):
        # flipping the stored observable bool must not masquerade as a
        # pass: adjudication re-derives the predicate + metrics from the
        # reloaded public artifact arrays
        out = _copy_study(completed_study, tmp_path)
        report_path = out / "level-02" / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["coupling"]["observable"] = False
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises(SourceStudyIdentityError):
            adjudicate_source_study(out)

    def test_adjudicate_refuses_removed_coupling_record(self,
                                                        completed_study,
                                                        tmp_path):
        out = _copy_study(completed_study, tmp_path)
        report_path = out / "level-02" / "report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        del report["coupling"]
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with pytest.raises((SourceStudyIdentityError,
                            SourceStudyWorkflowError)):
            load_source_study(out)


# --------------------------------------------------------------------------- #
# module CLI
# --------------------------------------------------------------------------- #


class TestCLI:

    def test_cli_execute_requires_explicit_backend_choice(self, tmp_path):
        out = tmp_path / "study"
        assert main(["prepare", "--out", str(out)]) == CLI_EXIT_OK
        # typed nonzero refusal, not a raised traceback and not success
        assert main(["execute", "--out", str(out)]) == CLI_EXIT_ERROR
        assert main(["execute", "--out", str(out),
                     "--backend", "foam"]) == CLI_EXIT_ERROR
        assert _read_status(out)["status"] == "not_executed"

    def test_cli_adjudicate_not_executed(self, tmp_path):
        out = tmp_path / "study"
        assert main(["prepare", "--out", str(out)]) == CLI_EXIT_OK
        # adjudicating a prepared/incomplete study is nonzero: it must
        # not pretend an actual source pass
        assert main(["adjudicate", "--out", str(out)]) == CLI_EXIT_INCOMPLETE

    def test_cli_adjudicate_missing_inventory_error(self, tmp_path):
        out = tmp_path / "study"
        assert main(["prepare", "--out", str(out)]) == CLI_EXIT_OK
        (out / "geometry_inventory.json").unlink()
        # typed refusal (CLI_EXIT_ERROR), never a fabricated success
        assert main(["adjudicate", "--out", str(out)]) == CLI_EXIT_ERROR


def _mock_backend_factory(captured, **backend_kwargs):
    """Mock the CapytaineBackend CLASS at the real factory boundary.

    EXPLICITLY LABELLED REHEARSAL (no actual solver is claimed or
    imported): the CLI's real production path still constructs the REAL
    public CapytaineExecutionLimits from the case-derived caps; only the
    backend class is a mock-rehearsal double (no BEM import/calls).
    For these CLI factory-boundary tests ONLY, the double's
    capabilities and raw dataset attrs claim the capytaine 3.0.0
    identity the prospective case expects, so the real production path
    is exercised end to end; every artifact they produce is rehearsal
    output, never scientific evidence.
    """

    class _FactoryBackend(MockSourceBackend):
        def __init__(self, *, cache_dir=None, execution_limits=None):
            super().__init__(
                backend_name="capytaine", backend_version="3.0.0",
                dataset_backend="capytaine",
                dataset_backend_version="3.0.0",
                **backend_kwargs)
            captured["cache_dir"] = cache_dir
            captured["execution_limits"] = execution_limits

    return _FactoryBackend


class TestCLIGate:

    def test_cli_execute_production_path_real_limits_mock_backend(
            self, tmp_path, monkeypatch):
        # EXPLICITLY LABELLED REHEARSAL: the factory-boundary double's
        # capabilities and raw attrs claim capytaine 3.0.0 (the
        # prospective case identity) so the REAL production path is
        # exercised; no actual solver is claimed, imported or called
        import anyhydro.capytaine_backend as cb_module
        from anyhydro import CapytaineExecutionLimits
        sys.modules.pop("capytaine", None)
        captured: dict = {}
        monkeypatch.setattr(cb_module, "CapytaineBackend",
                            _mock_backend_factory(captured))
        out = tmp_path / "study"
        assert main(["prepare", "--out", str(out)]) == CLI_EXIT_OK
        # the prospective case (levels 4 and 6 = 384/864 panels) drives
        # the case-derived explicit caps through the PUBLIC limits
        assert main(["execute", "--out", str(out),
                     "--backend", "capytaine"]) == CLI_EXIT_OK
        limits = captured["execution_limits"]
        assert isinstance(limits, CapytaineExecutionLimits)
        assert limits.max_total_panels == 864
        assert limits.max_finite_omegas == 3
        assert limits.max_wave_directions == 1
        assert limits.n_jobs == 1
        assert limits.n_threads == 1
        assert _read_status(out)["status"] == "executed"
        assert _read_status(out)["outcome"] == \
            OUTCOME_SOURCE_ADMISSION_PASSED_DEVELOPMENT_ONLY
        # no BEM package was ever imported on the mocked main path
        assert "capytaine" not in sys.modules

    def test_cli_execute_no_go_nonzero(self, tmp_path, monkeypatch):
        import anyhydro.capytaine_backend as cb_module
        captured: dict = {}
        monkeypatch.setattr(
            cb_module, "CapytaineBackend",
            _mock_backend_factory(captured, variant="psd_fail"))
        out = tmp_path / "study"
        assert main(["prepare", "--out", str(out)]) == CLI_EXIT_OK
        assert main(["execute", "--out", str(out),
                     "--backend", "capytaine"]) == CLI_EXIT_NO_GO
        # the report and artifacts are retained for the nonzero exit
        assert _read_status(out)["outcome"] == OUTCOME_NO_GO
        assert _read_report(out)["outcome"] == OUTCOME_NO_GO
        assert (out / "level-04").is_dir()
        assert (out / "level-06").is_dir()

    def test_cli_execute_cancelled_nonzero(self, tmp_path, monkeypatch):
        import anyhydro.capytaine_backend as cb_module
        captured: dict = {}
        monkeypatch.setattr(cb_module, "CapytaineBackend",
                            _mock_backend_factory(captured))
        out = tmp_path / "study"
        assert main(["prepare", "--out", str(out)]) == CLI_EXIT_OK
        assert main(["execute", "--out", str(out), "--backend", "capytaine",
                     "--cancel-before-level", "4"]) == CLI_EXIT_CANCELLED
        status = _read_status(out)
        assert status["status"] == "cancelled"
        assert status["outcome"] == OUTCOME_CANCELLED
        # cancellation preserves the evidence: no solves ran, no level
        # artifacts exist, the report records the cancellation
        assert _read_report(out)["outcome"] == OUTCOME_CANCELLED
        assert not (out / "level-04").exists()

    def test_cli_execute_failed_nonzero(self, tmp_path, monkeypatch):
        import anyhydro.capytaine_backend as cb_module
        captured: dict = {}
        monkeypatch.setattr(
            cb_module, "CapytaineBackend",
            _mock_backend_factory(captured, variant="fail_on_solve"))
        out = tmp_path / "study"
        assert main(["prepare", "--out", str(out)]) == CLI_EXIT_OK
        assert main(["execute", "--out", str(out),
                     "--backend", "capytaine"]) == CLI_EXIT_FAILED
        assert _read_status(out)["status"] == "failed"
        # prior level evidence is retained
        assert (out / "level-04" / "report.json").is_file()

    def test_cli_adjudicate_no_go_nonzero(self, tmp_path, monkeypatch):
        import anyhydro.capytaine_backend as cb_module
        captured: dict = {}
        monkeypatch.setattr(
            cb_module, "CapytaineBackend",
            _mock_backend_factory(captured, variant="psd_fail"))
        out = tmp_path / "study"
        main(["prepare", "--out", str(out)])
        main(["execute", "--out", str(out), "--backend", "capytaine"])
        assert main(["adjudicate", "--out", str(out)]) == CLI_EXIT_NO_GO

    def test_cli_adjudicate_incomplete_study_nonzero(self, tmp_path,
                                                    monkeypatch):
        # a failed (partial) study cannot be adjudicated to success: the
        # typed refusal is a nonzero exit, never a green gate
        import anyhydro.capytaine_backend as cb_module
        captured: dict = {}
        monkeypatch.setattr(
            cb_module, "CapytaineBackend",
            _mock_backend_factory(captured, variant="fail_on_solve"))
        out = tmp_path / "study"
        main(["prepare", "--out", str(out)])
        assert main(["execute", "--out", str(out),
                     "--backend", "capytaine"]) == CLI_EXIT_FAILED
        assert main(["adjudicate", "--out", str(out)]) == CLI_EXIT_ERROR
        assert _read_status(out)["status"] == "failed"
