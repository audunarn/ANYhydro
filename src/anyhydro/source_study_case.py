"""Source-study case record and refinement metrics (development tooling).

Immutable closed-world case record for the first-slice source-quality
mesh study, the prospective development example, the study inputs built
through the public :mod:`anyhydro.first_slice` fixture API, and the
cross-level refinement metric arithmetic. DEVELOPMENT TOOLING ONLY:
``qualification_scope`` is always
``"development_source_study_only"``; nothing here confers scientific
qualification. See :mod:`anyhydro.source_study` for the workflow.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from .dof import parse_dof_key
from .errors import HydroError
from .first_slice import (
    FIRST_SLICE_MAX_SUBDIVISIONS,
    FIRST_SLICE_MIN_SUBDIVISIONS,
    FIRST_SLICE_PROJECTED_DOFS,
    FIRST_SLICE_RHO,
    FIRST_SLICE_GRAVITY,
    FIRST_SLICE_WATER_DEPTH,
    FIRST_SLICE_FREE_SURFACE_LID_POLICY,
    FIRST_SLICE_SUBMERGED_LID_POLICY,
    make_first_slice_bodies,
    verify_first_slice_fixture,
)
from .operators import (
    CORRECTION_POLICIES,
    DerivedOperatorCriteria,
    DerivedOperatorSettings,
    DerivedHydroOperators,
    NormalizationSpec,
    parse_dof_key,
)
from .records import HydroRequest

__all__ = [
    "SOURCE_STUDY_FORMAT",
    "SOURCE_STUDY_MODULE_VERSION",
    "DEVELOPMENT_QUALIFICATION_SCOPE",
    "FIRST_SLICE_GEOMETRY_RECIPE",
    "FIRST_SLICE_SUBMERGED_LID_GEOMETRY_RECIPE",
    "study_lid_policy",
    "SOURCE_STUDY_QUANTITIES",
    "OUTCOME_PREPARED_NOT_EXECUTED",
    "OUTCOME_SOURCE_ADMISSION_PASSED_DEVELOPMENT_ONLY",
    "OUTCOME_NO_GO",
    "OUTCOME_FAILED",
    "OUTCOME_CANCELLED",
    "SourceStudyError",
    "SourceStudyCaseError",
    "SourceStudyWorkflowError",
    "SourceStudyIdentityError",
    "QuantityRefinementCriteria",
    "RefinementCriteria",
    "SourceStudyCase",
    "make_prospective_source_study_case",
    "study_operator_settings",
    "build_study_bodies",
    "build_study_request",
    "compare_level_refinement",
    "artifact_level_arrays",
    "crossbody_coupling_report",
]

#: Honest v2 case format: the closed-world case schema gained the
#: required hash-covered ``admission_declaration`` field. The v2
#: decoder explicitly does NOT support v1 records (typed refusal, no
#: automatic migration or restamping); existing v1 prepared/mocked
#: evidence on disk is left unchanged and is simply not decodable by
#: the v2 tooling. Nothing about v2 confers qualification.
SOURCE_STUDY_FORMAT = "anyhydro-source-study-v2"
SOURCE_STUDY_MODULE_VERSION = "0.1.0"
DEVELOPMENT_QUALIFICATION_SCOPE = "development_source_study_only"

#: Semantic identity of the physical geometry recipe (the fixed first-slice
#: two upright-box wetted-hull/downward-lid case). This is deliberately
#: separate from any discrete mesh identity: cross-level comparison is
#: justified by this shared recipe, never inferred from bounds.
FIRST_SLICE_GEOMETRY_RECIPE = "first-slice-two-box-wetted-hull-lid-v1"
FIRST_SLICE_SUBMERGED_LID_GEOMETRY_RECIPE = "first-slice-two-box-wetted-hull-submerged-lid-0p1m-v1"

def study_lid_policy(case):
    return (FIRST_SLICE_SUBMERGED_LID_POLICY
            if case.geometry_recipe == FIRST_SLICE_SUBMERGED_LID_GEOMETRY_RECIPE
            else FIRST_SLICE_FREE_SURFACE_LID_POLICY)

#: The refinement-compared quantities.
SOURCE_STUDY_QUANTITIES: Tuple[str, ...] = (
    "added_mass",
    "damping",
    "added_mass_infinite",
    "hydrostatic_stiffness",
    "excitation_force",
)

OUTCOME_PREPARED_NOT_EXECUTED = "prepared_not_executed"
OUTCOME_SOURCE_ADMISSION_PASSED_DEVELOPMENT_ONLY = (
    "source_admission_passed_development_only"
)
OUTCOME_NO_GO = "no_go"
OUTCOME_FAILED = "failed"
OUTCOME_CANCELLED = "cancelled"

_CASE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_MAX_FINITE_OMEGAS = 513
_MAX_WAVE_DIRECTIONS = 513
_MAX_LEVELS = 8


class SourceStudyError(HydroError):
    """Base error of the source-study development tooling."""


class SourceStudyCaseError(SourceStudyError):
    """The case record is invalid, not closed-world or not admissible."""


class SourceStudyWorkflowError(SourceStudyError):
    """The workflow refused an operation (state, paths, partial records)."""


class SourceStudyIdentityError(SourceStudyError):
    """Identity/controls mismatch: version, caps, grids or conventions."""


def _require(condition: bool, error: Exception) -> None:
    if not condition:
        raise error


def _canonical_json(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _finite_float(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise SourceStudyCaseError(f"{name} must be a float, not a bool")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise SourceStudyCaseError(f"{name} must be a float: {exc}") from exc
    if not np.isfinite(out):
        raise SourceStudyCaseError(f"{name} must be finite (got {out!r})")
    if positive and out <= 0.0:
        raise SourceStudyCaseError(f"{name} must be > 0 (got {out!r})")
    return out


def _int_control(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SourceStudyCaseError(
            f"{name} must be an integer (got {type(value).__name__}: {value!r})"
        )
    return int(value)


# --------------------------------------------------------------------------- #
# case record: closed world, immutable, hash-bound
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class QuantityRefinementCriteria:
    """Per-quantity normalized refinement criteria (all fields required).

    ``abs_tol`` and ``floor`` live in this quantity's OWN normalized unit
    (the normalized A, B, A_inf, hydrostatic or excitation unit under the
    case's declared normalization); there is never any cross-unit mixing.
    The pass rule against the finest reference is

        residual <= abs_tol + rel_tol * max(reference_scale, floor)

    where ``reference_scale`` is the largest normalized magnitude of the
    reference quantity (its own unit) and ``floor`` is the explicit
    near-zero floor.
    """

    abs_tol: float
    rel_tol: float
    floor: float
    unit: str

    def __post_init__(self) -> None:
        for name in ("abs_tol", "rel_tol"):
            object.__setattr__(
                self, name,
                _finite_float(getattr(self, name), f"refinement {name}"),
            )
            _require(getattr(self, name) >= 0.0,
                     SourceStudyCaseError(f"refinement {name} must be >= 0"))
        object.__setattr__(
            self, "floor",
            _finite_float(self.floor, "refinement floor", positive=True),
        )
        _require(
            isinstance(self.unit, str) and self.unit.strip(),
            SourceStudyCaseError("refinement unit must be a non-empty string"),
        )
        object.__setattr__(self, "unit", str(self.unit))

    def to_dict(self) -> dict:
        return {
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
            "floor": self.floor,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "QuantityRefinementCriteria":
        if not isinstance(data, Mapping):
            raise SourceStudyCaseError(
                "serialized quantity criteria must be a mapping")
        expected = {"abs_tol", "rel_tol", "floor", "unit"}
        unknown = set(data) - expected
        missing = expected - set(data)
        if unknown or missing:
            raise SourceStudyCaseError(
                f"serialized quantity criteria field mismatch (unknown: "
                f"{sorted(unknown)}, missing: {sorted(missing)}); refusing")
        return cls(**dict(data))


@dataclass(frozen=True)
class RefinementCriteria:
    """Caller-declared cross-level refinement criteria (no defaults).

    ``mesh_rel_tol`` is the declared study-level mesh criterion; each
    quantity's ``rel_tol`` may be at most this bound (tighter is allowed,
    looser is refused). The mesh criterion passes only when every
    quantity passes its own abs+relative+floor rule against the finest
    reference level.
    """

    mesh_rel_tol: float
    quantities: Mapping[str, QuantityRefinementCriteria]

    def __post_init__(self) -> None:
        mesh = _finite_float(self.mesh_rel_tol, "mesh_rel_tol")
        _require(mesh > 0.0,
                 SourceStudyCaseError("mesh_rel_tol must be > 0"))
        object.__setattr__(self, "mesh_rel_tol", mesh)
        _require(
            isinstance(self.quantities, Mapping),
            SourceStudyCaseError("refinement quantities must be a mapping"),
        )
        if set(self.quantities) != set(SOURCE_STUDY_QUANTITIES):
            raise SourceStudyCaseError(
                f"refinement quantities must be exactly "
                f"{list(SOURCE_STUDY_QUANTITIES)}, got "
                f"{sorted(self.quantities)}")
        frozen: Dict[str, QuantityRefinementCriteria] = {}
        for name in SOURCE_STUDY_QUANTITIES:
            item = self.quantities[name]
            if not isinstance(item, QuantityRefinementCriteria):
                raise SourceStudyCaseError(
                    f"refinement criteria for {name!r} must be a "
                    f"QuantityRefinementCriteria (got {type(item).__name__})")
            _require(
                item.rel_tol <= mesh,
                SourceStudyCaseError(
                    f"refinement rel_tol for {name!r} ({item.rel_tol}) is "
                    f"looser than the declared mesh criterion {mesh}"),
            )
            frozen[name] = item
        object.__setattr__(self, "quantities", dict(frozen))

    def to_dict(self) -> dict:
        return {
            "mesh_rel_tol": self.mesh_rel_tol,
            "quantities": {k: self.quantities[k].to_dict()
                           for k in SOURCE_STUDY_QUANTITIES},
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RefinementCriteria":
        if not isinstance(data, Mapping):
            raise SourceStudyCaseError(
                "serialized refinement criteria must be a mapping")
        expected = {"mesh_rel_tol", "quantities"}
        unknown = set(data) - expected
        missing = expected - set(data)
        if unknown or missing:
            raise SourceStudyCaseError(
                f"serialized refinement criteria field mismatch (unknown: "
                f"{sorted(unknown)}, missing: {sorted(missing)}); refusing")
        return cls(
            mesh_rel_tol=data["mesh_rel_tol"],
            quantities={
                k: QuantityRefinementCriteria.from_dict(v)
                for k, v in dict(data["quantities"]).items()
            },
        )


_CASE_SERIALIZED_KEYS = (
    "format",
    "case_id",
    "revision",
    "geometry_recipe",
    "levels",
    "omegas",
    "wave_directions",
    "include_zero_frequency",
    "include_infinite_frequency",
    "expected_backend_name",
    "expected_backend_version",
    "method",
    "method_declaration",
    "admission_declaration",
    "n_jobs",
    "n_threads",
    "retained_dofs",
    "normalization",
    "correction_policy",
    "operator_criteria",
    "refinement",
    "qualification_scope",
)


@dataclass(frozen=True, init=False)
class SourceStudyCase:
    """Immutable closed-world source-study case record.

    Every control is explicit and required; there are no permissive
    automatic defaults. The record is hash-bound (``content_hash`` over
    the canonical closed-world JSON) and accepts ONLY the fixed
    first-slice two-box geometry recipe — shape equivalence is never
    inferred from bounds. The retained DOF labels are exactly the
    first-slice projected surge/heave order.
    """

    case_id: str
    revision: str
    geometry_recipe: str
    levels: Tuple[int, ...]
    omegas: Tuple[float, ...]
    wave_directions: Tuple[float, ...]
    include_zero_frequency: bool
    include_infinite_frequency: bool
    expected_backend_name: str
    expected_backend_version: str
    method: Optional[str]
    method_declaration: str
    admission_declaration: str
    n_jobs: int
    n_threads: int
    retained_dofs: Tuple[str, ...]
    normalization: NormalizationSpec
    correction_policy: str
    operator_criteria: DerivedOperatorCriteria
    refinement: RefinementCriteria
    qualification_scope: str

    def __init__(self, *, case_id: str, revision: str, geometry_recipe: str,
                 levels, omegas, wave_directions,
                 include_zero_frequency: bool,
                 include_infinite_frequency: bool,
                 expected_backend_name: str, expected_backend_version: str,
                 method, method_declaration: str,
                 admission_declaration: str,
                 n_jobs: int, n_threads: int, retained_dofs,
                 normalization: NormalizationSpec,
                 correction_policy: str,
                 operator_criteria: DerivedOperatorCriteria,
                 refinement: RefinementCriteria, **unknown: Any) -> None:
        if unknown:
            raise SourceStudyCaseError(
                f"unknown source-study case field(s) {sorted(unknown)}; "
                "the case record is closed-world and refuses silent unknowns")
        for name, value in (("case_id", case_id), ("revision", revision),
                            ("geometry_recipe", geometry_recipe),
                            ("expected_backend_name", expected_backend_name),
                            ("expected_backend_version",
                             expected_backend_version),
                            ("method_declaration", method_declaration),
                            ("admission_declaration",
                             admission_declaration)):
            _require(isinstance(value, str) and value.strip(),
                     SourceStudyCaseError(f"{name} must be a non-empty string"))
            object.__setattr__(self, name, str(value))
        _require(bool(_CASE_ID_PATTERN.match(self.case_id)),
                 SourceStudyCaseError(
                     f"case_id {self.case_id!r} must match "
                     f"{_CASE_ID_PATTERN.pattern} (no path separators, no "
                     "traversal)"))
        _require(self.geometry_recipe in (FIRST_SLICE_GEOMETRY_RECIPE, FIRST_SLICE_SUBMERGED_LID_GEOMETRY_RECIPE),
                 SourceStudyCaseError(
                     f"geometry_recipe must be {FIRST_SLICE_GEOMETRY_RECIPE!r} "
                     f"(got {self.geometry_recipe!r}); this study accepts "
                     "only the fixed first-slice two-box case"))

        levels_tuple = tuple(_int_control(v, "level") for v in levels)
        _require(len(levels_tuple) >= 2,
                 SourceStudyCaseError(
                     "the exact level inventory needs at least two levels "
                     "(cross-level refinement compares against the finest)"))
        _require(len(set(levels_tuple)) == len(levels_tuple),
                 SourceStudyCaseError("levels must be unique"))
        _require(tuple(sorted(levels_tuple)) == levels_tuple,
                 SourceStudyCaseError("levels must be sorted ascending"))
        _require(len(levels_tuple) <= _MAX_LEVELS,
                 SourceStudyCaseError(
                     f"at most {_MAX_LEVELS} levels are supported"))
        for level in levels_tuple:
            _require(FIRST_SLICE_MIN_SUBDIVISIONS <= level
                     <= FIRST_SLICE_MAX_SUBDIVISIONS,
                     SourceStudyCaseError(
                         f"level {level} is outside the bounded "
                         f"subdivision range [{FIRST_SLICE_MIN_SUBDIVISIONS}, "
                         f"{FIRST_SLICE_MAX_SUBDIVISIONS}]"))
        object.__setattr__(self, "levels", levels_tuple)

        omegas_tuple = tuple(
            _finite_float(v, "omega", positive=True) for v in omegas)
        _require(1 <= len(omegas_tuple) <= _MAX_FINITE_OMEGAS,
                 SourceStudyCaseError(
                     f"the exact finite omega grid needs 1..{_MAX_FINITE_OMEGAS} "
                     "entries"))
        _require(all(omegas_tuple[i] < omegas_tuple[i + 1]
                     for i in range(len(omegas_tuple) - 1)),
                 SourceStudyCaseError("omegas must be strictly increasing"))
        object.__setattr__(self, "omegas", omegas_tuple)
        if study_lid_policy(self) == FIRST_SLICE_SUBMERGED_LID_POLICY:
            _require(max(self.omegas) <= 8.0, SourceStudyCaseError(
                "submerged_0p1m_v1 fixed-box policy admits only bands up to 8 rad/s"))

        dirs_tuple = tuple(
            _finite_float(v, "wave direction") for v in wave_directions)
        _require(1 <= len(dirs_tuple) <= _MAX_WAVE_DIRECTIONS,
                 SourceStudyCaseError(
                     "the exact heading grid needs at least one entry"))
        object.__setattr__(self, "wave_directions", dirs_tuple)

        _require(include_zero_frequency is False,
                 SourceStudyCaseError(
                     "include_zero_frequency must be False: the zero-frequency "
                     "limit is NOT computed in this study (the canonical "
                     "schema stores strictly positive finite frequencies)"))
        _require(include_infinite_frequency is True,
                 SourceStudyCaseError(
                     "include_infinite_frequency must be True: the study "
                     "requires an explicit A_inf source"))
        object.__setattr__(self, "include_zero_frequency", False)
        object.__setattr__(self, "include_infinite_frequency", True)

        if method is not None:
            _require(isinstance(method, str) and method.strip(),
                     SourceStudyCaseError(
                         "method must be None (the declared default) or a "
                         "non-empty string"))
            object.__setattr__(self, "method", str(method))
        else:
            object.__setattr__(self, "method", None)

        for name in ("n_jobs", "n_threads"):
            value = _int_control(n_jobs if name == "n_jobs" else n_threads,
                                 name)
            _require(value == 1,
                     SourceStudyCaseError(
                         f"{name} must be 1 (one job, one thread; no silent "
                         "resource overrides)"))
            object.__setattr__(self, name, value)

        retained = tuple(str(v) for v in retained_dofs)
        _require(retained == tuple(FIRST_SLICE_PROJECTED_DOFS),
                 SourceStudyCaseError(
                     "retained_dofs must be exactly the first-slice "
                     f"projected order {list(FIRST_SLICE_PROJECTED_DOFS)}"))
        object.__setattr__(self, "retained_dofs", retained)

        _require(isinstance(normalization, NormalizationSpec),
                 SourceStudyCaseError(
                     "normalization must be a NormalizationSpec instance"))
        _require(len(normalization.scales) == len(retained),
                 SourceStudyCaseError(
                     "normalization scales must have one entry per retained "
                     "DOF"))
        object.__setattr__(self, "normalization", normalization)

        _require(correction_policy in CORRECTION_POLICIES,
                 SourceStudyCaseError(
                     f"correction_policy must explicitly be one of "
                     f"{CORRECTION_POLICIES}, got {correction_policy!r}"))
        object.__setattr__(self, "correction_policy", str(correction_policy))

        _require(isinstance(operator_criteria, DerivedOperatorCriteria),
                 SourceStudyCaseError(
                     "operator_criteria must be a DerivedOperatorCriteria"))
        object.__setattr__(self, "operator_criteria", operator_criteria)
        _require(isinstance(refinement, RefinementCriteria),
                 SourceStudyCaseError(
                     "refinement must be a RefinementCriteria instance"))
        object.__setattr__(self, "refinement", refinement)
        object.__setattr__(self, "qualification_scope",
                           DEVELOPMENT_QUALIFICATION_SCOPE)

    # -- inventory ------------------------------------------------------- #

    def level_panel_inventory(self) -> Dict[int, int]:
        """Exact total panel count (both bodies, hull + lid) per level."""
        return {int(n): 24 * int(n) * int(n) for n in self.levels}

    def max_total_panels(self) -> int:
        return max(self.level_panel_inventory().values())

    def to_dict(self) -> dict:
        return {
            "format": SOURCE_STUDY_FORMAT,
            "case_id": self.case_id,
            "revision": self.revision,
            "geometry_recipe": self.geometry_recipe,
            "levels": [int(v) for v in self.levels],
            "omegas": [float(v) for v in self.omegas],
            "wave_directions": [float(v) for v in self.wave_directions],
            "include_zero_frequency": self.include_zero_frequency,
            "include_infinite_frequency": self.include_infinite_frequency,
            "expected_backend_name": self.expected_backend_name,
            "expected_backend_version": self.expected_backend_version,
            "method": self.method,
            "method_declaration": self.method_declaration,
            "admission_declaration": self.admission_declaration,
            "n_jobs": self.n_jobs,
            "n_threads": self.n_threads,
            "retained_dofs": list(self.retained_dofs),
            "normalization": self.normalization.to_dict(),
            "correction_policy": self.correction_policy,
            "operator_criteria": self.operator_criteria.to_dict(),
            "refinement": self.refinement.to_dict(),
            "qualification_scope": self.qualification_scope,
        }

    def content_hash(self) -> str:
        h = hashlib.sha256()
        # honest v2 hash prefix: the admission_declaration extension
        # changed the closed-world case schema, so the content hash is
        # namespaced at v2 and can never collide with a v1 record
        h.update(b"anyhydro-source-study-case-v2")
        payload = _canonical_json(self.to_dict()).encode("utf-8")
        h.update(len(payload).to_bytes(8, "little"))
        h.update(payload)
        return h.hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SourceStudyCase":
        if not isinstance(data, Mapping):
            raise SourceStudyCaseError(
                f"serialized case must be a mapping (got {type(data).__name__})")
        expected = set(_CASE_SERIALIZED_KEYS)
        unknown = set(data) - expected
        missing = expected - set(data)
        if unknown or missing:
            raise SourceStudyCaseError(
                f"serialized case field mismatch (unknown: {sorted(unknown)}, "
                f"missing: {sorted(missing)}); the case record is closed-world")
        if data["format"] != SOURCE_STUDY_FORMAT:
            if data["format"] == "anyhydro-source-study-v1":
                raise SourceStudyCaseError(
                    "serialized case format 'anyhydro-source-study-v1' is "
                    "EXPLICITLY UNSUPPORTED by the v2 decoder (the required "
                    "hash-covered admission_declaration changed the "
                    "closed-world case schema); there is no automatic "
                    "migration or restamping — rebuild the case record "
                    "explicitly at v2")
            raise SourceStudyCaseError(
                f"serialized case format must be {SOURCE_STUDY_FORMAT!r} "
                f"(got {data['format']!r})")
        _require(data["qualification_scope"] == DEVELOPMENT_QUALIFICATION_SCOPE,
                 SourceStudyCaseError(
                     "qualification_scope must be "
                     f"{DEVELOPMENT_QUALIFICATION_SCOPE!r}; development "
                     "source studies never confer scientific qualification"))
        return cls(
            case_id=data["case_id"],
            revision=data["revision"],
            geometry_recipe=data["geometry_recipe"],
            levels=data["levels"],
            omegas=data["omegas"],
            wave_directions=data["wave_directions"],
            include_zero_frequency=data["include_zero_frequency"],
            include_infinite_frequency=data["include_infinite_frequency"],
            expected_backend_name=data["expected_backend_name"],
            expected_backend_version=data["expected_backend_version"],
            method=data["method"],
            method_declaration=data["method_declaration"],
            admission_declaration=data["admission_declaration"],
            n_jobs=data["n_jobs"],
            n_threads=data["n_threads"],
            retained_dofs=data["retained_dofs"],
            normalization=NormalizationSpec.from_dict(data["normalization"]),
            correction_policy=data["correction_policy"],
            operator_criteria=DerivedOperatorCriteria.from_dict(
                data["operator_criteria"]),
            refinement=RefinementCriteria.from_dict(data["refinement"]),
        )


def make_prospective_source_study_case() -> SourceStudyCase:
    """The concrete PROSPECTIVE development case (parent-owned candidate).

    Revision ``prospective-candidate-2026-10-08-v2``: the unit
    declarations and work conjugacy are stated explicitly and
    correctly (q = S q_hat with S = L = 1 m makes q_hat DIMENSIONLESS;
    Q_hat = S^T Q carries N*m = J; A_hat/A_inf_hat kg*m^2; B_hat
    kg*m^2/s; H_hat N*m = J; the stored excitation is the force for a
    ONE-METRE reference wave, multiplied physically by the
    dimensionless eta/(1 m), so F_1m_hat has N*m = J — equivalently
    the per-metre-wave transfer has N). The numerical thresholds,
    arrays, S = [1, 1, 1, 1] and all criteria values are UNCHANGED from
    v1; only the declared units/conventions text and the hash-covered
    admission declaration evolve at v2 (adopted BEFORE any actual run;
    none has run).

    Levels 4 and 6 (384 and 864 total panels), omegas [2, 4, 6] rad/s,
    heading 0, explicit A_inf, zero-frequency NOT computed, translation
    length 1 m normalization S=[1,1,1,1], explicit reciprocal-average
    correction. Proposed admission criteria: reciprocity 1e-8 abs / 0.02
    rel / 1e-8 floor, correction 1e-8 abs / 0.01 rel / 1e-8 floor,
    derived symmetry 1e-10, damping PSD 1e-10 (each abs/floor in its own
    normalized A/A_inf or B unit, never a cross-unit norm); mesh
    criterion 0.02 with per-quantity near-zero floors 1e-9 in each
    quantity's own normalized unit. The mesh norm rule is a GLOBAL-band
    linf relative control: each sample's max-entry complex-absolute
    residual is compared against ``abs_tol + rel_tol * max(finest
    quantity's GLOBAL registered-band max-entry magnitude, floor)`` —
    never a relative percentage of each local sample. These values are
    a prospective candidate frozen BEFORE any future BEM; they are not
    authority to loosen scientific criteria after results, and this
    three-frequency two-level grid is NOT claimed adequate for
    radiation/RAO qualification — it only discriminates the old
    coarse-pilot PSD/reciprocity mesh hypothesis. Broader
    kernel/tail/time/dynamic qualification remains a distinct
    registered gate.
    """
    quantities = {
        "added_mass": QuantityRefinementCriteria(
            abs_tol=1e-9, rel_tol=0.02, floor=1e-9,
            unit="kg*m^2 (A_hat = S A S, S = 1 m translation length)"),
        "damping": QuantityRefinementCriteria(
            abs_tol=1e-9, rel_tol=0.02, floor=1e-9,
            unit="kg*m^2/s (B_hat = S B S)"),
        "added_mass_infinite": QuantityRefinementCriteria(
            abs_tol=1e-9, rel_tol=0.02, floor=1e-9,
            unit="kg*m^2 (A_inf_hat = S A_inf S)"),
        "hydrostatic_stiffness": QuantityRefinementCriteria(
            abs_tol=1e-9, rel_tol=0.02, floor=1e-9,
            unit="N*m = J (H_hat = S H S)"),
        "excitation_force": QuantityRefinementCriteria(
            abs_tol=1e-9, rel_tol=0.02, floor=1e-9,
            unit=("N*m = J (F_1m_hat = S^T F_1m for the one-metre "
                  "reference wave; the per-metre-wave transfer has N)")),
    }
    return SourceStudyCase(
        case_id="first-slice-source-quality-mesh-study",
        revision="prospective-candidate-2026-10-08-v2",
        geometry_recipe=FIRST_SLICE_GEOMETRY_RECIPE,
        levels=(4, 6),
        omegas=(2.0, 4.0, 6.0),
        wave_directions=(0.0,),
        include_zero_frequency=False,
        include_infinite_frequency=True,
        expected_backend_name="capytaine",
        expected_backend_version="3.0.0",
        method=None,
        method_declaration=(
            "unset method selects the pinned capytaine 3.0.0 default "
            "solver configuration (BEMSolver default matrix engine); no "
            "method override is declared"),
        admission_declaration=(
            "source admission requires the ACTUAL raw dataset attrs "
            "backend/backend_version to match the case-expected backend "
            "name/version (capabilities alone are not proof), and the "
            "actual ordered rotation_center/center_of_mass/"
            "center_of_buoyancy data variables to match the fixed "
            "first-slice fixture — reference centers (-2,0,0) m and "
            "(2,0,0) m, COM and COB each (cx, 0, -0.25) m, body and DOF "
            "axes in the fixed order — compared at atol 1e-12 m rtol 0, "
            "a prospective fixed-fixture ADMISSION numerical tolerance, "
            "not a meshing tolerance; metadata attrs alone cannot prove "
            "the actual arrays and no coordinate mutation or rounding is "
            "performed"),
        n_jobs=1,
        n_threads=1,
        retained_dofs=FIRST_SLICE_PROJECTED_DOFS,
        normalization=NormalizationSpec(
            name="translation-length-1m",
            unit_convention=(
                "q = S q_hat with S = diag(1, 1, 1, 1) m (declared "
                "translation length L = 1 m): q_hat is DIMENSIONLESS "
                "and Q_hat = S^T Q carries N*m = J; A_hat = S A S and "
                "A_inf_hat = S A_inf S carry kg*m^2; B_hat = S B S "
                "carries kg*m^2/s; H_hat = S H S carries N*m = J. The "
                "stored excitation is the force for a ONE-METRE "
                "reference wave: the physical excitation multiplies it "
                "by the dimensionless ratio eta/(1 m), so F_1m_hat = "
                "S^T F_1m carries N*m = J (equivalently the "
                "per-metre-wave transfer carries N). Mesh refinement "
                "semantics: per-sample max-entry complex-absolute "
                "residual compared against abs_tol + rel_tol * "
                "max(finest quantity's GLOBAL registered-band max-entry "
                "magnitude, floor) — a GLOBAL-band linf relative "
                "control, never a relative percentage of each local "
                "sample"),
            scales=(1.0, 1.0, 1.0, 1.0),
        ),
        correction_policy="reciprocal_average",
        operator_criteria=DerivedOperatorCriteria(
            reciprocity_abs_tol=1e-8,
            reciprocity_rel_tol=0.02,
            reciprocity_floor=1e-8,
            correction_abs_tol=1e-8,
            correction_rel_tol=0.01,
            correction_floor=1e-8,
            derived_symmetry_tol=1e-10,
            damping_psd_tol=1e-10,
        ),
        refinement=RefinementCriteria(mesh_rel_tol=0.02, quantities=quantities),
    )


# --------------------------------------------------------------------------- #
# study inputs through the public fixture API
# --------------------------------------------------------------------------- #


def study_operator_settings(case: SourceStudyCase) -> DerivedOperatorSettings:
    """The explicit public derived-operator settings of this case."""
    _require(isinstance(case, SourceStudyCase),
             SourceStudyCaseError("case must be a SourceStudyCase"))
    return DerivedOperatorSettings(
        correction_policy=case.correction_policy,
        retained_dofs=case.retained_dofs,
        normalization=case.normalization,
        criteria=case.operator_criteria,
    )


def build_study_bodies(case: SourceStudyCase, level: int):
    """Deterministic verified bodies for one registered level (public API)."""
    _require(isinstance(case, SourceStudyCase),
             SourceStudyCaseError("case must be a SourceStudyCase"))
    _require(_int_control(level, "level") in case.levels,
             SourceStudyWorkflowError(
                 f"level {level} is not registered in the case inventory "
                 f"{list(case.levels)}; no new levels are invented"))
    bodies = make_first_slice_bodies(subdivisions=int(level), lid_policy=study_lid_policy(case))
    verify_first_slice_fixture(bodies, subdivisions=int(level), lid_policy=study_lid_policy(case))
    return bodies


def build_study_request(case: SourceStudyCase) -> HydroRequest:
    """The exact case request (grids, fluid, explicit availability)."""
    _require(isinstance(case, SourceStudyCase),
             SourceStudyCaseError("case must be a SourceStudyCase"))
    settings: Dict[str, Any] = {}
    if study_lid_policy(case) == FIRST_SLICE_SUBMERGED_LID_POLICY:
        if max(case.omegas) > 8.0:
            raise SourceStudyCaseError("submerged_0p1m_v1 fixed-box policy admits only bands up to 8 rad/s")
        settings["auxiliary_lid_policy"] = FIRST_SLICE_SUBMERGED_LID_POLICY
    if case.method is not None:
        settings["method"] = case.method
    return HydroRequest(
        omegas=np.array(case.omegas, dtype=np.float64),
        wave_directions=np.array(case.wave_directions, dtype=np.float64),
        water_depth=FIRST_SLICE_WATER_DEPTH,
        rho=FIRST_SLICE_RHO,
        gravity=FIRST_SLICE_GRAVITY,
        include_zero_frequency=case.include_zero_frequency,
        include_infinite_frequency=case.include_infinite_frequency,
        backend_settings=settings,
    )


# --------------------------------------------------------------------------- #
# refinement metric arithmetic (study-tool responsibility)
# --------------------------------------------------------------------------- #


def _normalized_matrix(X: np.ndarray, scales: np.ndarray) -> np.ndarray:
    if X.ndim == 2:
        return scales[:, None] * X * scales[None, :]
    return scales[None, :, None] * X * scales[None, None, :]


def _normalized_vector(F: np.ndarray, scales: np.ndarray) -> np.ndarray:
    # Q_hat = S^T Q for the excitation phasors (per frequency/heading).
    if F.ndim == 2:
        return scales[None, :] * F
    return scales[None, None, :] * F


def compare_level_refinement(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    criteria: RefinementCriteria,
    scales,
) -> dict:
    """Complete normalized residuals of one level against the finest.

    ``reference``/``candidate`` map each quantity name to its RETAINED
    array exactly as exposed by the public derived-operator artifact
    (``raw_added_mass``/``raw_damping`` per frequency, A_inf and
    hydrostatics matrices, complex ``excitation_force`` per
    frequency/heading). Residuals are computed in the case's declared
    normalized coordinates; each quantity's threshold combines its own
    absolute term, relative term and explicit near-zero floor in its own
    normalized unit. Complete per-frequency/heading residuals are
    returned, never only counts.
    """
    _require(isinstance(criteria, RefinementCriteria),
             SourceStudyCaseError("criteria must be a RefinementCriteria"))
    s = np.asarray(scales, dtype=np.float64)
    _require(s.ndim == 1 and s.size > 0 and bool(np.all(np.isfinite(s)))
             and bool(np.all(s > 0.0)),
             SourceStudyCaseError("normalization scales must be positive "
                                  "and finite"))
    report: Dict[str, Any] = {"quantities": {}, "mesh_pass": True}
    for name in SOURCE_STUDY_QUANTITIES:
        _require(name in reference and name in candidate,
                 SourceStudyWorkflowError(
                     f"refinement quantity {name!r} is missing from the "
                     "level arrays; refusing to compare a partial record"))
        ref = np.asarray(reference[name])
        cand = np.asarray(candidate[name])
        _require(ref.shape == cand.shape,
                 SourceStudyWorkflowError(
                     f"refinement quantity {name!r} shape mismatch "
                     f"{ref.shape} vs {cand.shape}; only matched grids are "
                     "compared"))
        for label, arr in (("reference", ref), ("candidate", cand)):
            _require(bool(np.all(np.isfinite(np.asarray(arr.real)))),
                     SourceStudyWorkflowError(
                         f"{label} quantity {name!r} real part is not finite"))
            if np.iscomplexobj(arr):
                _require(bool(np.all(np.isfinite(np.asarray(arr.imag)))),
                         SourceStudyWorkflowError(
                             f"{label} quantity {name!r} imaginary part is "
                             "not finite"))
        if name == "excitation_force":
            ref_hat = _normalized_vector(ref, s)
            cand_hat = _normalized_vector(cand, s)
        else:
            ref_hat = _normalized_matrix(ref, s)
            cand_hat = _normalized_matrix(cand, s)
        diff = np.abs(ref_hat - cand_hat)
        if ref_hat.ndim == 3 and name != "excitation_force":
            # (omega, n, n) matrices: residual per frequency (max norm).
            residuals = [
                float(np.max(diff[i])) if diff[i].size else 0.0
                for i in range(diff.shape[0])
            ]
        elif ref_hat.ndim == 3:
            # (omega, dir, n) phasors: residual per frequency/heading.
            residuals = [
                float(np.max(diff[i, j])) if diff[i, j].size else 0.0
                for i in range(diff.shape[0])
                for j in range(diff.shape[1])
            ]
        else:
            residuals = [float(np.max(diff))] if diff.size else [0.0]
        scale = float(np.max(np.abs(ref_hat))) if ref_hat.size else 0.0
        qc = criteria.quantities[name]
        threshold = qc.abs_tol + qc.rel_tol * max(scale, qc.floor)
        passed = bool(all(r <= threshold for r in residuals))
        report["quantities"][name] = {
            "residuals": residuals,
            "reference_scale": scale,
            "threshold": threshold,
            "abs_tol": qc.abs_tol,
            "rel_tol": qc.rel_tol,
            "floor": qc.floor,
            "unit": qc.unit,
            "pass": passed,
        }
        if not passed:
            report["mesh_pass"] = False
    return report


def artifact_level_arrays(artifact: DerivedHydroOperators) -> dict:
    """The public retained arrays of a derived-operator artifact."""
    return {
        "added_mass": artifact.raw_added_mass,
        "damping": artifact.raw_damping,
        "added_mass_infinite": artifact.raw_added_mass_infinite,
        "hydrostatic_stiffness": artifact.hydrostatic_stiffness,
        "excitation_force": artifact.excitation_force,
    }


def crossbody_coupling_report(artifact: DerivedHydroOperators,
                              case: SourceStudyCase) -> dict:
    """Truthful retained cross-body coupling observability of one level.

    Study-tool metric over the PUBLIC artifact arrays only (no cloned
    operator arithmetic or proof): the retained A/B per finite frequency
    and A_inf are read exactly as stored, normalized by S-congruence
    with the case's fixed normalization (``A_hat = S A S``, never a
    mixed-SI max norm), and the off-body block max-norms are reported
    per frequency for every ordered body pair. The observability
    predicate is evaluated against the case's DECLARED per-quantity
    near-zero floors (no new tolerances): the fixed two-body physical
    case requires a non-negligible off-body block in at least one
    finite-frequency A or B matrix; near-zero entries at a particular
    omega are expected and never demanded per-entry or per-frequency.
    The predicate is stored separately from the artifact's own
    ``admitted`` verdict and never relabels the public artifact's truth.
    """
    _require(isinstance(artifact, DerivedHydroOperators),
             SourceStudyCaseError("artifact must be a DerivedHydroOperators"))
    _require(isinstance(case, SourceStudyCase),
             SourceStudyCaseError("case must be a SourceStudyCase"))
    retained = [str(k) for k in artifact.settings.retained_dofs]
    _require(tuple(retained) == tuple(case.retained_dofs),
             SourceStudyWorkflowError(
                 "coupling observability requires the artifact's retained "
                 "DOFs to match the case record"))
    groups: Dict[str, List[int]] = {}
    for index, label in enumerate(retained):
        body_id, _ = parse_dof_key(label)
        groups.setdefault(body_id, []).append(index)
    _require(len(groups) == 2 and all(len(g) >= 1 for g in groups.values()),
             SourceStudyWorkflowError(
                 "coupling observability is defined for the fixed "
                 "two-body case; the retained DOFs must span exactly two "
                 f"bodies (got {sorted(groups)})"))
    s = np.asarray(case.normalization.scales, dtype=np.float64)
    pairs = [(a, b) for a in groups for b in groups if a != b]
    floors = {
        "added_mass": case.refinement.quantities["added_mass"].floor,
        "damping": case.refinement.quantities["damping"].floor,
        "added_mass_infinite":
            case.refinement.quantities["added_mass_infinite"].floor,
    }
    blocks: Dict[str, Dict[str, list]] = {
        "added_mass": {}, "damping": {}, "added_mass_infinite": {},
    }
    for a, b in pairs:
        rows = np.asarray(groups[a], dtype=int)
        cols = np.asarray(groups[b], dtype=int)
        a_hat = _normalized_matrix(np.asarray(artifact.raw_added_mass), s)
        b_hat = _normalized_matrix(np.asarray(artifact.raw_damping), s)
        ainf_hat = _normalized_matrix(
            np.asarray(artifact.raw_added_mass_infinite), s)
        label = f"{a}::{b}"
        blocks["added_mass"][label] = [
            float(np.max(np.abs(a_hat[i][np.ix_(rows, cols)])))
            if cols.size and rows.size else 0.0
            for i in range(a_hat.shape[0])
        ]
        blocks["damping"][label] = [
            float(np.max(np.abs(b_hat[i][np.ix_(rows, cols)])))
            if cols.size and rows.size else 0.0
            for i in range(b_hat.shape[0])
        ]
        blocks["added_mass_infinite"][label] = float(
            np.max(np.abs(ainf_hat[np.ix_(rows, cols)])))
    observable = bool(
        any(norm > floors["added_mass"]
            for norms in blocks["added_mass"].values() for norm in norms)
        or any(norm > floors["damping"]
               for norms in blocks["damping"].values() for norm in norms))
    return {
        "bodies": sorted(groups),
        "omegas": [float(v) for v in np.asarray(artifact.omegas)],
        "blocks": blocks,
        "floors": floors,
        "predicate": (
            "at least one finite-frequency retained A or B off-body block "
            "max-norm exceeds its declared per-quantity normalized floor "
            "(A_inf off-body norms are reported as evidence; near-zero "
            "blocks at a particular omega do not fail the predicate)"),
        "observable": observable,
    }
