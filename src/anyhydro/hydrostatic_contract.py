"""Explicit immutable/validated hydrostatic contract transport.

ADR-004 (governing): mechanics owns dry gravity; the hydrodynamic source owns
buoyancy. All composition uses declared source force bases, baseline wrenches
and stiffness derivative semantics. This module transports those declarations
with a dataset; it does not compose anything.

Honesty rules enforced here:

- A missing contract is UNKNOWN. It is never silently treated as zero
  stiffness, a gravity-free source, or any particular role.
- No baseline wrench, displaced volume, CG/CB, source gravity or inertia is
  ever fabricated; only what the source actually declared is recorded.
- A declared-complete contract is transport metadata, NOT scientific
  qualification and NOT integrator admission.
- Unknown role/force-basis/derivative/frame semantics leave the affected
  future exact six-DOF composition UNADMITTED: the typed refusal in
  :func:`require_hydrostatic_composition` is the gate. No silent FULL
  subtraction or policy selection exists here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .errors import HydroError, RecordValidationError, SchemaError

HYDROSTATIC_CONTRACT_ATTR = "hydrostatic_contract"


class HydrostaticContractError(HydroError):
    """A hydrostatic source contract is missing, incomplete or mismatched.

    Unknown role/force-basis/derivative semantics are a typed refusal for
    the affected composition; they are never silently defaulted. Lives here
    rather than in ``errors`` so that module stays byte-identical to its
    baseline; it remains a :class:`HydroError` subclass.
    """

ROLE_UNKNOWN = "UNKNOWN"
ROLE_BUOYANCY_ONLY = "BUOYANCY_ONLY"
ROLE_FULL = "FULL"
HYDROSTATIC_ROLES = (ROLE_UNKNOWN, ROLE_BUOYANCY_ONLY, ROLE_FULL)

FORCE_BASIS_GENERALIZED_DOF = "GENERALIZED_DOF"
FORCE_BASIS_PHYSICAL_WRENCH = "PHYSICAL_WRENCH"
FORCE_BASIS_UNKNOWN = "UNKNOWN"
FORCE_BASES = (
    FORCE_BASIS_GENERALIZED_DOF,
    FORCE_BASIS_PHYSICAL_WRENCH,
    FORCE_BASIS_UNKNOWN,
)

STIFFNESS_DERIVATIVE_INCREMENTAL = "INCREMENTAL_ABOUT_EQUILIBRIUM"
STIFFNESS_DERIVATIVE_TOTAL = "TOTAL"
STIFFNESS_DERIVATIVE_UNKNOWN = "UNKNOWN"
STIFFNESS_DERIVATIVES = (
    STIFFNESS_DERIVATIVE_INCREMENTAL,
    STIFFNESS_DERIVATIVE_TOTAL,
    STIFFNESS_DERIVATIVE_UNKNOWN,
)

FULL_SOURCE_KEYS = ("mass", "center_of_gravity", "composition")

# Versioned immutable source binding: the contract is bound to the EXACT
# dataset content by a noncircular identity that excludes ONLY its own
# contract/binding payload (the ``hydrostatic_contract`` attribute). The
# binding declares target identity; it is not scientific qualification.
SOURCE_BINDING_VERSION = "hydrostatic_source_binding_v1"
SOURCE_BINDING_KEYS = ("binding_version", "dataset_content_hash")


def _deep_freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({k: _deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    return value


def _check_body_id(body_id) -> str:
    if not isinstance(body_id, str) or not body_id.strip():
        raise RecordValidationError("contract body ids must be non-empty strings")
    if "::" in body_id:
        raise RecordValidationError(
            f"contract body id {body_id!r} must not contain the DOF separator '::'"
        )
    return body_id


def _validate_wrench(wrench, body_id: str) -> tuple:
    try:
        arr = np.asarray(wrench, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise RecordValidationError(
            f"baseline wrench for body {body_id!r} must be a finite 6-vector "
            f"of numbers (got malformed input: {exc})"
        ) from exc
    if arr.shape != (6,) or not np.all(np.isfinite(arr)):
        raise RecordValidationError(
            f"baseline wrench for body {body_id!r} must be a finite 6-vector "
            "(force and moment components in the declared force basis)"
        )
    return tuple(float(v) for v in arr)


def _validate_source_binding(binding) -> Mapping[str, Any]:
    """Validate the versioned immutable source binding payload."""
    if not isinstance(binding, Mapping):
        raise RecordValidationError(
            "source_binding must be a JSON object with exactly "
            f"{list(SOURCE_BINDING_KEYS)} or null"
        )
    if set(binding) != set(SOURCE_BINDING_KEYS):
        raise RecordValidationError(
            f"source_binding must contain exactly {list(SOURCE_BINDING_KEYS)}, "
            f"got {sorted(binding)}"
        )
    if binding["binding_version"] != SOURCE_BINDING_VERSION:
        raise RecordValidationError(
            f"source_binding binding_version must be {SOURCE_BINDING_VERSION!r}, "
            f"got {binding['binding_version']!r}"
        )
    content_hash = binding["dataset_content_hash"]
    if (
        not isinstance(content_hash, str)
        or len(content_hash) != 64
        or any(c not in "0123456789abcdef" for c in content_hash)
    ):
        raise RecordValidationError(
            "source_binding dataset_content_hash must be a sha256 hex digest"
        )
    return MappingProxyType(
        {
            "binding_version": str(binding["binding_version"]),
            "dataset_content_hash": str(content_hash),
        }
    )


def _validate_full_source_mass_scalar(mass, where: str) -> float:
    if not isinstance(mass, (int, float)) or isinstance(mass, bool):
        raise RecordValidationError(f"{where} must be a float")
    mass = float(mass)
    if not np.isfinite(mass) or mass <= 0.0:
        raise RecordValidationError(
            f"{where} must be finite and > 0 (got {mass!r})"
        )
    return mass


def _validate_full_source_cg_vector(cg, where: str) -> Tuple[float, float, float]:
    ok_cg = (
        isinstance(cg, (list, tuple))
        and len(cg) == 3
        and all(
            isinstance(v, (int, float)) and not isinstance(v, bool) for v in cg
        )
        and all(np.isfinite(float(v)) for v in cg)
    )
    if not ok_cg:
        raise RecordValidationError(
            f"{where} must be a list of 3 finite floats"
        )
    return tuple(float(v) for v in cg)


@dataclass(frozen=True)
class HydrostaticContract:
    """Immutable, validated hydrostatic source contract.

    Fields
    - role: UNKNOWN, BUOYANCY_ONLY or FULL. BUOYANCY_ONLY composes with
      separately owned gravity; FULL additionally requires the exact source
      mass/CG/composition declaration in ``full_source``.
    - force_basis: declared semantics of the source force/stiffness rows.
      GENERALIZED_DOF rows are work-conjugate to the dataset DOF coordinates;
      PHYSICAL_WRENCH declares a physical wrench basis whose rate map is
      source-specific (ADR-004) and is never silently converted.
    - stiffness_derivative: whether the recorded hydrostatic stiffness is the
      incremental restoring derivative about equilibrium or a total
      derivative including baseline terms.
    - bodies: the body binding; must equal the dataset bodies when attached.
    - baseline_wrench: optional ACTUAL measured baseline wrench per body.
      Never fabricated; absent means not measured, not zero.
    - provenance: source identity (non-empty ``source`` string required).
    - full_source: required exactly for FULL: mass, center_of_gravity and a
      composition declaration. Rejected for other roles. Two declared shapes
      are transported: the legacy single-value shape (readable as declared
      incomplete metadata for multi-body contracts) and the per-body shape
      (mappings covering exactly the contract bodies). Neither shape admits
      composition in this slice.
    - source_binding: optional versioned immutable binding of this contract
      to the EXACT dataset content (noncircular identity excluding only the
      contract/binding payload itself). ``attach_hydrostatic_contract``
      binds; eligibility refuses unbound and stale/foreign bindings.
    """

    role: str
    force_basis: str
    stiffness_derivative: str
    bodies: Tuple[str, ...]
    baseline_wrench: Optional[Mapping[str, Tuple[float, ...]]]
    provenance: Mapping[str, Any]
    full_source: Optional[Mapping[str, Any]]
    source_binding: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if self.role not in HYDROSTATIC_ROLES:
            raise RecordValidationError(
                f"role must be one of {HYDROSTATIC_ROLES}, got {self.role!r}"
            )
        if self.force_basis not in FORCE_BASES:
            raise RecordValidationError(
                f"force_basis must be one of {FORCE_BASES}, got {self.force_basis!r}"
            )
        if self.stiffness_derivative not in STIFFNESS_DERIVATIVES:
            raise RecordValidationError(
                f"stiffness_derivative must be one of {STIFFNESS_DERIVATIVES}, "
                f"got {self.stiffness_derivative!r}"
            )

        bodies = tuple(_check_body_id(b) for b in self.bodies)
        if not bodies:
            raise RecordValidationError("contract bodies must not be empty")
        if len(set(bodies)) != len(bodies):
            raise RecordValidationError("contract bodies contains duplicates")
        object.__setattr__(self, "bodies", bodies)

        if self.baseline_wrench is not None:
            if not isinstance(self.baseline_wrench, Mapping):
                raise RecordValidationError(
                    "baseline_wrench must be a mapping of body_id -> 6-vector or None"
                )
            if set(self.baseline_wrench) != set(bodies):
                raise RecordValidationError(
                    "baseline_wrench must cover exactly the contract bodies "
                    f"{list(bodies)}, got {sorted(self.baseline_wrench)}"
                )
            frozen_wrench = MappingProxyType(
                {
                    body: _validate_wrench(self.baseline_wrench[body], body)
                    for body in bodies
                }
            )
            object.__setattr__(self, "baseline_wrench", frozen_wrench)

        if not isinstance(self.provenance, Mapping):
            raise RecordValidationError("provenance must be a JSON object")
        source = self.provenance.get("source")
        if not isinstance(source, str) or not source.strip():
            raise RecordValidationError(
                "provenance must contain a non-empty 'source' identity string"
            )
        try:
            json.dumps(dict(self.provenance), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise RecordValidationError(
                f"provenance must be JSON-serializable: {exc}"
            ) from exc
        object.__setattr__(self, "provenance", _deep_freeze(dict(self.provenance)))

        if self.role == ROLE_FULL:
            if not isinstance(self.full_source, Mapping):
                raise RecordValidationError(
                    "role FULL requires the exact source mass/CG/composition "
                    "declaration in full_source; an incomplete source is a "
                    "typed refusal, never a silent default"
                )
            if set(self.full_source) != set(FULL_SOURCE_KEYS):
                raise RecordValidationError(
                    f"full_source must contain exactly {list(FULL_SOURCE_KEYS)}, "
                    f"got {sorted(self.full_source)}"
                )
            mass = self.full_source["mass"]
            cg = self.full_source["center_of_gravity"]
            if isinstance(mass, Mapping) != isinstance(cg, Mapping):
                raise RecordValidationError(
                    "full_source mass and center_of_gravity must use the same "
                    "declared shape: either the single-value legacy shape or a "
                    "per-body mapping covering exactly the contract bodies"
                )
            composition = self.full_source["composition"]
            if not isinstance(composition, str) or not composition.strip():
                raise RecordValidationError(
                    "full_source composition must be a non-empty declaration string"
                )
            if not isinstance(mass, Mapping):
                # Legacy single-value shape: readable declared metadata. For a
                # multi-body contract it is incomplete per-body coverage and
                # never admits composition.
                frozen_full = MappingProxyType(
                    {
                        "mass": _validate_full_source_mass_scalar(
                            mass, "full_source mass"
                        ),
                        "center_of_gravity": _validate_full_source_cg_vector(
                            cg, "full_source center_of_gravity"
                        ),
                        "composition": str(composition),
                    }
                )
            else:
                # Per-body shape: exact coverage of the contract bodies.
                if set(mass) != set(bodies) or set(cg) != set(bodies):
                    raise RecordValidationError(
                        "full_source per-body mass and center_of_gravity must "
                        f"cover exactly the contract bodies {list(bodies)}, "
                        f"got mass {sorted(mass)} and "
                        f"center_of_gravity {sorted(cg)}"
                    )
                frozen_mass = {}
                for body in bodies:
                    frozen_mass[body] = _validate_full_source_mass_scalar(
                        mass[body], f"full_source mass for body {body!r}"
                    )
                frozen_cg = {}
                for body in bodies:
                    frozen_cg[body] = _validate_full_source_cg_vector(
                        cg[body], f"full_source center_of_gravity for body {body!r}"
                    )
                frozen_full = MappingProxyType(
                    {
                        "mass": MappingProxyType(frozen_mass),
                        "center_of_gravity": MappingProxyType(frozen_cg),
                        "composition": str(composition),
                    }
                )
            object.__setattr__(self, "full_source", frozen_full)
        else:
            if self.full_source is not None:
                raise RecordValidationError(
                    "full_source is only permitted for role FULL; a "
                    f"{self.role!r} source declaring mass/CG/composition is "
                    "contradictory and must be resolved by the source, not "
                    "recorded half-declared"
                )
            object.__setattr__(self, "full_source", None)

        if self.source_binding is not None:
            object.__setattr__(
                self,
                "source_binding",
                _validate_source_binding(self.source_binding),
            )

    def to_json(self) -> dict:
        if self.full_source is None:
            full_json = None
        else:
            mass = self.full_source["mass"]
            cg = self.full_source["center_of_gravity"]
            if isinstance(mass, Mapping):
                full_json = {
                    "mass": {b: mass[b] for b in self.bodies},
                    "center_of_gravity": {
                        b: list(cg[b]) for b in self.bodies
                    },
                    "composition": self.full_source["composition"],
                }
            else:
                full_json = {
                    "mass": mass,
                    "center_of_gravity": list(cg),
                    "composition": self.full_source["composition"],
                }
        return {
            "role": self.role,
            "force_basis": self.force_basis,
            "stiffness_derivative": self.stiffness_derivative,
            "bodies": list(self.bodies),
            "baseline_wrench": None
            if self.baseline_wrench is None
            else {b: list(self.baseline_wrench[b]) for b in self.bodies},
            "provenance": dict(self.provenance),
            "full_source": full_json,
            "source_binding": None
            if self.source_binding is None
            else dict(self.source_binding),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "HydrostaticContract":
        if not isinstance(data, Mapping):
            raise RecordValidationError(
                "hydrostatic contract must be a JSON object"
            )
        expected = {
            "role",
            "force_basis",
            "stiffness_derivative",
            "bodies",
            "baseline_wrench",
            "provenance",
            "full_source",
        }
        optional = {"source_binding"}
        unknown = set(data) - expected - optional
        missing = expected - set(data)
        if unknown or missing:
            raise RecordValidationError(
                f"hydrostatic contract must contain exactly {sorted(expected)} "
                f"plus optional {sorted(optional)}; missing {sorted(missing)}, "
                f"unknown {sorted(unknown)}"
            )
        # Container types are validated BEFORE any .items()/dict()/tuple()
        # traversal: malformed recognized payloads raise typed errors here,
        # never AttributeError/TypeError.
        bodies = data["bodies"]
        if not isinstance(bodies, (list, tuple)) or isinstance(
            bodies, (str, bytes)
        ):
            raise RecordValidationError(
                "contract bodies must be a list of body id strings"
            )
        baseline = data["baseline_wrench"]
        if baseline is not None:
            if not isinstance(baseline, Mapping):
                raise RecordValidationError(
                    "baseline_wrench must be a JSON object of body_id -> "
                    "6-vector or null"
                )
            frozen_baseline = {}
            for body, wrench in baseline.items():
                if not isinstance(body, str):
                    raise RecordValidationError(
                        "baseline_wrench keys must be body id strings"
                    )
                if not isinstance(wrench, (list, tuple)) or isinstance(
                    wrench, (str, bytes)
                ):
                    raise RecordValidationError(
                        f"baseline_wrench[{body!r}] must be a list of 6 "
                        "finite floats"
                    )
                frozen_baseline[body] = list(wrench)
            baseline = frozen_baseline
        provenance = data["provenance"]
        if not isinstance(provenance, Mapping):
            raise RecordValidationError("provenance must be a JSON object")
        full_source = data["full_source"]
        if full_source is not None and not isinstance(full_source, Mapping):
            raise RecordValidationError(
                "full_source must be a JSON object or null"
            )
        source_binding = data.get("source_binding")
        if source_binding is not None and not isinstance(
            source_binding, Mapping
        ):
            raise RecordValidationError(
                "source_binding must be a JSON object or null"
            )
        return cls(
            role=data["role"],
            force_basis=data["force_basis"],
            stiffness_derivative=data["stiffness_derivative"],
            bodies=tuple(bodies),
            baseline_wrench=baseline,
            provenance=dict(provenance),
            full_source=None if full_source is None else dict(full_source),
            source_binding=source_binding,
        )


def _decode_contract_attr(ds) -> Optional[dict]:
    raw = ds.attrs.get(HYDROSTATIC_CONTRACT_ATTR)
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SchemaError(
                f"attribute {HYDROSTATIC_CONTRACT_ATTR!r} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(decoded, dict):
            raise SchemaError(
                f"attribute {HYDROSTATIC_CONTRACT_ATTR!r} must be a JSON object"
            )
        return decoded
    raise SchemaError(f"attribute {HYDROSTATIC_CONTRACT_ATTR!r} must be a JSON object")


def validate_hydrostatic_contract_attr(ds) -> None:
    """Schema-level validation of the ``hydrostatic_contract`` attribute.

    Malformed metadata fails with :class:`SchemaError`; it is never ignored.
    """
    data = _decode_contract_attr(ds)
    if data is None:
        return
    try:
        contract = HydrostaticContract.from_json(data)
    except RecordValidationError as exc:
        raise SchemaError(
            f"attribute {HYDROSTATIC_CONTRACT_ATTR!r}: {exc}"
        ) from exc
    bodies = [str(b) for b in ds["body"].values.tolist()]
    if set(contract.bodies) != set(bodies):
        raise SchemaError(
            f"attribute {HYDROSTATIC_CONTRACT_ATTR!r} bodies must cover exactly "
            f"the dataset bodies {bodies}, got {list(contract.bodies)}"
        )


def contract_source_binding(ds) -> Mapping[str, Any]:
    """Noncircular source binding payload for the dataset's EXACT content.

    The hash excludes ONLY the contract/binding payload itself (the
    ``hydrostatic_contract`` attribute); it covers all arrays, axes and
    reference centers, coefficient frames, conventions, mesh/request/body
    identity and the transformation history. The binding declares target
    identity; it is not scientific qualification.
    """
    from .hashing import dataset_content_hash

    return {
        "binding_version": SOURCE_BINDING_VERSION,
        "dataset_content_hash": dataset_content_hash(
            ds, exclude_attrs=(HYDROSTATIC_CONTRACT_ATTR,)
        ),
    }


def hydrostatic_contract_binding_matches(ds) -> bool:
    """True when the attached contract is bound to the dataset's exact content.

    An unbound contract returns False; a stale/foreign binding (changed
    geometry, axes, coefficients or reference provenance) returns False.
    Transport stays possible either way; this is a detection helper, not a
    gate.
    """
    data = _decode_contract_attr(ds)
    if data is None:
        return False
    contract = HydrostaticContract.from_json(data)
    if contract.source_binding is None:
        return False
    return dict(contract.source_binding) == dict(contract_source_binding(ds))


def rebind_hydrostatic_contract(ds) -> str:
    """Canonical contract JSON rebound to the dataset's exact current content.

    Preserves every declared field and the original source identity and
    provenance; only the noncircular source binding is recomputed. Must be
    called only AFTER all content changes are complete. Returns the canonical
    JSON string for the ``hydrostatic_contract`` attribute.
    """
    data = _decode_contract_attr(ds)
    if data is None:
        raise HydrostaticContractError(
            "no hydrostatic contract is attached; nothing to rebind"
        )
    contract = HydrostaticContract.from_json(data)
    rebound = HydrostaticContract(
        role=contract.role,
        force_basis=contract.force_basis,
        stiffness_derivative=contract.stiffness_derivative,
        bodies=contract.bodies,
        baseline_wrench=contract.baseline_wrench,
        provenance=contract.provenance,
        full_source=contract.full_source,
        source_binding=contract_source_binding(ds),
    )
    return json.dumps(rebound.to_json(), sort_keys=True)


def read_hydrostatic_contract(ds) -> HydrostaticContract:
    """Read the attached contract; a missing contract is role UNKNOWN.

    UNKNOWN is never interpreted as zero stiffness, a gravity-free source or
    any other default.
    """
    from .schema import validate_hydro_dataset

    validate_hydro_dataset(ds)
    data = _decode_contract_attr(ds)
    if data is None:
        bodies = tuple(str(b) for b in ds["body"].values.tolist())
        return HydrostaticContract(
            role=ROLE_UNKNOWN,
            force_basis=FORCE_BASIS_UNKNOWN,
            stiffness_derivative=STIFFNESS_DERIVATIVE_UNKNOWN,
            bodies=bodies,
            baseline_wrench=None,
            provenance={"source": "none: no hydrostatic contract is attached"},
            full_source=None,
            source_binding=None,
        )
    return HydrostaticContract.from_json(data)


def attach_hydrostatic_contract(
    ds, contract: HydrostaticContract, *, rebind: bool = False
):
    """Return a new dataset with the validated contract attached.

    The input is not mutated. The contract body binding must equal the
    dataset bodies. The caller's declarations are bound to the EXACT dataset
    content via the versioned noncircular source binding. A contract that is
    already bound to different content is a typed refusal unless the caller
    passes ``rebind=True`` explicitly: a foreign/stale bound record is never
    silently rebound. Attaching a contract is a transport declaration only:
    it is not scientific qualification and not integrator admission.
    """
    from .schema import validate_hydro_dataset

    validate_hydro_dataset(ds)
    if not isinstance(contract, HydrostaticContract):
        raise HydrostaticContractError(
            "contract must be a HydrostaticContract record"
        )
    bodies = [str(b) for b in ds["body"].values.tolist()]
    if set(contract.bodies) != set(bodies):
        raise HydrostaticContractError(
            f"contract bodies {list(contract.bodies)} must cover exactly the "
            f"dataset bodies {bodies}"
        )
    binding = contract_source_binding(ds)
    if (
        contract.source_binding is not None
        and not rebind
        and dict(contract.source_binding) != dict(binding)
    ):
        raise HydrostaticContractError(
            "contract is already bound to different dataset content (bound "
            f"{contract.source_binding['dataset_content_hash']!r}, target "
            f"{binding['dataset_content_hash']!r}); rebinding a foreign or "
            "stale record requires an explicit rebind=True"
        )
    bound = HydrostaticContract(
        role=contract.role,
        force_basis=contract.force_basis,
        stiffness_derivative=contract.stiffness_derivative,
        bodies=contract.bodies,
        baseline_wrench=contract.baseline_wrench,
        provenance=contract.provenance,
        full_source=contract.full_source,
        source_binding=binding,
    )
    out = ds.copy(deep=True)
    out.attrs[HYDROSTATIC_CONTRACT_ATTR] = json.dumps(
        bound.to_json(), sort_keys=True
    )
    return validate_hydro_dataset(out)


def replace_baseline_wrench(
    contract: HydrostaticContract, new_baseline_wrench: Mapping[str, Tuple[float, ...]]
) -> str:
    """Canonical contract JSON with a transported baseline wrench substituted.

    Used by the coefficient frame map to transport a GENERALIZED_DOF baseline
    wrench with ``T.T``. The wrench is actual measured data being carried
    between equivalent coordinate representations, not fabricated.
    """
    updated = HydrostaticContract(
        role=contract.role,
        force_basis=contract.force_basis,
        stiffness_derivative=contract.stiffness_derivative,
        bodies=contract.bodies,
        baseline_wrench=new_baseline_wrench,
        provenance=contract.provenance,
        full_source=contract.full_source,
        source_binding=contract.source_binding,
    )
    return json.dumps(updated.to_json(), sort_keys=True)


@dataclass(frozen=True)
class HydrostaticCompositionEligibility:
    """Transport-level eligibility for the future exact six-DOF composition.

    ``admissible`` refers ONLY to the declared contract semantics being
    complete enough for ADR-004 composition. It is not scientific
    qualification, not solver admission and not a claim that any six-DOF
    consumer exists.
    """

    role: str
    admissible: bool
    reasons: Tuple[str, ...]


def hydrostatic_composition_eligibility(ds) -> HydrostaticCompositionEligibility:
    """Evaluate ADR-004 composition eligibility from the attached contract.

    This slice admits only complete, bound GENERALIZED_DOF declarations at
    the metadata level. Typed refusal reasons, never silent defaults:

    - role UNKNOWN (no or incomplete contract);
    - force basis or stiffness derivative semantics UNKNOWN;
    - missing baseline wrench: absent is UNMEASURED, not zero (a
      source-declared zero wrench is valid metadata and is never inferred);
    - PHYSICAL_WRENCH force basis: the source-specific rate/coordinate
      convention map and its implemented conversion are required first; no
      unverified prerequisite claim is recorded here;
    - role FULL: the declaration is transported as metadata only; the exact
      per-body decomposition/reference-only-gravity policy is not implemented
      or adopted in this slice;
    - unbound contract, or a stale/foreign source binding (geometry, axes,
      coefficients or reference provenance changed after binding).
    """
    contract = read_hydrostatic_contract(ds)
    reasons = []
    if contract.role == ROLE_UNKNOWN:
        reasons.append(
            "hydrostatic role is UNKNOWN: no (or an incomplete) contract is "
            "attached; UNKNOWN is not zero stiffness and not a gravity-free "
            "assumption"
        )
    if contract.force_basis == FORCE_BASIS_UNKNOWN:
        reasons.append("source force basis semantics are UNKNOWN")
    if contract.stiffness_derivative == STIFFNESS_DERIVATIVE_UNKNOWN:
        reasons.append("source stiffness derivative semantics are UNKNOWN")
    if contract.baseline_wrench is None:
        reasons.append(
            "baseline wrench is UNMEASURED (absent): missing is not zero; an "
            "actual measured baseline wrench per body is required for "
            "composition admission"
        )
    if contract.force_basis == FORCE_BASIS_PHYSICAL_WRENCH:
        reasons.append(
            "force basis PHYSICAL_WRENCH is not admitted: the source-specific "
            "rate/coordinate convention map and its implemented conversion "
            "are required first; this slice admits only complete, bound "
            "GENERALIZED_DOF declarations at the metadata level"
        )
    if contract.role == ROLE_FULL:
        reasons.append(
            "role FULL: the source mass/CG/composition declaration is "
            "transported as declared metadata only; the exact per-body "
            "decomposition/reference-only-gravity composition policy is "
            "not implemented or adopted in this slice"
        )
    if contract.role == ROLE_FULL and contract.full_source is None:
        reasons.append("role FULL lacks the required source mass/CG/composition")
    if contract.source_binding is None:
        reasons.append(
            "contract is not bound to the exact dataset content: no "
            f"{SOURCE_BINDING_VERSION} source binding is attached"
        )
    elif dict(contract.source_binding) != dict(contract_source_binding(ds)):
        reasons.append(
            "contract source binding is stale/foreign: bound dataset content "
            f"{contract.source_binding['dataset_content_hash']!r} does not "
            "match the actual dataset content (geometry, axes, coefficients "
            "or reference provenance changed after binding)"
        )
    return HydrostaticCompositionEligibility(
        role=contract.role,
        admissible=not reasons,
        reasons=tuple(reasons),
    )


def require_hydrostatic_composition(ds) -> HydrostaticCompositionEligibility:
    """Typed gate for the affected future exact six-DOF composition.

    Raises :class:`HydrostaticContractError` when the declared source
    semantics are incomplete. This refusal is the ADR-004 admission rule:
    unknown baseline/force/derivative/frame semantics leave the affected
    composition unadmitted; nothing is silently selected or subtracted.
    """
    eligibility = hydrostatic_composition_eligibility(ds)
    if not eligibility.admissible:
        raise HydrostaticContractError(
            "hydrostatic composition is not admissible: "
            + "; ".join(eligibility.reasons)
        )
    return eligibility
