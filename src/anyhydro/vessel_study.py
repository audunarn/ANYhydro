"""Versioned vessel source-study recipe for the six-DOF linear route (ADR-015 section 10, slice L6-4 H-d).

A ``VesselStudyRecipe`` binds everything a single-vessel six-DOF source study needs, immutably and hash-covered:

- per declared refinement level, the bound hull and auxiliary lid panel meshes (``HydroPanelMesh``; their content
  hashes are part of the recipe hash; levels are supplied, not generated - external hull panels such as the OrcaWave
  L01 mesh cannot be refined by a recipe);
- the reference pose (rotation centre = the linear reference point, centre of mass; world axes, wet geometry at its
  equilibrium pose) and the fluid (rho, g, depth);
- the frequency band and headings of the solves, the BEM method, and the declared producer (backend name, backend
  version, adapter version) that every solve is verified against;
- the radiation-kernel settings (exact piecewise-linear quadrature, linear-to-zero low tail, truncated high tail,
  Bartlett taper of length H, declared sample limit) and the normalised full-block reconstruction criteria
  (S = diag(1, 1, 1, 1/L, 1/L, 1/L)).

The study judges the full rigid block of one body (all six DOFs); retained-DOF subsets are selected later, at the
ANYmarine import. ``run_vessel_level`` solves one level through the Capytaine backend with the hydrostatic contract
attached (one solve per heading - the adapter admits one heading per solve), verifies every response against its
own request and the declared producer, builds the kernel and reports the normalised full-block gate, the contract
eligibility of every response and the kernel's Toeplitz-symbol diagnostic, bound to the recipe, dataset and kernel
identities. DEVELOPMENT TOOLING ONLY; nothing here confers qualification. Physical campaigns run only after
ANYmarine's L6-F2 gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .dof import body_dof_keys
from .errors import HydroError
from .records import HydroBody, HydroPanelMesh, HydroRequest

VESSEL_STUDY_FORMAT = "anyhydro-vessel-study-v2"
RIGID_DOFS = ("surge", "sway", "heave", "roll", "pitch", "yaw")
QUALIFICATION_SCOPE = "development_vessel_study_only"
_CRITERIA_KEYS = ("omega_min", "omega_max", "block_rel_tol", "modal_rel_tol", "energetic_fraction",
                  "cluster_fraction")
_PRODUCER_KEYS = ("backend", "backend_version", "adapter_version")


class VesselStudyError(HydroError):
    """The recipe is malformed or a study step is refused."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VesselStudyError(message)


def _number(value, name, *, positive=False, allow_inf=False) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float, np.floating, np.integer)),
             f"{name} must be a number")
    out = float(value)
    _require(math.isfinite(out) or (allow_inf and out == math.inf), f"{name} must be finite")
    _require(not positive or out > 0, f"{name} must be positive")
    return out


def _numbers(value, name, count=None):
    _require(isinstance(value, (list, tuple, np.ndarray)), f"{name} must be a sequence of numbers")
    out = tuple(_number(v, f"{name} entry") for v in value)
    _require(count is None or len(out) == count, f"{name} must have {count} entries")
    return out


@dataclass(frozen=True)
class VesselMeshLevel:
    """One declared refinement level: the bound wet hull and the optional auxiliary lid."""

    name: str
    hull: HydroPanelMesh
    lid: Optional[HydroPanelMesh] = None

    def __post_init__(self) -> None:
        _require(isinstance(self.name, str) and self.name.strip() != "", "level name must be non-empty")
        _require(isinstance(self.hull, HydroPanelMesh), "level hull must be a HydroPanelMesh")
        _require(self.lid is None or isinstance(self.lid, HydroPanelMesh), "level lid must be a HydroPanelMesh or None")

    def identity(self) -> dict:
        return {"name": self.name, "hull_mesh_hash": self.hull.mesh_hash,
                "lid_mesh_hash": None if self.lid is None else self.lid.mesh_hash}


@dataclass(frozen=True)
class KernelRecipe:
    dt: float
    n_time: int
    bartlett_window: float           # H (s): kernel x (1 - t/H)+, zero after H
    symmetry_tol: float
    passivity_tol: float
    max_time_samples: int            # declared execution limit (independent of n_time)
    low_tail_extension_name: str = "linear_to_zero_from_first_frequency"

    _KEYS = ("dt", "n_time", "bartlett_window", "symmetry_tol", "passivity_tol", "max_time_samples",
             "low_tail_extension_name")

    def __post_init__(self) -> None:
        object.__setattr__(self, "dt", _number(self.dt, "kernel dt", positive=True))
        for name in ("n_time", "max_time_samples"):
            value = getattr(self, name)
            _require(isinstance(value, int) and not isinstance(value, bool) and value >= 2,
                     f"kernel {name} must be an integer >= 2")
        _require(self.n_time <= self.max_time_samples, "kernel n_time exceeds the declared max_time_samples")
        object.__setattr__(self, "bartlett_window", _number(self.bartlett_window, "Bartlett window", positive=True))
        _require(self.bartlett_window <= (self.n_time - 1) * self.dt,
                 "the Bartlett window H must lie inside the kernel grid")
        for name in ("symmetry_tol", "passivity_tol"):
            object.__setattr__(self, name, _number(getattr(self, name), f"kernel {name}", positive=True))
        _require(isinstance(self.low_tail_extension_name, str) and self.low_tail_extension_name.strip() != "",
                 "low_tail_extension_name must be non-empty")
        self.settings()                                  # resolved settings validated now (limits, ceilings)

    def settings(self):
        from .radiation import (HIGH_TAIL_TRUNCATE, LOW_TAIL_LINEAR_TO_ZERO, QUADRATURE_PIECEWISE_LINEAR,
                                RadiationExecutionLimits, RadiationSettings, RadiationError, TIME_WINDOW_BARTLETT)
        try:
            return RadiationSettings(
                dt=self.dt, n_time=self.n_time, symmetry_tol=self.symmetry_tol, passivity_tol=self.passivity_tol,
                low_tail_policy=LOW_TAIL_LINEAR_TO_ZERO, extension_name=self.low_tail_extension_name,
                time_window=self.bartlett_window, time_window_shape=TIME_WINDOW_BARTLETT,
                quadrature_method=QUADRATURE_PIECEWISE_LINEAR, high_tail_policy=HIGH_TAIL_TRUNCATE,
                execution_limits=RadiationExecutionLimits(max_time_samples=self.max_time_samples))
        except RadiationError as error:
            raise VesselStudyError(f"kernel settings are not admissible: {error}") from error

    def to_dict(self) -> dict:
        return {key: getattr(self, key) for key in self._KEYS}


@dataclass(frozen=True)
class VesselStudyRecipe:
    title: str
    body_id: str
    levels: Tuple[VesselMeshLevel, ...]
    rotation_center: Tuple[float, float, float]
    center_of_mass: Tuple[float, float, float]
    rho: float
    gravity: float
    water_depth: float
    omegas: Tuple[float, ...]
    headings: Tuple[float, ...]
    length_scale: float              # L of the normalisation S = diag(1, 1, 1, 1/L, 1/L, 1/L)
    kernel: KernelRecipe
    full_block: Mapping[str, float]  # prospective criteria (frozen)
    producer: Mapping[str, str]      # backend, backend_version, adapter_version (verified on every response)
    method: str = "direct"

    def __post_init__(self) -> None:
        from .radiation import MAX_INPUT_FREQUENCIES, MIN_FREQUENCY_SAMPLES
        _require(isinstance(self.title, str) and self.title.strip() != "", "title must be non-empty")
        _require(isinstance(self.body_id, str) and self.body_id.strip() != "", "body_id must be non-empty")
        levels = tuple(self.levels)
        _require(len(levels) >= 1 and all(isinstance(level, VesselMeshLevel) for level in levels),
                 "at least one VesselMeshLevel is required")
        _require(len({level.name for level in levels}) == len(levels), "level names must be distinct")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "rotation_center", _numbers(self.rotation_center, "rotation_center", 3))
        object.__setattr__(self, "center_of_mass", _numbers(self.center_of_mass, "center_of_mass", 3))
        for name in ("rho", "gravity", "length_scale"):
            object.__setattr__(self, name, _number(getattr(self, name), name, positive=True))
        object.__setattr__(self, "water_depth", _number(self.water_depth, "water_depth", positive=True, allow_inf=True))
        omegas = _numbers(self.omegas, "omegas")
        _require(MIN_FREQUENCY_SAMPLES <= len(omegas) <= MAX_INPUT_FREQUENCIES and omegas[0] > 0
                 and all(b > a for a, b in zip(omegas, omegas[1:])),
                 f"omegas must be {MIN_FREQUENCY_SAMPLES}..{MAX_INPUT_FREQUENCIES} positive, strictly increasing values")
        object.__setattr__(self, "omegas", omegas)
        headings = _numbers(self.headings, "headings")
        _require(len(headings) >= 1 and len(set(headings)) == len(headings), "headings must be distinct")
        object.__setattr__(self, "headings", headings)
        _require(isinstance(self.kernel, KernelRecipe), "kernel must be a KernelRecipe")
        _require(self.method in ("direct", "indirect"), "method must be 'direct' or 'indirect'")
        _require(isinstance(self.full_block, Mapping) and set(self.full_block) == set(_CRITERIA_KEYS),
                 f"full_block criteria must contain exactly {list(_CRITERIA_KEYS)}")
        object.__setattr__(self, "full_block", MappingProxyType(
            {k: _number(self.full_block[k], f"criteria {k}", positive=True) for k in _CRITERIA_KEYS}))
        _require(isinstance(self.producer, Mapping) and set(self.producer) == set(_PRODUCER_KEYS)
                 and all(isinstance(self.producer[k], str) and self.producer[k].strip() for k in _PRODUCER_KEYS),
                 f"producer must declare exactly {list(_PRODUCER_KEYS)} as strings")
        object.__setattr__(self, "producer", MappingProxyType({k: self.producer[k] for k in _PRODUCER_KEYS}))
        self.full_block_criteria()                 # validated now
        _require(any(self.full_block["omega_min"] <= w <= self.full_block["omega_max"] for w in self.omegas),
                 "the full-block evaluation window contains no recipe frequency")
        self.request(self.headings[0])             # the request is admissible before any solve

    # ------------------------------------------------------------------ derived objects

    @property
    def retained_dofs(self) -> Tuple[str, ...]:
        """The study judges the full rigid block; subsets are selected at the ANYmarine import."""
        return RIGID_DOFS

    def dof_scales(self) -> Tuple[float, ...]:
        return (1.0, 1.0, 1.0) + (1.0 / self.length_scale,) * 3

    def full_block_criteria(self):
        from .radiation import FullBlockReconstructionCriteria, RadiationError
        try:
            return FullBlockReconstructionCriteria(dof_scales=self.dof_scales(), **dict(self.full_block))
        except RadiationError as error:
            raise VesselStudyError(f"full-block criteria are not admissible: {error}") from error

    def body(self, level: VesselMeshLevel) -> HydroBody:
        return HydroBody(body_id=self.body_id, name=self.title, mesh=level.hull, lid_mesh=level.lid,
                         rotation_center=self.rotation_center, center_of_mass=self.center_of_mass,
                         enabled_dofs=body_dof_keys(self.body_id))

    def request(self, heading: float) -> HydroRequest:
        return HydroRequest(omegas=np.asarray(self.omegas), wave_directions=np.array([heading]),
                            water_depth=self.water_depth, rho=self.rho, gravity=self.gravity,
                            include_zero_frequency=False, include_infinite_frequency=True,
                            backend_settings={"method": self.method, "hydrostatic_contract": "attach"})

    # ------------------------------------------------------------------ identity

    _KEYS = ("format", "qualification_scope", "title", "body_id", "levels", "rotation_center", "center_of_mass",
             "rho", "gravity", "water_depth", "omegas", "headings", "retained_dofs", "length_scale", "method",
             "kernel", "full_block", "producer")

    def to_dict(self) -> dict:
        return {
            "format": VESSEL_STUDY_FORMAT, "qualification_scope": QUALIFICATION_SCOPE, "title": self.title,
            "body_id": self.body_id, "levels": [level.identity() for level in self.levels],
            "rotation_center": list(self.rotation_center), "center_of_mass": list(self.center_of_mass),
            "rho": self.rho, "gravity": self.gravity,
            "water_depth": "inf" if math.isinf(self.water_depth) else self.water_depth,
            "omegas": list(self.omegas), "headings": list(self.headings), "retained_dofs": list(RIGID_DOFS),
            "length_scale": self.length_scale, "method": self.method, "kernel": self.kernel.to_dict(),
            "full_block": dict(self.full_block), "producer": dict(self.producer),
        }

    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], levels) -> "VesselStudyRecipe":
        """Rebuild a recipe from its identity dict and the supplied level meshes; strict (typed refusals only)."""
        _require(isinstance(data, Mapping) and set(data) == set(cls._KEYS),
                 f"a {VESSEL_STUDY_FORMAT} record must contain exactly {list(cls._KEYS)}")
        for key in ("levels", "retained_dofs", "omegas", "headings", "rotation_center", "center_of_mass"):
            _require(isinstance(data[key], (list, tuple)), f"{key} must be a list")
        for key in ("kernel", "full_block", "producer"):
            _require(isinstance(data[key], Mapping), f"{key} must be an object")
        _require(data["format"] == VESSEL_STUDY_FORMAT and data["qualification_scope"] == QUALIFICATION_SCOPE,
                 f"not a {VESSEL_STUDY_FORMAT} record")
        _require(list(data["retained_dofs"]) == list(RIGID_DOFS), "the study judges the full rigid block")
        levels = tuple(levels)
        _require([level.identity() for level in levels] == list(data["levels"]),
                 "the supplied level meshes do not match the recipe's bound mesh identities")
        kernel = data["kernel"]
        _require(isinstance(kernel, Mapping) and set(kernel) == set(KernelRecipe._KEYS),
                 f"kernel must contain exactly {list(KernelRecipe._KEYS)}")
        depth = data["water_depth"]
        recipe = cls(title=data["title"], body_id=data["body_id"], levels=levels,
                     rotation_center=data["rotation_center"], center_of_mass=data["center_of_mass"],
                     rho=data["rho"], gravity=data["gravity"], water_depth=math.inf if depth == "inf" else depth,
                     omegas=data["omegas"], headings=data["headings"], length_scale=data["length_scale"],
                     kernel=KernelRecipe(**kernel), full_block=data["full_block"], producer=data["producer"],
                     method=data["method"])
        _require(json.dumps(recipe.to_dict(), sort_keys=True) == json.dumps(dict(data), sort_keys=True),
                 "the recipe does not reproduce its identity record exactly")
        return recipe


@dataclass(frozen=True)
class VesselLevelResult:
    recipe_hash: str
    level: str
    datasets: Tuple[Any, ...]        # one per heading (hydrostatic contract attached)
    dataset_hashes: Tuple[str, ...]
    kernel: Any
    kernel_hash: str
    report: Any                      # FullBlockReconstructionReport (normalised) of the first-heading dataset
    contract_admissible: bool        # every response
    contract_reasons: Tuple[str, ...]
    symbol_min_eigenvalue: float     # min eig of dt (K_0 + 2 sum K_l cos(l theta)) on a grid (diagnostic)

    def summary(self) -> dict:
        s = self.report.summary()
        return {"recipe_hash": self.recipe_hash, "level": self.level, "dataset_hashes": list(self.dataset_hashes),
                "kernel_hash": self.kernel_hash, "kernel_source_dataset_hash": self.kernel.provenance["source_dataset_hash"],
                "headings": len(self.datasets), "full_block_pass": bool(s["overall_pass"]),
                "max_impedance_rel_error": s["max_impedance_rel_error"],
                "contract_admissible": self.contract_admissible, "contract_reasons": list(self.contract_reasons),
                "symbol_min_eigenvalue": self.symbol_min_eigenvalue,
                "kernel_passivity_violations": int(self.kernel.diagnostics.passivity_violations),
                "qualification_scope": QUALIFICATION_SCOPE}


def symbol_min_eigenvalue(kernel_values, dt: float, n_theta: int = 2049) -> float:
    K = np.asarray(kernel_values, dtype=float)
    theta = np.linspace(0.0, math.pi, n_theta)
    weights = 2.0 * np.cos(np.outer(theta, np.arange(K.shape[0])))
    weights[:, 0] = 1.0
    symbol = dt * np.einsum("tl,lij->tij", weights, K)
    symbol = 0.5 * (symbol + np.swapaxes(symbol, 1, 2))
    return float(np.min(np.linalg.eigvalsh(symbol)))


def _verify_response(recipe: VesselStudyRecipe, level: VesselMeshLevel, dataset, heading: float) -> None:
    """Each response must be the solve of its own request by the declared producer (labelled coordinates)."""
    _require(dataset.attrs.get("backend") == recipe.producer["backend"]
             and dataset.attrs.get("backend_version") == recipe.producer["backend_version"]
             and dataset.attrs.get("adapter_version") == recipe.producer["adapter_version"],
             f"the response was not produced by the declared producer {dict(recipe.producer)}")
    request = recipe.request(heading)
    _require(dataset.attrs.get("request_hash") == request.content_hash(),
             "the response is not the solve of the recipe's request (request hash: frequencies, heading, depth, "
             "rho, g, limits flags, backend settings)")
    identity = json.loads(dataset.attrs["body_identity"]) if isinstance(dataset.attrs.get("body_identity"), str) \
        else dict(dataset.attrs.get("body_identity", {}))
    body = recipe.body(level)
    expected = {"mesh_hash": body.mesh.mesh_hash,
                "lid_mesh_hash": None if body.lid_mesh is None else body.lid_mesh.mesh_hash,
                "rotation_center": [float(v) for v in body.rotation_center],
                "center_of_mass": [float(v) for v in body.center_of_mass],
                "enabled_dofs": list(body.enabled_dofs)}
    _require(identity.get(recipe.body_id) == expected,
             "the response body identity (hull/lid mesh, rotation centre, CG, DOFs) is not the recipe level's")
    _require(np.array_equal(np.asarray(dataset["omega"].values, dtype=float), np.asarray(recipe.omegas)),
             "the response frequencies are not the recipe's")
    _require(np.array_equal(np.asarray(dataset["wave_direction"].values, dtype=float), np.array([heading])),
             f"the response heading is not the requested {heading!r}")
    keys = list(body_dof_keys(recipe.body_id))
    for name in ("influenced_dof", "radiating_dof"):
        _require([str(k) for k in dataset[name].values] == keys, f"the response {name} is not the body's six DOFs")
    _require([str(b) for b in dataset["body"].values] == [recipe.body_id], "the response body is not the recipe body")


def run_vessel_level(recipe: VesselStudyRecipe, level_name: str, backend) -> VesselLevelResult:
    """Solve one level (one solve per heading), build the Bartlett-tapered kernel and run the normalised gate."""
    from .hashing import dataset_content_hash
    from .hydrostatic_contract import hydrostatic_composition_eligibility
    from .radiation import compare_reconstruction, radiation_kernel_from_dataset
    levels = {level.name: level for level in recipe.levels}
    _require(level_name in levels, f"unknown level {level_name!r}")
    recipe_hash = recipe.content_hash()                     # identity captured before any solve
    criteria = recipe.full_block_criteria()
    settings = recipe.kernel.settings()
    body = recipe.body(levels[level_name])
    datasets = []
    for heading in recipe.headings:
        dataset = backend.solve([body], recipe.request(heading))
        _verify_response(recipe, levels[level_name], dataset, heading)
        datasets.append(dataset)
    reference = datasets[0]
    for other in datasets[1:]:
        for name in ("added_mass", "radiation_damping", "added_mass_infinite", "hydrostatic_stiffness"):
            _require(other[name].identical(reference[name]),
                     f"labelled radiation/hydrostatic data differ between heading solves ({name})")
    reasons = []
    for dataset in datasets:
        eligibility = hydrostatic_composition_eligibility(dataset)
        reasons.extend(eligibility.reasons)
    kernel = radiation_kernel_from_dataset(reference, settings)
    report = compare_reconstruction(kernel, reference, criteria)
    cutoff = int(math.ceil(recipe.kernel.bartlett_window / recipe.kernel.dt))
    return VesselLevelResult(
        recipe_hash=recipe_hash, level=level_name, datasets=tuple(datasets),
        dataset_hashes=tuple(dataset_content_hash(d) for d in datasets), kernel=kernel,
        kernel_hash=kernel.content_hash(), report=report, contract_admissible=not reasons,
        contract_reasons=tuple(dict.fromkeys(reasons)),
        symbol_min_eigenvalue=symbol_min_eigenvalue(np.asarray(kernel.kernel)[:cutoff + 1], recipe.kernel.dt))
