"""Versioned vessel source-study recipe for the six-DOF linear route (ADR-015 section 10, slice L6-4 H-d).

A ``VesselStudyRecipe`` binds everything a single-vessel six-DOF source study needs, immutably and hash-covered:

- per declared refinement level, the bound hull and auxiliary lid panel meshes (``HydroPanelMesh``; their content
  hashes are part of the recipe hash; levels are supplied, not generated - external hull panels such as the OrcaWave
  L01 mesh cannot be refined by a recipe);
- the reference pose (rotation centre = the linear reference point, centre of mass; world axes, wet geometry at its
  equilibrium pose), the fluid (rho, g, depth) and the retained rigid DOFs;
- the frequency band and headings of the solves, the BEM method;
- the radiation-kernel settings (exact piecewise-linear quadrature, linear-to-zero low tail, truncated high tail,
  Bartlett taper of length H) and the normalised full-block reconstruction criteria (S = diag(1, 1/L)).

``run_vessel_level`` solves one level through the Capytaine backend with the hydrostatic contract attached (one solve
per heading - the adapter admits one heading per solve), builds the kernel and reports the normalised full-block gate,
the contract eligibility and the kernel's Toeplitz-symbol diagnostic. DEVELOPMENT TOOLING ONLY; nothing here confers
qualification. Physical campaigns run only after ANYmarine's L6-F2 gate.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from .dof import body_dof_keys
from .errors import HydroError
from .records import HydroBody, HydroPanelMesh, HydroRequest

VESSEL_STUDY_FORMAT = "anyhydro-vessel-study-v1"
RIGID_DOFS = ("surge", "sway", "heave", "roll", "pitch", "yaw")
QUALIFICATION_SCOPE = "development_vessel_study_only"


class VesselStudyError(HydroError):
    """The recipe is malformed or a study step is refused."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VesselStudyError(message)


def _floats(value, name, count=None):
    try:
        out = tuple(float(v) for v in value)
    except (TypeError, ValueError) as error:
        raise VesselStudyError(f"{name} must be numbers") from error
    _require(all(math.isfinite(v) for v in out), f"{name} must be finite")
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
    low_tail_extension_name: str = "linear_to_zero_from_first_frequency"

    def __post_init__(self) -> None:
        _require(self.dt > 0 and math.isfinite(self.dt), "kernel dt must be positive")
        _require(isinstance(self.n_time, int) and not isinstance(self.n_time, bool) and self.n_time >= 2,
                 "kernel n_time must be an integer >= 2")
        _require(0 < self.bartlett_window <= (self.n_time - 1) * self.dt,
                 "the Bartlett window H must lie inside the kernel grid")
        _require(self.symmetry_tol > 0 and self.passivity_tol > 0, "kernel tolerances must be positive")

    def settings(self):
        from .radiation import (HIGH_TAIL_TRUNCATE, LOW_TAIL_LINEAR_TO_ZERO, QUADRATURE_PIECEWISE_LINEAR,
                                RadiationExecutionLimits, RadiationSettings, TIME_WINDOW_BARTLETT)
        return RadiationSettings(
            dt=self.dt, n_time=self.n_time, symmetry_tol=self.symmetry_tol, passivity_tol=self.passivity_tol,
            low_tail_policy=LOW_TAIL_LINEAR_TO_ZERO, extension_name=self.low_tail_extension_name,
            time_window=self.bartlett_window, time_window_shape=TIME_WINDOW_BARTLETT,
            quadrature_method=QUADRATURE_PIECEWISE_LINEAR, high_tail_policy=HIGH_TAIL_TRUNCATE,
            execution_limits=RadiationExecutionLimits(max_time_samples=max(self.n_time, 1001)))


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
    retained_dofs: Tuple[str, ...]
    length_scale: float              # L of the normalisation S = diag(1, 1, 1, 1/L, 1/L, 1/L)
    kernel: KernelRecipe
    full_block: Mapping[str, float]  # omega_min, omega_max, block_rel_tol, modal_rel_tol, energetic/cluster fractions
    method: str = "direct"

    def __post_init__(self) -> None:
        _require(isinstance(self.title, str) and self.title.strip() != "", "title must be non-empty")
        _require(isinstance(self.body_id, str) and self.body_id.strip() != "", "body_id must be non-empty")
        levels = tuple(self.levels)
        _require(len(levels) >= 1 and all(isinstance(level, VesselMeshLevel) for level in levels),
                 "at least one VesselMeshLevel is required")
        _require(len({level.name for level in levels}) == len(levels), "level names must be distinct")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "rotation_center", _floats(self.rotation_center, "rotation_center", 3))
        object.__setattr__(self, "center_of_mass", _floats(self.center_of_mass, "center_of_mass", 3))
        for name in ("rho", "gravity", "length_scale"):
            _require(float(getattr(self, name)) > 0, f"{name} must be positive")
        _require(float(self.water_depth) > 0, "water_depth must be positive (inf for deep water)")
        omegas = _floats(self.omegas, "omegas")
        _require(len(omegas) >= 2 and all(b > a for a, b in zip(omegas, omegas[1:])) and omegas[0] > 0,
                 "omegas must be positive and strictly increasing")
        object.__setattr__(self, "omegas", omegas)
        headings = _floats(self.headings, "headings")
        _require(len(headings) >= 1 and len(set(headings)) == len(headings), "headings must be distinct")
        object.__setattr__(self, "headings", headings)
        dofs = tuple(self.retained_dofs)
        _require(dofs and all(d in RIGID_DOFS for d in dofs) and list(dofs) == sorted(set(dofs), key=RIGID_DOFS.index),
                 "retained_dofs must be distinct rigid DOFs in canonical order")
        object.__setattr__(self, "retained_dofs", dofs)
        _require(isinstance(self.kernel, KernelRecipe), "kernel must be a KernelRecipe")
        _require(self.method in ("direct", "indirect"), "method must be 'direct' or 'indirect'")
        keys = {"omega_min", "omega_max", "block_rel_tol", "modal_rel_tol", "energetic_fraction", "cluster_fraction"}
        _require(isinstance(self.full_block, Mapping) and set(self.full_block) == keys,
                 f"full_block criteria must contain exactly {sorted(keys)}")
        object.__setattr__(self, "full_block", {k: float(v) for k, v in self.full_block.items()})
        self.full_block_criteria()                 # validated now

    # ------------------------------------------------------------------ derived objects

    def dof_scales(self) -> Tuple[float, ...]:
        return (1.0, 1.0, 1.0) + (1.0 / float(self.length_scale),) * 3

    def full_block_criteria(self):
        from .radiation import FullBlockReconstructionCriteria
        return FullBlockReconstructionCriteria(dof_scales=self.dof_scales(), **self.full_block)

    def body(self, level: VesselMeshLevel) -> HydroBody:
        return HydroBody(body_id=self.body_id, name=self.title, mesh=level.hull, lid_mesh=level.lid,
                         rotation_center=self.rotation_center, center_of_mass=self.center_of_mass,
                         enabled_dofs=body_dof_keys(self.body_id))

    def request(self, heading: float) -> HydroRequest:
        return HydroRequest(omegas=np.asarray(self.omegas), wave_directions=np.array([heading]),
                            water_depth=float(self.water_depth), rho=float(self.rho), gravity=float(self.gravity),
                            include_zero_frequency=False, include_infinite_frequency=True,
                            backend_settings={"method": self.method, "hydrostatic_contract": "attach"})

    # ------------------------------------------------------------------ identity

    def to_dict(self) -> dict:
        return {
            "format": VESSEL_STUDY_FORMAT, "qualification_scope": QUALIFICATION_SCOPE, "title": self.title,
            "body_id": self.body_id, "levels": [level.identity() for level in self.levels],
            "rotation_center": list(self.rotation_center), "center_of_mass": list(self.center_of_mass),
            "rho": float(self.rho), "gravity": float(self.gravity),
            "water_depth": "inf" if math.isinf(float(self.water_depth)) else float(self.water_depth),
            "omegas": list(self.omegas), "headings": list(self.headings), "retained_dofs": list(self.retained_dofs),
            "length_scale": float(self.length_scale), "method": self.method,
            "kernel": {"dt": self.kernel.dt, "n_time": self.kernel.n_time,
                       "bartlett_window": self.kernel.bartlett_window, "symmetry_tol": self.kernel.symmetry_tol,
                       "passivity_tol": self.kernel.passivity_tol,
                       "low_tail_extension_name": self.kernel.low_tail_extension_name},
            "full_block": dict(sorted(self.full_block.items())),
        }

    def content_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], levels) -> "VesselStudyRecipe":
        """Rebuild a recipe from its identity dict and the supplied level meshes; refuses any mesh mismatch."""
        _require(isinstance(data, Mapping) and data.get("format") == VESSEL_STUDY_FORMAT,
                 f"not a {VESSEL_STUDY_FORMAT} record")
        levels = tuple(levels)
        _require([level.identity() for level in levels] == list(data["levels"]),
                 "the supplied level meshes do not match the recipe's bound mesh identities")
        depth = data["water_depth"]
        recipe = cls(title=data["title"], body_id=data["body_id"], levels=levels,
                     rotation_center=data["rotation_center"], center_of_mass=data["center_of_mass"],
                     rho=data["rho"], gravity=data["gravity"], water_depth=math.inf if depth == "inf" else depth,
                     omegas=data["omegas"], headings=data["headings"], retained_dofs=tuple(data["retained_dofs"]),
                     length_scale=data["length_scale"], kernel=KernelRecipe(**data["kernel"]),
                     full_block=data["full_block"], method=data["method"])
        _require(recipe.to_dict() == dict(data), "the recipe does not reproduce its identity record")
        return recipe


@dataclass(frozen=True)
class VesselLevelResult:
    recipe_hash: str
    level: str
    datasets: Tuple[Any, ...]        # one per heading (hydrostatic contract attached)
    kernel: Any
    report: Any                      # FullBlockReconstructionReport (normalised)
    contract_admissible: bool
    contract_reasons: Tuple[str, ...]
    symbol_min_eigenvalue: float     # min eig of dt (K_0 + 2 sum K_l cos(l theta)) on a grid (diagnostic)

    def summary(self) -> dict:
        s = self.report.summary()
        return {"recipe_hash": self.recipe_hash, "level": self.level, "headings": len(self.datasets),
                "full_block_pass": bool(s["overall_pass"]), "max_impedance_rel_error": s["max_impedance_rel_error"],
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


def run_vessel_level(recipe: VesselStudyRecipe, level_name: str, backend) -> VesselLevelResult:
    """Solve one level (one solve per heading), build the Bartlett-tapered kernel and run the normalised gate."""
    from .hydrostatic_contract import hydrostatic_composition_eligibility
    from .radiation import compare_reconstruction, radiation_kernel_from_dataset
    levels = {level.name: level for level in recipe.levels}
    _require(level_name in levels, f"unknown level {level_name!r}")
    body = recipe.body(levels[level_name])
    datasets = tuple(backend.solve([body], recipe.request(h)) for h in recipe.headings)
    reference = datasets[0]
    for other in datasets[1:]:
        for name in ("added_mass", "radiation_damping", "added_mass_infinite"):
            _require(np.array_equal(np.asarray(other[name].values), np.asarray(reference[name].values)),
                     f"radiation coefficients differ between heading solves ({name})")
    kernel = radiation_kernel_from_dataset(reference, recipe.kernel.settings())
    report = compare_reconstruction(kernel, reference, recipe.full_block_criteria())
    eligibility = hydrostatic_composition_eligibility(reference)
    cutoff = int(round(recipe.kernel.bartlett_window / recipe.kernel.dt))
    return VesselLevelResult(
        recipe_hash=recipe.content_hash(), level=level_name, datasets=datasets, kernel=kernel, report=report,
        contract_admissible=bool(eligibility.admissible), contract_reasons=tuple(eligibility.reasons),
        symbol_min_eigenvalue=symbol_min_eigenvalue(np.asarray(kernel.kernel)[:cutoff + 1], recipe.kernel.dt))
