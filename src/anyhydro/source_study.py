"""First-slice source-quality mesh study workflow (development tooling).

Prepare-only path (no backend construction/import), execute path (at
most ONE solve per registered level through an explicitly selected
public backend under explicit execution limits) and reload/adjudicate
path (the registered geometry/request inventory is re-verified against
the rebuilt expected inventory on every reload/execute/adjudicate
before any backend work, then the full public raw-source proof, reports
reproduced without any fresh BEM). DEVELOPMENT TOOLING ONLY: ``qualification_scope`` is always
``"development_source_study_only"``; no outcome of this study confers
scientific qualification, and this slice carries no BEM execution
authority — no solve is ever run here; the real Capytaine adapter is
only constructed on the explicit execute path under case-derived
explicit ``CapytaineExecutionLimits`` (the 100-panel DEFAULT refuses
the prospective refined levels; explicit limits admit them up to the
adapter's absolute 2048-panel software ceiling).

Only public anyhydro APIs are used: the first-slice fixture builders and
analytic verifier, the canonical schema validation, the dataset
persistence, and the public derived-operator construction, persistence
and raw-source proof. The refinement metric arithmetic lives in
:mod:`anyhydro.source_study_case` (new study-tool code operating on
public artifact arrays in the case's declared normalized units).
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from .errors import HydroError
from .dof import body_dof_keys
from .first_slice import (
    FIRST_SLICE_COM_RELATIVE,
    verify_first_slice_fixture,
)
from .hashing import dataset_content_hash
from .operators import (
    DerivedHydroOperators,
    derived_operators_from_dataset,
    load_derived_operators,
    save_derived_operators,
)
from .persistence import load_hydro_dataset, save_hydro_dataset
from .schema import (
    HEADING_CONVENTION,
    NORMAL_CONVENTION,
    PHASOR_CONVENTION,
    validate_for_time_domain,
)
from .source_study_case import (
    DEVELOPMENT_QUALIFICATION_SCOPE,
    FIRST_SLICE_GEOMETRY_RECIPE,
    FIRST_SLICE_SUBMERGED_LID_GEOMETRY_RECIPE,
    OUTCOME_CANCELLED,
    OUTCOME_FAILED,
    OUTCOME_NO_GO,
    OUTCOME_PREPARED_NOT_EXECUTED,
    OUTCOME_SOURCE_ADMISSION_PASSED_DEVELOPMENT_ONLY,
    SOURCE_STUDY_FORMAT,
    SOURCE_STUDY_MODULE_VERSION,
    SOURCE_STUDY_QUANTITIES,
    QuantityRefinementCriteria,
    RefinementCriteria,
    SourceStudyCase,
    SourceStudyCaseError,
    SourceStudyError,
    SourceStudyIdentityError,
    SourceStudyWorkflowError,
    artifact_level_arrays,
    study_lid_policy,
    build_study_bodies,
    build_study_request,
    compare_level_refinement,
    crossbody_coupling_report,
    make_prospective_source_study_case,
    study_operator_settings,
)

__all__ = [
    "SOURCE_STUDY_FORMAT",
    "SOURCE_STUDY_MODULE_VERSION",
    "DEVELOPMENT_QUALIFICATION_SCOPE",
    "FIRST_SLICE_GEOMETRY_RECIPE",
    "SOURCE_STUDY_QUANTITIES",
    "RECEIVED_ARCHIVE_FORMAT",
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
    "prepare_source_study",
    "execute_source_study",
    "load_source_study",
    "adjudicate_source_study",
    "CLI_EXIT_OK",
    "CLI_EXIT_ERROR",
    "CLI_EXIT_FAILED",
    "CLI_EXIT_CANCELLED",
    "CLI_EXIT_NO_GO",
    "CLI_EXIT_INCOMPLETE",
    "main",
]

_Status = Mapping[str, Any]


def _require(condition: bool, error: Exception) -> None:
    if not condition:
        raise error


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True, indent=2,
                  allow_nan=False)
        handle.write("\n")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError) as exc:
        raise SourceStudyWorkflowError(f"corrupt or unreadable {path}: {exc}") \
            from exc


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _level_dir(out_dir: Path, level: int) -> Path:
    return out_dir / f"level-{int(level):02d}"


def _verify_geometry_inventory(out_dir: Path, case: SourceStudyCase) -> None:
    """Bind the saved geometry/request inventory to the immutable case.

    The registered ``geometry_inventory.json`` is REQUIRED on every
    reload/execute/adjudicate and compared as one full canonical record
    against the inventory rebuilt deterministically from the hash-bound
    case through the PUBLIC first-slice APIs (exact per-level panel/
    vertex counts, per-body hull/lid mesh hashes, source geometry refs,
    request hash and analytic verification totals). Closed-world exact
    identity, no numeric smoothing: a missing, unreadable, unknown,
    removed or modified field — and any drift of the CURRENT built
    recipe under the same case hash — is a typed refusal BEFORE any
    backend capability check or solve. No autocorrection, no migration,
    no re-registration; a prepared case can never silently execute new
    geometry.
    """
    inventory_path = out_dir / "geometry_inventory.json"
    if not inventory_path.is_file():
        raise SourceStudyWorkflowError(
            f"missing geometry inventory: {inventory_path}; the registered "
            "input reproducibility record is required on every reload/"
            "execute/adjudicate; refusing")
    saved = _read_json(inventory_path)
    _require(isinstance(saved, Mapping),
             SourceStudyWorkflowError(
                 "geometry inventory must be a JSON object; refusing"))
    expected = _geometry_inventory(case)
    try:
        saved_canonical = _canonical_json(saved)
        expected_canonical = _canonical_json(expected)
    except ValueError as exc:
        raise SourceStudyWorkflowError(
            f"geometry inventory is not a finite canonical JSON record: "
            f"{exc}; refusing") from exc
    _require(saved_canonical == expected_canonical,
             SourceStudyIdentityError(
                 "geometry inventory mismatch: the saved inventory does not "
                 "match the inventory rebuilt from the immutable case "
                 "through the public first-slice APIs (missing, unknown or "
                 "modified level/panel/mesh-hash/request-hash fields, or "
                 "current-recipe drift under the same case hash); "
                 "refusing"))


def _load_case_record(out_dir: Path) -> Tuple[SourceStudyCase, str]:
    """Closed-world reload of the case record with hash verification.

    The saved geometry/request inventory is also re-verified against the
    rebuilt expected inventory on this shared route, so every reload,
    execute and adjudicate refuses a missing/tampered/unknown inventory
    (or current-recipe drift under the same case hash) BEFORE any
    backend capability check or solve.
    """
    case_path = out_dir / "case.json"
    if not case_path.is_file():
        raise SourceStudyWorkflowError(
            f"missing case record: {case_path}; refusing a partial study")
    record = _read_json(case_path)
    _require(isinstance(record, Mapping),
             SourceStudyWorkflowError("case record must be a JSON object"))
    expected = {"format", "case", "case_hash"}
    unknown = set(record) - expected
    missing = expected - set(record)
    if unknown or missing:
        raise SourceStudyWorkflowError(
            f"case record field mismatch (unknown: {sorted(unknown)}, "
            f"missing: {sorted(missing)}); refusing")
    _require(record["format"] == SOURCE_STUDY_FORMAT,
             SourceStudyWorkflowError(
                 f"case record format must be {SOURCE_STUDY_FORMAT!r}"))
    case = SourceStudyCase.from_dict(record["case"])
    _require(record["case_hash"] == case.content_hash(),
             SourceStudyIdentityError(
                 "case hash mismatch: the stored case record was changed "
                 "after preparation; refusing"))
    _verify_geometry_inventory(out_dir, case)
    return case, str(record["case_hash"])


def _load_status(out_dir: Path) -> _Status:
    status_path = out_dir / "status.json"
    if not status_path.is_file():
        raise SourceStudyWorkflowError(
            f"missing status record: {status_path}; refusing a partial study")
    status = _read_json(status_path)
    _require(isinstance(status, Mapping),
             SourceStudyWorkflowError("status record must be a JSON object"))
    expected = {"status", "outcome", "case_hash", "updated_utc"}
    unknown = set(status) - expected
    missing = expected - set(status)
    if unknown or missing:
        raise SourceStudyWorkflowError(
            f"status record field mismatch (unknown: {sorted(unknown)}, "
            f"missing: {sorted(missing)}); refusing")
    return dict(status)


def _write_status(out_dir: Path, status: str, outcome: Optional[str],
                  case_hash: str) -> None:
    _write_json(out_dir / "status.json", {
        "status": status,
        "outcome": outcome,
        "case_hash": case_hash,
        "updated_utc": _utc_now(),
    })


def _geometry_inventory(case: SourceStudyCase) -> dict:
    """Full per-level geometry + request inventory via the public fixture."""
    levels = {}
    for level in case.levels:
        bodies = build_study_bodies(case, level)
        request = build_study_request(case)
        verification = verify_first_slice_fixture(bodies, subdivisions=level, lid_policy=study_lid_policy(case))
        levels[str(int(level))] = {
            "subdivisions": int(level),
            "total_panels": 24 * int(level) * int(level),
            "hull_panels_per_body": 10 * int(level) * int(level),
            "lid_panels_per_body": 2 * int(level) * int(level),
            "hull_vertices_per_body": 5 * int(level) ** 2 + 2 * int(level) + 1,
            "lid_vertices_per_body": (int(level) + 1) ** 2,
            "mesh_hashes": {
                b.body_id: {
                    "hull": b.mesh.mesh_hash,
                    "lid": b.lid_mesh.mesh_hash,
                }
                for b in bodies
            },
            "source_geometry_refs": {
                b.body_id: list(b.mesh.source_geometry_refs) for b in bodies
            },
            "request_hash": request.content_hash(),
            "verification_totals": verification["totals"],
        }
    return {
        "geometry_recipe": case.geometry_recipe,
        "levels": levels,
        "request": {
            "omegas": list(case.omegas),
            "wave_directions": list(case.wave_directions),
            "include_zero_frequency": case.include_zero_frequency,
            "include_infinite_frequency": case.include_infinite_frequency,
            "backend_settings": dict(build_study_request(case).backend_settings),
        },
    }


def prepare_source_study(case: SourceStudyCase, out_dir) -> dict:
    """Prepare-only path: write the immutable case and full inventory.

    ``out_dir`` must be an EXCLUSIVE new directory. The status is
    ``not_executed``; no backend is constructed or imported and no
    scientific metric is claimed.
    """
    _require(isinstance(case, SourceStudyCase),
             SourceStudyCaseError("case must be a SourceStudyCase"))
    target = Path(out_dir)
    if target.exists():
        raise SourceStudyWorkflowError(
            f"output directory {target} already exists; prepare requires an "
            "exclusive new directory")
    target.mkdir(parents=True)
    case_hash = case.content_hash()
    _write_json(target / "case.json", {
        "format": SOURCE_STUDY_FORMAT,
        "case": case.to_dict(),
        "case_hash": case_hash,
    })
    _write_json(target / "geometry_inventory.json",
                _geometry_inventory(case))
    _write_status(target, "not_executed", OUTCOME_PREPARED_NOT_EXECUTED,
                  case_hash)
    return {
        "out_dir": str(target),
        "status": "not_executed",
        "outcome": OUTCOME_PREPARED_NOT_EXECUTED,
        "case_hash": case_hash,
        "qualification_scope": case.qualification_scope,
    }


class _SolveController:
    """One-solve-per-level guard around the explicitly selected backend."""

    def __init__(self, backend, case: SourceStudyCase) -> None:
        self._backend = backend
        self._case = case
        self.calls: Dict[int, int] = {int(n): 0 for n in case.levels}
        self.call_log: List[dict] = []

    def solve(self, level: int, bodies, request):
        n = int(level)
        _require(n in self.calls,
                 SourceStudyWorkflowError(
                     f"solve requested for unregistered level {n}; no new "
                     "levels are invented"))
        _require(self.calls[n] == 0,
                 SourceStudyWorkflowError(
                     f"level {n} was already solved once; no retries are "
                     "permitted"))
        self.calls[n] += 1
        started = _utc_now()
        dataset = self._backend.solve(bodies, request)
        self.call_log.append({
            "level": n,
            "started_utc": started,
            "finished_utc": _utc_now(),
            "solve_calls": 1,
        })
        return dataset


def _check_backend_identity(case: SourceStudyCase, backend,
                            execution_limits) -> dict:
    """Caps, version and capability checks BEFORE any backend work."""
    _require(hasattr(backend, "solve") and callable(backend.solve),
             SourceStudyIdentityError(
                 "backend must provide a public solve(bodies, request)"))
    capabilities = backend.capabilities()
    name = str(capabilities.backend_name)
    version = str(capabilities.backend_version)
    _require(name == case.expected_backend_name,
             SourceStudyIdentityError(
                 f"backend name {name!r} does not match the case-expected "
                 f"{case.expected_backend_name!r}"))
    _require(version == case.expected_backend_version,
             SourceStudyIdentityError(
                 f"backend version {version!r} does not match the "
                 f"case-expected {case.expected_backend_version!r}; "
                 "refusing before any solve"))
    # Required capabilities are checked exactly as the case requests
    # them; a backend advertising EXTRA unrequested capability is not a
    # mismatch. The fixed case never requests the zero-frequency limit:
    # include_zero_frequency=False means it is NOT REQUESTED/COMPUTED,
    # so supports_zero_frequency is only checked when the case actually
    # requests that feature.
    _require(bool(capabilities.supports_multibody) is True,
             SourceStudyIdentityError(
                 "backend does not support multibody solves; the fixed "
                 "two-body case requires them"))
    _require(bool(capabilities.supports_infinite_frequency) is True,
             SourceStudyIdentityError(
                 "backend does not support the infinite-frequency limit; "
                 "the case requires it"))
    _require(bool(capabilities.provides_added_mass_infinite) is True,
             SourceStudyIdentityError(
                 "backend does not provide an explicit added_mass_infinite; "
                 "the case requires it"))
    if case.include_zero_frequency:
        _require(bool(capabilities.supports_zero_frequency) is True,
                 SourceStudyIdentityError(
                     "the case requests the zero-frequency limit but the "
                     "backend does not support it"))
    for attr in ("max_total_panels", "max_finite_omegas",
                 "max_wave_directions", "n_jobs", "n_threads"):
        _require(hasattr(execution_limits, attr),
                 SourceStudyIdentityError(
                     f"execution limits must expose {attr!r}"))
    limits = {
        "max_total_panels": _int(execution_limits.max_total_panels,
                                 "max_total_panels"),
        "max_finite_omegas": _int(execution_limits.max_finite_omegas,
                                  "max_finite_omegas"),
        "max_wave_directions": _int(execution_limits.max_wave_directions,
                                    "max_wave_directions"),
        "n_jobs": _int(execution_limits.n_jobs, "n_jobs"),
        "n_threads": _int(execution_limits.n_threads, "n_threads"),
    }
    _require(limits["max_total_panels"] >= case.max_total_panels(),
             SourceStudyIdentityError(
                 f"execution limits admit only "
                 f"{limits['max_total_panels']} total panels; the case "
                 f"requires at least {case.max_total_panels()} (finest level "
                 f"{max(case.levels)}); refusing before any backend work"))
    _require(limits["max_finite_omegas"] >= len(case.omegas),
             SourceStudyIdentityError(
                 f"execution limits admit only "
                 f"{limits['max_finite_omegas']} finite frequencies; the "
                 f"case grid has {len(case.omegas)}"))
    _require(limits["max_wave_directions"] >= len(case.wave_directions),
             SourceStudyIdentityError(
                 f"execution limits admit only "
                 f"{limits['max_wave_directions']} headings; the case grid "
                 f"has {len(case.wave_directions)}"))
    _require(limits["n_jobs"] == case.n_jobs and
             limits["n_threads"] == case.n_threads,
             SourceStudyIdentityError(
                 "execution limits must match the case resource caps "
                 f"(n_jobs={case.n_jobs}, n_threads={case.n_threads})"))
    return {
        "backend_name": name,
        "backend_version": version,
        "execution_limits": limits,
        "zero_frequency_computed": bool(case.include_zero_frequency),
        "required_capabilities": {
            "supports_multibody": True,
            "supports_infinite_frequency": True,
            "provides_added_mass_infinite": True,
            "supports_zero_frequency": bool(case.include_zero_frequency),
        },
    }


def _int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise SourceStudyIdentityError(
            f"execution limit {name} must be an integer (got "
            f"{type(value).__name__}: {value!r})")
    return int(value)


#: Prospective fixed-fixture ADMISSION numerical tolerance (metres) for
#: the geometric center data variables — NOT a meshing tolerance. The
#: fixed fixture coordinates are exactly representable floats, so this
#: only guards serialization noise; no coordinate mutation or rounding
#: is ever performed.
FIXTURE_ADMISSION_ATOL = 1e-12


def _check_fixture_center_arrays(case: SourceStudyCase, dataset: xr.Dataset,
                                 bodies, level: int) -> None:
    """Prove the ACTUAL ordered center data variables match the fixture.

    The ``body_identity`` attrs alone cannot prove the actual arrays:
    the ordered ``rotation_center``/``center_of_mass``/
    ``center_of_buoyancy`` data variables are compared against the
    fixed first-slice fixture (reference centers (-2, 0, 0) m and
    (2, 0, 0) m; COM and COB each (cx, 0, -0.25) m) with the body and
    DOF axes in the fixed order, at ``atol = FIXTURE_ADMISSION_ATOL``
    with ``rtol = 0``.
    """
    del case  # the fixture is fixed; only the built bodies are needed
    expected_body_ids = [b.body_id for b in bodies]
    body_axis = [str(v) for v in np.asarray(dataset["body"].values)]
    _require(body_axis == expected_body_ids,
             SourceStudyIdentityError(
                 f"fixed-fixture mismatch at level {level}: dataset body "
                 f"axis {body_axis} does not match the fixed body order "
                 f"{expected_body_ids}"))
    xyz_axis = [str(v) for v in np.asarray(dataset["xyz"].values)]
    _require(xyz_axis == ["x", "y", "z"],
             SourceStudyIdentityError(
                 f"fixed-fixture mismatch at level {level}: dataset xyz "
                 f"axis is {xyz_axis}, expected ['x', 'y', 'z']"))
    expected_dofs = [key for b in bodies for key in body_dof_keys(b.body_id)]
    for axis in ("radiating_dof", "influenced_dof"):
        dof_axis = [str(v) for v in np.asarray(dataset[axis].values)]
        _require(dof_axis == expected_dofs,
                 SourceStudyIdentityError(
                     f"fixed-fixture mismatch at level {level}: dataset "
                     f"{axis} axis does not match the fixed body/DOF "
                     "order of the case bodies"))
    expected_centers = {
        "rotation_center": np.array(
            [b.rotation_center for b in bodies], dtype=np.float64),
        "center_of_mass": np.array(
            [b.center_of_mass for b in bodies], dtype=np.float64),
        # fixed fixture COB: (cx, 0, -0.25) m — the analytic centroid of
        # the 0.5 m draft box, coincident with the fixture COM offset
        "center_of_buoyancy": np.array(
            [[b.rotation_center[0] + FIRST_SLICE_COM_RELATIVE[0],
              b.rotation_center[1] + FIRST_SLICE_COM_RELATIVE[1],
              b.rotation_center[2] + FIRST_SLICE_COM_RELATIVE[2]]
             for b in bodies], dtype=np.float64),
    }
    for name, expected in expected_centers.items():
        _require(name in dataset.data_vars,
                 SourceStudyIdentityError(
                     f"fixed-fixture mismatch at level {level}: dataset "
                     f"is missing the {name!r} data variable"))
        actual = np.asarray(dataset[name].values, dtype=np.float64)
        _require(actual.shape == expected.shape,
                 SourceStudyIdentityError(
                     f"fixed-fixture mismatch at level {level}: dataset "
                     f"{name} has shape {actual.shape}, expected "
                     f"{expected.shape}"))
        _require(bool(np.allclose(actual, expected,
                                  rtol=0.0, atol=FIXTURE_ADMISSION_ATOL)),
                 SourceStudyIdentityError(
                     f"fixed-fixture mismatch at level {level}: the "
                     f"actual dataset {name} data variable "
                     f"{actual.tolist()} does not match the fixed "
                     f"first-slice fixture {expected.tolist()} "
                     f"(atol {FIXTURE_ADMISSION_ATOL} m, rtol 0; "
                     "metadata attrs alone cannot prove the actual "
                     "arrays)"))


#: Bounded forensic archive of a schema-invalid (or unsaved) RECEIVED
#: source. Clearly labelled rejected/received evidence: never an
#: admitted neutral dataset, never loadable as a qualified source.
RECEIVED_ARCHIVE_FORMAT = "anyhydro-source-study-received-source-archive-v1"
_FORENSIC_MAX_ELEMENTS = 32
_FORENSIC_MAX_STRING = 512


def _forensic_scalar(value: Any) -> Any:
    """One bounded JSON-safe forensic scalar (no NaN/Infinity leakage)."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, complex):
        return [_forensic_scalar(value.real), _forensic_scalar(value.imag)]
    if isinstance(value, float):
        return value if np.isfinite(value) else repr(value)
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    return str(value)[:_FORENSIC_MAX_STRING]


def _forensic_array(values) -> dict:
    """Bounded snapshot of one received array (capped, flagged)."""
    try:
        arr = np.asarray(values)
        flat = arr.ravel()[:_FORENSIC_MAX_ELEMENTS]
        elements = [_forensic_scalar(v) for v in flat.tolist()]
        return {
            "shape": [int(v) for v in arr.shape],
            "dtype": str(arr.dtype),
            "truncated": bool(arr.size > _FORENSIC_MAX_ELEMENTS),
            "elements": elements,
        }
    except Exception as exc:  # the snapshot itself must never mask the failure
        return {"snapshot_error": f"{type(exc).__name__}: {exc}"}


def _write_received_forensic_archive(level_dir: Path, received, level: int,
                                     failure: Exception) -> str:
    """Write the labelled rejected/received forensic archive; return name."""
    if isinstance(received, xr.Dataset):
        received_type = "xarray.Dataset"
        try:
            attrs = {str(k): _forensic_scalar(v)
                     for k, v in dict(received.attrs).items()}
        except Exception as exc:
            attrs = {"snapshot_error": f"{type(exc).__name__}: {exc}"}
        coords = {str(name): _forensic_array(received.coords[name].values)
                  for name in received.coords}
        variables = {str(name): _forensic_array(received[name].values)
                     for name in received.data_vars}
    else:
        received_type = type(received).__name__
        attrs, coords, variables = {}, {}, {}
    archive = {
        "format": RECEIVED_ARCHIVE_FORMAT,
        "status": ("rejected_received_schema_invalid"
                   if isinstance(failure, HydroError)
                   else "rejected_received_save_failed"),
        "admitted": False,
        "qualification_status": "unqualified_rejected_received",
        "level": int(level),
        "received_type": received_type,
        "failure_type": type(failure).__name__,
        "failure": f"{type(failure).__name__}: {failure}",
        "note": (
            "bounded forensic snapshot of a RECEIVED source that could not "
            "be saved through the public schema route; this is labelled "
            "rejected/received evidence, NOT an admitted neutral dataset "
            "and never a qualified source; the original typed failure is "
            "re-raised unchanged and the raw source is not mutated"),
        "attrs": attrs,
        "coords": coords,
        "data_variables": variables,
    }
    name = f"received-forensic-level-{int(level):02d}.json"
    _write_json(level_dir / name, archive)
    return name


def _persist_received_source(level_dir: Path, received, level: int) -> dict:
    """Preserve the RECEIVED raw source BEFORE identity admission.

    Schema-valid foreign-source data is saved through the PUBLIC
    ``save_hydro_dataset`` (its ``validate_hydro_dataset`` gate defines
    schema validity) under the same content-addressed raw name the
    admitted flow uses, so an identity-admission refusal (raw
    backend/version/center mismatch) leaves the received source on disk
    without marking it admitted — no level report is written for a
    refused level, so reload/adjudicate still refuse partial records.
    A schema-invalid dataset is preserved as a bounded, clearly
    labelled forensic archive and the ORIGINAL typed failure is
    re-raised unchanged. The raw source is never mutated.
    """
    if not isinstance(received, xr.Dataset):
        failure = SourceStudyWorkflowError(
            "backend solve must return an xarray Dataset")
        _write_received_forensic_archive(level_dir, received, level, failure)
        raise failure
    try:
        digest = dataset_content_hash(received)
        raw_path = level_dir / ("raw-" + digest + ".nc")
        save_hydro_dataset(received, raw_path)
    except Exception as exc:
        _write_received_forensic_archive(level_dir, received, level, exc)
        raise
    return {
        "kind": "saved_raw_dataset",
        "file": raw_path.name,
        "dataset_content_hash": digest,
    }


def _admit_dataset(case: SourceStudyCase, dataset: xr.Dataset, bodies,
                   request, level: int) -> dict:
    """Public fail-closed admission of one raw source dataset.

    Proves the ACTUAL raw dataset attrs backend/backend_version match
    the case-expected backend identity (capabilities alone are not
    proof) and the actual ordered center data variables match the fixed
    first-slice fixture, in addition to the conventions, qualification
    label, grids, request hash, mesh hashes and body identity.
    """
    _require(isinstance(dataset, xr.Dataset),
             SourceStudyWorkflowError(
                 "backend solve must return an xarray Dataset"))
    validate_for_time_domain(dataset)
    attrs = dict(dataset.attrs)
    for key, expected in (
        ("phasor_convention", PHASOR_CONVENTION),
        ("heading_convention", HEADING_CONVENTION),
        ("normal_convention", NORMAL_CONVENTION),
    ):
        _require(str(attrs.get(key)) == expected,
                 SourceStudyIdentityError(
                     f"force/geometric convention mismatch: dataset "
                     f"{key} is {attrs.get(key)!r}, expected {expected!r}"))
    _require(str(attrs.get("qualification_status")) == "unqualified",
             SourceStudyIdentityError(
                 "raw source claims qualification_status "
                 f"{attrs.get('qualification_status')!r}; development study "
                 "inputs must be honestly unqualified"))
    # the ACTUAL raw dataset attrs must match the case-expected backend
    # name/version: the capabilities checked before the solves are not
    # proof of what produced this dataset
    _require(str(attrs.get("backend")) == case.expected_backend_name,
             SourceStudyIdentityError(
                 f"raw source backend {attrs.get('backend')!r} does not "
                 f"match the case-expected "
                 f"{case.expected_backend_name!r}; refusing source "
                 "admission"))
    _require(str(attrs.get("backend_version"))
             == case.expected_backend_version,
             SourceStudyIdentityError(
                 f"raw source backend_version "
                 f"{attrs.get('backend_version')!r} does not match the "
                 f"case-expected {case.expected_backend_version!r}; "
                 "refusing source admission"))
    _require(bool(np.array_equal(
        np.asarray(dataset["omega"].values, dtype=np.float64),
        np.array(case.omegas, dtype=np.float64))),
        SourceStudyIdentityError(
            "grid mismatch: dataset omega grid does not match the case grid"))
    _require(bool(np.array_equal(
        np.asarray(dataset["wave_direction"].values, dtype=np.float64),
        np.array(case.wave_directions, dtype=np.float64))),
        SourceStudyIdentityError(
            "grid mismatch: dataset heading grid does not match the case "
            "grid"))
    _require(str(attrs.get("request_hash")) == request.content_hash(),
             SourceStudyIdentityError(
                 "control mismatch: dataset request_hash does not match the "
                 "case-built request"))
    _require(attrs.get("auxiliary_lid_policy", "free_surface_v1") == study_lid_policy(case),
             SourceStudyIdentityError("source auxiliary lid policy does not match the registered recipe"))
    expected_mesh = {b.body_id: b.mesh.mesh_hash for b in bodies}
    expected_lid = {
        b.body_id: (None if b.lid_mesh is None else b.lid_mesh.mesh_hash)
        for b in bodies
    }
    mesh_hashes = _decode_attr(attrs.get("mesh_hashes"))
    _require(mesh_hashes == expected_mesh,
             SourceStudyIdentityError(
                 f"geometric mismatch at level {level}: dataset mesh "
                 f"hashes {mesh_hashes} do not match the built bodies "
                 f"{expected_mesh}"))
    identity = _decode_attr(attrs.get("body_identity"))
    for b in bodies:
        entry = identity.get(b.body_id)
        _require(isinstance(entry, Mapping),
                 SourceStudyIdentityError(
                     f"body {b.body_id!r} missing from dataset body "
                     "identity"))
        _require(entry.get("mesh_hash") == b.mesh.mesh_hash
                 and entry.get("lid_mesh_hash") == expected_lid[b.body_id]
                 and [float(v) for v in entry.get("rotation_center", [])]
                 == [float(v) for v in b.rotation_center]
                 and [float(v) for v in entry.get("center_of_mass", [])]
                 == [float(v) for v in b.center_of_mass],
                 SourceStudyIdentityError(
                     f"body identity mismatch for {b.body_id!r}: the dataset "
                     "was not produced from the case bodies"))
    _check_fixture_center_arrays(case, dataset, bodies, level)
    return {
        "dataset_content_hash": dataset_content_hash(dataset),
        "mesh_hashes": expected_mesh,
        "request_hash": request.content_hash(),
        "source_backend": str(attrs.get("backend")),
        "source_backend_version": str(attrs.get("backend_version")),
    }


def _decode_attr(value) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _study_assumptions(case: SourceStudyCase) -> list:
    return [
        "development source study only; no scientific qualification is "
        "conferred by any outcome of this study",
        f"fixed geometry recipe {case.geometry_recipe}; cross-level "
        "comparison is justified by this shared recipe, never inferred "
        "from bounds",
        "zero-frequency limit not computed; A_inf is an explicit "
        "infinite-frequency source",
        f"normalization {case.normalization.name}: "
        f"{case.normalization.unit_convention}",
        f"source admission: {case.admission_declaration}",
        f"correction policy {case.correction_policy} applied only to the "
        "derived retained arrays; raw source arrays are never "
        "symmetrised, clamped or averaged on the raw route",
        f"refinement compared per frequency/heading against the finest "
        f"registered level {max(case.levels)} with per-quantity explicit "
        "near-zero floors in each quantity's own normalized unit",
        "the three-frequency two-level grid is not claimed adequate for "
        "radiation/RAO or time-domain qualification; those remain "
        "separate registered gates",
    ]


def _check_matched_levels(case: SourceStudyCase,
                          reference: DerivedHydroOperators,
                          candidate: DerivedHydroOperators) -> dict:
    """Only matched grids/conventions/recipe/normalization are compared."""
    _require(bool(np.array_equal(reference.omegas, candidate.omegas))
             and bool(np.array_equal(reference.wave_directions,
                                     candidate.wave_directions)),
             SourceStudyIdentityError(
                 "refinement comparison requires exactly matched omega and "
                 "heading grids"))
    _require(reference.source_dof_order == candidate.source_dof_order
             and reference.source_body_order == candidate.source_body_order,
             SourceStudyIdentityError(
                 "refinement comparison requires matched source DOF/body "
                 "order"))
    _require(reference.settings.to_dict() == candidate.settings.to_dict(),
             SourceStudyIdentityError(
                 "refinement comparison requires identical normalization, "
                 "policy and criteria"))
    _require(
        reference.provenance["phasor_convention"]
        == candidate.provenance["phasor_convention"]
        and reference.provenance["heading_convention"]
        == candidate.provenance["heading_convention"],
        SourceStudyIdentityError(
            "refinement comparison requires matched phasor/heading "
            "conventions"))
    return {
        "omegas": [float(v) for v in reference.omegas],
        "wave_directions": [float(v) for v in reference.wave_directions],
        "retained_dofs": list(case.retained_dofs),
        "normalization": case.normalization.to_dict(),
        "geometry_recipe": case.geometry_recipe,
    }


def execute_source_study(
    case: SourceStudyCase,
    backend,
    execution_limits,
    out_dir,
    *,
    should_cancel: Optional[Callable[[int], bool]] = None,
) -> dict:
    """Execute the registered study: at most ONE solve per level.

    Consumes the explicit case, checks caps/body identity/version BEFORE
    any backend work, saves content-addressed raw and derived artifacts
    per level (non-admitted artifacts are saved too), derives the
    retained operators through the PUBLIC API, compares only matched
    grids/conventions/recipe/normalization against the finest level and
    persists complete per-level and cross-level reports. Fail-closed on
    missing/version/corrupt/foreign-source/non-finite/inadmissible
    arrays; prior level and failure artifacts are retained; there is no
    automatic retry or campaign. ``should_cancel`` is consulted between
    owned stages (before each level); a cancellation preserves all
    evidence and records the ``cancelled`` outcome.
    """
    _require(isinstance(case, SourceStudyCase),
             SourceStudyCaseError("case must be a SourceStudyCase"))
    target = Path(out_dir)
    _require(target.is_dir(),
             SourceStudyWorkflowError(
                 f"study directory {target} does not exist; prepare first"))
    stored_case, case_hash = _load_case_record(target)
    _require(stored_case.content_hash() == case.content_hash(),
             SourceStudyIdentityError(
                 "the provided case does not match the prepared case record; "
                 "changed case/controls invalidate the study identity"))
    status = _load_status(target)
    _require(status["case_hash"] == case_hash,
             SourceStudyIdentityError(
                 "status record case hash mismatch; refusing"))
    _require(status["status"] == "not_executed",
             SourceStudyWorkflowError(
                 f"study directory status is {status['status']!r}; "
                 "re-execution over an existing study is refused (no "
                 "automatic retry or campaign; use a new exclusive "
                 "directory)"))

    identity = _check_backend_identity(case, backend, execution_limits)
    controller = _SolveController(backend, case)
    settings = study_operator_settings(case)
    log: List[dict] = [{"event": "identity_checks_passed", "utc": _utc_now(),
                        "detail": identity}]
    level_reports: Dict[str, dict] = {}
    artifacts: Dict[int, DerivedHydroOperators] = {}

    def _persist(outcome: str, refinement: Optional[dict]) -> None:
        _write_json(target / "study_report.json", {
            "format": SOURCE_STUDY_FORMAT,
            "module_version": SOURCE_STUDY_MODULE_VERSION,
            "case_hash": case_hash,
            "case_id": case.case_id,
            "revision": case.revision,
            "qualification_scope": case.qualification_scope,
            "outcome": outcome,
            "levels": level_reports,
            "refinement": refinement,
            "assumptions": _study_assumptions(case),
            "effective_controls": identity,
            "call_inventory": controller.call_log,
            "log": log,
        })
        _write_status(target,
                      "failed" if outcome == OUTCOME_FAILED
                      else ("cancelled" if outcome == OUTCOME_CANCELLED
                            else "executed"),
                      outcome, case_hash)

    for level in case.levels:
        try:
            if should_cancel is not None and should_cancel(int(level)):
                log.append({"event": "cancelled", "utc": _utc_now(),
                            "detail": {"before_level": int(level)}})
                _persist(OUTCOME_CANCELLED, None)
                return {
                    "outcome": OUTCOME_CANCELLED,
                    "levels": level_reports,
                    "case_hash": case_hash,
                }
            bodies = build_study_bodies(case, level)
            request = build_study_request(case)
            log.append({"event": "level_started", "utc": _utc_now(),
                        "detail": {"level": int(level),
                                   "panels": 24 * int(level) ** 2}})
            dataset = controller.solve(level, bodies, request)
            level_dir = _level_dir(target, level)
            level_dir.mkdir(parents=True, exist_ok=False)
            # RECEIVED source evidence is preserved BEFORE identity
            # admission: a raw backend/version/center mismatch must not
            # discard the source needed to diagnose the rejection
            received = _persist_received_source(level_dir, dataset, level)
            log.append({"event": "received_source_preserved",
                        "utc": _utc_now(),
                        "detail": {"level": int(level), **received}})
            admission = _admit_dataset(case, dataset, bodies, request, level)
            raw_path = level_dir / received["file"]
            raw_file_hash = _file_sha256(raw_path)
            artifact = derived_operators_from_dataset(dataset, settings)
            artifact_path = level_dir / (
                "derived-" + artifact.content_hash() + ".npz")
            save_derived_operators(artifact, artifact_path)
            # two-body coupling observability: truthful per-level metric
            # over the PUBLIC artifact arrays, stored SEPARATELY from the
            # artifact's own admission verdict (never relabels it)
            coupling = crossbody_coupling_report(artifact, case)
            level_report = {
                "level": int(level),
                "total_panels": 24 * int(level) ** 2,
                "mesh_hashes": admission["mesh_hashes"],
                "request_hash": admission["request_hash"],
                "dataset_content_hash": admission["dataset_content_hash"],
                "raw_file": raw_path.name,
                "raw_file_sha256": raw_file_hash,
                "derived_artifact": artifact_path.name,
                "derived_content_hash": artifact.content_hash(),
                "admitted": bool(artifact.admitted),
                "coupling": coupling,
                "qualification_status": artifact.qualification_status,
                "operator_report_summary": artifact.report.summary(),
                "source_backend": admission["source_backend"],
                "source_backend_version": admission["source_backend_version"],
                "solve_calls": 1,
            }
            _write_json(level_dir / "report.json", level_report)
            level_reports[str(int(level))] = level_report
            artifacts[int(level)] = artifact
            log.append({"event": "level_completed", "utc": _utc_now(),
                         "detail": {"level": int(level),
                                     "admitted": bool(artifact.admitted)}})
        except Exception as exc:  # fail closed, retain prior artifacts
            log.append({"event": "failed", "utc": _utc_now(),
                        "error": f"{type(exc).__name__}: {exc}"})
            _persist(OUTCOME_FAILED, None)
            raise

    # -- cross-level refinement against the finest registered level ------ #
    finest = max(case.levels)
    refinement: Dict[str, Any] = {
        "reference_level": int(finest),
        "mesh_rel_tol": case.refinement.mesh_rel_tol,
        "levels": {},
    }
    for level in case.levels:
        if int(level) == finest:
            continue
        report = compare_level_refinement(
            artifact_level_arrays(artifacts[finest]),
            artifact_level_arrays(artifacts[int(level)]),
            case.refinement,
            case.normalization.scales,
        )
        report["matched"] = _check_matched_levels(
            case, artifacts[finest], artifacts[int(level)])
        refinement["levels"][str(int(level))] = report

    reasons: List[str] = []
    for level in case.levels:
        entry = level_reports[str(int(level))]
        if not entry["admitted"]:
            reasons.append(
                f"level {level}: derived operators not admitted "
                "(reciprocity/correction/symmetry/PSD criteria)")
        if not entry["coupling"]["observable"]:
            reasons.append(
                f"level {level}: two-body coupling not observable (every "
                "finite-frequency retained A and B off-body block is "
                "negligible against the declared per-quantity normalized "
                "floors; a block-diagonal source cannot pass the fixed "
                "two-body physical case)")
    for name, report in refinement["levels"].items():
        if not report["mesh_pass"]:
            reasons.append(
                f"level {name}: mesh refinement criterion "
                f"{case.refinement.mesh_rel_tol} not met against the "
                f"finest level {finest}")
    outcome = (OUTCOME_SOURCE_ADMISSION_PASSED_DEVELOPMENT_ONLY
               if not reasons else OUTCOME_NO_GO)

    log.append({"event": "study_completed", "utc": _utc_now(),
                 "detail": {"outcome": outcome}})
    _persist(outcome, refinement)
    return {
        "outcome": outcome,
        "no_go_reasons": reasons,
        "levels": level_reports,
        "refinement": refinement,
        "case_hash": case_hash,
    }


def load_source_study(out_dir) -> dict:
    """Reload a study: case, status, raw datasets and derived artifacts.

    Every level artifact is verified with the PUBLIC full raw-source
    proof (``load_derived_operators(..., source_dataset=...)``
    rederives everything from the reloaded raw dataset; a matching
    recorded hash is necessary but never sufficient). Tampered, partial,
    corrupt, unknown or traversal-bearing records are refused.
    """
    target = Path(out_dir)
    _require(target.is_dir(),
             SourceStudyWorkflowError(
                 f"study directory {target} does not exist"))
    case, case_hash = _load_case_record(target)
    status = _load_status(target)
    _require(status["case_hash"] == case_hash,
             SourceStudyIdentityError(
                 "status record case hash does not match the case record"))
    expected_dirs = {_level_dir(target, n).name for n in case.levels}
    present_dirs = {p.name for p in target.iterdir() if p.is_dir()}
    unexpected = present_dirs - expected_dirs
    _require(not unexpected,
             SourceStudyWorkflowError(
                 f"study directory contains unknown level directory(ies) "
                 f"{sorted(unexpected)}; refusing a partial/foreign record"))
    if status["status"] == "not_executed":
        # prepared-only study: no level artifacts may exist yet
        _require(not (present_dirs & expected_dirs),
                 SourceStudyWorkflowError(
                     "the study is marked not_executed but contains level "
                     "artifacts; refusing an inconsistent record"))
        return {
            "case": case,
            "case_hash": case_hash,
            "status": status,
            "levels": {},
            "study_report": None,
        }
    levels: Dict[str, dict] = {}
    for level in case.levels:
        level_dir = _level_dir(target, level)
        _require(level_dir.is_dir(),
                 SourceStudyWorkflowError(
                     f"level directory {level_dir} is missing; refusing a "
                     "partial study"))
        report_path = level_dir / "report.json"
        _require(report_path.is_file(),
                 SourceStudyWorkflowError(
                     f"level report {report_path} is missing"))
        level_report = _read_json(report_path)
        _require(isinstance(level_report, Mapping),
                 SourceStudyWorkflowError(
                     "level report must be a JSON object"))
        for key in ("raw_file", "derived_artifact", "raw_file_sha256",
                    "dataset_content_hash", "derived_content_hash",
                    "admitted", "coupling"):
            _require(key in level_report,
                     SourceStudyWorkflowError(
                         f"level {level} report is missing field {key!r}; "
                         "refusing a partial record"))
        for key in ("raw_file", "derived_artifact"):
            part = str(level_report[key])
            _require("/" not in part and "\\" not in part
                     and ".." not in part,
                     SourceStudyWorkflowError(
                         "artifact file names must not contain path "
                         "separators (traversal refused)"))
        raw_path = level_dir / str(level_report["raw_file"])
        artifact_path = level_dir / str(level_report["derived_artifact"])
        _require(raw_path.is_file() and artifact_path.is_file(),
                 SourceStudyWorkflowError(
                     f"level {level} artifacts are missing; refusing a "
                     "partial study"))
        _require(_file_sha256(raw_path) == level_report["raw_file_sha256"],
                 SourceStudyIdentityError(
                     f"level {level} raw file bytes do not match the "
                     "recorded sha256; the artifact was modified"))
        dataset = load_hydro_dataset(raw_path)
        _require(dataset_content_hash(dataset)
                 == level_report["dataset_content_hash"],
                 SourceStudyIdentityError(
                     f"level {level} reloaded dataset content hash does not "
                     "match the recorded identity"))
        # reload proof: the same fail-closed admission identity checks
        # (raw backend/version attrs, fixed-fixture center arrays,
        # conventions, grids, request/mesh identity) are re-run on the
        # reloaded raw dataset against the deterministically rebuilt
        # bodies/request — a matching recorded hash is necessary but
        # never sufficient, and metadata attrs alone cannot prove the
        # actual arrays
        _admit_dataset(case, dataset, build_study_bodies(case, int(level)),
                       build_study_request(case), int(level))
        artifact = load_derived_operators(artifact_path,
                                          source_dataset=dataset)
        _require(artifact.content_hash()
                 == level_report["derived_content_hash"],
                 SourceStudyIdentityError(
                     f"level {level} derived artifact content hash does not "
                     "match the recorded identity"))
        levels[str(int(level))] = {
            "level": int(level),
            "level_report": dict(level_report),
            "dataset": dataset,
            "artifact": artifact,
        }
    study_report = None
    report_path = target / "study_report.json"
    if status["status"] != "not_executed":
        _require(report_path.is_file(),
                 SourceStudyWorkflowError(
                     f"study report {report_path} is missing for a "
                     f"{status['status']} study; refusing a partial record"))
        study_report = _read_json(report_path)
        _require(isinstance(study_report, Mapping),
                 SourceStudyWorkflowError(
                     "study report must be a JSON object"))
        _require(study_report.get("case_hash") == case_hash,
                 SourceStudyIdentityError(
                     "study report case hash mismatch; the report does not "
                     "belong to this case record"))
    return {
        "case": case,
        "case_hash": case_hash,
        "status": status,
        "levels": levels,
        "study_report": study_report,
    }


def adjudicate_source_study(out_dir) -> dict:
    """Adjudicate/reproduce a study's reports without any fresh BEM.

    Loads the saved case and artifacts (full public raw-source proof),
    recomputes admission and refinement from the reloaded public arrays
    and compares with the persisted reports. Any mismatch, missing
    level/array/report or tampered field is a typed refusal.
    """
    loaded = load_source_study(out_dir)
    case = loaded["case"]
    status = loaded["status"]
    if status["status"] == "not_executed":
        return {
            "outcome": OUTCOME_PREPARED_NOT_EXECUTED,
            "reproduced": True,
            "levels": {},
            "refinement": None,
            "case_hash": loaded["case_hash"],
        }
    _require(status["status"] in ("executed", "failed", "cancelled"),
             SourceStudyWorkflowError(
                 f"unknown study status {status['status']!r}; refusing"))
    levels = loaded["levels"]
    level_summaries: Dict[str, dict] = {}
    reasons: List[str] = []
    for name in sorted(levels):
        entry = levels[name]
        artifact = entry["artifact"]
        # the coupling predicate + metrics are RE-DERIVED from the
        # reloaded public artifact arrays; a stored bool or hash-only
        # bookkeeping change cannot masquerade as a pass
        coupling = crossbody_coupling_report(artifact, case)
        level_summaries[name] = {
            "level": entry["level"],
            "admitted": bool(artifact.admitted),
            "coupling": coupling,
            "qualification_status": artifact.qualification_status,
            "operator_report_summary": artifact.report.summary(),
        }
        _require(entry["level_report"]["admitted"] == bool(artifact.admitted),
                 SourceStudyIdentityError(
                     f"level {name}: persisted admission outcome does not "
                     "match the recomputed artifact state"))
        _require("coupling" in entry["level_report"],
                 SourceStudyWorkflowError(
                     f"level {name} report is missing the coupling "
                     "observability record; refusing a partial record"))
        _require(
            _canonical_json(entry["level_report"]["coupling"])
            == _canonical_json(coupling),
            SourceStudyIdentityError(
                f"level {name}: persisted coupling observability record "
                "does not match the recomputed off-body block metrics; "
                "the record was changed"))
        if not bool(artifact.admitted):
            reasons.append(f"level {name}: derived operators not admitted")
        if not coupling["observable"]:
            reasons.append(
                f"level {name}: two-body coupling not observable (every "
                "finite-frequency retained A and B off-body block is "
                "negligible against the declared per-quantity normalized "
                "floors; a block-diagonal source cannot pass the fixed "
                "two-body physical case)")
    artifacts = {entry["level"]: entry["artifact"]
                 for entry in levels.values()}
    finest = max(case.levels)
    refinement: Dict[str, Any] = {
        "reference_level": int(finest),
        "mesh_rel_tol": case.refinement.mesh_rel_tol,
        "levels": {},
    }
    for level in case.levels:
        if int(level) == finest:
            continue
        _require(int(level) in artifacts,
                 SourceStudyWorkflowError(
                     f"level {level} is required by the case inventory but "
                     "absent from the study; a missing level prevents "
                     "study-pass"))
        report = compare_level_refinement(
            artifact_level_arrays(artifacts[finest]),
            artifact_level_arrays(artifacts[int(level)]),
            case.refinement,
            case.normalization.scales,
        )
        report["matched"] = _check_matched_levels(
            case, artifacts[finest], artifacts[int(level)])
        refinement["levels"][str(int(level))] = report
        if not report["mesh_pass"]:
            reasons.append(
                f"level {level}: mesh refinement criterion not met against "
                f"the finest level {finest}")
    persisted = loaded["study_report"]
    if persisted.get("refinement") is not None:
        _require(_canonical_json(persisted["refinement"])
                 == _canonical_json(refinement),
                 SourceStudyIdentityError(
                     "persisted cross-level refinement report does not match "
                     "the recomputed residuals; the record was changed"))
    if status["status"] == "cancelled":
        outcome = OUTCOME_CANCELLED
    elif status["status"] == "failed":
        outcome = OUTCOME_FAILED
    else:
        outcome = (OUTCOME_SOURCE_ADMISSION_PASSED_DEVELOPMENT_ONLY
                   if not reasons else OUTCOME_NO_GO)
        _require(persisted.get("outcome") == outcome,
                 SourceStudyIdentityError(
                     "persisted study outcome does not match the "
                     "recomputed adjudication"))
    return {
        "outcome": outcome,
        "reproduced": True,
        "no_go_reasons": reasons,
        "levels": level_summaries,
        "refinement": refinement,
        "case_hash": loaded["case_hash"],
        "qualification_scope": case.qualification_scope,
    }


# --------------------------------------------------------------------------- #
# module CLI
# --------------------------------------------------------------------------- #

#: Typed module-CLI exit codes. Success (0) is reserved for the full
#: ``source_admission_passed_development_only`` outcome — which still
#: confers NO scientific qualification. Every no_go/failed/cancelled
#: outcome and any prepared/incomplete adjudication is a distinct nonzero
#: code, so no CI gate can go green on missing criteria while the
#: reports/artifacts are retained on disk.
CLI_EXIT_OK = 0
CLI_EXIT_ERROR = 1
CLI_EXIT_FAILED = 2
CLI_EXIT_CANCELLED = 3
CLI_EXIT_NO_GO = 4
CLI_EXIT_INCOMPLETE = 5

_OUTCOME_EXIT_CODES = {
    OUTCOME_SOURCE_ADMISSION_PASSED_DEVELOPMENT_ONLY: CLI_EXIT_OK,
    OUTCOME_NO_GO: CLI_EXIT_NO_GO,
    OUTCOME_FAILED: CLI_EXIT_FAILED,
    OUTCOME_CANCELLED: CLI_EXIT_CANCELLED,
    OUTCOME_PREPARED_NOT_EXECUTED: CLI_EXIT_INCOMPLETE,
}


def _cli_prepare(args) -> int:
    if args.case_json is not None:
        record = _read_json(Path(args.case_json))
        _require(isinstance(record, Mapping) and "case" in record,
                 SourceStudyCaseError(
                     "case JSON must be a prepared case record object"))
        case = SourceStudyCase.from_dict(record["case"])
    else:
        case = make_prospective_source_study_case()
    result = prepare_source_study(case, args.out)
    print(json.dumps(result, sort_keys=True, indent=2))
    # prepare succeeds with the explicit not_executed outcome; no
    # scientific metric is claimed
    return CLI_EXIT_OK


def _cli_execute(args) -> int:
    from .capytaine_backend import (  # lazy: only the explicit execute path
        CapytaineBackend,
        CapytaineExecutionLimits,
    )
    loaded = load_source_study(args.out)
    case = loaded["case"]
    _require(args.backend == "capytaine",
             SourceStudyWorkflowError(
                 "execute requires the explicit --backend capytaine choice; "
                 "no implicit backend is ever constructed"))
    # case-derived explicit caps through the PUBLIC execution limits and
    # backend constructor (the real production path; the 100-panel DEFAULT
    # would refuse the registered levels, explicit limits admit them up
    # to the adapter's absolute 2048-panel software ceiling)
    limits = CapytaineExecutionLimits(
        max_total_panels=case.max_total_panels(),
        max_finite_omegas=len(case.omegas),
        max_wave_directions=len(case.wave_directions),
        n_jobs=case.n_jobs,
        n_threads=case.n_threads,
    )
    backend = CapytaineBackend(execution_limits=limits)
    should_cancel = None
    if args.cancel_before_level is not None:
        cancel_level = _int(args.cancel_before_level, "cancel_before_level")
        _require(cancel_level in {int(n) for n in case.levels},
                 SourceStudyWorkflowError(
                     f"--cancel-before-level {cancel_level} is not a "
                     f"registered level {list(case.levels)}"))
        should_cancel = lambda level: int(level) >= cancel_level  # noqa: E731
    try:
        result = execute_source_study(case, backend, limits, args.out,
                                      should_cancel=should_cancel)
    except Exception as exc:  # the workflow persisted the failure evidence
        print(f"source-study execute failed ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return CLI_EXIT_FAILED
    print(json.dumps({k: v for k, v in result.items()
                      if k != "refinement"},
                     sort_keys=True, indent=2, default=str))
    return _OUTCOME_EXIT_CODES[result["outcome"]]


def _cli_adjudicate(args) -> int:
    result = adjudicate_source_study(args.out)
    print(json.dumps(result, sort_keys=True, indent=2, default=str))
    # adjudicating a prepared/incomplete study is an explicit nonzero
    # outcome: it must not pretend an actual source pass
    return _OUTCOME_EXIT_CODES[result["outcome"]]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Module CLI: prepare / execute / adjudicate.

    ``execute`` requires the explicit ``--backend capytaine`` choice; the
    choice expresses API intent but grants no scientific qualification,
    and this development slice carries no BEM execution authority. The
    CLI is a gate: only the full
    ``source_admission_passed_development_only`` outcome exits 0;
    no_go/failed/cancelled/incomplete outcomes and typed refusals exit
    with distinct nonzero codes while reports and artifacts are kept.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m anyhydro.source_study",
        description="First-slice source-quality mesh study (development "
                    "tooling; no scientific qualification)")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser(
        "prepare", help="write the immutable case + inventory to an "
                        "exclusive new directory (no backend is imported)")
    prep.add_argument("--out", required=True)
    prep.add_argument("--case-json", default=None,
                      help="optional prepared case record JSON (default: "
                           "the prospective development case)")
    prep.set_defaults(func=_cli_prepare)

    exe = sub.add_parser(
        "execute", help="execute the prepared study through an explicitly "
                        "selected public backend (one solve per level)")
    exe.add_argument("--out", required=True)
    exe.add_argument("--backend", default=None,
                    help="explicit backend choice (only 'capytaine')")
    exe.add_argument("--cancel-before-level", default=None, type=int,
                    help="cancel before this registered level (explicit "
                         "user intent; evidence is preserved)")
    exe.set_defaults(func=_cli_execute)

    adj = sub.add_parser(
        "adjudicate", help="reload and reproduce the study reports "
                           "without any fresh BEM")
    adj.add_argument("--out", required=True)
    adj.set_defaults(func=_cli_adjudicate)

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except SourceStudyError as exc:
        # typed refusal (missing/corrupt/foreign/tampered record, wrong
        # backend choice): nonzero, no fabricated success
        print(f"source-study CLI refusal ({type(exc).__name__}: {exc})",
              file=sys.stderr)
        return CLI_EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
