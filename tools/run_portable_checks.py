"""Portable base-CI selection for the neutral anyhydro package.

Runs exactly the allowlisted test modules with the optional ecosystem packages
(ANYloads, ANYfileio, Capytaine) made unimportable, so the selection behaves
identically on a clean base install (numpy/scipy/xarray + pytest, the declared
base dependencies) and on a developer runtime that also has the optional
packages installed.  Mock-only tests inject their doubles into ``sys.modules``
before importing, which the blockers do not interfere with.

The allowlist below is the single source of truth for the GitHub base CI job
(.github/workflows/ci.yml) and is verified by
tests/test_portable_ci_contract.py.  No ``-k`` expression is used: every test
collected from the selected modules runs, and no assertion is filtered out.

A green run of this selection is base-CI evidence only.  Scientific
qualification gates, real backend artifacts and optional ecosystem
integration are distinct evidence obligations and are NOT waived by it.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Optional ecosystem packages that a clean base install does not provide.
BLOCKED_OPTIONAL_PACKAGES = ("anyloads", "anyfileio", "capytaine")

# Test modules selected for portable base CI.  Every module runs with only the
# declared base dependencies; backend-dependent paths are mock-only.
PORTABLE_TEST_MODULES = (
    "test_artifact_bound_panel_profile",
    "test_backends",
    "test_cache_identity_repair",
    "test_check_energy_artifact",
    "test_check_operator_artifact",
    "test_coefficient_frames",
    "test_derived_operators",
    "test_execution_limits",
    "test_execution_limits_persistence_hash",
    "test_far_field_weak_modes",
    "test_finite_depth_john_oracle",
    "test_first_slice_fixture",
    "test_fixtures",
    "test_green_function",
    "test_hams_producer",
    "test_hams_reference",
    "test_hashing",
    "test_hydrostatic_contract",
    "test_imports",
    "test_linux_owned_process",
    "test_mean_drift",
    "test_orcaflex_export",
    "test_owned_process",
    "test_panel_accuracy_reference",
    "test_panel_integration_candidate",
    "test_persistence",
    "test_physical_green_field",
    "test_slow_drift",
    "test_portable_ci_contract",
    "test_prony_fit",
    "test_radiation",
    "test_radiation_full_block",
    "test_radiation_high_tail",
    "test_radiation_long_kernel",
    "test_radiation_tolerance_basis",
    "test_radiation_oscillatory",
    "test_radiation_parent_review",
    "test_reciprocity",
    "test_records",
    "test_retained_panel_energy_audit",
    "test_retained_block_criteria",
    "test_retained_panel_pairs",
    "test_retained_weak_neumann",
    "test_review_repairs",
    "test_saved_panel_diagnostics",
    "test_schema",
    "test_source_study",
    "test_submerged_lid",
    "test_surfaces",
    "test_thin_sheet_galerkin",
    "test_thin_sheet_galerkin_fs",
    "test_thin_sheet_kernels",
    "test_transform",
    "test_vessel_study",
)

# Test modules excluded from portable base CI, each with its concrete
# dependency/fixture reason and the evidence gate that still applies.
EXCLUDED_TEST_MODULES = {
    "test_build_e1_backend": (
        "reads gitignored local .development-evidence/w5-isolated-backend-20261010 "
        "upstream Capytaine Fortran sources at collection time; the evidence is "
        "deliberately not published. Remaining gate: isolated backend build and "
        "qualification campaign with published sources."
    ),
    "test_capytaine_backend": (
        "TestDirectCapytaineComparison executes real Capytaine solves, which need "
        "the optional anyhydro[capytaine] extra. Remaining gate: backend "
        "qualification against real Capytaine artifacts."
    ),
    "test_direct_far_field": (
        "real Capytaine direct/indirect solves (optional anyhydro[capytaine] extra) and "
        "imports test_capytaine_backend. Remaining gate: direct-method far-field "
        "qualification against real Capytaine solves."
    ),
    "test_direct_finite_depth": (
        "real Capytaine direct solves (optional anyhydro[capytaine] extra) and the "
        "John-series oracle in tools/. Remaining gate: direct-method finite-depth "
        "qualification against real Capytaine solves."
    ),
    "test_hydrostatic_contract_backend": (
        "real Capytaine hydrostatics and solves (optional anyhydro[capytaine] extra) and "
        "imports test_capytaine_backend. Remaining gate: ADR-015 L6-4 hydrostatic contract "
        "evidence against real Capytaine solves."
    ),
    "test_vessel_study_run": (
        "real Capytaine solves (optional anyhydro[capytaine] extra) and imports "
        "test_capytaine_backend. Remaining gate: ADR-015 L6-4 vessel source campaign "
        "after the ANYmarine L6-F2 gate."
    ),
    "test_mean_drift_export": (
        "real Capytaine Kochin-export solves (optional anyhydro[capytaine] extra) and "
        "imports test_capytaine_backend. Remaining gate: MD2 mean-drift validation "
        "evidence (V2-V5) against real Capytaine solves."
    ),
    "test_excitation_source": (
        "module-level 'from anyloads import RegularWave' needs the optional "
        "anyhydro[sea-state] extra. Remaining gate: sea-state integration "
        "qualification with ANYloads."
    ),
    "test_drift_series": (
        "module-level ANYloads imports (AiryComponent, WaveRealization) need the optional "
        "anyhydro[sea-state] extra. Remaining gate: N4 slow-drift evidence (Newman vs OrcaWave "
        "full QTFs on the barge, moored irregular-sea run)."
    ),
    "test_sea_excitation": (
        "module-level ANYloads imports (RegularWave, WaveRealization, JonswapSpectrum, "
        "DirectionalSpectrum) need the optional anyhydro[sea-state] extra. Remaining "
        "gate: sea-state integration qualification with ANYloads."
    ),
    "test_hams_backend": (
        "module-level anyfileio.hams import plus stored native HAMS output fixtures; "
        "native producers must never be executed to make unit tests pass. Remaining "
        "gate: HAMS lifecycle qualification with stored outputs and ANYfileio."
    ),
    "test_hams_field_backend": (
        "imports test_hams_backend, which needs anyfileio and stored native HAMS "
        "output fixtures. Remaining gate: HAMS field lifecycle qualification."
    ),
    "test_hams_fields": (
        "imports test_hams_import, which needs anyfileio. Remaining gate: HAMS "
        "field import qualification with ANYfileio."
    ),
    "test_hams_import": (
        "module-level anyfileio.hams import needs the optional anyhydro[wamit] "
        "extra. Remaining gate: HAMS import qualification with ANYfileio."
    ),
    "test_hams_inputs": (
        "module-level anyfileio.hams import needs the optional anyhydro[wamit] "
        "extra. Remaining gate: HAMS input qualification with ANYfileio."
    ),
    "test_hams_limit_backend": (
        "imports test_hams_backend, which needs anyfileio and stored native HAMS "
        "output fixtures. Remaining gate: HAMS limit lifecycle qualification."
    ),
    "test_hams_limits": (
        "imports test_hams_import, which needs anyfileio. Remaining gate: HAMS "
        "limit import qualification with ANYfileio."
    ),
    "test_process_receipt_metadata": (
        "imports test_hams_backend, which needs anyfileio and stored native HAMS "
        "output fixtures. Remaining gate: HAMS process receipt qualification."
    ),
    "test_hams_gravity_recipe": (
        "requires the HAMS_LIMIT_PRISTINE_SOURCE environment variable pointing at "
        "a pristine local HAMS native source tree that is not published. Remaining "
        "gate: native source recipe campaign with published sources."
    ),
    "test_hams_phase_recipe": (
        "requires the HAMS_LIMIT_PRISTINE_SOURCE environment variable pointing at "
        "a pristine local HAMS native source tree that is not published. Remaining "
        "gate: native source recipe campaign with published sources."
    ),
    "test_wamit_import": (
        "exercises the real anyfileio.wamit parser inside import_wamit7 (only the "
        "typed-failure test mocks it), needing the optional anyhydro[wamit] extra. "
        "Remaining gate: WAMIT import qualification with ANYfileio."
    ),
}


class _OptionalDependencyBlocker:
    """Meta-path finder that makes blocked optional packages unimportable.

    Modules already present in ``sys.modules`` (for example mock doubles that
    tests inject) are returned by the import system without consulting
    finders, so mock-only tests are unaffected.  Real imports fail with
    ModuleNotFoundError exactly as on a clean base install.
    """

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        if root in BLOCKED_OPTIONAL_PACKAGES:
            raise ModuleNotFoundError(
                f"portable base CI selection must not import optional package {root!r}"
            )
        return None


def _blocked_distribution_name(name) -> bool:
    normalized = str(name).lower().replace("_", "-").split(" ")[0]
    return normalized in BLOCKED_OPTIONAL_PACKAGES


def _blocked_module_name(name) -> bool:
    return str(name).split(".")[0] in BLOCKED_OPTIONAL_PACKAGES


def install_optional_dependency_blockers() -> None:
    """Simulate a clean base install for optional ecosystem packages."""
    sys.meta_path.insert(0, _OptionalDependencyBlocker())
    import importlib.metadata as metadata
    import importlib.util

    original_version = metadata.version
    original_distribution = metadata.distribution
    original_find_spec = importlib.util.find_spec

    def version(name, *args, **kwargs):
        if _blocked_distribution_name(name):
            raise metadata.PackageNotFoundError(name)
        return original_version(name, *args, **kwargs)

    def distribution(name, *args, **kwargs):
        if _blocked_distribution_name(name):
            raise metadata.PackageNotFoundError(name)
        return original_distribution(name, *args, **kwargs)

    def find_spec(name, *args, **kwargs):
        if _blocked_module_name(name):
            return None
        return original_find_spec(name, *args, **kwargs)

    metadata.version = version
    metadata.distribution = distribution
    importlib.util.find_spec = find_spec


def select_modules(slice_spec: str | None = None) -> list[str]:
    """Return the ordered selection, optionally one deterministic slice of it."""
    modules = list(PORTABLE_TEST_MODULES)
    if not slice_spec:
        return modules
    index_text, _, total_text = slice_spec.partition("/")
    index, total = int(index_text), int(total_text)
    if not 1 <= index <= total:
        raise SystemExit(f"--slice expects N/M with 1 <= N <= M, got {slice_spec!r}")
    size, remainder = divmod(len(modules), total)
    start = (index - 1) * size + min(index - 1, remainder)
    stop = start + size + (1 if index <= remainder else 0)
    return modules[start:stop]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    slice_spec = None
    if "--list" in args:
        for name in select_modules():
            print(name)
        return 0
    if "--slice" in args:
        position = args.index("--slice")
        slice_spec = args[position + 1]
        del args[position : position + 2]
    modules = select_modules(slice_spec)
    print(
        f"portable base CI selection: {len(modules)} of {len(PORTABLE_TEST_MODULES)} "
        f"allowlisted modules, {len(EXCLUDED_TEST_MODULES)} excluded with documented "
        f"reasons; blocked optional packages: {', '.join(BLOCKED_OPTIONAL_PACKAGES)}"
    )
    install_optional_dependency_blockers()
    paths = [str(REPO_ROOT / "tests" / f"{name}.py") for name in modules]
    exit_code = pytest.main(["-q", "-p", "no:cacheprovider", *paths, *args])
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
