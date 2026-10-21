# Portable base CI selection

This document defines what the ANYhydro GitHub base CI job
(`.github/workflows/ci.yml`, Windows/Linux, Python 3.11/3.13) runs, why some
test modules are excluded, and which evidence obligations remain. The machine
readable manifest lives in `tools/run_portable_checks.py` and is enforced by
`tests/test_portable_ci_contract.py`.

## Scope

The base package declares only `numpy`, `scipy` and `xarray`. The `dev` extra
additionally provides `pytest`, `build` and `threadpoolctl>=3.6`: the panel
integration candidate imports `threadpoolctl` at solve time to pin native
thread pools, so a clean `.[dev]` install must provide it. Base CI therefore
verifies the portable neutral contracts:
records, schema, coordinates, frame maps, hydrostatics, persistence,
transforms, radiation, operators, conventions, ownership, surfaces, and the
eligible mock-only public workflows. It installs `.[dev]` only — no optional
ecosystem packages.

## Portable legacy oracle

Golden legacy content hashes bind exact artifact bytes. Recomputed analytic
fixtures and kernel quadrature are not bit-stable across numeric environments
(demonstrated 2026-10-10: with identical numpy/scipy/xarray versions, the
two-body fixture differs in `excitation_force` between Windows and Linux, and
the full-12 kernel hash differs across three environments from identical
dataset bytes). The literal golden hashes are therefore verified against
frozen pre-refactor artifact bytes under `tests/fixtures/legacy-compatibility/`
(see `provenance.json` there and `docs/PLATFORM_CI_REPAIR.md`), while the live
generators keep their independent numerical checks and tolerances in the tests.

A green base CI run is base-CI evidence only. Scientific qualification gates,
real backend artifacts and optional ecosystem integration are distinct
evidence obligations and remain mandatory before those capabilities are
claimed; they are not silently waived by a green base CI.

## Mechanism

`tools/run_portable_checks.py` holds an explicit module allowlist and runs
`pytest` on exactly those files. There is no `-k` expression: every collected
test in the selected modules runs and no assertion is filtered out. Selection
changes are explicit edits to the allowlist, verified by the contract test.

To make the selection behave identically on a clean base install and on a
developer runtime that has optional packages installed, the runner installs
import blockers for the optional ecosystem packages (`anyloads`, `anyfileio`,
`capytaine`) and for the licensed OrcaFlex API `OrcFxAPI` (the runner never executes
licensed simulations; tests that need it skip through `pytest.importorskip`) and blocks their `importlib.metadata` lookups before invoking
pytest. Mock-only tests inject their doubles into `sys.modules` before
importing, which the blockers do not interfere with, so accidental real
optional/native execution fails loudly instead of silently passing locally.

## Selected modules (portable base CI)

| Module | Coverage |
| --- | --- |
| test_artifact_bound_panel_profile | artifact-bound panel profile contracts (mock Capytaine) |
| test_backends | backend diagnostics/admission contracts |
| test_cache_identity_repair | cache identity repair (mocked solve, no real backend) |
| test_check_energy_artifact | mocked energy artifact route |
| test_check_operator_artifact | mocked operator probe route |
| test_coefficient_frames | coefficient frames and reference maps |
| test_derived_operators | derived operators and admission/refusal |
| test_execution_limits | execution limit validation (mocked solves) |
| test_execution_limits_persistence_hash | limits persistence hashing (no backend objects) |
| test_first_slice_fixture | first-slice fixture contracts |
| test_fixtures | development fixture contracts |
| test_hams_producer | producer build identity/lifecycle (synthetic Python fixture, never native HAMS) |
| test_hams_reference | signed public reference fixtures (no native execution) |
| test_hashing | dataset/request/mesh hashing |
| test_hydrostatic_contract | hydrostatic composition contracts |
| test_imports | base-import isolation (no optional Capytaine/Qt at import time) |
| test_linux_owned_process | Linux process ownership (skips on Windows) |
| test_mean_drift | far-field mean drift formulas, FFT derivative, MacCamy-Fuchs/Havelock cylinder anchor (scipy, no BEM) |
| test_orcaflex_export | OrcaFlex vessel-type export data, conventions and refusals (pure); the OrcFxAPI round-trip and end-to-end checks always skip under this runner (OrcFxAPI is blocked) and run only under a plain pytest call on a licensed machine; the read-back verifier is covered by pure writer-double tests |
| test_owned_process | owned process lifecycle (stdlib subprocess) |
| test_panel_accuracy_reference | panel accuracy reference contracts (mock Capytaine) |
| test_panel_integration_candidate | panel integration candidate admission (mocked backend) |
| test_persistence | persistence round trips |
| test_slow_drift | Newman difference-frequency QTF from the far-field mean drift (numpy/xarray only, no BEM) |
| test_portable_ci_contract | this selection contract |
| test_radiation | radiation kernels and admission |
| test_radiation_full_block | band-consistent A_inf, body-diagonal tail and full-block reconstruction criteria |
| test_radiation_high_tail | declared power-law high-frequency kernel tail |
| test_radiation_long_kernel | long radiation kernel quadrature |
| test_basis_sym_golden | hash guard: absolute-basis, unsymmetrised recipes/settings keep their hashes |
| test_kernel_symmetrisation | declared producer-side kernel symmetrisation: symmetric part, recorded bounded skew, hash binding, fail-closed record |
| test_vessel_study_basis | tolerance basis and kernel symmetrisation in the vessel-study recipe (the Capytaine end-to-end tests skip without the extra) |
| test_radiation_tolerance_basis | absolute vs per-sample relative kernel tolerance basis and its diagnostics binding |
| test_radiation_oscillatory | oscillatory radiation quadrature |
| test_radiation_parent_review | radiation parent review contracts |
| test_reciprocity | pairwise reciprocity diagnostic of radiation matrices (owner decision 2026-10-13) |
| test_records | neutral record contracts |
| test_review_repairs | review repair contracts |
| test_saved_panel_diagnostics | saved panel diagnostics (mock backend) |
| test_schema | schema validation |
| test_source_study | source-quality mesh study (mock-rehearsal only) |
| test_submerged_lid | submerged lid policy (mocked solve) |
| test_surfaces | thin-sheet surface semantics |
| test_transform | coordinate transforms |
| test_vessel_study | vessel source-study recipe identity, immutability and strictness (no BEM) |
| test_far_field_weak_modes | far-field damping and the far-field weak-mode operator policy; backend check skips without Capytaine |
| test_finite_depth_john_oracle | independent John-series finite-depth oracle; actual-record checks skip without local evidence |
| test_physical_green_field | physical Green-field discriminator (mocked native route) |
| test_green_function | exact propagating imaginary part of the Green function; real-Capytaine checks skip without the extra |
| test_prony_fit | validated finite-depth Prony decomposition; real-Capytaine checks skip without the extra |
| test_retained_block_criteria | W5 retained-block criteria R1-R8; actual 03c check skips without local evidence |
| test_retained_panel_energy_audit | retained panel energy audit tool |
| test_retained_panel_pairs | retained panel-pair quadrature discriminator |
| test_retained_weak_neumann | retained-field weak Neumann experiment (guarded) |
| test_thin_sheet_galerkin | linear Galerkin thin-sheet panels, plan TS2: analytic triangle 1/R, thin-disc added mass |
| test_thin_sheet_galerkin_fs | free-surface Galerkin sheets, plan TS2: complex load vector, exact wave Hessian (Bessel integrals), image kernel, blocked assembly, near-surface refusal; Capytaine-dependent checks skip without the extra |
| test_thin_sheet_kernels | thin-sheet (dipole panel) kernels, plan TS1: exact vortex-ring Rankine operator; real-Capytaine wave/image checks skip without the extra |

## Excluded modules and remaining gates

| Module | Concrete dependency/fixture reason | Remaining gate |
| --- | --- | --- |
| test_build_e1_backend | reads gitignored local `.development-evidence/w5-isolated-backend-20261010` upstream Capytaine Fortran sources at collection time; the evidence is deliberately not published | isolated backend build and qualification campaign with published sources |
| test_capytaine_backend | `TestDirectCapytaineComparison` executes real Capytaine solves needing the optional `anyhydro[capytaine]` extra | backend qualification against real Capytaine artifacts |
| test_direct_finite_depth | real Capytaine direct solves needing the optional `anyhydro[capytaine]` extra; uses the John-series oracle in `tools/` | direct-method finite-depth qualification against real Capytaine solves |
| test_direct_far_field | real Capytaine direct/indirect solves needing the optional `anyhydro[capytaine]` extra; imports `test_capytaine_backend` | direct-method far-field qualification against real Capytaine solves |
| test_hydrostatic_contract_backend | real Capytaine hydrostatics and solves needing the optional `anyhydro[capytaine]` extra; imports `test_capytaine_backend` | ADR-015 L6-4 hydrostatic contract evidence against real Capytaine solves |
| test_vessel_study_run | real Capytaine solves needing the optional `anyhydro[capytaine]` extra; imports `test_capytaine_backend` | ADR-015 L6-4 vessel source campaign after the ANYmarine L6-F2 gate |
| test_mean_drift_export | real Capytaine Kochin-export solves needing the optional `anyhydro[capytaine]` extra; imports `test_capytaine_backend` | MD2 mean-drift validation evidence (V2-V5) against real Capytaine solves |
| test_excitation_source | module-level `from anyloads import RegularWave` needs the optional `anyhydro[sea-state]` extra | sea-state integration qualification with ANYloads |
| test_drift_series | module-level ANYloads imports need the optional `anyhydro[sea-state]` extra | N4 slow-drift evidence (Newman vs OrcaWave full QTFs, moored irregular-sea run) |
| test_sea_excitation | module-level ANYloads imports need the optional `anyhydro[sea-state]` extra | sea-state integration qualification with ANYloads |
| test_hams_backend | module-level `anyfileio.hams` import plus stored native HAMS output fixtures; native producers must never be executed to make unit tests pass | HAMS lifecycle qualification with stored outputs and ANYfileio |
| test_hams_field_backend | imports `test_hams_backend` (ANYfileio plus stored native outputs) | HAMS field lifecycle qualification |
| test_hams_fields | imports `test_hams_import` (ANYfileio) | HAMS field import qualification with ANYfileio |
| test_hams_import | module-level `anyfileio.hams` import needs the optional `anyhydro[wamit]` extra | HAMS import qualification with ANYfileio |
| test_hams_inputs | module-level `anyfileio.hams` import needs the optional `anyhydro[wamit]` extra | HAMS input qualification with ANYfileio |
| test_hams_limit_backend | imports `test_hams_backend` (ANYfileio plus stored native outputs) | HAMS limit lifecycle qualification |
| test_hams_limits | imports `test_hams_import` (ANYfileio) | HAMS limit import qualification with ANYfileio |
| test_process_receipt_metadata | imports `test_hams_backend` (ANYfileio plus stored native outputs) | HAMS process receipt qualification |
| test_hams_gravity_recipe | requires `HAMS_LIMIT_PRISTINE_SOURCE` pointing at a pristine local HAMS native source tree that is not published | native source recipe campaign with published sources |
| test_hams_phase_recipe | requires `HAMS_LIMIT_PRISTINE_SOURCE` pointing at a pristine local HAMS native source tree that is not published | native source recipe campaign with published sources |
| test_wamit_import | exercises the real `anyfileio.wamit` parser inside `import_wamit7` (only the typed-failure test mocks it), needing the optional `anyhydro[wamit]` extra | WAMIT import qualification with ANYfileio |

## Local verification

The same runner is used locally with the existing Windows runtime; because
that runtime also has the optional packages installed, the import blockers
provide the clean-install proof for the selection. `--slice N/M` splits the
ordered allowlist into deterministic chunks so receipt runs fit bounded
per-process check budgets; the union of the slices is exactly the allowlist
(verified by the contract test). GitHub's effective clean environment is the
final authority and is inspected after push.
