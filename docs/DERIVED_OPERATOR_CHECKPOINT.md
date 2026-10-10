# Derived operator checkpoint (worker, 2026-10-08)

## Repair round (second worker, 2026-10-08) - early checkpoint

Parent review (derived-operator-parent-review-01.md, evidence
derived-operator-parent-01) confirmed read: all three parent
counterexamples fail against the initial implementation (normalized-PSD
gate, hash-only source proof, missing producing source controls). Repair
design fixed, implementation starting; no parent failure rerun before
repairs, per brief. Scope unchanged: operators/radiation/tests/docs only,
no siblings, no BEM/dynamics, no Git. Budget: 102.487697900013 s
remaining of the original 120 s (17.0003020999866 s measured + 0.512 s
conservative static charge), 20 s final-verification reserve, all checks
via the exact harness (run_check.py, per-process <= 15 s, sequential),
evidence derived-operator-worker-06 onward. Planned repairs:

1. PSD admission on normalized Bhat = S^T B S (finite thresholds in
   normalized coordinates); complete eigenbasis plus all negative modes;
   SI mechanic arrays unchanged; normalized vector -> physical direction
   S v documented; raw-full eig/alignment relabelled informational
   unscaled mixed-SI diagnostics.
2. matches_raw_source becomes a full rederivation proof (typed
   verify_raw_source; predicate returns False): revalidate, hash, then
   recompute P, all projected arrays, report and provenance under the
   immutable stored settings and compare exactly. Optional
   source_dataset keyword on the loader and on both radiation
   consumers; consumers holding the raw source MUST invoke the proof.
3. Complete verbatim source attrs (backend_settings/method/rho/g/
   water_depth/units as present, nothing invented) plus relevant
   variable attrs preserved in provenance; constructor/reader enforce
   12-DOF/513-frequency caps and full-6N body/DOF-order consistency
   before freezing effective arrays; load failures typed.
4. Complete frequency-indexed reciprocity/correction residuals and
   thresholds in the immutable report; maxima demoted to summaries.
5. Criteria documented as operating in each operator's own normalized
   units; no scientific thresholds or data changed; the full-12
   radiation kernel baseline hash stays authoritative and untouched.

Artifact format bumped to anyhydro-derived-operators-v2 (module 1.1.0):
the report field set changes, so old files must be refused by version,
not silently misread.

---

Early checkpoint per assignment. Status: inspection complete, design fixed,
implementation starting. This file is a progress record, not a permission
gate. Final status is recorded in the handoff section at the bottom.

## Concrete findings

- `radiation.py` builds kernels in `radiation_kernel_from_dataset` as one
  monologue: validation, low tail, kernel quadrature, diagnostics,
  assumptions, provenance. `RadiationKernel` itself already accepts any
  1..12 DOF axis (`MAX_DOFS = 12`), so a retained-coordinate kernel needs
  no kernel-class change.
- `RadiationSettings` / `ReconstructionCriteria` / artifact hashing and the
  scipy-NetCDF save/load are reusable as-is; the serialized-settings reader
  already refuses unknown top-level keys.
- `schema.validate_hydro_dataset` deliberately requires all 6 DOFs per
  body, so a retained-4 artifact must be its own artifact type, never a
  forged raw dataset. Confirmed: no schema change needed or allowed.
- `dof.parse_dof_key` gives label-driven selection; `P` can be built purely
  by label matching with no hardcoded numeric positions.
- The independent exponential-kernel oracle lives in
  `tests/test_radiation.py` (`make_oracle_dataset`, `_oracle_matrices`,
  513-frequency grid, 1001-sample kernel) and is reused for the new tests.
- Baseline `derived-operator-baseline.json` (parent-captured) matches the
  current working tree for every file I touch; dirty/untracked state is
  preserved, no Git operations.

## API / internal approach

New module `src/anyhydro/operators.py` (imports only dof/errors/hashing/
schema + numpy/xarray; no Qt/ANYsolver/Capytaine at base import):

- `ProjectionSpec`: ordered retained DOF labels; exact 0/1 selector
  `P` (n_full x n_ret) built by label match against the source DOF axis;
  refuses empty/duplicate/unknown/malformed labels. `q_full = P q_ret`,
  `Q_ret = P^T Q_full`.
- `NormalizationSpec`: declared fixed positive diagonal scales `S` per
  retained DOF plus a mandatory convention name and unit convention string;
  refuses bool/non-finite/non-positive scales. Congruence is strictly
  `X_hat = S^T X S`.
- `CORRECTION_POLICIES = ("none", "reciprocal_average")`; reciprocal average
  `(X + X^T)/2` applied to A/B/Ainf only, never hydrostatics/excitation, no
  PSD clamping. No default policy: `DerivedOperatorSettings` requires an
  explicit choice.
- `DerivedOperatorCriteria`: all-explicit finite nonnegative thresholds -
  raw normalized reciprocity defect (abs + rel against a declared
  per-matrix global normalization with an explicit near-zero floor),
  normalized correction magnitude (same threshold shape), derived symmetry
  tolerance (fixed retained coordinates), retained B PSD eigenvalue
  tolerance (fixed coordinates, sign-invariant under congruence).
- `DerivedOperatorReport`: complete residual/threshold/pass arrays for
  reciprocity and correction, derived symmetry defects, retained B
  eigenvalues with negative-mode vectors/alignment, plus a separate
  raw-full diagnostic section (full B min eigenvalues, full asymmetries,
  excluded-DOF PSD defects) that never feeds the retained pass/fail.
- `DerivedHydroOperators`: immutable artifact holding source dataset hash,
  raw source DOF/body order, `P`, raw retained A/B/Ainf, derived A/B/Ainf,
  projected unmodified hydrostatics/excitation/wave/frequency coordinates,
  full source provenance (backend/method/request/mesh/lid/equilibrium/
  reference/phasor/transform metadata) and the report. `content_hash()`
  covers all data, settings, criteria and results. `admitted` is always
  recomputed from the report, never a stored bit.
- Atomic `save_derived_operators` / `load_derived_operators` (deterministic
  npz, `allow_pickle=False`, strict format-version and exact-field checks,
  hash verification, and reader-side recomputation of P, derived arrays and
  the report; mismatch is a typed refusal).

`radiation.py` changes (behavior-preserving for the existing full-6N path):

- Extract the private array core of `radiation_kernel_from_dataset` into
  `_kernel_from_arrays(...)` (low tail, kernel quadrature, diagnostics,
  base assumptions); the dataset bridge calls it with exactly its current
  inputs so old hashes/results are bit-identical.
- New `radiation_kernel_from_derived_operators(derived, settings)`: typed
  non-success unless `derived.admitted`; caller supplies B(0) in retained
  coordinates; same numerical kernel implementation; distinct provenance
  (`source_kind="derived_operators"`, derived content hash, raw source
  dataset hash, policy/S/criteria identity).
- New `compare_derived_reconstruction(kernel, derived, criteria)`: bound to
  the derived content hash AND the raw source hash plus retained
  DOF/policy/S/criteria identity; compares reconstructed A+B against the
  derived reference (never raw source relabelled); returns the existing
  public `ReconstructionReport`.

`__init__.py`: export the new public names.

## Intended edits

- Add `src/anyhydro/operators.py`.
- Edit `src/anyhydro/radiation.py` (shared core + two derived entry points,
  no change to existing public behavior).
- Edit `src/anyhydro/__init__.py` exports.
- Add `tests/test_derived_operators.py` (focused tests per brief).
- Add `docs/DERIVED_OPERATORS.md`; minimal README note.

## Remaining work / risks

- Implementation, tests, harness runs, docs (in that order).
- Risk: npz roundtrip of complex excitation and object-free string arrays
  needs care (`allow_pickle=False`); mitigated by storing labels as JSON
  metadata and arrays as raw float/complex.
- Risk: exact old-hash compatibility of the refactored dataset bridge;
  mitigated by capturing the pre-refactor full-12 oracle kernel hash first
  and asserting it in the new test.
- Reserved scientific choices: none taken; criteria are caller-declared
  with no defaults, per contract.

## Handoff (final)

Implemented and verified (author verification only; not qualification or
final acceptance).

### Changed files (all within assigned ownership)

- `src/anyhydro/operators.py` — new module (errors, `NormalizationSpec`,
  `DerivedOperatorCriteria`, `DerivedOperatorSettings`, `build_projection`,
  `DerivedOperatorReport`, `DerivedHydroOperators`,
  `derived_operators_from_dataset`, atomic pickle-free
  `save_derived_operators`/`load_derived_operators` with reader-side
  recomputation).
- `src/anyhydro/radiation.py` — extracted shared private core
  `_kernel_from_arrays` (dataset bridge passes identical inputs; old
  full-12 artifact hash bit-identical, asserted in tests); new public
  `radiation_kernel_from_derived_operators` and
  `compare_derived_reconstruction`; shared `_compare_window`.
- `src/anyhydro/__init__.py` — exports for the new public names.
- `tests/test_derived_operators.py` — new focused suite (30 tests).
- `docs/DERIVED_OPERATORS.md` — new usage/conventions/refusal doc.
- `README.md` — one documentation-index line added.
- No edits to schema/backends/capytaine_backend/records/fixtures/hashing/
  persistence, siblings, TASK/HANDOFF, old evidence, Git state or .idea.

### Checks actually run (harness, exact wall, exits, counts)

All runs via `run_check.py`, cwd `C:/Github/ANYhydro`, PYTHONPATH
`C:/Github/ANYhydro/src`, thread env pinned to 1, per-process timeout 15 s:

- `derived-operator-worker-01`: pre-refactor full-12 oracle kernel content
  hash capture (10bd41545ea153aa94cae508bcf1da3b7d0f3e2145553812de73060ccfc9091c);
  capture script had a later typo (exit 1) after the hash printed — recorded
  as a failed attempt, hash itself captured.
- `derived-operator-worker-02`: pytest derived+radiation, wall 4.51 s,
  exit 1, 4 failed / 55 passed.
- `derived-operator-worker-03`: same suite after first repairs, wall
  3.77 s, exit 1, 1 failed / 58 passed.
- `derived-operator-worker-04`: same suite, wall 3.30 s, exit 0,
  59 passed (30 derived + 29 radiation).
- `derived-operator-worker-05`: pytest derived+radiation+imports, wall
  4.15 s, exit 0, 61 passed (adds 2 base-import neutrality tests).
- Cumulative harness process wall approx. 17.0 s of the 120 s allowance
  (including the 1.28 s pre-checkpoint spend); no other numerical runs.

### Failures found and repaired

1. `NormalizationSpec` accepted `+inf` scales (positivity check alone);
   added an explicit `np.isfinite` refusal. Real source fix.
2. Mixed `[True, 1.0, ...]` scale lists coerced to float64 before the bool
   check; added an element-wise bool refusal before array conversion. Real
   source fix.
3. Test regex mismatch on the str-label refusal (message wording); test
   fixed to match the actual typed message.
4. Bogus immutability assertion (`.__class__` read cannot raise); replaced
   with a `FrozenInstanceError` assignment check.
5. Wrong-source test built a dataset-path kernel with a 4x4 B(0); now uses
   the full 12x12 oracle B(0) so the intended provenance refusal is reached.

### Remaining limits / gaps

- npz artifacts are content-hash verified but not byte-deterministic (zip
  timestamps); the contract requires exact content roundtrip, which holds.
- `MAX_SOURCE_DOFS`/`MAX_SOURCE_FREQUENCIES` are duplicated in
  `operators.py` and `radiation.py` (deliberate: avoids an import cycle;
  commented at both sites).
- Development fixtures only: no mesh/impedance/response/source/convergence
  qualification claimed; Ainf + physical dry mass gate remains
  consumer-owned; response/impedance correction and mesh acceptance stay
  open for later registered qualification.
- Reserved scientific choices: none taken; all criteria caller-declared.

### Evidence paths

- `C:/ANYworkspaceWork/anymarine-evidence/derived-operator-worker-01` …
  `derived-operator-worker-05` (result.json, stdout.log, stderr.log each).
- Baseline: `C:/ANYworkspaceWork/anymarine-evidence/derived-operator-baseline.json`.

---

## Handoff addendum (final, continuation slice 2026-10-08)

Truthful continuation record: the second worker's wrapper terminated at 70
turns with no updated checkpoint covering the post-handoff parent runs.
Parent-02 (evidence derived-operator-parent-02, started 2026-10-08T13:51:57Z,
harness wall 2.745483599996078 s, exit 1, 4 passed / 1 failed) confirmed the
three original counterexamples closed and isolated the last actionable
defect: `test_normalized_negative_mode_vectors_are_eigenvectors` failed with
max absolute mismatch 74.11182742 on the coupled indefinite 4x4 fixture.
All earlier failed evidence above is preserved unchanged.

### Defect and repair (this slice)

`_retained_report_arrays` selected negative modes as `eigvecs[neg]`, i.e.
ROWS of the per-frequency eigenvector matrices, while `numpy.linalg.eigh`
returns eigenvectors as COLUMNS. Repair in `src/anyhydro/operators.py`:

- Each negative mode is now the column
  `eigvecs[freq_idx, :, basis_idx]`, shape `(n_modes, n)`; the full basis
  `damping_eigenvectors` keeps the eigh column convention.
- The eigensolver now always receives the explicit symmetric part
  `0.5*(B_hat + B_hat^T)`, never silently one triangle of a potentially
  nonsymmetric `B_hat` (defensive: reachable inputs are already symmetric
  because `none` refuses asymmetric raw and `reciprocal_average`
  symmetrizes).
- No stored-data, clamping, policy, tolerance, schema or artifact-format
  changes; `anyhydro-derived-operators-v2` field set unchanged (report
  vector contents change, so new-artifact content hashes change).

### Changed files (all within assigned ownership)

- `src/anyhydro/operators.py` — the two-line mode-vector/eigensolver fix
  plus the symmetrization and column-convention comments/docstrings.
- `tests/test_derived_operators.py` — new regression
  `test_coupled_negative_mode_vectors_satisfy_eigen_equation` (coupled
  indefinite retained block, declared scales 1/2/3/4, multiple negative
  modes per frequency: `B_hat v = lambda v` for every reported mode, unit
  norm, exact match with the eigenbasis column). Suite now 71 tests.
- `docs/DERIVED_OPERATORS.md` — symmetrized eigensolver input, eigh column
  convention, `damping_negative_mode_vectors` shape, optional loader
  `source_dataset` keyword semantics.
- `docs/DERIVED_OPERATOR_CHECKPOINT.md` — this addendum only.

### Checks actually run (harness, exact wall, exits, counts)

Both via `run_check.py` with PYTHONPATH `C:/Github/ANYhydro/src`, thread
env pinned to 1, and TEMP=TMP=`C:/ANYworkspaceWork/anymarine-evidence/tmp`
(the run-06 permission failures were caused by omitted TEMP/TMP; no system
ACL/config touched). stderr empty in both runs.

- `derived-operator-worker-09`: pytest
  `tests/test_derived_operators.py tests/test_radiation.py
  tests/test_imports.py`, cwd `C:/Github/ANYhydro`, timeout 15 s, harness
  wall 4.388859900005627 s, exit 0, 71 passed in 2.39 s (70 prior + 1 new
  regression).
- `derived-operator-worker-10`: pytest
  `test_derived_operator_parent.py` (parent five-test file: positive
  control, forged-projection hash, source controls, all negative-mode
  eigenvectors, constructor cap), cwd evidence dir, timeout 12 s, harness
  wall 2.411551099998178 s, exit 0, 5 passed in 0.86 s (was 4 passed /
  1 failed in parent-02).

Worker budget: measured cumulative 28.7592437999847 s + 6.800411100003805 s
(these two harness walls) = 35.5596548999885 s, plus the 0.512 s conservative
upper static charge (upper bound only, not precise process wall —
uncertainty preserved); approx. 83.93 s of the original 120 s allowance
remained unspent at handoff. Parent budget usage 6.245259599993006 s of 40 s
(parent-02) is the parent's own record.

### Open limits / gaps (unchanged plus this slice)

- Verification ran against the source tree via PYTHONPATH, matching the
  parent-02 environment; the parent will independently build/install/verify
  from the wheel after terminal. No installed-wheel check claimed here.
- npz artifacts remain content-hash verified, not byte-deterministic.
- `MAX_SOURCE_DOFS`/`MAX_SOURCE_FREQUENCIES` duplication (operators/
  radiation, deliberate import-cycle avoidance) unchanged.
- Development fixtures only: no mesh/impedance/response/source/convergence
  qualification; Ainf + physical dry mass gate remains consumer-owned.
- No Git operations, no sibling edits, no BEM/solver dynamics, no network.
- Reserved scientific choices: none taken; criteria caller-declared.

## Parent installed verification — 2026-10-08

119 unique installed cases passed across two attempts. First attempt114pass with
four missing-fixture errors and one missing-helper failure in disposable harness;
five repaired cases passed, source untouched and failures retained. All five
independent parent counterexamples pass, including coupled negative-mode
vectors and positive normalization control. Source owner71+5 checks passed.
Installed wheel/source14Python SHA match; public oldfull12 kernel hash preserved.
Existing actual1002-frame nonzero import/child/save/replay regression passed with
changed dependency; stored actualBEMpilot still refuses. No fresh BEM/qualification.

Evidence root C:/ANYworkspaceWork/anymarine-evidence. Candidate wheel:
wheels/derived-operators-v1/anyhydro-0.1.0-py3-none-any.whl. Final commands/identities/
counts/failures/exactprocesswall: derived-operator-final-manifest.json.
Worker35.559654800/120s measured plus0.512 static uppercharge uncertainty; parent
27.983808500/40s. Source scope confirmed, no marine/sourcebackend changes. Full
roadmap unfinished; next consumer integration and scientific firstslice admission.
Direct-provenance refusal and crossplatform eigenbasis/roundoff checks remain open.
