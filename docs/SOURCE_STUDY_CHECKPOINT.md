# Source-study checkpoint (final update, implementation complete)

Status: implementation complete, all focused checks green through the
mandated harness. This is a development checkpoint, not an acceptance
report. No BEM execution is authorised or has run in this slice; the raw
96-panel pilot stays frozen. The early inspection findings and candidate
design below were implemented as proposed; the record splits into
`source_study_case.py` (case record + refinement metrics) and
`source_study.py` (workflow + CLI) for size reasons.

## Check evidence (all through the mandated harness)

- `source-study-worker-01` — golden n=2 hash capture (mesh/request
  hashes pinned in `tests/test_first_slice_fixture.py`).
- `source-study-worker-02..04` — test authoring/repair iterations
  (import re-export fix, `_require` definition fix).
- `source-study-worker-05` — `tests/test_source_study.py`: 44 passed
  in 2.5 s (mocked/analytic only).
- `source-study-worker-06` — `tests/test_first_slice_fixture.py`
  (legacy + parametric/golden regression): 21 passed in 0.6 s.
- `source-study-worker-07` — module CLI prepare + adjudicate of a
  prepared study: return codes 0/0, `CAPYTAINE_IMPORTED False`.

## Concrete findings from inspection

- `first_slice.py` builds meshes from `(origin, u, v)` face specs with a
  float-tuple vertex pool. Float keys are exact only because n=2 coordinates
  are dyadic; non-binary levels (n=6) need exact rational keys to share
  vertices across faces without rounding/snapping coordinates.
- Golden invariants to preserve bit-identically at default n=2: mesh arrays,
  `mesh_hash`, `source_geometry_refs`, `request_hash`/`content_hash` of
  `make_first_slice_request()`, and all no-arg outputs.
- Public APIs to build on (no internals cloned): `assemble_hydro_dataset`,
  `validate_hydro_dataset`/`validate_for_time_domain`, `CapytaineBackend`
  capabilities/execution limits (DEFAULT 100 total panels; the absolute
  software ceiling for EXPLICIT limits is 2048, so the prospective
  384/864-panel levels remain usable through case-derived explicit
  `CapytaineExecutionLimits` — still no BEM execution authority in this
  slice), `DerivedOperatorSettings`/`Criteria`/
  `NormalizationSpec`, `derived_operators_from_dataset`,
  `save/load_derived_operators`, `DerivedHydroOperators.verify_raw_source`
  (full rederivation proof, not hash-only).
- Dataset attrs carry phasor/heading/normal conventions, body identity,
  mesh hashes, request hash, qualification status — usable for convention
  and identity mismatch checks without touching private arithmetic.

## Proposed record/workflow (candidate design)

1. `first_slice.py`: public keyword `subdivisions=n` (default 2) on
   `make_first_slice_bodies` and `verify_first_slice_fixture`; bounded
   integer levels 1..16, bool/float/non-finite/oversize refused before
   allocation; exact `Fraction` vertex keys for cross-face sharing at
   non-binary levels; counts hull 10n^2 panels / 5n^2+2n+1 vertices, lid
   2n^2 / (n+1)^2, total 24n^2; generalized analytic verifier (face roles,
   normals, volume, COB, waterplane — not just counts).
2. `source_study.py` (new): immutable closed-world `SourceStudyCase`
   (case id/revision, fixed box geometry recipe semantic ID, exact level
   inventory, omega/heading grids, zero/infinite availability, expected
   backend name/version 3.0.0 + default method, job/thread caps, retained
   DOFs, NormalizationSpec, DerivedOperatorSettings/Criteria, per-quantity
   refinement criteria with explicit floors/units,
   `qualification_scope="development_source_study_only"`), JSON closed
   world with unknown-field refusal and content hash;
   `make_prospective_source_study_case()` for levels (4, 6), omegas
   (2, 4, 6), heading (0,), S=[1,1,1,1] translation-length 1 m,
   reciprocal_average, proposed admission criteria exactly as briefed.
3. Workflow: `prepare_source_study` (exclusive new out dir, status
   `not_executed`, no backend construction/import), `execute_source_study`
   (caps/body identity/version checks BEFORE backend work, at most one
   solve per registered level, content-addressed raw dataset + derived
   operator artifacts saved even when non-admitted, per-level and
   cross-level reports, fail-closed, no retry, cancellation preserves
   evidence), `load_source_study` + `adjudicate_source_study` (reload and
   reproduce reports without fresh BEM, full public raw-source proof,
   tamper/traversal/corrupt refusal). Refinement metric arithmetic is new
   study-tool code: normalized per-quantity (A/B/Ainf/H/F) per-frequency/
   heading residuals vs the finest level with abs+relative+floor rules in
   each quantity's own normalized unit.
4. CLI: `python -m anyhydro.source_study prepare|execute|adjudicate` and
   `examples/source_study.py`.

## Differences from prior slices

No changes to operators/radiation/capytaine_backend numerical truth; only
public APIs are consumed. The adapter's DEFAULT 100-panel budget still
refuses the prospective levels; case-derived EXPLICIT limits admit them
up to the absolute 2048-panel software ceiling (a prospective
execution-control contract change adopted by the parent — forward
software capacity, NOT new run authority; this slice still executes no
BEM).

## Intended edits

`src/anyhydro/first_slice.py` (parametric), new `src/anyhydro/source_study.py`,
`src/anyhydro/__init__.py` exports, new `tests/test_source_study.py`,
additions to `tests/test_first_slice_fixture.py`, new
`examples/source_study.py`, `docs/SOURCE_STUDY.md`, this checkpoint file,
README section.

## Remaining work / reserved choices

- None inside this slice's boundary. Docs (`docs/SOURCE_STUDY.md`),
  README section, `examples/source_study.py` and the checkpoint are in
  place; all focused checks passed (see evidence above).
- Reserved for the parent: adoption of the prospective case/criteria as the
  scientific case, any real BEM run authority, radiation/time-domain
  qualification. The proposed criterion values are the parent-owned
  prospective candidate from the brief, not loosened by results.
- Known design boundary: `load_source_study` refuses partial records, so a
  `failed`/`cancelled` study with missing level directories cannot be
  reloaded (its logs/evidence remain on disk); adjudication of such
  studies is correspondingly refused rather than silently forgiven.

## Final handoff (continuation session, parent-review repairs + ceiling scope extension)

Outcome: all parent-review defects repaired with connected regressions,
the adopted execution-control ceiling extension implemented, and all
focused checks green through the mandated harness. Still a development
source study only: no BEM import/solve ran, the raw 96-panel pilot stays
frozen, and nothing here confers scientific qualification.

### Adopted scope extension (parent-owned execution contract)

`capytaine_backend.py` EXECUTION-CONTROL constants/docstrings only:
`_CEILING_TOTAL_PANELS` extended 100 -> 2048 (absolute software ceiling
for EXPLICIT limits); `MAX_TOTAL_PANELS = 100` DEFAULT preserved with all
default metadata/input/cache hashes unchanged
(`test_execution_limits_persistence_hash.py` green unchanged). No
backend physics/solver/methods/numerical output/provenance schema change.
Forward software capacity, NOT new run authority: cases above 2048
panels are refused honestly, no autounbounded profile. Docs updated
(`CAPYTAINE_ADAPTER.md`, `README.md`, `SOURCE_STUDY.md`, this
checkpoint's earlier mistaken "unusable 100 ceiling" statements
corrected).

### Parent-review defect repairs

1. Capability checks (`_check_backend_identity`): required
   capabilities are now checked exactly as the case requests them.
   `supports_zero_frequency` is checked only if the case requests the
   zero-frequency limit (it never does; `include_zero_frequency=False`
   means NOT REQUESTED/COMPUTED, so a backend advertising extra zero
   capability is no longer rejected); multibody, explicit
   infinite-frequency and explicit `added_mass_infinite` are required
   before any solve. The report records `zero_frequency_computed: false`
   explicitly.
2. CLI gate (`main`/`_cli_execute`/`_cli_adjudicate`): typed exit codes
   `CLI_EXIT_OK/ERROR/FAILED/CANCELLED/NO_GO/INCOMPLETE` (0..5). Success
   0 only for the full `source_admission_passed_development_only`
   outcome; no_go/failed/cancelled/incomplete and typed refusals are
   distinct nonzero codes while reports/artifacts are retained.
   `execute` maps case-derived caps through the REAL public
   `CapytaineExecutionLimits` -> `CapytaineBackend` production path
   (tested with the backend class mocked at the real factory boundary:
   real limits instance asserted at 864 panels / 3 omegas / 1 heading /
   1 job / 1 thread for the levels-4/6 case, no BEM import/calls).
   New `--cancel-before-level` expresses explicit cancellation intent.
3. Two-body coupling scientific gate: new
   `crossbody_coupling_report` (public artifact arrays only,
   S-congruence with the fixed case normalization, per-frequency
   retained A/B off-body block max-norms plus A_inf off-body norms,
   predicate against the case's DECLARED per-quantity near-zero floors —
   no new tolerances invented; at least one finite-frequency A or B
   off-body block must be non-negligible). Stored per level separately
   from `artifact.admitted` (never relabels the public artifact's truth);
   study PASS now requires required inventory + operator admission +
   coupling observable + refinement. Adjudication re-derives the
   predicate + metrics from the reloaded public arrays; a flipped or
   removed stored coupling record is a typed refusal. A block-diagonal
   otherwise-identical positive-PSD mock source produces NoGo with
   preserved raw + operator artifacts, and reload/adjudicate reproduce
   the same NoGo.

### Changed files (this continuation)

- `src/anyhydro/capytaine_backend.py` — ceiling constant + docstrings
  (execution-control only).
- `src/anyhydro/source_study_case.py` — new
  `crossbody_coupling_report` (+ `parse_dof_key` import, export).
- `src/anyhydro/source_study.py` — capability checks, coupling gate in
  execute/adjudicate/load, typed CLI exit codes, `--cancel-before-level`,
  module docstring.
- `src/anyhydro/__init__.py` — exports `crossbody_coupling_report`,
  `CLI_EXIT_*`.
- `tests/test_source_study.py` — capability mock overrides + 4
  capability tests, `block_diagonal` mock variant + 4 coupling-gate
  tests, CLI gate tests (production-path real-limits/mock-backend,
  no_go/cancelled/failed/incomplete nonzero codes); 2 changed-contract
  CLI tests updated (raise -> typed nonzero).
- `tests/test_execution_limits.py` — above-ceiling 101 -> 2049 where
  the contract prospectively changes; 864/1536/2048 admit; new default
  100-panel refusal vs explicit 864 admission class (mocked solve).
- `docs/CAPYTAINE_ADAPTER.md`, `docs/SOURCE_STUDY.md`,
  `docs/SOURCE_STUDY_CHECKPOINT.md`, `README.md`,
  `examples/source_study.py` — limit/gate documentation.

### Checks (all through the mandated harness, unique evidence dirs)

- `source-study-worker-09`: `tests/test_source_study.py` — 58 passed
  in 4.35 s (0 failed).
- `source-study-worker-10`: `tests/test_execution_limits.py` +
  `tests/test_execution_limits_persistence_hash.py` — 52 passed in
  0.64 s (0 failed; default-100 hashes unchanged).
- `source-study-worker-11`: `tests/test_first_slice_fixture.py` +
  `tests/test_imports.py` — 23 passed in 1.13 s (0 failed; golden
  n=2 hashes and base-import purity intact).
- `source-study-worker-12`: module-CLI main-path gate check — prepare
  exit 0, adjudicate-of-prepared exit 5 (`CLI_EXIT_INCOMPLETE`),
  `CAPYTAINE_IMPORTED False`.
- `source-study-worker-13`: final combined run of all four impacted
  suites — 131 passed in 4.66 s (0 failed).

Timing: cumulative process-wall 48.52 s of the 180 s ORIGINAL worker
allowance (workers 01..07: 23.694 s parent-measured; 08: 4.68 s;
09: 6.86 s; 10: 2.57 s; 11: 3.05 s; 12: 0.72 s; 13: 6.95 s); ~131.5 s
remains. Parent 60 s verification reserve untouched. No BEM legacy
adapter tests or other BEM examples were run.

### Gaps / remaining reserved decisions (truthful labels)

- VERIFIED (assigned criteria, mocked/analytic only): ceiling
  admission/refusal arithmetic, default-100 refusal, capability
  request-scoping, CLI exit-code gate, coupling observability gate and
  its reload/adjudicate re-derivation, golden fixture hashes,
  base-import purity.
- PARTIAL: the production-path CLI execute test mocks the
  `CapytaineBackend` class at the real factory boundary — the REAL
  backend's solve path is intentionally never exercised (no BEM
  authority this slice).
- UNVERIFIED / reserved for the parent: scientific adoption of the
  prospective case/criteria (W5/full qualification unmet), any actual
  BEM run authority, cross-repository acceptance. The three-frequency
  two-level grid remains not claimed adequate for radiation/RAO or
  time-domain qualification.

## Early checkpoint (v2 units/admission delta round, worker14+ session)

Parent found three concrete contract deltas; prior repairs stand, no
redo. Exact existing `source-study-worker-01..13` result.json
`wall_seconds` sum: 43.8437955999979752 s of the 180 s ORIGINAL
allowance; 136.1562044000020248 s remain. Parent 60 s untouched.
MaxTurns75; final 20 turns / 30 s reserve.

Concrete deltas and intended edits (no arithmetic/tolerance change, no
raw dataset units/schema change, no new BEM/import backend, no
cryptographic attestation, no new hydrostatic threshold):

1. Prospective case units are scientifically wrong. Adopt template
   revision `prospective-candidate-2026-10-08-v2` BEFORE any actual run
   (none has run): declare q = S q_hat (S = L = 1 m) so q_hat is
   DIMENSIONLESS; Q_hat = S^T Q carries N*m = J; A_hat/A_inf_hat
   kg*m^2; B_hat kg*m^2/s; H_hat N*m = J; stored excitation is the
   force for a ONE-METRE reference wave multiplied by the physical
   dimensionless eta/(1 m); F_1m_hat has N*m = J (per-metre-wave
   transfer N). Record in hash-covered `normalization.unit_convention`
   and per-quantity `unit` strings; numerical thresholds/arrays/S1
   unchanged; old prepared evidence untouched on disk. The ADOPTED mesh
   norm rule is recorded exactly: per-sample max-entry complex-absolute
   residual vs `abs_tol + rel_tol * max(finest quantity's GLOBAL
   registered-band max-entry magnitude, floor)` — a GLOBAL-band linf
   relative control, NOT relative 2% of each local sample. Focused
   near-zero/multi-frequency tests pin this (the existing
   `compare_level_refinement` arithmetic already implements it; only
   declarations/tests/docs change).
2. `_admit_dataset` reports raw backend/version but never proves them:
   add typed refusal unless the actual raw dataset attrs
   `backend`/`backend_version` match the case `expected_backend_*`
   (capabilities alone are not proof), and re-run the same admission
   proof at reload (`load_source_study` rebuilds the deterministic
   bodies/request per level and re-admits — covers adjudicate).
   Analytic/mock tests switch to honest `mock-rehearsal` /
   `0.0-mock-rehearsal` identities for case/capabilities/raw attrs;
   ONLY the CLI factory-boundary mock tests may claim capytaine 3.0.0
   in capabilities and raw attrs, with explicit rehearsal labeling in
   the test (no actual solver claim). Mismatched raw backend/version
   refusals tested on execute and reload; failure evidence retained
   under the existing fail-closed behavior.
3. `_admit_dataset` checks body_identity attrs but not the ACTUAL
   ordered `rotation_center`/`center_of_mass`/`center_of_buoyancy`
   data variables: add a fixed-fixture proof (reference centers
   (-2,0,0)/(2,0,0) m; COM/COB each (cx,0,-0.25) m; body/DOF axes in
   the fixed order) compared at atol 1e-12 m rtol 0 — a prospective
   fixed-fixture ADMISSION numerical tolerance, not a meshing
   tolerance; no coordinate mutation/rounding. Described in a new
   hash-covered required case text field `admission_declaration`
   (closed-world schema evolves at v2; v1 records are invalidated by
   identity, left untouched on disk). Mutation tests (each array,
   unchanged metadata/requesthash, recorded hashes made consistent so
   the ARRAY check itself fires) on production execute and
   reload/adjudicate. No public operator PSD/sourceproof changes; the
   analytic H reference stays for parent assessment; no new
   configurable hydrostatic threshold/schema.

Remaining work: implement the above in `source_study_case.py`,
`source_study.py`, `tests/test_source_study.py`; update
`docs/SOURCE_STUDY.md`, README study section, this checkpoint and
`anymarine-evidence/source-study-final-handoff.md`; run the impacted
mocked/analytic suites through the mandated harness (worker14 onward,
one process/thread, <=15 s and remaining budget).

## Early checkpoint 2026-10-08 (turn-6, deltas A/B of the latest parent brief)

Prior 75-turn wrapper terminated at the CLI check; implementation and
checks 14 (5 fail) -> 15 (67 pass) -> 16 (75 pass) -> 17 (CLI prepare
0 / adjudicate 5 / no Capytaine) are preserved and NOT rerun. Remaining
original worker budget: 118.4372877999849 s of the 180 s allowance
(measured sum of source-study-worker-01..17 result.json), no renewal,
parent 60 s untouched. Scope of this continuation (no siblings, no Git,
no backend numerics/BEM/import changes, no old-pilot reruns):

A) Honest v2 case format. `SOURCE_STUDY_FORMAT` becomes
`anyhydro-source-study-v2` and the content-hash prefix becomes
`anyhydro-source-study-case-v2` (the case already carries the
`admission_declaration` field and revision
`prospective-candidate-2026-10-08-v2`; only the format string and hash
prefix were still v1). The v2 decoder refuses v1 records with a typed
`SourceStudyCaseError` naming v1 as explicitly unsupported — no
automatic migration, no restamping. Existing v1 prepared/mocked
evidence on disk is left unchanged and simply becomes undecodable by
the v2 tooling. Physical fixture, default mesh/request hashes and all
numerical thresholds are untouched; v2 confers no qualification.

B) Received-source evidence BEFORE identity admission. Today
`execute_source_study` runs `_admit_dataset` before
`save_hydro_dataset`, so a raw backend/version/center mismatch leaves
only a log entry and discards the source needed to diagnose the
rejection. Chosen representation (internal route, no pickle/external
objects, no raw-source mutation, public schema untouched):

- The level directory is created immediately after the solve, and the
  RECEIVED dataset is persisted BEFORE `_admit_dataset` runs.
- Schema-valid foreign sources (pass the public
  `validate_hydro_dataset` gate inside `save_hydro_dataset`) are saved
  through the public `save_hydro_dataset` to the same content-addressed
  `raw-<dataset_content_hash>.nc` name the admitted flow uses — on
  admission success the flow is byte-identical to before; on refusal
  the received raw file simply survives, unmarked as admitted (no
  report.json is written for a refused level, so reload/adjudicate
  still refuse partial records).
- Schema-invalid datasets (the public save route raises, e.g. the
  schema's own rotation_center/center_of_mass array-vs-attr proof)
  get a bounded forensic `received-forensic-level-NN.json` archive in
  the level directory: clearly labelled
  `anyhydro-source-study-received-source-archive-v1`,
  `status: rejected_received_schema_invalid`, `admitted: false`, with
  bounded (capped, truncation-flagged) arrays/coords/attrs and the
  original typed failure text. It is a labelled rejected/received
  record, never an admitted neutral dataset, never loadable as a
  qualified source. The ORIGINAL typed failure is re-raised unchanged
  so the failed state and failure log semantics are maintained.
- A `received_source_preserved` log event (kind + file name) is
  appended so the failure log names the preserved evidence; at least
  the source data plus the failure log survive backend/version/center
  mutations.

Focused regressions to add/update (only impacted ones; passed suites
are not rerun): v1-format typed refusal + honest v2 hash-prefix pin;
received raw file survives backend/version/center refusals (updated
assertions in the three existing mismatch tests); forensic archive on
schema-invalid rejection; cancellation prior-level evidence. Docs
(SOURCE_STUDY.md, README, this checkpoint, final handoff) updated at
the end; final 20 turns / 30 s reserved for checks/docs/handoff.

## Final checkpoint (registered-inventory binding repair, worker23+ session)

Parent's independent check `source-study-parent-05` FAILED both cases:
a missing or modified `geometry_inventory.json` was silently ignored by
`load_source_study` (no typed refusal). Registered input
reproducibility is therefore fixed BEFORE source readiness acceptance.

### Defect and repair

`_load_case_record` (the shared closed-world reload route used by
execute, load and adjudicate) verified only `case.json`; the registered
`geometry_inventory.json` was never required or compared. Repair
(`src/anyhydro/source_study.py` only, no new public API):

- New `_verify_geometry_inventory(out_dir, case)`: the registered
  `geometry_inventory.json` is REQUIRED on every reload/execute/
  adjudicate; the saved record is compared as ONE full canonical record
  (`_canonical_json`, closed-world exact identity, no numeric
  smoothing) against the inventory rebuilt deterministically from the
  hash-bound case through the PUBLIC first-slice APIs
  (`_geometry_inventory(case)`: exact per-level panel/vertex counts,
  per-body hull/lid mesh hashes, source geometry refs, request hash,
  analytic verification totals).
- Missing/unreadable/non-object inventory is a typed
  `SourceStudyWorkflowError`; any missing, unknown, removed or
  modified field, non-finite tampering, or CURRENT-recipe drift under
  the same case hash is a typed `SourceStudyIdentityError`.
- The check runs inside `_load_case_record` BEFORE
  `_check_backend_identity` and any solve, so execute refuses before
  caps/solve and a prepared case can never silently execute new
  geometry; prepared, cancelled, failed and completed studies all take
  the same shared route, and a missing COMPLETE-study inventory also
  refuses.
- No autocorrection/migration/re-registration; bad evidence is
  preserved; v2 case format/hash/prefix, geometry recipe bytes and
  default n=2 hashes unchanged; no BEM import/solve.

### Focused regressions (tests/test_source_study.py)

New `TestGeometryInventoryBinding` (10 tests): prepared and completed
reload/adjudicate bind the inventory; missing inventory refused
(prepared and completed); corrupt inventory refused; modified level
panel count / per-body mesh hash / request hash refused; unknown extra
field and removed field refused (closed world); execute refuses a
missing or tampered inventory BEFORE any capabilities/solve (sentinel
backend, status stays `not_executed`); simulated current-recipe drift
under the same case hash (TESTING HOOK ONLY: monkeypatched
`_geometry_inventory`, no production hook) refused on load and on
execute-before-solve. New CLI test
`test_cli_adjudicate_missing_inventory_error` (typed `CLI_EXIT_ERROR`).

### Checks (all through the mandated harness, workers 23..26)

- `source-study-worker-23`: full `tests/test_source_study.py` —
  86 passed in 7.43 s (wall 9.959927300005802 s), 0 failed.
- `source-study-worker-24`: module-CLI gate — prepare exit 0,
  adjudicate-of-prepared exit 5, adjudicate with the inventory removed
  exit 1 (`CLI_EXIT_ERROR`), `CAPYTAINE_IMPORTED false`
  (wall 0.7694062999944435 s).
- `source-study-worker-25`: the parent registration
  `source-study-registration-v2` still loads and adjudicates
  reproduced (`not_executed`; inventory unchanged; wall
  0.7412479000049643 s).
- `source-study-worker-26`: `tests/test_imports.py` — 2 passed
  (base-import purity; wall 2.746440899994923 s).

### Exact timing accounting (result.json wall_seconds)

Workers 01..22: 80.06772949999597 s (unchanged, not rerun). This
session: 23: 9.959927300005802; 24: 0.7694062999944435;
25: 0.7412479000049643; 26: 2.746440899994923 — session sum
14.2170224000001328 s. Cumulative 94.2847518999961 s of the 180 s
ORIGINAL allowance; 85.7152481000039 s remains. Parent 60 s reserve
untouched. No renewal, no probes/installs, no BEM suites run.

### Gaps / reserved (truthful labels)

- VERIFIED (mocked/analytic only): inventory binding on every
  reload/execute/adjudicate before caps/solve; missing/tampered/
  unknown/drift refusals; registration-v2 compatibility; CLI typed
  refusal; base-import purity.
- UNVERIFIED / reserved for the parent: scientific adoption of the
  prospective v2 case/criteria, any actual BEM run authority,
  independent installed verification
  (`test_source_study_independent.py` is parent-run; not executed or
  source-copied here), cross-repository acceptance. Development
  source readiness is NOT scientific qualification.

## Parent independent installed assessment, 2026-10-08

Final installed22cases passed source-study-parent-08 (full log in external evidence);
source/installed hashes equal, defaultn2 historicalhashes exact, independent analytic
geometry/work/norm/oracles plus productionCLI realconstructor/capabilities with ONLY
solve mocked, admittedcoupled12->4 reload, PSD/blockdiag/phaseNoGo, actualrawbackend/
center rejection+receivedevidence, cancellationpartial and inventory missing/changed
refusal. Parent05 failures preserved before sharedinventory repair. Cap backendAST
identical to priorwheel except explicitceiling100->2048+docs. Case7d5639a97b8fec37c0981fe6ff8960ee3acffcec517274b6ce708050a4ac41af stillnot_executed;
registered case + verifiedmanifest under C:/ANYworkspaceWork/anymarine-evidence.
Developmenttooling assessment met on pinnedWindowsruntime. No actualBEM performed;
scientificqualification/unregisteredcampaignauthority notconferred. Freshsource-run
allowance pendinguserquestion; priorpilotconsumed/frozen. Governingliving TASK in
ANYmarine records current exact accounting and remaining full-roadmap work.


## Actual coupled source study completed — 2026-10-09

Previous goal turn progress: source transport implementation verified. This turn
progress: prospectively adopted unchanged immutable development case
7d5639a97b8fec37c0981fe6ff8960ee3acffcec517274b6ce708050a4ac41af and completed ONE new real Capytaine3.0.0 source-study
execute. Full goal ACTIVE/incomplete. qualification-evidence skill applied using
existing public prepare/execute/reload/adjudicate supervisor and frozen criteria.
New output physical-source-real-20261009; all historical96-panel pilot, studies,
failed evidence and resource pools remain unchanged. No automatic retry.

Actual candidate: installed ANYhydro adapter0.2.0, wheel identity frozen before
execute (physical-source-candidate-identities.json). Two bodies, full coupled12
raw DOFs at subdivision4/6 (384/864 total panels), omega2/4/6 rad/s, heading0,
explicitA_inf, no zero-frequency, pinned defaultBEM, one job/thread. Public solve
called ONCE per registered level; each call assembles its finite/limit BEM problems.
Execute35.9058795000019s exit0, adjudicate1.7803106999999727s exit0/reproducedtrue.
Outcome source_admission_passed_development_only: retained surge/heave4-DOF
reciprocity, permitted reciprocal-average correction, symmetry, PSD and all five
quantity mesh criteria pass. Retained damping minima0.03477/0.05352; tightest
reported refinement A7.5388 residual versus7.8286 threshold under GLOBAL-band
max-entry control, NOT a local2% accuracy or asymptotic convergence claim.

Raw saved sources preserve all12 coupling terms: max cross-body A66.2043/66.0688
and B707.268/697.273. Exact source declaration/request/body inventory verification
passes both actual saved datasets; deterministic three-component amplitudes/phases
compiled through public source-bound builder agree with independent complex sums
to2.2737367544323206e-13 (rtol/atol1e-12), without phase regeneration or BEM.
Received raw NetCDF and derived sources reloaded/rederived during adjudication;
no block-diagonal replacement, source averaging, clamping or relabeling.

Material remaining source limitation: full12 raw symmetric damping has negative
modes (minimum-0.022656142104575363 at384 panels and-0.015483134983758425 at864).
Saved-source power inspection with q=S qhat, translation lengths1/2m, rotations
1rad, Bhat=S.T B S (kg*m2/s) confirms negative quadratic dissipation directions
primarily common pitch at omega2/4; omega6 minimum is positive. Work identity
errors<=1.36e-16W. Harmonic cycle-mean dissipation is half v.T B v; sign unchanged.
This is not a mixed-unit display artifact. Full-six source remains UNADMITTED;
four-DOF projected pass cannot justify full12 passivity or nonlinear qualification.
Further full-space source-quality route needs a prospective discriminating case/
physical derivation, not automatic mesh escalation or eigenvalue clamping.
The three-frequency band cannot qualify broadband radiation memory, RAO/free
decay or complete time responses. Equilibrium/hydrostatic/source/native binding,
lines/contact, HAMS/external reference and W3-W10 acceptance remain outstanding.

Independent OpenAI Read assessment confirmed registered scope/completeness and
highlighted those limits. Supporting review only, no different-family scientific
acceptance. Stderr contains retained Unicode logging-handler failures while
emitting Capytaine warnings; both numerical datasets validated and study terminal
pass reproduced. Future evidence logging should use explicit UTF-8; no rerun for
cosmetic logs. Finite depth150m remains the registered condition despite deep-
water performance warning; no silent infinite-depth substitution.

NEW180s pool used42.02055310000287s, remaining137.97944689999713s, separate from all older pools.
One registered two-level real attempt consumed; no further BEM in this slice.
All owned processes terminal. No product physics/solver changes or publication.
Initial stage-patch syntax failed before mutation/launch, corrected; no numerical
failure discarded. Evidence: C:/ANYworkspaceWork/anymarine-evidence/physical-source-*
process records; physical-source-real-20261009 case/inventory/raw/derived/reports;
physical-source-inspection.json; physical-source-full-space-power-clarified.json;
physical-source-independent-review.md; physical-source-handoff.json. The next
connected implementation should consume those saved sources through strict
equilibrium/hydrostatic/radiation binding while preserving full-six refusal.
