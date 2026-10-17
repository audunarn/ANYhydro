# First-slice source-quality mesh study (development tooling)

Development-only source study tooling for the deterministic first-slice
two-box fixture. It registers a parametric deterministic mesh-refinement
study as an immutable closed-world case record, prepares it without any
backend, executes it through an explicitly selected public backend under
explicit execution limits (at most ONE solve per registered level), and
reloads/adjudicates the saved evidence without any fresh BEM.

**Nothing here confers scientific qualification.** The
`qualification_scope` of every case is always
`development_source_study_only`. This slice carries no BEM execution
authority: no solve is ever run here. The real Capytaine adapter is only
constructed on the explicit execute path under case-derived explicit
`CapytaineExecutionLimits` (the 100-panel DEFAULT refuses the
prospective refined levels; explicit limits admit them up to the
adapter's absolute software panel ceiling, docs/CAPYTAINE_ADAPTER.md), and the
frozen raw 96-panel pilot is neither rerun nor re-stamped.

## Owned modules

- `anyhydro.first_slice` — the physical fixture plus the parametric
  deterministic refinement: public keyword `subdivisions=n` (default 2,
  bit-identical to the historical fixture hashes), bounded integer levels
  1..16 (bool/float/non-finite/oversize refused before allocation), exact
  rational vertex keys for cross-face vertex sharing at non-binary levels
  (no coordinate rounding), and a generalized analytic verifier
  (face roles, outward normals, volume, COB, waterplane — not just
  counts). Per body: hull `10n^2` panels / `5n^2+2n+1` vertices, lid
  `2n^2` panels / `(n+1)^2` vertices; both bodies total `24n^2` panels.
- `anyhydro.source_study_case` — the immutable hash-bound case record
  (`SourceStudyCase`), the per-quantity refinement criteria, the
  prospective development case, the study inputs built through the public
  fixture API, and the normalized cross-level refinement metric
  arithmetic.
- `anyhydro.source_study` — the public workflow (prepare / execute /
  load / adjudicate) and the module CLI
  (`python -m anyhydro.source_study ...`).

Only public anyhydro APIs are used on the data route: the fixture
builders and analytic verifier, the canonical schema validation, the
dataset persistence, and the public derived-operator construction,
persistence and full raw-source proof. The refinement metric arithmetic
is new study-tool code operating on public artifact arrays in the case's
declared normalized units; no private operator/radiation internals are
touched, and no normalized diagnostics/projection/passivity arithmetic
is cloned.

## Case record contract

`SourceStudyCase` is closed-world and immutable: every control is
explicit and required (no permissive automatic defaults), unknown fields
are refused, and the record is hash-bound (`content_hash` over the
canonical JSON). The case format is honestly
`anyhydro-source-study-v2` and the content hash is namespaced with the
`anyhydro-source-study-case-v2` prefix: the required hash-covered
`admission_declaration` field changed the closed-world schema, so v1
records are EXPLICITLY UNSUPPORTED by the v2 decoder (typed
`SourceStudyCaseError`, no automatic migration or restamping — a v1
record must be rebuilt explicitly at v2). Existing v1 prepared/mocked
evidence on disk is left unchanged and is simply not decodable by the
v2 tooling; nothing about v2 confers qualification. The case accepts
ONLY the fixed first-slice two-box geometry
recipe (`first-slice-two-box-wetted-hull-lid-v1`); cross-level
comparison is justified by that shared recipe, never inferred from
bounds. Required controls include:

- exact level inventory (sorted, unique, bounded subdivision levels);
- exact finite omega and heading grids (strictly increasing, finite);
- explicit availability: zero-frequency NOT computed, A_inf explicit;
- expected backend name/version (capytaine 3.0.0) and declared method;
- the hash-covered `admission_declaration` text (raw backend/version
  attr proof + fixed-fixture center-array proof, v2);
- one job / one thread resource caps;
- retained DOFs exactly the first-slice projected surge/heave order;
- `NormalizationSpec`, explicit `DerivedOperatorSettings` criteria and
  explicit per-quantity normalized refinement floors/units;
- `qualification_scope="development_source_study_only"`.

## Prospective development case (parent-owned candidate)

`make_prospective_source_study_case()` (revision
`prospective-candidate-2026-10-08-v2`, adopted BEFORE any actual run —
none has run; the numerical thresholds, arrays, S = [1, 1, 1, 1] and
all criteria values are unchanged from v1):

- levels 4 and 6 (384 and 864 total panels);
- omegas [2, 4, 6] rad/s, heading 0, A_inf explicit, zero-frequency NOT
  computed;
- translation-length-1 m normalization S = [1, 1, 1, 1];
- explicit `reciprocal_average` correction policy;
- proposed admission criteria: reciprocity 1e-8 abs / 0.02 rel / 1e-8
  floor, correction 1e-8 abs / 0.01 rel / 1e-8 floor, derived symmetry
  1e-10, damping PSD 1e-10 — each abs/floor in its own normalized
  A/A_inf or B unit, never a cross-unit norm;
- mesh criterion 0.02 with per-quantity abs 1e-9 / rel 0.02 / floor 1e-9
  in each quantity's own normalized unit.

### Declared units and work conjugacy (v2)

With S = diag(1, 1, 1, 1) m (declared translation length L = 1 m):

- `q = S q_hat` — the normalized retained coordinates `q_hat` are
  DIMENSIONLESS;
- `Q_hat = S^T Q` — normalized retained force/moment components carry
  N*m = J;
- `A_hat = S A S` and `A_inf_hat = S A_inf S` carry kg*m^2;
- `B_hat = S B S` carries kg*m^2/s;
- `H_hat = S H S` carries N*m = J;
- the stored excitation is the force for a ONE-METRE reference wave:
  the physical excitation multiplies it by the dimensionless ratio
  eta/(1 m), so `F_1m_hat = S^T F_1m` carries N*m = J (equivalently the
  per-metre-wave transfer carries N).

These declarations live in the hash-covered
`normalization.unit_convention` text and in each per-quantity
refinement `unit` string, so they are part of the case identity. The
raw dataset units/schema are unchanged.

### Mesh norm rule (adopted semantics, no arithmetic change)

The refinement control is a GLOBAL-band linf relative control: each
sample's max-entry complex-absolute residual is compared against

```
abs_tol + rel_tol * max(finest quantity's GLOBAL registered-band
max-entry magnitude, floor)
```

where the reference scale is the largest normalized magnitude of the
reference quantity over its ENTIRE registered band (all frequencies /
headings) in that quantity's own normalized unit. It is never a
relative percentage of each local sample: a large-magnitude frequency
dominates the threshold for residuals on near-zero frequencies, and
when the entire band is near zero the explicit per-quantity floor
governs. Focused near-zero/multi-frequency tests pin this semantics.

### Source admission proof (raw backend identity + fixed fixture)

Admission of each raw dataset (and the same proof re-run at reload)
requires:

- the ACTUAL raw dataset attrs `backend`/`backend_version` to match the
  case-expected backend name/version — the capabilities checked before
  the solves are not proof of what produced the dataset;
- the actual ordered `rotation_center`/`center_of_mass`/
  `center_of_buoyancy` data variables to match the fixed first-slice
  fixture — reference centers (-2, 0, 0) m and (2, 0, 0) m, COM and COB
  each (cx, 0, -0.25) m, body and DOF axes in the fixed order —
  compared at atol 1e-12 m with rtol 0. This is a prospective
  fixed-fixture ADMISSION numerical tolerance, not a meshing tolerance;
  metadata attrs alone cannot prove the actual arrays, and no
  coordinate mutation or rounding is performed.

Both checks are described in the hash-covered case text field
`admission_declaration` (closed-world schema at v2; v1 case records
are invalidated by identity and left untouched on disk).

These values are a prospective candidate frozen BEFORE any future BEM.
They are not authority to loosen scientific criteria after results. The
three-frequency two-level grid is NOT claimed adequate for
radiation/RAO or time-domain qualification — it only discriminates the
old coarse-pilot PSD/reciprocity mesh hypothesis. Broader
kernel/tail/time/dynamic qualification remains a distinct registered
gate. The parent reviews/adopts the scientific case before any real
run.

## Workflow

```bash
# prepare-only: exclusive new directory, no backend constructed/imported
python -m anyhydro.source_study prepare --out <new-dir>

# execute: explicit backend choice; one solve per registered level
python -m anyhydro.source_study execute --out <dir> --backend capytaine

# reload + reproduce the reports without any fresh BEM
python -m anyhydro.source_study adjudicate --out <dir>
```

- **prepare** writes the immutable case record (hash-bound), the full
  per-level geometry + request inventory (exact panel/vertex counts,
  mesh hashes, source geometry refs, request hash, analytic
  verification totals) and a `not_executed` status. No backend is
  constructed or imported and no scientific metric is claimed. The
  inventory is a REGISTERED input reproducibility record: it is
  required and re-verified verbatim on every later
  reload/execute/adjudicate (no autocorrection, no migration, no
  re-registration).
- **execute** requires the registered `geometry_inventory.json` to
  match EXACTLY the inventory rebuilt from the hash-bound case through
  the public first-slice APIs (closed-world canonical identity, no
  numeric smoothing): a missing, unreadable, unknown, removed or
  modified inventory — or current-recipe drift under the same case
  hash — is a typed refusal BEFORE any backend capability check or
  solve, so a prepared case can never silently execute new geometry.
  Execute then consumes the explicit case, checks caps/body
  identity/version/capabilities BEFORE any backend work (required
  capabilities are checked exactly as the case requests them: the
  fixed case requires multibody, the explicit infinite-frequency limit
  and an explicit `added_mass_infinite`; the zero-frequency limit is NOT
  requested, so a backend advertising extra zero capability is not a
  mismatch and the report states explicitly that zero frequency was not
  computed), performs at most ONE solve
  per registered level (no retries, no new levels), admits each raw
  dataset fail-closed (conventions, qualification label, grids, request
  hash, mesh hashes, body identity, the ACTUAL raw backend/version
  attrs against the case expectation, and the actual ordered center
  data variables against the fixed fixture at atol 1e-12 m rtol 0),
  saves content-addressed raw and
  derived artifacts per level (non-admitted artifacts are saved too),
  derives the retained operators through the PUBLIC API, evaluates the
  two-body coupling observability gate (truthful per-level retained
  A/B/A_inf off-body block norms per frequency in the case's declared
  normalized units, stored separately from the artifact's own admission
  verdict; a block-diagonal source cannot pass the fixed two-body
  physical case), compares only
  matched grids/conventions/recipe/normalization against the finest
  level, and persists complete per-level and cross-level reports with
  assumptions, logs, effective controls and the call inventory.
- **received-source evidence BEFORE admission** — the RECEIVED raw
  dataset of every solve is persisted BEFORE identity admission, so a
  raw backend/version/center mismatch can never discard the source
  needed to diagnose the rejection. Schema-valid foreign-source data is
  saved through the PUBLIC `save_hydro_dataset` (its
  `validate_hydro_dataset` gate defines schema validity) under the same
  content-addressed `raw-<dataset_content_hash>.nc` name the admitted
  flow uses; on admission refusal the received raw file simply
  survives, unmarked as admitted (no level report is written for a
  refused level, so reload/adjudicate still refuse partial records). A
  schema-invalid dataset is preserved as a bounded, clearly labelled
  forensic archive (`received-forensic-level-NN.json`, format
  `anyhydro-source-study-received-source-archive-v1`, status
  `rejected_received_schema_invalid`, `admitted: false`, capped
  arrays/coords/attrs with truncation flags and the original typed
  failure text) — labelled rejected/received evidence, never an
  admitted neutral dataset, never loadable as a qualified source, with
  no pickle or external objects. The original typed failure is
  re-raised unchanged and the raw source is never mutated; a
  `received_source_preserved` log event names the preserved file, so at
  least the source data plus the failure log survive
  backend/version/center mutations. Canonical raw/derived level
  reports exist only for admitted levels; failed partial records
  cannot pass load/adjudicate.
- **load/adjudicate** first re-verify the registered geometry/request
  inventory against the inventory rebuilt from the immutable case
  (the same shared closed-world route as execute: a missing, tampered,
  unknown or drifted inventory is a typed refusal for prepared and
  completed studies alike), then reload the saved case and artifacts with
  the public full raw-source proof (`load_derived_operators(...,
  source_dataset=...)` rederives everything; a matching recorded hash is
  necessary but never sufficient), re-run the same fail-closed
  admission proof on each reloaded raw dataset against the
  deterministically rebuilt bodies/request (raw backend/version attrs
  and the fixed-fixture center arrays included — consistent hash
  bookkeeping cannot masquerade a mutated array or foreign backend as
  intact), recompute admission, the coupling
  observability predicate + metrics and refinement
  and compare with the persisted reports. Tampered, partial, corrupt,
  unknown or traversal-bearing records are refused; a changed stored
  bool or hash-only bookkeeping change cannot masquerade as a pass.

### Outcomes

- `prepared_not_executed` — prepared only;
- `source_admission_passed_development_only` — every level admitted,
  the two-body coupling observable and
  every quantity inside its declared refinement criterion (development
  evidence only; a mesh pass can never imply radiation/time-domain
  qualification);
- `no_go` — criteria failure (non-admitted level, coupling not
  observable or mesh criterion not
  met); complete residuals and failure evidence are preserved;
- `failed` — fail-closed refusal (missing/version/corrupt/foreign
  source/non-finite/inadmissible arrays); prior level and failure
  artifacts are retained, no automatic retry;
- `cancelled` — caller cancellation between owned stages; logs and
  evidence are preserved.

The module CLI is a gate with typed exit codes: only the full
`source_admission_passed_development_only` outcome exits `0`;
`no_go`/`failed`/`cancelled`/prepared-incomplete adjudications and
typed refusals exit with distinct nonzero codes
(`CLI_EXIT_NO_GO`/`CLI_EXIT_FAILED`/`CLI_EXIT_CANCELLED`/
`CLI_EXIT_INCOMPLETE`/`CLI_EXIT_ERROR`) while reports and artifacts are
retained — no CI gate can go green on missing criteria.

The full 12-source-DOF raw record always remains available; the retained
route never symmetrises, clamps or averages raw arrays, and residuals and
eigenmode reports stay truthful (no fabricated zeros).

## Checks (mocked/analytic only)

`tests/test_source_study.py` (86 tests: case record, refinement metrics,
prepare/execute/load/adjudicate workflow, capability checks, the
two-body coupling observability gate, the registered geometry/request
inventory binding and the typed CLI exit-code gate),
`tests/test_execution_limits.py` (panel ceiling 2048 for explicit
limits, default 100-panel refusal, caps before cache), the
parametric/golden regressions in `tests/test_first_slice_fixture.py`
and the base-import purity checks in `tests/test_imports.py` run
through the mandated evidence harness with explicitly labelled
mock-rehearsal backends only — no real Capytaine import or solve.
Latest observed evidence (this inventory-binding repair session):
`anymarine-evidence/source-study-worker-23` (full source-study suite,
86 passed in 7.43 s), `source-study-worker-24` (module-CLI gate:
prepare exit 0, adjudicate-of-prepared exit 5, adjudicate with a
missing inventory exit 1, no Capytaine import), `source-study-worker-25`
(the parent registration `source-study-registration-v2` still loads
and reproduces: inventory unchanged), `source-study-worker-26`
(`tests/test_imports.py`, 2 passed). Earlier sessions' evidence:
`source-study-worker-01..22`.


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
