## GUI publication complete; numerical development paused - 2026-10-10

Definitive restart: C:/Github/ANYmarine/CHECKPOINT.md. Published source baselines
ANYmarine 8e97dbb97e1c85f35d25134fd883ea3197aae27f, ANYhydro 1bb3b82cd94ad3229156dc3da51b56da410396d6 on main;
remote identities verified. Installed GPU startup/status/saved waves pass.
Status39, startup22, waves43, foundation54 pass/1 skip; hydro77 pass.
W5 remains NO_GO, native energy-point-02 consumed/failed/unqualified and audit
100/100 consistency only. No new scientific run. Historical limits preserved;
new GUI pools frozen at handoff. Root coordinates Mistral; no Astra.
Measured public note docs/GUI_MAIN_HANDOFF_20261010.md; full local evidence
C:/ANYworkspaceWork/anymarine-evidence/gui-main-handoff-20261010/.
Resume first with bounded stored-array case/energy diagnosis, not a native retry.
Saved UTC 2026-10-10T19:05:39.458648+00:00. Historical notes below retained verbatim.

# W2 handoff — ANYhydro Capytaine development adapter

Date: 2026-10-08. Repository: C:/ANYworkspaceWork/ANYhydro. Mode: Edit.
Owner: Mistral worker. Owned files: all repository files except AGENTS.md,
.github/workflows/ci.yml, LICENSE. No Git commits/pushes, no sibling edits,
no nested workers, no campaigns, no environment modification. Runtimes and
the run_check helper were used unchanged.

## State

**Complete for this slice and verified by execution** (worker verification;
development evidence only, not scientific qualification and not final
integration acceptance).

## What this session did (continuing the previous run)

The previous run had added `src/anyhydro/capytaine_backend.py`, the public
exports and a first adapter test file, but ended without running anything.
This session:

1. Fixed the constant typo `CAPYTAINDE_ADAPTER_VERSION` ->
   `CAPYTAINE_ADAPTER_VERSION` (5 occurrences in
   `src/anyhydro/capytaine_backend.py`) before any published use. The
   published value "0.1.0" is unchanged, so cache identity content and any
   stored dataset attrs are unaffected.
2. Ran the adapter tests for the first time (first tiny solve) — all green
   on the first run; no adapter code fixes were needed.
3. Ran the full suite on both runtimes (with and without Capytaine).
4. Cleaned a leftover dead expression in `examples/headless_two_body.py`
   and ran it end to end (solve -> validate -> save .nc -> reload ->
   hash-stable round trip -> imported-dataset replay).
5. Bumped `pyproject.toml` build requirement to `setuptools>=77` (required
   for the SPDX `license = "MPL-2.0"` string; LICENSE file is the
   parent-supplied one; dev extra already contains `build`).
6. Updated `README.md`: status section now states the adapter exists, plus a
   new "Capytaine adapter: invocation and capability limits" section and a
   pointer to `docs/CAPYTAINE_ADAPTER.md`.
7. Updated `CHECKPOINT.md` to reflect the actual state and next step.

## Adapter scope (unchanged from the previous run, now execution-verified)

- One combined `cpt.Multibody` `fill_dataset` call: full 6N coupled
  `added_mass`/`radiation_damping` with cross-body blocks preserved;
  complex `excitation_force` in the canonical phasor/heading conventions
  (verified against installed Capytaine 3.0.0 source; no sign flips).
- `hydrostatic_stiffness`: linear buoyancy-only (Capytaine
  `compute_hydrostatic_stiffness`, `mass=0`, per sub-body, block-assembled).
  The dataset attr `hydrostatic_stiffness_definition` states explicitly
  that the gravity restoring of the dry body is NOT included and is owned
  by the solver.
- `added_mass_infinite` (opt-in): explicit per-radiating-DOF `omega=inf`
  symbolic radiation solves, finite-checked, with source provenance.
- Typed refusals, never silent: zero-frequency limit
  (`UnsupportedOperationError`), unverified normals, non-wetted geometry
  (hull above free surface / below bottom, lid off z=0), non-finite solver
  output (`BackendSolveError` — no invented replacement coefficients),
  unknown/non-string settings, budgets exceeded (`CapabilityError`),
  missing optional library (`CapabilityError` from lazy probe).
- Enforced development budgets: <= 100 panels total, <= 3 finite omegas,
  <= 1 heading, `n_jobs=n_threads=1`.
- Content-addressed NetCDF cache (identity: mesh/lid hashes, rotation
  centers, COMs, enabled DOFs, full request, settings, exact Capytaine and
  adapter/schema versions); validated load, corruption/mismatch = miss;
  stores canonical datasets only, never pickled backend objects; a failed
  cache write never invalidates a correct solve.
- Caller immutability: source records are copied/read-only; verified by
  test.

## Tests

`tests/test_capytaine_backend.py` (29 tests, `pytest.importorskip` at
module top so base CI runs without the backend):

- Capabilities honesty; unknown/non-string settings; zero-frequency,
  frequency/heading/panel budget refusals; unverified normals; panels above
  free surface / below bottom; lid off z=0; missing-dependency diagnostics
  and typed solve refusal; honest diagnostics when available.
- Real tiny solves: canonical validation, full 12x12 with nonzero
  cross-body blocks, body-qualified DOF labels, explicit finite
  `added_mass_infinite` with provenance, honest hydrostatics labeling,
  caller inputs unchanged, deep and finite depth both solve.
- Direct Capytaine comparison: adapter output vs an independent direct
  `cpt.Multibody` run with identical panels — added mass, damping,
  excitation, per-body hydrostatics and per-DOF omega=inf rows all match
  (rtol 1e-10). This is a development adapter-mapping check, not a
  scientific qualification.
- Cache: hit avoids solve (fresh solve monkeypatched to fail), changed
  rotation center / request / physical constants each invalidate, corrupted
  entry is a miss and re-solves, cache file is NetCDF not pickle,
  `cache_dir` setting override, no cache files without `cache_dir`.

## Checks actually run (unique .evidence dirs, bounded runner TEMP/TMP)

Runtimes (unchanged): Capytaine-enabled
`C:/ANYworkspaceWork/anymarine-evidence/runtime/Scripts/python.exe`
(Python 3.13.9, capytaine 3.0.0, numpy 2.5.3, xarray 2026.9.0, pytest 9.1.1)
and base `C:/Github/ANYsolver/.venv/Scripts/python.exe` (no capytaine).
`PYTHONPATH=C:/ANYworkspaceWork/ANYhydro/src`, `OMP_NUM_THREADS=1`,
timeout 120 s each, `TEMP`/`TMP` bounded to
`.evidence/tmp/workspace-tmp`, pytest `--basetemp` bounded inside
`.evidence/tmp/`.

1. `.evidence/2026-10-08-adapter-tests-a` — adapter tests only, Capytaine
   runtime: exit 0, wall 12.5 s, **29 passed, 0 failed**.
2. `.evidence/2026-10-08-pytest-full-cpt-b` — full suite, Capytaine
   runtime: exit 0, wall 13.1 s, **158 passed, 0 failed**.
3. `.evidence/2026-10-08-pytest-full-base-c` — full suite, base runtime
   (no capytaine): exit 0, wall 8.5 s, **129 passed, 1 skipped, 0 failed**
   (the skip is the adapter module via `importorskip`: base CI works with
   no backend installed).
4. `.evidence/2026-10-08-example-headless-a` —
   `python examples/headless_two_body.py --out .../two_body.nc`:
   exit 0, wall 8.2 s. Output confirms 2 bodies, 12 DOFs, (1, 12, 12)
   added mass, nonzero cross-body block, finite `added_mass_infinite` with
   provenance, .nc saved and reloaded with stable content hash, replay
   backend round trip OK.

External `C:/ANYworkspaceWork/anymarine-evidence/test_hydro_review.py`
was NOT rerun this session: it does not reference the adapter, and nothing
it tests (records/schema/backends surface) changed. Last evidence
`.evidence/2026-10-08-independent-suite-c` (this repository): 5 passed,
exit 0.

Each evidence dir contains `result.json` (exact command, cwd, started UTC,
wall seconds, timeout, status, exit code, effective environment),
`stdout.log`, `stderr.log`. No pipelines masked exit codes.

## Remaining W2 acceptance gaps (honest)

- The adapter is a development adapter: no mesh convergence study, no
  passivity check, no independent hydrostatic qualification. Development
  fixtures and these tests confer no scientific qualification.
- The direct-vs-adapter comparison covers one tiny two-box geometry at
  1 frequency, 1 heading, deep and finite depth; it validates adapter
  mapping, not physics.
- Panel pressure, Kochin functions, free-surface elevation and mean drift
  remain disabled pending their own tests (`capabilities()` reports False).
- Zero-frequency limit remains a typed refusal by design (schema stores
  strictly positive finite frequencies only).
- Only Python 3.13.9 was exercised; wheels were not built/installed by the
  worker — parent builds/installs wheels (setuptools>=77 now declared for
  the SPDX license string) and exercises the public paths.
- `qualification_status` is always `unqualified` from this adapter; the
  `qualified` enum value exists but is never written here.

No blockers. No Git operations, no sibling writes, no nested workers were
performed.

## Paused relocation — 2026-10-08
User designated C:/Github/ANYhydro as the canonical repository and paused development. Implementation is transferred preserving destination Git metadata. Old C:/ANYworkspaceWork/ANYhydro remains a backup. Evidence remains C:/ANYworkspaceWork/anymarine-evidence. No new hydro tests or development ran during relocation. Shared pause/restart state is in C:/Github/ANYmarine/TASK.md.


User confirmed MPL-2.0 applies to all repositories; this repository already uses MPL-2.0. Development remains paused.


Prospective 2026-10-08 transition: user resumed canonical ANYmarine/ANYhydro development;
old pause statements above are historical. Shared TASK.md remains authoritative.
New Mistral ownership is neutral coefficient frames/hydrostatic contract transport,
analytic tests/docs, with separately parent-owned ANYmarine consumer guards.
No source/BEM authority renewed; pending physicalstudy remains unexecuted.


## Excitation source declaration/transport — 2026-10-09

Progress: source transport IMPLEMENTED and verified against development criteria;
full plan ACTIVE/incomplete. New ANYhydro excitation_source module/public API
serializes complete immutable HydroRequest values and explicit finite/infinite
depth, verifies raw dataset/request/body mesh/lid/reference/full coupled identity,
binds unit elevation amplitude/global phase origin/datum/zero-speed/conventions,
and compiles existing shared-wave excitation using verified intended environment.
No implicit defaults or missing-sidecar recovery, no native fixture gate relaxed.
Only finite-depth synthesis currently admitted; saved infinite-depth request
transport is supported. Source-bound series retains the raw qualification label;
a declared audit/status is transport metadata, not scientific truth or native
admission. Equilibrium geometry/source-study/hydrostatic/radiation/native physical
compilation remain required before full demonstrator acceptance.

Adopted binding order: excitation declaration before hydrostatic attachment;
excitation binding explicitly excludes its own and hydrostatic transport payloads
to avoid cycles, while the existing hydrostatic binding includes the excitation
declaration. Source hashes bind coefficient arrays and all remaining provenance.
Existing hydrostatic-bound input refuses new attachment and requires explicit
rebuild. Tested combined binding preservation; legacy schema/hashes unchanged.
NetCDF JSON string/dict source metadata hashes identically.

Capytaine adapter0.2.0 emits audited conventions only for exact3.0.0 and canonical
enabled full-six inventory. Existing subset/reordered raw solves still succeed
without source declaration. Cache identity retains enabled order: no cross-order
metadata hits. Fresh/warm public hashes match and warm returns strip private stamp.
Independent Read review found and resolved those production/cache compatibility
defects; final delta had no unresolved actionable findings. No numerical science
by synthetic injection, no different-family scientific gate claimed.

Checks actually run: final installed source/sea/cache54 passed; affected source
hash/persistence/hydrostatic88 passed; source new tests22 passed before final
three added compatibility/cache cases; final source ordered-cache10 passed.
Standard isolated build and pip install succeed. 21 source/wheel/installed
Python files match; cold installed serialization works with ANYloads/ANYsolver/
Qt/viewer/Capytaine blocked. Wheel SHA256 e43f09131679cda77459e8b82353c6c13bb41abbb8b881dd3d726d005cf04549.
All checks NO BEM or mechanical integration, one thread/process. NEW90s pool
used44.50724880001144s, unused45.49275119998856s; no prior authority renewed.
Failed01qualified-DOF handling and fixture settings;02JSON persistence plus mock
request limits;03--noconftest missing fixture setup retained. Fixed without
weakening physical laws, tolerances, source qualification or execution limits.

Owned changes: ANYhydro/src/anyhydro/excitation_source.py, __init__.py,
capytaine_backend.py, hashing.py; tests/test_excitation_source.py; source-admission
contract docs. ANYmarine changes this slice are task/handoff documentation only.
Mistral unavailable, parent fallback; no Solver/Loads/viewer product edits.
Existing dirty/untracked work preserved, no commit/publication, all owned jobs
terminal. Evidence: C:/ANYworkspaceWork/anymarine-evidence/excitation-source-*/
result.json/stdout.log/stderr.log, excitation-source-reviewed-installed-tests.xml,
excitation-source-final-identities.json, excitation-source-handoff.json,
excitation-source-independent-review.md. Earlier identity/report evidence retained.

Next connected milestone: bind raw source contracts to producing equilibrium and
admitted hydrostatic/radiation artifacts in shared ANYmarine project/worker
compilation, then execute prospectively defined physical acceptance cases. Full
nonlinear lines/contact, independent HAMS comparison, W3/W4 science and complete
headless W8/GUI W9/installed Windows/Linux W10 requirements remain pending.


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
