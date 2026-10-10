# First-slice BEM preparation checkpoint

Revision: 2026-10-08, first-slice preparation owner (Mistral). Final state:
implemented and verified against the assigned slice criteria (author
verification, not acceptance). Repair revision, same date and connected
owner: parent findings 1-3 repaired — see the "Repair" section (hash
identity fix in `hashing.py`, heading-claim retraction, worker-06 breach
recorded; no BEM executed during repair). The early-checkpoint findings and design
are retained below; results and evidence follow.

## Results (all evidence under C:/ANYworkspaceWork/anymarine-evidence/)

- New focused tests: 61 passed
  (`tests/test_execution_limits.py`, `tests/test_first_slice_fixture.py`;
  worker-04, exit 0, wall 2.69 s). Earlier repair iterations preserved:
  worker-01 (collection error, 2.54 s), worker-02 (8 failures, 3.23 s),
  worker-03 (2 failures, 2.81 s).
- Existing adapter admission/import tests: 15 passed (worker-05,
  3.06 s). Full existing `tests/test_capytaine_backend.py` including real
  BEM solves: 29 passed (worker-06, 10.83 s) — BREACH OF EXECUTION
  AUTHORITY: the brief authorised exactly ONE real BEM execution (the
  pilot); worker-06 (`runtime/Scripts/python.exe -m pytest
  tests/test_capytaine_backend.py -q`, cwd `C:/Github/ANYhydro`, wall
  10.83 s, exit 0, evidence `first-slice-prep-worker-06/`) executed
  additional real BEM solves and was NOT authorised. No further BEM of
  any kind — legacy tests or fixtures included — is permitted.
- ONE real Capytaine 3.0.0 BEM pilot through the public source anyhydro
  path (`PYTHONPATH=src`, proven by `anyhydro_file` in the JSON):
  `first-slice-prep-bem-01`, exit 0, wall 2.52 s (harness timeout 25 s),
  solve 1.34 s. All declared checks passed: fixture verification (96
  panels), full 12x12 finite coefficients, nonzero cross-body blocks
  (max cross added mass 67.7 kg, cross damping 749.8 N.s/m), explicit
  omega=inf A_inf source, analytical heave buoyancy stiffness exactly
  10055.25 N/m on both bodies (relative error 0.0, tolerance 1e-10), no
  input mutations. Dataset NetCDF + JSON with geometry/request/source
  hashes: dataset content hash
  `815208beac59b01bca5062721986dbf2105bfb7d4e5b158f990b9c1b3abebbf4`,
  request hash `61d2a6c200c5dd85d5cb81dd4249fbe5b53e00fb033912c7cd0e79760f1c9b67`.
- Raw symmetry/passivity (reported, NOT a gate, no tuning): added-mass
  asymmetry 0.87-3.6% relative, damping asymmetry 1.7-4.3%, A_inf
  asymmetry 0.59%; one slightly negative damping diagonal at omega=2
  (-5.6e-4 N.s/m, raw observation); B(omega_max=6) diagonal reported in
  full in the JSON (9.5 to 1495.7 N.s/m), not claimed small. Capytaine
  warned that panel radius (0.373 m) exceeds wavelength/8 at omega=6 — a
  development-fixture resolution caveat, retained in stderr.log.
- Worker process-wall budget: ~27.7 s of the 90 s allowance used
  (2.54+3.23+2.81+2.69+3.06+2.52+10.83), one process/thread,
  OMP/OPENBLAS/MKL=1, TEMP/TMP under the evidence tree; all runs through
  `run_check.py` with exact commands, stdout/stderr and wall times on
  disk.

## Repair (2026-10-08, same connected implementation owner)

Parent findings 1-3 repaired; no BEM of any kind was executed (the worker-06
breach above is the last BEM on record and is not authorised retroactively).

### Finding 1: execution_limits hash identity — root cause and fix

Investigation before editing (evidence `first-slice-prep-repair-01/`,
script `first_slice_repair_discriminate.py`, wall 1.26 s, exit 0): the
parent's hypothesis was corrected — `execution_limits` is NOT a dict in
memory. `schema.build_hydro_dataset` JSON-encodes `extra_attrs` dict values
(`json.dumps(..., sort_keys=True)`, default separators), so the freshly
solved dataset carries it as a JSON *string*; `persistence._decode_attrs`
already decodes it back to a *dict* on load. The defect was in hashing only:
`execution_limits` was absent from `JSON_DATASET_ATTRS`, so the in-memory
string was hashed as `str::<raw text>` while the reloaded dict was
canonicalized as `json::<compact sorted>` — two different canonical forms
for the same content. Discrimination on the retained pilot NC (read-only,
never solved) proved this is the sole difference: the reloaded dataset hashed
to `15d7daba...` (matching the parent's installed value exactly) and the
reconstructed in-memory form (execution_limits restored to the pilot's
original string) hashed to `815208be...` (matching the pilot's in-memory hash
exactly), with `execution_limits` the only attribute whose canonical form
differed.

Fix: one line in `src/anyhydro/hashing.py` — `execution_limits` added to
`JSON_DATASET_ATTRS`. `persistence.py` needed no change (its decode rule was
already canonical); `schema.py` untouched; no arrays, no original evidence,
no retrospective modification of the pilot record. The historical in-memory
hash `815208be...` is pre-fix evidence and will not reproduce by design;
post-fix, in-memory, reloaded and re-saved forms all hash identically.

Post-fix verification (evidence `first-slice-prep-repair-03/`, script
`first_slice_repair_verify.py`, wall 1.20 s, exit 0): on the retained pilot
dataset, reconstructed in-memory form, reloaded form and a scratch re-saved
+ reloaded form (re-save under the repair temp dir only; original NC
untouched) all hash to `15d7daba...`; the reloaded hash is unchanged from
the parent's installed value; the synthetic fixture without the attribute
keeps its pre-fix hash `2db117e6...` (existing dataset hashes unchanged).

Regression tests: NEW `tests/test_execution_limits_persistence_hash.py`
(3 tests, fully mocked synthetic fixture; no BEM, no capytaine import):
string/dict forms hash identically; save/load and re-save preserve the hash
and the effective limits; the no-execution-limits fixture hash is pinned to
its pre-repair constant. Scoped run with the inspected BEM-free
`test_hashing.py`, `test_persistence.py`, `test_schema.py`:
45 passed, exit 0, wall 3.01 s (evidence `first-slice-prep-repair-02/`).

Pilot script: `examples/first_slice_bem_check.py` now records roundtrip
evidence for FUTURE executions (`persistence_roundtrip` summary block and a
`persistence_roundtrip_hash_identity` declared check: reload, re-save,
reload again; identical hash and effective limits). The pilot was NOT
rerun; no new BEM.

### Finding 2: heading prose

`schema.py` `HEADING_CONVENTION` and `docs/CONVENTIONS.md` were verified by
the parent to be already correct; the claims of an outstanding heading
mismatch in this checkpoint are retracted above. No string, schema or hash
was changed; numerical conventions unchanged.

### Finding 3: worker-06 breach and raw retained diagnostics

The breach is recorded in the Results section above with the exact command
and evidence path; it is not authorised. Raw retained diagnostics
(read-only, `first-slice-prep-parent-diagnostics-01/`,
`first-slice-retained-diagnostics.json`) fail solver symmetry and remain
UNQUALIFIED raw observations — no gate, no clamp, no symmetrisation, no
tolerance change: A_inf relative asymmetry 0.0007958026; max added-mass
asymmetry 0.0184723760 (omega 4 rad/s); max damping asymmetry 0.0430789203
(omega 2 rad/s); symmetric-part damping minimum eigenvalue -0.0667438505 at
omega 2 rad/s. The first-slice fixture and radiation are NOT qualified.

### Repair wall accounting

This repair session used 5.47 s of process wall through `run_check.py`
(1.26 + 3.01 + 1.20 s, runs `first-slice-prep-repair-01..03`), one
process/thread, `OMP/OPENBLAS/MKL_NUM_THREADS=1`, `TEMP/TMP` under the
evidence tree, `PYTHONPATH=src`. Combined with the previous session's
27.68 s (seven runs), 33.15 s of the 90 s allowance is used; the final
verification (1.20 s) ran inside the 15 s reserve. The parent's separate
30 s installed verification allowance is untouched.

### Remaining uncertainty for the parent

`cache_identity` (JSON string stamped only on cache copies in
`capytaine_backend.solve`) is not in `JSON_DATASET_ATTRS`; a cache-stamped
dataset reloaded directly would exhibit the same str/dict canonical
asymmetry for that attribute. It never appears on `solve()` returns or the
pilot dataset, so no pinned hash is affected; left for the parent to
adjudicate (same one-line treatment would apply).

## Changed files (owned boundary only)

- `src/anyhydro/capytaine_backend.py` — `CapytaineExecutionLimits`,
  constructor injection, effective-limits provenance (capabilities,
  diagnostics, cache identity, dataset attrs), caps before cache lookup,
  new downward-lid admission check, heading prose fix.
- `src/anyhydro/first_slice.py` — NEW deterministic fixture + analytic
  verification + case constants.
- `src/anyhydro/__init__.py` — exports.
- `tests/test_execution_limits.py`, `tests/test_first_slice_fixture.py` —
  NEW focused tests (mocked solves, no BEM in unit tests).
- `tests/test_execution_limits_persistence_hash.py` — NEW repair regression
  tests (mocked; hash identity across execution_limits persistence).
- `src/anyhydro/hashing.py` — repair: `execution_limits` added to
  `JSON_DATASET_ATTRS` (one line; the only source change of the repair).
- `examples/first_slice_bem_check.py` — NEW pilot script; repair: added
  persistence roundtrip evidence for future executions (pilot not rerun).
- `docs/FIRST_SLICE_PREPARATION_CHECKPOINT.md` (this file),
  `docs/CAPYTAINE_ADAPTER.md`, `README.md` — documentation.

No marine/solver/sibling edits, no TASK.md edit, no Git operations, no
dense BEM, no W5 qualification claim, no time-domain runs.

## Concrete findings (verified by direct reads this session)

- Baseline `C:/Github/ANYhydro` is fully untracked (no commits); existing
  adapter `src/anyhydro/capytaine_backend.py` (664 lines) enforces module
  constants `MAX_TOTAL_PANELS=100`, `MAX_FINITE_OMEGAS=3`,
  `MAX_WAVE_DIRECTIONS=1` inside `solve`, before cache lookup.
- Installed Capytaine 3.0.0 (runtime
  `C:/ANYworkspaceWork/anymarine-evidence/runtime`) pins the lid convention:
  `bodies/bodies.py:98` forces `lid_mesh.with_normal_vector_going_down(...)`
  and `Mesh.generate_lid` builds lids with `normal=(0.0, 0.0, -1.0)`. So
  waterplane lid normals must point DOWN. Capytaine would silently flip a
  wrong-oriented lid; the adapter must not rely on that (no silent
  symmetrisation of source geometry).
- Capytaine hydrostatics (`bodies/hydrostatics.py`): heave-heave stiffness
  is `rho*g*waterplane_area` of the immersed hull;
  `waterplane_area = -integral(n_z dS)` (`meshes/surface_integrals.py`).
  For the open-top wetted box (bottom + 4 sides, top edges at z=0) this is
  exactly 1.0 m^2, the divergence-theorem volume is exactly 0.5 m^3 (the
  missing top at z=0 contributes zero) and the COB is exactly
  `(cx, 0, -0.25)`. Analytical heave buoyancy stiffness:
  `rho*g*A_wp = 1025*9.81*1.0 = 10055.25 N/m` (the brief's "10055 N/m" is
  read as this exact value from the case constants).
- Runtime has pytest 9.1.1, numpy 2.5.3, xarray 2026.9.0, scipy 1.18.1,
  Python 3.13.9, Capytaine 3.0.0.

## Design and edit boundary

Owned edits (nothing else is touched):

1. `src/anyhydro/capytaine_backend.py` — new frozen public
   `CapytaineExecutionLimits` (defaults 100 panels / 3 finite frequencies /
   1 heading / n_jobs=n_threads=1; ceilings 100 / 513 / 1; n_jobs and
   n_threads pinned to 1). Rejects bool, fractional (non-Integral), finite
   floats, non-finite, negative, zero and unknown controls with typed
   `CapabilityError` at construction — before any Capytaine import or cache
   lookup. `CapytaineBackend(execution_limits=...)` takes it explicitly;
   effective limits appear in `capabilities()`, `diagnose()`, the cache
   identity and dataset provenance attrs. All caps still apply before cache
   lookup. New admission check: lid face normals must already point
   downward (typed `GeometryValidationError`), so the adapter never depends
   on Capytaine's silent auto-flip. Heading prose corrected to: heading
   beta=0 propagates along +X, beta=pi/2 along +Y, headings measured
   counterclockwise from +X toward +Y (numeric conventions unchanged).
2. New `src/anyhydro/first_slice.py` — deterministic two-body wetted-hull +
   lid fixture: body ids `first-slice-a` / `first-slice-b`, refs
   (-2,0,0) / (2,0,0) m, COM relative (0,0,-0.25), upright 1 m x 1 m box,
   draft 0.5 m, hull = bottom + four sides over z in [-0.5, 0], each of the
   five faces subdivided 2x2 quads -> 2 triangles each => 40 hull panels per
   body; waterplane lid 1 m^2 at z=0, 2x2 quads -> 8 lid panels per body,
   normals DOWN; 96 panels total; no dry top. Shared vertices where
   practical (25 hull vertices, 9 lid vertices per body). Analytic
   verification (normals, topology, areas, volume 0.5 m^3, COB z=-0.25,
   waterplane area 1.0) plus case constants (rho 1025, g 9.81, depth 150,
   dry mass 512.5 kg, reference weight 5027.625 N, heave buoyancy stiffness
   10055.25 N/m, omegas [2,4,6] rad/s, heading 0, surge/heave projection
   metadata only).
3. `src/anyhydro/__init__.py` — export `CapytaineExecutionLimits` and the
   fixture module symbols.
4. New focused tests: `tests/test_execution_limits.py`,
   `tests/test_first_slice_fixture.py` (unit/mocked; no BEM execution).
5. New `examples/first_slice_bem_check.py` — the ONE real Capytaine 3.0.0
   pilot through the public source anyhydro path (fixture 96 panels,
   omegas [2,4,6], heading 0, explicit A_inf, n_jobs=n_threads=1), writing
   dataset NetCDF + JSON (geometry/request/source/version hashes, raw
   symmetry/passivity report, limits) and preserving all output on failure.
6. `docs/CAPYTAINE_ADAPTER.md`, `README.md` — limits/fixture/heading-prose
   documentation updates.

Explicitly NOT edited: `schema.py` (parent read verified its
`HEADING_CONVENTION` prose and `docs/CONVENTIONS.md` heading prose are
ALREADY correct — "wave propagation direction from +X toward +Y, radians";
the earlier claim of an ambiguous/outstanding heading mismatch was wrong and
is retracted; the strings and their hashes are untouched and numeric
conventions are unchanged), `fixtures.py`, `radiation.py`,
marine/solver siblings, `TASK.md`, Git.

## Reserved choices (parent may override)

- The analytical heave stiffness is taken as `rho*g*A_wp = 10055.25 N/m`
  (the brief's "10055" read as the exact product of the case constants);
  the pilot checks it at relative 1e-10.
- The lid record's `outward_normals_verified` is set to `False` honestly:
  lid normals are deliberately downward (pinned Capytaine convention), i.e.
  NOT outward from the enclosed volume; the fixture's verification report
  certifies downward lid normals explicitly and the backend enforces the
  downward orientation geometrically at admission.
- `docs/CONVENTIONS.md` heading prose is left untouched (outside owned
  boundary); flagged here for the parent.

## Remaining work

None within this slice's authorised scope. Outstanding items for the
parent (not blockers, reserved choices recorded above):

- The 513-frequency explicit-limit admission is implemented and tested
  (unit-level, mocked) but never executed here, as instructed.
- Heading prose: no outstanding mismatch exists. Parent read verified
  `schema.py` `HEADING_CONVENTION` and `docs/CONVENTIONS.md` are already
  correct; nothing remains to change (this checkpoint's earlier claim of an
  outstanding heading mismatch was wrong and is retracted above).
- The raw asymmetry/negative-damping observations and the omega=6 mesh
  resolution caveat are evidence for the parent's later qualification
  decisions; no gate was tuned to them.
- Mechanical surge link, FD/TD projections and any dense/W5 runs remain
  with their own contracts; nothing in this slice alters application
  caps, radiation settings or the solver.


## Parent reconciliation — supersedes incomplete cache-repair handoff

The earlier assertion that cache_identity never appears on public solve returns
was false: the warm cache path returned the stamp. The candidate removes only
that validated private attribute on return, preserving the disk stamp; hashing
now canonicalizes deliberate raw-cache JSON/dict forms. New mocked public-path
cache tests verify fresh/warm hash identity and mismatch refusals. Parent built
and verified final installed candidate:117passed (Capytaine import blocked),
canonical and installed source hashes match. Original pilot bytes unchanged.

Worker cache repair used an incorrect runtime/harness and produced three failed
repository-local evidence attempts; preserve them. Two worker turns ended without
verified handoff. Parent verified final candidate because execution was unreliable.
No further BEM was run; earlier extra-BEM authority breach remains recorded.
Measured resource totals and runtime-probe accounting uncertainty are in
C:/ANYworkspaceWork/anymarine-evidence/first-slice-preparation-final-manifest.json.
This is development verification only; raw pilot fails numerical admission and
RAO/free-decay qualification remains outstanding. TASK.md in ANYmarine governs
the full goal and next milestone. No active owned workers or Git publication.
