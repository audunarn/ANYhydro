## Final verification amendment, 2026-10-11

The repaired tool passes28 focused tests on Windows and Linux. Root independently
verified hand-computed complex hull rows, unchanged lid means, full coupling
and nonmutating refusals for input, existing outputs and symlinks. Final guarded
actual execution is recorded at
`.development-evidence/w5-weak-neumann-candidate-20261011/root-final-case-02/`.
It is explicitly completed_unqualified; its numerical diagnostics exactly match
the earlier run. All eight case-file hashes match the independently verified
original retained artifact, and parent summary identity is unchanged. Two
retained solves total; no Green evaluation or new kernel assembly. Scientific
closure remains FAIL, full damping has negative directions, W5 stays NO_GO.
Root adjudication and precise closed resource accounts:
`.development-evidence/w5-weak-neumann-candidate-20261011/root-acceptance.json`.
This experiment changed the test space at the same quadrature. Its work/residual
decomposition is algebraic and cannot establish a Green conservation identity.
Next check a small matched Green field against physical PDE, free-surface/seabed
boundaries and outgoing-radiation behavior before another full-case campaign.
This note provides no execution authority for that next investigation.

# Retained weak Neumann candidate (experimental, 2026-10-11)

Status: **experimental discriminator, delivered and retained.** This
candidate is not a qualified production repair, confers no scientific
qualification, and does not change any capability decision: general
admission is false and **W5 NO_GO is unchanged** regardless of the outcome
below. One development case at one frequency is all that was examined.

## What was tested

The retained-field physical-hull potential-tested (Petrov-Galerkin) Neumann
candidate of the adopted contract
(`w5-weak-neumann-candidate-20261011/contract.md`), implemented in
`tools/solve_retained_weak_neumann.py`. For the frozen case
`energy-point-native-coupled-03c` (case hash
`6d78c0fc3c6a4d5743777888e20cb0d7f1f278e40e6fae0ca42d98d36929556c`,
864 sources = 720 physical hull + 144 artificial internal lid, 4 receiver
Gauss nodes, 12 body DOFs), the physical-hull rows of the mean Neumann
system were replaced by equations projected onto the potentials generated
by the hull source columns themselves, using exactly the stored operators:

- hull rows `sum_q S_q[H,J]^H diag(w[H,q]) K_q[H,:]` and
  RHS `sum_q S_q[H,J]^H diag(w[H,q]) g[H,q,:]`, `J = H`, conjugated test
  side;
- lid rows unchanged (original panel means
  `sum_q diag(w[L,q]/area[L]) K_q[L,:]`, zero RHS on the lid);
- stored `K` already contains the +0.5 jump and is used as stored (never
  added again); all 864 source columns and the full 12-DOF right-hand side
  are retained; no coefficient averaging, symmetrization, clipping or
  tolerance change anywhere.

The system was solved once (row-equilibrated `np.linalg.solve`, equation
residual 1.1e-15, per-DOF relative residual <= 1.7e-13), and node
potential, pressure `p = i rho omega phi`, normal-velocity residual,
panel-mean residual and the physical-hull force
`F[d,e] = -sum_hq w conj(g_d) p_e` were recovered through the exact stored
operators. Diagnostics use the unchanged source-pole/angular far-power
formulas at fixed angular counts 64/128/256 and the frozen 12-vector rate
recovered from the parent `summary.json` (retained coordinates 0, 2, 6,
8). Closure verdicts use the original gate normalization unchanged
(relative 0.03, absolute 1e-8, near-zero floor 1e-8,
scale = max(|far|, |near|)).

## Verification

- 28 focused tests in `tests/test_retained_weak_neumann.py` (synthetic
  fixtures, at most 8 panels/8 DOFs) pass: hand-computed complex hull
  assembly, conjugate test side, original lid rows, jump-not-doubled,
  interleaved panel ordering with full 8-DOF coupling, synthetic
  solve/recovery identities (pressure, residual, force, near-work
  identity, lid rows equal to the mean equations), refusals (missing
  operator, case-hash mismatch, manifest tampering, nonfinite operator,
  lid trace violation, output inside the immutable tree, existing output),
  input immutability with fresh output, singular-solve failure metadata,
  and status-saved-before-solve under a simulated cancellation; plus
  production-path regressions: `main()` with `--output CASE`, a
  pre-existing output holding old status and sentinels, and an output
  symlink/junction resolving into the case tree all refuse with
  byte-identical case and output inventories pre/post, a missing case
  creates no output, a refusal after the run owns its fresh output is
  recorded only there, parent `summary.json` authentication
  (omega/rho/water_depth agreement with case.json, dispersion-relation
  fit for k, exact registered frozen-rate vector — a matching case_hash
  and length alone is refused), and the fixed actual CLI control gates
  (changed/duplicate/reordered/subset retained DOFs, angular counts,
  virtual cap and expected-case-hash overrides are refused before any
  output is created; a fresh module with the real registered controls
  admits no synthetic case).
- The tool labels every completed computation `completed_unqualified`
  with `source_admission: false` and `qualification: false` in both
  status.json and the report; no completed state is presented as a
  validated engineering result. Failure records are written only to
  the fresh output directory the run itself created and owns.
- The actual run authenticated the manifest (exact key/shape/dtype/sha256),
  geometry, operators pre/post (file hashes identical before and after:
  no input mutation), measured a single-thread runtime before assembly
  (threadpoolctl: 1 pool, 1 thread), ran under the 1536 MiB virtual
  address-space cap on Linux, and wrote complete artifacts to a fresh
  output directory
  (`.development-evidence/w5-weak-neumann-candidate-20261011/candidate-run-01/`).
  That run predates the production-path repair; its `completed` status
  state is a historical record, not a qualification, and is preserved
  untouched alongside the first code/input hashes.
- Original values recomputed from the stored fields reproduce the prior
  retained-energy audit exactly (near 0.0008646213777890402 W, far(256)
  0.00488087632611286 W, all-panel flux 0.0011483402107147767 W, residual
  work 0.0002837188329358522 W, node residual max 0.7817856039608815 m/s),
  confirming the frozen rate and evidence provenance.

## Actual results (frozen rate, 12-vector from the parent summary)

| quantity | original | candidate |
| --- | --- | --- |
| near mechanical power (W) | 0.0008646213777890402 | 0.002284044935573978 |
| far power, angular 256 (W) | 0.00488087632611286 | 0.0041185373103440025 |
| closure far - near (W) | 0.00401625494832382 | 0.0018344923747700245 |
| relative closure gap | 82.3% | 44.5% |
| closure verdict (3% gate) | fail | fail |
| physical-hull flux (W) | 0.0011652401525908407 | 0.002273129781007771 |
| all-panel flux (W) | 0.0011483402107147767 | 0.002263278413860424 |
| pressure-residual work (W) | +0.0002837188329358522 | -0.000020766521719465 |
| node residual max (m/s) | 0.7817856039608815 | 1.1618169394054858 |
| panel-mean residual max (m/s) | 1.87e-15 | 0.2810655290959646 |

Per retained coordinate (unit rates), both original and candidate pass the
3% closure gate; the candidate gaps are slightly smaller (surge 0.97% /
1.15% and heave 1.14% / 1.14%, against original 2.15% / 1.87% and 1.37% /
1.38% for the two bodies).

Candidate vs original source densities differ by 20.3% relative; the
force matrix differs by 1.9% relative. Unclipped symmetric damping
eigenvalues (`B = -Re F`): full 12x12 has four small negative values
(minimum -1.43e-3), the retained 4x4 block is positive with minimum
1.9e-6; `B` asymmetry is 0.99% relative. No passivity, reciprocity or
conservation is claimed; nothing was clipped or forced.

## Interpretation

1. **The candidate does not close the frozen-rate energy defect.** At the
   cancellation-sensitive frozen rate the closure verdict remains fail:
   the relative gap drops from 82.3% to 44.5%, an improvement by about
   half, but still far outside the unchanged 3% gate (candidate near
   0.002284 W and physical-hull flux 0.002273 W against far(256)
   0.004119 W; diff 1.83e-3 W against limit 1.24e-4 W). A ~44.5%
   remaining relative frozen gap together with a changed kinematic
   response (source densities differ 20.3% relative, node residual max
   1.16 vs 0.78 m/s, and the unclipped full 12x12 symmetric damping
   spectrum still contains four small negative values, minimum
   -1.43e-3) means the verdict is FAIL. A passing single case could not
   have established W5 anyway; this failing case is the delivered
   discriminator. No claim to fit HAMS or any external reference is
   made or implied.
2. **The candidate's pressure-residual work is small, but the
   physical-hull versus radiative discrepancy remains.** At the frozen
   rate the pressure-residual work is -2.08e-5 W over all panels, of
   which the physical HULL carries -1.09e-5 W and the artificial lid
   -9.85e-6 W (quoted separately; the original's hull residual work was
   +3.01e-4 W). The identity flux = work + residual work holds to
   machine precision, but this is an algebraic decomposition of the
   discrete arrays that is valid for arbitrary discrete fields — it is
   not a conservation or Green-accuracy check and cannot certify the
   stored boundary relation. The remaining 1.83e-3 W sits between the
   near-field physical-hull pressure work and the source-pole far
   power of the candidate's own field.
3. **This experiment changed the TEST SPACE at the SAME quadrature.**
   Replacing the hull test space with the conjugated hull-source
   potentials reweights exactly the quadrature the pair discriminator
   probed; it does not repair the source/receiver quadrature itself.
   It therefore cannot exclude hull quadrature error, kernel error, or
   a different test space as the source of the remaining defect, and
   per-coordinate closure was already passing before. The candidate
   result narrows what this exact alternative test space achieves
   (about half the frozen-rate gap); it does not support any universal
   conclusion that further test-space changes are indicated or not
   indicated.
4. **Trade-offs are visible and retained.** The candidate raises the
   frozen-rate near power from 0.00086 W to 0.00228 W (the frozen HAMS
   reference mean power is 0.00159 W; the candidate overshoots it — this
   is context, not an acceptance comparison), lowers the far power by
   15.6%, and increases the node residual maximum (1.16 vs 0.78 m/s). Its
   panel-mean residual is nonzero by construction (0.28 m/s): a
   Petrov-Galerkin candidate does not satisfy the hull panel-mean
   equations, and a vanishing mean residual was never pointwise
   acceptance.

## Uncertainty and limits

One case, one frequency, one discretization; the far power uses the
existing frozen source-pole formula (pole/angular agreement validates
algebra, not physics); the artificial lid is not a physical wetted
boundary and no auxiliary lid flux is physical outward flux; the
hull-source test space does not contain every lid-generated
physical-hull potential; HAMS/full-source comparison and any missing
evidence remain unmet. Nothing here admits the source, changes
tolerances, or promotes any capability.

## Next action

Retain the candidate as a failed-but-informative discriminator: it is a
scientific FAIL at the frozen rate, not a production repair, and the
historical `candidate-run-01` completed record is evidence, not a
qualification. What this experiment establishes is narrow: one exact
alternative hull test space at the same stored quadrature roughly halves
the frozen-rate far-versus-near defect without closing it. Which
mathematical or physical source action follows — an independent
evaluation of the stored Green relation, a kernel/convergence
discriminator, a full-source/HAMS comparison, or another test space —
is the root's decision; this slice does not authorize an automatic
further kernel/convergence campaign or any further test-space change.
