# Retained panel-pair discriminator

`tools/audit_retained_panel_pairs.py` inspects a frozen development case.
It is diagnostic tooling; W5 remains NO_GO and production admission remains
false. The original physical 3%/absolute 1e-8/floor 1e-8 criteria are unchanged.

## Scope and identity

The independent primitive is the constant unit source over a physical triangle:
`G_R = -1/(4*pi*r)`, `grad_x G_R = (x-y)/(4*pi*r^3)`.
This direct Rankine term is only one part of the finite-water Green function.
There are at most 12 unordered pairs, both directions and the original four
receiver nodes: 96 unique node/source comparisons. Each uses fixed 16/32/64
controls. No native/full Green evaluation, BEM assembly, source solve,
Capytaine, HAMS or Prony execution occurs.

The tool verifies snapshot manifest array hashes, shapes and dtypes; finite
geometry, positive weights/areas, face indices, unit oriented normals, ownership
and hull masks; file identity before/after execution; and a fresh output resolved
outside the immutable case and its three ancestor trees, including symlinks.
Linux requires measured one-thread numerical runtime before loading operators.
Its 1536 MiB RLIMIT_AS is a virtual address-space cap, not a resident-memory claim.

## Reconstruction and independent reference

Sequentially load four stored operators and form `M = sum_q diag(w_q) S_q`.
Rank unordered pairs by `abs(real(M_ij-M_ji))`, preserving worst pairs and
reporting body identity, hull/lid roles, adjacency and source/receiver geometry.

The analytic triangle potential uses boundary edge antiderivatives; its
receiver gradient uses tangential edge terms and signed solid angle. An
independent projection-anchored radial integration removes the internal
near-coplanar ridge. Radial moments are exact; angular integration uses
Gauss-Legendre 16/32/64. For radius R and signed plane height h:

```
potential moment = sqrt(R^2+h^2) - abs(h)
normal derivative moment = h/sqrt(R^2+h^2) - sign(h)
tangential gradient moment = -dhat * [asinh(R/abs(h))-R/sqrt(R^2+h^2)]
```

For an in-plane receiver outside the triangle, the common divergent tangential
constant cancels across signed sectors. A zero-area sector on an edge line
contributes zero for a valid off-plane receiver. Actual on/near-triangle
singularities and degenerate geometry remain refusals.

Neither radial control calls the analytic candidate. The original centroid
Duffy 16/32/64 data remain labelled, including failures. Symmetric 4x4 source
and receiver quadrature checks assembly algebra only.

## Prospective diagnostic criteria

Each node must satisfy both fixed-level convergence and analytic/control match:

```
scalar difference <= 1e-10 + 1e-7*abs(reference scalar)
gradient norm difference <= 1e-8 + 1e-6*norm(reference gradient)
```

Store 16-to-32 and 32-to-64 differences. The scalar observed error estimate
is the maximum of both differences and the analytic/control mismatch.
Conservatively sum all eight absolute receiver-weighted estimates for pair
uncertainty; no RSS independence is assumed. Quantitative attribution requires
every scalar and gradient check to pass and uncertainty <=
`0.1*max(abs(stored skew), abs(reference skew), 1e-12)`.
These are observed estimates, not rigorous error bounds.

## Verified development result, 2026-10-11

Final focused tests: 25 passed, including closed forms, scaling, rotation,
orientation, additivity, independent volume quadrature, injected candidate
gradient error, snapshot mutations and production-path refusals.

Final retained-case report:
`C:/ANYworkspaceWork/anymarine-evidence/w5-panel-pair-discriminator-20261011/root-final-case-04/`.
Frozen case hash: `6d78c0fc3c6a4d5743777888e20cb0d7f1f278e40e6fae0ca42d98d36929556c`.
Reconstructed weighted-S asymmetry is exactly 0.03446422663563036.
All 12 selected pairs pass the diagnostic controls, with 96 analytic,
288 radial and 288 legacy Duffy evaluations. Measured native thread count is
one; case input files are unchanged.

The 12 nonadjacent same-body hull/lid pairs capture only 3.4021% of the
total absolute real skew mass. Their hull and lid normals are perpendicular.
Each stored skew is approximately -1.61321313e-5; the mixed exact-source,
four-receiver-node Rankine skew is -1.61828348e-5, leaving +5.07035125e-8
unexplained (about 0.3143% of stored magnitude). Conservative uncertainty is
approximately 1.79e-10 per pair.

This establishes the mixed quadrature contribution on the selected pairs.
It does not establish a global cause, total Green accuracy, physical energy
closure or a production repair. Across the full operator, approximately
99.4% of real skew mass is same-body; the largest category is hull/hull.
Next assess a source/receiver-consistent discretization and its jump/normal
trace contracts before proposing a repair. No forced symmetry, eigenvalue
clipping, tolerance change or qualification claim is adopted.

## Historical failures

Original Duffy-only actual-case runs 01/02 remained partial. Worker03 timed out
with 21 tests passing and one failing, without a new handoff; its candidate is
archived. Root repaired the off-centroid fixture and independent controls.
Radial finite differences and ordinary edge quadrature were insufficient at
the edge-projected near-coplanar nodes; these failed/intermediate checks and
reports remain. The final radial angular gradient control resolves that
reference obstruction within the original fixed-level bounds.
All evidence is under `w5-panel-pair-discriminator-20261011` in the workspace
and ANYhydro ignored development-evidence directory.
