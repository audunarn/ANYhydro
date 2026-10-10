# Independent Read review — frozen Green-function accuracy profile and panel-integration caller

## Evidence-to-source identity (verified)

- `sha256sum`: `src/anyhydro/panel_accuracy_reference.py` = `4c1167bb…9adba`, `src/anyhydro/panel_integration_candidate.py` = `9d959903…e973c2` — byte-identical to the wheel hashes, `module_sha256`, and `implementation_modules` recorded in `docs/reviews/w5-tabulated-profile-20261010.json`. The code reviewed is the code that produced the evidence.
- Recorded "48 passed" logs match exactly the parametrized instance count of the two panel test files (31 + 17 = 48), i.e., a scoped run — not the 716-def full suite. The packet does not state this scope.
- Recorded candidate claims check out against the code path and runtime counters: tabulated profile + 4-row probe → 1 construction, micro + probe = 2 native calls, return before batches/solve (`panel_integration_candidate.py:346-350`, `panel_accuracy_reference.py:429-447`); `numerical_systems: 0`, `completed_probe_unqualified` terminal state, no `radiation.npz`. Guard arithmetic is internally consistent (limit = 1e-8 + 1e-6·4.7655e-3 = 1.4765e-8; error 2.709e-9; original-step errors 2.529e-9/2.530e-9 pass).

## Prioritized findings

**P1-1. The default (no-profile) route is blocked by its own self-FD guard on the real backend — record this explicitly.**
`panel_integration_candidate.py:353-366` gates on step 1e-5 with the unchanged `1e-8+1e-6*scale` limit. The recorded `native_self_fd_errors` at step 1e-5 is 1.377e-7 vs limit 1.48e-8 — an order of magnitude fail. The seven-step sequence plateaus at 1.37833e-7 (decreasing, converging), which is a *systematic* inconsistency, not roundoff growth: the native potential S carries smooth absolute error ≤ 2.13e-9 (`potential_error`) whose spatial derivative (~1.35e-7) disagrees with the separately returned native gradient, while that gradient matches the reference to 2.5e-9 and the reference potential's own FD converges to 3.2e-13. So native S and native gradient are mutually inconsistent at ~1.4e-7; the finer table did not change this. Consequence: only the profile route can currently reach full batches. Remedy: no threshold change (original thresholds preserved); add an offline decomposition of the saved `accuracy-comparison.npz` (`native_fd` vs `reference_fd`) attributing the plateau to S table-interpolation error, and state the default-route blockage in the docs.

**P1-2. The imaginary-operator replacement is never cross-validated — and the saved probe already contains the data to do it with zero native calls.**
`panel_integration_candidate.py:385` replaces native Im with `imaginary_pole` (normalization verified against `docs/RADIATED_POWER_CONTRACT.md`: Im G_prop = −kZZ′J0(kR)/(2D), stable Z form, `panel_integration_candidate.py:76-87`; far power 2πρωk/D·∫|H|²dθ with H = (1/4π)∫σZ e^{−ik(…)}dS matches at `:185`). But no recorded evidence compares the analytic pole to the native imaginary part. `native-operator-probe.npz` holds raw native s (4×864 complex) acquired *before* replacement. An offline Im(s_native)-vs-`imaginary_pole` comparison at those 4 receivers validates the pole against the pinned native evaluator at 3456 source integrals. Expected caveat per the power contract: native Im has known coincident-row sign reversals — deviations there are themselves diagnostic. This is the cheapest uncompleted validation available from saved evidence.

**P1-3. The +0.5 diagonal jump and adjoint sign convention are asserted, never validated.**
`panel_integration_candidate.py:389` adds 0.5 to the real normal-gradient diagonal with `diagonal_term_in_double_layer=False`; `docs/panel-integration-candidate.md` asserts ".5 for all admitted submerged panels". The mock backend cannot test this (smooth mock), the guard tests only the real gradient at off-panel receivers, and zero systems were solved. This is a hard open gate for full-operator development; it needs its own bounded check (one-panel analytic limit, or comparison against Capytaine's own assembled diagonal with the native term enabled) before any full point.

**P2-4. `operator_probe` lacks an upfront node-count refusal.**
`panel_accuracy_reference.py:429-431` slices `points.reshape(-1,3)[:count]`; if the mesh has fewer flattened nodes than `count`, the slice silently shortens and fails later as a generic shape error ('invalid native operator probe') instead of a refusal. Remedy: explicit `count <= points.shape[0]*points.shape[1]` check; add a test.

**P2-5. Guard covers only the real gradient; reference imaginary part is dead evidence.**
`panel_accuracy_reference.py:250` compares `gradient…real` to `self.gradient.real`. The bundle's complex gradient imaginary part is loaded, hashed, and convergence-checked but never used, and the native imaginary gradient is never compared to anything. Consistent with the doc wording, but either disclose the unused imaginary reference or record an ungated comparison (ties into P1-2).

**P2-6. Potential agreement is ungated.**
`potential_error` 2.13e-9 is recorded (`panel_accuracy_reference.py:264`) but not enforced; the operator's pressure comes from S. Disclosed in the doc ("retained without claiming a certified bound") — acceptable, but trend it in future registrations rather than gating retroactively.

**P2-7. Test-coverage gaps against documented claims.**
Untested: cancellation between full batches / before solve / before completion (checkpoints exist at `panel_integration_candidate.py:373,394,410`; only after-micro/after-probe/after-table are tested); tabulated-profile `check_micro` failure path (only untabulated tested — shared code, low risk); `directional_evidence` refusal paths (`:154-170`); probe count exceeding node inventory (P2-4); after-probe table mutation (verify exists in `operator_probe` but untested).

**P3-8. Claim judgment — "local reference guard passed" is same-family convergence evidence, not independent acceptance.**
The reference bundle's `backend` identity is the *same pinned* Capytaine 3.0.0 native library, produced at orders 64/128. The guard therefore establishes internal consistency/convergence of one implementation. The record is honest about this (`qualified:false`, `producer_attested:false`, interpretation limits), and the method name says "declared-converged-reference". Judge: guard pass is a necessary condition only; the claim as recorded stands and must not be promoted.

**P3-9. The 8 zero-radius probe rows are explained, not anomalous.**
`r` is horizontal-only (`panel_accuracy_reference.py:391`); receiver nodes are panel 0's Gauss nodes, each coinciding with its own source node plus the horizontally coincident lid node — the lid grid (2n² panels over the same 1 m square, `first_slice.py`) shares the bottom grid's xy node locations at z=−0.1 vs −0.5. Confirmable offline from saved probe indices/coordinates.

**P3-10. Two unverified assumptions worth labeling as such.**
(a) "Special singular-panel integrator triggers at face centers, not shared Gauss nodes" — an assertion about native internals; the packet discloses `instrumented_native_trace:false`. (b) The bundle's 43-receiver center is reconstructed as the 3-vertex mean (`panel_accuracy_reference.py:139-143`) while the no-profile micro uses Capytaine's `mesh.faces_centers` (`panel_integration_candidate.py:343`); if Capytaine's quad-encoded triangle center is a 4-vertex mean, the doc's "original off-panel center" wording is inaccurate (the profile path itself is self-consistent either way). The mock test bakes in the 3-vertex mean, so tests cannot detect a divergence. Native Capytaine source is outside this repo — unverifiable here.

## Unavailable checks and unresolved gates

- Not performed (prohibited): pytest execution, native construction/Green evaluations, numerical runs. Test-pass claims rest on the recorded scoped logs plus hash identity.
- Unresolved: W5/source admission, energy/PSD/qualification (all correctly marked unmeasured); full operator and the 0.5-jump/adjoint validation (P1-3); full-point affordability (probe timing has unknown fixed-overhead share — correctly disclaimed); producer attestation and different-family independence; default-route self-FD blockage (P1-1).

## Efficient next steps for blocked full-operator development

1. Offline, zero-native-call, from saved evidence: (a) Im(s_native) vs `imaginary_pole` at the 4 probe receivers (P1-2); (b) self-FD plateau decomposition from `accuracy-comparison.npz` (P1-1); (c) zero-radius row confirmation from probe coordinates (P3-9).
2. Then a separately bounded registration for one 16-row probe (API max, still zero systems) to tighten per-row cost scaling — or directly one full batch (864 receivers, one call, zero systems), the only honest affordability measurement.
3. Only after 1–2 and P1-3's jump validation: a new bounded registration for one full radiation point; all original W5/W7 gates unchanged.
4. Optional, needs its own authority: evidence-backed reuse of the preserved `frozen-tabulation.npz` to avoid the 110.6 s reconstruction in repeated runs — currently `tabulation_cache_dir=None` is contractual; do not do it implicitly.

## Reviewed scope, tools, disposition

- Scope: `docs/reviews/w5-tabulated-profile-20261010.json`, `docs/panel-integration-candidate.md`, `docs/RADIATED_POWER_CONTRACT.md`, `src/anyhydro/panel_accuracy_reference.py`, `src/anyhydro/panel_integration_candidate.py`, `tests/test_panel_accuracy_reference.py`, `tests/test_panel_integration_candidate.py`, plus read-only greps into `first_slice.py` and `capytaine_backend.py` (lazy Capytaine import confirmed; base imports clean of Qt/ANYsolver/Capytaine).
- Successful tools: `read_file` (7 files), `bash` read-only (`sha256sum`, `grep`, `ls`, `sed`). No writes, no Git mutation, no numerical execution, no external/sibling reads.
- Disposition: no correctness error found in the reviewed accuracy-profile or caller code; physics conventions (phasors, pole normalization, far power, work-conjugate traces, full 12×12 coupled matrices, immutable input/provenance contracts, original thresholds) verified against the contract and preserved. The recorded candidate claims are supported by the evidence and honestly bounded. The material risks are the unvalidated imaginary replacement and diagonal jump (both addressable from saved evidence or one bounded registration), the default-route self-FD blockage, and the small refusal/coverage gaps (P2-4, P2-7). No edits made; findings are for the integration owner.