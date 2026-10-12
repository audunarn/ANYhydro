# Finite-depth Prony decomposition: W5 free-surface attribution and repair

Status: development evidence, 2026-10-12. Qualification is not claimed.

## Finding

The W5 physical Green-field check (`tools/check_physical_green_field.py`,
kh = 34.4, h = 150 m) passed every PDE predicate except the free-surface Robin
condition, which failed at all four surface samples by 2.67–3.15 times the
unchanged criterion `abs 1e-9 + rel 1e-4`.

An independent oracle (`tools/check_finite_depth_john_oracle.py`, John
eigenfunction series, no Capytaine) reproduces the saved native real potential
and vertical gradient at all 11 base receivers of both sources to ~1e-13
relative as

    S_native = -1/(4 pi) [ Gauss4(G_exact) - Gauss4(M_eps) + MIX ]

* `M_eps` is the field of the Prony fit error of the frozen 4-term
  decomposition (MSE 7.1e-5, max |eps| 0.07 on [-0.1, 20]);
* `MIX` is the kernel's one-point centroid rule for Rankine-type images
  against the 4-node rule for wave terms.

Only these two terms violate the free-surface condition. The measured residual
equals the prediction to ~1e-12; the Prony error is 97–100 % of it. With the
frozen coefficients the native field differs from the exact one by ~1 % in the
far field and up to 31 % at deep interior/bottom receivers.

Cause: Capytaine 3.0's default `python` decomposition accepts the first fit
whose mean-square error is below 1e-4 on a randomly perturbed grid. That admits
pointwise errors of 0.02–0.08 at every kh tested (0.3–2000).

## Repair

`anyhydro.prony_fit` fits deterministically (Prony's method, 4–30 terms) and
accepts a fit only on its measured maximum pointwise error on an 8001-point
grid of [-0.1, 20] plus the tail window [20, 60]: target 1e-4, acceptance
5e-4, otherwise `CapabilityError`. The removable pole at x = kh is evaluated in
60-digit decimal arithmetic; kh >= 1e5 and omega = inf use Capytaine's
asymptotic form. `validated_delhommeau()` injects it through
`find_best_exponential_decomposition` and records
`finite_depth_prony_decomposition = anyhydro-validated-prony-v1` in the Green
function settings. The production Capytaine backend uses it (adapter 0.3.0;
cache identity includes the method).

Evidence (C:/ANYworkspaceWork/anymarine-evidence/w5-prony-attribution-20261012):

| | Robin ratios hull / lid | max field error vs exact |
|---|---|---|
| frozen (native control, bitwise equal to the record) | 2.67, 2.90 / 2.92, 3.15 | 31 % |
| refit 15–18 terms (native, pinned E1 core) | 0.0005, 0.0018 / 0.090, 0.053 | 1.3e-5 |

The remaining lid floor (~0.09) is the centroid/Gauss mixing term.

Impact on BEM coefficients (10 x 5 x 2 m box, h = 150 m, omega 0.3–2.0):
added mass and damping change by 1e-4 to 2e-3 relative. Near/far energy
consistency (far-field Kochin flux over near-field damping) of the same box
moves by at most 2e-3 (heave, omega = 2) and typically 1e-5 to 1e-4 between a
deterministic upstream 4-term fit and the validated fit: the fit-error field is
non-radiating and mainly affects added mass. This repairs a real accuracy
defect but does not explain the 1.35–2.15 % energetic-mode near/far gaps of the
retained W5 coupled point 03c, which therefore have another cause.

## Known limits

* For kh below about 8 the remainder has an algebraic tail
  `ref(x) ~ (4 kh tanh kh - A)/x` that no exponential sum represents beyond the
  fit range (upstream or validated). Measured with the validated fit: field
  error <= ~2e-4 and nearest-surface Robin ratio 1.0–1.3 at kh = 1–3.
* kh <= 0.3 is refused (best 30-term fit 4e-3); Capytaine refuses kh <= 0.1.
* The artifact-bound W5 reference profile still binds the frozen original
  producer coefficients by design; comparisons against that reference inherit
  its fit error.

## Exact imaginary part (adapter 0.4.0)

W5 evidence (anymarine-evidence/w5-route-comparison-20261012) showed that Capytaine's
default tabulated wave term gives a pressure-integrated damping matrix with a
negative eigenvalue for two boxes 4 m apart (omega = 1.5 rad/s, h = 150 m): about
-0.011 against a largest eigenvalue of 206, unchanged by mesh refinement or a
finer table, while far-field (Kochin) damping is positive (+0.0015). Without
tabulation the near-field value becomes +0.0015 and near/far agree on every mode,
but the solve is about 140 times slower. The imaginary part of the outgoing
finite-depth kernel is exactly the propagating mode
`Im G = -(k/(2D)) Z(x3) Z(xi3) J0(k r)`; `anyhydro.green_function` evaluates it
analytically with the same source quadrature, keeping the tabulated real part.
Capytaine's untabulated imaginary part agrees to ~1e-8, except for a source and
receiver both exactly on z = 0 (waterline lid self panels), where upstream flips the
sign (+k/2 per unit area instead of the continuous -k/2); the exact term is
continuous there. With realistic lid refinements this changes damping by < 0.03 %.
Matrix evaluation costs about 1.3 times the default (independent measurement); the
two-box damping becomes positive and far-field consistent.
