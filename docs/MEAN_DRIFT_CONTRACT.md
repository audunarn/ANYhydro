# Mean wave drift, far field (contract MD0)

Status: ADOPTED by the owner 2026-10-15 (rev 4) (programme plan
`C:/ANYworkspaceWork/anymarine-evidence/mean-drift-20261015/PLAN.md`, slice MD0). Rev 2/3 answer Codex review rounds 1/2
(session 01a13f6a-6dfd-7a41-95e2-2319e7702c71); the coefficients below were confirmed against Codex's independent
derivation in round 2. Rev 4 (owner, 2026-10-15) records the MD2 outcome: implemented in `anyhydro.mean_drift`
(MD2 commit 2911852), finite depth admitted after V3, and the clarifications below. Development only; nothing here qualifies drift loads.

## Scope
Mean (zero difference-frequency) second-order drift loads on one body in regular waves, unidirectional (equal heading
pairs), far-field (momentum conservation): surge force, sway force, yaw moment. Fixed bodies and freely floating
bodies with caller-supplied motions. Out of scope: bidirectional pairs, heave/roll/pitch drift, near-field or
control-surface methods, full QTFs, slow-drift forcing, multi-body drift (refused).

## 1. Own implementation; Capytaine as deep-water cross-check
Capytaine 3.0.0's `far_field_mean_drift_force` uses finite-depth coefficients c1 = 2 pi rho w and
c2 = 2 pi rho k^2 tanh(kh)^2 / D (D = tanh(kh) + kh sech^2(kh)). With exact MacCamy-Fuchs cylinder Kochin data it
reproduces the independent drift series in infinite depth but not in finite depth (ka = 1: kh = 0.5 gives 0.493 vs
1.231, kh = 1 gives 0.739 vs 1.032; the finite-depth series is the deep series times 1 + 2kh / sinh 2kh). An
energy/momentum derivation consistent with ANYhydro's source-Kochin normalisation and radiation-energy formula gives
c1 = 2 pi rho w / tanh(kh) and c2 = 2 pi rho k^2 / D (same structure for the yaw terms). Therefore:
- `anyhydro.mean_drift` implements the far-field surge/sway/yaw formulas itself (Maruo/Newman structure as in
  Capytaine) with these coefficients; MD2 must re-derive them independently (written note) and confirm them with the
  finite-depth MacCamy-Fuchs series before finite depth is admitted. Rev 4: done in MD2 (V3 ratios within 0.09 % of
  1 + 2kh / sinh 2kh at the refined mesh); finite depth is admitted (`FINITE_DEPTH_ADMITTED`). Where
  tanh(kh) >= 1 - 1e-6 the deep-water (asymptotic) coefficients are used while the dataset keeps its actual depth and
  wavenumbers, and the substitution is recorded.
- Capytaine's routine is a cross-check in infinite depth only. Because it interpolates linearly and differentiates by
  finite differences (yaw derivative factor sin(m dtheta)/(m dtheta)), agreement is judged by convergence with theta
  refinement within declared tolerances; an exact (round-off) algebra check uses the same interpolation and
  differentiation as Capytaine in a separate test.
- The suspected Capytaine finite-depth defect is recorded; reporting it upstream is an owner decision (publication).

## 2. Kochin export (Capytaine backend, opt-in)
- Backend setting `kochin: "export"` (string). Absent: request hash, cache key, schema version and dataset unchanged.
  Present: the dataset gains `kochin_radiation(omega, radiating_dof, theta)`,
  `kochin_diffraction(omega, wave_direction, theta)` (complex), the float64 scalars `water_depth` (may be +inf for
  deep water), `rho`, `g` (each equal to the `wave_environment` attribute value) and
  `wavenumber(omega)` (dispersion w^2 = g k tanh(kh) checked to 1e-12 relative), and dataset-level attributes for the
  Kochin route, reference point and theta grid (dataset level, because the complex split/recombine persistence
  helpers drop variable attributes; or those helpers are fixed to preserve them - decided in MD2, round-trip tested).
- Origins (one convention for everything): Kochin reference point = world origin (0, 0) for radiation and
  diffraction (the backend's current default); incident-wave phase origin = world origin (ANYhydro/Capytaine
  incident potential); radiating-DOF basis = the dataset rotation centre. The bounding-box centre is used only to
  size the theta grid (bandwidth), never as a Kochin reference.
- theta grid: uniform periodic theta_j = 2 pi j / N, N = `far_field_theta_count(...)` (>= 720, N > 4 k R + 64 with R
  measured about the bounding-box centre plus |centre| so that world-origin Kochin harmonics are resolved).
- Routes: indirect - `compute_kochin` on the source strengths (hull and lid); direct - `green_identity_kochin` on the
  physical hull only (lid excluded) with the scattered potential and its boundary condition for diffraction
  (sign fixed by the diffraction problem's boundary condition dphi_s/dn = -dphi_i/dn).

## 3. Motions
`motions(omega, wave_direction, radiating_dof)` complex, per unit wave amplitude, x = Re{X exp(-i w t)}, about the
dataset rotation centre, translations m/m, rotations rad/m. Admission (refuse otherwise): exactly equal omega values,
headings equal after normalisation, body-qualified radiating DOFs equal to the dataset's, all finite; no xarray
alignment. A fixed body is the explicit all-zero array. Provenance: content hash of the motion array and its
coordinates, the equation-of-motion source (matrices hash, raw or derived damping), and "fixed" for zero motions.
Rev 4: source "fixed" exactly when all motions are zero; nonzero motions need either source "prescribed" (motions
given directly, e.g. validation) or `matrices_hash` (non-empty) and `damping` in {raw, derived}; otherwise refused.

## 4. Drift computation
- Headings normalised modulo 2 pi for the computation (caller labels kept on output); equivalent duplicate headings
  refused.
- Formulas (beta the heading, H' = dH/dtheta, t = tanh(kh), D = t + kh sech^2(kh)):
      H = i (H_D + sum_j X_j H_R,j)                                  (phase factor verified in MD2)
      (F_x, F_y) = -c1 Im H(beta) (cos beta, sin beta) - c2 int_0^{2 pi} |H|^2 (cos theta, sin theta) dtheta
      M_z,0 = (c1 / k) Re H'(beta) - (c2 / k) Im int_0^{2 pi} conj(H) H' dtheta
      c1 = 2 pi rho w / t,  c2 = 2 pi rho k^2 / D   (c1 / k = 2 pi rho g / w, c2 / k = 2 pi rho k / D)
  H at the heading by trigonometric (DFT) interpolation on the periodic grid, H' by spectral differentiation,
  integrals by the periodic trapezoid rule (no theta extension). The Kochin function is numerically resolved, not
  exactly band-limited (its Fourier series has infinite tails): the theta-convergence check (N vs 2N) bounds the
  discretisation error.
- Yaw computed about the world z-axis through the origin, then transferred to the dataset rotation centre c:
  M_z,c = M_z,0 - c_x F_y + c_y F_x.
- Output: optional variable `mean_drift_far_field(omega, wave_direction, drift_dof)` real, drift_dof =
  body-qualified (surge, sway, yaw), units N/m^2 and N.m/m^2 per unit amplitude squared (unit attributes),
  provenance attributes (method, coefficients version, Kochin route, motion hash, depth treatment). The imaginary
  part of the diagonal must be <= 1e-9 of the per-DOF peak (else refuse). `mean_drift_force` (all influenced DOFs)
  stays unused until a near-field method exists (reconciles PLAN rev 2).

## 5. Schema, cache, transformations
- Schema 1.3.0 (1.2.0 is already the surface-dataset schema, `anyhydro.surfaces.SURFACE_DATASET_SCHEMA`) is emitted
  only for datasets that carry the new variables. Rev 4: the Kochin export is single-body without surface semantics;
  schema 1.3.0 refuses `surface_identity` (surface datasets stay 1.2.0 and cannot carry Kochin or drift variables).
  Legacy free-form `kochin_radiation`/`kochin_diffraction` remain allowed in 1.1.0 as before.
  Without the export a dataset keeps its existing version (1.1.0 or 1.2.0), its legacy adapter/schema identity fields
  and its cache identity. Older readers refuse 1.3.0 (validator behaviour, documented). The 1.3.0 validator checks
  dims, dtypes, finiteness, drift_dof membership and unit attributes of the new variables.
- `shift_dataset_reference_point` and every frame/reference transformation refuse datasets carrying Kochin or drift
  variables, before any mutation (an explicit map may come later).

## 6. Validation (MD2; criteria declared in the MD2 plan before running)
- MacCamy-Fuchs: bottom-mounted vertical cylinder (seabed to surface; very long cylinder for deep water), drift
  series with declared truncation convergence; ka in {0.5, 1, 2}, kh in {0.5, 1, 2, inf}; Kochin data from the series
  (formula check) and from meshed BEM at declared refinement levels (route check).
- Direct vs indirect: complex diffraction and radiation Kochin functions on the same mesh, lid and no lid, away from
  irregular frequencies.
- Nonzero intrinsic yaw: a rectangular box at oblique headings. Two separate translation tests:
  (a) reference-point transfer for one solution: M_z,c = M_z,0 - c_x F_y + c_y F_x;
  (b) physical translation of the body by a (world-origin incident phase kept): F' = F,
      M'_z,0 = M_z,0 + a_x F_y - a_y F_x, and with the output centre moving with the body its local yaw is unchanged;
      for prescribed or free motions whose basis centre moves with the body the motions carry the translation phase
      X' = exp(i k a . d(beta)) X, d(beta) = (cos beta, sin beta).
  Known complex motions and amplitude-squared scaling; absolute tolerances for symmetry-zero components; theta
  (N vs 2N) and mesh refinement.
- Deep-water cross-check against Capytaine's routine.
- L01 vs OrcaWave momentum conservation in MD3 (criteria in the MD3 plan).
- Rev 4 MD2 outcome (evidence mean-drift-20261015/PLAN.md): V2-V5 passed on the declared meshes (cylinder vertical
  panel size <= 0.1 m; analytic deep drift 0.664926 rho g A^2 a at ka = 1). Capytaine 3.0's indirect method with a
  lid mesh diverges under refinement (external defect, recorded); the direct route is used for drift.

## 7. Limits
Linear free surface (z = 0); far field gives no heave/roll/pitch drift; drift near resonance is as accurate as the
supplied motions.
