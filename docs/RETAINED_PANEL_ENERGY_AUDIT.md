# Retained panel energy audit

`tools/audit_retained_panel_energy.py` reads completed development-case arrays,
reconstructs pressure/Neumann residual/force from saved operators, and compares
mechanical work with the source-pole far power. It performs no Green evaluation,
BEM assembly, new source solve, HAMS execution or qualification. Core package
imports and physical solver implementation are unchanged.

Use a new output directory outside the immutable input tree:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python tools/audit_retained_panel_energy.py \
  --case-dir /path/to/completed/case \
  --parent-dir /path/to/completed \
  --output-dir /path/to/new-audit
```

The case supplies `input.npz`, `radiation.npz`, `case.json`, `status.json`,
`operator-N.npz`, and (unless `--skip-native`) `native-operator-N.npz`.
The parent supplies `summary.json` and `energy-comparison.json`;
`retained-blocks.npz` is optional. NPZ loading forbids pickle. Input array
shapes, finiteness, manifest hashes, case identity, reference completeness and
positive weights are checked. Reports contain actual file hashes and runtime
settings. Linux requires a positive virtual address-space cap, at most1536MiB;
this is not a measured RSS bound. Shell supervisors must enforce wall timeout.

Defaults retain surge/heave indices0,2,6,8 and angular counts64,128,256.
Velocities are peak phasors with exp(-i omega t), and F/V=-B+i omega A.
The work convention is P=-Re(v^H Fv)/2. For H=A/(4*pi), far power is
2*pi*rho*omega*k/D times the angular integral of |H|^2. The saved source-pole
quadratic form is -rho*omega/2 times REAL sum(w*conj(sigma)*(Im(S)@sigma));
there is no extra factor of pi. Agreement of these constructions checks shared
algebra, not independent physical accuracy. Deleting lid sources changes
interference; hull-only/full power is not an additive lid fraction.

The retained03c audit reproduced mechanical power0.0008646213777890402W and
far power0.00488087632611286W. Coordinate near/far discrepancies span about
1.35–2.15%; cancellation amplifies the selected rate (gross/net about28641),
but does not explain away the broader closure defect. The retained symmetric
damping minimum eigenvalue is -0.0012336511193195333. Saved real normal traces
and their +0.5 jump reproduce their producer exactly; that is not an accuracy
oracle for the underlying evaluator or formulation.

## Discrete work/adjoint diagnosis

The audit now decomposes the closure defect with exact complex-work algebra
derived from the recorded phasor convention exp(-i omega t),
p = i rho omega phi, actual normal velocity u = g + r (g the prescribed rigid
trace, r the stored node residual), quadrature weights w and panel areas
A_i = sum_q w_iq:

* flux = +.5 Re(sum_iq w p conj(u)) over all panels, and
  work = +.5 Re(sum_iq w p conj(g)) over hull panels, so
  flux = work + residual work exactly (u = g + r; g vanishes on the lid).
* flux = flux_panel_mean + flux_covariance exactly: with area-weighted panel
  means pbar_i = sum_q (w_iq/A_i) p_iq (likewise ubar), the weighted-mean
  cross terms vanish per panel, so the pointwise power splits into
  .5 Re(sum_i A_i pbar_i conj(ubar_i)) plus the within-panel
  pressure/normal-velocity covariance
  .5 Re(sum_iq w (p-pbar) conj(u-ubar)). The same split applies to work.
* far = -(rho omega/2) Re(v^H M_pole v) from the stored Im(S). Because the
  saved source strengths sigma are panel-constant, this pole form is already
  the panel-mean quadrature sum_i A_i conj(sigma)(mean_Im(S) sigma); there is
  no separate panel-mean far object.
* Defect decomposition (normalization: relative to far and to the pointwise
  defect): far - flux = (far - flux_panel_mean) - flux_covariance, and
  far - work = (far - work_panel_mean) - work_covariance.

The mean-enforced saved trace satisfies mean_K sigma = gbar to machine
precision, so ubar = gbar and the panel-mean flux is a mean-picture object on
the same footing as the pole form. The decision is data-driven: if removing
the within-panel covariance does not even halve |far - flux|, the defect
`persists_in_saved_trace_green_relation`; if the pointwise relation closes
while the panel-mean one does not, the covariance is what carries closure
(`within_panel_covariance_accounted`); machine-zero cases are labeled
`discrete_green_relation_closed`. The 1e-9 relative scale is a machine-algebra
zero for fabricated exact fixtures, not a convergence floor or waiver.

Supporting diagnostics: the force equals the negative motion-coordinate work
matrix (F = -A_work exactly for the real rigid trace — the unconjugated force
pairing versus the conjugated power pairing); projected retained 4x4
flux/work/residual/pole matrices with their panel-mean and covariance parts.
All projected matrices are exported in one public motion-coordinate
convention A[e,d] = sum_iq w conj(G[i,e]) P[i,d], so caller mean power is
.5 Re(v^H A v) for arbitrary complex velocity phasors; the pressure-first
stencil sum_d P conj(G_e) is the transpose of A. The pole form
Sigma^H W Im(S) Sigma is already motion-coordinate and is not transposed
with the pressure-first flux matrices. The full mean-power Hermitian matrix
.5 (A + A^H)/2 includes the .5 before Hermitianization, so its eigenvalues
are mean-power scale (half the work-form eigenvalues); pole eigenvalues are
Hermitian parts of -(rho omega/2) Sigma^H W Im(S) Sigma and are already
mean-power scale. A fabricated complex counterexample is tested through the
public audit fields: v = [1, i], G = I, P = [[1+2i,3+4i],[5+6i,7+8i]], equal
weights give boundary/negative-force mean power 5, while the wrongly oriented
untransposed matrix gives 3.

Weighted-S reciprocity: the meaningful same-space measure is the quadrature
sum Sum_q diag(w_q) S_q (complex and imaginary-only asymmetry). Per-quadrature
diag(w_q) S_q entries are retained as stencil information only — the target
node is fixed while the source integrates all nodes, so per-quadrature
asymmetry is not a reciprocity failure claim; a fabricated fixture tests
that per-quadrature asymmetry can cancel exactly in the summed operator.
On the retained case the summed imaginary asymmetry is machine zero while
per-quadrature entries are O(0.2-0.3). The far-form Hermitian asymmetry is
reported separately. All are postprocessing of stored operators; none
re-derives the imaginary pole, and none confers qualification. A persistent
panel-mean defect locates the discrepancy inside the stored discrete
relation; without new independent evidence it does not distinguish kernel
accuracy from discretization.

Evidence and current execution state live in ANYmarine/TASK.md and
`C:/ANYworkspaceWork/anymarine-evidence/w5-retained-energy-audit-20261011/`.
W5 remains NO_GO. The next diagnostic concerns the discrete work/adjoint
relation of panel-mean Neumann enforcement and pointwise pressure recovery.
No tolerance change, refinement campaign or source repair is admitted here.


Physical-boundary adjudication: all-panel flux includes an artificial internal
lid and is not automatically the exterior-fluid boundary integral. The audit
also reports hull-only flux, lid flux and far-minus-hull flux for every motion.
Neither a computational flux gap nor weightedS asymmetry uniquely separates
evaluator error from discretization. Weight symmetrization alone is not a
proved Galerkin formulation or an energy-identity proof; no such repair is
adopted. Next localize source/target pairs contributing to summed weightedS
asymmetry and obtain an independently matched integral/trace discriminator.
Current discrete-work evidence is in
`C:/ANYworkspaceWork/anymarine-evidence/w5-discrete-work-audit-20261011/`;
the preceding retained audit and all failed attempts remain immutable.
