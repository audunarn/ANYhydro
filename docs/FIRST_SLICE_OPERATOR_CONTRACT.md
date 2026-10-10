# First-slice BEM/operator admission contract

Status: adopted interface/derivation contract, 2026-10-08. This is a development
contract; operator development API implemented; case-specific criteria, mesh/tail convergence
and scientific qualification remain outstanding. Governing roadmap and living
record: ANYmarine/docs/IMPLEMENTATION_PLAN.md and ANYmarine/TASK.md. Preserve
the existing W2 mapping, W5 A+B reconstruction/RAO/free-decay/refinement and W7
independent-backend gates. W7 is not moved ahead of the primary first-slice gate.

## Raw evidence and constrained mechanics

The adapter keeps complete raw 6N A(omega), B(omega), excitation and explicit
Ainf, exact identities, equilibrium hull/lid meshes, frame/phasor conventions,
method/backend/version/request/settings and source hashes. Never average/clamp
in the adapter or change raw coefficients to satisfy solver admission.
Machine-precision agreement with a direct backend call verifies mapping only.

The first mechanical model retains ordered surge/heave for each of two bodies.
Define P explicitly by source DOF labels, with q_full=P q_retained and
Q_retained=P^T Q_full. Its operators are P^T X P and its excitation P^T F.
This constrained model is distinct from a freely moving full6N model. Raw full
datasets survive intact. Excluded DOFs do not supply a mechanical reference for
this slice. Reports distinguish full-source diagnostics from retained-subspace
admission; an excluded-DOF failure must never be reported as a retained pass or
silently corrected.

The public derived-operator artifact must preserve the raw dataset hash,
exact P/DOF/body order, equilibrium/mesh/lid/reference evidence, transformations,
raw retained arrays and derived arrays, settings and criteria/results. It is
a separate unqualified artifact, not raw solver output relabelled as qualified.
Its data/criteria are immutable and hash-bound. Radiation and application
services must consume its public contract without copying hydrodynamic truth.

## Work-conjugate normalization and reciprocity

For a fixed declared positive diagonal scale S, q=S q_hat,
Q_hat=S^T Q and X_hat=S^T X S preserve virtual work. Declare S and its units
prospectively. One consistent full rigid-body normalization takes translational
scales equal to a declared body length L and rotational scales equal to one:
q_hat=(u/L,theta), Q_hat=(L F,M); all normalized force components have torque
units. Alternatively q_hat=(u,L theta), Q_hat=(F,M/L) uses
S=diag(I,I/L). Do not multiply rotational inertia blocks by L^2 while claiming
the latter normalization. The first retained4 model contains translations only.

An explicitly selected reciprocal-average policy may construct
X_rec=(X_raw+X_raw^T)/2 for A, B and Ainf. It preserves the symmetric part and
satisfies reciprocity to floating arithmetic. It commutes with real coordinate
congruence: sym(S^T X S)=S^T sym(X) S. This identity does not prove that the
discarded error is small or that the raw mesh is accurate. No default automatic
processing, no PSD eigenvalue clamping, no fitted inertia or invented damping.
Hydrostatics/excitation are not silently modified by this policy.

Before admission require prospective limits on normalized correction magnitude,
raw reciprocity defects, derived impedance/response change and mesh convergence.
Use declared dimensional/normalized absolute floors plus relative terms; do not
divide by near-zero coupling coefficients or combine mixed SI-unit blocks in an
unexplained maximum norm. Record complete residuals, not only counts. Any failed
criterion is a typed non-success and preserves artifacts. The existing strict
mechanical symmetry tolerance remains unchanged.

## Passivity and radiation

PSD of the symmetric retained B is required under declared normalization and
roundoff controls. Eigenvalue sign survives nonsingular congruence; eigenvalue
magnitudes and relative severity do not. Declare magnitude thresholds in fixed
coordinates, record negative-mode vectors/alignment and refuse inadmissible B;
reciprocity averaging is not a passivity repair. Ainf plus physical dry mass
must satisfy the existing positive total-mass gate.

Do not conflate three operations:
1. A nonnegative continuous spectral density B(omega), even with a sharp
   frequency cutoff, produces a passive continuous convolution from rest.
2. Finite positive quadrature weights with PSD B samples produce a cosine-sum
   kernel with a Gram/storage interpretation. Its recurrences and frequency
   accuracy still require investigation.
3. A finite time window, horizon/interpolation or discrete temporal rule can
   alter reconstruction and passivity. Source PSD alone does not qualify the
   delivered sampled/truncated convolution or mechanical evolution.

Counterexample to a claimed compulsory cutoff failure: scalar B=b>0 on
0<=omega<=Omega and zero beyond yields
K(t)=(2b/pi)sin(Omega*t)/t (continuous limit K(0)=2b*Omega/pi).
For zero prehistory its radiation work through T is
W(T)=(1/pi) integral_0^Omega b*|integral_0^T v(t)exp(i*omega*t)dt|^2 domega>=0.
More generally this expression holds with B as the Hermitian quadratic form.
Thus a nonzero endpoint and abrupt frequency cutoff alone do not force negative
work. They can cause accuracy/tail errors; independently measure those errors.

With the present average-acceleration Newmark method, mechanical work over a
step uses delta_q=dt*(v_n+v_(n-1))/2 and
delta_W_memory=0.5*(Fmem_n+Fmem_(n-1))^T delta_q.
Use this convention with the solver's left-hand resisting-force sign and
zero-memory initial state. Report mechanical kinetic plus restoring energy,
external work, other losses, radiation work and closure error. Returning stored
fluid energy can give negative instantaneous power or a decreasing cumulative
work segment; neither alone proves non-passivity. Preserve the actual net-energy
free-decay gate and prospectively define numerical/near-zero normalization.
Equation residual balance alone is not an energy qualification.

## Prospective case admission still required

Before numerical qualification, freeze:
- exact case geometry/panel levels, declared omega/heading grids, method/backend,
  physical dry mass/equilibrium/reference buoyancy/gravity and linear link;
- firstslice P, normalization, correction/passivity/mesh criteria and near-zero
  handling, frequencies away from singular static null modes as appropriate;
- explicit B(0) or named low-frequency continuation, high-tail treatment, Ainf
  source, kernel horizon/window, dt, run duration and exact frame/resource limits;
- both A and B reconstruction criteria within the admitted frequency window;
- free-decay initial state and energy/closure criterion;
- regular-wave forcing convention, transient removal/steady-window extraction,
  common-surge drift treatment and small-excursion envelope;
- retained-subspace FD reference using identical admitted operators, physical
  mass/link/hydrostatics and exp(-i omega t);
- unchanged RAO amplitude<3%, phase<3deg and relevant refinement2% gates,
  execution allowance and independent scientific assessment.

No finite small mesh, three-frequency grid, ten-second run or unnamed tail is
presumed sufficient. Numerical tests are selected to discriminate mesh,
formulation, spectral-band and temporal error; refinement need not be monotone,
and failure is retained rather than used to tune the gates. W7 HAMS-MREL
comparison remains an additional required later gate. This contract grants no
new BEM or qualification-campaign execution authority.


