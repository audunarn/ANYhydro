# Radiated-power diagnostic contract — Capytaine 3.0.0

Status: independently assessed mathematical/runtime normalization contract.
This is not a source admission, qualified physical energy balance or new backend
capability claim. The current physical retained W5 source remains NO_GO.
ANYhydro owns hydrodynamic data and diagnostics; ANYmarine consumes its public
services. Qt and mechanics integration do not own this formulation.

## Scope, phasors and work

Use fixed equilibrium, zero forward speed, linear potential flow, positive finite
omega, rho, g and depth (infinite depth allowed), and peak phasors exp(-i omega t).
V=-i omega Xi; p=i rho omega Phi. Capytaine radiation solutions have unit
DISPLACEMENT amplitude: combine sources and Kochin values with Xi coefficients,
including explicit dimensional normalization, rather than velocity coefficients.
For a physical fluid-force phasor F=(omega^2 A+i omega B)Xi, mean body-to-fluid
power is -0.5 Re(V^H F)=0.5 Re[V^H(B-i omega A)V]. For real V, or reciprocal
real A, this is 0.5 Re(V^H B V). Do not discard the A term for arbitrary complex
velocity and nonsymmetric raw A. Earlier V^T B V records are rate quadratics,
twice the mean power; retain them and explicitly label any new mean quantity.
Use named body/DOF axes and explicit work-conjugate maps. For normalized motion
V=S vhat, Bhat=S^T B S and Ahat=S^T A S for real S. This contract's physical
target has four translational DOFs and S=diag(1 m); mixed coordinates require
their own declared dimensional scales and force map.

## Derivation and finite-depth normalization

Let omega^2=g k tanh(kh), Z(z)=cosh[k(z+h)]/cosh(kh),
D=tanh(kh)+kh sech^2(kh). The vertical modal norm is
N=integral[-h,0]Z(z)^2 dz=D/(2k). With Laplacian G=delta and the Rankine
part -1/(4 pi r), the outgoing propagating Green mode is
G_prop=-i Z(z)Z(z') H0^(1)(kR)/(4N). Its imaginary part is exactly
-k Z(z)Z(z') J0(kR)/(2D); evanescent finite-depth modes are real.
The pinned public compute_kochin normalization is
H(theta)=(1/(4 pi)) integral sigma Z(z') exp[-i k(x'cos(theta)+y'sin(theta))]dS.
It implies Phi_far=-(2/D)sqrt(2 pi k/R)Z(z)H(theta)exp[i(kR+pi/4)].
Mean pressure-work flux integrated over the far cylinder, or surface-wave
energy rho g |eta|^2/2 times group speed g D/(2 omega), gives

    P_far=(2 pi rho omega k/D) integral[0,2pi] |H_total(theta)|^2 dtheta.

H_total has units m^3/s; power has units W. No extra tanh^2(kh) appears.
In infinite depth Z=exp(kz), D=1. Stable finite-depth evaluation can use
Z=(exp(kz)+exp[-k(z+2h)])/(1+exp[-2kh]) and
sech^2(kh)=4 exp[-2kh]/(1+exp[-2kh])^2, with z in[-h,0].
Capytaine compute_kochin instead uses exp(kz) for kh>=20. Retain its actual
branch in provenance; near-seabed relative error need not be small. Quantify
the exact-mode difference before claiming general finite-depth support.

The official theory manual's eq54 has the factor2 above while eq55/56 omit it;
the printed finite-depth drift coefficient is also inconsistent with this energy
identity. The independent modal derivation and pinned Green check settle THIS
power normalization, not any drift-force implementation.
Primary reference: https://capytaine.org/stable/theory_manual/theory.html
Pinned local code identities and actual settings are in the evidence below.

## Source fields, ordering and persistence

Public indirect solve(keep_details=True) supplies sources, potential and pressure;
direct solves have no equivalent source representation and must refuse this path.
Retain every source face INCLUDING lids. Zero lid boundary velocity does not imply
zero lid source strength. Hull pressure integrates mechanical force; lid rows
are not extra physical buoyancy or load. Persist sources/centers/areas/vertices/
faces/normals, hull/lid identity and ordering, exact DOF shapes/boundary conditions,
unit amplitude, body centers, fluid/omega/k, Green/solver settings and raw forces.
Record producing geometry/request/source/result identities and conventions. No
field may be fabricated from coefficient matrices, silently dropped or inferred
from names/coordinates alone. Saved fields must reload with exact arrays/axes.
The existing ANYhydro backend provides_kochin=False remains unchanged until a
public optional source-detail service is implemented and checked end to end.

For center-panel c_i=sigma_i area_i Z_i, an independent angular oracle is
integral |H|^2 dtheta=(1/(8 pi)) sum_ij c_i conj(c_j) J0(k|x_i-x_j|).
This validates compute_kochin's center representation, not its panel-discretization
accuracy against the continuous body field. Use endpoint-excluded uniform64/128/256
angles; prospectively require successive power differences <=1e-12 W+1e-6*scale.
Check physical near/far discrepancy against separately frozen case-specific
diagnostic controls; never replace measured B by a positive Gram matrix or use
positive far power to waive PSD, refinement, RAO or energy gates.

## Evidence and next bounded outcome

Evidence root: C:/ANYworkspaceWork/anymarine-evidence/w5-radiated-power-20261010/.
Independent non-author GPT-6-Astra high derivation: astra-review.md.
Default-tabulation accuracy attempt01 FAILED its prospective1e-5 relative
imaginary-G criterion (max5.437480707143856e-4); it remains protected.
One registered untabulated check passes12 imaginary comparisons (max9.0141e-7)
and6 Rankine comparisons (max2.7828e-8), unchanged criteria. This shows setting
sensitivity, not a bound on the production body's damping error; production
settings remain unchanged. Synthetic public Kochin checks pass24 predicates:
phase/amplitude combination, angular/Bessel identity, reference shift, lid
inclusion sensitivity, missing-source refusal and nonsymmetric-A complex work.
No BEM solve, physical-source acquisition or time integration occurred.

Next: implement or explicitly evidence the public source-detail route, then
prospectively register ONE frozen-direction physical comparison at level6,
omega1.5, identical geometry/fluid/default Green, indirect keep_details=True.
A combined radiation DOF may reduce four basis solves to one only after its
shape, BC and force projection are independently checked. Its temporary named
diagnostic mode must not alter the application model or replace saved matrices.
Compare measured near force and far flux, saved indirect directional coefficients,
BC residuals and lid contribution. Preserve failure; choose the next concrete
repair/reference decision from this discriminator. Any acquisition needs a new
explicitly bounded registered numerical scope; remaining ordinary budget is
not BEM authority. The separate omega8 refinement failure remains open.


## Physical directional diagnosis and explicit next candidate â€” 2026-10-10

Evidence: w5-direction-power-20261010/ under the existing evidence root. Exactly
one indirect combined-mode source was acquired; immediate immutable raw fields
survived a postprocessing failure. Repaired assessment did not rerun the solve.
Mean near work is -0.00763276035 W,far flux+0.00252682334 W. Boundary/pressure/
angular checks pass; full original B-action and archive reproduction criteria
remain FAILED. Default weighted imaginary S is reciprocal but its quadratic
source work is negative. The preserved full untabulated evaluation timed out.

The exact-panel trace has a coincident free-surface imaginary sign discontinuity
with either table setting. All144 lid S/K imaginary diagonals show the reversed
sign. Publisher-verified Capytaine3.0.0 source archive SHA256:
2651fd1cd66624d6b5f56e84f5d0001eda34b5d108f38c39d908f425f20bc968.
libs/libDelhommeau/src/Green_wave.f90:88-97 routes coincident r,z to the special
singularity integral;152-157 sets its imaginary sign and derivative. The negative
Green normalization in matrices.f90:311-320 matches the observed reversal. No
installed library or coefficient was patched. Diagonal attribution explains
22.4835% of the imaginary source-work defect; the remainder remains unresolved.
Do not mistake that local source-code diagnosis for whole-operator qualification.

Root adopts Astra's independently supported NEXT DEVELOPMENT CANDIDATE:
an explicit auxiliary-lid source recipe at z=-0.1m for the fixed .5m-draft boxes.
This is a numerical irregular-frequency representation; physical hull/fluid,
gravity,drymass,buoyancy-only hydrostatics and mechanical links stay fixed. Keep
legacy recipe/default/hash and failures; carry the new lid identity through source
case,persistence,admission and application preparation. The pinned public
lowest_lid_position estimate at8rad/s must be checked on actual geometry before
execution and does not itself prove IR removal. A new first directional point
requires separately registered authority; all full-source/response gates follow.
This is not acceptance,an infinitesimal trace offset or an implicit project edit.

Primary context: Capytaine3.0.0 body documentation permits slightly submerged
lids and excludes them from hydrostatics:
https://capytaine.org/stable/user_manual/body.html#lid-mesh .
The documented -0.1m example and experimental free-surface warning are in the
official version2.2.1 mesh guide:
https://capytaine.org/v2.2.1/user_manual/mesh.html#extracting-or-generating-a-lid .
The current source recipe candidate must earn its own evidence. Generic ANYhydro
source-field/Kochin transport and backend capability flags remain unchanged.
