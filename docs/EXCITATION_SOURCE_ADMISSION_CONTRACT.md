# Excitation source admission contract

Revision 1, 2026-10-09. Governing requirements for the next physical-source
integration milestone. Implementation and scientific acceptance are pending.
The existing development builders and fixture-only native conversion retain
their current public contracts; this document does not admit a BEM source.

## Observed state and scope

The installed Capytaine version is 3.0.0. Its
[theory manual](https://capytaine.org/v3.0.0/theory_manual/theory.html#mathematical-problem)
uses a negative-time phasor and an incident potential with positive spatial
phase. The same expressions appear in the inspected installed
`capytaine/bem/airy_waves.py`. At the still-water surface, the zero-speed relation
`eta = i omega Phi / g` gives unit incident elevation amplitude at global
`(x,y)=(0,0)`. This amplitude/origin conclusion is a derivation from those
expressions, not numerical validation of a diffraction solve.

The current ANYhydro Capytaine adapter supplies depth, density and gravity to
`fill_dataset`, and preserves the resulting complex excitation without a sign
flip. Its neutral assembly exports request and mesh hashes but does not explicitly
export the reference incident amplitude, phase origin or source environment as an
excitation contract. `build_excitation_series` therefore requires caller
declarations. A hash alone cannot recover the missing request values.

The existing `PotentialFlowDevelopmentData` conversion admits only
`development_fixture` series. It checks full DOF order and GLOBAL axes, but does
not prove source reference-center or environment compatibility. Keep that
restriction until the physical-source admission path below is implemented and
accepted. Never relabel fixtures or unqualified sources to pass it.

## Amplitude, phase and units

Store the reference incident elevation amplitude `a_ref=1 m` explicitly. The
canonical excitation vector `X_ref` contains forces in N and moments in N.m for
that reference wave; it is not implicitly a wave-height-normalized vector.
Equivalently, the transfer vector `X_ref/a_ref` has units N/m and N.m/m.
For saved component amplitude `a_j` in metres, the physical formula is

```text
F_H(t) = sum((a_j / a_ref) Re{X_ref(omega_j,beta_j)
                            exp(i phase_j) exp(-i omega_j t)})
```

The existing SI implementation multiplies by the numeric metre amplitude,
which is the same ratio when `a_ref=1 m`. Do not introduce a factor of two for
wave height or another amplitude factor during backend or native conversion.

Both source coefficients and shared realization must use propagation headings
from +X toward +Y, radians, and the negative-time phasor. Their current admitted
phase origins must both be global `(0,0)` at still-water level zero. A nonzero or
unknown origin is a refusal, not a silent reinterpretation. If an explicit origin
translation is implemented later, its multiplier must follow

```text
eta_source = a_ref exp(i k d.(x-o_source))
eta_wave   = a_j   exp(i[k d.(x-o_wave)+phase_j])
multiplier = (a_j/a_ref) exp(i[phase_j+k d.(o_source-o_wave)])
```

That extension needs its own phase tests and versioned policy. It is not present
in the current builder. No encounter-frequency or positive-time conversion is
implied by array shape or a backend name.

## Source binding and admission

The physical-source consumer must receive the authoritative raw dataset,
canonical HydroRequest, HydroBody inventory, producing equilibrium snapshot and
source-study/qualification evidence. It must verify all of the following before
compiling coefficients or a native provider:

1. Recomputed request identity matches the dataset request hash. Recover depth,
   density, gravity, frequency and heading inventory from the saved request, not
   from defaults. If the request sidecar is missing, refuse admission.
2. Recomputed hull/lid mesh hashes, body identities, qualified influenced and
   radiating DOFs and exact reference centers match the source. A complete coupled
   matrix is required; body-by-body diagonal replacement is not equivalent.
3. The producing equilibrium pose and material hydrodynamic reference points map
   to the stored coefficient-frame centers. GLOBAL axes are required by the
   current series/native route. A center mismatch cannot be repaired by merely
   changing metadata; use the public work-conjugate reframe operation, retain its
   settings and resulting hash, then rebind dependent artifacts explicitly.
4. Source depth, gravity, density and still-water datum match the intended
   incident-wave environment. Depth classification is explicit; a finite number
   cannot stand for infinite depth. Current effects follow the policy below.
5. Backend version, adapter identity, audited convention evidence, reference
   amplitude, phasor, heading, origin and normal conventions are explicit. Unknown
   or contradictory values fail with a typed reason. Importing an external
   dataset does not supply absent conventions.
6. Existing scientific source requirements remain satisfied: complete coupling,
   refinement/reciprocity/passivity criteria, explicit A_inf provenance and
   settings, reconstructed added mass and damping, hydrostatic ownership/baseline
   semantics and source-specific motion bounds. Passing the development source
   study alone does not confer full qualification or six-DOF source admission.
7. The same materialized wave realization supplies every component to excitation
   synthesis and incident Morison kinematics. Persist amplitudes, phases, bins,
   direction weights and the realization hash. Saving/reopening never regenerates
   phases. Frequencies remain inside the admitted coefficient range; directions
   require the existing exact match modulo 2pi with 1e-12 rad conversion allowance.

A structured source contract should record these declarations and source
evidence references as data only. Bind it noncircularly to the exact dataset and
request/body/equilibrium identities. The binding itself proves identity, not
truth of a scientific claim. Reuse the existing hash and frame services rather
than duplicating them. Do not change schema 1.1.0, historical files or existing
hashes in place; introduce an explicit versioned contract extension. Preserve
the hydrostatic contract's own binding and rebuild dependent artifacts against
the resulting dataset identity when the new contract is attached.

Native source compilation must retain a distinct source-admission result. Do
not weaken the existing fixture-only provider to accept arbitrary `unqualified`
or `qualified` labels. An author-written status string is insufficient evidence.
The producing raw source and admitted derived/radiation artifacts must share a
verified source identity. Keep amplitude synthesis separate from radiation memory
and do not embed a second owner for added inertia or buoyancy.

## Current and approximation policy

The planned initial environmental model superposes prescribed still-water Airy
wave velocity and steady current for relative Morison drag. It retains
still-water wave dispersion and zero-speed BEM coefficients. State that
approximation explicitly in the project, job snapshot and result diagnostics.
It does not qualify current-modified diffraction/radiation or a Doppler shift.
Separate wave/current drag diagnostics must still sum to the one assembled drag.

Do not substitute Capytaine's forward-speed option for general multibody current.
Its [3.0.0 forward-speed theory](https://capytaine.org/v3.0.0/theory_manual/theory.html#problem-with-forward-speed)
introduces additional assumptions, encounter frequency and changed boundary and
pressure relations. The documentation limits the supplementary rigid-DOF
boundary treatment to a single body. Admitting that model requires a separate
physical and numerical contract. Nonzero current is not silently used to change
the current source request or radiation kernel.

Exact finite-rotation attachment mechanics does not enlarge the validity of the
equilibrium potential-flow source. Preserve the declared source-specific motion
envelope and invalidate coefficients when equilibrium geometry changes.

## Prospective verification

Before running a new qualification campaign, select and freeze its physical
case, normalization, near-zero floors, tolerances and execution authority. The
existing source-study and scientific thresholds remain unchanged and apply only
to their registered scope. Do not borrow the two-box surge/heave criterion as
acceptance for an arbitrary six-DOF nonlinear model.

Development checks must cover amplitude versus height, phases 0 and pi/2,
independent complex sums, both propagation axes, exact saved phases, changed
source/geometry/environment/reference identities, missing sidecars, duplicate
owners, ambiguous headings, out-of-range frequencies and unknown conventions.
Use the existing 1e-12 algebra criterion for phasor/unit identities and the
existing native development trajectory tolerances without relaxation. Numerical
BEM accuracy, actual diffraction fields and full-response acceptance need the
registered physical evidence and independent assessment; algebra is insufficient.

Source/Wheel/installed verification must exercise the public compiler and saved
project/worker route once this declaration is admitted by ANYmarine. Verify
stale-result handling, cancellation and authoritative replay arrays. A unit
helper pass does not establish application integration.

## Execution and remaining implementation

This contract review launches no numerical process and grants no campaign or
backend retry. The native/environment pool has 2.5280093000055786 actual
process-wall seconds remaining out of 200. The single disturbed-wave BEM attempt
remains consumed and unverified. Standard build/install commands also have
preserved timeout failures. Sufficient new execution authority is required before
the remaining numerical, installed-artifact and physical-source gates can run.

Remaining connected work: versioned source-contract transport and verification;
backend emission of audited normalization/environment declarations; shared
project/worker compiler admission; GUI editing of saved realizations; full source
and equilibrium/radiation qualification; nonlinear lines/contact and the complete
headless W8 demonstrator before full GUI W9 and installed desktop W10 acceptance.

## Implementation delta — 2026-10-09

The preceding revision-1 observations and spent-budget statements are historical.
Current source transport is implemented; physical/native admission and scientific
qualification remain pending. No historical campaign authority is renewed.

Public `request_to_dict` / `request_from_dict` preserve all HydroRequest values,
including nested backend settings and explicit finite/infinite depth. A missing
request sidecar is refused, rather than reconstructed from a hash or defaults.
`attach_excitation_source` returns a detached dataset carrying versioned unit
incident-elevation amplitude, global origin/datum, zero-speed, SI and phasor/
heading/normal declarations, request values, backend identity and convention audit
reference. `verify_excitation_source` requires authoritative request and body
inventory and checks full coupled axes, hull/lid hashes, centers, exact source
frequencies/headings and unchanged raw dataset identity. It verifies transport
and algebra; a status or evidence string does not establish scientific truth.

The raw binding excludes its own payload and hydrostatic transport declaration
explicitly to avoid a cycle. Attach excitation before the separately bound
hydrostatic contract. Adding that hydrostatic contract preserves the raw source
binding; its existing binding includes the excitation declaration. Attempting to
attach excitation to a hydrostatic-bound dataset refuses and requires an explicit
rebuild. Stiffness, baseline-independent coefficient data, reference coordinates,
all other provenance and qualification labels remain bound. Reframing or changing
coefficient data invalidates this declaration; no metadata-only rebase exists.
This extension changes hashes only when present; schema 1.1.0 and legacy hashes
are preserved. JSON string/dict transport hashes canonically through NetCDF.

Capytaine development adapter 0.2.0 emits the declaration only for the audited
exact backend version 3.0.0. Untested versions retain raw output without an audit
declaration. Cache identity includes adapter revision; fresh/warm public hashes
remain equal. Existing cache stamps stay private. No BEM was executed by the
transport checks; injected synthetic results exercise the public solve path and
retain their development-fixture status.

`build_source_bound_excitation_series` checks those identities and requires the
intended environment explicitly: depth_kind, water_depth, rho, gravity,
still_water_level and current_policy. The current policy is
`still_water_airy_plus_prescribed_current_for_morison_drag`. Only the existing
finite-depth/global-origin route is admitted for synthesis; infinite-depth
request transport is supported but Airy synthesis remains a typed refusal.
The function uses the existing amplitude/phase builder, preserving saved phases,
full DOF order, range/heading refusals and source qualification status.

It returns an existing DEVELOPMENT excitation series. It does not manufacture
equilibrium, hydrostatics, A_inf or a radiation kernel, grant native source
admission, confer qualification, or substitute for source-study evidence. The
fixture-only native provider restriction remains intact. Shared project/worker
physical-source compilation, equilibrium and radiation qualification, full
nonlinear response and installed desktop acceptance remain outstanding.



Review correction: automatic backend emission also requires the canonical full
enabled six-DOF inventory. Existing raw solves with subsets or reordered enabled
DOFs remain valid and return without the new declaration. Dedicated injected
public-solve cases preserve this behavior; no raw solve is rejected after BEM
solely because the new declaration cannot be emitted.

Cache identity now retains enabled DOF order explicitly. Canonical and reordered
inventories cannot share declaration metadata or suppress declaration emission
through a warm hit; alternating requests are covered without executing BEM.
