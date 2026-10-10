# Experimental panel-integration source

`anyhydro.panel_integration_candidate.run_panel_integration_candidate` is a
versioned finite radiation candidate, not a `HydrodynamicsBackend`, canonical
dataset, qualified capability or GUI result. Existing Capytaine defaults, cache
identities and source study gates remain unchanged. Capytaine is imported lazily;
the exact tested version is3.0.0. Own implementation is MPL-2.0.

The complete candidate uses positive four-node tensor Gauss quadrature on each
collapsed triangle for source integration AND receiver testing/pressure forces.
Native public `Mesh.with_quadrature`, `Mesh.extract_faces` and
`Delhommeau.evaluate` supply the real single-layer operator and receiver gradient.
The imaginary operator AND gradient use the finite-depth propagating pole in
`RADIATED_POWER_CONTRACT.md`, integrated with identical source quadrature weights.
Geometric area, first and second moments are checked on the actual native nodes.
These checks establish rigid geometric moments, not exact pressure/Green integrals.

All quadrature nodes must be below z=-1e-6m; free-surface panels are refused before
backend import/evaluation. The pinned native low-frequency special singular-panel
integrator triggers at face centers, not shared Gauss nodes. Coincident quadrature
nodes on z=0 lids would create a REAL singularity; imaginary replacement is not a
repair. No point displacement or gap clamp is used. The existing explicit
`submerged_0p1m_v1` fixture is a supported candidate input; its earlier stock source
failure is retained. Hull-only inputs have no irregular-frequency-freedom claim.

Each receiver quadrature node gets its own source-panel real normal jump(.5 for
all admitted submerged panels). The native evaluator's flattened-index diagonal
is disabled. Mean Neumann equations are tested with receiver weights divided by
panel area. RHS uses exact world rigid velocities about the declared body reference
points; auxiliary lids have zero mechanical RHS. All six DOFs per body are kept in
one coupled system. Forces use actual node rigid traces, pressure and AREA weights;
there is no center-only moment approximation or normalized force weight.

With peak velocity columns and exp(-i omega t), p=i rho omega phi and
F/velocity=i omega A-B, hence A=Im(F)/omega and B=-Re(F). Complex directional
mechanical power uses the full force action, preserving asymmetric added mass.
Outgoing wave amplitudes use every numerical source and the same source quadrature.
Pole source/far equality is constructed and never independent acceptance.

The record exposes raw node residuals, mean residuals and actual pressure-work
residuals, with signed and absolute contributions on hull/all boundaries. A passing
mean residual proves only the mean system was solved. It does not establish pointwise
BC accuracy, pressure-weighted accuracy, radiated-power balance, PSD or mesh/band
qualification. No thresholds, coefficients or modes are corrected to manufacture
a pass. The frozen physical point must be independently assessed before any full
campaign; all original W5/W7 gates remain.

Execution requires explicit `CapytaineExecutionLimits` and a new artifact directory.
There are at most two one-triangle real-gradient finite-difference checks followed
by four full Green batches and one full6N radiation system. A failed micro-check
(prospective1e-8 +1e-6*gradient scale) stops before the full batches. Input/case hashes,
effective native settings/threads, failed derivative arrays, each native S/full
gradient immediately after acquisition (before validation/replacement), derived batches
and raw coupled results are saved. Arrays returned to callers are immutable. Job
supervision owns the process-tree wall limit; callers must register their authority
before execution. Saving `completed_unqualified` evidence does not produce a
completed validated application result.

Diffraction, A-infinity, zero frequency, sheets, generic panels and production
admission are outside this candidate, and are not silently omitted from a request.
The API takes one explicitly finite radiation point rather than accepting a broader
request and dropping unsupported work.


## Opt-in frozen reference profile and terminal operator probe

`anyhydro.panel_accuracy_reference.FrozenUntabulatedProfile.load(directory)`
loads a portable three-file declared-evidence bundle. It does not import a BEM
backend. Format `anyhydro-panel-accuracy-reference-v1` requires `manifest.json`,
`input.npz`, and `reference.npz`, bounded regular files and non-pickle numeric
arrays. Integrity hashes do not establish independent truth or producer
attestation. Full producing hull/lid snapshots, body order, reference centers,
fluid/frequency, exact k/kh, actual lambda/amplitude2x4 coefficients, pinned
Capytaine3.0.0 native library and three backend source identities must match.
Arrays are copied into immutable bytes; changes to bundle files after loading
cannot alter the running profile. An incompatible native build requires its own
reference evidence, rather than accepting an arbitrary platform substitution.

The manifest stores `format`, `files` (SHA256 of both NPZ files), `case_binding`,
`backend`, `k_hex`, `kh_hex`, `orders`, `steps`, `provenance`, and
`coefficient_identity`. `case_binding` contains inventory, body_ids,
rotation_centers, omega, water_depth, rho, gravity and snapshot_arrays as saved
by the candidate. `backend` contains capytaine, native_library_sha256, and
source_sha256 for green_functions/delhommeau.py, tools/prony_decomposition.py,
and meshes/geometry.py. Input arrays contain all eight candidate model arrays
plus coefficients and receivers. Reference arrays contain complex128
potential_low/high (43,) and gradient_low/high (43,3) at orders64/128.
Convergence is checked numerically at fixed1e-13+1e-11*max(abs(low),abs(high)),
never a manifest pass flag or adjustable tolerance. Differences are empirical,
not certified error bounds. The receiver grid is the original off-panel center
followed by plus/minus x/y/z at each fixed step1e-3,5e-4,2.5e-4,1e-4,5e-5,
1e-5,5e-6. Original steps are not discarded in favor of larger passing steps.

Pass the profile explicitly as `accuracy_profile=profile` to
`run_panel_integration_candidate`. This prospectively compares the native real
gradient at all43 receivers against the resolved reference, using unchanged
1e-8+1e-6*max(abs(native_center_real_gradient)). Native self-finite-difference
errors at every step are preserved separately, including their failures.
Potential/reference differences are retained without claiming a certified bound.
The default two-call native self-FD guard is unchanged. The opt-in evaluator uses
public inherited Green evaluation with zero-size tables, actual frozen Prony
coefficients and strict method/kh admission; no new fitting, RNG edits, fallback
or backend patching. Force integration and all full-point scientific checks
remain unchanged. Local guard success does not confer W5 or continuum acceptance.

For feasibility only, pass `operator_probe_receivers=N` with the profile.
N must be an integer1..16 (bool rejected); it is exactly N native receiver-point
rows, the first N flattened panel/quadrature nodes. The probe evaluates the full
source mesh once, saves indices/coordinates/raw potential and gradients, and
immediately returns `PanelOperatorProbe`. It allocates no full operator system,
solves no radiation equations and writes no radiation result. Its distinct
terminal state is `completed_probe_unqualified`. Record preparation, micro and
probe process durations separately; sample timing does not establish full-point
affordability. An execution registration can impose a narrower limit than this
API. Current development registration permits at most four probe rows/two native
calls/zero systems; it does not authorize any full radiation point.

`cancel_requested` is a cooperative callable checked before acquisition, after
the micro, after a probe and between full batches/before solve/completion.
Cancellation raises `PanelCandidateCancelled`, saves a `cancelled` state when
an artifact directory exists, and preserves acquired arrays. External process
supervision remains responsible for interrupting a native call or timeout.
This experimental route does not register a completed application analysis.

The reference potential FD checks at ORIGINAL1e-5/5e-6 are mandatory
against native center gradient under that same unchanged scale/limit. Both
candidate and accuracy-evaluator module hashes enter the opt-in case identity.


## Separate finer-tabulated candidate

`FrozenTabulatedProfile.load(reference_bundle)` explicitly constructs a different
profile identity, `anyhydro-panel-accuracy-finer-tabulation-v1`, from the same
three-file independent-reference bundle. Calling the untabulated loader remains
unchanged. Only these two named profile types are admitted; arbitrary subclasses
are refused. The fixed candidate uses1352 radial and744 vertical coordinates,
scaled_nemoh3/rmax100/zmin-251/1001 integration points/newer/low_freq/float64/
python/cacheNone through the public constructor and inherited evaluate. Doubling
counts does not assert nested grids or an improvement. No extra fitting or
reference calculation occurs; frozen coefficients and unchanged original-step
guard still govern. Guard failure stops before the probe without a refinement.

The planned profile hashes every constructor setting and both implementation
modules. Before construction, status records its attempt and intended settings.
Success preserves actual grids/table arrays and SHA256 metadata, freezes their
buffers, and binds those identities into a FINAL producing case hash. The
pre-construction identity remains recorded as planned_case_hash; it is not a
completed case. Input/reference arrays and both planned/final identities remain
evidence. Hash/immutability checks occur before and after acquisition; mutation
fails with saved raw outputs. Construction, freezing and serialization timings
are recorded separately and all remain included in process-wall accounting.
Timeout before arrays exist retains the attempted construction/status. External
process supervision owns hard cancellation; cooperative cancellation after
table preservation keeps table evidence and prevents a micro/probe launch.

Source-derived inventories record actual table-bound predicates and zero-radius
gradient limits for four finite-depth wave-image families, at actual source
quadrature coordinates. They disclose outside-table asymptotic use and explicitly
do not claim an instrumented native trace. Rankine/Prony contributions are not
table lookups. Table resolution alone says nothing about those approximations.
Every acquired micro/probe retains raw arrays; a probe is still an unqualified
terminal sample with no radiation interface/system solve. This route changes no
default, force, energy, residual, PSD, source-admission or scientific gate.

The prospective development registration permits one positive-size construction,
one43receiver micro and, only on guard success, oneâ‰¤4receiver point-row/full864
source probe. It authorizes no full point, sweep, oracle or automatic refinement.
One measured candidate outcome determines a new separately bounded next action.

Native backend identity is checked before construction. The acquisition callback
saves raw numeric table arrays and attempted settings immediately after the
constructor returns, before post-construction settings/shape/finiteness/range
checks; failed validation preserves those raw arrays. Validated immutable copies
are separately preserved and checked before/after native evaluation.

The 2026-10-10 installed candidate passed its unchanged independent-reference
guard and four-row terminal probe (one table/two Green calls/zero systems).
Construction took110.62115019999328s and the probe0.042752300010761246s;
the full113.85892939999758s process includes preparation and evidence handling.
Native self-FD remains failed; full-operator affordability and all scientific gates
remain open. See docs/reviews/w5-tabulated-profile-20261010.json for the recorded
input/evidence subset and ANYmarine TASK for Mistral disposition and protected
history. This packet does not attest independent oracle production.


## Saved-tabulation reuse consumer and headless saved-evidence diagnostics

`anyhydro.saved_tabulation_profile.FrozenSavedTabulatedProfile.load(reference_bundle,
saved_directory)` is a distinct consumer identity,
`anyhydro-panel-accuracy-saved-tabulation-v1`, never selected implicitly. It
validates the producer's frozen-tabulation archive, tabulation receipt, producing
case and accuracy profile against the independent reference bundle before any
backend construction: bounded regular files, hash-before-parse, producer
descriptor/backend identities, receipt/archive/table cross-identities,
producing-case hash recomputation and reference binding. Producer module
identities are historical provenance; consumer module identities are separate and
current. Reuse goes through the public Delhommeau constructor with an exclusively
owned per-job cache directory materialized in the exact public cache schema; loaded
tables are validated, frozen immutable and hash-checked before and after every
native evaluation. Cache-load observation is expected-source-path bookkeeping
observed with the mock backend only; no instrumented native trace and no native
zero-construction claim is made. Missing/corrupt or inconsistent producer inputs refuse before cache construction.
Post-load mutation, constructor failure or cancellation can preserve an already
prepared cache and acquired evidence; they never become completed radiation results. The old `FrozenTabulatedProfile` producer route is unchanged.

`anyhydro.saved_panel_diagnostics.run_saved_panel_diagnostics(inputs, output,
identities)` validates the pinned saved evidence byte-for-byte against
input-identities.json (including the pinned raw native micro packet
accuracy-native-micro.npz), cross-checks declared producing-case snapshots
against the actual input arrays before the producing-case self-hash, confirms
the coincident horizontal receiver pairs, compares the finite-depth imaginary
pole against the saved native imaginary S at the exact saved receivers and
source quadrature, and records signed finite-difference defect identities with
potential/reference/native gradient distinctions. Complex-valued statistics
use magnitude-based metrics only: RMS=sqrt(mean(abs(values)**2)), L2=norm and
max_abs=max(abs), with signed extrema reported separately for the real and
imaginary components; real arrays retain the existing signed_min/signed_max.
No complex-to-real cast, lexicographic complex extremum or new threshold occurs.
It makes zero native calls,
constructs no backend, fits nothing, introduces no acceptance threshold and
performs no retroactive imaginary acceptance.

Measured 2026-10-10 (development evidence, mock-verified; native not executed):
focused contract checks 73 passed / 0 failed through the w5-unblock harness with
numpy ComplexWarning promoted to error (attempts focused-panel-tests-b through
-f: 17, 10, 2, 0, 0 failures; JUnit checks-source.xml). The headless
diagnostics on the pinned inputs completed all thirteen stages in 0.29s
process wall with zero native calls and zero ComplexWarnings under
warning-as-error: 4 receivers over
864 source panels, all 4 receivers with two coincident nodes (8 coincident panel
entries); imaginary-pole max relative difference on significant entries
4.80e-05, with known coincident row sign reversals themselves diagnostic; signed
FD defects recorded per original step (native-minus-reference plateau about
1.381e-07) without causal attribution, because the earlier untabulated self-FD
comparison also failed. Inputs remained byte-identical to
input-identities.json afterwards (15 pinned files, zero mismatches). Reports:
.development-evidence/w5-unblock-20261010/offline-runs/saved-diagnostics-a/report.json
(pre-fix, warnings retained) and
.development-evidence/w5-unblock-20261010/offline-runs/saved-diagnostics-b/report.json
(post-fix, warning-as-error, complex statistics contract), each with raw arrays
imaginary-pole-comparison.npz and fd-signed-defects.npz.
Development fixtures confer no scientific qualification; jump, sign and
full-point scientific gates remain open.

Parent installed integration confirms73 focused checks,36 source/wheel/installed
module identities and actual original-table loader admission without importing
Capytaine. The installed saved-diagnostic service reproduces preserved reportb
physical metrics exactly with actual effective thread pools all1. It makes zero
native calls; no actual cached constructor/Green/system integration was performed.
Final evidence and exact accounting are under ANYmarine TASK and
C:/ANYworkspaceWork/anymarine-evidence/w5-mistral-unblock-20261010.
