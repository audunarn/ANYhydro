# Radiation preprocessing — development contract

Owner: `src/anyhydro/radiation.py`. Maturity: **development only**.
Governing contract:
`C:/ANYworkspaceWork/anymarine-evidence/radiation-contract.md`.
Nothing in this slice confers scientific qualification; existing W5
reconstruction/RAO/free-decay gates and their original thresholds remain
pending with the parent.

## Conventions and equations

ANYhydro uses `Re{X exp(-i*omega*t)}` and influence-row/radiating-column
matrices. The positive left-hand radiation term is `A_inf*a + K*v` with the
causal kernel

```
K(t)   = (2/pi) * integral_0^infinity B(omega) cos(omega*t) d omega
B_rec(w) = integral_0^infinity K(t) cos(w*t) dt
A_rec(w) = A_inf - integral_0^infinity K(t) sin(w*t) dt / w     (w > 0)
A_rec(0) = A_inf - integral_0^infinity t*K(t) dt                (explicit limit)
```

The sine sign follows from substituting the phasor into the causal
convolution; it is never flipped to match a positive-exponent convention.
Independent oracle (used by tests and `examples/radiation_check.py`):

```
B(w) = C*lambda/(lambda^2 + w^2)
A(w) = A_inf - C/(lambda^2 + w^2)
K(t) = C*exp(-lambda*t)          (closed form)
B(0) = C/lambda,  A(0) = A_inf - C/lambda^2
```

Primary reference for the Cummins/cosine transform:
https://wec-sim.github.io/WEC-Sim/main/theory/theory.html

## Route

```
validated neutral dataset (validate_for_time_domain: explicit A_inf + source)
  -> radiation_kernel_from_dataset(ds, settings)   full-coupled causal kernel
  -> kernel.reconstruct(omegas)                     BOTH A and B
  -> compare_reconstruction(kernel, ds, criteria)  criteria-based diagnostics
  -> save_radiation_kernel / load_radiation_kernel  immutable artifact + reproduction
```

## Settings (`RadiationSettings`, all explicit, immutable)

- `dt`, `n_time`: uniform time grid `t_k = k*dt` starting at zero, `dt > 0`,
  `2 <= n_time <= 1001` by default, or up to the explicitly declared
  `execution_limits.max_time_samples` (hard ceiling 20001).
- `time_window`: optional explicit kernel time window (kernel zero beyond
  it), recorded as an assumption. Default `None`.
- `low_tail_policy`:
  - `"explicit_zero_sample"` (default): requires `zero_frequency_damping`,
    a full coupled `B(0)` matrix supplied by the caller. The final
    finite-frequency sample is never substituted.
  - `"linear_to_zero_extension"`: requires a separately named
    `extension_name`; extends B linearly from the first grid frequency to
    `B(0) = 0`, recorded as an assumption.
- `high_tail_policy`:
  - `"truncate"` (default): B is zero above the last grid frequency
    (exact old serialized settings and hashes).
  - `"power_law_extension"`: requires `quadrature_method=
    "piecewise_linear_exact"`, a finite `high_tail_exponent > 1`, an
    `high_tail_upper` above the grid and a separately named
    `high_tail_name`. B continues as `B(omega_max) (omega_max/nu)^n` on
    geometric samples (ratio 1.01) up to `high_tail_upper`, zero beyond,
    recorded as an assumption. It is a positive multiple of B(omega_max),
    so symmetry and PSD are inherited. Truncation at a frequency where B is
    still large leaves an added-mass offset of about
    `(2/pi) int_{omega_max}^inf B(nu)/nu^2 dnu` (W5 evidence
    `w5-physical-20261012b`); the declared tail removes it for the
    coefficients that follow a power law. Oscillating inter-body coupling
    terms are not power laws; tail adequacy stays a separately assessed gate.
  - `high_tail_scope` (power-law tail only): `"full"` (default) or
    `"body_diagonal"`, which continues each body's own block and truncates
    inter-body coupling (oscillating in omega, not a power law). A principal
    block of a PSD matrix is PSD, so passivity is inherited.
- `a_inf_policy`: `"computed"` (default; the source A_inf) or
  `"band_consistent"` with `a_inf_calibration_min/max`: the time-domain
  A_inf is the computed A_inf plus the mean over the calibration-band source
  frequencies of `A - A_rec` (Ogilvie-consistent with this band-limited
  kernel). The computed A_inf, the difference and the calibration
  frequencies are stored in the provenance;
  `verify_band_consistent_added_mass_infinite` proves them (computed value
  equals the source exactly, band mean of `A - A_rec` vanishes) and consumers
  never trust the stored value without it. The W5 workflow requires the
  calibration band below the reconstruction window.
- `symmetry_tol`, `passivity_tol`: explicit validation tolerances. There is
  no blanket scientific default.
- `tolerance_basis`: `"absolute"` (default; omitted from serialized settings,
  so old hashes are unchanged) takes the two tolerances as SI values.
  `"relative_to_damping_scale"` takes them as fractions of the max spectral
  norm of the checked damping samples (passivity, damping symmetry) or of the
  source-grid added-mass samples (added-mass symmetry). The scales and the
  effective tolerances are recorded in the diagnostics (`damping_scale`,
  `added_mass_scale`, `effective_*_tol`; present only for the relative basis).
  Violations are still counted and reported, never clamped or repaired.
- `execution_limits`: optional frozen public `RadiationExecutionLimits`
  (default `max_time_samples=1001`, minimum 2, hard ceiling 20001).
  Default `None` keeps the development cap and the exact old serialized
  settings and hashes. Explicit limits admit a longer kernel for that one
  construction, are validated before any allocation (bool, fractional,
  non-finite, unknown and out-of-range controls are typed refusals), and
  participate in the settings dict, the artifact content hash, provenance,
  assumptions and persistence. `RadiationSettings.from_dict` /
  `load_radiation_kernel` faithfully reconstruct and enforce them: a
  longer-than-1001 artifact whose declared controls were deleted refuses.
  Limits are execution bounds only: no scientific qualification, no
  loosened tolerances, no new tail/physics policy, no fake padding or
  cropped history.

## Quadrature and grids

Deterministic trapezoidal quadrature throughout. The frequency grid is the
dataset's finite strictly increasing grid (>= 3 samples) extended by the
explicit zero-frequency sample. By default the high tail is **explicitly
truncated at `omega_max`**; with `high_tail_policy="power_law_extension"` the
declared tail is appended up to `high_tail_upper` (which must satisfy
`high_tail_upper*dt <= pi`) and truncated there. There is no fitting,
symmetrization or clamping.
NaN/Inf anywhere and a missing `A_inf` are typed refusals.

## Diagnostics (`RadiationDiagnostics`, errors preserved, never clamped)

- Tail: `damping_at_omega_max`, `max_abs_damping_at_omega_max`,
  `omega_max`, `max_frequency_spacing`, `omega_max_times_dt` (compare with
  pi for time-grid resolution of the highest frequency),
  `kernel_duration`, `duration_over_spacing_period`
  (`kernel_duration * max_spacing / (2*pi)`, the time-aliasing scale of the
  frequency grid).
- Symmetry: `max_damping_asymmetry`, `max_added_mass_asymmetry`,
  `damping_symmetry_violations`, `added_mass_symmetry_violations` counted
  against `symmetry_tol`.
- Passivity: `min_damping_eigenvalue` (min over frequencies of the min
  eigenvalue of `(B+B^T)/2`), `passivity_violations` counted against
  `passivity_tol`.

## Reconstruction criteria (`ReconstructionCriteria`, caller-declared)

Comparison happens on the reference grid frequencies inside the finite
window `[omega_min, omega_max]` against the kernel's own source dataset
(content-hash bound). Per matrix coefficient:

```
max_error_ij <= abs_tol + rel_tol * max_over_window(|reference_ij|)
```

Added mass and damping have separate absolute tolerances (different units);
near-zero references are covered by the absolute allowance. Both matrices
must pass (`overall_pass`); there is no admission on damping alone and no
default criteria object. `ReconstructionReport` records observed max
errors, per-coefficient thresholds, pass flags, failing coefficient lists
and worst error/threshold ratios (`summary()`).

## Artifact (`RadiationKernel`)

Frozen dataclass; all arrays are defensive copies backed by immutable
buffers (truly read-only: `setflags(write=True)` fails on the arrays and on
views of them) and the provenance mapping is deep-frozen. Provenance
preserves the source dataset
content hash, request hash, mesh hashes, body identity, reference
transformations, `A_inf` source, phasor convention, source qualification
status, schema version and backend. (The source dataset's `created_utc` is
deliberately not copied: the content hash already binds the artifact to the
dataset identity, and excluding the timestamp keeps artifact reproduction
deterministic across processes.) Assumptions
record the low-tail policy, the high-tail truncation or declared extension, the time window and
the non-inheritance of qualification. `qualification_status` is always
`"unqualified"`: kernel artifacts cannot inherit qualified status from a
source dataset.

Serialization is deterministic NetCDF (scipy engine, `NETCDF3_64BIT`,
atomic replace, no timestamps): saving the same artifact twice produces
byte-identical files, and `load_radiation_kernel` verifies a stored
content hash (`RadiationSerializationError` on mismatch or tampering).

## Development limits (typed `RadiationLimitError`)

- <= 513 input frequencies, <= 1001 time samples, <= 12 DOFs.
- Explicit longer kernels: `RadiationExecutionLimits(max_time_samples=N)`
  with `2 <= N <= 20001` admits `n_time` up to `N` for that construction
  only; the global default cap stays 1001.
- One numerical process/thread; no BEM runs in this slice.

## Tests and example

- `tests/test_radiation.py`: oracle kernel match, both-matrix
  reconstruction, sign convention (not flipped), explicit zero-frequency
  limit, low-tail policies, refusals (missing `A_inf`, NaN/Inf, < 3
  frequencies), all three limits, immutability/defensive copies,
  provenance and non-inherited qualification, deterministic verified
  serialization and tamper rejection, source-bound comparison, criteria
  mechanics (threshold formula, absolute allowance, no
  damping-only admission), symmetry/passivity/tail diagnostics, time
  window.
- `tests/test_radiation_long_kernel.py`: explicit long-kernel controls —
  default-1001 refusal, explicit 1002/1003 admission on the analytic
  oracle, limits validation (bool/fractional/non-finite/unknown/min/
  ceiling), immutability, hash and provenance participation, NetCDF
  reload equality, deleted-controls refusal, and preservation of the
  pre-change default-settings content hash.
- `examples/radiation_check.py`: full 12x12 coupled two-body oracle through
  the complete route with declared criteria and artifact reproduction.

Development fixtures and these tests confer no scientific qualification.

## Full-block reconstruction criteria (W5 K1-K3)

`reconstruction_criteria_from_dict` accepts the legacy per-coefficient form
or `form="full_block"` (`FullBlockReconstructionCriteria`: window,
`block_rel_tol`, `modal_rel_tol`, `energetic_fraction`, `cluster_fraction`).
`compare_derived_reconstruction` then returns a
`FullBlockReconstructionReport`: per window frequency the Frobenius relative
errors of Z = B + i omega A, A and B (K1) and the largest energetic-mode
relative error of A and B (K2, clusters in spectral norm), plus reported-only
per-coefficient errors and the A_inf difference (K3). The raw-dataset route
keeps the per-coefficient form.
