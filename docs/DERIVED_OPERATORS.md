# Derived hydrodynamic operators (development slice)

`anyhydro.operators` builds an explicit, unqualified *derived* hydrodynamic
operator artifact: a caller-chosen retained subset of DOFs, projected from a
validated neutral full-6N dataset, with an explicit correction policy,
declared normalization, caller-declared admission criteria and a complete
recomputable report. `anyhydro.radiation` then consumes an *admitted* derived
artifact through the same numerical kernel implementation as the dataset
path. This slice is development-only: nothing here establishes mesh,
impedance, response, source or convergence qualification.

The raw full-6N source dataset is immutable and never forged: a retained
artifact is its own artifact type (`anyhydro-derived-operators-v2`), never a
relabeled raw dataset, and `schema.validate_hydro_dataset` still requires all
6 DOFs per body for raw data.

## Projection and normalization conventions

- Retained DOFs are ordered body-qualified labels (`body_id::dof`), exact and
  unique, and must be a subset of the source DOF axis. No numeric positions
  are hardcoded anywhere.
- `build_projection` builds the exact 0/1 selector `P` (n_full x n_ret) by
  label match. Conventions: `q_full = P q_ret` (generalized forces/flows
  embed), `Q_ret = P^T Q_full` (work-conjugate retrieval). Full coupled
  matrices are preserved: the raw retained blocks are stored unmodified
  alongside the derived ones.
- `NormalizationSpec` declares a fixed positive diagonal scale `S` per
  retained DOF plus mandatory `name` and `unit_convention` prose. Congruence
  is strictly `X_hat = S^T X S` (and back `X = S^{-T} X_hat S^{-1}`); there
  is no other normalization path. Scales must be finite, strictly positive
  floats; bools, non-finite and empty inputs are refused.

## Correction policy (explicit, no defaults)

`DerivedOperatorSettings` requires an explicit `correction_policy`:

- `CORRECTION_NONE` (`"none"`): the derived A/B/Ainf are the raw retained
  blocks, unmodified. Asymmetric inadmissible operators are still refused
  for radiation consumption (admission fails), never silently repaired.
- `CORRECTION_RECIPROCAL_AVERAGE` (`"reciprocal_average"`): strictly
  `(X + X^T)/2` applied to A, B and Ainf only. Hydrostatics, excitation,
  wave and frequency coordinates are never modified. There is no PSD
  clamping, no tuning, and no default policy: omitting the policy is a
  settings error.
- `CORRECTION_FAR_FIELD_WEAK_MODES`
  (`"reciprocal_average_far_field_weak_modes"`): the reciprocal average,
  then, per frequency in the declared normalized coordinates, the symmetric
  damping's block on its weak subspace (eigenvalues below
  `WEAK_MODE_FRACTION = 1e-3` times the largest, the adopted W5 energetic
  threshold; orthonormal basis `V_w`) is replaced by the far-field damping
  compressed onto it: `B + V_w (V_w^T F V_w - Lambda_w) V_w^T`
  (basis-independent for repeated weak eigenvalues). Requires the dataset
  variable `radiation_damping_far_field` (Capytaine adapter >= 0.5.0,
  indirect method); the retained far-field matrix is stored in the artifact
  (`raw_damping_far_field`) and its source description in the provenance,
  so the invariant and the raw-source proof are recomputed exactly.
  Far-field (Kochin energy-flux) damping is positive semidefinite by
  construction and is the accurate estimator for nearly non-radiating
  modes, which pressure integration cannot resolve at practical meshes (W5
  evidence `w5-physical-20261012`). Energetic modes, A and A_inf are
  untouched. Artifacts of the other policies are unchanged (no new array).

## Criteria and the report

`DerivedOperatorCriteria` has no scientific defaults: every threshold is
declared, finite and nonnegative by the caller.

**Numeric convention (fixed).** Every admission criterion operates in the
declared normalized coordinates `X_hat = S^T X S`. Each absolute threshold
is therefore interpreted in that operator's own normalized unit — the
`A_hat` and `B_hat` units differ, and there is no cross-operator mixing.
The relative term uses a per-matrix global scale (the largest normalized
magnitude in the matrix concerned) with an explicit positive near-zero
floor: `threshold = abs_tol + rel_tol * max(scale, floor)`. The stored
mechanical arrays (raw and derived A/B/Ainf, hydrostatics, excitation)
stay in SI retained coordinates; only the admission arithmetic moves to
the normalized coordinates, because a finite numeric threshold is not
invariant under congruence even though the eigenvalue sign is.

- Raw normalized reciprocity defects: `|X_hat - X_hat^T|` of the raw
  retained A, B and A_inf, complete and frequency-indexed.
- Normalized correction magnitudes: how far the policy moved the
  operators, same abs + rel threshold shape, complete and
  frequency-indexed.
- Derived symmetry tolerance: exact symmetry of the derived A/B/Ainf in
  normalized coordinates.
- Retained B PSD: eigenvalue tolerance of `B_hat = S^T B_derived S` in
  normalized coordinates. The eigensolver always receives the explicit
  symmetric part `0.5*(B_hat + B_hat^T)` — never silently one triangle
  of a potentially nonsymmetric input. Negative modes are never clamped.

`DerivedOperatorReport` keeps the complete frequency-indexed
residual/threshold/pass arrays for reciprocity and correction (shape
`(n_w, n, n)` for the frequency-dependent A and B, `(n, n)` for A_inf),
the symmetry defects, the complete normalized B eigenbasis
(`damping_eigenvalues` and `damping_eigenvectors` per frequency) and
**every** negative retained mode (all eigenvalues below
`-damping_psd_tol`, with its frequency index, basis index, eigenvalue and
normalized eigenvector — not only the worst mode per frequency).
Eigenvectors follow the `numpy.linalg.eigh` column convention:
`damping_eigenvectors[o][:, b]` is the eigenvector for basis index `b`,
and `damping_negative_mode_vectors` has shape `(n_modes, n)` where row
`k` is exactly the column
`damping_eigenvectors[freq_idx[k], :, basis_idx[k]]` at unit L2 norm
(each mode satisfies `sym(B_hat) v = lambda v`).
Maxima
appear in `summary()` only, never instead of the full arrays. A normalized
eigenvector `v_hat` maps to the physical retained direction `q = S v_hat`.

The raw-full diagnostic section (`raw_full_unscaled_*`, labelled
`unscaled_mixed_si_informational`) carries full-source minimum damping
eigenvalues, asymmetries, excluded-DOF PSD defects with negative-mode
vectors and retained alignment. Those arrays are computed in unscaled
mixed-SI coordinates (they lack the retained normalization `S`): they are
informational only, never a normalized admission or severity measure, and
can never be mistaken for a retained failure or pass.

`admitted` is always recomputed from the report; there is no stored pass
bit. Failed criteria still produce a retained, inspectable artifact and
report, but radiation consumption raises a typed
`DerivedOperatorAdmissionError`.

## Raw-source proof (rederivation, never the hash alone)

`verify_raw_source(dataset)` is the public source proof: it revalidates
the dataset, checks the recorded source hash, then **recomputes the
complete public derivation** — the label-built selector `P`, every raw
projected A/B/Ainf/H/F, the frequency and wave grids, the source DOF and
body order, the derived arrays, the report and the producing provenance —
under the immutable stored settings, and compares every stored value
exactly. `matches_raw_source(dataset)` is its boolean form. A matching
source hash alone proves nothing: an artifact whose data came from a
different dataset is refused even if its recorded hash was forged.
Non-admitted artifacts verify exactly like admitted ones; source identity
and failure evidence are preserved.

Consumers holding the raw source MUST invoke this proof:
`load_derived_operators(path, source_dataset=...)` runs it at load time
(the keyword is optional; the proof runs exactly when the raw source is
supplied, and a failed proof is a typed
`DerivedOperatorSerializationError`),
and both radiation consumers accept the same keyword
(`radiation_kernel_from_derived_operators(..., source_dataset=...)`,
`compare_derived_reconstruction(..., source_dataset=...)`); a failed
proof is a typed refusal in each case.

## Artifact and persistence

`derived_operators_from_dataset(dataset, settings)` records: the source
dataset content hash, raw source DOF/body order, `P`, raw retained
A/B/Ainf, derived A/B/Ainf, projected unmodified hydrostatics, excitation,
wave and frequency coordinates, and complete provenance. The provenance
preserves the actual complete source attributes verbatim
(`source_attrs`: backend settings and method, fluid rho/g/depth, unit
controls — as present, nothing invented) plus the relevant variable
attributes (`source_variable_attrs`, e.g. the Ainf source evidence),
alongside the explicit identity keys (request, mesh, lid, equilibrium,
reference points, phasor and transform metadata).

Hard limits and consistency are enforced by the builder **and by the
artifact constructor/reader** before any effective array freezes:
<= 12 source DOFs, <= 513 source frequencies, and full-6N body/DOF-order
consistency (every source body carries exactly its six DOFs). Loader
failures are typed `DerivedOperatorSerializationError`.

`save_derived_operators` / `load_derived_operators` are atomic and
pickle-free (`allow_pickle=False`). The loader refuses unknown or missing
fields, wrong format versions, non-npz content, and any content-hash
mismatch; it then recomputes `P`, the policy application and the retained
report from the stored raw arrays rather than trusting stored results.
`content_hash()` covers all data, settings, criteria and results,
including the complete report arrays and the raw-full coordinate label.
The raw source dataset remains separate and unchanged.

## Retained-coordinate radiation

- `radiation_kernel_from_derived_operators(derived, settings)`: typed
  non-success (`DerivedOperatorAdmissionError`) unless the artifact is
  admitted. The caller supplies `zero_frequency_damping` (B(0)) in retained
  coordinates; the Ainf + physical dry mass gate remains consumer-owned and
  no mass is invented here. The kernel uses the same private numerical core
  (`_kernel_from_arrays`) as the dataset path — no copied hydrodynamics, no
  monkeypatching, no schema relaxation. Provenance is distinct:
  `source_kind="derived_operators"`, derived content hash, raw source
  dataset hash, and policy/normalization/criteria identity. Derived kernels
  persist and reload through the existing public immutable
  `RadiationKernel` API. Both this function and the comparator accept the
  mandatory-in-intent `source_dataset` keyword (see "Raw-source proof").
- `compare_derived_reconstruction(kernel, derived, criteria)`: public A+B
  reconstruction comparator bound to the derived content hash AND the raw
  source dataset hash plus retained DOF order, policy, S and criteria
  identity. It compares against the derived reference — never the raw
  source relabelled — and refuses kernels or artifacts from any other
  source.

## Refusal paths (typed, tested)

Empty, duplicate, malformed or unknown retained DOF labels; non-finite,
non-positive, bool or empty scales; unknown settings keys; missing or
invalid criteria; non-finite criteria; missing Ainf evidence; hard-limit
violations (12 DOFs / 513 frequencies, enforced by builder, constructor
and reader); inconsistent body/DOF order; non-admitted artifacts at
radiation consumption; asymmetric operators under the `none` policy;
negative retained B in normalized coordinates (reported with the complete
eigenbasis and every negative mode, never clamped, refused); tampered,
truncated, unknown-field, wrong-version or foreign-format files; failed
raw-source proof when source_dataset is supplied at the loader and both radiation consumers (forged
source hashes included); wrong-source kernels at the comparator.

## Status

Development fixture evidence only. Author verification is not
qualification or final acceptance; response/impedance correction and mesh
acceptance requirements remain open for later registered qualification.

Applications holding the raw source must supply source_dataset for verification;
the optional parameter does not authenticate origin when omitted. Current source
and installed verification is development evidence in the pinned Windows runtime;
Linux/eigenbasis roundoff portability and scientific qualification remain open.

