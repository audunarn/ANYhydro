# Coefficient frames and the hydrostatic contract

Revision: hydro-frames slice, 2026-10-08. DEVELOPMENT documentation of the
neutral data model. Nothing here confers scientific qualification or
integrator admission.

## 1. Coefficient frames

Each body's six DOFs (`body_id::surge` ... `body_id::yaw`) keep their stable
body-qualified identity. Their coordinate values are components in a
recorded per-body coefficient basis about a recorded world reference center:

- dataset attribute `coefficient_frames` (JSON object, optional):
  `body_id -> {"axes": 3x3, "reference_center": [x, y, z]}`
- `axes` columns are the coefficient-basis directions expressed in WORLD
  axes; the matrix must be a proper rotation (orthogonality and
  determinant +1 validated at 1e-12; reflections, shears and non-finite
  input fail typed; there is no silent orthonormalization).
- `reference_center` must equal the stored `rotation_center` variable: it is
  world geometry and never moves under a frame change.

A dataset without the attribute follows the existing published GLOBAL
convention (identity axes). Its semantics, hashes and persisted files are
unchanged by this extension. Malformed metadata fails schema validation; it
is never ignored.

## 2. The fixed EQUILIBRIUM-LINEAR frame map

For a frame change `(E_old, O_old) -> (E_new, O_new)` per body:

```text
Q     = E_old.T @ E_new
d_old = E_old.T @ (O_new - O_old)
T     = [[Q, skew(d_old) @ Q],     q_old = T @ q_new
       [0,        Q           ]]

M_new = Ti.T @ M_old @ Tr     (A, B, C, A_inf; leading omega axis retained)
F_new = Ti.T @ F_old          (excitation and recognized force splits)
```

The global map is block diagonal over bodies, so cross-body blocks transform
as `T_i.T A_ij T_j` and are preserved. Arbitrary independent influenced and
radiating axis orderings are supported. Work is invariant:
`F_new . q_new = F_old . q_old`.

Public API (`anyhydro.frames`, re-exported at package level):

- `rotate_axes(ds, new_axes)` - change axes only; the argument-free legacy
  form remains a typed `UnsupportedOperationError` refusal.
- `reframe_dataset(ds, new_axes=None, new_reference_centers=None)` - the
  general map (axes and/or centers).
- `coefficient_frame_axes(ds)` - validated `body_id -> world-basis 3x3`
  mapping (identity for legacy GLOBAL datasets).
- `require_global_dof_axes(ds)` - typed `TransformError` gate for consumers
  that only admit GLOBAL DOF axes; succeeds for legacy datasets without
  recorded axes.
- `frame_change_matrix(old, new)` - the per-body 6x6 map.

`shift_dataset_reference_point` honors recorded bases: the shift acts in the
body's coefficient basis (`d_old = E_old.T (O_new - O_old)`), and the frame
metadata's reference center moves with the `rotation_center` variable.
Legacy datasets without metadata keep the exact world-shift semantics,
provenance records and hashes.

## 3. Critical semantics - what this map is NOT

These transformations are CONSTANT equilibrium-linear coefficient coordinate
changes. C congruence and `T.T` baseline transport do NOT by themselves
implement an exact finite-pose reference/chart shift: that requires baseline
and connection terms and a matching consumer pose map (ADR-004). Every frame
change appends a reproducible provenance record to `reference_transformations`
with the old/new axes and centers (the inverse map is restorable from the
record), the map identity, settings and the pre-transform source content
hash. The content hash after an inverse map differs from the original: the
transformation history is part of the dataset's identity and is honestly
recorded, never claimed away.

Geometry (mesh vertices, COM, CB, reference centers), wave headings and the
phasor convention stay in WORLD axes. A coefficient frame change alters DOF
coordinate bases only - never mesh geometry or wave phase.

## 4. Mapped and refused quantities

Mapped exactly like the excitation force when present:
`froude_krylov_force`, `diffraction_force`, `mean_drift_force`.

Refused BEFORE any mutation (typed `TransformError`, input never mutated)
because their exact mapping is not defined in this slice:
`panel_pressure`, `field_velocity`, `free_surface_elevation`,
`kochin_radiation`, `kochin_diffraction`. No silent dropping or relabeling.

## 5. The hydrostatic contract

ADR-004 is governing: mechanics owns dry gravity; the hydrodynamic source
owns buoyancy. The dataset attribute `hydrostatic_contract` (JSON object,
optional) transports the source's explicit declaration via the immutable,
validated `HydrostaticContract` record:

- `role`: `UNKNOWN`, `BUOYANCY_ONLY` or `FULL`. A missing contract is
  UNKNOWN - never zero stiffness, never a gravity-free assumption.
- `force_basis`: `GENERALIZED_DOF` (rows work-conjugate to the dataset DOF
  coordinates), `PHYSICAL_WRENCH` (source-specific physical rate map,
  ADR-004) or `UNKNOWN`.
- `stiffness_derivative`: `INCREMENTAL_ABOUT_EQUILIBRIUM`, `TOTAL` or
  `UNKNOWN`.
- `bodies`: body binding; must equal the dataset bodies when attached.
- `baseline_wrench`: optional ACTUAL measured baseline wrench per body.
  Absent means not measured, not zero. Never fabricated. A missing baseline
  leaves composition UNADMITTED; a source-declared zero wrench is valid
  metadata and is never inferred from absence.
- `provenance`: source identity (non-empty `source` string required).
- `full_source`: required exactly for `FULL`: `mass`, `center_of_gravity`
  and a `composition` declaration. Rejected for other roles. Two declared
  shapes are transported: the legacy single-value shape (readable as
  declared incomplete metadata on multi-body contracts) and the per-body
  shape (mappings covering exactly the contract bodies; mixed shapes are
  refused). Neither shape admits composition in this slice.
- `source_binding`: optional versioned immutable binding
  (`hydrostatic_source_binding_v1`) of the contract to the EXACT dataset
  content. The binding hash is noncircular: it excludes ONLY the contract/
  binding payload itself and covers all arrays, axes/reference centers,
  coefficient frames, conventions, mesh/request/body identity and the
  transformation history. `attach_hydrostatic_contract` binds; a contract
  already bound to different content is refused unless the caller passes
  `rebind=True` explicitly. The binding declares target identity, not
  qualification.

Public API (`anyhydro.hydrostatic_contract`, re-exported at package level):
`attach_hydrostatic_contract`, `read_hydrostatic_contract`,
`contract_source_binding`, `hydrostatic_contract_binding_matches`,
`rebind_hydrostatic_contract`, `hydrostatic_composition_eligibility`,
`require_hydrostatic_composition` (typed `HydrostaticContractError`
refusal).

Admission rule (`hydrostatic_composition_eligibility`, this slice): only
complete, bound `GENERALIZED_DOF` declarations are admissible at the
metadata level. Typed refusal reasons, never silent defaults:

- role, force basis or derivative semantics UNKNOWN;
- missing baseline wrench (UNMEASURED, not zero);
- `PHYSICAL_WRENCH` force basis: the source-specific rate/coordinate
  convention map and its implemented conversion are required first; no
  unverified prerequisite claim is recorded;
- role FULL: the declaration is transported as metadata only; the exact
  per-body decomposition/reference-only-gravity policy is not implemented
  or adopted in this slice;
- unbound contract, or a stale/foreign source binding (geometry, axes,
  coefficients or reference provenance changed after binding).

Honesty rules:

- No baseline wrench, displaced volume, CG/CB, source gravity or inertia is
  ever fabricated; only what the source declared is recorded.
- A declared-complete contract is transport metadata, NOT scientific
  qualification and NOT integrator admission.
- Unknown role/force-basis/derivative/frame semantics leave the affected
  future exact six-DOF composition UNADMITTED
  (`require_hydrostatic_composition` refuses); no silent FULL subtraction or
  policy selection exists.
- The current Capytaine adapter's mass=0 buoyancy-only behavior is
  unchanged and its narrative note is preserved; this slice claims no
  double-gravity defect and sources no new quantities.

## 6. Interaction between frames and the contract

A GENERALIZED_DOF baseline wrench attached to a dataset is actual measured
data in dataset DOF coordinates; a frame change transports it with the exact
`T.T` wrench map and records provenance. A contract with a baseline wrench
and a `PHYSICAL_WRENCH` or `UNKNOWN` force basis refuses the frame change
before mutation: the wrench's coordinate semantics are not exactly mapped by
the equilibrium-linear map. A contract whose source binding is stale/foreign
relative to the input dataset also refuses before mutation: a frame change
never silently rebinds a foreign or stale bound record. After all array,
axes/centers and history changes the attached contract is explicitly
rebound to the reframed dataset's exact content, preserving the original
source identity and provenance; the noncircular binding hash excludes the
contract payload itself, so bound reference settings stay unambiguous.

## 7. Schema, hashing, persistence

Both attributes are optional extensions of schema 1.1.0: absent on legacy
datasets (unchanged hashes and persisted files), present they are validated
(malformed metadata fails) and hashed as canonical JSON - tamper-evident,
invariant under body/DOF reordering, and stable across the NetCDF
round-trip. A schema version bump that would alter legacy hash, default or
cache binding is a parent decision, not adopted here.
