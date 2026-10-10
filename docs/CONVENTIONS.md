# ANYhydro canonical conventions and invariants

Revision: W1 slice, 2026-10-08. These conventions are enforced by the schema
validator and tests. Backend outputs must be normalized to them before entering
a hydrodynamic dataset.

## 1. Coordinates and units

- SI units throughout. Right-handed global system, +Z upward.
- Mean free surface at z = 0. Water depth is a positive scalar; seabed
  elevation is negative in global coordinates.
- Angles are radians internally.
- Units recorded in dataset attributes: length `m`, force `N`, moment `N.m`.

## 2. Rigid DOF ordering

Per body, the canonical DOF order is:

```text
surge, sway, heave, roll, pitch, yaw
```

DOF identity is body-qualified and stable: `body_id::surge`, ...,
`body_id::yaw`. Unqualified names such as `"Surge"` are never used as
persistent identity. The separator `::` is reserved; body identifiers must not
contain it.

## 3. Complex convention

Canonical phasor convention:

```text
x(t) = Re{ X * exp(-i * omega * t) }
```

All complex data (excitation and optional force splits) follow this convention.
`excitation_force` is complex; radiation matrices are real.

## 4. Wave heading convention

Wave headings are stored in one canonical propagation convention: the
direction of propagation measured from +X toward +Y, in radians. Conversion
happens only at backend boundaries.

## 5. Panel normal convention

Panel normals point outward (out of the body, into the fluid). Mesh records
carry `outward_normals_verified`: a retained flag that is true only when the
normals were actually validated. No code path claims normal verification
without performing it.

## 6. Frequency coordinates

`omega` coordinates are finite, positive and strictly increasing. The zero
and infinite frequency limits are separate concerns: they are requested via
`HydroRequest.include_zero_frequency` / `include_infinite_frequency` and, for
the infinite limit, admitted through the optional `added_mass_infinite`
variable. A finite-frequency `omega` axis never contains 0 or `inf`.

## 7. Full matrix preservation

For N bodies, hydrodynamic data live in the complete 6N-DOF space. Off-diagonal
terms A_ij, B_ij (i != j), including cross-body blocks, are part of the
production model and must survive validation, transformation, serialization
and reordering. Validation rejects incomplete DOF axes.

## 8. Mechanical authority

ANYsolver owns dry mass, center of gravity and inertia. ANYhydro stores
`center_of_mass` only as a geometric reference record; no mass or inertia
variable exists in the schema, and unknown variables are rejected. Capytaine's
automatically estimated rigid-body inertia is never production dry mass.

## 9. Reference-point translation map (adopted contract)

For a shift `r = new_origin - old_origin` with axes unchanged, per body:

```text
T = [[I, skew(r)],   q_old = T q_new
     [0,     I  ]]

F_new = T^T F_old
A_new = T^T A_old T      (likewise B, C, A_inf)
```

where `skew(r) * theta = r x theta`. The global map is block diagonal over
bodies, so cross-body blocks transform as `T_i^T A_ij T_j` and are preserved.
Work is invariant: `F_new . q_new = F_old . q_old`.

After mapping, the stored `rotation_center` is updated to the new origin.
`center_of_mass` and `center_of_buoyancy` are global positions and do not move.

Cautions:

- Translational and rotational entries carry different units; never compare or
  normalize mixed norms without explicit scales.
- Reference-point translations are supported with axes unchanged. Rotation of
  axes is supported only with explicit proper orthonormal bases via
  `rotate_axes(ds, new_axes)` / `reframe_dataset`; the argument-free legacy
  form remains a typed refusal. Mirrored frames are always refused.
- Mapping appends a provenance record to the dataset attribute
  `reference_transformations`.

## 10. Identity and provenance

Every dataset records `schema_version`, `backend`, `backend_version`,
`backend_commit` (unknown is recorded honestly as `"unknown"`), convention
strings, `mesh_hashes`, `request_hash`, `created_utc` and
`qualification_status`. Development fixtures are marked
`development_fixture` and confer no scientific qualification.

## 11. Optional dependencies

Capytaine and Qt are optional and never imported at base import time. No
backend object or pickle payload is ever stored in persisted files.

## 12. Coefficient frames (optional 1.1.0 extension)

Each body's six DOFs keep their stable body-qualified identity; their
coordinate values are components in a recorded per-body proper orthonormal
coefficient basis (columns in WORLD axes) about a recorded world reference
center, stored in the optional `coefficient_frames` attribute. Datasets
without the attribute follow the published GLOBAL convention and are
unchanged. The fixed EQUILIBRIUM-LINEAR frame map
(`Q = E_old.T E_new`, `d_old = E_old.T (O_new - O_old)`,
`T = [[Q, skew(d_old) Q], [0, Q]]`, `q_old = T q_new`) is a constant
equilibrium-linear coefficient coordinate change - NOT an exact finite-pose
reference/chart shift (ADR-004). Geometry, wave headings and the phasor
convention stay in WORLD axes. See
`docs/COEFFICIENT_FRAMES_AND_HYDROSTATICS.md`.

## 13. Hydrostatic contract (optional 1.1.0 extension)

The optional `hydrostatic_contract` attribute transports the source's
declared role (UNKNOWN, BUOYANCY_ONLY or FULL), force basis, stiffness
derivative semantics, body binding, optional actual baseline wrench,
provenance, the FULL source declaration (legacy single-value or per-body
shape) and the versioned source binding. A missing contract is UNKNOWN -
never zero stiffness and never a gravity-free assumption. A missing baseline
wrench is UNMEASURED, not zero. No baseline, displaced volume, CG/CB, gravity
or inertia is fabricated. Unknown semantics, a missing baseline, a
PHYSICAL_WRENCH basis without an admitted source rate/coordinate convention
map, role FULL (decomposition policy not implemented in this slice) and an
unbound or stale/foreign source binding each leave the affected future exact
six-DOF composition unadmitted (typed refusal); declared metadata is not
scientific qualification and not integrator admission. `attach` binds the
caller's declarations to the exact dataset content (noncircular identity
excluding only the contract payload); rebinding a foreign/stale bound record
requires an explicit `rebind=True`. ADR-004 is governing.

