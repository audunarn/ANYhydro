# ANYhydro hydrodynamic data schema

Revision: W2 slice, 2026-10-08. Schema version: `1.1.0` (unreleased
foundation; compatibility changes under this unreleased version are
documented here).

## 1. Coordinates

| Coordinate          | Dtype     | Constraints |
|---------------------|-----------|-------------|
| `omega`             | float64   | 1-D, finite, > 0, strictly increasing |
| `wave_direction`    | float64   | 1-D, finite (radians, propagation from +X toward +Y) |
| `radiating_dof`     | str       | 1-D, body-qualified `body_id::dof`, complete 6 per body, no duplicates |
| `influenced_dof`    | str       | same set and constraints as `radiating_dof` |
| `body`              | str       | 1-D, unique, non-empty, exactly the bodies implied by the DOF axes |
| `xyz`               | str       | exactly `["x", "y", "z"]` |

DOF completeness: for every body on the `body` axis, all six canonical DOFs
(surge, sway, heave, roll, pitch, yaw) must be present on both DOF axes.

## 2. Mandatory variables

| Variable                | Dims                                   | Dtype      |
|-------------------------|----------------------------------------|------------|
| `added_mass`            | (omega, influenced_dof, radiating_dof) | float64    |
| `radiation_damping`     | (omega, influenced_dof, radiating_dof) | float64    |
| `excitation_force`      | (omega, wave_direction, influenced_dof)| complex128 |
| `hydrostatic_stiffness` | (influenced_dof, radiating_dof)        | float64    |
| `rotation_center`       | (body, xyz)                            | float64    |
| `center_of_mass`        | (body, xyz)                            | float64    |
| `center_of_buoyancy`    | (body, xyz)                            | float64    |

All arrays must be finite. The full 6N x 6N coupled space is mandatory;
cross-body blocks are retained, never discarded.

## 3. Optional variables

`added_mass_infinite(influenced_dof, radiating_dof)` float64: the infinite
frequency added-mass limit. If present it must carry a non-empty `source`
attribute recording its provenance (backend limit solve or explicitly
validated extraction, with source identity and generation settings).

Time-domain admission requires `added_mass_infinite`. A frequency-only
dataset is valid but is rejected by `validate_for_time_domain`; the last
finite-frequency sample is never silently substituted.

Other optional variables (dims checked where canonical, finiteness always):
`froude_krylov_force`, `diffraction_force`, `mean_drift_force`
(omega, wave_direction, influenced_dof); `panel_pressure`,
`field_velocity`, `free_surface_elevation`, `kochin_radiation`,
`kochin_diffraction`.

Unknown variables are rejected. In particular, dry mass, inertia and any
mechanical-mass authority are excluded from this schema.

## 4. Mandatory dataset attributes

| Attribute             | Value / constraint |
|-----------------------|--------------------|
| `schema_version`      | `"1.1.0"` |
| `backend`             | backend name (honest `"unknown"` allowed) |
| `backend_version`     | honest `"unknown"` allowed |
| `backend_commit`      | honest `"unknown"` allowed |
| `phasor_convention`   | `"x(t) = Re{X * exp(-i*omega*t)}"` |
| `heading_convention`  | `"wave propagation direction from +X toward +Y, radians"` |
| `normal_convention`   | `"outward"` |
| `length_unit`         | `"m"` |
| `force_unit`          | `"N"` |
| `moment_unit`         | `"N.m"` |
| `mesh_hashes`         | JSON object mapping every body_id to its panel mesh hash |
| `body_identity`       | JSON object mapping every body_id to `{mesh_hash, lid_mesh_hash, rotation_center, center_of_mass, enabled_dofs}` |
| `request_hash`        | stable hash of the originating `HydroRequest` |
| `created_utc`         | ISO 8601 UTC timestamp |
| `qualification_status`| one of `development_fixture`, `unqualified`, `qualified` |

`body_identity` (new in 1.1.0) records exactly what the producing side knew
about each body: hull and lid mesh hashes, reference frame (rotation
center), center of mass and the enabled DOF set. It must be consistent with
`mesh_hashes` and the `rotation_center` / `center_of_mass` variables, and it
deliberately contains no dry mass or mechanical data. Importers use it to
reject altered frame/COM/lid/DOF identity with a typed
`BodyIdentityMismatchError` instead of silently admitting changed bodies.
`shift_dataset_reference_point` keeps `body_identity` consistent when the
reference point moves.

`reference_transformations` (optional) is a JSON array of applied
reference-point transformation records.

## 5. Records

`HydroPanelMesh` (frozen): `vertices` (n, 3) float64 finite; `faces`
(n_faces, max_size) int32 with `-1` padding beyond each row's `face_sizes`;
`face_sizes` uint8 in [3, 255]; `body_id`; `outward_normals_verified` (retained
flag, true only after actual validation); `source_geometry_refs`;
`mesh_hash` (deterministic content hash, verified on construction).

`HydroBody` (frozen): `body_id` (no `::`), `name`, `mesh` (matching body_id),
optional `lid_mesh` (matching body_id), `rotation_center`, `center_of_mass`,
`enabled_dofs` (body-qualified, unique, non-empty subset of the six DOFs).

`HydroRequest` (frozen): `omegas` (finite, > 0, strictly increasing),
`wave_directions` (finite), `water_depth` (positive finite, or `+inf` for
deep water), `rho` > 0 finite, `gravity` > 0 finite,
`include_zero_frequency`, `include_infinite_frequency`, `backend_settings`
(JSON-serializable mapping, deep-frozen: nested mappings and sequences are
immutable too, and the request hash is invariant under freezing). Zero and
infinite frequency limits are separate flags; they never appear inside
`omegas`.

Duplicate or incomplete IDs are rejected at record construction and at schema
validation.

## 6. Persistence format

NetCDF via the scipy engine, format `NETCDF3_64BIT`:

- complex variables are split explicitly into `<name>_real` and
  `<name>_imag` float64 variables and recombined exactly on load;
- nested provenance attributes (`mesh_hashes`, `body_identity`,
  `reference_transformations`, `backend_settings`) are stored as JSON-encoded
  strings;
- no backend objects, no pickle payloads;
- writes are atomic (temp file in the target directory, then `os.replace`);
- loads are validated against this schema before use.

## 7. Hashing

- `mesh_hash`: deterministic SHA-256 over body_id, vertices, faces,
  face_sizes with explicit little-endian byte encoding. Never based on Python
  object addresses.
- `request_hash`: deterministic SHA-256 over request scalars, flags, arrays
  and canonical JSON of `backend_settings`.
- `dataset_content_hash`: deterministic SHA-256 over labels (coordinates),
  data, conventions and provenance attributes, excluding `created_utc`. DOF
  and body axes are canonicalized before hashing, so the hash is invariant
  under consistent reordering.
