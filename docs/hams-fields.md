# HAMS finite point fields (development)

`anyhydro.hams_fields` provides canonical, source-bound incident, total,
scattered and radiation responses at explicit exterior WORLD points. It is
development transport, not a qualified diffraction/radiation analysis.

Only GNU gravity-consistent source aggregate
`077b0ec28d35339763cb2d4aaf079ee7da335919fe5f0486aa035faefd602a88`
and profile `hams_mrel_cf9583c_phase_limit_gravity_corrected_finite_v3`
admit this route. The caller binds its executable and exact input/raw file
hashes. Existing finite and limit profiles retain their meanings. This source
has no admitted infinite-frequency/zero-frequency transport; do not substitute
older-source Ainf into its datasets. A corrected native Windows build and
physical W5/W7 comparison remain open. Saved Linux evidence can be imported
on Windows, but its ELF producer cannot execute through the Windows backend.

Pass `field_points=((x, y, z), ...)` to `HamsBackend` or `prepare_hams_deck`.
Points are snapshotted before execution and bound into the input manifest.
The complete coupled source route requires at least two bodies, finite positive
frequencies, and distinct finite points: exact z=0 gives elevation, z<-1e-6
gives dynamic wave pressure in Pa, above the bed. Hull/lid bounding boxes and
the pinned source's tolerant ONBODY condition are conservatively excluded.
Near-surface ambiguous points, interior/on-body evaluation, attached sheets
and panel-pressure loads are refused. The point cap is
`HamsExecutionLimits.max_field_points` (default 4096), checked before launch.

`import_hams_fields(...)` reconstructs and verifies its primary coefficient
dataset against the same raw evidence; arbitrary parent datasets refuse.
`HamsFieldData` contains immutable bytes-backed arrays:

- incident and total: `(omega, heading, point)`;
- radiation: `(omega, point, target_dof)`;
- scattered: derived exactly as `total - incident`.

All arrays use `exp(-i*omega*t)`, WORLD phase origin (0,0), and target equilibrium
coefficient frames. Total already contains incident. Surface wave responses
are metres per incident amplitude metre; pressure wave responses are Pa per
incident amplitude metre. Radiation multiplies target displacement (m or rad),
not velocity: native rotational columns receive reference-length scaling, all
native complex values are conjugated, submerged values receive rho*g once,
and the source motion map acts on the right. No extra omega multiplier applies.
Pressure excludes hydrostatic pressure. Radiation/field point transport does
not establish large-excursion hydrodynamics or scientific accuracy.

`save_hams_fields(fields, new_path)` publishes a new hash-bound JSON artifact;
`load_hams_fields(path, expected_parent_hash=...)` verifies schema, units,
quantities and content identity. Reframing uses `reframe_hams_fields`, alongside
`reframe_dataset`, and verifies old/new parents, body/DOF order, frames and
frequency/heading coordinates before applying the radiation motion map.

Backend runs persist `fields.json`, reopen it, and bind its bytes/content/parent
to `run.json` and `completed.json`. After the last cancellation callback,
producer/input/raw/result/field/receipt identities and the deadline are checked
before completion. Failed/cancelled jobs retain evidence without a completion
seal. `load_completed_hams_fields(directory, bodies=..., request=...,
producer_build=...)` verifies the whole saved run and reimports the raw fields,
so resealing a semantically changed sidecar does not admit it. The coefficient
loader also refuses missing/tampered requested fields. An unsigned receipt is
not independent producer attestation or qualification.

This slice reused the saved corrected .7rad/s, two-body acquisition. It executed
no BEM or mechanics jobs. Lifecycle tests use private Python fixtures copying
stored output; they do not prove a native producer or the Qt desktop workflow.
The earlier source acquisition-ordering gap, original residual failures and
W5 NO_GO remain protected. ANYmarine's next connected slice is source-tagged
stored-wave replay through ANY3dView; Capytaine field bindings remain distinct.
