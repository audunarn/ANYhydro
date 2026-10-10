# ANYhydro Capytaine development adapter

Revision: W2 slice, 2026-10-08. Module:
`src/anyhydro/capytaine_backend.py` (`CapytaineBackend`).

## Status and scope

This is a DEVELOPMENT adapter only. It produces development evidence and
exercises the canonical data model end to end; it confers no scientific
qualification. Mesh convergence, passivity and independent hydrostatic
qualification remain outstanding.

Verified against installed Capytaine 3.0.0 source (no guessing):

- Conventions: incident potential `exp(+i k (x cos b + y sin b))` with
  `exp(-i w t)` time factor (`bem/airy_waves.py`) — phasor `exp(-i w t)`;
  heading `beta = 0` propagates along +X, `beta = pi/2` propagates along
  +Y, and headings are measured counterclockwise from +X toward +Y —
  identical to the anyhydro canonical conventions. No sign flips are
  applied anywhere in the adapter.
- `cpt.Mesh` faces are per-face index lists; anyhydro's `-1`-padded arrays are
  converted without mutating the source records.
- Waterplane lid normals point DOWN (pinned Capytaine convention:
  `FloatingBody.__init__` forces `with_normal_vector_going_down` and
  `Mesh.generate_lid` uses `normal=(0, 0, -1)`). The adapter requires
  source lids to already have downward normals and refuses anything else,
  so it never relies on Capytaine's silent orientation flip and source
  panel identity is retained.
- `cpt.rigid_body_dofs(rotation_center=...)` names are capitalized
  (Surge..Yaw); `cpt.Multibody` renames sub-body DOFs to
  `<body_id>__<Dof>`, which the adapter maps back to the canonical
  body-qualified keys `<body_id>::<dof>`.
- `RadiationProblem(omega=np.inf)` is Capytaine's symbolic infinite-frequency
  limit (valid without forward speed).

## What the adapter produces

- Full 6N coupled `added_mass`, `radiation_damping` (one combined
  `cpt.Multibody` `fill_dataset` call, `hydrostatics=False`, `n_jobs=1`,
  `n_threads=1`, `progress_bar=False`): cross-body off-diagonal blocks are
  preserved, never discarded.
- `excitation_force` (complex, canonical phasor/heading conventions).
- `hydrostatic_stiffness`: Capytaine's linear buoyancy-only hydrostatic
  stiffness (`compute_hydrostatic_stiffness`) computed per sub-body with
  `mass=0` and block-assembled. This is NOT the total dry-body restoring:
  the gravity restoring of the dry body is owned separately by the solver.
  The dataset attribute `hydrostatic_stiffness_definition` states this
  explicitly.
- `added_mass_infinite` (only when the request sets
  `include_infinite_frequency=True`): explicit per-radiating-DOF
  `omega=inf` radiation solves, with the exact Capytaine version recorded in
  the variable's `source` provenance.

## Typed refusals (never silent)

- Zero-frequency limit (`include_zero_frequency=True`):
  `UnsupportedOperationError`. The canonical schema stores strictly
  positive finite frequencies only; `supports_zero_frequency` is always
  False.
- Unverified outward normals: `GeometryValidationError` — the adapter never
  flips arbitrary geometry.
- Non-wetted geometry (hull panels above the free surface or below the
  bottom, lid panels off the z=0 plane, lid panels not pointing DOWN):
  `GeometryValidationError` — the BEM geometry must already be
  equilibrium-wetted with correctly oriented lids; the adapter never clips
  and never flips source geometry.
- Invalid (non-finite) solver output: `BackendSolveError` — no physical
  replacement coefficients are ever invented.
- Unknown backend settings: `CapabilityError`. Supported keys:
  `method` (BEM method passed to `fill_dataset`), `cache_dir` (cache
  directory override). Unknown keys are refused, not silently ignored.
- Execution limits: `CapytaineExecutionLimits` (see below). Bool,
  fractional, non-finite, negative, zero and unknown limit controls raise
  `CapabilityError` at construction — before any Capytaine import or
  cache lookup.

## Execution limits

`CapytaineExecutionLimits` is the frozen public limits type, passed
explicitly to `CapytaineBackend(execution_limits=...)`. Defaults preserve
the tiny development budgets: 100 total panels, 3 finite frequencies,
1 heading, `n_jobs=n_threads=1`. Explicit limits may admit up to 2048
total panels (the absolute software panel ceiling; the registered
384/864-panel source-study meshes require explicit limits and are still
refused by the 100-panel DEFAULT) and up to 513 finite frequencies for
later registered cases (never executed by default); the heading (1)
ceiling stays fixed in this first-slice preparation and
`n_jobs`/`n_threads` are pinned to 1 — no
silent thread or environment overrides. Cases above the current
2048-panel execution capacity are refused honestly; there is no
autounbounded profile. All caps apply before cache
lookup, and the effective limits are recorded in `capabilities()`,
`diagnose()`, the cache identity and the dataset provenance attributes
(`development_limits`, `execution_limits`). A limits profile confers no
qualification: these are execution bounds only.

## First-slice fixture

`anyhydro.first_slice` provides the deterministic two-body wetted-hull +
lid development fixture used by the first-slice preparation pilot
(`examples/first_slice_bem_check.py`): body ids `first-slice-a` /
`first-slice-b`, references (-2, 0, 0) / (2, 0, 0) m, upright 1 m x 1 m
boxes with 0.5 m draft, hull = bottom + four sides over z in [-0.5, 0]
(40 hull panels per body), waterplane lid at z=0 with downward normals
(8 lid panels per body), 96 panels total, submerged volume 0.5 m^3, COB
z = -0.25, waterplane area 1.0 m^2 — all verified analytically by
`verify_first_slice_fixture`. Case constants (rho 1025, g 9.81, depth
150, dry mass 512.5 kg, heave buoyancy stiffness 10055.25 N/m) are
fixture/evidence metadata; dry physics is owned separately by the
mechanical case contract. The fixture confers no scientific
qualification.

## Cache

Optional content-addressed NetCDF cache (`cache_dir` constructor argument or
`cache_dir` backend setting; no cache when unset). The cache identity covers
hull and lid mesh hashes, body ids, rotation centers, centers of mass,
enabled DOFs, the full request (frequencies, heading, water depth, rho,
gravity, limit flags), the settings, the effective execution limits, the
exact Capytaine version and the
adapter/schema versions — a cached entry can never be reused for changed
input. Entries are written atomically and validated on load (schema
validation plus exact identity and request-hash match); a corrupted or
mismatched entry is a miss and the solve runs fresh. The cache stores
validated canonical datasets only — never pickled backend objects — and
there is no silent fallback to an alternate backend on failure.

## Capability honesty

`capabilities()` reports `supports_zero_frequency=False`,
`supports_infinite_frequency=True`, `provides_added_mass_infinite=True`, and
`False` for panel pressure, Kochin, free-surface and mean drift: those
optional pressure/local-flow postprocessing capabilities stay disabled
pending their own tests. `diagnose()` reports Capytaine availability without
importing it (the import happens lazily inside `solve` only; the base
`anyhydro` import never touches Capytaine).
