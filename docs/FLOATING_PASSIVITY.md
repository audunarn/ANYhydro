# Floating-family damping passivity: 60 m barge attribution

Status: development evidence, 2026-10-30. Qualification is not claimed. Evidence:
`C:/ANYworkspaceWork/evidence/barge60-passivity-fix-20261030` (retained diagnostics, native chunks and production preparation).

## Finding

The 60 x 5 x 1 m hull radiates almost the same far field in sway and roll, so sym(B) has a near-null sway+roll direction.
Its far-field diagnostic value is 3.5e-11 of the largest eigenvalue at omega 0.56 and about 1e-5 at omega 3.1; the largest
eigenvalue is pitch. The near-field value in that direction carries ordinary BEM error, about 1 % of the sway/roll entries,
so the raw matrix fails the relative 1e-9 gate wherever the true value is below that error.

The actual source has one panel row across the 1 m draft, three to four cells across the beam, and triangles only.

Controlled diagnostics found that replay repair, a different Prony fit, exact imaginary kernel, a direct method on the coarse
source, and the tested receiver-integration prototypes were each insufficient alone. These bounded comparisons do not exclude
every possible discretization or kernel error. The first box irregular-frequency estimate is about 3.3 rad/s, above the tested
band; it does not explain the observed low-frequency failures.

Relative smallest eigenvalue, ANYhydro Green function, 60 x 5 x 1 m box, direct method, gate -1e-9:

| nx ny nz | panels | omega .565 | .974 | 1.301 | 1.792 |
|---|---|---|---|---|---|
| 48 4 1 | 296 | -1.9e-8 | -7.3e-8 | | -8.7e-7 |
| 48 16 2 | 1024 | -7.3e-10 | -3.6e-9 | | -2.5e-8 |
| 48 32 2 | 1856 | -4.9e-10 | -2.7e-9 | -1.5e-8 | -2.5e-8 |
| 48 32 4 | 2176 | -9.47e-11 | -2.2e-10 | +6.4e-10 | +6.8e-8 |

The indirect route does not converge in this direction: -9.2e-8 and -1.1e-6 at 1024 panels (omega .565 and 1.792), and it
worsens in some sampled refinements. The final explicit 2176-quad direct profile subsequently passed all 32 original realization
frequencies, 0.56450493 through 3.10068650 rad/s. Worst relative eigenvalue: -2.5343925518937395e-10, against the unchanged
-1e-9 gate. No coefficient symmetrization, clipping, replacement or DOF rescaling was applied. Passing this gate establishes the
tested admission, not convergence of every near-null eigenvalue or physical qualification.

## Direct recipe

`DIRECT_RECIPE` in `floating_radiation_fields.py`: ANYhydro Green function and the direct method. `ACTIVE_RECIPE` stays the
legacy recipe. Equations on panel values: (1/2 I + D) phi = S u; off-body phi(x) = S_x u - D_x phi with the replayed kernel;
p = i w rho phi; A and B from the same panel pressures. The `*_sources` slots hold the Neumann data u for this recipe.

The recipe is part of the request, so legacy records stay recoverable as themselves, are never current for the new recipe, and
never donate radiation data. Transient source profiles keep the indirect recipe and refuse the direct one.

## Refined direct quad profile

Explicit development route: all 32 damping matrices pass. The original sea state remains refused by a different admission.

- `FloatingSettings.source_profile = dict(name='refined_direct_quad_v1', resolution=[nx, ny, nz], full_panel_limit=N)`, N <= 2500.
  With a profile, `panel_size` stays 0 and `max_panels` stays 600. `None` keeps the default route and its identities.
- `anymesher.floating_quad_source` builds native quads from the wet plates of the geometry owner. The admitted scope is one bottom
  and four walls of a world-axis box reaching z = 0, within 1e-9 of the longest edge. A world-axis-aligned 90-degree yaw may be
  admitted; resolution follows world axes. Other poses/shapes, partial or extra plates refuse.
- `SourceManifest.triangles` holds one convex (4,3) polygon per face and is checked against `vertices[faces]` and the profile count.
  Overlap integrates those polygons with one pressure per native face. Triangle-only hydrostatics use `source_triangles_closed`, which
  fans them with `anygeometry.floating_overlap.fan_triangles`.
- The profile selects `DIRECT_RECIPE` and `source_capacity='refined_direct_quad_v1'` in the family request. The capacity is part of
  the cache and donor identity and sets the 2500 ceiling in `_pressure_body.prepare_body`; every other request keeps 600.

Reflection acceleration was not implemented or assessed. All 2176 physical hull panels are solved without moving the world
wave-phase origin.

## Production disposition

The frozen 60 m structural mesh, explicit 307500 kg mass model and actual owner equilibrium were used by the production worker
with native BEM solves disabled. All 32 exact family requests recovered. Response, static balance and source dynamic closure
passed their existing checks. The full conservative relative-waterline envelope was 2.238 m versus 1 m clearance, so preparation
refused before mapping/history publication. Hs=1 m, Tp=6 s and 1 m draft were not changed. No loaded project or FE result was
published; full Apply, offline load reopening and structural frame solves remain unverified for this sea state.

Independent Opus 5.5/medium reviews and their reconciliation are retained alongside the evidence. The final narrow repairs add
exact family-request/frequency refusal and a native panel-pressure re-solve using the replayed kernel. Focused final checks:
12 hydro tests and 22 FEM tests passed. Earlier timeouts remain evidence. No release or engineering qualification is claimed.
