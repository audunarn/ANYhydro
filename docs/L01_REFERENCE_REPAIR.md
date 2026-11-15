# Explicit waterline lid and source quadrature

The previous L01 development case used an inset158-panel lid at z=-0.10m.
Saved pressures reproduce its middle-frequency negative damping, so output
serialization is not the explanation. A fitted950-triangle reference diagnostic
at z=0, with the exact same2988 physical panels, makes damping positive at that
frequency. Scaled damping difference toOrcaWave falls from about4.34% to0.17%
with the original centroid source rule; four-node source quadrature gives0.23%.
These are single-frequency diagnostics, not convergence or production admission.
Changing elevation, coverage and lid discretization together does not isolate
elevation as the cause.

## Public development API

Install the optional `waterline-lid` dependencies. Generate the mathematical
surface explicitly, using an already wetted/outward-oriented physical hull:

```python
from anyhydro.direct_lid_profile import create_source_profile, WATERLINE_PROFILE_METHOD
from anyhydro.floating_radiation_fields import solve_family_snapshot, DIRECT_LID_RECIPE

profile = create_source_profile(
    vertices, faces, surface_ids, lid_z=0., water_depth=100.,
    method=WATERLINE_PROFILE_METHOD, waterline_cell_size=2.,
)
snapshot = solve_family_snapshot(
    vertices, faces, surface_ids, omega,
    heading=0., water_depth=100., rho=1025., gravity=9.81,
    Q0=rotation_body_to_world, cg_world=cg_world,
    recipe=DIRECT_LID_RECIPE, direct_lid_profile=profile,
    source_capacity="faceted_quad_hull_v1",  # explicit existing capacity, when needed
    # Optional: source_quadrature="Gauss-Legendre 2",
)
```

`waterline_cell_size` controls the interior-point lattice, not a guaranteed
triangle-quality or maximum-edge criterion. The existing ANYmesher constrained
planar triangulator is used with its explicit Python backend. Every original
waterline vertex/segment is retained; interior points are added without moving
the physical hull. Coverage and boundary connectivity are checked. Multiple
loops/holes, nonzero-height waterline boundaries, oversized generation work and
full sources over4000 panels refuse explicitly. Private meshes are not packaged.

The new method is `anymesher-constrained-waterline-lid-v1`. Its cell size,
complete geometry and mask are content-hashed; executable reconstruction must
regenerate the same canonical profile. The legacy profile method and its
serialized content remain unchanged when the new option is absent.

`source_quadrature="Gauss-Legendre 2"` selects four source integration points
per face for explicit direct families. It is part of request, donor/cache and
provider identities and is reconstructed during off-body recovery. Absent means
the historical centroid rule. No default numerical recipe changes. Pressure,
six coupled A/B entries per column and excitation are all obtained from the same
solve. The lid has zero prescribed rigid motion and is excluded from physical
pressure integration; its complete solved potentials remain in field recovery.

## Verification and measured limits

Focused tests cover concave waterline coverage, exact boundaries, work/panel
refusals, neutral imports, legacy identities, canonical tampering, real seven-RHS
public solves, pressure-to-force closure, cache/donor separation and retained
field recovery. The focused native-enabled suite passed27 tests. After independent
review repairs, the final affected module passed8 tests with the native gate on,
and6 passed/2 skipped with it off. Historical28 no-lid and18 inset-lid cached
families reconstruct unchanged request and stored snapshot identities.

The public shared-mesher profile has910 mathematical triangles at z=0. Its
three-frequency comparison preserves the exact original2988 physical panels,
original frame and environment. The table uses dimensionally scaled matrix norms
with the reference matrix in the denominator. The raw eigenvalue ratio is the
minimum eigenvalue of sym(B) divided by its largest absolute eigenvalue; the
unchanged gate is -1e-9. Raw stored matrices are never symmetrized or projected.

| Lid / source quadrature | omega rad/s | B difference toOrcaWave | toWadam | Raw ratio | Gate |
|---|---:|---:|---:|---:|---|
| Fitted910 / centroid | 1.05538 | 0.102% | 0.155% | +2.80e-5 | pass |
| Fitted910 / centroid | 1.95531 | 0.204% | 0.597% | +2.09e-5 | pass |
| Fitted910 / centroid | 2.77344 | 7.031% | 7.412% | -4.94e-5 | FAIL |
| Fitted910 / four-node | 2.77344 | 3.424% | 3.844% | -2.67e-6 | FAIL |
| Inset158 / four-node | 2.77344 | 13.690% | 11.459% | +1.30e-5 | pass |

The fitted four-node run stopped at its high-frequency failure; no low/middle
points of that public configuration were calculated. The final inset four-node
public check covers only the high frequency. Earlier manual middle-frequency
diagnostics remain separate evidence. Do not splice these configurations into a
production family or infer full-band acceptance from the positive rows.

Pressure-to-coefficient closure, cache roundtrip and all seven retained-field
recoveries without BEM re-solving passed on the measured public cases. Closure
establishes internal consistency, not reference accuracy: high-frequency
diffraction-pressure differences remain approximately45–56% against the references
for the fitted cases. These discrepancies remain unresolved.

An author-separated OpenAI source review was reconciled, including native test
scheduling, a pinned legacy-schema digest and direct-without-lid quadrature
coverage. This is not different-family review or scientific acceptance.

Raw matrices are preserved; no damping projection, pressure fitting, omitted
modes, reference substitution, physical/mass/sea changes or tolerance relaxation.

Native L01 remains unadmitted pending measured full-band/convergence evidence
and the existing consumer/production gates. No qualification or release claim.

## User acceptance of the selected reference difference — 2026-11-02

The user accepts the measured13.7% radiation-damping difference for the original
158-panel inset lid atz=-0.10m with `source_quadrature="Gauss-Legendre 2"`, at
omega2.7734372644972387rad/s. The exact scaled Frobenius difference toOrcaWave is
13.69025990983783%; toWadam it is11.458761080765898%. This comparison uses the
reference denominator and the original2988 physical panels. Its full raw damping
matrix passes the unchanged -1e-9 gate (relative minimum+1.3048363099703375e-5).

This records case-specific acceptance of that damping comparison. Other reported
quantities remain measured evidence; no blanket tolerance is inferred. The failed
fitted-lid configurations remain failed. The original32-frequency sea has only
one verified cached family under this exact inset/four-node configuration;31
frequencies remain unverified. No full-band or production qualification is implied.

## Authorized full-band attempt — 2026-11-02

The user authorized a separate3600-second cumulative check, at most120 seconds
per missing frequency, one thread and180 seconds reserved for verification,
stopping on first failure. The verified index27 family was reused. Checking the
highest missing frequency first, index31 atomega3.1006864992461756rad/s failed:
raw relative damping minimum-3.521474382656075e-5 versus the unchanged-1e-9 gate.
The run stopped with1 cached pass,1 newly calculated failure and30 frequencies
unrun. Full-band passivity therefore remains unverified.

The failed matrix is finite and its directional damping form from coefficients
(-5356.400193468222) matches the pressure-derived form(-5356.400193468223).
Source geometry, request identity, physical inputs and numerical source pins
were preserved. The failed raw snapshot remains evidence; it is not admitted.
The user's acceptance of the13.7% comparison atindex27 remains case-specific.

The execution harness received an author-separated OpenAI Read review. Cleanup,
pause and budget-reserve edge cases were repaired and passed5 synthetic controller
regressions. An explicit optimization0 audit rechecked the saved arrays without
new BEM solves. The original frequency execution did not record its optimization
flag; the later audit does not retroactively establish that setting. All actual
process-tree cleanups were verified. Total supervised process-wall use85.7587259s;
no additional frequency or reference/structural solve was launched after failure.

## Targeted repair follow-up — 2026-11-03

The explicit experimental source rules `anyhydro-bilinear-gauss-3-v1` and
`anyhydro-bilinear-gauss-4-v1` integrate the bilinear face map with9 or16nodes
and its area Jacobian. They preserve physical panels, receiver collocation and
legacy defaults/identities. Rules are included in request/provider/cache/donor
identities and rebuilt for retained recovery. Folded face maps and altered or
malformed native quadrature caches are refused. These options do not guarantee
discrete passivity or confer admission.

The nine-node rule passed the previously failing index31 but failed index27:
raw relative damping minimum-1.1151254547178393e-5 and damping difference25.4265%
against OrcaWave. It is rejected as the L01 repair. The original accepted13.7%
comparison is preserved for its original158-panel lid/four-node snapshot.

A separate candidate uses the existing native four-node rule with a canonical
230-panel inset lid atz=-0.10m and `faces_max_radius=1.61`. The physical hull
remains2988panels. Both resolution and coverage change: section coverage rises
from76.5537% to79.9575%, but boundary Hausdorff distance grows from3.0323m to
3.9216m. Both lids are connected with no measured outside area or overlapping
panels. Radius1 and1.6 were refused by the unchanged512MiB generation ceiling;
the admitted candidate preserves that ceiling and4000source-panel capacity.

This candidate passed indices31 and27, including cache reload, all7 retained
fields and pressure/coefficient closure. Atindex27 its damping differences are
11.1182% against OrcaWave and8.8172% against Wadam. This is a measured
lid-sensitivity result, not a claim that integration error alone caused the
original failure. Evidence and
the explicit profile are under
`C:/ANYworkspaceWork/evidence/l01-high-frequency-repair-20261103`.

The exact230-panel-lid/four-node candidate now passes all32 original frequencies.
The independent saved-data audit recovered all32 coherent snapshots, checked
physical geometry, finite raw matrices/pressures, provider identity, both pressure
closures, cache consolidation and all7retained states with BEM solves prohibited.
The smallest raw relative damping minimum is+3.1358009587885296e-8; radiation
closure is at most4.044598042498816e-16 and diffraction closure at most
1.7563407570665798e-15. All6protected inputs and46legacy cache identities remain
unchanged. This verifies the sampled development passivity/closure/retention
domain; it does not establish mesh convergence, continuous-frequency stability,
full reference agreement or production qualification.

Scaled damping-matrix differences at low/middle/high reference frequencies are
0.3296%/1.5158%/11.1182% against OrcaWave and
0.2742%/1.1177%/8.8172% against Wadam. Radiation-pressure differences against
OrcaWave are0.4174%/1.2390%/2.0848%. Incident-plus-diffraction pressure agreement
remains poorer:0.8873%/14.4559%/49.3339% against OrcaWave and
0.7707%/12.0879%/39.4547% against Wadam, using area-weighted panel norms.
No new acceptance of these pressure differences is inferred from the user's
original13.7%damping acceptance. No reference or structural solve, default
recipe change, install, commit, push or release action accompanied this check.
