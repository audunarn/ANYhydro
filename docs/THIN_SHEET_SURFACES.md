# Explicit hydrodynamic surfaces — development interface

`SurfaceSemantics` distinguishes `body_boundary`, `thin_sheet` and
`numerical_lid`, one primary `potential_flow`/`morison`/`none` owner and an
optional correction. `HydroPanelMesh.surface` retains stable surface/body/source
identities. `HydroBody.attached_surfaces` shares its body's existing six DOFs.
Unchanged legacy constructors, meshes and datasets retain their exact identities.
`migrate_surface_role` is explicit; ambiguous drag-only roles require a decision.
No open mesh, structural shell or numerical lid silently becomes a physical plate.

Sheets use an oriented planar midsurface: face winding points minus-to-plus,
pressure jump is `p_minus-p_plus`, and traction is that jump times the normal.
Connectivity stores the oriented boundary edges. Every connected component must
be open; duplicate/coincident opposing faces, inconsistent winding, nonmanifold
edges, degenerate panels and integer narrowing are refused. These checks do not
qualify general self-intersection, near-coincidence or plate-edge discretization.
`sheet_pressure_load` integrates supplied panel pressures and moments about an
explicit world reference. It does not calculate pressures or claim a sheet BEM.

Physical plate mass and any declared finite-thickness buoyancy stay in the
existing mechanical model exactly once. The surface schema admits neither mass
nor displaced-volume fields. Potential flow plus explicit plate drag is allowed
as an ownership declaration; full Morison sheet inertia or a member equivalent
diameter is refused. `PlateDragCorrection` specifies reference area, coefficient,
wave-plus-current velocity relative to the surface and normal projection. Its
force law is not implemented by this record slice.

Datasets with explicit surface semantics use **schema 1.2.0**; ordinary datasets
remain 1.1.0. Older readers refuse the new version. A current reader refuses
surface metadata relabeled as legacy, or a 1.2.0 dataset missing its inventory.
Surface semantics enter mesh hashes; surface inventory, source provenance and
pressure contract enter canonical dataset hashes. The imported backend matches
the complete inventory against actual caller records, in addition to the existing
hull/lid/reference/DOF/request identities. Hull-only data cannot serve sheets.

Coefficient-only external replay requires recorded pinned source/version/commit,
capability evidence SHA-256, declared verified genuine full hull–sheet/cross-body
coupling and the signed pressure convention. `sheet_source_provenance` is a
**declared evidence reference**, not independent scientific adjudication. Validation
does not execute an external solver, authenticate that evidence, promote the
dataset's qualification status or establish actual Capytaine/HAMS sheet support.
Scientific admission must inspect the referenced source/evidence under D10–D14.
Separate hull and plate sums or thickness approximations are not admitted as
genuine coupled sheet data by this route.

Local pressure is optional. If present, `pressure_jump` is complex128 on
`(omega, wave_direction, sheet_panel)`, with complete surface/face identities.
Its sealed dataset-level `pressure_jump_contract` pins **world total excitation
pressure jump in Pa/m**, under unit complex incident amplitude of 1 m and the
dataset's phasor/sign convention. Radiation pressure or other decomposition,
normalization and units are refused by this version. The dataset-level contract
survives NetCDF complex splitting; scalar world pressures remain unchanged under
reference shifts and coefficient-axis rotations while generalized coefficients
are mapped through the existing work-conjugate interfaces.

`HydroBackendCapabilities` exposes explicit evidence states for
`native_thin_sheet`, `mixed_hull_sheet`, `multibody_with_sheets` and
`sheet_pressure_jump`. All four Capytaine adapter states remain **unsupported**.
Native requests fail before cache, geometry processing or Capytaine import; there
is no one-sided, doubled-panel, thickness, omitted-force or zero-fill substitution.
The imported backend can transport a supplied pressure field (`implemented`),
but cannot solve new sheets. No backend has been numerically qualified here.

Verification evidence and independent review:
`C:/ANYworkspaceWork/anymarine-evidence/sheet-boundary-checks` and
`sheet-boundary-review.md`. Application surface persistence/geometry import is
the next connected slice; existing application routes cannot yet supply these
new plate records. Baseline gates and separate D10–D14 remain unchanged.
