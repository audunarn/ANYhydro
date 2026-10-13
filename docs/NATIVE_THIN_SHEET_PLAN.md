# Native thin-sheet (dipole panel) BEM on Capytaine kernels — plan

Owner decision 2026-10-13: dipole (zero-thickness) panels are developed in ANYhydro on Capytaine's
Green-function kernels (pinned Capytaine 3.0.0, not modified). A desktop GUI job for this route
comes later. The assessment that led here is
anymarine-evidence/dipole-bem-assessment-20261013/REPORT.md. The surface records and dataset contracts
already exist (THIN_SHEET_SURFACES.md). This plan makes the `native_thin_sheet`,
`mixed_hull_sheet`, `multibody_with_sheets` and `sheet_pressure_jump` capabilities of the Capytaine
route real. The development slices are named TS1-TS6 so as not to clash with the qualification
ladder D0-D14 in ANYmarine's IMPLEMENTATION_PLAN.md; their evidence feeds D10-D12.

## Formulation (WAMIT-style dipole panels, direct method)
- Unknowns: phi on hull (thick) panels; the potential jump m = phi_plus - phi_minus on sheet panels
  (oriented midsurface, normal minus to plus, THIN_SHEET_SURFACES.md).
- Hull rows: the existing direct-method equation, plus sheet columns int m dG/dn_xi dS (double
  layer, without a jump term).
- Sheet rows: d/dn_x of the representation equals the prescribed normal velocity, which is
  continuous across the sheet (V.n for radiation, -dphi_I/dn for diffraction).
  - The Rankine part of d/dn_x of the sheet double layer is hypersingular. For a flat constant-
    strength panel it equals the velocity of a vortex ring on the panel edges (Biot-Savart), and
    the same holds for its free-surface image.
  - The wave part needs second derivatives of the regular wave term. A first route is central
    differences of Capytaine's gradient at shifted collocation points.
- Loads from the jump: F = -i w rho int m n dS (signs per CONVENTIONS.md, verified in TS3).
  Sheets add no buoyancy or hydrostatics and have no interior (no irregular frequencies).

## Slices and acceptance (limits declared before each run; failures recorded, never tuned)
- TS1 Kernels.
  - Capytaine's G normalization pinned empirically.
  - Doublet-panel (vortex-ring) velocity vs an independent quadrature of grad(double layer) at
    off-panel points: round-off level.
  - Self term at a disc centre vs the analytic vortex-ring value.
  - Wave-term Hessian by central differences: O(h^2) Richardson convergence, trace 0 (harmonic), and
    tabulated vs exact (untabulated) evaluation measured.
- TS2 Isolated submerged sheet, infinite depth (D10). A deeply submerged thin disc gives the heave
  added mass 8/3 rho a^3 in the infinite-fluid limit, with edge-mesh refinement convergence. Plus
  agreement with OrcaWave on a submerged plate case built via OrcFxAPI.
- TS3 Jump/side reversal and signed force, moment and work mapping (D11). Reversing the sheet
  orientation leaves all coefficients unchanged, with the pressure jump sign reversed.
- TS4 Mixed hull plus sheet, and surface-piercing sheets (D12). Reciprocity (A, B symmetric) and
  dissipation (B positive semidefinite); agreement with OrcaWave dipole cases (simple skirt and
  vertical plate models built via the API).
- TS5 Finite depth. Agreement with OrcaWave at a finite depth.
- TS6 Adapter admission: capability states implemented with evidence, dataset schema 1.2.0 with
  the pressure jump. Then the C2 floater vs OrcaWave (panel ceiling decision then), and the ANYmarine
  Morison RAO comparison.
- Later: GUI Capytaine job; D13/D14 in ANYmarine.

Open owner decisions: the panel ceiling for the C2 floater (about 7800 panels, before TS6); porous
sheets (later).
