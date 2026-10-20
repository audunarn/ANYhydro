# OrcaFlex vessel-type export (slice O1b)

`anyhydro.orcaflex_export` writes a validated single-body ANYhydro dataset into an OrcaFlex 11.6c **vessel type**
(frequency-dependent added mass and damping, A_inf, first-order load RAOs, hydrostatic stiffness, optional far-field mean
drift, optional displacement RAOs). Every convention below was proved against OrcaFlex 11.6c by the O1a probes (local
evidence, not part of this repository). OrcFxAPI is an **optional, licensed** dependency: the module imports without it,
the pure data layer needs no licence, and the OrcFxAPI tests skip cleanly without it.

OrcaFlex outputs, saved `.dat`/`.yml` models, help extractions and any other Orcina data never enter this repository.

## API

```python
from anyhydro import orcaflex_export as ox

props = ox.DeclaredMassProperties(mass=..., centre_of_gravity=(x, y, z), inertia=[[...], [...], [...]])  # kg, m, kg m^2
data = ox.vessel_type_data(dataset, props, length_scale=L, gravity=9.81,
                           displacement_raos=None, motions_source=None,
                           include_mean_drift=True)                               # pure, no OrcFxAPI
vt = ox.export_vessel_type(model, dataset, props, name="Barge", length_scale=L, gravity=9.81)   # OrcFxAPI writer
```

- `vessel_type_data` validates, composes, symmetrises, converts and returns plain data (scalars, indexed items, RAO and
  added-mass tables, the zeroed load-coefficient blocks, a provenance record). `write_vessel_type(model, data, name=...)`
  sets exactly those items and reads them back (see Read-back); `export_vessel_type` is the two combined.
- Errors are `OrcaFlexExportError` (a `HydroError`). Nothing is converted silently.
- The provenance JSON (dataset content hash, conventions, unit factors, declared inputs, skew ratios, headings, warnings,
  writer and OrcFxAPI version) is stored in the vessel type's tag `anyhydro_provenance`; the dataset content hash is also
  in the tag `anyhydro_dataset_content_hash`. Tags are saved with the `.dat` file.

## Declared inputs (all required, no defaults)

| Input | Rule |
|---|---|
| mass, centre of gravity (world), inertia about the CoG in world axes | the dataset carries no mass properties (ANYsolver owns them) |
| `length_scale` | normalises rotations (1/L) for the skew bound; also written as the vessel type `Length` |
| `gravity` | the single g; must equal every gravity the dataset declares (contract provenance - required -, `wave_environment`, scalar `g`) |

Water density comes from the dataset (contract provenance, `wave_environment` or scalar `rho`; all present values must agree).
The writer refuses a model whose `environment.Density` differs from it (the explicit `DisplacedVolume` balances the mass
only at that density). OrcaFlex's own g is 9.80665 m/s^2; the relative difference to the declared g is recorded.

## Conventions and units (ANYhydro SI to OrcaFlex "SI": te, kN, m, s)

| Quantity | Conversion | OrcaFlex item | Convention |
|---|---|---|---|
| model units | - | `general.UnitsSystem == "SI"` (checked, never changed) | te, kN, m, s; angles in degrees in tables |
| mass, inertia, added mass, A_inf | x 1e-3 | `Mass`, `MomentOfInertiaTensorX/Y/Z`, `AddedMassMatrix*` | inertia about the CoG |
| damping, stiffness, forces | x 1e-3 | `Damping*`, `HydrostaticStiffness*` | angles per rad, no angle conversion |
| frequency | x 1 | `WavesReferredToBy = "frequency (rad/s)"`, `RAOPeriodOrFrequency`, `AMDPeriodOrFrequency` | rad/s |
| phase | degrees(arg X), no sign change | `RAOPhaseConvention = lags`, `RAOPhaseUnitsConvention = degrees`, `RAOPhaseRelativeToConvention = crest` | ANYhydro `x = Re{X exp(-i w t)}` equals OrcaFlex `x = A cos(w t - phase)` |
| heading | degrees(wave_direction) mod 360 | `RAODirection` | propagation direction from vessel +x toward +y; equal headings modulo 360 are merged only if their data are equal, else refused |
| load RAOs | x 1e-3 (kN/m, kN.m/m) | `SelectedRAOs = "Wave load RAO"`, `RAO*Amp/Phase` | wrench about the rotation centre = `RAOOrigin` |
| phase origin | - | `RAOPhaseOriginX/Y/Z = (0, 0, 0)` | the incident-wave phase origin is the world origin, **not** the RAO origin |
| displacement RAOs (optional) | x 1 (m/m, rad/m) | `"Displacement RAO"`, `RAOResponseUnits = radians`, `RAOWaveUnit = amplitude` | motion of the RAO origin; supplied by the caller as complex `(omega, heading, 6)` |
| mean drift | x 1e-3 | `"Wave drift QTF"`, `WaveDriftQTFMethod = Newman's approximation` | surge, sway, yaw from `mean_drift_far_field` (yaw about the rotation centre); heave, roll, pitch zero (declared); amplitudes are signed |
| hydrostatic stiffness | x 1e-3 | `HydrostaticStiffnessz/Rx/Ry` (upper triangle) | heave, roll, pitch block of the stiffness about `StiffnessInertiaDampingRefOrigin` = rotation centre |
| volume, buoyancy | - | `DisplacedVolume`, `CentreOfBuoyancyX/Y/Z` written explicitly | datum `HydrostaticReferenceOriginDatumPositionz` = rotation-centre height; vessel at world position 0 |
| infinite frequency | - | AMD row at `OrcinaInfinity()` holds A_inf (damping 0); each RAO table gets a zero-response row there | |
| symmetry | - | `Symmetry = "None"` | arbitrary headings, at least two |

Time-domain note: OrcaFlex results carry a time-stamp offset of about 0.29 dt against the exact wave (O1a probe B0), identical
for loads and motions; it vanishes with dt.

## Restoring: buoyancy plus gravity (owner design rule 1)

`K = sym(chart_stiffness(contract, K_source)) + K_gravity(declared mass, CoG, g)`, with the ADR-004 gravity block exactly as
ANYmarine composes it (`-W (c_z I - c e_z^T) - 1/2 skew(tau)`, `tau = -W c x e_z`, `c` = CoG relative to the rotation
centre), then the heave/roll/pitch block is written. Only a bound `BUOYANCY_ONLY` contract with a measured baseline wrench
is composed (same gate as `require_hydrostatic_composition`). The vessel must be in equilibrium against the baseline wrench
within 1e-3 W (force) and 1e-3 W L (moment), the dataset centre of buoyancy must agree with the baseline wrench within
1e-3 L, and `|mass - rho V| <= 1e-3 rho V` with `V = F_z,baseline / (rho g)`. The entries of the composed stiffness outside
the written block (surge/sway/yaw rows and columns, e.g. the gravity yaw coupling) are not written; their normalised
Frobenius ratio is recorded as `not_exported_entries_frobenius_ratio`.

## Symmetrisation (owner design rule 4)

OrcaFlex stores only symmetric A, B, A_inf, K. Each operator (every frequency sample of A and B) is split in normalised
coordinates (translations 1, rotations 1/L) into symmetric and skew parts; the symmetric part is written, the skew ratio is
recorded (`skew_ratios`), and a ratio above 1e-2 is refused.

## Refusals (typed `OrcaFlexExportError`)

multibody; non-GLOBAL coefficient frames; unknown phasor, heading or unit conventions; invalid dataset or non-finite data;
missing `added_mass_infinite`; missing, UNKNOWN or non-`BUOYANCY_ONLY` hydrostatic contract; gravity or density missing,
inconsistent or not equal to the declared g; mass imbalance beyond 1e-3; weight/buoyancy not in equilibrium; inconsistent
centre of buoyancy; skew above the bound; non-diagonal inertia (products of inertia: the OrcaFlex sign convention is not
probed); fewer than two distinct headings; equal headings with different data; displacement RAOs of the wrong shape, grid
or with non-finite values; displacement RAOs without a `motions_source` (or a source without motions); a present but
undecodable `wave_environment` attribute (it is never treated as absent); non-SI model; model density different from the dataset's; an existing object name.
Heading gaps above 45 degrees and the RAO frequency band are recorded as warnings (OrcaFlex warns and extrapolates).

## Not exported

Surge, sway and yaw hydrostatic stiffness (moorings and horizontal restoring belong in the OrcaFlex model); full QTFs and
sum-frequency QTFs; current, wind and other-damping coefficients (a new vessel type holds Orcina's default tanker there; these blocks are
**zeroed, not exported**: `OtherDampingLinearCoeff*`, `OtherDampingQuadraticCoeff*`, `CurrentCoeff*Area/AreaMoment`,
the `CurrentCoeff*` and `WindCoeff*` tables and `CurrentYawRateDrag*Factor`, so no default data acts if a user enables the
effect; the list is recorded in the provenance under `load_coefficient_blocks` and verified in the read-back); all Vessel-object settings (position, heading, motion
mode: for the load-RAO path use `PrimaryMotion = Calculated (6 DOF)`, `SuperimposedMotion = None`, and
`IncludeWaveLoad1stOrder`); the impulse-response controls `AMDTimeStep`, `AMDCutoffTime`, `AMDCutoffTolerance` (OrcaFlex
defaults, to be set and recorded by the O1c study); multibody; non-zero datum heel and trim.

Unused RAO sets (displacement RAOs when none are supplied, the drift QTF when no drift is present) are reset to zero tables
because a new vessel type is pre-filled with Orcina's default vessel.

## Displacement RAO provenance

`displacement_raos` needs a declared `motions_source` (non-empty text). The provenance records the source, the DOF keys and
a SHA-256 of the canonical motion array (complex128, little-endian, C order, `-0.0` normalised) together with its omega and
wave-direction coordinates and shape (the pattern of `anyhydro.mean_drift`), so exports of different motions never share a
provenance or tag identity.

## Read-back

After writing, `_verify_written` reads back and compares everything written: all scalars and indexed items; the zeroed
load blocks; every RAO set (response origin, phase origin, the exact set of directions and frequencies per direction,
amplitudes and phases, phases compared as unit phasors where the amplitude is non-zero, the infinite-frequency row); unused
sets must read as the zero tables; every added-mass and damping row including the infinite-frequency row. Tables are
matched by value because OrcaFlex re-sorts them; extra or missing directions or frequencies are mismatches. A mismatch
raises `OrcaFlexExportError` and destroys the half-written object. The verifier is covered without a licence by pure
writer-double tests that corrupt each category; the OrcFxAPI tests additionally prove the physical moment transfer.
