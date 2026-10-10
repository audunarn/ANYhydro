# Corrected HAMS coefficient import

`anyhydro.hams_import.import_hams_coefficients` transports explicitly bound
HAMS-MREL coefficients into the neutral dataset. It does not execute a producer.
The public immutable `HamsImportContract` declares source body order, WORLD
coefficient frames, hull/lid identities, CB records, fluid/request identities,
all input and coefficient hashes, executable identity and corrected compiled
source identity. `input_directory`, `native_directory` and the executable path
are required so the declared identities are checked against actual bytes.

This profile is separate from `import_wamit7`, which continues to require
standard WAMIT periods and dimensional reference-length powers. HAMS output
type3 is angular frequency in rad/s. The pinned HAMS coefficient writers use
these literal conversions, regardless of their configured reference length:

```
A = rho * Ahat
B = rho * omega * Bhat
C = rho * g * Chat
F / incident_elevation_amplitude = rho * g * conjugate(.3 rectangular complex)
```

The signed excitation conversion follows the pinned source: `VINP` stores
`psi = physical_potential / i`, its incident spatial phase is positive along
the propagation heading, and verified solid-outward winding becomes native
inward-solid normals. Canonical `exp(-i omega t)` pressure integration therefore
gives `F = i*EXFC`; the writer emits `-i*conjugate(EXFC)/(rho*g*a)`.
`HydroStaticMulti` initializes incident origin WORLD X=Y=0 and amplitude1.
An independent underside-pressure crest fixture checks the upward force sign.
Source: [pinned incident potential](https://github.com/vaibhavraghavan/HAMS-MREL/blob/cf9583c1609a5d1b904d24e09dff81e383b2ffad/src/PatclVelct.f90)
and [pinned force writers](https://github.com/vaibhavraghavan/HAMS-MREL/blob/cf9583c1609a5d1b904d24e09dff81e383b2ffad/src/PotentWavForceMulti.f90).

The supported source has the two redundant-degree phase corrections documented
in [hams-producer-phase.md](hams-producer-phase.md), with an optional shared GNU
directory inquiry port. The profile identifies all33 compiled source files by
their aggregate hash. The caller also supplies the separately recorded build's
executable hash. Hashes bind declarations; they do not independently attest that
a binary was compiled from the claimed source or qualify its accuracy.

Admission checks capture strict control, panel and mechanical syntax; verify
WORLD vertices, connectivity, zero LCS, reference centers, COM and explicit lid
policy; reject extra mode/input files; and require all18 native acquisition
files. Native radiation rows follow frequency/body-pair order and native
excitation follows frequency/heading/body order. Every row is checked against
`.1`/`.3` at the combined literal printing bounds. This prevents a full reordered
`.1` table from concealing missing native acquisition. `.hst` must equal the
represented input CRS self blocks divided by `rho*g`, with zero cross-body
blocks. No missing entries, symmetrization, phase repairs or zero padding exist.

The profile requires total diffraction, full rigid six-DOF bodies, no symmetry,
native rho1000/g9.80665 and WORLD source axes. It refuses finite depths above
500m because pinned VINP and DINP use inconsistent branches there. Deck
preparation also refuses this regime and finite frequencies entering native
zero/infinite limit substitutions. Zero/infinite-frequency sentinel transport,
pressure/elevation/velocity fields, sheets and generalized modes require
separate contracts. A missing A-infinity remains absent and time-domain
validation refuses it; no last-frequency substitute is made.

Imported arrays retain full coupled matrices and use the existing work-conjugate
frame/reference transport. The dataset and its signed excitation provenance
survive save/reopen with the same content hash. Restoring ownership stays
UNKNOWN: a supplied CRS table is not independent buoyancy/gravity/preload
evidence, and composition refuses until an explicit physical contract exists.

Verification evidence:
`C:/ANYworkspaceWork/anymarine-evidence/hams-profile-20261010/`.
74 focused source checks,222 final installed regression checks and26 additional
installed frame/persistence checks pass. All27 package Python files match source,
wheel and installed bytes. The stored GNU multibody candidate imports and
save/reopens through the installed API with7 input/25 output files unchanged and
18 native files verified. Optional syntax, GUI, solver, loads and BEM dependencies
can be absent during imports. No numerical job was run in this slice.

Outputs remain unqualified. This establishes source-derived coefficient
transport and refusal behavior, not convergence, passivity, physical source
agreement, Windows producer support, complete backend execution or W7 acceptance.
