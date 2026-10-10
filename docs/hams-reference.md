# Pinned HAMS-MREL radiation reference preparation

Development tooling, not an admitted production backend or W7 acceptance.
`prepare_hams_first_slice_reference(new_directory, subdivisions=6, omega=1.5)`
writes a new two-box radiation deck and immutable SHA-bound sidecar without
launching a solver. `case.verify()` refuses changed inputs and extra generalized
mode files. `verify_hams_executable(path)` requires the published Windows v2.2
asset SHA256 4abcddfa4d1ef1f75f58377bf55bd6cc14f97be4ac4ae3f0fdd35c9daa7f0fcb.
Its associated source tag commit is cf9583c1609a5d1b904d24e09dff81e383b2ffad;
this does not establish compiler provenance. Keep the Apache-2.0 upstream binary
external to these MPL-2.0 sources.

The fixed release has rho=1000 and g=9.80665. This radiation-only route maps the
target omega to omega_native=omega_target*sqrt(g_native/g_target), preserving
omega^2/g, geometry and depth. `.1` is explicitly OUFT=3 (native angular frequency,
not period), reference length1m, influenced rows/radiating columns; dimensional
A=rho_target*a and B=rho_target*omega_target*b. Target rho1025/g9.81 are recorded
separately. No excitation, hydrostatic or motion equivalence is claimed. Auxiliary
mechanical input is explicitly synthetic and its outputs must not be imported.
HAMS computes inward-solid normals from unchanged outward geometric winding;
its own numerical waterplane is z=0, distinct from Capytaine's submerged lid.

Execution is separately registered; this API grants none. If executing, supply
exactly two arguments Input and Output with cwd the case directory: the upstream
reordering routine hard-codes Output/Wamit_format/AmssDamp.1. Keep failed and
partial output. Process exit alone is not success. The strict reader requires
both `read_hams_radiation_point(case, path_to_AmssDamp_1, Hams_format_directory)`
and every native radiation file, preventing final .1 reordering from concealing
missing rows through zero padding. It retains all144 signed coupled entries,
asymmetry and decimal rounding bounds. Real-rate mean power returns the value
and its conservative printed-precision bound; no near/far assessment, correction,
frequency-limit coefficients or full canonical hydrodynamic dataset is supplied.

Authoritative pin/source/runtime evidence lives outside repositories at
C:/ANYworkspaceWork/anymarine-evidence/hams-reference-20261010.

The official Windows asset invokes Unix `mkdir -p` internally. Preparation now pre-creates both output subdirectories; the failed original launch is preserved. This repairs launcher portability without modifying the binary or numerical inputs.
