# Immutable HAMS deck preparation

`anyhydro.hams_inputs.prepare_hams_deck(directory, bodies, request, mechanics,
executable_path=...)` returns a sealed `HamsPreparedDeck` without launching a
solver. `mechanics` maps every body ID to an explicit public
`anyfileio.hams.HamsMechanicalInput`. ANYfileIO remains optional at module/base
import; invoking preparation without that syntax package raises CapabilityError.

The route pins HAMS-MREL v2.2 source commit and official executable bytes already
recorded by `hams_reference`. The caller must supply a readable matching
executable; none is downloaded, copied into a wheel or run. Native rho1000 and
g9.80665 are required. Fluid similarity, backend-setting guessing and canonical
zero-frequency requests are refused.

The caller's unique body order and full ordered six rigid DOFs are retained.
WORLD-aligned vertices, reference centers, center of mass, hull/lid hashes and
request hash bind the deck. Vertices retain their WORLD coordinates with zero
LCS origins, including WORLD Z: pinned multi-body input ignores origin Z.
Reference centers must be exactly representable in the native three-decimal
field; geometry is never silently moved. Triangles/quads retain winding; hull
outward-normal verification is required. Thin sheets, generalized modes and
non-potential-flow hull ownership are refused.

Irregular-frequency removal requires an explicit WORLD-z=0 lid for every body.
A supplied lid cannot be silently omitted. Missing/submerged lids refuse.
Mechanical matrices are WORLD-aligned about each body's declared reference
center, with an explicit WORLD center of mass. They are caller inputs and do not
establish independent fluid hydrostatic evidence. No identity/zero mechanics
defaults are invented.

By default native E12.5 matrix quantization refuses. Explicit
`allow_mechanical_rounding=True` records caller mechanical identity and per-matrix
changed counts/maximum absolute changes alongside exact emitted file hashes.
This opt-in is not a scientific error tolerance or acceptance.

Preparation exclusively creates a new directory, precreates native output
directories and writes `case.json` last. Interrupted writes leave incomplete
evidence; a valid returned seal is required. `deck.verify(bodies=..., request=..., mechanics=...,
executable_path=...)` refuses changed/missing inputs, extra mode files, sidecar
changes and caller/mechanical/executable identity changes. Verification returns a fresh
manifest rather than mutable owned state.

One-body native filenames and numbered multibody filenames are emitted.
Frequency-limit acquisition is requested only for explicit Ainf requests.
No output is admitted: the pinned HAMS excitation phase inconsistency remains
unresolved. Prepared decks and unit tests are development evidence, not W7
cross-solver qualification. Source gates and historical failures remain intact.
