# Pinned excitation-phase producer correction

HAMS-MREL v2.2 source commit
`cf9583c1609a5d1b904d24e09dff81e383b2ffad` multiplies `ATAN2D` by `180/pi`
in both rigid-body excitation writers. [GNU documents ATAN2D's degree result](https://gcc.gnu.org/onlinedocs/gfortran/ATAN2D.html).
The separate source candidate removes that extra conversion. It changes emitted
`.3` phase only; native excitation, magnitude and rectangular components remain
unaltered. No imported result is patched or redundant source data bypassed.

`tools/prepare_hams_phase_candidate.py` prepares an external candidate from a
pristine user-supplied source tree. It verifies the exact compiled-source
inventory and all33 SHA256 identities in `hams_phase_source_recipe.json`, refuses
overwrite/nested destinations/unknown source, preserves upstream notices and
writes a source manifest. It downloads, builds and executes nothing. Example:

```powershell
python tools/prepare_hams_phase_candidate.py --source <pristine-pinned-source> --output <new-candidate> --gnu-compatible
```

The GNU option changes the Intel-specific directory inquiry to `INQUIRE(FILE=dir)`.
Apply that same IO-only change to a comparison baseline; do not compare a ported
candidate against the untouched Intel binary. GNU13.3 directory-existence
behavior was verified before the paired calls. The recipe's explicit source
order comes from the pinned upstream compile script. Existing GNU OpenMP and
LAPACK/BLAS runtime libraries were used; no compiler/packages were installed.

Measured Linux/WSL development evidence: paired GNU builds, one0.7rad/s frequency,
heading0deg, two identical four-subdivision boxes, one thread, no limits. The
seven input files are byte-identical before/after both calls. Both output
inventories contain144 radiation,12 excitation and144 supplied-restoring rows.
Legacy output has12 redundant phase errors; candidate has0 errors and no
indeterminate phase rows. Only `ExcForce.3` differs;24 other raw output files are
byte-identical, including native complex excitation/radiation. All printed `.3`
fields except phase are identical. Standalone quadrant/axis arithmetic uses the
same native format; its zero row is indeterminate, not phase-agreement evidence.

Evidence: `C:/ANYworkspaceWork/anymarine-evidence/hams-producer-phase-20261010`,
including source/executable/library identities, commands, process-wall budgets,
failed unmodified-source GNU build, independent review and raw paired outputs.
This demonstrates redundant-output consistency for the measured multibody case.
Single-body writer has the same source correction but has not had a BEM run.

The published Windows binary is unchanged and still refused for inconsistent
excitation. This GNU candidate is unqualified, external to the Python wheel,
and not accepted by the existing official-executable preparation pin. Canonical
phasor, signed force/incident-origin mapping, Windows producer build, full
frequency/spacing/depth comparisons and W7 acceptance remain open. Complete
hydrostatics are supplied input, not independent scientific evidence.

ANYhydro wrapper/tool sources retain MPL-2.0. External HAMS source retains its
Apache-2.0 notices; this recipe does not relicense or vendor that source/binary.

## Opt-in native limit-row reorder repair

The pinned multibody radiation writer emits four fixed-width fields for limits
and five for finite frequencies. Its original reorder routine requests five
values using advancing list-directed input: it can consume the following row's
frequency as damping, discard that row and fill missing coefficients with
initialized zeros. This prevents trustworthy native limit acquisition.

```text
python tools/prepare_hams_phase_candidate.py --source <pristine-pinned-source> --output <new-candidate> --gnu-compatible --preserve-frequency-limits
```

This separately versioned GNU-only source option replaces the reorder routine
using a hash-pinned `hams_reorder_limits.f90.inc` asset. The default phase-only
source profiles remain unchanged. All33 original compiled-source hashes and the
replacement hash are verified before creating output. New source hashes are
not automatically admitted by HamsProducerBuild or the canonical importer.

The replacement reads one physical fixed-width record, validates native real
and integer literal syntax, finite values, bounds, complete unique matrices and
distinct contiguous printed-frequency blocks. It rejects collisions and malformed
records instead of normalizing implied decimals/embedded blanks. Four-field
limits remain four-field; no computed limit damping is invented. For angular
output type3, native0 is zero frequency and native-1 is infinite frequency.
This differs from standard WAMIT period sentinels and requires separate import
admission. Parsed finite A/B values and full cross-body matrices are retained.

Validated output is staged and closed first. The exact original bytes are saved
to `.before-reorder`, then GNU native rename publishes the staged table. Failures
exit nonzero; raw input and any incomplete staging/backup evidence remain.
Preexisting staging/backup paths refuse. These extra artifacts must be retained
by producer supervision. This is trusted producer I/O, not an executable sandbox
or protection against unrelated concurrent filesystem mutation.

Development evidence:36 actual standalone Fortran routine executions across the
initial and lexically repaired candidates. The original defect reproduces,
finite-only reordered output is byte-identical, and repaired mixed limit/finite
tables retain their complete inventory. Malformed fields, duplicate/missing
coefficients, merged native fields, no final newline, preexisting stage and an
injected rename failure are covered. GNU13.3 compiles and links the complete33-
source repaired HAMS candidate with the previously identified Netlib libraries.
The resulting producer has not been executed. No HAMS/BEM/mechanics solver call
or physical qualification ran in this repair slice. Independent Astra Read
review and all failures are saved in
`C:/ANYworkspaceWork/anymarine-evidence/hams-limit-reorder-20261010`.

The earlier source-derived finite coefficient transport and native supervision
now exist separately in ANYhydro. This new source repair still requires actual
limit acquisition, explicit corrected producer/profile identity, complete native
consistency and preserved Ainf provenance before time-domain use. Corrected
Windows availability, physical W5/W7 evidence, fields and sheets remain open.

### Subsequent actual acquisition and transport -- 2026-10-10

The later hams-limits-20261010 slice executed the exact limit-preserving GNU
candidate once and admitted a separate source-bound development transport
profile. Native0/A0,-1/Ainf and finite inventories are complete; finite arrays
are unchanged. Ainf is canonical with source/transformation provenance, while
incidental A0 is retained as unadmitted source evidence. Limit excitation NaNs
remain raw/unadmitted. See hams-backend.md and ANYmarine/TASK.md. Earlier statements
about unexecuted candidates describe their historical slices; scientific W5/W7
qualification and corrected Windows availability remain open.


## Prospective gravity-consistent candidate (2026-10-10)

The earlier profiles retain a known finite-depth inconsistency: DISPERSION uses
local9.807 while Const_mod and incident/finite-Green calculations use9.80665.
Source-bound incident elevation has factor9.807/9.80665. Relabeling or output
renormalization cannot repair the inconsistent finite Green inputs.

An additional explicit `--consistent-gravity` option requires both
`--gnu-compatible` and `--preserve-frequency-limits`. It verifies the pinned
DISPERSION routine and imports existing Const_mod.G, removes localG and its DATA
initialization, leaving the root algorithm and every other source unchanged
relative to the previous limit-preserving candidate. It writes a separately
identified `anyhydro-hams-gravity-source-candidate-v3` manifest. Older source
profiles and output remain intact. The unused FinGreen local gravity is untouched.

New aggregate source identity:
`077b0ec28d35339763cb2d4aaf079ee7da335919fe5f0486aa035faefd602a88`.
GNU13.3 executable:
`05e049f26f6ee023e84a35dc0943a52711bff8b0be0c1ab0a55e1f9b5a7210b7`.
This candidate remains explicitly **unadmitted** by HamsProducerBuild/backend
profiles. No Windows build, corrected limit acquisition or field API is claimed.

Evidence `C:/ANYworkspaceWork/anymarine-evidence/hams-fields-20261010` preserves
the original and corrected byte-identical-input acquisitions, all raw fields,
source/build/runtime hashes and numerical/ordinary ledgers. Actual extracted
DISPERSION/VINP checks at .2,.7,1.2rad/s,150m include1,200 roots and48 incident
values. Four initial normalized-residual failures are retained: high-order tan
argument/storage sensitivity requires the prospectively reviewed representation
allowance and separate smooth-phase oracle. A precision audit found Windows
longdouble was binary64, invalidating its high-precision claim. The corrected
acquisition preceded that intended prerequisite. A subsequent60-digit Decimal
oracle on exact binary64 inputs gives root distances at most2.325ULP; this is
retrospective evidence, not retroactive prerequisite satisfaction. An ordering-harness defect is separately preserved/repaired.
The pre-existing higher-order iteration formula was not changed.

The corrected .7rad/s two-body source emits16 radiation and8total field rows,
plus8incident rows at eight exterior surface/submerged points. Incident real,
imaginary, magnitude and phase match the analytic consistent-gravity model and
native float32/default-CMPLX/printing behavior; four surface magnitudes print1.
Full144radiation,12excitation,144supplied-restoring row inventories remain intact.
Radiation and excitation coefficients change; restoring bytes remain identical.
Exact differences and both failed/passed observations are saved, never rescaled.

The neutral ANYfileIO field syntax reader preserves these rows; canonical field
import, units/frame transport, validated sidecars/backend/Qt integration remain
the next connected slice. Total is incident+scattered under ISOL1, so adding
incident again is invalid. Field-point pressure is not panel-pressure capability.
Existing W5 power/PSD/finite-depth failures remain open and unexplained by this
source correction. Normalization evidence does not qualify scattering/radiation.


## Finite field transport profile adopted prospectively (2026-10-10)

The subsequent canonical transport slice now admits aggregate077b0ec... in
HamsProducerBuild, finite input preparation/import and the backend, explicitly
unqualified. Earlier unadmitted-status observations above describe their dated
source-prerequisite slice. This adoption does not erase its ordering gap, admit
corrected Ainf/zero transport, qualify fields, or claim a native Windows build.
See [HAMS point fields](hams-fields.md) for exact units, phase, point/refusal,
parent-frame, persistence and replay contracts. No new numerical solve was used.
