# W5 precision candidate review (2026-10-10)

Worker: Mistral Edit (continuation of the w5-precision-candidate-20261010
session; the initial 360 s launcher timed out on inference budget, not a
scientific failure). Ownership per
`C:/ANYworkspaceWork/anymarine-evidence/w5-precision-candidate-20261010/edit.ps1`:
`tools/native_precision`, this review, and new evidence under
`.development-evidence/w5-precision-candidate-20261010`. Upstream bundle,
root check.py, manifests/pools (except runner updates), old evidence,
runtime, siblings and Git untouched; no commit/push, no nested workers.

## What was delivered

### 1. Source precision audit (conservative inventory, not runtime proof)

`tools/native_precision/fortran_precision_audit.py` v1.0.0 over all 21 pinned
upstream/libs Fortran files (attempt `w5prec-audit-01`; full machine-readable
report `results/precision-audit-report.json`, summary
`results/precision-audit-summary.txt`):

- 519 findings: 303 unsuffixed real literals, 158 default-kind expressions,
  29 default REAL declarations, 26 REAL intrinsics without KIND, 3 CMPLX
  intrinsics without KIND.
- 141 material unsuffixed literals with float32 rounding recorded.
- Path classification: 350 "active" (newer/low_freq/scaled_nemoh3 paths),
  56 alternative (FinGreen3D/LiangWuNoblesse), 113 unknown auxiliary.

**Explicit scope statement.** The classification is a conservative FILE
INVENTORY by path, NOT proof of runtime calls. In particular, frozen
`old_Prony_decomposition.f90` (137 findings) is not currently invoked by the
active path, and `float32.f90` is not the active float64 configuration.
"active: 350" is therefore not "350 measured active hazards", and the 141
material literals are an inventory of float32-rounded constants, NOT 141
proven physics errors. No coefficient was changed on the basis of this
inventory.

### 2. Minimal candidate (PI/EULER_GAMMA only)

`tools/native_precision/build_candidate_patch.py` v1.0.0 (attempts
`w5prec-candidate-01..04`): copy of upstream `constants.f90` with exactly two
changed lines (1-based lines 26 and 28, confirmed correct against a
0-vs-1 indexing bug found by unittest `w5prec-tests-01`, now fixed in the
test):

- `PI = 3.141592653589793238462643383279502884_PRE`
- `EULER_GAMMA = 0.577215664901532860606512090082402431_PRE`

Apache-2.0 header preserved verbatim; public interface (module CONSTANTS,
18 parameters, IS_INFINITY) unchanged; parent/candidate/patch sha256 in
`candidate/patch-manifest.json` (parent
df6c9ae10f8f5c60fd397c5fe691c8ac527e971081d6d474138862eaea71cd96, candidate
5b8233205963e84710629ee9395ccb7e78ddb138bca411cfe8defb481b6b66bf, patch
b06e2caa70f124d4693da3b40f316d30e31ad5287d31801783e08e7ed1a16d6f).
Upstream original immutable; refusal/idempotency/immutability guaranteed and
tested. `MINUS_ONE_OVER_FOURPI` auto-corrects as a derived expression.

### 3. ONE source-only compiler probe (executed once)

`tools/native_precision/probe_constants_source.py` v2.0.1. Five pre-execution
repairs were applied and mock-tested BEFORE the single launch: direct
subprocess argv (no JSON shell escaping of the bash script; shlex.quoted
paths), explicit VARIANT markers with strict exactly-two-block parsing and
numeric kind/value validation (nonzero exit on missing/malformed/nonfinite/
wrong values), temp/source/upstream/output overlap and nonempty-output
refusals with source hashes verified before/after, one-thread env exported
inside the WSL shell, and full receipts (exact argv array, compiler version,
source/driver/precision/executable identities).

Actual probe: attempt `w5prec-probe-02` (attempt `w5prec-probe-01` was a
pre-execution path refusal; no compiler was launched there). GNU Fortran
(Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0 in WSL Ubuntu-24.04 under GNU
`timeout --kill-after=3s 100s`, wall 4.21 s, one thread. Receipt:
`results/constants-source-probe-receipt.json`.

| quantity | baseline | candidate |
|---|---|---|
| PI | 3.1415927410125732 (float32 rounding of literal) | 3.141592653589793 (= math.pi, diff 0.0) |
| EULER_GAMMA | 0.5772156715393066 | 0.5772156649015329 (diff 0.0 vs reference) |
| kinds (PI/GAMMA/PRE) | 8/8/8 | 8/8/8 |
| executable sha256 | 9f978cf9100b4a1a1ffaa722ff041ec511148d69ed61b9833b3f07fb93701d35 | 064d19a5da3b041bf66d56fc8f54909192f4c357f8a2cff6a2c54dedc27468a7 |

Sources unchanged before/after (hashes match pinned identities). The
executable hashes are in the receipt stdout (sha256sum lines) and were
re-hashed from the preserved artifacts into
`results/constants-source-probe-executable-hashes.json`; the receipt's parsed
`executable_sha256` field is null because the probe ran with tool v2.0.0 whose
parser missed the `./` prefix (fixed in v2.0.1, covered by mocks; no second
compiler probe was launched).

**Explicit scope statement.** This standalone GNU constant probe proves
source-kind correction of the two named constants only. It does NOT qualify
ABI, a full backend build, Green-function physics, the historical energy/PSD
failure (cause unestablished), or any numerical criterion. The failed ideal
solidangle criterion (1.3907937745916144e-8 > 5.097905158670353e-9) is KEPT;
no tolerance weakening, no empirical rescaling, no compiler default-real
promotion, no output rescaling.

### 4. Tests

73 unittests (stdlib only, no compiler in tests; run attempts
`w5prec-tests-01` (50 tests, 49 pass / 1 fail, preserved),
`w5prec-tests-02` (caught a real main() KeyError, fixed),
`w5prec-tests-03`, `w5prec-tests-04`: 73/73 pass). Coverage: lexical
(comment/string/integer ignoring), parser, refusal (bounds, missing sources,
temp/source/upstream/output overlap, nonempty temp, existing receipt),
idempotent patch, hashes, source immutability, upstream identity pinning,
public interface unchanged, strict probe parsing/validation via mocks.

## Proposed distinct backend build recipe (NOT executed; prerequisites missing)

Adoption of the candidate into any native backend requires, at minimum:

1. A distinctly identified backend target (name/version/SHA), separate from
   the failed historical bundles (which are preserved, not reused).
2. Full packaging/templates for the pinned capytaine-3.0.0 libs — NOT present
   in the pinned bundle; exact missing prerequisites: upstream packaging
   metadata, build templates, and a Windows-side Fortran compiler (none found;
   only WSL gfortran 13.3.0 exists).
3. Fresh producer/table/reference admission: no cached table reuse, fresh
   tabulation producer identity, and reference admission per repo contract.
4. New native SHA recorded; current runtime NOT mutated.
5. Re-run of the ordinary criteria (including the currently failed ideal
   solidangle criterion) against the newly built backend before any
   qualification claim.

## Reserved decisions (not adopted by this worker)

- All other audited hazards (further unsuffixed literals, default-kind
  intrinsics/declarations in active paths) — listed in the audit report,
  unchanged.
- Any external ABI/public behavior/numerical model change beyond the two
  named constants.
- Backend build recipe execution and any admission decisions.

## Budget/attempt accounting

Ordinary pool 180 s, ~6.2 s used cumulative (~173.8 s remaining), not
renewed. All checks through root `check.py` with the exact named runtime
`C:/ANYworkspaceWork/anymarine-evidence/runtime/Scripts/python.exe`, unique
attempt names, one job/thread, max 120 s/process, WSL child GNU timeout
<= 100 s kill-after 3 s. No unaccounted experiments; no numerical authority
consumed.
