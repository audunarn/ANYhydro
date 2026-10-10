# Long-kernel checkpoint (explicit radiation execution limits)

Status: implemented and verified (development slice; no scientific
qualification). This checkpoint records the connected loop that admits
explicitly longer radiation kernels through the public builder while
preserving the default cap and all old hashes/behavior.

## Scope and ownership

- Owned files: `src/anyhydro/radiation.py`, `src/anyhydro/__init__.py`
  (exports), `tests/test_radiation_long_kernel.py`, this checkpoint and
  the radiation guidance in `README.md` / `docs/RADIATION_DEVELOPMENT.md`.
- No other source subsystem, Capytaine/backend/physics/mesh/operator
  changes. Public core remains Qt/ANYsolver/Capytaine-free at base import.

## Public surface

- `RadiationExecutionLimits` (frozen, keyword-only):
  - `max_time_samples`, default `1001` (= `MAX_TIME_SAMPLES`, unchanged),
    minimum `MIN_TIME_SAMPLES = 2`, hard ceiling
    `MAX_TIME_SAMPLES_CEILING = 20001`.
  - Typed refusals before any allocation: bool, fractional/non-integral,
    non-finite, wrong-type, unknown controls (`unknown execution-limit
    control(s)`), below-minimum and above-ceiling values.
  - `as_dict()` / `describe()` for provenance and persistence.
- `RadiationSettings.execution_limits` (optional, default `None`):
  - `None`: effective cap stays 1001; `to_dict()` keeps the exact old
    8-key serialized form (no new key), so all old settings hashes,
    artifact content hashes and stored artifacts remain valid.
  - Explicit: governs `n_time` for that construction only; recorded in
    the settings dict, the artifact content hash, `provenance`
    (`execution_limits`) and `assumptions`; no other validation,
    tolerance or tail/physics policy is loosened, and there is no fake
    padding or cropped history.
- `RadiationSettings.from_dict` / `load_radiation_kernel`: faithfully
  reconstruct explicit limits (unknown serialized controls refuse) and
  enforce the effective bound. Deleting the declared controls from a
  longer-than-1001 artifact is a typed refusal; artifacts with
  `n_time <= 1001` and no field keep the old identity. Unknown
  top-level keys in the serialized settings are a typed
  `RadiationSettingsError` raised before any settings/kernel
  allocation (fail-closed compatibility with the old
  `RadiationSettings(**settings_dict)` reader);
  `load_radiation_kernel` wraps them as `RadiationSerializationError`
  without mutating the artifact.

## Verification (exact runtime and harness, single sequential process)

- Pre-change reference hash of the default-settings oracle kernel:
  `8a40ee08c0568484bb78f666ab74d307bcdf87ecbc8d4e7f9c10daff448e208e`
  (computed before any source edit; evidence:
  `anymarine-evidence/long-kernel-worker-02`, run `pre_hash.py`).
- Post-change: `pytest tests/test_radiation_long_kernel.py
  tests/test_radiation.py tests/test_radiation_parent_review.py -q`
  -> 71 passed, exit 0 (evidence: `long-kernel-worker-02/run-d-all-radiation`).
  The focused file reproduces the pre-change hash exactly and covers:
  default-1002 refusal; explicit 1002/1003 builder success on the
  analytic exponential oracle; immutable arrays/limits; content hash
  changes with declared controls (4 distinct control sets -> 4 distinct
  hashes, deterministic repeat); NetCDF reload equality/hash/metadata and
  finite A+B reconstruction on the public path; wrong/missing/unknown/
  bool/fractional/non-finite/too-large/below-min limits typed refusal;
  deleted-controls refusal from a >1001 artifact; old-identity
  roundtrip for `<=1001` artifacts without the field.
- No execution beyond 1003 samples; ceiling 20001/20002 validated at
  construction only, before allocation. No global cap increase, no
  monkeypatching of limits, no BEM/Capytaine import.

## Reader fail-closed repair (worker-03, 2026-10-08)

- `RadiationSettings.from_dict` again refuses any top-level key outside
  the exact serialized fields (`SETTINGS_SERIALIZED_KEYS`: dt, n_time,
  symmetry_tol, passivity_tol, low_tail_policy, zero_frequency_damping,
  extension_name, time_window, execution_limits) with a typed
  `RadiationSettingsError` before settings/kernel allocation. This
  restores the fail-closed behavior of the old
  `RadiationSettings(**settings_dict)` reader that the selecting
  `from_dict` had silently dropped. No scientific, source-array or
  tolerance change; all old valid serialized/hash forms remain valid.
- Regression coverage through the public `from_dict` (unrecognized tail
  flag plus assorted unknown keys) and the public
  `load_radiation_kernel` (tampered artifact: typed
  `RadiationSerializationError`, sha256-identical artifact bytes after
  the refusal, untampered artifact still loads with its exact content
  hash).
- Verification (exact runtime and harness, evidence
  `anymarine-evidence/long-kernel-worker-03`):
  `pytest tests/test_radiation_long_kernel.py -q` -> 42 passed,
  exit 0, wall 2.624s (`run-a-longkernel`);
  `pytest tests/test_radiation.py -q` -> 30 passed, exit 0,
  wall 2.531s (`run-b-radiation-suite`). No BEM/Capytaine import, no
  execution beyond 1003 samples.

## Gaps / next steps (owned by the parent)

- Parent proves the actual ANYmarine longer-import path after installing
  both final wheels; this checkpoint closes only the ANYhydro dependency
  gap.
- Development fixtures and tests confer no scientific qualification.

## Parent installed cross-package verification — 2026-10-08

Final anyhydro and anymarine wheels in evidence/wheels/lid-longrun-v2 installed
offline and all canonical Python source SHA values matched installed source.
Actual public1003-sample radiation kernel, explicit limits and retained4 mapping
fed nonzero1002-frame hydro import/save/reopen/child/authoritative replay/save-as.
87 unique installed cases passed across preserved attempts. Final installed
long-settings roundtrip and unknown top-level typed refusal passed after the
reader repair; author42 focused+30 regression passes remain evidence.
ANYhydro total18.011518900/45s; shared parent29.503520500/30s. Full precise ledgers:
C:/ANYworkspaceWork/anymarine-evidence/lid-longrun-final-manifest.json.
This closes the parent dependency gap above; no BEM execution or scientific
qualification. Raw pilot reciprocal/PSD refusal and all historical failures kept.
