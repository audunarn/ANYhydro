# Final published restart point - 2026-10-10

GUI/publication handoff COMPLETE; broader numerical development remains PAUSED.
This is the current restart entry point; earlier sections are historical evidence.
Saved UTC 2026-10-10T20:11:30.683224+00:00. Source and CI baselines published on main:
- ANYmarine 77c308bdc875c43ad4065f2332a5d64fde23d5a1
  https://github.com/audunarn/ANYmarine/actions/runs/38079388189 - SUCCESS (Windows/Linux3.11/3.13).
- ANYhydro 4f10a55d8b45b57249b0aa8ca37c3b1cf1b44022
  https://github.com/audunarn/ANYhydro/actions/runs/38082088864 - SUCCESS (portable base CI Windows/Linux3.11/3.13).
A documentation-only commit records this final checkpoint. On resume verify the
latest main plus source identities; do not mistake older baseline hashes below
for the current branch head. Canonical roots C:/Github/ANYmarine and
C:/Github/ANYhydro; root coordinates Mistral directly, no Astra.

Latest GUI: C:/Github/ANYmarine/run_gui.py loads the bundled model on GPU startup;
Help > Development status exposes current capabilities and failed/unqualified
W5 evidence. Stored incident/disturbed wave display remains available. Restart
an old app process to see the installed updates. Tested runtime:
C:/ANYworkspaceWork/anymarine-evidence/runtime/Scripts/python.exe.
The installed marine wheel includes the Linux symlink-root safety repair;
the installed hydro wheel matches the final process-lifecycle repair.
Actual GPU first paint/status/waves,39 status tests,22 startup/menu tests,
43 wave regressions and original54 headless passes/1 Windows symlink skip are
preserved evidence. Root-link repair: real Linux filesystem probe passes;
affected persistence/startup29 pass/1 platform skip; GitHub clean matrix SUCCESS.
Hydro portable selection:34 modules,796 local passes/18 Linux-only skips;
GitHub verifies the actual Linux legs too.15 dependency/artifact exclusions are
explicit in ANYhydro docs/PORTABLE_CI.md, with remaining evidence obligations.
No scientific test or tolerance was weakened and no native campaign rerun.
Numerical/hash/fixture production algorithms remain unchanged. The prospective
platform CI repair changes Linux process supervision, its two lifecycle test
modules and legacy-oracle tests (7 of101 protected
files changed;94 unchanged). Exact before/after
identities are recorded in PLATFORM_CI_REPAIR.md.

W5 remains NO_GO/general numerical admission false. The terminal one-shot
energy-point-02 failed added-mass/damping/mean-power reference comparisons.
The100/100 independent stored-array audit reproduces those failures and is
consistency evidence, not physical acceptance. Audit final handoff is incomplete;
no completed extra scientific diagnostic is recorded. Full roadmap incomplete.

FIRST ACTION ON EXPLICIT RESUME: verify current guidance/source/evidence/dirty
ownership and assign Mistral a bounded stored-array physical-case/reference,
auxiliary-lid/projection and energy-accounting diagnosis. Choose a discriminating
check before a prospective new execution decision. Do not restart consumed
energy-point-02, flip signs, clamp matrices, loosen tolerances or assume a cause.
Keep all remaining scientific, optional-backend, desktop and thin-sheet gates.

All owned workers terminal and execution pools frozen. Native one-shot consumed;
no old remaining seconds revived. The CI worker omitted exploratory checks from
its nominal tally: recovered accounting conservatively exceeds360s (about372s;
exact process wall missing for two ad-hoc calls). Gap/overrun retained without
retrospective allowance. Separate prospective platform-repair checks are
recorded in PLATFORM_CI_REPAIR.md; every pool is now frozen. See also
docs/CLEAN_CI_HANDOFF_20261010.md.

Published compact evidence: ANYhydro docs/development-status/; measured GUI note
docs/GUI_MAIN_HANDOFF_20261010.md and clean-CI note in both repos. Full failed
attempts/raw arrays/receipts preserved locally under their historical paths;
current delivery evidence C:/ANYworkspaceWork/anymarine-evidence/
gui-main-handoff-20261010/, gui-ci-repair-20261010/ and
gui-ci-platform-repair-20261010/ and gui-ci-final-tail-repair-20261010/.
No raw evidence/caches
deleted or published. MPL-2.0 retained; exact Git source bytes preserved.

---

## GUI publication complete; numerical development paused - 2026-10-10

Definitive restart: C:/Github/ANYmarine/CHECKPOINT.md. Published source baselines
ANYmarine 8e97dbb97e1c85f35d25134fd883ea3197aae27f, ANYhydro 1bb3b82cd94ad3229156dc3da51b56da410396d6 on main;
remote identities verified. Installed GPU startup/status/saved waves pass.
Status39, startup22, waves43, foundation54 pass/1 skip; hydro77 pass.
W5 remains NO_GO, native energy-point-02 consumed/failed/unqualified and audit
100/100 consistency only. No new scientific run. Historical limits preserved;
new GUI pools frozen at handoff. Root coordinates Mistral; no Astra.
Measured public note docs/GUI_MAIN_HANDOFF_20261010.md; full local evidence
C:/ANYworkspaceWork/anymarine-evidence/gui-main-handoff-20261010/.
Resume first with bounded stored-array case/energy diagnosis, not a native retry.
Saved UTC 2026-10-10T19:05:39.458648+00:00. Historical notes below retained verbatim.

# W2 checkpoint (final, 2026-10-08)

Date: 2026-10-08. Owner: Mistral worker (W1+W2 repository ownership,
excluding AGENTS.md, .github/workflows/ci.yml, LICENSE).

## Actual state (all verified by execution this session)

- `src/anyhydro/capytaine_backend.py` — development Capytaine 3.0 adapter,
  complete: combined `cpt.Multibody` solve with full 6N coupled matrices,
  buoyancy-only block-assembled hydrostatics, explicit omega=inf
  `added_mass_infinite` with provenance, typed refusals (zero frequency,
  unverified normals, non-wetted geometry, budgets, unknown settings,
  missing optional library), content-addressed validated NetCDF cache,
  caller immutability.
- Typo fixed before published use: `CAPYTAINDE_ADAPTER_VERSION` renamed to
  `CAPYTAINE_ADAPTER_VERSION` (5 occurrences; the published value "0.1.0"
  is unchanged, so cache identity content is unaffected).
- `tests/test_capytaine_backend.py` — 29 tests, all passing on the
  Capytaine-enabled runtime, including a direct `cpt.Multibody` A/B/F
  comparison with identical panels (adapter-mapping check, not scientific
  qualification), cache hit/invalidation/corruption, NetCDF cache
  round-trip, refusal paths, and input immutability.
- `examples/headless_two_body.py` — runnable, saves/loads/replays a .nc;
  leftover dead expression removed.
- `pyproject.toml` — build requirement bumped to `setuptools>=77` for the
  SPDX `license = "MPL-2.0"` string; LICENSE file present (parent-supplied).
- `README.md` — Capytaine adapter invocation and enforced capability limits
  documented; status section updated.
- `docs/CAPYTAINE_ADAPTER.md` — accurate adapter contract (verified against
  the module; no changes needed).

## Evidence (this session, unique .evidence dirs, bounded runner TEMP/TMP)

- `2026-10-08-adapter-tests-a`: adapter tests, Capytaine runtime, exit 0,
  12.5 s, 29 passed.
- `2026-10-08-pytest-full-cpt-b`: full suite, Capytaine runtime, exit 0,
  13.1 s, 158 passed.
- `2026-10-08-pytest-full-base-c`: full suite, base runtime (no capytaine),
  exit 0, 8.5 s, 129 passed, 1 skipped (adapter module via importorskip —
  base CI works without the backend).
- `2026-10-08-example-headless-a`: headless example, exit 0, 8.2 s, .nc
  saved, reloaded hash-stable, replay backend round trip OK.

External `anymarine-evidence/test_hydro_review.py` NOT rerun: it does not
touch the adapter or any file changed this session (last evidence
`2026-10-08-independent-suite-c`: 5 passed).

## Actual next step

Parent: build/install wheels (setuptools>=77 now declared) and exercise the
public paths. Remaining W2 acceptance gaps are listed in HANDOFF.md
(mesh convergence, passivity, independent hydrostatic qualification,
disabled post-processing capabilities, single-geometry mapping check).
