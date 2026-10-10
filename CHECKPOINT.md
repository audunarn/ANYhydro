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
