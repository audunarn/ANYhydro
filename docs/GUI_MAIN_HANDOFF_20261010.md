# GUI/main development handoff validation - 2026-10-10

Current source and bundled examples are a development snapshot, not an engineering
release. W5 remains NO_GO and general numerical admission false. No native
scientific calculation or retry was run for this publication. Broader numerical
development remains paused at the user's request.

The GUI now exposes Help > Development status and a permanent startup status
line. It lists the current application routes and reports the failed hydrodynamic
reference comparison separately from the completed 100/100 stored-array audit.
Opening the status performs no calculation and does not alter project results.
The snapshot date is 2026-10-10; corrupt data fails clearly.

Measured checks in the isolated Windows Python3.13 runtime:
- Status loader and Qt integration:39 passed after provenance/corrupt-data repairs.
- Startup/example/menu regressions:22 passed.
- Wave-loading/replay regressions:43 passed.
- ANYmarine published headless CI selection:54 passed,1 skipped; the symlink refusal test
  is skipped because Windows symlinks are unavailable (not counted as a pass).
- ANYhydro records/schema/transforms:64 passed; persistence13 passed after a
  shared-temp-directory permission error was repaired in the check harness.
- Both wheels built and installed with no dependency downloads.69 ANYmarine
  and37 ANYhydro installed package files matched wheel and source bytes exactly.
- Actual Windows GPU desktop: startup shows the two-body model on first paint;
  Help status shows19 capability rows; stored incident/disturbed waves render;
  no job is launched. Offscreen checks do not replace this observed desktop path.

Wheel identities:
- ANYmarine SHA256 f22ccc74ce89f224e58e9fd5024682983b1c7e03845eed034492c2bcc34d3554
- ANYhydro SHA256 6de48afb55e559988a1e9dcbc0be30a341ef9a023c2ded91138c9bc4268485ec

Full command/input-hash/runtime/wall/exit receipts, failed check attempts,
screenshots and Mistral handoff are preserved locally in
C:/ANYworkspaceWork/anymarine-evidence/gui-main-handoff-20261010/.
The initial desktop observer used the viewer's height property as a callable;
the corrected observer uses the public size API. That failed observer is
preserved and does not establish an application failure.
New check-budget accounting was reconciled from unique terminal receipts after
overlapping ledger writes; no test was rerun to reconstruct accounting.

See ANYmarine docs/DEVELOPMENT_STATUS.md and ANYhydro
docs/development-status/energy-point-02-snapshot.json for the portable status
and compact source evidence. Raw local evidence/caches are preserved outside Git.
No qualified-source admission, physical validation, full GUI qualification,
Linux desktop result or release approval is claimed by this source publication.
