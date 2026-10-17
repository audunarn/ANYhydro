"""Source-study example CLI: prepare / execute / adjudicate.

Thin wrapper over the public module CLI
(``python -m anyhydro.source_study``). Development tooling only — no
scientific qualification is implied or claimed, and this slice carries
no BEM execution authority: ``execute`` requires the explicit
``--backend capytaine`` choice and constructs the real adapter under
case-derived explicit ``CapytaineExecutionLimits`` (the 100-panel
DEFAULT refuses the prospective refined levels; explicit limits admit
them up to the adapter's absolute software panel ceiling, docs/CAPYTAINE_ADAPTER.md). The CLI
is a gate: only the full
``source_admission_passed_development_only`` outcome exits 0;
no_go/failed/cancelled/incomplete outcomes exit with distinct nonzero
codes while reports and artifacts are kept.

Run (headless, from the repository root):

    python examples/source_study.py prepare --out <exclusive-new-dir>
    python examples/source_study.py adjudicate --out <dir>
"""

from __future__ import annotations

import sys

from anyhydro.source_study import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
