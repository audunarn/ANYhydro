# ANYhydro licence and provenance

## Current distribution

Original ANYhydro source code in the Apache transition release and later is offered under the Apache License, Version 2.0, included in the repository `LICENSE`. Project attribution is in `NOTICE`. The licence covers original implementations throughout the package, including Green-function correction, thin-sheet/dipole kernels, radiation processing, interfaces, and development utilities, not only Capytaine adapters. **Permissive licensing is an intentional decision for all owned code included in the release.**

This document is informational and is not a substitute for the actual licence text or an assessment of third-party rights. Confirm copyright ownership, employer rights, external contributions, copied code and source/data notices before the first Apache-licensed release.

## Historical releases

ANYhydro revisions published under MPL-2.0 retain their original licensing and grants. The change is not retroactive and does not remove existing MPL rights. A historical artifact should be interpreted under the licence included with that artifact; do not rewrite historical tags, wheels, or source archives.

## Other projects and data

- **Capytaine 3.x** is separately maintained by the Capytaine project and offered under Apache-2.0. It is an optional dependency, not ANYhydro-owned implementation. Consult its own licence and notices.
- **HAMS**, native executables, platform libraries, proprietary hydrodynamic tools, and optional file-format providers are separate software with separate rights. No HAMS executable/source is relicensed by this transition.
- **ANYsolver**, **ANYloads**, **ANYfileIO**, **ANYgeometry** and other ANY components retain their respective repository licences, including MPL-2.0 where applicable. An import does not confer their copyrights on ANYhydro.
- **Examples, imported reference results, original test artifacts and datasets** may carry separate attribution or external-data terms. Do not treat externally generated OrcaWave/WAMIT/HAMS outputs, standards text or third-party geometry as automatically Apache-licensed merely because a test refers to them.
- Project names, trademarks, branding, or endorsements are not licensed by Apache-2.0.

## Repository and release controls

1. Preserve authorship and licence notices from third-party code; do not remove them in a mechanical SPDX sweep.
2. For code owned and authored by the ANYhydro copyright holder, use `SPDX-License-Identifier: Apache-2.0` as appropriate. External material keeps its own notice.
3. Confirm the source archive and wheel include `LICENSE` and `NOTICE` (and any required third-party notices) before publication.
4. Release records must identify the first Apache revision/version explicitly and retain the last MPL revision/version.
5. No change to numerical mechanics, acceptance evidence or qualification follows merely from changing the licence.

## Strategic boundary

ANYhydro remains an open numerical-data and interoperability layer. This licence choice does **not** predetermine the licences or commercial strategy for ANYmarine, ANYfem, ANYworkbenchAI, or any independent proprietary workflows. Those products may consume ANYhydro subject to the applicable software licences.
