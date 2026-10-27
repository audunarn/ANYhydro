# ANYhydro

Neutral frequency-domain hydrodynamic data model, validation, reference-point
transformations and persistence for the ANY marine ecosystem.

ANYhydro owns:

- immutable hydrodynamic panel mesh, body and request records;
- body-qualified DOF identity (`body_id::surge`, ..., `body_id::yaw`);
- the canonical hydrodynamic `xarray.Dataset` schema (full coupled `6N`-DOF
  matrices, complex excitation, explicit conventions and provenance);
- reference-point translation maps for forces and matrices;
- deterministic content hashing and lossless NetCDF persistence;
- an imported-dataset backend that replays validated stored results.

ANYhydro does **not** own dry mass, center of gravity or inertia: mechanical
authority stays in ANYsolver. Physical mass never enters this package.

## Status

Development slice (roadmap Waves 1-2). The synthetic two-body fixture is a
development fixture and confers no scientific qualification. Optional
Capytaine and Qt are never imported at base import time.

A development Capytaine 3.0 BEM adapter (`anyhydro.CapytaineBackend`) is
included: it fills the canonical dataset with full coupled 6N matrices from
one combined multibody solve. It is development evidence only — no
scientific qualification is implied or claimed.

## Install

```bash
pip install .            # base: numpy, scipy, xarray
pip install .[dev]       # + pytest, build
pip install .[capytaine] # optional BEM backend dependency (>=3.0,<3.1)
```

## Capytaine adapter: invocation and capability limits

```python
import numpy as np
from anyhydro import CapytaineBackend, HydroRequest

backend = CapytaineBackend()
dataset = backend.solve(bodies, request)   # bodies: sequence of HydroBody
```

or run the headless example, which solves two tiny submerged boxes, saves
the canonical dataset to NetCDF, reloads it and replays it through the
imported-dataset backend:

```bash
python examples/headless_two_body.py --out two_body.nc
```

Enforced development limits (typed `CapabilityError` beyond them):

- explicit frozen `CapytaineExecutionLimits` passed to
  `CapytaineBackend(execution_limits=...)`; defaults: <= 100 panels total
  (hull + lid), <= 3 finite frequencies, <= 1 heading,
  `n_jobs=n_threads=1`. Explicit limits may admit up to 2048 total panels
  (absolute software ceiling; the registered 384/864-panel source-study
  meshes need explicit limits) and up to 513 finite
  frequencies for later registered cases; the heading ceiling stays at
  1 and jobs/threads are pinned to 1. Bool, fractional, non-finite,
  negative, zero and unknown limit controls are refused at construction,
  before any Capytaine import or cache lookup; all caps apply before cache
  lookup and the effective limits are recorded in diagnostics, cache
  identity and dataset provenance;
- zero-frequency limit: typed refusal (the schema stores strictly positive
  finite frequencies only);
- unverified outward normals, non-wetted geometry and lids whose normals
  do not point DOWN (the pinned Capytaine lid convention): typed refusal,
  never silently flipped or clipped;
- unknown backend settings: refused (supported: `method` = `direct` (default since adapter 0.6.0) or `indirect`, `cache_dir`, `auxiliary_lid_policy`).

The deterministic first-slice two-body wetted-hull/lid fixture (96 panels,
analytically verified geometry) lives in `anyhydro.first_slice`; the
single real-backend pilot is `examples/first_slice_bem_check.py`.

`hydrostatic_stiffness` is linear buoyancy-only (Capytaine
`compute_hydrostatic_stiffness` with `mass=0`, block-assembled per body);
the gravity restoring of the dry body is NOT included and is owned
separately by the solver. `added_mass_infinite` (opt-in via
`include_infinite_frequency=True`) comes from explicit per-radiating-DOF
`omega=inf` limit solves with recorded provenance. Panel pressure, Kochin,
free-surface and mean-drift capabilities are disabled pending their own
tests. See `docs/CAPYTAINE_ADAPTER.md` for the full adapter contract.

## Quick example

```python
import anyhydro

bodies = anyhydro.fixtures.make_fixture_bodies()
request = anyhydro.fixtures.make_fixture_request()
ds = anyhydro.fixtures.make_two_body_fixture()

anyhydro.schema.validate_hydro_dataset(ds)
anyhydro.schema.validate_for_time_domain(ds)  # requires added_mass_infinite

anyhydro.persistence.save_hydro_dataset(ds, "fixture.nc")
ds2 = anyhydro.persistence.load_hydro_dataset("fixture.nc")

shifted = anyhydro.transform.shift_dataset_reference_point(
    ds, {"fixture-body-alpha": (1.0, 0.0, -0.5)}
)
```

## Radiation preprocessing (development slice)

`anyhydro.radiation` turns a validated neutral dataset with an explicit
`added_mass_infinite` (non-empty source; the last finite-frequency sample is
never substituted) into the full-coupled causal radiation kernel for the
`Re{X exp(-i*omega*t)}` convention, reconstructs BOTH added mass and
damping, and compares them against the reference under caller-declared
per-coefficient criteria:

```python
from anyhydro.radiation import (
    RadiationSettings, ReconstructionCriteria,
    radiation_kernel_from_dataset, compare_reconstruction,
    save_radiation_kernel, load_radiation_kernel,
)

settings = RadiationSettings(
    dt=0.03, n_time=1001,
    symmetry_tol=1e-9, passivity_tol=1e-9,
    low_tail_policy="explicit_zero_sample",       # requires B(0) below
    zero_frequency_damping=B_at_zero,             # explicit zero-frequency sample
)
kernel = radiation_kernel_from_dataset(dataset, settings)   # validate_for_time_domain inside
a_rec, b_rec = kernel.reconstruct(omegas)                  # A_inf - int K sin(wt)dt/w, int K cos(wt)dt

criteria = ReconstructionCriteria(
    omega_min=0.1, omega_max=8.0,
    added_mass_abs_tol=2.0, added_mass_rel_tol=1e-2,       # separate units
    damping_abs_tol=1.0, damping_rel_tol=1e-2,
)
report = compare_reconstruction(kernel, dataset, criteria)  # BOTH matrices must pass

save_radiation_kernel(kernel, "kernel.nc")   # deterministic, immutable artifact
kernel2 = load_radiation_kernel("kernel.nc")  # content-hash verified
```

Explicit longer kernels: the default cap stays 1001 time samples. A
caller may explicitly declare a larger per-construction cap through
frozen public `RadiationExecutionLimits` (default `max_time_samples=1001`,
minimum 2, hard ceiling 20001; bool/fractional/non-finite/unknown
controls are typed refusals before any allocation):

```python
from anyhydro import RadiationExecutionLimits

settings = RadiationSettings(
    dt=0.03, n_time=1003,                       # longer than the default cap
    execution_limits=RadiationExecutionLimits(max_time_samples=1003),
    ...                                         # all other controls unchanged
)
```

`execution_limits=None` (the default) preserves the exact old serialized
settings, hashes and behavior. Explicit limits are recorded in the
settings dict, the artifact content hash, provenance and assumptions, and
are faithfully reconstructed and enforced on NetCDF reload: deleting the
declared controls from a longer-than-1001 artifact is a typed refusal.
Limits are execution bounds only — no scientific qualification, no
loosened tolerances, no fake padding or cropped history.

Enforced development limits (typed `RadiationLimitError`): <= 513 input
frequencies, <= 1001 time samples (explicitly extendable per construction
via `RadiationExecutionLimits`, hard ceiling 20001), <= 12 DOFs. The
kernel is computed by
deterministic trapezoidal quadrature; the high tail is explicitly truncated
at `omega_max` (diagnosed, never extrapolated or fitted); the default
low-tail policy requires an explicit zero-frequency damping sample, with an
optional separately named linear-to-zero `B(0)=0` extension recorded as an
assumption. Symmetry/passivity violations are diagnosed with explicit
tolerances and preserved, never clamped. Kernel artifacts are always
`unqualified`: they cannot inherit a qualified status from the source
dataset. Run the analytic coupled example with
`python examples/radiation_check.py`. See
`docs/RADIATION_DEVELOPMENT.md` for the full contract mapping.

## Source-quality mesh study (development tooling)

`anyhydro.source_study` registers the first-slice source-quality mesh
study as an immutable closed-world case record with parametric
deterministic mesh refinement (`anyhydro.first_slice` public keyword
`subdivisions=n`, default 2 bit-identical to the historical fixture),
and provides the complete public workflow:

```bash
python -m anyhydro.source_study prepare --out <new-dir>   # no backend import
python -m anyhydro.source_study execute --out <dir> --backend capytaine
python -m anyhydro.source_study adjudicate --out <dir>    # no fresh BEM
```

Prepare writes the hash-bound case and full geometry/request inventory
with a `not_executed` status. Execute checks caps/body
identity/version BEFORE any backend work, performs at most ONE solve
per registered level, persists the RECEIVED raw source BEFORE identity
admission (schema-valid sources through the public save route under the
same content-addressed name; schema-invalid ones as a bounded, clearly
labelled rejected/received forensic archive — never an admitted neutral
dataset, with the original typed failure re-raised unchanged), admits
each raw dataset fail-closed, saves
content-addressed raw and derived artifacts (non-admitted ones too)
and persists complete per-level and cross-level refinement reports.
Load/adjudicate reproduce every report through the public full
raw-source proof — a matching hash is necessary but never sufficient.
The case format is honestly `anyhydro-source-study-v2` (content-hash
prefix `anyhydro-source-study-case-v2`); v1 records are explicitly
unsupported by the v2 decoder — typed refusal, no automatic migration.

The prospective development case (levels 4 and 6 = 384/864 panels,
omegas [2, 4, 6], heading 0, explicit A_inf, translation-length-1 m
normalization, explicit reciprocal-average) is a parent-owned candidate
frozen before any future BEM. Every outcome is
`development_source_study_only`: nothing confers scientific
qualification, and this slice carries no BEM execution authority. See
`docs/SOURCE_STUDY.md` for the full contract.

## Documentation

- `docs/CONVENTIONS.md` — canonical conventions and invariants.
- `docs/HYDRO_DATA_SCHEMA.md` — dataset schema, persistence format and hashing.
- `docs/CAPYTAINE_ADAPTER.md` — Capytaine development adapter contract.
- `docs/FINITE_DEPTH_PRONY.md` — validated finite-depth Prony decomposition
  used by the adapter, and the W5 free-surface attribution behind it.
- `docs/RADIATION_DEVELOPMENT.md` — radiation preprocessing development contract.
- `docs/DERIVED_OPERATORS.md` — derived retained-DOF operator artifacts and
  retained-coordinate radiation (development slice).
- `docs/SOURCE_STUDY.md` — first-slice source-quality mesh study contract
  (development tooling).

## License

MPL-2.0, matching the ANY ecosystem.


### Development free-surface preview

`anyhydro.wave_preview` exposes neutral `incident_field`, `validate_field` and
`positions_at_time` records with SI/WORLD geometry and exp(-i omega t) phasors.
Incident fields consume public ANYloads Airy definitions; importing the module
does not import Qt, ANYsolver or Capytaine. `compute_diffraction_preview` lazily
uses exact Capytaine3.0.0 for one fixed-body diffraction problem, with at most
two bodies,48 immersed panels and1024 surface points. This optional development
service does not change the coefficient backend's capability declarations.

Output records bind captured project/geometry/wave hashes, grid, Capytaine
identity and explicit Green-function settings. The direct Delhommeau option
disables tabulation/cache and retains the1001-point integration rule. Elevation
is incident plus scattered phasors; no moving-body radiation or nonlinear
water entry is represented. The first real attempt timed out during default
tabulation startup; revised direct-integration diffraction remains unverified.
Transport tests use declared fixture arrays and confer no scientific acceptance.

Development synthesis from shared saved ANYloads realizations is available through the optional sea-state extra: [contract and examples](docs/SEA_STATE.md). This preserves source status and does not confer source admission.

Pinned radiation reference preparation and strict coupled output parsing: [HAMS reference tooling](docs/hams-reference.md). Development only; no solver is launched and W7 is not accepted.

## Licensing

From the Apache transition release onward, original ANYhydro source is distributed under the Apache License, Version 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)). Previously published MPL-2.0 revisions retain the rights granted with those revisions; the licence change is prospective and does not revoke earlier permissions.

This is a separate project from Capytaine. Capytaine 3.x is an optional dependency independently licensed by its maintainers under Apache-2.0, not source code owned by ANYhydro. Other optional dependencies, third-party sources, data, and native executables retain their own terms. See [licensing and provenance](docs/LICENSING.md).

The Apache-2.0 grant covers the original ANYhydro code, including optional in-house numerical methods (such as thin-sheet panels and Green-function corrections), not only the neutral adapter. It does not grant rights to third-party software, reference data or trademarks.
