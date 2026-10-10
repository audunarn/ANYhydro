# Corrected HAMS native supervision (development)

The gravity-consistent source has a separate finite-only field route; see
[point fields](hams-fields.md). Explicit `field_points` are immutable run inputs.
Requested fields must pass import, persistence/reopen and saved-run semantic
verification before completion. `load_completed_hams_fields` returns the stored
artifact. Current source-specific Ainf admission remains limited to the older
limit-corrected source; the gravity-consistent source refuses it.

`anyhydro.hams_backend.HamsBackend` implements the shared headless backend
protocol. It executes one explicitly configured corrected HAMS build and returns
a validated canonical dataset. This is an **unqualified finite-frequency rigid
six-DOF route**, preserving all coupled body blocks. Ordinary lifecycle fixtures
copy previously stored output; they do not establish numerical accuracy or
availability of a corrected Windows producer.

```python
from anyhydro.hams_producer import HamsProducerBuild
from anyhydro.hams_backend import HamsBackend, HamsExecutionLimits

build = HamsProducerBuild(
    executable_path=corrected_executable,
    executable_sha256=measured_executable_sha256,
    source_tree_sha256=verified_corrected_compiled_source_sha256,
    platform="windows",  # or "linux", matching the actual executable/runtime
    build_description=recorded_compiler_flags_and_libraries,
)
backend = HamsBackend(
    build, mechanical_records, world_centers_of_buoyancy,
    workroot=case_directory,
    limits=HamsExecutionLimits(wall_seconds=60, threads=1),
    should_cancel=cancellation_callback,
    irregular_frequency_removal=True,
)
dataset = backend.solve(bodies, request)
```

The executable hash, pinned source commit, admitted corrected source-tree profile,
native platform, version and build description identify the producer. They are
explicit declarations, not independent build attestation or qualification.
`build.verify()` verifies actual executable bytes/header without executing.
`backend.diagnose()` additionally checks native platform availability. Imports
require no Qt, ANYsolver or Capytaine; constructing the backend requires the
optional ANYfileIO HAMS syntax interface already used by input preparation.

The original `prepare_hams_deck` default retains its published-executable pin.
An explicit `producer_build=build` creates a separately versioned corrected deck.
Unknown profiles, changed executable/input bytes and inconsistent declarations
refuse. Merely preparing or verifying a deck never starts a producer. There is
no public arbitrary command-prefix, shell or WSL launch interface. A Linux GNU
build cannot run directly in the Windows application.

Each solve captures body/request arrays and configuration, creates a unique
case and exclusively claims `run.json`. A backend instance rejects overlapping
or reentrant calls. Multiple instances need application-level scheduling.
Effective thread environment settings are recorded, along with the exact launch
command, identities, PID, monotonic elapsed time and failures. Production launches
only the bound executable with `Input Output`; test-only private overrides run
Python fixtures. This supervision is process ownership, not a sandbox for an
untrusted executable. The application must supply an authorized, trusted build.

Receipts progress through queued/running/cancelling/cancelled/failed/completed.
Admission failures before exclusive claim create no job receipt. Cancellation
and deadlines are checked before launch, while polling, during import/persistence
and immediately before the completion seal. Failed, cancelled or incomplete
cases retain logs and raw outputs. Cleanup/receipt errors preserve the primary
failure. A zero process exit alone is insufficient: complete native coefficients,
strict canonical import, input identity, save/reopen identity and raw artifact
hashes must agree. `completed.json` is written last as the completion commit;
later cancellation is handled by the consumer's own admission rules. No result
is returned without a seal. Reproduction needs a fresh directory; rerun and
preexisting publication files, links/junctions/nonregular artifacts refuse.

`load_completed_hams_run(directory, bodies=..., request=..., producer_build=...)`
revalidates the input deck, producer, receipt, seal, raw inventory and stored
dataset. Stale requests/bodies, changed data or missing sidecars refuse. Hashes
bind the captured artifacts; they do not make an unsigned receipt independent
scientific evidence.

Windows 10+ uses atomic process-creation JobList assignment, a noninherited
kill-on-close Job Object, and a whitelist of standard I/O handles. The producer
belongs to its job before its first thread runs; supervisor termination closes
the job and terminates its descendants. No suspended-but-unowned fallback is
used. See Microsoft's [atomic JobList explanation](https://devblogs.microsoft.com/oldnewthing/20230209-00/?p=107812/),
[attribute-list API](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-updateprocthreadattribute)
and [Job Object lifecycle](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).
Linux launches a new session and terminates its owned process group, without
thread-unsafe pre-execution callbacks; see [Python subprocess documentation](https://docs.python.org/3/library/subprocess.html).
Linux supervisor-death containment and Qt worker admission remain separate,
unfinished integration work. Native HAMS must not be wired into the current Qt
worker until its permitted-child contract and closure/cancellation evidence are
updated together.

Zero/infinite-frequency limits, infinite-frequency added mass, pressure/wave
fields, thin sheets, symmetry and generalized modes remain unsupported. Finite
depths above 500 m and frequencies entering native limit substitutions refuse.
Supplied mechanical restoring remains UNKNOWN physical hydrostatic ownership;
composition and time-domain admission require the existing scientific contracts.
W5 physical NO_GO evidence and all W7 qualification obligations remain unchanged.

SPDX-License-Identifier: MPL-2.0

## Explicit limit-preserving source profile (development)

The exact GNU phase/limit source identity
`a2608dc7bbb56176917b1fc13804e77412ef5aefce26740a8366556217ad5940`
admits `hams_mrel_cf9583c_phase_limit_corrected_omega_v2`. Older phase-only
identities retain finite-only behavior. Supply `include_infinite_frequency=True`
in HydroRequest and the explicit matching HamsProducerBuild; no automatic build
selection or platform conversion is provided. Preparation enables both native
limit branches. HamsExecutionLimits.max_frequencies counts finite samples plus
both mandatory limit calculations. The backend advertises infinite-frequency
support and Ainf only for this source profile. Saved-run identity and cancellation
rules apply unchanged.

Native angular coordinate0 is A0; -1 is Ainf. This differs from standard WAMIT
period0. Import requires both complete, unique full coupled limit matrices,
four-field serialized rows, exact native inventories/order and printing agreement.
`added_mass_infinite` equals rho times the native-1 matrix, then uses the recorded
work-conjugate map. Its source names the sentinel, profile, .1 hash and source
identity; dataset reference_transformations retain transformation settings.

`wamit_import.native_source.zero_frequency_evidence` retains full incidental A0
in source-frame SI values, source body order/frames/density and exact radiation
file text/hash. This is persisted/hash-bound evidence, not a canonical operator;
include_zero_frequency=True still refuses. No limit damping or excitation is
synthesized. Native unrequested auxiliary limit rows are structurally validated,
hash-bound and explicitly unadmitted, including recorded nonfinite scalar counts.
Requested finite coefficients and both added-mass limits must be finite.

One actual GNU acquisition plus installed transport/lifecycle tests demonstrates
this development route. Its finite coefficients match earlier saved arrays,
Ainf and A0 survive save/reopen, and24NaN scalars in infinite-limit excitation
remain excluded. This establishes transport only: Ainf accuracy, physical W5/W7
gates, corrected Windows producer, Qt child admission and Linux supervisor-death
containment remain open. Evidence: hams-limits-20261010 under the task evidence root.
