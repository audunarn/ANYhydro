## Root actual-run assessment (2026-10-11)

51 focused tests pass on Windows and Linux. The actual bound native check
(root-native-01) exited1 after1.1356318999896757s with
ANYHYDRO-NATIVE-E1: zeta outside the tested envelope. No physical predicates
were measured. The partial running output and external failure receipt are
preserved; no complete evaluator-call count is inferred from the fatal stop.

A read-only domain reconstruction finds220 unsupported of752 planned image
arguments for each source. The kernel supports Re(zeta)<=-30/|Im|<=10, or
Re(zeta)>-.5/|Im|<=10/|zeta|<=8/nonzero. Far and mid-depth receivers exceed
that domain. Next provide typed preflight and a reviewed numerical extension
or reference route; keep the guard and scientific criteria intact. Surface
or bottom-only checks cannot substitute for the missing physical checks.

Evidence: C:/ANYworkspaceWork/anymarine-evidence/w5-physical-green-discriminator-20261011
and hydro .development-evidence/w5-physical-green-discriminator-20261011.
This experimental tool is not a production repair or qualification result.

# Physical Green field discriminator (experimental tool)

Status: EXPERIMENTAL development tool. It confers no scientific
qualification, no production admission and no W5 acceptance. A
successful run is recorded as `completed_unqualified`; the
`qualification` field is always `false` regardless of gate outcomes.
Actual native Green execution is ROOT-ONLY (see below).

Owned files:

- `tools/check_physical_green_field.py` - core diagnostics + guarded CLI
- `tests/test_physical_green_field.py` - real pytest suite (synthetic
  oracles only; no native execution)
- `docs/PHYSICAL_GREEN_FIELD_DISCRIMINATOR.md` - this document

## Purpose

Deliver one coherent matched-source physical discriminator for the
retained composite Green route, without any BEM solve. For two
deterministic one-triangle source cases taken from the actual frozen
case input (first vertical hull triangle, face 72; first horizontal
internal lid triangle, face 720, z = -0.1), it evaluates the actual
receiver fields and reports:

- Laplace (divergence of the gradient, central differences),
- free-surface Robin residual,
- seabed no-flux residual,
- gradient-vs-finite-difference consistency,
- outgoing far-field amplitude/phase vs the analytic Hankel asymptote,
- outgoing (Sommerfeld) radial-derivative residual,

for BOTH the raw native complex field (clearly named `native_raw`,
reported descriptively, NOT the retained route) and the retained
composite field (`composite`), plus a numerical call inventory.

## Governing physics

Time convention `exp(-i omega t)`. Away from the source the potential is
harmonic; boundary conditions:

    Laplace:      grad^2 Phi = 0                     (away from source)
    free surface: d_z Phi - (omega^2/g) Phi = 0       at z = 0
    seabed:       d_z Phi = 0                         at z = -h
    outgoing:     Phi ~ exp(+i k R)                   as R -> infinity

Frozen case: `omega = 1.5`, `h = 150`, `g = 9.81`,
`k = 0x1.d5b98a919d5b9p-3`. The EXACT frozen k is used, and only
after an independent dispersion consistency check: the independent
bisection solve of `omega^2/g = k*tanh(kh)` must agree with the frozen
literal within the known ULP gap (<= 2 ULP; the bisection result is
recorded in `k_solved_independently_hex` and may differ by 1 ULP),
and the frozen k itself must satisfy the dispersion relation within
16 ULP of `omega^2/g` (residual and tolerance recorded). `k.hex()`
must equal the frozen hex exactly; the bound constructor
(`ArtifactBoundPanelProfile.green_function`) independently enforces
the same equality against the reference manifest `k_hex`. The case
`omega`/`water_depth`/`gravity` must equal the frozen actual criteria
exactly.

## Retained composite route

    S_comp    = native_S.real    + 1j * pole_S
    grad_comp = native_grad.real + 1j * pole_grad

`pole_S`/`pole_grad` come from the UNCHANGED public
`anyhydro.panel_integration_candidate.imaginary_pole` evaluated on the
same receivers in the same call sequence. The pole term is smooth (no
normal jump); the single normal-trace jump lives in the native real
part, so no doubled jump occurs off-surface. All gradients are taken
with respect to receivers. The per-panel gradient layout is
`(3, n_receivers, n_panels)`; the TOTAL field used by every diagnostic
sums panels over axis 2 (and `S` over axis 1).

The built-in imaginary pole identity is NOT an independent physical
accuracy test: the pole satisfies the propagating-mode boundary
conditions by construction. The independent oracles are the analytic
Hankel field, finite differences and the far-field asymptote below.

## Far-field oracle (single source triangle, d = (cos t, sin t))

    Z(z)     = cosh(k(z+h))/cosh(kh)
    D        = tanh(kh) + kh*sech(kh)^2
               = tanh(kh) + kh*(1 - tanh(kh)^2)
    A(theta) = sum_q w_q Z(y_q.z) exp(-i k d.y_q.xy)
    Phi_far  = -i k/(2D) sqrt(2/(pi k R)) exp(i(kR - pi/4))
               Z(x.z) A(theta)

This is the outgoing Hankel asymptotic consistent with
`Im G = -k/(2D) Z(x.z) Z(y.z) J0(kR)` (current pole + power
normalization). `D` is `d/dk [k*tanh(kh)]`; the candidate's earlier
formula `th + kh*(1-th)^2` was WRONG and is fixed here (regression
test: `test_dispersion_factor_rejects_candidate_wrong_formula`).

The comparison target is the TOTAL composite field (wave + real
Rankine/image part), with no amplitude or phase fitting, no
normalization, no clipping, no symmetrization and no invented
conservation assertion. It is an exact-asymptotic comparison at finite
kR (100 and 200), not a finite-distance equality.

## Receiver layout (per source case, deterministic)

- 4 far receivers: kR in {100, 200} x headings {0.3, 1.9}, z = -0.05h
- 3 interior receivers (exterior, z in (-h, 0), away from the source)
- 2 free-surface receivers (z = 0)
- 2 seabed receivers (z = -h)
- 36 central-difference shifts of the interior receivers:
  2 steps x 3 directions x 2 signs x 3 points

Total 47 receivers -> 1 batch per source case, 2 batches across the
two source cases (budget: <= 24 batches total, <= 96 receivers per
batch, exactly ONE source panel per native call, 4 Gauss2 source
nodes -> 188 source-quadrature interactions per case, 376 total).

## Proposed gates (root adopts BEFORE any actual execution)

Both finite-difference steps (h = 0.01/k and h = 0.005/k, i.e.
FD_STEPS_K = (0.01, 0.005)) must pass; no extrapolation
and no picked step. Normalizations use a floor of 1e-12 potential
units.

- far-field relative error at kR = 200: <= 3% (floor 1e-12); errors at
  kR = 100 are recorded but not gated.
- Sommerfeld radial residual |dPhi/dR - i k Phi| <= 3% normalized by
  max(k|Phi|, |dPhi/dR|, 1e-12), at all far receivers.
- Laplace (divergence of grad via central differences): rel 1e-3 +
  abs 1e-9, normalized by max(k^2 |Phi|, |Phi|/d^2, 1e-12) with d the
  distance to the source.
- gradient-vs-FD: rel 1e-4 + abs 1e-9, normalized by
  max(|grad|, |Phi|/d, 1e-12).
- free-surface Robin |d_z Phi - nu Phi| with nu = k*tanh(kh) =
  omega^2/g: rel 1e-4 + abs 1e-9, normalized by
  max(|grad_z|, nu|Phi|, 1e-12).
- seabed |grad_z Phi|: rel 1e-4 + abs 1e-9, normalized by
  max(|grad|, k|Phi|, 1e-12).

Truncation sensitivity is reported: per-step residuals and scales are
saved for both steps, and the gate requires both.

### Interpretation caveat (receiver depth)

The far receivers sit at z = -0.05h = -7.5, where Z(z) ~= 0.18. The
propagating-wave reference amplitude is therefore reduced by ~5x
relative to a near-surface receiver, which makes the far-field
amplitude gate more sensitive to any algebraically decaying real-field
component. For the actual finite-depth field (kh ~= 34) the leading 1/R
Rankine term is cancelled by the free-surface image (the real part
decays algebraically faster), so the gates are expected to be robust,
but the kR = 100 amplitude error and the receiver depth are recorded
so root can judge. The synthetic test fixture deliberately uses a
SMALL evanescent stand-in amplitude (1e-3) because a fixture-scale
non-wave component would otherwise dominate the kR = 100 Sommerfeld
residual; this is a fixture choice, not a physics claim.

## Call inventory (counted BEFORE any evaluation)

Printed as JSON progress before any evaluation, saved in the output
record BEFORE any native call (`planned_progress_saved`), and enforced
against actual counts:

- planned batches, receivers per batch, actual provider calls
  (enforced against the plan; a mismatch is a typed refusal),
- constructor calls: ONE shared gf construction for ALL source cases
  (the first case plans the constructor call, later cases plan 0;
  `constructor_calls_actual` is 1),
- Prony coefficient-override uses: planned 1 per case, and the
  OBSERVED count is recorded separately (`coefficient_override`).
  The observed count comes from a counting wrapper installed on the
  constructed gf instance that delegates to the pinned
  `find_best_exponential_decomposition` unchanged; if the attribute is
  not observable the record says `unmeasured` - never a false actual
  claim,
- hidden probe calls (0; `profile.check_micro` probes are refused by
  the budget),
- source-quadrature interactions (receivers x Gauss2 nodes; 188 per
  case, 376 total),
- public pole evaluations (1 per source case),
- native E1 primitive evaluations inside the kernel: UNMEASURED and
  reported as such (they are not observable from this tool).

The preview budget (progress) and the evaluation budget are SEPARATE
NativeBudget objects, so planning is not double-counted; both are
recorded (`budget` and `evaluation_budget`).

## Case identity and input trust

- `case.json` is verified by CANONICALLY RECOMPUTING its case hash
  (sha256 over sorted-key compact JSON of every field except
  `case_hash` itself, the producer's convention); the stored field
  alone is never trusted, and the recomputed hash must equal the
  expected frozen hash.
- the `input.npz` arrays are verified against the case identity
  `snapshot_arrays` entries (shape, dtype, sha256 of the
  ascontiguousarray bytes), binding the input to the known trusted
  case artifact.
- the reference bundle is a DIRECTORY (the
  `ArtifactBoundPanelProfile.load` contract); its regular files
  (manifest/input/reference) are fingerprinted by relative path,
  together with the binding and sidecar files, before and after the
  run (immutability). The binding's `reference_manifest_sha256`
  bundle hash and format are preserved in the record.
- trust is stated as LIMITED to the actual known case and its pinned
  artifacts (`trust_scope` in the record); it is NOT a generic
  qualification of the backend, the route or the solver.

## Source geometry translation

Each source triangle is translated horizontally so its centroid sits
at the global origin (z preserved). This is adopted for THIS isolated
source check only, under horizontal translation invariance of the
homogeneous flat fluid; the original vertices and the translation
delta are recorded (`vertices_original`, `vertices_translated`,
`translation_delta_xy`), height/shape/quadrature are preserved, and NO
global coupled-case claim is made.

## Native source mesh (actual route)

The one-triangle source mesh uses the canonical Capytaine face
`[[0, 1, 2, 2]]` (the existing panel route's
`np.column_stack((triangles, triangles[:, -1]))` form) and
`.with_quadrature("Gauss-Legendre 2")`. The native 4-point source
quadrature is then ASSERTED to match the saved translated quadrature
points/weights (rtol 1e-13); a centroid default (1 point) is refused
with a typed error. A per-case `mesh_receipts` entry records the
faces, the quadrature method, the point count and the match booleans.

## CLI

    python tools/check_physical_green_field.py \
        --case-input  <case>/input.npz \
        --case-json   <case>/case.json \
        --reference   <reference-bundle-dir> \
        --binding     <native-binding.json> \
        --sidecar     <native-sidecar.json> \
        --output      <fresh-output.json> \
        [--source {hull,lid,both}] [--hull-face 72] [--lid-face 720] \
        [--expected-case-hash 6d78c0fc...] \
        [--max-batches 24] [--receivers-per-batch 96] \
        [--plan-only] [--allow-native-execution]

`--max-batches`/`--receivers-per-batch` may TIGHTEN the fixed caps
(24 batches, 96 receivers per batch, exactly 1 source panel per
native call) but any attempt to INCREASE them is refused before
anything runs.

Frozen case hash:
`6d78c0fc3c6a4d5743777888e20cb0d7f1f278e40e6fae0ca42d98d36929556c`
(actual case on Ubuntu 24:
`/var/tmp/anyhydro-native-coupled-20261011/results/energy-point-native-coupled-03c/case`).
Reference bundle:
`.development-evidence/w5-unblock-20261010/inputs/reference-bundle`;
binding/sidecar:
`.development-evidence/w5-native-coupled-followup-20261011/inputs/native-{binding,sidecar}-followup-03.json`.

### Modes

- `--plan-only`: writes plan/counts/metadata only. No profile load, no
  native call. Exit 0.
- default (no `--allow-native-execution`): refuses with status
  `native_execution_not_authorized`, exit 3. No capytaine import.
- `--allow-native-execution` (ROOT ONLY): loads
  `ArtifactBoundPanelProfile.load(reference, binding, sidecar)` with
  the reference as a bundle DIRECTORY, constructs the pinned
  zero-table Green function ONCE via
  `profile.green_function(Delhommeau, capytaine, k=..., water_depth=...)`
  (shared by both source cases; the Prony override use is OBSERVED via
  a delegating counting wrapper), builds a one-triangle canonical
  `capytaine.Mesh` per source case (faces `[[0, 1, 2, 2]]`,
  `.with_quadrature("Gauss-Legendre 2")`, asserted against the saved
  translated quadrature) and evaluates
  `gf.evaluate(receivers, mesh, free_surface=0.0, water_depth=...,
  wavenumber=k, adjoint_double_layer=True, early_dot_product=False,
  diagonal_term_in_double_layer=False)` batch by batch. The worker
  NEVER passes this flag; native execution happens only after root
  inspects this tool and adopts the controls.

### Guards

- fresh output: an existing output path (checked on the RESOLVED
  path, so `..` forms cannot dodge it) is refused (exit 4). The output
  remains a fresh JSON FILE owned by THIS run alone; only that file's
  status is ever modified;
- output location: containment is evaluated on RESOLVED paths - no
  string prefixes, no `..` escape. Refused inside `src/`, `tests/`,
  `tools/`, `docs/`, inside the actual case directory, inside the
  reference bundle directory, and inside any `.development-evidence`
  or `ANYworkspaceWork/anymarine-evidence` subtree other than this
  task's own scope. The task scope is listed in BOTH platform forms
  (`C:/ANYworkspaceWork/...` and `/mnt/c/ANYworkspaceWork/...`) so the
  Linux root runtime resolves it correctly;
- bad arguments (guard refusals, invalid case/inputs) leave the
  filesystem inventory COMPLETELY unchanged: no directories are
  created and no bytes are written; the output parent is created
  lazily on the first write to this run's own fresh output file;
- symlink refusal: any symlinked component in the output chain, a
  symlinked input, or a symlinked reference (or file inside it) is
  refused;
- reference kind: the reference must be a bundle DIRECTORY; a regular
  file is refused;
- case hash: CANONICALLY RECOMPUTED from the case.json contents (the
  stored field is not trusted) and compared with the expected hash;
- input arrays: verified against the case identity `snapshot_arrays`
  shape/dtype/sha256 entries;
- fixed caps: `--max-batches`/`--receivers-per-batch` may tighten but
  never increase the frozen caps;
- input immutability: SHA-256 fingerprints of all input files and of
  every regular file in the reference directory, before and after the
  evaluation; a mismatch records status `input_mutated` (exit 6);
- progress preservation: the plan/counts/metadata record is written to
  the run-owned output BEFORE any native call
  (`planned_progress_saved`), so a hard crash can only lose later
  fields;
- typed failures: `NativeUnavailableError` (a `CapabilityError`) when
  the native provider is unavailable - no fallback, no broadening;
  invalid source arrays, non-positive parameters and non-finite
  provider output are refused BEFORE any pass is recorded;
- source verification: the hull face must be vertical (|n_z| <= 0.25),
  the lid face must be a downward horizontal lid at z = -0.1, and the
  public `triangle_quadrature` must reproduce the stored Gauss2
  nodes/weights (receipt) - all deterministic checks.

### Runtime measurements

`metadata.runtime` records MEASURED values taken before any native
call: the thread env caps, `os.cpu_count()`, the process affinity size
(`sched_getaffinity`, where available), the RLIMIT_AS soft/hard
limits (where available) and, when threadpoolctl is installed, the
live threadpool info. Anything unavailable on the platform is
reported as such, never invented. Compiled native E1 primitive call
counts remain UNMEASURED and are reported as such.

### Output record

Complete diagnostics are saved regardless of gate outcome: all
residuals and scales per FD step, far errors per kR, receivers,
original and translated source geometry with the translation delta,
quadrature and mesh receipts, plan/actual counts (preview and
evaluation budgets kept separate), measured runtime settings,
effective gf settings and evaluator case metadata (actual mode),
canonical case-hash/array verification, reference-directory
fingerprints and the binding bundle hash, input fingerprints, and a
`status`:

- `plan_only`, `completed_unqualified` (exit 0),
- `native_execution_not_authorized` (3),
- `guard_refused` (4),
- `native_unavailable` (5),
- `input_mutated` (6),
- `evaluation_failed` (7, partial diagnostics preserved).

The progress record (`planned_progress_saved`) is written to the
run-owned output BEFORE any native call and is then overwritten by
the final status of the same file. `qualification` is `false` in
every status. Gate failures do not change the status: they are
diagnostic results for root, and a source failure is a useful
result, never a licence to broaden the route, alter thresholds or
tune amplitudes.

## Tests

`tests/test_physical_green_field.py` (51 tests) uses ONLY synthetic
independent analytic oracles (exact scipy Hankel wave field plus a
BC-consistent real evanescent mode as the native-real stand-in; the
pole always comes from the unchanged public helper). No
mirror-implementation-only assertions, no native execution, no
availability probes. Covered: the corrected dispersion factor against
a finite difference of `k*tanh(kh)` (and rejection of the candidate's
wrong formula), the exact frozen-k hex policy (independent solve
within the known ULP gap plus dispersion consistency), finite-depth
mode boundary conditions, the far oracle against an exact Hankel
field, the source phase shift `exp(-i k d.y)`, full composite-vs-
oracle equality, plan/actual count enforcement, both-FD-steps gate
logic (a perturbation on the second step only must fail the gate),
per-panel gradient shape and sum, typed caller refusals, reference-
directory fingerprinting, resolved-containment guard logic (no string
prefix, no `..` escape), and every CLI guard at the CALLER level:
fresh output, output inside the case/reference directories, prefix
attacks on the task scope, fixed-cap increase refusal, bad arguments
leaving the inventory unchanged, canonical case-hash recompute
catching a tampered field, input.npz snapshot-array mismatch, the
no-flag native refusal, and a MOCK-POSITIVE main regression in which
mocked public capytaine/profile modules (reference bundle as a real
DIRECTORY fixture) let the actual-route CLI reach `gf.evaluate` with 2
calls x 47 receivers, 376 source-quadrature interactions, canonical
`[[0, 1, 2, 2]]` faces, the same Gauss-Legendre 2 weights, ONE shared
constructor and an OBSERVED coefficient-override count.

The synthetic CLI fixtures pin their OWN canonically recomputed case
hash via `--expected-case-hash` (the CLI default remains the actual
frozen case hash above); fixtures with a wrong or stale stored hash
are still refused by the canonical recompute. Final receipt: 51
passed / 0 failed under the metered development harness
(`worker_repair` pool, attempt `finalize-pytest-01`, pytest wall
0.58 s). Development fixtures do not confer scientific
qualification; actual native execution remains root-only.
