# Corrected-artifact finite-depth wave check — 2026-10-10

The public finite-depth evaluator passes the original real-gradient/reference-FD
gates on the frozen43-point/one-triangle case. Maximum real-gradient error is
1.152189420261518e-10 against unchanged1.4765493367017705e-8. The original last two
reference finite-difference checks pass. This is a local reference result only.

Four of seven native-self-FD diagnostics fail; the three coarser steps pass.
Failed errors:6.49546692860388e-8,3.6132035518459185e-7,2.8522062529064834e-7,
2.402847169976838e-7. All original steps/limits/flags remain explicit. Native complex
potential error1.8583027781501893e-10 and imaginary-gradient error4.336808689942018e-19
are diagnostics, not new acceptance criteria. Root's exact stored-array decomposition
localizes the small-step residual to finite differences of native/reference potential
differences; direct Rankine fine-step FD errors are about1e-12. It does not uniquely
identify a source cause or imply a universal noise law.

Read-verified source mechanism: Delhommeau_integrals.f90's exp_e1 function explicitly
uses complex(kind(1e0)) intermediates and narrows its input, despite the float64
backend. Independent Mistral Read supports the reference pass and diagnosis but
root limits its stronger causal/universal claims; full report/disposition retained.
No source change was made in this wave slice.

Mistral delivered the connected checker/tests. Initial298.3618400999985-second
run produced the candidate but no tests/handoff; focused204.5473525999987-second
continuation finished45 mocks, all passing on first run (2.703012199999648 seconds
process wall). Shared-runner prospective hashes match checker/helper/test source.
52 preexisting protected files remain unchanged. Three supervised source/input
inspection checks also retain exact accounting; they consumed no native authority.

Actual native run: ctor1/emptytable1/Rankine1/wave1/fixed-coefficient override1,
default-fit0/cache0/positive-table0/system0, all pools1. Child wall
1.3207687639999999 seconds; Windows/WSL outer5.31107680000423
seconds. One-shot admission consumed; original bytes and failed diagnostics preserved.
Inherited public constructor/evaluate/Rankine execute the actual corrected artifact.
The public-method override supplies declared immutable Prony coefficients from the
original mathematical reference; old producer/backend bytes are untouched and
separate from new evaluator identity. No default-fitting qualification follows.
The saved remainder is total minus direct Rankine, including reflected/image terms.

Exact backend ID capytaine-3.0.0-anyhydro-precision-pi-gamma-v1/linux-x86_64,
wheel806bd98a22bb5adf8ac109b5e023ed9b4ba3858ef50151e92fecd667d252df84,
f64 extension37c66565f9fd3643ca51840dd22f126547233dbf7bd292ef5095b0bc69f46311.
Existing isolated Linux runtime reused read-only; application/build runtimes unchanged.
Authoritative evidence: .development-evidence/w5-wave-artifact-20261010/criteria.json,
inputs/,native-admission.json.consumed,results/native-once/,root-wave-adjudication.json,
native-linux-receipt.json,review-disposition.md and attempts/. The summary's all_passed
means reference gates; four self-FD failures are separately and explicitly preserved.

All pools frozen: ordinary3.506499299983261s/240, native outer
5.31107680000423s/180; worker/review accounting separate. No unused
authority carries forward. No unchanged successful test/reference case rerun needed.

Next Mistral-owned coherent outcome: isolate PRE-kind E1 arithmetic while preserving
formulas/coefficient values/branches, build a separately identified wheel and perform
one matched changed-input micro under fresh admission. Root adopts source/identity
contracts; results determine whether precision is resolved. Then measure the operator
route and complete a coupled energy point, rather than indefinitely repeating micro
diagnostics. Full-operator investigation is permitted; acceptance needs actual gates.
W5 remains NO_GO. Default Prony, below-cutoff, full-wave/operator/passivity/energy/
PSD/radiation/RAO, full W6–W10 and thin-sheet scope remain open. Full goal ACTIVE.
