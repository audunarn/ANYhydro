# Family pressure bank: hull pressures from a retained floating field family

Status: development, 2026-11-13. Qualification is not claimed. Source: `src/anyhydro/family_pressure_bank.py`;
tests: `tests/test_family_pressure_bank.py`. Exported from `anyhydro`: `floating_family_to_pressure_bank`, `linear_rao`.

## Purpose

Composes an `ImportedPressureBank` (`orcawave_pressure.py`) from the per-frequency hull panel pressures that a coherent
floating field family cache already holds. It never solves, never imports Capytaine and never imports Qt (the test
`test_no_heavy_imports_at_base_import` checks the base import). Missing frequencies are refused, not computed.

## Inputs

`floating_family_to_pressure_bank(cache, request, omegas, rao_source=None, *, body_id=0, composition=None, input_files=None)`

- `cache`, `request`: family cache folder and the family request; each frequency is read through
  `floating_radiation_fields.recover_family_snapshot` and the set is checked with `validate_family_coherence`.
- `omegas`: strictly increasing, finite, 1-D, rad/s. Every omega must exist in the cache (`WaveFieldMissing` otherwise).
- `rao_source`: `None` (fixed diffraction), an `(n_omega, 6)` complex array, or a callable `omega -> (6,)`.
- `composition`: optional; may only confirm the choice implied by `rao_source`.
- `input_files`: optional `{role: {'name', 'sha256'}}`, recorded verbatim in the provenance.

## Conventions

Inherited from the family record: phasor `exp(-i omega t)`, SI, unit-amplitude Airy wave, fixed world phase origin,
normals outward from the hull. Pressure is the fluid pressure, so the load on the structure is `-p n a`.

- `p_inc+dif` (`panel_pressure_incident_diffraction`): incident plus fixed-body diffraction pressure.
- `p_rad[k]` (`panel_pressure_radiation`): fluid pressure per unit displacement of mean-body coordinate k, `p = i omega rho phi`.
- `fixed_diffraction`: `p = p_inc+dif`.
- `total_with_source_motion`: `p = p_inc+dif + sum_k xi_k p_rad_k`, `x(t) = Re[xi exp(-i omega t)]`, mean body axes through the
  mean CG. The caller supplies `xi`; nothing is assumed.
- `linear_rao(omega, A, B, excitation, M, C, extra_stiffness=None)` solves `(-w^2 (M+A) - i w B + C) xi = F` (6x6 inputs,
  omega > 0) for use with ANYhydro coefficients. Any other explicit RAO source (for example an imported one) works as well.
- `snapshot_wrench_operator(snap)` gives `W` (n_faces, 6); generalized force in the mean body axes is `-W^T p`.

## Provenance and hashes

`provenance_json` of the bank holds `source='anyhydro_floating_family_cache'`, recipe, composition and its formula,
`snapshot_ids` (one per frequency), `rao_sha256` and `rao_note` (None for fixed diffraction), `input_files`, the family
`conventions`, and the generalized axes `Q0` and `cg_world`. The bank carries heading (mod 2 pi), water depth, rho, gravity and
`body_id`; panel ids are `0..n-1`, surface ids `body:<id>`.

## Refusals (`FamilyPressureBankError`, a `ValueError`)

Non-increasing or non-finite omegas; RAO that is not a finite `(n_omega, 6)` complex array; composition that contradicts the
RAO source; family record without native hull panel pressures; geometry arrays that differ between frequencies;
`linear_rao` with wrong shapes, omega <= 0 or a non-finite result. Snapshot validation and coherence errors come from
`floating_radiation_fields`. A warped quad is carried unchanged and is refused later by `validate_geometry` (geometry owner).

## Connection to the pressure-source selector

The bank is the same `ImportedPressureBank` that the OrcaWave import produces, so it serializes and validates identically.
ANYfem's `bank_from_family_spec` (`ANYfem/src/anyfem/application/orcawave_pressure.py`, spec type
`anyhydro_family_pressure_source_v1`) calls this function from a JSON spec (cache, request, omegas, optional `rao_file`; RAO
frequencies must equal the pressure frequencies exactly) and then runs `bank.validate_geometry()`. The selector wiring in the
ANYfem GUI was not checked for this document: not verified.

## Example

```python
from anyhydro import floating_family_to_pressure_bank, linear_rao
xi = [linear_rao(w, A[k], B[k], F[k], M, C) for k, w in enumerate(omegas)]
bank = floating_family_to_pressure_bank(cache_dir, request, omegas, xi)   # total_with_source_motion
fixed = floating_family_to_pressure_bank(cache_dir, request, omegas)      # fixed_diffraction
```

## Test

`python -m pytest tests/test_family_pressure_bank.py -q --basetemp <one reused folder>`. Eight tests; the last
(`test_local_refined_lid_cache_...`) reads a local L01 cache and its availability on other machines is not verified.

## Known limits

- Depends on `validate_family_coherence`, which is in the uncommitted diff of `floating_radiation_fields.py`; on a clean
  `origin/main` the import fails.
- RAO is never interpolated; no frequency is synthesized.
- Closure against force is tested on synthetic panels and one local L01 cache only. No convergence or OrcaWave agreement claim.
