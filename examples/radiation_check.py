"""Analytic coupled radiation example (development, no scientific qualification).

Builds a full 12x12 coupled two-body dataset from the independent
exponential-kernel oracle

    B(w) = C*lambda/(lambda^2 + w^2)
    A(w) = A_inf - C/(lambda^2 + w^2)
    K(t) = C*exp(-lambda*t)            (closed form)

then runs the complete radiation-preprocessing route:

    validated neutral dataset
      -> explicit A_inf/source-preserving full-coupled causal kernel
      -> reconstruction of BOTH added mass and damping
      -> criteria-based diagnostics
      -> immutable serialized artifact and reproduction

The declared tolerances are this example's own caller criteria, chosen for
the discretization below; they are not a scientific default. Observed max
errors are printed regardless of pass/fail.

Run:
    python examples/radiation_check.py --out radiation_kernel.nc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import anyhydro
from anyhydro.radiation import (
    RadiationKernel,
    ReconstructionCriteria,
    RadiationSettings,
    compare_reconstruction,
    load_radiation_kernel,
    radiation_kernel_from_dataset,
    save_radiation_kernel,
)
from anyhydro.fixtures import make_fixture_bodies
from anyhydro.records import HydroRequest
from anyhydro.schema import assemble_hydro_dataset

# --- discretization (within the development caps: 513 freqs / 1001 time / 12 DOF)
# Valid regime: kernel duration * frequency spacing ~= 1 (spacing-vs-duration
# diagnostic), omega_max*dt < pi, T >> 1/lambda.
N_OMEGA = 513
OMEGA_MIN, OMEGA_MAX = 0.05, 50.0
N_TIME, DT = 1001, 0.01  # duration 10 s >> 1/lambda
WINDOW = (0.5, 8.0)  # caller-declared comparison window


def oracle_matrices(n_dof: int):
    """Full coupled C, lambda, A_inf with nonzero cross-body blocks."""
    C = np.empty((n_dof, n_dof))
    lam = np.empty((n_dof, n_dof))
    for i in range(n_dof):
        for j in range(n_dof):
            same_body = (i // 6) == (j // 6)
            if i == j:
                C[i, j] = 900.0 + 50.0 * (i % 6)
                lam[i, j] = 1.0 + 0.08 * (i % 6)
            elif same_body:
                C[i, j] = 80.0 + 6.0 * ((i * 7 + j * 3) % 11)
                lam[i, j] = 0.9 + 0.05 * ((i + j) % 9)
            else:
                C[i, j] = 30.0 + 5.0 * ((i * 5 + j * 11) % 7)
                lam[i, j] = 1.1 + 0.04 * ((i * 3 + j * 7) % 5)
    a_inf = 2.0 * C / lam**2 + 100.0  # explicit limit, distinct from every A(w)
    return C, lam, a_inf


def build_oracle_dataset():
    bodies = make_fixture_bodies()
    n_dof = 6 * len(bodies)
    C, lam, a_inf = oracle_matrices(n_dof)
    omegas = np.linspace(OMEGA_MIN, OMEGA_MAX, N_OMEGA)
    request = HydroRequest(
        omegas=omegas,
        wave_directions=np.array([0.0]),
        water_depth=150.0,
        rho=1025.0,
        gravity=9.81,
        include_zero_frequency=False,
        include_infinite_frequency=False,
        backend_settings={"fixture": "radiation-exponential-oracle"},
    )
    w2 = omegas[:, None, None] ** 2
    B = C[None] * lam[None] / (lam[None] ** 2 + w2)
    A = a_inf[None] - C[None] / (lam[None] ** 2 + w2)
    centers = np.array([b.rotation_center for b in bodies], dtype=np.float64)
    coms = np.array([b.center_of_mass for b in bodies], dtype=np.float64)
    ds = assemble_hydro_dataset(
        bodies,
        request,
        added_mass=A,
        radiation_damping=B,
        excitation_force=np.zeros((N_OMEGA, 1, n_dof), dtype=np.complex128),
        hydrostatic_stiffness=np.zeros((n_dof, n_dof)),
        rotation_center=centers,
        center_of_mass=coms,
        center_of_buoyancy=centers,
        added_mass_infinite=a_inf,
        added_mass_infinite_source=(
            "analytic exponential-kernel oracle: explicit limit, not a backend solve"
        ),
        backend="exponential-oracle",
        qualification_status="development_fixture",
    )
    anyhydro.validate_for_time_domain(ds)
    return ds, C, lam, a_inf


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="radiation_kernel.nc", help="artifact NetCDF path")
    args = parser.parse_args()

    print("=== ANYhydro radiation development example (no scientific qualification) ===")
    ds, C, lam, a_inf = build_oracle_dataset()
    print(f"dataset: {ds.sizes['omega']} frequencies in "
          f"[{ds['omega'].values[0]}, {ds['omega'].values[-1]}], "
          f"{len(ds['influenced_dof'])} coupled DOFs, "
          f"qualification_status={ds.attrs['qualification_status']!r}")

    settings = RadiationSettings(
        dt=DT,
        n_time=N_TIME,
        symmetry_tol=1.0e-9,
        passivity_tol=1.0e-9,
        low_tail_policy="explicit_zero_sample",
        zero_frequency_damping=C / lam,  # oracle B(0)
    )
    kernel = radiation_kernel_from_dataset(ds, settings)
    print(f"kernel: {kernel.kernel.shape} on t in [0, {kernel.time[-1]}] s, "
          f"dofs={kernel.n_dof}, artifact status={kernel.qualification_status!r}")

    d = kernel.diagnostics
    print("tail diagnostics: omega_max=%.6g, max|B(omega_max)|=%.6g, "
          "omega_max*dt=%.4g (pi=%.4f), max spacing=%.4g, "
          "duration*spacing/(2pi)=%.4g"
          % (d.omega_max, d.max_abs_damping_at_omega_max, d.omega_max_times_dt,
             np.pi, d.max_frequency_spacing, d.duration_over_spacing_period))
    print("symmetry/passivity (errors preserved): max|B-B^T|=%.3g, "
          "max|A-A^T|=%.3g, min eig((B+B^T)/2)=%.6g, "
          "damping symmetry violations=%d, added mass symmetry violations=%d, "
          "passivity violations=%d"
          % (d.max_damping_asymmetry, d.max_added_mass_asymmetry,
             d.min_damping_eigenvalue, d.damping_symmetry_violations,
             d.added_mass_symmetry_violations, d.passivity_violations))

    # closed-form kernel check (the oracle's analytic kernel is known); the
    # explicit omega_max tail truncation is absolute-small, so compare on the
    # kernel's own scale max(C) = max K(0)
    t = kernel.time
    analytic_kernel = C[None] * np.exp(-lam[None] * t[:, None, None])
    kernel_err = float(np.max(np.abs(kernel.kernel - analytic_kernel)) / np.max(C))
    print(f"kernel vs closed form C*exp(-lambda*t): max |error| / max(C) = {kernel_err:.3e}")

    # reconstruction of BOTH matrices against the analytic oracle; the w=0
    # explicit limit is reported separately (the finite kernel duration,
    # capped by the spacing-vs-duration criterion, limits its accuracy)
    w_check = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    a_rec, b_rec = kernel.reconstruct(w_check)
    w2 = w_check[:, None, None] ** 2
    a_ref = a_inf[None] - C[None] / (lam[None] ** 2 + w2)
    b_ref = C[None] * lam[None] / (lam[None] ** 2 + w2)
    pos = w_check > 0.0
    print(f"reconstruction vs oracle (w in {list(w_check[pos])}): "
          f"max|A_rec-A|={float(np.max(np.abs(a_rec[pos] - a_ref[pos]))):.3e}, "
          f"max|B_rec-B|={float(np.max(np.abs(b_rec[pos] - b_ref[pos]))):.3e}")
    print(f"explicit w=0 limit: |A_rec(0)-A(0)|max="
          f"{float(np.max(np.abs(a_rec[0] - a_ref[0]))):.3e}, "
          f"|B_rec(0)-B(0)|max={float(np.max(np.abs(b_rec[0] - b_ref[0]))):.3e} "
          f"(limited by the finite kernel duration)")

    # strict zero-limit formula verification on the analytically known sampled
    # exponential kernel (independent of the bridge's omega_max band
    # truncation), plus the documented finite-band w=0 discrepancy of the
    # bridge kernel itself (reported, never admitted by loosening criteria)
    sampled = RadiationKernel(
        settings=settings,
        time=t,
        dof_keys=kernel.dof_keys,
        body_ids=kernel.body_ids,
        kernel=analytic_kernel,
        added_mass_infinite=a_inf,
        diagnostics=kernel.diagnostics,
        provenance=dict(kernel.provenance),
        assumptions=kernel.assumptions,
    )
    a0_ref = a_inf - C / lam**2
    b0_ref = C / lam
    a0_sampled, b0_sampled = sampled.reconstruct(np.array([0.0]))
    sampled_a_rel = float(np.max(np.abs(a0_sampled[0] - a0_ref) / np.abs(a0_ref)))
    sampled_b_rel = float(np.max(np.abs(b0_sampled[0] - b0_ref) / np.abs(b0_ref)))
    band_a_rel = float(np.max(np.abs(a_rec[0] - a0_ref) / np.abs(a0_ref)))
    print(f"zero-limit formula on sampled exponential kernel (strict rtol=1e-2): "
          f"max rel err A={sampled_a_rel:.3e}, B={sampled_b_rel:.3e}")
    print(f"finite-band bridge kernel w=0 discrepancy (documented, not admitted): "
          f"max rel err A={band_a_rel:.3e}")
    if sampled_a_rel >= 1.0e-2 or sampled_b_rel >= 1.0e-2:
        raise SystemExit("strict zero-limit verification on the sampled exponential kernel failed")

    # criteria-based diagnostics in the caller-declared window
    criteria = ReconstructionCriteria(
        omega_min=WINDOW[0],
        omega_max=WINDOW[1],
        added_mass_abs_tol=2.0,     # kg
        added_mass_rel_tol=1.0e-2,
        damping_abs_tol=2.0,         # N s/m
        damping_rel_tol=1.0e-2,
    )
    report = compare_reconstruction(kernel, ds, criteria)
    summary = report.summary()
    print(f"criteria comparison in window {WINDOW}: "
          f"{summary['n_window_frequencies']} frequencies, "
          f"overall_pass={summary['overall_pass']}")
    print(f"observed max errors: added mass={summary['max_added_mass_error']:.3e}, "
          f"damping={summary['max_damping_error']:.3e}; "
          f"worst error/threshold ratios: "
          f"A={summary['added_mass_worst_error_to_threshold_ratio']:.3e}, "
          f"B={summary['damping_worst_error_to_threshold_ratio']:.3e}")
    print(f"failing coefficients: added mass={summary['failing_added_mass_coefficients']}, "
          f"damping={summary['failing_damping_coefficients']}")

    # immutable serialized artifact and reproduction
    out = Path(args.out)
    save_radiation_kernel(kernel, out)
    reloaded = load_radiation_kernel(out)
    same = reloaded.content_hash() == kernel.content_hash()
    print(f"artifact saved to {out} ({out.stat().st_size} bytes); "
          f"reproduction content-hash match: {same}")
    print(f"artifact hash: {kernel.content_hash()}")
    print(f"source dataset hash: {kernel.provenance['source_dataset_hash']}")
    print(f"A_inf source: {kernel.provenance['a_inf_source']}")
    for assumption in kernel.assumptions:
        print(f"assumption: {assumption}")

    if not same:
        raise SystemExit("artifact reproduction failed")
    if not summary["overall_pass"]:
        raise SystemExit("declared reconstruction criteria not met")
    print("example complete: development diagnostics only, no scientific acceptance claim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
