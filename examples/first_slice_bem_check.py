"""ONE tiny real Capytaine 3.0.0 BEM pilot through the public anyhydro path.

First-slice preparation pilot (development evidence, NOT a qualification
run): the deterministic two-body wetted-hull/lid fixture (96 panels),
fixed frequencies [2, 4, 6] rad/s, heading 0, rho 1025, g 9.81, depth 150,
explicit added_mass_infinite requested, n_jobs=n_threads=1, default
CapytaineExecutionLimits (100 panels / 3 frequencies / 1 heading).

Outputs into --out (all preserved on failure):
- first_slice_prep_bem_01.nc  : the canonical solved dataset (NetCDF)
- first_slice_prep_bem_01.json: geometry/request/source/version hashes,
  effective imports, diagnostics, raw symmetry/passivity report, effective
  limits, per-check results and timings.

Persistence roundtrip evidence (hash identity): the solved dataset is
reloaded from the NetCDF, re-saved and reloaded again; the content hash and
the effective execution limits must be identical across all three forms.


Exit code 0 only if every declared check passes; the summary never claims
scientific qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, help="evidence output directory")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "first_slice_prep_bem_01.json"
    nc_path = out / "first_slice_prep_bem_01.nc"

    t0 = time.perf_counter()
    summary = {
        "pilot": "first-slice-prep-bem-01",
        "qualification": (
            "unqualified development pilot; no scientific qualification, "
            "no W5 gate, no dense BEM claim"
        ),
        "checks": {},
        "timings_s": {},
    }
    all_pass = True

    def check(name: str, ok: bool, detail) -> bool:
        nonlocal all_pass
        summary["checks"][name] = {"pass": bool(ok), "detail": detail}
        if not ok:
            all_pass = False
        return bool(ok)

    try:
        import anyhydro
        from anyhydro import (
            CapytaineBackend,
            CapytaineExecutionLimits,
            dataset_content_hash,
            load_hydro_dataset,
            make_first_slice_bodies,
            make_first_slice_request,
            save_hydro_dataset,
            validate_hydro_dataset,
            verify_first_slice_fixture,
        )
        import capytaine
        import scipy
        import xarray

        summary["effective_imports"] = {
            "python": platform.python_version(),
            "anyhydro_version": anyhydro.__version__,
            "anyhydro_file": str(Path(anyhydro.__file__).resolve()),
            "capytaine_version": capytaine.__version__,
            "capytaine_file": str(Path(capytaine.__file__).resolve()),
            "numpy_version": np.__version__,
            "xarray_version": xarray.__version__,
            "scipy_version": scipy.__version__,
        }
        src = Path(anyhydro.__file__).resolve().parent
        summary["source_hashes_sha256"] = {
            p.name: _sha256_file(p)
            for p in sorted(src.glob("*.py"))
        }

        # -- fixture and analytic verification --------------------------- #
        bodies = make_first_slice_bodies()
        geometry_report = verify_first_slice_fixture(bodies)
        check("fixture_verification", True, geometry_report["totals"])

        request = make_first_slice_request(include_infinite_frequency=True)
        limits = CapytaineExecutionLimits()
        backend = CapytaineBackend(execution_limits=limits, cache_dir=str(out / "cache"))
        summary["effective_limits"] = limits.as_dict()
        summary["diagnostics"] = [
            {"code": d.code, "severity": d.severity, "message": d.message,
             "details": dict(d.details)}
            for d in backend.diagnose()
        ]

        request_hash_before = request.content_hash()
        mesh_hashes_before = {
            b.body_id: {"hull": b.mesh.mesh_hash, "lid": b.lid_mesh.mesh_hash}
            for b in bodies
        }
        vertices_before = [b.mesh.vertices.copy() for b in bodies]
        faces_before = [b.mesh.faces.copy() for b in bodies]

        # -- the ONE real BEM solve -------------------------------------- #
        t_solve = time.perf_counter()
        dataset = backend.solve(bodies, request)
        summary["timings_s"]["solve"] = time.perf_counter() - t_solve

        # -- canonical dataset and outputs ------------------------------- #
        validate_hydro_dataset(dataset)
        save_hydro_dataset(dataset, nc_path)
        summary["dataset"] = {
            "path": str(nc_path),
            "content_hash": dataset_content_hash(dataset),
            "request_hash": dataset.attrs["request_hash"],
            "backend": dataset.attrs["backend"],
            "backend_version": dataset.attrs["backend_version"],
            "qualification_status": dataset.attrs["qualification_status"],
            "development_limits": dataset.attrs["development_limits"],
            "execution_limits": dataset.attrs["execution_limits"],
        }

        # -- persistence roundtrip evidence (hash identity) --------------- #
        in_memory_hash = summary["dataset"]["content_hash"]
        reloaded = load_hydro_dataset(nc_path)
        reloaded_hash = dataset_content_hash(reloaded)
        resave_path = out / "first_slice_prep_bem_01_roundtrip.nc"
        save_hydro_dataset(reloaded, resave_path)
        resaved_reloaded = load_hydro_dataset(resave_path)
        resaved_reloaded_hash = dataset_content_hash(resaved_reloaded)
        el_in_memory = dataset.attrs["execution_limits"]
        el_in_memory_dict = (
            json.loads(el_in_memory) if isinstance(el_in_memory, str)
            else el_in_memory
        )
        roundtrip = {
            "in_memory_content_hash": in_memory_hash,
            "reloaded_content_hash": reloaded_hash,
            "resaved_reloaded_content_hash": resaved_reloaded_hash,
            "execution_limits_in_memory": el_in_memory,
            "execution_limits_reloaded": reloaded.attrs["execution_limits"],
            "execution_limits_resaved_reloaded": (
                resaved_reloaded.attrs["execution_limits"]
            ),
        }
        summary["persistence_roundtrip"] = roundtrip
        check(
            "persistence_roundtrip_hash_identity",
            in_memory_hash == reloaded_hash
            and reloaded_hash == resaved_reloaded_hash
            and el_in_memory_dict == reloaded.attrs["execution_limits"]
            and el_in_memory_dict == resaved_reloaded.attrs["execution_limits"],
            roundtrip,
        )

        # -- declared pilot checks ---------------------------------------- #
        n_dof = 12
        am = dataset["added_mass"].values
        rd = dataset["radiation_damping"].values
        ex = dataset["excitation_force"].values
        check(
            "full_12x12_finite_coefficients",
            am.shape == (3, n_dof, n_dof) and rd.shape == (3, n_dof, n_dof)
            and ex.shape == (3, 1, n_dof),
            {"added_mass": list(am.shape), "radiation_damping": list(rd.shape),
             "excitation_force": list(ex.shape)},
        )
        check(
            "nonzero_cross_body_blocks",
            bool(
                np.any(am[:, 0:6, 6:12] != 0.0)
                and np.any(am[:, 6:12, 0:6] != 0.0)
                and np.any(rd[:, 0:6, 6:12] != 0.0)
            ),
            {
                "added_mass_cross_max_abs": float(np.abs(am[:, 0:6, 6:12]).max()),
                "damping_cross_max_abs": float(np.abs(rd[:, 0:6, 6:12]).max()),
            },
        )
        ainf = dataset["added_mass_infinite"].values
        ainf_source = str(dataset["added_mass_infinite"].attrs["source"])
        check(
            "explicit_ainf_source",
            ainf.shape == (n_dof, n_dof)
            and "omega=inf" in ainf_source
            and "capytaine" in ainf_source
            and np.any(ainf[0:6, 6:12] != 0.0),
            {"shape": list(ainf.shape), "source": ainf_source},
        )

        # analytical heave buoyancy stiffness: rho*g*A_wp = 10055.25 N/m
        expected_k = 1025.0 * 9.81 * 1.0
        hs = dataset["hydrostatic_stiffness"].values
        heave_values = [float(hs[2, 2]), float(hs[8, 8])]
        rel_errors = [abs(v - expected_k) / abs(expected_k) for v in heave_values]
        check(
            "analytical_heave_buoyancy_stiffness",
            all(e <= 1e-10 for e in rel_errors),
            {
                "expected_n_per_m": expected_k,
                "measured_n_per_m": heave_values,
                "relative_errors": rel_errors,
                "tolerance_relative": 1e-10,
            },
        )

        # no mutations of the caller's records
        unchanged = (
            request.content_hash() == request_hash_before
            and all(
                b.mesh.mesh_hash == mesh_hashes_before[b.body_id]["hull"]
                and b.lid_mesh.mesh_hash == mesh_hashes_before[b.body_id]["lid"]
                for b in bodies
            )
            and all(
                np.array_equal(b.mesh.vertices, v0)
                and np.array_equal(b.mesh.faces, f0)
                for b, v0, f0 in zip(bodies, vertices_before, faces_before)
            )
        )
        check("no_input_mutations", unchanged, {"request_hash": request_hash_before})

        # -- raw symmetry / passivity report (reported, NOT a gate) -------- #
        def _asym(x):
            scale = float(np.abs(x).max())
            return float(np.abs(x - x.T).max() / scale) if scale > 0.0 else 0.0

        raw = {
            "note": (
                "raw measured report only; no gate tuning, no qualification; "
                "B(omega_max) is reported, not claimed small"
            ),
            "per_omega": [],
        }
        for k, omega in enumerate(dataset["omega"].values):
            raw["per_omega"].append(
                {
                    "omega_rad_s": float(omega),
                    "added_mass_asymmetry_relative": _asym(am[k]),
                    "damping_asymmetry_relative": _asym(rd[k]),
                    "damping_diagonal_min": float(np.diag(rd[k]).min()),
                    "damping_diagonal_max": float(np.diag(rd[k]).max()),
                }
            )
        raw["ainf_asymmetry_relative"] = _asym(ainf)
        raw["damping_at_omega_max_diagonal"] = [
            float(v) for v in np.diag(rd[-1])
        ]
        summary["raw_symmetry_passivity_report"] = raw

        summary["geometry"] = geometry_report
        summary["request"] = {
            "omegas": [float(v) for v in request.omegas],
            "wave_directions": [float(v) for v in request.wave_directions],
            "water_depth": float(request.water_depth),
            "rho": float(request.rho),
            "gravity": float(request.gravity),
            "include_infinite_frequency": bool(request.include_infinite_frequency),
            "content_hash": request_hash_before,
        }
        summary["mesh_hashes"] = mesh_hashes_before
        summary["all_checks_pass"] = all_pass
        summary["timings_s"]["total"] = time.perf_counter() - t0
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "pilot": summary["pilot"],
                    "all_checks_pass": all_pass,
                    "dataset": nc_path.name,
                    "summary_json": json_path.name,
                    "solve_seconds": round(summary["timings_s"]["solve"], 3),
                }
            )
        )
        return 0 if all_pass else 1
    except Exception:
        summary["all_checks_pass"] = False
        summary["error"] = traceback.format_exc()
        summary["timings_s"]["total"] = time.perf_counter() - t0
        try:
            json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        except Exception:
            pass
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
