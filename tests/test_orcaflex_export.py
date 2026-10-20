# SPDX-License-Identifier: MPL-2.0
"""OrcaFlex vessel-type export (slice O1b): pure data tests, refusals, and OrcFxAPI round-trip tests.

The pure tests need neither OrcFxAPI nor a licence. The OrcFxAPI tests skip cleanly without the package or a licence.
Hand-computed expectations follow docs/ORCAFLEX_EXPORT.md (conventions proved by the O1a probes).
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

from anyhydro import (HydroError, HydrostaticContract, attach_hydrostatic_contract, dataset_content_hash,
                      reframe_dataset)
from anyhydro import orcaflex_export as ox
from anyhydro.fixtures import FIXTURE_BODY_IDS, make_fixture_bodies, make_fixture_request, make_two_body_fixture
from anyhydro.schema import MEAN_DRIFT_UNITS, assemble_hydro_dataset

BODY = FIXTURE_BODY_IDS[0]
RHO, G = 1025.0, 9.81
OMEGA = np.array([0.3, 0.5, 0.7, 0.9, 1.2])
HEADINGS = np.radians([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0])
RC = np.array([3.0, -2.0, -1.0])               # rotation centre = RAO origin = reference origin
COB = np.array([0.5, 0.25, -2.0])
COG = np.array([0.5, 0.25, 0.5])                # same horizontal position as the centre of buoyancy
LENGTH = 20.0
DOFS = ("surge", "sway", "heave", "roll", "pitch", "yaw")


def _sym_pd(rng, n, scale, floor):
    a = rng.normal(size=(n, n))
    return a @ a.T * scale + np.eye(n) * floor


def default_arrays(headings=HEADINGS, *, heavy=False, duplicate_of=None, seed=20261020):
    rng = np.random.default_rng(seed)
    nw, nh = len(OMEGA), len(headings)
    A = np.stack([_sym_pd(rng, 6, 1e4, 1e6) for _ in range(nw)])
    B = np.stack([_sym_pd(rng, 6, 1e3, 1e4) for _ in range(nw)])
    scale6 = np.array([2e5, 2e5, 4e5, 3e6, 3e6, 3e6])
    X = (rng.normal(size=(nw, nh, 6)) + 1j * rng.normal(size=(nw, nh, 6))) * scale6
    drift = rng.uniform(-5e3, 5e4, size=(nw, nh, 3)) * np.array([1.0, 1.0, 3.0])
    if duplicate_of is not None:                       # the last heading repeats an earlier one (e.g. 2 pi = 0)
        X[:, -1] = X[:, duplicate_of]
        drift[:, -1] = drift[:, duplicate_of]
    K = np.zeros((6, 6))
    if heavy:
        K[2, 2], K[3, 3], K[4, 4], K[2, 4], K[4, 2] = 5e9, 3e12, 8e12, 2e10, 2e10
    else:
        K[2, 2], K[3, 3], K[4, 4], K[2, 4], K[4, 2] = 5e5, 3e6, 8e6, 1e5, 1e5
    Ainf = A.max(axis=0) * 1.1 + np.eye(6) * 1e5
    return dict(A=A, B=B, X=X, drift=drift, K=K, Ainf=Ainf)


def _body():
    return dataclasses.replace(make_fixture_bodies()[0], rotation_center=tuple(RC), center_of_mass=tuple(COG))


def _add_drift(ds, drift):
    """Schema 1.3.0 Kochin extension with a far-field mean drift block (synthetic Kochin data; only the drift is used)."""
    import xarray as xr
    from anyhydro.green_function import solve_wavenumber
    n_theta = 720
    nw, nh = ds["omega"].size, ds["wave_direction"].size
    out = ds.assign_coords(theta=2.0 * np.pi * np.arange(n_theta) / n_theta,
                           drift_dof=[f"{BODY}::{d}" for d in ("surge", "sway", "yaw")])
    out["kochin_radiation"] = (("omega", "radiating_dof", "theta"), np.zeros((nw, 6, n_theta), dtype=np.complex128))
    out["kochin_diffraction"] = (("omega", "wave_direction", "theta"), np.zeros((nw, nh, n_theta), dtype=np.complex128))
    out["wavenumber"] = (("omega",), np.array([solve_wavenumber(float(w), G, math.inf) for w in OMEGA]))
    out["water_depth"] = ((), np.float64(math.inf))
    out["rho"] = ((), np.float64(RHO))
    out["g"] = ((), np.float64(G))
    out["mean_drift_far_field"] = (("omega", "wave_direction", "drift_dof"), np.asarray(drift, dtype=np.float64))
    out.attrs["schema_version"] = "1.3.0"
    out.attrs["wave_environment"] = json.dumps({"water_depth": "inf", "rho": RHO, "g": G})
    out.attrs["kochin_export"] = json.dumps({"route": "synthetic", "reference_point": [0.0, 0.0], "theta_count": n_theta})
    out.attrs["mean_drift_far_field"] = json.dumps(
        {"format": "synthetic", "method": "far_field_momentum_conservation", "units": MEAN_DRIFT_UNITS,
         "motions": {"source": "fixed", "motion_hash": "synthetic"}, "yaw_reference": [float(v) for v in RC]})
    return out


def make_dataset(*, headings=HEADINGS, volume=100.0, contract="ok", with_drift=False, with_ainf=True, role="BUOYANCY_ONLY",
                 force_basis="GENERALIZED_DOF", prov_gravity=G, prov_rho=RHO, baseline_fz_scale=1.0, **overrides):
    """A validated single-body dataset (GLOBAL axes) with a bound BUOYANCY_ONLY hydrostatic contract."""
    arrays = default_arrays(headings, duplicate_of=overrides.pop("duplicate_of", None),
                            heavy=bool(overrides.pop("heavy", False)))
    arrays.update({k: v for k, v in overrides.items() if k in arrays})
    request = dataclasses.replace(make_fixture_request(), omegas=OMEGA, wave_directions=np.asarray(headings, dtype=float),
                                  water_depth=math.inf, rho=RHO, gravity=G)
    ds = assemble_hydro_dataset(
        [_body()], request, added_mass=arrays["A"], radiation_damping=arrays["B"], excitation_force=arrays["X"],
        hydrostatic_stiffness=arrays["K"], rotation_center=RC[None, :], center_of_mass=COG[None, :],
        center_of_buoyancy=COB[None, :], added_mass_infinite=arrays["Ainf"] if with_ainf else None,
        added_mass_infinite_source="synthetic test construction" if with_ainf else None,
        backend="synthetic-test", backend_version="1", backend_commit="none", qualification_status="development_fixture")
    if with_drift:
        ds = _add_drift(ds, arrays["drift"])
    if contract is None:
        return ds
    fz = RHO * G * volume * baseline_fz_scale
    wrench = (0.0, 0.0, fz) + tuple(np.cross(COB - RC, [0.0, 0.0, fz]))
    provenance = {"source": "synthetic test declaration"}
    if prov_gravity is not None:
        provenance["gravity"] = prov_gravity
    if prov_rho is not None:
        provenance["rho"] = prov_rho
    return attach_hydrostatic_contract(ds, HydrostaticContract(
        role=role, force_basis=force_basis, stiffness_derivative="INCREMENTAL_ABOUT_EQUILIBRIUM", bodies=(BODY,),
        baseline_wrench={BODY: wrench}, provenance=provenance,
        full_source=({"mass": 1.0, "center_of_gravity": [0.0, 0.0, 0.0], "composition": "x"} if role == "FULL" else None)))


def mass_props(volume=100.0, *, inertia=None, mass=None, cog=COG):
    return ox.DeclaredMassProperties(mass=RHO * volume if mass is None else mass, centre_of_gravity=tuple(cog),
                                     inertia=np.diag([2.0e6, 9.0e6, 1.0e7]) if inertia is None else inertia)


def export_data(ds=None, props=None, **kw):
    kw.setdefault("length_scale", LENGTH)
    kw.setdefault("gravity", G)
    return ox.vessel_type_data(make_dataset() if ds is None else ds, mass_props() if props is None else props, **kw)


def skew_ratio(M, L=LENGTH):
    S = np.diag([1, 1, 1, 1 / L, 1 / L, 1 / L])
    Mn = S @ M @ S
    return float(np.linalg.norm(0.5 * (Mn - Mn.T)) / np.linalg.norm(Mn))


# ------------------------------------------------------------------------------------------------ module contract
def test_module_imports_without_orcfxapi():
    code = ("import sys; sys.modules['OrcFxAPI'] = None; import anyhydro.orcaflex_export as m; "
            "assert callable(m.vessel_type_data) and callable(m.write_vessel_type) and callable(m.export_vessel_type)")
    src = os.path.dirname(os.path.dirname(os.path.abspath(ox.__file__)))
    env = dict(os.environ, PYTHONPATH=src + os.pathsep + os.environ.get("PYTHONPATH", ""))
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_error_is_a_hydro_error():
    assert issubclass(ox.OrcaFlexExportError, HydroError)


def test_unit_factors_are_declared():
    f = ox.UNIT_FACTORS
    assert f["mass"] == f["inertia"] == f["added_mass"] == 1e-3
    assert f["force"] == f["damping"] == f["stiffness"] == f["load_rao"] == f["mean_drift"] == 1e-3
    assert f["displacement_rao_translation"] == 1.0 and f["displacement_rao_rotation"] == 1.0
    assert f["phase"] == "degrees(arg X), no sign change" and f["heading"].startswith("degrees(wave_direction) mod 360")


# ------------------------------------------------------------------------------------------------ pure data
def test_scalar_items_units_and_origins():
    data = export_data()
    s = data["scalars"]
    assert data["units_system"] == "SI"
    assert s["Mass"] == pytest.approx(RHO * 100.0 * 1e-3)
    assert [s["CentreOfMassX"], s["CentreOfMassY"], s["CentreOfMassZ"]] == list(COG)
    assert s["DisplacedVolume"] == pytest.approx(100.0, rel=1e-12)
    assert [s["CentreOfBuoyancyX"], s["CentreOfBuoyancyY"], s["CentreOfBuoyancyZ"]] == list(COB)
    assert [s["StiffnessInertiaDampingRefOriginx"], s["StiffnessInertiaDampingRefOriginy"],
            s["StiffnessInertiaDampingRefOriginz"]] == list(RC)
    assert s["HydrostaticReferenceOriginDatumPositionz"] == RC[2]
    assert s["HydrostaticStiffnessMethod"] == "Displacement" and s["AMDMethod"] == "Frequency dependent"
    assert s["Symmetry"] == "None"
    assert (s["WavesReferredToBy"], s["RAOPhaseConvention"], s["RAOPhaseUnitsConvention"],
            s["RAOPhaseRelativeToConvention"], s["RAOResponseUnits"], s["RAOWaveUnit"]) == (
        "frequency (rad/s)", "lags", "degrees", "crest", "radians", "amplitude")
    # diagonal inertia, only the upper triangle; kg m^2 -> te m^2
    assert data["indexed"]["MomentOfInertiaTensorX"] == [2.0e3]
    assert data["indexed"]["MomentOfInertiaTensorY"] == [0.0, 9.0e3]
    assert data["indexed"]["MomentOfInertiaTensorZ"] == [0.0, 0.0, 1.0e4]


def test_hydrostatic_stiffness_is_buoyancy_plus_gravity_hand_computed():
    arrays = default_arrays()
    data = export_data()
    W = RHO * 100.0 * G
    cz = COG[2] - RC[2]                                  # 1.5 m: the gravity block is -W cz on roll and pitch
    K = arrays["K"][2:5, 2:5] + np.diag([0.0, -W * cz, -W * cz])
    got = np.asarray(data["matrices"]["hydrostatic_stiffness"])
    assert got == pytest.approx(K * 1e-3, rel=1e-12)
    assert got[1, 1] == pytest.approx((3e6 - W * 1.5) * 1e-3)
    ix = data["indexed"]
    assert ix["HydrostaticStiffnessz"] == [pytest.approx(got[0, 0])]
    assert ix["HydrostaticStiffnessRx"] == pytest.approx([got[0, 1], got[1, 1]])
    assert ix["HydrostaticStiffnessRy"] == pytest.approx([got[0, 2], got[1, 2], got[2, 2]])
    assert data["provenance"]["hydrostatic_composition"]["gravity_roll_pitch_diagonal_kN"] == pytest.approx(-W * cz * 1e-3)


def test_surge_sway_yaw_stiffness_is_not_written_and_is_recorded():
    data = export_data()
    names = set(data["scalars"]) | set(data["indexed"])
    assert not {"HydrostaticStiffnessx", "HydrostaticStiffnessy", "HydrostaticStiffnessRz"} & names
    dropped = data["provenance"]["hydrostatic_composition"]["not_exported_entries_frobenius_ratio"]
    assert 0.0 < dropped < 1.0 and math.isfinite(dropped)      # the gravity yaw coupling 1/2 W c is dropped, and recorded


def test_physical_wrench_basis_uses_the_anyhydro_chart_conversion():
    from anyhydro import read_hydrostatic_contract
    from anyhydro.hydrostatic_contract import RATE_MAP_SPATIAL_EXP, chart_stiffness
    ds = make_dataset()
    base = read_hydrostatic_contract(ds)
    phys = HydrostaticContract(role="BUOYANCY_ONLY", force_basis="PHYSICAL_WRENCH", rate_map=RATE_MAP_SPATIAL_EXP,
                               stiffness_derivative=base.stiffness_derivative, bodies=base.bodies,
                               baseline_wrench=dict(base.baseline_wrench), provenance=dict(base.provenance),
                               full_source=None)
    ds2 = attach_hydrostatic_contract(make_dataset(contract=None), phys)
    data = export_data(ds2, length_scale=100.0)       # the chart skew terms are scaled by 1/L^2
    contract = read_hydrostatic_contract(ds2)
    chart = chart_stiffness(contract, BODY, default_arrays()["K"])
    sym = 0.5 * (chart + chart.T)
    W = RHO * 100.0 * G
    expect = sym[2:5, 2:5] + np.diag([0.0, -W * (COG[2] - RC[2]), -W * (COG[2] - RC[2])])
    assert np.asarray(data["matrices"]["hydrostatic_stiffness"]) == pytest.approx(expect * 1e-3, rel=1e-12)
    assert data["provenance"]["hydrostatic_composition"]["force_basis"] == "PHYSICAL_WRENCH"
    assert data["provenance"]["skew_ratios"]["K_buoyancy_chart"] > 0.0                  # the -1/2 skew(M) chart term


def test_load_rao_phase_amplitude_and_headings():
    arrays = default_arrays()
    arrays["X"][1, 2, 0] = 3.0e5 * np.exp(-2.0j)               # omega 0.5, heading 90 deg, surge
    arrays["X"][3, 5, 4] = 1.0e6 * np.exp(1.0j)                 # omega 0.9, heading 225 deg, pitch
    data = export_data(make_dataset(X=arrays["X"]))
    r = data["rao_sets"]["Wave load RAO"]
    assert list(r["directions_deg"]) == pytest.approx([0, 45, 90, 135, 180, 225, 270, 315])
    assert list(r["frequencies"]) == list(OMEGA)
    assert r["amplitude"].shape == (8, 5, 6) and r["phase_deg"].shape == (8, 5, 6)
    assert r["amplitude"][2, 1, 0] == pytest.approx(300.0)                    # kN/m
    assert r["phase_deg"][2, 1, 0] == pytest.approx(math.degrees(-2.0))       # arg X, no sign change
    assert r["amplitude"][5, 3, 4] == pytest.approx(1000.0)                   # kN.m/m
    assert r["phase_deg"][5, 3, 4] == pytest.approx(math.degrees(1.0))
    X = arrays["X"]
    assert np.allclose(r["amplitude"], np.abs(X.transpose(1, 0, 2)) * 1e-3)
    assert np.allclose(r["phase_deg"], np.degrees(np.angle(X.transpose(1, 0, 2))))


def test_rao_origin_phase_origin_and_infinite_frequency_row():
    data = export_data()
    load = data["rao_sets"]["Wave load RAO"]
    assert tuple(load["origin"]) == tuple(RC)                  # the excitation is a wrench about the rotation centre
    assert tuple(load["phase_origin"]) == (0.0, 0.0, 0.0)      # incident phase origin is the world origin, NOT the RAO origin
    assert load["infinite_frequency_row"] == "zero response"


def test_heading_mod_360_and_duplicate_2pi_is_dropped():
    headings = np.array([-np.pi / 2, 0.0, np.pi / 2, np.pi, 2.0 * np.pi])
    ds = make_dataset(headings=headings, duplicate_of=1)
    X = np.asarray(ds["excitation_force"].transpose("omega", "wave_direction", "influenced_dof").values)
    data = export_data(ds)
    r = data["rao_sets"]["Wave load RAO"]
    assert list(r["directions_deg"]) == pytest.approx([0.0, 90.0, 180.0, 270.0])
    assert np.allclose(r["amplitude"][3], np.abs(X[:, 0, :]) * 1e-3)       # -90 deg -> 270
    assert np.allclose(r["amplitude"][0], np.abs(X[:, 1, :]) * 1e-3)       # the first 0 is kept
    prov = data["provenance"]["headings"]
    assert prov["dropped_duplicates"] == [{"dropped_index": 4, "kept_index": 1, "direction_deg": 0.0}]


def test_headings_unsorted_are_sorted_and_gap_warning_is_recorded():
    data = export_data(make_dataset(headings=np.radians([180.0, 0.0])))
    assert list(data["rao_sets"]["Wave load RAO"]["directions_deg"]) == [0.0, 180.0]
    assert any("45" in w for w in data["provenance"]["warnings"])


def test_added_mass_damping_tables_units_and_infinite_row():
    arrays = default_arrays()
    data = export_data()
    amd = data["added_mass_damping"]
    assert list(amd["frequencies"]) == list(OMEGA)
    assert np.allclose(amd["added_mass"], arrays["A"] * 1e-3)
    assert np.allclose(amd["damping"], arrays["B"] * 1e-3)
    assert np.allclose(amd["added_mass_infinite"], arrays["Ainf"] * 1e-3)
    assert amd["infinite_frequency_row"] == "added_mass_infinite, damping zero"


def test_symmetrisation_and_skew_records():
    arrays = default_arrays()
    K = arrays["K"].copy()
    K[2, 4] += 1.0e3
    K[4, 2] -= 1.0e3
    A = arrays["A"].copy()
    A[2, 0, 3] += 2.0e3
    A[2, 3, 0] -= 2.0e3
    data = export_data(make_dataset(K=K, A=A))
    skew = data["provenance"]["skew_ratios"]
    assert skew["K_buoyancy_chart"] == pytest.approx(skew_ratio(K))
    assert skew["A_max_over_frequency"] == pytest.approx(max(skew_ratio(a) for a in A))
    assert skew["A_argmax_omega"] == OMEGA[2]
    assert skew["bound"] == 1e-2 and skew["B_max_over_frequency"] < 1e-12 and skew["A_inf"] < 1e-12
    # the written heave/pitch coupling is the mean of the two source entries
    assert data["matrices"]["hydrostatic_stiffness"][0][2] == pytest.approx(1e5 * 1e-3)
    sym = 0.5 * (A[2] + A[2].T)
    assert np.allclose(data["added_mass_damping"]["added_mass"][2], sym * 1e-3)


def test_mean_drift_tables():
    arrays = default_arrays()
    ds = make_dataset(with_drift=True)
    data = export_data(ds)
    q = data["rao_sets"]["Wave drift QTF"]
    assert q is not None and q["phase_deg"] is None
    assert tuple(q["origin"]) == tuple(RC)
    assert data["scalars"]["WaveDriftQTFMethod"] == "Newman's approximation"
    d = arrays["drift"].transpose(1, 0, 2) * 1e-3          # (heading, omega, [surge, sway, yaw])
    assert np.allclose(q["amplitude"][:, :, 0], d[:, :, 0]) and np.allclose(q["amplitude"][:, :, 1], d[:, :, 1])
    assert np.allclose(q["amplitude"][:, :, 5], d[:, :, 2])
    assert np.all(q["amplitude"][:, :, [2, 3, 4]] == 0.0)       # heave, roll, pitch drift declared zero
    assert (q["amplitude"] < 0).any()                           # negative drift is representable (probed)
    assert data["provenance"]["mean_drift"]["status"] == "exported"
    assert data["provenance"]["mean_drift"]["heave_roll_pitch"] == "zero (not provided by the far-field method)"


def test_mean_drift_absent_or_disabled_is_not_exported():
    absent = export_data()
    assert absent["rao_sets"]["Wave drift QTF"] is None
    assert absent["provenance"]["mean_drift"]["status"] == "not present in the dataset"
    off = export_data(make_dataset(with_drift=True), include_mean_drift=False)
    assert off["rao_sets"]["Wave drift QTF"] is None
    assert off["provenance"]["mean_drift"]["status"] == "excluded by the caller"


def test_displacement_raos_optional_input():
    assert export_data()["rao_sets"]["Displacement RAO"] is None
    assert export_data()["provenance"]["displacement_raos"] == "not supplied"
    rng = np.random.default_rng(1)
    mot = (rng.normal(size=(5, 8, 6)) + 1j * rng.normal(size=(5, 8, 6))) * np.array([0.5, 0.5, 0.5, 0.02, 0.02, 0.02])
    mot[0, 3, 3] = 0.02 * np.exp(0.5j)                         # roll, rad/m
    data = export_data(displacement_raos=mot)
    r = data["rao_sets"]["Displacement RAO"]
    assert tuple(r["origin"]) == tuple(RC) and tuple(r["phase_origin"]) == (0.0, 0.0, 0.0)
    assert r["amplitude"].shape == (8, 5, 6)
    assert np.allclose(r["amplitude"], np.abs(mot.transpose(1, 0, 2)))
    assert np.allclose(r["phase_deg"], np.degrees(np.angle(mot.transpose(1, 0, 2))))
    assert r["amplitude"][3, 0, 3] == pytest.approx(0.02)       # rotations stay in rad/m (RAOResponseUnits = radians)
    assert r["phase_deg"][3, 0, 3] == pytest.approx(math.degrees(0.5))
    assert data["provenance"]["displacement_raos"]["units"] == "m/m translations, rad/m rotations about the rotation centre"


def test_displacement_raos_as_dataarray_must_match_the_grid():
    import xarray as xr
    ds = make_dataset()
    keys = [f"{BODY}::{d}" for d in DOFS]
    mot = np.ones((5, 8, 6), dtype=complex)
    good = xr.DataArray(mot, dims=("omega", "wave_direction", "radiating_dof"),
                        coords={"omega": OMEGA, "wave_direction": HEADINGS, "radiating_dof": keys})
    assert export_data(ds, displacement_raos=good)["rao_sets"]["Displacement RAO"] is not None
    bad = good.assign_coords(omega=OMEGA * (1 + 1e-9))
    with pytest.raises(ox.OrcaFlexExportError, match="omega"):
        export_data(ds, displacement_raos=bad)
    with pytest.raises(ox.OrcaFlexExportError, match="shape"):
        export_data(ds, displacement_raos=np.ones((5, 7, 6), dtype=complex))
    with pytest.raises(ox.OrcaFlexExportError, match="finite"):
        export_data(ds, displacement_raos=np.full((5, 8, 6), np.nan, dtype=complex))


def test_provenance_record():
    ds = make_dataset(with_drift=True)
    data = export_data(ds)
    p = data["provenance"]
    assert p["format"] == ox.FORMAT
    assert p["dataset_content_hash"] == dataset_content_hash(ds)
    assert p["conventions"]["phasor"] == "x(t) = Re{X * exp(-i*omega*t)}"
    assert p["conventions"]["orcaflex"]["units_system"] == "SI"
    assert p["unit_factors"] == ox.UNIT_FACTORS
    assert p["declared"]["gravity"] == G and p["declared"]["length_scale"] == LENGTH
    assert p["declared"]["mass_properties"]["mass_kg"] == RHO * 100.0
    assert p["declared"]["rho"] == RHO and p["declared"]["displaced_volume_m3"] == pytest.approx(100.0)
    assert p["omega_band_rad_s"] == [OMEGA[0], OMEGA[-1]]
    assert p["single_body"] == BODY
    assert json.loads(json.dumps(p)) == p                                  # plain JSON data
    json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in data["scalars"].items()})


def test_mass_balance_boundary_is_one_per_mille():
    export_data(props=mass_props(mass=RHO * 100.0 * 1.0009))
    export_data(props=mass_props(mass=RHO * 100.0 * 0.9991))
    with pytest.raises(ox.OrcaFlexExportError, match="mass"):
        export_data(props=mass_props(mass=RHO * 100.0 * 1.0011))
    with pytest.raises(ox.OrcaFlexExportError, match="mass"):
        export_data(props=mass_props(mass=RHO * 100.0 * 0.9989))


# ------------------------------------------------------------------------------------------------ refusals
def test_refuses_multibody():
    with pytest.raises(ox.OrcaFlexExportError, match="single body|multibody"):
        export_data(make_two_body_fixture())


def test_refuses_non_global_frames():
    ds = reframe_dataset(make_dataset(contract=None), new_axes={BODY: np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])})
    with pytest.raises(ox.OrcaFlexExportError, match="GLOBAL|global"):
        export_data(ds)


def test_refuses_missing_a_inf():
    with pytest.raises(ox.OrcaFlexExportError, match="added_mass_infinite"):
        export_data(make_dataset(with_ainf=False))


def test_refuses_missing_or_unusable_contract():
    with pytest.raises(ox.OrcaFlexExportError, match="contract|UNKNOWN"):
        export_data(make_dataset(contract=None))
    with pytest.raises(ox.OrcaFlexExportError, match="BUOYANCY_ONLY|FULL|composition"):
        export_data(make_dataset(role="FULL"))


def test_refuses_gravity_mismatch():
    with pytest.raises(ox.OrcaFlexExportError, match="gravity"):
        export_data(gravity=9.80665)
    with pytest.raises(ox.OrcaFlexExportError, match="gravity"):
        export_data(make_dataset(prov_gravity=None))                       # the contract must declare its g
    with pytest.raises(ox.OrcaFlexExportError, match="gravity"):
        export_data(make_dataset(with_drift=True, prov_gravity=9.8))       # contract vs declared
    with pytest.raises(ox.OrcaFlexExportError, match="gravity"):
        export_data(gravity=float("nan"))


def test_refuses_unknown_density():
    with pytest.raises(ox.OrcaFlexExportError, match="rho"):
        export_data(make_dataset(prov_rho=None))


def test_refuses_mass_imbalance():
    with pytest.raises(ox.OrcaFlexExportError, match="mass"):
        export_data(props=mass_props(mass=RHO * 100.0 * 1.02))


def test_refuses_horizontal_cog_cob_misalignment():
    with pytest.raises(ox.OrcaFlexExportError, match="equilibrium"):
        export_data(props=mass_props(cog=COG + np.array([1.0, 0.0, 0.0])))


def test_refuses_skew_above_the_bound():
    arrays = default_arrays()
    K = arrays["K"].copy()
    K[2, 4], K[4, 2] = 3.0e6, -3.0e6
    with pytest.raises(ox.OrcaFlexExportError, match="K_buoyancy_chart"):
        export_data(make_dataset(K=K))
    A = arrays["A"].copy()
    A[1, 0, 1], A[1, 1, 0] = 6.0e5, -6.0e5
    with pytest.raises(ox.OrcaFlexExportError, match="added mass"):
        export_data(make_dataset(A=A))
    B = arrays["B"].copy()
    B[3, 2, 4], B[3, 4, 2] = 1.0e5, -1.0e5
    with pytest.raises(ox.OrcaFlexExportError, match="damping"):
        export_data(make_dataset(B=B))
    Ainf = arrays["Ainf"].copy()
    Ainf[0, 2], Ainf[2, 0] = 2.0e6, -2.0e6
    with pytest.raises(ox.OrcaFlexExportError, match="A_inf"):
        export_data(make_dataset(Ainf=Ainf))


def test_refuses_non_diagonal_inertia():
    inertia = np.diag([2.0e6, 9.0e6, 1.0e7])
    inertia[0, 2] = inertia[2, 0] = 1.0e5
    with pytest.raises(ox.OrcaFlexExportError, match="products of inertia"):
        export_data(props=mass_props(inertia=inertia))


def test_refuses_non_finite_and_invalid_declarations():
    with pytest.raises(ox.OrcaFlexExportError, match="finite"):
        mass_props(mass=float("nan"))
    with pytest.raises(ox.OrcaFlexExportError, match="finite"):
        mass_props(cog=(0.0, float("inf"), 0.0))
    with pytest.raises(ox.OrcaFlexExportError, match="positive"):
        mass_props(mass=-1.0)
    with pytest.raises(ox.OrcaFlexExportError, match="symmetric"):
        mass_props(inertia=np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))
    with pytest.raises(ox.OrcaFlexExportError, match="length_scale"):
        export_data(length_scale=0.0)
    with pytest.raises(ox.OrcaFlexExportError, match="length_scale"):
        export_data(length_scale=float("nan"))
    bad = make_dataset(contract=None)
    bad["added_mass"].values[0, 0, 0] = np.nan
    with pytest.raises(ox.OrcaFlexExportError):
        export_data(bad)


def test_refuses_unknown_phasor_and_heading_convention():
    ds = make_dataset(contract=None)
    ds.attrs["phasor_convention"] = "x(t) = Re{X * exp(+i*omega*t)}"
    with pytest.raises(ox.OrcaFlexExportError, match="phasor"):
        export_data(ds)
    ds = make_dataset(contract=None)
    ds.attrs["heading_convention"] = "wave source direction"
    with pytest.raises(ox.OrcaFlexExportError, match="heading"):
        export_data(ds)


def test_refuses_a_single_heading_and_conflicting_duplicates():
    with pytest.raises(ox.OrcaFlexExportError, match="two"):
        export_data(make_dataset(headings=np.array([0.0])))
    arrays = default_arrays(np.array([0.0, np.pi / 2, 2.0 * np.pi]))
    with pytest.raises(ox.OrcaFlexExportError, match="duplicate"):
        export_data(make_dataset(headings=np.array([0.0, np.pi / 2, 2.0 * np.pi]), X=arrays["X"]))


# ------------------------------------------------------------------------------------------------ OrcFxAPI
@pytest.fixture()
def orcfx():
    o = pytest.importorskip("OrcFxAPI")
    try:
        model = o.Model()
        model.CreateObject(o.ObjectType.VesselType)
    except Exception as error:                      # noqa: BLE001 - any licence or runtime failure skips cleanly
        pytest.skip(f"OrcFxAPI is installed but not usable here: {str(error)[:120]}")
    return o


def _new_model(o, **env):
    model = o.Model()
    model.threadCount = 1
    g = model.general
    g.StageCount = 2
    g.StageDuration[0] = 10.0
    g.StageDuration[1] = env.pop("duration", 60.0)
    g.ImplicitConstantTimeStep = env.pop("dt", 0.05)
    e = model.environment
    e.SeabedOriginDepth = 1000.0
    e.WaveType = "Airy"
    for k, v in env.items():
        setattr(e, k, v)
    return model


def _read_table(vt, which, index):
    vt.SelectedRAOs = which
    vt.SelectedRAODirectionIndex = index
    rows = {}
    for r in range(vt.GetDataRowCount("RAOPeriodOrFrequency")):
        w = vt.GetData("RAOPeriodOrFrequency", r)
        amp = [vt.GetData(f"RAO{n}Amp", r) for n in ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")]
        try:
            ph = [vt.GetData(f"RAO{n}Phase", r) for n in ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw")]
        except Exception:                           # noqa: BLE001 - drift QTF tables carry no phase
            ph = None
        rows[w] = (np.array(amp), None if ph is None else np.array(ph))
    return vt.RAODirection, rows


def _compare_rao_set(o, vt, which, spec, phase=True):
    vt.SelectedRAOs = which
    assert vt.NumberOfRAODirections == len(spec["directions_deg"])
    by_dir = {}
    for i in range(vt.NumberOfRAODirections):
        d, rows = _read_table(vt, which, i)
        by_dir[round(d, 6)] = rows
    for k, d in enumerate(spec["directions_deg"]):
        rows = by_dir[round(float(d), 6)]                                   # by value: saved tables are re-sorted
        for j, w in enumerate(spec["frequencies"]):
            amp, ph = rows[float(w)]
            assert amp == pytest.approx(spec["amplitude"][k, j], rel=1e-9, abs=1e-12)
            if phase and spec["phase_deg"] is not None:
                z = np.exp(1j * np.radians(ph)) * np.where(amp == 0.0, 0.0, 1.0)
                z0 = np.exp(1j * np.radians(spec["phase_deg"][k, j])) * np.where(spec["amplitude"][k, j] == 0.0, 0.0, 1.0)
                assert np.allclose(z, z0, atol=1e-9)
        assert rows[o.OrcinaInfinity()][0] == pytest.approx(np.zeros(6))      # zero response at infinite frequency


def _check_vessel_type(o, vt, data):
    vt.SelectedDraught = vt.GetData("DraughtName", 0)
    for name, value in data["scalars"].items():
        got = vt.GetData(name, -1)
        if isinstance(value, str):
            assert str(got).strip().lower() == value.lower(), name
        else:
            assert got == pytest.approx(value, rel=1e-12, abs=1e-12), name
    for name, rows in data["indexed"].items():
        for i, value in enumerate(rows):
            assert vt.GetData(name, i) == pytest.approx(value, rel=1e-12, abs=1e-12), (name, i)
    for which in ("Wave load RAO", "Displacement RAO", "Wave drift QTF"):
        spec = data["rao_sets"][which]
        if spec is None:
            continue
        vt.SelectedRAOs = which
        assert [vt.RAOOriginX, vt.RAOOriginY, vt.RAOOriginZ] == pytest.approx(list(spec["origin"]))
        if spec["phase_origin"] is not None:
            assert [vt.RAOPhaseOriginX, vt.RAOPhaseOriginY, vt.RAOPhaseOriginZ] == pytest.approx(list(spec["phase_origin"]))
        _compare_rao_set(o, vt, which, spec, phase=which != "Wave drift QTF")
    amd = data["added_mass_damping"]
    rows = {vt.GetData("AMDPeriodOrFrequency", i): i for i in range(vt.NumberOfAddedMassAndDampingPeriodsOrFrequencies)}
    assert set(rows) == {o.OrcinaInfinity(), *[float(w) for w in amd["frequencies"]]}
    cols = ("X", "Y", "Z", "Rx", "Ry", "Rz")
    for key, m_a, m_d in [(float(w), amd["added_mass"][j], amd["damping"][j]) for j, w in enumerate(amd["frequencies"])] + [
            (o.OrcinaInfinity(), amd["added_mass_infinite"], np.zeros((6, 6)))]:
        vt.SelectedAddedMassAndDampingPeriodOrFrequencyIndex = rows[key]
        for c, name in enumerate(cols):
            for r in range(6):
                assert vt.GetData("AddedMassMatrix" + name, r) == pytest.approx(m_a[r, c], rel=1e-12, abs=1e-12)
                assert vt.GetData("Damping" + name, r) == pytest.approx(m_d[r, c], rel=1e-12, abs=1e-12)


def _stored_provenance(vt):
    return json.loads(vt.tags[ox.PROVENANCE_TAG])


def test_round_trip_through_orcfxapi(orcfx, tmp_path):
    o = orcfx
    ds = make_dataset(with_drift=True)
    arrays = default_arrays()
    mot = (arrays["X"] * 1e-6).astype(complex)
    model = _new_model(o)
    vt = ox.export_vessel_type(model, ds, mass_props(), name="Barge", length_scale=LENGTH, gravity=G, displacement_raos=mot)
    data = export_data(ds, displacement_raos=mot)
    assert vt.Name == "Barge"
    _check_vessel_type(o, vt, data)
    stored = _stored_provenance(vt)
    assert stored["dataset_content_hash"] == dataset_content_hash(ds)
    assert {k: v for k, v in stored.items() if k != "writer"} == data["provenance"]
    assert stored["writer"]["OrcFxAPI_DLL_version"] == o.DLLVersion()
    assert vt.tags[ox.HASH_TAG] == dataset_content_hash(ds)
    path = str(tmp_path / "vessel.dat")
    model.SaveData(path)
    reloaded = o.Model(path)
    vt2 = reloaded["Barge"]
    _check_vessel_type(o, vt2, data)
    assert _stored_provenance(vt2) == stored


def test_round_trip_without_optional_blocks_resets_orcina_defaults(orcfx):
    o = orcfx
    model = _new_model(o)
    vt = ox.export_vessel_type(model, make_dataset(), mass_props(), name="Plain", length_scale=LENGTH, gravity=G)
    data = export_data()
    _check_vessel_type(o, vt, data)
    for which in ("Displacement RAO", "Wave drift QTF"):             # reset, not left at Orcina's default vessel
        vt.SelectedRAOs = which
        for i in range(vt.NumberOfRAODirections):
            vt.SelectedRAODirectionIndex = i
            for r in range(vt.GetDataRowCount("RAOPeriodOrFrequency")):
                assert all(vt.GetData(f"RAO{n}Amp", r) == 0.0 for n in ("Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"))
    assert vt.GetData("NumberOfDraughts", -1) == 1
    assert vt.Mass == pytest.approx(RHO * 100.0 * 1e-3)


def test_writer_refuses_a_non_si_model_and_duplicate_names(orcfx):
    o = orcfx
    model = _new_model(o)
    model.general.UnitsSystem = "US"
    with pytest.raises(ox.OrcaFlexExportError, match="SI"):
        ox.export_vessel_type(model, make_dataset(), mass_props(), name="V", length_scale=LENGTH, gravity=G)
    model = _new_model(o)
    ox.export_vessel_type(model, make_dataset(), mass_props(), name="V", length_scale=LENGTH, gravity=G)
    with pytest.raises(ox.OrcaFlexExportError, match="name"):
        ox.export_vessel_type(model, make_dataset(), mass_props(), name="V", length_scale=LENGTH, gravity=G)


def test_writer_refuses_an_environment_density_that_differs_from_the_dataset(orcfx):
    model = _new_model(orcfx)
    model.environment.Density = 1.0
    with pytest.raises(ox.OrcaFlexExportError, match="density"):
        ox.export_vessel_type(model, make_dataset(), mass_props(), name="V", length_scale=LENGTH, gravity=G)


def _fit(t, h, w):
    tn = (t - t.mean()) / (t.max() - t.min())
    cols = [np.cos(w * t), np.sin(w * t), np.ones_like(t)] + [tn ** k for k in range(1, 4)]
    (a, b, *_), *_ = np.linalg.lstsq(np.column_stack(cols), h, rcond=None)
    return np.hypot(a, b) * np.exp(1j * np.arctan2(b, a))          # x = R cos(w t - phi) = Re{R e^{i phi} e^{-i w t}}


@pytest.mark.parametrize("iw,ih", [(2, 1), (4, 6)])
def test_end_to_end_load_rao_time_history_and_moment_transfer(orcfx, iw, ih):
    """The exported vessel in an OrcaFlex regular wave reproduces Re{X exp(-i w t)} (amplitude ratio 1) and the
    moment transfer M_c + r_c x F about the vessel origin (o1a_probe_g, G1)."""
    o = orcfx
    volume = 2.0e9 / RHO                                              # a very heavy vessel: motions stay ~1e-4
    ds = make_dataset(volume=volume, heavy=True)
    w, beta = float(OMEGA[iw]), float(HEADINGS[ih])
    model = _new_model(o, WavePeriod=2.0 * math.pi / w, WaveHeight=2.0,
                       WaveDirection=(math.degrees(beta) if math.degrees(beta) <= 180 else math.degrees(beta) - 360.0))
    vt = ox.export_vessel_type(model, ds, mass_props(volume, inertia=np.diag([2e11, 1e12, 1e12])), name="Heavy",
                               length_scale=LENGTH, gravity=G)
    v = model.CreateObject(o.ObjectType.Vessel)
    v.VesselType = vt.Name
    v.PrimaryMotion = "Calculated (6 DOF)"
    v.SuperimposedMotion = "None"
    v.IncludeWaveLoad1stOrder = "Yes"
    v.IncludeWaveDriftLoad2ndOrder = "No"
    v.IncludeAddedMassAndDamping = "No"
    model.RunSimulation()
    per = o.Period(1)
    t = v.SampleTimes(per)
    got = np.array([_fit(t, v.TimeHistory("Wave (1st order) " + n, per), w)
                    for n in ("Lx force", "Ly force", "Lz force", "Lx moment", "Ly moment", "Lz moment")])
    Xc = np.asarray(ds["excitation_force"].transpose("omega", "wave_direction", "influenced_dof").values)[iw, ih] * 1e-3
    expected = np.concatenate([Xc[:3], Xc[3:] + np.cross(RC, Xc[:3])])
    assert np.abs(got) / np.abs(expected) == pytest.approx(np.ones(6), abs=5e-3)
    artefact = math.degrees(0.0143 * w)                                # documented result time stamp offset (O1a B0)
    assert np.degrees(np.angle(got / expected)) == pytest.approx(np.full(6, artefact), abs=0.4)
