# SPDX-License-Identifier: MPL-2.0
"""Newman difference-frequency QTF from the far-field mean drift (docs/MEAN_DRIFT_CONTRACT.md section 8, slice N1)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from anyhydro.mean_drift import MEAN_DRIFT_FORMAT
from anyhydro.schema import MEAN_DRIFT_UNITS
from anyhydro.slow_drift import NEWMAN_QTF_FORMAT, SlowDriftError, newman_qtf, newman_values, qtf_at

OMEGA = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
HEADINGS = np.array([0.0, np.pi / 2])
CENTRE = [1.0, -0.5, 0.0]


def _drift():
    w = OMEGA[:, None, None]
    return np.concatenate([1e4 * w ** 2 * np.ones((1, 2, 1)),                        # surge: positive, rising
                           -3e3 * w * np.ones((1, 2, 1)),                             # sway: negative
                           np.array([-50.0, -10.0, 20.0, 40.0, 60.0])[:, None, None] * np.ones((1, 2, 1))],
                          axis=2)                                                      # yaw: sign change


def _dataset(drift=None, headings=HEADINGS, dofs=("b::surge", "b::sway", "b::yaw"), record=None):
    drift = _drift() if drift is None else drift
    ds = xr.Dataset({"mean_drift_far_field": (("omega", "wave_direction", "drift_dof"), drift),
                     "rotation_center": (("body", "xyz"), np.array([CENTRE]))},
                    coords={"omega": OMEGA, "wave_direction": headings, "drift_dof": list(dofs), "body": ["b"],
                            "xyz": ["x", "y", "z"]})
    meta = dict(format=MEAN_DRIFT_FORMAT, method="far_field_momentum_conservation", units=MEAN_DRIFT_UNITS,
                yaw_reference=CENTRE, motions=dict(source="fixed", motion_hash="x"))
    if record is not None:
        meta = record(meta)
    if meta is not None:
        ds.attrs["mean_drift_far_field"] = json.dumps(meta)
    return ds


@pytest.mark.parametrize("method", ["geometric", "arithmetic"])
def test_diagonal_is_the_mean_drift_and_the_qtf_is_symmetric(method):
    ds = _dataset()
    q = newman_qtf(ds, method)["newman_qtf"].values
    D = ds["mean_drift_far_field"].values
    assert np.array_equal(np.einsum("iihk->ihk", q), D)
    assert np.array_equal(q, np.transpose(q, (1, 0, 2, 3)))


def test_geometric_rule_values_signs_and_zeros():
    D = np.array([4.0, 9.0, -1.0, -16.0, 0.0, -0.0])
    T = newman_values(D, "geometric")
    assert T[0, 1] == pytest.approx(6.0)                    # sqrt(4 * 9), positive
    assert T[2, 3] == pytest.approx(-4.0)                   # both negative: -sqrt(16)
    assert T[1, 2] == 0.0 and T[2, 1] == 0.0                # opposite signs -> 0 (declared)
    assert T[0, 4] == 0.0 and T[4, 4] == 0.0 and T[5, 5] == 0.0
    assert newman_values(D, "arithmetic")[1, 2] == pytest.approx(4.0)


@pytest.mark.parametrize("method", ["geometric", "arithmetic"])
def test_extreme_finite_values_keep_an_exact_diagonal(method):
    """Review round 1: no intermediate overflow or underflow."""
    D = np.array([1e200, 1e-200, 1e308, -1e308, np.nextafter(0.0, 1.0)])     # incl. the smallest subnormal
    T = newman_values(D, method)
    assert np.all(np.isfinite(T))
    assert np.array_equal(np.diag(T), D)
    if method == "geometric":
        assert T[0, 1] == pytest.approx(1.0)                # sqrt(1e200 * 1e-200)


def test_complex_drift_is_refused_not_truncated():
    with pytest.raises(SlowDriftError, match="real"):
        newman_values(np.array([1.0 + 2.0j, 3.0]))


def test_record_binds_the_source_the_array_and_the_method():
    ds = _dataset()
    out = newman_qtf(ds, "geometric")
    rec = json.loads(out.attrs["newman_qtf"])
    assert rec["format"] == NEWMAN_QTF_FORMAT and rec["method"] == "geometric"
    assert rec["source_mean_drift"]["format"] == MEAN_DRIFT_FORMAT
    changed = ds.copy(deep=True)
    changed["mean_drift_far_field"].values[0, 0, 0] *= 1.001
    assert json.loads(newman_qtf(changed).attrs["newman_qtf"])["drift_sha256"] != rec["drift_sha256"]


def test_lookup_uses_each_axis_and_grid_points_only():
    out = newman_qtf(_dataset())
    T = qtf_at(out, 0.5, 0.7)
    assert T.shape == (2, 3)
    assert T[0, 0] == pytest.approx(1e4 * 0.5 * 0.7)
    sliced = out.isel(omega_2=slice(1, None))                # review round 1: axes are searched independently
    assert np.array_equal(qtf_at(sliced, 0.4, 0.5), qtf_at(out, 0.4, 0.5))
    with pytest.raises(SlowDriftError, match="not a grid point"):
        qtf_at(out, 0.55, 0.7)
    with pytest.raises(SlowDriftError, match="not a grid point of omega_2"):
        qtf_at(sliced, 0.4, 0.4)


@pytest.mark.parametrize("change, message", [
    (dict(record=lambda m: None), "record"),
    (dict(drift=np.full((5, 2, 3), np.nan)), "finite"),
    (dict(dofs=("b::surge", "b::heave", "b::yaw")), "qualified"),
    (dict(record=lambda m: dict(m, units="kN")), "units"),
    (dict(record=lambda m: dict(m, motions={})), "motion"),
    (dict(record=lambda m: dict(m, yaw_reference=[0.0, 0.0, 0.0])), "yaw reference"),
    (dict(headings=np.array([0.0, 2 * np.pi])), "duplicate headings"),
    (dict(headings=np.array([0.0, np.nan])), "finite"),
])
def test_inadmissible_mean_drift_blocks_are_refused(change, message):
    with pytest.raises(SlowDriftError, match=message):
        newman_qtf(_dataset(**change))


def test_other_refusals():
    with pytest.raises(SlowDriftError, match="unknown Newman method"):
        newman_qtf(_dataset(), "harmonic")
    with pytest.raises(SlowDriftError, match="lacks 'mean_drift_far_field'"):
        newman_qtf(xr.Dataset())
    for omega in (OMEGA[::-1], np.array([0.4, 0.5, 0.5, 0.7, 0.8])):
        with pytest.raises(SlowDriftError, match="strictly increasing"):
            newman_qtf(_dataset().assign_coords(omega=omega))


# ---------------------------------------------------------------------------------------- N2b: stored datasets

def _full_dataset():
    """A full schema 1.3.0 single-body dataset (synthetic fixture body; zero Kochin functions, as the Capytaine
    export lays them out) with the mean-drift block; JSON attributes as JSON strings, the computed form."""
    from anyhydro import fixtures
    from anyhydro.records import HydroRequest
    from anyhydro.schema import KOCHIN_SCHEMA_VERSION, assemble_hydro_dataset
    body = fixtures.make_fixture_bodies()[0]
    request = HydroRequest(omegas=OMEGA.copy(), wave_directions=HEADINGS.copy(), water_depth=np.inf, rho=1025.0,
                           gravity=9.81, include_zero_frequency=False, include_infinite_frequency=False,
                           backend_settings={"fixture": "n2b-stored"})
    n = len(OMEGA)
    A = np.tile(np.eye(6), (n, 1, 1)) * 1e3
    ds = assemble_hydro_dataset(
        (body,), request, added_mass=A, radiation_damping=0.1 * A, excitation_force=np.ones((n, 2, 6), complex),
        hydrostatic_stiffness=np.eye(6), rotation_center=np.array([body.rotation_center]),
        center_of_mass=np.array([body.center_of_mass]), center_of_buoyancy=np.array([body.rotation_center]),
        added_mass_infinite=None, added_mass_infinite_source=None, backend="synthetic", backend_version="1",
        backend_commit="unknown", qualification_status="development_fixture")
    n_theta = 720
    ds = ds.assign_coords(theta=2.0 * np.pi * np.arange(n_theta) / n_theta,
                          drift_dof=[f"{body.body_id}::{d}" for d in ("surge", "sway", "yaw")])
    ds["kochin_radiation"] = (("omega", "radiating_dof", "theta"), np.zeros((n, 6, n_theta), complex))
    ds["kochin_diffraction"] = (("omega", "wave_direction", "theta"), np.zeros((n, 2, n_theta), complex))
    ds["wavenumber"] = (("omega",), OMEGA ** 2 / 9.81)
    for name, value in (("water_depth", np.inf), ("rho", 1025.0), ("g", 9.81)):
        ds[name] = ((), np.float64(value))
    ds.attrs["wave_environment"] = json.dumps(dict(water_depth="inf", rho=1025.0, g=9.81))
    ds.attrs["kochin_export"] = json.dumps(dict(route="zero test fixture", reference_point=[0.0, 0.0],
                                                theta_count=n_theta))
    ds.attrs["schema_version"] = KOCHIN_SCHEMA_VERSION
    ds["mean_drift_far_field"] = (("omega", "wave_direction", "drift_dof"), _drift())
    ds.attrs["mean_drift_far_field"] = json.dumps(dict(
        format=MEAN_DRIFT_FORMAT, method="far_field_momentum_conservation", units=MEAN_DRIFT_UNITS,
        yaw_reference=[float(v) for v in body.rotation_center], motions=dict(source="fixed", motion_hash="x")))
    return ds


def test_a_stored_dataset_hashes_alike_and_gives_the_same_qtf(tmp_path):
    """A loaded dataset carries the JSON records decoded: the QTF and the content hash must not depend on that."""
    from anyhydro import dataset_content_hash, load_hydro_dataset, save_hydro_dataset
    ds = _full_dataset()
    save_hydro_dataset(ds, tmp_path / "ds.nc")
    loaded = load_hydro_dataset(tmp_path / "ds.nc")
    assert isinstance(loaded.attrs["mean_drift_far_field"], dict)
    assert dataset_content_hash(loaded) == dataset_content_hash(ds)
    for method in ("geometric", "arithmetic"):
        a, b = newman_qtf(ds, method), newman_qtf(loaded, method)
        assert np.array_equal(a["newman_qtf"].values, b["newman_qtf"].values)
        assert json.loads(a.attrs["newman_qtf"]) == json.loads(b.attrs["newman_qtf"])


def test_a_record_that_is_neither_json_text_nor_a_mapping_is_refused():
    ds = _dataset()
    ds.attrs["mean_drift_far_field"] = "{not json"
    with pytest.raises(SlowDriftError, match="not JSON"):
        newman_qtf(ds)
    ds.attrs["mean_drift_far_field"] = 3
    with pytest.raises(SlowDriftError, match="record"):
        newman_qtf(ds)
