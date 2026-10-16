# SPDX-License-Identifier: MPL-2.0
"""Kochin export (Capytaine backend) and far-field mean drift on real solves: admission, provenance, persistence."""
from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from anyhydro import mean_drift as md
from anyhydro.errors import CapabilityError, SchemaError

capytaine = pytest.importorskip("capytaine", reason="optional dependency not installed")

from anyhydro import CapytaineBackend                               # noqa: E402
from anyhydro.capytaine_backend import CapytaineExecutionLimits    # noqa: E402
from anyhydro.persistence import load_hydro_dataset, save_hydro_dataset   # noqa: E402
from anyhydro.transform import shift_dataset_reference_point        # noqa: E402
from anyhydro.errors import TransformError                          # noqa: E402

from test_capytaine_backend import _box_body, _dev_request           # noqa: E402

CENTER = (0.3, -0.2, -0.3)


def _solve(settings, depth=float("inf"), omegas=(0.8, 1.2), headings=(0.0,)):
    body = _box_body("b0", CENTER)
    limits = CapytaineExecutionLimits(max_total_panels=100, max_finite_omegas=len(omegas),
                                      max_wave_directions=len(headings))
    request = _dev_request(omegas=np.array(omegas), wave_directions=np.array(headings), water_depth=depth,
                           backend_settings=settings)
    return CapytaineBackend(execution_limits=limits).solve([body], request)


@pytest.fixture(scope="module")
def exported():
    return _solve({"kochin": "export", "method": "direct"})


def test_legacy_requests_are_unchanged():
    ds = _solve({"method": "direct"})
    assert ds.attrs["schema_version"] == "1.1.0"
    assert not any(name in ds for name in ("kochin_radiation", "kochin_diffraction", "wavenumber"))


def test_unknown_kochin_setting_is_refused():
    with pytest.raises(CapabilityError, match="kochin"):
        _solve({"kochin": "yes"})


def test_export_carries_world_origin_kochin_and_environment(exported):
    ds = exported
    assert ds.attrs["schema_version"] == "1.3.0"
    assert ds["kochin_radiation"].dims == ("omega", "radiating_dof", "theta")
    assert ds["kochin_diffraction"].dims == ("omega", "wave_direction", "theta")
    export = json.loads(ds.attrs["kochin_export"])
    assert export["reference_point"] == [0.0, 0.0] and export["theta_count"] == ds["theta"].size
    env = json.loads(ds.attrs["wave_environment"])
    assert env == {"water_depth": "inf", "rho": 1025.0, "g": 9.81}
    # the exported radiation Kochin reproduces the stored far-field damping
    from anyhydro.green_function import far_field_damping
    again = far_field_damping(ds["kochin_radiation"].values, ds["omega"].values, ds["wavenumber"].values,
                              float("inf"), 1025.0)
    assert np.allclose(again, ds["radiation_damping_far_field"].values, rtol=1e-12,
                       atol=1e-12 * np.abs(again).max())


def test_export_round_trips_through_persistence(exported, tmp_path):
    path = tmp_path / "k.nc"
    save_hydro_dataset(exported, path)
    back = load_hydro_dataset(path)
    assert back.attrs["schema_version"] == "1.3.0"
    assert np.array_equal(back["kochin_diffraction"].values, exported["kochin_diffraction"].values)
    drift = md.far_field_mean_drift(back, _zero_motions(back), {"source": "fixed"})
    save_hydro_dataset(drift, tmp_path / "d.nc")
    again = load_hydro_dataset(tmp_path / "d.nc")
    assert np.array_equal(again["mean_drift_far_field"].values, drift["mean_drift_far_field"].values)


def _zero_motions(ds, scale=0.0):
    shape = (ds["omega"].size, ds["wave_direction"].size, ds["radiating_dof"].size)
    return xr.DataArray(np.full(shape, scale, dtype=complex), dims=("omega", "wave_direction", "radiating_dof"),
                        coords={n: ds[n].values for n in ("omega", "wave_direction", "radiating_dof")})


def test_fixed_body_drift_and_yaw_reference(exported):
    ds = md.far_field_mean_drift(exported, _zero_motions(exported), {"source": "fixed"})
    F = ds["mean_drift_far_field"].values
    assert F.shape == (2, 1, 3) and np.all(np.isfinite(F))
    meta = json.loads(ds.attrs["mean_drift_far_field"])
    assert meta["yaw_reference"] == list(CENTER) and meta["motions"]["source"] == "fixed"
    assert meta["depth_treatment"] == ["infinite"]


def test_motion_admission_is_exact(exported):
    bad = _zero_motions(exported).assign_coords(omega=exported["omega"].values * (1 + 1e-12))
    with pytest.raises(md.MeanDriftError, match="omega"):
        md.far_field_mean_drift(exported, bad, {"source": "x"})
    with pytest.raises(md.MeanDriftError, match="source"):
        md.far_field_mean_drift(exported, _zero_motions(exported), {})
    nan = _zero_motions(exported, scale=np.nan)
    with pytest.raises(md.MeanDriftError, match="finite"):
        md.far_field_mean_drift(exported, nan, {"source": "x"})


def test_reference_shift_refuses_kochin_datasets(exported):
    with pytest.raises(TransformError, match="reference shift refused"):
        shift_dataset_reference_point(exported, {"b0": (0.0, 0.0, 0.0)})


def test_finite_depth_export_gives_finite_depth_drift():
    ds = _solve({"kochin": "export", "method": "direct"}, depth=3.0)
    assert json.loads(ds.attrs["wave_environment"])["water_depth"] == 3.0 and float(ds["water_depth"]) == 3.0
    out = md.far_field_mean_drift(ds, _zero_motions(ds), {"source": "fixed"})
    assert json.loads(out.attrs["mean_drift_far_field"])["depth_treatment"] == ["finite"]
    assert np.all(np.isfinite(out["mean_drift_far_field"].values))


def _motions(ds, seed=3):
    rng = np.random.default_rng(seed)
    shape = (ds["omega"].size, ds["wave_direction"].size, ds["radiating_dof"].size)
    values = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * 0.4
    return xr.DataArray(values, dims=("omega", "wave_direction", "radiating_dof"),
                        coords={n: ds[n].values for n in ("omega", "wave_direction", "radiating_dof")})


def test_moving_body_drift_uses_phase_factor_radiation_and_yaw_transfer(exported):
    X = _motions(exported)
    ds = md.far_field_mean_drift(exported, X, {"source": "prescribed"})
    got = ds["mean_drift_far_field"].values
    env = json.loads(exported.attrs["wave_environment"])
    HD, HR = exported["kochin_diffraction"].values, exported["kochin_radiation"].values
    for i, omega in enumerate(exported["omega"].values):
        H = 1j * (HD[i, 0] + X.values[i, 0] @ HR[i])                # explicit contract composition
        fx, fy, mz0, _ = md.drift_from_kochin(H, 0.0, float(omega), float(exported["wavenumber"].values[i]),
                                              float("inf"), env["rho"])
        expected = (fx, fy, mz0 - CENTER[0] * fy + CENTER[1] * fx)
        assert np.allclose(got[i, 0], expected, rtol=1e-12, atol=1e-12 * max(map(abs, expected)))
    # radiation matters and the phase factor matters (guards against dropping either)
    fixed = md.far_field_mean_drift(exported, _zero_motions(exported), {"source": "fixed"})
    assert not np.allclose(got, fixed["mean_drift_far_field"].values, rtol=1e-3)
    H0 = HD[0, 0] + X.values[0, 0] @ HR[0]
    no_i = md.drift_from_kochin(H0, 0.0, float(exported["omega"].values[0]),
                                float(exported["wavenumber"].values[0]), float("inf"), env["rho"])
    assert not np.isclose(no_i[0], got[0, 0, 0], rtol=1e-3)


def test_motion_provenance_rules(exported):
    X = _motions(exported)
    with pytest.raises(md.MeanDriftError, match="'fixed' given with nonzero"):
        md.far_field_mean_drift(exported, X, {"source": "fixed"})
    with pytest.raises(md.MeanDriftError, match="must have provenance source 'fixed'"):
        md.far_field_mean_drift(exported, _zero_motions(exported), {"source": "eom"})
    with pytest.raises(md.MeanDriftError, match="matrices_hash"):
        md.far_field_mean_drift(exported, X, {"source": "eom", "damping": "raw"})
    with pytest.raises(md.MeanDriftError, match="damping"):
        md.far_field_mean_drift(exported, X, {"source": "eom", "matrices_hash": "abc"})
    ok = md.far_field_mean_drift(exported, X, {"source": "eom", "matrices_hash": "abc", "damping": "derived"})
    assert json.loads(ok.attrs["mean_drift_far_field"])["motions"]["matrices_hash"] == "abc"


def test_equivalent_motion_headings_are_admitted():
    ds = _solve({"kochin": "export", "method": "direct"}, omegas=(0.8,), headings=(0.0,))
    motions = _zero_motions(ds).assign_coords(wave_direction=[2 * np.pi])
    out = md.far_field_mean_drift(ds, motions, {"source": "fixed"})
    assert np.array_equal(out["wave_direction"].values, ds["wave_direction"].values)


def test_export_carries_environment_scalars(exported):
    assert float(exported["rho"]) == 1025.0 and float(exported["g"]) == 9.81
    assert np.isinf(float(exported["water_depth"]))


def test_schema_refuses_foreign_drift_dofs_units_and_coarse_theta(exported):
    from anyhydro.schema import validate_hydro_dataset
    good = md.far_field_mean_drift(exported, _zero_motions(exported), {"source": "fixed"})
    for dofs in (["surge", "sway", "yaw"], ["ghost::surge", "ghost::sway", "ghost::yaw"]):
        with pytest.raises(SchemaError, match="drift_dof"):
            validate_hydro_dataset(good.assign_coords(drift_dof=dofs))
    meta = json.loads(good.attrs["mean_drift_far_field"])
    meta["units"] = "bananas"
    bad = good.copy()
    bad.attrs["mean_drift_far_field"] = json.dumps(meta)
    with pytest.raises(SchemaError, match="units"):
        validate_hydro_dataset(bad)
    coarse = exported.isel(theta=slice(0, None, exported["theta"].size // 8))
    export = json.loads(coarse.attrs["kochin_export"])
    export["theta_count"] = int(coarse["theta"].size)
    coarse.attrs["kochin_export"] = json.dumps(export)
    with pytest.raises(SchemaError, match="at least 720"):
        validate_hydro_dataset(coarse)
    with pytest.raises(SchemaError, match="scalar 'rho'"):
        validate_hydro_dataset(exported.assign(rho=np.float64(1000.0)))


def test_dispersion_check_admits_newton_wavenumbers_but_refuses_wrong_ones():
    from anyhydro.schema import validate_hydro_dataset
    ds = _solve({"kochin": "export", "method": "direct"}, depth=3.0)
    k = ds["wavenumber"].values
    validate_hydro_dataset(ds.assign(wavenumber=("omega", k * (1 + 1e-10))))     # Newton-level differences
    with pytest.raises(SchemaError, match="dispersion"):
        validate_hydro_dataset(ds.assign(wavenumber=("omega", k * (1 + 1e-6))))


def test_newman_qtf_from_a_real_drift_dataset(exported):
    """N1: the Newman QTF built from a real far-field drift reproduces it on the diagonal and binds its record."""
    import json
    from anyhydro.slow_drift import newman_qtf
    ds = md.far_field_mean_drift(exported, _zero_motions(exported), {"source": "fixed"})
    q = newman_qtf(ds)
    diag = np.einsum("iihk->ihk", q["newman_qtf"].values)
    assert np.array_equal(diag, ds["mean_drift_far_field"].values)
    assert json.loads(q.attrs["newman_qtf"])["source_mean_drift"]["motions"]["source"] == "fixed"

