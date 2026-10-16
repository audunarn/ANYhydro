# SPDX-License-Identifier: MPL-2.0
"""Newman slow-drift force series from a wave realization (docs/MEAN_DRIFT_CONTRACT.md section 8, slice N2)."""
from __future__ import annotations

import math

import numpy as np
import pytest

anyloads = pytest.importorskip("anyloads")
from anyloads import AiryComponent, WaveRealization  # noqa: E402

from anyhydro.slow_drift import (DriftSeries, SlowDriftError, build_drift_series, newman_values,  # noqa: E402
                                 verify_drift_binding)

from test_slow_drift import HEADINGS, OMEGA, _drift  # noqa: E402


def _dataset():
    """A full single-body hydro dataset (synthetic fixture body) on the N1 test grid with the N1 drift block."""
    import json
    from anyhydro import fixtures
    from anyhydro.mean_drift import MEAN_DRIFT_FORMAT
    from anyhydro.records import HydroRequest
    from anyhydro.schema import MEAN_DRIFT_UNITS, assemble_hydro_dataset
    body = fixtures.make_fixture_bodies()[0]
    request = HydroRequest(omegas=OMEGA.copy(), wave_directions=HEADINGS.copy(), water_depth=np.inf, rho=1025.0,
                           gravity=9.81, include_zero_frequency=False, include_infinite_frequency=False,
                           backend_settings={"fixture": "n2-drift-series"})
    n = len(OMEGA)
    A = np.tile(np.eye(6), (n, 1, 1)) * 1e3
    ds = assemble_hydro_dataset(
        (body,), request, added_mass=A, radiation_damping=0.1 * A, excitation_force=np.ones((n, 2, 6), complex),
        hydrostatic_stiffness=np.eye(6), rotation_center=np.array([body.rotation_center]),
        center_of_mass=np.array([body.center_of_mass]), center_of_buoyancy=np.array([body.rotation_center]),
        added_mass_infinite=None, added_mass_infinite_source=None, backend="synthetic", backend_version="1",
        backend_commit="unknown", qualification_status="development_fixture")
    ds = ds.assign_coords(drift_dof=[f"{body.body_id}::{d}" for d in ("surge", "sway", "yaw")])
    ds["mean_drift_far_field"] = (("omega", "wave_direction", "drift_dof"), _drift())
    ds.attrs["mean_drift_far_field"] = json.dumps(dict(
        format=MEAN_DRIFT_FORMAT, method="far_field_momentum_conservation", units=MEAN_DRIFT_UNITS,
        yaw_reference=[float(v) for v in body.rotation_center], motions=dict(source="fixed", motion_hash="x")))
    return ds


def _waves(*components):
    return WaveRealization(tuple(AiryComponent(*c) for c in components))


def _direct(series, t):
    """Reference: the plain O(N^2) double sum with the interpolated D and the declared Newman rule."""
    out = np.zeros((len(t), 3))
    theta = t[:, None] * series.omegas[None, :] - series.phases[None, :]
    for k in range(3):
        T = newman_values(series.drift[:, k], series.method)
        for i in range(len(series.omegas)):
            for j in range(len(series.omegas)):
                out[:, k] += (series.amplitudes[i] * series.amplitudes[j] * T[i, j]
                              * np.cos(theta[:, i] - theta[:, j]))
    return out


T_GRID = np.linspace(0.0, 400.0, 801)


@pytest.mark.parametrize("method", ["geometric", "arithmetic"])
def test_one_component_gives_the_constant_mean_drift(method):
    ds = _dataset()
    s = build_drift_series(ds, _waves((1.5, 0.6, 0.0, 0.7)), method)
    D = ds["mean_drift_far_field"].values[2, 0, :]                     # omega 0.6 is a grid point, heading 0
    assert np.allclose(s.evaluate(T_GRID), 1.5 ** 2 * D, rtol=1e-12, atol=0)
    assert np.allclose(s.mean(), 1.5 ** 2 * D, rtol=1e-12, atol=0)


def test_two_components_give_the_mean_and_the_beat_with_the_elevation_phase_convention():
    """Elevation a cos(w t - phi) (ANYloads at the origin): the slow part is a1 a2 T cos((w1 - w2) t - (phi1 - phi2))."""
    ds = _dataset()
    a1, w1, p1, a2, w2, p2 = 1.2, 0.5, 0.3, 0.8, 0.7, -1.1
    s = build_drift_series(ds, _waves((a1, w1, 0.0, p1), (a2, w2, 0.0, p2)))
    D = ds["mean_drift_far_field"].values[:, 0, 0]                     # surge, positive
    D1, D2 = D[1], D[3]
    expected = (a1 ** 2 * D1 + a2 ** 2 * D2
                + 2 * a1 * a2 * math.sqrt(D1 * D2) * np.cos((w1 - w2) * T_GRID - (p1 - p2)))
    assert np.allclose(s.evaluate(T_GRID)[:, 0], expected, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("method", ["geometric", "arithmetic"])
def test_many_components_with_mixed_signs_equal_the_double_sum(method):
    rng = np.random.default_rng(7)
    comps = [(float(rng.uniform(0.1, 1.0)), float(w), 90.0, float(rng.uniform(-math.pi, math.pi)))
             for w in np.linspace(0.41, 0.79, 23)]                       # off-grid: D interpolated
    s = build_drift_series(_dataset(), _waves(*comps), method)
    assert np.any(s.drift[:, 2] < 0) and np.any(s.drift[:, 2] > 0)     # yaw changes sign inside the band
    assert np.allclose(s.evaluate(T_GRID), _direct(s, T_GRID), rtol=1e-10, atol=1e-6)
    assert np.allclose(s.mean(), (s.amplitudes[:, None] ** 2 * s.drift).sum(axis=0))


def test_serialisation_and_binding():
    ds = _dataset()
    waves = _waves((1.0, 0.45, 0.0, 0.1), (0.7, 0.62, 0.0, 2.0))
    s = build_drift_series(ds, waves)
    back = DriftSeries.from_dict(s.to_dict())
    assert back.content_hash == s.content_hash
    assert np.array_equal(back.evaluate(T_GRID), s.evaluate(T_GRID))
    assert verify_drift_binding(s, ds, waves)
    with pytest.raises(SlowDriftError, match="does not match"):
        verify_drift_binding(s, ds, _waves((1.0, 0.45, 0.0, 0.1), (0.7, 0.62, 0.0, 2.1)))
    changed = ds.copy(deep=True)
    changed["mean_drift_far_field"].values[1, 0, 0] *= 1.01
    with pytest.raises(SlowDriftError, match="does not match"):
        verify_drift_binding(s, changed, waves)
    with pytest.raises(SlowDriftError, match="exact named schema"):
        DriftSeries.from_dict(dict(s.to_dict(), extra=1))


@pytest.mark.parametrize("waves, message", [
    (((1.0, 0.5, 0.0, 0.0), (1.0, 0.6, 90.0, 0.0)), "unidirectional"),
    (((1.0, 0.5, 45.0, 0.0),), "exact dataset heading"),
    (((1.0, 0.95, 0.0, 0.0),), "outside the stored"),
    (((1.0, 0.3, 0.0, 0.0),), "outside the stored"),
])
def test_refusals(waves, message):
    with pytest.raises(SlowDriftError, match=message):
        build_drift_series(_dataset(), _waves(*waves))


def test_other_refusals():
    with pytest.raises(SlowDriftError, match="WaveRealization"):
        build_drift_series(_dataset(), [AiryComponent(1.0, 0.5)])
    with pytest.raises(SlowDriftError, match="unknown Newman method"):
        build_drift_series(_dataset(), _waves((1.0, 0.5, 0.0, 0.0)), "harmonic")
    s = build_drift_series(_dataset(), _waves((1.0, 0.5, 360.0, 0.0)))   # 360 deg == the dataset heading 0
    assert s.heading_radians == float(HEADINGS[0])
    assert OMEGA[0] <= s.omegas[0] <= OMEGA[-1]
