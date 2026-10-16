# SPDX-License-Identifier: MPL-2.0
"""Newman difference-frequency QTF from the far-field mean drift (docs/MEAN_DRIFT_CONTRACT.md section 8, slice N1)."""
from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from anyhydro.mean_drift import MEAN_DRIFT_FORMAT
from anyhydro.slow_drift import NEWMAN_QTF_FORMAT, SlowDriftError, newman_qtf, newman_values, qtf_at

OMEGA = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
HEADINGS = np.array([0.0, np.pi / 2])


def _dataset(drift=None, record=True):
    if drift is None:
        w = OMEGA[:, None, None]
        drift = np.concatenate([1e4 * w ** 2 * np.ones((1, 2, 1)),                 # surge: positive, rising
                                -3e3 * w * np.ones((1, 2, 1)),                      # sway: negative
                                np.array([-50.0, -10.0, 20.0, 40.0, 60.0])[:, None, None] * np.ones((1, 2, 1))],
                               axis=2)                                               # yaw: sign change
    ds = xr.Dataset({"mean_drift_far_field": (("omega", "wave_direction", "drift_dof"), drift)},
                    coords={"omega": OMEGA, "wave_direction": HEADINGS,
                            "drift_dof": ["b::surge", "b::sway", "b::yaw"]})
    if record:
        ds.attrs["mean_drift_far_field"] = json.dumps(dict(format=MEAN_DRIFT_FORMAT, motions=dict(source="fixed",
                                                                                                    motion_hash="x")))
    return ds


@pytest.mark.parametrize("method", ["geometric", "arithmetic"])
def test_diagonal_is_the_mean_drift_and_the_qtf_is_symmetric(method):
    ds = _dataset()
    q = newman_qtf(ds, method)["newman_qtf"].values
    D = ds["mean_drift_far_field"].values
    assert np.array_equal(np.einsum("iihk->ihk", q), D)
    assert np.array_equal(q, np.transpose(q, (1, 0, 2, 3)))


def test_geometric_rule_values_and_sign():
    D = np.array([4.0, 9.0, -1.0, -16.0])
    T = newman_values(D, "geometric")
    assert T[0, 1] == pytest.approx(6.0)                    # sqrt(4 * 9), positive
    assert T[2, 3] == pytest.approx(-4.0)                   # both negative: -sqrt(16)
    assert T[1, 2] == 0.0 and T[2, 1] == 0.0                # opposite signs -> 0 (declared)
    assert newman_values(D, "arithmetic")[1, 2] == pytest.approx(4.0)


def test_record_binds_the_source_and_declares_the_method():
    out = newman_qtf(_dataset(), "geometric")
    rec = json.loads(out.attrs["newman_qtf"])
    assert rec["format"] == NEWMAN_QTF_FORMAT and rec["method"] == "geometric"
    assert rec["source_mean_drift"]["format"] == MEAN_DRIFT_FORMAT
    assert list(out["omega_1"].values) == list(OMEGA) and list(out["drift_dof"].values)[2] == "b::yaw"


def test_lookup_is_on_grid_points_only():
    out = newman_qtf(_dataset())
    T = qtf_at(out, 0.5, 0.7)
    assert T.shape == (2, 3)
    assert T[0, 0] == pytest.approx(1e4 * 0.5 * 0.7)        # sqrt(1e4 0.25 x 1e4 0.49)
    with pytest.raises(SlowDriftError, match="not a grid point"):
        qtf_at(out, 0.55, 0.7)


@pytest.mark.parametrize("change, message", [
    (dict(record=False), "record"),
    (dict(drift=np.full((5, 2, 3), np.nan)), "finite"),
])
def test_refusals(change, message):
    with pytest.raises(SlowDriftError, match=message):
        newman_qtf(_dataset(**change))
    with pytest.raises(SlowDriftError, match="unknown Newman method"):
        newman_qtf(_dataset(), "harmonic")
    with pytest.raises(SlowDriftError, match="lacks 'mean_drift_far_field'"):
        newman_qtf(xr.Dataset())
    bad = _dataset().assign_coords(omega=OMEGA[::-1])
    with pytest.raises(SlowDriftError, match="strictly increasing"):
        newman_qtf(bad)
