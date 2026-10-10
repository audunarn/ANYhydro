"""Focused explicit long-kernel control tests (development slice).

Explicit longer radiation kernels through the public builder with frozen
``RadiationExecutionLimits``: default-1001 refusal, explicit 1002/1003
admission on the analytic exponential oracle, immutability, hash and
provenance participation, NetCDF save/reload equality, typed refusals and
preservation of all old hashes/behavior (``execution_limits=None``
identity). These are development tests: they confer no scientific
qualification. No BEM/Capytaine import, no monkeypatching of limits, no
execution beyond 1003 samples.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

import anyhydro
from anyhydro import radiation as rad
from anyhydro.fixtures import make_fixture_bodies
from anyhydro.records import HydroRequest
from anyhydro.schema import assemble_hydro_dataset

# Reference content hash of the default-settings (execution_limits=None)
# oracle kernel, computed with the pre-change code via the exact harness
# (evidence: anymarine-evidence/long-kernel-worker-02, run pre_hash.py).
# The post-change build must reproduce it exactly.
REFERENCE_KERNEL_HASH = "8a40ee08c0568484bb78f666ab74d307bcdf87ecbc8d4e7f9c10daff448e208e"

OLD_SETTINGS_KEYS = {
    "dt", "n_time", "symmetry_tol", "passivity_tol", "low_tail_policy",
    "zero_frequency_damping", "extension_name", "time_window",
}

REF_OMEGAS = np.linspace(0.05, 20.0, 61)
REF_DT = 0.05
REF_N_TIME = 201


def _ref_matrices(n_dof: int = 12):
    """Full coupled C, lambda, A_inf (same structure as the radiation oracle)."""
    C = np.empty((n_dof, n_dof))
    lam = np.empty((n_dof, n_dof))
    for i in range(n_dof):
        for j in range(n_dof):
            same_body = (i // 6) == (j // 6)
            if i == j:
                C[i, j] = 800.0 + 40.0 * (i % 6)
                lam[i, j] = 1.0 + 0.1 * (i % 6)
            elif same_body:
                C[i, j] = 60.0 + 5.0 * ((i * 7 + j * 3) % 11)
                lam[i, j] = 0.9 + 0.05 * ((i + j) % 9)
            else:
                C[i, j] = 25.0 + 4.0 * ((i * 5 + j * 11) % 7)
                lam[i, j] = 1.1 + 0.04 * ((i * 3 + j * 7) % 5)
    a_inf = 2.0 * C / lam**2 + 50.0
    return C, lam, a_inf


def make_ref_dataset():
    C, lam, a_inf = _ref_matrices()
    bodies = make_fixture_bodies()[: 2]
    request = HydroRequest(
        omegas=np.asarray(REF_OMEGAS, dtype=np.float64),
        wave_directions=np.array([0.0]),
        water_depth=150.0,
        rho=1025.0,
        gravity=9.81,
        include_zero_frequency=False,
        include_infinite_frequency=False,
        backend_settings={"fixture": "long-kernel-exponential-oracle"},
    )
    n_w = request.omegas.shape[0]
    w2 = request.omegas[:, None, None] ** 2
    B = C[None] * lam[None] / (lam[None] ** 2 + w2)
    A = a_inf[None] - C[None] / (lam[None] ** 2 + w2)
    centers = np.array([b.rotation_center for b in bodies], dtype=np.float64)
    coms = np.array([b.center_of_mass for b in bodies], dtype=np.float64)
    return assemble_hydro_dataset(
        bodies,
        request,
        added_mass=A,
        radiation_damping=B,
        excitation_force=np.zeros((n_w, 1, 12), dtype=np.complex128),
        hydrostatic_stiffness=np.zeros((12, 12)),
        rotation_center=centers,
        center_of_mass=coms,
        center_of_buoyancy=centers,
        added_mass_infinite=a_inf,
        added_mass_infinite_source="long-kernel test oracle: analytic exponential kernel, not a backend solve",
        backend="exponential-oracle",
        qualification_status="development_fixture",
    )


def make_ref_settings(**overrides):
    C, lam, _ = _ref_matrices()
    defaults = dict(
        dt=REF_DT,
        n_time=REF_N_TIME,
        symmetry_tol=1.0e-9,
        passivity_tol=1.0e-9,
        low_tail_policy=rad.LOW_TAIL_EXPLICIT_ZERO,
        zero_frequency_damping=C / lam,  # oracle B(0)
    )
    defaults.update(overrides)
    return rad.RadiationSettings(**defaults)


@pytest.fixture(scope="module")
def ref_ds():
    return make_ref_dataset()


# --------------------------------------------------------------------------- #
# RadiationExecutionLimits admission (typed, before any allocation)
# --------------------------------------------------------------------------- #


class TestExecutionLimitsValidation:
    def test_defaults_preserve_1001(self):
        limits = rad.RadiationExecutionLimits()
        assert limits.max_time_samples == 1001
        assert limits.as_dict() == {"max_time_samples": 1001}
        assert "max_time_samples=1001" in limits.describe()

    def test_frozen(self):
        limits = rad.RadiationExecutionLimits(max_time_samples=1003)
        with pytest.raises(Exception):
            limits.max_time_samples = 1005  # type: ignore[misc]

    def test_unknown_control_rejected(self):
        with pytest.raises(rad.RadiationSettingsError, match="unknown execution-limit"):
            rad.RadiationExecutionLimits(bogus=1, other="x")

    @pytest.mark.parametrize("bad", [True, False, 1002.5, 1002.0, float("inf"),
                                     float("-inf"), float("nan"), "1003", None])
    def test_bool_fractional_nonfinite_wrong_type_rejected(self, bad):
        with pytest.raises(rad.RadiationSettingsError, match="must be an integer"):
            rad.RadiationExecutionLimits(max_time_samples=bad)

    def test_min_two_enforced(self):
        with pytest.raises(rad.RadiationLimitError, match="at least 2"):
            rad.RadiationExecutionLimits(max_time_samples=1)
        assert rad.RadiationExecutionLimits(max_time_samples=2).max_time_samples == 2

    def test_hard_ceiling_20001_boundary(self):
        # ceiling boundary validation only, before any allocation
        assert rad.RadiationExecutionLimits(max_time_samples=20001).max_time_samples == 20001
        with pytest.raises(rad.RadiationLimitError, match="ceiling"):
            rad.RadiationExecutionLimits(max_time_samples=20002)

    def test_numpy_integer_admitted(self):
        assert rad.RadiationExecutionLimits(
            max_time_samples=np.int64(1003)).max_time_samples == 1003

    def test_public_export(self):
        assert anyhydro.RadiationExecutionLimits is rad.RadiationExecutionLimits
        assert anyhydro.MAX_TIME_SAMPLES == 1001
        assert anyhydro.MAX_TIME_SAMPLES_CEILING == 20001
        assert anyhydro.MIN_TIME_SAMPLES == 2


# --------------------------------------------------------------------------- #
# settings integration: default refusal, explicit admission, wrong objects
# --------------------------------------------------------------------------- #


class TestSettingsLimitsIntegration:
    def test_default_refuses_1002(self):
        with pytest.raises(rad.RadiationLimitError, match="1001"):
            make_ref_settings(n_time=1002)

    def test_explicit_default_limits_also_refuse_1002(self):
        with pytest.raises(rad.RadiationLimitError, match="1001"):
            make_ref_settings(n_time=1002,
                              execution_limits=rad.RadiationExecutionLimits())

    def test_limits_below_n_time_refuse(self):
        with pytest.raises(rad.RadiationLimitError, match="1002"):
            make_ref_settings(n_time=1003,
                              execution_limits=rad.RadiationExecutionLimits(max_time_samples=1002))

    @pytest.mark.parametrize("wrong", [{"max_time_samples": 1003}, 1003, "limits", [1003]])
    def test_wrong_limits_object_refused(self, wrong):
        with pytest.raises(rad.RadiationSettingsError,
                           match="RadiationExecutionLimits instance"):
            make_ref_settings(execution_limits=wrong)

    def test_explicit_1002_and_1003_builder_success(self, ref_ds):
        for n_time, cap in ((1002, 1002), (1003, 1003)):
            settings = make_ref_settings(
                n_time=n_time,
                execution_limits=rad.RadiationExecutionLimits(max_time_samples=cap),
            )
            kernel = rad.radiation_kernel_from_dataset(ref_ds, settings)
            assert kernel.time.size == n_time
            assert kernel.kernel.shape == (n_time, 12, 12)
            assert kernel.settings is settings
            assert kernel.settings.execution_limits.max_time_samples == cap
            assert np.all(np.isfinite(kernel.kernel))
            assert np.array_equal(kernel.time, np.arange(n_time) * REF_DT)
            # no fake padding or cropped history: the full declared grid
            assert kernel.settings.n_time == n_time == kernel.time.size

    def test_reconstruction_finite_public_path(self, ref_ds):
        settings = make_ref_settings(
            n_time=1003,
            execution_limits=rad.RadiationExecutionLimits(max_time_samples=1003),
        )
        kernel = rad.radiation_kernel_from_dataset(ref_ds, settings)
        a_rec, b_rec = kernel.reconstruct(np.array([0.0, 0.5, 2.0]))
        assert a_rec.shape == (3, 12, 12) and b_rec.shape == (3, 12, 12)
        assert np.all(np.isfinite(a_rec)) and np.all(np.isfinite(b_rec))

    def test_immutable_arrays_and_limits(self, ref_ds):
        settings = make_ref_settings(
            n_time=1002,
            execution_limits=rad.RadiationExecutionLimits(max_time_samples=1002),
        )
        kernel = rad.radiation_kernel_from_dataset(ref_ds, settings)
        for array in (kernel.time, kernel.kernel, kernel.added_mass_infinite,
                      kernel.diagnostics.frequency_grid,
                      kernel.settings.zero_frequency_damping):
            with pytest.raises(ValueError):
                array.setflags(write=True)
        with pytest.raises(Exception):
            kernel.settings.execution_limits.max_time_samples = 20001  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            kernel.settings.n_time = 10  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# hash, provenance, persistence and old-identity preservation
# --------------------------------------------------------------------------- #


class TestHashProvenancePersistence:
    def test_old_fixture_hash_unchanged(self, ref_ds):
        # default settings (execution_limits=None) reproduce the pre-change
        # content hash exactly: all old hashes/behavior are preserved
        kernel = rad.radiation_kernel_from_dataset(ref_ds, make_ref_settings())
        assert kernel.content_hash() == REFERENCE_KERNEL_HASH

    def test_none_settings_dict_keeps_old_identity(self, ref_ds):
        kernel = rad.radiation_kernel_from_dataset(ref_ds, make_ref_settings())
        data = kernel.settings.to_dict()
        assert set(data) == OLD_SETTINGS_KEYS  # no new key when limits are None
        assert "execution_limits" not in json.loads(
            anyhydro.radiation._kernel_to_dataset(kernel).attrs["settings"])
        assert "execution_limits" not in kernel.provenance
        assert not any("execution_limits" in a for a in kernel.assumptions)

    def test_hash_changes_with_declared_controls(self, ref_ds):
        none_kernel = rad.radiation_kernel_from_dataset(ref_ds, make_ref_settings())
        cap1001 = rad.radiation_kernel_from_dataset(
            ref_ds, make_ref_settings(execution_limits=rad.RadiationExecutionLimits()))
        cap1003 = rad.radiation_kernel_from_dataset(
            ref_ds, make_ref_settings(execution_limits=rad.RadiationExecutionLimits(max_time_samples=1003)))
        cap1005 = rad.radiation_kernel_from_dataset(
            ref_ds, make_ref_settings(execution_limits=rad.RadiationExecutionLimits(max_time_samples=1005)))
        hashes = {none_kernel.content_hash(), cap1001.content_hash(),
                  cap1003.content_hash(), cap1005.content_hash()}
        assert len(hashes) == 4  # every distinct declared control set is distinct
        # determinism: same declared controls, same hash
        again = rad.radiation_kernel_from_dataset(
            ref_ds, make_ref_settings(execution_limits=rad.RadiationExecutionLimits(max_time_samples=1003)))
        assert again.content_hash() == cap1003.content_hash()

    def test_provenance_records_declared_limits(self, ref_ds):
        kernel = rad.radiation_kernel_from_dataset(
            ref_ds, make_ref_settings(execution_limits=rad.RadiationExecutionLimits(max_time_samples=1003)))
        assert kernel.provenance["execution_limits"] == {"max_time_samples": 1003}
        assert any("execution_limits=max_time_samples=1003" in a for a in kernel.assumptions)
        assert "execution_limits" in kernel.settings.to_dict()

    def test_netcdf_roundtrip_long_kernel(self, ref_ds, tmp_path):
        settings = make_ref_settings(
            n_time=1003,
            execution_limits=rad.RadiationExecutionLimits(max_time_samples=1003),
        )
        kernel = rad.radiation_kernel_from_dataset(ref_ds, settings)
        path = tmp_path / "long_kernel.nc"
        rad.save_radiation_kernel(kernel, path)
        loaded = rad.load_radiation_kernel(path)
        assert loaded.content_hash() == kernel.content_hash()
        assert np.array_equal(loaded.kernel, kernel.kernel)
        assert np.array_equal(loaded.time, kernel.time)
        assert dict(loaded.provenance) == dict(kernel.provenance)
        assert loaded.assumptions == kernel.assumptions
        assert loaded.settings.execution_limits is not None
        assert loaded.settings.execution_limits.max_time_samples == 1003
        assert loaded.settings.n_time == 1003
        a1, b1 = kernel.reconstruct(np.array([0.5, 2.0]))
        a2, b2 = loaded.reconstruct(np.array([0.5, 2.0]))
        assert np.array_equal(a1, a2) and np.array_equal(b1, b2)

    def test_netcdf_roundtrip_none_limits_old_identity(self, ref_ds, tmp_path):
        kernel = rad.radiation_kernel_from_dataset(ref_ds, make_ref_settings())
        path = tmp_path / "old_kernel.nc"
        rad.save_radiation_kernel(kernel, path)
        loaded = rad.load_radiation_kernel(path)
        assert loaded.content_hash() == REFERENCE_KERNEL_HASH
        assert loaded.settings.execution_limits is None

    def test_from_dict_reconstructs_and_enforces_limits(self):
        settings = make_ref_settings(execution_limits=rad.RadiationExecutionLimits(max_time_samples=1003))
        rebuilt = rad.RadiationSettings.from_dict(settings.to_dict())
        assert rebuilt.to_dict() == settings.to_dict()
        assert rebuilt.execution_limits is not None
        assert rebuilt.execution_limits.max_time_samples == 1003
        # enforcement: a >1001 dict whose declared controls were deleted refuses
        long_dict = settings.to_dict()
        long_dict["n_time"] = 1002
        del long_dict["execution_limits"]
        with pytest.raises(rad.RadiationLimitError, match="1001"):
            rad.RadiationSettings.from_dict(long_dict)

    def test_deleting_controls_from_long_artifact_refuses(self, ref_ds, tmp_path):
        import xarray as xr

        settings = make_ref_settings(
            n_time=1002,
            execution_limits=rad.RadiationExecutionLimits(max_time_samples=1002),
        )
        kernel = rad.radiation_kernel_from_dataset(ref_ds, settings)
        path = tmp_path / "long_kernel.nc"
        rad.save_radiation_kernel(kernel, path)
        tampered = xr.load_dataset(path, engine="scipy")
        stripped = json.loads(tampered.attrs["settings"])
        del stripped["execution_limits"]
        tampered.attrs["settings"] = json.dumps(stripped, sort_keys=True,
                                                separators=(",", ":"))
        tampered.to_netcdf(tmp_path / "stripped.nc", engine="scipy",
                           format="NETCDF3_64BIT")
        with pytest.raises(rad.RadiationSerializationError, match="1001"):
            rad.load_radiation_kernel(tmp_path / "stripped.nc")

    def test_tampered_unknown_limit_control_refuses(self, ref_ds, tmp_path):
        import xarray as xr

        settings = make_ref_settings(
            n_time=1002,
            execution_limits=rad.RadiationExecutionLimits(max_time_samples=1002),
        )
        kernel = rad.radiation_kernel_from_dataset(ref_ds, settings)
        path = tmp_path / "long_kernel.nc"
        rad.save_radiation_kernel(kernel, path)
        tampered = xr.load_dataset(path, engine="scipy")
        poisoned = json.loads(tampered.attrs["settings"])
        poisoned["execution_limits"] = {"max_time_samples": 1002, "bogus": 7}
        tampered.attrs["settings"] = json.dumps(poisoned, sort_keys=True,
                                                separators=(",", ":"))
        tampered.to_netcdf(tmp_path / "poisoned.nc", engine="scipy",
                           format="NETCDF3_64BIT")
        with pytest.raises(rad.RadiationSerializationError, match="unknown"):
            rad.load_radiation_kernel(tmp_path / "poisoned.nc")


# --------------------------------------------------------------------------- #
# fail-closed reader compatibility: unknown top-level keys are refused
# --------------------------------------------------------------------------- #


class TestReaderFailClosedUnknownTopLevelKeys:
    def test_from_dict_rejects_unrecognized_tail_flag(self):
        data = make_ref_settings().to_dict()
        data["low_tail_flag"] = "keep"  # unrecognized tail flag
        with pytest.raises(rad.RadiationSettingsError, match="unknown top-level"):
            rad.RadiationSettings.from_dict(data)

    @pytest.mark.parametrize("extra", [
        {"tail_policy": "linear_to_zero_extension"},
        {"execution_limit": {"max_time_samples": 1003}},
        {"max_time_samples": 1003},
        {"n_times": 1002},
    ])
    def test_from_dict_rejects_assorted_unknown_keys(self, extra):
        data = make_ref_settings().to_dict()
        data.update(extra)
        with pytest.raises(rad.RadiationSettingsError, match="unknown top-level"):
            rad.RadiationSettings.from_dict(data)

    def test_from_dict_admits_exact_serialized_fields(self):
        settings = make_ref_settings(
            execution_limits=rad.RadiationExecutionLimits(max_time_samples=1003))
        rebuilt = rad.RadiationSettings.from_dict(settings.to_dict())
        assert rebuilt.to_dict() == settings.to_dict()

    def test_load_tampered_unknown_key_refuses_without_artifact_mutation(
            self, ref_ds, tmp_path):
        import hashlib
        import xarray as xr

        settings = make_ref_settings(
            n_time=1002,
            execution_limits=rad.RadiationExecutionLimits(max_time_samples=1002),
        )
        kernel = rad.radiation_kernel_from_dataset(ref_ds, settings)
        path = tmp_path / "long_kernel.nc"
        rad.save_radiation_kernel(kernel, path)
        tampered = xr.load_dataset(path, engine="scipy")
        poisoned = json.loads(tampered.attrs["settings"])
        poisoned["low_tail_flag"] = "keep"  # unrecognized tail flag
        tampered.attrs["settings"] = json.dumps(poisoned, sort_keys=True,
                                                separators=(",", ":"))
        poisoned_path = tmp_path / "poisoned.nc"
        tampered.to_netcdf(poisoned_path, engine="scipy", format="NETCDF3_64BIT")

        before = hashlib.sha256(poisoned_path.read_bytes()).hexdigest()
        with pytest.raises(rad.RadiationSerializationError,
                           match="unknown top-level"):
            rad.load_radiation_kernel(poisoned_path)
        after = hashlib.sha256(poisoned_path.read_bytes()).hexdigest()
        assert before == after  # the refusal mutates nothing on disk
        # the untampered artifact still loads with its exact content hash
        loaded = rad.load_radiation_kernel(path)
        assert loaded.content_hash() == kernel.content_hash()
