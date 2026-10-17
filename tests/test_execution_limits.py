"""CapytaineExecutionLimits admission and integration tests (no BEM run).

All solves are mocked: these tests exercise limit validation, constructor
admission, budget enforcement before cache lookup, and provenance wiring
without importing capytaine or executing any BEM solve.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import anyhydro.capytaine_backend as cb
from anyhydro import (
    CapytaineBackend,
    CapytaineExecutionLimits,
    CapabilityError,
    HydroRequest,
    assemble_hydro_dataset,
    make_first_slice_bodies,
    make_first_slice_request,
    validate_hydro_dataset,
)

LIMIT_FIELDS = (
    "max_total_panels",
    "max_finite_omegas",
    "max_wave_directions",
    "n_jobs",
    "n_threads",
)


class TestLimitValidation:
    def test_defaults_preserve_tiny_budgets(self):
        limits = CapytaineExecutionLimits()
        assert limits.max_total_panels == 100
        assert limits.max_finite_omegas == 3
        assert limits.max_wave_directions == 1
        assert limits.n_jobs == 1
        assert limits.n_threads == 1
        assert limits.as_dict() == {
            "max_total_panels": 100,
            "max_finite_omegas": 3,
            "max_wave_directions": 1,
            "n_jobs": 1,
            "n_threads": 1,
        }
        assert "max_total_panels=100" in limits.describe()

    @pytest.mark.parametrize("field", LIMIT_FIELDS)
    def test_bool_rejected(self, field):
        with pytest.raises(CapabilityError, match="must be an integer"):
            CapytaineExecutionLimits(**{field: True})

    @pytest.mark.parametrize("field", LIMIT_FIELDS)
    def test_fractional_rejected(self, field):
        with pytest.raises(CapabilityError, match="must be an integer"):
            CapytaineExecutionLimits(**{field: 3.5})

    @pytest.mark.parametrize("field", LIMIT_FIELDS)
    def test_integral_valued_float_rejected(self, field):
        # 3.0 is not fractional, but only Integral types are admitted:
        # floats are refused outright, no silent coercion.
        with pytest.raises(CapabilityError, match="must be an integer"):
            CapytaineExecutionLimits(**{field: 3.0})

    @pytest.mark.parametrize("field", LIMIT_FIELDS)
    def test_nonfinite_rejected(self, field):
        for bad in (float("inf"), float("-inf"), float("nan")):
            with pytest.raises(CapabilityError, match="must be an integer"):
                CapytaineExecutionLimits(**{field: bad})

    @pytest.mark.parametrize("field", LIMIT_FIELDS)
    def test_negative_and_zero_rejected(self, field):
        for bad in (-1, 0):
            with pytest.raises(CapabilityError, match="positive"):
                CapytaineExecutionLimits(**{field: bad})

    def test_panel_ceiling_12288(self):
        # absolute software ceiling for EXPLICIT limits (raised 2026-10-12
        # for the W5 extended-band levels 10/15 = 2400/5400 panels, and
        # 2026-10-17 to 12288 by owner decision); 12289 is refused
        assert CapytaineExecutionLimits(max_total_panels=100).max_total_panels == 100
        assert CapytaineExecutionLimits(max_total_panels=864).max_total_panels == 864
        assert CapytaineExecutionLimits(max_total_panels=1536).max_total_panels == 1536
        assert CapytaineExecutionLimits(max_total_panels=2048).max_total_panels == 2048
        assert CapytaineExecutionLimits(max_total_panels=6144).max_total_panels == 6144
        assert CapytaineExecutionLimits(max_total_panels=12288).max_total_panels == 12288
        with pytest.raises(CapabilityError, match="ceiling"):
            CapytaineExecutionLimits(max_total_panels=12289)

    def test_frequency_ceiling_513(self):
        assert CapytaineExecutionLimits(max_finite_omegas=513).max_finite_omegas == 513
        with pytest.raises(CapabilityError, match="ceiling"):
            CapytaineExecutionLimits(max_finite_omegas=514)

    def test_heading_ceiling_1(self):
        with pytest.raises(CapabilityError, match="ceiling"):
            CapytaineExecutionLimits(max_wave_directions=2)

    def test_one_job_one_thread_pinned(self):
        with pytest.raises(CapabilityError, match="ceiling"):
            CapytaineExecutionLimits(n_jobs=2)
        with pytest.raises(CapabilityError, match="ceiling"):
            CapytaineExecutionLimits(n_threads=2)

    def test_unknown_control_rejected(self):
        with pytest.raises(CapabilityError, match="unknown execution-limit"):
            CapytaineExecutionLimits(bogus=1, other="x")

    def test_numpy_integer_admitted(self):
        limits = CapytaineExecutionLimits(max_finite_omegas=np.int64(5))
        assert limits.max_finite_omegas == 5

    def test_frozen(self):
        limits = CapytaineExecutionLimits()
        with pytest.raises(Exception):
            limits.max_finite_omegas = 10  # type: ignore[misc]

    def test_validation_precedes_backend_import(self, monkeypatch):
        # even a completely broken capytaine probe cannot matter: limit
        # validation raises before any availability check or import
        monkeypatch.setattr(
            cb, "_capytaine_spec", lambda: (_ for _ in ()).throw(AssertionError())
        )
        with pytest.raises(CapabilityError):
            CapytaineExecutionLimits(max_finite_omegas=-3)


class TestBackendConstructor:
    def test_default_limits_when_unset(self):
        backend = CapytaineBackend()
        assert backend._limits.as_dict() == CapytaineExecutionLimits().as_dict()

    def test_explicit_limits_accepted(self):
        limits = CapytaineExecutionLimits(max_finite_omegas=5)
        assert CapytaineBackend(execution_limits=limits)._limits is limits

    def test_non_limits_object_rejected(self):
        with pytest.raises(CapabilityError, match="CapytaineExecutionLimits"):
            CapytaineBackend(execution_limits={"max_finite_omegas": 5})

    def test_invalid_limits_rejected_at_construction(self):
        with pytest.raises(CapabilityError, match="ceiling"):
            CapytaineBackend(execution_limits=CapytaineExecutionLimits(max_total_panels=12289))


# --------------------------------------------------------------------- #
# mocked-solve helpers
# --------------------------------------------------------------------- #


def _stub_dataset(bodies, request):
    """A valid canonical dataset with zero coefficients (no BEM involved)."""
    n_dof = 6 * len(bodies)
    n_omega = int(request.omegas.shape[0])
    n_dir = int(request.wave_directions.shape[0])
    return assemble_hydro_dataset(
        bodies,
        request,
        added_mass=np.zeros((n_omega, n_dof, n_dof)),
        radiation_damping=np.zeros((n_omega, n_dof, n_dof)),
        excitation_force=np.zeros((n_omega, n_dir, n_dof), dtype=np.complex128),
        hydrostatic_stiffness=np.zeros((n_dof, n_dof)),
        rotation_center=np.array([b.rotation_center for b in bodies]),
        center_of_mass=np.array([b.center_of_mass for b in bodies]),
        center_of_buoyancy=np.array([b.center_of_mass for b in bodies]),
        backend="stub",
        backend_version="0",
        qualification_status="development_fixture",
    )


def _mock_fresh(monkeypatch, dataset):
    def _fake(self, bodies, request, settings):
        return dataset
    monkeypatch.setattr(CapytaineBackend, "_solve_fresh", _fake)


def _wide_request(n_omega=4):
    base = make_first_slice_request(include_infinite_frequency=False)
    return HydroRequest(
        omegas=np.linspace(1.0, 1.0 + 0.5 * (n_omega - 1), n_omega),
        wave_directions=base.wave_directions,
        water_depth=base.water_depth,
        rho=base.rho,
        gravity=base.gravity,
        include_zero_frequency=False,
        include_infinite_frequency=False,
        backend_settings={},
    )


class TestAdmissionAndCache:
    def test_default_backend_refuses_four_frequencies(self, monkeypatch):
        bodies = make_first_slice_bodies()
        request = _wide_request(4)
        _mock_fresh(monkeypatch, _stub_dataset(bodies, request))
        with pytest.raises(CapabilityError, match="frequency budget"):
            CapytaineBackend().solve(bodies, request)

    def test_explicit_limits_admit_four_frequencies(self, monkeypatch):
        bodies = make_first_slice_bodies()
        request = _wide_request(4)
        _mock_fresh(monkeypatch, _stub_dataset(bodies, request))
        backend = CapytaineBackend(
            execution_limits=CapytaineExecutionLimits(max_finite_omegas=5)
        )
        dataset = backend.solve(bodies, request)
        validate_hydro_dataset(dataset)

    def test_caps_apply_before_cache_lookup(self, tmp_path, monkeypatch):
        # a cache entry created under wide limits must NOT be served to a
        # default-limits backend for an over-budget request
        bodies = make_first_slice_bodies()
        request = _wide_request(4)
        _mock_fresh(monkeypatch, _stub_dataset(bodies, request))
        wide = CapytaineBackend(
            cache_dir=str(tmp_path),
            execution_limits=CapytaineExecutionLimits(max_finite_omegas=5),
        )
        wide.solve(bodies, request)
        assert len(list(tmp_path.glob("*.nc"))) == 1
        default = CapytaineBackend(cache_dir=str(tmp_path))
        with pytest.raises(CapabilityError, match="frequency budget"):
            default.solve(bodies, request)

    def test_warm_cache_hit_under_same_limits(self, tmp_path, monkeypatch):
        bodies = make_first_slice_bodies()
        request = make_first_slice_request(include_infinite_frequency=False)
        _mock_fresh(monkeypatch, _stub_dataset(bodies, request))
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        first = backend.solve(bodies, request)
        # any fresh solve now must fail: only the cache can answer
        def _boom(self, bodies, request, settings):
            raise AssertionError("cache miss: fresh solve was called")
        monkeypatch.setattr(CapytaineBackend, "_solve_fresh", _boom)
        second = backend.solve(bodies, request)
        assert second.attrs["request_hash"] == first.attrs["request_hash"]

    def test_cache_identity_records_effective_limits(self, tmp_path):
        bodies = make_first_slice_bodies()
        request = make_first_slice_request(include_infinite_frequency=False)
        limits = CapytaineExecutionLimits(max_finite_omegas=7)
        backend = CapytaineBackend(execution_limits=limits)
        identity = backend._cache_identity(bodies, request, {}, "3.0.0")
        assert identity["execution_limits"] == limits.as_dict()
        # the limits change the cache key: provenance is part of identity
        other = CapytaineBackend()._cache_identity(bodies, request, {}, "3.0.0")
        assert CapytaineBackend._cache_key(identity) != CapytaineBackend._cache_key(other)

    def test_cached_entry_carries_limits_provenance(self, tmp_path, monkeypatch):
        bodies = make_first_slice_bodies()
        request = make_first_slice_request(include_infinite_frequency=False)
        _mock_fresh(monkeypatch, _stub_dataset(bodies, request))
        backend = CapytaineBackend(cache_dir=str(tmp_path))
        backend.solve(bodies, request)
        from anyhydro import load_hydro_dataset

        cached = load_hydro_dataset(next(tmp_path.glob("*.nc")))
        raw = cached.attrs["cache_identity"]
        stored = raw if isinstance(raw, dict) else json.loads(raw)
        assert stored["execution_limits"] == CapytaineExecutionLimits().as_dict()


class TestPanelBudgetDefaultRefusal:
    """Default 100-panel limit still refuses >100-panel cases; only
    EXPLICIT limits admit the registered refined meshes (no BEM run)."""

    def _study_bodies_request(self):
        bodies = make_first_slice_bodies(subdivisions=6)  # 864 panels
        request = make_first_slice_request(include_infinite_frequency=False)
        return bodies, request

    def test_default_limits_refuse_864_panel_case(self, monkeypatch):
        bodies, request = self._study_bodies_request()
        _mock_fresh(monkeypatch, _stub_dataset(bodies, request))
        with pytest.raises(CapabilityError, match="panel budget"):
            CapytaineBackend().solve(bodies, request)

    def test_explicit_limits_admit_864_panel_case(self, monkeypatch):
        bodies, request = self._study_bodies_request()
        _mock_fresh(monkeypatch, _stub_dataset(bodies, request))
        backend = CapytaineBackend(
            execution_limits=CapytaineExecutionLimits(max_total_panels=864)
        )
        dataset = backend.solve(bodies, request)
        validate_hydro_dataset(dataset)


class TestProvenanceAndDiagnostics:
    def test_diagnose_reports_effective_limits(self):
        limits = CapytaineExecutionLimits(max_finite_omegas=5)
        codes = {d.code: d for d in CapytaineBackend(execution_limits=limits).diagnose()}
        details = codes["development_budgets"].details
        assert details["effective_limits"] == limits.as_dict()
        assert details["default_limits"] == CapytaineExecutionLimits().as_dict()

    def test_capabilities_note_records_limits(self):
        limits = CapytaineExecutionLimits(max_finite_omegas=5)
        notes = CapytaineBackend(execution_limits=limits).capabilities().notes
        assert any(limits.describe() in note for note in notes)
        assert any("no qualification" in note for note in notes)

    def test_solve_fresh_wires_limits_into_fill_dataset(self, monkeypatch):
        # structural check: the fill_dataset call site uses the effective
        # limits (n_jobs/n_threads), not hard-coded values
        source = open(cb.__file__, encoding="utf-8").read()
        assert "n_jobs=self._limits.n_jobs" in source
        assert "n_threads=self._limits.n_threads" in source
