"""Coefficient frame tests: physical oracles, full cross-body maps, refusals.

Oracle tolerance: 1e-11 normalized with a declared floor of 1e-30
(adopted W1 gate). Development fixtures confer no scientific qualification.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from anyhydro import (
    CoefficientFrame,
    SchemaError,
    TransformError,
    UnsupportedOperationError,
    coefficient_frame_axes,
    dataset_content_hash,
    frame_change_matrix,
    load_hydro_dataset,
    reframe_dataset,
    require_global_dof_axes,
    rotate_axes,
    save_hydro_dataset,
    shift_dataset_reference_point,
    validate_hydro_dataset,
)
from anyhydro.fixtures import FIXTURE_BODY_IDS, make_two_body_fixture

R_TOL = 1e-11
FLOOR = 1e-30  # declared normalization floor

# Pre-edit baseline evidence (hydro-frames-worker-01/check-01-baseline):
# legacy fixture/shift hashes must remain unchanged by this extension.
BASELINE_FIXTURE_HASH = (
    "2db117e6b2b19b5bf9278cb21410d3aff2e266cc33f14bbc2646e05cd6fd52b9"
)
BASELINE_SHIFTED_HASH = (
    "70813430e6343da3d67dda4ca8dc7c8d3a900e409ffb3c643502e015236c6022"
)

ALPHA, BETA = FIXTURE_BODY_IDS[0], FIXTURE_BODY_IDS[1]


def _rot(axis, angle) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _velocity_map_T(E_old, O_old, E_new, O_new) -> np.ndarray:
    """Independent oracle: build T column-wise from the physical velocity
    field of each unit new-frame DOF, expressed in the old frame."""
    E_old = np.asarray(E_old, dtype=np.float64)
    E_new = np.asarray(E_new, dtype=np.float64)
    O_old = np.asarray(O_old, dtype=np.float64)
    O_new = np.asarray(O_new, dtype=np.float64)
    T = np.zeros((6, 6))
    for k in range(6):
        q_new = np.zeros(6)
        q_new[k] = 1.0
        u_new, th_new = q_new[:3], q_new[3:]
        omega_world = E_new @ th_new
        v_Oold = E_new @ u_new + np.cross(omega_world, O_old - O_new)
        T[:3, k] = E_old.T @ v_Oold
        T[3:, k] = E_old.T @ omega_world
    return T


def _norm_err(a, b) -> float:
    scale = max(float(np.max(np.abs(b))), FLOOR)
    return float(np.max(np.abs(a - b))) / scale


def _new_frames():
    return {
        ALPHA: (_rot((0.0, 0.0, 1.0), 0.35), (1.0, -0.5, 0.25)),
        BETA: (_rot((1.0, 2.0, -0.5), -1.1), (12.0, 0.0, -0.5)),
    }


def _reframe(ds):
    frames = _new_frames()
    return reframe_dataset(
        ds,
        new_axes={b: frames[b][0] for b in frames},
        new_reference_centers={b: frames[b][1] for b in frames},
    )


OLD_FRAMES = {
    # legacy fixture frames: GLOBAL axes, stored reference centers
    ALPHA: (np.eye(3), (0.0, 0.0, 0.0)),
    BETA: (np.eye(3), (12.0, 0.0, -0.5)),
}


def _block_T(keys) -> np.ndarray:
    """Manual global map from the velocity-field oracle, arbitrary DOF order."""
    from anyhydro import parse_dof_key

    new_frames = _new_frames()
    n = len(keys)
    T = np.zeros((n, n))
    pos = {k: i for i, k in enumerate(keys)}
    for body in {parse_dof_key(k)[0] for k in keys}:
        E_old, O_old = OLD_FRAMES[body]
        E_new, O_new = new_frames[body]
        block = _velocity_map_T(E_old, O_old, E_new, O_new)
        positions = [pos[f"{body}::{d}"] for d in
                     ("surge", "sway", "heave", "roll", "pitch", "yaw")]
        T[np.ix_(positions, positions)] = block
    return T


class TestFrameMapOracles:
    def _random_case(self, seed):
        rng = np.random.default_rng(seed)
        E_old = _rot(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
        E_new = _rot(rng.normal(size=3), rng.uniform(-np.pi, np.pi))
        O_old = rng.normal(size=3) * 10.0
        O_new = rng.normal(size=3) * 10.0
        return E_old, O_old, E_new, O_new

    def test_velocity_field_oracle_matches_closed_form(self):
        for seed in range(4):
            E_old, O_old, E_new, O_new = self._random_case(seed)
            T = frame_change_matrix(
                CoefficientFrame(E_old, O_old), CoefficientFrame(E_new, O_new)
            )
            oracle = _velocity_map_T(E_old, O_old, E_new, O_new)
            assert np.allclose(T, oracle, atol=1e-12, rtol=0.0), seed

    def test_physical_wrench_oracle(self):
        rng = np.random.default_rng(7)
        for seed in range(4):
            E_old, O_old, E_new, O_new = self._random_case(seed)
            T = frame_change_matrix(
                CoefficientFrame(E_old, O_old), CoefficientFrame(E_new, O_new)
            )
            f_world = rng.normal(size=3) * 1.0e4
            m_world = rng.normal(size=3) * 5.0e3
            F_old = np.concatenate([E_old.T @ f_world, E_old.T @ m_world])
            F_new = T.T @ F_old
            m_new_world = m_world + np.cross(O_old - O_new, f_world)
            expected = np.concatenate(
                [E_new.T @ f_world, E_new.T @ m_new_world]
            )
            assert _norm_err(F_new, expected) <= R_TOL, seed

    def test_virtual_work_oracle(self):
        rng = np.random.default_rng(11)
        for seed in range(4):
            E_old, O_old, E_new, O_new = self._random_case(seed)
            T = frame_change_matrix(
                CoefficientFrame(E_old, O_old), CoefficientFrame(E_new, O_new)
            )
            q_new = np.concatenate(
                [rng.normal(size=3), rng.normal(size=3) * 0.01]
            )
            F_old = np.concatenate(
                [rng.normal(size=3) * 1.0e4, rng.normal(size=3) * 5.0e3]
            )
            q_old = T @ q_new
            F_new = T.T @ F_old
            work_new = float(F_new @ q_new)
            work_old = float(F_old @ q_old)
            scale = max(abs(work_new), abs(work_old), FLOOR)
            assert abs(work_new - work_old) / scale <= R_TOL, seed


class TestDatasetFrameChange:
    def test_full_12x12_matches_manual_oracle(self, two_body_dataset):
        ds = two_body_dataset
        reframed = _reframe(ds)
        infl = [str(k) for k in ds["influenced_dof"].values.tolist()]
        rad = [str(k) for k in ds["radiating_dof"].values.tolist()]
        Ti = _block_T(infl)
        Tr = _block_T(rad)
        for var in ("added_mass", "radiation_damping"):
            expected = np.einsum("ai,oab,bj->oij", Ti, ds[var].values, Tr)
            assert _norm_err(reframed[var].values, expected) <= R_TOL, var
        expected_c = Ti.T @ ds["hydrostatic_stiffness"].values @ Tr
        assert _norm_err(reframed["hydrostatic_stiffness"].values, expected_c) <= R_TOL
        expected_ainf = Ti.T @ ds["added_mass_infinite"].values @ Tr
        assert _norm_err(reframed["added_mass_infinite"].values, expected_ainf) <= R_TOL
        expected_f = np.einsum("ai,owa->owi", Ti, ds["excitation_force"].values)
        assert _norm_err(reframed["excitation_force"].values, expected_f) <= R_TOL

    def test_independent_axis_permutations(self, two_body_dataset):
        ds = two_body_dataset
        canonical = _reframe(ds)
        infl = [str(k) for k in ds["influenced_dof"].values.tolist()]
        rng = np.random.default_rng(3)
        perm_i = list(rng.permutation(infl))
        perm_r = list(rng.permutation(infl))
        shuffled = ds.sel(influenced_dof=perm_i, radiating_dof=perm_r)
        reframed = _reframe(shuffled)
        for var in ("added_mass", "radiation_damping"):
            for ki in infl:
                for kj in infl:
                    a = canonical[var].sel(influenced_dof=ki, radiating_dof=kj).values
                    b = reframed[var].sel(influenced_dof=ki, radiating_dof=kj).values
                    assert _norm_err(b, a) <= R_TOL, (var, ki, kj)
        for ki in infl:
            a = canonical["excitation_force"].sel(influenced_dof=ki).values
            b = reframed["excitation_force"].sel(influenced_dof=ki).values
            assert _norm_err(b, a) <= R_TOL, ki

    def test_sequential_inverse_equivalence(self, two_body_dataset):
        ds = two_body_dataset
        frames = _new_frames()
        old_centers = {
            b: tuple(
                float(v)
                for v in ds["rotation_center"].sel(body=b).values
            )
            for b in (ALPHA, BETA)
        }
        reframed = _reframe(ds)
        back = reframe_dataset(
            reframed,
            new_axes={ALPHA: np.eye(3), BETA: np.eye(3)},
            new_reference_centers=old_centers,
        )
        for var in ("added_mass", "radiation_damping", "hydrostatic_stiffness",
                    "added_mass_infinite", "excitation_force"):
            assert _norm_err(back[var].values, ds[var].values) <= R_TOL, var
        # history differs: the hash is NOT claimed unchanged after inverse
        assert dataset_content_hash(back) != dataset_content_hash(ds)

    def test_geometry_and_conventions_unchanged(self, two_body_dataset):
        ds = two_body_dataset
        reframed = _reframe(ds)
        for b, (_, center) in _new_frames().items():
            assert np.allclose(
                reframed["rotation_center"].sel(body=b).values, center
            ), b
        for var in ("center_of_mass", "center_of_buoyancy"):
            assert np.array_equal(reframed[var].values, ds[var].values), var
        assert reframed.attrs["phasor_convention"] == ds.attrs["phasor_convention"]
        assert reframed.attrs["heading_convention"] == ds.attrs["heading_convention"]
        assert np.array_equal(reframed["omega"].values, ds["omega"].values)
        assert np.array_equal(
            reframed["wave_direction"].values, ds["wave_direction"].values
        )

    def test_split_mapping(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        rng = np.random.default_rng(5)
        shape = ds["excitation_force"].shape
        for name in ("froude_krylov_force", "diffraction_force", "mean_drift_force"):
            ds[name] = (
                ("omega", "wave_direction", "influenced_dof"),
                (rng.normal(size=shape) + 1j * rng.normal(size=shape)) * 100.0,
            )
        reframed = _reframe(ds)
        infl = [str(k) for k in ds["influenced_dof"].values.tolist()]
        Ti = _block_T(infl)
        for name in ("froude_krylov_force", "diffraction_force", "mean_drift_force"):
            expected = np.einsum("ai,owa->owi", Ti, ds[name].values)
            assert _norm_err(reframed[name].values, expected) <= R_TOL, name

    def test_kochin_refusal_before_mutation(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        ds["kochin_radiation"] = (
            ("omega", "wave_direction"),
            np.ones((ds.sizes["omega"], ds.sizes["wave_direction"])),
        )
        before = dataset_content_hash(ds)
        with pytest.raises(TransformError, match="kochin_radiation"):
            _reframe(ds)
        assert dataset_content_hash(ds) == before

    def test_pressure_field_refusal(self, two_body_dataset):
        ds = two_body_dataset.copy(deep=True)
        ds["panel_pressure"] = (
            ("omega", "wave_direction", "influenced_dof"),
            np.ones(ds["excitation_force"].shape),
        )
        with pytest.raises(TransformError, match="panel_pressure"):
            _reframe(ds)

    def test_provenance_and_metadata(self, two_body_dataset):
        ds = two_body_dataset
        pre_hash = dataset_content_hash(ds)
        reframed = _reframe(ds)
        frames = _new_frames()
        recorded = json.loads(reframed.attrs["coefficient_frames"])
        for b, (E, O) in frames.items():
            assert np.allclose(np.array(recorded[b]["axes"]), E), b
            assert np.allclose(recorded[b]["reference_center"], O), b
        transforms = json.loads(reframed.attrs["reference_transformations"])
        assert len(transforms) == 2
        for record in transforms:
            assert record["kind"] == "coefficient_frame_change"
            assert record["map_id"] == "equilibrium_linear_v1"
            assert record["source_content_hash"] == pre_hash
            assert "inverse" in record and "old_axes" in record and "new_axes" in record
        axes = coefficient_frame_axes(reframed)
        for b, (E, _) in frames.items():
            assert np.allclose(axes[b], E), b

    def test_require_global_dof_axes_gate(self, two_body_dataset):
        ds = two_body_dataset
        assert require_global_dof_axes(ds) is ds
        axes = coefficient_frame_axes(ds)
        for b in (ALPHA, BETA):
            assert np.array_equal(axes[b], np.eye(3)), b
        reframed = _reframe(ds)
        with pytest.raises(TransformError, match="non-global"):
            require_global_dof_axes(reframed)

    def test_rotate_axes_only(self, two_body_dataset):
        ds = two_body_dataset
        frames = _new_frames()
        rotated = rotate_axes(ds, {b: frames[b][0] for b in frames})
        assert np.array_equal(
            rotated["rotation_center"].values, ds["rotation_center"].values
        )
        recorded = json.loads(rotated.attrs["coefficient_frames"])
        for b, (E, _) in frames.items():
            assert np.allclose(np.array(recorded[b]["axes"]), E), b
            assert np.allclose(
                recorded[b]["reference_center"],
                ds["rotation_center"].sel(body=b).values,
            ), b

    def test_rotate_axes_argument_free_refusal(self, two_body_dataset):
        with pytest.raises(UnsupportedOperationError):
            rotate_axes(two_body_dataset)
        with pytest.raises(UnsupportedOperationError):
            rotate_axes(None)

    def test_reframe_requires_target(self, two_body_dataset):
        with pytest.raises(TransformError, match="nothing to reframe"):
            reframe_dataset(two_body_dataset)

    def test_bad_inputs_refused_no_mutation(self, two_body_dataset):
        ds = two_body_dataset
        before = dataset_content_hash(ds)
        shear = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        reflection = np.diag([1.0, 1.0, -1.0])
        bad_axes = {
            "shear": {ALPHA: shear, BETA: np.eye(3)},
            "reflection": {ALPHA: reflection, BETA: np.eye(3)},
            "nonfinite": {ALPHA: np.eye(3), BETA: np.array([[np.nan, 0, 0], [0, 1, 0], [0, 0, 1]])},
            "incomplete": {ALPHA: np.eye(3)},
            "foreign": {ALPHA: np.eye(3), BETA: np.eye(3), "other": np.eye(3)},
            "shape": {ALPHA: np.eye(4), BETA: np.eye(3)},
        }
        for label, axes in bad_axes.items():
            with pytest.raises(TransformError):
                rotate_axes(ds, axes)
            with pytest.raises(TransformError):
                reframe_dataset(ds, new_axes=axes)
        with pytest.raises(TransformError):
            reframe_dataset(
                ds,
                new_reference_centers={ALPHA: (0.0, 0.0, 0.0), BETA: (np.inf, 0, 0)},
            )
        assert dataset_content_hash(ds) == before


class TestFramedShift:
    def _framed(self, ds):
        frames = _new_frames()
        return rotate_axes(ds, {b: frames[b][0] for b in frames})

    def test_shift_matches_reframe_on_framed_dataset(self, two_body_dataset):
        framed = self._framed(two_body_dataset)
        new_centers = {ALPHA: (2.0, 1.0, -1.0), BETA: (10.0, 3.0, 0.0)}
        shifted = shift_dataset_reference_point(framed, new_centers)
        reframed = reframe_dataset(framed, new_reference_centers=new_centers)
        for var in ("added_mass", "radiation_damping", "hydrostatic_stiffness",
                    "added_mass_infinite", "excitation_force"):
            assert np.allclose(
                shifted[var].values, reframed[var].values, rtol=R_TOL, atol=0.0
            ), var
        recorded = json.loads(shifted.attrs["coefficient_frames"])
        for b, c in new_centers.items():
            assert np.allclose(recorded[b]["reference_center"], c), b
        transforms = json.loads(shifted.attrs["reference_transformations"])
        shift_records = [t for t in transforms if t["kind"] == "translation_shift"]
        assert len(shift_records) == 2
        assert all(t["d_coordinates"] == "old_coefficient_basis" for t in shift_records)

    def test_legacy_shift_record_unchanged(self, two_body_dataset):
        ds = two_body_dataset
        shifted = shift_dataset_reference_point(
            ds, {ALPHA: (1.0, -0.5, 0.25), BETA: (12.0, 0.0, -0.5)}
        )
        assert "coefficient_frames" not in shifted.attrs
        transforms = json.loads(shifted.attrs["reference_transformations"])
        assert all("d_coordinates" not in t for t in transforms)


class TestLegacyUnchanged:
    def test_legacy_hashes_match_preedit_baseline(self):
        from anyhydro.fixtures import FIXTURE_BODY_IDS

        ds = make_two_body_fixture()
        assert dataset_content_hash(ds) == BASELINE_FIXTURE_HASH
        shifted = shift_dataset_reference_point(
            ds,
            {FIXTURE_BODY_IDS[0]: (1.0, -0.5, 0.25), FIXTURE_BODY_IDS[1]: (12.0, 0.0, -0.5)},
        )
        assert dataset_content_hash(shifted) == BASELINE_SHIFTED_HASH


class TestSchemaAndSerialization:
    def test_schema_rejects_malformed_frames(self, two_body_dataset):
        reframed = _reframe(two_body_dataset)
        payload = json.loads(reframed.attrs["coefficient_frames"])
        cases = []
        bad = json.loads(json.dumps(payload))
        bad[ALPHA]["axes"][0][0] = 5.0  # shear
        cases.append(json.dumps(bad))
        bad = json.loads(json.dumps(payload))
        bad[ALPHA]["axes"] = np.diag([1.0, 1.0, -1.0]).tolist()  # reflection
        cases.append(json.dumps(bad))
        bad = json.loads(json.dumps(payload))
        bad[ALPHA]["reference_center"] = [999.0, 0.0, 0.0]  # center mismatch
        cases.append(json.dumps(bad))
        bad = json.loads(json.dumps(payload))
        del bad[BETA]  # incomplete coverage
        cases.append(json.dumps(bad))
        cases.append("{not json")
        for raw in cases:
            ds = reframed.copy(deep=True)
            ds.attrs["coefficient_frames"] = raw
            with pytest.raises(SchemaError):
                validate_hydro_dataset(ds)

    def test_hash_detects_frames_tamper(self, two_body_dataset):
        reframed = _reframe(two_body_dataset)
        tampered = reframed.copy(deep=True)
        payload = json.loads(tampered.attrs["coefficient_frames"])
        payload[ALPHA]["axes"] = _rot((1.0, 0.0, 0.0), 0.2).tolist()
        tampered.attrs["coefficient_frames"] = json.dumps(payload, sort_keys=True)
        assert dataset_content_hash(tampered) != dataset_content_hash(reframed)

    def test_hash_invariant_under_reorder_with_frames(self, two_body_dataset):
        from anyhydro import reorder_dataset

        reframed = _reframe(two_body_dataset)
        dofs = [str(k) for k in reframed["influenced_dof"].values.tolist()]
        shuffled = reframed.sel(
            influenced_dof=dofs[6:12] + dofs[0:6],
            radiating_dof=dofs[6:12] + dofs[0:6],
        )
        assert dataset_content_hash(shuffled) == dataset_content_hash(reframed)
        assert (
            dataset_content_hash(reorder_dataset(shuffled))
            == dataset_content_hash(reframed)
        )

    def test_persistence_roundtrip_with_frames(self, two_body_dataset, tmp_path):
        reframed = _reframe(two_body_dataset)
        path = tmp_path / "reframed.nc"
        save_hydro_dataset(reframed, path)
        loaded = load_hydro_dataset(path)
        assert dataset_content_hash(loaded) == dataset_content_hash(reframed)
        # decoded dict form validates and reads back identically
        validate_hydro_dataset(loaded)
        axes = coefficient_frame_axes(loaded)
        for b, (E, _) in _new_frames().items():
            assert np.allclose(axes[b], E), b
        for var in ("added_mass", "excitation_force", "hydrostatic_stiffness"):
            assert np.array_equal(loaded[var].values, reframed[var].values), var
