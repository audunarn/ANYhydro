"""Reference-point transformation tests.

Work invariance target: 1e-11 relative tolerance (adopted W1 gate).
"""

from __future__ import annotations

import numpy as np
import pytest

from anyhydro import (
    UnsupportedOperationError,
    dataset_content_hash,
    map_force,
    map_matrix,
    reorder_dataset,
    shift_dataset_reference_point,
    shift_matrix,
    skew,
)
from anyhydro.fixtures import FIXTURE_BODY_IDS


R_TOL = 1e-11

BETA_CENTER = (12.0, 0.0, -0.5)


def _shift_alpha(ds, r=(1.0, -0.5, 0.25)):
    """Shift only body alpha; beta keeps its existing rotation center."""
    return shift_dataset_reference_point(
        ds, {FIXTURE_BODY_IDS[0]: r, FIXTURE_BODY_IDS[1]: BETA_CENTER}
    )


class TestShiftMatrices:
    def test_skew_cross_product_identity(self):
        r = np.array([1.0, -2.0, 0.5])
        theta = np.array([0.3, 0.7, -1.1])
        assert np.allclose(skew(r) @ theta, np.cross(r, theta), atol=0.0)

    def test_shift_matrix_structure(self):
        T = shift_matrix((0.4, 0.5, 0.6))
        assert np.allclose(T[:3, :3], np.eye(3))
        assert np.allclose(T[:3, 3:], skew((0.4, 0.5, 0.6)))
        assert np.allclose(T[3:, 3:], np.eye(3))
        assert np.allclose(T[3:, :3], 0.0)

    def test_analytic_force_shift(self):
        # force f at old origin, no moment: after shifting origin by r the
        # moment about the new origin is f x r... equivalently F_new = T^T F_old
        r = (2.0, 0.0, 0.0)
        f = np.array([0.0, 3.0, 0.0])
        F_old = np.concatenate([f, np.zeros(3)])
        F_new = map_force(F_old, shift_matrix(r))
        # moment about new origin = (point - new_origin) x f = (-r) x f
        assert np.allclose(F_new[:3], f)
        assert np.allclose(F_new[3:], np.cross(-np.asarray(r), f))

    def test_matrix_mapping_matches_manual(self):
        r = (0.7, -0.3, 0.2)
        T = shift_matrix(r)
        A = np.random.default_rng(0).normal(size=(6, 6))
        assert np.allclose(map_matrix(A, T, T), T.T @ A @ T)


class TestDatasetShift:
    def test_shift_round_trip(self, two_body_dataset):
        ds = two_body_dataset
        shifted = _shift_alpha(ds)
        # shift back
        back = shift_dataset_reference_point(
            shifted,
            {
                FIXTURE_BODY_IDS[0]: tuple(
                    ds["rotation_center"].sel(body=FIXTURE_BODY_IDS[0]).values
                ),
                FIXTURE_BODY_IDS[1]: tuple(
                    ds["rotation_center"].sel(body=FIXTURE_BODY_IDS[1]).values
                ),
            },
        )
        for var in ("added_mass", "radiation_damping", "hydrostatic_stiffness",
                    "added_mass_infinite", "excitation_force", "rotation_center"):
            assert np.allclose(back[var].values, ds[var].values, rtol=R_TOL, atol=0.0), var

    def test_work_invariance_force_and_displacement(self, two_body_dataset):
        ds = two_body_dataset
        rng = np.random.default_rng(42)
        # scaled generalized displacement: translations ~ m, rotations ~ rad
        q_new = np.concatenate(
            [rng.normal(0, 1.0, size=6), rng.normal(0, 0.01, size=6)]
        )
        shifted = _shift_alpha(ds)
        # excitation force at first omega/heading on both datasets
        F_old = ds["excitation_force"].isel(omega=0, wave_direction=0).values
        F_new = shifted["excitation_force"].isel(omega=0, wave_direction=0).values
        work_old = np.vdot(q_new, F_old)  # F_old . q_new in old frame pairing
        # proper work pairing: F_old with q_old = T q_new
        T = np.zeros((12, 12))
        r = np.array([1.0, -0.5, 0.25])
        T[:6, :6] = shift_matrix(r)
        T[6:, 6:] = np.eye(6)
        q_old = T @ q_new
        work_check = np.vdot(q_old, F_old)
        work_new = np.vdot(q_new, F_new)
        scale = max(abs(work_old), abs(work_new), abs(work_check), 1.0)
        assert abs(work_new - work_check) / scale < R_TOL

    def test_shift_preserves_cross_body_blocks(self, two_body_dataset):
        shifted = _shift_alpha(two_body_dataset)
        cross = shifted["added_mass"].values[:, 0:6, 6:12]
        assert np.all(cross != 0.0)
        cross_back = shifted["added_mass"].values[:, 6:12, 0:6]
        assert np.all(cross_back != 0.0)

    def test_shift_updates_rotation_center_only(self, two_body_dataset):
        ds = two_body_dataset
        shifted = _shift_alpha(ds)
        new_rc = shifted["rotation_center"].sel(body=FIXTURE_BODY_IDS[0]).values
        assert np.allclose(new_rc, (1.0, -0.5, 0.25))
        # global geometric reference points do not move
        for var in ("center_of_mass", "center_of_buoyancy"):
            assert np.array_equal(shifted[var].values, ds[var].values), var
        # untouched body keeps its rotation center
        rc_beta = shifted["rotation_center"].sel(body=FIXTURE_BODY_IDS[1]).values
        assert np.array_equal(
            rc_beta, ds["rotation_center"].sel(body=FIXTURE_BODY_IDS[1]).values
        )

    def test_shift_records_provenance(self, two_body_dataset):
        import json

        shifted = _shift_alpha(two_body_dataset)
        transforms = json.loads(shifted.attrs["reference_transformations"])
        assert len(transforms) == 2
        assert transforms[0]["kind"] == "translation_shift"
        assert transforms[0]["body_id"] == FIXTURE_BODY_IDS[0]

    def test_shift_does_not_mutate_input(self, two_body_dataset):
        ds = two_body_dataset
        before = dataset_content_hash(ds)
        _shift_alpha(ds)
        assert dataset_content_hash(ds) == before

    def test_shift_rejects_incomplete_body_coverage(self, two_body_dataset):
        with pytest.raises(Exception):
            shift_dataset_reference_point(
                two_body_dataset, {FIXTURE_BODY_IDS[0]: (0.0, 0.0, 0.0)}
            )

    def test_shift_rejects_nonfinite_center(self, two_body_dataset):
        with pytest.raises(Exception):
            shift_dataset_reference_point(
                two_body_dataset,
                {FIXTURE_BODY_IDS[0]: (np.nan, 0.0, 0.0),
                 FIXTURE_BODY_IDS[1]: (0.0, 0.0, 0.0)},
            )


class TestReordering:
    def test_reorder_is_identity_on_canonical_fixture(self, two_body_dataset):
        ds = two_body_dataset
        reordered = reorder_dataset(ds)
        assert np.array_equal(
            reordered["added_mass"].values, ds["added_mass"].values
        )
        assert dataset_content_hash(reordered) == dataset_content_hash(ds)

    def test_reorder_restores_shuffled_axes(self, two_body_dataset):
        ds = two_body_dataset
        dofs = [str(d) for d in ds["influenced_dof"].values.tolist()]
        shuffled = dofs[6:12] + dofs[0:6]
        shuffled_ds = ds.sel(influenced_dof=shuffled, radiating_dof=shuffled)
        reordered = reorder_dataset(shuffled_ds)
        # values per stable identity are unchanged
        for key in dofs:
            a = shuffled_ds["excitation_force"].sel(influenced_dof=key).values
            b = reordered["excitation_force"].sel(influenced_dof=key).values
            assert np.array_equal(a, b), key
        assert dataset_content_hash(reordered) == dataset_content_hash(ds)

    def test_reorder_keeps_excitation_consistent(self, two_body_dataset):
        ds = two_body_dataset
        dofs = [str(d) for d in ds["influenced_dof"].values.tolist()]
        shuffled = list(reversed(dofs))
        shuffled_ds = ds.sel(influenced_dof=shuffled)
        reordered = reorder_dataset(shuffled_ds)
        for key in dofs:
            assert np.array_equal(
                shuffled_ds["excitation_force"].sel(influenced_dof=key).values,
                reordered["excitation_force"].sel(influenced_dof=key).values,
            )


class TestUnsupported:
    def test_axis_rotation_explicitly_unsupported(self):
        from anyhydro.transform import rotate_axes

        with pytest.raises(UnsupportedOperationError):
            rotate_axes(None)
