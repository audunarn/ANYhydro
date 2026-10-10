"""Regression: dataset hash identity across the execution_limits attribute.

The first-slice pilot exposed a hash-identity defect:
``schema.build_hydro_dataset`` stores ``extra_attrs`` dict values as JSON
strings, so ``execution_limits`` is a *string* in the freshly solved dataset,
while persistence decodes it back to a *dict* on load. Content hashing must
canonicalize both representations identically, and save/load/re-save must
preserve the hash and the effective limits.

All data here is the synthetic development fixture; no BEM solve, no
capytaine import, no backend objects.
"""

from __future__ import annotations

import json

import anyhydro
from anyhydro.fixtures import make_two_body_fixture

LIMITS = {
    "max_finite_omegas": 3,
    "max_total_panels": 100,
    "max_wave_directions": 1,
    "n_jobs": 1,
    "n_threads": 1,
}

# Pre-repair content hash of the plain synthetic fixture (which carries no
# execution_limits attribute), recorded in evidence run
# first-slice-prep-repair-01 before the hashing repair: datasets without the
# new attribute must keep their hash.
FIXTURE_HASH_WITHOUT_EXECUTION_LIMITS = (
    "2db117e6b2b19b5bf9278cb21410d3aff2e266cc33f14bbc2646e05cd6fd52b9"
)


def _mocked_solved_dataset():
    """Synthetic fixture carrying the backend's provenance attributes."""
    ds = make_two_body_fixture()
    ds.attrs["adapter"] = "anyhydro-capytaine-development-adapter"
    ds.attrs["adapter_version"] = "test"
    ds.attrs["development_limits"] = (
        "max_total_panels=100, max_finite_omegas=3, max_wave_directions=1"
    )
    # exactly what schema.build_hydro_dataset does for extra_attrs dicts:
    # a JSON string with sort_keys and default separators
    ds.attrs["execution_limits"] = json.dumps(LIMITS, sort_keys=True)
    return ds


class TestExecutionLimitsHashIdentity:
    def test_string_and_dict_forms_hash_identically(self):
        string_form = _mocked_solved_dataset()
        dict_form = string_form.copy()
        dict_form.attrs["execution_limits"] = json.loads(
            string_form.attrs["execution_limits"]
        )
        assert isinstance(string_form.attrs["execution_limits"], str)
        assert isinstance(dict_form.attrs["execution_limits"], dict)
        assert (
            anyhydro.dataset_content_hash(string_form)
            == anyhydro.dataset_content_hash(dict_form)
        )

    def test_roundtrip_preserves_hash_and_effective_limits(self, tmp_path):
        ds = _mocked_solved_dataset()
        before = anyhydro.dataset_content_hash(ds)
        path = tmp_path / "mocked.nc"
        anyhydro.save_hydro_dataset(ds, path)
        loaded = anyhydro.load_hydro_dataset(path)
        assert anyhydro.dataset_content_hash(loaded) == before
        assert loaded.attrs["execution_limits"] == LIMITS
        # re-save the reloaded dataset and load again: identity is stable
        path2 = tmp_path / "mocked-resaved.nc"
        anyhydro.save_hydro_dataset(loaded, path2)
        reloaded = anyhydro.load_hydro_dataset(path2)
        assert anyhydro.dataset_content_hash(reloaded) == before
        assert reloaded.attrs["execution_limits"] == LIMITS

    def test_dataset_without_execution_limits_hash_unchanged(self):
        # Portable legacy oracle (gui-ci-platform-repair-20261010): the golden
        # hash binds exact artifact bytes; a newly generated reference
        # artifact reproducing the legacy golden canonical content hash is
        # frozen under tests/fixtures/legacy-compatibility/ (see
        # provenance.json there; the original historical serialized file
        # bytes were not recovered).
        # The recomputed synthetic fixture is not bit-stable across numeric
        # environments, so the literal golden is verified on identical
        # immutable inputs.
        from pathlib import Path

        legacy = (
            Path(__file__).parent
            / "fixtures"
            / "legacy-compatibility"
            / "two_body_fixture_legacy.nc"
        )
        ds = anyhydro.load_hydro_dataset(legacy)
        assert "execution_limits" not in ds.attrs
        assert (
            anyhydro.dataset_content_hash(ds)
            == FIXTURE_HASH_WITHOUT_EXECUTION_LIMITS
        )
