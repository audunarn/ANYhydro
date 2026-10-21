"""Shared fixtures for the anyhydro test suite."""

from __future__ import annotations

import pytest

import anyhydro
from anyhydro.fixtures import make_fixture_bodies, make_fixture_request, make_two_body_fixture


@pytest.fixture(scope="session")
def fixture_bodies():
    return make_fixture_bodies()


@pytest.fixture(scope="session")
def fixture_request():
    return make_fixture_request()


@pytest.fixture()
def two_body_dataset():
    return make_two_body_fixture()


@pytest.fixture()
def two_body_dataset_no_a_inf():
    return make_two_body_fixture(include_added_mass_infinite=False)


# --- test tiers (CI speed-up S2/S3): ``slow`` mark from the recorded duration table, and an id filter ----------
def _ci_tiers():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "tools" / "ci_tiers.py"
    spec = importlib.util.spec_from_file_location("anyhydro_ci_tiers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: test recorded as slow in reports/ci/durations.json (full tier only)")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Mark the tests the duration table records as slow, before ``-m`` deselection, so ``-m slow`` and
    ``-m "not slow"`` partition the suite."""
    tiers = _ci_tiers()
    table = tiers.load_table()
    for item in items:
        if tiers.is_slow(table, item.nodeid):
            item.add_marker(pytest.mark.slow)


@pytest.hookimpl(trylast=True)
def pytest_collection_finish(session):
    """With ``ANYHYDRO_TEST_IDS_FILE=<file>`` (one node id per line) keep only those of the selected tests
    (``tools/dev_checks.py`` gives each parallel job its share this way)."""
    import os
    from pathlib import Path

    ids_file = os.environ.get("ANYHYDRO_TEST_IDS_FILE")
    if not ids_file:
        return
    keep_ids = set(Path(ids_file).read_text(encoding="utf-8").splitlines())
    keep = [item for item in session.items if item.nodeid in keep_ids]
    dropped = [item for item in session.items if item.nodeid not in keep_ids]
    if dropped:
        session.config.hook.pytest_deselected(items=dropped)
        session.items[:] = keep
