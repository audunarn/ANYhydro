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
