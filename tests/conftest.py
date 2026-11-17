import pytest

from tests.synthetic import synthetic_table


@pytest.fixture
def table():
    """A fresh synthetic stats table & tests may modify it."""
    return synthetic_table()
