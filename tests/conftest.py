import pytest

from sixth.data import synthetic_bars
from sixth.graph import HypothesisGraph


@pytest.fixture
def graph(tmp_path):
    g = HypothesisGraph(str(tmp_path / "test.sqlite"))
    yield g
    g.close()


@pytest.fixture(scope="session")
def bars():
    return synthetic_bars(n=1500, seed=11)
