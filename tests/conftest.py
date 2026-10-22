import pytest

from featurescope import Project
from featurescope.synthetic import make_dataset


@pytest.fixture(scope="session")
def df():
    return make_dataset(3000, n_slides=3, seed=1)


@pytest.fixture
def project(tmp_path, df):
    return Project.create(tmp_path / "proj", df)
