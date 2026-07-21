import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edf.data.synthetic import generate_synthetic_dataset  # noqa: E402
from edf.features.build_features import build_feature_frame  # noqa: E402


@pytest.fixture(scope="session")
def raw_dataset():
    return generate_synthetic_dataset(start="2023-01-01", end="2023-04-01", seed=1)


@pytest.fixture(scope="session")
def feature_frame(raw_dataset):
    return build_feature_frame(raw_dataset)
