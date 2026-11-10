import numpy as np
import pytest

from hedge.market.gbm import simulate_gbm
from hedge.market.heston import simulate_heston
from hedge.seeds import (
    SEED_RANGES,
    TestSeedError,
    contains_test_seeds,
    split_seeds,
    standard_normals,
)


def test_seed_ranges_do_not_overlap():
    names = list(SEED_RANGES)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ra, rb = SEED_RANGES[a], SEED_RANGES[b]
            assert ra.stop <= rb.start or rb.stop <= ra.start, f"{a} and {b} seeds overlap"


def test_preregistered_ranges():
    # Fixed in docs/preregistration.md. Changing these invalidates the pre-registration.
    assert SEED_RANGES["test"] == range(10_000, 20_000)
    assert SEED_RANGES["validation"] == range(20_000, 30_000)
    assert SEED_RANGES["train"] == range(100_000, 200_000)


def test_split_sizes():
    assert split_seeds("train").size == 100_000
    assert split_seeds("validation").size == 10_000
    assert split_seeds("test").size == 10_000
    assert not contains_test_seeds(split_seeds("train"))
    assert not contains_test_seeds(split_seeds("validation"))
    assert contains_test_seeds(split_seeds("test"))


def test_simulators_refuse_test_seeds():
    with pytest.raises(TestSeedError):
        simulate_gbm([9_999, 10_000], n_steps=5, dt=1 / 252)
    with pytest.raises(TestSeedError):
        simulate_heston([19_999], n_steps=5, dt=1 / 252)
    simulate_gbm([9_999, 20_000], n_steps=5, dt=1 / 252)  # either side of the range is fine


def test_one_seed_one_path():
    # A path depends only on its own seed, not on the batch it is generated in.
    a = standard_normals([3, 7, 11], 50)
    b = standard_normals([11, 3], 50)
    np.testing.assert_array_equal(a[2], b[0])
    np.testing.assert_array_equal(a[0], b[1])
    assert not np.allclose(a[0], a[1])
    # A longer draw extends the same stream.
    np.testing.assert_array_equal(standard_normals([3], 80)[0, :50], a[0])
