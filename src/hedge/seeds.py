"""Seed ranges for every path set, and the per-path random draws.

One seed makes one path. Path ``i`` of a split is always built from seed ``start + i``, so a
path is the same whichever batch it is generated in, and two scenarios that share seeds share
the same underlying shocks (common random numbers).

The ranges below are fixed by ``docs/preregistration.md``. Test seeds are never used for
training, tuning or unit tests; ``tests/test_seeds.py`` fails if the ranges overlap.
"""

from __future__ import annotations

import numpy as np

# Half-open ranges [start, stop).
SEED_RANGES: dict[str, range] = {
    "dev": range(0, 10_000),  # unit tests, notebooks, the demo's "Try it" page
    "test": range(10_000, 20_000),  # opened once, after the agents-frozen tag
    "validation": range(20_000, 30_000),  # reward choice, tuning, early stopping
    "train": range(100_000, 200_000),  # training only
}

SPLITS = ("train", "validation", "test")


class TestSeedError(RuntimeError):
    """Raised when test seeds are requested without saying so explicitly."""

    __test__ = False  # not a pytest class


def split_seeds(split: str, n_paths: int | None = None) -> np.ndarray:
    """Seeds for a split, in order. ``n_paths`` takes the first n (default: all)."""
    rng = SEED_RANGES[split]
    n = len(rng) if n_paths is None else n_paths
    if not 0 < n <= len(rng):
        raise ValueError(f"split {split!r} has {len(rng)} seeds, asked for {n}")
    return np.arange(rng.start, rng.start + n, dtype=np.int64)


def contains_test_seeds(seeds) -> bool:
    seeds = np.asarray(seeds)
    rng = SEED_RANGES["test"]
    return bool(((seeds >= rng.start) & (seeds < rng.stop)).any())


def guard_test_seeds(seeds, allow_test: bool) -> None:
    """Refuse to touch test seeds unless the caller opts in."""
    if not allow_test and contains_test_seeds(seeds):
        raise TestSeedError(
            "These seeds include test seeds (10,000 to 19,999). Test paths are opened once, "
            "after the agents-frozen tag. Pass allow_test=True only for that final run."
        )


def standard_normals(seeds, n_draws: int) -> np.ndarray:
    """Array of shape (len(seeds), n_draws): row i holds the first n draws of seed i's stream."""
    seeds = np.asarray(seeds, dtype=np.int64)
    if seeds.ndim != 1:
        raise ValueError("seeds must be one-dimensional")
    out = np.empty((seeds.size, n_draws))
    for i, seed in enumerate(seeds):
        np.random.Generator(np.random.PCG64(int(seed))).standard_normal(n_draws, out=out[i])
    return out
