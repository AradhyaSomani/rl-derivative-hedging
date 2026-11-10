"""Do nothing: hold no shares."""

from __future__ import annotations

import numpy as np


def no_hedge_holdings(paths: np.ndarray) -> np.ndarray:
    paths = np.atleast_2d(paths)
    return np.zeros((paths.shape[0], paths.shape[1] - 1))
