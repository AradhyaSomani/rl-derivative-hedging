import numpy as np

from hedge.market.gbm import simulate_gbm

DT = 1 / 252


def test_shape_and_start():
    paths = simulate_gbm(np.arange(100), n_steps=30, dt=DT, s0=100.0)
    assert paths.shape == (100, 31)
    assert (paths[:, 0] == 100.0).all()
    assert (paths > 0).all()


def test_reproducible():
    a = simulate_gbm(np.arange(50), 30, DT)
    b = simulate_gbm(np.arange(50), 30, DT)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[10:20], simulate_gbm(np.arange(10, 20), 30, DT))


def test_log_return_moments_match_theory():
    mu, sigma, n_paths, n_steps = 0.08, 0.2, 5_000, 30
    paths = simulate_gbm(np.arange(n_paths), n_steps, DT, mu=mu, sigma=sigma)
    log_returns = np.diff(np.log(paths), axis=1).ravel()
    n = log_returns.size
    mean_theory = (mu - 0.5 * sigma**2) * DT
    var_theory = sigma**2 * DT
    # 4 standard errors: the mean has s.e. sqrt(var / n), the variance s.e. var * sqrt(2 / n).
    assert abs(log_returns.mean() - mean_theory) < 4 * np.sqrt(var_theory / n)
    assert abs(log_returns.var(ddof=1) - var_theory) < 4 * var_theory * np.sqrt(2 / n)


def test_terminal_price_is_a_martingale_when_drift_is_zero():
    n_paths = 5_000
    s_T = simulate_gbm(np.arange(n_paths), 30, DT, mu=0.0, sigma=0.2)[:, -1]
    assert abs(s_T.mean() - 100.0) < 4 * s_T.std(ddof=1) / np.sqrt(n_paths)
