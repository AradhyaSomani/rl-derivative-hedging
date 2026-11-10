# Results

No agents are trained yet. This file holds the Phase 1 baseline numbers on **validation**
paths; test paths have not been generated or opened.

## Baselines on validation paths (10,000 paths per scenario)

Costs are per option on a stock at 100; the premium is 2.752 on GBM and 2.721 on Heston.
Regenerate with `make paths baselines`.

| Scenario | Strategy | Mean cost | Std | CVaR 95% | Txn cost | Turnover |
| --- | --- | --- | --- | --- | --- | --- |
| gbm_0bps | no_hedge | 0.014 | 4.176 | 12.178 | 0.000 | 0.00 |
| gbm_0bps | bs_delta | -0.001 | 0.433 | 1.008 | 0.000 | 2.72 |
| gbm_0bps | no_trade_band (width 0.01, to edge) | -0.002 | 0.438 | 0.994 | 0.000 | 2.48 |
| gbm_5bps | no_hedge | 0.014 | 4.176 | 12.178 | 0.000 | 0.00 |
| gbm_5bps | bs_delta | 0.136 | 0.441 | 1.186 | 0.137 | 2.72 |
| gbm_5bps | no_trade_band (width 0.02, to edge) | 0.112 | 0.457 | 1.151 | 0.116 | 2.29 |
| gbm_10bps | no_hedge | 0.014 | 4.176 | 12.178 | 0.000 | 0.00 |
| gbm_10bps | bs_delta | 0.273 | 0.451 | 1.368 | 0.274 | 2.72 |
| gbm_10bps | no_trade_band (width 0.03, to edge) | 0.212 | 0.485 | 1.305 | 0.217 | 2.14 |
| gbm_20bps | no_hedge | 0.014 | 4.176 | 12.178 | 0.000 | 0.00 |
| gbm_20bps | bs_delta | 0.547 | 0.482 | 1.742 | 0.549 | 2.72 |
| gbm_20bps | no_trade_band (width 0.04, to edge) | 0.403 | 0.529 | 1.592 | 0.408 | 2.02 |
| heston_0bps | no_hedge | -0.026 | 3.696 | 9.984 | 0.000 | 0.00 |
| heston_0bps | bs_delta | -0.010 | 0.539 | 1.243 | 0.000 | 2.73 |
| heston_0bps | no_trade_band (width 0.005, to edge) | -0.011 | 0.540 | 1.239 | 0.000 | 2.61 |
| heston_5bps | no_hedge | -0.026 | 3.696 | 9.984 | 0.000 | 0.00 |
| heston_5bps | bs_delta | 0.128 | 0.537 | 1.397 | 0.138 | 2.73 |
| heston_5bps | no_trade_band (width 0.01, to edge) | 0.115 | 0.541 | 1.380 | 0.126 | 2.50 |
| heston_10bps | no_hedge | -0.026 | 3.696 | 9.984 | 0.000 | 0.00 |
| heston_10bps | bs_delta | 0.266 | 0.539 | 1.557 | 0.276 | 2.73 |
| heston_10bps | no_trade_band (width 0.02, to edge) | 0.221 | 0.552 | 1.514 | 0.234 | 2.31 |
| heston_20bps | no_hedge | -0.026 | 3.696 | 9.984 | 0.000 | 0.00 |
| heston_20bps | bs_delta | 0.542 | 0.551 | 1.893 | 0.552 | 2.73 |
| heston_20bps | no_trade_band (width 0.04, to edge) | 0.397 | 0.591 | 1.769 | 0.413 | 2.04 |

Reading it:

- Hedging works: delta hedging cuts the standard deviation of cost from about 4.2 to 0.43 on
  GBM. On Heston the Black-Scholes delta leaves more risk (0.54), which is the room question 2
  is about.
- Costs move the mean almost one for one: 10 bps adds 0.27 to the delta hedge's mean cost and
  lifts its CVaR 95% from 1.01 to 1.37 on GBM.
- Phase 1 gate: the tuned no-trade band beats plain delta on CVaR 95% at 10 bps on both
  markets (1.305 vs 1.368 on GBM, 1.514 vs 1.557 on Heston), with a lower mean cost.
- Caveat: the band is tuned on these same validation paths, so its numbers are slightly
  flattering. The small edge it shows at 0 bps is that selection effect, not a real gain.

## What didn't work

Nothing to report yet.
