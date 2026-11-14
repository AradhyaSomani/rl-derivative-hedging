# Pre-registration

**Status: tagged `prereg-v1` and `agents-frozen` on 2026-11-14**, both on the commit that
holds the validation results, and before the test paths were generated. The tag therefore
shows that the plan and the setup were fixed before the test run, not before the validation
experiments; section 8, item 22 says which items were written after validation results were
known. Changes since the tag are in section 9.

## 1. Questions

1. With proportional transaction costs, does an RL agent achieve lower tail loss (CVaR 95%)
   than Black-Scholes delta hedging and a no-trade-band hedge, at the same or lower average
   cost?
2. When prices follow Heston but the baseline assumes Black-Scholes, how much of the RL
   agent's gain comes from learning around the wrong model?
3. Does model-free RL (PPO, SAC) match deep hedging, which optimises the hedging loss directly
   through the simulator?

## 2. Setting

The hedger has sold one European call and trades the underlying once a day until expiry.

| Item | Value |
| --- | --- |
| Option | Short European call, at the money (K = S0 = 100), 30 trading days to expiry |
| Rebalancing | Daily: 30 steps, dt = 1/252 year |
| Holding | Between 0 and 1 share per option |
| Costs | Proportional: cost rate times traded value, on every trade including opening and closing |
| Interest rate | 0 |
| Drift | 0 (equal to the interest rate), so no strategy can earn a risk premium by under-hedging |
| Premium | The option's fair price in the scenario's own model: Black-Scholes on GBM, semi-analytic Heston on Heston |

Hedging P&L per path, with holdings h, h before time 0 and at expiry both 0:

    P&L = premium - max(S_T - K, 0) + sum_{t<T} h_t (S_{t+1} - S_t) - sum_{t<=T} rate * S_t * |h_t - h_{t-1}|

Hedging cost is minus the P&L. All strategies are scored by the one function
`hedge.pnl.hedge_pnl`.

## 3. Scenarios

Two markets times four cost levels, eight scenarios, each a file in `config/scenarios/`.

| Market | Parameters |
| --- | --- |
| GBM | volatility 20% |
| Heston | v0 = 0.04, kappa = 2.0, theta = 0.04, xi = 0.3, rho = -0.7; Euler with full truncation, 8 sub-steps per day |

Cost levels: 0, 5, 10 and 20 bps of traded value.

## 4. Seeds and splits

One seed makes one path (`hedge.seeds`). The same seeds are used in every scenario.

| Split | Seeds | Paths | Used for |
| --- | --- | --- | --- |
| Test | 10,000 to 19,999 | 10,000 | Scored once, after the `agents-frozen` tag |
| Validation | 20,000 to 29,999 | 10,000 | Reward choice, tuning, early stopping, band width |
| Train | 100,000 to 199,999 | 100,000 | Training only |
| Dev | 0 to 9,999 | | Unit tests, notebooks, the demo |

The simulators refuse test seeds unless explicitly allowed, and a unit test fails if the
ranges overlap. RL and deep hedging training seeds: 0, 1, 2, 3, 4.

## 5. Strategies

- No hedge.
- Black-Scholes delta, rebalanced every step, using the true volatility on GBM and the square
  root of the current variance on Heston (the misspecified hedge).
- No-trade band around that delta. Width and whether to trade to delta or to the band edge
  are chosen per scenario for lowest validation CVaR 95%, from the grid in
  `config/agents/no_trade_band.yaml`.
- PPO and SAC (TD3 optional), two hidden layers of 64, rewards `quadratic`, `mean_variance`
  and `terminal_cvar` with lambda in {0.1, 0.5, 1, 2}, chosen on validation paths.
- Deep hedging network trained on CVaR 95% of the P&L, early-stopped on validation CVaR.

## 6. Metrics and what counts as support

Computed on hedging cost over the 10,000 test paths of each scenario.

| Metric | Definition |
| --- | --- |
| Mean cost | Average hedging cost |
| Std of cost | Sample standard deviation |
| CVaR 95% | Mean of the worst 5% of costs (the worst 500 of 10,000) |
| Transaction cost | Mean total cost paid per path |
| Turnover | Mean shares traded per path, including opening and closing |
| Seed spread | Range of CVaR across the 5 training seeds |

Headline metrics are CVaR 95% and standard deviation; mean cost is always reported beside
them. Every RL number is the median of 5 training seeds with the range.

| Claim | Test | Counts as support |
| --- | --- | --- |
| RL lowers tail risk under costs | CVaR 95%, RL vs BS delta, GBM at 10 bps | Bootstrap 95% CI of the difference excludes 0, median over 5 seeds |
| RL is not just a smarter band | RL vs tuned no-trade band, same scenario | CVaR lower and mean cost no higher |
| Gains grow when the model is wrong | RL gain over delta on Heston vs on GBM | Larger gain on Heston, CIs not overlapping |
| RL matches deep hedging | CVaR of PPO/SAC vs deep hedging | Within 5% of deep hedging's CVaR |
| The agent generalises | Agent trained at 10 bps, tested at 5 and 20 bps | Still beats delta at both |

Null results and failed rewards or settings are reported in `docs/results.md`.

## 7. Stress tests (after the main test run)

Train at 20% volatility and test at 10% and 30%; train on GBM and test on Heston; rare jumps
added to test paths only.

## 8. Defaults chosen while building, for Aradhya to confirm before tagging

1. **Heston base parameters** (section 3). A common textbook-style set that satisfies the
   Feller condition; it is not taken from a specific paper. The deep hedging paper's own set
   could be used instead.
2. **Second contract for the check: a put**, same strike and maturity. The engine and
   baselines already support it (`option.kind: put`).
3. **Interest rate 0 in v1.** Scenario configs with a non-zero rate are rejected.
4. **Drift 0.** Keeps mean hedging cost equal to transaction cost, so mean and risk are easy
   to read. A positive drift is a possible later stress test.
5. **Trading-day clock**: 30 trading days, 252 per year.
6. **Train, validation and dev seed ranges.** Only the test range came from the task list.
7. **Baselines before the freeze are scored on validation paths**, not test paths, so the
   test set really is opened once. The task list's Phase 1 says to run them on test paths.
8. **Path files are keyed by a hash of the market, maturity and time grid** rather than the
   whole scenario, so the four cost levels share one file.
9. **The band may trade to its edge** as well as to delta; the choice is tuned with the width.

10. **Action range.** The agent's action is in [-1, 1] and is mapped linearly onto a holding
    of 0 to 1 share, because stable-baselines3 works best with symmetric actions. The task
    list says `Box(0, 1)`; the two are equivalent.
11. **Step P&L uses a Black-Scholes mark of the option** (true volatility on GBM, current
    variance on Heston). The marks cancel over an episode, so the total always equals the P&L
    engine's; they only decide how the total is spread across steps for the rewards.
12. **Reward definitions** (`hedge.env.rewards`), with gross = step P&L before costs:
    - `quadratic`: -gross^2 - lambda * cost. The squared term is before costs so that costs
      are not counted twice.
    - `mean_variance`: pnl - (lambda / 2) * pnl^2, with pnl after costs.
    - `terminal_cvar`: 0 until expiry, then -L - lambda * (w + max(L - w, 0) / 0.05), where L
      is the episode's hedging cost and w is a fixed threshold. A single episode has no CVaR,
      so this uses the Rockafellar-Uryasev form, whose average is an upper bound on CVaR 95%
      that is tight when w equals the VaR.

13. **Agent action.** The frozen agents output a correction of at most 0.25 shares around
    the BS delta (`action_mode: delta_offset`) instead of the holding itself. Chosen on
    validation CVaR (GBM 10 bps); both versions are reported in `docs/results.md`.
14. **Frozen line-up** (`config/eval.yaml`): PPO with the tuned hyperparameters and the
    quadratic reward at lambda 1, on all 8 scenarios; SAC with default hyperparameters on the
    two 10 bps scenarios only; deep hedging on all 8. Five training seeds each (0 to 4).
15. **Budgets.** PPO 1,000,000 steps. SAC 100,000 steps, because one SAC step costs about ten
    PPO steps on a CPU. Deep hedging at most 1,000 gradient steps of 5,000 episodes with
    early stopping on validation CVaR (patience 6 checks, one every 50 steps).
16. **Hyperparameter search.** 30 Optuna trials for PPO on GBM 10 bps, one seed, scored on
    validation CVaR, over learning rate, entropy coefficient, discount and a constant or
    linear learning-rate schedule. No search for SAC, for deep hedging or on Heston.
17. **Deep hedging loss**: CVaR 95% of the hedging cost over the batch. Same observations,
    network size and action as the RL agents.
18. **How a learned strategy is reported.** Each metric is the median over the 5 seeds, with
    the minimum and maximum CVaR. The confidence interval for the CVaR gap to BS delta is a
    paired bootstrap (2,000 resamples of the paths, seed 0) for the seed with the median
    validation CVaR. The median seed is picked on validation, also for the test run.
19. **"Matches deep hedging"** (question 3) means a CVaR 95% within 5% of deep hedging's.
20. **Stress tests** (`config/eval.yaml`): volatility 30% and 10% on GBM, Heston paths for
    GBM-trained agents, and jumps on GBM and Heston (intensity 2 per year, log-jump mean
    -0.05, standard deviation 0.08, not compensated, so the jumps also pull the price down
    a little on average). All at 10 bps, with the option premium kept at its unstressed value.
21. **Not done:** TD3, DDPG, recurrent policies, a put as second contract and a non-zero
    rate were planned or possible and are cut; see "What didn't work" in `docs/results.md`.
22. **The final test run has not been made.** It is left for Aradhya, after reviewing this
    section and tagging `agents-frozen`. Items 13 to 21 were written after the validation
    results were known, so they record what was done; they are not predictions.

## 9. Changes after the tag

- **2026-11-14, the test run.** `make final-test` was run once. Results are in
  `docs/results.md`. No strategy, config or setting was changed afterwards.
- **2026-11-14, correction to section 8, item 18.** Its last sentence says the seed whose
  confidence interval is reported is picked on validation, also for the test run. The code
  that ran does something slightly different: on each split it reports the seed whose CVaR is
  the median of the five on that split. The rule is the median, not the best, so it does not
  favour any strategy, and the medians over seeds in the tables do not depend on it. The run
  was not repeated.
- **2026-11-14, code change after the tag.** `python -m hedge.cli export-sample` now empties
  `app/sample` before writing, so the demo's sample holds only the test results. This does
  not touch any result.
- **2026-11-14, review of section 8.** Aradhya reviewed section 8 and confirmed items 1 to 12
  as written. It is not recorded whether this review happened before or after the
  `prereg-v1` and `agents-frozen` tags were made, so it should not be read as a check made
  before tagging.
- **2026-11-14, what items 13 to 21 are.** Items 13 to 21 were written after the validation
  results had been seen. They describe choices that were made with those results in view, so
  they are a record, not a prediction. The two most exposed to being fitted to the results
  are item 13 (the delta-offset action, chosen because it scored better on validation) and
  item 19 (the 5% threshold for "matches deep hedging", set when the validation gap was
  already known). The test paths had not been generated or scored when any of them was
  written.
- **2026-11-14, planned exploratory stress test.** After the final test run, the stress tests
  will be repeated with Heston paths drawn from the parameter set used in the deep hedging
  paper (Buehler et al., 2019) in place of the textbook-style set in section 3. This was
  decided after the test results were known, so it is post hoc and will be labelled
  exploratory wherever it is reported. It changes no agent, setting or headline number.
- **2026-11-14, second test run.** `make final-test` was run a second time, at commit
  `588acf4`, with only documentation changed since the first run (commit `2defcfe`). The
  saved result tables (`final_test.csv`, `final_test_table.csv`, `stress_test_table.csv`)
  are identical to the first run's in every number; only the recorded code commit differs.
  No strategy, config or setting was changed between or after the runs. The test paths have
  now been scored twice, not once, and the results are the same.
- **2026-11-14, exploratory stress test run.** The stress test planned above was added as a
  separate `exploratory_stress` section of `config/eval.yaml` and run with
  `python -m hedge.cli stress --split validation --exploratory`. It uses the parameters of
  Buehler et al. (2019), section 5.2 (alpha = 1, b = 0.04, sigma = 2, v0 = 0.04, rho = -0.7),
  that is kappa = 1 and xi = 2 here, and repeats the three Heston stresses: agents trained on
  Heston 10 bps, agents trained on GBM 10 bps, and Heston 10 bps with the preregistered jumps. It runs on validation seeds, so the test seeds were not used
  again. No agent was retrained, no setting changed, and the preregistered stress results are
  in separate files that it does not write to. Results are in `docs/results.md` under
  "Exploratory".
