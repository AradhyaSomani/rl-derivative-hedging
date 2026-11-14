# Reading notes

Study notes for every row of the Learning Plan in the task list, each ending with the
"done when" check answered for this project. The point is not to replace the readings but to
say what to look for in each one and where the idea shows up in the code.

**What these notes are based on.** They were written from general knowledge of these
standard texts and from this repository's code; the sources were not reread while writing
them. Chapter numbers in Hull change between editions, so chapters are named by topic. Every
number quoted from the project was produced by running the project's own code.

Theory already worked through in [`theory.md`](theory.md) is linked rather than repeated.

## 1. Options, Black-Scholes, the Greeks (Phase 0)

**Read:** Hull, *Options, Futures, and Other Derivatives*: the chapter on the Black-Scholes-Merton
model and the chapter on the Greek letters.

**What to take from it.**

- The Black-Scholes assumptions: constant volatility, continuous trading, no costs, a
  constant interest rate. The project breaks the first (Heston) and the third (costs) on
  purpose, and approximates the second by trading daily.
- The pricing argument is replication: a portfolio of shares and cash that matches the option
  over each instant can be held without risk, so it must earn the risk-free rate. That gives
  the PDE; the formula is its solution.
- The Greeks and the hedging identity with zero rate, $`\Theta = -\tfrac12\sigma^2S^2\Gamma`$.
  Delta is the hedge ratio; gamma is why a discretely rebalanced hedge still leaks money.
- Hull's tables of a delta hedge rebalanced weekly: the cost of hedging comes out close to the
  Black-Scholes price, and its spread shrinks as rebalancing gets more frequent. That is the
  same experiment as the project's "1,000 steps, no costs, standard deviation close to 0" test.

**Done when you can price a call and its delta by hand.** Done in
[`theory.md` section 1](theory.md): $`d_1 = 0.03450`$, $`\Delta = 0.51376`$, price 2.7524 for the
project's at-the-money 30-day call, matching `hedge.market.black_scholes`.

## 2. Delta hedging P&L (Phase 0)

**Read:** Hull, the chapter on the Greek letters, the sections on delta hedging and on the
cost of hedging; then your own spreadsheet.

**What to take from it.**

- The seller's P&L is premium, minus payoff, plus the gains on the shares, minus costs.
  Everything else is bookkeeping, and most hedging-project bugs are in that bookkeeping.
- The hedge buys after the stock rises and sells after it falls. That "buy high, sell low"
  loss is not a mistake: it is what the premium pays for.
- Costs scale with how much is traded, and a delta hedge trades the most when gamma is large:
  near the strike, close to expiry.

**Done when you can hedge one path in Excel and explain every cash flow.** The workbook
[`spreadsheet_hedge.xlsx`](spreadsheet_hedge.xlsx) hedges path seed 0 at 10 bps with live
formulas, explains every row in words, and reproduces the Python walkthrough's P&L of +0.1170
to 12 decimal places. Its "Read me" sheet walks through each cash flow and lists things to
change (cost rate, the shocks) and what should happen.

## 3. Monte Carlo simulation (Phase 1)

**Read:** numpy's `random.Generator` documentation; any Monte Carlo chapter (Glasserman,
*Monte Carlo Methods in Financial Engineering*, chapter 1 is the classic).

**What to take from it.**

- Monte Carlo error falls like $`1/\sqrt{n}`$: four times the paths for half the error. With
  10,000 paths a CVaR 95% is an average of only 500 numbers, which is why the project puts
  bootstrap intervals on it.
- `np.random.default_rng(seed)` gives an independent, reproducible stream per seed. The legacy
  global `np.random.seed` is shared state and easy to disturb.
- GBM can be simulated exactly in log space, so the step size adds no bias
  (`hedge.market.gbm`). Heston cannot; the project uses full truncation for the variance.

**Done when you can explain why fixed seeds matter.** Three reasons, all used in this project:

1. *Reproducibility.* Path $`i`$ of a split is always built from seed `start + i`
   (`hedge.seeds`), so anyone rerunning the code gets the same paths and the same table.
2. *A clean test set.* Seeds 10,000 to 19,999 are reserved for the final test and the code
   refuses to touch them unless asked (`guard_test_seeds`). Without fixed ranges you could not
   prove the test paths were never used in tuning.
3. *Common random numbers.* Every strategy and every cost level is scored on the same shocks,
   so the difference between two strategies is not drowned in path-to-path noise. That is
   why the bootstrap in `hedge.eval.final.bootstrap_cvar_gap` is paired.

## 4. Heston model (Phase 1)

**Read:** S. Heston, "A Closed-Form Solution for Options with Stochastic Volatility with
Applications to Bond and Currency Options", *Review of Financial Studies* 6(2), 1993,
introduction and model sections.

**What to take from it.**

- Why stochastic volatility: Black-Scholes cannot produce the skew seen in prices, and a
  correlation between price and volatility shocks can.
- The variance follows a mean-reverting square-root process, which keeps it positive under
  the Feller condition and keeps the model tractable.
- The closed form comes from the characteristic function; the price is an integral that is
  computed numerically (the project's `hedge.market.heston.heston_price`).

**Done when you can say what each of the five parameters does.** Done in
[`theory.md` section 3](theory.md), with a table of implied volatilities computed by moving
one parameter at a time.

## 5. Gymnasium environments (Phase 2)

**Read:** Gymnasium docs, "Make your own custom environment", and the `Env` API page.

**What to take from it.**

- An environment defines `observation_space`, `action_space`, `reset(seed=...)` returning
  `(obs, info)`, and `step(action)` returning `(obs, reward, terminated, truncated, info)`.
- `terminated` means the task ended for a reason inside the problem (the option expired);
  `truncated` means it was cut off from outside (a time limit). The difference matters for
  value bootstrapping.
- Seed the environment's own `self.np_random` through `super().reset(seed=seed)`.

**Done when it passes `check_env`.** `tests/test_env.py::test_gymnasium_check_env` runs
`gymnasium.utils.env_checker.check_env` on the hedging environment for every reward type and
action mode, and is part of the 121-test suite.

## 6. RL basics: MDPs, value functions, policy gradients (Phase 3)

**Read:** Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., free online),
chapter 3 (finite MDPs), chapter 6 (temporal-difference learning), chapter 13 (policy gradient
methods).

**What to take from it.**

- *Chapter 3.* State, action, reward, return, discounting; the value function $`v_\pi(s)`$ and
  action value $`q_\pi(s,a)`$; the Bellman equations. Hedging fits as an MDP if the state
  includes everything the best trade depends on: spot, time left, current holding (because of
  costs) and, under Heston, the variance.
- *Chapter 6.* TD learning updates a value estimate from the next estimate instead of waiting
  for the episode's end. Sarsa is on-policy, Q-learning is off-policy. SAC's critics are a
  descendant of this.
- *Chapter 13.* Policy gradient: adjust the policy's parameters in the direction that makes
  high-return actions more likely (REINFORCE), with a baseline or critic to cut variance
  (actor-critic). PPO is an actor-critic policy-gradient method that limits how far one update
  moves the policy.

**Done when you can explain on-policy versus off-policy.** An *on-policy* method learns only
from data generated by the policy it is currently improving, so after each update the old
data is thrown away (PPO collects a rollout, does a few epochs on it, discards it). An
*off-policy* method can learn from data generated by any policy, including older versions of
itself, so it keeps a replay buffer and reuses every sample many times (SAC, Q-learning,
DQN). Off-policy is usually far more sample-efficient; on-policy is usually more stable and
easier to tune. The Pendulum exercise shows the gap directly: SAC first scored better than -200
after 4,000 environment steps, tuned PPO after 30,000, and PPO with default settings not at
all in 200,000 (see [`pendulum.md`](pendulum.md)).

## 7. PPO and SAC in practice (Phase 3)

**Read:** stable-baselines3 docs: "Reinforcement Learning Tips and Tricks", the PPO and SAC
pages, and the TensorBoard integration page. rl-baselines3-zoo for tuned hyperparameters.

**What to take from it.**

- Normalise observations and rescale actions to $`[-1, 1]`$; the hedging env does both.
- Evaluate on a separate environment with deterministic actions, and report several seeds:
  RL results vary a lot from seed to seed. The project trains 5 seeds per scenario.
- Default hyperparameters are a starting point, not an answer. PPO on Pendulum is the standard
  example: with SB3's defaults it learns little, with the zoo's tuned set it solves the task.
- Shape the reward carefully; a sparse terminal reward is much harder to learn from than a
  per-step one. That is the trade-off behind the project's three reward types.

**Done when you can train PPO on Pendulum and read its TensorBoard curves.** The script and
the curves are in [`pendulum.md`](pendulum.md) and `notebooks/pendulum_warmup.py`, with a guide to the curves worth watching.

## 8. Risk measures: VaR, CVaR, entropic risk (Phase 3)

**Read:** any risk management text (Hull's *Risk Management and Financial Institutions*,
chapters on VaR and expected shortfall), and section 3 of the deep hedging paper on convex
risk measures.

**What to take from it.**

- VaR at 95% is the 95th percentile of the loss: "losses are below this 95% of the time". It
  says nothing about how bad the other 5% are and is not subadditive.
- CVaR 95% (expected shortfall) is the average of the worst 5% of losses. It is a coherent,
  convex risk measure, which is what lets deep hedging optimise it by gradient descent.
- Entropic risk, $`\frac1a\log E[e^{a\,\text{loss}}]`$, is exponential utility in disguise; $`a`$
  sets the risk aversion. The project implements it (`entropic_loss`) but uses CVaR.

**Done when you can compute CVaR from a sample by hand.** Here are the hedging costs (minus the
P&L) of the BS delta hedge at 10 bps on the 20 development paths with seeds 0 to 19, sorted:

    -0.2023 -0.1725 -0.1170 -0.1045 -0.0377  0.0703  0.1008  0.1199  0.1287  0.2263
     0.2859  0.3320  0.5654  0.5867  0.6274  0.6384  0.6965  0.7485  0.7904  0.8747

(Seed 0 is the path in the spreadsheet: cost -0.1170, which is P&L +0.1170.)

- Mean cost: 0.3079. Most of it is trading costs; without costs the mean would be near 0.
- CVaR 80%: the worst 20% of 20 is 4 paths, so (0.8747 + 0.7904 + 0.7485 + 0.6965) / 4 =
  **0.7775**.
- CVaR 95%: the worst 5% of 20 is one path, so **0.8747**. With so few paths the tail is a
  single number, which is why the project scores on 10,000 paths (500 in the tail).
- `hedge.eval.metrics.cvar` uses exactly this rule (average the $`\lceil n(1-\alpha)\rceil`$
  largest) and gives the same numbers.

## 9. Deep hedging (Phase 4)

**Read:** Buehler, Gonon, Teichmann and Wood, "Deep Hedging", *Quantitative Finance* 19(8),
2019. Setup and results are summarised in [`theory.md` section 4](theory.md).

**Done when you can write the loss for one batch in PyTorch.** For a batch of paths
`paths` (shape `[B, N+1]`) and the network's holdings `h` (shape `[B, N]`):

```python
payoff = torch.relu(paths[:, -1] - K)
gain = (h * (paths[:, 1:] - paths[:, :-1])).sum(dim=1)
flat = torch.zeros_like(h[:, :1])
trades = torch.diff(torch.cat([flat, h, flat], dim=1), dim=1)   # includes buy at 0, sell at N
costs = kappa * (paths * trades.abs()).sum(dim=1)
cost = -(premium - payoff + gain - costs)                         # hedging cost per path
k = math.ceil(0.05 * B)
loss = torch.topk(cost, k).values.mean()                         # CVaR 95% of the batch
loss.backward()
```

This is `torch_hedge_pnl` plus `cvar_loss` in `hedge.agents.deep_hedging`. The key point:
the whole P&L is differentiable in the holdings, so the gradient flows from the tail of the
batch straight back into the network, with no reward signal or value function needed. The
`topk` means only the worst 5% of paths in each batch get a gradient.

## 10. Bootstrap confidence intervals (Phase 4)

**Read:** scipy's `scipy.stats.bootstrap` documentation; any introduction to the bootstrap
(Efron and Tibshirani's book is the original).

**What to take from it.**

- Resample the paths with replacement, recompute the statistic, repeat; the spread of the
  recomputed values estimates the statistic's sampling error.
- For a difference between two strategies on the *same* paths, resample path indices once and
  use them for both (a paired bootstrap). Unpaired resampling throws away the common random
  numbers and gives intervals that are far too wide.
- Percentile intervals on a tail statistic like CVaR are rough with small samples; 10,000
  paths with 500 in the tail is enough for the gaps the project reports.

**Done when you can put a CI on a CVaR difference.** `hedge.eval.final.bootstrap_cvar_gap`
does it: paired resamples of path indices, CVaR 95% of each strategy on each resample, the
2.5% and 97.5% quantiles of the gap. `make eval` writes the intervals into the results tables.

## 11. FastAPI and Streamlit (Phase 5)

**Read:** the official FastAPI tutorial (first steps, path and query parameters, response
models) and Streamlit's "Get started".

**Done when you can ship a page that reads from the API.** `src/hedge/api/main.py` and
`app/streamlit_app.py` are that page; hosting it is still on Aradhya's list.
