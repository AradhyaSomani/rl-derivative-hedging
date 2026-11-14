# What this project is and how it works

A plain-language walkthrough of everything in this repository, written to be read top to
bottom. The exact numbers are in [results.md](results.md), the maths in [theory.md](theory.md)
and the plan in [preregistration.md](preregistration.md).

## 1. The problem in one paragraph

A bank sells a customer a call option: the right to buy a stock at 100 in 30 days. If the
stock ends at 110 the bank owes 10; if it ends below 100 the bank owes nothing. The bank
received about 2.75 for this, and does not want to gamble on the outcome. So every day it
holds some shares of the stock to offset the option: this is **hedging**. The textbook says
exactly how many shares to hold (the option's **delta**), but the textbook assumes trading is
free and that we know how the market behaves. Neither is true. The question of this project
is whether a hedger that **learns** from simulated experience does better than the textbook
once trading costs money and the market model is wrong.

## 2. Hedging basics

- **Delta.** How much the option's value moves when the stock moves by 1. An at-the-money
  call has a delta of about 0.5, so the seller holds about half a share. As the stock rises
  the delta rises towards 1, and as it falls the delta falls towards 0.
- **Why hedging loses money by design.** Following delta means buying shares after the stock
  went up and selling after it went down. That steady loss is what the premium pays for. If
  the hedge is done well, the premium received and the hedging losses roughly cancel on every
  path, whatever the stock does.
- **Trading costs.** Each trade costs a small fraction of its value. The project uses 0, 5,
  10 and 20 basis points (a basis point is 0.01%). Delta hedging trades every day, so it pays
  costs every day. Trading less saves costs but tracks the option less closely. That
  trade-off is the whole game.
- **Hedging cost.** For each path we compute the seller's final profit or loss and flip the
  sign, so a positive number is a loss. Every strategy is judged on the distribution of this
  number over 10,000 paths.
- **Mean and tail.** The **mean** cost says how expensive a hedge is on average. The **tail**
  says how bad the bad days are. We measure the tail with **CVaR 95%**: the average of the
  worst 5% of outcomes. A hedge exists to control the tail, so CVaR is the headline number.

## 3. The simulated markets

We need many price paths, so we generate them from two models (`src/hedge/market/`).

- **GBM** (geometric Brownian motion): the Black-Scholes world. Volatility is a constant 20%.
  Here the textbook delta has the right model.
- **Heston**: volatility itself moves randomly and tends to rise when the stock falls, as in
  real equity markets. Here the textbook delta, which assumes constant volatility, has the
  wrong model.

Two markets times four cost levels gives the **eight scenarios** in `config/scenarios/`.

Each path is generated from its own seed number, so "path 12,345" is the same path forever, on
any machine. The seeds are split into fixed ranges:

| Range | Used for |
| --- | --- |
| 100,000 to 199,999 | Training the agents |
| 20,000 to 29,999 | Validation: every choice made while building |
| 10,000 to 19,999 | Test: scored once at the very end |

## 4. The accounting: one P&L engine

`src/hedge/pnl.py` is the single place where profit and loss is computed:

    P&L = premium - payoff + gains on the shares held - trading costs

Every strategy, simple or learned, is scored by this one function on the same paths. It is
tested against an independent cash-account calculation, so a result can never come from one
strategy being measured differently from another.

## 5. The baselines

Simple strategies to beat (`src/hedge/baselines/`):

- **No hedge.** Hold nothing. Cheap on average, terrible in the tail (CVaR about 12.7).
- **BS delta.** The textbook: rebalance to delta every day.
- **No-trade band.** Only trade when the holding has drifted far enough from delta. This is
  the honest competitor: it is simple, it knows about costs, and practitioners use it. Its
  width is tuned on validation paths for each scenario.

Any learned hedger that cannot beat the band has not earned its complexity.

## 6. The environment: turning hedging into a game

Reinforcement learning needs a game to play. `src/hedge/env/hedging_env.py` is that game, in
the standard gymnasium format:

- **One episode** is one option's life: 30 days on one simulated path.
- **What the agent sees** each day: time left, how far the stock is from the strike, its
  current holding, the Black-Scholes delta, and on Heston the current variance.
- **What the agent does:** it outputs a correction of at most 0.25 shares around the
  Black-Scholes delta. This was a design choice made on validation: letting the agent pick
  the holding from scratch worked worse, because it first had to rediscover delta.
- **The reward** it receives each day. Three were tried (`src/hedge/env/rewards.py`):
  - *Quadratic:* penalise the squared daily P&L and the trading cost. This one worked.
  - *Mean-variance:* reward P&L, penalise its square. Much worse.
  - *Terminal CVaR:* nothing until expiry, then a tail penalty. Worst, because one signal
    per 30 steps is too little to learn from.

The environment does no accounting of its own. A scripted agent that just returns delta
reproduces the delta baseline exactly, which is how we know the game is honest.

## 7. The learned hedgers

- **PPO** (proximal policy optimisation). The standard reinforcement-learning workhorse. It
  plays episodes, sees which actions led to better rewards, and nudges its policy network
  that way. Trained for 1,000,000 days of simulated trading per run.
- **SAC** (soft actor-critic). A different RL algorithm that reuses old experience from a
  memory buffer, so it needs about ten times fewer simulated days, but it does far more
  computation per day. On a laptop it ends up no faster than PPO.
- **Deep hedging.** Not RL. Because we wrote the simulator and the P&L formula, we can
  compute the tail risk of 5,000 whole episodes at once and push the gradient straight back
  through the P&L into the network (`src/hedge/agents/deep_hedging.py`). The network is
  trained directly on CVaR 95%, the thing we actually care about. It needs a market it can
  differentiate through, which RL does not.

All three use the same small network (two layers of 64 units), the same inputs and the same
action. Each was trained 5 times with different random seeds, because one training run can be
lucky.

## 8. How the comparison was kept honest

This is the part that separates a result from a story.

1. **Plan first.** The questions and the measures were written in `preregistration.md`.
2. **Validation for every choice.** Rewards, action space, hyperparameters, band width: all
   picked on validation paths.
3. **Freeze.** The chosen setup was committed and tagged `agents-frozen` in git.
4. **One test run.** Only then were the test paths generated, by a command that refuses to
   run without that tag. They were scored once, on 2026-11-14, and nothing was changed after.
5. **Risk, seeds and intervals.** Every learned strategy is reported as the median of 5
   seeds. The comparison with delta comes with a 95% bootstrap interval, so we can say
   whether a gap is real or noise.
6. **What failed is written down.** See "What didn't work" in `results.md`.

The test paid for itself. On validation PPO appeared to beat delta in one scenario; on test
that win was gone. Without a held-out set we would have reported it.

## 9. What we found

CVaR 95% at 10 basis points on the test paths (lower is better):

| Strategy | GBM | Heston |
| --- | --- | --- |
| BS delta | 1.369 | 1.552 |
| No-trade band | 1.322 | 1.503 |
| PPO | 1.396 | 1.629 |
| SAC | 1.408 | 1.602 |
| Deep hedging | 1.170 | 1.376 |

- **Deep hedging wins everywhere.** Its tail risk is 10% to 18% below delta's in all eight
  scenarios, at the same or lower average cost, and below the band's too.
- **RL did not beat the textbook on risk.** PPO and SAC are 7% to 12% cheaper on average at
  10 basis points because they learned to trade less, but their tail is no better than
  delta's anywhere, and the simple band beats them everywhere.
- **RL did not match deep hedging:** 18% to 22% worse on tail risk.
- **Stress tests.** With volatility at 30% instead of 20%, with Heston paths for agents
  trained on GBM, and with sudden jumps added, deep hedging still has the lowest tail risk.
  Jumps hurt every hedger about equally: nobody can hedge a jump by trading once a day.

### Honest caveats

- **It is not an equal fight.** Deep hedging saw about 150 times more simulated data and
  optimises the tail directly. PPO and SAC optimise a daily reward that only stands in for
  it. The fair reading is "if you can differentiate through your simulator, do that", not
  "RL is bad".
- **Deep hedging lowers the tail but raises the standard deviation.** It reshapes the
  distribution, it does not shrink it. Judged on standard deviation, delta would look better.
- **Everything is simulated.** One option, one maturity, daily trading, two models. A real
  market has no known model to train on, which is exactly where deep hedging's requirement
  is hardest to meet.
- **The interest rate is zero** (section 8, item 3 of the preregistration), and the drift is
  zero with it. A positive rate would add financing of the stock position and discounting of
  the payoff, and was not tested.
- **The RL tuning was modest:** one search of 30 trials, on GBM only. A bigger effort might
  close some of the gap.
- **The agents start from delta** and can move at most 0.25 shares from it, so this measures
  how much a learner can improve on delta, not whether it can find hedging from nothing.

## 10. How the repository fits together

| Folder | What is in it |
| --- | --- |
| `config/` | Every scenario, agent and experiment as a small text file. Nothing is hard-coded |
| `src/hedge/market/` | The two simulators, Black-Scholes and Heston pricing |
| `src/hedge/pnl.py` | The one P&L engine |
| `src/hedge/baselines/` | No hedge, delta, no-trade band |
| `src/hedge/env/` | The gymnasium environment and the rewards |
| `src/hedge/agents/` | Training for PPO, SAC and deep hedging, experiments, tuning |
| `src/hedge/eval/` | Metrics, the final evaluation, stress tests, figures |
| `src/hedge/db/`, `api/` | A PostgreSQL store for all results and a FastAPI service over it |
| `app/` | The Streamlit demo and the small results sample it ships with |
| `tests/` | 121 tests, run automatically on every push |
| `docs/` | Plan, theory, results, this report, figures |

The flow is: `make paths`, `make baselines`, `make train`, `make eval`, `make stress`,
`make report`, and once only `make final-test`. The demo (`make app`) has five pages: a race
between two hedgers on one path, the cost distributions, cost against tail risk, a page where
you change the cost and volatility and get fresh paths, and a summary.

## 11. What to say in an interview

**Thirty seconds.** "I tested whether learned hedgers beat Black-Scholes delta hedging for a
short call once there are transaction costs and model error. I built the simulators, a single
tested P&L engine and a gymnasium environment, trained PPO, SAC and a deep hedging network,
and evaluated them on held-out paths that were opened once after freezing the setup. Deep
hedging cut CVaR 95% by 10 to 18 percent in all eight scenarios. PPO and SAC made hedging
cheaper but not safer, and lost to a simple no-trade band."

**Questions to be ready for.**

- *Why CVaR and not variance?* A hedge exists to limit bad outcomes. Variance punishes good
  and bad surprises equally. And the answer changes with the measure: deep hedging has a
  lower CVaR but a higher standard deviation than delta.
- *Why did RL lose?* Its reward is per day, while the goal is a property of the whole
  distribution. The reward that targeted the tail directly gave too sparse a signal to learn
  from. Deep hedging optimises the goal itself and gets exact gradients.
- *Then why would anyone use RL?* When you cannot differentiate through the market: real
  market impact, an exchange simulator, anything that is a black box.
- *Why does the learned hedge beat delta even with zero costs?* With daily rather than
  continuous trading, delta is not optimal for a tail measure. The gain comes with a higher
  standard deviation.
- *How do you know you did not overfit?* Separate seeds for training, validation and test;
  the test set was created after a git tag and scored once; and one validation result did
  fail on test, which I report.
- *What is the no-trade band and why include it?* A cost-aware rule that only trades when
  far from delta. Beating delta under costs is easy; beating the band is the real bar.
- *What would you do next?* A CVaR-aware RL method to make the comparison fairer, more
  hedging instruments, real data for the stress tests, and a longer hyperparameter search on
  Heston.

**What not to claim.** Not that RL cannot hedge, not that this would work on a real desk as
is, and not that the learned hedger is better in every sense: it is better on the tail.
