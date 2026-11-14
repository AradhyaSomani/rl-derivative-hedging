# Reading notes

Study notes for the eleven rows of the Learning Plan in the task list. Each entry says what to
read, explains the key ideas in more depth than the reading list does, works one example with
numbers from this project, shows where the idea lives in the code and the results, and ends
with the row's "done when" check answered for this project.

Theory already worked through in [`theory.md`](theory.md) is linked rather than repeated.

## What these notes are based on

The first version of these notes was written from general knowledge, without rereading the
sources. For this version the sources that are free on the web were fetched and read on
2026-11-14, and the notes were checked against them. Books that are not freely available were
not reread, so what is said about them is from background knowledge and is marked
"(background)".

| Source | Status | Link |
|---|---|---|
| Buehler, Gonon, Teichmann, Wood, "Deep Hedging" (2019), sections 1 to 5 | Read (arXiv version) | [arXiv:1802.03042](https://arxiv.org/abs/1802.03042) |
| Heston, "A Closed-Form Solution for Options with Stochastic Volatility" (1993) | Read; the scanned copy's equations are partly illegible, so the formulas below follow the standard statement | [PDF (Imperial College copy)](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf) |
| stable-baselines3, "Reinforcement Learning Tips and Tricks" | Read | [SB3 docs](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) |
| stable-baselines3, PPO and SAC pages | Read | [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html), [SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) |
| OpenAI Spinning Up, PPO and SAC pages | Read | [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html), [SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html) |
| Gymnasium, "Create a custom environment" | Read | [Gymnasium docs](https://gymnasium.farama.org/introduction/create_custom_env/) |
| numpy, random `Generator` | Read | [numpy docs](https://numpy.org/doc/stable/reference/random/generator.html) |
| scipy, `scipy.stats.bootstrap` | Read | [scipy docs](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) |
| FastAPI tutorial, "Response Model" | Read | [FastAPI docs](https://fastapi.tiangolo.com/tutorial/response-model/) |
| Streamlit, "Caching overview" | Read | [Streamlit docs](https://docs.streamlit.io/develop/concepts/architecture/caching) |
| Künzi-Bay and Mayer (2006), which restates the Rockafellar and Uryasev CVaR formula | Read (the 2000 original is paywalled) | [PDF](https://core.ac.uk/download/159146995.pdf) |
| Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed. | Background (free online, chapters not reread) | [book page](http://incompleteideas.net/book/the-book-2nd.html) |
| Hull, *Options, Futures, and Other Derivatives*; Hull, *Risk Management and Financial Institutions* | Background (not freely available) | |
| Glasserman, *Monte Carlo Methods in Financial Engineering*; Efron and Tibshirani, *An Introduction to the Bootstrap* | Background (not freely available) | |

Every number quoted from this project was produced by running the project's own code. "Dev
seeds" means seeds 0 to 9,999, which are reserved for checks like these; no test path was used
for anything in these notes.

---

## 1. Options, Black-Scholes, the Greeks (Phase 0)

**Read:** Hull, the chapter on the Black-Scholes-Merton model and the chapter on the Greek
letters (background; chapter numbers change between editions, so they are named by topic).

**The key ideas.**

- **The assumptions.** Constant volatility, continuous trading, no costs, a constant interest
  rate, a lognormal price. This project breaks the first on purpose (Heston), breaks the
  third on purpose (costs of 5 to 20 bps), approximates the second by trading once a day,
  and keeps the rate at zero (a limitation listed in `report.md`).
- **Replication is the whole argument.** Hold $`\Delta`$ shares against a short option. Over
  an instant the shares cancel the option's exposure to the price, so the combination is
  riskless and must earn the risk-free rate. Writing that down gives the Black-Scholes PDE;
  the formula is its solution with the payoff as the boundary condition. The price does not
  depend on the stock's expected return, which is why the project can set the drift to zero
  without changing any price.
- **The Greeks are sensitivities.** Delta $`\partial V/\partial S`$ is the hedge ratio. Gamma
  $`\partial^2 V/\partial S^2`$ measures how fast the hedge ratio goes stale, so it measures
  how much a hedge rebalanced once a day leaks. Theta is time decay. Vega is the sensitivity
  to volatility, the risk a stock-only hedge cannot remove (see the paper-Heston stress test
  in section 9).
- **Theta pays for gamma.** With a zero rate the PDE reads
  $`\Theta + \tfrac12\sigma^2S^2\Gamma = 0`$. A delta-hedged short option loses on big moves
  (gamma) and gains from time decay (theta); with the right volatility the two cancel on
  average.

**Worked example.** For the project's option (at the money, 30 trading days,
$`\sigma = 20\%`$, zero rate), [`theory.md` section 1](theory.md) gives $`d_1 = 0.03450`$,
$`\Delta = 0.51376`$ and price 2.7524. The same inputs give $`\Gamma = 0.05778`$. The identity
then says $`\Theta = -\tfrac12 (0.2)^2 (100)^2 (0.05778) = -11.556`$ per year, and a finite
difference of `black_scholes.price` in time gives -11.556. Per trading day that is about
-0.046: the option loses 4.6 cents of value a day while the price stands still, which the
seller keeps.

**How the hedging error behaves with the rebalancing frequency** (Hull's tables show the same
effect for a weekly hedge; background). BS delta hedge of the project's option, dev seeds 0
to 4,999, premium at the BS price:

| Rebalances in 30 days | Std of P&L, no costs | CVaR 95%, no costs | Mean cost at 10 bps | CVaR 95% at 10 bps |
|---|---|---|---|---|
| 30 (daily, the project) | 0.428 | 1.011 | 0.276 | 1.373 |
| 120 | 0.219 | 0.507 | 0.453 | 1.134 |
| 1,000 | 0.076 | 0.167 | 1.105 | 1.875 |

Without costs the error halves each time the number of steps is multiplied by four, the
$`1/\sqrt{n}`$ rate. With costs the trade-off appears: more trading means less hedging error
but more paid in costs, and at 1,000 steps the costs dominate. That tension is what the whole
project is about.

**In the code:** `hedge.market.black_scholes` (price, delta, gamma, vega; put-call parity is
tested in `tests/test_black_scholes.py`); `hedge.baselines.delta.bs_delta_holdings`; the
"1,000 steps, no costs" check is in `tests/test_pnl.py`.

**Done when you can price a call and its delta by hand.** Done in
[`theory.md` section 1](theory.md), matching `hedge.market.black_scholes` to the digits shown.

## 2. Delta hedging P&L (Phase 0)

**Read:** Hull, the sections on delta hedging and on the cost of hedging (background); then
your own spreadsheet.

**The key ideas.**

- **The seller's P&L is four terms:** premium received, minus the payoff at expiry, plus the
  gains on the shares held, minus trading costs. `hedge.pnl.hedge_pnl` returns exactly these
  (`premium`, `payoff`, `hedge_gain`, `txn_cost`) and `pnl` is their sum. Most hedging bugs
  are in this bookkeeping: an off-by-one between a holding and the price move it earns, or
  forgetting the opening and closing trades.
- **Timing.** The holding chosen at time $`t`$ (knowing $`S_t`$) earns $`h_t(S_{t+1} - S_t)`$.
  It never sees $`S_{t+1}`$. The holdings array has one column fewer than the prices array
  for this reason.
- **"Buy high, sell low" is the cost of convexity.** A short call's delta rises when the price
  rises, so the hedger buys after rises and sells after falls. Each round trip loses a little;
  over the life of the option those losses add up, on average, to the premium. That is what
  the premium pays for.
- **Costs are proportional to trading.** Cost per trade is rate x price x shares traded. A
  delta hedge trades most where gamma is large, near the strike and close to expiry, and it
  also pays to open the position (about half a share on day 0) and to close it at expiry.

**Worked example: the first two days of path seed 0** (GBM 10 bps, the spreadsheet's path).
Prices 100.0000, 100.1506, 99.9761. The project's environment spreads the P&L over the days by
marking the option at its BS value: $`M_0 = 2.75242`$, $`M_1 = 2.78417`$, $`M_2 = 2.64687`$.

- Day 0: buy $`\Delta_0 = 0.51376`$ shares at 100. Cost $`0.001 \times 100 \times 0.51376 =
  0.05138`$.
- Day 0 to 1: the shares gain $`0.51376 \times 0.1506 = 0.07737`$; the short option loses
  $`2.78417 - 2.75242 = 0.03175`$. Gross $`+0.04562`$; net of the cost, $`-0.00576`$.
- Day 1: the new delta is 0.52237, so buy 0.00861 shares; cost 0.00086.
- Day 1 to 2: the price falls 0.1745; the shares lose $`0.52237 \times 0.1745 = 0.09115`$
  and the option's mark falls 0.13730 (part price move, part one day of theta), so gross
  $`+0.04615`$ and net $`+0.04529`$.

These are the numbers `HedgingEnv.step` returns in `info["step_pnl"]` and `info["step_cost"]`
for this path, and the 30 daily amounts add up to the spreadsheet's total of +0.1170.

**In the code:** `hedge.pnl.hedge_pnl` and `hedge.pnl.cash_ledger`; `hedge.env.rewards` (the
step P&L with marks; `tests/test_env.py` checks that the steps sum to the engine's total).

**Done when you can hedge one path in Excel and explain every cash flow.** The workbook
[`spreadsheet_hedge.xlsx`](spreadsheet_hedge.xlsx) hedges path seed 0 at 10 bps with live
formulas, explains every row, and reproduces the Python walkthrough's P&L of +0.1170 to 12
decimal places.

## 3. Monte Carlo simulation (Phase 1)

**Read:** numpy's [random `Generator` page](https://numpy.org/doc/stable/reference/random/generator.html)
(read); Glasserman, *Monte Carlo Methods in Financial Engineering*, chapter 1 (background).

**The key ideas.**

- **The error falls like $`1/\sqrt{n}`$.** The standard error of a mean is
  $`\text{sd}/\sqrt{n}`$: four times the paths for half the error. Tail statistics are noisier
  than means because they use only part of the sample.
- **`default_rng` and `Generator`.** The numpy page calls `default_rng` "the recommended
  constructor for the random number class `Generator`". A `Generator` holds its own state, so
  two generators never disturb each other, unlike the legacy global `np.random.seed`. An
  integer seed is passed through `SeedSequence` to set up the bit generator, and
  `Generator.spawn` makes independent child streams. One caveat from the page: `Generator`
  has "no compatibility guarantee" across numpy versions, so "same seed, same numbers" is
  only promised within one numpy version.
- **Exact against discretised simulation.** GBM can be simulated exactly in log space, so the
  daily step adds no bias (`hedge.market.gbm`). Heston cannot: the project uses an Euler
  scheme with 8 sub-steps a day and full truncation of the variance (section 4).

**Worked example: how noisy is a CVaR?** BS delta at 10 bps, dev seeds:

| Paths | Mean cost | Its standard error | CVaR 95% | Its standard error (bootstrap) |
|---|---|---|---|---|
| 1,000 | 0.258 | 0.014 | 1.322 | 0.063 |
| 10,000 | 0.273 | 0.005 | 1.364 | 0.019 |

The CVaR's standard error is about four times the mean's at the same sample size, because
CVaR 95% averages only the worst 5% (50 or 500 paths). With 10,000 paths it is about 0.02,
which is why the project's headline gaps of 0.15 to 0.25 are clearly resolved and gaps under
about 0.05 often are not.

**Done when you can explain why fixed seeds matter.** Three reasons, all used in this project:

1. *Reproducibility.* Path $`i`$ of a split is always built from seed `start + i`
   (`hedge.seeds`), so anyone rerunning the code gets the same paths and the same table.
   That is how `make eval` on the Mac matched the cloud numbers exactly.
2. *A clean test set.* Seeds 10,000 to 19,999 are reserved for the final test and the code
   refuses them unless asked (`guard_test_seeds`). Without fixed ranges there would be no way
   to show the test paths were never used in tuning.
3. *Common random numbers.* Every strategy and every cost level is scored on the same shocks,
   so the difference between two strategies is not drowned in path-to-path noise. Section 10
   measures how much this narrows the confidence intervals.

## 4. Heston model (Phase 1)

**Read:** S. Heston, "A Closed-Form Solution for Options with Stochastic Volatility with
Applications to Bond and Currency Options", *Review of Financial Studies* 6(2), 1993
([PDF](https://www.ma.imperial.ac.uk/~ajacquie/IC_Num_Methods/IC_Num_Methods_Docs/Literature/Heston.pdf)),
sections 1 to 3 (read).

**The model**, in the project's notation (`hedge.market.heston`):

```math
dS_t = \mu S_t\,dt + \sqrt{v_t}\,S_t\,dW^1_t, \qquad
dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW^2_t, \qquad
d\langle W^1, W^2\rangle_t = \rho\,dt.
```

Heston writes $`\kappa^*, \theta^*`$ for the risk-neutral mean reversion and long-run variance
and $`\sigma`$ for what the project calls $`\xi`$. The deep hedging paper writes
$`\alpha, b, \sigma`$ for $`\kappa, \theta, \xi`$.

**The key ideas.**

- **Why stochastic volatility.** Black-Scholes with one volatility cannot match the prices of
  options at different strikes. Heston's section 3 explains what the extra parameters do:
  "The correlation parameter $`\rho`$ positively affects the skewness of spot returns", so a
  negative $`\rho`$ makes falls larger and raises the price of low-strike options (the skew);
  and $`\sigma`$ "controls the volatility of volatility": without it returns are normal, with
  it their kurtosis rises, which raises far-from-the-money prices and lowers near-the-money
  ones. The conclusion notes that Black-Scholes gives "virtually identical" prices for
  at-the-money options; the differences are in the wings.
- **The closed form.** The call price has the shape $`S P_1 - K P(t,T) P_2`$, where $`P_1`$
  and $`P_2`$ are probabilities. Heston guesses that the characteristic function is the
  exponential of a function linear in $`\log S`$ and $`v`$ (the paper's appendix), solves
  ordinary differential equations for the coefficients, and recovers $`P_1, P_2`$ by Fourier
  inversion, one numerical integral each. `hedge.market.heston.heston_price` does this with
  `scipy.integrate.quad`.
- **Feller and simulation.** The variance stays strictly positive when
  $`2\kappa\theta \ge \xi^2`$ (background; this condition is not in Heston's paper). When it
  fails the variance touches zero, and an Euler step can push it below zero. "Full truncation"
  keeps the negative state but uses $`\max(v, 0)`$ in every term, which keeps the scheme
  stable.

**Worked example: the five parameters in prices.** Implied volatilities of 30-day calls,
computed with `heston_price` and inverted through Black-Scholes:

| Strike | Project set ($`\kappa=2, \xi=0.3`$) | Paper set ($`\kappa=1, \xi=2`$) |
|---|---|---|
| 90 | 22.2% | 27.1% |
| 95 | 21.0% | 21.0% |
| 100 | 19.8% | 13.7% |
| 105 | 18.6% | 12.3% |
| 110 | 17.4% | 14.9% |

Both sets have $`v_0 = \theta = 0.04`$ (20% volatility) and $`\rho = -0.7`$. The project set
gives a gentle downward skew around 20%. The paper's volatility of variance of 2 makes the
smile very curved: the at-the-money option is cheap (13.7%) because most paths are calm, and
the 90 strike is expensive (27.1%) because the rare volatile paths are also the falling ones.
Feller holds for the project set ($`2\kappa\theta = 0.16 > \xi^2 = 0.09`$) and fails badly for
the paper set ($`0.08 < 4`$); on about half of the paper set's simulated days the variance
sits at zero.

**Done when you can say what each of the five parameters does.**

- $`v_0`$: the variance today; it sets the short-dated at-the-money level.
- $`\theta`$: the long-run variance the process returns to; it sets longer-dated levels.
- $`\kappa`$: how fast $`v`$ returns to $`\theta`$ (a half-life of $`\ln 2/\kappa`$ years,
  about 4 months for $`\kappa = 2`$).
- $`\xi`$: volatility of variance; more $`\xi`$ means fatter tails and a more curved smile.
- $`\rho`$: correlation of price and variance shocks; negative means falls come with rising
  volatility, which tilts the smile into a skew.

More detail, with one-at-a-time sensitivities, is in [`theory.md` section 3](theory.md).

## 5. Gymnasium environments (Phase 2)

**Read:** Gymnasium, ["Create a custom environment"](https://gymnasium.farama.org/introduction/create_custom_env/)
(read), and the `Env` API page.

**The key ideas.**

- **The contract.** An environment subclasses `gymnasium.Env`, declares `observation_space`
  and `action_space` in `__init__`, and implements `reset(seed=None, options=None)` returning
  `(observation, info)` and `step(action)` returning
  `(observation, reward, terminated, truncated, info)`.
- **Seeding.** The tutorial's `reset` starts with `super().reset(seed=seed)`, with the comment
  "IMPORTANT: Must call this first to seed the random number generator"; all randomness then
  comes from `self.np_random`.
- **`terminated` against `truncated`.** `terminated` means the task ended for a reason inside
  the problem; `truncated` means it was cut off from outside, typically by a time limit (the
  tutorial registers `max_episode_steps=300` to prevent infinite episodes, and the SB3 tips
  page says to handle timeouts by returning `truncated = True`). The difference matters for
  value learning: after a real end there is no future value, after a cut-off there is, and
  the learner should still bootstrap from the next state.
- **`check_env`** "will catch many common issues" (the tutorial), but it is not proof that
  the environment is right; the project's own tests check the finance.

**Worked example: one reset and one step** of `HedgingEnv` on GBM 10 bps, path seed 0, with
the delta-offset action and action 0 (which means "hold BS delta"):

- `observation_space` is `Box([-1, -10, -1, -1], [1, 10, 1, 1])` for the features
  `tau, log_moneyness, holding, bs_delta`; `action_space` is `Box(-1, 1, (1,))`.
- `reset` returns `[1, 0, -1, 0.0275]`: all the time left, at the money, holding nothing (the
  bottom of the allowed range), and delta 0.51376 scaled to [-1, 1].
- The first `step` buys 0.51376 shares and returns reward -0.053457. The quadratic reward is
  $`-(\text{gross})^2 - \lambda\cdot\text{cost} = -0.04562^2 - 0.05138`$, section 2's numbers.
  `terminated` and `truncated` are both false.
- After day 30, `terminated` is true (the option has expired, a real end) and `truncated` is
  always false: the episode's length is part of the problem, not an outside limit.

**In the code:** `hedge.env.hedging_env.HedgingEnv` and `hedge.env.rewards`; `reset` calls
`super().reset(seed=seed)` and draws training paths from `self.np_random`.

**Done when it passes `check_env`.** `tests/test_env.py::test_gymnasium_check_env` runs
`check_env` for every reward type and action mode.

## 6. RL basics: MDPs, value functions, policy gradients (Phase 3)

**Read:** Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd ed.
([free online](http://incompleteideas.net/book/the-book-2nd.html)), chapters 3, 6 and 13
(background; not reread for these notes).

**The key ideas.**

- **Chapter 3, finite MDPs.** An agent sees a state, takes an action, gets a reward and a next
  state. The return is $`G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots`$. The value
  $`v_\pi(s)`$ is the expected return from $`s`$ under policy $`\pi`$, and $`q_\pi(s,a)`$ the
  same after first taking $`a`$. The Bellman equation writes a value as one reward plus the
  discounted value of where you land.
- **The Markov property is a modelling choice.** Hedging is an MDP only if the state holds
  everything the best trade depends on: time left, price against strike, the current holding
  (because costs depend on how far you move) and, under Heston, the variance. That is exactly
  the project's observation.
- **Chapter 6, temporal-difference learning.** Instead of waiting for the end of the episode,
  update toward one reward plus the current estimate of the next state's value:
  $`V(s) \leftarrow V(s) + \alpha[r + \gamma V(s') - V(s)]`$. This bootstraps, which lowers
  variance and adds bias. SARSA updates toward the action actually taken next (on-policy),
  Q-learning toward the best next action (off-policy). SAC's critics descend from this.
- **Chapter 13, policy gradients.** Parameterise the policy directly and push its parameters
  toward actions that did better than expected:
  $`\nabla J \propto E[\nabla \log \pi(a|s)\,(G - b(s))]`$. The baseline $`b`$ (often a
  learned value, the "critic") leaves the expected gradient unchanged but cuts its variance.
  PPO is an actor-critic method of this kind.

**Worked example: what the discount does here.** The tuned PPO uses $`\gamma = 0.925`$
(`config/agents/ppo_tuned.yaml`, from the Optuna search). Its effective horizon is about
$`1/(1-\gamma) = 13`$ days, and a reward 29 days ahead is weighted $`0.925^{29} = 0.10`$; with
SB3's default 0.99 it would get 0.75. A short horizon suits the quadratic reward, which
charges each day's hedging error and cost on that day, so little of what matters is far
ahead. It would be a poor fit for the `terminal_cvar` reward, where everything is paid at
expiry; that reward trained worst in the reward grid (`docs/results.md`, Phase 3).

**Done when you can explain on-policy versus off-policy.** An *on-policy* method learns only
from data generated by the policy it is currently improving, so after each update the old
data is thrown away; Spinning Up says plainly that "PPO is an on-policy algorithm". An
*off-policy* method can learn from data generated by any policy, including older versions of
itself, so it keeps a replay buffer and reuses each sample many times (SAC, Q-learning, DQN).
Off-policy is usually far more sample-efficient; on-policy is usually more stable and easier
to tune. The Pendulum exercise shows the gap: SAC first scored better than -200 after 4,000
environment steps, tuned PPO after 30,000, and PPO with default settings not at all in
200,000 (see [`pendulum.md`](pendulum.md)). On the hedging task SAC reached CVaR 1.398 in
100,000 steps, a level PPO needed far more steps for (`docs/results.md`, "SAC against PPO").

## 7. PPO and SAC in practice (Phase 3)

**Read:** stable-baselines3, ["Reinforcement Learning Tips and Tricks"](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html),
the [PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) and
[SAC](https://stable-baselines3.readthedocs.io/en/master/modules/sac.html) pages, and Spinning
Up on [PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) and
[SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html) (all read).

**PPO, the idea.** Collect a batch of experience with the current policy, estimate each
action's advantage $`A`$ (how much better it was than expected), and improve the policy on that
batch for a few epochs without letting it move far from the policy that collected the data.
Spinning Up's simplified clipped objective for one sample is

```math
L = \min\left(r A,\ g(\epsilon, A)\right), \qquad
r = \frac{\pi_\theta(a \mid s)}{\pi_{\theta_k}(a \mid s)}, \qquad
g(\epsilon, A) = \begin{cases} (1+\epsilon)A & A \ge 0 \\ (1-\epsilon)A & A < 0. \end{cases}
```

With $`\epsilon = 0.2`$ (SB3's default `clip_range`): if $`A = 2`$ and the new policy makes the
action 1.5 times as likely, $`L = \min(3.0, 2.4) = 2.4`$, so there is nothing to gain from
pushing the ratio past 1.2. If $`A = -1`$ and the ratio has fallen to 0.6,
$`L = \min(-0.6, -0.8) = -0.8`$, so there is nothing to gain from pushing it below 0.8. In
Spinning Up's words, clipping "serves as a regularizer by removing incentives for the policy
to change dramatically".

**SAC, the idea.** Maximise reward plus an entropy bonus,
$`E[\sum_t \gamma^t (r_t + \alpha H(\pi(\cdot|s_t)))]`$, so the policy stays as random as it can
while still doing well. Two Q-networks are trained toward
$`y = r + \gamma(1-d)\,(\min_j Q_{\text{targ},j}(s', \tilde a') - \alpha \log \pi(\tilde a'|s'))`$,
with $`\tilde a'`$ drawn from the current policy; taking the smaller of the two counters
Q-learning's over-estimation. Actions are $`\tanh(\mu(s) + \sigma(s)\,\xi)`$, a squashed
Gaussian, so they are bounded and gradients can flow through the sampling (the
"reparameterization trick"). SB3 tunes $`\alpha`$ automatically by default
(`ent_coef='auto'`) toward a target entropy of minus the action dimension, here -1 (that value
is from SB3's source, `sac.py`; the docs page does not state it).

**Practical advice from the SB3 tips page, and what the project did with it.**

| Advice | In this project |
|---|---|
| "Rescale your actions so that they lie in [-1, 1]"; normalise observations when their bounds are known | Action in [-1, 1]; every observation scaled to about [-1, 1] (`HedgingEnv` docstring) |
| Evaluate on a separate environment; try `deterministic=True` for PPO | Evaluation runs on validation paths in `mode="eval"` with `deterministic=True` (`hedge.eval.evaluate.sb3_predict`) |
| "You should always do several runs to obtain quantitative results" | 5 training seeds per scenario; medians and ranges reported |
| Default hyperparameters often fail on new problems; start from the RL Zoo or tune | 30-trial Optuna search for PPO (`config/experiments/tune_ppo_offset.yaml`); SAC kept at defaults, a stated limitation |
| For continuous actions on one process: SAC, TD3 or TQC; with many processes: PPO | PPO with 8 parallel environments, SAC with 1 |
| Start with a shaped reward and a simpler problem | The per-step quadratic reward beat the terminal CVaR reward; the delta-offset action (start from delta) beat the raw holding |

SB3's defaults, for reference: PPO `learning_rate` 3e-4, `n_steps` 2048, `batch_size` 64,
`n_epochs` 10, `gamma` 0.99, `gae_lambda` 0.95, `clip_range` 0.2, `ent_coef` 0; SAC
`learning_rate` 3e-4, `buffer_size` 1,000,000, `batch_size` 256, `tau` 0.005, `gamma` 0.99,
one gradient step per environment step. The tuned PPO differs mainly in a learning rate about
six times smaller (4.7e-5) and the shorter discount discussed in section 6.

**Done when you can train PPO on Pendulum and read its TensorBoard curves.** The script and
curves are in [`pendulum.md`](pendulum.md) and `notebooks/pendulum_warmup.py`, with a guide to
the curves worth watching.

## 8. Risk measures: VaR, CVaR, entropic risk (Phase 3)

**Read:** Hull, *Risk Management and Financial Institutions*, the chapters on VaR and expected
shortfall (background); the deep hedging paper, section 3 (read); the Rockafellar and Uryasev
formula as restated by [Künzi-Bay and Mayer (2006)](https://core.ac.uk/download/159146995.pdf)
(read).

**The key ideas.**

- **VaR at 95%** is the 95th percentile of the loss: "losses are below this 95% of the time".
  It says nothing about how bad the other 5% are, and it is not subadditive (two positions can
  each have a small VaR and a large combined one).
- **CVaR at 95%** (expected shortfall) is the average loss in the worst 5%. It is a coherent
  risk measure, and it is convex, which is what lets deep hedging minimise it by gradient
  descent.
- **The deep hedging paper's definition** (Definition 3.1): a convex risk measure is monotone
  (a better position needs less cash), convex ("diversification works"), cash-invariant
  ($`\rho(X + c) = \rho(X) - c`$, so $`\rho(X)`$ is the cash that makes the position
  acceptable) and normalised ($`\rho(0) = 0`$). The hedging problem is
  $`\pi(X) = \inf_\delta \rho(X + (\delta\cdot S)_T - C_T(\delta))`$, the lowest risk trading
  can reach, and the indifference price is $`p(Z) = \pi(-Z) - \pi(0)`$: the cash that makes
  you indifferent between selling the option and not.
- **Optimised certainty equivalents** (the paper's equation 3.5):
  $`\rho(X) = \inf_w \{w + E[\ell(-X - w)]\}`$ for a convex loss function $`\ell`$. With
  $`\ell(x) = \max(x, 0)/(1-\alpha)`$ this is CVaR, the Rockafellar and Uryasev formula, and
  the minimising $`w`$ is the VaR. With an exponential $`\ell`$ it is the entropic risk
  $`\tfrac1a \log E[e^{aL}]`$, exponential utility in disguise.

**Worked example: CVaR from a sample by hand.** The hedging costs (minus the P&L) of the BS
delta hedge at 10 bps on the 20 dev paths with seeds 0 to 19, sorted:

    -0.2023 -0.1725 -0.1170 -0.1045 -0.0377  0.0703  0.1008  0.1199  0.1287  0.2263
     0.2859  0.3320  0.5654  0.5867  0.6274  0.6384  0.6965  0.7485  0.7904  0.8747

(Seed 0 is the path in the spreadsheet: cost -0.1170, which is P&L +0.1170.)

- Mean cost: 0.3079. Most of it is trading costs; without costs the mean would be near 0.
- CVaR 80%: the worst 20% of 20 is 4 paths, so (0.8747 + 0.7904 + 0.7485 + 0.6965) / 4 =
  **0.7775**.
- CVaR 95%: the worst 5% of 20 is one path, so **0.8747**.
- *The Rockafellar and Uryasev check.* Minimising
  $`w + \tfrac{1}{0.2}\,\overline{\max(L - w, 0)}`$ over $`w`$ on a fine grid gives 0.7775 at
  $`w = 0.6384`$, the fifth-largest cost: the minimum is the CVaR 80% and the minimiser sits
  at the VaR. The project's `terminal_cvar` reward uses this trick, with a fixed threshold
  standing in for $`w`$ (`hedge.env.rewards`).
- *VaR.* numpy's interpolated 80% quantile is 0.6500, between 0.6384 and 0.6965.
- *Entropic risk* $`\tfrac1a\log\overline{e^{aL}}`$: 0.367 with $`a = 1`$, 0.549 with
  $`a = 5`$, 0.739 with $`a = 20`$. A small $`a`$ stays close to the mean; a large $`a`$
  approaches the worst loss.
- `hedge.eval.metrics.cvar` uses the same rule (average the $`\lceil n(1-\alpha)\rceil`$
  largest) and gives the same numbers.

**Done when you can compute CVaR from a sample by hand.** Done above.

## 9. Deep hedging (Phase 4)

**Read:** Buehler, Gonon, Teichmann and Wood, "Deep Hedging", *Quantitative Finance* 19(8),
2019 ([arXiv:1802.03042](https://arxiv.org/abs/1802.03042), read). A summary is also in
[`theory.md` section 4](theory.md).

**What the paper does.**

- **Setup (section 2).** A market of hedging instruments, trading costs of several kinds
  (proportional, fixed, and costs on delta and vega when trading options), and a liability to
  hedge. The strategy at each date may use only what is known at that date.
- **Objective (section 3).** Minimise a convex risk measure of the hedged position, as in
  section 8. The resulting price is risk-adjusted, not the risk-neutral expectation.
- **Approximation (section 4).** Replace the strategy by neural networks. Proposition 4.9: as
  the networks grow, the network price converges to the true optimum, so restricting to
  networks loses nothing in principle.
- **Implementation (section 5.1).** 30 trading days, daily rebalancing ($`T = 30/365`$). One
  network per day, each with two hidden layers of $`d + 15`$ nodes, ReLU and batch
  normalisation, trained with Adam (learning rate 0.005, batch size 256) in TensorFlow.
- **The Heston experiments (section 5.2).** Heston with mean reversion 1, long-run variance
  0.04, volatility of variance 2, $`v_0 = 0.04`$, $`\rho = -0.7`$, $`s_0 = 100`$. The hedger
  trades the stock **and a variance swap**, so it can hedge volatility risk as well as price
  risk. For the at-the-money call the risk-neutral price is 1.69; with CVaR at level 0.5 the
  deep-hedging price is 1.94, and at 0.99 it is 3.49. The strategy trained for CVaR 0.99 has a
  higher mean loss than the one trained for 0.5 (0.26 against 0.15) but a smaller 0.99 tail
  (1.80 against 2.36): the risk measure you train on is the risk you get.
- **Costs (section 5.3).** With proportional costs $`\varepsilon`$, theory says the price
  premium for costs grows like $`\varepsilon^{2/3}`$; the fitted exponents in the paper are
  0.67 (Black-Scholes) and 0.71 (Heston).
- **High dimension (section 5.4).** Five independent Heston markets (ten instruments) hedged
  after a few hours of training, to show that the method scales.

**How the project differs.**

| | Paper | This project |
|---|---|---|
| Instruments | Stock and variance swap | Stock only |
| Network | One per day, $`d+15`$ nodes, ReLU, batch norm | One shared network with time to expiry as an input, 64 x 64, tanh |
| Action | Holdings | BS delta plus a bounded offset (`delta_offset`) |
| Training | Adam, lr 0.005, batch 256 | Adam, lr 0.001, batch 5,000 paths, early stopping on validation |
| Risk measure | CVaR at 0.5 and 0.99; entropic | CVaR 95% |
| Year | 365 days | 252 days |

**Connecting to the results.**

- Deep hedging has the lowest CVaR in all 8 test scenarios, 10 to 18% below BS delta
  (`docs/results.md`): the paper's main claim, reproduced in a smaller setting.
- The cost exponent does **not** reproduce, and should not be expected to. On validation
  paths the CVaR increase from 0 to 5, 10 and 20 bps grows with an exponent of about 1 for
  every strategy (deep hedging 0.97, BS delta 1.02, band 0.96 on GBM; similar on Heston). The
  $`\varepsilon^{2/3}`$ law is about small costs with continuous trading, where the best
  hedger widens its no-trade band as costs rise; with one trade a day and costs of a few bps,
  turnover hardly changes and costs grow in proportion. (This reading is an interpretation,
  not something the paper tests.)
- **Exploratory stress test with the paper's Heston set** (validation paths, post hoc;
  `docs/results.md`, "Exploratory: Heston with the deep hedging paper's parameters"). On
  paths with the paper's parameters every frozen hedger's CVaR roughly triples and the gaps
  between strategies shrink to a few percent. A check made for these notes suggests why: most
  of that risk is volatility risk, which a stock-only hedge cannot remove. On dev seeds, a BS
  delta hedge sold at the fair price and traded with **no** costs has a CVaR 95% of 5.08 in
  the paper's market, against 1.23 in the project's; realised volatility over the 30 days
  ranges from 2.6% to 44.7% (5th to 95th percentile) in the paper's market. The paper avoids
  this by giving its hedger a variance swap.

**Done when you can write the loss for one batch in PyTorch.** For a batch of paths `paths`
(shape `[B, N+1]`) and the network's holdings `h` (shape `[B, N]`):

```python
payoff = torch.relu(paths[:, -1] - K)
gain = (h * (paths[:, 1:] - paths[:, :-1])).sum(dim=1)
flat = torch.zeros_like(h[:, :1])
trades = torch.diff(torch.cat([flat, h, flat], dim=1), dim=1)  # includes buy at 0, sell at N
costs = kappa * (paths * trades.abs()).sum(dim=1)
cost = -(premium - payoff + gain - costs)  # hedging cost per path
k = math.ceil(0.05 * B)
loss = torch.topk(cost, k).values.mean()  # CVaR 95% of the batch
loss.backward()
```

This is `torch_hedge_pnl` plus `cvar_loss` in `hedge.agents.deep_hedging`. The whole P&L is
differentiable in the holdings, so the gradient flows from the tail of the batch straight
back into the network, with no reward signal or value function. Because of `topk`, only the
worst 5% of paths in each batch get a gradient, which is one reason the project uses batches
of 5,000 paths (250 in the tail) where the paper, at CVaR 0.5, could use 256. Swapping
`cvar_loss` for `entropic_loss` gives the paper's other objective.

## 10. Bootstrap confidence intervals (Phase 4)

**Read:** scipy's [`scipy.stats.bootstrap`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
page (read); Efron and Tibshirani, *An Introduction to the Bootstrap* (background).

**The key ideas.**

- **The bootstrap.** Resample the data with replacement, recompute the statistic, repeat; the
  spread of the recomputed values estimates the statistic's sampling error.
- **Paired resampling.** With `paired=True` scipy resamples indices and applies the same
  indices to every array. For a difference between two strategies scored on the same paths
  this keeps the common random numbers; resampling each strategy's paths separately throws
  them away and overstates the uncertainty.
- **Interval methods.** scipy offers `'percentile'`, `'basic'` and `'BCa'` (the default,
  bias-corrected and accelerated) and defaults to 9,999 resamples. The page notes that the
  plain percentile interval "is rarely used in practice"; BCa corrects for bias and skew in
  the bootstrap distribution. The project uses percentile intervals with 2,000 resamples,
  which the comparison below shows is enough here.

**Worked example: the CVaR gap of deep hedging to BS delta** on GBM 10 bps validation paths
(deep hedging seed 0, 10,000 paths; the two strategies' costs have correlation 0.82):

| Method | Gap | 95% interval | Width |
|---|---|---|---|
| Paired percentile, the project's `bootstrap_cvar_gap` | -0.215 | -0.242 to -0.189 | 0.053 |
| Paired percentile, `scipy.stats.bootstrap` | -0.215 | -0.242 to -0.189 | 0.053 |
| Paired BCa, `scipy.stats.bootstrap` | -0.215 | -0.243 to -0.191 | 0.052 |
| Unpaired percentile | -0.215 | -0.264 to -0.172 | 0.092 |

The project's function and scipy agree exactly (same generator, same resamples), BCa barely
moves the interval, and pairing makes it about 1.7 times narrower. The first version of these
notes said unpaired intervals would be "far too wide"; 1.7 times is the measured factor here.
Pairing matters most for small gaps, where an unpaired interval would often cross zero.

**Done when you can put a CI on a CVaR difference.** `hedge.eval.final.bootstrap_cvar_gap`
does it; this call reproduces it with scipy (`f` returns the CVaR 95% gap along the last axis,
so scipy can evaluate many resamples at once):

```python
def f(a, b, axis=-1):
    def tail(x):
        return np.sort(x, axis=axis)[..., -500:].mean(axis=axis)

    return tail(a) - tail(b)


res = stats.bootstrap(
    (deep_cost, delta_cost), f, paired=True, vectorized=True, n_resamples=2000, method="BCa"
)
```

## 11. FastAPI and Streamlit (Phase 5)

**Read:** the FastAPI tutorial, in particular ["Response Model"](https://fastapi.tiangolo.com/tutorial/response-model/),
and Streamlit's ["Caching overview"](https://docs.streamlit.io/develop/concepts/architecture/caching)
(both read).

**The key ideas.**

- **FastAPI path operations.** A decorated function such as `@app.get("/results")` becomes an
  endpoint; typed parameters become path or query parameters and are validated
  automatically. Request bodies are Pydantic models.
- **Response models.** Declaring the return type (or `response_model=`) makes FastAPI
  validate, document and **filter** the output: "It will limit and filter the output data to
  what is defined in the return type. This is particularly important for security." If both
  are given, `response_model` wins.
- **Streamlit reruns the script.** "Streamlit runs your script from top to bottom at every
  user interaction or code change." Anything slow must therefore be cached: `st.cache_data`
  for data such as DataFrames (each caller gets its own copy, so changing it is safe) and
  `st.cache_resource` for shared objects such as models or database connections (one
  instance, so changes are shared by everyone).

**In the code.** `src/hedge/api/main.py` has the read endpoints (`/scenarios`, `/strategies`,
`/results`, `/frontier`, `/evaluations`, `/runs/{run_id}/curve`, `/stress`, `/health`) and two
that compute, whose request bodies are the Pydantic models `HedgeRequest` and
`SimulateRequest`. `app/streamlit_app.py` wraps its data loaders in `@st.cache_data`, so
changing a widget reruns the page without rereading the files.

**Done when you can ship a page that reads from the API.** `src/hedge/api/main.py` and
`app/streamlit_app.py` are that page; hosting it is still on Aradhya's list.
