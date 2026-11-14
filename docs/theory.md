# Theory notes

The four pieces of theory the project rests on, each tied to a number the code produces so it
can be checked. Project setting throughout: a short at-the-money call, $`S_0 = K = 100`$,
30 trading days to expiry ($`T = 30/252`$), volatility 20%, interest rate 0.

## 1. Black-Scholes call price and delta, derived once

**Model.** Under the pricing measure the stock follows
$`dS_t = r S_t\,dt + \sigma S_t\,dW_t`$, so at expiry

```math
S_T = S \exp\!\Big(\big(r - \tfrac{1}{2}\sigma^2\big)T + \sigma\sqrt{T}\,Z\Big), \qquad Z \sim N(0, 1).
```

**Price as an expectation.** The call is worth the discounted expected payoff:

```math
C = e^{-rT}\,E\big[(S_T - K)^+\big] = e^{-rT}E\big[S_T\,1_{S_T > K}\big] - K e^{-rT}P(S_T > K).
```

The option is exercised when $`S_T > K`$, which is the same as $`Z > -d_2`$ with

```math
d_2 = \frac{\ln(S/K) + (r - \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_1 = d_2 + \sigma\sqrt{T}.
```

*Second term.* $`P(Z > -d_2) = N(d_2)`$, so it equals $`K e^{-rT} N(d_2)`$.

*First term.* Write $`\varphi`$ for the standard normal density. Then

```math
e^{-rT}E\big[S_T\,1_{Z > -d_2}\big]
= S\int_{-d_2}^{\infty} e^{-\frac{1}{2}\sigma^2 T + \sigma\sqrt{T}z}\,\varphi(z)\,dz
= S\int_{-d_2}^{\infty} \varphi\big(z - \sigma\sqrt{T}\big)\,dz
= S\,N\big(d_2 + \sigma\sqrt{T}\big) = S\,N(d_1).
```

The middle step completes the square: $`-\tfrac{1}{2}z^2 + \sigma\sqrt{T}z - \tfrac{1}{2}\sigma^2T
= -\tfrac{1}{2}(z - \sigma\sqrt{T})^2`$.

**Result.**

```math
C = S\,N(d_1) - K e^{-rT} N(d_2).
```

**Delta.** Differentiating with respect to $`S`$ gives three terms:

```math
\frac{\partial C}{\partial S} = N(d_1) + S\,\varphi(d_1)\frac{\partial d_1}{\partial S}
- K e^{-rT}\varphi(d_2)\frac{\partial d_2}{\partial S}.
```

Both $`d_1`$ and $`d_2`$ have the same derivative, $`1/(S\sigma\sqrt{T})`$, so the last two terms
cancel if $`S\,\varphi(d_1) = K e^{-rT}\varphi(d_2)`$. They do:

```math
\frac{\varphi(d_2)}{\varphi(d_1)} = e^{\frac{1}{2}(d_1^2 - d_2^2)}
= e^{\,d_1\sigma\sqrt{T} - \frac{1}{2}\sigma^2 T} = e^{\ln(S/K) + rT} = \frac{S}{K}e^{rT}.
```

So $`\Delta = \partial C/\partial S = N(d_1)`$.

**Why delta is the hedge, and why it is not perfect here.** The seller is short the call and
holds $`h`$ shares. Over a short step the position changes by

```math
h\,\delta S - \Big(\Delta\,\delta S + \tfrac{1}{2}\Gamma\,(\delta S)^2 + \Theta\,\delta t\Big).
```

Choosing $`h = \Delta`$ removes the term in $`\delta S`$. What is left is
$`-\tfrac{1}{2}\Gamma(\delta S)^2 - \Theta\,\delta t`$. With $`r = 0`$ the Black-Scholes equation
says $`\Theta = -\tfrac{1}{2}\sigma^2 S^2\Gamma`$, so the leftover is

```math
\tfrac{1}{2}\Gamma S^2\Big(\sigma^2\,\delta t - \big(\delta S/S\big)^2\Big).
```

Trading continuously, $`(\delta S/S)^2 = \sigma^2\,\delta t`$ and the leftover vanishes. Trading
once a day it does not: the seller gains on quiet days and loses on days with a large move.
That leftover is the whole reason a cost-free daily delta hedge still has a standard deviation
of 0.43 in `docs/results.md`, and it is what a learned hedger can try to reshape.

**Check against the project.** With $`S = K`$ and $`r = 0`$, $`d_1 = \tfrac{1}{2}\sigma\sqrt{T}`$
and $`d_2 = -d_1`$:

| Quantity | Value |
| --- | --- |
| $`\sqrt{T}`$ | 0.34503 |
| $`d_1`$ | 0.03450 |
| Delta $`N(d_1)`$ | 0.51376 |
| Price $`100\,(N(d_1) - N(-d_1))`$ | 2.7524 |

The premium the project uses on GBM is 2.7524 and the first holding of the delta hedge is
0.5138 (see the ledger below).

## 2. Hedging P&L for one path, step by step

**Conventions.** The seller receives the premium at time 0, holds $`h_t`$ shares from day
$`t`$ to day $`t+1`$, and pays the payoff at expiry. Every trade costs a fraction $`\kappa`$ of
the value traded (10 basis points is $`\kappa = 0.001`$), including the first purchase and the
final sale of whatever is still held. The interest rate is 0, so cash earns nothing.

**Formula.** The profit and loss (P&L) of the seller, with $`h_{-1} = 0`$ and $`h_N = 0`$ (the position is closed at expiry):

```math
\text{profit} = \underbrace{C_0}_{\text{premium}} - \underbrace{(S_N - K)^+}_{\text{payoff}}
+ \underbrace{\sum_{t=0}^{N-1} h_t\,(S_{t+1} - S_t)}_{\text{hedge gain}}
- \underbrace{\kappa\sum_{t=0}^{N} S_t\,\lvert h_t - h_{t-1}\rvert}_{\text{trading costs}}.
```

"Hedging cost" in the results is this P&L with the sign flipped, so a positive cost is a loss.

**The same thing as a cash account.** Start with cash equal to the premium. Each day, buying
$`h_t - h_{t-1}`$ shares moves $`-(h_t - h_{t-1})S_t`$ of cash and costs
$`\kappa S_t\lvert h_t - h_{t-1}\rvert`$. At expiry sell everything and pay the payoff. The final
cash is the P&L. The engine (`hedge.pnl.hedge_pnl`) uses the formula, an independent ledger
(`hedge.pnl.cash_ledger`) uses the cash account, and a test checks they agree.

**One path worked through.** GBM at 10 basis points, path seed 0, delta hedge. Selected rows of

    python -m hedge.cli walkthrough --scenario config/scenarios/gbm_10bps.yaml --seed 0 --csv ledger.csv

| Day | Spot | Holding | Trade | Cash flow from stock | Trading cost | Cash |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 100.0000 | 0.5138 | +0.5138 | -51.3762 | 0.0514 | -48.6752 |
| 1 | 100.1506 | 0.5224 | +0.0086 | -0.8620 | 0.0009 | -49.5381 |
| 2 | 99.9761 | 0.5119 | -0.0105 | +1.0501 | 0.0011 | -48.4890 |
| 8 | 103.5457 | 0.7321 | +0.0727 | -7.5321 | 0.0075 | -71.0986 |
| 13 | 97.3373 | 0.3108 | -0.2176 | +21.1830 | 0.0212 | -29.3522 |
| 29 | 95.0285 | 0.0000 | -0.0058 | +0.5472 | 0.0005 | +0.1144 |
| 30 | 95.2849 | 0.0000 | -0.00003 | +0.0025 | 0.0000 | +0.1170 |

Day 0: cash is premium minus the purchase minus its cost,
2.7524 - 51.3762 - 0.0514 = -48.6752. The hedge buys as the stock rises (day 8) and sells as
it falls (day 13): it always buys high and sells low, which is where the premium goes.

Totals for the path:

| Term | Value |
| --- | --- |
| Premium received | +2.7524 |
| Payoff paid (stock ends at 95.28, below the strike) | 0.0000 |
| Hedge gain $`\sum h_t (S_{t+1} - S_t)`$ | -2.4375 |
| Trading costs (2.007 shares traded in total) | -0.1980 |
| **P&L** | **+0.1170** |

The option expired worthless, yet the seller kept only 0.117 of the 2.752 premium. The hedge
lost 2.44 buying high and selling low, which is the price of having been protected had the
stock gone up instead, and costs took another 0.20.

**Step-by-step P&L in the environment.** The agents need a reward every day, so the
environment also marks the option at its Black-Scholes value each day. Those marks cancel when
summed over the path, so the total is exactly the number above; a test checks this for every
reward type.

## 3. Heston's five parameters and what each does to the smile

**Model.** The variance $`v_t`$ is random and pulled back towards a long-run level:

```math
dS_t = r S_t\,dt + \sqrt{v_t}\,S_t\,dW^1_t, \qquad
dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW^2_t, \qquad
d\langle W^1, W^2\rangle_t = \rho\,dt.
```

(This $`\kappa`$ is the mean-reversion speed, not the cost rate of section 2.)

| Parameter | Project value | Meaning | Effect on the implied-volatility smile |
| --- | --- | --- | --- |
| $`v_0`$ | 0.04 | Variance today (20% volatility) | Moves the whole smile up or down, most at short maturities |
| $`\theta`$ | 0.04 | Long-run variance | Sets the level at long maturities; little effect at 30 days |
| $`\kappa`$ | 2 | Speed at which variance returns to $`\theta`$ | Higher: variance has less time to wander, so the smile flattens |
| $`\xi`$ | 0.3 | Volatility of variance | Higher: fatter tails, so more skew and more curvature |
| $`\rho`$ | -0.7 | Correlation of price and variance shocks | Negative: volatility rises when the stock falls, so the smile slopes down (the equity skew). Zero: a symmetric smile |

**Feller condition.** If $`2\kappa\theta \ge \xi^2`$ the variance never reaches zero. Here
$`2 \times 2 \times 0.04 = 0.16 \ge 0.09`$, so it holds.

**The effects, computed.** Implied volatility in percent for a 30-day option, from the
project's own Heston pricer (`hedge.market.heston.heston_price`), changing one parameter at a
time:

| Case | K = 90 | 95 | 100 | 105 | 110 |
| --- | --- | --- | --- | --- | --- |
| Project values | 22.24 | 21.01 | 19.77 | 18.55 | 17.43 |
| $`v_0 = 0.0625`$ (25% volatility) | 26.35 | 25.31 | 24.29 | 23.28 | 22.33 |
| $`\theta = 0.0625`$ | 22.75 | 21.57 | 20.39 | 19.23 | 18.16 |
| $`\kappa = 8`$ | 21.84 | 20.83 | 19.84 | 18.87 | 17.96 |
| $`\xi = 0.6`$ | 23.98 | 21.68 | 19.17 | 16.70 | 15.26 |
| $`\rho = 0`$ | 20.24 | 19.92 | 19.81 | 19.91 | 20.17 |
| $`\rho = -0.9`$ | 22.72 | 21.29 | 19.76 | 18.11 | 16.29 |

Reading the table: raising $`v_0`$ lifts every strike by 4 to 5 points, raising $`\theta`$ by
the same amount lifts them by only 0.6 because 30 days is short. Doubling $`\xi`$ doubles the
slope (the gap between strikes 95 and 105 goes from 2.5 to 5.0 points). Setting $`\rho = 0`$
removes the slope and leaves a small symmetric smile.

**Why the Heston premium (2.721) is below Black-Scholes (2.752).** An at-the-money call is
close to linear in volatility, which is the square root of average variance. The average
variance over the option's life is random with mean 0.04, and the square root is concave, so
the expected volatility is below 20% (Jensen's inequality). The at-the-money implied
volatility is 19.77%, not 20%.

**What this means for the delta baseline.** On Heston paths the baseline uses the
Black-Scholes delta with the current volatility $`\sqrt{v_t}`$. That ignores $`\rho`$: when the
stock falls, volatility tends to rise, which pushes the call's value back up. The call's true
sensitivity to the stock is therefore lower than its Black-Scholes delta, and the baseline
holds slightly too many shares. This is the "wrong model" in the project's second question.

## 4. Deep hedging paper: setup and results

H. Buehler, L. Gonon, J. Teichmann and B. Wood, "Deep Hedging", *Quantitative Finance* 19(8),
2019, pages 1271 to 1291 ([arXiv:1802.03042](https://arxiv.org/abs/1802.03042)).

**The idea.** Classical hedging starts from a model in which the option can be replicated
exactly and derives the hedge from its Greeks. With trading costs, limits or other frictions
there is no exact replication, so the paper turns hedging into an optimisation problem: among
all trading strategies, find the one that minimises a risk measure of the final P&L, costs
included,

```math
\pi(-Z) = \inf_{\delta}\ \rho\big(-Z + (\delta \cdot S)_T - C_T(\delta)\big),
```

where $`Z`$ is the payoff owed, $`(\delta\cdot S)_T`$ the trading gains, $`C_T`$ the trading costs
and $`\rho`$ a convex risk measure. The strategy $`\delta`$ is a neural network, trained by
simulating paths and following the gradient of the risk measure through the P&L. The same
quantity defines a price: the indifference price is the extra cash that makes selling the
option as acceptable as not selling it, $`\pi(-Z) - \pi(0)`$.

**Their setup** (section 5 of the paper):

- Market: Heston, with the stock and an idealised variance swap as hedging instruments, 30
  trading days, daily rebalancing.
- Payoffs: an at-the-money call and a call spread; a larger example with five independent
  Heston models.
- Risk measures: CVaR (they call it average value at risk) at several levels, and the entropic
  risk measure, which corresponds to exponential utility.
- Network: a separate small feed-forward network for each trading day, whose inputs are the
  market state and the previous holding. The previous holding matters once trading costs
  exist, because the best trade depends on what is already held.
- Training: Adam on simulated paths.

**Their results.**

- Without costs, at a moderate risk level, the network's strategy is very close to the model
  hedge, so the method recovers the classical answer when the classical answer applies.
- More risk aversion (a higher CVaR level) gives a higher price and a strategy that cuts the
  extreme losses.
- With proportional costs $`\varepsilon`$, the price rises above the cost-free price roughly
  like $`\varepsilon^{2/3}`$, the rate known from the theory of small transaction costs. They
  measure a slope of about 0.67 in Black-Scholes and about 0.71 in Heston.
- The cost of the method depends mainly on the number of hedging instruments, not on the
  size of the portfolio being hedged, so it scales to large books.

**How this project's version differs.**

| | Paper | This project |
| --- | --- | --- |
| Hedging instruments | Stock and variance swap | Stock only |
| Network | One per trading day | One shared across all days, with time to expiry as an input |
| Output | The holding | A correction of at most 0.25 shares around the Black-Scholes delta |
| Loss | CVaR at several levels, entropic | CVaR 95% only |
| Markets | Heston (and Black-Scholes for the cost study) | GBM and Heston, four cost levels each |
| Compared against | The model hedge | BS delta, a tuned no-trade band, PPO and SAC |

**What carried over.** The project sees the same pattern on its own test paths: the trained
network lowers CVaR 95% against the delta hedge in every scenario (10% to 18%), its advantage
is there even at zero cost because the objective is the tail and not the variance, and the
tail risk of every strategy rises with the cost level. The part the paper does not cover, and
the project adds, is the comparison with step-by-step reinforcement learning, which did not
match it.
