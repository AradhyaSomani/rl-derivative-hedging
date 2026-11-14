"""Streamlit demo: watch the hedgers on one path, then on 10,000, then change the market.

    streamlit run app/streamlit_app.py

Reads finished results from ``data/results`` (or the committed sample in ``app/sample``).
The "Try it" page computes in memory with the frozen agents and writes nothing.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:  # lets the app run on a host without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

import os  # noqa: E402

os.chdir(ROOT)  # configs and results are addressed relative to the project root

from hedge import results  # noqa: E402
from hedge.eval.metrics import cvar  # noqa: E402

results.configure_environment()

# Fixed colour per strategy, so a strategy keeps its colour on every page.
COLORS = {
    "BS delta": "#2a78d6",
    "No-trade band": "#eb6834",
    "PPO": "#1baf7a",
    "Deep hedging": "#eda100",
    "SAC": "#6250d6",
    "No hedge": "#52514e",
}
SHAPES = {
    "BS delta": "square",
    "Deep hedging": "circle",
    "No-trade band": "diamond",
    "PPO": "triangle-up",
    "SAC": "cross",
    "No hedge": "triangle-down",
}
ORDER = list(COLORS)
MUTED = "#52514e"


def color_scale(present: list[str]) -> alt.Scale:
    domain = [name for name in ORDER if name in present]
    return alt.Scale(domain=domain, range=[COLORS[name] for name in domain])


def shape_scale(present: list[str]) -> alt.Scale:
    domain = [name for name in ORDER if name in present]
    return alt.Scale(domain=domain, range=[SHAPES[name] for name in domain])


@st.cache_data
def load_summary() -> tuple[pd.DataFrame, str]:
    return results.summary_table(), results.available_split()


@st.cache_data
def load_costs(scenario: str) -> pd.DataFrame:
    return results.costs(scenario)


@st.cache_data
def load_steps(scenario: str) -> pd.DataFrame:
    return results.steps(scenario)


@st.cache_data
def load_stress() -> pd.DataFrame:
    return results.stress_table()


@st.cache_data(show_spinner=False)
def run_simulation(market: str, cost_bps: float, volatility: float, first_seed: int) -> dict:
    from hedge.eval.live import simulate_comparison

    return simulate_comparison(market, cost_bps, volatility, 1_000, first_seed)


def scenario_picker(summary: pd.DataFrame, key: str, default: str = "heston_10bps") -> str:
    names = sorted(summary["scenario"].unique(), key=lambda s: (s.split("_")[0], len(s), s))
    index = names.index(default) if default in names else 0
    return st.selectbox(
        "Scenario",
        names,
        index=index,
        key=key,
        help="The market model and the trading cost in basis points of traded value.",
    )


def split_note(split: str) -> None:
    if split == "validation":
        st.caption(
            "These numbers are on validation paths. The test paths have not been opened yet; "
            "they are scored once, after the setup is frozen."
        )
    else:
        st.caption("Final results on the 10,000 test paths per scenario, scored once.")


# Page 1 -------------------------------------------------------------------------------------
def page_race(summary: pd.DataFrame, split: str) -> None:
    st.header("Race: two hedgers, one path")
    st.write(
        "You sold one call option. Each day a hedger chooses how many shares to hold against "
        "it. Both hedgers below face the same price path and pay the same trading costs."
    )
    left, middle, right = st.columns(3)
    with left:
        scenario = scenario_picker(summary, "race_scenario")
    steps = load_steps(scenario)
    challengers = [
        n for n in ORDER if n in set(steps["label"]) and n not in ("BS delta", "No hedge")
    ]
    with middle:
        default = challengers.index("Deep hedging") if "Deep hedging" in challengers else 0
        challenger = st.selectbox("Racing against BS delta", challengers, index=default)
    with right:
        path_id = st.selectbox("Path", sorted(steps["path_id"].unique()))

    pair = steps[(steps["path_id"] == path_id) & steps["label"].isin(["BS delta", challenger])]
    n_days = int(pair["step"].max()) + 1
    if "race_day" not in st.session_state:
        st.session_state.race_day = n_days
    play = st.button("Play", type="primary")
    slot = st.empty()

    def draw(day: int) -> None:
        shown = pair[pair["step"] < day]
        present = ["BS delta", challenger]
        base = alt.Chart(shown).encode(
            x=alt.X("step:Q", title="Day", scale=alt.Scale(domain=[0, n_days - 1])),
            color=alt.Color(
                "label:N",
                scale=color_scale(present),
                title=None,
                legend=alt.Legend(orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Hedger"),
                alt.Tooltip("step:Q", title="Day"),
                alt.Tooltip("spot:Q", title="Share price", format=".2f"),
                alt.Tooltip("holding:Q", title="Shares held", format=".3f"),
                alt.Tooltip("cum_pnl:Q", title="Running P&L", format=".3f"),
            ],
        )
        holding = base.mark_line(interpolate="step-after", strokeWidth=2).encode(
            y=alt.Y("holding:Q", title="Shares held", scale=alt.Scale(domain=[0, 1]))
        )
        pnl = base.mark_line(strokeWidth=2).encode(
            y=alt.Y("cum_pnl:Q", title="Profit and loss so far")
        )
        with slot.container():
            a, b = st.columns(2)
            a.altair_chart(holding.properties(height=300, title="Shares held"), width="stretch")
            b.altair_chart(
                pnl.properties(height=300, title="Running P&L, option marked at its value"),
                width="stretch",
            )
            cols = st.columns(4)
            for i, name in enumerate(present):
                rows = shown[shown["label"] == name]
                if rows.empty:
                    continue
                traded = rows["holding"].diff().abs().fillna(rows["holding"].abs()).sum()
                cols[2 * i].metric(f"{name}: P&L so far", f"{rows['cum_pnl'].iloc[-1]:+.3f}")
                cols[2 * i + 1].metric(f"{name}: shares traded", f"{traded:.2f}")

    if play:
        for day in range(1, n_days + 1):
            draw(day)
            time.sleep(0.12)
        st.session_state.race_day = n_days
    else:
        draw(st.session_state.race_day)
    st.caption(
        "Watch how often each one trades. Every trade costs money, so a hedger that trades "
        "less keeps more of the premium, at the price of following the option less closely."
    )
    split_note(split)


# Page 2 -------------------------------------------------------------------------------------
def page_distributions(summary: pd.DataFrame, split: str) -> None:
    st.header("Distributions: what each hedge cost across 10,000 paths")
    st.write(
        "Hedging cost is what the hedge lost you on a path. The right tail is what matters: "
        "CVaR 95% is the average of the worst 5% of outcomes."
    )
    scenario = scenario_picker(summary, "dist_scenario")
    costs = load_costs(scenario)
    options = [n for n in ORDER if n in set(costs["label"]) and n != "No hedge"]
    default = [n for n in ("BS delta", "No-trade band", "Deep hedging") if n in options]
    chosen = st.multiselect(
        "Strategies (up to three, so the histograms stay readable)",
        options,
        default=default,
        max_selections=3,
    )
    if not chosen:
        st.info("Pick at least one strategy.")
        return
    data = costs[costs["label"].isin(chosen)]
    lo, hi = np.quantile(data["cost"], [0.001, 0.999])
    edges = np.linspace(lo, hi, 61)  # shared bins and axes for every strategy
    rows, marks = [], []
    for name in chosen:
        values = data.loc[data["label"] == name, "cost"].to_numpy()
        counts, _ = np.histogram(values, bins=edges)
        # One extra row closes the last step of the outline.
        rows.append(
            pd.DataFrame(
                {
                    "label": name,
                    "cost": edges,
                    "cost_end": np.append(edges[1:], edges[-1]),
                    "paths": np.append(counts, counts[-1]),
                }
            )
        )
        marks.append({"label": name, "value": values.mean(), "kind": "Mean"})
        marks.append({"label": name, "value": cvar(values), "kind": "CVaR 95%"})
    hist, marks = pd.concat(rows), pd.DataFrame(marks)
    bars = (
        alt.Chart(hist)
        .mark_area(interpolate="step-after", fillOpacity=0.22, line={"strokeWidth": 1.5})
        .encode(
            x=alt.X("cost:Q", title="Hedging cost per option", scale=alt.Scale(domain=[lo, hi])),
            y=alt.Y("paths:Q", title="Paths", stack=None),
            color=alt.Color(
                "label:N",
                scale=color_scale(chosen),
                title=None,
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Strategy"),
                alt.Tooltip("cost:Q", title="Cost from", format=".2f"),
                alt.Tooltip("cost_end:Q", title="to", format=".2f"),
                alt.Tooltip("paths:Q", title="Paths"),
            ],
        )
    )
    rules = (
        alt.Chart(marks)
        .mark_rule(strokeWidth=2)
        .encode(
            x="value:Q",
            color=alt.Color("label:N", scale=color_scale(chosen), legend=None),
            strokeDash=alt.StrokeDash(
                "kind:N",
                scale=alt.Scale(domain=["Mean", "CVaR 95%"], range=[[4, 3], [1, 0]]),
                title="Line",
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Strategy"),
                alt.Tooltip("kind:N", title="Line"),
                alt.Tooltip("value:Q", title="Value", format=".3f"),
            ],
        )
    )
    st.altair_chart((bars + rules).properties(height=380), width="stretch")
    st.caption("Dashed lines mark each strategy's mean cost, solid lines its CVaR 95%.")

    table = (
        summary[summary["scenario"] == scenario].set_index("label").reindex(ORDER).dropna(how="all")
    )
    st.subheader("Results table")
    show = table[
        ["mean_cost", "std_cost", "cvar95", "cvar95_min", "cvar95_max", "mean_turnover", "n_seeds"]
    ]
    show.columns = ["Mean cost", "Std", "CVaR 95%", "CVaR min", "CVaR max", "Turnover", "Seeds"]
    st.dataframe(show.style.format("{:.3f}", subset=show.columns[:6]), width="stretch")
    st.caption(
        "Learned strategies show the median over 5 training seeds, with the range of CVaR "
        "across seeds. Baselines have no training seed."
    )
    split_note(split)


# Page 3 -------------------------------------------------------------------------------------
def page_frontier(summary: pd.DataFrame, split: str) -> None:
    st.header("Frontier: cost against tail risk")
    st.write(
        "Every point is a strategy at one cost level. Down and to the left is better: cheaper "
        "on average and with a smaller bad tail."
    )
    market = st.radio("Market", ["gbm", "heston"], format_func=str.upper, horizontal=True)
    data = summary[(summary["market"] == market) & (summary["strategy"] != "no_hedge")].copy()
    present = [n for n in ORDER if n in set(data["label"])]
    base = alt.Chart(data).encode(
        x=alt.X("mean_cost:Q", title="Mean hedging cost", scale=alt.Scale(zero=False)),
        y=alt.Y("cvar95:Q", title="CVaR 95% of hedging cost", scale=alt.Scale(zero=False)),
        color=alt.Color("label:N", scale=color_scale(present), title="Strategy"),
    )
    lines = base.mark_line(strokeWidth=2).encode(order="cost_bps:Q")
    points = base.mark_point(
        size=90, filled=True, opacity=1, stroke="#fcfcfb", strokeWidth=2
    ).encode(
        shape=alt.Shape("label:N", scale=shape_scale(present), title="Strategy"),
        tooltip=[
            alt.Tooltip("label:N", title="Strategy"),
            alt.Tooltip("cost_bps:Q", title="Cost (bps)"),
            alt.Tooltip("mean_cost:Q", title="Mean cost", format=".3f"),
            alt.Tooltip("cvar95:Q", title="CVaR 95%", format=".3f"),
            alt.Tooltip("cvar95_min:Q", title="CVaR, best seed", format=".3f"),
            alt.Tooltip("cvar95_max:Q", title="CVaR, worst seed", format=".3f"),
        ],
    )
    spread = base.mark_rule(strokeWidth=1.2).encode(y="cvar95_min:Q", y2="cvar95_max:Q")
    cost_labels = (
        alt.Chart(data[data["label"] == "BS delta"])
        .mark_text(dx=10, dy=12, align="left", color=MUTED, fontSize=11)
        .encode(x="mean_cost:Q", y="cvar95:Q", text=alt.Text("cost_bps:Q", format=".0f"))
    )
    st.altair_chart((lines + spread + points + cost_labels).properties(height=440), width="stretch")
    st.caption(
        "Lines join one strategy across cost levels of 0, 5, 10 and 20 basis points (numbered "
        "on the BS delta line). Vertical bars show the range over 5 training seeds."
    )

    stress = load_stress()
    if not stress.empty:
        st.subheader("Stress tests: a market the hedgers were not trained on")
        names = {
            "vol_up": "Volatility 30% (trained at 20%)",
            "vol_down": "Volatility 10% (trained at 20%)",
            "model_swap": "Heston paths (trained on GBM)",
            "jumps_gbm": "Rare jumps added (GBM)",
            "jumps_heston": "Rare jumps added (Heston)",
        }
        test = st.selectbox("Stress test", list(names), format_func=names.get)
        rows = stress[(stress["stress_test"] == test) & (stress["strategy"] != "no_hedge")]
        rows = rows.set_index("label").reindex(ORDER).dropna(how="all")
        show = rows[["cvar95_base", "cvar95", "cvar95_worsening"]]
        show.columns = ["CVaR 95% in training market", "CVaR 95% under stress", "Worsening"]
        st.dataframe(show.style.format("{:.3f}"), width="stretch")
        st.caption("A robust hedger's tail risk worsens less than BS delta's.")
    split_note(split)


# Page 4 -------------------------------------------------------------------------------------
def page_try_it() -> None:
    st.header("Try it: change the market")
    st.write(
        "Pick a cost level and a volatility. The page simulates 1,000 fresh paths and hedges "
        "each one with every strategy. The agents are the ones trained at the nearest cost "
        "level and at 20% volatility, so moving the sliders away from that tests them."
    )
    a, b, c = st.columns(3)
    market = a.radio("Market", ["gbm", "heston"], format_func=str.upper, horizontal=True)
    cost_bps = b.slider("Trading cost (basis points)", 0, 30, 10)
    volatility = c.slider("Volatility (%)", 10, 40, 20) / 100
    if "try_seed" not in st.session_state:
        st.session_state.try_seed = 0
    if st.button("New paths"):
        st.session_state.try_seed = (st.session_state.try_seed + 1_000) % 9_000
    start = time.time()
    with st.spinner("Simulating 1,000 paths"):
        out = run_simulation(market, float(cost_bps), volatility, st.session_state.try_seed)
    elapsed = time.time() - start
    table = pd.DataFrame(
        {
            name: {
                "Mean cost": m["mean_cost"],
                "Std": m["std_cost"],
                "CVaR 95%": m["cvar95"],
                "Trading cost paid": m["mean_txn_cost"],
                "Turnover": m["mean_turnover"],
            }
            for name, m in out["strategies"].items()
        }
    ).T.reindex([n for n in ORDER if n in out["strategies"]])
    st.dataframe(table.style.format("{:.3f}"), width="stretch")

    hedged = table.drop(index="No hedge", errors="ignore").reset_index(names="label")
    present = list(hedged["label"])
    chart = (
        alt.Chart(hedged)
        .mark_point(size=120, filled=True, opacity=1, stroke="#fcfcfb", strokeWidth=2)
        .encode(
            x=alt.X("Mean cost:Q", scale=alt.Scale(zero=False)),
            y=alt.Y("CVaR 95%:Q", scale=alt.Scale(zero=False)),
            color=alt.Color("label:N", scale=color_scale(present), legend=None),
            shape=alt.Shape("label:N", scale=shape_scale(present), legend=None),
            tooltip=[
                alt.Tooltip("label:N", title="Strategy"),
                alt.Tooltip("Mean cost:Q", format=".3f"),
                alt.Tooltip("CVaR 95%:Q", format=".3f"),
            ],
        )
    )
    text = chart.mark_text(dx=12, align="left", color=MUTED, fontSize=12).encode(
        text="label:N", color=alt.value(MUTED), shape=alt.value("circle")
    )
    st.altair_chart((chart + text).properties(height=340), width="stretch")
    st.caption(
        f"Option premium received: {out['premium']:.3f}. Agents trained on "
        f"{out['agents_trained_on']}. With only 1,000 paths, CVaR is the average of the worst "
        f"50, so it moves from one set of paths to the next. Computed in {elapsed:.1f} seconds."
    )


# Page 5 -------------------------------------------------------------------------------------
def page_about(summary: pd.DataFrame, split: str) -> None:
    st.header("What this is")
    st.write(
        "This project asks whether a learned hedger can protect a sold option more cheaply, "
        "and with a smaller bad tail, than the textbook rule (Black-Scholes delta hedging) "
        "once trading costs exist and the market model is wrong."
    )
    st.markdown(
        "- **BS delta** rebalances to the textbook hedge every day.\n"
        "- **No-trade band** only trades when it has drifted far enough from delta. It is the "
        "honest competitor: cheap, simple and aware of costs.\n"
        "- **PPO and SAC** are reinforcement-learning agents trained with a per-step reward.\n"
        "- **Deep hedging** trains a network directly on the tail risk of the whole episode."
    )
    readme = ROOT / "docs" / "results.md"
    if readme.exists():
        with st.expander("Full results write-up, including what didn't work"):
            st.markdown(readme.read_text())
    split_note(split)


def main() -> None:
    st.set_page_config(page_title="RL derivative hedging", layout="wide")
    try:
        summary, split = load_summary()
    except FileNotFoundError:
        st.error(
            "No results found. Run `make eval` (or `python -m hedge.cli eval --split validation "
            "--scenarios config/scenarios/*.yaml`) first, or restore `app/sample`."
        )
        return
    pages = {
        "Race": lambda: page_race(summary, split),
        "Distributions": lambda: page_distributions(summary, split),
        "Frontier": lambda: page_frontier(summary, split),
        "Try it": page_try_it,
        "About": lambda: page_about(summary, split),
    }
    st.sidebar.title("RL derivative hedging")
    choice = st.sidebar.radio("Page", list(pages))
    st.sidebar.caption("Hedging a short call option under trading costs.")
    pages[choice]()


main()
