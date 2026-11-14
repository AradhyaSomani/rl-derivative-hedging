"""Plots for the write-up. Static PNGs, written to ``docs/figures/``.

Colour is used for identity only: each strategy keeps one colour in every figure, other
series take the free slots in a fixed order, dark grey for reference series (baselines).
Every series is also labelled directly or by marker shape, so no reading depends on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]
# A strategy keeps one colour in every figure it appears in.
STRATEGY_COLORS = {
    "BS delta": SERIES[0],
    "No-trade band": SERIES[1],
    "PPO": SERIES[2],
    "Deep hedging": SERIES[3],
    "SAC": SERIES[4],
}


def _colors(labels) -> list[str]:
    """One colour per label: the strategy's own colour if it has one, else the next free slot."""
    free = [c for c in SERIES if c not in {STRATEGY_COLORS.get(str(x)) for x in labels}]
    return [STRATEGY_COLORS.get(str(x)) or free.pop(0) for x in labels]


REFERENCE = "#52514e"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
FIGURES_DIR = Path("docs/figures")
BASELINE_MARKERS = {"no_hedge": "X", "bs_delta": "s", "no_trade_band": "D"}
BASELINE_LABELS = {"no_hedge": "No hedge", "bs_delta": "BS delta", "no_trade_band": "No-trade band"}


def _axes(figsize=(7.2, 4.4)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    _style(ax)
    return fig, ax


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _save(fig, file: Path) -> Path:
    file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return file


def learning_curve(
    curves: dict[str, pd.DataFrame],
    references: dict[str, float],
    title: str,
    file: Path,
    log_x: bool = False,
    x_label: str = "Training steps",
) -> Path:
    """Validation CVaR 95% against training steps, one line per run (at most three).

    ``references`` are horizontal lines, e.g. the delta hedge's and the band's CVaR.
    """
    fig, ax = _axes()
    if log_x:  # step 0 cannot be drawn on a log axis
        curves = {label: c[c["step"] > 0].reset_index(drop=True) for label, c in curves.items()}
    for color, (label, curve) in zip(_colors(curves), curves.items(), strict=False):
        ax.plot(curve["step"], curve["val_cvar95"], color=color, linewidth=2, label=label)
        ax.annotate(
            f"{label}  {curve['val_cvar95'].iloc[-1]:.2f}",
            (curve["step"].iloc[-1], curve["val_cvar95"].iloc[-1]),
            xytext=(-4 if log_x else 6, 14), textcoords="offset points",
            ha="right" if log_x else "left", va="center", fontsize=9, color=INK,
        )  # fmt: skip
    x_max = max(curve["step"].iloc[-1] for curve in curves.values())
    # Label the higher reference above its line and the lower one below, so they never collide.
    ordered = sorted(references.items(), key=lambda item: -item[1])
    x_min = min(c["step"][c["step"] > 0].min() for c in curves.values())
    for i, (label, value) in enumerate(ordered):
        ax.axhline(value, color=REFERENCE, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(
            f"{label}  {value:.2f}", (x_max, value), xytext=(6, 5 if i == 0 else -12),
            textcoords="offset points", fontsize=9, color=MUTED,
        )  # fmt: skip
    values = np.concatenate([c["val_cvar95"].to_numpy() for c in curves.values()])
    top = min(values.max(), 4 * max(references.values())) if references else values.max()
    ax.set_ylim(0, top * 1.08)
    if log_x:
        ax.set_xscale("log")
        ax.set_xlim(x_min / 1.5, x_max * 12)
    else:
        ax.set_xlim(0, x_max * 1.22)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Validation CVaR 95% of hedging cost")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:g}M" if x >= 1e6 else f"{x / 1e3:g}k")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    return _save(fig, file)


def holdings_vs_delta(
    holdings: np.ndarray, deltas: np.ndarray, prices: np.ndarray, title: str, file: Path
) -> Path:
    """The agent's holding against BS delta on a few sample paths (one small panel per path)."""
    n = holdings.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.9), sharey=True, facecolor=SURFACE)
    days = np.arange(holdings.shape[1])
    for i, ax in enumerate(np.atleast_1d(axes)):
        _style(ax)
        ax.step(days, deltas[i], where="post", color=REFERENCE, linewidth=1.5, label="BS delta")
        ax.step(days, holdings[i], where="post", color=SERIES[0], linewidth=2, label="Agent")
        ax.set_title(f"Path {i}: ends at {prices[i, -1]:.1f}", fontsize=9, color=MUTED, loc="left")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Day")
    np.atleast_1d(axes)[0].set_ylabel("Shares held")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    return _save(fig, file)


def frontier(table: pd.DataFrame, title: str, file: Path, group: str = "reward") -> Path:
    """Mean cost against CVaR 95%: down and left is better.

    ``table`` needs columns strategy, mean_cost, cvar95 and, for learned agents, ``group``
    (at most three distinct values, which get the three colours) and ``label``. Baselines are
    the rows whose strategy is a key of ``BASELINE_MARKERS``. A no-hedge row far off the scale
    is left out and noted in the title by the caller.
    """
    fig, ax = _axes((7.2, 5.0))
    agents = table[~table["strategy"].isin(BASELINE_MARKERS)]
    for color, (name, rows) in zip(SERIES, agents.groupby(group, sort=False), strict=False):
        ax.scatter(
            rows["mean_cost"], rows["cvar95"], s=70, color=color, edgecolor=SURFACE,
            linewidth=2, label=str(name), zorder=3,
        )  # fmt: skip
        for _, row in rows.iterrows():
            ax.annotate(
                str(row["label"]), (row["mean_cost"], row["cvar95"]), xytext=(6, 5),
                textcoords="offset points", fontsize=8, color=MUTED,
            )  # fmt: skip
    for strategy, marker in BASELINE_MARKERS.items():
        rows = table[table["strategy"] == strategy]
        if rows.empty:
            continue
        ax.scatter(
            rows["mean_cost"], rows["cvar95"], s=80, marker=marker, color=REFERENCE,
            edgecolor=SURFACE, linewidth=2, zorder=4,
        )  # fmt: skip
        ax.annotate(
            BASELINE_LABELS[strategy], (rows["mean_cost"].iloc[0], rows["cvar95"].iloc[0]),
            xytext=(8, -12), textcoords="offset points", fontsize=9, color=INK,
        )  # fmt: skip
    ax.set_xlabel("Mean hedging cost")
    ax.set_ylabel("CVaR 95% of hedging cost")
    ax.legend(frameon=False, fontsize=9, title=group.capitalize(), title_fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    return _save(fig, file)


def cost_frontier(table: pd.DataFrame, title: str, file: Path) -> Path:
    """Mean cost against CVaR 95% as the cost level rises, one line per strategy (at most four).

    Strategies are told apart by marker shape as well as colour.

    ``table`` has columns strategy_label, cost_bps, mean_cost, cvar95, and optionally
    cvar95_min / cvar95_max (the range over training seeds, drawn as a vertical bar).
    """
    fig, ax = _axes((7.2, 5.0))
    markers = ["s", "D", "^", "o"]
    groups = table.groupby("strategy_label", sort=False)
    names = [name for name, _ in groups]
    for color, marker, (name, rows) in zip(_colors(names), markers, groups, strict=False):
        rows = rows.sort_values("cost_bps")
        ax.plot(rows["mean_cost"], rows["cvar95"], color=color, linewidth=2, zorder=2)
        ax.scatter(
            rows["mean_cost"], rows["cvar95"], s=60, marker=marker, color=color,
            edgecolor=SURFACE, linewidth=2, label=str(name), zorder=3,
        )  # fmt: skip
        if "cvar95_min" in rows and rows["cvar95_min"].notna().any():
            ax.vlines(
                rows["mean_cost"], rows["cvar95_min"], rows["cvar95_max"], color=color,
                linewidth=1.2, zorder=1,
            )  # fmt: skip
    # Label the cost level once, on the first strategy's points.
    first = next(iter(groups))[1]
    for _, row in first.iterrows():
        ax.annotate(
            f"{row['cost_bps']:g} bps", (row["mean_cost"], row["cvar95"]), xytext=(8, -12),
            textcoords="offset points", ha="left", fontsize=8, color=MUTED,
        )  # fmt: skip
    ax.set_xlabel("Mean hedging cost")
    ax.set_ylabel("CVaR 95% of hedging cost")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    return _save(fig, file)


def cost_histogram(costs: dict[str, np.ndarray], title: str, file: Path) -> Path:
    """Overlapping histograms of hedging cost on shared bins, with mean and CVaR 95% marked.

    At most four strategies. Dashed vertical lines are means, solid ones CVaR 95%.
    """
    fig, ax = _axes((7.2, 4.4))
    everything = np.concatenate(list(costs.values()))
    lo, hi = np.quantile(everything, [0.001, 0.999])
    bins = np.linspace(lo, hi, 61)
    top = 0.0
    for color, (label, values) in zip(_colors(costs), costs.items(), strict=False):
        tail = float(np.sort(values)[-max(1, int(np.ceil(values.size * 0.05 - 1e-9))) :].mean())
        counts, _, _ = ax.hist(
            values, bins=bins, histtype="stepfilled", color=color, alpha=0.25, linewidth=0
        )
        ax.hist(
            values, bins=bins, histtype="step", color=color, linewidth=1.5,
            label=f"{label}: mean {values.mean():.2f}, CVaR {tail:.2f}",
        )  # fmt: skip
        top = max(top, counts.max())
        ax.axvline(values.mean(), color=color, linewidth=1.5, linestyle=(0, (4, 3)))
        ax.axvline(tail, color=color, linewidth=1.8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, top * 1.32)
    ax.set_xlabel("Hedging cost per option (dashed line: mean, solid line: CVaR 95%)")
    ax.set_ylabel("Paths")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    return _save(fig, file)
