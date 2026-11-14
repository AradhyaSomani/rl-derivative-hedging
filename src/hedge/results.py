"""File-based access to finished results, for the demo and the write-up.

Results are read from ``data/results`` when it exists (a full local run) and otherwise from
``app/sample``, a small committed copy made by ``python -m hedge.cli export-sample`` so the
demo can be deployed without re-running anything. Test results are preferred once the final
run exists; until then everything shown is on validation paths.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from hedge.eval.final import load_eval_config

FULL_DIR = Path("data/results")
SAMPLE_DIR = Path("app/sample")
BASELINE_LABELS = {"no_hedge": "No hedge", "bs_delta": "BS delta", "no_trade_band": "No-trade band"}


def results_dir() -> Path:
    if any(FULL_DIR.glob("final_*_table.csv")):
        return FULL_DIR
    return SAMPLE_DIR


def runs_dir() -> Path:
    """Where the trained models are; also exported to HEDGE_RUNS_DIR for ``hedge.eval.live``."""
    full = Path("data/runs")
    return full if full.exists() and any(full.iterdir()) else SAMPLE_DIR / "runs"


def available_split(directory: Path | None = None) -> str:
    directory = directory or results_dir()
    return "test" if (directory / "final_test_table.csv").exists() else "validation"


def labels() -> dict[str, str]:
    """Strategy name -> display label."""
    learned = {e["strategy"]: e["label"] for e in load_eval_config()["learned"]}
    return {**BASELINE_LABELS, **learned}


def _decorate(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["label"] = table["strategy"].map(labels()).fillna(table["strategy"])
    if "scenario" in table:
        table["market"] = table["scenario"].str.split("_").str[0]
        table["cost_bps"] = table["scenario"].str.extract(r"_(\d+)bps")[0].astype(float)
    return table


def summary_table(split: str | None = None) -> pd.DataFrame:
    """Median over training seeds per scenario and strategy."""
    directory = results_dir()
    split = split or available_split(directory)
    return _decorate(pd.read_csv(directory / f"final_{split}_table.csv"))


def stress_table(split: str | None = None) -> pd.DataFrame:
    directory = results_dir()
    split = split or available_split(directory)
    file = directory / f"stress_{split}_table.csv"
    return _decorate(pd.read_csv(file)) if file.exists() else pd.DataFrame()


def median_seeds(split: str | None = None) -> pd.DataFrame:
    """For each scenario and strategy, the seed whose CVaR is the median (-1 for baselines)."""
    directory = results_dir()
    split = split or available_split(directory)
    rows = pd.read_csv(directory / f"final_{split}.csv")
    rows["seed"] = rows["seed"].fillna(-1).astype(int)
    rows = rows.sort_values("cvar95")
    picked = rows.groupby(["scenario", "strategy"]).apply(
        lambda g: g.iloc[(len(g) - 1) // 2]["seed"], include_groups=False
    )
    return picked.rename("seed").reset_index()


def _median_only(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    if frame["seed"].nunique() <= 1 and (frame["seed"] == -1).all():
        return frame
    keep = median_seeds(split)
    return frame.merge(keep, on=["scenario", "strategy", "seed"])


def costs(scenario: str, split: str | None = None) -> pd.DataFrame:
    """Per-path hedging cost for every strategy on one scenario (median seed of each)."""
    directory = results_dir()
    split = split or available_split(directory)
    frame = pd.read_parquet(
        directory / f"costs_{split}.parquet", filters=[("scenario", "==", scenario)]
    )
    frame = _median_only(frame, split)
    frame["cost"] = -frame["pnl"]
    return _decorate(frame)


def steps(scenario: str, split: str | None = None) -> pd.DataFrame:
    """Step-by-step detail of the sample paths on one scenario (median seed of each)."""
    directory = results_dir()
    split = split or available_split(directory)
    frame = pd.read_parquet(
        directory / f"steps_{split}.parquet", filters=[("scenario", "==", scenario)]
    )
    return _decorate(_median_only(frame, split))


def export_sample(sample_dir: Path = SAMPLE_DIR) -> list[Path]:
    """Copy what the demo needs into ``app/sample``: tables, median-seed costs, steps, models."""
    source = FULL_DIR
    split = available_split(source)
    if sample_dir.exists():  # rebuilt from scratch, so files of an earlier split do not linger
        shutil.rmtree(sample_dir)
    sample_dir.mkdir(parents=True)
    written = []
    for name in (f"final_{split}.csv", f"final_{split}_table.csv", f"stress_{split}_table.csv"):
        if (source / name).exists():
            written.append(Path(shutil.copy(source / name, sample_dir / name)))
    keep = median_seeds(split)
    for name, columns in (
        (f"costs_{split}.parquet", ["scenario", "strategy", "seed", "path_id", "pnl", "txn_cost"]),
        (f"steps_{split}.parquet", None),
    ):
        frame = pd.read_parquet(source / name, columns=columns)
        frame = frame.merge(keep, on=["scenario", "strategy", "seed"])
        float_cols = frame.select_dtypes("float64").columns
        frame[float_cols] = frame[float_cols].astype("float32")
        frame.to_parquet(sample_dir / name, index=False)
        written.append(sample_dir / name)
    for _, row in keep[keep["seed"] >= 0].iterrows():
        run = Path("data/runs") / row["strategy"] / row["scenario"] / f"seed{row['seed']}"
        target = sample_dir / "runs" / row["strategy"] / row["scenario"] / f"seed{row['seed']}"
        target.mkdir(parents=True, exist_ok=True)
        for file in ("run.json", "eval_validation.json", "model.zip", "model.pt"):
            if (run / file).exists():
                written.append(Path(shutil.copy(run / file, target / file)))
    return written


def configure_environment() -> None:
    """Point the live-computation code at whichever model folder exists."""
    os.environ.setdefault("HEDGE_RUNS_DIR", str(runs_dir()))
