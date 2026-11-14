"""FastAPI service over the PostgreSQL experiment record.

    export DATABASE_URL=postgresql://user:password@localhost:5432/hedging
    uvicorn hedge.api.main:app --reload        # docs at http://localhost:8000/docs

Read-only apart from the two POST endpoints, which compute in memory and write nothing.
Results are served for one split: ``?split=test`` or ``?split=validation``. When the split is
left out, test results are used if the final run has been loaded and validation otherwise.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Literal

import psycopg
from fastapi import FastAPI, HTTPException
from psycopg.rows import dict_row
from pydantic import BaseModel, Field

from hedge.eval import live

Split = Literal["test", "validation"]


def _connect() -> psycopg.Connection:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise HTTPException(503, "DATABASE_URL is not set")
    try:
        return psycopg.connect(url, row_factory=dict_row)
    except psycopg.OperationalError as err:
        raise HTTPException(503, f"database unavailable: {err}") from err


def _query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with _connect() as conn:
        return conn.execute(sql, params).fetchall()


def _split(split: Split | None) -> str:
    """The requested split, or the best available one."""
    if split:
        return split
    rows = _query(
        "SELECT 1 FROM evaluations e JOIN path_sets ps USING (path_set_id) "
        "WHERE ps.split = 'test' LIMIT 1"
    )
    return "test" if rows else "validation"


MODELS: dict[str, Any] = {"loaded": [], "error": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the frozen agents once at start-up, so POST /hedge answers quickly."""
    try:
        for cost in live.TRAINED_COSTS:
            name = f"gbm_{cost}bps"
            MODELS["loaded"] += [f"{label}@{name}" for label in live.frozen_agents(name)]
    except Exception as err:  # noqa: BLE001 - report any load failure through /health
        MODELS["error"] = str(err)
    yield


app = FastAPI(
    title="RL derivative hedging",
    description="Experiment results, and the frozen hedging agents, behind one small API.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    """Database and model status."""
    try:
        n = _query("SELECT count(*) AS n FROM evaluations")[0]["n"]
        database = {"ok": True, "evaluations": n}
    except HTTPException as err:
        database = {"ok": False, "detail": err.detail}
    models = {"ok": MODELS["error"] is None and bool(MODELS["loaded"]), **MODELS}
    return {"status": "ok" if database["ok"] and models["ok"] else "degraded",
            "database": database, "models": models}  # fmt: skip


@app.get("/scenarios")
def scenarios() -> list[dict[str, Any]]:
    """Scenarios with market model, option and cost."""
    return _query(
        "SELECT sc.scenario_id, sc.name, sc.cost_bps, sc.steps_per_day, sc.config_hash, "
        "mm.name AS market_model, mm.model, mm.params, "
        "os.kind AS option_kind, os.moneyness, os.maturity_days "
        "FROM scenarios sc JOIN market_models mm USING (market_model_id) "
        "JOIN option_specs os USING (option_spec_id) ORDER BY mm.model, sc.cost_bps"
    )


@app.get("/strategies")
def strategies() -> list[dict[str, Any]]:
    """Strategies with kind, reward and lambda."""
    return _query(
        "SELECT strategy_id, name, kind, reward, risk_lambda, band_width, obs_features, "
        "hyperparams FROM strategies ORDER BY strategy_id"
    )


RESULTS_SQL = """
SELECT sc.name AS scenario, mm.model AS market, sc.cost_bps, st.name AS strategy, st.kind,
       count(*) AS n_evaluations, count(e.run_id) AS n_seeds,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY m.mean_cost) AS mean_cost,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY m.std_cost) AS std_cost,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY m.cvar95) AS cvar95,
       min(m.cvar95) AS cvar95_min, max(m.cvar95) AS cvar95_max,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY m.mean_turnover) AS mean_turnover,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY m.cvar_vs_delta) AS cvar_vs_delta,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY m.cvar_vs_delta_lo) AS cvar_vs_delta_lo,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY m.cvar_vs_delta_hi) AS cvar_vs_delta_hi
FROM eval_metrics m
JOIN evaluations e USING (eval_id)
JOIN path_sets ps ON ps.path_set_id = e.path_set_id AND ps.split = %s
JOIN scenarios sc USING (scenario_id)
JOIN market_models mm USING (market_model_id)
JOIN strategies st ON st.strategy_id = e.strategy_id
WHERE e.stress_test IS NULL {where}
GROUP BY sc.name, mm.model, sc.cost_bps, st.name, st.kind, st.strategy_id
ORDER BY mm.model, sc.cost_bps, st.strategy_id
"""


@app.get("/results")
def results(scenario: str | None = None, split: Split | None = None) -> list[dict[str, Any]]:
    """Rows of the results table: median over training seeds per strategy and scenario."""
    chosen = _split(split)
    if scenario:
        return _query(RESULTS_SQL.format(where="AND sc.name = %s"), (chosen, scenario))
    return _query(RESULTS_SQL.format(where=""), (chosen,))


@app.get("/frontier")
def frontier(
    market: Literal["gbm", "heston"] = "gbm", split: Split | None = None
) -> list[dict[str, Any]]:
    """Mean cost and CVaR for every strategy and cost level of one market."""
    rows = _query(RESULTS_SQL.format(where="AND mm.model = %s"), (_split(split), market))
    keep = ("scenario", "cost_bps", "strategy", "kind", "mean_cost", "cvar95", "cvar95_min",
            "cvar95_max", "n_seeds")  # fmt: skip
    return [{k: row[k] for k in keep} for row in rows]


@app.get("/evaluations")
def evaluations(scenario: str | None = None, split: Split | None = None) -> list[dict[str, Any]]:
    """Evaluation ids, to use with the distribution and path endpoints."""
    sql = (
        "SELECT e.eval_id, sc.name AS scenario, st.name AS strategy, tr.seed, e.stress_test, "
        "ps.split, m.mean_cost, m.cvar95 FROM evaluations e "
        "JOIN path_sets ps USING (path_set_id) JOIN scenarios sc USING (scenario_id) "
        "JOIN strategies st ON st.strategy_id = e.strategy_id "
        "LEFT JOIN training_runs tr ON tr.run_id = e.run_id "
        "JOIN eval_metrics m USING (eval_id) WHERE ps.split = %s"
    )
    params: tuple = (_split(split),)
    if scenario:
        sql += " AND sc.name = %s"
        params += (scenario,)
    return _query(sql + " ORDER BY e.eval_id", params)


@app.get("/evaluations/{eval_id}/distribution")
def distribution(eval_id: int) -> dict[str, Any]:
    """Hedging cost per path, for histograms."""
    rows = _query(
        "SELECT path_id, -pnl AS cost FROM eval_paths WHERE eval_id = %s ORDER BY path_id",
        (eval_id,),
    )
    if not rows:
        raise HTTPException(404, f"no per-path results for evaluation {eval_id}")
    return {"eval_id": eval_id, "n_paths": len(rows), "costs": [r["cost"] for r in rows]}


@app.get("/evaluations/{eval_id}/paths/{path_id}")
def path_detail(eval_id: int, path_id: int) -> dict[str, Any]:
    """Step-by-step spot, delta, holding and P&L for one of the stored sample paths."""
    steps = _query(
        "SELECT step, spot, tau, bs_delta, holding, step_cost, cum_pnl FROM eval_steps "
        "WHERE eval_id = %s AND path_id = %s ORDER BY step",
        (eval_id, path_id),
    )
    if not steps:
        raise HTTPException(404, "step detail is stored only for the first 25 paths")
    summary = _query(
        "SELECT premium, payoff, hedge_gain, txn_cost, pnl, turnover FROM eval_paths "
        "WHERE eval_id = %s AND path_id = %s",
        (eval_id, path_id),
    )
    return {"eval_id": eval_id, "path_id": path_id, "summary": summary[0], "steps": steps}


@app.get("/runs/{run_id}/curve")
def run_curve(run_id: int) -> dict[str, Any]:
    """Training curve for one run."""
    run = _query(
        "SELECT tr.run_id, st.name AS strategy, sc.name AS scenario, tr.seed, tr.total_steps, "
        "tr.wall_time_s, tr.status, tr.code_commit FROM training_runs tr "
        "JOIN strategies st USING (strategy_id) JOIN scenarios sc USING (scenario_id) "
        "WHERE tr.run_id = %s",
        (run_id,),
    )
    if not run:
        raise HTTPException(404, f"no training run {run_id}")
    curve = _query(
        "SELECT step, mean_reward, val_mean_cost, val_cvar95 FROM training_curve "
        "WHERE run_id = %s ORDER BY step",
        (run_id,),
    )
    return {**run[0], "curve": curve}


@app.get("/stress")
def stress(split: Split | None = None) -> list[dict[str, Any]]:
    """Stress-test results: median over training seeds per stress test and strategy."""
    return _query(
        "SELECT e.stress_test, sc.name AS scenario, st.name AS strategy, "
        "count(e.run_id) AS n_seeds, "
        "percentile_cont(0.5) WITHIN GROUP (ORDER BY m.mean_cost) AS mean_cost, "
        "percentile_cont(0.5) WITHIN GROUP (ORDER BY m.cvar95) AS cvar95, "
        "min(m.cvar95) AS cvar95_min, max(m.cvar95) AS cvar95_max "
        "FROM eval_metrics m JOIN evaluations e USING (eval_id) "
        "JOIN path_sets ps ON ps.path_set_id = e.path_set_id AND ps.split = %s "
        "JOIN scenarios sc USING (scenario_id) "
        "JOIN strategies st ON st.strategy_id = e.strategy_id "
        "WHERE e.stress_test IS NOT NULL "
        "GROUP BY e.stress_test, sc.name, st.name, st.strategy_id "
        "ORDER BY e.stress_test, st.strategy_id",
        (_split(split),),
    )


class HedgeRequest(BaseModel):
    spot: float = Field(gt=0, examples=[102.0])
    strike: float = Field(gt=0, examples=[100.0])
    days_left: int = Field(ge=1, le=30, examples=[12])
    volatility: float = Field(gt=0, le=2, examples=[0.2])
    current_holding: float = Field(ge=0, le=1, examples=[0.5])
    cost_bps: float = Field(ge=0, le=100, examples=[10])


@app.post("/hedge")
def hedge(request: HedgeRequest) -> dict[str, Any]:
    """The frozen agents' holding and the BS delta for one market state."""
    try:
        return live.hedge_decision(**request.model_dump())
    except ValueError as err:
        raise HTTPException(422, str(err)) from err


class SimulateRequest(BaseModel):
    market: Literal["gbm", "heston"] = "gbm"
    cost_bps: float = Field(default=10, ge=0, le=100)
    volatility: float | None = Field(default=None, gt=0, le=2)
    n_paths: int = Field(default=1000, ge=1, le=live.MAX_PATHS)
    first_seed: int = Field(default=0, ge=0)
    include_costs: bool = False


@app.post("/simulate")
def simulate(request: SimulateRequest) -> dict[str, Any]:
    """Run N fresh paths for the chosen settings and compare every strategy (at most 1,000)."""
    return live.simulate_comparison(**request.model_dump())
