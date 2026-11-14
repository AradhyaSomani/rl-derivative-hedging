-- Experiment record for the API and the demo. Simulated paths stay in Parquet files.
-- Every result row records the config hash and the code commit that produced it.

CREATE TYPE market_model AS ENUM ('gbm', 'heston', 'historical');
CREATE TYPE option_kind AS ENUM ('call', 'put');
CREATE TYPE path_split AS ENUM ('train', 'validation', 'test');
CREATE TYPE strategy_kind AS ENUM
    ('no_hedge', 'bs_delta', 'no_trade_band', 'ppo', 'sac', 'td3', 'deep_hedging', 'scripted_delta');
CREATE TYPE run_status AS ENUM ('running', 'finished', 'failed');

-- Setup
CREATE TABLE market_models (
    market_model_id SERIAL PRIMARY KEY,
    name           TEXT UNIQUE NOT NULL,            -- 'gbm_vol20', 'heston_base'
    model          market_model NOT NULL,
    params         JSONB NOT NULL,                  -- s0, r, mu, sigma | v0, kappa, theta, xi, rho
    data_source    TEXT                             -- for 'historical'
);

CREATE TABLE option_specs (
    option_spec_id SERIAL PRIMARY KEY,
    kind           option_kind NOT NULL,
    is_short       BOOLEAN NOT NULL DEFAULT TRUE,
    moneyness      REAL NOT NULL DEFAULT 1.0,       -- K / S0
    maturity_days  SMALLINT NOT NULL
);

CREATE TABLE scenarios (
    scenario_id    SERIAL PRIMARY KEY,
    name           TEXT UNIQUE NOT NULL,            -- 'heston_10bps'
    market_model_id INT NOT NULL REFERENCES market_models,
    option_spec_id INT NOT NULL REFERENCES option_specs,
    cost_bps       REAL NOT NULL,
    steps_per_day  SMALLINT NOT NULL DEFAULT 1,
    config_hash    TEXT NOT NULL
);

CREATE TABLE path_sets (
    path_set_id    SERIAL PRIMARY KEY,
    scenario_id    INT  NOT NULL REFERENCES scenarios,
    split          path_split NOT NULL,
    n_paths        INT  NOT NULL,
    seed_start     INT  NOT NULL,
    seed_end       INT  NOT NULL,
    parquet_path   TEXT NOT NULL,
    UNIQUE (scenario_id, split)
);

-- Strategies and training
CREATE TABLE strategies (
    strategy_id    SERIAL PRIMARY KEY,
    name           TEXT UNIQUE NOT NULL,            -- 'ppo_quadratic_l0.5'
    kind           strategy_kind NOT NULL,
    reward         TEXT,                            -- 'quadratic', 'mean_variance', 'terminal_cvar'
    risk_lambda    REAL,
    band_width     REAL,                            -- no-trade band
    obs_features   TEXT[],                          -- for ablations
    hyperparams    JSONB
);

CREATE TABLE training_runs (
    run_id         SERIAL PRIMARY KEY,
    strategy_id    INT  NOT NULL REFERENCES strategies,
    scenario_id    INT  NOT NULL REFERENCES scenarios,
    seed           INT  NOT NULL,
    total_steps    BIGINT,
    wall_time_s    REAL,
    status         run_status NOT NULL DEFAULT 'running',
    checkpoint     TEXT,                            -- saved weights
    tb_log_dir     TEXT,
    code_commit    TEXT NOT NULL,
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (strategy_id, scenario_id, seed)
);

CREATE TABLE training_curve (
    run_id         INT    NOT NULL REFERENCES training_runs ON DELETE CASCADE,
    step           BIGINT NOT NULL,
    mean_reward    REAL,
    val_mean_cost  REAL,
    val_cvar95     REAL,
    PRIMARY KEY (run_id, step)
);

-- Evaluation
CREATE TABLE evaluations (
    eval_id        SERIAL PRIMARY KEY,
    strategy_id    INT  NOT NULL REFERENCES strategies,
    run_id         INT  REFERENCES training_runs,   -- NULL for baselines
    path_set_id    INT  NOT NULL REFERENCES path_sets,
    stress_test    TEXT,                            -- NULL, 'vol_up', 'model_swap', 'jumps'
    code_commit    TEXT NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE eval_paths (
    eval_id        INT  NOT NULL REFERENCES evaluations ON DELETE CASCADE,
    path_id        INT  NOT NULL,
    premium        REAL NOT NULL,
    payoff         REAL NOT NULL,
    hedge_gain     REAL NOT NULL,
    txn_cost       REAL NOT NULL,
    pnl            REAL NOT NULL,                   -- premium - payoff + hedge_gain - txn_cost
    turnover       REAL NOT NULL,
    PRIMARY KEY (eval_id, path_id)
);

CREATE TABLE eval_steps (                          -- only for the demo's sample paths
    eval_id        INT      NOT NULL REFERENCES evaluations ON DELETE CASCADE,
    path_id        INT      NOT NULL,
    step           SMALLINT NOT NULL,
    spot           REAL     NOT NULL,
    tau            REAL     NOT NULL,
    bs_delta       REAL     NOT NULL,
    holding        REAL     NOT NULL,
    step_cost      REAL     NOT NULL,
    cum_pnl        REAL     NOT NULL,
    PRIMARY KEY (eval_id, path_id, step)
);

CREATE TABLE eval_metrics (
    eval_id        INT  PRIMARY KEY REFERENCES evaluations ON DELETE CASCADE,
    n_paths        INT  NOT NULL,
    mean_cost      REAL NOT NULL,
    std_cost       REAL NOT NULL,
    var95          REAL NOT NULL,
    cvar95         REAL NOT NULL,
    mean_txn_cost  REAL NOT NULL,
    mean_turnover  REAL NOT NULL,
    cvar_vs_delta  REAL,                            -- difference vs BS delta, same paths
    cvar_vs_delta_lo REAL,                          -- bootstrap 95% CI
    cvar_vs_delta_hi REAL
);

CREATE INDEX ON training_runs (scenario_id, strategy_id);
CREATE INDEX ON evaluations (path_set_id, strategy_id);

-- Results table for the README: median over seeds per strategy and scenario
CREATE VIEW results_table AS
SELECT sc.name AS scenario,
       st.name AS strategy,
       count(*)                                                AS n_seeds,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY m.mean_cost) AS mean_cost,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY m.std_cost)  AS std_cost,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY m.cvar95)    AS cvar95,
       min(m.cvar95) AS cvar95_min,
       max(m.cvar95) AS cvar95_max
FROM eval_metrics m
JOIN evaluations e ON e.eval_id = m.eval_id
JOIN path_sets  ps ON ps.path_set_id = e.path_set_id AND ps.split = 'test'
JOIN scenarios  sc ON sc.scenario_id = ps.scenario_id
JOIN strategies st ON st.strategy_id = e.strategy_id
WHERE e.stress_test IS NULL
GROUP BY sc.name, st.name;
