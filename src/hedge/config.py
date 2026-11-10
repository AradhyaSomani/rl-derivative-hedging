"""Scenario configs. Every run is launched from a YAML file in ``config/``, not edited code."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from hedge.market import black_scholes as bs
from hedge.market.heston import heston_price

_HESTON_KEYS = ("v0", "kappa", "theta", "xi", "rho")


def config_hash(obj: Any) -> str:
    """Short, stable hash of a config: 10 hex characters of the SHA-256 of its canonical JSON."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:10]


@dataclass(frozen=True)
class Scenario:
    """One market, one option, one cost level."""

    name: str
    market: dict[str, Any]
    option: dict[str, Any]
    grid: dict[str, Any]
    cost_bps: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Scenario:
        scenario = cls(
            name=raw["name"],
            market=dict(raw["market"]),
            option=dict(raw["option"]),
            grid=dict(raw["grid"]),
            cost_bps=float(raw["cost_bps"]),
        )
        if scenario.model not in ("gbm", "heston"):
            raise ValueError(f"unknown market model {scenario.model!r}")
        if scenario.market.get("r", 0.0) != 0.0:
            raise ValueError("the P&L engine assumes a zero interest rate in v1")
        if scenario.cost_bps < 0:
            raise ValueError("cost_bps must be non-negative")
        return scenario

    @classmethod
    def from_yaml(cls, path: str | Path) -> Scenario:
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    # Derived quantities
    @property
    def model(self) -> str:
        return self.market["model"]

    @property
    def kind(self) -> str:
        return self.option["kind"]

    @property
    def s0(self) -> float:
        return float(self.market["s0"])

    @property
    def strike(self) -> float:
        return self.s0 * float(self.option["moneyness"])

    @property
    def n_steps(self) -> int:
        return int(self.option["maturity_days"]) * int(self.grid["steps_per_day"])

    @property
    def dt(self) -> float:
        return 1.0 / (float(self.grid["days_per_year"]) * int(self.grid["steps_per_day"]))

    @property
    def maturity(self) -> float:
        return self.n_steps * self.dt

    @property
    def cost_rate(self) -> float:
        return self.cost_bps / 10_000.0

    @property
    def heston_params(self) -> dict[str, float]:
        return {k: float(self.market[k]) for k in _HESTON_KEYS}

    def premium(self) -> float:
        """What the option is sold for: the fair price in the scenario's own market model."""
        if self.model == "gbm":
            return float(
                bs.price(self.s0, self.strike, self.maturity, self.market["sigma"], 0.0, self.kind)
            )
        return heston_price(
            self.s0, self.strike, self.maturity, **self.heston_params, r=0.0, kind=self.kind
        )

    # Hashes
    def paths_config(self) -> dict[str, Any]:
        """Everything that determines the simulated paths. Cost level and option kind do not."""
        return {
            "market": self.market,
            "maturity_days": self.option["maturity_days"],
            "grid": self.grid,
        }

    @property
    def paths_hash(self) -> str:
        return config_hash(self.paths_config())

    @property
    def hash(self) -> str:
        return config_hash(
            {
                "market": self.market,
                "option": self.option,
                "grid": self.grid,
                "cost_bps": self.cost_bps,
            }
        )
