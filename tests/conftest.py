from pathlib import Path

import pytest

from hedge.config import Scenario

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_DIR = ROOT / "config" / "scenarios"


@pytest.fixture(scope="session")
def gbm_scenario() -> Scenario:
    return Scenario.from_yaml(SCENARIO_DIR / "gbm_10bps.yaml")


@pytest.fixture(scope="session")
def heston_scenario() -> Scenario:
    return Scenario.from_yaml(SCENARIO_DIR / "heston_10bps.yaml")
