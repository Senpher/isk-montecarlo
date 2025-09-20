"""Configuration objects for the ISK Monte Carlo simulator."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional


class Granularity(str, Enum):
    """Supported simulation step sizes."""

    MONTHLY = "monthly"
    DAILY = "daily"


class ReturnModelType(str, Enum):
    """Available stochastic return models."""

    NORMAL = "normal"
    SPLIT_T = "split_t"


@dataclass(slots=True)
class ReturnModelConfig:
    """Parameters that describe the desired return distribution."""

    model: ReturnModelType = ReturnModelType.SPLIT_T
    arith_mean_annual: float = 0.1164
    stdev_annual: float = 0.1949
    df: float = 5.0
    asym: float = 1.3
    monthly_df: Optional[float] = None
    monthly_asym: Optional[float] = None
    daily_df: Optional[float] = None
    daily_asym: Optional[float] = None


@dataclass(slots=True)
class SimulationConfig:
    """Parameters that drive the wealth-accumulation simulation."""

    start_balance: float = 1_100_000.0
    monthly_deposit: float = 25_000.0
    target_real: float = 15_000_000.0
    inflation: float = 0.02
    isk_tax: float = 0.009
    years: int = 40
    sims: int = 10_000
    granularity: Granularity = Granularity.MONTHLY
    trading_days_per_year: int = 250
    seed: Optional[int] = 123
    progress: bool = True


@dataclass(slots=True)
class PlotConfig:
    """Options that control how plots are rendered and persisted."""

    bins: int = 40
    subset_paths: int = 200
    truncate_paths: bool = True
    save_plots: Optional[Path] = None
    show: bool = False


@dataclass(slots=True)
class CAGRConfig:
    """Parameters for stand-alone equity CAGR simulations."""

    years: int = 40
    sims: int = 12_000


DEFAULT_RETURN_MODEL = ReturnModelConfig()
DEFAULT_SIMULATION = SimulationConfig()
DEFAULT_PLOT = PlotConfig()
DEFAULT_CAGR = CAGRConfig()
