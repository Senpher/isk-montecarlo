"""Simulation engines for ISK Monte Carlo experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
from numpy.random import Generator, default_rng

from .config import Granularity, ReturnModelConfig, SimulationConfig
from .models import build_return_sampler

try:
    from tqdm import trange
except Exception:  # pragma: no cover - tqdm optional
    trange = None


@dataclass(slots=True)
class SimulationResult:
    """Holds full simulation paths and useful metadata."""

    paths_real: np.ndarray
    crossing_index: np.ndarray
    years_axis: np.ndarray
    periods_per_year: float
    target: float

    def valid_crossings(self) -> np.ndarray:
        mask = self.crossing_index >= 0
        return self.crossing_index[mask]

    def years_to_target(self) -> np.ndarray:
        crossings = self.valid_crossings()
        if crossings.size == 0:
            return np.array([], dtype=float)
        return crossings / self.periods_per_year

    def percentile_years(self, percentiles: Iterable[float]) -> np.ndarray:
        years = self.years_to_target()
        percentiles = list(percentiles)
        if years.size == 0:
            return np.full(len(percentiles), np.nan)
        return np.percentile(years, percentiles)

    def mean_path(self) -> np.ndarray:
        return np.nanmean(self.paths_real, axis=0)

    def median_path(self) -> np.ndarray:
        return np.nanmedian(self.paths_real, axis=0)


def _iter_range(total: int, *, show_progress: bool) -> Iterable[int]:
    if show_progress and trange is not None:
        return trange(total, desc="Simulations")
    return range(total)


def _inflation_factor(rate: float, years: np.ndarray) -> np.ndarray:
    return np.asarray((1.0 + rate) ** years, dtype=np.float64)


def _apply_isk_tax(returns: np.ndarray, isk_tax: float, periods_per_year: float) -> np.ndarray:
    return returns - isk_tax / periods_per_year


def _run_simulation(
    config: SimulationConfig,
    model_config: ReturnModelConfig,
    granularity: Granularity,
    *,
    rng: Generator,
) -> SimulationResult:
    if granularity is Granularity.MONTHLY:
        periods_per_year = 12
        periods = config.years * periods_per_year
        years_axis = np.arange(periods, dtype=float) / periods_per_year
        deposit_years = years_axis
        draw_returns = build_return_sampler(
            model_config,
            Granularity.MONTHLY,
            trading_days_per_year=config.trading_days_per_year,
            rng=rng,
        )
    else:
        periods_per_year = float(config.trading_days_per_year)
        periods = int(config.years * periods_per_year)
        years_axis = np.arange(periods, dtype=float) / periods_per_year
        deposit_years = np.floor(years_axis * 12.0) / 12.0
        draw_returns = build_return_sampler(
            model_config,
            Granularity.DAILY,
            trading_days_per_year=config.trading_days_per_year,
            rng=rng,
        )

    paths_real = np.zeros((config.sims, periods), dtype=float)
    crossing_index = np.full(config.sims, -1, dtype=int)

    inflation_years = years_axis if granularity is Granularity.MONTHLY else years_axis
    inflation_factor = _inflation_factor(config.inflation, inflation_years)

    progress_iter = _iter_range(config.sims, show_progress=config.progress)
    for sim_idx in progress_iter:
        returns = draw_returns(periods)
        returns = _apply_isk_tax(returns, config.isk_tax, periods_per_year)
        balance_nominal = config.start_balance
        crossed = False
        for period in range(periods):
            balance_nominal *= 1.0 + returns[period]
            if granularity is Granularity.MONTHLY:
                deposit_year = deposit_years[period]
                deposit = config.monthly_deposit * (1.0 + config.inflation) ** deposit_year
                balance_nominal += deposit
            else:
                is_month_end = (period == periods - 1) or (
                    deposit_years[period + 1] > deposit_years[period]
                )
                if is_month_end:
                    deposit_year = deposit_years[period]
                    deposit = config.monthly_deposit * (1.0 + config.inflation) ** deposit_year
                    balance_nominal += deposit

            real_balance = balance_nominal / inflation_factor[period]
            paths_real[sim_idx, period] = real_balance

            if not crossed and real_balance >= config.target_real:
                crossing_index[sim_idx] = period
                crossed = True

    return SimulationResult(
        paths_real=paths_real,
        crossing_index=crossing_index,
        years_axis=years_axis,
        periods_per_year=periods_per_year,
        target=config.target_real,
    )


def run_simulation(
    sim_config: SimulationConfig,
    model_config: ReturnModelConfig,
) -> SimulationResult:
    """Run the requested simulation and return the full result set."""

    rng = default_rng(sim_config.seed)
    return _run_simulation(sim_config, model_config, sim_config.granularity, rng=rng)


def run_monthly_simulation(
    sim_config: SimulationConfig,
    model_config: ReturnModelConfig,
) -> SimulationResult:
    config = replace(sim_config, granularity=Granularity.MONTHLY)
    rng = default_rng(config.seed)
    return _run_simulation(config, model_config, Granularity.MONTHLY, rng=rng)


def run_daily_simulation(
    sim_config: SimulationConfig,
    model_config: ReturnModelConfig,
) -> SimulationResult:
    config = replace(sim_config, granularity=Granularity.DAILY)
    rng = default_rng(config.seed)
    return _run_simulation(config, model_config, Granularity.DAILY, rng=rng)


def find_cross_index(series: np.ndarray, threshold: float) -> int:
    indices = np.nonzero(series >= threshold)[0]
    if indices.size == 0:
        return -1
    return int(indices[0])
