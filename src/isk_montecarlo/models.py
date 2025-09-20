"""Return models used in the ISK Monte Carlo simulator."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.random import Generator

from .config import Granularity, ReturnModelConfig, ReturnModelType


def _scale_mean(mean: float, periods: int) -> float:
    return mean / periods


def _scale_stdev(stdev: float, periods: int) -> float:
    return stdev / np.sqrt(periods)


def _sample_student_t(df: float, size: int, rng: Generator) -> np.ndarray:
    z = rng.normal(0.0, 1.0, size)
    u = rng.chisquare(df, size)
    return z / np.sqrt(u / df)


def _calibrate_split_t(
    df: float, asym: float, rng: Generator, *, size: int = 200_000
) -> Callable[[int], np.ndarray]:
    baseline = _sample_student_t(df, size, rng)
    stretched = np.where(baseline < 0, asym * baseline, baseline / asym)
    mean = stretched.mean()
    std = stretched.std(ddof=1)

    def draw(n: int) -> np.ndarray:
        sample = _sample_student_t(df, n, rng)
        stretched_sample = np.where(sample < 0, asym * sample, sample / asym)
        return (stretched_sample - mean) / std

    return draw


def build_return_sampler(
    config: ReturnModelConfig,
    granularity: Granularity,
    *,
    trading_days_per_year: int,
    rng: Generator,
) -> Callable[[int], np.ndarray]:
    """Return a callable that generates arithmetic returns for the requested granularity."""

    if granularity is Granularity.MONTHLY:
        periods = 12
        df = config.monthly_df or config.df
        asym = config.monthly_asym or config.asym
    else:
        periods = trading_days_per_year
        df = config.daily_df or config.df
        asym = config.daily_asym or config.asym

    mean = _scale_mean(config.arith_mean_annual, periods)
    stdev = _scale_stdev(config.stdev_annual, periods)

    if config.model is ReturnModelType.NORMAL:
        return lambda n: rng.normal(mean, stdev, n)

    if config.model is ReturnModelType.SPLIT_T:
        standardized = _calibrate_split_t(df=df, asym=asym, rng=rng)
        return lambda n: mean + stdev * standardized(n)

    raise ValueError(f"Unsupported return model: {config.model}")


def generate_equity_cagr(
    config: ReturnModelConfig,
    *,
    years: int,
    sims: int,
    trading_days_per_year: int,
    rng: Generator,
) -> np.ndarray:
    """Simulate equity-only CAGRs using the configured return model."""

    draw_daily = build_return_sampler(
        config,
        Granularity.DAILY,
        trading_days_per_year=trading_days_per_year,
        rng=rng,
    )
    days = years * trading_days_per_year
    cagr = np.empty(sims)
    for idx in range(sims):
        returns = draw_daily(days)
        growth = float(np.prod(1.0 + returns))
        cagr[idx] = growth ** (1.0 / years) - 1.0
    return cagr
