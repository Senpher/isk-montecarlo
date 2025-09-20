"""Run monthly ISK Monte Carlo simulations with configurable parameters (parallel capable)."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, List

import numpy as np
from numpy.random import default_rng

from isk_montecarlo.config import (
    Granularity,
    PlotConfig,
    ReturnModelConfig,
    ReturnModelType,
    SimulationConfig,
)
from isk_montecarlo.models import build_return_sampler
from isk_montecarlo.plots import (
    maybe_show,
    plot_paths,
    plot_return_distribution,
    plot_years_to_target,
    save_figure,
)
from isk_montecarlo.simulate import SimulationResult, run_monthly_simulation

if TYPE_CHECKING:
    import numpy as np


# ---------------------------
# CLI
# ---------------------------


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-balance", type=float, default=1_100_000.0)
    parser.add_argument("--monthly-deposit", type=float, default=25_000.0)
    parser.add_argument("--target-real", type=float, default=15_000_000.0)
    parser.add_argument("--inflation", type=float, default=0.02)
    parser.add_argument("--isk-tax", type=float, default=0.009)
    parser.add_argument("--years", type=int, default=40)
    parser.add_argument("--sims", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--model", choices=[m.value for m in ReturnModelType], default=ReturnModelType.SPLIT_T.value
    )
    parser.add_argument("--arith-mean-annual", type=float, default=0.1164)
    parser.add_argument("--stdev-annual", type=float, default=0.1949)
    parser.add_argument(
        "--df", type=positive_float, default=5.0, help="Degrees of freedom for split-t"
    )
    parser.add_argument(
        "--asym", type=positive_float, default=1.3, help="Asymmetry (>1 => heavier left tail)"
    )
    parser.add_argument("--df-monthly", type=positive_float, default=None)
    parser.add_argument("--asym-monthly", type=positive_float, default=None)

    parser.add_argument("--distribution-sample", type=int, default=60_000)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--no-truncate", dest="truncate", action="store_false")
    parser.set_defaults(truncate=True)
    parser.add_argument("--save-plots", type=Path)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)

    # NEW: parallelism
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of parallel workers (processes)"
    )

    return parser.parse_args()


# ---------------------------
# Config builders
# ---------------------------


def build_configs(
    args: argparse.Namespace,
) -> tuple[SimulationConfig, ReturnModelConfig, PlotConfig]:
    model_type = ReturnModelType(args.model)
    return_config = ReturnModelConfig(
        model=model_type,
        arith_mean_annual=args.arith_mean_annual,
        stdev_annual=args.stdev_annual,
        df=args.df,
        asym=args.asym,
        monthly_df=args.df_monthly,
        monthly_asym=args.asym_monthly,
    )

    sim_config = SimulationConfig(
        start_balance=args.start_balance,
        monthly_deposit=args.monthly_deposit,
        target_real=args.target_real,
        inflation=args.inflation,
        isk_tax=args.isk_tax,
        years=args.years,
        sims=args.sims,
        granularity=Granularity.MONTHLY,
        trading_days_per_year=250,  # retained for consistency with daily config usage
        seed=args.seed,
        progress=args.progress,
    )

    plot_config = PlotConfig(
        bins=args.bins,
        subset_paths=args.subset,
        truncate_paths=args.truncate,
        save_plots=args.save_plots,
        show=args.show,
    )
    return sim_config, return_config, plot_config


# ---------------------------
# Return sampling for the distribution chart
# ---------------------------


def sample_monthly_returns(
    config: SimulationConfig,
    model_config: ReturnModelConfig,
    *,
    sample_size: int,
) -> "np.ndarray":
    rng_seed = None if config.seed is None else config.seed + 1
    rng = default_rng(rng_seed)
    sampler = build_return_sampler(
        model_config,
        Granularity.MONTHLY,
        trading_days_per_year=config.trading_days_per_year,
        rng=rng,
    )
    returns = sampler(sample_size)
    # Apply ISK tax drag monthly for the displayed distribution (consistent with sim)
    return returns - config.isk_tax / 12.0


# ---------------------------
# Parallel helpers (Windows-friendly: top-level for pickling)
# ---------------------------


def _split_work(total: int, workers: int) -> List[int]:
    w = max(1, workers)
    base = total // w
    rem = total % w
    parts = [base + (1 if i < rem else 0) for i in range(w)]
    return [p for p in parts if p > 0]


def _run_monthly_chunk(
    sim_config: SimulationConfig,
    model_config: ReturnModelConfig,
    sims: int,
    seed: int,
) -> SimulationResult:
    sub = replace(sim_config, sims=sims, seed=seed, granularity=Granularity.MONTHLY)
    return run_monthly_simulation(sub, model_config)


def _merge_results(parts: List[SimulationResult]) -> SimulationResult:
    if not parts:
        raise ValueError("No partial results to merge")
    first = parts[0]
    for p in parts[1:]:
        if not np.array_equal(p.years_axis, first.years_axis):
            raise ValueError("years_axis mismatch between chunks")
        if p.periods_per_year != first.periods_per_year:
            raise ValueError("periods_per_year mismatch between chunks")
        if p.target != first.target:
            raise ValueError("target mismatch between chunks")
    paths = np.concatenate([p.paths_real for p in parts], axis=0)
    crossing = np.concatenate([p.crossing_index for p in parts], axis=0)
    return SimulationResult(
        paths_real=paths,
        crossing_index=crossing,
        years_axis=first.years_axis,
        periods_per_year=first.periods_per_year,
        target=first.target,
    )


def _parallel_monthly(
    sim_config: SimulationConfig,
    model_config: ReturnModelConfig,
    workers: int,
) -> SimulationResult:
    if workers <= 1 or sim_config.sims <= 1:
        return run_monthly_simulation(sim_config, model_config)

    chunks = _split_work(sim_config.sims, workers)
    base_seed = sim_config.seed if sim_config.seed is not None else 123
    seeds = [base_seed + 100_003 * i for i in range(len(chunks))]

    results: List[SimulationResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_run_monthly_chunk, sim_config, model_config, chunk, seed)
            for chunk, seed in zip(chunks, seeds)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    return _merge_results(results)


# ---------------------------
# Main
# ---------------------------


def main() -> None:
    args = parse_args()
    sim_config, model_config, plot_config = build_configs(args)

    # MONTHLY SIMULATION (parallel)
    result = _parallel_monthly(sim_config, model_config, workers=max(1, args.workers))

    # Return distribution sample (single-process is fine; it’s fast)
    monthly_returns = sample_monthly_returns(
        sim_config,
        model_config,
        sample_size=args.distribution_sample,
    )

    figures: List = []
    fig_dist = plot_return_distribution(monthly_returns, bins=args.bins)
    figures.append(("monthly_distribution", fig_dist))
    fig_paths = plot_paths(
        result,
        plot_config,
        rng=default_rng(sim_config.seed + 2 if sim_config.seed is not None else None),
        title="Monthly Monte Carlo (real SEK) — split-t or normal returns",
    )
    figures.append(("monthly_paths", fig_paths))
    fig_hist = plot_years_to_target(result, plot_config)
    figures.append(("monthly_years_to_target", fig_hist))

    if plot_config.save_plots:
        for name, fig in figures:
            path = plot_config.save_plots / f"{name}.png"
            save_figure(fig, path)

    maybe_show([fig for _, fig in figures], show=plot_config.show)


if __name__ == "__main__":
    main()
