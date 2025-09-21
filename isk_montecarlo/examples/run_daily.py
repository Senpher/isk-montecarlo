"""Run daily ISK Monte Carlo simulations and equity CAGR experiments (parallel capable)
   + show the ACTUAL daily return distribution used by the simulator (after ISK tax)."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import List

import numpy as np
from numpy.random import default_rng

from isk_montecarlo.config import (
    Granularity,
    PlotConfig,
    ReturnModelConfig,
    ReturnModelType,
    SimulationConfig,
)
from isk_montecarlo.models import build_return_sampler, generate_equity_cagr
from isk_montecarlo.plots import (
    maybe_show,
    plot_cagr_distribution,
    plot_paths,
    plot_years_to_target,
    plot_return_distribution,
    save_figure,
)
from isk_montecarlo.simulate import SimulationResult, run_daily_simulation


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
    # target & savings
    parser.add_argument("--start-balance", type=float, default=1_100_000.0)
    parser.add_argument("--monthly-deposit", type=float, default=25_000.0)
    parser.add_argument("--target-real", type=float, default=15_000_000.0)
    # macro / account
    parser.add_argument("--inflation", type=float, default=0.02)
    parser.add_argument("--isk-tax", type=float, default=0.009)
    # horizon / sims
    parser.add_argument("--years", type=int, default=40)
    parser.add_argument("--sims", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--trading-days-per-year", type=int, default=250)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)

    # return model params
    parser.add_argument("--model", choices=[m.value for m in ReturnModelType], default=ReturnModelType.SPLIT_T.value)
    parser.add_argument("--arith-mean-annual", type=float, default=0.1164)
    parser.add_argument("--stdev-annual", type=float, default=0.1949)
    parser.add_argument("--df", type=positive_float, default=5.0)
    parser.add_argument("--asym", type=positive_float, default=1.2)
    parser.add_argument("--df-daily", type=positive_float, default=None)
    parser.add_argument("--asym-daily", type=positive_float, default=None)

    # plots / extras
    parser.add_argument("--cagr-sims", type=int, default=12_000)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--no-truncate", dest="truncate", action="store_false")
    parser.set_defaults(truncate=True)
    parser.add_argument("--save-plots", type=Path)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)

    # Daily distribution plot controls
    parser.add_argument("--dist-xlim", type=float, nargs=2, metavar=("XMIN", "XMAX"),
                        help="x-axis zoom for daily distribution plot (e.g. -0.03 0.03 for ±3%)")
    parser.add_argument("--dist-bins", type=int, default=180,
                        help="bins for daily distribution histogram")

    # Show the *actual daily* sampling distribution (after-tax)
    parser.add_argument("--distribution-sample", type=int, default=60_000,
                        help="number of daily draws to visualize from the exact sampler the sim uses (after ISK tax)")

    # parallelism control
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (processes)")

    return parser.parse_args()


# ---------------------------
# Parallel helpers
# ---------------------------

def _run_daily_chunk(
    sim_config: SimulationConfig, model_config: ReturnModelConfig, sims: int, seed: int
) -> SimulationResult:
    """Run a daily simulation chunk with given sims and seed."""
    sub = replace(sim_config, sims=sims, seed=seed, granularity=Granularity.DAILY)
    return run_daily_simulation(sub, model_config)


def _generate_cagr_chunk(
    model_config: ReturnModelConfig,
    years: int,
    sims: int,
    trading_days_per_year: int,
    seed: int,
) -> np.ndarray:
    # Always pass rng—our generate_equity_cagr requires it.
    rng = default_rng(seed)
    return generate_equity_cagr(
        model_config,
        years=years,
        sims=sims,
        trading_days_per_year=trading_days_per_year,
        rng=rng,
    )


def _split_work(total: int, workers: int) -> List[int]:
    """Split `total` simulations into `workers` nearly equal parts."""
    w = max(1, workers)
    base = total // w
    rem = total % w
    parts = [base + (1 if i < rem else 0) for i in range(w)]
    return [p for p in parts if p > 0]


def _merge_results(parts: List[SimulationResult]) -> SimulationResult:
    """Concatenate partial SimulationResult objects along simulation axis."""
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


def _parallel_daily(
    sim_config: SimulationConfig,
    model_config: ReturnModelConfig,
    workers: int,
) -> SimulationResult:
    """Parallel daily simulation by chunking sims across processes."""
    if workers <= 1 or sim_config.sims <= 1:
        return run_daily_simulation(sim_config, model_config)

    chunks = _split_work(sim_config.sims, workers)
    base_seed = sim_config.seed if sim_config.seed is not None else 123
    seeds = [base_seed + 100_003 * i for i in range(len(chunks))]

    results: List[SimulationResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_run_daily_chunk, sim_config, model_config, chunk, seed)
            for chunk, seed in zip(chunks, seeds)
        ]
        for fut in as_completed(futures):
            results.append(fut.result())

    return _merge_results(results)


def _parallel_cagr(
    model_config: ReturnModelConfig,
    years: int,
    sims: int,
    trading_days_per_year: int,
    workers: int,
    base_seed: int | None,
) -> np.ndarray:
    """Parallel equity CAGR sampling and concatenation."""
    if workers <= 1 or sims <= 1:
        return _generate_cagr_chunk(model_config, years, sims, trading_days_per_year, base_seed or 123)

    chunks = _split_work(sims, workers)
    seed0 = base_seed if base_seed is not None else 123
    seeds = [seed0 + 200_003 * i for i in range(len(chunks))]

    parts: List[np.ndarray] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_generate_cagr_chunk, model_config, years, chunk, trading_days_per_year, seed)
            for chunk, seed in zip(chunks, seeds)
        ]
        for fut in as_completed(futures):
            parts.append(fut.result())

    return np.concatenate(parts, axis=0)


# ---------------------------
# Daily return distribution sampler (AFTER ISK tax)
# ---------------------------

def sample_daily_returns(
    config: SimulationConfig,
    model_config: ReturnModelConfig,
    *,
    sample_size: int,
) -> np.ndarray:
    """Sample daily arithmetic returns from the exact sampler used in the sim, minus daily ISK-tax drag."""
    rng = default_rng((config.seed or 0) + 3)
    sampler = build_return_sampler(
        model_config,
        Granularity.DAILY,
        trading_days_per_year=config.trading_days_per_year,
        rng=rng,
    )
    r = sampler(sample_size)
    # apply the same daily ISK-tax drag as in the sim
    return r - config.isk_tax / config.trading_days_per_year


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
        daily_df=args.df_daily,
        daily_asym=args.asym_daily,
    )

    sim_config = SimulationConfig(
        start_balance=args.start_balance,
        monthly_deposit=args.monthly_deposit,
        target_real=args.target_real,
        inflation=args.inflation,
        isk_tax=args.isk_tax,
        years=args.years,
        sims=args.sims,
        granularity=Granularity.DAILY,
        trading_days_per_year=args.trading_days_per_year,
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
# Main
# ---------------------------

def main() -> None:
    args = parse_args()
    sim_config, model_config, plot_config = build_configs(args)

    # 1) Daily distribution (after ISK tax) — this is the *actual* sampling distribution used
    daily_draws = sample_daily_returns(
        sim_config,
        model_config,
        sample_size=args.distribution_sample,
    )

    figures: List = []
    # Use --dist-bins for the distribution; try to pass --dist-xlim if the plotting function supports it
    try:
        fig_dist = plot_return_distribution(
            daily_draws,
            bins=args.dist_bins,
            xlim=tuple(args.dist_xlim) if args.dist_xlim else None,  # works if your helper supports xlim
        )
    except TypeError:
        # fallback if your plot_return_distribution doesn't accept xlim
        fig_dist = plot_return_distribution(daily_draws, bins=args.dist_bins)
    figures.append(("daily_distribution", fig_dist))

    # 2) Daily simulation (parallel)
    result = _parallel_daily(sim_config, model_config, workers=max(1, args.workers))

    fig_paths = plot_paths(
        result,
        plot_config,
        rng=default_rng(sim_config.seed + 1 if sim_config.seed is not None else None),
        title="Daily Monte Carlo (real SEK) — split-t or normal returns",
    )
    figures.append(("daily_paths", fig_paths))
    fig_hist = plot_years_to_target(result, plot_config)
    figures.append(("daily_years_to_target", fig_hist))

    # 3) Equity-only CAGR (parallel)
    cagrs = _parallel_cagr(
        model_config,
        years=sim_config.years,
        sims=args.cagr_sims,
        trading_days_per_year=sim_config.trading_days_per_year,
        workers=max(1, args.workers),
        base_seed=(sim_config.seed + 2) if sim_config.seed is not None else None,
    )
    fig_cagr = plot_cagr_distribution(cagrs, bins=plot_config.bins)
    figures.append(("daily_cagr", fig_cagr))

    # Save/Show
    if plot_config.save_plots:
        for name, fig in figures:
            path = plot_config.save_plots / f"{name}.png"
            save_figure(fig, path)

    maybe_show([fig for _, fig in figures], show=plot_config.show)


if __name__ == "__main__":
    main()
