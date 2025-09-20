"""Run daily ISK Monte Carlo simulations and equity CAGR experiments."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from numpy.random import default_rng

from isk_montecarlo.config import (
    Granularity,
    PlotConfig,
    ReturnModelConfig,
    ReturnModelType,
    SimulationConfig,
)
from isk_montecarlo.models import generate_equity_cagr
from isk_montecarlo.plots import (
    maybe_show,
    plot_cagr_distribution,
    plot_paths,
    plot_years_to_target,
    save_figure,
)
from isk_montecarlo.simulate import run_daily_simulation


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
    parser.add_argument("--trading-days-per-year", type=int, default=250)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument(
        "--model", choices=[m.value for m in ReturnModelType], default=ReturnModelType.SPLIT_T.value
    )
    parser.add_argument("--arith-mean-annual", type=float, default=0.1164)
    parser.add_argument("--stdev-annual", type=float, default=0.1949)
    parser.add_argument("--df", type=positive_float, default=5.0)
    parser.add_argument("--asym", type=positive_float, default=1.2)
    parser.add_argument("--df-daily", type=positive_float, default=None)
    parser.add_argument("--asym-daily", type=positive_float, default=None)

    parser.add_argument("--cagr-sims", type=int, default=12_000)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--no-truncate", dest="truncate", action="store_false")
    parser.set_defaults(truncate=True)
    parser.add_argument("--save-plots", type=Path)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    sim_config, model_config, plot_config = build_configs(args)
    result = run_daily_simulation(sim_config, model_config)

    figures: List = []
    fig_paths = plot_paths(
        result,
        plot_config,
        rng=default_rng(sim_config.seed + 1 if sim_config.seed is not None else None),
        title="Daily Monte Carlo (real SEK) — split-t or normal returns",
    )
    figures.append(("daily_paths", fig_paths))
    fig_hist = plot_years_to_target(result, plot_config)
    figures.append(("daily_years_to_target", fig_hist))

    cagr_rng = default_rng(sim_config.seed + 2 if sim_config.seed is not None else None)
    cagrs = generate_equity_cagr(
        model_config,
        years=sim_config.years,
        sims=args.cagr_sims,
        trading_days_per_year=sim_config.trading_days_per_year,
        rng=cagr_rng,
    )
    fig_cagr = plot_cagr_distribution(cagrs, bins=plot_config.bins)
    figures.append(("daily_cagr", fig_cagr))

    if plot_config.save_plots:
        for name, fig in figures:
            path = plot_config.save_plots / f"{name}.png"
            save_figure(fig, path)

    maybe_show([fig for _, fig in figures], show=plot_config.show)


if __name__ == "__main__":
    main()
