"""Visualize Swedish tax, pension, and BTP1 contributions across salary levels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import _paths  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from isk_montecarlo.plots import maybe_show, save_figure
from isk_montecarlo.tax import TaxParameters, TaxProfiles, compute_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-salary", type=float, default=250_000.0, help="Maximum monthly salary to plot"
    )
    parser.add_argument(
        "--points", type=int, default=2_000, help="Number of salary points to evaluate"
    )
    parser.add_argument("--save-plots", type=Path)
    parser.add_argument("--show", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def plot_income_and_pension(profiles: TaxProfiles) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(
        profiles.gross_monthly,
        profiles.net_monthly,
        label="After-tax income (Stockholm 2025 model)",
    )
    ax.plot(
        profiles.gross_monthly,
        profiles.public_pension,
        label="Public pension contribution (18.5% of PGI)",
    )
    ax.plot(profiles.gross_monthly, profiles.btp1_low, label="BTP1: 6.5% part (≤ 7.5 IBB)")
    ax.plot(profiles.gross_monthly, profiles.btp1_high, label="BTP1: 32% part (7.5–30 IBB)")
    ax.plot(profiles.gross_monthly, profiles.btp1_total, label="BTP1: total contribution")

    ymax = ax.get_ylim()[1]
    for label, xval in profiles.monthly_breakpoints().items():
        ax.axvline(x=xval, linestyle="--")
        ax.text(xval, ymax * 0.05, label, rotation=90, va="bottom", ha="right")

    ax.set_title("Sweden 2025 (Stockholm): Net income & pension vs monthly salary")
    ax.set_xlabel("Monthly gross salary (SEK)")
    ax.set_ylabel("SEK per month")
    ax.grid(True, linestyle=":")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_tax_rates(profiles: TaxProfiles) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(profiles.gross_monthly, profiles.average_rate, label="Average tax rate")
    ax.plot(profiles.gross_monthly, profiles.marginal_rate, label="Marginal tax rate")

    ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1.0
    for label, xval in profiles.monthly_breakpoints().items():
        ax.axvline(x=xval, linestyle="--")
        ax.text(xval, ymax * 0.05, label, rotation=90, va="bottom", ha="right")

    ax.set_title("Sweden 2025 (Stockholm): Average & marginal tax rates")
    ax.set_xlabel("Monthly gross salary (SEK)")
    ax.set_ylabel("Tax rate")
    ax.grid(True, linestyle=":")
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    params = TaxParameters()
    gross = np.linspace(0.0, args.max_salary, args.points)
    profiles = compute_profiles(gross, params=params)

    figures: List = []
    fig_income = plot_income_and_pension(profiles)
    figures.append(("income_pension", fig_income))
    fig_rates = plot_tax_rates(profiles)
    figures.append(("tax_rates", fig_rates))

    if args.save_plots:
        for name, fig in figures:
            path = args.save_plots / f"{name}.png"
            save_figure(fig, path)

    maybe_show([fig for _, fig in figures], show=args.show)


if __name__ == "__main__":
    main()
