"""Plotting helpers for the ISK Monte Carlo simulator."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter
from numpy.random import Generator, default_rng
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from .config import PlotConfig
from .simulate import SimulationResult, find_cross_index


def _get_axes(ax: Optional[Axes]) -> tuple[Figure, Axes]:
    if ax is None:
        fig, axis = plt.subplots()
    else:
        axis = ax
        fig = ax.figure
    return fig, axis


def plot_return_distribution(
    returns: np.ndarray,
    *,
    bins: int = 180,
    xlim: tuple[float, float] | None = None,
    normalize_max: bool = True,
    show_mean: bool = True,
    inset: bool = True,
    inset_xlim: tuple[float, float] | None = None,
):
    """
    Histogram of per-period returns (e.g., daily). Draws a small inset but *no connector lines*.

    Args:
        returns: arithmetic returns per period (after ISK drag if you passed those in).
        bins: number of histogram bins.
        xlim: zoom for main axes.
        normalize_max: normalize bar heights to 0–1 (else density).
        show_mean: vertical dashed line at sample mean.
        inset: show a small inset of the central region.
        inset_xlim: x-limits for the inset (e.g. (-0.015, 0.015)).
    """
    hist_counts, bin_edges = np.histogram(returns, bins=bins)
    if normalize_max:
        heights = hist_counts / (hist_counts.max() if hist_counts.max() else 1)
        y_label = "Normalized frequency"
        y_lim = (0, 1)
        title = "Implied Return Distribution — Normalized height (max = 1)"
    else:
        widths = np.diff(bin_edges)
        heights = hist_counts / (hist_counts.sum() * widths.mean())
        y_label = "Density"
        y_lim = None
        title = "Implied Return Distribution — Density"

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    width = (bin_edges[1] - bin_edges[0]) * 0.98

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(centers, heights, width=width, edgecolor="black", alpha=0.8)
    if show_mean:
        m = returns.mean()
        ax.axvline(m, linestyle="--", label=f"Mean ~{m:.2%}")

    ax.set_title(title)
    ax.set_xlabel("Return per period")
    ax.set_ylabel(y_label)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if show_mean:
        ax.legend()
    ax.grid(alpha=0.3)

    # ----- inset WITHOUT connector lines -----
    if inset:
        axins = inset_axes(ax, width="38%", height="38%", loc="upper left", borderpad=1.0)
        axins.bar(centers, heights, width=width, edgecolor="black", alpha=0.8)
        if show_mean:
            axins.axvline(m, linestyle="--")
        axins.xaxis.set_major_formatter(PercentFormatter(1.0))
        axins.set_ylim(ax.get_ylim())
        if inset_xlim is not None:
            axins.set_xlim(*inset_xlim)
        else:
            # sensible default: zoom around the mode/mean area
            axins.set_xlim(-0.015, 0.015)
        axins.grid(alpha=0.2)
        # no mark_inset/ConnectionPatch → no diagonal lines

    return fig


def plot_paths(
    result: SimulationResult,
    plot_config: PlotConfig,
    *,
    rng: Optional[Generator] = None,
    title: str,
    ylabel: str = "Portfolio value (real SEK)",
) -> Figure:
    """Plot a subset of simulated wealth paths together with the mean/median."""

    fig, axis = plt.subplots(figsize=(9, 6))
    rng = default_rng() if rng is None else rng

    subset = min(plot_config.subset_paths, result.paths_real.shape[0])
    indices = rng.choice(result.paths_real.shape[0], subset, replace=False)
    for idx in indices:
        cross = result.crossing_index[idx]
        if plot_config.truncate_paths and cross >= 0:
            axis.plot(
                result.years_axis[: cross + 1],
                result.paths_real[idx, : cross + 1],
                linewidth=0.7,
                alpha=0.2,
            )
        else:
            axis.plot(result.years_axis, result.paths_real[idx, :], linewidth=0.7, alpha=0.2)

    median_path = result.median_path()
    mean_path = result.mean_path()
    median_cross = find_cross_index(median_path, result.target)
    mean_cross = find_cross_index(mean_path, result.target)

    def _plot_line(path: np.ndarray, cross: int, **kwargs: float) -> None:
        if plot_config.truncate_paths and cross >= 0:
            axis.plot(result.years_axis[: cross + 1], path[: cross + 1], **kwargs)
        else:
            axis.plot(result.years_axis, path, **kwargs)

    _plot_line(median_path, median_cross, linewidth=2.2, linestyle="--", label="Median path (real)")
    _plot_line(mean_path, mean_cross, linewidth=2.2, linestyle="-", label="Mean path (real)")

    axis.axhline(result.target, linestyle=":", label=f"Target: {result.target:,.0f} SEK (real)")
    axis.set_title(title)
    axis.set_xlabel("Years from today")
    axis.set_ylabel(ylabel)
    axis.legend()
    axis.grid(alpha=0.3)
    return fig


def plot_years_to_target(
    result: SimulationResult,
    plot_config: PlotConfig,
    *,
    ax: Optional[Axes] = None,
) -> Figure:
    """Plot the distribution of years required to hit the wealth target."""

    years = result.years_to_target()
    fig, axis = _get_axes(ax)
    if years.size == 0:
        axis.text(
            0.5, 0.5, "Target not reached", transform=axis.transAxes, ha="center", va="center"
        )
        axis.set_axis_off()
        return fig

    percentiles = result.percentile_years((10, 50, 90))
    axis.hist(years, bins=plot_config.bins, edgecolor="black", alpha=0.7, density=True)
    axis.axvline(percentiles[0], linestyle="--", label=f"10th %ile ~{percentiles[0]:.1f} yrs")
    axis.axvline(percentiles[1], linestyle="--", label=f"Median ~{percentiles[1]:.1f} yrs")
    axis.axvline(percentiles[2], linestyle="--", label=f"90th %ile ~{percentiles[2]:.1f} yrs")
    axis.set_title("Distribution of years to hit the target (real SEK)")
    axis.set_xlabel("Years to reach target")
    axis.set_ylabel("Probability density")
    axis.grid(alpha=0.3)
    axis.legend()
    return fig


def plot_cagr_distribution(
    cagrs: np.ndarray, *, bins: int = 60, ax: Optional[Axes] = None
) -> Figure:
    """Plot the simulated equity CAGR distribution."""

    fig, axis = _get_axes(ax)
    axis.hist(cagrs, bins=bins, edgecolor="black", alpha=0.7)
    median = float(np.median(cagrs))
    p10, p90 = np.percentile(cagrs, [10, 90])
    axis.axvline(median, linestyle="--", label=f"Median CAGR ~{median:.2%}")
    axis.axvline(p10, linestyle=":", label=f"10th %ile ~{p10:.2%}")
    axis.axvline(p90, linestyle=":", label=f"90th %ile ~{p90:.2%}")
    axis.set_title("Equity CAGR distribution")
    axis.set_xlabel("Annualized return")
    axis.set_ylabel("Frequency")
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.grid(alpha=0.3)
    axis.legend()
    return fig


def save_figure(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def maybe_show(figures: Iterable[Figure], *, show: bool) -> None:
    if show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)
