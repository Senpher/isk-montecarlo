# isk-montecarlo

Monte Carlo simulator for Swedish FIRE with **ISK** accounts:

- Inflation-indexed monthly deposits
- ISK tax drag
- Normal or **fat-tailed, left-skewed** (split-t) return models
- Daily or monthly granularity
- Reproducible charts (paths, histograms, time-to-target)

## Quick start

```bash
pip install -r requirements.txt
python examples/run_monthly.py  # or examples/run_daily.py
```

The repo ships with an optional `environment.yml` (conda) and `.devcontainer/` setup for GitHub Codespaces. Both install the same dependencies as `requirements.txt`.

## Project structure

```
isk-montecarlo/
├─ src/isk_montecarlo/
│  ├─ config.py          # dataclasses for parameters & defaults
│  ├─ models.py          # normal & split-t return generators
│  ├─ simulate.py        # monthly/daily engines with ISK/inflation logic
│  ├─ plots.py           # histograms, wealth-path charts, CAGR plots
│  └─ tax.py             # Swedish tax & pension contribution helpers
├─ examples/
│  ├─ run_monthly.py     # argparse CLI for monthly Monte Carlo
│  ├─ run_daily.py       # argparse CLI for daily Monte Carlo + CAGR
│  └─ plot_tax.py        # progressive tax & pension contribution charts
├─ notebooks/            # optional exploratory notebooks
├─ requirements.txt
├─ environment.yml
├─ Makefile
└─ .devcontainer/devcontainer.json
```

## Key CLI parameters

### Core financial

- `--start-balance` (e.g., `1_100_000`)
- `--monthly-deposit` (today’s SEK; indexed to inflation internally)
- `--target-real` (e.g., `15_000_000`)
- `--inflation` (default `0.02`)
- `--isk-tax` (default `0.009`)

### Return model

- `--model` = `normal` | `split_t`
- Arithmetic mean: `--arith-mean-annual` (default `0.1164`)
- Volatility: `--stdev-annual` (default `0.1949`)
- Split-t extras: `--df`, `--asym` plus per-frequency overrides (`--df-monthly`, `--asym-monthly`, `--df-daily`, `--asym-daily`)

### Simulation

- `--granularity` is implied by the script (`run_monthly.py` or `run_daily.py`)
- `--trading-days-per-year` (daily script, default `250`)
- `--years` (default `40`)
- `--sims` (default `10_000`)
- `--seed` (reproducibility)
- `--progress/--no-progress`
- `--truncate/--no-truncate` (plotting only)

### Plotting & outputs

- `--save-plots` (folder path)
- `--show/--no-show` (render to screen)
- `--bins` (histogram bins)
- `--subset` (number of paths to draw, e.g., `200`)

### Tax utilities

Run `python examples/plot_tax.py` for progressive tax, marginal/average rate, and pension contribution charts across a salary range. Use `--max-salary` and `--points` to change the grid.

## Sensible defaults

- Annual arithmetic mean: **0.1164**
- Annual stdev: **0.1949**
- Inflation: **0.02**
- ISK tax: **0.009**
- Split-t default: `df=5`, `asym=1.3`
- Horizon: **40 years**
- Sims: **10,000**
- Seed: **123**

## Development helpers

- `make setup` – install dependencies via pip
- `make run-monthly` – run the monthly CLI with defaults
- `make run-daily` – run the daily CLI with defaults

All Python code is formatted with `ruff format` and linted with `ruff check`. Run both commands before committing changes.
