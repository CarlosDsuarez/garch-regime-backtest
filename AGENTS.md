# AGENTS.md

## Cursor Cloud specific instructions

This is a single-entry-point Python quant backtest (GARCH regime + QQQ SMA crossover). There is no web server, database, or frontend — the "app" is a CLI pipeline that downloads data, fits a model, runs a backtest, and writes charts/CSVs to `outputs/`.

### Environment
- Python 3.12 with a virtualenv at `.venv/` (created by the startup update script). Run everything through it, e.g. `.venv/bin/python main.py`, or activate with `source .venv/bin/activate`.
- The system package `python3.12-venv` is required for `python3 -m venv` and is preinstalled in the VM snapshot.
- Dependencies are pinned/ranged in `requirements.txt` (arch, yfinance, pandas, numpy, matplotlib, seaborn, scipy, statsmodels).

### Run
- Run the full pipeline: `.venv/bin/python main.py`. It prints a per-regime performance report and writes `outputs/{equity_curves,regime_performance,sma_signals,monthly_heatmap}.png` plus `outputs/{regime_metrics,backtest_full}.csv`.

### Data source (non-obvious)
- The loader first looks for a Project-1 CSV at `~/Desktop/garch-nasdaq-anomaly/data/qqq_returns.csv`. That path does not exist in the cloud VM, so it always falls back to downloading QQQ from **yfinance** (2010→today). This means `main.py` requires outbound network access.
- Plain HTTP requests to Yahoo endpoints can return HTTP 429, but yfinance uses `curl_cffi` browser impersonation and downloads succeed in practice. If a run fails on data download, it is almost always transient Yahoo rate-limiting — just re-run.
- Because data is fetched live, the exact numbers and the committed files in `outputs/` change on every run. Treat `outputs/` regeneration as expected noise; do not commit refreshed `outputs/` unless that is the intent.

### Lint / test / build
- There is no configured linter and no automated test suite. Use `.venv/bin/python -m py_compile main.py garch_loader.py strategy.py backtest.py metrics.py visualizations.py` as a quick compile check.
