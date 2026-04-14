"""
main.py
-------
Orquestador principal del sistema GARCH Regime Backtest — QQQ SMA Crossover.

Ejecución:
    python main.py
"""

from __future__ import annotations

import sys

# Forzar UTF-8 en stdout para soportar caracteres especiales en Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

import pandas as pd
import numpy as np

import garch_loader
import strategy
import backtest
import metrics
import visualizations

OUTPUT_DIR   = Path(__file__).parent / "outputs"
PROJECT_ROOT = Path(__file__).parent


def _fmt_pct(v: float) -> str:
    return f"{v*100:+.1f}%" if pd.notna(v) else "N/A"


def _fmt_f(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if pd.notna(v) else "N/A"


def print_report(
    backtest_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    source: str = "yfinance (fallback)",
) -> None:
    """
    Imprime el reporte de consola con métricas globales y por régimen.

    Args:
        backtest_df: DataFrame del backtest.
        metrics_df:  DataFrame de métricas por régimen.
        source:      Descripción de la fuente de datos.
    """
    total_row = metrics_df.loc["TOTAL"] if "TOTAL" in metrics_df.index else pd.Series()
    bm        = metrics.compute_benchmark_metrics(backtest_df)

    start_dt  = backtest_df.index[0].strftime("%Y-%m-%d")
    end_dt    = backtest_df.index[-1].strftime("%Y-%m-%d")
    n_days    = len(backtest_df)

    strat_ret = total_row.get("ret_anual", float("nan"))
    strat_sh  = total_row.get("sharpe",   float("nan"))
    strat_dd  = total_row.get("max_dd",   float("nan"))
    alpha     = strat_ret - bm["ret_anual"]

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     GARCH REGIME BACKTEST — QQQ SMA CROSSOVER (20/50)   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Período        : {start_dt} → {end_dt} ({n_days} días)")
    print(f"  Capital inicial: $100,000")
    print(f"  Costos         : 5 bps por operación")
    print(f"  Fuente GARCH   : {source}")
    print()

    print("  PERFORMANCE TOTAL:")
    print(f"    Estrategia   → Ret: {_fmt_pct(strat_ret)} | "
          f"Sharpe: {_fmt_f(strat_sh)} | Max DD: {_fmt_pct(strat_dd)}")
    print(f"    Buy & Hold   → Ret: {_fmt_pct(bm['ret_anual'])} | "
          f"Sharpe: {_fmt_f(bm['sharpe'])} | Max DD: {_fmt_pct(bm['max_dd'])}")
    print(f"    Alpha neto   → {_fmt_pct(alpha)} anualizado")
    print()

    regime_rows = metrics_df.drop(index="TOTAL", errors="ignore")

    print("  PERFORMANCE POR RÉGIMEN:")
    print("  ┌─────────────────┬──────┬────────┬────────┬────────┬────────┐")
    print("  │ Régimen         │ Días │ Ret An │ Sharpe │ Max DD │WinRate │")
    print("  ├─────────────────┼──────┼────────┼────────┼────────┼────────┤")

    for regime_name, row in regime_rows.iterrows():
        dias  = int(row.get("dias",      0))
        ret_a = row.get("ret_anual",     float("nan"))
        sh    = row.get("sharpe",        float("nan"))
        dd    = row.get("max_dd",        float("nan"))
        wr    = row.get("win_rate",      float("nan"))

        ret_str = f"{ret_a*100:+.1f}%" if pd.notna(ret_a) else "N/A   "
        sh_str  = f"{sh:.2f}"          if pd.notna(sh)    else "N/A  "
        dd_str  = f"{dd*100:.1f}%"     if pd.notna(dd)    else "N/A   "
        wr_str  = f"{wr*100:.0f}%"     if pd.notna(wr)    else "N/A"

        name_t = regime_name[:15].ljust(15)
        print(
            f"  │ {name_t} │ {dias:4d} │ {ret_str:>6} │ {sh_str:>6} │ "
            f"{dd_str:>6} │ {wr_str:>5}  │"
        )

    print("  └─────────────────┴──────┴────────┴────────┴────────┴────────┘")
    print()

    valid_regimes = regime_rows.dropna(subset=["sharpe"])
    if not valid_regimes.empty:
        best     = valid_regimes["sharpe"].idxmax()
        worst    = valid_regimes["sharpe"].idxmin()
        best_sh  = valid_regimes.loc[best,  "sharpe"]
        worst_sh = valid_regimes.loc[worst, "sharpe"]
        print("  CONCLUSIÓN ACCIONABLE:")
        print(f"    Mejor régimen  : {best} (Sharpe: {_fmt_f(best_sh)})")
        print(f"    Peor régimen   : {worst} (Sharpe: {_fmt_f(worst_sh)})")
        print(f"    Recomendación  : Aplicar estrategia en '{best}', "
              f"pausar en '{worst}'")
    print()

    print(f"  Outputs guardados en: {OUTPUT_DIR.resolve()}")
    print("  ══════════════════════════════════════════════════════════════")
    print()


def main() -> None:
    """Pipeline principal del GARCH Regime Backtest."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Carga de datos y clasificación GARCH ───────────────────────────────
    df_master = garch_loader.load_garch_data()

    source = (
        "Proyecto 1 (~/Desktop/garch-nasdaq-anomaly/)"
        if garch_loader.RETURNS_CSV.exists()
        else "yfinance (fallback)"
    )

    # ── 2. Señales SMA Crossover ──────────────────────────────────────────────
    df_signals = strategy.generate_signals(df_master)

    # ── 3. Motor de backtest ──────────────────────────────────────────────────
    df_backtest = backtest.run_backtest(df_signals)

    # ── 4. Métricas por régimen ───────────────────────────────────────────────
    df_metrics = metrics.compute_metrics(df_backtest)

    # ── 5. Visualizaciones ────────────────────────────────────────────────────
    visualizations.plot_all(df_backtest, df_metrics)

    # ── 6. Guardar CSVs ───────────────────────────────────────────────────────
    metrics_path  = OUTPUT_DIR / "regime_metrics.csv"
    backtest_path = OUTPUT_DIR / "backtest_full.csv"

    df_metrics.to_csv(metrics_path)
    df_backtest.to_csv(backtest_path)
    print(f"[INFO] regime_metrics.csv  → {metrics_path}")
    print(f"[INFO] backtest_full.csv   → {backtest_path}")

    # ── 7. Reporte de consola ─────────────────────────────────────────────────
    print_report(df_backtest, df_metrics, source=source)

    # ── 8. Ruta absoluta del proyecto ─────────────────────────────────────────
    print(f"  Proyecto ubicado en: {PROJECT_ROOT.resolve()}")
    print()


if __name__ == "__main__":
    main()
