"""
backtest.py
-----------
Motor de backtest con costos de transacción, equity curve y drawdown.
"""

import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float = 100_000,
    cost_bps: float = 5,
) -> pd.DataFrame:
    """
    Ejecuta el backtest de la estrategia SMA Crossover vs Buy & Hold.

    Descarta los primeros 60 días de warmup (GARCH + SMAs) antes de calcular
    rendimientos acumulados, para operar completamente out-of-sample.

    Costos de transacción:
        5 bps aplicados en cada día donde la señal cambia de estado.

    Args:
        df: DataFrame con señales generadas por strategy.generate_signals().
        initial_capital: Capital inicial en USD (default: 100,000).
        cost_bps: Costo por operación en puntos básicos (default: 5).

    Returns:
        DataFrame con columnas:
            date | signal | log_return | strategy_return | cum_strategy |
            cum_benchmark | regime_code | regime_name | drawdown
    """
    # ── Selección de columnas y limpieza ─────────────────────────────────────
    cols = [
        "close", "log_return", "signal", "signal_shifted",
        "vol_cond", "z_score", "regime_code", "regime_name",
        "sma_20", "sma_50",
    ]
    out = df[cols].copy()

    # Eliminar filas sin señal válida (warmup SMA/GARCH)
    # CAUTION: dropna sobre signal_shifted elimina días sin posición definida
    out = out.dropna(subset=["signal_shifted", "log_return"]).copy()

    # ── Retorno de la estrategia ──────────────────────────────────────────────
    # CAUTION: strategy_return usa signal_shifted (posición del día anterior) ×
    #          log_return del día actual → sin lookahead bias
    out["strategy_return"] = out["signal_shifted"] * out["log_return"]

    # ── Costos de transacción ─────────────────────────────────────────────────
    # CAUTION: trade_days detecta cambios en la señal retrasada; se resta el
    #          costo solo en esos días para no introducir bias direccional
    trade_days = out["signal_shifted"].diff().abs() > 0
    out.loc[trade_days, "strategy_return"] -= cost_bps / 10_000

    # ── Equity curves ─────────────────────────────────────────────────────────
    out["cum_strategy"]  = initial_capital * (1 + out["strategy_return"]).cumprod()
    out["cum_benchmark"] = initial_capital * (1 + out["log_return"]).cumprod()

    # ── Drawdown de la estrategia ─────────────────────────────────────────────
    rolling_peak = out["cum_strategy"].cummax()
    out["drawdown"] = out["cum_strategy"] / rolling_peak - 1

    # ── Drawdown del benchmark ────────────────────────────────────────────────
    rolling_peak_bm = out["cum_benchmark"].cummax()
    out["drawdown_benchmark"] = out["cum_benchmark"] / rolling_peak_bm - 1

    out.index.name = "date"
    return out
