"""
metrics.py
----------
Calcula métricas de performance por régimen GARCH y para el total del período.
"""

import numpy as np
import pandas as pd
from scipy import stats


REGIME_ORDER = [
    "Vol Comprimida",
    "Normal",
    "Vol Elevada",
    "Anomalía",
    "Anomalía Extrema",
    "TOTAL",
]


def _metrics_for_slice(
    returns: pd.Series,
    cum_ret: pd.Series,
    total_days: int,
) -> dict:
    """
    Calcula todas las métricas para una serie de retornos.

    Args:
        returns: Serie de retornos diarios de la estrategia.
        cum_ret: Serie de retorno acumulado de la estrategia (base 1).
        total_days: Número total de días del backtest (para % del período).

    Returns:
        Diccionario con todas las métricas calculadas.
    """
    n = len(returns)
    if n == 0:
        return {k: np.nan for k in [
            "ret_anual", "vol_anual", "sharpe", "sortino",
            "max_dd", "calmar", "win_rate", "profit_factor",
            "dias", "pct_periodo",
        ]}

    mean_daily   = returns.mean()
    std_daily    = returns.std(ddof=1)
    ret_anual    = (1 + mean_daily) ** 252 - 1
    vol_anual    = std_daily * np.sqrt(252)

    # Sharpe (sin tasa libre de riesgo para simplicidad)
    sharpe = ret_anual / vol_anual if vol_anual > 0 else np.nan

    # Sortino: solo días negativos como downside
    downside     = returns[returns < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = ret_anual / (downside_std * np.sqrt(252)) if downside_std and downside_std > 0 else np.nan

    # Max drawdown sobre el slice acumulado
    rolling_peak = cum_ret.cummax()
    dd_series    = cum_ret / rolling_peak - 1
    max_dd       = dd_series.min()

    # Calmar ratio
    calmar = ret_anual / abs(max_dd) if max_dd < 0 else np.nan

    # Win rate: días con posición abierta y retorno positivo
    active = returns[returns != 0]
    win_rate = (active > 0).sum() / len(active) if len(active) > 0 else np.nan

    # Profit factor
    gains  = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    profit_factor = gains / losses if losses > 0 else np.nan

    return {
        "ret_anual"    : ret_anual,
        "vol_anual"    : vol_anual,
        "sharpe"       : sharpe,
        "sortino"      : sortino,
        "max_dd"       : max_dd,
        "calmar"       : calmar,
        "win_rate"     : win_rate,
        "profit_factor": profit_factor,
        "dias"         : n,
        "pct_periodo"  : n / total_days,
    }


def compute_metrics(backtest_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula métricas de performance para cada régimen GARCH y para el total.

    Args:
        backtest_df: DataFrame producido por backtest.run_backtest().

    Returns:
        DataFrame con régimen como índice y métricas como columnas.
        Incluye fila 'TOTAL' al final.
    """
    total_days = len(backtest_df)

    # Construir equity base-1 para cálculos de drawdown por slice
    base1 = (1 + backtest_df["strategy_return"]).cumprod()

    rows = {}

    # ── Métricas por régimen ──────────────────────────────────────────────────
    for regime_name, group in backtest_df.groupby("regime_name"):
        idx     = group.index
        returns = group["strategy_return"]
        cum_ret = base1.loc[idx]
        rows[regime_name] = _metrics_for_slice(returns, cum_ret, total_days)

    # ── Métricas totales ──────────────────────────────────────────────────────
    rows["TOTAL"] = _metrics_for_slice(
        backtest_df["strategy_return"], base1, total_days
    )

    df_metrics = pd.DataFrame(rows).T

    # Ordenar según orden canónico
    ordered_idx = [r for r in REGIME_ORDER if r in df_metrics.index]
    df_metrics  = df_metrics.loc[ordered_idx]

    # Tipos numéricos
    numeric_cols = [
        "ret_anual", "vol_anual", "sharpe", "sortino",
        "max_dd", "calmar", "win_rate", "profit_factor",
        "dias", "pct_periodo",
    ]
    df_metrics[numeric_cols] = df_metrics[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    return df_metrics


def compute_benchmark_metrics(backtest_df: pd.DataFrame) -> dict:
    """
    Calcula métricas de performance del benchmark Buy & Hold.

    Args:
        backtest_df: DataFrame producido por backtest.run_backtest().

    Returns:
        Diccionario con ret_anual, vol_anual, sharpe, max_dd.
    """
    returns  = backtest_df["log_return"]
    base1    = (1 + returns).cumprod()

    mean_d   = returns.mean()
    std_d    = returns.std(ddof=1)
    ret_a    = (1 + mean_d) ** 252 - 1
    vol_a    = std_d * np.sqrt(252)
    sharpe   = ret_a / vol_a if vol_a > 0 else np.nan

    peak     = base1.cummax()
    max_dd   = (base1 / peak - 1).min()

    return {
        "ret_anual": ret_a,
        "vol_anual": vol_a,
        "sharpe"   : sharpe,
        "max_dd"   : max_dd,
    }
