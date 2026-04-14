"""
visualizations.py
-----------------
Genera los 4 gráficos del sistema de backtest GARCH-Regime.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin pantalla

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

OUTPUT_DIR = Path(__file__).parent / "outputs"

# ── Paleta de colores por régimen ─────────────────────────────────────────────
REGIME_COLORS = {
    "Vol Comprimida" : "#AED6F1",
    "Normal"         : "#ABEBC6",
    "Vol Elevada"    : "#FAD7A0",
    "Anomalía"       : "#F0B27A",
    "Anomalía Extrema": "#F1948A",
}
REGIME_ORDER = [
    "Vol Comprimida", "Normal", "Vol Elevada", "Anomalía", "Anomalía Extrema"
]


def _shade_regimes(ax: plt.Axes, df: pd.DataFrame, col: str = "regime_name") -> None:
    """Sombrea el fondo del eje según el régimen activo en cada tramo."""
    dates   = df.index
    regimes = df[col].values
    start   = 0
    for i in range(1, len(regimes)):
        changed  = regimes[i] != regimes[i - 1]
        last_day = (i == len(regimes) - 1)
        if changed or last_day:
            end_idx = i if changed else i + 1
            color   = REGIME_COLORS.get(regimes[start], "#FFFFFF")
            ax.axvspan(
                dates[start],
                dates[min(end_idx, len(dates) - 1)],
                alpha=0.25, color=color, linewidth=0,
            )
            start = i


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 1 — Equity Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_equity_curves(backtest_df: pd.DataFrame) -> None:
    """
    Equity curve estrategia vs Buy&Hold con fondo sombreado por régimen.
    Subplot inferior: drawdown de ambas curvas.

    Args:
        backtest_df: DataFrame de backtest.run_backtest().
    """
    fig, axes = plt.subplots(
        2, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.suptitle(
        "QQQ SMA Crossover (20/50) — Equity Curve por Régimen GARCH",
        fontsize=14, fontweight="bold", y=0.98,
    )

    ax1 = axes[0]
    _shade_regimes(ax1, backtest_df)

    ax1.plot(
        backtest_df.index, backtest_df["cum_strategy"],
        color="#1A5276", linewidth=1.5, label="Estrategia SMA 20/50",
    )
    ax1.plot(
        backtest_df.index, backtest_df["cum_benchmark"],
        color="#7F8C8D", linewidth=1.2, alpha=0.8, label="Buy & Hold QQQ",
    )
    ax1.set_ylabel("Valor Portfolio ($)", fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    patches = [
        mpatches.Patch(color=c, alpha=0.5, label=r)
        for r, c in REGIME_COLORS.items()
    ]
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(
        handles + patches,
        labels + [p.get_label() for p in patches],
        loc="upper left", fontsize=8, ncol=2,
    )

    ax2 = axes[1]
    _shade_regimes(ax2, backtest_df)

    ax2.fill_between(
        backtest_df.index, backtest_df["drawdown"] * 100, 0,
        color="#C0392B", alpha=0.6, label="Drawdown Estrategia",
    )
    ax2.fill_between(
        backtest_df.index, backtest_df["drawdown_benchmark"] * 100, 0,
        color="#7F8C8D", alpha=0.35, label="Drawdown Benchmark",
    )
    ax2.set_ylabel("Drawdown (%)", fontsize=11)
    ax2.set_xlabel("Fecha", fontsize=11)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out = OUTPUT_DIR / "equity_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] equity_curves.png → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 2 — Regime Performance
# ─────────────────────────────────────────────────────────────────────────────
def plot_regime_performance(
    backtest_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    """
    Tabla heatmap de métricas por régimen + barras de retorno anualizado.

    Args:
        backtest_df: DataFrame del backtest.
        metrics_df:  DataFrame de metrics.compute_metrics().
    """
    m = metrics_df.drop(index="TOTAL", errors="ignore").copy()
    order = [r for r in REGIME_ORDER if r in m.index]
    m = m.loc[order]

    fig = plt.figure(figsize=(16, 10))
    gs  = fig.add_gridspec(2, 1, height_ratios=[1.2, 1], hspace=0.45)
    fig.suptitle(
        "Performance por Régimen GARCH — SMA Crossover (20/50)",
        fontsize=14, fontweight="bold",
    )

    ax_table = fig.add_subplot(gs[0])
    heat_cols = {
        "Ret Anual (%)": m["ret_anual"] * 100,
        "Sharpe"       : m["sharpe"],
        "Sortino"      : m["sortino"],
        "Max DD (%)"   : m["max_dd"] * 100,
        "Win Rate (%)" : m["win_rate"] * 100,
        "Días"         : m["dias"],
        "% Período"    : m["pct_periodo"] * 100,
    }
    heat_df = pd.DataFrame(heat_cols, index=m.index)

    norm_df = heat_df.copy()
    norm_df["Max DD (%)"] = -norm_df["Max DD (%)"]

    norm_min    = norm_df.min()
    norm_max    = norm_df.max()
    norm_scaled = (norm_df - norm_min) / (norm_max - norm_min + 1e-9)

    sns.heatmap(
        norm_scaled,
        annot=heat_df.round(2).astype(str),
        fmt="",
        cmap="RdYlGn",
        ax=ax_table,
        linewidths=0.5,
        cbar=False,
        annot_kws={"size": 10, "weight": "bold"},
    )
    ax_table.set_title("Métricas por Régimen (verde=mejor, rojo=peor)", fontsize=11)
    ax_table.set_xticklabels(ax_table.get_xticklabels(), rotation=20, ha="right")

    ax_bar = fig.add_subplot(gs[1])

    std_by_regime = (
        backtest_df.groupby("regime_name")["strategy_return"]
        .std()
        .reindex(order)
        * np.sqrt(252)
    )

    colors = [REGIME_COLORS.get(r, "#AED6F1") for r in order]
    ax_bar.bar(
        range(len(order)),
        m["ret_anual"] * 100,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        yerr=std_by_regime.values * 100,
        capsize=4,
        error_kw={"elinewidth": 1.2, "ecolor": "#333333"},
        zorder=3,
    )
    ax_bar.axhline(0, color="black", linewidth=0.8, linestyle="-")

    bm_ret = ((1 + backtest_df["log_return"].mean()) ** 252 - 1) * 100
    ax_bar.axhline(
        bm_ret, color="#E74C3C", linewidth=1.5, linestyle="--",
        label=f"Buy & Hold Ret. ({bm_ret:.1f}%)", zorder=4,
    )

    ax_bar.set_xticks(range(len(order)))
    ax_bar.set_xticklabels(order, rotation=15, ha="right")
    ax_bar.set_ylabel("Retorno Anualizado (%)", fontsize=11)
    ax_bar.set_title("Retorno Anualizado por Régimen ± 1 STD", fontsize=11)
    ax_bar.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=1))
    ax_bar.legend(fontsize=9)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    out = OUTPUT_DIR / "regime_performance.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] regime_performance.png → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 3 — SMA Signals con régimen
# ─────────────────────────────────────────────────────────────────────────────
def plot_sma_signals(backtest_df: pd.DataFrame) -> None:
    """
    Precio QQQ con SMAs, zonas de señal y fondo sombreado por régimen.
    Muestra el último año de datos para legibilidad.

    Args:
        backtest_df: DataFrame del backtest.
    """
    last_date  = backtest_df.index[-1]
    start_zoom = last_date - pd.DateOffset(years=1)
    df_zoom    = backtest_df.loc[start_zoom:]

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.suptitle(
        "QQQ — SMA Crossover con Régimen GARCH (Último año)",
        fontsize=14, fontweight="bold",
    )

    _shade_regimes(ax, df_zoom)

    long_mask = df_zoom["signal_shifted"] == 1
    for grp in _contiguous_mask(df_zoom.index, long_mask):
        ax.axvspan(grp[0], grp[-1], alpha=0.15, color="#27AE60", linewidth=0, zorder=2)

    flat_mask = df_zoom["signal_shifted"] == 0
    for grp in _contiguous_mask(df_zoom.index, flat_mask):
        ax.axvspan(grp[0], grp[-1], alpha=0.08, color="#7F8C8D", linewidth=0, zorder=2)

    ax.plot(df_zoom.index, df_zoom["close"],  color="#17202A", linewidth=1.0,
            label="QQQ Close", zorder=5)
    ax.plot(df_zoom.index, df_zoom["sma_20"], color="#2980B9", linewidth=1.4,
            label="SMA 20", zorder=6)
    ax.plot(df_zoom.index, df_zoom["sma_50"], color="#E67E22", linewidth=1.4,
            label="SMA 50", zorder=6)

    ax.set_ylabel("Precio ($)", fontsize=11)
    ax.set_xlabel("Fecha", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    regime_patches = [
        mpatches.Patch(color=c, alpha=0.5, label=r)
        for r, c in REGIME_COLORS.items()
    ]
    long_patch = mpatches.Patch(color="#27AE60", alpha=0.4, label="Señal Long")
    flat_patch = mpatches.Patch(color="#7F8C8D", alpha=0.3, label="Señal Flat")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles + [long_patch, flat_patch] + regime_patches,
        labels  + ["Señal Long", "Señal Flat"] + [p.get_label() for p in regime_patches],
        loc="upper left", fontsize=8, ncol=2,
    )

    plt.tight_layout()
    out = OUTPUT_DIR / "sma_signals.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] sma_signals.png → {out}")


def _contiguous_mask(index: pd.DatetimeIndex, mask: pd.Series) -> list[list]:
    """Agrupa índices contiguos donde mask es True."""
    groups: list[list] = []
    current: list = []
    for dt, val in zip(index, mask):
        if val:
            current.append(dt)
        else:
            if current:
                groups.append(current)
                current = []
    if current:
        groups.append(current)
    return groups


# ─────────────────────────────────────────────────────────────────────────────
# Gráfico 4 — Monthly Heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_monthly_heatmap(backtest_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    """
    Heatmap de retornos mensuales (años × meses) con columna de retorno anual.

    Args:
        backtest_df: DataFrame del backtest.
        metrics_df:  DataFrame de métricas (para título con Sharpe y Max DD).
    """
    monthly = (
        backtest_df["strategy_return"]
        .resample("ME")
        .apply(lambda x: (1 + x).prod() - 1)
    )

    df_m = monthly.to_frame("ret")
    df_m["year"]  = df_m.index.year
    df_m["month"] = df_m.index.month

    pivot = df_m.pivot(index="year", columns="month", values="ret")

    month_names = [
        "Ene", "Feb", "Mar", "Abr", "May", "Jun",
        "Jul", "Ago", "Sep", "Oct", "Nov", "Dic",
    ]
    pivot.columns = [month_names[c - 1] for c in pivot.columns]

    pivot["Anual"] = pivot.apply(
        lambda row: (1 + row.dropna()).prod() - 1, axis=1
    )

    fig, ax = plt.subplots(figsize=(18, max(6, len(pivot) * 0.5 + 3)))

    total_row    = metrics_df.loc["TOTAL"] if "TOTAL" in metrics_df.index else pd.Series()
    sharpe_total = total_row.get("sharpe", float("nan"))
    maxdd_total  = total_row.get("max_dd", float("nan"))

    fig.suptitle(
        f"Retornos Mensuales — SMA Crossover QQQ  |  "
        f"Sharpe Total: {sharpe_total:.2f}  |  Max DD: {maxdd_total*100:.1f}%",
        fontsize=13, fontweight="bold",
    )

    annot_data = pivot.map(
        lambda v: f"{v*100:.1f}%" if pd.notna(v) else ""
    )

    finite_vals = pivot.values[np.isfinite(pivot.values.astype(float))]
    vmax = max(abs(finite_vals).max(), 0.01) if len(finite_vals) > 0 else 0.1

    cmap = sns.diverging_palette(10, 133, as_cmap=True)

    sns.heatmap(
        pivot * 100,
        annot=annot_data,
        fmt="",
        cmap=cmap,
        center=0,
        vmin=-vmax * 100,
        vmax=vmax * 100,
        linewidths=0.4,
        linecolor="#CCCCCC",
        ax=ax,
        cbar_kws={"label": "Retorno (%)"},
        annot_kws={"size": 8},
    )

    ax.set_xlabel("Mes", fontsize=11)
    ax.set_ylabel("Año", fontsize=11)
    ax.set_title("Rojo = negativo · Blanco = cero · Verde = positivo", fontsize=10)

    if "Anual" in pivot.columns:
        anual_col_idx = list(pivot.columns).index("Anual")
        ax.add_patch(
            plt.Rectangle(
                (anual_col_idx, 0), 1, len(pivot),
                fill=False, edgecolor="black", linewidth=2,
                transform=ax.transData,
            )
        )

    plt.tight_layout()
    out = OUTPUT_DIR / "monthly_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] monthly_heatmap.png → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────
def plot_all(backtest_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    """
    Genera y guarda los 4 gráficos del sistema.

    Args:
        backtest_df: DataFrame completo del backtest.
        metrics_df:  DataFrame de métricas por régimen.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n[INFO] Generando gráficos...")
    plot_equity_curves(backtest_df)
    plot_regime_performance(backtest_df, metrics_df)
    plot_sma_signals(backtest_df)
    plot_monthly_heatmap(backtest_df, metrics_df)
    print("[INFO] Todos los gráficos generados.\n")
