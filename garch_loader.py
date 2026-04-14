"""
garch_loader.py
---------------
Carga datos QQQ desde el Proyecto 1 (~/Desktop/garch-nasdaq-anomaly/) o desde
yfinance como fallback, re-calibra GARCH(1,1) con dist='t', clasifica cada día
en un régimen de volatilidad y retorna un DataFrame maestro.
"""

import warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

warnings.filterwarnings("ignore")

# ── Constantes de régimen ─────────────────────────────────────────────────────
REGIME_CODES = {
    0: "Vol Comprimida",
    1: "Normal",
    2: "Vol Elevada",
    3: "Anomalía",
    4: "Anomalía Extrema",
}

PROJECT1_PATH = Path.home() / "Desktop" / "garch-nasdaq-anomaly"
RETURNS_CSV   = PROJECT1_PATH / "data" / "qqq_returns.csv"


def _classify_regime(z: float) -> int:
    """Asigna código de régimen según z-score de volatilidad condicional."""
    if z < -1.5:
        return 0
    elif z < 1.5:
        return 1
    elif z < 2.5:
        return 2
    elif z < 3.5:
        return 3
    else:
        return 4


def _fit_garch(returns: pd.Series) -> pd.Series:
    """
    Ajusta GARCH(1,1) con distribución t de Student y extrae volatilidad
    condicional anualizada.

    Args:
        returns: Serie de log-retornos diarios (en %).

    Returns:
        Serie de volatilidad condicional anualizada.
    """
    am = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="t", rescale=False)
    res = am.fit(disp="off", show_warning=False)
    # sigma en % diario → anualizar y convertir a decimal
    sigma_annual = res.conditional_volatility * np.sqrt(252) / 100
    return sigma_annual


def load_garch_data() -> pd.DataFrame:
    """
    Carga o descarga datos QQQ, re-calibra GARCH(1,1) y clasifica regímenes.

    Intenta leer ~/Desktop/garch-nasdaq-anomaly/data/qqq_returns.csv.
    Si no existe, descarga QQQ desde yfinance (2010-01-01 hasta hoy).

    Returns:
        DataFrame con columnas:
            date | close | log_return | vol_cond | z_score |
            regime_code | regime_name
    """
    # ── 1. Fuente de datos ────────────────────────────────────────────────────
    source = "Proyecto 1"
    df_raw: pd.DataFrame

    if RETURNS_CSV.exists():
        try:
            df_raw = pd.read_csv(RETURNS_CSV, parse_dates=True, index_col=0)
            # Normalizar nombres de columnas (el P1 puede usar distintas convenciones)
            df_raw.columns = [c.lower().strip() for c in df_raw.columns]
            required = {"close", "log_return"}
            if not required.issubset(set(df_raw.columns)):
                raise ValueError(f"Columnas faltantes: {required - set(df_raw.columns)}")
            df_raw.index.name = "date"
            print(f"[INFO] Datos cargados desde Proyecto 1: {RETURNS_CSV}")
        except Exception as exc:
            print(f"[WARN] Error leyendo Proyecto 1 ({exc}). Usando fallback yfinance.")
            source = "yfinance (fallback)"
            df_raw = _download_qqq()
    else:
        print("[WARN] Proyecto 1 no encontrado en ~/Desktop/garch-nasdaq-anomaly/")
        print("[WARN] Usando fallback: descargando QQQ desde yfinance (2010-hoy).")
        source = "yfinance (fallback)"
        df_raw = _download_qqq()

    # ── 2. Limpieza básica ────────────────────────────────────────────────────
    df = df_raw[["close", "log_return"]].dropna().copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # ── 3. Re-ajuste GARCH(1,1) con dist='t' ─────────────────────────────────
    print("[INFO] Ajustando GARCH(1,1) con dist='t'...")
    vol_cond = _fit_garch(df["log_return"])
    df["vol_cond"] = vol_cond.values  # volatilidad condicional anualizada

    # ── 4. Z-score rodante (ventana=60) ──────────────────────────────────────
    # CAUTION: rolling z-score usa solo información pasada → sin lookahead bias
    roll_mean = df["vol_cond"].rolling(60).mean()
    roll_std  = df["vol_cond"].rolling(60).std()
    df["z_score"] = (df["vol_cond"] - roll_mean) / roll_std

    # ── 5. Clasificación de régimen ───────────────────────────────────────────
    df["regime_code"] = df["z_score"].apply(_classify_regime)
    df["regime_name"] = df["regime_code"].map(REGIME_CODES)

    # Asegurar que el índice se llame 'date'
    df.index.name = "date"

    print(f"[INFO] Datos listos: {len(df)} días | Fuente: {source}")
    print(f"[INFO] Periodo: {df.index[0].date()} -> {df.index[-1].date()}", flush=True)

    return df


def _download_qqq() -> pd.DataFrame:
    """
    Descarga datos históricos de QQQ desde yfinance.

    Returns:
        DataFrame con columnas 'close' y 'log_return'.
    """
    ticker = yf.Ticker("QQQ")
    raw = ticker.history(start="2010-01-01", auto_adjust=True)
    if raw.empty:
        raise RuntimeError("yfinance no devolvió datos para QQQ.")

    df = pd.DataFrame()
    df["close"] = raw["Close"]
    # CAUTION: log_return en día T usa close[T] y close[T-1] → sin lookahead
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df.index.name = "date"
    df.dropna(inplace=True)
    return df
