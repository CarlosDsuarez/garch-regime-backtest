"""
strategy.py
-----------
Implementa la estrategia SMA Crossover (20/50) sobre precios de cierre QQQ.
"""

import pandas as pd
import numpy as np


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera señales de la estrategia SMA Crossover (20/50).

    Lógica:
        - SMA_20 > SMA_50 → señal = +1 (long)
        - SMA_20 ≤ SMA_50 → señal = 0  (flat, sin posición)
        - Sin posiciones short.
        - La señal se desplaza un día hacia adelante para eliminar lookahead bias:
          la señal calculada en el cierre del día D se ejecuta al cierre del día D+1.

    Args:
        df: DataFrame maestro con columnas 'close', 'log_return', 'regime_code',
            'regime_name', 'vol_cond', 'z_score'.

    Returns:
        DataFrame con las columnas originales más:
            sma_20 | sma_50 | signal | signal_shifted
    """
    out = df.copy()

    # ── SMAs sobre precio de cierre ───────────────────────────────────────────
    out["sma_20"] = out["close"].rolling(window=20).mean()
    out["sma_50"] = out["close"].rolling(window=50).mean()

    # ── Señal raw (0 o 1) ─────────────────────────────────────────────────────
    # CAUTION: señal raw calculada con datos hasta el cierre de ese día (no futuro)
    out["signal"] = np.where(out["sma_20"] > out["sma_50"], 1, 0)

    # ── Shift: evitar lookahead bias ──────────────────────────────────────────
    # CAUTION: signal_shifted[T] = signal[T-1]; la posición del día T se decide
    # con la información disponible al cierre del día T-1.
    out["signal_shifted"] = out["signal"].shift(1)

    # Los primeros 50 días no tienen SMAs completas → NaN → eliminar del análisis
    out.loc[out["sma_50"].isna(), ["signal", "signal_shifted"]] = np.nan

    return out
