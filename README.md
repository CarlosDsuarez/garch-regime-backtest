# GARCH Regime Backtest — QQQ SMA Crossover (20/50)

Sistema de backtest que clasifica cada día del NASDAQ (QQQ) en un régimen de volatilidad
usando GARCH(1,1), y evalúa el rendimiento de una estrategia SMA Crossover segmentada
por régimen para determinar en qué entorno de volatilidad la estrategia genera alpha real.

## Prerequisitos

### Proyecto 1 (recomendado)
Este proyecto reutiliza datos del **Proyecto 1 — GARCH NASDAQ Anomaly Detector**,
que debe existir en:

```
~/Desktop/garch-nasdaq-anomaly/
└── data/
    └── qqq_returns.csv   ← requerido
```

Si el Proyecto 1 **no existe**, el sistema descargará QQQ automáticamente desde
yfinance (2010–hoy) e imprimirá una advertencia en consola.

### Dependencias Python
```bash
pip install -r requirements.txt
```

Requiere Python 3.10+.

## Estructura del Proyecto

```
garch-regime-backtest/
├── main.py              ← pipeline principal (punto de entrada)
├── garch_loader.py      ← carga datos, ajusta GARCH, clasifica regímenes
├── strategy.py          ← señales SMA Crossover (20/50)
├── backtest.py          ← motor de backtest con costos de transacción
├── metrics.py           ← cálculo de métricas por régimen
├── visualizations.py    ← 4 gráficos de análisis
├── data/                ← datos locales opcionales
├── outputs/             ← gráficos, tablas y tearsheet generados
├── requirements.txt
└── README.md
```

## Ejecución

```bash
cd ~/Desktop/garch-regime-backtest
python main.py
```

## Outputs Generados

| Archivo | Descripción |
|---------|-------------|
| `outputs/equity_curves.png` | Equity curve vs Buy&Hold + drawdown, sombreado por régimen |
| `outputs/regime_performance.png` | Heatmap de métricas + barras de retorno por régimen |
| `outputs/sma_signals.png` | Precio QQQ con SMAs y zonas de señal (último año) |
| `outputs/monthly_heatmap.png` | Heatmap de retornos mensuales (años × meses) |
| `outputs/regime_metrics.csv` | Tabla de métricas por régimen (5 regímenes + TOTAL) |
| `outputs/backtest_full.csv` | DataFrame completo del backtest día a día |

## Regímenes GARCH

| Régimen | Z-score | Código |
|---------|---------|--------|
| Vol Comprimida | Z < -1.5 | 0 |
| Normal | -1.5 ≤ Z < 1.5 | 1 |
| Vol Elevada | 1.5 ≤ Z < 2.5 | 2 |
| Anomalía | 2.5 ≤ Z < 3.5 | 3 |
| Anomalía Extrema | Z ≥ 3.5 | 4 |

## Estrategia

- **SMA Crossover (20/50)**: Largo cuando SMA_20 > SMA_50, plano en caso contrario.
- **Sin posiciones short**.
- **Costos de transacción**: 5 bps por operación (cambio de señal).
- **Sin lookahead bias**: la señal se desplaza 1 día (`signal.shift(1)`).

## Notas Técnicas

- Los primeros ~60 días se descartan como warmup (GARCH + SMAs).
- GARCH(1,1) con distribución t de Student (`dist='t'`).
- Z-score calculado con ventana rodante de 60 días.
- Volatilidad condicional anualizada: `sigma_t × √252`.
