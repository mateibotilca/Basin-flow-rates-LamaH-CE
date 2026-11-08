# CP2 summary - baselines on the development period

**Protocol.** One forecast per issue date t (end of day t, inputs <= t), target = mean discharge of day t+2; samples belong to the hydrological year (Oct-Sep) of the target. Development HY1982-2010 is split into 6 blocked leave-years-out folds (1982-1986, 1987-1991, 1992-1996, 1997-2001, 2002-2006, 2007-2010); training targets within 32 days of the test block are purged. Hold-out HY2011-2017 is not used. Alert thresholds (Q95, Q99), climatology and RF are fitted on each fold's training targets only. Test predictions of all folds are pooled; [intervals] are 90 % percentile intervals from 1000 bootstrap resamples of whole HY. RF: 300 trees, min_samples_leaf 5, max_features 0.33, mean of seeds [0, 1, 2], no tuning.

| Basin | Metric | Persistence | Climatology | RF-A (engineered) | RF-B (raw lags) |
|---|---|---|---|---|---|
| 533 | NSE | 0.80 [0.76, 0.83] | 0.75 [0.72, 0.78] | 0.89 [0.87, 0.91] | 0.89 [0.87, 0.91] |
| 533 | CSI@Q95 | 0.36 [0.31, 0.40] | 0.00 [0.00, 0.00] | 0.45 [0.40, 0.50] | 0.45 [0.40, 0.50] |
| 533 | CSI@Q99 | 0.14 [0.09, 0.19] | 0.00 [0.00, 0.00] | 0.18 [0.11, 0.23] | 0.12 [0.05, 0.18] |
| 332 | NSE | 0.11 [-0.02, 0.26] | 0.05 [0.01, 0.08] | 0.43 [0.37, 0.50] | 0.42 [0.36, 0.49] |
| 332 | CSI@Q95 | 0.33 [0.27, 0.39] | 0.00 [0.00, 0.00] | 0.38 [0.32, 0.44] | 0.36 [0.29, 0.43] |
| 332 | CSI@Q99 | 0.17 [0.10, 0.24] | 0.00 [0.00, 0.00] | 0.12 [0.07, 0.16] | 0.12 [0.06, 0.16] |
| 195 | NSE | -0.18 [-0.33, -0.01] | 0.06 [0.04, 0.08] | 0.27 [0.22, 0.31] | 0.26 [0.21, 0.31] |
| 195 | CSI@Q95 | 0.21 [0.16, 0.24] | 0.00 [0.00, 0.00] | 0.21 [0.17, 0.26] | 0.22 [0.18, 0.26] |
| 195 | CSI@Q99 | 0.07 [0.01, 0.13] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.03] |

**Observations** (full table: `table_cp2_main.md`)

- Highest NSE per basin (ties at 2 decimals joined by '='): 533 RF-A (engineered) = RF-B (raw lags) 0.89; 332 RF-A (engineered) 0.43; 195 RF-A (engineered) 0.27.
- Highest CSI per basin - CSI@Q95: 533 RF-A (engineered) = RF-B (raw lags) 0.45; 332 RF-A (engineered) 0.38; 195 RF-B (raw lags) 0.22 | CSI@Q99: 533 RF-A (engineered) 0.18; 332 Persistence 0.17; 195 Persistence 0.07. Climatology issues no alert at any threshold in all basins (POD = CSI = 0, FAR undefined).
- CSI@Q99 90 % intervals span 0.00-0.15 (persistence and RF); 104-112 observed Q99 exceedance days per basin over 29 HY.
- RF-A (engineered) minus RF-B (raw lags): NSE 533 -0.001, 332 +0.008, 195 +0.006; CSI@Q95 533 -0.00, 332 +0.02, 195 -0.00; CSI@Q99 533 +0.06, 332 +0.00, 195 -0.01.
- RF-A (seed 0) NSE, blocked vs LOYO folds (`table_cp2_loyo.csv`): 195 0.265 vs 0.265, 332 0.426 vs 0.430, 533 0.890 vs 0.891.
