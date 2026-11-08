# IC3 H7 - mini digital twin for 48 h flood alerting (LamaH-CE, daily)

Decision supported: the duty hydrologist issues a flood warning when the discharge
forecast for day t+2 exceeds the alert threshold. This repository builds the twin
layer by layer (observations -> representation -> learning -> prediction ->
interpretation -> decision), one layer per checkpoint.

## Repository layout

```
configs/    basins.yaml - basins, inputs, paths (the only file you edit)
            cp2.yaml - CP2 protocol, features, baselines, metrics
src/data/   CP1 observation layer: readers, selection, ingestion, figures
src/features/    CP2 samples, targets, representations A and B
src/evaluation/  CP2 folds + hold-out guard, metrics + bootstrap, figures
src/baselines/   CP2 persistence, climatology, Random Forest, entry point run_cp2
tests/      smoke_test.py - runs the whole pipeline on fake data
            test_cp2.py - leakage and protocol tests (pytest)
notebooks/  exploration only (nothing the results depend on)
results/    tables and figures per checkpoint (results/cp1/ ...)
paper/      article sources
data/       NOT in git: processed tables are regenerated from LamaH-CE
```

## CP1 - how to run (CPU, a few minutes)

1. Open this folder in VS Code and create an environment (terminal in VS Code):

   Windows (PowerShell):
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
   macOS / Linux:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Check the environment on fake data (no LamaH needed):
   ```
   python -m tests.smoke_test
   ```
   The last line must be `SMOKE TEST OK`.

3. Point the config at your data: in `configs/basins.yaml` set `lamah_root` to the
   folder that contains `A_basins_total_upstrm`, `D_gauges`, ... (forward slashes).

4. Check that every needed file exists and see the real column names:
   ```
   python -m src.data.inspect_files --config configs/basins.yaml
   ```

5. (Optional) Reproduce the basin selection:
   ```
   python -m src.data.select_basins --config configs/basins.yaml
   ```
   -> `results/cp1/basin_candidates.csv`

6. Build the observation layer:
   ```
   python -m src.data.build_dataset --config configs/basins.yaml
   ```
   -> `data/processed/basin_<id>.csv` (aligned daily table per basin)
   -> `results/cp1/basin_summary.csv`, `gap_report.csv`, `events_Q99.csv`
   -> `results/cp1/fig_cp1_record_overview.png/.pdf`, `fig_cp1_regimes.png/.pdf`

## CP2 - how to run (CPU, ~10 min on an M1)

CP2 adds the representation layer, the baselines and the evaluation protocol. Every
decision (lead time, folds, purge, thresholds, features, RF hyperparameters, bootstrap)
is in `configs/cp2.yaml`; the code reads it and hard-codes none of it.

1. Install the new dependencies (`scikit-learn`, `pytest`):
   ```
   pip install -r requirements.txt
   ```
2. Check the environment and the protocol on fake / synthetic data:
   ```
   python -m tests.smoke_test      # CP1 + CP2 end-to-end, last line: SMOKE TEST OK
   pytest -q                       # leakage and protocol tests (tests/test_cp2.py)
   ```
3. Run the baselines on the processed tables of CP1 (`data/processed/basin_<id>.csv`):
   ```
   python -m src.baselines.run_cp2 --config configs/cp2.yaml
   ```
   -> `results/cp2/table_cp2_main.csv/.md`, `SUMMARY.md`, `folds.csv`,
      `metrics_by_fold.csv`, `metrics_pooled.csv`, `predictions.csv.gz`, `run_info.json`,
      `fig_cp2_protocol.png/.pdf`, `fig_cp2_skill.png/.pdf`, `isel_qceq_days.csv`
4. (Robustness check) RF on pure leave-one-year-out folds, representation A, seed 0:
   ```
   python -m src.baselines.run_cp2 --config configs/cp2.yaml --folds loyo
   ```
   -> `results/cp2/table_cp2_loyo.csv`, `folds_loyo.csv` (SUMMARY.md is refreshed)

`--quick` (1 seed, representation A only) writes to `results/cp2/quick/` for debugging.

Protocol in short: one sample per issue date t, target = q_target(t+2), assigned to the
hydrological year (Oct-Sep) of the target. Development HY1982-2010 -> 6 blocked
leave-years-out folds with a 32-day purge around each test block; hold-out HY2011-2017
is never loaded at CP2 (a guard raises unless `allow_holdout=True`, reserved for CP4).
Thresholds (Q95, Q99), climatology and RF are fitted on each fold's training targets only.
Metrics are computed once on the pooled test predictions of all folds, with 90 %
intervals from a bootstrap over whole hydrological years.

## Design decisions fixed at CP1

* **Issue time:** end of day t (24:00 UTC). LamaH daily values aggregate
  00:00-23:59 UTC, so the day-t mean of every gauge and forcing is known at issue time.
* **Target:** mean discharge of day t+2 at the outlet gauge (m3/s).
* **Inputs:** only data up to day t. ERA5-Land is a reanalysis (observations), not a
  forecast, so using it for t+1/t+2 is a "perfect forecast" upper bound, reported
  separately in the CP4 ablation - never in the main results.
* **Gaps:** -999 -> NaN on an explicit daily calendar; no extra interpolation here.
* **Threshold:** Q99 in the CP1 tables is descriptive (full record). The alert
  threshold used for modelling is computed on training years only (CP2).

## Data and licence

LamaH-CE daily (Klingler et al., 2021, ESSD 13, 4529-4565,
https://doi.org/10.5194/essd-13-4529-2021; data https://doi.org/10.5281/zenodo.4525244),
CC BY 4.0. Runoff data providers that must be cited when using LamaH-CE:
BAFU, CHMI, GKD, HZB, LUBW (see the dataset's Info/References file).
# Basin-flow-rates-LamaH-CE
