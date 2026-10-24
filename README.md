# IC3 H7 - mini digital twin for 48 h flood alerting (LamaH-CE, daily)

Decision supported: the duty hydrologist issues a flood warning when the discharge
forecast for day t+2 exceeds the alert threshold. This repository builds the twin
layer by layer (observations -> representation -> learning -> prediction ->
interpretation -> decision), one layer per checkpoint.

## Repository layout

```
configs/    basins.yaml - basins, inputs, paths (the only file you edit)
src/data/   CP1 observation layer: readers, selection, ingestion, figures
tests/      smoke_test.py - runs the whole pipeline on fake data
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
