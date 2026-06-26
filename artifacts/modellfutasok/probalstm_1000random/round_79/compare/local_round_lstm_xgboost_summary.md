# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m2-dust2.csv`
- round_num: `17`
- rows: `207`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.563527 | 0.325349 | 0.850845 | 0.067633 | 0.436473 |
| xgboost | 0.478705 | 0.235259 | 0.659353 | 0.888889 | 0.521295 |

## Closer Per Tick

- lstm: `4`
- xgboost: `203`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
