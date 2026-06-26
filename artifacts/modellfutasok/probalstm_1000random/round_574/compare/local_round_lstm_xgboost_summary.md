# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m2-dust2.csv`
- round_num: `3`
- rows: `116`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.067374 | 0.005556 | 0.070353 | 1.000000 | 0.932626 |
| xgboost | 0.039028 | 0.001772 | 0.039944 | 1.000000 | 0.960972 |

## Closer Per Tick

- lstm: `0`
- xgboost: `116`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
