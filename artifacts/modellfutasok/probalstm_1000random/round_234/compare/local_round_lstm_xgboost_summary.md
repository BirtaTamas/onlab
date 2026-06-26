# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2/blast-bounty-2025-season-2-nrg-vs-vitality-bo3-7fVRmFXct71SEzAlz8IKvC/nrg-vs-vitality-m2-dust2.csv`
- round_num: `5`
- rows: `264`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.373490 | 0.167449 | 0.498814 | 0.787879 | 0.373490 |
| xgboost | 0.396842 | 0.191610 | 0.547697 | 0.666667 | 0.396842 |

## Closer Per Tick

- lstm: `204`
- xgboost: `60`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
