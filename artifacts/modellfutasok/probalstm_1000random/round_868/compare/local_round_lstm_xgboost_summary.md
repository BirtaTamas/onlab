# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_bounty_season_2_finals/blast-bounty-2025-season-2-finals-vitality-vs-the-mongolz-bo3-JVS9HKMAkaZTRHkoiRSMP6/vitality-vs-the-mongolz-m1-mirage.csv`
- round_num: `4`
- rows: `145`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.413509 | 0.186771 | 0.571662 | 0.910345 | 0.586491 |
| xgboost | 0.444524 | 0.215904 | 0.620786 | 0.441379 | 0.555476 |

## Closer Per Tick

- lstm: `102`
- xgboost: `43`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
