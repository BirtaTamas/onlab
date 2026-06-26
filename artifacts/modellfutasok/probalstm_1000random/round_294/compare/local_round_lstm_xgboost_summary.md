# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `9`
- rows: `110`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.178115 | 0.042956 | 0.204512 | 1.000000 | 0.821885 |
| xgboost | 0.213669 | 0.060373 | 0.252035 | 1.000000 | 0.786331 |

## Closer Per Tick

- lstm: `95`
- xgboost: `15`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
