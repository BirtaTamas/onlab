# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-lynn-vision-vs-ninja-bo3-zpPbzx1DSQhVYC3-qoelpd/lynn-vision-vs-ninja-m2-inferno.csv`
- round_num: `18`
- rows: `186`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.268115 | 0.085530 | 0.324551 | 1.000000 | 0.731885 |
| xgboost | 0.244392 | 0.079413 | 0.297586 | 1.000000 | 0.755608 |

## Closer Per Tick

- lstm: `84`
- xgboost: `102`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
