# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-rare-atom-vs-nomads-bo3-2A6RLk5ZJnfAwsBhy_Qbbv/rare-atom-vs-nomads-m1-mirage.csv`
- round_num: `9`
- rows: `116`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.262685 | 0.090314 | 0.325915 | 0.991379 | 0.737315 |
| xgboost | 0.252997 | 0.089021 | 0.316297 | 0.974138 | 0.747003 |

## Closer Per Tick

- lstm: `42`
- xgboost: `74`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
