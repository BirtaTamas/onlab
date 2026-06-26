# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62/tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `5`
- rows: `149`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.200702 | 0.134528 | 0.335201 | 0.711409 | 0.200702 |
| xgboost | 0.180341 | 0.105518 | 0.273691 | 0.718121 | 0.180341 |

## Closer Per Tick

- lstm: `91`
- xgboost: `58`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
