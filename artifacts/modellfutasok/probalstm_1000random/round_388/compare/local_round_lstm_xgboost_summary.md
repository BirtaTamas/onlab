# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-tyloo-vs-rare-atom-bo3-8GB1HWZtKOlh9_707n2A62/tyloo-vs-rare-atom-m2-inferno.csv`
- round_num: `14`
- rows: `167`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.500148 | 0.290492 | 0.756933 | 0.209581 | 0.500148 |
| xgboost | 0.501449 | 0.288196 | 0.753885 | 0.245509 | 0.501449 |

## Closer Per Tick

- lstm: `106`
- xgboost: `61`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
