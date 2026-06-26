# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-flyquest-vs-nomads-bo3-rjDbNQ6hoJ50qwkbItjOHm/flyquest-vs-nomads-m2-mirage.csv`
- round_num: `4`
- rows: `250`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.141686 | 0.041052 | 0.169790 | 1.000000 | 0.858314 |
| xgboost | 0.128478 | 0.041312 | 0.158543 | 0.900000 | 0.871522 |

## Closer Per Tick

- lstm: `56`
- xgboost: `194`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `lstm`
Winner by logloss: `xgboost`
