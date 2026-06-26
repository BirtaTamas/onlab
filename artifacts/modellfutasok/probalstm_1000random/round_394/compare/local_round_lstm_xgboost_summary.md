# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-flyquest-vs-nomads-bo3-rjDbNQ6hoJ50qwkbItjOHm/flyquest-vs-nomads-m2-mirage.csv`
- round_num: `5`
- rows: `284`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.345912 | 0.188244 | 0.505621 | 0.626761 | 0.345912 |
| xgboost | 0.275850 | 0.118970 | 0.364364 | 0.823944 | 0.275850 |

## Closer Per Tick

- lstm: `101`
- xgboost: `183`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
