# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-flyquest-vs-nomads-bo3-rjDbNQ6hoJ50qwkbItjOHm/flyquest-vs-nomads-m2-mirage.csv`
- round_num: `7`
- rows: `112`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.105673 | 0.025207 | 0.121815 | 1.000000 | 0.894327 |
| xgboost | 0.111174 | 0.036390 | 0.136477 | 1.000000 | 0.888826 |

## Closer Per Tick

- lstm: `31`
- xgboost: `81`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
