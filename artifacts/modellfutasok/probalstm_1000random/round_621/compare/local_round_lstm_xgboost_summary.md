# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/asian_champions_league/hero-esports-asian-champions-league-2025-flyquest-vs-nomads-bo3-rjDbNQ6hoJ50qwkbItjOHm/flyquest-vs-nomads-m2-mirage.csv`
- round_num: `9`
- rows: `239`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.319384 | 0.198972 | 0.513281 | 0.569038 | 0.319384 |
| xgboost | 0.299360 | 0.165998 | 0.452016 | 0.564854 | 0.299360 |

## Closer Per Tick

- lstm: `138`
- xgboost: `101`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
