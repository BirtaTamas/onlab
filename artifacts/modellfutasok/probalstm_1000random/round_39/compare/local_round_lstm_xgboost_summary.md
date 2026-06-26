# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `3`
- rows: `112`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.362237 | 0.165153 | 0.500351 | 0.866071 | 0.637763 |
| xgboost | 0.161937 | 0.042814 | 0.191833 | 0.946429 | 0.838063 |

## Closer Per Tick

- lstm: `0`
- xgboost: `112`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
