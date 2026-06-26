# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `1`
- rows: `146`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.350485 | 0.152698 | 0.463948 | 1.000000 | 0.649515 |
| xgboost | 0.333500 | 0.146345 | 0.443118 | 0.904110 | 0.666500 |

## Closer Per Tick

- lstm: `68`
- xgboost: `78`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
