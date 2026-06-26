# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-spirit-vs-flyquest-bo3-fQI-qOiPd1cRkmhkz0Xs5h/spirit-vs-flyquest-m1-mirage.csv`
- round_num: `12`
- rows: `161`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.449816 | 0.242956 | 0.699890 | 0.583851 | 0.550184 |
| xgboost | 0.430356 | 0.218201 | 0.648463 | 0.708075 | 0.569644 |

## Closer Per Tick

- lstm: `68`
- xgboost: `93`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
