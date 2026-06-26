# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-mouz-vs-falcons-bo3-OIe4ELGS25ekkV8Rf6FbR4/mouz-vs-falcons-m3-mirage.csv`
- round_num: `19`
- rows: `153`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.454676 | 0.264264 | 0.687229 | 0.274510 | 0.545324 |
| xgboost | 0.362365 | 0.171014 | 0.491677 | 0.908497 | 0.637635 |

## Closer Per Tick

- lstm: `3`
- xgboost: `150`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
