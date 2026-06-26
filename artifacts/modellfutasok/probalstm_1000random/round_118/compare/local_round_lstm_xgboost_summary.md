# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `6`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.372049 | 0.154367 | 0.482839 | 1.000000 | 0.627951 |
| xgboost | 0.364861 | 0.150963 | 0.473123 | 1.000000 | 0.635139 |

## Closer Per Tick

- lstm: `80`
- xgboost: `150`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
