# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `14`
- rows: `278`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.275093 | 0.112032 | 0.358042 | 0.758993 | 0.275093 |
| xgboost | 0.301356 | 0.126051 | 0.401057 | 0.755396 | 0.301356 |

## Closer Per Tick

- lstm: `210`
- xgboost: `68`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
