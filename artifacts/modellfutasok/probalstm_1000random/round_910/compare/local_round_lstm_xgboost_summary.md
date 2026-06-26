# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-pain-vs-rare-atom-bo3-Rmb0_mvtIpTmOfUIJVjwOw/pain-vs-rare-atom-m1-inferno.csv`
- round_num: `12`
- rows: `129`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.269234 | 0.123956 | 0.374851 | 0.767442 | 0.269234 |
| xgboost | 0.178219 | 0.066549 | 0.230464 | 0.829457 | 0.178219 |

## Closer Per Tick

- lstm: `4`
- xgboost: `125`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
