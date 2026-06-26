# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_2/fissure-playground-2-faze-vs-pain-bo3-N7fBU9m4mxAF0UgZPywYDX/faze-vs-pain-m1-nuke.csv`
- round_num: `21`
- rows: `230`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.130489 | 0.027774 | 0.148233 | 1.000000 | 0.869511 |
| xgboost | 0.101668 | 0.023035 | 0.116864 | 1.000000 | 0.898332 |

## Closer Per Tick

- lstm: `21`
- xgboost: `209`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
