# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-nrg-vs-aurora-bo3-qymu5EnF_DYwHSVf1aSLaG/nrg-vs-aurora-m1-inferno.csv`
- round_num: `10`
- rows: `128`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.724799 | 0.564637 | 1.602396 | 0.140625 | 0.275201 |
| xgboost | 0.664834 | 0.465985 | 1.192969 | 0.125000 | 0.335166 |

## Closer Per Tick

- lstm: `30`
- xgboost: `98`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
