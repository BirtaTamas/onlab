# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `3`
- rows: `190`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.012848 | 0.000439 | 0.013073 | 1.000000 | 0.012848 |
| xgboost | 0.039526 | 0.004319 | 0.041881 | 1.000000 | 0.039526 |

## Closer Per Tick

- lstm: `157`
- xgboost: `33`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
