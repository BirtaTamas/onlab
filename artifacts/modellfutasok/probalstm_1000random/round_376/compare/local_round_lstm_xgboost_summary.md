# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `4`
- rows: `309`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.571111 | 0.342505 | 0.917801 | 0.288026 | 0.571111 |
| xgboost | 0.559518 | 0.328061 | 0.890263 | 0.197411 | 0.559518 |

## Closer Per Tick

- lstm: `156`
- xgboost: `153`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
