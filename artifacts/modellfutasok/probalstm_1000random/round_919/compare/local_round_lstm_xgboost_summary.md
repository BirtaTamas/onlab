# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m1-inferno.csv`
- round_num: `20`
- rows: `114`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.293418 | 0.121898 | 0.381285 | 0.938596 | 0.293418 |
| xgboost | 0.257535 | 0.106259 | 0.337279 | 0.675439 | 0.257535 |

## Closer Per Tick

- lstm: `67`
- xgboost: `47`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
