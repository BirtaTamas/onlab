# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `33`
- rows: `210`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.430940 | 0.218674 | 0.606820 | 0.471429 | 0.430940 |
| xgboost | 0.303276 | 0.110954 | 0.380290 | 0.852381 | 0.303276 |

## Closer Per Tick

- lstm: `11`
- xgboost: `199`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
