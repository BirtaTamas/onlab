# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-flyquest-vs-furia-bo3-kDRQKndVW9qgvAgGZjUFS9/flyquest-vs-furia-m2-dust2.csv`
- round_num: `18`
- rows: `149`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.269577 | 0.075956 | 0.317038 | 1.000000 | 0.730423 |
| xgboost | 0.233359 | 0.057886 | 0.268587 | 1.000000 | 0.766641 |

## Closer Per Tick

- lstm: `18`
- xgboost: `131`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
