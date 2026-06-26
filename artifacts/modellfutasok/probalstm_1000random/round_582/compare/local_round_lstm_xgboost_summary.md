# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-3dmax-vs-mibr-bo3-O12tFfVag47APQdKBJkGZl/3dmax-vs-mibr-m2-ancient-p3.csv`
- round_num: `8`
- rows: `197`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.104952 | 0.022377 | 0.118254 | 1.000000 | 0.104952 |
| xgboost | 0.211376 | 0.069355 | 0.257215 | 1.000000 | 0.211376 |

## Closer Per Tick

- lstm: `197`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
