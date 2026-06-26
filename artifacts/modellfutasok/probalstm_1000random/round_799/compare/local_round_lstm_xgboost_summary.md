# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne_stage_1/iem-cologne-2025-stage-1-3dmax-vs-mibr-bo3-O12tFfVag47APQdKBJkGZl/3dmax-vs-mibr-m2-ancient-p3.csv`
- round_num: `3`
- rows: `133`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.174944 | 0.071953 | 0.227052 | 1.000000 | 0.174944 |
| xgboost | 0.218666 | 0.096807 | 0.292190 | 0.819549 | 0.218666 |

## Closer Per Tick

- lstm: `133`
- xgboost: `0`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
