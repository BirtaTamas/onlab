# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-furia-vs-b8-bo3-3h93b_qbGndTgDFTW66Ud1/furia-vs-b8-m1-mirage.csv`
- round_num: `13`
- rows: `131`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.541433 | 0.310650 | 0.849637 | 0.603053 | 0.458567 |
| xgboost | 0.412437 | 0.193454 | 0.575158 | 0.832061 | 0.587563 |

## Closer Per Tick

- lstm: `3`
- xgboost: `128`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
