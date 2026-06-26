# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-b8-vs-lynn-vision-bo3-Whl3pjYuIoHffY1VOn8vws/b8-vs-lynn-vision-m1-dust2.csv`
- round_num: `11`
- rows: `136`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.826137 | 0.716485 | 2.118345 | 0.058824 | 0.173863 |
| xgboost | 0.660631 | 0.464841 | 1.164424 | 0.058824 | 0.339369 |

## Closer Per Tick

- lstm: `2`
- xgboost: `134`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
