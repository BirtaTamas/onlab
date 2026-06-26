# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_2/blasttv-austin-major-2025-stage-2-b8-vs-lynn-vision-bo3-Whl3pjYuIoHffY1VOn8vws/b8-vs-lynn-vision-m1-dust2.csv`
- round_num: `7`
- rows: `229`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.396267 | 0.164958 | 0.515080 | 0.956332 | 0.396267 |
| xgboost | 0.522496 | 0.280151 | 0.750652 | 0.174672 | 0.522496 |

## Closer Per Tick

- lstm: `226`
- xgboost: `3`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
