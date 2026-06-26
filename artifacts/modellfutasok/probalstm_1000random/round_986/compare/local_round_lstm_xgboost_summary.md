# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_austin_major_stage_1/blasttv-austin-major-2025-stage-1-imperial-vs-legacy-bo3-GRvbnL5Q4zT_JzAd-0AXgo/imperial-vs-legacy-m3-mirage.csv`
- round_num: `2`
- rows: `127`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.073710 | 0.006278 | 0.077061 | 1.000000 | 0.926290 |
| xgboost | 0.032361 | 0.001192 | 0.032974 | 1.000000 | 0.967639 |

## Closer Per Tick

- lstm: `22`
- xgboost: `105`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
