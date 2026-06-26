# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m2-inferno.csv`
- round_num: `19`
- rows: `197`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.225642 | 0.060188 | 0.262840 | 1.000000 | 0.774358 |
| xgboost | 0.207694 | 0.055893 | 0.242359 | 1.000000 | 0.792306 |

## Closer Per Tick

- lstm: `59`
- xgboost: `138`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
