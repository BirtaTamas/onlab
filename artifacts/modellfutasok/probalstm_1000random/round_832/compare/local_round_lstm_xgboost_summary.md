# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m2-inferno.csv`
- round_num: `6`
- rows: `191`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.471065 | 0.245343 | 0.677373 | 0.486911 | 0.528935 |
| xgboost | 0.410196 | 0.181413 | 0.544857 | 0.863874 | 0.589804 |

## Closer Per Tick

- lstm: `37`
- xgboost: `154`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
