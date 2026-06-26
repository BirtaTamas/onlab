# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m3-train.csv`
- round_num: `7`
- rows: `205`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.650232 | 0.489789 | 1.262491 | 0.195122 | 0.650232 |
| xgboost | 0.668428 | 0.473816 | 1.202909 | 0.136585 | 0.668428 |

## Closer Per Tick

- lstm: `80`
- xgboost: `125`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
