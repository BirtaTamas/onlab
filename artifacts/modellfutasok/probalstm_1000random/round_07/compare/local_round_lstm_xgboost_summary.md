# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m3-train.csv`
- round_num: `17`
- rows: `218`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.502011 | 0.263672 | 0.717850 | 0.261468 | 0.497989 |
| xgboost | 0.447810 | 0.211861 | 0.610790 | 0.949541 | 0.552190 |

## Closer Per Tick

- lstm: `3`
- xgboost: `215`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
