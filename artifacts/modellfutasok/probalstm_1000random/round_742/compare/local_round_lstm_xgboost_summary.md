# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m3-train.csv`
- round_num: `13`
- rows: `210`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.354459 | 0.149717 | 0.462976 | 1.000000 | 0.645541 |
| xgboost | 0.249750 | 0.091212 | 0.314735 | 0.980952 | 0.750250 |

## Closer Per Tick

- lstm: `18`
- xgboost: `192`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
