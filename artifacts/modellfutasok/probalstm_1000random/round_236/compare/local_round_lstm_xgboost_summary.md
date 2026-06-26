# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-lynn-vision-vs-3dmax-bo3-peFr0yEP4eKTMrYfeYqBZK/lynn-vision-vs-3dmax-m3-train.csv`
- round_num: `23`
- rows: `249`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.352945 | 0.150084 | 0.460992 | 0.987952 | 0.352945 |
| xgboost | 0.457186 | 0.245854 | 0.660342 | 0.236948 | 0.457186 |

## Closer Per Tick

- lstm: `245`
- xgboost: `4`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
