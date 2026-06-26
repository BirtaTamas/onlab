# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-lynn-vision-vs-furia-bo3-RhNzrLTGYeGsl1rd1jweWL/lynn-vision-vs-furia-m2-anubis.csv`
- round_num: `18`
- rows: `183`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.066354 | 0.027099 | 0.086100 | 1.000000 | 0.066354 |
| xgboost | 0.086691 | 0.035849 | 0.114435 | 0.885246 | 0.086691 |

## Closer Per Tick

- lstm: `182`
- xgboost: `1`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
