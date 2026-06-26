# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-lynn-vision-vs-furia-bo3-RhNzrLTGYeGsl1rd1jweWL/lynn-vision-vs-furia-m2-anubis.csv`
- round_num: `10`
- rows: `132`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.397165 | 0.186405 | 0.541414 | 0.477273 | 0.602835 |
| xgboost | 0.368129 | 0.164216 | 0.491481 | 0.909091 | 0.631871 |

## Closer Per Tick

- lstm: `17`
- xgboost: `115`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
