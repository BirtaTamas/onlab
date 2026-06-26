# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-furia-vs-lynn-vision-bo3-KVSQ5iZB0TjTG70slfdqOB/furia-vs-lynn-vision-m2-overpass.csv`
- round_num: `12`
- rows: `175`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.261876 | 0.125979 | 0.360940 | 0.605714 | 0.261876 |
| xgboost | 0.290814 | 0.140649 | 0.403666 | 0.571429 | 0.290814 |

## Closer Per Tick

- lstm: `158`
- xgboost: `17`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
