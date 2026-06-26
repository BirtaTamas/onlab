# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-nrg-bo3-WMQcRUwgyUmu57EEkX9f3P/falcons-vs-nrg-m1-train.csv`
- round_num: `7`
- rows: `144`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.048574 | 0.006443 | 0.052269 | 1.000000 | 0.048574 |
| xgboost | 0.099093 | 0.018713 | 0.109997 | 1.000000 | 0.099093 |

## Closer Per Tick

- lstm: `137`
- xgboost: `7`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
