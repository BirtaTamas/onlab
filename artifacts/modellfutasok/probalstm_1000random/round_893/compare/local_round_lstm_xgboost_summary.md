# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_dallas/iem-dallas-2025-falcons-vs-nrg-bo3-WMQcRUwgyUmu57EEkX9f3P/falcons-vs-nrg-m1-train.csv`
- round_num: `5`
- rows: `238`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.295106 | 0.146143 | 0.409236 | 0.605042 | 0.295106 |
| xgboost | 0.326562 | 0.159678 | 0.451033 | 0.441176 | 0.326562 |

## Closer Per Tick

- lstm: `184`
- xgboost: `54`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
