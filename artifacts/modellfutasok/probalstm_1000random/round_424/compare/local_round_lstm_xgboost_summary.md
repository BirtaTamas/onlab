# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-mouz-vs-falcons-bo3-OIe4ELGS25ekkV8Rf6FbR4/mouz-vs-falcons-m3-mirage.csv`
- round_num: `16`
- rows: `182`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.215145 | 0.113460 | 0.311870 | 0.725275 | 0.215145 |
| xgboost | 0.227976 | 0.114832 | 0.321150 | 0.725275 | 0.227976 |

## Closer Per Tick

- lstm: `141`
- xgboost: `41`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
