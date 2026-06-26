# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-mouz-vs-falcons-bo3-OIe4ELGS25ekkV8Rf6FbR4/mouz-vs-falcons-m3-mirage.csv`
- round_num: `12`
- rows: `252`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.413823 | 0.231907 | 0.628706 | 0.396825 | 0.413823 |
| xgboost | 0.457011 | 0.267097 | 0.711210 | 0.305556 | 0.457011 |

## Closer Per Tick

- lstm: `184`
- xgboost: `68`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
