# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-natus-vincere-bo3-jwAddb1WR9PRMQexpSMSG8/the-mongolz-vs-natus-vincere-m2-ancient.csv`
- round_num: `6`
- rows: `204`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.165760 | 0.060614 | 0.208133 | 1.000000 | 0.834240 |
| xgboost | 0.177876 | 0.074987 | 0.232765 | 1.000000 | 0.822124 |

## Closer Per Tick

- lstm: `95`
- xgboost: `109`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
