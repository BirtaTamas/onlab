# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_cologne/iem-cologne-2025-the-mongolz-vs-natus-vincere-bo3-jwAddb1WR9PRMQexpSMSG8/the-mongolz-vs-natus-vincere-m2-ancient.csv`
- round_num: `13`
- rows: `128`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.293518 | 0.116864 | 0.379184 | 0.757812 | 0.293518 |
| xgboost | 0.375095 | 0.171724 | 0.505231 | 0.640625 | 0.375095 |

## Closer Per Tick

- lstm: `116`
- xgboost: `12`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
