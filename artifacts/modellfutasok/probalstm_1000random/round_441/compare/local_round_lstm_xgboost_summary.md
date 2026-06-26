# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-g2-vs-virtuspro-bo3-lXkBTaEEYeJRsoa-wcGwPP/g2-vs-virtus-pro-m3-dust2.csv`
- round_num: `4`
- rows: `141`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.162806 | 0.039072 | 0.186910 | 1.000000 | 0.837194 |
| xgboost | 0.176535 | 0.052453 | 0.210823 | 0.971631 | 0.823465 |

## Closer Per Tick

- lstm: `59`
- xgboost: `82`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
