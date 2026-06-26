# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m3-inferno.csv`
- round_num: `8`
- rows: `301`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.176440 | 0.056186 | 0.214650 | 0.970100 | 0.176440 |
| xgboost | 0.191807 | 0.061281 | 0.233500 | 0.930233 | 0.191807 |

## Closer Per Tick

- lstm: `183`
- xgboost: `118`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
