# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-vitality-vs-virtuspro-bo3-8Z0L17IYJlstHvIADVy9G9/vitality-vs-virtus-pro-m3-mirage.csv`
- round_num: `3`
- rows: `120`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.282771 | 0.104200 | 0.354995 | 0.966667 | 0.717229 |
| xgboost | 0.335215 | 0.155074 | 0.453130 | 0.850000 | 0.664785 |

## Closer Per Tick

- lstm: `76`
- xgboost: `44`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
