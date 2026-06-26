# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m2-mirage.csv`
- round_num: `12`
- rows: `114`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.189779 | 0.050099 | 0.221655 | 1.000000 | 0.810221 |
| xgboost | 0.222395 | 0.074113 | 0.273988 | 1.000000 | 0.777605 |

## Closer Per Tick

- lstm: `82`
- xgboost: `32`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
