# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/iem_chengdu/iem-chengdu-2025-faze-vs-virtuspro-bo3-YDlVsCnS6YPgcr85bBYoPq/faze-vs-virtus-pro-m2-mirage.csv`
- round_num: `6`
- rows: `165`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.382992 | 0.198134 | 0.559985 | 0.630303 | 0.617008 |
| xgboost | 0.343513 | 0.166895 | 0.479559 | 0.890909 | 0.656487 |

## Closer Per Tick

- lstm: `33`
- xgboost: `132`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
