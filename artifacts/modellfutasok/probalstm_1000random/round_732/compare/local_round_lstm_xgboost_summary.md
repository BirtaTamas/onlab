# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `26`
- rows: `222`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.462938 | 0.281594 | 0.762841 | 0.283784 | 0.462938 |
| xgboost | 0.472191 | 0.270056 | 0.753164 | 0.238739 | 0.472191 |

## Closer Per Tick

- lstm: `75`
- xgboost: `147`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
