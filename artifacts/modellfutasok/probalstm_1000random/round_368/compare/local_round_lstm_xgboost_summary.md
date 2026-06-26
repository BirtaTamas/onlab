# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `25`
- rows: `213`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.769944 | 0.630161 | 1.895138 | 0.234742 | 0.230056 |
| xgboost | 0.773436 | 0.634434 | 1.827230 | 0.220657 | 0.226564 |

## Closer Per Tick

- lstm: `98`
- xgboost: `115`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `xgboost`
