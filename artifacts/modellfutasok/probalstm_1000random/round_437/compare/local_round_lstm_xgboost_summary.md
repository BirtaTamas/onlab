# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `7`
- rows: `264`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.098789 | 0.017136 | 0.108669 | 1.000000 | 0.098789 |
| xgboost | 0.178305 | 0.049411 | 0.209215 | 1.000000 | 0.178305 |

## Closer Per Tick

- lstm: `263`
- xgboost: `1`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
