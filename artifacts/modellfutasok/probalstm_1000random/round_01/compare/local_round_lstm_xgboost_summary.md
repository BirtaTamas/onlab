# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `15`
- rows: `206`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.236347 | 0.112448 | 0.324274 | 0.684466 | 0.236347 |
| xgboost | 0.257216 | 0.112970 | 0.342527 | 0.825243 | 0.257216 |

## Closer Per Tick

- lstm: `134`
- xgboost: `72`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
