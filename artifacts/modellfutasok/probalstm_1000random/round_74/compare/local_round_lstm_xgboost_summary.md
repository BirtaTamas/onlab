# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `13`
- rows: `215`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.184083 | 0.054950 | 0.223145 | 1.000000 | 0.815917 |
| xgboost | 0.119826 | 0.040931 | 0.150022 | 1.000000 | 0.880174 |

## Closer Per Tick

- lstm: `0`
- xgboost: `215`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
