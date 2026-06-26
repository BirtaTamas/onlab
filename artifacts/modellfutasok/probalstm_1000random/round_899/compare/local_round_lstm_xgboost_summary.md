# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-vitality-vs-faze-bo3-cXlexQJNK-6GX9ddkYcv53/vitality-vs-faze-m1-mirage.csv`
- round_num: `34`
- rows: `169`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.210693 | 0.111906 | 0.302193 | 0.644970 | 0.210693 |
| xgboost | 0.215862 | 0.110663 | 0.305643 | 0.633136 | 0.215862 |

## Closer Per Tick

- lstm: `125`
- xgboost: `44`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `xgboost`
Winner by logloss: `lstm`
