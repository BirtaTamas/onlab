# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `3`
- rows: `288`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.094193 | 0.015214 | 0.102844 | 1.000000 | 0.094193 |
| xgboost | 0.126453 | 0.023773 | 0.140166 | 1.000000 | 0.126453 |

## Closer Per Tick

- lstm: `227`
- xgboost: `61`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
