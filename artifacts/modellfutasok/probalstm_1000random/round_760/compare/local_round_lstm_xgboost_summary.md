# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `1`
- rows: `209`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.320468 | 0.141378 | 0.427094 | 0.741627 | 0.320468 |
| xgboost | 0.360319 | 0.165277 | 0.489033 | 0.483254 | 0.360319 |

## Closer Per Tick

- lstm: `162`
- xgboost: `47`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
