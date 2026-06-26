# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london_finals/blast-open-london-2025-finals-faze-vs-g2-bo3-ldI7_iFRuThMOXF8zIbBwX/faze-vs-g2-m1-inferno.csv`
- round_num: `2`
- rows: `162`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.080770 | 0.024293 | 0.096549 | 1.000000 | 0.080770 |
| xgboost | 0.075522 | 0.019253 | 0.087337 | 1.000000 | 0.075522 |

## Closer Per Tick

- lstm: `131`
- xgboost: `31`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
