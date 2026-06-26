# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_rivals_season_1/blast-rivals-2025-season-1-mouz-vs-pain-bo3-Ao8EIC0rxvFkpkJ5bGImFu/mouz-vs-pain-m1-nuke.csv`
- round_num: `19`
- rows: `212`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.408426 | 0.179195 | 0.539692 | 1.000000 | 0.591574 |
| xgboost | 0.362704 | 0.145539 | 0.464919 | 1.000000 | 0.637296 |

## Closer Per Tick

- lstm: `4`
- xgboost: `208`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
