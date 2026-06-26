# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-spirit-vs-furia-bo3-_zQK5XUu10iN1JLmPA8zQ4/spirit-vs-furia-m2-nuke.csv`
- round_num: `20`
- rows: `265`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.372602 | 0.194528 | 0.532150 | 0.562264 | 0.372602 |
| xgboost | 0.366881 | 0.183284 | 0.517142 | 0.535849 | 0.366881 |

## Closer Per Tick

- lstm: `141`
- xgboost: `124`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
