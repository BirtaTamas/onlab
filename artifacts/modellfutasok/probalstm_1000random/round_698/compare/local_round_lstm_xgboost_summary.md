# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-flyquest-vs-legacy-bo3-FlEa8e0vdBrf1ft_mNbThh/flyquest-vs-legacy-m2-nuke.csv`
- round_num: `14`
- rows: `149`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.227035 | 0.066258 | 0.270286 | 1.000000 | 0.772965 |
| xgboost | 0.170013 | 0.043587 | 0.197376 | 1.000000 | 0.829987 |

## Closer Per Tick

- lstm: `4`
- xgboost: `145`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
