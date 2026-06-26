# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/blast_open_london/blast-open-london-2025-flyquest-vs-legacy-bo3-FlEa8e0vdBrf1ft_mNbThh/flyquest-vs-legacy-m2-nuke.csv`
- round_num: `6`
- rows: `121`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.039358 | 0.002963 | 0.040937 | 1.000000 | 0.039358 |
| xgboost | 0.051564 | 0.004561 | 0.054010 | 1.000000 | 0.051564 |

## Closer Per Tick

- lstm: `99`
- xgboost: `22`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
