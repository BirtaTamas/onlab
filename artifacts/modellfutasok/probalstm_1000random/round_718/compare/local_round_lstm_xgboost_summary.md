# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `11`
- rows: `103`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.303496 | 0.116046 | 0.384337 | 1.000000 | 0.696504 |
| xgboost | 0.200670 | 0.059841 | 0.239958 | 1.000000 | 0.799330 |

## Closer Per Tick

- lstm: `0`
- xgboost: `103`
- tie: `0`

Winner by mean absolute error: `xgboost`
Winner by Brier score: `xgboost`
Winner by logloss: `xgboost`
