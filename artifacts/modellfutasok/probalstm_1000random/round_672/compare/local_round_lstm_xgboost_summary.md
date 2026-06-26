# LSTM vs XGBoost Local Round Comparison

- csv_path: `processed_full/fissure_playground_1/fissure-playground-1-tyloo-vs-saw-bo3-Ik7_qO98IbLkRUwtJ3asIP/tyloo-vs-saw-m1-nuke.csv`
- round_num: `29`
- rows: `127`

## Metrics

| model | mean_abs_error | brier | logloss | accuracy@0.5 | mean_probability |
|---|---:|---:|---:|---:|---:|
| lstm | 0.420967 | 0.199633 | 0.578845 | 0.685039 | 0.579033 |
| xgboost | 0.480726 | 0.274496 | 0.734674 | 0.677165 | 0.519274 |

## Closer Per Tick

- lstm: `101`
- xgboost: `26`
- tie: `0`

Winner by mean absolute error: `lstm`
Winner by Brier score: `lstm`
Winner by logloss: `lstm`
